//! Inkling 2 benches: tensor-unit bf16 GEMM at Inkling's dense shapes, the DSpark hold block as byte-equivalent bf16
//! matvecs, the draft head's real hold and absorb, the core EXL3 expert projections on recorded MoE inputs, and the EXL3 decode chain (usage).
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const tf = @import("tensorfold");
const ik = tf.inkling2;
const exl3 = @import("tensorfold").exl3_projection;

const usage =
    \\tf-inkling-bench gemm                                         tensor-unit GEMM at the dense shapes, M 512/2048/4096
    \\tf-inkling-bench draft                                        the DSpark hold block as bf16 matvecs
    \\tf-inkling-bench dspark <model>                               the DSpark draft head's hold at depth 5 and 7 and an absorb of one row on <model>'s real weights
    \\                                                              (the embedding, the unembedding and mtp.safetensors; no layers): median of 9 runs each
    \\tf-inkling-bench dense [rows...]                              48 layers' dense matvecs (default rows 1 2 4 6 8 16), per shape at 1 and 8 rows, unembed at 1 row
    \\tf-inkling-bench experts <model> <layer> <xm.bin> <eids.bin> <rows> <row0> <act.bin> <out>
    \\                                                              w13 and w2 on recorded MoE inputs, bf16 out
    \\tf-inkling-bench xd <model> <layer> <xm.bin> <eids.bin> <rows> [check] [parts]
    \\                                                              the EXL3 decode chain on a layer's recorded MoE inputs: microseconds per layer call
    \\                                                              (median of 7 rounds after 2 warm-ups, 48 windows chained) at each row count of <rows>
    \\                                                              (one count or a comma list, e.g. 1,2,4,6,8,16), after a comparison of its outputs with the
    \\                                                              projections' path; `check` stops after the comparison, `parts` also times each dispatch alone
    \\tf-inkling-bench xd-dump <model> <layer> <xm.bin> <eids.bin> <rows> <row0> <act.bin> <out>
    \\                                                              the decode chain on recorded rows, every pick weighing 1/8: bf16 files <out>.{gu,y,y_own,moe,moe_own}.bin
    \\                                                              (y: w2 per pick on act.bin, the input `experts` measures; y_own: on the chain's own act),
    \\                                                              and <out>.{proj_y_own,proj_moe_own}.bin: the projections' path on its own act
    \\
;

const MvArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };
const TRIALS = 5;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    if (args.len >= 2 and std.mem.eql(u8, args[1], "gemm")) return gemm(init);
    if (args.len >= 2 and std.mem.eql(u8, args[1], "draft")) return draft(init);
    if (args.len >= 2 and std.mem.eql(u8, args[1], "dense")) return dense(init, args[2..]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "dspark")) return @import("bench_dspark.zig").run(init, args);
    if (args.len == 10 and std.mem.eql(u8, args[1], "experts")) return experts(init, args);
    if (args.len == 10 and std.mem.eql(u8, args[1], "xd-dump")) return xdDumpLeg(init, args);
    if (args.len >= 7 and std.mem.eql(u8, args[1], "xd")) return xdLeg(init, args);
    std.debug.print("{s}", .{usage});
    return error.UnknownMode;
}

fn fill(b: mtl.Buffer, n: usize, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (b.slice(u16, n)) |*x| {
        const f: f32 = (r.float(f32) * 2 - 1) * 0.05;
        x.* = @intCast(@as(u32, @bitCast(f)) >> 16);
    }
}

fn median(ms: []f64) f64 {
    std.mem.sort(f64, ms, {}, std.sort.asc(f64));
    return ms[ms.len / 2];
}

/// GPU ms of `reps` encodes of `body` in one command buffer, median over TRIALS after one discarded warm-up.
fn timed(q: mtl.Queue, reps: usize, ctx: anytype, comptime body: fn (@TypeOf(ctx), mtl.ComputeEncoder, usize) void) !f64 {
    var ms: [TRIALS]f64 = undefined;
    try trials(q, reps, ctx, body, &ms);
    return median(&ms);
}

/// Each trial's GPU ms of `reps` encodes of `body`, after one discarded warm-up.
fn trials(q: mtl.Queue, reps: usize, ctx: anytype, comptime body: fn (@TypeOf(ctx), mtl.ComputeEncoder, usize) void, ms: *[TRIALS]f64) !void {
    for (0..TRIALS + 1) |t| {
        const cb = q.commandBuffer();
        const e = cb.compute(.serial);
        for (0..reps) |r| body(ctx, e, r);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("bench: {s}", .{msg});
            return error.GpuFailed;
        }
        if (t > 0) ms[t - 1] = (cb.gpuEnd() - cb.gpuStart()) * 1e3 / @as(f64, @floatFromInt(reps));
    }
}

const Gemm = struct { pipes: [2][2]mtl.Pipeline, x: mtl.Buffer, w: mtl.Buffer, y: mtl.Buffer, m: usize, n: usize, k: usize };

fn gemmBody(g: *const Gemm, e: mtl.ComputeEncoder, _: usize) void {
    const tn = (g.n + 127) / 128;
    const tm = (g.m + 63) / 64;
    e.setPipeline(g.pipes[@intFromBool(g.m % 64 == 0)][@intFromBool(g.n % 128 == 0)]);
    e.setBuffer(g.x, 0, 0);
    e.setBuffer(g.w, 0, 1);
    var p: [16]i32 = @splat(0);
    const vals = [_]usize{ g.m, g.n, g.k, g.k, g.k, g.n, tn, tm, 2, g.k / 256 };
    for (vals, 0..) |v, i| p[i] = @intCast(v);
    e.setBytes(std.mem.asBytes(&p), 2);
    e.setBuffer(g.y, 0, 3);
    e.dispatchThreads(mtl.Size.of(tn * 4 * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
}

fn gemm(init: std.process.Init) !void {
    const device = try mtl.Device.init();
    const q = try device.queue();
    if (!tf.llama.contracts.available(device)) return error.NeedsTensorUnits;
    const src = try tf.llama.contracts.gemmSource(device, init.gpa);
    defer init.gpa.free(src);
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var g: Gemm = undefined;
    for (0..4) |i| {
        var nm: [16]u8 = undefined;
        g.pipes[i / 2][i % 2] = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrintSentinel(&nm, "ll_gemm_{d}{d}", .{ i / 2, i % 2 }, 0), false);
    }
    const shapes = [_][3]usize{ .{ 6144, 4608, 0 }, .{ 1024, 4608, 0 }, .{ 768, 4608, 0 }, .{ 48, 4608, 0 }, .{ 4608, 6144, 0 }, .{ 384, 4608, 0 }, .{ 201024, 4608, 0 } };
    const names = [_][]const u8{ "q|gate", "k|v", "r", "sink", "o", "router", "unembed" };
    for (shapes, names) |sh, name| for ([_]usize{ 512, 2048, 4096 }) |m| {
        g.m = m;
        g.n = sh[0];
        g.k = sh[1];
        g.x = try device.buffer(m * g.k * 2, mtl.ResourceOptions.shared);
        g.w = try device.buffer(g.n * g.k * 2, mtl.ResourceOptions.shared);
        g.y = try device.buffer(m * g.n * 2, mtl.ResourceOptions.shared);
        defer for ([_]mtl.Buffer{ g.x, g.w, g.y }) |b| b.deinit();
        fill(g.x, m * g.k, 1);
        fill(g.w, g.n * g.k, 2);
        const ms = try timed(q, if (g.n > 100000) 2 else 8, &g, gemmBody);
        const tflops = 2.0 * @as(f64, @floatFromInt(m * g.n * g.k)) / (ms * 1e-3) / 1e12;
        std.debug.print("{{\"gemm\": \"{s}\", \"m\": {d}, \"n\": {d}, \"k\": {d}, \"ms\": {d:.3}, \"tflops\": {d:.1}}}\n", .{ name, m, g.n, g.k, ms, tflops });
    };
}

const Mv = struct { pipe: [6]mtl.Pipeline, x: mtl.Buffer, y: mtl.Buffer };
/// `kern` forces the kernel's row capacity (1, 2, 4, 8, 16 or 32); 0 picks the smallest that holds `rows`, as the forward does.
const MvOp = struct { w: mtl.Buffer, rows: u32, n: u32, k: u32, kern: u32 = 0 };

fn mvEncode(mv: *const Mv, e: mtl.ComputeEncoder, op: MvOp) void {
    const cap = if (op.kern != 0) op.kern else op.rows;
    const idx: usize = if (cap <= 1) 0 else if (cap <= 2) 1 else if (cap <= 4) 2 else if (cap <= 8) 3 else if (cap <= 16) 4 else 5;
    e.setPipeline(mv.pipe[idx]);
    e.setBuffer(mv.x, 0, 0);
    e.setBuffer(op.w, 0, 1);
    e.setValue(MvArgs{ .rows = op.rows, .n = op.n, .k = op.k, .ldy = op.n }, 2);
    e.setBuffer(mv.y, 0, 3);
    e.dispatchGroups(mtl.Size.of((op.n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
}

const Phase = struct { mv: *const Mv, ops: []const MvOp };

fn phaseBody(p: Phase, e: mtl.ComputeEncoder, _: usize) void {
    for (p.ops) |op| mvEncode(p.mv, e, op);
}

fn mvSetup(device: mtl.Device) !struct { mv: Mv, lib: mtl.Library } {
    const lib = try mtl.Library.fromSource(device, ks.inkling2_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
    var mv: Mv = undefined;
    for ([_][:0]const u8{ "ik_matvec_1", "ik_matvec_2", "ik_matvec_4", "ik_matvec_8", "ik_matvec_16", "ik_matvec_32" }, 0..) |n, i| mv.pipe[i] = try mtl.Pipeline.init(device, lib, n, false);
    mv.x = try device.buffer(8 * 18432 * 2, mtl.ResourceOptions.shared);
    mv.y = try device.buffer(8 * 201024 * 2, mtl.ResourceOptions.shared);
    fill(mv.x, 8 * 18432, 3);
    return .{ .mv = mv, .lib = lib };
}

/// The decode step's dense matvecs: every layer's q, gate, k, v, r, sink, o and router with its own weights (9.55 GB,
/// so nothing stays in cache), at the given window widths in the given order; then each shape alone at 1 and 8 rows,
/// and the unembed for one row.
fn dense(init: std.process.Init, widths: []const [:0]const u8) !void {
    const device = try mtl.Device.init();
    const q = try device.queue();
    const s = try mvSetup(device);
    defer s.lib.deinit();
    const mv = s.mv;
    var bufs: std.ArrayList(mtl.Buffer) = .empty;
    defer {
        for (bufs.items) |b| b.deinit();
        bufs.deinit(init.gpa);
    }
    const shapes = [_][2]usize{ .{ 6144, 4608 }, .{ 6144, 4608 }, .{ 1024, 4608 }, .{ 1024, 4608 }, .{ 768, 4608 }, .{ 48, 4608 }, .{ 4608, 6144 }, .{ 384, 4608 } };
    var ops: [48 * shapes.len]MvOp = undefined;
    var bytes: f64 = 0;
    for (0..48) |l| for (shapes, 0..) |sh, j| {
        const b = try device.buffer(sh[0] * sh[1] * 2, mtl.ResourceOptions.shared);
        try bufs.append(init.gpa, b);
        fill(b, sh[0] * sh[1], 200 + l * shapes.len + j);
        ops[l * shapes.len + j] = .{ .w = b, .rows = 1, .n = @intCast(sh[0]), .k = @intCast(sh[1]) };
        bytes += @floatFromInt(sh[0] * sh[1] * 2);
    };
    // each width is `R` (the forward's kernel choice) or `R@K` (R rows through the K-row kernel)
    var list: [32][2]u32 = undefined;
    var nw: usize = 0;
    for (widths) |w| {
        var it = std.mem.splitScalar(u8, w, '@');
        const r = try std.fmt.parseInt(u32, it.first(), 10);
        list[nw] = .{ r, if (it.next()) |kk| try std.fmt.parseInt(u32, kk, 10) else 0 };
        nw += 1;
    }
    if (nw == 0) for ([_]u32{ 1, 2, 4, 6, 8, 16 }) |r| {
        list[nw] = .{ r, 0 };
        nw += 1;
    };
    for (list[0..nw]) |rk| {
        for (&ops) |*op| {
            op.rows = rk[0];
            op.kern = rk[1];
        }
        var t: [TRIALS]f64 = undefined;
        try trials(q, 2, Phase{ .mv = &mv, .ops = &ops }, phaseBody, &t);
        const all = t;
        const ms = median(&t);
        std.debug.print("{{\"dense_rows\": {d}, \"kernel\": {d}, \"ms\": {d:.3}, \"trials\": [{d:.2}, {d:.2}, {d:.2}, {d:.2}, {d:.2}], \"gb\": {d:.2}, \"gbps\": {d:.0}}}\n", .{ rk[0], rk[1], ms, all[0], all[1], all[2], all[3], all[4], bytes / 1e9, bytes / 1e6 / ms });
    }
    for (&ops) |*op| op.kern = 0;
    const names = [_][]const u8{ "q", "gate", "k", "v", "r", "sink", "o", "router" };
    for (shapes, names, 0..) |sh, name, j| for ([_]u32{ 1, 8 }) |rows| {
        var one: [48]MvOp = undefined;
        for (&one, 0..) |*op, l| {
            op.* = ops[l * shapes.len + j];
            op.rows = rows;
        }
        const ms = try timed(q, 2, Phase{ .mv = &mv, .ops = &one }, phaseBody);
        const b: f64 = @floatFromInt(48 * sh[0] * sh[1] * 2);
        std.debug.print("{{\"shape\": \"{s}\", \"n\": {d}, \"k\": {d}, \"rows\": {d}, \"ms48\": {d:.3}, \"gbps\": {d:.0}}}\n", .{ name, sh[0], sh[1], rows, ms, b / 1e6 / ms });
    };
    const unembed = [_]MvOp{.{ .w = try device.buffer(201024 * 4608 * 2, mtl.ResourceOptions.shared), .rows = 1, .n = 201024, .k = 4608 }};
    try bufs.append(init.gpa, unembed[0].w);
    fill(unembed[0].w, 201024 * 4608, 300);
    const ms = try timed(q, 4, Phase{ .mv = &mv, .ops = &unembed }, phaseBody);
    std.debug.print("{{\"unembed1_ms\": {d:.3}, \"gbps\": {d:.0}}}\n", .{ ms, 201024.0 * 4608 * 2 / 1e6 / ms });
}

fn draft(init: std.process.Init) !void {
    const device = try mtl.Device.init();
    const q = try device.queue();
    const s = try mvSetup(device);
    defer s.lib.deinit();
    const mv = s.mv;
    var gpa_ops: std.ArrayList(MvOp) = .empty;
    defer gpa_ops.deinit(init.gpa);
    var bufs: std.ArrayList(mtl.Buffer) = .empty;
    defer {
        for (bufs.items) |b| b.deinit();
        bufs.deinit(init.gpa);
    }
    const W = struct {
        fn make(d: mtl.Device, a: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer), n: usize, k: usize, seed: u64) !mtl.Buffer {
            const b = try d.buffer(n * k * 2, mtl.ResourceOptions.shared);
            fill(b, n * k, seed);
            try list.append(a, b);
            return b;
        }
    };
    // five draft layers, eight rows: q, k, v, o, gate, up, down (each layer its own weights)
    var layer_ops: [5 * 7]MvOp = undefined;
    for (0..5) |l| {
        const shapes = [_][2]usize{ .{ 6144, 4608 }, .{ 1024, 4608 }, .{ 1024, 4608 }, .{ 4608, 6144 }, .{ 18432, 4608 }, .{ 18432, 4608 }, .{ 4608, 18432 } };
        for (shapes, 0..) |sh, j| layer_ops[l * 7 + j] = .{ .w = try W.make(device, init.gpa, &bufs, sh[0], sh[1], 10 + l * 7 + j), .rows = 8, .n = @intCast(sh[0]), .k = @intCast(sh[1]) };
    }
    const unembed = [_]MvOp{.{ .w = try W.make(device, init.gpa, &bufs, 201024, 4608, 90), .rows = 7, .n = 201024, .k = 4608 }};
    const fold_w = try W.make(device, init.gpa, &bufs, 201024, 256, 91);
    var fold: [7]MvOp = undefined;
    for (&fold) |*f| f.* = .{ .w = fold_w, .rows = 1, .n = 201024, .k = 256 };
    var absorb: [10]MvOp = undefined;
    for (0..5) |l| for (0..2) |j| {
        absorb[l * 2 + j] = .{ .w = try W.make(device, init.gpa, &bufs, 1024, 4608, 100 + l * 2 + j), .rows = 1, .n = 1024, .k = 4608 };
    };
    const t_layers = try timed(q, 4, Phase{ .mv = &mv, .ops = &layer_ops }, phaseBody);
    const t_unembed = try timed(q, 4, Phase{ .mv = &mv, .ops = &unembed }, phaseBody);
    const t_fold = try timed(q, 4, Phase{ .mv = &mv, .ops = &fold }, phaseBody);
    const t_absorb = try timed(q, 4, Phase{ .mv = &mv, .ops = &absorb }, phaseBody);
    var all: std.ArrayList(MvOp) = .empty;
    defer all.deinit(init.gpa);
    try all.appendSlice(init.gpa, &layer_ops);
    try all.appendSlice(init.gpa, &unembed);
    try all.appendSlice(init.gpa, &fold);
    try all.appendSlice(init.gpa, &absorb);
    const t_all = try timed(q, 4, Phase{ .mv = &mv, .ops = all.items }, phaseBody);
    std.debug.print("{{\"draft_layers_ms\": {d:.3}, \"unembed7_ms\": {d:.3}, \"fold7_ms\": {d:.3}, \"absorb1_ms\": {d:.3}, \"hold_total_ms\": {d:.3}}}\n", .{ t_layers, t_unembed, t_fold, t_absorb, t_all });
}

fn experts(init: std.process.Init, args: []const [:0]const u8) !void {
    const layer = try std.fmt.parseInt(u32, args[3], 10);
    const rows = try std.fmt.parseInt(usize, args[6], 10);
    const row0 = try std.fmt.parseInt(usize, args[7], 10);
    const slot: usize = for (ik.forward.xm_layers, 0..) |l, k| {
        if (l == layer) break k;
    } else return error.LayerNotRecorded;
    const H: usize = 4608;
    const I: usize = 2304;
    const E: usize = 384;
    const L: usize = 48;
    const device = try mtl.Device.init();
    const q = try device.queue();
    var p = try ik.expert_pack.Pack.read(init.gpa, init.io, args[2], L, E);
    defer p.deinit();
    var ck = ik.weights.Checkpoint.init(init.gpa);
    defer ck.deinit();
    var widths: [5]u32 = @splat(0);
    const table = try ik.weights.expertTable(init.gpa, device, &ck, args[2], p.layers[layer], layer, H, I, E, &widths);
    defer table.deinit();
    const pipes_ = try exl3.Pipelines.load(device);
    var pipes = pipes_;
    defer pipes.deinit();
    const xm_all = try std.Io.Dir.cwd().readFileAlloc(init.io, args[4], init.gpa, .unlimited);
    defer init.gpa.free(xm_all);
    const eids_all = try std.Io.Dir.cwd().readFileAlloc(init.io, args[5], init.gpa, .unlimited);
    defer init.gpa.free(eids_all);
    const act = try std.Io.Dir.cwd().readFileAlloc(init.io, args[8], init.gpa, .unlimited);
    defer init.gpa.free(act);
    const S = rows * 8;
    const x = try device.buffer(rows * H * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    const ids = try device.buffer(S * 4, mtl.ResourceOptions.shared);
    defer ids.deinit();
    const xa = try device.buffer(S * I * 2, mtl.ResourceOptions.shared);
    defer xa.deinit();
    const gu = try device.buffer(S * 2 * I * 2, mtl.ResourceOptions.shared);
    defer gu.deinit();
    const y = try device.buffer(S * H * 2, mtl.ResourceOptions.shared);
    defer y.deinit();
    const scratch = try device.buffer(try exl3.scratchBytes(H, 2 * I, S), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const xm16 = std.mem.bytesAsSlice(u16, xm_all);
    const e32 = std.mem.bytesAsSlice(u32, eids_all);
    for (0..rows) |r| @memcpy(x.slice(u16, rows * H)[r * H ..][0..H], xm16[((row0 + r) * 2 + slot) * H ..][0..H]);
    for (0..rows) |r| @memcpy(ids.slice(u32, S)[r * 8 ..][0..8], e32[((row0 + r) * L + layer) * 8 ..][0..8]);
    if (act.len != S * I * 2) return error.ActSize;
    @memcpy(xa.slice(u8, S * I * 2), act);
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    e.useResource(table, mtl.ResourceUsage.read);
    for (ck.shards.items) |sh| e.useResource(sh.buffer, mtl.ResourceUsage.read);
    try exl3.encodeIndexed(&pipes, e, .{ .descriptors = .{ .buffer = table }, .count = E, .k = H, .n = 2 * I }, .{ .buffer = x }, .{ .buffer = gu }, .{ .buffer = scratch }, .{ .count = S, .x_stride = H, .y_stride = 2 * I, .routes = .{ .ids = .{ .buffer = ids }, .repeat = 8 } });
    try exl3.encodeIndexed(&pipes, e, .{ .descriptors = .{ .buffer = table, .offset = E * @sizeOf(exl3.ExpertDescriptor) }, .count = E, .k = I, .n = H }, .{ .buffer = xa }, .{ .buffer = y }, .{ .buffer = scratch }, .{ .count = S, .x_stride = I, .y_stride = H, .routes = .{ .ids = .{ .buffer = ids } } });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("experts: {s}", .{msg});
        return error.GpuFailed;
    }
    const out = try std.fmt.allocPrint(init.gpa, "{s}.gu.bin", .{args[9]});
    defer init.gpa.free(out);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out, .data = gu.slice(u8, S * 2 * I * 2) });
    const out2 = try std.fmt.allocPrint(init.gpa, "{s}.y.bin", .{args[9]});
    defer init.gpa.free(out2);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out2, .data = y.slice(u8, S * H * 2) });
    std.debug.print("{{\"layer\": {d}, \"rows\": {d}, \"slots\": {d}, \"k2\": {d}, \"k3\": {d}}}\n", .{ layer, rows, S, widths[2], widths[3] });
}

// ---- the EXL3 decode chain on a layer's recorded MoE inputs ----

const xd_calls = 48; // windows of rows one measurement chains, one a call, each with its own rows and picks
const xd_rounds = 7;
const xd_warmups = 2;
const xd_hidden: usize = 4608;
const xd_inner: usize = 2304;
const xd_experts: usize = 384;
const xd_taps_layers: usize = 48; // layers in a taps file's expert-pick rows

/// Which of the chain's four dispatches to encode; `dbg` takes the mid and finish kernels that also store gu and ey.
const Run = struct { pair: bool = true, mid: bool = true, down: bool = true, finish: bool = true, dbg: bool = false };

/// The chain's pipelines, its layer's descriptor table and its own scratch for up to 16 rows.
const Xd = struct {
    lib: mtl.Library,
    pair: mtl.Pipeline,
    mid: mtl.Pipeline,
    down: mtl.Pipeline,
    finish: mtl.Pipeline,
    mid_dbg: mtl.Pipeline,
    finish_dbg: mtl.Pipeline,
    table: mtl.Buffer,
    ew: mtl.Buffer,
    planes: mtl.Buffer,
    act: mtl.Buffer,
    sums: mtl.Buffer,
    moe: mtl.Buffer,
    gu: mtl.Buffer,
    ey: mtl.Buffer,

    const max_slots = 16 * 8;

    fn init(device: mtl.Device, table: mtl.Buffer) !Xd {
        const lib = try mtl.Library.fromSource(device, ks.inkling2_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        const H = xd_hidden;
        const I = xd_inner;
        const xd: Xd = .{
            .lib = lib,
            .pair = try mtl.Pipeline.init(device, lib, "ik_xd_pair", false),
            .mid = try mtl.Pipeline.init(device, lib, "ik_xd_mid", false),
            .down = try mtl.Pipeline.init(device, lib, "ik_xd_down", false),
            .finish = try mtl.Pipeline.init(device, lib, "ik_xd_finish", false),
            .mid_dbg = try mtl.Pipeline.init(device, lib, "ik_xd_mid_dbg", false),
            .finish_dbg = try mtl.Pipeline.init(device, lib, "ik_xd_finish_dbg", false),
            .table = table,
            .ew = try device.buffer(max_slots * 4, mtl.ResourceOptions.shared),
            .planes = try device.buffer(max_slots * 2 * H * 4, mtl.ResourceOptions.shared),
            .act = try device.buffer(max_slots * I * 2, mtl.ResourceOptions.shared),
            .sums = try device.buffer(max_slots * H * 4, mtl.ResourceOptions.shared),
            .moe = try device.buffer(16 * H * 2, mtl.ResourceOptions.shared),
            .gu = try device.buffer(max_slots * H * 2, mtl.ResourceOptions.shared),
            .ey = try device.buffer(max_slots * H * 2, mtl.ResourceOptions.shared),
        };
        @memset(xd.ew.slice(f32, max_slots), 0.125); // every pick weighs 1/8
        return xd;
    }

    fn deinit(xd: Xd) void {
        for ([_]mtl.Pipeline{ xd.pair, xd.mid, xd.down, xd.finish, xd.mid_dbg, xd.finish_dbg }) |p| p.deinit();
        for ([_]mtl.Buffer{ xd.ew, xd.planes, xd.act, xd.sums, xd.moe, xd.gu, xd.ey }) |b| b.deinit();
        xd.lib.deinit();
    }

    /// The dispatches `run` selects (all four by default) for `rows` rows of `x` and picks `ids` (from byte offsets `xo`
    /// and `io`); `run.dbg` also stores gu and the per-pick ey.
    fn encode(xd: *const Xd, e: mtl.ComputeEncoder, x: mtl.Buffer, xo: usize, ids: mtl.Buffer, io: usize, rows: usize, run: Run) void {
        const S = rows * 8;
        const args = ik.forward.Args{ .rows = @intCast(S), .cap = 4, .thr = 25 };
        const w2 = xd_experts * @sizeOf(exl3.ExpertDescriptor);
        if (run.pair) {
            e.setPipeline(xd.pair);
            e.setValue(args, 0);
            e.setBuffer(x, xo, 1);
            e.setBuffer(ids, io, 2);
            e.setBuffer(xd.table, 0, 3);
            e.setBuffer(xd.planes, 0, 4);
            e.dispatchGroups(mtl.Size.of(144, S, 2), mtl.Size.of(128, 1, 1));
        }
        if (run.mid) {
            e.setPipeline(if (run.dbg) xd.mid_dbg else xd.mid);
            e.setValue(args, 0);
            e.setBuffer(xd.planes, 0, 1);
            e.setBuffer(ids, io, 2);
            e.setBuffer(xd.table, 0, 3);
            e.setBuffer(xd.act, 0, 4);
            if (run.dbg) e.setBuffer(xd.gu, 0, 5);
            e.dispatchGroups(mtl.Size.of(36, S, 1), mtl.Size.of(32, 1, 1));
        }
        if (run.down) {
            e.setPipeline(xd.down);
            e.setValue(args, 0);
            e.setBuffer(xd.act, 0, 1);
            e.setBuffer(ids, io, 2);
            e.setBuffer(xd.table, w2, 3);
            e.setBuffer(xd.sums, 0, 4);
            e.dispatchGroups(mtl.Size.of(144, S, 1), mtl.Size.of(128, 1, 1));
        }
        if (run.finish) {
            e.setPipeline(if (run.dbg) xd.finish_dbg else xd.finish);
            e.setValue(args, 0);
            e.setBuffer(xd.sums, 0, 1);
            e.setBuffer(ids, io, 2);
            e.setBuffer(xd.table, w2, 3);
            e.setBuffer(xd.ew, 0, 4);
            e.setBuffer(xd.moe, 0, 5);
            if (run.dbg) e.setBuffer(xd.ey, 0, 6);
            e.dispatchGroups(mtl.Size.of(36, rows, 1), mtl.Size.of(256, 1, 1));
        }
    }
};

/// `xd_calls` windows of `rows` consecutive recorded rows of a layer: each call's rows and picks, in call order.
const XdWindows = struct {
    rows: usize,
    x: mtl.Buffer,
    ids: mtl.Buffer,
    /// Mean distinct experts a call picks.
    distinct: f64,

    /// Call k's first row is `first + (53 k + 17) mod (T - first - rows + 1)`: windows spread over the generated rows.
    fn init(device: mtl.Device, rows: usize, first: usize, T: usize, xm: []align(1) const u16, picks: []align(1) const u32, slot: usize, layer: usize) !XdWindows {
        if (T < first + rows) return error.SequenceTooShort;
        const H = xd_hidden;
        const x = try device.buffer(xd_calls * rows * H * 2, mtl.ResourceOptions.shared);
        errdefer x.deinit();
        const ids = try device.buffer(xd_calls * rows * 8 * 4, mtl.ResourceOptions.shared);
        errdefer ids.deinit();
        var distinct: usize = 0;
        for (0..xd_calls) |k| {
            const row = first + (k * 53 + 17) % (T - first - rows + 1);
            var seen: [xd_experts]bool = @splat(false);
            for (0..rows) |r| {
                @memcpy(x.slice(u16, xd_calls * rows * H)[(k * rows + r) * H ..][0..H], xm[((row + r) * 2 + slot) * H ..][0..H]);
                const p = picks[((row + r) * xd_taps_layers + layer) * 8 ..][0..8];
                for (p, 0..) |e, j| {
                    if (e >= xd_experts) return error.BadExpertId;
                    ids.slice(u32, xd_calls * rows * 8)[(k * rows + r) * 8 + j] = e;
                    seen[e] = true;
                }
            }
            for (seen) |s| distinct += @intFromBool(s);
        }
        return .{ .rows = rows, .x = x, .ids = ids, .distinct = @as(f64, @floatFromInt(distinct)) / xd_calls };
    }

    fn deinit(w: XdWindows) void {
        w.x.deinit();
        w.ids.deinit();
    }
};

/// Microseconds a call takes: the median over `xd_rounds` command buffers of `xd_calls` chained calls of the
/// dispatches `run` selects, after `xd_warmups` rounds that are not counted.
fn xdTime(q: mtl.Queue, xd: *const Xd, ck: *const ik.weights.Checkpoint, win: XdWindows, run: Run) !f64 {
    var us: [xd_rounds]f64 = undefined;
    for (0..xd_rounds + xd_warmups) |round| {
        const cb = q.commandBuffer();
        const e = cb.compute(.serial);
        e.useResource(xd.table, mtl.ResourceUsage.read);
        for (ck.shards.items) |sh| e.useResource(sh.buffer, mtl.ResourceUsage.read);
        for (0..xd_calls) |k| xd.encode(e, win.x, k * win.rows * xd_hidden * 2, win.ids, k * win.rows * 8 * 4, win.rows, run);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("xd: {s}", .{msg});
            return error.GpuFailed;
        }
        if (round >= xd_warmups) us[round - xd_warmups] = (cb.gpuEnd() - cb.gpuStart()) * 1e6 / xd_calls;
    }
    return median(&us);
}

const Cmp = struct { differ: usize = 0, max_ulp: u32 = 0, max_ulp_big: u32 = 0, max_rel: f64 = 0, nonfinite: usize = 0 };

fn ulpKey(v: u16) i32 {
    return if (v & 0x8000 != 0) -@as(i32, v & 0x7fff) else @as(i32, v);
}

fn bf16Value(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

/// bf16 arrays against each other (`b` the reference): elements that differ, the largest step between two (and the
/// largest among elements with |b| of at least 1e-3 of b's rms: a sum that cancels to near zero differs by many steps
/// at no cost to the output), the largest difference relative to max(|b|, rms of b), and non-finite values in `a`.
fn cmp16(a: []const u16, b: []const u16) Cmp {
    var c: Cmp = .{};
    var ss: f64 = 0;
    for (b) |y| ss += bf16Value(y) * bf16Value(y);
    const rms = @sqrt(ss / @as(f64, @floatFromInt(@max(b.len, 1))));
    for (a, b) |x, y| {
        if (x & 0x7f80 == 0x7f80) c.nonfinite += 1;
        if (x != y) {
            const d: u32 = @abs(ulpKey(x) - ulpKey(y));
            c.differ += 1;
            c.max_ulp = @max(c.max_ulp, d);
            if (@abs(bf16Value(y)) >= 1e-3 * rms) c.max_ulp_big = @max(c.max_ulp_big, d);
            c.max_rel = @max(c.max_rel, @abs(bf16Value(x) - bf16Value(y)) / @max(@abs(bf16Value(y)), rms));
        }
    }
    return c;
}

/// The projections' path on up to 16 rows: w13, the capped SwiGLU, w2 and the weighted sum (every pick 1/8) as separate
/// dispatches, with its own buffers.
const XdProj = struct {
    pipes: exl3.Pipelines,
    swiglu: mtl.Pipeline,
    combine: mtl.Pipeline,
    scratch: mtl.Buffer,
    gu: mtl.Buffer,
    act: mtl.Buffer,
    y: mtl.Buffer,
    moe: mtl.Buffer,

    fn init(device: mtl.Device, xd: *const Xd) !XdProj {
        const slots = Xd.max_slots;
        return .{
            .pipes = try exl3.Pipelines.load(device),
            .swiglu = try mtl.Pipeline.init(device, xd.lib, "ik_swiglu", false),
            .combine = try mtl.Pipeline.init(device, xd.lib, "ik_combine", false),
            .scratch = try device.buffer(try exl3.scratchBytes(xd_hidden, 2 * xd_inner, slots), mtl.ResourceOptions.shared),
            .gu = try device.buffer(slots * 2 * xd_inner * 2, mtl.ResourceOptions.shared),
            .act = try device.buffer(slots * xd_inner * 2, mtl.ResourceOptions.shared),
            .y = try device.buffer(slots * xd_hidden * 2, mtl.ResourceOptions.shared),
            .moe = try device.buffer(16 * xd_hidden * 2, mtl.ResourceOptions.shared),
        };
    }

    fn deinit(p: *XdProj) void {
        p.pipes.deinit();
        p.swiglu.deinit();
        p.combine.deinit();
        for ([_]mtl.Buffer{ p.scratch, p.gu, p.act, p.y, p.moe }) |b| b.deinit();
    }

    /// w2 of `slots` picks (`ids` from byte offset `io`) over the act rows of `act` into `out`.
    fn w2(p: *const XdProj, e: mtl.ComputeEncoder, xd: *const Xd, act: mtl.Buffer, out: mtl.Buffer, ids: mtl.Buffer, io: usize, slots: usize) !void {
        const table: exl3.View = .{ .buffer = xd.table, .offset = xd_experts * @sizeOf(exl3.ExpertDescriptor) };
        try exl3.encodeIndexed(&p.pipes, e, .{ .descriptors = table, .count = xd_experts, .k = xd_inner, .n = xd_hidden }, .{ .buffer = act }, .{ .buffer = out }, .{ .buffer = p.scratch }, .{ .count = slots, .x_stride = xd_inner, .y_stride = xd_hidden, .routes = .{ .ids = .{ .buffer = ids, .offset = io } } });
    }

    /// `rows` rows of `x` (from byte offset `xo`) with picks `ids` (from `io`) into gu, act, y and moe.
    fn encode(p: *const XdProj, e: mtl.ComputeEncoder, xd: *const Xd, x: mtl.Buffer, xo: usize, ids: mtl.Buffer, io: usize, rows: usize) !void {
        const S = rows * 8;
        const table: exl3.View = .{ .buffer = xd.table };
        try exl3.encodeIndexed(&p.pipes, e, .{ .descriptors = table, .count = xd_experts, .k = xd_hidden, .n = 2 * xd_inner }, .{ .buffer = x, .offset = xo }, .{ .buffer = p.gu }, .{ .buffer = p.scratch }, .{ .count = S, .x_stride = xd_hidden, .y_stride = 2 * xd_inner, .routes = .{ .ids = .{ .buffer = ids, .offset = io }, .repeat = 8 } });
        e.setPipeline(p.swiglu);
        e.setValue(ik.forward.Args{ .rows = @intCast(S), .width = xd_inner, .cap = 4, .thr = 25 }, 0);
        e.setBuffer(p.gu, 0, 1);
        e.setBuffer(p.act, 0, 2);
        e.dispatchThreads(mtl.Size.of(xd_inner, S, 1), mtl.Size.of(256, 1, 1));
        try p.w2(e, xd, p.act, p.y, ids, io, S);
        e.setPipeline(p.combine);
        e.setValue(ik.forward.Args{ .rows = @intCast(rows), .width = xd_hidden, .queries = 8 }, 0);
        e.setBuffer(p.y, 0, 1);
        e.setBuffer(xd.ew, 0, 2);
        e.setBuffer(p.moe, 0, 3);
        e.dispatchThreads(mtl.Size.of(xd_hidden, rows, 1), mtl.Size.of(256, 1, 1));
    }
};

/// Call 0 of `win` through the chain and through the projections' path: each bf16 intermediate and the output against
/// each other, and the projections' w2 on the chain's own act against the chain's w2 (the two w2 legs on one input), one
/// JSON line.
fn xdCheck(device: mtl.Device, q: mtl.Queue, xd: *const Xd, ck: *const ik.weights.Checkpoint, win: XdWindows, layer: usize) !void {
    const H = xd_hidden;
    const I = xd_inner;
    const rows = win.rows;
    const S = rows * 8;
    var proj = try XdProj.init(device, xd);
    defer proj.deinit();
    const y_same = try device.buffer(S * H * 2, mtl.ResourceOptions.shared);
    defer y_same.deinit();
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    e.useResource(xd.table, mtl.ResourceUsage.read);
    for (ck.shards.items) |sh| e.useResource(sh.buffer, mtl.ResourceUsage.read);
    xd.encode(e, win.x, 0, win.ids, 0, rows, .{ .dbg = true });
    try proj.encode(e, xd, win.x, 0, win.ids, 0, rows);
    try proj.w2(e, xd, xd.act, y_same, win.ids, 0, S);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("xd check: {s}", .{msg});
        return error.GpuFailed;
    }
    const g = cmp16(xd.gu.slice(u16, S * H), proj.gu.slice(u16, S * H));
    const a = cmp16(xd.act.slice(u16, S * I), proj.act.slice(u16, S * I));
    const k = cmp16(xd.ey.slice(u16, S * H), proj.y.slice(u16, S * H));
    const k2 = cmp16(xd.ey.slice(u16, S * H), y_same.slice(u16, S * H));
    const m = cmp16(xd.moe.slice(u16, rows * H), proj.moe.slice(u16, rows * H));
    std.debug.print("{{\"xd_check\": {{\"layer\": {d}, \"rows\": {d}, ", .{ layer, rows });
    printCmp("gu", g, S * H);
    printCmp("act", a, S * I);
    printCmp("ey", k, S * H);
    printCmp("ey_same_act", k2, S * H);
    printCmp("moe", m, rows * H);
    std.debug.print("\"nonfinite\": {d}}}}}\n", .{g.nonfinite + a.nonfinite + k.nonfinite + m.nonfinite});
    // the two w2 legs' f32 sums before the output rotation, on the chain's own act: the projections' path keeps its
    // sums in the scratch after its rotated input (rounded up to 16 bytes)
    const at = (I * S * 2 + 15) / 16 * 16;
    const theirs: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, proj.scratch.slice(u8, at + S * H * 4)[at..]));
    const ours = xd.sums.slice(f32, S * H);
    var sq: f64 = 0;
    var dq: f64 = 0;
    var dmax: f64 = 0;
    var differ: usize = 0;
    for (ours, theirs) |o, t| {
        const d = @abs(@as(f64, o) - @as(f64, t));
        sq += @as(f64, t) * @as(f64, t);
        dq += d * d;
        dmax = @max(dmax, d);
        differ += @intFromBool(o != t);
    }
    const rms = @sqrt(sq / @as(f64, @floatFromInt(S * H)));
    std.debug.print("{{\"xd_sums_same_act\": {{\"layer\": {d}, \"rows\": {d}, \"differ\": {d}, \"of\": {d}, \"rel_rms\": {e:.2}, \"max_over_rms\": {e:.2}}}}}\n", .{ layer, rows, differ, S * H, @sqrt(dq / sq), dmax / rms });
}

/// One comparison as a JSON member, a comma after it.
fn printCmp(name: []const u8, c: Cmp, of: usize) void {
    std.debug.print("\"{s}\": {{\"differ\": {d}, \"max_ulp\": {d}, \"max_ulp_big\": {d}, \"max_rel\": {e:.2}, \"of\": {d}}}, ", .{ name, c.differ, c.max_ulp, c.max_ulp_big, c.max_rel, of });
}

/// The row counts of a comma list.
fn xdRows(text: []const u8, out: *[8]usize) !usize {
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |t| {
        if (n == out.len) return error.TooManyRowCounts;
        out[n] = try std.fmt.parseInt(usize, t, 10);
        if (out[n] == 0 or out[n] > 16) return error.BadRows;
        n += 1;
    }
    if (n == 0) return error.BadRows;
    return n;
}

/// The first generated row of a taps file: the prompt length less one, from the sibling tokens file; zero without it.
fn xdFirstRow(init: std.process.Init, xm_path: []const u8, T: usize) !usize {
    const suffix = ".xm.bin";
    if (!std.mem.endsWith(u8, xm_path, suffix)) return 0;
    const path = try std.fmt.allocPrint(init.gpa, "{s}.tokens.json", .{xm_path[0 .. xm_path.len - suffix.len]});
    defer init.gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(init.io, path, init.gpa, .limited(64 << 20)) catch |e| switch (e) {
        error.FileNotFound => return 0,
        else => return e,
    };
    defer init.gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, init.gpa, text, .{});
    defer parsed.deinit();
    const prompt = parsed.value.object.get("prompt") orelse return error.BadTokens;
    if (prompt != .integer or prompt.integer < 1 or prompt.integer > T) return error.BadTokens;
    return @intCast(prompt.integer - 1);
}

/// Layer `layer`'s descriptor table from the pack at `model` (the checkpoint `ck` holds its files), checked for the chain.
fn xdTable(init: std.process.Init, device: mtl.Device, model: []const u8, layer: u32, ck: *ik.weights.Checkpoint, widths: *[5]u32) !mtl.Buffer {
    var p = try ik.expert_pack.Pack.read(init.gpa, init.io, model, xd_taps_layers, xd_experts);
    defer p.deinit();
    const table = try ik.weights.expertTable(init.gpa, device, ck, model, p.layers[layer], layer, xd_hidden, xd_inner, xd_experts, widths);
    errdefer table.deinit();
    if (!ik.weights.xdOk(table.slice(exl3.ExpertDescriptor, 2 * xd_experts), xd_experts, xd_hidden, xd_inner)) return error.LayerNotXd;
    return table;
}

/// A taps file pair read whole: MoE inputs [T][2][H] bf16 and every layer's picks [T][48][8] u32.
const XdTaps = struct {
    xm_bytes: []u8,
    eids_bytes: []u8,
    T: usize,

    fn read(init: std.process.Init, xm_path: []const u8, eids_path: []const u8) !XdTaps {
        const xm_bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, xm_path, init.gpa, .unlimited);
        errdefer init.gpa.free(xm_bytes);
        const eids_bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, eids_path, init.gpa, .unlimited);
        errdefer init.gpa.free(eids_bytes);
        if (xm_bytes.len % 2 != 0 or eids_bytes.len % 4 != 0) return error.BadTaps;
        const per_row = xd_taps_layers * 8 * 4;
        if (eids_bytes.len == 0 or eids_bytes.len % per_row != 0) return error.BadEids;
        const T = eids_bytes.len / per_row;
        if (xm_bytes.len != T * 2 * xd_hidden * 2) return error.BadXm;
        return .{ .xm_bytes = xm_bytes, .eids_bytes = eids_bytes, .T = T };
    }

    fn deinit(t: XdTaps, gpa: std.mem.Allocator) void {
        gpa.free(t.xm_bytes);
        gpa.free(t.eids_bytes);
    }

    fn xm(t: XdTaps) []align(1) const u16 {
        return std.mem.bytesAsSlice(u16, t.xm_bytes);
    }

    fn picks(t: XdTaps) []align(1) const u32 {
        return std.mem.bytesAsSlice(u32, t.eids_bytes);
    }
};

/// The index of `layer` on the taps' MoE-input layer axis.
fn xdSlot(layer: u32) !usize {
    for (ik.forward.xm_layers, 0..) |l, k| {
        if (l == layer) return k;
    }
    return error.LayerNotRecorded;
}

/// tf-inkling-bench xd: a layer's experts through the decode chain on recorded rows. Per row count: the check line,
/// then (unless `check` ends the arguments) the timing line {"decode": {"layer", "rows", "us", "distinct"}}.
fn xdLeg(init: std.process.Init, args: []const [:0]const u8) !void {
    const layer = try std.fmt.parseInt(u32, args[3], 10);
    const slot = try xdSlot(layer);
    var counts: [8]usize = undefined;
    const nc = try xdRows(args[6], &counts);
    var check_only = false;
    var parts = false;
    for (args[7..]) |option| {
        if (std.mem.eql(u8, option, "check")) {
            check_only = true;
        } else if (std.mem.eql(u8, option, "parts")) {
            parts = true;
        } else return error.UnknownOption;
    }
    const device = try mtl.Device.init();
    const q = try device.queue();
    var ck = ik.weights.Checkpoint.init(init.gpa);
    defer ck.deinit();
    var widths: [5]u32 = @splat(0);
    const table = try xdTable(init, device, args[2], layer, &ck, &widths);
    defer table.deinit();
    const taps = try XdTaps.read(init, args[4], args[5]);
    defer taps.deinit(init.gpa);
    const first = try xdFirstRow(init, args[4], taps.T);
    var xdc = try Xd.init(device, table);
    defer xdc.deinit();
    std.debug.print("{{\"xd\": {{\"layer\": {d}, \"k2\": {d}, \"k3\": {d}, \"rows_in_taps\": {d}, \"first_row\": {d}}}}}\n", .{ layer, widths[2], widths[3], taps.T, first });
    for (counts[0..nc]) |rows| {
        const win = try XdWindows.init(device, rows, first, taps.T, taps.xm(), taps.picks(), slot, layer);
        defer win.deinit();
        try xdCheck(device, q, &xdc, &ck, win, layer);
        if (check_only) continue;
        const us = try xdTime(q, &xdc, &ck, win, .{});
        std.debug.print("{{\"decode\": {{\"layer\": {d}, \"rows\": {d}, \"us\": {d:.1}, \"distinct\": {d:.1}}}}}\n", .{ layer, rows, us, win.distinct });
        if (parts) {
            const only = [4]Run{ .{ .mid = false, .down = false, .finish = false }, .{ .pair = false, .down = false, .finish = false }, .{ .pair = false, .mid = false, .finish = false }, .{ .pair = false, .mid = false, .down = false } };
            var each: [4]f64 = undefined;
            for (only, &each) |run, *t| t.* = try xdTime(q, &xdc, &ck, win, run);
            std.debug.print("{{\"decode_parts\": {{\"layer\": {d}, \"rows\": {d}, \"pair_us\": {d:.1}, \"mid_us\": {d:.1}, \"down_us\": {d:.1}, \"finish_us\": {d:.1}}}}}\n", .{ layer, rows, each[0], each[1], each[2], each[3] });
        }
    }
}

/// One command buffer of the dispatches `run` selects, waited.
fn xdRun(q: mtl.Queue, xd: *const Xd, ck: *const ik.weights.Checkpoint, x: mtl.Buffer, xo: usize, ids: mtl.Buffer, io: usize, rows: usize, run: Run) !void {
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    e.useResource(xd.table, mtl.ResourceUsage.read);
    for (ck.shards.items) |sh| e.useResource(sh.buffer, mtl.ResourceUsage.read);
    xd.encode(e, x, xo, ids, io, rows, run);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("xd dump: {s}", .{msg});
        return error.GpuFailed;
    }
}

fn writeU16(init: std.process.Init, prefix: []const u8, name: []const u8, data: []const u16) !void {
    const path = try std.fmt.allocPrint(init.gpa, "{s}.{s}.bin", .{ prefix, name });
    defer init.gpa.free(path);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = std.mem.sliceAsBytes(data) });
}

/// tf-inkling-bench xd-dump: `rows` recorded rows from `row0` through the decode chain in windows of 16 (every window
/// size gives the same bits), each pick weighing 1/8. Writes bf16 files `<out>.<name>.bin`: gu [slots][4608] (w13's
/// output), y [slots][4608] (w2's per-pick output on the act in `act.bin`, bf16 [slots][2304], the input the
/// projections' path's w2 is measured on), y_own (the same on the chain's own act), moe, moe_own [rows][4608] (the
/// weighted sums of y and y_own), and proj_y_own, proj_moe_own: the projections' path from the same rows, on its own act.
fn xdDumpLeg(init: std.process.Init, args: []const [:0]const u8) !void {
    const H = xd_hidden;
    const I = xd_inner;
    const layer = try std.fmt.parseInt(u32, args[3], 10);
    const slot = try xdSlot(layer);
    const rows = try std.fmt.parseInt(usize, args[6], 10);
    const row0 = try std.fmt.parseInt(usize, args[7], 10);
    const S = rows * 8;
    const device = try mtl.Device.init();
    const q = try device.queue();
    var ck = ik.weights.Checkpoint.init(init.gpa);
    defer ck.deinit();
    var widths: [5]u32 = @splat(0);
    const table = try xdTable(init, device, args[2], layer, &ck, &widths);
    defer table.deinit();
    const taps = try XdTaps.read(init, args[4], args[5]);
    defer taps.deinit(init.gpa);
    if (rows == 0 or row0 + rows > taps.T) return error.RowsOutsideTaps;
    const act_bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[8], init.gpa, .unlimited);
    defer init.gpa.free(act_bytes);
    if (act_bytes.len != S * I * 2) return error.ActSize;
    const act_ref = std.mem.bytesAsSlice(u16, act_bytes);
    const xs = try device.buffer(rows * H * 2, mtl.ResourceOptions.shared);
    defer xs.deinit();
    const ids = try device.buffer(S * 4, mtl.ResourceOptions.shared);
    defer ids.deinit();
    for (0..rows) |r| {
        @memcpy(xs.slice(u16, rows * H)[r * H ..][0..H], taps.xm()[((row0 + r) * 2 + slot) * H ..][0..H]);
        for (taps.picks()[((row0 + r) * xd_taps_layers + layer) * 8 ..][0..8], 0..) |e, j| {
            if (e >= xd_experts) return error.BadExpertId;
            ids.slice(u32, S)[r * 8 + j] = e;
        }
    }
    const gu = try init.gpa.alloc(u16, S * H);
    defer init.gpa.free(gu);
    const y = try init.gpa.alloc(u16, S * H);
    defer init.gpa.free(y);
    const y_own = try init.gpa.alloc(u16, S * H);
    defer init.gpa.free(y_own);
    const moe = try init.gpa.alloc(u16, rows * H);
    defer init.gpa.free(moe);
    const moe_own = try init.gpa.alloc(u16, rows * H);
    defer init.gpa.free(moe_own);
    var xdc = try Xd.init(device, table);
    defer xdc.deinit();
    var proj = try XdProj.init(device, &xdc);
    defer proj.deinit();
    const proj_y = try init.gpa.alloc(u16, S * H);
    defer init.gpa.free(proj_y);
    const proj_moe = try init.gpa.alloc(u16, rows * H);
    defer init.gpa.free(proj_moe);
    var w0: usize = 0;
    while (w0 < rows) : (w0 += 16) {
        const n: usize = @min(16, rows - w0);
        const s0 = w0 * 8;
        const sn = n * 8;
        const xo = w0 * H * 2;
        const io = s0 * 4;
        try xdRun(q, &xdc, &ck, xs, xo, ids, io, n, .{ .down = false, .finish = false, .dbg = true });
        @memcpy(gu[s0 * H ..][0 .. sn * H], xdc.gu.slice(u16, sn * H));
        try xdRun(q, &xdc, &ck, xs, xo, ids, io, n, .{ .pair = false, .mid = false, .dbg = true });
        @memcpy(y_own[s0 * H ..][0 .. sn * H], xdc.ey.slice(u16, sn * H));
        @memcpy(moe_own[w0 * H ..][0 .. n * H], xdc.moe.slice(u16, n * H));
        @memcpy(xdc.act.slice(u16, sn * I), act_ref[s0 * I ..][0 .. sn * I]);
        try xdRun(q, &xdc, &ck, xs, xo, ids, io, n, .{ .pair = false, .mid = false, .dbg = true });
        @memcpy(y[s0 * H ..][0 .. sn * H], xdc.ey.slice(u16, sn * H));
        @memcpy(moe[w0 * H ..][0 .. n * H], xdc.moe.slice(u16, n * H));
        {
            const cb = q.commandBuffer();
            const e = cb.compute(.serial);
            e.useResource(xdc.table, mtl.ResourceUsage.read);
            for (ck.shards.items) |sh| e.useResource(sh.buffer, mtl.ResourceUsage.read);
            try proj.encode(e, &xdc, xs, xo, ids, io, n);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.log.err("xd dump: {s}", .{msg});
                return error.GpuFailed;
            }
        }
        @memcpy(proj_y[s0 * H ..][0 .. sn * H], proj.y.slice(u16, sn * H));
        @memcpy(proj_moe[w0 * H ..][0 .. n * H], proj.moe.slice(u16, n * H));
    }
    try writeU16(init, args[9], "gu", gu);
    try writeU16(init, args[9], "y", y);
    try writeU16(init, args[9], "y_own", y_own);
    try writeU16(init, args[9], "moe", moe);
    try writeU16(init, args[9], "moe_own", moe_own);
    try writeU16(init, args[9], "proj_y_own", proj_y);
    try writeU16(init, args[9], "proj_moe_own", proj_moe);
    std.debug.print("{{\"xd_dump\": {{\"layer\": {d}, \"rows\": {d}, \"slots\": {d}, \"k2\": {d}, \"k3\": {d}}}}}\n", .{ layer, rows, S, widths[2], widths[3] });
}
