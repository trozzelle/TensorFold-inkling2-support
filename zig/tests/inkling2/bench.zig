//! Inkling 2 benches: tensor-unit bf16 GEMM at Inkling's dense shapes, the DSpark hold block as byte-equivalent bf16
//! matvecs, and the core EXL3 expert projections on recorded MoE inputs (usage).
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const tf = @import("tensorfold");
const ik = tf.inkling2;
const exl3 = @import("tensorfold").exl3_projection;

const usage =
    \\tf-inkling-bench gemm                                         tensor-unit GEMM at the dense shapes, M 512/2048/4096
    \\tf-inkling-bench draft                                        the DSpark hold block as bf16 matvecs
    \\tf-inkling-bench dense [rows...]                              48 layers' dense matvecs (default rows 1 2 4 6 8 16), per shape at 1 and 8 rows, unembed at 1 row
    \\tf-inkling-bench experts <model> <layer> <xm.bin> <eids.bin> <rows> <row0> <act.bin> <out>
    \\                                                              w13 and w2 on recorded MoE inputs, bf16 out
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
    if (args.len == 10 and std.mem.eql(u8, args[1], "experts")) return experts(init, args);
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
