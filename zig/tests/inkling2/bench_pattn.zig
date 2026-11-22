//! tf-inkling-bench's `pattn` mode: the prompt attention at a chunk's rows (4,096 by default) on local and global layers, at
//! chunk starts 0 and `rows`, on synthetic inputs: the engine's dispatch (the relative-bias table built and read in pieces of
//! 4,096, 1,024, 512 and 256 rows), the bias formed in the kernel instead, and the table built on the tensor units (whose
//! bits differ). Each figure is the best of nine timed command buffers of `reps` calls after one that is not counted
//! (`layer_ms`; another process's work on the GPU only adds time), and their median (`median_ms`): GPU milliseconds a layer.
const std = @import("std");
const mtl = @import("metal");
const cases = @import("cases_pattn.zig");
const tf = @import("tensorfold");
const ap = tf.inkling2.attention_prefill;

const trials = 9;
const Hq = cases.Hq;
const D = cases.D;
const E = cases.E;
const REL = cases.REL;
const KV = cases.KV;

fn fill(b: mtl.Buffer, n: usize, r: std.Random, scale: f32) void {
    for (b.slice(u16, n)) |*x| {
        const f: f32 = (@as(f32, @floatFromInt(r.int(i16))) / 32768.0) * scale;
        x.* = @intCast(@as(u32, @bitCast(f)) >> 16);
    }
}

fn median(v: []f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return v[v.len / 2];
}

/// Keys the rows p0 .. p0 + rows see over all heads, as (QK + PV) flops: 4 D a key.
fn flops(rows: usize, p0: usize, global: bool) f64 {
    var keys: f64 = 0;
    for (0..rows) |t| {
        const p = p0 + t;
        keys += @floatFromInt(if (global) p + 1 else @min(p + 1, E));
    }
    return keys * Hq * 4 * D;
}

const Bufs = struct {
    q: mtl.Buffer,
    sink: mtl.Buffer,
    reln: mtl.Buffer,
    gate: mtl.Buffer,
    out: mtl.Buffer,
    keys: mtl.Buffer,
    vals: mtl.Buffer,
    proj: mtl.Buffer,
    projt: mtl.Buffer,
    table: mtl.Buffer,

    fn init(device: mtl.Device, r: std.Random, rows: usize, cap: usize) !Bufs {
        const mk = struct {
            fn f(d: mtl.Device, bytes: usize) !mtl.Buffer {
                return d.buffer(bytes, mtl.ResourceOptions.shared);
            }
        }.f;
        const b: Bufs = .{
            .q = try mk(device, rows * Hq * D * 2),
            .sink = try mk(device, rows * Hq * 2),
            .reln = try mk(device, rows * Hq * REL * 2),
            .gate = try mk(device, rows * Hq * D * 2),
            .out = try mk(device, rows * Hq * D * 2),
            .keys = try mk(device, cap * KV * D * 2),
            .vals = try mk(device, cap * KV * D * 2),
            .proj = try mk(device, REL * E * 2),
            .projt = try mk(device, E * REL * 4),
            .table = try mk(device, ap.tableBytes(rows)),
        };
        fill(b.q, rows * Hq * D, r, 3);
        fill(b.sink, rows * Hq, r, 2);
        fill(b.reln, rows * Hq * REL, r, 2);
        fill(b.gate, rows * Hq * D, r, 3);
        fill(b.keys, cap * KV * D, r, 3);
        fill(b.vals, cap * KV * D, r, 1.7);
        fill(b.proj, REL * E, r, 1);
        const src = b.proj.slice(u16, REL * E);
        for (b.projt.slice(f32, E * REL), 0..) |*x, k| x.* = @bitCast(@as(u32, src[(k % REL) * E + k / REL]) << 16);
        return b;
    }

    fn deinit(b: Bufs) void {
        for ([_]mtl.Buffer{ b.q, b.sink, b.reln, b.gate, b.out, b.keys, b.vals, b.proj, b.projt, b.table }) |x| x.deinit();
    }

    fn io(b: Bufs) cases.Io {
        return .{ .q = .{ .buf = b.q }, .k = .{ .buf = b.keys }, .v = .{ .buf = b.vals }, .sink = .{ .buf = b.sink }, .gate = .{ .buf = b.gate }, .out = .{ .buf = b.out } };
    }
};

/// What one timed call encodes: the engine's dispatch in pieces of `chunk` rows; the formed-bias kernel; the table build
/// alone, the attention reading a built table alone; or the tensor-unit table build and the attention reading it.
const Mode = union(enum) { chunk: usize, formed, build, attn, mma };

fn encodeMode(e: mtl.ComputeEncoder, p: *const cases.Pipes, b: Bufs, mode: Mode, a: cases.Args) void {
    const reln: ap.B = .{ .buf = b.reln };
    const proj: ap.B = .{ .buf = b.proj };
    const table: ap.B = .{ .buf = b.table };
    switch (mode) {
        .chunk => |piece| ap.encodeChunk(e, &p.engine, a, b.io(), reln, proj, table, piece),
        .formed => cases.encodeFormed(e, p.formed, a, b.io(), reln, .{ .buf = b.projt }),
        .build => ap.encodeBias(e, p.engine.bias, a, reln, proj, table),
        .attn => ap.encodeAttn(e, p.engine.attn, a, b.io(), table),
        .mma => {
            cases.encodeBiasMma(e, p.bias_mma, a, reln, proj, table);
            ap.encodeAttn(e, p.engine.attn, a, b.io(), table);
        },
    }
}

const Ms = struct { best: f64, median: f64 };

fn gpuMs(q: mtl.Queue, p: *const cases.Pipes, b: Bufs, mode: Mode, a: cases.Args, reps: usize) !Ms {
    var ms: [trials]f64 = undefined;
    for (0..trials + 1) |t| {
        const cb = q.commandBuffer();
        const e = cb.compute(.serial);
        for (0..reps) |_| encodeMode(e, p, b, mode, a);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("bench: {s}", .{msg});
            return error.GpuFailed;
        }
        if (t > 0) ms[t - 1] = (cb.gpuEnd() - cb.gpuStart()) * 1e3 / @as(f64, @floatFromInt(reps));
    }
    var best = ms[0];
    for (ms) |x| best = @min(best, x);
    return .{ .best = best, .median = median(&ms) };
}

const Form = struct { name: []const u8, mode: Mode, split: bool = false };

pub fn run(init: std.process.Init, args: []const [:0]const u8) !void {
    const rows: usize = if (args.len >= 1) try std.fmt.parseInt(usize, args[0], 10) else 4096;
    const device = try mtl.Device.init();
    const q = try device.queue();
    var pipes = try cases.Pipes.init(device, init.gpa);
    defer pipes.deinit();
    var prng = std.Random.DefaultPrng.init(0xa77e);
    const r = prng.random();
    const forms = [_]Form{
        .{ .name = "table, one piece", .mode = .{ .chunk = rows }, .split = true },
        .{ .name = "table, 1024-row pieces", .mode = .{ .chunk = 1024 } },
        .{ .name = "table, 512-row pieces", .mode = .{ .chunk = 512 } },
        .{ .name = "table, 256-row pieces", .mode = .{ .chunk = 256 } },
        .{ .name = "formed in the kernel", .mode = .formed },
        .{ .name = "table built on the tensor units (bits differ)", .mode = .mma },
    };
    var total: [forms.len]f64 = @splat(0);
    for ([_]usize{ 0, rows }) |p0| {
        const cap = (p0 + rows + 63) / 64 * 64 + 64;
        const b = try Bufs.init(device, r, rows, cap);
        defer b.deinit();
        for ([_]bool{ false, true }) |global| {
            const a: cases.Args = .{ .rows = @intCast(rows), .p0 = @intCast(p0), .global = @intFromBool(global) };
            const fl = flops(rows, p0, global);
            const reps: usize = if (global and p0 > 0) 2 else 4;
            for (forms, 0..) |form, fi| {
                const layer = try gpuMs(q, &pipes, b, form.mode, a, reps);
                total[fi] += layer.best * if (global) @as(f64, 8) else 40;
                std.debug.print("{{\"pattn\": \"{s}\", \"p0\": {d}, \"rows\": {d}, \"form\": \"{s}\", \"layer_ms\": {d:.3}, \"median_ms\": {d:.3}, \"tflops\": {d:.1}", .{ if (global) "global" else "local", p0, rows, form.name, layer.best, layer.median, fl / (layer.best * 1e-3) / 1e12 });
                if (form.split) {
                    const build = try gpuMs(q, &pipes, b, .build, a, reps);
                    const attn = try gpuMs(q, &pipes, b, .attn, a, reps);
                    std.debug.print(", \"build_ms\": {d:.3}, \"attn_ms\": {d:.3}", .{ build.best, attn.best });
                }
                std.debug.print("}}\n", .{});
            }
        }
    }
    for (forms, total) |form, t| std.debug.print("{{\"pattn_48_layers\": \"{s}\", \"chunks\": 2, \"rows\": {d}, \"ms\": {d:.1}}}\n", .{ form.name, rows, t });
}
