//! Inkling 2's dense prompt GEMM against FP64 at every dense shape and row count, and its variants against each other:
//! the same row computed by the M-aligned and the M-edge variant, at any row position in a tile, in a run of any
//! length, with the N edge variant (a narrower weight) or without, gives the same bits.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const h = @import("harness.zig");
const dp = tf.inkling2.dense_prefill;

const Shape = struct { name: []const u8, n: usize, k: usize, out: dp.Out = .bf16 };
const shapes = [_]Shape{
    .{ .name = "q|gate", .n = 6144, .k = 4608 },
    .{ .name = "k|v", .n = 1024, .k = 4608 },
    .{ .name = "r", .n = 768, .k = 4608 },
    .{ .name = "sink", .n = 48, .k = 4608 },
    .{ .name = "o", .n = 4608, .k = 6144 },
    .{ .name = "router", .n = 384, .k = 4608, .out = .f32 },
    .{ .name = "dyn conv", .n = 32, .k = 4608 },
};
const row_counts = [_]usize{ 1, 7, 31, 64, 65, 513, 4096 };
/// The most rows any case runs (4,097) and the NaN rows past a run's last row.
const max_rows = 4097;
const pad_rows = 64;
const total_rows = max_rows + pad_rows;
/// Byte offset of the weight and output in their buffers: matrices of a packed buffer do not start at zero.
const byte_off = 4096;

const Run = struct {
    g: *const h.Gpu,
    pipes: *const dp.Pipes,
    x: mtl.Buffer,
    w: mtl.Buffer,
    y: mtl.Buffer,
    shape: Shape,

    /// rows `n` at x row `x0`: out rows (row stride `ldy`) from the output's start; weight rows `nn` of the matrix.
    fn go(r: Run, x0: usize, n: usize, nn: usize, ldy: usize) !void {
        const cb = r.g.queue.commandBuffer();
        const e = cb.compute(.serial);
        dp.gemmRows(e, r.pipes, .{ .buf = r.x, .off = x0 * r.shape.k * 2 }, .{ .buf = r.w, .off = byte_off, .n = nn, .k = r.shape.k }, .{ .buf = r.y, .off = byte_off }, r.shape.out, n, ldy);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("ik_gemm: {s}", .{msg});
            return error.GpuFailed;
        }
    }

    fn esz(r: Run) usize {
        return if (r.shape.out == .f32) 4 else 2;
    }
    fn outBytes(r: Run) []u8 {
        return r.y.slice(u8, byte_off + total_rows * r.shape.n * r.esz())[byte_off..];
    }
    fn fillOut(r: Run) void {
        @memset(r.y.slice(u8, byte_off + total_rows * r.shape.n * r.esz()), 0xa5);
    }
};

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    var pipes = try dp.Pipes.init(g.device, std.heap.page_allocator);
    defer pipes.deinit();
    // the simdgroup-matrix text older GPUs run builds here too; its fragment layout is theirs, so only this GPU's runs
    if (g.device.tensorUnits()) {
        var older = try dp.Pipes.initUnits(g.device, std.heap.page_allocator, false);
        older.deinit();
    }
    for (shapes) |s| try one(g, r, &pipes, s);
}

fn one(g: *const h.Gpu, r: std.Random, pipes: *const dp.Pipes, s: Shape) !void {
    const xb = try g.buf(total_rows * s.k * 2);
    const wb = try g.buf(byte_off + s.n * s.k * 2);
    const yb = try g.buf(byte_off + total_rows * s.n * 4);
    defer for ([_]mtl.Buffer{ xb, wb, yb }) |b| b.deinit();
    const xs = h.fill16(xb, total_rows * s.k, r, 1);
    const ws = h.fill16(wb, (byte_off + s.n * s.k * 2) / 2, r, 0.05)[byte_off / 2 ..];
    const run_ = Run{ .g = g, .pipes = pipes, .x = xb, .w = wb, .y = yb, .shape = s };
    try fp64(run_, r, xs, ws);
    try variants(run_);
}

/// A sorted sample of 0..limit: all of it when small, else the ends, `picks` random indices and both sides of `picks`
/// random multiples of `every` (the tile edges).
fn sample(a: std.mem.Allocator, r: std.Random, limit: usize, every: usize, picks: usize) ![]usize {
    var list: std.ArrayList(usize) = .empty;
    if (limit <= 128) {
        for (0..limit) |i| try list.append(a, i);
        return list.toOwnedSlice(a);
    }
    try list.appendSlice(a, &.{ 0, 1, limit - 2, limit - 1 });
    for (0..picks) |_| {
        try list.append(a, r.uintLessThan(usize, limit));
        const edge = every * (1 + r.uintLessThan(usize, (limit - 1) / every));
        try list.appendSlice(a, &.{ edge - 1, edge });
    }
    std.mem.sort(usize, list.items, {}, std.sort.asc(usize));
    var w: usize = 0;
    for (list.items, 0..) |v, i| if (i == 0 or v != list.items[i - 1]) {
        list.items[w] = v;
        w += 1;
    };
    list.shrinkRetainingCapacity(w);
    return list.toOwnedSlice(a);
}

fn fp64(rn: Run, r: std.Random, xs: []u16, ws: []const u16) !void {
    const s = rn.shape;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [64]u8 = undefined;
    var c: h.Check = .{ .name = try std.fmt.bufPrint(&buf, "gemm {s} vs FP64", .{s.name}) };
    const x32 = try a.alloc(f32, s.k);
    const w32 = try a.alloc(f32, s.k);
    for (row_counts) |n| {
        // rows n.. of x are NaN: a run never reads or writes past its last row
        const ps = xs[n * s.k ..][0 .. pad_rows * s.k];
        const keep = try a.dupe(u16, ps);
        @memset(ps, 0x7fc0);
        rn.fillOut();
        try rn.go(0, n, s.n, s.n);
        @memcpy(ps, keep);
        const rows = try sample(a, r, n, 64, 24);
        const cols = try sample(a, r, s.n, 128, 40);
        for (rows) |t| {
            for (0..s.k) |i| x32[i] = @floatCast(h.fb(xs[t * s.k + i]));
            for (cols) |j| {
                for (0..s.k) |i| w32[i] = @floatCast(h.fb(ws[j * s.k + i]));
                var sum: f64 = 0;
                var sq: f64 = 0;
                for (0..s.k) |i| {
                    const p = @as(f64, x32[i]) * @as(f64, w32[i]);
                    sum += p;
                    sq += p * p;
                }
                const rms = @sqrt(sq);
                if (s.out == .f32) {
                    c.near(std.mem.bytesAsSlice(f32, rn.outBytes())[t * s.n + j], sum, 1e-5, rms);
                } else {
                    c.b16a(std.mem.bytesAsSlice(u16, rn.outBytes())[t * s.n + j], sum, 1, rms * 1e-5);
                }
            }
        }
        // nothing past the last row, and nothing past the matrix's columns in a row
        const bytes = rn.outBytes();
        var touched: usize = 0;
        for (bytes[n * s.n * rn.esz() ..][0 .. 64 * s.n * rn.esz()]) |b| touched += @intFromBool(b != 0xa5);
        c.n += 1;
        c.bad += @intFromBool(touched != 0);
    }
    try c.done();
}

/// Every pair of runs below computes the same rows: their bits agree row for row (column for column over N).
fn variants(rn: Run) !void {
    const s = rn.shape;
    const a = std.heap.page_allocator;
    var buf: [64]u8 = undefined;
    var c: h.Check = .{ .name = try std.fmt.bufPrint(&buf, "gemm {s} variants", .{s.name}) };
    const rowb = s.n * rn.esz();
    // the reference: 4,097 rows (the M-edge variant), then 4,096 rows at x row 1 (M a multiple of 64)
    rn.fillOut();
    try rn.go(0, 4097, s.n, s.n);
    const edge = try a.dupe(u8, rn.outBytes()[0 .. 4097 * rowb]);
    defer a.free(edge);
    rn.fillOut();
    try rn.go(0, 4096, s.n, s.n);
    c.same(rn.outBytes()[0 .. 4096 * rowb], edge[0 .. 4096 * rowb]);
    rn.fillOut();
    try rn.go(1, 4096, s.n, s.n);
    c.same(rn.outBytes()[0 .. 4096 * rowb], edge[rowb .. 4097 * rowb]);
    const cuts = [_][2]usize{ .{ 0, 4095 }, .{ 0, 37 }, .{ 4000, 37 }, .{ 4059, 38 }, .{ 0, 1 }, .{ 1, 1 }, .{ 63, 1 }, .{ 4096, 1 }, .{ 31, 2 }, .{ 100, 64 }, .{ 33, 65 }, .{ 7, 513 } };
    for (cuts) |cut| {
        rn.fillOut();
        try rn.go(cut[0], cut[1], s.n, s.n);
        c.same(rn.outBytes()[0 .. cut[1] * rowb], edge[cut[0] * rowb ..][0 .. cut[1] * rowb]);
    }
    // the N edge variant against the aligned one: a weight one row short of a tile multiple, in a row of the same pitch
    if (s.n % 128 == 0) {
        const nn = s.n - 1;
        for ([_][2]usize{ .{ 0, 4096 }, .{ 0, 4095 }, .{ 5, 37 } }) |cut| {
            rn.fillOut();
            try rn.go(cut[0], cut[1], nn, s.n);
            const got = rn.outBytes();
            var same = true;
            for (0..cut[1]) |t| {
                same = same and std.mem.eql(u8, got[t * rowb ..][0 .. nn * rn.esz()], edge[(cut[0] + t) * rowb ..][0 .. nn * rn.esz()]);
                same = same and std.mem.allEqual(u8, got[t * rowb + nn * rn.esz() ..][0..rn.esz()], 0xa5);
            }
            c.n += 1;
            c.bad += @intFromBool(!same);
        }
    }
    try c.done();
}
