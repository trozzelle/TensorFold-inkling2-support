//! Inkling 2 kernel checks: split-K decode attention against an FP64 reference, a window's rows against the same rows
//! run one at a time, and a row's blocks against the same row with empty blocks around them.
const std = @import("std");
const mtl = @import("metal");
const h = @import("harness.zig");

const Hq = 48;
const KV = 8;
const D = 128;
const E = 1024; // relative-bias extent, also the local layers' window
const KEYS = 256; // keys in one block
const PART_SIMDGROUPS = 24; // the partial kernel's threadgroup: 8 key groups of 32 keys, 3 pairs of q heads
const SLOT = D + 2; // floats in one (row, head, block) partial: m, l, acc[D]
const ROWS = 16; // a decode window's rows
const CAP = 3200; // cached keys; the deepest window's last row is at 3078
const MAX_BLOCKS = (CAP + KEYS - 1) / KEYS + 3; // the cache's blocks and three empty ones
const OUT_ROW = Hq * D * 2; // bytes of one row's attention output

/// Random inputs for ROWS rows over a CAP-key cache, and the buffers the two kernels write. Scores spread over several
/// units, so blocks differ in their maxima and the merge's rescaling matters.
const Inputs = struct {
    qn: mtl.Buffer,
    keys: mtl.Buffer,
    vals: mtl.Buffer,
    pb: mtl.Buffer,
    sink: mtl.Buffer,
    gate: mtl.Buffer,
    att: mtl.Buffer,
    part: mtl.Buffer,
    qv: []u16,
    kv: []u16,
    vv: []u16,
    pbv: []u16,
    sv: []u16,
    gv: []u16,

    fn init(g: *const h.Gpu, r: std.Random) !Inputs {
        const qn = try g.buf(ROWS * Hq * D * 2);
        const keys = try g.buf(CAP * KV * D * 2);
        const vals = try g.buf(CAP * KV * D * 2);
        const pb = try g.buf(ROWS * Hq * E * 2);
        const sink = try g.buf(ROWS * Hq * 2);
        const gate = try g.buf(ROWS * Hq * D * 2);
        const att = try g.buf(ROWS * OUT_ROW);
        const part = try g.buf(ROWS * Hq * MAX_BLOCKS * SLOT * 4);
        return .{
            .qn = qn,
            .keys = keys,
            .vals = vals,
            .pb = pb,
            .sink = sink,
            .gate = gate,
            .att = att,
            .part = part,
            .qv = h.fill16(qn, ROWS * Hq * D, r, 2.5),
            .kv = h.fill16(keys, CAP * KV * D, r, 2.5),
            .vv = h.fill16(vals, CAP * KV * D, r, 1.0),
            .pbv = h.fill16(pb, ROWS * Hq * E, r, 2.0),
            .sv = h.fill16(sink, ROWS * Hq, r, 2.0),
            .gv = h.fill16(gate, ROWS * Hq * D, r, 3.0),
        };
    }

    fn deinit(x: Inputs) void {
        for ([_]mtl.Buffer{ x.qn, x.keys, x.vals, x.pb, x.sink, x.gate, x.att, x.part }) |b| b.deinit();
    }

    /// The attention bytes of rows `first` .. `first + n`.
    fn out(x: Inputs, first: usize, n: usize) []u8 {
        return x.att.slice(u8, ROWS * OUT_ROW)[first * OUT_ROW ..][0 .. n * OUT_ROW];
    }
};

/// The first block and the block count that cover rows pos .. pos + rows - 1: from the block of the first row's
/// earliest visible key to the block of the last row's position.
fn blocksOf(pos: u32, rows: u32, window: u32) struct { f0: u32, n: u32 } {
    const lo = if (window > 0 and pos + 1 > window) pos + 1 - window else 0;
    const f0 = lo / KEYS;
    return .{ .f0 = f0, .n = (pos + rows - 1) / KEYS - f0 + 1 };
}

/// ik_attn_part, then ik_attn_merge, over `rows` rows that start at input row `row` and position `pos`, with blocks
/// f0 .. f0 + n. The partials and the output start as NaN, so anything the kernels leave unwritten shows.
fn attend(g: *const h.Gpu, x: Inputs, row: usize, rows: u32, pos: u32, window: u32, f0: u32, n: u32) !void {
    @memset(x.part.slice(f32, rows * Hq * n * SLOT), std.math.nan(f32));
    @memset(x.out(row, rows), 0xff);
    const a: h.Args = .{ .rows = rows, .heads = Hq, .kv = KV, .dim = D, .pos = pos, .window = window, .extent = E, .blocks = n, .queries = f0, .scale = @floatCast(1 / @sqrt(@as(f64, D))) };
    try g.run("ik_attn_part", a, &.{ .{ .buf = x.qn, .off = row * Hq * D * 2 }, .{ .buf = x.keys }, .{ .buf = x.vals }, .{ .buf = x.pb, .off = row * Hq * E * 2 }, .{ .buf = x.part } }, .{ KV, rows, n }, .{ 32, PART_SIMDGROUPS, 1 }, true);
    try g.run("ik_attn_merge", a, &.{ .{ .buf = x.part }, .{ .buf = x.sink, .off = row * Hq * 2 }, .{ .buf = x.gate, .off = row * Hq * D * 2 }, .{ .buf = x.att, .off = row * OUT_ROW } }, .{ Hq, rows, 1 }, .{ 32, 1, 1 }, true);
}

/// A run's output must be all written and finite, so equal bytes below cannot be equal leftovers.
fn finite(c: *h.Check, bytes: []const u8) void {
    c.n += 1;
    for (std.mem.bytesAsSlice(u16, bytes)) |v| {
        if (std.math.isNan(h.fb(v))) {
            c.bad += 1;
            return;
        }
    }
}

/// FP64 softmax over the sink logit and the visible keys, with the relative bias and the output gate, as the one-pass
/// attention case: positions across block edges and the bias extent, for one row and for a full window.
fn caseSplitRef(g: *const h.Gpu, r: std.Random) !void {
    const x = try Inputs.init(g, r);
    defer x.deinit();
    const alloc = std.heap.page_allocator;
    const qf = try alloc.alloc(f64, x.qv.len);
    const kf = try alloc.alloc(f64, x.kv.len);
    const vf = try alloc.alloc(f64, x.vv.len);
    defer for ([_][]f64{ qf, kf, vf }) |s| alloc.free(s);
    for (qf, x.qv) |*o, v| o.* = h.fb(v);
    for (kf, x.kv) |*o, v| o.* = h.fb(v);
    for (vf, x.vv) |*o, v| o.* = h.fb(v);
    const scale = 1 / @sqrt(@as(f64, D));
    var logits: [CAP]f64 = undefined;
    for ([_]u32{ 1024, 0 }) |window| {
        var c: h.Check = .{ .name = if (window > 0) "split attention local" else "split attention global" };
        for ([_]u32{ 0, 255, 256, 700, 1500, 3000 }) |pos| for ([_]u32{ 1, ROWS }) |rows| {
            const bad = c.bad;
            const grid = blocksOf(pos, rows, window);
            try attend(g, x, 0, rows, pos, window, grid.f0, grid.n);
            const att = x.att.slice(u16, ROWS * Hq * D);
            for (0..rows) |t| for (0..Hq) |hd| {
                const p = pos + t;
                const lo = if (window > 0 and p + 1 > window) p + 1 - window else 0;
                const kvh = hd / (Hq / KV);
                const qh = qf[(t * Hq + hd) * D ..][0..D];
                const sink = h.fb(x.sv[t * Hq + hd]);
                var m = sink;
                for (lo..p + 1) |j| {
                    const kr = kf[(j * KV + kvh) * D ..][0..D];
                    var s: f64 = 0;
                    for (qh, kr) |qd, kd| s += qd * kd;
                    s *= scale;
                    if (p - j < E) s += h.fb(x.pbv[(t * Hq + hd) * E + p - j]);
                    logits[j - lo] = s;
                    m = @max(m, s);
                }
                var z = @exp(sink - m);
                var o: [D]f64 = @splat(0);
                for (lo..p + 1) |j| {
                    const w = @exp(logits[j - lo] - m);
                    z += w;
                    const vr = vf[(j * KV + kvh) * D ..][0..D];
                    for (&o, vr) |*od, vd| od.* += w * vd;
                }
                for (0..D) |d| {
                    const gt = h.fb(x.gv[(t * Hq + hd) * D + d]);
                    // two rounding points (o, then o times the gate), as the one-pass attention's check: a flip at the
                    // first can move the second by two steps, and outputs near zero keep the f32 sums' absolute error
                    c.b16a(att[(t * Hq + hd) * D + d], h.rb(o[d] / z) / (1 + @exp(-gt)), 2, 1e-6);
                }
            };
            if (c.bad != bad) std.debug.print("  failing: pos {d}, {d} rows\n", .{ pos, rows });
        };
        try c.done();
    }
}

/// A window's rows equal the same rows run one at a time (each with its own first block and block count) in bytes, over
/// windows that start and end near block edges and the bias extent; a NaN in row 1's query stays in row 1.
fn caseSplitWindow(g: *const h.Gpu, r: std.Random) !void {
    const x = try Inputs.init(g, r);
    defer x.deinit();
    const saved = try std.heap.page_allocator.alloc(u8, ROWS * OUT_ROW);
    defer std.heap.page_allocator.free(saved);
    for ([_]u32{ 1024, 0 }) |window| {
        var c: h.Check = .{ .name = if (window > 0) "split window local" else "split window global" };
        for ([_]u32{ 0, 250, 1020, 1270, 2040, 3063 }) |pos| {
            const bad = c.bad;
            const grid = blocksOf(pos, ROWS, window);
            try attend(g, x, 0, ROWS, pos, window, grid.f0, grid.n);
            @memcpy(saved, x.out(0, ROWS));
            finite(&c, saved);
            for (0..ROWS) |t| {
                const one = blocksOf(pos + @as(u32, @intCast(t)), 1, window);
                try attend(g, x, t, 1, pos + @as(u32, @intCast(t)), window, one.f0, one.n);
                finite(&c, x.out(t, 1));
                c.same(x.out(t, 1), saved[t * OUT_ROW ..][0..OUT_ROW]);
            }
            const nan_at = Hq * D;
            const kept = x.qv[nan_at];
            x.qv[nan_at] = 0x7fc0;
            try attend(g, x, 0, ROWS, pos, window, grid.f0, grid.n);
            x.qv[nan_at] = kept;
            for (0..ROWS) |t| if (t != 1) c.same(x.out(t, 1), saved[t * OUT_ROW ..][0..OUT_ROW]);
            if (c.bad != bad) std.debug.print("  failing: pos {d}\n", .{pos});
        }
        try c.done();
    }
}

/// Blocks a row does not see change nothing: the same row with three empty blocks before its own, after them, or both
/// (the partials start as NaN, so an empty block must write its own marker) gives the plain run's bytes.
fn caseSplitBlocks(g: *const h.Gpu, r: std.Random) !void {
    const x = try Inputs.init(g, r);
    defer x.deinit();
    var plain: [OUT_ROW]u8 = undefined;
    for ([_]u32{ 1024, 0 }) |window| {
        var c: h.Check = .{ .name = if (window > 0) "split blocks local" else "split blocks global" };
        for ([_]u32{ 700, 3000 }) |pos| {
            const grid = blocksOf(pos, 1, window);
            try attend(g, x, 0, 1, pos, window, grid.f0, grid.n);
            @memcpy(&plain, x.out(0, 1));
            finite(&c, &plain);
            for ([_][2]u32{ .{ 3, 0 }, .{ 0, 3 }, .{ 3, 3 } }) |extra| {
                if (grid.f0 < extra[0]) continue; // a global row starts at block 0: nothing lies before it
                try attend(g, x, 0, 1, pos, window, grid.f0 - extra[0], grid.n + extra[0] + extra[1]);
                c.same(x.out(0, 1), &plain);
            }
        }
        try c.done();
    }
}

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    try caseSplitRef(g, r);
    try caseSplitWindow(g, r);
    try caseSplitBlocks(g, r);
}
