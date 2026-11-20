//! Inkling 2 convolution kernel checks: static and dynamic taps against FP64, the ring, and chunked runs against one.
const std = @import("std");
const mtl = @import("metal");
const h = @import("harness.zig");

const C = 1024;
const rows = 6;
const pos0: i64 = 37;

/// Input at position q: this chunk's row from pos0 on, the ring's before it, zero before position 0.
fn src(x: []const u16, ringv: []const u16, c: usize, q: i64) f64 {
    if (q < 0) return 0;
    if (q >= pos0) return h.fb(x[@as(usize, @intCast(q - pos0)) * C + c]);
    return h.fb(ringv[@as(usize, @intCast(@mod(q, 32))) * C + c]);
}

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    const x = try g.buf(rows * C * 2);
    const ringb = try g.buf(32 * C * 2);
    const w = try g.buf(C * 4 * 2);
    const y = try g.buf(rows * C * 2);
    const taps = try g.buf(4 * C * 2);
    const dyn = try g.buf(rows * 8 * 4 * 2);
    defer for ([_]mtl.Buffer{ x, ringb, w, y, taps, dyn }) |b| b.deinit();
    const xs = h.fill16(x, rows * C, r, 3);
    const rs = h.fill16(ringb, 32 * C, r, 3);
    const ws = h.fill16(w, C * 4, r, 0.7);
    const grid: [3]usize = .{ C, rows, 1 };
    const group: [3]usize = .{ 256, 1, 1 };
    var c: h.Check = .{ .name = "conv static" };
    const a = h.Args{ .rows = rows, .width = C, .pos = @intCast(pos0), .scale = @sqrt(5.0) };
    try g.run("ik_conv", a, &.{ .{ .buf = x }, .{ .buf = ringb }, .{ .buf = w }, .{ .buf = y } }, grid, group, false);
    for (0..rows) |t| for (0..C) |ch| {
        var s: f64 = 0;
        for (0..4) |j| s += h.fb(ws[ch * 4 + j]) * src(xs, rs, ch, pos0 + @as(i64, @intCast(t)) - 3 + @as(i64, @intCast(j)));
        c.b16(y.slice(u16, rows * C)[t * C + ch], (h.rb(s) + h.fb(xs[t * C + ch])) / @sqrt(5.0));
    };
    try c.done();
    try chunks(g, "ik_conv", a, &.{ .{ .buf = x }, .{ .buf = ringb }, .{ .buf = w } }, 0, x, ringb, y);

    // dynamic: taps [4, C] plus each row's projection [8 heads, 4 lags]
    const ts = h.fill16(taps, 4 * C, r, 0.5);
    const ds = h.fill16(dyn, rows * 32, r, 0.5);
    var cd: h.Check = .{ .name = "conv dynamic" };
    const ad = h.Args{ .rows = rows, .width = C, .pos = @intCast(pos0), .heads = 8, .scale = 3 };
    try g.run("ik_conv_dyn", ad, &.{ .{ .buf = x }, .{ .buf = ringb }, .{ .buf = taps }, .{ .buf = dyn }, .{ .buf = y } }, grid, group, false);
    for (0..rows) |t| for (0..C) |ch| {
        const hh = ch / (C / 8);
        var s: f64 = 0;
        for (0..4) |lag| s += src(xs, rs, ch, pos0 + @as(i64, @intCast(t)) - @as(i64, @intCast(lag))) * (h.fb(ts[lag * C + ch]) + h.fb(ds[t * 32 + hh * 4 + lag]));
        cd.b16(y.slice(u16, rows * C)[t * C + ch], (s + h.fb(xs[t * C + ch])) / 3);
    };
    try cd.done();
    try chunks(g, "ik_conv_dyn", ad, &.{ .{ .buf = x }, .{ .buf = ringb }, .{ .buf = taps } }, 32 * 2, dyn, ringb, y);

    // the ring: the last min(rows, 32) inputs at their positions mod 32, the rest untouched
    const before = try std.heap.page_allocator.dupe(u16, rs);
    defer std.heap.page_allocator.free(before);
    try g.run("ik_ring", a, &.{ .{ .buf = x }, .{ .buf = ringb } }, grid, group, false);
    var cr: h.Check = .{ .name = "ring" };
    for (0..32) |slot| for (0..C) |ch| {
        var want = before[slot * C + ch];
        for (0..rows) |t| if (@mod(pos0 + @as(i64, @intCast(t)), 32) == slot) {
            want = xs[t * C + ch];
        };
        cr.n += 1;
        if (ringb.slice(u16, 32 * C)[slot * C + ch] != want) cr.bad += 1;
    };
    try cr.done();
}

/// A convolution's chunks give a one-dispatch run's bits: rows 0..2, `ik_ring`, then rows 3..5 at pos0 + 3 against the
/// 6-row output `whole` (still in `y`). `lead` binds x, ring and the taps; `side` is the per-row projection when the
/// convolution is dynamic (`side_row` bytes a row, else 0 and `x` itself).
fn chunks(g: *const h.Gpu, name: [:0]const u8, a: h.Args, lead: []const h.Bind, side_row: usize, side: mtl.Buffer, ringb: mtl.Buffer, whole: mtl.Buffer) !void {
    const dynamic = side_row != 0;
    const ring2 = try g.buf(32 * C * 2);
    const out = try g.buf(rows * C * 2);
    defer ring2.deinit();
    defer out.deinit();
    @memcpy(ring2.slice(u8, 32 * C * 2), ringb.slice(u8, 32 * C * 2));
    const x = lead[0].buf;
    const taps = lead[2].buf;
    const half = rows / 2;
    var c: h.Check = .{ .name = if (dynamic) "conv dynamic chunks" else "conv chunks" };
    const wide: [3]usize = .{ C, half, 1 };
    const group: [3]usize = .{ 256, 1, 1 };
    var first = a;
    first.rows = half;
    var second = a;
    second.rows = rows - half;
    second.pos = a.pos + half;
    const tail = half * C * 2;
    if (dynamic) {
        try g.run(name, first, &.{ .{ .buf = x }, .{ .buf = ring2 }, .{ .buf = taps }, .{ .buf = side }, .{ .buf = out } }, wide, group, false);
        try g.run("ik_ring", first, &.{ .{ .buf = x }, .{ .buf = ring2 } }, wide, group, false);
        try g.run(name, second, &.{ .{ .buf = x, .off = tail }, .{ .buf = ring2 }, .{ .buf = taps }, .{ .buf = side, .off = half * side_row }, .{ .buf = out, .off = tail } }, wide, group, false);
    } else {
        try g.run(name, first, &.{ .{ .buf = x }, .{ .buf = ring2 }, .{ .buf = taps }, .{ .buf = out } }, wide, group, false);
        try g.run("ik_ring", first, &.{ .{ .buf = x }, .{ .buf = ring2 } }, wide, group, false);
        try g.run(name, second, &.{ .{ .buf = x, .off = tail }, .{ .buf = ring2 }, .{ .buf = taps }, .{ .buf = out, .off = tail } }, wide, group, false);
    }
    c.same(out.slice(u8, rows * C * 2), whole.slice(u8, rows * C * 2));
    try c.done();
}
