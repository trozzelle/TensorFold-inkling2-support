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

    for ([_]Kind{ .static, .dynamic, .mlp }) |kind| for ([_][2]u32{ .{ 1, 37 }, .{ 6, 37 }, .{ 8, 29 }, .{ 16, 60 }, .{ 16, 3 }, .{ 16, 0 } }) |cs| try fusedCase(g, r, kind, cs[0], cs[1]);
}

/// The decode path's fused kernels: `static` is ik_conv_ring, `dynamic` ik_conv_dyn_ring, `mlp` ik_conv_out.
const Kind = enum { static, dynamic, mlp };
const fused_rows = 16;

const Setup = struct { x: mtl.Buffer, w: mtl.Buffer, taps: mtl.Buffer, dyn: mtl.Buffer, res: mtl.Buffer, ym: mtl.Buffer };

/// One chain over rows t0.. of the setup's buffers (`a.rows` rows at `a.pos`): the fused kernel, or the kernels it
/// replaces (ik_conv or ik_conv_dyn, ik_ring, and for the MLP branch ik_out) into `y`.
fn exec(g: *const h.Gpu, kind: Kind, fused: bool, a: h.Args, s: Setup, ring: mtl.Buffer, y: mtl.Buffer, t0: usize) !void {
    const grid: [3]usize = .{ C, a.rows, 1 };
    const group: [3]usize = .{ 256, 1, 1 };
    const xb: h.Bind = .{ .buf = s.x, .off = t0 * C * 2 };
    const rb: h.Bind = .{ .buf = ring };
    const yb: h.Bind = .{ .buf = y, .off = t0 * C * 2 };
    const db: h.Bind = .{ .buf = s.dyn, .off = t0 * 32 * 2 };
    const sb: h.Bind = .{ .buf = s.res, .off = t0 * C * 4 };
    const mb: h.Bind = .{ .buf = s.ym, .off = t0 * C * 2 };
    const wb: h.Bind = .{ .buf = s.w };
    const tb: h.Bind = .{ .buf = s.taps };
    const keep: [3]usize = .{ C, @min(a.rows, 32), 1 };
    switch (kind) {
        .static => if (fused) {
            try g.run("ik_conv_ring", a, &.{ xb, rb, wb, yb }, grid, group, false);
        } else {
            try g.run("ik_conv", a, &.{ xb, rb, wb, yb }, grid, group, false);
            try g.run("ik_ring", a, &.{ xb, rb }, keep, group, false);
        },
        .dynamic => if (fused) {
            try g.run("ik_conv_dyn_ring", a, &.{ xb, rb, tb, db, yb }, grid, group, false);
        } else {
            try g.run("ik_conv_dyn", a, &.{ xb, rb, tb, db, yb }, grid, group, false);
            try g.run("ik_ring", a, &.{ xb, rb }, keep, group, false);
        },
        .mlp => if (fused) {
            try g.run("ik_conv_out", a, &.{ xb, rb, wb, sb, yb }, grid, group, false);
        } else {
            try g.run("ik_conv", a, &.{ xb, rb, wb, mb }, grid, group, false);
            try g.run("ik_ring", a, &.{ xb, rb }, keep, group, false);
            try g.run("ik_out", a, &.{ sb, mb, yb }, grid, group, false);
        },
    }
}

fn copyBytes(dst: mtl.Buffer, from: mtl.Buffer, n: usize) void {
    @memcpy(dst.slice(u8, n), from.slice(u8, n));
}

/// A fused kernel gives the bits of the chain it replaces, in `chunk` rows at `pos`: outputs and the advanced ring, with a
/// NaN row 1 (the same bits, and rows 0 and 5 on stay as without it), and one row a dispatch over the same rows.
fn fusedCase(g: *const h.Gpu, r: std.Random, kind: Kind, chunk: u32, pos: u32) !void {
    const ybytes = fused_rows * C * 2;
    const rbytes = 32 * C * 2;
    const x = try g.buf(ybytes);
    const w = try g.buf(C * 4 * 2);
    const taps = try g.buf(4 * C * 2);
    const dyn = try g.buf(fused_rows * 8 * 4 * 2);
    const res = try g.buf(fused_rows * C * 4);
    const ym = try g.buf(ybytes);
    const ring0 = try g.buf(rbytes);
    const ring_ref = try g.buf(rbytes);
    const ring_fused = try g.buf(rbytes);
    const ring_step = try g.buf(rbytes);
    const y_ref = try g.buf(ybytes);
    const y_fused = try g.buf(ybytes);
    const y_step = try g.buf(ybytes);
    defer for ([_]mtl.Buffer{ x, w, taps, dyn, res, ym, ring0, ring_ref, ring_fused, ring_step, y_ref, y_fused, y_step }) |b| b.deinit();
    const xs = h.fill16(x, fused_rows * C, r, 3);
    _ = h.fill16(w, C * 4, r, 0.7);
    _ = h.fill16(taps, 4 * C, r, 0.5);
    _ = h.fill16(dyn, fused_rows * 32, r, 0.5);
    _ = h.fill32(res, fused_rows * C, r, 4);
    _ = h.fill16(ring0, 32 * C, r, 3);
    const s: Setup = .{ .x = x, .w = w, .taps = taps, .dyn = dyn, .res = res, .ym = ym };
    const a = h.Args{ .rows = chunk, .width = C, .pos = pos, .heads = 8, .scale = if (kind == .dynamic) 3 else @sqrt(5.0) };
    var name: [64]u8 = undefined;
    var c: h.Check = .{ .name = std.fmt.bufPrint(&name, "fused {s} {d} rows at {d}", .{ @tagName(kind), chunk, pos }) catch unreachable };

    copyBytes(ring_ref, ring0, rbytes);
    copyBytes(ring_fused, ring0, rbytes);
    try exec(g, kind, false, a, s, ring_ref, y_ref, 0);
    try exec(g, kind, true, a, s, ring_fused, y_fused, 0);
    const yn = chunk * C * 2;
    c.same(y_fused.slice(u8, yn), y_ref.slice(u8, yn));
    c.same(ring_fused.slice(u8, rbytes), ring_ref.slice(u8, rbytes));

    // a NaN row 1 stays within the rows and the ring slot it reaches
    if (chunk > 1) {
        const clean_y = try std.heap.page_allocator.dupe(u8, y_fused.slice(u8, yn));
        const clean_ring = try std.heap.page_allocator.dupe(u8, ring_fused.slice(u8, rbytes));
        defer std.heap.page_allocator.free(clean_y);
        defer std.heap.page_allocator.free(clean_ring);
        const kept = try std.heap.page_allocator.dupe(u16, xs[C .. 2 * C]);
        defer std.heap.page_allocator.free(kept);
        @memset(xs[C .. 2 * C], 0x7fc0);
        copyBytes(ring_ref, ring0, rbytes);
        copyBytes(ring_fused, ring0, rbytes);
        try exec(g, kind, false, a, s, ring_ref, y_ref, 0);
        try exec(g, kind, true, a, s, ring_fused, y_fused, 0);
        c.same(y_fused.slice(u8, yn), y_ref.slice(u8, yn));
        c.same(ring_fused.slice(u8, rbytes), ring_ref.slice(u8, rbytes));
        const row = C * 2;
        for (0..chunk) |t| if (t == 0 or t >= 5) c.same(y_fused.slice(u8, yn)[t * row ..][0..row], clean_y[t * row ..][0..row]);
        for (0..32) |slot| if (slot != (pos + 1) % 32) c.same(ring_fused.slice(u8, rbytes)[slot * row ..][0..row], clean_ring[slot * row ..][0..row]);
        @memcpy(xs[C .. 2 * C], kept);
    }

    // one row a dispatch, as decode steps
    copyBytes(ring_ref, ring0, rbytes);
    copyBytes(ring_step, ring0, rbytes);
    try exec(g, kind, false, a, s, ring_ref, y_ref, 0);
    for (0..chunk) |t| {
        var one = a;
        one.rows = 1;
        one.pos = pos + @as(u32, @intCast(t));
        try exec(g, kind, true, one, s, ring_step, y_step, t);
    }
    c.same(y_step.slice(u8, yn), y_ref.slice(u8, yn));
    c.same(ring_step.slice(u8, rbytes), ring_ref.slice(u8, rbytes));
    try c.done();
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
