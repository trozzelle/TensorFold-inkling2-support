//! Inkling 2 kernel checks: embedding norm, RMS norms, depth attention, residual adds and softcap.
const std = @import("std");
const mtl = @import("metal");
const h = @import("harness.zig");

const H = 4608;

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    try caseEmbed(g, r);
    try caseRmsNorm(g, r);
    try caseNormF32(g, r);
    try caseDepth(g, r);
    try caseResidSoftcap(g, r);
}

/// Three ids gathered from a 16-row table and weighted-RMS-normed; each row alone gives the batch's bits.
fn caseEmbed(g: *const h.Gpu, r: std.Random) !void {
    const rows = 3;
    const V = 16;
    const ids = try g.buf(rows * 4);
    const table = try g.buf(V * H * 2);
    const w = try g.buf(H * 2);
    const y = try g.buf(rows * H * 2);
    defer for ([_]mtl.Buffer{ ids, table, w, y }) |b| b.deinit();
    const id = ids.slice(u32, rows);
    id[0] = 5;
    id[1] = 0;
    id[2] = 15;
    const ts = h.fill16(table, V * H, r, 3.0);
    const ws = h.fill16(w, H, r, 1.5);
    const a: h.Args = .{ .rows = rows, .width = H, .eps = 1e-6 };
    try g.run("ik_embed", a, &.{ .{ .buf = ids }, .{ .buf = table }, .{ .buf = w }, .{ .buf = y } }, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, true);
    var c: h.Check = .{ .name = "embed" };
    const ys = y.slice(u16, rows * H);
    for (0..rows) |t| {
        const x = ts[id[t] * H ..][0..H];
        var ss: f64 = 0;
        for (x) |v| ss += h.fb(v) * h.fb(v);
        const inv = 1 / @sqrt(ss / H + 1e-6);
        for (0..H) |d| c.b16(ys[t * H + d], h.fb(ws[d]) * (h.fb(x[d]) * inv));
    }
    const all = try std.heap.page_allocator.dupe(u8, y.slice(u8, rows * H * 2));
    defer std.heap.page_allocator.free(all);
    for (0..rows) |t| {
        var one = a;
        one.rows = 1;
        try g.run("ik_embed", one, &.{ .{ .buf = ids, .off = t * 4 }, .{ .buf = table }, .{ .buf = w }, .{ .buf = y, .off = t * H * 2 } }, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, true);
        c.same(y.slice(u8, rows * H * 2)[t * H * 2 ..][0 .. H * 2], all[t * H * 2 ..][0 .. H * 2]);
    }
    try c.done();
}

fn caseRmsNorm(g: *const h.Gpu, r: std.Random) !void {
    const rows = 5;
    const x = try g.buf(rows * H * 2);
    const w = try g.buf(H * 2);
    const y = try g.buf(rows * H * 4);
    const z = try g.buf(rows * H * 2);
    defer for ([_]mtl.Buffer{ x, w, y, z }) |b| b.deinit();
    const xs = h.fill16(x, rows * H, r, 3.0);
    const ws = h.fill16(w, H, r, 1.5);
    try g.run("ik_rms", .{ .rows = rows, .width = H, .eps = 1e-6 }, &.{ .{ .buf = x }, .{ .buf = y } }, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, true);
    try g.run("ik_norm", .{ .rows = rows, .width = H, .stride = H, .eps = 1e-6 }, &.{ .{ .buf = x }, .{ .buf = w }, .{ .buf = z } }, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, true);
    var c: h.Check = .{ .name = "rms, norm" };
    for (0..rows) |t| {
        var ss: f64 = 0;
        for (0..H) |d| ss += h.fb(xs[t * H + d]) * h.fb(xs[t * H + d]);
        const inv = 1 / @sqrt(ss / H + 1e-6);
        for (0..H) |d| {
            c.f(y.slice(f32, rows * H)[t * H + d], h.fb(xs[t * H + d]) * inv, 1e-5);
            c.b16(z.slice(u16, rows * H)[t * H + d], h.fb(ws[d]) * (h.fb(xs[t * H + d]) * inv));
        }
    }
    try c.done();
    try h.independent(g, "ik_norm", .{ .rows = rows, .width = H, .stride = H, .eps = 1e-6 }, x, H * 2, &.{w}, z, H * 2, rows, true);
}

/// f32 inputs (IK_F32_IN) for both norms, the norm's rows a stride wider than the width.
fn caseNormF32(g: *const h.Gpu, r: std.Random) !void {
    const rows = 3;
    const stride = H + 64;
    const x = try g.buf(rows * stride * 4);
    const w = try g.buf(H * 2);
    const y = try g.buf(rows * H * 4);
    const z = try g.buf(rows * H * 2);
    defer for ([_]mtl.Buffer{ x, w, y, z }) |b| b.deinit();
    const xs = h.fill32(x, rows * stride, r, 3.0);
    const ws = h.fill16(w, H, r, 1.5);
    try g.run("ik_norm", .{ .rows = rows, .width = H, .stride = stride, .flags = 1, .eps = 1e-6 }, &.{ .{ .buf = x }, .{ .buf = w }, .{ .buf = z } }, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, true);
    var c: h.Check = .{ .name = "norm f32 in" };
    for (0..rows) |t| {
        var ss: f64 = 0;
        for (0..H) |d| ss += @as(f64, xs[t * stride + d]) * xs[t * stride + d];
        const inv = 1 / @sqrt(ss / H + 1e-6);
        for (0..H) |d| c.b16(z.slice(u16, rows * H)[t * H + d], h.fb(ws[d]) * (@as(f64, xs[t * stride + d]) * inv));
    }
    // ik_rms reads its rows densely.
    try g.run("ik_rms", .{ .rows = rows, .width = H, .flags = 1, .eps = 1e-6 }, &.{ .{ .buf = x }, .{ .buf = y } }, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, true);
    for (0..rows) |t| {
        var ss: f64 = 0;
        for (0..H) |d| ss += @as(f64, xs[t * H + d]) * xs[t * H + d];
        const inv = 1 / @sqrt(ss / H + 1e-6);
        for (0..H) |d| c.f(y.slice(f32, rows * H)[t * H + d], @as(f64, xs[t * H + d]) * inv, 1e-5);
    }
    try c.done();
}

fn caseDepth(g: *const h.Gpu, r: std.Random) !void {
    const rows = 3;
    const K = 4;
    const Q = 3;
    const stride = 8; // the blocks' rows apart, as the scratch's prompt_rows
    const blocks = try g.buf(K * stride * H * 4);
    const part = try g.buf(rows * H * 4);
    const q = try g.buf(Q * H * 4);
    const out = try g.buf(rows * Q * H * 4);
    defer for ([_]mtl.Buffer{ blocks, part, q, out }) |b| b.deinit();
    const bs = h.fill32(blocks, K * stride * H, r, 1.0);
    const ps = h.fill32(part, rows * H, r, 1.0);
    const qs = h.fill32(q, Q * H, r, 0.2);
    const scale = 1 / @sqrt(@as(f64, H));
    try g.run("ik_depth", .{ .rows = rows, .width = H, .blocks = K, .queries = Q, .stride = stride, .scale = @floatCast(scale) }, &.{ .{ .buf = blocks }, .{ .buf = part }, .{ .buf = q }, .{ .buf = out } }, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, true);
    var c: h.Check = .{ .name = "depth" };
    for (0..rows) |t| for (0..Q) |j| {
        var w: [K + 1]f64 = undefined;
        for (0..K + 1) |i| {
            var s: f64 = 0;
            for (0..H) |d| s += (if (i < K) bs[(i * stride + t) * H + d] else ps[t * H + d]) * qs[j * H + d];
            w[i] = s * scale;
        }
        var m: f64 = -std.math.inf(f64);
        for (w) |v| m = @max(m, v);
        var z: f64 = 0;
        for (&w) |*v| {
            v.* = @exp(v.* - m);
            z += v.*;
        }
        for (0..H) |d| {
            var o: f64 = 0;
            for (0..K + 1) |i| o += w[i] / z * (if (i < K) bs[(i * stride + t) * H + d] else ps[t * H + d]);
            c.near(out.slice(f32, rows * Q * H)[(t * Q + j) * H + d], o, 1e-4, 1e-2);
        }
    };
    try c.done();
}

fn caseResidSoftcap(g: *const h.Gpu, r: std.Random) !void {
    const rows = 4;
    const V = 1000;
    const hb = try g.buf(rows * H * 2);
    const ya = try g.buf(rows * H * 2);
    const res = try g.buf(rows * H * 4);
    const z = try g.buf(rows * V * 2);
    defer for ([_]mtl.Buffer{ hb, ya, res, z }) |b| b.deinit();
    const hs = h.fill16(hb, rows * H, r, 4);
    const ys = h.fill16(ya, rows * H, r, 4);
    var c: h.Check = .{ .name = "resid, out, softcap" };
    for ([_]u32{ 0, 4 }) |flag| {
        try g.run("ik_resid", .{ .rows = rows, .width = H, .flags = flag }, &.{ .{ .buf = hb }, .{ .buf = ya }, .{ .buf = res } }, .{ H, rows, 1 }, .{ 256, 1, 1 }, false);
        for (0..rows * H) |i| c.f(res.slice(f32, rows * H)[i], (if (flag != 0) 0 else h.fb(hs[i])) + h.fb(ys[i]), 1e-6);
    }
    // After the boundary run `res` equals `ya`, so the output is bf16(ya + ya).
    try g.run("ik_out", .{ .rows = rows, .width = H }, &.{ .{ .buf = res }, .{ .buf = ya }, .{ .buf = hb } }, .{ H, rows, 1 }, .{ 256, 1, 1 }, false);
    for (0..rows * H) |i| c.b16(hb.slice(u16, rows * H)[i], h.fb(ys[i]) + h.fb(ys[i]));
    // ik_copy: the output rows copied bit for bit.
    try g.run("ik_copy", .{ .rows = rows, .width = H }, &.{ .{ .buf = hb }, .{ .buf = ya } }, .{ H, rows, 1 }, .{ 256, 1, 1 }, false);
    c.same(hb.slice(u8, rows * H * 2), ya.slice(u8, rows * H * 2));
    const zs = h.fill16(z, rows * V, r, 80);
    const before = try std.heap.page_allocator.dupe(u16, zs);
    defer std.heap.page_allocator.free(before);
    try g.run("ik_softcap", .{ .rows = rows, .width = V, .extent = 990, .cap = 30 }, &.{.{ .buf = z }}, .{ V, rows, 1 }, .{ 256, 1, 1 }, false);
    for (0..rows) |t| for (0..V) |i| {
        const got = z.slice(u16, rows * V)[t * V + i];
        if (i >= 990) {
            c.n += 1;
            if (got != 0xff80) c.bad += 1; // -inf
        } else c.b16(got, 30 * std.math.tanh(h.fb(before[t * V + i]) / 30));
    };
    try c.done();
}
