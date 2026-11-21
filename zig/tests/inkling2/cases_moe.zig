//! Inkling 2 kernel checks: router, routing, expert tiles, capped SwiGLU and the weighted combine.
const std = @import("std");
const mtl = @import("metal");
const h = @import("harness.zig");

const H = 4608;

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    try caseRoute(g, r);
    try caseTilesSwigluCombine(g, r);
}

fn caseRoute(g: *const h.Gpu, r: std.Random) !void {
    const rows = 5;
    const E = 384;
    const x = try g.buf(rows * H * 2);
    const w = try g.buf(E * H * 2);
    const logits = try g.buf(rows * E * 4);
    const bias = try g.buf(E * 4);
    const ids = try g.buf(rows * 8 * 4);
    const wt = try g.buf(rows * 8 * 4);
    defer for ([_]mtl.Buffer{ x, w, logits, bias, ids, wt }) |b| b.deinit();
    const xs = h.fill16(x, rows * H, r, 1);
    const ws = h.fill16(w, E * H, r, 0.05);
    const bs = h.fill32(bias, E, r, 0.3);
    try g.run("ik_router", .{ .rows = rows, .width = E, .dim = H }, &.{ .{ .buf = x }, .{ .buf = w }, .{ .buf = logits } }, .{ (E + 7) / 8, rows, 1 }, .{ 256, 1, 1 }, true);
    var c: h.Check = .{ .name = "router, route" };
    const ls = logits.slice(f32, rows * E);
    for (0..rows) |t| for (0..E) |e| {
        var s: f64 = 0;
        for (0..H) |d| s += h.fb(xs[t * H + d]) * h.fb(ws[e * H + d]);
        c.near(ls[t * E + e], s, 1e-4, 1.0);
    };
    try g.run("ik_route", .{ .rows = rows, .width = E, .eps = 1e-20, .scale = 8 }, &.{ .{ .buf = logits }, .{ .buf = bias }, .{ .buf = ids }, .{ .buf = wt } }, .{ rows * 32, 1, 1 }, .{ 32, 1, 1 }, false);
    for (0..rows) |t| { // the reference on the kernel's own logits: top-8 of sigmoid + bias, ties to the lower id
        var score: [E]f64 = undefined;
        var taken: [E]bool = @splat(false);
        for (0..E) |e| score[e] = 1 / (1 + @exp(-@as(f64, ls[t * E + e])));
        var picked: [8]usize = undefined;
        var total: f64 = 0;
        for (0..8) |k| {
            var best: usize = E;
            for (0..E) |e| if (!taken[e] and (best == E or score[e] + bs[e] > score[best] + bs[best])) {
                best = e;
            };
            taken[best] = true;
            picked[k] = best;
            total += score[best];
        }
        for (0..8) |k| {
            c.n += 1;
            if (ids.slice(u32, rows * 8)[t * 8 + k] != picked[k]) c.bad += 1;
            c.f(wt.slice(f32, rows * 8)[t * 8 + k], score[picked[k]] / (total + 1e-20) * 8, 1e-6);
        }
    }
    try c.done();
}

fn caseTilesSwigluCombine(g: *const h.Gpu, r: std.Random) !void {
    const rows = 7;
    const E = 384;
    const S = rows * 8;
    const T = h.state.tileRows(S, E);
    const ids = try g.buf(S * 4);
    const texp = try g.buf(T * 4);
    const tslot = try g.buf(T * 4);
    const ttok = try g.buf(T * 4);
    defer for ([_]mtl.Buffer{ ids, texp, tslot, ttok }) |b| b.deinit();
    const is = ids.slice(u32, S);
    for (is, 0..) |*x, i| x.* = @intCast((i * 37 + i / 8) % 11); // few experts, several picks each
    try g.run("ik_tiles", .{ .rows = S, .width = E, .queries = 8, .stride = @intCast(T) }, &.{ .{ .buf = ids }, .{ .buf = texp }, .{ .buf = tslot }, .{ .buf = ttok } }, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, true);
    var c: h.Check = .{ .name = "tiles" };
    var seen: [S]bool = @splat(false);
    const te = texp.slice(u32, T);
    for (0..T / 8) |tile| { // each tile one expert or padding; each slot once, with its token, ascending
        var expert: ?u32 = null;
        var last: i64 = -1;
        for (0..8) |j| {
            const i = tile * 8 + j;
            if (te[i] == 0xffffffff) continue;
            c.n += 1;
            if ((expert != null and expert.? != te[i]) or is[tslot.slice(u32, T)[i]] != te[i] or ttok.slice(u32, T)[i] != tslot.slice(u32, T)[i] / 8 or seen[tslot.slice(u32, T)[i]] or tslot.slice(u32, T)[i] <= last) c.bad += 1;
            expert = te[i];
            last = tslot.slice(u32, T)[i];
            seen[tslot.slice(u32, T)[i]] = true;
        }
    }
    for (seen) |s| if (!s) {
        c.bad += 1;
    };
    try c.done();
    // SwiGLU on interleaved rows, then the weighted combine
    const I = 2304;
    const gu = try g.buf(S * 2 * I * 2);
    const act = try g.buf(S * I * 2);
    const ey = try g.buf(S * H * 2);
    const wt = try g.buf(S * 4);
    const moe = try g.buf(rows * H * 2);
    defer for ([_]mtl.Buffer{ gu, act, ey, wt, moe }) |b| b.deinit();
    const gs = h.fill16(gu, S * 2 * I, r, 30);
    try g.run("ik_swiglu", .{ .rows = S, .width = I, .cap = 4, .thr = 25 }, &.{ .{ .buf = gu }, .{ .buf = act } }, .{ I, S, 1 }, .{ 256, 1, 1 }, false);
    var cs: h.Check = .{ .name = "swiglu, combine" };
    for (0..S) |s| for (0..I) |i| {
        const gg = h.fb(gs[s * 2 * I + 2 * i]);
        const u = h.fb(gs[s * 2 * I + 2 * i + 1]);
        cs.b16(act.slice(u16, S * I)[s * I + i], 4 * std.math.tanh(gg / 4) * (1 / (1 + @exp(-gg))) * (25 * std.math.tanh(u / 25)));
    };
    const es = h.fill16(ey, S * H, r, 2);
    const ws = h.fill32(wt, S, r, 1);
    try g.run("ik_combine", .{ .rows = rows, .width = H, .queries = 8 }, &.{ .{ .buf = ey }, .{ .buf = wt }, .{ .buf = moe } }, .{ H, rows, 1 }, .{ 256, 1, 1 }, false);
    for (0..rows) |t| for (0..H) |d| {
        var s: f64 = 0;
        for (0..8) |k| s += h.fb(es[(t * 8 + k) * H + d]) * ws[t * 8 + k];
        cs.b16(moe.slice(u16, rows * H)[t * H + d], s);
    };
    try cs.done();
}
