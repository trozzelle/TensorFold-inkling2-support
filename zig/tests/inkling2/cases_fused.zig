//! Inkling 2 kernel checks: fused norm and depth-attention chains. Each fused kernel is run beside the chain of
//! kernels on copies of the same inputs and must give every output buffer bit for bit; each row alone gives the
//! window's bits, and a NaN row stays in its own outputs.
const std = @import("std");
const mtl = @import("metal");
const h = @import("harness.zig");

const H = 4608;
const stride = 24; // the blocks' rows apart (the scratch uses its prefill rows)
const eps_norm: f32 = 1e-5; // away from the RMS's 1e-6, so a swapped epsilon shows
const eps_rms: f32 = 1e-6;
const f32_in: u32 = 1; // forward.zig's flags
const boundary_flag: u32 = 4;
const nan16: u16 = 0x7fc0;
const nan32: u32 = 0x7fc00000;

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    for ([_]usize{ 1, 7, 16 }) |rows| {
        try caseRdn(g, r, rows, 3, 4, false, false); // a global layer
        try caseRdn(g, r, rows, 3, 4, true, false); // a global layer at a block boundary
        try caseRdn(g, r, rows, 1, 9, false, false); // a local layer with the most blocks
        try caseRdn(g, r, rows, 3, 4, true, true); // f32 input
        try caseRdn(g, r, rows, 0, 4, false, false); // no query: the RMS alone
        try caseRrdn(g, r, rows, 4, false);
        try caseRrdn(g, r, rows, 4, true);
        try caseRrdn(g, r, rows, 8, true); // the most sources
        try caseRrdn(g, r, rows, 0, true); // the first block
    }
}

fn clone(g: *const h.Gpu, src: mtl.Buffer, bytes: usize) !mtl.Buffer {
    const b = try g.buf(bytes);
    @memcpy(b.slice(u8, bytes), src.slice(u8, bytes));
    return b;
}

fn scale() f32 {
    return @floatCast(1 / @sqrt(@as(f64, H)));
}

/// The RMS destination's bytes for `rows` rows: block slot K's rows at a boundary, else the nr buffer.
fn nrdBytes(nr: mtl.Buffer, blocks: mtl.Buffer, boundary: bool, K: usize, rows: usize) []u8 {
    if (boundary) return blocks.slice(u8, (K * stride + rows) * H * 4)[K * stride * H * 4 ..];
    return nr.slice(u8, rows * H * 4);
}

/// The binding of the RMS destination from row `row`.
fn nrdBind(nr: mtl.Buffer, blocks: mtl.Buffer, boundary: bool, K: usize, from: usize) h.Bind {
    if (boundary) return .{ .buf = blocks, .off = (K * stride + from) * H * 4 };
    return .{ .buf = nr, .off = from * H * 4 };
}

/// ik_rdn's buffers: the queries' weights in three slots (from one [q][H] buffer; slots past q reuse the last weights).
fn rdnBinds(hin: h.Bind, blocks: h.Bind, nrd: h.Bind, query: mtl.Buffer, norms: mtl.Buffer, q: usize, xin: h.Bind) [8]h.Bind {
    const o1: usize = @as(usize, @min(1, q - 1)) * H * 2;
    const o2: usize = @as(usize, @min(2, q - 1)) * H * 2;
    return .{ hin, blocks, nrd, .{ .buf = query }, .{ .buf = norms }, .{ .buf = norms, .off = o1 }, .{ .buf = norms, .off = o2 }, xin };
}

fn row(s: []const u8, t: usize, bytes: usize) []const u8 {
    return s[t * bytes ..][0..bytes];
}

/// The fused attention-input chain against ik_rms, ik_depth and Q ik_norm calls.
fn caseRdn(g: *const h.Gpu, r: std.Random, rows: usize, Q: u32, K: u32, boundary: bool, f32input: bool) !void {
    const esz: usize = if (f32input) 4 else 2;
    const q = @max(Q, 1);
    const bbytes = (K + 1) * stride * H * 4;
    const hb = try g.buf(rows * H * esz);
    const blocks_a = try g.buf(bbytes); // the separate kernels' chain
    const nr_a = try g.buf(rows * H * 4);
    const xin_a = try g.buf(q * rows * H * 2);
    const ai = try g.buf(rows * q * H * 4);
    const query = try g.buf(q * H * 4);
    const norms = try g.buf(q * H * 2);
    if (f32input) _ = h.fill32(hb, rows * H, r, 3.0) else _ = h.fill16(hb, rows * H, r, 3.0);
    _ = h.fill32(blocks_a, (K + 1) * stride * H, r, 1.0); // slot K too: the RMS must overwrite it at a boundary
    _ = h.fill32(query, q * H, r, 0.2);
    _ = h.fill16(norms, q * H, r, 1.5);
    const blocks_init = try clone(g, blocks_a, bbytes);
    const blocks_b = try clone(g, blocks_a, bbytes); // the fused kernel's
    const nr_b = try g.buf(rows * H * 4);
    const xin_b = try g.buf(q * rows * H * 2);
    const blocks_c = try clone(g, blocks_a, bbytes); // one row at a time
    const nr_c = try g.buf(rows * H * 4);
    const xin_c = try g.buf(q * H * 2);
    const blocks_d = try clone(g, blocks_a, bbytes); // a NaN row 1
    const nr_d = try g.buf(rows * H * 4);
    const xin_d = try g.buf(q * rows * H * 2);
    defer for ([_]mtl.Buffer{ hb, blocks_a, nr_a, xin_a, ai, query, norms, blocks_init, blocks_b, nr_b, xin_b, blocks_c, nr_c, xin_c, blocks_d, nr_d, xin_d }) |b| b.deinit();
    const in_flag: u32 = if (f32input) f32_in else 0;
    const fused: h.Args = .{ .rows = @intCast(rows), .width = H, .queries = Q, .blocks = K, .stride = stride, .flags = in_flag | if (boundary) boundary_flag else 0, .eps = eps_norm, .scale = scale() };
    const grid = .{ rows, 1, 1 };
    const one = .{ 1024, 1, 1 };

    const nrd_a = nrdBind(nr_a, blocks_a, boundary, K, 0);
    try g.run("ik_rms", .{ .rows = @intCast(rows), .width = H, .flags = in_flag, .eps = eps_rms }, &.{ .{ .buf = hb }, nrd_a }, grid, one, true);
    if (Q > 0) {
        try g.run("ik_depth", .{ .rows = @intCast(rows), .width = H, .blocks = K, .queries = Q, .stride = stride, .scale = scale() }, &.{ .{ .buf = blocks_a }, nrd_a, .{ .buf = query }, .{ .buf = ai } }, grid, one, true);
        for (0..Q) |j| try g.run("ik_norm", .{ .rows = @intCast(rows), .width = H, .stride = Q * H, .flags = f32_in, .eps = eps_norm }, &.{ .{ .buf = ai, .off = j * H * 4 }, .{ .buf = norms, .off = j * H * 2 }, .{ .buf = xin_a, .off = j * rows * H * 2 } }, grid, one, true);
    }
    const kernel: [:0]const u8 = if (Q > 1) "ik_rdn3" else "ik_rdn"; // the instance for the layer's query count
    try g.run(kernel, fused, &rdnBinds(.{ .buf = hb }, .{ .buf = blocks_b }, nrdBind(nr_b, blocks_b, boundary, K, 0), query, norms, q, .{ .buf = xin_b }), grid, one, true);

    var name: [64]u8 = undefined;
    var c: h.Check = .{ .name = std.fmt.bufPrint(&name, "rdn rows {d} Q {d} K {d}{s}{s}", .{ rows, Q, K, if (boundary) " boundary" else "", if (f32input) " f32" else "" }) catch "rdn" };
    c.same(blocks_a.slice(u8, bbytes), blocks_b.slice(u8, bbytes));
    c.same(nr_a.slice(u8, rows * H * 4), nr_b.slice(u8, rows * H * 4));
    c.same(xin_a.slice(u8, Q * rows * H * 2), xin_b.slice(u8, Q * rows * H * 2));

    // Each row alone gives the window's bits (a one-row call lays its queries H apart).
    const xb = xin_b.slice(u8, q * rows * H * 2);
    for (0..rows) |t| {
        var one_row = fused;
        one_row.rows = 1;
        try g.run(kernel, one_row, &rdnBinds(.{ .buf = hb, .off = t * H * esz }, .{ .buf = blocks_c, .off = t * H * 4 }, nrdBind(nr_c, blocks_c, boundary, K, t), query, norms, q, .{ .buf = xin_c }), .{ 1, 1, 1 }, one, true);
        for (0..Q) |j| c.same(row(xin_c.slice(u8, q * H * 2), j, H * 2), row(xb, j * rows + t, H * 2));
    }
    c.same(blocks_c.slice(u8, bbytes), blocks_b.slice(u8, bbytes));
    c.same(nr_c.slice(u8, rows * H * 4), nr_b.slice(u8, rows * H * 4));

    // A NaN in row 1 stays in row 1.
    if (rows > 1) {
        if (f32input) hb.slice(u32, rows * H)[H] = nan32 else hb.slice(u16, rows * H)[H] = nan16;
        try g.run(kernel, fused, &rdnBinds(.{ .buf = hb }, .{ .buf = blocks_d }, nrdBind(nr_d, blocks_d, boundary, K, 0), query, norms, q, .{ .buf = xin_d }), grid, one, true);
        const nb = nrdBytes(nr_b, blocks_b, boundary, K, rows);
        const nd = nrdBytes(nr_d, blocks_d, boundary, K, rows);
        for (0..rows) |t| if (t != 1) {
            c.same(row(nd, t, H * 4), row(nb, t, H * 4));
            for (0..Q) |j| c.same(row(xin_d.slice(u8, q * rows * H * 2), j * rows + t, H * 2), row(xb, j * rows + t, H * 2));
        };
    }
    try c.done();
}

/// The fused MLP-input chain against ik_resid, ik_rms, ik_depth and ik_norm. K blocks, plus the boundary's slot.
fn caseRrdn(g: *const h.Gpu, r: std.Random, rows: usize, K: u32, boundary: bool) !void {
    const bbytes = (K + 1) * stride * H * 4;
    const hb = try g.buf(rows * H * 2);
    const ya = try g.buf(rows * H * 2);
    const blocks = try g.buf(bbytes);
    const query = try g.buf(H * 4);
    const norm = try g.buf(H * 2);
    _ = h.fill16(hb, rows * H, r, 3.0);
    _ = h.fill16(ya, rows * H, r, 3.0);
    _ = h.fill32(blocks, (K + 1) * stride * H, r, 1.0);
    _ = h.fill32(query, H, r, 0.2);
    _ = h.fill16(norm, H, r, 1.5);
    const res_a = try g.buf(rows * H * 4); // the separate kernels' chain
    const ns_a = try g.buf(rows * H * 4);
    const mi = try g.buf(rows * H * 4);
    const xm_a = try g.buf(rows * H * 2);
    const res_b = try g.buf(rows * H * 4); // the fused kernel's
    const ns_b = try g.buf(rows * H * 4);
    const xm_b = try g.buf(rows * H * 2);
    const res_c = try g.buf(rows * H * 4); // one row at a time
    const ns_c = try g.buf(rows * H * 4);
    const xm_c = try g.buf(rows * H * 2);
    const res_d = try g.buf(rows * H * 4); // a NaN row 1
    const ns_d = try g.buf(rows * H * 4);
    const xm_d = try g.buf(rows * H * 2);
    defer for ([_]mtl.Buffer{ hb, ya, blocks, query, norm, res_a, ns_a, mi, xm_a, res_b, ns_b, xm_b, res_c, ns_c, xm_c, res_d, ns_d, xm_d }) |b| b.deinit();
    const flag: u32 = if (boundary) boundary_flag else 0;
    const nr: u32 = @intCast(rows);
    const grid = .{ rows, 1, 1 };
    const one = .{ 1024, 1, 1 };

    try g.run("ik_resid", .{ .rows = nr, .width = H, .flags = flag }, &.{ .{ .buf = hb }, .{ .buf = ya }, .{ .buf = res_a } }, .{ H, rows, 1 }, .{ 256, 1, 1 }, false);
    try g.run("ik_rms", .{ .rows = nr, .width = H, .flags = f32_in, .eps = eps_rms }, &.{ .{ .buf = res_a }, .{ .buf = ns_a } }, grid, one, true);
    try g.run("ik_depth", .{ .rows = nr, .width = H, .blocks = K + @intFromBool(boundary), .queries = 1, .stride = stride, .scale = scale() }, &.{ .{ .buf = blocks }, .{ .buf = ns_a }, .{ .buf = query }, .{ .buf = mi } }, grid, one, true);
    try g.run("ik_norm", .{ .rows = nr, .width = H, .stride = H, .flags = f32_in, .eps = eps_norm }, &.{ .{ .buf = mi }, .{ .buf = norm }, .{ .buf = xm_a } }, grid, one, true);
    const fused: h.Args = .{ .rows = nr, .width = H, .blocks = K, .stride = stride, .flags = flag, .eps = eps_norm, .scale = scale() };
    try g.run("ik_rrdn", fused, &.{ .{ .buf = hb }, .{ .buf = ya }, .{ .buf = res_b }, .{ .buf = blocks }, .{ .buf = ns_b }, .{ .buf = query }, .{ .buf = norm }, .{ .buf = xm_b } }, grid, one, true);

    var name: [64]u8 = undefined;
    var c: h.Check = .{ .name = std.fmt.bufPrint(&name, "rrdn rows {d} K {d}{s}", .{ rows, K, if (boundary) " boundary" else "" }) catch "rrdn" };
    c.same(res_a.slice(u8, rows * H * 4), res_b.slice(u8, rows * H * 4));
    c.same(ns_a.slice(u8, rows * H * 4), ns_b.slice(u8, rows * H * 4));
    c.same(xm_a.slice(u8, rows * H * 2), xm_b.slice(u8, rows * H * 2));

    for (0..rows) |t| {
        var one_row = fused;
        one_row.rows = 1;
        try g.run("ik_rrdn", one_row, &.{ .{ .buf = hb, .off = t * H * 2 }, .{ .buf = ya, .off = t * H * 2 }, .{ .buf = res_c, .off = t * H * 4 }, .{ .buf = blocks, .off = t * H * 4 }, .{ .buf = ns_c, .off = t * H * 4 }, .{ .buf = query }, .{ .buf = norm }, .{ .buf = xm_c, .off = t * H * 2 } }, .{ 1, 1, 1 }, one, true);
    }
    c.same(res_c.slice(u8, rows * H * 4), res_b.slice(u8, rows * H * 4));
    c.same(ns_c.slice(u8, rows * H * 4), ns_b.slice(u8, rows * H * 4));
    c.same(xm_c.slice(u8, rows * H * 2), xm_b.slice(u8, rows * H * 2));

    if (rows > 1) {
        hb.slice(u16, rows * H)[H] = nan16;
        try g.run("ik_rrdn", fused, &.{ .{ .buf = hb }, .{ .buf = ya }, .{ .buf = res_d }, .{ .buf = blocks }, .{ .buf = ns_d }, .{ .buf = query }, .{ .buf = norm }, .{ .buf = xm_d } }, grid, one, true);
        for (0..rows) |t| if (t != 1) {
            c.same(row(res_d.slice(u8, rows * H * 4), t, H * 4), row(res_b.slice(u8, rows * H * 4), t, H * 4));
            c.same(row(ns_d.slice(u8, rows * H * 4), t, H * 4), row(ns_b.slice(u8, rows * H * 4), t, H * 4));
            c.same(row(xm_d.slice(u8, rows * H * 2), t, H * 2), row(xm_b.slice(u8, rows * H * 2), t, H * 2));
        };
    }
    try c.done();
}
