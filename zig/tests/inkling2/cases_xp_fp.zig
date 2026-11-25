//! Tensor-unit prefill routed-expert checks for NVFP4 and MXFP8 at Mini shapes (H=2304, I=1152, E=128, topk=8):
//! end-to-end MoE outputs against FP64 reference, strict row independence across batch sizes, and NaN containment.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const nvfp4 = tf.quant.nvfp4;
const fp8 = tf.quant.fp8;
const xpf = tf.inkling2.experts_fp_prefill;
const h = @import("harness.zig");

const H = 2304;
const I = 1152;
const E = 128;
const TOPK = 8;
const ROWS_MAX = 128;
const SLOTS_MAX = ROWS_MAX * TOPK;
const CAPS: xpf.Caps = .{ .gate = 4.0, .up = 25.0 };
const BATCH_ROWS = [_]usize{ 1, 7, 37, 128 };

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    const gpa = std.heap.page_allocator;
    if (!xpf.available(g.device)) {
        std.debug.print("SKIP xp_fp: no tensor units on this GPU\n", .{});
        return;
    }
    var pipes = try xpf.Pipes.init(g.device, gpa);
    defer pipes.deinit();
    const scratch = try g.buf(xpf.xpBytes(ROWS_MAX, H, E));
    defer scratch.deinit();
    const s = xpf.Xp.init(scratch, 0, ROWS_MAX, H, E);

    try caseNvfp4(g, gpa, &pipes, s, r);
    try caseMxfp8(g, gpa, &pipes, s, r);
}

fn swigluRef(g: f64, u: f64) f64 {
    const cg = std.math.tanh(g / 4.0) * 4.0;
    const cu = std.math.tanh(u / 25.0) * 25.0;
    return cg * (1.0 / (1.0 + @exp(-g))) * cu;
}

fn e8m0Ref(code: u8) f64 {
    if (code == 255) return std.math.nan(f64);
    const bits: u32 = if (code == 0) 0x00400000 else @as(u32, code) << 23;
    return @as(f32, @bitCast(bits));
}

fn poison(b: mtl.Buffer) void {
    @memset(b.slice(u8, b.length()), 0xff);
}

/// Skewed picks (Zipf skew 1.2) so larger batches form multi-block (>16) and multi-window (>32) experts, while
/// forcing row 0 and row 1 to share expert 0 when n >= 2 so NaN containment exercises shared windows directly.
fn fillPicks(r: std.Random, ids: []u32, ew: []f32, n: usize) void {
    var cdf: [E]f64 = undefined;
    var total: f64 = 0;
    for (&cdf, 0..) |*c, k| {
        total += 1.0 / std.math.pow(f64, @as(f64, @floatFromInt(k + 1)), 1.2);
        c.* = total;
    }
    for (0..n) |row| {
        var k: usize = 0;
        if (row < 2 and n >= 2) {
            ids[row * TOPK] = 0;
            k = 1;
        }
        while (k < TOPK) {
            const u = r.float(f64) * total;
            var lo: usize = 0;
            var hi: usize = E - 1;
            while (lo < hi) {
                const mid = (lo + hi) / 2;
                if (cdf[mid] < u) lo = mid + 1 else hi = mid;
            }
            const pick: u32 = @intCast(lo);
            if (std.mem.indexOfScalar(u32, ids[row * TOPK ..][0..k], pick) != null) continue;
            ids[row * TOPK + k] = pick;
            k += 1;
        }
    }
    for (ew) |*w| w.* = 0.05 + r.float(f32) * 0.25;
}

fn submit(g: *const h.Gpu, cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("ik_xpf: {s}", .{msg});
        return error.GpuFailed;
    }
    _ = g;
}

fn sampledRows(n: usize, out: *[4]usize) []const usize {
    if (n == 1) {
        out[0] = 0;
        return out[0..1];
    }
    if (n <= 7) {
        out[0] = 0;
        out[1] = 1;
        out[2] = n - 1;
        return out[0..3];
    }
    out[0] = 0;
    out[1] = 1;
    out[2] = n / 2;
    out[3] = n - 1;
    return out[0..4];
}

fn caseNvfp4(g: *const h.Gpu, gpa: std.mem.Allocator, p: *const xpf.Pipes, s: xpf.Xp, r: std.Random) !void {
    const w13_bytes = E * 2 * I * (H / 2);
    const s13_bytes = E * 2 * I * (H / 16);
    const w2_bytes = E * H * (I / 2);
    const s2_bytes = E * H * (I / 16);
    const xm = try g.buf(ROWS_MAX * H * 2);
    const eids = try g.buf(SLOTS_MAX * 4);
    const ew = try g.buf(SLOTS_MAX * 4);
    const moe = try g.buf(ROWS_MAX * H * 2);
    const moe1 = try g.buf(H * 2);
    const w13 = try g.buf(w13_bytes);
    const s13 = try g.buf(s13_bytes);
    const g13 = try g.buf(E * 4);
    const w2 = try g.buf(w2_bytes);
    const s2 = try g.buf(s2_bytes);
    const g2 = try g.buf(E * 4);
    defer for ([_]mtl.Buffer{ xm, eids, ew, moe, moe1, w13, s13, g13, w2, s2, g2 }) |b| b.deinit();

    const xms = h.fill16(xm, ROWS_MAX * H, r, 0.5);
    const w13s = w13.slice(u8, w13_bytes);
    r.bytes(w13s);
    const s13s = s13.slice(u8, s13_bytes);
    r.bytes(s13s);
    for (s13s) |*sc| sc.* = @intCast(24 + (sc.* % 48));
    const g13s = g13.slice(f32, E);
    for (g13s) |*gs| gs.* = @floatCast(0.01 + r.float(f64) * 0.05);

    const w2s = w2.slice(u8, w2_bytes);
    r.bytes(w2s);
    const s2s = s2.slice(u8, s2_bytes);
    r.bytes(s2s);
    for (s2s) |*sc| sc.* = @intCast(24 + (sc.* % 48));
    const g2s = g2.slice(f32, E);
    for (g2s) |*gs| gs.* = @floatCast(0.01 + r.float(f64) * 0.05);

    const layer: xpf.Nvfp4Layer = .{
        .w13 = .{ .weight = .{ .buffer = w13, .offset = 0, .bytes = w13_bytes, .dtype = .u8 }, .scale = .{ .buffer = s13, .offset = 0, .bytes = s13_bytes, .dtype = .f8_e4m3 }, .global_scale = g13, .n = 2 * I, .k = H },
        .w2 = .{ .weight = .{ .buffer = w2, .offset = 0, .bytes = w2_bytes, .dtype = .u8 }, .scale = .{ .buffer = s2, .offset = 0, .bytes = s2_bytes, .dtype = .f8_e4m3 }, .global_scale = g2, .n = H, .k = I },
        .experts = E,
    };

    const act_ref = try gpa.alloc(f64, TOPK * I);
    defer gpa.free(act_ref);
    var c_ref: h.Check = .{ .name = "xp_fp_nvfp4_ref" };
    var c_ind: h.Check = .{ .name = "xp_fp_nvfp4_row_indep" };
    var c_nan: h.Check = .{ .name = "xp_fp_nvfp4_nan_contain" };

    for (BATCH_ROWS) |rows| {
        const slots = rows * TOPK;
        const ids = eids.slice(u32, slots);
        const weights = ew.slice(f32, slots);
        fillPicks(r, ids, weights, rows);

        poison(s.counts.buf);
        poison(moe);
        const cb = g.queue.commandBuffer();
        const enc = cb.compute(.serial);
        xpf.encodeNvfp4(enc, p, layer, .{ .buf = xm }, .{ .ids = .{ .buf = eids }, .weights = .{ .buf = ew } }, .{ .buf = moe }, s, rows, CAPS);
        try submit(g, cb, enc);

        const got_moe = moe.slice(u16, rows * H);
        var sbuf: [4]usize = undefined;
        const check_rows = sampledRows(rows, &sbuf);
        for (check_rows) |row| {
            for (0..TOPK) |k| {
                const e = ids[row * TOPK + k];
                for (0..I) |i| {
                    var sg_v: f64 = 0;
                    var su_v: f64 = 0;
                    const rg: usize = (@as(usize, e) * 2 * I + 2 * i);
                    const ru: usize = rg + 1;
                    const wg = w13s[rg * (H / 2) ..][0 .. H / 2];
                    const wu = w13s[ru * (H / 2) ..][0 .. H / 2];
                    const scg = s13s[rg * (H / 16) ..][0 .. H / 16];
                    const scu = s13s[ru * (H / 16) ..][0 .. H / 16];
                    for (0..H / 16) |tk| {
                        const fg = fp8.e4m3(scg[tk]);
                        const fu = fp8.e4m3(scu[tk]);
                        const k0 = tk * 16;
                        for (0..8) |bi| {
                            const bg = wg[tk * 8 + bi];
                            const bu = wu[tk * 8 + bi];
                            const x0 = h.fb(xms[row * H + k0 + 2 * bi]);
                            const x1 = h.fb(xms[row * H + k0 + 2 * bi + 1]);
                            sg_v += x0 * (nvfp4.e2m1(bg & 0xF) * fg) + x1 * (nvfp4.e2m1(bg >> 4) * fg);
                            su_v += x0 * (nvfp4.e2m1(bu & 0xF) * fu) + x1 * (nvfp4.e2m1(bu >> 4) * fu);
                        }
                    }
                    act_ref[k * I + i] = h.rb(swigluRef(h.rb(sg_v * g13s[e]), h.rb(su_v * g13s[e])));
                }
            }
            for (0..32) |idx| {
                const d = (idx * 71) % H;
                var acc: f64 = 0;
                for (0..TOPK) |k| {
                    const e = ids[row * TOPK + k];
                    const r2: usize = @as(usize, e) * H + d;
                    const wr = w2s[r2 * (I / 2) ..][0 .. I / 2];
                    const sr = s2s[r2 * (I / 16) ..][0 .. I / 16];
                    var dot: f64 = 0;
                    for (0..I / 16) |tk| {
                        const sc = fp8.e4m3(sr[tk]);
                        const k0 = tk * 16;
                        for (0..8) |bi| {
                            const b = wr[tk * 8 + bi];
                            dot += act_ref[k * I + k0 + 2 * bi] * (nvfp4.e2m1(b & 0xF) * sc) +
                                act_ref[k * I + k0 + 2 * bi + 1] * (nvfp4.e2m1(b >> 4) * sc);
                        }
                    }
                    acc += h.rb(dot * g2s[e]) * @as(f64, weights[row * TOPK + k]);
                }
                c_ref.b16a(got_moe[row * H + d], acc, 2, 1e-4);
            }
        }

        if (rows == 37 or rows == 128) {
            const batch_bytes = try gpa.dupe(u8, moe.slice(u8, rows * H * 2));
            defer gpa.free(batch_bytes);
            for (check_rows) |row| {
                poison(s.counts.buf);
                poison(moe1);
                const cb1 = g.queue.commandBuffer();
                const e1 = cb1.compute(.serial);
                xpf.encodeNvfp4(e1, p, layer, .{ .buf = xm, .off = row * H * 2 }, .{ .ids = .{ .buf = eids, .off = row * TOPK * 4 }, .weights = .{ .buf = ew, .off = row * TOPK * 4 } }, .{ .buf = moe1 }, s, 1, CAPS);
                try submit(g, cb1, e1);
                c_ind.same(moe1.slice(u8, H * 2), batch_bytes[row * H * 2 ..][0 .. H * 2]);
            }

            if (rows == 37) {
                const kept = xms[1 * H];
                xms[1 * H] = 0x7fc0;
                poison(s.counts.buf);
                poison(moe);
                const cbn = g.queue.commandBuffer();
                const en = cbn.compute(.serial);
                xpf.encodeNvfp4(en, p, layer, .{ .buf = xm }, .{ .ids = .{ .buf = eids }, .weights = .{ .buf = ew } }, .{ .buf = moe }, s, rows, CAPS);
                try submit(g, cbn, en);
                xms[1 * H] = kept;
                const nan_bytes = moe.slice(u8, rows * H * 2);
                for (0..rows) |row| if (row != 1) {
                    c_nan.same(nan_bytes[row * H * 2 ..][0 .. H * 2], batch_bytes[row * H * 2 ..][0 .. H * 2]);
                };
                // Confirm row 1 actually turned non-finite so the check isn't vacuous.
                const row1 = moe.slice(u16, rows * H)[1 * H ..][0..H];
                c_nan.n += 1;
                if (!std.math.isNan(h.fb(row1[0]))) c_nan.bad += 1;
            }
        }
    }

    try c_ref.done();
    try c_ind.done();
    try c_nan.done();
}

fn caseMxfp8(g: *const h.Gpu, gpa: std.mem.Allocator, p: *const xpf.Pipes, s: xpf.Xp, r: std.Random) !void {
    const w13_bytes = E * 2 * I * H;
    const s13_bytes = E * 2 * I * (H / 32);
    const w2_bytes = E * H * I;
    const s2_bytes = E * H * (I / 32);
    const xm = try g.buf(ROWS_MAX * H * 2);
    const eids = try g.buf(SLOTS_MAX * 4);
    const ew = try g.buf(SLOTS_MAX * 4);
    const moe = try g.buf(ROWS_MAX * H * 2);
    const moe1 = try g.buf(H * 2);
    const w13 = try g.buf(w13_bytes);
    const s13 = try g.buf(s13_bytes);
    const w2 = try g.buf(w2_bytes);
    const s2 = try g.buf(s2_bytes);
    defer for ([_]mtl.Buffer{ xm, eids, ew, moe, moe1, w13, s13, w2, s2 }) |b| b.deinit();

    const xms = h.fill16(xm, ROWS_MAX * H, r, 0.5);
    const w13s = w13.slice(u8, w13_bytes);
    r.bytes(w13s);
    for (w13s) |*b| if (b.* & 0x7F == 0x7F) {
        b.* = 0x38;
    };
    const s13s = s13.slice(u8, s13_bytes);
    r.bytes(s13s);
    for (s13s) |*sc| sc.* = @intCast(113 + (sc.* % 6));

    const w2s = w2.slice(u8, w2_bytes);
    r.bytes(w2s);
    for (w2s) |*b| if (b.* & 0x7F == 0x7F) {
        b.* = 0x38;
    };
    const s2s = s2.slice(u8, s2_bytes);
    r.bytes(s2s);
    for (s2s) |*sc| sc.* = @intCast(113 + (sc.* % 6));

    const layer: xpf.Mxfp8Layer = .{
        .w13 = .{ .weight = .{ .buffer = w13, .offset = 0, .bytes = w13_bytes, .dtype = .f8_e4m3 }, .scale = .{ .buffer = s13, .offset = 0, .bytes = s13_bytes, .dtype = .u8 }, .n = 2 * I, .k = H },
        .w2 = .{ .weight = .{ .buffer = w2, .offset = 0, .bytes = w2_bytes, .dtype = .f8_e4m3 }, .scale = .{ .buffer = s2, .offset = 0, .bytes = s2_bytes, .dtype = .u8 }, .n = H, .k = I },
        .experts = E,
    };

    const act_ref = try gpa.alloc(f64, TOPK * I);
    defer gpa.free(act_ref);
    var c_ref: h.Check = .{ .name = "xp_fp_mxfp8_ref" };
    var c_ind: h.Check = .{ .name = "xp_fp_mxfp8_row_indep" };
    var c_nan: h.Check = .{ .name = "xp_fp_mxfp8_nan_contain" };

    for (BATCH_ROWS) |rows| {
        const slots = rows * TOPK;
        const ids = eids.slice(u32, slots);
        const weights = ew.slice(f32, slots);
        fillPicks(r, ids, weights, rows);

        poison(s.counts.buf);
        poison(moe);
        const cb = g.queue.commandBuffer();
        const enc = cb.compute(.serial);
        xpf.encodeMxfp8(enc, p, layer, .{ .buf = xm }, .{ .ids = .{ .buf = eids }, .weights = .{ .buf = ew } }, .{ .buf = moe }, s, rows, CAPS);
        try submit(g, cb, enc);

        const got_moe = moe.slice(u16, rows * H);
        var sbuf: [4]usize = undefined;
        const check_rows = sampledRows(rows, &sbuf);
        for (check_rows) |row| {
            for (0..TOPK) |k| {
                const e = ids[row * TOPK + k];
                for (0..I) |i| {
                    var sg_v: f64 = 0;
                    var su_v: f64 = 0;
                    const rg: usize = @as(usize, e) * 2 * I + 2 * i;
                    const ru: usize = rg + 1;
                    const wg = w13s[rg * H ..][0..H];
                    const wu = w13s[ru * H ..][0..H];
                    const scg = s13s[rg * (H / 32) ..][0 .. H / 32];
                    const scu = s13s[ru * (H / 32) ..][0 .. H / 32];
                    for (0..H / 32) |bk| {
                        const fg = e8m0Ref(scg[bk]);
                        const fu = e8m0Ref(scu[bk]);
                        const k0 = bk * 32;
                        for (0..32) |j| {
                            const xv = h.fb(xms[row * H + k0 + j]);
                            sg_v += xv * (fp8.e4m3(wg[k0 + j]) * fg);
                            su_v += xv * (fp8.e4m3(wu[k0 + j]) * fu);
                        }
                    }
                    act_ref[k * I + i] = h.rb(swigluRef(h.rb(sg_v), h.rb(su_v)));
                }
            }
            for (0..32) |idx| {
                const d = (idx * 71) % H;
                var acc: f64 = 0;
                for (0..TOPK) |k| {
                    const e = ids[row * TOPK + k];
                    const r2: usize = @as(usize, e) * H + d;
                    const wr = w2s[r2 * I ..][0..I];
                    const sr = s2s[r2 * (I / 32) ..][0 .. I / 32];
                    var dot: f64 = 0;
                    for (0..I / 32) |bk| {
                        const sc = e8m0Ref(sr[bk]);
                        const k0 = bk * 32;
                        for (0..32) |j| dot += act_ref[k * I + k0 + j] * (fp8.e4m3(wr[k0 + j]) * sc);
                    }
                    acc += h.rb(dot) * @as(f64, weights[row * TOPK + k]);
                }
                c_ref.b16a(got_moe[row * H + d], acc, 2, 1e-4);
            }
        }

        if (rows == 37 or rows == 128) {
            const batch_bytes = try gpa.dupe(u8, moe.slice(u8, rows * H * 2));
            defer gpa.free(batch_bytes);
            for (check_rows) |row| {
                poison(s.counts.buf);
                poison(moe1);
                const cb1 = g.queue.commandBuffer();
                const e1 = cb1.compute(.serial);
                xpf.encodeMxfp8(e1, p, layer, .{ .buf = xm, .off = row * H * 2 }, .{ .ids = .{ .buf = eids, .off = row * TOPK * 4 }, .weights = .{ .buf = ew, .off = row * TOPK * 4 } }, .{ .buf = moe1 }, s, 1, CAPS);
                try submit(g, cb1, e1);
                c_ind.same(moe1.slice(u8, H * 2), batch_bytes[row * H * 2 ..][0 .. H * 2]);
            }

            if (rows == 37) {
                const kept = xms[1 * H];
                xms[1 * H] = 0x7fc0;
                poison(s.counts.buf);
                poison(moe);
                const cbn = g.queue.commandBuffer();
                const en = cbn.compute(.serial);
                xpf.encodeMxfp8(en, p, layer, .{ .buf = xm }, .{ .ids = .{ .buf = eids }, .weights = .{ .buf = ew } }, .{ .buf = moe }, s, rows, CAPS);
                try submit(g, cbn, en);
                xms[1 * H] = kept;
                const nan_bytes = moe.slice(u8, rows * H * 2);
                for (0..rows) |row| if (row != 1) {
                    c_nan.same(nan_bytes[row * H * 2 ..][0 .. H * 2], batch_bytes[row * H * 2 ..][0 .. H * 2]);
                };
                const row1 = moe.slice(u16, rows * H)[1 * H ..][0..H];
                c_nan.n += 1;
                if (!std.math.isNan(h.fb(row1[0]))) c_nan.bad += 1;
            }
        }
    }

    try c_ref.done();
    try c_ind.done();
    try c_nan.done();
}
