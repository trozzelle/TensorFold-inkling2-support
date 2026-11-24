//! NVFP4 and MXFP8 routed-expert kernel checks against FP64 references, row independence, and NaN containment.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const nvfp4 = tf.quant.nvfp4;
const fp8 = tf.quant.fp8;
const h = @import("harness.zig");

const H = 2304;
const I = 1152;
const E = 12; // compact synthetic expert table with Inkling 2 Mini K/N dimensions
const TOPK = 8;
const ROWS = 4;
const SLOTS = ROWS * TOPK;

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    try caseNvfp4(g, r);
    try caseMxfp8(g, r);
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

fn caseNvfp4(g: *const h.Gpu, r: std.Random) !void {
    const xm = try g.buf(ROWS * H * 2);
    const eids = try g.buf(SLOTS * 4);
    const w13 = try g.buf(E * 2 * I * (H / 2));
    const s13 = try g.buf(E * 2 * I * (H / 16));
    const g13 = try g.buf(E * 4);
    const act = try g.buf(SLOTS * I * 2);
    const w2 = try g.buf(E * H * (I / 2));
    const s2 = try g.buf(E * H * (I / 16));
    const g2 = try g.buf(E * 4);
    const ey = try g.buf(SLOTS * H * 2);
    const act1 = try g.buf(TOPK * I * 2);
    const ey1 = try g.buf(TOPK * H * 2);
    defer for ([_]mtl.Buffer{ xm, eids, w13, s13, g13, act, w2, s2, g2, ey, act1, ey1 }) |b| b.deinit();

    const xms = h.fill16(xm, ROWS * H, r, 0.5);
    const ids = eids.slice(u32, SLOTS);
    for (ids) |*id| id.* = r.uintLessThan(u32, E);
    const w13s = w13.slice(u8, E * 2 * I * (H / 2));
    r.bytes(w13s);
    const s13s = s13.slice(u8, E * 2 * I * (H / 16));
    for (s13s) |*sc| sc.* = @intCast(24 + r.uintLessThan(u8, 48)); // positive finite e4m3 scales
    const g13s = g13.slice(f32, E);
    for (g13s) |*gs| gs.* = @floatCast(0.01 + r.float(f64) * 0.05);

    const w2s = w2.slice(u8, E * H * (I / 2));
    r.bytes(w2s);
    const s2s = s2.slice(u8, E * H * (I / 16));
    for (s2s) |*sc| sc.* = @intCast(24 + r.uintLessThan(u8, 48));
    const g2s = g2.slice(f32, E);
    for (g2s) |*gs| gs.* = @floatCast(0.01 + r.float(f64) * 0.05);

    const a13: h.Args = .{ .rows = SLOTS, .width = I, .dim = H, .queries = TOPK, .cap = 4.0, .thr = 25.0 };
    try g.run("ik_nvfp4_w13_swiglu", a13, &.{ .{ .buf = xm }, .{ .buf = eids }, .{ .buf = w13 }, .{ .buf = s13 }, .{ .buf = g13 }, .{ .buf = act } }, .{ (I + 3) / 4, SLOTS, 1 }, .{ 128, 1, 1 }, true);
    const a2: h.Args = .{ .rows = SLOTS, .width = H, .dim = I };
    try g.run("ik_nvfp4_w2", a2, &.{ .{ .buf = act }, .{ .buf = eids }, .{ .buf = w2 }, .{ .buf = s2 }, .{ .buf = g2 }, .{ .buf = ey } }, .{ (H + 3) / 4, SLOTS, 1 }, .{ 128, 1, 1 }, true);

    var c13: h.Check = .{ .name = "nvfp4_w13_swiglu" };
    const acts = act.slice(u16, SLOTS * I);
    for (0..SLOTS) |slot| {
        const e = ids[slot];
        const tok = slot / TOPK;
        for (0..32) |idx| { // sample 32 output pairs across [0..I)
            const i = (idx * 37) % I;
            var sg_v: f64 = 0;
            var su_v: f64 = 0;
            for (0..H) |k| {
                const xv = h.fb(xms[tok * H + k]);
                const bg = w13s[(e * 2 * I + 2 * i) * (H / 2) + k / 2];
                const bu = w13s[(e * 2 * I + 2 * i + 1) * (H / 2) + k / 2];
                const ng: u8 = if (k & 1 == 0) bg & 0xF else bg >> 4;
                const nu: u8 = if (k & 1 == 0) bu & 0xF else bu >> 4;
                sg_v += xv * nvfp4.e2m1(ng) * fp8.e4m3(s13s[(e * 2 * I + 2 * i) * (H / 16) + k / 16]);
                su_v += xv * nvfp4.e2m1(nu) * fp8.e4m3(s13s[(e * 2 * I + 2 * i + 1) * (H / 16) + k / 16]);
            }
            const want = swigluRef(h.rb(sg_v * g13s[e]), h.rb(su_v * g13s[e]));
            c13.b16a(acts[slot * I + i], want, 2, 1e-4);
        }
    }
    try c13.done();

    var c2: h.Check = .{ .name = "nvfp4_w2" };
    const eys = ey.slice(u16, SLOTS * H);
    for (0..SLOTS) |slot| {
        const e = ids[slot];
        for (0..32) |idx| {
            const d = (idx * 71) % H;
            var acc: f64 = 0;
            for (0..I) |k| {
                const b = w2s[(e * H + d) * (I / 2) + k / 2];
                const nib: u8 = if (k & 1 == 0) b & 0xF else b >> 4;
                acc += h.fb(acts[slot * I + k]) * nvfp4.e2m1(nib) * fp8.e4m3(s2s[(e * H + d) * (I / 16) + k / 16]);
            }
            c2.b16a(eys[slot * H + d], acc * g2s[e], 1, 1e-4);
        }
    }
    try c2.done();

    // Row independence: each row run alone (1 row = TOPK slots) gives bit-identical act and ey to the batch
    const all_act = try std.heap.page_allocator.dupe(u8, act.slice(u8, SLOTS * I * 2));
    defer std.heap.page_allocator.free(all_act);
    const all_ey = try std.heap.page_allocator.dupe(u8, ey.slice(u8, SLOTS * H * 2));
    defer std.heap.page_allocator.free(all_ey);

    var ci: h.Check = .{ .name = "nvfp4_row_indep" };
    const a13_one: h.Args = .{ .rows = TOPK, .width = I, .dim = H, .queries = TOPK, .cap = 4.0, .thr = 25.0 };
    const a2_one: h.Args = .{ .rows = TOPK, .width = H, .dim = I };
    for (0..ROWS) |row| {
        try g.run("ik_nvfp4_w13_swiglu", a13_one, &.{ .{ .buf = xm, .off = row * H * 2 }, .{ .buf = eids, .off = row * TOPK * 4 }, .{ .buf = w13 }, .{ .buf = s13 }, .{ .buf = g13 }, .{ .buf = act1 } }, .{ (I + 3) / 4, TOPK, 1 }, .{ 128, 1, 1 }, true);
        try g.run("ik_nvfp4_w2", a2_one, &.{ .{ .buf = act1 }, .{ .buf = eids, .off = row * TOPK * 4 }, .{ .buf = w2 }, .{ .buf = s2 }, .{ .buf = g2 }, .{ .buf = ey1 } }, .{ (H + 3) / 4, TOPK, 1 }, .{ 128, 1, 1 }, true);
        ci.same(act1.slice(u8, TOPK * I * 2), all_act[row * TOPK * I * 2 ..][0 .. TOPK * I * 2]);
        ci.same(ey1.slice(u8, TOPK * H * 2), all_ey[row * TOPK * H * 2 ..][0 .. TOPK * H * 2]);
    }
    try ci.done();

    // NaN containment on row 1
    const kept = xms[1 * H];
    xms[1 * H] = 0x7fc0;
    try g.run("ik_nvfp4_w13_swiglu", a13, &.{ .{ .buf = xm }, .{ .buf = eids }, .{ .buf = w13 }, .{ .buf = s13 }, .{ .buf = g13 }, .{ .buf = act } }, .{ (I + 3) / 4, SLOTS, 1 }, .{ 128, 1, 1 }, true);
    try g.run("ik_nvfp4_w2", a2, &.{ .{ .buf = act }, .{ .buf = eids }, .{ .buf = w2 }, .{ .buf = s2 }, .{ .buf = g2 }, .{ .buf = ey } }, .{ (H + 3) / 4, SLOTS, 1 }, .{ 128, 1, 1 }, true);
    var cn: h.Check = .{ .name = "nvfp4_nan_contain" };
    for (0..ROWS) |row| if (row != 1) {
        cn.same(act.slice(u8, SLOTS * I * 2)[row * TOPK * I * 2 ..][0 .. TOPK * I * 2], all_act[row * TOPK * I * 2 ..][0 .. TOPK * I * 2]);
        cn.same(ey.slice(u8, SLOTS * H * 2)[row * TOPK * H * 2 ..][0 .. TOPK * H * 2], all_ey[row * TOPK * H * 2 ..][0 .. TOPK * H * 2]);
    };
    xms[1 * H] = kept;
    try cn.done();
}

fn caseMxfp8(g: *const h.Gpu, r: std.Random) !void {
    const xm = try g.buf(ROWS * H * 2);
    const eids = try g.buf(SLOTS * 4);
    const w13 = try g.buf(E * 2 * I * H);
    const s13 = try g.buf(E * 2 * I * (H / 32));
    const act = try g.buf(SLOTS * I * 2);
    const w2 = try g.buf(E * H * I);
    const s2 = try g.buf(E * H * (I / 32));
    const ey = try g.buf(SLOTS * H * 2);
    const act1 = try g.buf(TOPK * I * 2);
    const ey1 = try g.buf(TOPK * H * 2);
    defer for ([_]mtl.Buffer{ xm, eids, w13, s13, act, w2, s2, ey, act1, ey1 }) |b| b.deinit();

    const xms = h.fill16(xm, ROWS * H, r, 0.5);
    const ids = eids.slice(u32, SLOTS);
    for (ids) |*id| id.* = r.uintLessThan(u32, E);
    const w13s = w13.slice(u8, E * 2 * I * H);
    for (w13s) |*b| {
        const v = r.int(u8);
        b.* = if (v & 0x7F == 0x7F) 0x38 else v; // finite e4m3
    }
    const s13s = s13.slice(u8, E * 2 * I * (H / 32));
    for (s13s) |*sc| sc.* = @intCast(113 + r.uintLessThan(u8, 6)); // typical UE8M0 range

    const w2s = w2.slice(u8, E * H * I);
    for (w2s) |*b| {
        const v = r.int(u8);
        b.* = if (v & 0x7F == 0x7F) 0x38 else v;
    }
    const s2s = s2.slice(u8, E * H * (I / 32));
    for (s2s) |*sc| sc.* = @intCast(113 + r.uintLessThan(u8, 6));

    const a13: h.Args = .{ .rows = SLOTS, .width = I, .dim = H, .queries = TOPK, .cap = 4.0, .thr = 25.0 };
    try g.run("ik_mxfp8_w13_swiglu", a13, &.{ .{ .buf = xm }, .{ .buf = eids }, .{ .buf = w13 }, .{ .buf = s13 }, .{ .buf = act } }, .{ (I + 3) / 4, SLOTS, 1 }, .{ 128, 1, 1 }, true);
    const a2: h.Args = .{ .rows = SLOTS, .width = H, .dim = I };
    try g.run("ik_mxfp8_w2", a2, &.{ .{ .buf = act }, .{ .buf = eids }, .{ .buf = w2 }, .{ .buf = s2 }, .{ .buf = ey } }, .{ (H + 3) / 4, SLOTS, 1 }, .{ 128, 1, 1 }, true);

    var c13: h.Check = .{ .name = "mxfp8_w13_swiglu" };
    const acts = act.slice(u16, SLOTS * I);
    for (0..SLOTS) |slot| {
        const e = ids[slot];
        const tok = slot / TOPK;
        for (0..32) |idx| {
            const i = (idx * 37) % I;
            var sg_v: f64 = 0;
            var su_v: f64 = 0;
            for (0..H) |k| {
                const xv = h.fb(xms[tok * H + k]);
                sg_v += xv * fp8.e4m3(w13s[(e * 2 * I + 2 * i) * H + k]) * e8m0Ref(s13s[(e * 2 * I + 2 * i) * (H / 32) + k / 32]);
                su_v += xv * fp8.e4m3(w13s[(e * 2 * I + 2 * i + 1) * H + k]) * e8m0Ref(s13s[(e * 2 * I + 2 * i + 1) * (H / 32) + k / 32]);
            }
            const want = swigluRef(h.rb(sg_v), h.rb(su_v));
            c13.b16a(acts[slot * I + i], want, 2, 1e-4);
        }
    }
    try c13.done();

    var c2: h.Check = .{ .name = "mxfp8_w2" };
    const eys = ey.slice(u16, SLOTS * H);
    for (0..SLOTS) |slot| {
        const e = ids[slot];
        for (0..32) |idx| {
            const d = (idx * 71) % H;
            var acc: f64 = 0;
            for (0..I) |k| acc += h.fb(acts[slot * I + k]) * fp8.e4m3(w2s[(e * H + d) * I + k]) * e8m0Ref(s2s[(e * H + d) * (I / 32) + k / 32]);
            c2.b16a(eys[slot * H + d], acc, 1, 1e-4);
        }
    }
    try c2.done();

    // Row independence: each row run alone (1 row = TOPK slots) gives bit-identical act and ey to the batch
    const all_act = try std.heap.page_allocator.dupe(u8, act.slice(u8, SLOTS * I * 2));
    defer std.heap.page_allocator.free(all_act);
    const all_ey = try std.heap.page_allocator.dupe(u8, ey.slice(u8, SLOTS * H * 2));
    defer std.heap.page_allocator.free(all_ey);

    var ci: h.Check = .{ .name = "mxfp8_row_indep" };
    const a13_one: h.Args = .{ .rows = TOPK, .width = I, .dim = H, .queries = TOPK, .cap = 4.0, .thr = 25.0 };
    const a2_one: h.Args = .{ .rows = TOPK, .width = H, .dim = I };
    for (0..ROWS) |row| {
        try g.run("ik_mxfp8_w13_swiglu", a13_one, &.{ .{ .buf = xm, .off = row * H * 2 }, .{ .buf = eids, .off = row * TOPK * 4 }, .{ .buf = w13 }, .{ .buf = s13 }, .{ .buf = act1 } }, .{ (I + 3) / 4, TOPK, 1 }, .{ 128, 1, 1 }, true);
        try g.run("ik_mxfp8_w2", a2_one, &.{ .{ .buf = act1 }, .{ .buf = eids, .off = row * TOPK * 4 }, .{ .buf = w2 }, .{ .buf = s2 }, .{ .buf = ey1 } }, .{ (H + 3) / 4, TOPK, 1 }, .{ 128, 1, 1 }, true);
        ci.same(act1.slice(u8, TOPK * I * 2), all_act[row * TOPK * I * 2 ..][0 .. TOPK * I * 2]);
        ci.same(ey1.slice(u8, TOPK * H * 2), all_ey[row * TOPK * H * 2 ..][0 .. TOPK * H * 2]);
    }
    try ci.done();

    // NaN containment on row 1
    const kept = xms[1 * H];
    xms[1 * H] = 0x7fc0;
    try g.run("ik_mxfp8_w13_swiglu", a13, &.{ .{ .buf = xm }, .{ .buf = eids }, .{ .buf = w13 }, .{ .buf = s13 }, .{ .buf = act } }, .{ (I + 3) / 4, SLOTS, 1 }, .{ 128, 1, 1 }, true);
    try g.run("ik_mxfp8_w2", a2, &.{ .{ .buf = act }, .{ .buf = eids }, .{ .buf = w2 }, .{ .buf = s2 }, .{ .buf = ey } }, .{ (H + 3) / 4, SLOTS, 1 }, .{ 128, 1, 1 }, true);
    var cn: h.Check = .{ .name = "mxfp8_nan_contain" };
    for (0..ROWS) |row| if (row != 1) {
        cn.same(act.slice(u8, SLOTS * I * 2)[row * TOPK * I * 2 ..][0 .. TOPK * I * 2], all_act[row * TOPK * I * 2 ..][0 .. TOPK * I * 2]);
        cn.same(ey.slice(u8, SLOTS * H * 2)[row * TOPK * H * 2 ..][0 .. TOPK * H * 2], all_ey[row * TOPK * H * 2 ..][0 .. TOPK * H * 2]);
    };
    xms[1 * H] = kept;
    try cn.done();
}
