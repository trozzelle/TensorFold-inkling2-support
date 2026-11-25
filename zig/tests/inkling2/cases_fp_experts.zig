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
    try caseGroupedNvfp4(g, r);
    try caseGroupedMxfp8(g, r);
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

const G_E = 128;
const G_ROWS_MAX = 16;
const G_SLOTS_MAX = G_ROWS_MAX * TOPK;
const G_ROW_COUNTS = [_]usize{ 1, 2, 4, 8, 16 };

// Scratch layout for ik_fp_group (256-byte aligned subregions, 2304 bytes total).
const SCRATCH_U_EID: usize = 0; // [128] u32
const SCRATCH_U_OFF: usize = 512; // [128] u32
const SCRATCH_U_CNT: usize = 1024; // [128] u32
const SCRATCH_U_SLOTS: usize = 1536; // [128] u32
const SCRATCH_DARGS: usize = 2048; // [8] u32 (w13 at +0, w2 at +16)
const SCRATCH_BYTES: usize = 2304;

/// Skewed routing picks with heavy collisions on experts 0 and 1 (up to 10 slots) to exercise 4-slot tile splitting
/// and shared tiles between row 0 and row 1 for NaN containment.
fn fillHeavyPicks(r: std.Random, ids: []u32, rows: usize) void {
    var cdf: [G_E]f64 = undefined;
    var total: f64 = 0;
    for (&cdf, 0..) |*c, k| {
        total += 1.0 / std.math.pow(f64, @as(f64, @floatFromInt(k + 1)), 1.25);
        c.* = total;
    }
    for (0..rows) |row| {
        var k: usize = 0;
        if (row < @min(rows, 10)) {
            ids[row * TOPK + k] = 0;
            k += 1;
        }
        if (rows >= 6 and row < @min(rows, 6)) {
            ids[row * TOPK + k] = 1;
            k += 1;
        }
        if (row == rows - 1 and k < TOPK) {
            ids[row * TOPK + k] = G_E - 1;
            k += 1;
        }
        while (k < TOPK) {
            const u = r.float(f64) * total;
            var lo: usize = 0;
            var hi: usize = G_E - 1;
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
}

fn checkGroupTable(scratch: mtl.Buffer, ids: []const u32, slots: usize, rows: usize) !u32 {
    const raw = scratch.slice(u32, SCRATCH_BYTES / 4);
    const u_eid = raw[SCRATCH_U_EID / 4 ..][0..G_SLOTS_MAX];
    const u_off = raw[SCRATCH_U_OFF / 4 ..][0..G_SLOTS_MAX];
    const u_cnt = raw[SCRATCH_U_CNT / 4 ..][0..G_SLOTS_MAX];
    const u_slots = raw[SCRATCH_U_SLOTS / 4 ..][0..G_SLOTS_MAX];
    const dargs = raw[SCRATCH_DARGS / 4 ..][0..8];

    const u = dargs[1];
    if (dargs[0] != (I + 3) / 4 or dargs[2] != 1 or dargs[4] != (H + 3) / 4 or dargs[5] != u or dargs[6] != 1)
        return error.KernelMismatch;
    if (u == 0 or u > slots) return error.KernelMismatch;

    var seen: [G_SLOTS_MAX]bool = @splat(false);
    var total_slots: usize = 0;
    var split_found = false;
    var prev_eid: u32 = 0;
    for (0..u) |idx| {
        const e = u_eid[idx];
        const off = u_off[idx];
        const cnt = u_cnt[idx];
        if (e >= G_E or cnt == 0 or cnt > 4 or off + cnt > slots) return error.KernelMismatch;
        if (idx > 0 and e < prev_eid) return error.KernelMismatch;
        if (idx > 0 and e == prev_eid) split_found = true;
        prev_eid = e;
        total_slots += cnt;
        for (0..cnt) |m| {
            const s = u_slots[off + m];
            if (s >= slots or seen[s] or ids[s] != e) return error.KernelMismatch;
            seen[s] = true;
        }
    }
    if (total_slots != slots) return error.KernelMismatch;
    if (rows >= 8 and !split_found) return error.KernelMismatch;
    return u;
}

fn runGroupedNvfp4(
    g: *const h.Gpu,
    p_grp: mtl.Pipeline,
    p_w13: mtl.Pipeline,
    p_w2: mtl.Pipeline,
    xm: mtl.Buffer,
    eids: mtl.Buffer,
    scratch: mtl.Buffer,
    w13: mtl.Buffer,
    s13: mtl.Buffer,
    g13: mtl.Buffer,
    act: mtl.Buffer,
    w2: mtl.Buffer,
    s2: mtl.Buffer,
    g2: mtl.Buffer,
    ey: mtl.Buffer,
    slots: usize,
    reps: usize,
) !f64 {
    const a_grp: h.Args = .{ .rows = @intCast(slots), .width = G_E, .dim = I, .stride = H };
    const a13: h.Args = .{ .rows = @intCast(slots), .width = I, .dim = H, .queries = TOPK, .cap = 4.0, .thr = 25.0 };
    const a2: h.Args = .{ .rows = @intCast(slots), .width = H, .dim = I };
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    for (0..reps) |_| {
        e.setPipeline(p_grp);
        e.setValue(a_grp, 0);
        e.setBuffer(eids, 0, 1);
        e.setBuffer(scratch, SCRATCH_U_EID, 2);
        e.setBuffer(scratch, SCRATCH_U_OFF, 3);
        e.setBuffer(scratch, SCRATCH_U_CNT, 4);
        e.setBuffer(scratch, SCRATCH_U_SLOTS, 5);
        e.setBuffer(scratch, SCRATCH_DARGS, 6);
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));

        e.setPipeline(p_w13);
        e.setValue(a13, 0);
        e.setBuffer(xm, 0, 1);
        e.setBuffer(scratch, SCRATCH_U_EID, 2);
        e.setBuffer(scratch, SCRATCH_U_OFF, 3);
        e.setBuffer(scratch, SCRATCH_U_CNT, 4);
        e.setBuffer(scratch, SCRATCH_U_SLOTS, 5);
        e.setBuffer(w13, 0, 6);
        e.setBuffer(s13, 0, 7);
        e.setBuffer(g13, 0, 8);
        e.setBuffer(act, 0, 9);
        e.dispatchIndirect(scratch, SCRATCH_DARGS, mtl.Size.of(128, 1, 1));

        e.setPipeline(p_w2);
        e.setValue(a2, 0);
        e.setBuffer(act, 0, 1);
        e.setBuffer(scratch, SCRATCH_U_EID, 2);
        e.setBuffer(scratch, SCRATCH_U_OFF, 3);
        e.setBuffer(scratch, SCRATCH_U_CNT, 4);
        e.setBuffer(scratch, SCRATCH_U_SLOTS, 5);
        e.setBuffer(w2, 0, 6);
        e.setBuffer(s2, 0, 7);
        e.setBuffer(g2, 0, 8);
        e.setBuffer(ey, 0, 9);
        e.dispatchIndirect(scratch, SCRATCH_DARGS + 16, mtl.Size.of(128, 1, 1));
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("grouped nvfp4: {s}", .{msg});
        return error.GpuFailed;
    }
    return (cb.gpuEnd() - cb.gpuStart()) * 1e6 / @as(f64, @floatFromInt(reps));
}

fn runUngroupedNvfp4(
    g: *const h.Gpu,
    p_w13: mtl.Pipeline,
    p_w2: mtl.Pipeline,
    xm: mtl.Buffer,
    eids: mtl.Buffer,
    w13: mtl.Buffer,
    s13: mtl.Buffer,
    g13: mtl.Buffer,
    act: mtl.Buffer,
    w2: mtl.Buffer,
    s2: mtl.Buffer,
    g2: mtl.Buffer,
    ey: mtl.Buffer,
    slots: usize,
    reps: usize,
) !f64 {
    const a13: h.Args = .{ .rows = @intCast(slots), .width = I, .dim = H, .queries = TOPK, .cap = 4.0, .thr = 25.0 };
    const a2: h.Args = .{ .rows = @intCast(slots), .width = H, .dim = I };
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    for (0..reps) |_| {
        e.setPipeline(p_w13);
        e.setValue(a13, 0);
        e.setBuffer(xm, 0, 1);
        e.setBuffer(eids, 0, 2);
        e.setBuffer(w13, 0, 3);
        e.setBuffer(s13, 0, 4);
        e.setBuffer(g13, 0, 5);
        e.setBuffer(act, 0, 6);
        e.dispatchGroups(mtl.Size.of((I + 3) / 4, slots, 1), mtl.Size.of(128, 1, 1));

        e.setPipeline(p_w2);
        e.setValue(a2, 0);
        e.setBuffer(act, 0, 1);
        e.setBuffer(eids, 0, 2);
        e.setBuffer(w2, 0, 3);
        e.setBuffer(s2, 0, 4);
        e.setBuffer(g2, 0, 5);
        e.setBuffer(ey, 0, 6);
        e.dispatchGroups(mtl.Size.of((H + 3) / 4, slots, 1), mtl.Size.of(128, 1, 1));
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("ungrouped nvfp4: {s}", .{msg});
        return error.GpuFailed;
    }
    return (cb.gpuEnd() - cb.gpuStart()) * 1e6 / @as(f64, @floatFromInt(reps));
}

fn runGroupedMxfp8(
    g: *const h.Gpu,
    p_grp: mtl.Pipeline,
    p_w13: mtl.Pipeline,
    p_w2: mtl.Pipeline,
    xm: mtl.Buffer,
    eids: mtl.Buffer,
    scratch: mtl.Buffer,
    w13: mtl.Buffer,
    s13: mtl.Buffer,
    act: mtl.Buffer,
    w2: mtl.Buffer,
    s2: mtl.Buffer,
    ey: mtl.Buffer,
    slots: usize,
) !void {
    const a_grp: h.Args = .{ .rows = @intCast(slots), .width = G_E, .dim = I, .stride = H };
    const a13: h.Args = .{ .rows = @intCast(slots), .width = I, .dim = H, .queries = TOPK, .cap = 4.0, .thr = 25.0 };
    const a2: h.Args = .{ .rows = @intCast(slots), .width = H, .dim = I };
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);

    e.setPipeline(p_grp);
    e.setValue(a_grp, 0);
    e.setBuffer(eids, 0, 1);
    e.setBuffer(scratch, SCRATCH_U_EID, 2);
    e.setBuffer(scratch, SCRATCH_U_OFF, 3);
    e.setBuffer(scratch, SCRATCH_U_CNT, 4);
    e.setBuffer(scratch, SCRATCH_U_SLOTS, 5);
    e.setBuffer(scratch, SCRATCH_DARGS, 6);
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));

    e.setPipeline(p_w13);
    e.setValue(a13, 0);
    e.setBuffer(xm, 0, 1);
    e.setBuffer(scratch, SCRATCH_U_EID, 2);
    e.setBuffer(scratch, SCRATCH_U_OFF, 3);
    e.setBuffer(scratch, SCRATCH_U_CNT, 4);
    e.setBuffer(scratch, SCRATCH_U_SLOTS, 5);
    e.setBuffer(w13, 0, 6);
    e.setBuffer(s13, 0, 7);
    e.setBuffer(act, 0, 8);
    e.dispatchIndirect(scratch, SCRATCH_DARGS, mtl.Size.of(128, 1, 1));

    e.setPipeline(p_w2);
    e.setValue(a2, 0);
    e.setBuffer(act, 0, 1);
    e.setBuffer(scratch, SCRATCH_U_EID, 2);
    e.setBuffer(scratch, SCRATCH_U_OFF, 3);
    e.setBuffer(scratch, SCRATCH_U_CNT, 4);
    e.setBuffer(scratch, SCRATCH_U_SLOTS, 5);
    e.setBuffer(w2, 0, 6);
    e.setBuffer(s2, 0, 7);
    e.setBuffer(ey, 0, 8);
    e.dispatchIndirect(scratch, SCRATCH_DARGS + 16, mtl.Size.of(128, 1, 1));

    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("grouped mxfp8: {s}", .{msg});
        return error.GpuFailed;
    }
}

fn caseGroupedNvfp4(g: *const h.Gpu, r: std.Random) !void {
    const p_grp = try mtl.Pipeline.init(g.device, g.lib, "ik_fp_group", false);
    const p_u13 = try mtl.Pipeline.init(g.device, g.lib, "ik_nvfp4_w13_swiglu", false);
    const p_u2 = try mtl.Pipeline.init(g.device, g.lib, "ik_nvfp4_w2", false);
    const p_g13 = try mtl.Pipeline.init(g.device, g.lib, "ik_nvfp4_w13_swiglu_g", false);
    const p_g2 = try mtl.Pipeline.init(g.device, g.lib, "ik_nvfp4_w2_g", false);
    defer for ([_]mtl.Pipeline{ p_grp, p_u13, p_u2, p_g13, p_g2 }) |p| p.deinit();

    const xm = try g.buf(G_ROWS_MAX * H * 2);
    const eids = try g.buf(G_SLOTS_MAX * 4);
    const scratch = try g.buf(SCRATCH_BYTES);
    const w13 = try g.buf(G_E * 2 * I * (H / 2));
    const s13 = try g.buf(G_E * 2 * I * (H / 16));
    const g13 = try g.buf(G_E * 4);
    const w2 = try g.buf(G_E * H * (I / 2));
    const s2 = try g.buf(G_E * H * (I / 16));
    const g2 = try g.buf(G_E * 4);
    const act_ref = try g.buf(G_SLOTS_MAX * I * 2);
    const ey_ref = try g.buf(G_SLOTS_MAX * H * 2);
    const act_g = try g.buf(G_SLOTS_MAX * I * 2);
    const ey_g = try g.buf(G_SLOTS_MAX * H * 2);
    defer for ([_]mtl.Buffer{ xm, eids, scratch, w13, s13, g13, w2, s2, g2, act_ref, ey_ref, act_g, ey_g }) |b| b.deinit();

    const xms = h.fill16(xm, G_ROWS_MAX * H, r, 0.5);
    r.bytes(w13.slice(u8, G_E * 2 * I * (H / 2)));
    for (s13.slice(u8, G_E * 2 * I * (H / 16))) |*sc| sc.* = @intCast(24 + r.uintLessThan(u8, 48));
    for (g13.slice(f32, G_E)) |*gs| gs.* = @floatCast(0.01 + r.float(f64) * 0.05);
    r.bytes(w2.slice(u8, G_E * H * (I / 2)));
    for (s2.slice(u8, G_E * H * (I / 16))) |*sc| sc.* = @intCast(24 + r.uintLessThan(u8, 48));
    for (g2.slice(f32, G_E)) |*gs| gs.* = @floatCast(0.01 + r.float(f64) * 0.05);

    var cg: h.Check = .{ .name = "nvfp4_grouped_bit_exact" };
    for (G_ROW_COUNTS) |rows| {
        const slots = rows * TOPK;
        const ids = eids.slice(u32, slots);
        fillHeavyPicks(r, ids, rows);

        @memset(scratch.slice(u8, SCRATCH_BYTES), 0xff);
        @memset(act_g.slice(u8, slots * I * 2), 0xff);
        @memset(ey_g.slice(u8, slots * H * 2), 0xff);

        _ = try runUngroupedNvfp4(g, p_u13, p_u2, xm, eids, w13, s13, g13, act_ref, w2, s2, g2, ey_ref, slots, 1);
        _ = try runGroupedNvfp4(g, p_grp, p_g13, p_g2, xm, eids, scratch, w13, s13, g13, act_g, w2, s2, g2, ey_g, slots, 1);
        const tiles = try checkGroupTable(scratch, ids, slots, rows);

        cg.same(act_g.slice(u8, slots * I * 2), act_ref.slice(u8, slots * I * 2));
        cg.same(ey_g.slice(u8, slots * H * 2), ey_ref.slice(u8, slots * H * 2));

        if (rows == 4 or rows == 8) {
            const us_u = try runUngroupedNvfp4(g, p_u13, p_u2, xm, eids, w13, s13, g13, act_ref, w2, s2, g2, ey_ref, slots, 5);
            const us_g = try runGroupedNvfp4(g, p_grp, p_g13, p_g2, xm, eids, scratch, w13, s13, g13, act_g, w2, s2, g2, ey_g, slots, 5);
            std.debug.print("INFO nvfp4_grouped rows={d} slots={d} tiles={d}: ungrouped {d:.1} us, grouped {d:.1} us ({d:.2}x)\n", .{
                rows, slots, tiles, us_u, us_g, us_u / us_g,
            });
        }
    }
    try cg.done();

    // NaN containment on rows=8 where row 1 shares expert 0 inside the same tile as rows 0, 2, 3
    const nan_rows = 8;
    const nan_slots = nan_rows * TOPK;
    fillHeavyPicks(r, eids.slice(u32, nan_slots), nan_rows);
    _ = try runGroupedNvfp4(g, p_grp, p_g13, p_g2, xm, eids, scratch, w13, s13, g13, act_ref, w2, s2, g2, ey_ref, nan_slots, 1);
    const kept = xms[1 * H];
    xms[1 * H] = 0x7fc0;
    _ = try runGroupedNvfp4(g, p_grp, p_g13, p_g2, xm, eids, scratch, w13, s13, g13, act_g, w2, s2, g2, ey_g, nan_slots, 1);
    xms[1 * H] = kept;
    var cn: h.Check = .{ .name = "nvfp4_grouped_nan_contain" };
    for (0..nan_rows) |row| if (row != 1) {
        cn.same(act_g.slice(u8, nan_slots * I * 2)[row * TOPK * I * 2 ..][0 .. TOPK * I * 2], act_ref.slice(u8, nan_slots * I * 2)[row * TOPK * I * 2 ..][0 .. TOPK * I * 2]);
        cn.same(ey_g.slice(u8, nan_slots * H * 2)[row * TOPK * H * 2 ..][0 .. TOPK * H * 2], ey_ref.slice(u8, nan_slots * H * 2)[row * TOPK * H * 2 ..][0 .. TOPK * H * 2]);
    };
    try cn.done();
}

fn caseGroupedMxfp8(g: *const h.Gpu, r: std.Random) !void {
    const p_grp = try mtl.Pipeline.init(g.device, g.lib, "ik_fp_group", false);
    const p_g13 = try mtl.Pipeline.init(g.device, g.lib, "ik_mxfp8_w13_swiglu_g", false);
    const p_g2 = try mtl.Pipeline.init(g.device, g.lib, "ik_mxfp8_w2_g", false);
    defer for ([_]mtl.Pipeline{ p_grp, p_g13, p_g2 }) |p| p.deinit();

    const xm = try g.buf(G_ROWS_MAX * H * 2);
    const eids = try g.buf(G_SLOTS_MAX * 4);
    const scratch = try g.buf(SCRATCH_BYTES);
    const w13 = try g.buf(G_E * 2 * I * H);
    const s13 = try g.buf(G_E * 2 * I * (H / 32));
    const w2 = try g.buf(G_E * H * I);
    const s2 = try g.buf(G_E * H * (I / 32));
    const act_ref = try g.buf(G_SLOTS_MAX * I * 2);
    const ey_ref = try g.buf(G_SLOTS_MAX * H * 2);
    const act_g = try g.buf(G_SLOTS_MAX * I * 2);
    const ey_g = try g.buf(G_SLOTS_MAX * H * 2);
    defer for ([_]mtl.Buffer{ xm, eids, scratch, w13, s13, w2, s2, act_ref, ey_ref, act_g, ey_g }) |b| b.deinit();

    const xms = h.fill16(xm, G_ROWS_MAX * H, r, 0.5);
    const w13s = w13.slice(u8, G_E * 2 * I * H);
    r.bytes(w13s);
    for (w13s) |*b| {
        if ((b.* & 0x7F) == 0x7F) b.* = 0x38;
    }
    const s13s = s13.slice(u8, G_E * 2 * I * (H / 32));
    for (s13s) |*sc| sc.* = @intCast(113 + r.uintLessThan(u8, 6));
    const w2s = w2.slice(u8, G_E * H * I);
    r.bytes(w2s);
    for (w2s) |*b| {
        if ((b.* & 0x7F) == 0x7F) b.* = 0x38;
    }
    const s2s = s2.slice(u8, G_E * H * (I / 32));
    for (s2s) |*sc| sc.* = @intCast(113 + r.uintLessThan(u8, 6));

    // Exercise the MXFP8 slow path (sc_code > 246 and E4M3 NaN weight byte) on two colliding blocks.
    s13s[0] = 247;
    @memset(w13s[0..32], 0x01);
    w2s[0] = 0x7F;

    var cg: h.Check = .{ .name = "mxfp8_grouped_bit_exact" };
    for (G_ROW_COUNTS) |rows| {
        const slots = rows * TOPK;
        const ids = eids.slice(u32, slots);
        fillHeavyPicks(r, ids, rows);

        @memset(scratch.slice(u8, SCRATCH_BYTES), 0xff);
        @memset(act_g.slice(u8, slots * I * 2), 0xff);
        @memset(ey_g.slice(u8, slots * H * 2), 0xff);

        const a13: h.Args = .{ .rows = @intCast(slots), .width = I, .dim = H, .queries = TOPK, .cap = 4.0, .thr = 25.0 };
        const a2: h.Args = .{ .rows = @intCast(slots), .width = H, .dim = I };
        try g.run("ik_mxfp8_w13_swiglu", a13, &.{ .{ .buf = xm }, .{ .buf = eids }, .{ .buf = w13 }, .{ .buf = s13 }, .{ .buf = act_ref } }, .{ (I + 3) / 4, slots, 1 }, .{ 128, 1, 1 }, true);
        try g.run("ik_mxfp8_w2", a2, &.{ .{ .buf = act_ref }, .{ .buf = eids }, .{ .buf = w2 }, .{ .buf = s2 }, .{ .buf = ey_ref } }, .{ (H + 3) / 4, slots, 1 }, .{ 128, 1, 1 }, true);

        try runGroupedMxfp8(g, p_grp, p_g13, p_g2, xm, eids, scratch, w13, s13, act_g, w2, s2, ey_g, slots);
        _ = try checkGroupTable(scratch, ids, slots, rows);

        cg.same(act_g.slice(u8, slots * I * 2), act_ref.slice(u8, slots * I * 2));
        cg.same(ey_g.slice(u8, slots * H * 2), ey_ref.slice(u8, slots * H * 2));
    }
    try cg.done();

    // NaN containment on rows=8 where row 1 shares expert 0 inside the same tile as rows 0, 2, 3
    const nan_rows = 8;
    const nan_slots = nan_rows * TOPK;
    fillHeavyPicks(r, eids.slice(u32, nan_slots), nan_rows);
    try runGroupedMxfp8(g, p_grp, p_g13, p_g2, xm, eids, scratch, w13, s13, act_ref, w2, s2, ey_ref, nan_slots);
    const kept = xms[1 * H];
    xms[1 * H] = 0x7fc0;
    try runGroupedMxfp8(g, p_grp, p_g13, p_g2, xm, eids, scratch, w13, s13, act_g, w2, s2, ey_g, nan_slots);
    xms[1 * H] = kept;
    var cn: h.Check = .{ .name = "mxfp8_grouped_nan_contain" };
    for (0..nan_rows) |row| if (row != 1) {
        cn.same(act_g.slice(u8, nan_slots * I * 2)[row * TOPK * I * 2 ..][0 .. TOPK * I * 2], act_ref.slice(u8, nan_slots * I * 2)[row * TOPK * I * 2 ..][0 .. TOPK * I * 2]);
        cn.same(ey_g.slice(u8, nan_slots * H * 2)[row * TOPK * H * 2 ..][0 .. TOPK * H * 2], ey_ref.slice(u8, nan_slots * H * 2)[row * TOPK * H * 2 ..][0 .. TOPK * H * 2]);
    };
    try cn.done();
}
