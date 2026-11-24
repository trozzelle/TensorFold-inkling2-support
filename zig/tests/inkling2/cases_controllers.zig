//! Inkling 2 looped-core controller kernel checks: ik_copy_f32, ik_msg_depth, ik_msggate_prep,
//! ik_msggate_apply_norm, ik_msggate_prep3, and ik_msggate_apply_norm3 against FP64 and 1-slot references,
//! row independence, and NaN containment at H=2304, rank=64.
const std = @import("std");
const mtl = @import("metal");
const h = @import("harness.zig");

const H = 2304;
const RANK = 64;
const ROWS = 5;
const flag_global: u32 = 2;

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    try caseCopyF32(g, r);
    try caseMsgDepth(g, r);
    try caseMsgGate(g, r, H, 0, 0, 1e-6); // local layer stride = H, msg slot 0
    try caseMsgGate(g, r, 3 * H, 2, 2, 1e-5); // global layer stride = 3*H, query 2 / msg slot 2 (v), distinct norm eps
    try caseMsgGate3(g, r, false, 1e-6); // local 3-slot: flags = 0, stride = H
    try caseMsgGate3(g, r, true, 1e-5); // global 3-slot: flags = IK_GLOBAL, stride = 3*H, distinct norm eps
}

fn caseCopyF32(g: *const h.Gpu, r: std.Random) !void {
    const src = try g.buf(ROWS * H * 4);
    const dst = try g.buf(ROWS * H * 4);
    const one_dst = try g.buf(H * 4);
    defer for ([_]mtl.Buffer{ src, dst, one_dst }) |b| b.deinit();

    const xs = h.fill32(src, ROWS * H, r, 3.0);
    const a: h.Args = .{ .rows = ROWS, .width = H };
    try g.run("ik_copy_f32", a, &.{ .{ .buf = src }, .{ .buf = dst } }, .{ H, ROWS, 1 }, .{ 256, 1, 1 }, false);

    var c: h.Check = .{ .name = "copy_f32" };
    c.same(dst.slice(u8, ROWS * H * 4), src.slice(u8, ROWS * H * 4));

    const all = try std.heap.page_allocator.dupe(u8, dst.slice(u8, ROWS * H * 4));
    defer std.heap.page_allocator.free(all);
    for (0..ROWS) |row| {
        try g.run("ik_copy_f32", .{ .rows = 1, .width = H }, &.{ .{ .buf = src, .off = row * H * 4 }, .{ .buf = one_dst } }, .{ H, 1, 1 }, .{ 256, 1, 1 }, false);
        c.same(one_dst.slice(u8, H * 4), all[row * H * 4 ..][0 .. H * 4]);
    }
    const kept = xs[1 * H];
    xs[1 * H] = std.math.nan(f32);
    try g.run("ik_copy_f32", a, &.{ .{ .buf = src }, .{ .buf = dst } }, .{ H, ROWS, 1 }, .{ 256, 1, 1 }, false);
    for (0..ROWS) |row| if (row != 1) c.same(dst.slice(u8, ROWS * H * 4)[row * H * 4 ..][0 .. H * 4], all[row * H * 4 ..][0 .. H * 4]);
    xs[1 * H] = kept;
    try c.done();
}

fn caseMsgDepth(g: *const h.Gpu, r: std.Random) !void {
    const stride = 8;
    const blocks = try g.buf(1 * stride * H * 4);
    const partial = try g.buf(ROWS * H * 4);
    const q = try g.buf(4 * H * 4);
    const out = try g.buf(ROWS * 4 * H * 4);
    const one_out = try g.buf(4 * H * 4);
    defer for ([_]mtl.Buffer{ blocks, partial, q, out, one_out }) |b| b.deinit();

    const bs = h.fill32(blocks, stride * H, r, 1.5);
    const ps = h.fill32(partial, ROWS * H, r, 1.5);
    const qs = h.fill32(q, 4 * H, r, 1.5);
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, H)));
    const a: h.Args = .{ .rows = ROWS, .width = H, .blocks = 1, .queries = 4, .stride = stride, .scale = scale };
    try g.run("ik_msg_depth", a, &.{ .{ .buf = blocks }, .{ .buf = partial }, .{ .buf = q }, .{ .buf = out } }, .{ ROWS, 1, 1 }, .{ 1024, 1, 1 }, true);
    const ys = out.slice(f32, ROWS * 4 * H);

    var c: h.Check = .{ .name = "msg_depth" };
    for (0..ROWS) |row| {
        for (0..4) |j| {
            var d0: f64 = 0;
            var d1: f64 = 0;
            for (0..H) |d| {
                d0 += @as(f64, bs[row * H + d]) * qs[j * H + d];
                d1 += @as(f64, ps[row * H + d]) * qs[j * H + d];
            }
            const l0 = d0 * scale;
            const l1 = d1 * scale;
            const m = @max(l0, l1);
            const e0 = @exp(l0 - m);
            const e1 = @exp(l1 - m);
            const w0 = e0 / (e0 + e1);
            const w1 = e1 / (e0 + e1);
            for (0..H) |d| {
                const want = w0 * bs[row * H + d] + w1 * ps[row * H + d];
                c.near(ys[(row * 4 + j) * H + d], want, 1e-4, 1.0);
            }
        }
    }

    // Row independence: each row alone matches the batch's bits.
    const all = try std.heap.page_allocator.dupe(u8, out.slice(u8, ROWS * 4 * H * 4));
    defer std.heap.page_allocator.free(all);
    var one_a = a;
    one_a.rows = 1;
    for (0..ROWS) |row| {
        try g.run("ik_msg_depth", one_a, &.{ .{ .buf = blocks, .off = row * H * 4 }, .{ .buf = partial, .off = row * H * 4 }, .{ .buf = q }, .{ .buf = one_out } }, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, true);
        c.same(one_out.slice(u8, 4 * H * 4), all[row * 4 * H * 4 ..][0 .. 4 * H * 4]);
    }

    // NaN containment: NaN in row 1 leaves all other rows bit-identical.
    const kept = ps[1 * H];
    ps[1 * H] = std.math.nan(f32);
    try g.run("ik_msg_depth", a, &.{ .{ .buf = blocks }, .{ .buf = partial }, .{ .buf = q }, .{ .buf = out } }, .{ ROWS, 1, 1 }, .{ 1024, 1, 1 }, true);
    for (0..ROWS) |row| if (row != 1) c.same(out.slice(u8, ROWS * 4 * H * 4)[row * 4 * H * 4 ..][0 .. 4 * H * 4], all[row * 4 * H * 4 ..][0 .. 4 * H * 4]);
    ps[1 * H] = kept;
    try c.done();
}

fn caseMsgGate(g: *const h.Gpu, r: std.Random, h_stride: u32, h_slot: usize, msg_slot: usize, norm_eps: f32) !void {
    const hbuf = try g.buf(ROWS * h_stride * 4);
    const mbuf = try g.buf(ROWS * 4 * H * 4);
    const ctrl_in = try g.buf(ROWS * 2 * H * 2);
    const one_in = try g.buf(2 * H * 2);
    const zbuf = try g.buf(ROWS * RANK * 2);
    const fw = try g.buf(H * RANK * 2);
    const gw = try g.buf(H * RANK * 2);
    const nw = try g.buf(H * 2);
    const out = try g.buf(ROWS * H * 2);
    const one_out = try g.buf(H * 2);
    defer for ([_]mtl.Buffer{ hbuf, mbuf, ctrl_in, one_in, zbuf, fw, gw, nw, out, one_out }) |b| b.deinit();

    const hs = h.fill32(hbuf, ROWS * h_stride, r, 2.0);
    const ms = h.fill32(mbuf, ROWS * 4 * H, r, 2.0);
    const zs = h.fill16(zbuf, ROWS * RANK, r, 0.3);
    const fws = h.fill16(fw, H * RANK, r, 0.25);
    const gws = h.fill16(gw, H * RANK, r, 0.25);
    const nws = h.fill16(nw, H, r, 1.5);

    const h_off = h_slot * H * 4;
    const m_off = msg_slot * H * 4;
    const prep_a: h.Args = .{ .rows = ROWS, .width = H, .stride = h_stride };
    try g.run("ik_msggate_prep", prep_a, &.{ .{ .buf = hbuf, .off = h_off }, .{ .buf = mbuf, .off = m_off }, .{ .buf = ctrl_in } }, .{ ROWS, 1, 1 }, .{ 1024, 1, 1 }, true);
    const apply_a: h.Args = .{ .rows = ROWS, .width = H, .dim = RANK, .stride = h_stride, .eps = norm_eps };
    try g.run("ik_msggate_apply_norm", apply_a, &.{ .{ .buf = hbuf, .off = h_off }, .{ .buf = mbuf, .off = m_off }, .{ .buf = zbuf }, .{ .buf = fw }, .{ .buf = gw }, .{ .buf = nw }, .{ .buf = out } }, .{ ROWS, 1, 1 }, .{ 1024, 1, 1 }, true);

    var cp: h.Check = .{ .name = "msggate_prep" };
    var ca: h.Check = .{ .name = "msggate_apply_norm" };
    const cins = ctrl_in.slice(u16, ROWS * 2 * H);
    const outs = out.slice(u16, ROWS * H);
    var blended: [H]f64 = undefined;
    for (0..ROWS) |row| {
        var ssh: f64 = 0;
        var ssm: f64 = 0;
        for (0..H) |d| {
            const hv: f64 = hs[row * h_stride + h_slot * H + d];
            const mv: f64 = ms[(row * 4 + msg_slot) * H + d];
            ssh += hv * hv;
            ssm += mv * mv;
        }
        const rh = 1.0 / @sqrt(ssh / H + 1e-6);
        const rm = 1.0 / @sqrt(ssm / H + 1e-6);
        var ssb: f64 = 0;
        for (0..H) |d| {
            const hv: f64 = hs[row * h_stride + h_slot * H + d];
            const mv: f64 = ms[(row * 4 + msg_slot) * H + d];
            cp.b16(cins[row * 2 * H + d], hv * rh);
            cp.b16(cins[row * 2 * H + H + d], mv * rm);
            var df: f64 = 0;
            var dg: f64 = 0;
            for (0..RANK) |rk| {
                const zr = h.fb(zs[row * RANK + rk]);
                df += zr * h.fb(fws[d * RANK + rk]);
                dg += zr * h.fb(gws[d * RANK + rk]);
            }
            const bv = (1.0 - std.math.tanh(df)) * hv + std.math.tanh(dg) * mv;
            blended[d] = bv;
            ssb += bv * bv;
        }
        const rinv = 1.0 / @sqrt(ssb / H + @as(f64, norm_eps));
        for (0..H) |d| ca.b16(outs[row * H + d], h.fb(nws[d]) * (blended[d] * rinv));
    }
    try cp.done();
    try ca.done();

    // Row independence: each row alone matches the batch's bits for both prep and apply_norm.
    const all_in = try std.heap.page_allocator.dupe(u8, ctrl_in.slice(u8, ROWS * 2 * H * 2));
    defer std.heap.page_allocator.free(all_in);
    const all_out = try std.heap.page_allocator.dupe(u8, out.slice(u8, ROWS * H * 2));
    defer std.heap.page_allocator.free(all_out);

    var ci: h.Check = .{ .name = "msggate_independent" };
    var one_prep = prep_a;
    one_prep.rows = 1;
    var one_apply = apply_a;
    one_apply.rows = 1;
    for (0..ROWS) |row| {
        const rh_off = (row * h_stride + h_slot * H) * 4;
        const rm_off = (row * 4 + msg_slot) * H * 4;
        try g.run("ik_msggate_prep", one_prep, &.{ .{ .buf = hbuf, .off = rh_off }, .{ .buf = mbuf, .off = rm_off }, .{ .buf = one_in } }, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, true);
        ci.same(one_in.slice(u8, 2 * H * 2), all_in[row * 2 * H * 2 ..][0 .. 2 * H * 2]);
        try g.run("ik_msggate_apply_norm", one_apply, &.{ .{ .buf = hbuf, .off = rh_off }, .{ .buf = mbuf, .off = rm_off }, .{ .buf = zbuf, .off = row * RANK * 2 }, .{ .buf = fw }, .{ .buf = gw }, .{ .buf = nw }, .{ .buf = one_out } }, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, true);
        ci.same(one_out.slice(u8, H * 2), all_out[row * H * 2 ..][0 .. H * 2]);
    }
    try ci.done();

    // NaN containment on row 1 for both prep and apply_norm.
    const kept = hs[1 * h_stride + h_slot * H];
    hs[1 * h_stride + h_slot * H] = std.math.nan(f32);
    try g.run("ik_msggate_prep", prep_a, &.{ .{ .buf = hbuf, .off = h_off }, .{ .buf = mbuf, .off = m_off }, .{ .buf = ctrl_in } }, .{ ROWS, 1, 1 }, .{ 1024, 1, 1 }, true);
    try g.run("ik_msggate_apply_norm", apply_a, &.{ .{ .buf = hbuf, .off = h_off }, .{ .buf = mbuf, .off = m_off }, .{ .buf = zbuf }, .{ .buf = fw }, .{ .buf = gw }, .{ .buf = nw }, .{ .buf = out } }, .{ ROWS, 1, 1 }, .{ 1024, 1, 1 }, true);
    var cn: h.Check = .{ .name = "msggate_nan_contain" };
    for (0..ROWS) |row| if (row != 1) {
        cn.same(ctrl_in.slice(u8, ROWS * 2 * H * 2)[row * 2 * H * 2 ..][0 .. 2 * H * 2], all_in[row * 2 * H * 2 ..][0 .. 2 * H * 2]);
        cn.same(out.slice(u8, ROWS * H * 2)[row * H * 2 ..][0 .. H * 2], all_out[row * H * 2 ..][0 .. H * 2]);
    };
    hs[1 * h_stride + h_slot * H] = kept;
    try cn.done();
}

fn caseMsgGate3(g: *const h.Gpu, r: std.Random, glob: bool, norm_eps: f32) !void {
    const h_stride: u32 = if (glob) 3 * H else H;
    const flags: u32 = if (glob) flag_global else 0;
    const hbuf = try g.buf(ROWS * h_stride * 4);
    const mbuf = try g.buf(ROWS * 4 * H * 4);
    const ref_in = try g.buf(3 * ROWS * 2 * H * 2);
    const got_in = try g.buf(3 * ROWS * 2 * H * 2);
    const one_in = try g.buf(3 * 2 * H * 2);
    const zbuf = try g.buf(3 * ROWS * RANK * 2);
    const one_z = try g.buf(3 * RANK * 2);
    const fw0 = try g.buf(H * RANK * 2);
    const fw1 = try g.buf(H * RANK * 2);
    const fw2 = try g.buf(H * RANK * 2);
    const gw0 = try g.buf(H * RANK * 2);
    const gw1 = try g.buf(H * RANK * 2);
    const gw2 = try g.buf(H * RANK * 2);
    const nw0 = try g.buf(H * 2);
    const nw1 = try g.buf(H * 2);
    const nw2 = try g.buf(H * 2);
    const ref_out = try g.buf(3 * ROWS * H * 2);
    const got_out = try g.buf(3 * ROWS * H * 2);
    const one_out = try g.buf(3 * H * 2);
    defer for ([_]mtl.Buffer{ hbuf, mbuf, ref_in, got_in, one_in, zbuf, one_z, fw0, fw1, fw2, gw0, gw1, gw2, nw0, nw1, nw2, ref_out, got_out, one_out }) |b| b.deinit();

    const hs = h.fill32(hbuf, ROWS * h_stride, r, 2.0);
    _ = h.fill32(mbuf, ROWS * 4 * H, r, 2.0);
    const zs = h.fill16(zbuf, 3 * ROWS * RANK, r, 0.3);
    const fws = [3]mtl.Buffer{ fw0, fw1, fw2 };
    const gws = [3]mtl.Buffer{ gw0, gw1, gw2 };
    const nws = [3]mtl.Buffer{ nw0, nw1, nw2 };
    for (0..3) |s| {
        _ = h.fill16(fws[s], H * RANK, r, 0.25);
        _ = h.fill16(gws[s], H * RANK, r, 0.25);
        _ = h.fill16(nws[s], H, r, 1.5);
    }

    // Reference: 3 calls of ik_msggate_prep and 3 calls of ik_msggate_apply_norm.
    const prep1_a: h.Args = .{ .rows = ROWS, .width = H, .stride = h_stride };
    const apply1_a: h.Args = .{ .rows = ROWS, .width = H, .dim = RANK, .stride = h_stride, .eps = norm_eps };
    for (0..3) |slot| {
        const h_off = if (glob) slot * H * 4 else 0;
        const m_off = slot * H * 4;
        const in_off = slot * ROWS * 2 * H * 2;
        const z_off = slot * ROWS * RANK * 2;
        const out_off = slot * ROWS * H * 2;
        const nw_buf = if (glob) nws[slot] else nws[0];
        try g.run("ik_msggate_prep", prep1_a, &.{ .{ .buf = hbuf, .off = h_off }, .{ .buf = mbuf, .off = m_off }, .{ .buf = ref_in, .off = in_off } }, .{ ROWS, 1, 1 }, .{ 1024, 1, 1 }, true);
        try g.run("ik_msggate_apply_norm", apply1_a, &.{ .{ .buf = hbuf, .off = h_off }, .{ .buf = mbuf, .off = m_off }, .{ .buf = zbuf, .off = z_off }, .{ .buf = fws[slot] }, .{ .buf = gws[slot] }, .{ .buf = nw_buf }, .{ .buf = ref_out, .off = out_off } }, .{ ROWS, 1, 1 }, .{ 1024, 1, 1 }, true);
    }

    // 3-slot kernels: must be bit-for-bit identical to the 3 single-slot calls.
    const prep3_a: h.Args = .{ .rows = ROWS, .width = H, .stride = h_stride, .flags = flags };
    const apply3_a: h.Args = .{ .rows = ROWS, .width = H, .dim = RANK, .stride = h_stride, .flags = flags, .eps = norm_eps };
    try g.run("ik_msggate_prep3", prep3_a, &.{ .{ .buf = hbuf }, .{ .buf = mbuf }, .{ .buf = got_in } }, .{ ROWS, 3, 1 }, .{ 1024, 1, 1 }, true);
    try g.run("ik_msggate_apply_norm3", apply3_a, &.{ .{ .buf = hbuf }, .{ .buf = mbuf }, .{ .buf = zbuf }, .{ .buf = fw0 }, .{ .buf = fw1 }, .{ .buf = fw2 }, .{ .buf = gw0 }, .{ .buf = gw1 }, .{ .buf = gw2 }, .{ .buf = nw0 }, .{ .buf = nw1 }, .{ .buf = nw2 }, .{ .buf = got_out } }, .{ ROWS, 3, 1 }, .{ 1024, 1, 1 }, true);

    var cp: h.Check = .{ .name = if (glob) "msggate_prep3_global" else "msggate_prep3_local" };
    cp.same(got_in.slice(u8, 3 * ROWS * 2 * H * 2), ref_in.slice(u8, 3 * ROWS * 2 * H * 2));
    try cp.done();

    var ca: h.Check = .{ .name = if (glob) "msggate_apply_norm3_global" else "msggate_apply_norm3_local" };
    ca.same(got_out.slice(u8, 3 * ROWS * H * 2), ref_out.slice(u8, 3 * ROWS * H * 2));
    try ca.done();

    // Row independence: each row alone matches the batch's bits for all 3 slots.
    const all_in = try std.heap.page_allocator.dupe(u8, got_in.slice(u8, 3 * ROWS * 2 * H * 2));
    defer std.heap.page_allocator.free(all_in);
    const all_out = try std.heap.page_allocator.dupe(u8, got_out.slice(u8, 3 * ROWS * H * 2));
    defer std.heap.page_allocator.free(all_out);

    var ci: h.Check = .{ .name = if (glob) "msggate3_independent_global" else "msggate3_independent_local" };
    var one_prep = prep3_a;
    one_prep.rows = 1;
    var one_apply = apply3_a;
    one_apply.rows = 1;
    const ozs = one_z.slice(u16, 3 * RANK);
    for (0..ROWS) |row| {
        const rh_off = row * h_stride * 4;
        const rm_off = row * 4 * H * 4;
        for (0..3) |slot| @memcpy(ozs[slot * RANK ..][0..RANK], zs[(slot * ROWS + row) * RANK ..][0..RANK]);
        try g.run("ik_msggate_prep3", one_prep, &.{ .{ .buf = hbuf, .off = rh_off }, .{ .buf = mbuf, .off = rm_off }, .{ .buf = one_in } }, .{ 1, 3, 1 }, .{ 1024, 1, 1 }, true);
        try g.run("ik_msggate_apply_norm3", one_apply, &.{ .{ .buf = hbuf, .off = rh_off }, .{ .buf = mbuf, .off = rm_off }, .{ .buf = one_z }, .{ .buf = fw0 }, .{ .buf = fw1 }, .{ .buf = fw2 }, .{ .buf = gw0 }, .{ .buf = gw1 }, .{ .buf = gw2 }, .{ .buf = nw0 }, .{ .buf = nw1 }, .{ .buf = nw2 }, .{ .buf = one_out } }, .{ 1, 3, 1 }, .{ 1024, 1, 1 }, true);
        for (0..3) |slot| {
            ci.same(one_in.slice(u8, 3 * 2 * H * 2)[slot * 2 * H * 2 ..][0 .. 2 * H * 2], all_in[(slot * ROWS + row) * 2 * H * 2 ..][0 .. 2 * H * 2]);
            ci.same(one_out.slice(u8, 3 * H * 2)[slot * H * 2 ..][0 .. H * 2], all_out[(slot * ROWS + row) * H * 2 ..][0 .. H * 2]);
        }
    }
    try ci.done();

    // NaN containment on row 1 across all active h slots for both prep3 and apply_norm3.
    const h_slots: usize = if (glob) 3 else 1;
    var kept: [3]f32 = undefined;
    for (0..h_slots) |s| {
        kept[s] = hs[1 * h_stride + s * H];
        hs[1 * h_stride + s * H] = std.math.nan(f32);
    }
    try g.run("ik_msggate_prep3", prep3_a, &.{ .{ .buf = hbuf }, .{ .buf = mbuf }, .{ .buf = got_in } }, .{ ROWS, 3, 1 }, .{ 1024, 1, 1 }, true);
    try g.run("ik_msggate_apply_norm3", apply3_a, &.{ .{ .buf = hbuf }, .{ .buf = mbuf }, .{ .buf = zbuf }, .{ .buf = fw0 }, .{ .buf = fw1 }, .{ .buf = fw2 }, .{ .buf = gw0 }, .{ .buf = gw1 }, .{ .buf = gw2 }, .{ .buf = nw0 }, .{ .buf = nw1 }, .{ .buf = nw2 }, .{ .buf = got_out } }, .{ ROWS, 3, 1 }, .{ 1024, 1, 1 }, true);
    var cn: h.Check = .{ .name = if (glob) "msggate3_nan_contain_global" else "msggate3_nan_contain_local" };
    for (0..3) |slot| {
        for (0..ROWS) |row| if (row != 1) {
            cn.same(got_in.slice(u8, 3 * ROWS * 2 * H * 2)[(slot * ROWS + row) * 2 * H * 2 ..][0 .. 2 * H * 2], all_in[(slot * ROWS + row) * 2 * H * 2 ..][0 .. 2 * H * 2]);
            cn.same(got_out.slice(u8, 3 * ROWS * H * 2)[(slot * ROWS + row) * H * 2 ..][0 .. H * 2], all_out[(slot * ROWS + row) * H * 2 ..][0 .. H * 2]);
        };
    }
    for (0..h_slots) |s| hs[1 * h_stride + s * H] = kept[s];
    try cn.done();
}
