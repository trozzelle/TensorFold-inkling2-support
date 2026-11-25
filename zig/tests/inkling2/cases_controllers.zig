//! Inkling 2 looped-core controller kernel checks: ik_copy_f32, ik_msg_depth, ik_msggate_prep,
//! ik_msggate_apply_norm, ik_msggate_prep3, ik_msggate_apply_norm3, ik_msggate_fg_nax + ik_msggate_blend_norm,
//! ik_msggate_fg_decode + ik_norm, ik_msggate_fg_decode3 + ik_norm3_f32, ik_loop_attn_prep, ik_loop_attn_prep3,
//! and ik_loop_mlp_prep against FP64 and unfused references, row independence, and NaN containment at H=2304, rank=64.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const h = @import("harness.zig");
const cp_mod = tf.inkling2.ctrl_prefill;

const H = 2304;
const RANK = 64;
const ROWS = 5;
const flag_f32_in: u32 = 1;
const flag_global: u32 = 2;
const flag_boundary: u32 = 4;

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    try caseCopyF32(g, r);
    try caseMsgDepth(g, r);
    try caseMsgGate(g, r, H, 0, 0, 1e-6); // local layer stride = H, msg slot 0
    try caseMsgGate(g, r, 3 * H, 2, 2, 1e-5); // global layer stride = 3*H, query 2 / msg slot 2 (v), distinct norm eps
    try caseMsgGate3(g, r, false, 1e-6); // local 3-slot: flags = 0, stride = H
    try caseMsgGate3(g, r, true, 1e-5); // global 3-slot: flags = IK_GLOBAL, stride = 3*H, distinct norm eps
    try caseFgNaxBlendNorm(g, r);
    try caseFgDecode(g, r);
    try caseFgDecode3(g, r, false, 1e-6);
    try caseFgDecode3(g, r, true, 1e-5);
    try caseLoopAttnPrep(g, r, false);
    try caseLoopAttnPrep(g, r, true);
    try caseLoopMlpPrep(g, r, false);
    try caseLoopMlpPrep(g, r, true);
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

fn runFgNax(g: *const h.Gpu, pipes: *const cp_mod.Pipes, z: mtl.Buffer, z_off: usize, fw: mtl.Buffer, gw: mtl.Buffer, df: mtl.Buffer, dg: mtl.Buffer, rows: usize) !void {
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    cp_mod.encodeFg(e, pipes, .{ .buf = z, .off = z_off }, .{ .buf = fw }, .{ .buf = gw }, .{ .buf = df }, .{ .buf = dg }, rows, H, RANK);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("ik_msggate_fg_nax: {s}", .{msg});
        return error.GpuFailed;
    }
}

fn caseFgNaxBlendNorm(g: *const h.Gpu, r: std.Random) !void {
    const a = std.heap.page_allocator;
    var pipes = try cp_mod.Pipes.init(g.device, a);
    defer pipes.deinit();
    if (g.device.tensorUnits()) {
        var older = try cp_mod.Pipes.initUnits(g.device, a, false);
        older.deinit();
    }

    const max_rows: usize = 128;
    const pad_rows: usize = 32;
    const h_stride: u32 = 3 * H;
    const h_slot: usize = 1;
    const msg_slot: usize = 2;
    const norm_eps: f32 = 1e-5;

    const hbuf = try g.buf(max_rows * h_stride * 4);
    const mbuf = try g.buf(max_rows * 4 * H * 4);
    const zbuf = try g.buf((max_rows + pad_rows) * RANK * 2);
    const fw = try g.buf(H * RANK * 2);
    const gw = try g.buf(H * RANK * 2);
    const nw = try g.buf(H * 2);
    const df = try g.buf((max_rows + pad_rows) * H * 4);
    const dg = try g.buf((max_rows + pad_rows) * H * 4);
    const out = try g.buf(max_rows * H * 2);
    const one_df = try g.buf(H * 4);
    const one_dg = try g.buf(H * 4);
    const one_out = try g.buf(H * 2);
    defer for ([_]mtl.Buffer{ hbuf, mbuf, zbuf, fw, gw, nw, df, dg, out, one_df, one_dg, one_out }) |b| b.deinit();

    const hs = h.fill32(hbuf, max_rows * h_stride, r, 2.0);
    const ms = h.fill32(mbuf, max_rows * 4 * H, r, 2.0);
    const zs = h.fill16(zbuf, (max_rows + pad_rows) * RANK, r, 0.3);
    const fws = h.fill16(fw, H * RANK, r, 0.25);
    const gws = h.fill16(gw, H * RANK, r, 0.25);
    const nws = h.fill16(nw, H, r, 1.5);

    const h_off = h_slot * H * 4;
    const m_off = msg_slot * H * 4;
    var blended: [H]f64 = undefined;
    for ([_]usize{ 1, 17, 64, 128 }) |rows| {
        // Poison rows past `rows` with NaN in z and sentinel in df/dg to verify no OOB read or write.
        const z_pad = zs[rows * RANK ..][0 .. pad_rows * RANK];
        const keep_z = try a.dupe(u16, z_pad);
        defer a.free(keep_z);
        @memset(z_pad, 0x7fc0);
        @memset(df.slice(u8, (rows + pad_rows) * H * 4), 0xa5);
        @memset(dg.slice(u8, (rows + pad_rows) * H * 4), 0xa5);

        try runFgNax(g, &pipes, zbuf, 0, fw, gw, df, dg, rows);
        @memcpy(z_pad, keep_z);

        const blend_a: h.Args = .{ .rows = @intCast(rows), .width = H, .stride = h_stride, .eps = norm_eps };
        try g.run("ik_msggate_blend_norm", blend_a, &.{ .{ .buf = hbuf, .off = h_off }, .{ .buf = mbuf, .off = m_off }, .{ .buf = df }, .{ .buf = dg }, .{ .buf = nw }, .{ .buf = out } }, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, true);

        var name: [64]u8 = undefined;
        var c: h.Check = .{ .name = try std.fmt.bufPrint(&name, "fg_nax_blend_norm rows {d}", .{rows}) };
        const outs = out.slice(u16, rows * H);
        const dfs = df.slice(f32, rows * H);
        const dgs = dg.slice(f32, rows * H);
        for (0..rows) |row| {
            var ssb: f64 = 0;
            for (0..H) |d| {
                const hv: f64 = hs[row * h_stride + h_slot * H + d];
                const mv: f64 = ms[(row * 4 + msg_slot) * H + d];
                var rf: f64 = 0;
                var rg: f64 = 0;
                for (0..RANK) |rk| {
                    const zr = h.fb(zs[row * RANK + rk]);
                    rf += zr * h.fb(fws[d * RANK + rk]);
                    rg += zr * h.fb(gws[d * RANK + rk]);
                }
                c.near(dfs[row * H + d], rf, 1e-4, 1.0);
                c.near(dgs[row * H + d], rg, 1e-4, 1.0);
                const bv = (1.0 - std.math.tanh(rf)) * hv + std.math.tanh(rg) * mv;
                blended[d] = bv;
                ssb += bv * bv;
            }
            const rinv = 1.0 / @sqrt(ssb / H + @as(f64, norm_eps));
            for (0..H) |d| c.b16n(outs[row * H + d], h.fb(nws[d]) * (blended[d] * rinv), 2);
        }
        // Nothing written past `rows`.
        var touched: usize = 0;
        for (df.slice(u8, (rows + pad_rows) * H * 4)[rows * H * 4 ..]) |b| touched += @intFromBool(b != 0xa5);
        for (dg.slice(u8, (rows + pad_rows) * H * 4)[rows * H * 4 ..]) |b| touched += @intFromBool(b != 0xa5);
        c.n += 1;
        c.bad += @intFromBool(touched != 0);
        try c.done();
    }

    // Row independence (1 row vs 128 rows) and NaN containment on row 1.
    try runFgNax(g, &pipes, zbuf, 0, fw, gw, df, dg, max_rows);
    const blend_all: h.Args = .{ .rows = max_rows, .width = H, .stride = h_stride, .eps = norm_eps };
    try g.run("ik_msggate_blend_norm", blend_all, &.{ .{ .buf = hbuf, .off = h_off }, .{ .buf = mbuf, .off = m_off }, .{ .buf = df }, .{ .buf = dg }, .{ .buf = nw }, .{ .buf = out } }, .{ max_rows, 1, 1 }, .{ 1024, 1, 1 }, true);
    const all_df = try a.dupe(u8, df.slice(u8, max_rows * H * 4));
    const all_dg = try a.dupe(u8, dg.slice(u8, max_rows * H * 4));
    const all_out = try a.dupe(u8, out.slice(u8, max_rows * H * 2));
    defer for ([_][]u8{ all_df, all_dg, all_out }) |s| a.free(s);

    var ci: h.Check = .{ .name = "fg_nax_blend_norm_independent" };
    for ([_]usize{ 0, 1, 15, 16, 17, 31, 32, 63, 64, 127 }) |row| {
        try runFgNax(g, &pipes, zbuf, row * RANK * 2, fw, gw, one_df, one_dg, 1);
        ci.same(one_df.slice(u8, H * 4), all_df[row * H * 4 ..][0 .. H * 4]);
        ci.same(one_dg.slice(u8, H * 4), all_dg[row * H * 4 ..][0 .. H * 4]);
        const rh_off = (row * h_stride + h_slot * H) * 4;
        const rm_off = (row * 4 + msg_slot) * H * 4;
        try g.run("ik_msggate_blend_norm", .{ .rows = 1, .width = H, .stride = h_stride, .eps = norm_eps }, &.{ .{ .buf = hbuf, .off = rh_off }, .{ .buf = mbuf, .off = rm_off }, .{ .buf = one_df }, .{ .buf = one_dg }, .{ .buf = nw }, .{ .buf = one_out } }, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, true);
        ci.same(one_out.slice(u8, H * 2), all_out[row * H * 2 ..][0 .. H * 2]);
    }
    try ci.done();

    const kept_z = zs[1 * RANK];
    zs[1 * RANK] = 0x7fc0;
    try runFgNax(g, &pipes, zbuf, 0, fw, gw, df, dg, max_rows);
    try g.run("ik_msggate_blend_norm", blend_all, &.{ .{ .buf = hbuf, .off = h_off }, .{ .buf = mbuf, .off = m_off }, .{ .buf = df }, .{ .buf = dg }, .{ .buf = nw }, .{ .buf = out } }, .{ max_rows, 1, 1 }, .{ 1024, 1, 1 }, true);
    var cn: h.Check = .{ .name = "fg_nax_blend_norm_nan_contain" };
    for (0..max_rows) |row| if (row != 1) {
        cn.same(df.slice(u8, max_rows * H * 4)[row * H * 4 ..][0 .. H * 4], all_df[row * H * 4 ..][0 .. H * 4]);
        cn.same(dg.slice(u8, max_rows * H * 4)[row * H * 4 ..][0 .. H * 4], all_dg[row * H * 4 ..][0 .. H * 4]);
        cn.same(out.slice(u8, max_rows * H * 2)[row * H * 2 ..][0 .. H * 2], all_out[row * H * 2 ..][0 .. H * 2]);
    };
    zs[1 * RANK] = kept_z;
    try cn.done();
}

fn caseFgDecode(g: *const h.Gpu, r: std.Random) !void {
    const max_rows: usize = 16;
    const h_stride: u32 = H;
    const msg_slot: usize = 3;
    const norm_eps: f32 = 1e-6;

    const hbuf = try g.buf(max_rows * h_stride * 4);
    const mbuf = try g.buf(max_rows * 4 * H * 4);
    const zbuf = try g.buf(max_rows * RANK * 2);
    const fw = try g.buf(H * RANK * 2);
    const gw = try g.buf(H * RANK * 2);
    const nw = try g.buf(H * 2);
    const out_v = try g.buf(max_rows * H * 4);
    const out = try g.buf(max_rows * H * 2);
    const one_v = try g.buf(H * 4);
    const one_out = try g.buf(H * 2);
    defer for ([_]mtl.Buffer{ hbuf, mbuf, zbuf, fw, gw, nw, out_v, out, one_v, one_out }) |b| b.deinit();

    const hs = h.fill32(hbuf, max_rows * h_stride, r, 2.0);
    const ms = h.fill32(mbuf, max_rows * 4 * H, r, 2.0);
    const zs = h.fill16(zbuf, max_rows * RANK, r, 0.3);
    const fws = h.fill16(fw, H * RANK, r, 0.25);
    const gws = h.fill16(gw, H * RANK, r, 0.25);
    const nws = h.fill16(nw, H, r, 1.5);
    const m_off = msg_slot * H * 4;

    var blended: [H]f64 = undefined;
    for ([_]usize{ 1, 4, 8, 16 }) |rows| {
        const fg_a: h.Args = .{ .rows = @intCast(rows), .width = H, .dim = RANK, .stride = h_stride };
        try g.run("ik_msggate_fg_decode", fg_a, &.{ .{ .buf = hbuf }, .{ .buf = mbuf, .off = m_off }, .{ .buf = zbuf }, .{ .buf = fw }, .{ .buf = gw }, .{ .buf = out_v } }, .{ (H + 31) / 32, (rows + 15) / 16, 1 }, .{ 256, 1, 1 }, true);
        const norm_a: h.Args = .{ .rows = @intCast(rows), .width = H, .stride = H, .flags = flag_f32_in, .eps = norm_eps };
        try g.run("ik_norm", norm_a, &.{ .{ .buf = out_v }, .{ .buf = nw }, .{ .buf = out } }, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, true);

        var name: [64]u8 = undefined;
        var c: h.Check = .{ .name = try std.fmt.bufPrint(&name, "fg_decode_norm rows {d}", .{rows}) };
        const outs = out.slice(u16, rows * H);
        for (0..rows) |row| {
            var ssb: f64 = 0;
            for (0..H) |d| {
                const hv: f64 = hs[row * h_stride + d];
                const mv: f64 = ms[(row * 4 + msg_slot) * H + d];
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
            for (0..H) |d| c.b16(outs[row * H + d], h.fb(nws[d]) * (blended[d] * rinv));
        }
        try c.done();
    }

    // 1-row vs 16-row bit-identical row independence and NaN containment.
    const a = std.heap.page_allocator;
    const all_v = try a.dupe(u8, out_v.slice(u8, max_rows * H * 4));
    const all_out = try a.dupe(u8, out.slice(u8, max_rows * H * 2));
    defer a.free(all_v);
    defer a.free(all_out);

    var ci: h.Check = .{ .name = "fg_decode_independent" };
    for (0..max_rows) |row| {
        const rh_off = row * h_stride * 4;
        const rm_off = (row * 4 + msg_slot) * H * 4;
        try g.run("ik_msggate_fg_decode", .{ .rows = 1, .width = H, .dim = RANK, .stride = h_stride }, &.{ .{ .buf = hbuf, .off = rh_off }, .{ .buf = mbuf, .off = rm_off }, .{ .buf = zbuf, .off = row * RANK * 2 }, .{ .buf = fw }, .{ .buf = gw }, .{ .buf = one_v } }, .{ (H + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, true);
        ci.same(one_v.slice(u8, H * 4), all_v[row * H * 4 ..][0 .. H * 4]);
        try g.run("ik_norm", .{ .rows = 1, .width = H, .stride = H, .flags = flag_f32_in, .eps = norm_eps }, &.{ .{ .buf = one_v }, .{ .buf = nw }, .{ .buf = one_out } }, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, true);
        ci.same(one_out.slice(u8, H * 2), all_out[row * H * 2 ..][0 .. H * 2]);
    }
    try ci.done();

    const kept_z = zs[1 * RANK];
    zs[1 * RANK] = 0x7fc0;
    try g.run("ik_msggate_fg_decode", .{ .rows = max_rows, .width = H, .dim = RANK, .stride = h_stride }, &.{ .{ .buf = hbuf }, .{ .buf = mbuf, .off = m_off }, .{ .buf = zbuf }, .{ .buf = fw }, .{ .buf = gw }, .{ .buf = out_v } }, .{ (H + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, true);
    try g.run("ik_norm", .{ .rows = max_rows, .width = H, .stride = H, .flags = flag_f32_in, .eps = norm_eps }, &.{ .{ .buf = out_v }, .{ .buf = nw }, .{ .buf = out } }, .{ max_rows, 1, 1 }, .{ 1024, 1, 1 }, true);
    var cn: h.Check = .{ .name = "fg_decode_nan_contain" };
    for (0..max_rows) |row| if (row != 1) {
        cn.same(out_v.slice(u8, max_rows * H * 4)[row * H * 4 ..][0 .. H * 4], all_v[row * H * 4 ..][0 .. H * 4]);
        cn.same(out.slice(u8, max_rows * H * 2)[row * H * 2 ..][0 .. H * 2], all_out[row * H * 2 ..][0 .. H * 2]);
    };
    zs[1 * RANK] = kept_z;
    try cn.done();
}

fn caseFgDecode3(g: *const h.Gpu, r: std.Random, glob: bool, norm_eps: f32) !void {
    const max_rows: usize = 16;
    const h_stride: u32 = if (glob) 3 * H else H;
    const flags: u32 = if (glob) flag_global else 0;
    const tag: []const u8 = if (glob) "global" else "local";

    const hbuf = try g.buf(max_rows * h_stride * 4);
    const mbuf = try g.buf(max_rows * 4 * H * 4);
    const zbuf = try g.buf(3 * max_rows * RANK * 2);
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
    const out_v = try g.buf(3 * max_rows * H * 4);
    const out = try g.buf(3 * max_rows * H * 2);
    const one_v = try g.buf(3 * H * 4);
    const one_out = try g.buf(3 * H * 2);
    defer for ([_]mtl.Buffer{ hbuf, mbuf, zbuf, one_z, fw0, fw1, fw2, gw0, gw1, gw2, nw0, nw1, nw2, out_v, out, one_v, one_out }) |b| b.deinit();

    const hs = h.fill32(hbuf, max_rows * h_stride, r, 2.0);
    const ms = h.fill32(mbuf, max_rows * 4 * H, r, 2.0);
    const zs = h.fill16(zbuf, 3 * max_rows * RANK, r, 0.3);
    const fws = [3][]u16{ h.fill16(fw0, H * RANK, r, 0.25), h.fill16(fw1, H * RANK, r, 0.25), h.fill16(fw2, H * RANK, r, 0.25) };
    const gws = [3][]u16{ h.fill16(gw0, H * RANK, r, 0.25), h.fill16(gw1, H * RANK, r, 0.25), h.fill16(gw2, H * RANK, r, 0.25) };
    const nws = [3][]u16{ h.fill16(nw0, H, r, 1.5), h.fill16(nw1, H, r, 1.5), h.fill16(nw2, H, r, 1.5) };

    var blended: [H]f64 = undefined;
    for ([_]usize{ 1, 4, 8, 16 }) |rows| {
        const fg_a: h.Args = .{ .rows = @intCast(rows), .width = H, .dim = RANK, .stride = h_stride, .flags = flags };
        try g.run("ik_msggate_fg_decode3", fg_a, &.{ .{ .buf = hbuf }, .{ .buf = mbuf }, .{ .buf = zbuf }, .{ .buf = fw0 }, .{ .buf = fw1 }, .{ .buf = fw2 }, .{ .buf = gw0 }, .{ .buf = gw1 }, .{ .buf = gw2 }, .{ .buf = out_v } }, .{ (H + 31) / 32, (rows + 15) / 16, 3 }, .{ 256, 1, 1 }, true);
        const norm_a: h.Args = .{ .rows = @intCast(rows), .width = H, .flags = flags, .eps = norm_eps };
        try g.run("ik_norm3_f32", norm_a, &.{ .{ .buf = out_v }, .{ .buf = nw0 }, .{ .buf = nw1 }, .{ .buf = nw2 }, .{ .buf = out } }, .{ rows, 3, 1 }, .{ 1024, 1, 1 }, true);

        var name: [64]u8 = undefined;
        var c: h.Check = .{ .name = try std.fmt.bufPrint(&name, "fg_decode3_{s} rows {d}", .{ tag, rows }) };
        const outs = out.slice(u16, 3 * rows * H);
        for (0..3) |slot| {
            const h_slot = if (glob) slot else 0;
            const nw_s = if (glob) nws[slot] else nws[0];
            for (0..rows) |row| {
                var ssb: f64 = 0;
                for (0..H) |d| {
                    const hv: f64 = hs[row * h_stride + h_slot * H + d];
                    const mv: f64 = ms[(row * 4 + slot) * H + d];
                    var df: f64 = 0;
                    var dg: f64 = 0;
                    for (0..RANK) |rk| {
                        const zr = h.fb(zs[(slot * rows + row) * RANK + rk]);
                        df += zr * h.fb(fws[slot][d * RANK + rk]);
                        dg += zr * h.fb(gws[slot][d * RANK + rk]);
                    }
                    const bv = (1.0 - std.math.tanh(df)) * hv + std.math.tanh(dg) * mv;
                    blended[d] = bv;
                    ssb += bv * bv;
                }
                const rinv = 1.0 / @sqrt(ssb / H + @as(f64, norm_eps));
                for (0..H) |d| c.b16(outs[(slot * rows + row) * H + d], h.fb(nw_s[d]) * (blended[d] * rinv));
            }
        }
        try c.done();
    }

    // 1-row vs 16-row bit-identical row independence and NaN containment.
    const a = std.heap.page_allocator;
    const all_v = try a.dupe(u8, out_v.slice(u8, 3 * max_rows * H * 4));
    const all_out = try a.dupe(u8, out.slice(u8, 3 * max_rows * H * 2));
    defer a.free(all_v);
    defer a.free(all_out);

    var ci: h.Check = .{ .name = if (glob) "fg_decode3_independent_global" else "fg_decode3_independent_local" };
    const ozs = one_z.slice(u16, 3 * RANK);
    for (0..max_rows) |row| {
        const rh_off = row * h_stride * 4;
        const rm_off = row * 4 * H * 4;
        for (0..3) |slot| @memcpy(ozs[slot * RANK ..][0..RANK], zs[(slot * max_rows + row) * RANK ..][0..RANK]);
        try g.run("ik_msggate_fg_decode3", .{ .rows = 1, .width = H, .dim = RANK, .stride = h_stride, .flags = flags }, &.{ .{ .buf = hbuf, .off = rh_off }, .{ .buf = mbuf, .off = rm_off }, .{ .buf = one_z }, .{ .buf = fw0 }, .{ .buf = fw1 }, .{ .buf = fw2 }, .{ .buf = gw0 }, .{ .buf = gw1 }, .{ .buf = gw2 }, .{ .buf = one_v } }, .{ (H + 31) / 32, 1, 3 }, .{ 256, 1, 1 }, true);
        try g.run("ik_norm3_f32", .{ .rows = 1, .width = H, .flags = flags, .eps = norm_eps }, &.{ .{ .buf = one_v }, .{ .buf = nw0 }, .{ .buf = nw1 }, .{ .buf = nw2 }, .{ .buf = one_out } }, .{ 1, 3, 1 }, .{ 1024, 1, 1 }, true);
        for (0..3) |slot| {
            ci.same(one_v.slice(u8, 3 * H * 4)[slot * H * 4 ..][0 .. H * 4], all_v[(slot * max_rows + row) * H * 4 ..][0 .. H * 4]);
            ci.same(one_out.slice(u8, 3 * H * 2)[slot * H * 2 ..][0 .. H * 2], all_out[(slot * max_rows + row) * H * 2 ..][0 .. H * 2]);
        }
    }
    try ci.done();

    var kept_z: [3]u16 = undefined;
    for (0..3) |slot| {
        kept_z[slot] = zs[(slot * max_rows + 1) * RANK];
        zs[(slot * max_rows + 1) * RANK] = 0x7fc0;
    }
    try g.run("ik_msggate_fg_decode3", .{ .rows = max_rows, .width = H, .dim = RANK, .stride = h_stride, .flags = flags }, &.{ .{ .buf = hbuf }, .{ .buf = mbuf }, .{ .buf = zbuf }, .{ .buf = fw0 }, .{ .buf = fw1 }, .{ .buf = fw2 }, .{ .buf = gw0 }, .{ .buf = gw1 }, .{ .buf = gw2 }, .{ .buf = out_v } }, .{ (H + 31) / 32, 1, 3 }, .{ 256, 1, 1 }, true);
    try g.run("ik_norm3_f32", .{ .rows = max_rows, .width = H, .flags = flags, .eps = norm_eps }, &.{ .{ .buf = out_v }, .{ .buf = nw0 }, .{ .buf = nw1 }, .{ .buf = nw2 }, .{ .buf = out } }, .{ max_rows, 3, 1 }, .{ 1024, 1, 1 }, true);
    var cn: h.Check = .{ .name = if (glob) "fg_decode3_nan_contain_global" else "fg_decode3_nan_contain_local" };
    for (0..3) |slot| {
        for (0..max_rows) |row| if (row != 1) {
            cn.same(out_v.slice(u8, 3 * max_rows * H * 4)[(slot * max_rows + row) * H * 4 ..][0 .. H * 4], all_v[(slot * max_rows + row) * H * 4 ..][0 .. H * 4]);
            cn.same(out.slice(u8, 3 * max_rows * H * 2)[(slot * max_rows + row) * H * 2 ..][0 .. H * 2], all_out[(slot * max_rows + row) * H * 2 ..][0 .. H * 2]);
        };
    }
    for (0..3) |slot| zs[(slot * max_rows + 1) * RANK] = kept_z[slot];
    try cn.done();
}

fn caseLoopAttnPrep(g: *const h.Gpu, r: std.Random, glob: bool) !void {
    const rows: usize = 7;
    const stride: u32 = 16;
    const K: u32 = 4;
    const B: u32 = 2;
    const Q: u32 = if (glob) 3 else 1;
    const flags: u32 = if (glob) flag_global else 0;
    const scale_h: f32 = @floatCast(1.0 / @sqrt(@as(f64, H)));
    const kernel: [:0]const u8 = if (glob) "ik_loop_attn_prep3" else "ik_loop_attn_prep";

    const hb = try g.buf(rows * H * 2);
    const bank_blocks = try g.buf(B * stride * H * 4);
    const bank_partial = try g.buf(rows * H * 4);
    const msg_q = try g.buf(4 * H * 4);
    const blocks = try g.buf((K + 1) * stride * H * 4);
    const attn_q = try g.buf(Q * H * 4);

    const ref_msg = try g.buf(rows * 4 * H * 4);
    const ref_nrd = try g.buf(rows * H * 4);
    const ref_ai = try g.buf(rows * Q * H * 4);
    const ref_in = try g.buf(3 * rows * 2 * H * 2);

    const got_msg = try g.buf(rows * 4 * H * 4);
    const got_nrd = try g.buf(rows * H * 4);
    const got_ai = try g.buf(rows * Q * H * 4);
    const got_in = try g.buf(3 * rows * 2 * H * 2);

    const one_msg = try g.buf(4 * H * 4);
    const one_nrd = try g.buf(H * 4);
    const one_ai = try g.buf(Q * H * 4);
    const one_in = try g.buf(3 * 2 * H * 2);
    defer for ([_]mtl.Buffer{ hb, bank_blocks, bank_partial, msg_q, blocks, attn_q, ref_msg, ref_nrd, ref_ai, ref_in, got_msg, got_nrd, got_ai, got_in, one_msg, one_nrd, one_ai, one_in }) |b| b.deinit();

    const hs = h.fill16(hb, rows * H, r, 3.0);
    _ = h.fill32(bank_blocks, B * stride * H, r, 1.5);
    _ = h.fill32(bank_partial, rows * H, r, 1.5);
    _ = h.fill32(msg_q, 4 * H, r, 0.5);
    _ = h.fill32(blocks, (K + 1) * stride * H, r, 1.0);
    _ = h.fill32(attn_q, Q * H, r, 0.5);

    const grid = .{ rows, 1, 1 };
    const one = .{ 1024, 1, 1 };
    const nr: u32 = @intCast(rows);

    // Unfused 4-kernel reference chain.
    try g.run("ik_msg_depth", .{ .rows = nr, .width = H, .blocks = B, .queries = 4, .stride = stride, .scale = scale_h }, &.{ .{ .buf = bank_blocks }, .{ .buf = bank_partial }, .{ .buf = msg_q }, .{ .buf = ref_msg } }, grid, one, true);
    try g.run("ik_rms", .{ .rows = nr, .width = H, .eps = 1e-6 }, &.{ .{ .buf = hb }, .{ .buf = ref_nrd } }, grid, one, true);
    try g.run("ik_depth", .{ .rows = nr, .width = H, .blocks = K, .queries = Q, .stride = stride, .scale = scale_h }, &.{ .{ .buf = blocks }, .{ .buf = ref_nrd }, .{ .buf = attn_q }, .{ .buf = ref_ai } }, grid, one, true);
    try g.run("ik_msggate_prep3", .{ .rows = nr, .width = H, .stride = Q * H, .flags = flags }, &.{ .{ .buf = ref_ai }, .{ .buf = ref_msg }, .{ .buf = ref_in } }, .{ rows, 3, 1 }, one, true);

    // Fused prelude kernel.
    const fused_a: h.Args = .{ .rows = nr, .width = H, .dim = B, .blocks = K, .queries = Q, .stride = stride, .scale = scale_h };
    try g.run(kernel, fused_a, &.{ .{ .buf = hb }, .{ .buf = bank_blocks }, .{ .buf = bank_partial }, .{ .buf = msg_q }, .{ .buf = blocks }, .{ .buf = attn_q }, .{ .buf = got_msg }, .{ .buf = got_nrd }, .{ .buf = got_ai }, .{ .buf = got_in } }, grid, one, true);

    var c: h.Check = .{ .name = if (glob) "loop_attn_prep3_bitexact" else "loop_attn_prep_bitexact" };
    c.same(got_msg.slice(u8, rows * 4 * H * 4), ref_msg.slice(u8, rows * 4 * H * 4));
    c.same(got_nrd.slice(u8, rows * H * 4), ref_nrd.slice(u8, rows * H * 4));
    c.same(got_ai.slice(u8, rows * Q * H * 4), ref_ai.slice(u8, rows * Q * H * 4));
    c.same(got_in.slice(u8, 3 * rows * 2 * H * 2), ref_in.slice(u8, 3 * rows * 2 * H * 2));

    // Row independence.
    const a = std.heap.page_allocator;
    const all_msg = try a.dupe(u8, got_msg.slice(u8, rows * 4 * H * 4));
    const all_nrd = try a.dupe(u8, got_nrd.slice(u8, rows * H * 4));
    const all_ai = try a.dupe(u8, got_ai.slice(u8, rows * Q * H * 4));
    const all_in = try a.dupe(u8, got_in.slice(u8, 3 * rows * 2 * H * 2));
    defer for ([_][]u8{ all_msg, all_nrd, all_ai, all_in }) |s| a.free(s);

    var one_a = fused_a;
    one_a.rows = 1;
    for (0..rows) |row| {
        try g.run(kernel, one_a, &.{ .{ .buf = hb, .off = row * H * 2 }, .{ .buf = bank_blocks, .off = row * H * 4 }, .{ .buf = bank_partial, .off = row * H * 4 }, .{ .buf = msg_q }, .{ .buf = blocks, .off = row * H * 4 }, .{ .buf = attn_q }, .{ .buf = one_msg }, .{ .buf = one_nrd }, .{ .buf = one_ai }, .{ .buf = one_in } }, .{ 1, 1, 1 }, one, true);
        c.same(one_msg.slice(u8, 4 * H * 4), all_msg[row * 4 * H * 4 ..][0 .. 4 * H * 4]);
        c.same(one_nrd.slice(u8, H * 4), all_nrd[row * H * 4 ..][0 .. H * 4]);
        c.same(one_ai.slice(u8, Q * H * 4), all_ai[row * Q * H * 4 ..][0 .. Q * H * 4]);
        for (0..3) |slot| c.same(one_in.slice(u8, 3 * 2 * H * 2)[slot * 2 * H * 2 ..][0 .. 2 * H * 2], all_in[(slot * rows + row) * 2 * H * 2 ..][0 .. 2 * H * 2]);
    }

    // NaN containment on row 1.
    const kept = hs[1 * H];
    hs[1 * H] = 0x7fc0;
    try g.run(kernel, fused_a, &.{ .{ .buf = hb }, .{ .buf = bank_blocks }, .{ .buf = bank_partial }, .{ .buf = msg_q }, .{ .buf = blocks }, .{ .buf = attn_q }, .{ .buf = got_msg }, .{ .buf = got_nrd }, .{ .buf = got_ai }, .{ .buf = got_in } }, grid, one, true);
    for (0..rows) |row| if (row != 1) {
        c.same(got_msg.slice(u8, rows * 4 * H * 4)[row * 4 * H * 4 ..][0 .. 4 * H * 4], all_msg[row * 4 * H * 4 ..][0 .. 4 * H * 4]);
        c.same(got_nrd.slice(u8, rows * H * 4)[row * H * 4 ..][0 .. H * 4], all_nrd[row * H * 4 ..][0 .. H * 4]);
        c.same(got_ai.slice(u8, rows * Q * H * 4)[row * Q * H * 4 ..][0 .. Q * H * 4], all_ai[row * Q * H * 4 ..][0 .. Q * H * 4]);
        for (0..3) |slot| c.same(got_in.slice(u8, 3 * rows * 2 * H * 2)[(slot * rows + row) * 2 * H * 2 ..][0 .. 2 * H * 2], all_in[(slot * rows + row) * 2 * H * 2 ..][0 .. 2 * H * 2]);
    };
    hs[1 * H] = kept;
    try c.done();
}

fn caseLoopMlpPrep(g: *const h.Gpu, r: std.Random, boundary: bool) !void {
    const rows: usize = 7;
    const stride: u32 = 16;
    const K: u32 = 4;
    const bflag: u32 = if (boundary) flag_boundary else 0;
    const scale_h: f32 = @floatCast(1.0 / @sqrt(@as(f64, H)));
    const nr: u32 = @intCast(rows);

    const hb = try g.buf(rows * H * 2);
    const ya = try g.buf(rows * H * 2);
    const blocks = try g.buf((K + 1) * stride * H * 4);
    const query = try g.buf(H * 4);
    const mbuf = try g.buf(rows * 4 * H * 4);

    const ref_res = try g.buf(rows * H * 4);
    const ref_ns = try g.buf(rows * H * 4);
    const ref_mi = try g.buf(rows * H * 4);
    const ref_in = try g.buf(rows * 2 * H * 2);

    const got_res = try g.buf(rows * H * 4);
    const got_ns = try g.buf(rows * H * 4);
    const got_mi = try g.buf(rows * H * 4);
    const got_in = try g.buf(rows * 2 * H * 2);

    const one_res = try g.buf(H * 4);
    const one_ns = try g.buf(H * 4);
    const one_mi = try g.buf(H * 4);
    const one_in = try g.buf(2 * H * 2);
    defer for ([_]mtl.Buffer{ hb, ya, blocks, query, mbuf, ref_res, ref_ns, ref_mi, ref_in, got_res, got_ns, got_mi, got_in, one_res, one_ns, one_mi, one_in }) |b| b.deinit();

    const hs = h.fill16(hb, rows * H, r, 3.0);
    _ = h.fill16(ya, rows * H, r, 3.0);
    _ = h.fill32(blocks, (K + 1) * stride * H, r, 1.0);
    _ = h.fill32(query, H, r, 0.5);
    _ = h.fill32(mbuf, rows * 4 * H, r, 2.0);

    const grid = .{ rows, 1, 1 };
    const one = .{ 1024, 1, 1 };
    const m_off = 3 * H * 4;

    // Unfused 4-kernel reference chain.
    try g.run("ik_resid", .{ .rows = nr, .width = H, .flags = bflag }, &.{ .{ .buf = hb }, .{ .buf = ya }, .{ .buf = ref_res } }, .{ H, rows, 1 }, .{ 256, 1, 1 }, false);
    try g.run("ik_rms", .{ .rows = nr, .width = H, .flags = flag_f32_in, .eps = 1e-6 }, &.{ .{ .buf = ref_res }, .{ .buf = ref_ns } }, grid, one, true);
    try g.run("ik_depth", .{ .rows = nr, .width = H, .blocks = K + @intFromBool(boundary), .queries = 1, .stride = stride, .scale = scale_h }, &.{ .{ .buf = blocks }, .{ .buf = ref_ns }, .{ .buf = query }, .{ .buf = ref_mi } }, grid, one, true);
    try g.run("ik_msggate_prep", .{ .rows = nr, .width = H, .stride = H }, &.{ .{ .buf = ref_mi }, .{ .buf = mbuf, .off = m_off }, .{ .buf = ref_in } }, grid, one, true);

    // Fused prelude kernel.
    const fused_a: h.Args = .{ .rows = nr, .width = H, .blocks = K, .stride = stride, .flags = bflag, .scale = scale_h };
    try g.run("ik_loop_mlp_prep", fused_a, &.{ .{ .buf = hb }, .{ .buf = ya }, .{ .buf = got_res }, .{ .buf = blocks }, .{ .buf = got_ns }, .{ .buf = query }, .{ .buf = mbuf, .off = m_off }, .{ .buf = got_mi }, .{ .buf = got_in } }, grid, one, true);

    var c: h.Check = .{ .name = if (boundary) "loop_mlp_prep_boundary_bitexact" else "loop_mlp_prep_bitexact" };
    c.same(got_res.slice(u8, rows * H * 4), ref_res.slice(u8, rows * H * 4));
    c.same(got_ns.slice(u8, rows * H * 4), ref_ns.slice(u8, rows * H * 4));
    c.same(got_mi.slice(u8, rows * H * 4), ref_mi.slice(u8, rows * H * 4));
    c.same(got_in.slice(u8, rows * 2 * H * 2), ref_in.slice(u8, rows * 2 * H * 2));

    // Row independence.
    const a = std.heap.page_allocator;
    const all_res = try a.dupe(u8, got_res.slice(u8, rows * H * 4));
    const all_ns = try a.dupe(u8, got_ns.slice(u8, rows * H * 4));
    const all_mi = try a.dupe(u8, got_mi.slice(u8, rows * H * 4));
    const all_in = try a.dupe(u8, got_in.slice(u8, rows * 2 * H * 2));
    defer for ([_][]u8{ all_res, all_ns, all_mi, all_in }) |s| a.free(s);

    var one_a = fused_a;
    one_a.rows = 1;
    for (0..rows) |row| {
        const rm_off = (row * 4 + 3) * H * 4;
        try g.run("ik_loop_mlp_prep", one_a, &.{ .{ .buf = hb, .off = row * H * 2 }, .{ .buf = ya, .off = row * H * 2 }, .{ .buf = one_res }, .{ .buf = blocks, .off = row * H * 4 }, .{ .buf = one_ns }, .{ .buf = query }, .{ .buf = mbuf, .off = rm_off }, .{ .buf = one_mi }, .{ .buf = one_in } }, .{ 1, 1, 1 }, one, true);
        c.same(one_res.slice(u8, H * 4), all_res[row * H * 4 ..][0 .. H * 4]);
        c.same(one_ns.slice(u8, H * 4), all_ns[row * H * 4 ..][0 .. H * 4]);
        c.same(one_mi.slice(u8, H * 4), all_mi[row * H * 4 ..][0 .. H * 4]);
        c.same(one_in.slice(u8, 2 * H * 2), all_in[row * 2 * H * 2 ..][0 .. 2 * H * 2]);
    }

    // NaN containment on row 1 (inject into ya so boundary mode also sees NaN in row 1).
    const yas = ya.slice(u16, rows * H);
    const kept_h = hs[1 * H];
    const kept_y = yas[1 * H];
    hs[1 * H] = 0x7fc0;
    yas[1 * H] = 0x7fc0;
    try g.run("ik_loop_mlp_prep", fused_a, &.{ .{ .buf = hb }, .{ .buf = ya }, .{ .buf = got_res }, .{ .buf = blocks }, .{ .buf = got_ns }, .{ .buf = query }, .{ .buf = mbuf, .off = m_off }, .{ .buf = got_mi }, .{ .buf = got_in } }, grid, one, true);
    for (0..rows) |row| if (row != 1) {
        c.same(got_res.slice(u8, rows * H * 4)[row * H * 4 ..][0 .. H * 4], all_res[row * H * 4 ..][0 .. H * 4]);
        c.same(got_ns.slice(u8, rows * H * 4)[row * H * 4 ..][0 .. H * 4], all_ns[row * H * 4 ..][0 .. H * 4]);
        c.same(got_mi.slice(u8, rows * H * 4)[row * H * 4 ..][0 .. H * 4], all_mi[row * H * 4 ..][0 .. H * 4]);
        c.same(got_in.slice(u8, rows * 2 * H * 2)[row * 2 * H * 2 ..][0 .. 2 * H * 2], all_in[row * 2 * H * 2 ..][0 .. 2 * H * 2]);
    };
    hs[1 * H] = kept_h;
    yas[1 * H] = kept_y;
    try c.done();
}
