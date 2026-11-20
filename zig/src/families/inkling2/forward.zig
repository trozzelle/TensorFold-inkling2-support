//! Inkling 2 rows through every layer in one command buffer at mlx-lm's rounding points; rows stay independent.
const std = @import("std");
const mtl = @import("metal");
const exl3 = @import("../../core/exl3_projection.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const Model = @import("model.zig").Model;
const Glue = @import("model.zig").Glue;

/// glue.metal's ik_args, field for field.
pub const Args = extern struct { rows: u32 = 0, width: u32 = 0, heads: u32 = 0, kv: u32 = 0, dim: u32 = 0, pos: u32 = 0, blocks: u32 = 0, queries: u32 = 0, stride: u32 = 0, flags: u32 = 0, window: u32 = 0, extent: u32 = 0, eps: f32 = 0, scale: f32 = 0, cap: f32 = 0, alpha: f32 = 0, floor_n: f32 = 0, thr: f32 = 0 };
pub const f32_in: u32 = 1;
pub const global: u32 = 2;
pub const boundary: u32 = 4;

comptime {
    std.debug.assert(@sizeOf(Args) == 72);
}

pub const B = struct { buf: mtl.Buffer, off: usize = 0 };
fn b(buf: mtl.Buffer) B {
    return .{ .buf = buf };
}
fn bo(buf: mtl.Buffer, off: usize) B {
    return .{ .buf = buf, .off = off };
}
fn t(x: wts.Tensor) B {
    return .{ .buf = x.buffer, .off = x.offset };
}

/// What a forward writes besides the cache: finals, logits of the last row or every row, every layer's output.
pub const Out = struct { finals: bool = false, logits: enum { none, last, all } = .last, dump: ?mtl.Buffer = null, timing: ?*Timing = null };
pub const Timing = struct { encode: f64 = 0, gpu: f64 = 0 };

/// core/bf16_matvec.metal's tf_bf16_mv_args.
const MvArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };

const Enc = struct {
    e: mtl.ComputeEncoder,
    m: *const Model,
    s: *st.Scratch,
    cache: *st.Cache,
    n: usize,

    fn glue(x: Enc, g: Glue, a: Args, bufs: []const B, grid: [3]usize, group: [3]usize, groups: bool) void {
        x.e.setPipeline(x.m.pipe(g));
        x.e.setValue(a, 0);
        for (bufs, 1..) |r, slot| x.e.setBuffer(r.buf, r.off, slot);
        const g3 = mtl.Size.of(grid[0], grid[1], grid[2]);
        const t3 = mtl.Size.of(group[0], group[1], group[2]);
        if (groups) x.e.dispatchGroups(g3, t3) else x.e.dispatchThreads(g3, t3);
    }

    /// out = in W^T for `rows` rows in slabs of at most 32 (one row's bits do not depend on the slab).
    fn mv(x: Enc, d: wts.Dense, in: B, out: B, rows: usize) void {
        var r: usize = 0;
        while (r < rows) {
            const k = @min(rows - r, 32);
            const g: Glue = if (k <= 1) .matvec_1 else if (k <= 2) .matvec_2 else if (k <= 4) .matvec_4 else if (k <= 8) .matvec_8 else if (k <= 16) .matvec_16 else .matvec_32;
            x.e.setPipeline(x.m.pipe(g));
            x.e.setBuffer(in.buf, in.off + r * d.k * 2, 0);
            x.e.setBuffer(d.t.buffer, d.t.offset, 1);
            x.e.setValue(MvArgs{ .rows = @intCast(k), .n = d.n, .k = d.k, .ldy = d.n }, 2);
            x.e.setBuffer(out.buf, out.off + r * d.n * 2, 3);
            x.e.dispatchGroups(mtl.Size.of((d.n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            r += k;
        }
    }

    /// A short convolution of `in` (width `w`) into `out` over ring `which` of layer `l`, then the ring advanced.
    fn conv(x: Enc, cv: wts.Conv, in: B, dyn_in: B, out: B, l: usize, which: usize, w: usize, kv: u32) void {
        const ring = bo(x.cache.conv, x.cache.ringAt(l, which));
        const n = x.n;
        var a = Args{ .rows = @intCast(n), .width = @intCast(w), .pos = @intCast(x.cache.len), .heads = kv };
        switch (cv) {
            .static => |taps| {
                a.scale = @sqrt(5.0);
                x.glue(.conv, a, &.{ in, ring, t(taps), out }, .{ w, n, 1 }, .{ 256, 1, 1 }, false);
            },
            .dynamic => |d| {
                x.mv(d.proj, dyn_in, b(x.s.dyn), n);
                a.scale = 3;
                x.glue(.conv_dyn, a, &.{ in, ring, t(d.taps), b(x.s.dyn), out }, .{ w, n, 1 }, .{ 256, 1, 1 }, false);
            },
        }
        x.glue(.ring, a, &.{ in, ring }, .{ w, @min(n, st.ring), 1 }, .{ 256, 1, 1 }, false);
    }

    /// The routed experts' grouped tiles through one EXL3 projection (in_ids: tokens for w13, slots for w2).
    fn experts(x: Enc, w: exl3.Experts, in: B, k: usize, in_rows: usize, out: B, slots: usize, rows: usize, in_ids: mtl.Buffer) !void {
        try exl3.encodeIndexed(&x.m.exl3, x.e, w, .{ .buffer = in.buf, .offset = in.off }, .{ .buffer = out.buf, .offset = out.off }, .{ .buffer = x.s.exl3 }, .{
            .count = rows,
            .x_stride = k,
            .y_stride = w.n,
            .grouped = true,
            .input_ids = .{ .buffer = in_ids },
            .output_ids = .{ .buffer = x.s.tslot },
            .input_rows = in_rows,
            .output_rows = slots,
            .routes = .{ .ids = .{ .buffer = x.s.texp } },
        });
    }
};

/// `ids` at positions cache.len.. through the loaded layers (cache.len unmoved): one command buffer, waited.
pub fn run(m: *const Model, s: *st.Scratch, cache: *st.Cache, ids: []const u32, out: Out) !void {
    const c = &m.config.text;
    const n = ids.len;
    if (n == 0 or n > st.prompt_rows or (out.logits == .all and n > st.window_rows)) return error.Inkling2RowsExceeded;
    if (cache.len + n > cache.capacity) return error.Inkling2ContextFull;
    if ((out.finals or out.logits != .none) and m.layers != c.layers) return error.Inkling2PartialModel;
    for (ids) |id| if (id >= c.vocab) return error.InvalidInkling2Token;
    @memcpy(s.ids.slice(u32, n), ids);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const t0 = if (out.timing != null) seconds() else 0;
    const cb = m.queue.commandBuffer();
    const x = Enc{ .e = cb.compute(.serial), .m = m, .s = s, .cache = cache, .n = n };
    for (m.weights.expert_buffers) |buf| x.e.useResource(buf, mtl.ResourceUsage.read);
    for (m.weights.tables) |buf| x.e.useResource(buf, mtl.ResourceUsage.read);
    const H: usize = c.hidden;
    const nr: u32 = @intCast(n);
    x.glue(.embed, .{ .rows = nr, .width = c.hidden, .eps = c.eps }, &.{ b(s.ids), t(m.weights.embed.t), t(m.weights.embed_norm), b(s.h) }, .{ n, 1, 1 }, .{ 1024, 1, 1 }, true);
    for (0..m.layers) |i| {
        try layer(x, @intCast(i));
        if (out.dump) |d| x.glue(.copy, .{ .rows = nr, .width = c.hidden }, &.{ b(s.h), bo(d, i * n * H * 2) }, .{ H, n, 1 }, .{ 256, 1, 1 }, false);
    }
    if (out.finals or out.logits != .none) final(x);
    if (out.logits != .none) {
        const first: usize = if (out.logits == .last) n - 1 else 0;
        const rows = n - first;
        x.mv(m.weights.unembed, bo(s.finals, first * H * 2), b(s.logits), rows);
        x.glue(.softcap, .{ .rows = @intCast(rows), .width = c.vocab, .extent = c.unpadded_vocab, .cap = c.logit_cap }, &.{b(s.logits)}, .{ c.vocab, rows, 1 }, .{ 256, 1, 1 }, false);
    }
    x.e.end();
    if (out.timing) |tm| tm.encode += seconds() - t0;
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2 forward: {s}", .{msg});
        return error.GpuFailed;
    }
    if (out.timing) |tm| tm.gpu += cb.gpuSeconds();
}

fn layer(x: Enc, i: u32) !void {
    const m = x.m;
    const s = x.s;
    const cache = x.cache;
    const c = &m.config.text;
    const p = m.plan.at(i);
    const w = &m.weights.layers[i];
    const n = x.n;
    const nr: u32 = @intCast(n);
    const H: usize = c.hidden;
    const hd = p.heads;
    const kvw: usize = @as(usize, hd.kv) * hd.dim;
    const K: u32 = p.blocks_in;
    const stride: u32 = st.prompt_rows;
    const slab: usize = st.prompt_rows * H * 4;
    const glob = p.kind == .global;
    const pos: u32 = @intCast(cache.len);
    const one = .{ 1024, 1, 1 };
    const scale_h: f32 = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(H))));
    // the input's unweighted norm; a boundary layer's goes straight into its block slot
    const nrd = if (p.boundary) bo(s.blocks, K * slab) else b(s.nr);
    x.glue(.rms, .{ .rows = nr, .width = c.hidden, .eps = 1e-6 }, &.{ b(s.h), nrd }, .{ n, 1, 1 }, one, true);
    const xin = [3]mtl.Buffer{ s.xq, s.xk, s.xv };
    const Q: u32 = if (glob) 3 else 1;
    if (w.attn_query) |q| {
        x.glue(.depth, .{ .rows = nr, .width = c.hidden, .blocks = K, .queries = Q, .stride = stride, .scale = scale_h }, &.{ b(s.blocks), nrd, t(q), b(s.ai) }, .{ n, 1, 1 }, one, true);
        for (0..Q) |j| x.glue(.norm, .{ .rows = nr, .width = c.hidden, .stride = Q * c.hidden, .flags = f32_in, .eps = c.eps }, &.{ bo(s.ai, j * H * 4), t(w.in_norm[j].?), b(xin[j]) }, .{ n, 1, 1 }, one, true);
    } else for (0..Q) |j| x.glue(.norm, .{ .rows = nr, .width = c.hidden, .stride = c.hidden, .eps = c.eps }, &.{ b(s.h), t(w.in_norm[j].?), b(xin[j]) }, .{ n, 1, 1 }, one, true);
    const xq = b(s.xq);
    const xk = if (glob) b(s.xk) else xq;
    const xv = if (glob) b(s.xv) else xq;
    x.mv(w.q, xq, b(s.q), n);
    x.mv(w.k, xk, b(s.k), n);
    x.mv(w.v, xv, b(s.v), n);
    x.mv(w.r, xq, b(s.r), n);
    x.mv(w.gate, xq, b(s.gate), n);
    x.mv(w.sink, xq, b(s.sink), n);
    x.conv(w.k_conv, b(s.k), xk, b(s.kc), i, 0, kvw, hd.kv);
    x.conv(w.v_conv, b(s.v), xv, b(s.vc), i, 1, kvw, hd.kv);
    const keys = bo(cache.kv, cache.keys(i));
    const values = bo(cache.kv, cache.values(i));
    x.glue(.prep, .{ .rows = nr, .heads = hd.q, .kv = hd.kv, .dim = hd.dim, .pos = pos, .flags = if (glob) global else 0, .eps = c.eps, .cap = c.qk_clip, .thr = c.rel_clip, .alpha = c.log_alpha, .floor_n = @floatFromInt(c.log_floor) }, &.{ b(s.q), b(s.kc), b(s.vc), b(s.r), t(w.q_scale), t(w.rel_scale), b(s.qn), keys, values, b(s.reln) }, .{ 2 * hd.q + 2 * hd.kv, n, 1 }, .{ 32, 1, 1 }, true);
    x.glue(.relbias, .{ .rows = nr, .heads = hd.q, .extent = p.extent, .dim = c.d_rel }, &.{ b(s.reln), t(w.rel_proj), b(s.pb) }, .{ p.extent, hd.q, n }, .{ 256, 1, 1 }, false);
    x.glue(.attn, .{ .rows = nr, .heads = hd.q, .kv = hd.kv, .dim = hd.dim, .pos = pos, .window = if (glob) 0 else c.window, .extent = p.extent, .scale = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(hd.dim)))) }, &.{ b(s.qn), keys, values, b(s.pb), b(s.sink), b(s.gate), b(s.att) }, .{ hd.q, n, 1 }, .{ 32, 1, 1 }, true);
    x.mv(w.o, b(s.att), b(s.o), n);
    x.conv(.{ .static = w.attn_conv }, b(s.o), b(s.o), b(s.ya), i, 2, H, 0);
    x.glue(.resid, .{ .rows = nr, .width = c.hidden, .flags = if (p.boundary) boundary else 0 }, &.{ b(s.h), b(s.ya), b(s.res) }, .{ H, n, 1 }, .{ 256, 1, 1 }, false);
    x.glue(.rms, .{ .rows = nr, .width = c.hidden, .flags = f32_in, .eps = 1e-6 }, &.{ b(s.res), b(s.ns) }, .{ n, 1, 1 }, one, true);
    x.glue(.depth, .{ .rows = nr, .width = c.hidden, .blocks = K + @intFromBool(p.boundary), .queries = 1, .stride = stride, .scale = scale_h }, &.{ b(s.blocks), b(s.ns), t(w.mlp_query), b(s.mi) }, .{ n, 1, 1 }, one, true);
    x.glue(.norm, .{ .rows = nr, .width = c.hidden, .stride = c.hidden, .flags = f32_in, .eps = c.eps }, &.{ b(s.mi), t(w.mlp_norm), b(s.xm) }, .{ n, 1, 1 }, one, true);
    x.glue(.router, .{ .rows = nr, .width = c.experts, .dim = c.hidden }, &.{ b(s.xm), t(w.router.t), b(s.logits32) }, .{ (c.experts + 7) / 8, n, 1 }, .{ 256, 1, 1 }, true);
    x.glue(.route, .{ .rows = nr, .width = c.experts, .eps = 1e-20, .scale = c.route_scale }, &.{ b(s.logits32), t(w.bias), b(s.eids), b(s.ew) }, .{ n * 32, 1, 1 }, .{ 32, 1, 1 }, false);
    const slots = n * c.topk;
    const rows = st.tileRows(slots, c.experts);
    x.glue(.tiles, .{ .rows = @intCast(slots), .width = c.experts, .queries = c.topk, .stride = @intCast(rows) }, &.{ b(s.eids), b(s.texp), b(s.tslot), b(s.ttok) }, .{ 1, 1, 1 }, one, true);
    const I: usize = c.expert_inter;
    try x.experts(w.w13, b(s.xm), H, n, b(s.gu), slots, rows, s.ttok);
    x.glue(.swiglu, .{ .rows = @intCast(slots), .width = c.expert_inter, .cap = c.gate_cap, .thr = c.up_cap }, &.{ b(s.gu), b(s.act) }, .{ I, slots, 1 }, .{ 256, 1, 1 }, false);
    try x.experts(w.w2, b(s.act), I, slots, b(s.ey), slots, rows, s.tslot);
    x.glue(.combine, .{ .rows = nr, .width = c.hidden, .queries = c.topk }, &.{ b(s.ey), b(s.ew), b(s.moe) }, .{ H, n, 1 }, .{ 256, 1, 1 }, false);
    x.conv(.{ .static = w.mlp_conv }, b(s.moe), b(s.moe), b(s.ym), i, 3, H, 0);
    x.glue(.out, .{ .rows = nr, .width = c.hidden }, &.{ b(s.res), b(s.ym), b(s.h) }, .{ H, n, 1 }, .{ 256, 1, 1 }, false);
}

/// The output query over every block and the last layer's normed output, then the final norm into s.finals.
fn final(x: Enc) void {
    const m = x.m;
    const s = x.s;
    const c = &m.config.text;
    const n = x.n;
    const nr: u32 = @intCast(n);
    const one = .{ 1024, 1, 1 };
    const K: u32 = @intCast(st.maxBlocks(c));
    x.glue(.rms, .{ .rows = nr, .width = c.hidden, .eps = 1e-6 }, &.{ b(s.h), b(s.nr) }, .{ n, 1, 1 }, one, true);
    x.glue(.depth, .{ .rows = nr, .width = c.hidden, .blocks = K, .queries = 1, .stride = st.prompt_rows, .scale = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(c.hidden)))) }, &.{ b(s.blocks), b(s.nr), t(m.weights.out_query), b(s.fin) }, .{ n, 1, 1 }, one, true);
    x.glue(.norm, .{ .rows = nr, .width = c.hidden, .stride = c.hidden, .flags = f32_in, .eps = c.eps }, &.{ b(s.fin), t(m.weights.norm), b(s.finals) }, .{ n, 1, 1 }, one, true);
}

fn seconds() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}
