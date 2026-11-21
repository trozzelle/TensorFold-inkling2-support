//! Inkling 2 rows through every layer in one command buffer at mlx-lm's rounding points; rows stay independent.
const std = @import("std");
const mtl = @import("metal");
pub const counters = @import("../../core/gpu_counters.zig");
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
/// `feature_taps` is what the DSpark head reads: each row's eight taps, [n][8][hidden] bf16 at the buffer: the hidden
/// states entering the `tap_layers` and the finals, which it forces on (a prompt chunk then runs the final stage too).
/// `prompt` marks rows that feed a prompt: they always take the one-kernel-per-op path, whatever their count; other
/// forwards of at most `window_rows` rows take the decode path (fused norms, stacked attention-input matvec, split-K
/// attention).
/// `spans` records each stage group's GPU time by class (the `profile` mode); `repeat` runs the loaded layers that many
/// times in the one command buffer (profile only: the cache's position stays, so the repeats re-read the same rows).
pub const Out = struct { finals: bool = false, logits: enum { none, last, all } = .last, dump: ?mtl.Buffer = null, timing: ?*Timing = null, taps: ?Taps = null, feature_taps: ?B = null, prompt: bool = false, spans: ?*counters.Recorder = null, repeat: u32 = 1 };
/// `encode` and `gpu` add up over forwards; `committed`, `started`, `ended` and `woke` are the last forward's clocks in
/// `seconds()`'s timebase: the host's commit, the GPU's start and end, and the host's return from the wait.
pub const Timing = struct { encode: f64 = 0, gpu: f64 = 0, committed: f64 = 0, started: f64 = 0, ended: f64 = 0, woke: f64 = 0 };
/// Most spans one layer opens (a global layer with dynamic convolutions is the most), so a recorder's capacity is
/// `spans_per_layer * layers * repeat` plus the final's.
pub const spans_per_layer = 32;
/// Rows recorded for the drafter and kernel studies: the inputs of the drafter's tapped layers, every layer's expert
/// picks, and the MoE inputs of `xm_layers` (layouts: h [7][n][H] bf16, eids [layers][n][8] u32, xm [2][n][H] bf16).
pub const Taps = struct { h: mtl.Buffer, eids: mtl.Buffer, xm: ?mtl.Buffer = null };
/// The target layers whose input the DSpark head reads; its eighth tap is the finals.
pub const tap_layers = [_]u32{ 5, 11, 17, 23, 29, 35, 41 };

comptime {
    std.debug.assert(tap_layers.len + 1 == st.feature_taps);
}

/// Layers whose MoE inputs the expert benches replay: one all-K3 layer and one mixed K2/K3 layer.
pub const xm_layers = [_]u32{ 0, 30 };

/// core/bf16_matvec.metal's tf_bf16_mv_args.
const MvArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };
/// stack.metal's ik_seg and ik_mvs_args.
const IkSeg = extern struct { x: u64, w: u64, y: u64, n: u32, k: u32, ldy: u32, first: u32 };
const MvsArgs = extern struct { rows: u32, nseg: u32, total: u32 };
/// One matvec of a stacked dispatch: `w` applied to the rows at `x`, into the rows at `y` (row stride `w.n`).
const Seg = struct { x: B, w: wts.Dense, y: B };
/// ik_attn_part's threadgroup: 24 simdgroups, 8 key groups of 32 keys (a 256-key block) by 3 pairs of q heads (a key head's six).
const part_group = [3]usize{ 32, 24, 1 };

const Enc = struct {
    e: *mtl.ComputeEncoder,
    m: *const Model,
    s: *st.Scratch,
    cache: *st.Cache,
    n: usize,
    /// Whether this forward takes the decode path.
    decode: bool = false,
    spans: ?*counters.Recorder = null,

    /// The weights' resources, declared on the current encoder (a span's encoder rotation drops them).
    fn resources(x: Enc) void {
        for (x.m.weights.expert_buffers) |buf| x.e.useResource(buf, mtl.ResourceUsage.read);
        for (x.m.weights.tables) |buf| x.e.useResource(buf, mtl.ResourceUsage.read);
    }

    /// Opens a span of `kind` when recording; spans never nest.
    fn begin(x: Enc, kind: counters.Class) !void {
        const sp = x.spans orelse return;
        const before = x.e.*;
        try sp.beginSpan(x.e, kind);
        if (before.id != x.e.id or before.m4 != x.e.m4) x.resources();
    }

    fn end(x: Enc) !void {
        const sp = x.spans orelse return;
        const before = x.e.*;
        try sp.endSpan(x.e);
        if (before.id != x.e.id or before.m4 != x.e.m4) x.resources();
    }

    fn glue(x: Enc, g: Glue, a: Args, bufs: []const B, grid: [3]usize, group: [3]usize, groups: bool) void {
        x.e.setPipeline(x.m.pipe(g));
        x.e.setValue(a, 0);
        for (bufs, 1..) |r, slot| x.e.setBuffer(r.buf, r.off, slot);
        const g3 = mtl.Size.of(grid[0], grid[1], grid[2]);
        const t3 = mtl.Size.of(group[0], group[1], group[2]);
        if (groups) x.e.dispatchGroups(g3, t3) else x.e.dispatchThreads(g3, t3);
    }

    /// The `n` hidden rows at `src` as tap `k` of the rows of eight taps at `to` ([n][8][hidden] bf16).
    fn tap(x: Enc, to: B, src: B, k: usize) void {
        const H: usize = x.m.config.text.hidden;
        x.glue(.tap, .{ .rows = @intCast(x.n), .width = @intCast(H), .stride = @intCast(st.feature_taps * H) }, &.{ src, bo(to.buf, to.off + k * H * 2) }, .{ H, x.n, 1 }, .{ 256, 1, 1 }, false);
    }

    /// out = in W^T for `rows` rows in slabs of at most 32 (one row's bits do not depend on the slab).
    fn mv(x: Enc, d: wts.Dense, in: B, out: B, rows: usize) void {
        var r: usize = 0;
        while (r < rows) {
            const k = @min(rows - r, 32);
            const g: Glue = if (k <= 8) .matvec_8 else if (k <= 16) .matvec_16 else .matvec_32;
            x.e.setPipeline(x.m.pipe(g));
            x.e.setBuffer(in.buf, in.off + r * d.k * 2, 0);
            x.e.setBuffer(d.t.buffer, d.t.offset, 1);
            x.e.setValue(MvArgs{ .rows = @intCast(k), .n = d.n, .k = d.k, .ldy = d.n }, 2);
            x.e.setBuffer(out.buf, out.off + r * d.n * 2, 3);
            x.e.dispatchGroups(mtl.Size.of((d.n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            r += k;
        }
    }

    /// Up to `max_segs` matvecs over the same k in one dispatch (at most `window_rows` rows). The segment table lives
    /// in layer `slot`'s part of `s.segs`, so no slot is rewritten within a command buffer; the kernel reaches its
    /// buffers by address, so they are declared on the encoder here.
    fn mvs(x: Enc, slot: usize, segs: []const Seg, rows: usize) void {
        std.debug.assert(segs.len <= st.max_segs and rows <= st.window_rows);
        const off = slot * st.seg_slot;
        const bytes = x.s.segs.slice(u8, off + st.seg_slot)[off..][0 .. segs.len * @sizeOf(IkSeg)];
        const table: []IkSeg = @alignCast(std.mem.bytesAsSlice(IkSeg, bytes));
        var first: u32 = 0;
        for (segs, table) |sg, *e| {
            e.* = .{ .x = sg.x.buf.gpuAddress() + sg.x.off, .w = sg.w.t.buffer.gpuAddress() + sg.w.t.offset, .y = sg.y.buf.gpuAddress() + sg.y.off, .n = sg.w.n, .k = sg.w.k, .ldy = sg.w.n, .first = first };
            first += sg.w.n;
            x.e.useResource(sg.x.buf, mtl.ResourceUsage.read);
            x.e.useResource(sg.w.t.buffer, mtl.ResourceUsage.read);
            x.e.useResource(sg.y.buf, mtl.ResourceUsage.write);
        }
        x.e.setPipeline(x.m.pipe(if (rows <= 8) .mvs_8 else .mvs_16));
        x.e.setBuffer(x.s.segs, off, 0);
        x.e.setValue(MvsArgs{ .rows = @intCast(rows), .nseg = @intCast(segs.len), .total = first }, 1);
        x.e.dispatchGroups(mtl.Size.of((first + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// A short convolution of `in` (width `w`) into `out` over ring `which` of layer `l`, then the ring advanced; a
    /// dynamic one's projection is a dense span, the convolution an attention_prepare span. The decode path advances the
    /// ring in the convolution's own dispatch.
    fn conv(x: Enc, cv: wts.Conv, in: B, dyn_in: B, out: B, l: usize, which: usize, w: usize, kv: u32) !void {
        const ring = bo(x.cache.conv, x.cache.ringAt(l, which));
        const n = x.n;
        var a = Args{ .rows = @intCast(n), .width = @intCast(w), .pos = @intCast(x.cache.len), .heads = kv };
        switch (cv) {
            .static => |taps| {
                a.scale = @sqrt(5.0);
                try x.begin(.attention_prepare);
                x.glue(if (x.decode) .conv_ring else .conv, a, &.{ in, ring, t(taps), out }, .{ w, n, 1 }, .{ 256, 1, 1 }, false);
            },
            .dynamic => |d| {
                try x.begin(.dense);
                x.mv(d.proj, dyn_in, b(x.s.dyn), n);
                try x.end();
                a.scale = 3;
                try x.begin(.attention_prepare);
                x.glue(if (x.decode) .conv_dyn_ring else .conv_dyn, a, &.{ in, ring, t(d.taps), b(x.s.dyn), out }, .{ w, n, 1 }, .{ 256, 1, 1 }, false);
            },
        }
        if (!x.decode) x.glue(.ring, a, &.{ in, ring }, .{ w, @min(n, st.ring), 1 }, .{ 256, 1, 1 }, false);
        try x.end();
    }

    /// The decode path's MLP branch in one dispatch: the convolution of `in` over ring 3 of layer `l`, the ring
    /// advanced, and the layer's output `h` = bf16(`res` + the convolution), in a norm span.
    fn convOut(x: Enc, taps: wts.Tensor, in: B, res: B, h: B, l: usize, w: usize) !void {
        std.debug.assert(x.decode);
        const ring = bo(x.cache.conv, x.cache.ringAt(l, 3));
        const n = x.n;
        const a = Args{ .rows = @intCast(n), .width = @intCast(w), .pos = @intCast(x.cache.len), .scale = @sqrt(5.0) };
        try x.begin(.norm);
        x.glue(.conv_out, a, &.{ in, ring, t(taps), res, h }, .{ w, n, 1 }, .{ 256, 1, 1 }, false);
        try x.end();
    }

    /// The routed experts' grouped tiles through one EXL3 projection (in_ids: tokens for w13, slots for w2).
    fn experts(x: Enc, w: exl3.Experts, in: B, k: usize, in_rows: usize, out: B, slots: usize, rows: usize, in_ids: mtl.Buffer) !void {
        try exl3.encodeIndexed(&x.m.exl3, x.e.*, w, .{ .buffer = in.buf, .offset = in.off }, .{ .buffer = out.buf, .offset = out.off }, .{ .buffer = x.s.exl3 }, .{
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

    /// The layer's routed experts for `rows` rows through the EXL3 decode chain (xd.metal): w13, the capped SwiGLU, w2 and
    /// the weighted sum in four dispatches, from s.xm with the picks and weights in s.eids and s.ew into s.moe. The
    /// kernels reach trellis and scales through the layer's descriptor tables, whose buffers the forward declares.
    fn xd(x: Enc, l: *const wts.Layer, rows: usize) void {
        const c = &x.m.config.text;
        const s = x.s;
        std.debug.assert(rows <= st.window_rows);
        const slots = rows * c.topk;
        const a = Args{ .rows = @intCast(slots), .cap = c.gate_cap, .thr = c.up_cap };
        const w13 = bo(l.w13.descriptors.buffer, l.w13.descriptors.offset);
        const w2 = bo(l.w2.descriptors.buffer, l.w2.descriptors.offset);
        // two 16-column output tiles a pair or down threadgroup, one simdgroup per 128 columns in mid and finish
        x.glue(.xd_pair, a, &.{ b(s.xm), b(s.eids), w13, b(s.planes) }, .{ l.w13.n / 32, slots, 2 }, .{ 128, 1, 1 }, true);
        x.glue(.xd_mid, a, &.{ b(s.planes), b(s.eids), w13, b(s.act) }, .{ l.w13.n / 128, slots, 1 }, .{ 32, 1, 1 }, true);
        x.glue(.xd_down, a, &.{ b(s.act), b(s.eids), w2, b(s.sums) }, .{ l.w2.n / 32, slots, 1 }, .{ 128, 1, 1 }, true);
        x.glue(.xd_finish, a, &.{ b(s.sums), b(s.eids), w2, b(s.ew), b(s.moe) }, .{ l.w2.n / 128, rows, 1 }, .{ 256, 1, 1 }, true);
    }
};

/// `ids` at positions cache.len.. through the loaded layers (cache.len unmoved): the command buffer waited. A decode step
/// commits its first `m.first_chunk` layers in a command buffer of their own and encodes the rest meanwhile, so the GPU
/// starts while the host is still encoding; the queue runs the two in order, the one's results the other's inputs.
pub fn run(m: *const Model, s: *st.Scratch, cache: *st.Cache, ids: []const u32, out: Out) !void {
    const c = &m.config.text;
    const n = ids.len;
    if (n == 0 or n > st.prompt_rows or (out.logits == .all and n > st.window_rows)) return error.Inkling2RowsExceeded;
    if (cache.len + n > cache.capacity) return error.Inkling2ContextFull;
    const decode = !out.prompt and n <= st.window_rows;
    // the decode path's split attention keeps one partial for each 256-key block of the cache so far
    if (decode and cache.len + n > s.capacity) return error.Inkling2ScratchShort;
    if ((out.finals or out.feature_taps != null or out.logits != .none) and m.layers != c.layers) return error.Inkling2PartialModel;
    for (ids) |id| if (id >= c.vocab) return error.InvalidInkling2Token;
    @memcpy(s.ids.slice(u32, n), ids);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const t0 = if (out.timing != null) seconds() else 0;
    var cb = m.queue.commandBuffer();
    var head: ?mtl.CommandBuffer = null; // the step's first command buffer, when it is committed before the rest is encoded
    var head_commit: f64 = 0;
    var enc = cb.compute(.serial);
    const x = Enc{ .e = &enc, .m = m, .s = s, .cache = cache, .n = n, .decode = decode, .spans = out.spans };
    if (out.spans) |sp| sp.beginCommand(cb) catch |err| {
        enc.end();
        return err;
    };
    errdefer if (out.spans) |sp| sp.cancelEncoding(&enc) catch {};
    x.resources();
    const H: usize = c.hidden;
    const nr: u32 = @intCast(n);
    x.glue(.embed, .{ .rows = nr, .width = c.hidden, .eps = c.eps }, &.{ b(s.ids), t(m.weights.embed.t), t(m.weights.embed_norm), b(s.h) }, .{ n, 1, 1 }, .{ 1024, 1, 1 }, true);
    const total = @as(usize, m.layers) * out.repeat;
    const split: usize = if (decode and out.spans == null and m.first_chunk < total) m.first_chunk else 0;
    for (0..total) |li| {
        const i = li % m.layers;
        if (out.taps != null or out.feature_taps != null) for (tap_layers, 0..) |l, k| if (l == i) {
            if (out.taps) |tp| x.glue(.copy, .{ .rows = nr, .width = c.hidden }, &.{ b(s.h), bo(tp.h, k * n * H * 2) }, .{ H, n, 1 }, .{ 256, 1, 1 }, false);
            if (out.feature_taps) |ft| x.tap(ft, b(s.h), k);
        };
        try layer(x, @intCast(i));
        if (out.taps) |tp| {
            // eight u32 picks a row are sixteen 16-bit words: the bf16 copy moves them bit for bit
            x.glue(.copy, .{ .rows = nr, .width = 16 }, &.{ b(s.eids), bo(tp.eids, i * n * 32) }, .{ 16, n, 1 }, .{ 16, 1, 1 }, false);
            if (tp.xm) |xmb| for (xm_layers, 0..) |l, k| if (l == i) {
                x.glue(.copy, .{ .rows = nr, .width = c.hidden }, &.{ b(s.xm), bo(xmb, k * n * H * 2) }, .{ H, n, 1 }, .{ 256, 1, 1 }, false);
            };
        }
        if (out.dump) |d| x.glue(.copy, .{ .rows = nr, .width = c.hidden }, &.{ b(s.h), bo(d, i * n * H * 2) }, .{ H, n, 1 }, .{ 256, 1, 1 }, false);
        if (li + 1 == split) {
            enc.end();
            head_commit = seconds();
            cb.commit();
            head = cb;
            cb = m.queue.commandBuffer();
            enc = cb.compute(.serial);
            x.resources();
        }
    }
    if (out.finals or out.feature_taps != null or out.logits != .none) try final(x);
    if (out.feature_taps) |ft| x.tap(ft, b(s.finals), tap_layers.len);
    if (out.logits != .none) {
        const first: usize = if (out.logits == .last) n - 1 else 0;
        const rows = n - first;
        try x.begin(.dense);
        x.mv(m.weights.unembed, bo(s.finals, first * H * 2), b(s.logits), rows);
        try x.end();
        try x.begin(.activation);
        x.glue(.softcap, .{ .rows = @intCast(rows), .width = c.vocab, .extent = c.unpadded_vocab, .cap = c.logit_cap }, &.{b(s.logits)}, .{ c.vocab, rows, 1 }, .{ 256, 1, 1 }, false);
        try x.end();
    }
    if (out.spans) |sp| try sp.finishEncoding(&enc) else enc.end();
    if (out.timing) |tm| {
        tm.committed = if (head != null) head_commit else seconds();
        tm.encode += seconds() - t0;
    }
    const warm = decode and m.warm_wait and out.spans == null;
    if (head == null) head_commit = if (warm) seconds() else 0;
    cb.commit();
    // spans are a profile's: its steps are waited for as before
    if (warm) waitWarm(cb, s.expected[n], head_commit) else cb.wait();
    if (out.timing) |tm| tm.woke = seconds();
    for ([_]?mtl.CommandBuffer{ head, cb }) |one| if (one) |f| if (f.failure()) |msg| {
        std.log.err("inkling2 forward: {s}", .{msg});
        return error.GpuFailed;
    };
    const began = (head orelse cb).gpuStart();
    if (out.timing) |tm| {
        tm.gpu += cb.gpuSeconds() + if (head) |f| f.gpuSeconds() else 0;
        tm.started = began;
        tm.ended = cb.gpuEnd();
    }
    if (warm) {
        // a shorter step is believed at once and a longer one by a quarter: one slow step (another process on the GPU, a
        // page-in) must not make the next sleep run past the GPU's end, while a lasting slowdown is followed in a few steps
        const span = cb.gpuEnd() - began;
        const old = s.expected[n];
        s.expected[n] = if (old == 0 or span < old) span else old + (span - old) / 4;
    }
}

/// A host core that blocked through a whole GPU step runs the work after it (the draw) at a fraction of its speed until it
/// has been busy for about 15 ms. A decode step therefore sleeps through the start of its GPU time and spins on the
/// command buffer's status for the last `warm_spin` seconds, with `warm_slack` more for the sleep's wake-up and the step's
/// own variation. `expected` is the GPU time this row count's recent steps took (0: none yet, so the wait is all spin).
const warm_spin = 0.015;
const warm_slack = 0.0015;

fn waitWarm(cb: mtl.CommandBuffer, expected: f64, committed: f64) void {
    const nap = committed + expected - warm_spin - warm_slack - seconds();
    if (expected > 0 and nap > 0) {
        const ts: std.c.timespec = .{ .sec = @intFromFloat(nap), .nsec = @intFromFloat((nap - @floor(nap)) * 1e9) };
        _ = std.c.nanosleep(&ts, null);
    }
    // a step far past its expected time stops burning the core: the blocking wait below takes over
    const give_up = seconds() + 2 * (warm_spin + warm_slack);
    while (seconds() < give_up) switch (cb.status()) {
        .completed, .@"error" => break,
        else => std.atomic.spinLoopHint(),
    };
    cb.wait(); // returns at once after the spin saw the end; it also orders the GPU's writes before the host's reads
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
    const Q: u32 = if (glob) 3 else 1;
    // the decode path's three inputs share s.xq, a query's rows a window apart
    const xrow: usize = n * H * 2;
    const xin = [3]B{ b(s.xq), if (x.decode) bo(s.xq, xrow) else b(s.xk), if (x.decode) bo(s.xq, 2 * xrow) else b(s.xv) };
    const xq = xin[0];
    const xk = if (glob) xin[1] else xq;
    const xv = if (glob) xin[2] else xq;
    try x.begin(.norm);
    if (x.decode) {
        const nm = w.in_norm;
        const norms = [3]B{ t(nm[0].?), t(nm[1] orelse nm[0].?), t(nm[2] orelse nm[0].?) };
        // the global layers' three queries take the wider instance of the kernel
        x.glue(if (w.attn_query != null and Q > 1) .rdn3 else .rdn, .{ .rows = nr, .width = c.hidden, .queries = if (w.attn_query != null) Q else 0, .blocks = K, .stride = stride, .eps = c.eps, .scale = scale_h }, &.{ b(s.h), b(s.blocks), nrd, if (w.attn_query) |q| t(q) else b(s.nr), norms[0], norms[1], norms[2], xq }, .{ n, 1, 1 }, one, true);
    } else x.glue(.rms, .{ .rows = nr, .width = c.hidden, .eps = 1e-6 }, &.{ b(s.h), nrd }, .{ n, 1, 1 }, one, true);
    if (w.attn_query) |q| {
        if (!x.decode) {
            x.glue(.depth, .{ .rows = nr, .width = c.hidden, .blocks = K, .queries = Q, .stride = stride, .scale = scale_h }, &.{ b(s.blocks), nrd, t(q), b(s.ai) }, .{ n, 1, 1 }, one, true);
            for (0..Q) |j| x.glue(.norm, .{ .rows = nr, .width = c.hidden, .stride = Q * c.hidden, .flags = f32_in, .eps = c.eps }, &.{ bo(s.ai, j * H * 4), t(w.in_norm[j].?), xin[j] }, .{ n, 1, 1 }, one, true);
        }
    } else for (0..Q) |j| x.glue(.norm, .{ .rows = nr, .width = c.hidden, .stride = c.hidden, .eps = c.eps }, &.{ b(s.h), t(w.in_norm[j].?), xin[j] }, .{ n, 1, 1 }, one, true);
    try x.end();
    try x.begin(.dense);
    if (x.decode) {
        x.mvs(i, &.{ .{ .x = xq, .w = w.q, .y = b(s.q) }, .{ .x = xk, .w = w.k, .y = b(s.k) }, .{ .x = xv, .w = w.v, .y = b(s.v) }, .{ .x = xq, .w = w.r, .y = b(s.r) }, .{ .x = xq, .w = w.gate, .y = b(s.gate) }, .{ .x = xq, .w = w.sink, .y = b(s.sink) } }, n);
    } else {
        x.mv(w.q, xq, b(s.q), n);
        x.mv(w.k, xk, b(s.k), n);
        x.mv(w.v, xv, b(s.v), n);
        x.mv(w.r, xq, b(s.r), n);
        x.mv(w.gate, xq, b(s.gate), n);
        x.mv(w.sink, xq, b(s.sink), n);
    }
    try x.end();
    try x.conv(w.k_conv, b(s.k), xk, b(s.kc), i, 0, kvw, hd.kv);
    try x.conv(w.v_conv, b(s.v), xv, b(s.vc), i, 1, kvw, hd.kv);
    const keys = bo(cache.kv, cache.keys(i));
    const values = bo(cache.kv, cache.values(i));
    try x.begin(.attention_prepare);
    x.glue(.prep, .{ .rows = nr, .heads = hd.q, .kv = hd.kv, .dim = hd.dim, .pos = pos, .flags = if (glob) global else 0, .eps = c.eps, .cap = c.qk_clip, .thr = c.rel_clip, .alpha = c.log_alpha, .floor_n = @floatFromInt(c.log_floor) }, &.{ b(s.q), b(s.kc), b(s.vc), b(s.r), t(w.q_scale), t(w.rel_scale), b(s.qn), keys, values, b(s.reln) }, .{ 2 * hd.q + 2 * hd.kv, n, 1 }, .{ 32, 1, 1 }, true);
    // the decode path's attention forms the bias itself from reln and the projection
    if (!x.decode) x.glue(.relbias, .{ .rows = nr, .heads = hd.q, .extent = p.extent, .dim = c.d_rel }, &.{ b(s.reln), t(w.rel_proj), b(s.pb) }, .{ p.extent, hd.q, n }, .{ 256, 1, 1 }, false);
    try x.end();
    try x.begin(.attention);
    const window: u32 = if (glob) 0 else c.window;
    const scale: f32 = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(hd.dim))));
    if (x.decode) {
        // 256-key blocks at absolute positions, from the block of the first row's first visible key to the block of the
        // last row's position: the blocks a row's keys do not lie in are empty for it and skipped by the merge, so its
        // bits are the same alone or in any window
        const first: u32 = if (window > 0 and pos + 1 > window) pos + 1 - window else 0;
        const f0 = first / st.key_block;
        const blocks = (pos + nr - 1) / st.key_block - f0 + 1;
        const a: Args = .{ .rows = nr, .heads = hd.q, .kv = hd.kv, .dim = hd.dim, .pos = pos, .window = window, .extent = p.extent, .blocks = blocks, .queries = f0, .scale = scale };
        x.glue(.attn_part, a, &.{ b(s.qn), keys, values, b(s.reln), t(w.rel_proj), b(s.part) }, .{ hd.kv, n, blocks }, part_group, true);
        x.glue(.attn_merge, a, &.{ b(s.part), b(s.sink), b(s.gate), b(s.att) }, .{ hd.q, n, 1 }, .{ 32, 1, 1 }, true);
    } else {
        x.glue(.attn, .{ .rows = nr, .heads = hd.q, .kv = hd.kv, .dim = hd.dim, .pos = pos, .window = window, .extent = p.extent, .scale = scale }, &.{ b(s.qn), keys, values, b(s.pb), b(s.sink), b(s.gate), b(s.att) }, .{ hd.q, n, 1 }, .{ 32, 1, 1 }, true);
    }
    try x.end();
    try x.begin(.dense);
    x.mv(w.o, b(s.att), b(s.o), n);
    try x.end();
    try x.conv(.{ .static = w.attn_conv }, b(s.o), b(s.o), b(s.ya), i, 2, H, 0);
    try x.begin(.norm);
    if (x.decode) {
        x.glue(.rrdn, .{ .rows = nr, .width = c.hidden, .blocks = K, .stride = stride, .flags = if (p.boundary) boundary else 0, .eps = c.eps, .scale = scale_h }, &.{ b(s.h), b(s.ya), b(s.res), b(s.blocks), b(s.ns), t(w.mlp_query), t(w.mlp_norm), b(s.xm) }, .{ n, 1, 1 }, one, true);
    } else {
        x.glue(.resid, .{ .rows = nr, .width = c.hidden, .flags = if (p.boundary) boundary else 0 }, &.{ b(s.h), b(s.ya), b(s.res) }, .{ H, n, 1 }, .{ 256, 1, 1 }, false);
        x.glue(.rms, .{ .rows = nr, .width = c.hidden, .flags = f32_in, .eps = 1e-6 }, &.{ b(s.res), b(s.ns) }, .{ n, 1, 1 }, one, true);
        x.glue(.depth, .{ .rows = nr, .width = c.hidden, .blocks = K + @intFromBool(p.boundary), .queries = 1, .stride = stride, .scale = scale_h }, &.{ b(s.blocks), b(s.ns), t(w.mlp_query), b(s.mi) }, .{ n, 1, 1 }, one, true);
        x.glue(.norm, .{ .rows = nr, .width = c.hidden, .stride = c.hidden, .flags = f32_in, .eps = c.eps }, &.{ b(s.mi), t(w.mlp_norm), b(s.xm) }, .{ n, 1, 1 }, one, true);
    }
    try x.end();
    try x.begin(.route);
    x.glue(.router, .{ .rows = nr, .width = c.experts, .dim = c.hidden }, &.{ b(s.xm), t(w.router.t), b(s.logits32) }, .{ (c.experts + 7) / 8, n, 1 }, .{ 256, 1, 1 }, true);
    x.glue(.route, .{ .rows = nr, .width = c.experts, .eps = 1e-20, .scale = c.route_scale }, &.{ b(s.logits32), t(w.bias), b(s.eids), b(s.ew) }, .{ n * 32, 1, 1 }, .{ 32, 1, 1 }, false);
    const slots = n * c.topk;
    const rows = st.tileRows(slots, c.experts);
    // the decode chain reads the picks as routed: only the grouped projections need them sorted into tiles
    const chain = x.decode and w.xd_ok;
    if (!chain) x.glue(.tiles, .{ .rows = @intCast(slots), .width = c.experts, .queries = c.topk, .stride = @intCast(rows) }, &.{ b(s.eids), b(s.texp), b(s.tslot), b(s.ttok) }, .{ 1, 1, 1 }, one, true);
    try x.end();
    const I: usize = c.expert_inter;
    try x.begin(.experts);
    if (chain) {
        x.xd(w, n);
    } else {
        try x.experts(w.w13, b(s.xm), H, n, b(s.gu), slots, rows, s.ttok);
        x.glue(.swiglu, .{ .rows = @intCast(slots), .width = c.expert_inter, .cap = c.gate_cap, .thr = c.up_cap }, &.{ b(s.gu), b(s.act) }, .{ I, slots, 1 }, .{ 256, 1, 1 }, false);
        try x.experts(w.w2, b(s.act), I, slots, b(s.ey), slots, rows, s.tslot);
        x.glue(.combine, .{ .rows = nr, .width = c.hidden, .queries = c.topk }, &.{ b(s.ey), b(s.ew), b(s.moe) }, .{ H, n, 1 }, .{ 256, 1, 1 }, false);
    }
    try x.end();
    if (x.decode) {
        try x.convOut(w.mlp_conv, b(s.moe), b(s.res), b(s.h), i, H);
    } else {
        try x.conv(.{ .static = w.mlp_conv }, b(s.moe), b(s.moe), b(s.ym), i, 3, H, 0);
        try x.begin(.norm);
        x.glue(.out, .{ .rows = nr, .width = c.hidden }, &.{ b(s.res), b(s.ym), b(s.h) }, .{ H, n, 1 }, .{ 256, 1, 1 }, false);
        try x.end();
    }
}

/// The output query over every block and the last layer's normed output, then the final norm into s.finals.
fn final(x: Enc) !void {
    const m = x.m;
    const s = x.s;
    const c = &m.config.text;
    const n = x.n;
    const nr: u32 = @intCast(n);
    const one = .{ 1024, 1, 1 };
    const K: u32 = @intCast(st.maxBlocks(c));
    try x.begin(.norm);
    x.glue(.rms, .{ .rows = nr, .width = c.hidden, .eps = 1e-6 }, &.{ b(s.h), b(s.nr) }, .{ n, 1, 1 }, one, true);
    x.glue(.depth, .{ .rows = nr, .width = c.hidden, .blocks = K, .queries = 1, .stride = st.prompt_rows, .scale = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(c.hidden)))) }, &.{ b(s.blocks), b(s.nr), t(m.weights.out_query), b(s.fin) }, .{ n, 1, 1 }, one, true);
    x.glue(.norm, .{ .rows = nr, .width = c.hidden, .stride = c.hidden, .flags = f32_in, .eps = c.eps }, &.{ b(s.fin), t(m.weights.norm), b(s.finals) }, .{ n, 1, 1 }, one, true);
    try x.end();
}

/// Host seconds on the clock the command buffers' GPU start and end times use (mach absolute time).
pub fn seconds() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.UPTIME_RAW, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}
