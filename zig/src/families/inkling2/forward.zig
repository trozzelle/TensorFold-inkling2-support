//! Inkling 2 rows through every layer at mlx-lm's rounding points; rows stay independent. A prompt chunk's layers go through a chunk
//! path in command buffers committed as they are encoded: the tensor-unit GEMM, attention and experts' chain where the load has them,
//! and for what lacks them (a GPU without tensor units, or experts the chain cannot take) the per-operation kernels; a decode window
//! of up to 16 rows takes the decode path.
const std = @import("std");
const mtl = @import("metal");
pub const counters = @import("../../core/gpu_counters.zig");
const exl3 = @import("../../core/exl3_projection.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const Model = @import("model.zig").Model;
const Glue = @import("model.zig").Glue;
const Legacy = @import("model.zig").Legacy;
const dense_prefill = @import("dense_prefill.zig");
const attention = @import("attention_prefill.zig");
const experts_prefill = @import("experts_prefill.zig");

/// glue.metal's ik_args, field for field.
pub const Args = extern struct { rows: u32 = 0, width: u32 = 0, heads: u32 = 0, kv: u32 = 0, dim: u32 = 0, pos: u32 = 0, blocks: u32 = 0, queries: u32 = 0, stride: u32 = 0, flags: u32 = 0, window: u32 = 0, extent: u32 = 0, ring: u32 = 0, eps: f32 = 0, scale: f32 = 0, cap: f32 = 0, alpha: f32 = 0, floor_n: f32 = 0, thr: f32 = 0 };
pub const f32_in: u32 = 1;
pub const global: u32 = 2;
pub const boundary: u32 = 4;

comptime {
    std.debug.assert(@sizeOf(Args) == 76);
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

/// `v` as the view type of one of the chunk kernels' modules (each holds a buffer and a byte offset).
fn view(comptime V: type, v: B) V {
    return .{ .buf = v.buf, .off = v.off };
}

/// What a forward writes besides the cache: finals, logits of the last row or every row, every layer's output.
/// `feature_taps` is what the DSpark head reads: each row's eight taps, [n][8][hidden] bf16 at the buffer: the hidden
/// states entering the `tap_layers` and the finals, which it forces on (a prompt chunk then runs the final stage too).
/// `prompt` marks rows that feed a prompt: they always take a chunk path, whatever their count: on the tensor units the GEMM for every
/// dense projection, the tiled attention and the experts' chain (the per-row grouped projections for a layer the chain cannot take),
/// and without tensor units the per-operation kernels; either way a row's bits do not depend on the rows or the chunk it is in;
/// other forwards of at most `window_rows` rows take the decode path (fused norms, stacked attention-input matvec, split-K
/// attention, the decode expert chain), and larger ones a chunk path.
/// `spans` records each stage group's GPU time by class (the `profile` and `prefill` modes), across the forward's command buffers;
/// `repeat` runs the loaded layers that many times in the one forward (profile only: the cache's position stays, so the repeats
/// re-read the same rows).
/// Diagnostic per-class ablation for timing: `stub` swaps each dispatch in the class for an empty kernel on the same grid,
/// `skip` drops the dispatch. Off by default (`Out.ablate == null`); when set, outputs are for timing only.
pub const Ablation = struct {
    pub const Mode = enum { run, stub, skip };
    mode: [counters.class_count]Mode = @splat(.run),
    now: ?counters.Class = null,
};

pub const Out = struct { finals: bool = false, logits: enum { none, last, all } = .last, dump: ?mtl.Buffer = null, timing: ?*Timing = null, taps: ?Taps = null, feature_taps: ?B = null, prompt: bool = false, spans: ?*counters.Recorder = null, repeat: u32 = 1, ablate: ?*Ablation = null, legacy: ?Legacy = null };
/// `encode`, `gpu` and `buffers` (the command buffers used) add up over forwards, and `longest` is the longest single command
/// buffer's GPU seconds so far; `committed`, `started`, `ended` and `woke` are the last forward's clocks in `seconds()`'s
/// timebase: the host's first commit, the GPU's start of the first command buffer and end of the last, and the host's return from
/// the wait.
pub const Timing = struct { encode: f64 = 0, gpu: f64 = 0, committed: f64 = 0, started: f64 = 0, ended: f64 = 0, woke: f64 = 0, buffers: u32 = 0, longest: f64 = 0 };
/// Rows times layers one command buffer of a chunk path holds at most. A prompt row takes about 10 microseconds a layer on the
/// tensor units (a 4,096-row layer 37 to 45 ms of GPU time on an M5 Ultra) and about 0.3 ms on the per-operation kernels (8,202
/// tokens through 48 layers in 126 s on that Mac), so this is about 30 ms or about a second of GPU time. The GPU kills a command
/// buffer that runs past its timeout (a kill at 15.6 s has been seen), and a 4,096-row chunk through 48 layers on the per-operation
/// kernels in one command buffer would take about a minute.
const buffer_row_layers = 3072;
/// Command buffers one forward uses at most: a queue holds 64 that are not yet complete, and creating one past that waits for
/// another to finish.
pub const max_buffers = 48;

/// Layers a chunk-path forward of `rows` rows commits after, of `total` layers: as many as keep a command buffer within
/// `buffer_row_layers` (at least one), and enough that no more than `max_buffers` buffers make the forward. At or past `total`
/// the forward is one command buffer.
pub fn commitEvery(rows: usize, total: usize) usize {
    const by_rows = @max(1, buffer_row_layers / rows);
    const by_count = (total + max_buffers - 1) / max_buffers;
    return @max(by_rows, by_count);
}

/// The command buffers a forward committed before its last, in order, and the host's clock at the first commit.
const Committed = struct {
    bufs: [max_buffers]mtl.CommandBuffer = undefined,
    n: usize = 0,
    first: f64 = 0,

    fn commit(c: *Committed, cb: mtl.CommandBuffer) void {
        std.debug.assert(c.n < c.bufs.len);
        if (c.n == 0) c.first = seconds();
        cb.commit();
        c.bufs[c.n] = cb;
        c.n += 1;
    }

    fn all(c: *const Committed) []const mtl.CommandBuffer {
        return c.bufs[0..c.n];
    }
};

fn checked(f: mtl.CommandBuffer) !void {
    if (f.failure()) |msg| {
        std.log.err("inkling2 forward: {s}", .{msg});
        return error.GpuFailed;
    }
}

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
/// ik_attn_part_w's threadgroup: 6 simdgroups, one per q head in a key head's group, sharing each 32-key slice in threadgroup memory.
const part_w_group = [3]usize{ 32, 6, 1 };

/// out = in W^T on `e` for `rows` rows of the bf16 [d.n, d.k] matrix `d` (torch's Linear layout), in slabs of at most 32:
/// the row-count kernel of the slab, whose bits do not depend on the slab. `in` rows are d.k wide, `out` rows d.n.
pub fn matvec(e: mtl.ComputeEncoder, m: *const Model, d: wts.Dense, in: B, out: B, rows: usize) void {
    var r: usize = 0;
    while (r < rows) {
        const k = @min(rows - r, 32);
        const g: Glue = if (k <= 8) .matvec_8 else if (k <= 16) .matvec_16 else .matvec_32;
        e.setPipeline(m.pipe(g));
        e.setBuffer(in.buf, in.off + r * d.k * 2, 0);
        e.setBuffer(d.t.buffer, d.t.offset, 1);
        e.setValue(MvArgs{ .rows = @intCast(k), .n = d.n, .k = d.k, .ldy = d.n }, 2);
        e.setBuffer(out.buf, out.off + r * d.n * 2, 3);
        e.dispatchGroups(mtl.Size.of((d.n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        r += k;
    }
}

const Enc = struct {
    e: *mtl.ComputeEncoder,
    m: *const Model,
    s: *st.Scratch,
    cache: *st.Cache,
    n: usize,
    /// Whether this forward takes the decode path.
    decode: bool = false,
    /// Whether this forward's rows run on the tensor-unit chunk kernels (the GEMM and the attention): a chunk, on a load that has them.
    tensor: bool = false,
    spans: ?*counters.Recorder = null,
    ablate: ?*Ablation = null,
    legacy: Legacy = .{},

    /// The weights' resources, declared on the current encoder (a span's encoder rotation drops them).
    fn resources(x: Enc) void {
        for (x.m.weights.expert_buffers) |buf| x.e.useResource(buf, mtl.ResourceUsage.read);
        for (x.m.weights.tables) |buf| x.e.useResource(buf, mtl.ResourceUsage.read);
    }

    fn ablateMode(x: Enc) Ablation.Mode {
        const ab = x.ablate orelse return .run;
        const kind = ab.now orelse return .run;
        return ab.mode[@backingInt(kind)];
    }

    /// Opens a span of `kind` when recording; spans never nest.
    fn begin(x: Enc, kind: counters.Class) !void {
        if (x.ablate) |ab| ab.now = kind;
        const sp = x.spans orelse return;
        const before = x.e.*;
        try sp.beginSpan(x.e, kind);
        if (before.id != x.e.id or before.m4 != x.e.m4) x.resources();
    }

    fn end(x: Enc) !void {
        if (x.ablate) |ab| ab.now = null;
        const sp = x.spans orelse return;
        const before = x.e.*;
        try sp.endSpan(x.e);
        if (before.id != x.e.id or before.m4 != x.e.m4) x.resources();
    }

    fn glue(x: Enc, g: Glue, a: Args, bufs: []const B, grid: [3]usize, group: [3]usize, groups: bool) void {
        const g3 = mtl.Size.of(grid[0], grid[1], grid[2]);
        const t3 = mtl.Size.of(group[0], group[1], group[2]);
        switch (x.ablateMode()) {
            .skip => return,
            .stub => {
                x.e.setPipeline(x.m.pipe(.nop));
                if (groups) x.e.dispatchGroups(g3, t3) else x.e.dispatchThreads(g3, t3);
                return;
            },
            .run => {},
        }
        x.e.setPipeline(x.m.pipe(g));
        x.e.setValue(a, 0);
        for (bufs, 1..) |r, slot| x.e.setBuffer(r.buf, r.off, slot);
        if (groups) x.e.dispatchGroups(g3, t3) else x.e.dispatchThreads(g3, t3);
    }

    /// The `n` hidden rows at `src` as tap `k` of the rows of eight taps at `to` ([n][8][hidden] bf16), in a taps span.
    fn tap(x: Enc, to: B, src: B, k: usize) !void {
        const H: usize = x.m.config.text.hidden;
        try x.begin(.taps);
        x.glue(.tap, .{ .rows = @intCast(x.n), .width = @intCast(H), .stride = @intCast(st.feature_taps * H) }, &.{ src, bo(to.buf, to.off + k * H * 2) }, .{ H, x.n, 1 }, .{ 256, 1, 1 }, false);
        try x.end();
    }

    /// out = in W^T for `rows` rows in slabs of at most 32 (one row's bits do not depend on the slab).
    fn mv(x: Enc, d: wts.Dense, in: B, out: B, rows: usize) void {
        switch (x.ablateMode()) {
            .skip => return,
            .stub => {
                var r: usize = 0;
                while (r < rows) {
                    const k = @min(rows - r, 32);
                    x.e.setPipeline(x.m.pipe(.nop));
                    x.e.dispatchGroups(mtl.Size.of((d.n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                    r += k;
                }
                return;
            },
            .run => matvec(x.e.*, x.m, d, in, out, rows),
        }
    }

    /// out = in W^T for `rows` rows of the bf16 [d.n, d.k] matrix `d`, bf16 out: the chunk kernels' tensor-unit GEMM (bf16 operands,
    /// f32 sums in an order that does not depend on the rows or the tile), or the matvec slabs of the decode and per-operation paths.
    fn proj(x: Enc, d: wts.Dense, in: B, out: B, rows: usize) void {
        if (x.tensor) return x.gemm(d, in, out, rows, .bf16);
        x.mv(d, in, out, rows);
    }

    /// The tensor-unit GEMM into rows of `d.n` values of `kind` (the router's logits are f32).
    fn gemm(x: Enc, d: wts.Dense, in: B, out: B, rows: usize, kind: dense_prefill.Out) void {
        std.debug.assert(x.tensor);
        dense_prefill.gemmRows(x.e.*, &x.m.chunk.?.dense, view(dense_prefill.Mat, in), .{ .buf = d.t.buffer, .off = d.t.offset, .n = d.n, .k = d.k }, view(dense_prefill.Mat, out), kind, rows, d.n);
    }

    /// Up to `max_segs` matvecs over the same k in one dispatch (at most `window_rows` rows). The segment table lives
    /// in layer `slot`'s part of `s.segs`, so no slot is rewritten within a command buffer; the kernel reaches its
    /// buffers by address, so they are declared on the encoder here.
    fn mvs(x: Enc, slot: usize, segs: []const Seg, rows: usize) void {
        std.debug.assert(segs.len <= st.max_segs and rows <= st.window_rows);
        switch (x.ablateMode()) {
            .skip => return,
            .stub => {
                var total: u32 = 0;
                for (segs) |sg| total += sg.w.n;
                x.e.setPipeline(x.m.pipe(.nop));
                x.e.dispatchGroups(mtl.Size.of((total + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                return;
            },
            .run => {},
        }
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
                x.proj(d.proj, dyn_in, b(x.s.dyn), n);
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

    /// The layer's routed experts for the chunk's rows through the tensor-unit chain (experts_prefill.zig): the picks sorted by
    /// expert, w13, the capped SwiGLU, w2 and the weighted sum, from s.xm with the picks and weights in s.eids and s.ew into
    /// s.moe, in the scratch's stage. The chain needs the encoder to be serial, the layer's trellis, scales and descriptor
    /// tables declared on it (`resources`), and every expert of both tables K2 or K3 (the layer's `xd_ok`).
    fn xp(x: Enc, l: *const wts.Layer) void {
        const c = &x.m.config.text;
        const s = x.s;
        const e = experts_prefill;
        std.debug.assert(x.tensor and l.xd_ok);
        e.encode(x.e.*, &x.m.chunk.?.experts, .{ .w13 = l.w13, .w2 = l.w2 }, view(e.B, b(s.xm)), .{ .ids = view(e.B, b(s.eids)), .weights = view(e.B, b(s.ew)) }, view(e.B, b(s.moe)), s.xp(), x.n, .{ .gate = c.gate_cap, .up = c.up_cap });
    }
};

/// `ids` at positions cache.len.. through the loaded layers (cache.len unmoved): the last command buffer waited. A decode step
/// commits its first `m.first_chunk` layers in a command buffer of their own and encodes the rest meanwhile, so the GPU
/// starts while the host is still encoding. A chunk's layers go in command buffers of bounded rows times layers (`commitEvery`),
/// each committed as soon as it is encoded. The queue runs the buffers in order, the one's results the other's inputs.
pub fn run(m: *const Model, s: *st.Scratch, cache: *st.Cache, ids: []const u32, out: Out) !void {
    const c = &m.config.text;
    const n = ids.len;
    if (n == 0 or n > s.prefill_rows or (out.logits == .all and n > st.window_rows)) return error.Inkling2RowsExceeded;
    if (out.prompt and n > cache.rows) return error.Inkling2CacheRingShort;
    if (cache.len + n > cache.capacity) return error.Inkling2ContextFull;
    const decode = !out.prompt and n <= st.window_rows;
    // the decode path's split attention keeps one partial for each 256-key block of the cache so far
    if (decode and cache.len + n > s.capacity) return error.Inkling2ScratchShort;
    if ((out.finals or out.feature_taps != null or out.logits != .none) and m.layers != c.layers) return error.Inkling2PartialModel;
    // a chunk takes the path the model does, and the scratch's buffers are made for it
    if (!decode and s.path != m.promptPath()) return error.Inkling2ScratchForAnotherPromptPath;
    for (ids) |id| if (id >= c.vocab) return error.InvalidInkling2Token;
    @memcpy(s.ids.slice(u32, n), ids);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const t0 = if (out.timing != null) seconds() else 0;
    var cb = m.queue.commandBuffer();
    var done: Committed = .{}; // the command buffers committed before `cb`, when the forward is in several
    var enc = cb.compute(.serial);
    var open = true; // `enc` is not yet ended
    const x = Enc{ .e = &enc, .m = m, .s = s, .cache = cache, .n = n, .decode = decode, .tensor = !decode and m.chunk != null, .spans = out.spans, .ablate = out.ablate, .legacy = out.legacy orelse m.legacy };
    if (out.spans) |sp| sp.beginCommand(cb) catch |err| {
        enc.end();
        return err;
    };
    errdefer if (out.spans) |sp| sp.cancelEncoding(&enc) catch {};
    // past a commit the GPU is running work on the scratch: an error hands nothing back until that has finished
    errdefer for (done.all()) |f| f.wait();
    errdefer if (open and out.spans == null) enc.end();
    x.resources();
    const H: usize = c.hidden;
    const nr: u32 = @intCast(n);
    x.glue(.embed, .{ .rows = nr, .width = c.hidden, .eps = c.eps }, &.{ b(s.ids), t(m.weights.embed.t), t(m.weights.embed_norm), b(s.h) }, .{ n, 1, 1 }, .{ 1024, 1, 1 }, true);
    const total = @as(usize, m.layers) * out.repeat;
    const split: usize = if (decode and out.spans == null and m.first_chunk < total) m.first_chunk else 0;
    const every: usize = if (decode) 0 else commitEvery(n, total);
    for (0..total) |li| {
        const i = li % m.layers;
        if (out.taps != null or out.feature_taps != null) for (tap_layers, 0..) |l, k| if (l == i) {
            if (out.taps) |tp| x.glue(.copy, .{ .rows = nr, .width = c.hidden }, &.{ b(s.h), bo(tp.h, k * n * H * 2) }, .{ H, n, 1 }, .{ 256, 1, 1 }, false);
            if (out.feature_taps) |ft| try x.tap(ft, b(s.h), k);
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
        if (li + 1 == split or (every != 0 and li + 1 < total and (li + 1) % every == 0)) {
            if (out.spans) |sp| try sp.endCommand(&enc) else enc.end();
            open = false;
            done.commit(cb);
            cb = m.queue.commandBuffer();
            if (out.spans) |sp| try sp.continueCommand(cb);
            enc = cb.compute(.serial);
            open = true;
            x.resources();
        }
    }
    if (out.finals or out.feature_taps != null or out.logits != .none) try final(x);
    if (out.feature_taps) |ft| try x.tap(ft, b(s.finals), tap_layers.len);
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
    open = false;
    if (out.timing) |tm| {
        tm.committed = if (done.n > 0) done.first else seconds();
        tm.encode += seconds() - t0;
    }
    const warm = decode and m.warm_wait and out.spans == null;
    const committed_at = if (done.n > 0) done.first else if (warm or out.timing != null) seconds() else 0;
    cb.commit();
    // spans are a profile's: its steps are waited for as before
    if (warm) waitWarm(cb, s.expected[n], committed_at) else cb.wait();
    if (out.timing) |tm| tm.woke = seconds();
    // the buffers before the last completed before it; the status keeps this from waiting for what is done
    for (done.all()) |f| switch (f.status()) {
        .completed, .@"error" => {},
        else => f.wait(),
    };
    for (done.all()) |f| try checked(f);
    try checked(cb);
    const began = (if (done.n > 0) done.bufs[0] else cb).gpuStartOr(committed_at);
    if (out.timing) |tm| {
        var spans: Spans = .{ .after = committed_at };
        for (done.all()) |f| spans.add(f.gpuStart(), f.gpuEnd());
        spans.add(cb.gpuStart(), cb.gpuEnd());
        tm.gpu += spans.gpu;
        tm.buffers += @intCast(done.n + 1);
        tm.longest = @max(tm.longest, spans.longest);
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

pub fn waitWarm(cb: mtl.CommandBuffer, expected: f64, committed: f64) void {
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
    const stride: u32 = @intCast(s.prefill_rows);
    const slab: usize = s.prefill_rows * H * 4;
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
            // the depth mix of the attention input: at the front of the stage on the chunk kernels (the attention and the experts use it
            // after the norms have read it), a buffer of its own on the per-operation path
            const mix = if (x.tensor) s.stage else s.ai;
            x.glue(.depth, .{ .rows = nr, .width = c.hidden, .blocks = K, .queries = Q, .stride = stride, .scale = scale_h }, &.{ b(s.blocks), nrd, t(q), b(mix) }, .{ n, 1, 1 }, one, true);
            for (0..Q) |j| x.glue(.norm, .{ .rows = nr, .width = c.hidden, .stride = Q * c.hidden, .flags = f32_in, .eps = c.eps }, &.{ bo(mix, j * H * 4), t(w.in_norm[j].?), xin[j] }, .{ n, 1, 1 }, one, true);
        }
    } else for (0..Q) |j| x.glue(.norm, .{ .rows = nr, .width = c.hidden, .stride = c.hidden, .eps = c.eps }, &.{ b(s.h), t(w.in_norm[j].?), xin[j] }, .{ n, 1, 1 }, one, true);
    try x.end();
    try x.begin(.dense);
    if (x.decode) {
        x.mvs(i, &.{ .{ .x = xq, .w = w.q, .y = b(s.q) }, .{ .x = xk, .w = w.k, .y = b(s.k) }, .{ .x = xv, .w = w.v, .y = b(s.v) }, .{ .x = xq, .w = w.r, .y = b(s.r) }, .{ .x = xq, .w = w.gate, .y = b(s.gate) }, .{ .x = xq, .w = w.sink, .y = b(s.sink) } }, n);
    } else {
        x.proj(w.q, xq, b(s.q), n);
        x.proj(w.k, xk, b(s.k), n);
        x.proj(w.v, xv, b(s.v), n);
        x.proj(w.r, xq, b(s.r), n);
        x.proj(w.gate, xq, b(s.gate), n);
        x.proj(w.sink, xq, b(s.sink), n);
    }
    try x.end();
    try x.conv(w.k_conv, b(s.k), xk, b(s.kc), i, 0, kvw, hd.kv);
    try x.conv(w.v_conv, b(s.v), xv, b(s.vc), i, 1, kvw, hd.kv);
    const keys = bo(cache.kv, cache.keys(i));
    const values = bo(cache.kv, cache.values(i));
    const ring = cache.ring(i);
    try x.begin(.attention_prepare);
    x.glue(.prep, .{ .rows = nr, .heads = hd.q, .kv = hd.kv, .dim = hd.dim, .pos = pos, .flags = if (glob) global else 0, .ring = ring, .eps = c.eps, .cap = c.qk_clip, .thr = c.rel_clip, .alpha = c.log_alpha, .floor_n = @floatFromInt(c.log_floor) }, &.{ b(s.q), b(s.kc), b(s.vc), b(s.r), t(w.q_scale), t(w.rel_scale), b(s.qn), keys, values, b(s.reln) }, .{ 2 * hd.q + 2 * hd.kv, n, 1 }, .{ 32, 1, 1 }, true);
    // the decode path's attention forms the relative bias itself from reln and the projection, and so do the chunk kernels (as a table
    // built right before the attention reads it); the per-operation path builds the table here
    if (!x.decode and !x.tensor) x.glue(.relbias, .{ .rows = nr, .heads = hd.q, .extent = p.extent, .dim = c.d_rel }, &.{ b(s.reln), t(w.rel_proj), b(s.pb) }, .{ p.extent, hd.q, n }, .{ 256, 1, 1 }, false);
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
        const a: Args = .{ .rows = nr, .heads = hd.q, .kv = hd.kv, .dim = hd.dim, .pos = pos, .window = window, .extent = p.extent, .ring = ring, .blocks = blocks, .queries = f0, .scale = scale };
        // Shared-KV global window attention wins once there are enough 256-key blocks to saturate the GPU; shorter contexts
        // keep per-row threadgroups for full occupancy.
        const shared_part = glob and !x.legacy.window_attention and ((n >= 3 and blocks >= 96) or (n >= 2 and blocks >= 256));
        if (shared_part) {
            x.glue(.attn_part_w, a, &.{ b(s.qn), keys, values, b(s.reln), t(w.rel_proj), b(s.part) }, .{ hd.kv, 1, blocks }, part_w_group, true);
        } else {
            x.glue(.attn_part, a, &.{ b(s.qn), keys, values, b(s.reln), t(w.rel_proj), b(s.part) }, .{ hd.kv, n, blocks }, part_group, true);
        }
        x.glue(.attn_merge, a, &.{ b(s.part), b(s.sink), b(s.gate), b(s.att) }, .{ hd.q, n, 1 }, .{ 32, 1, 1 }, true);
    } else if (x.tensor) {
        // the table of the chunk's relative biases is built in the stage and read by the tiled attention, 32 query rows of a head
        // a threadgroup; every view is at the chunk's first row
        const io: attention.Io = .{ .q = view(attention.B, b(s.qn)), .k = view(attention.B, keys), .v = view(attention.B, values), .sink = view(attention.B, b(s.sink)), .gate = view(attention.B, b(s.gate)), .out = view(attention.B, b(s.att)) };
        attention.encodeChunk(x.e.*, &m.chunk.?.attention, .{ .rows = nr, .p0 = pos, .global = @intFromBool(glob), .ring = ring }, io, view(attention.B, b(s.reln)), view(attention.B, t(w.rel_proj)), view(attention.B, b(s.stage)), s.prefill_rows);
    } else {
        // one simdgroup a (head, row) over the keys one by one, reading the table the per-operation path built
        x.glue(.attn, .{ .rows = nr, .heads = hd.q, .kv = hd.kv, .dim = hd.dim, .pos = pos, .window = window, .extent = p.extent, .ring = ring, .scale = scale }, &.{ b(s.qn), keys, values, b(s.pb), b(s.sink), b(s.gate), b(s.att) }, .{ hd.q, n, 1 }, .{ 32, 1, 1 }, true);
    }
    try x.end();
    try x.begin(.dense);
    x.proj(w.o, b(s.att), b(s.o), n);
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
    if (x.tensor) {
        x.gemm(w.router, b(s.xm), b(s.logits32), n, .f32);
    } else x.glue(.router, .{ .rows = nr, .width = c.experts, .dim = c.hidden }, &.{ b(s.xm), t(w.router.t), b(s.logits32) }, .{ (c.experts + 7) / 8, n, 1 }, .{ 256, 1, 1 }, true);
    x.glue(.route, .{ .rows = nr, .width = c.experts, .eps = 1e-20, .scale = c.route_scale }, &.{ b(s.logits32), t(w.bias), b(s.eids), b(s.ew) }, .{ n * 32, 1, 1 }, .{ 32, 1, 1 }, false);
    // a layer's experts take a chain when it can (the decode chain in a window, the tensor-unit chain in a chunk on the chunk kernels)
    // and the per-row path (ik_tiles, the grouped projections, the SwiGLU and the combine) when it cannot or the chunk has no chain: the
    // chains read the picks as routed and sort them themselves, and only the grouped projections need them sorted into tiles
    const slots = n * c.topk;
    const rows = st.tileRows(slots, c.experts);
    const chain = w.xd_ok and (x.decode or x.tensor);
    if (!chain) x.glue(.tiles, .{ .rows = @intCast(slots), .width = c.experts, .queries = c.topk, .stride = @intCast(rows) }, &.{ b(s.eids), b(s.texp), b(s.tslot), b(s.ttok) }, .{ 1, 1, 1 }, one, true);
    try x.end();
    const I: usize = c.expert_inter;
    try x.begin(.experts);
    if (chain) {
        if (x.tensor) x.xp(w) else x.xd(w, n);
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
    try x.begin(.final);
    x.glue(.rms, .{ .rows = nr, .width = c.hidden, .eps = 1e-6 }, &.{ b(s.h), b(s.nr) }, .{ n, 1, 1 }, one, true);
    x.glue(.depth, .{ .rows = nr, .width = c.hidden, .blocks = K, .queries = 1, .stride = @intCast(s.prefill_rows), .scale = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(c.hidden)))) }, &.{ b(s.blocks), b(s.nr), t(m.weights.out_query), b(s.fin) }, .{ n, 1, 1 }, one, true);
    x.glue(.norm, .{ .rows = nr, .width = c.hidden, .stride = c.hidden, .flags = f32_in, .eps = c.eps }, &.{ b(s.fin), t(m.weights.norm), b(s.finals) }, .{ n, 1, 1 }, one, true);
    try x.end();
}

/// Host seconds on the clock the command buffers' GPU start and end times use (mach absolute time).
pub fn seconds() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.UPTIME_RAW, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

/// GPU seconds of buffers run in order on one queue: one with no reported start runs from `after` (last end or commit).
const Spans = struct {
    after: f64,
    gpu: f64 = 0,
    longest: f64 = 0,

    fn add(s: *Spans, start: f64, end: f64) void {
        const one = end - mtl.startOr(start, s.after);
        s.after = end;
        s.gpu += one;
        s.longest = @max(s.longest, one);
    }
};

test "inkling2 forward: a buffer with no reported GPU start is timed from its commit or the end of the buffer ahead of it" {
    const uptime = 86_400.0; // a day of uptime: what end - 0 would have counted
    // reported starts are taken as they are
    var reported: Spans = .{ .after = uptime };
    reported.add(uptime + 0.5, uptime + 2);
    try std.testing.expectEqual(@as(f64, 1.5), reported.gpu);
    // the first buffer without a start runs from its commit
    var first: Spans = .{ .after = uptime };
    first.add(0, uptime + 2);
    try std.testing.expectEqual(@as(f64, 2), first.gpu);
    // in a chain each one without a start runs from the end of the one ahead of it, whatever that one reported
    var chain: Spans = .{ .after = uptime };
    chain.add(uptime + 1, uptime + 3);
    chain.add(0, uptime + 4);
    chain.add(uptime + 4.5, uptime + 5);
    chain.add(0, uptime + 7);
    try std.testing.expectEqual(@as(f64, 2 + 1 + 0.5 + 2), chain.gpu);
    try std.testing.expectEqual(@as(f64, 2), chain.longest);
    try std.testing.expect(chain.gpu < 10);
}
