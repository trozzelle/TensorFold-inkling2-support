//! Inkling 2's DSpark draft head as a lane drafter over the target's taps. Each committed row's eight taps are mixed, normed and
//! projected into five draft layers' context keys and values, kept in a ring of the newest 1,023 positions a layer. A hold runs
//! one block of eight rows (the pending token's embedding, then seven mask rows) through those layers, takes the target's
//! unembedding of rows 1 to 7 as base logits, and names the proposals with the Markov GRU: each step folds the previous token,
//! a state and two features into a latent whose product with the folded unembedding (dspark_weights.fold) joins the base
//! logits before the softcap, the padded vocabulary's mask and the argmax.
//!
//! The arithmetic is the reference's (omlx/patches/inkling2/dspark.py on mlx 0.32): kernels/metal/inkling2/dspark.metal holds the
//! rounding points; the layers' projections are the target's bf16 matvec kernels or, with the weights quantized at load (the
//! default, as the reference runs them), core's 8-bit tiles (dspark_qmv.zig); the target's embedding and unembedding serve the block.
//! An absorb commits its command buffer without waiting; the hold behind it runs after it on the same queue and is the round's one
//! host wait. The hold's proposals do not depend on its depth: a depth takes the first proposals of the same block pass.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const mix = @import("dspark_alpha.zig");
const ks = @import("kernel_sources");
const cfg = @import("config.zig");
const dw = @import("dspark_weights.zig");
const Qmv = @import("dspark_qmv.zig").Qmv;
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;
const Glue = @import("model.zig").Glue;

const dr = lanes.drafter;
const Stream = lanes.Stream;
const B = fwd.B;

// The shapes the kernels are written for.
const hidden = 4608;
const heads = 48;
const kv_heads = 8;
const head_dim = 128;
const kv_width = kv_heads * head_dim;
const rank = 256;
const taps = 8;
/// Tap mixes of a row: one for each of the five draft layers, and the Markov head's.
const mixes = 6;
const draft_layers = 5;
/// Bytes of a row's taps: [8][hidden] bf16.
const tap_row_bytes = taps * hidden * 2;
/// Rows of a hold's block: the pending token's and seven mask rows.
pub const block = 8;
/// Drafts a hold makes at most: one for each of the block's rows after the first.
pub const max_depth = block - 1;
/// Newest context positions a block attends to, and a layer ring's slots: its modulus, with `ring_pad` more that mirror the first
/// slots (ikd_ctx_prep) so the attention reads 8 slots in a row past the last without wrapping.
const window = 1023;
const ring = 1031;
const ring_pad = 7;
/// Keys of one attention block (the kernels' IKD_KEYS); a (row, head) has at most `max_blocks` of them, the block's own among them.
const key_block = 256;
const max_blocks = 7;
const part_slot = 130;
/// Rows one absorb pass projects at most (a prompt chunk's).
const absorb_rows = 512;
/// Ids of the vocabulary that groups of 256 ids' argmax partials cover.
const pick_group = 256;

/// The prior the depth rule starts from: how likely draft j is kept when drafts 1 to j - 1 were, over recorded greedy runs of
/// the reference head.
const prior = [max_depth]f64{ 0.857, 0.859, 0.861, 0.875, 0.901, 0.902, 0.905 };

/// The fields of a draft layer that are projections (dw.Proj).
const proj_fields = [_][]const u8{ "q", "k", "v", "k_ctx", "v_ctx", "o", "gate", "up", "down" };

const kernel_names = [_][:0]const u8{ "ikd_mix", "ikd_ctx_prep", "ikd_block_prep", "ikd_attn_part", "ikd_attn_merge", "ikd_swiglu", "ikd_add", "ikd_rnn_proj", "ikd_gru", "ikd_fold_pick", "ikd_pick", "ikd_fold_t" };
const Kernel = enum { mix, ctx_prep, block_prep, attn_part, attn_merge, swiglu, add, rnn_proj, gru, fold_pick, pick, fold_t };

comptime {
    const members = @typeInfo(Kernel).@"enum".field_names;
    std.debug.assert(members.len == kernel_names.len);
    for (members, kernel_names) |member, name| std.debug.assert(std.mem.eql(u8, "ikd_" ++ member, name));
}

// The kernels' argument structs, field for field.
const MixArgs = extern struct { rows: u32, width: u32, tap_stride: u32, mix_stride: u32, eps: f32 };
const PrepArgs = extern struct { rows: u32, heads: u32, kv: u32, pos: u32, ring: u32, eps: f32, base_log2: f32 };
const AttnArgs = extern struct { heads: u32, kv: u32, end: u32, window: u32, ring: u32, blocks: u32, scale: f32 };
const EwArgs = extern struct { rows: u32, width: u32 };
const ProjArgs = extern struct { n: u32, at: u32 };
const FoldArgs = extern struct { vocab: u32, valid: u32, flags: u32, cap: f32 };
const PickArgs = extern struct { groups: u32, step: u32 };
/// ikd_fold_pick's flag: also write every id's logit.
const write_logits: u32 = 1;

/// Blocks a (row, head) of the attention has when the block starts at position `end` and attends the newest `win` positions
/// before it: the 256-key blocks, anchored at absolute positions, that meet [end - win, end), then the block's own keys; just
/// those when nothing precedes it.
fn attnBlocks(end: u32, win: u32) u32 {
    const lo = if (end > win) end - win else 0;
    if (end == lo) return 1;
    return (end - 1) / key_block - lo / key_block + 2;
}

/// The drafter's scratch: absorb's mixed rows and context keys and values, the block's activations, base logits and the
/// Markov step's working buffers. Every field is a buffer, so a residency hook joins them all.
const Scratch = struct {
    mixed: mtl.Buffer, // [6][rows][hidden]: the tap mixes of an absorb's rows
    kc: mtl.Buffer, // [rows][kv_width]: a layer's context keys and values
    vc: mtl.Buffer,
    h0: mtl.Buffer, // [8][hidden]: the block's input rows; rows 1 to 7 are mask_embed from the start
    h: mtl.Buffer, // [8][hidden]: the block's residual stream
    x: mtl.Buffer, // [8][hidden]: normed rows
    y: mtl.Buffer, // [8][hidden]: a branch's output
    q: mtl.Buffer, // [8][heads * head_dim]
    qn: mtl.Buffer, // [8][heads * head_dim]: the rotated queries
    att: mtl.Buffer, // [8][heads * head_dim]: the attention's output
    k: mtl.Buffer, // [8][kv_width]: the block's keys and values, as projected and as normed and rotated
    v: mtl.Buffer,
    kb: mtl.Buffer,
    vb: mtl.Buffer,
    gate: mtl.Buffer, // [8][inter]
    up: mtl.Buffer,
    act: mtl.Buffer,
    part: mtl.Buffer, // f32 [8][heads][max_blocks][130]: the attention's partials
    hn: mtl.Buffer, // [8][hidden]: the final norm of rows 1 to 7
    base: mtl.Buffer, // [7][vocab]: the unembedding of those rows
    proj: mtl.Buffer, // [3 * rank]: the Markov projection
    latent: mtl.Buffer, // [rank]
    best: mtl.Buffer, // f32 [groups]: each group of ids' best logit and its id
    best_id: mtl.Buffer, // u32 [groups]
    /// The steps' logits, when asked for: bf16 [7][vocab].
    capture: ?mtl.Buffer = null,

    /// The buffers made up front, in `bytes`' order.
    const made = [_][]const u8{ "mixed", "kc", "vc", "h0", "h", "x", "y", "q", "qn", "att", "k", "v", "kb", "vb", "gate", "up", "act", "part", "hn", "base", "proj", "latent", "best", "best_id" };

    /// Bytes of each buffer for a vocabulary of `vocab` ids and an MLP of width `inter`.
    fn bytes(vocab: usize, inter: usize) [made.len]usize {
        const rows = block;
        const groups = (vocab + pick_group - 1) / pick_group;
        return .{
            mixes * absorb_rows * hidden * 2,
            absorb_rows * kv_width * 2,
            absorb_rows * kv_width * 2,
            rows * hidden * 2,
            rows * hidden * 2,
            rows * hidden * 2,
            rows * hidden * 2,
            rows * heads * head_dim * 2,
            rows * heads * head_dim * 2,
            rows * heads * head_dim * 2,
            rows * kv_width * 2,
            rows * kv_width * 2,
            rows * kv_width * 2,
            rows * kv_width * 2,
            rows * inter * 2,
            rows * inter * 2,
            rows * inter * 2,
            rows * heads * max_blocks * part_slot * 4,
            rows * hidden * 2,
            max_depth * vocab * 2,
            3 * rank * 2,
            rank * 2,
            groups * 4,
            groups * 4,
        };
    }

    /// Every buffer made, and the block's input rows ready: row 0 is the pending token's embedding, each hold's to write, and
    /// rows 1 to 7 are the mask embedding.
    fn init(device: mtl.Device, vocab: usize, w: *const dw.Weights) !Scratch {
        const sz = bytes(vocab, w.layers[0].gate.n());
        var s: Scratch = undefined;
        s.capture = null;
        var done: [made.len]bool = @splat(false);
        errdefer inline for (made, 0..) |f, i| if (done[i]) @field(s, f).deinit();
        inline for (made, 0..) |f, i| {
            @field(s, f) = try device.buffer(@max(sz[i], 16), mtl.ResourceOptions.shared);
            done[i] = true;
        }
        const rows = s.h0.slice(u16, block * hidden);
        for (1..block) |r| @memcpy(rows[r * hidden ..][0..hidden], w.mask_embed.host(u16));
        return s;
    }

    fn deinit(s: *Scratch) void {
        inline for (made) |f| @field(s.*, f).deinit();
        if (s.capture) |c| c.deinit();
    }
};

comptime {
    std.debug.assert(@typeInfo(Scratch).@"struct".field_names.len == Scratch.made.len + 1);
}

/// One stream's draft state: each layer's context ring, the Markov feature of its newest absorbed row, the GRU's state and
/// the hold's token row (the pending token, then its proposals).
const Lane = struct {
    k: [draft_layers]mtl.Buffer,
    v: [draft_layers]mtl.Buffer,
    f_rnn: mtl.Buffer,
    state: mtl.Buffer,
    seq: mtl.Buffer,
    end: u64 = 0, // context rows absorbed: positions [from, end)
    from: u64 = 0, // the first position held (a resumed prompt state's drafter context is not kept)
    held: [max_depth]u32 = @splat(0),
    held_len: u32 = 0,
};

/// How deep a served round drafts (INKLING_DRAFT_DEPTH): a whole number from 1 to `max_depth` holds every round at that depth,
/// `auto` leaves it to the lane core's cost rule, which chooses among 1 to `max_depth` by what the drafts and the window
/// forwards cost.
pub const Depth = union(enum) {
    fixed: u8,
    auto,

    /// What an unset or blank setting means.
    pub const default: Depth = .{ .fixed = 4 };

    /// The setting's text: null or blank is the default, `auto` (any case) or a whole number from 1 to `max_depth`.
    pub fn parse(text: ?[]const u8) error{BadDraftDepth}!Depth {
        const t = std.mem.trim(u8, text orelse "", " \t");
        if (t.len == 0) return default;
        if (std.ascii.eqlIgnoreCase(t, "auto")) return .auto;
        const n = std.fmt.parseInt(u8, t, 10) catch return error.BadDraftDepth;
        if (n == 0 or n > max_depth) return error.BadDraftDepth;
        return .{ .fixed = n };
    }

    /// Drafts a hold makes at most: the drafter's `Options.depth`, which it offers the depth rule and times its drafts at.
    pub fn most(d: Depth) u32 {
        return switch (d) {
            .fixed => |n| n,
            .auto => max_depth,
        };
    }

    /// The lane core's `Model.fixed_depth` (null: the cost rule chooses).
    pub fn fixedDepth(d: Depth) ?u8 {
        return switch (d) {
            .fixed => |n| n,
            .auto => null,
        };
    }
};

/// The lane core's facts for a load that drafts: `target`'s through `drafted` (the drafter's depth, step cost and prior among
/// them) and `depth`'s fixed depth. `target` should carry the window forwards' timed costs, which the cost rule of `auto` reads.
pub fn laneFacts(drafted: *const lanes.drafted.Drafted, target: lanes.Model, depth: Depth) lanes.Model {
    var m = drafted.facts(target);
    m.fixed_depth = depth.fixedDepth();
    return m;
}

/// `d` in front of `target`: the target is told the drafter's taps, so its forwards write them, and its logprob rows for a
/// stream's first token stay on offer.
pub fn wrap(gpa: std.mem.Allocator, target: lanes.backend.Backend, d: *Drafter) !lanes.drafted.Drafted {
    var w = try lanes.drafted.Drafted.init(gpa, target, d.drafter());
    w.first_rows = true;
    return w;
}

pub const Options = struct {
    /// Drafts a round at most (1 to 7): what `facts` offers the depth rule, and the cost of a draft is priced against.
    depth: u32 = 4,
};

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    model: *Model,
    w: *dw.Weights,
    mtp: cfg.Mtp,
    options: Options,
    pipes: [kernel_names.len]mtl.Pipeline,
    s: Scratch,
    /// The tiles of the 8-bit projections, when the layers are quantized.
    qmv: ?Qmv = null,
    lanes: std.AutoHashMapUnmanaged(*Stream, *Lane) = .empty,
    follow: ?mtl.ResidencySet.Follow = null,
    /// The last absorb's command buffer while it may still run (committed, not waited for).
    pending: ?mtl.CommandBuffer = null,
    /// What the load-time hold took, wall milliseconds, and a draft's price from it (the hold over `options.depth` drafts).
    hold_ms: f64 = 0,
    step_ms: f64 = 0,
    /// The GPU seconds of the last hold's and the last settled absorb's command buffers.
    hold_gpu: f64 = 0,
    absorb_gpu: f64 = 0,
    /// The GPU seconds a hold is expected to take (0: none yet), for the wait: the target's rule (forward.zig's `waitWarm`), a
    /// shorter hold believed at once and a longer one by a quarter.
    expected: f64 = 0,

    /// The draft head over `model`'s taps with `weights` (the model's `draft`): its kernels, scratch and, when the head is not
    /// yet folded, the fold; then one hold timed for `facts`.
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, model: *Model, weights: *dw.Weights, options: Options) !*Drafter {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const mtp = model.config.mtp orelse return error.NoDraftHead;
        if (!supported(&mtp, &model.config.text)) return error.UnsupportedDSpark;
        if (options.depth == 0 or options.depth > max_depth) return error.DraftDepth;
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .device = device, .model = model, .w = weights, .mtp = mtp, .options = options, .pipes = undefined, .s = undefined };
        {
            const lib = try mtl.Library.fromSource(device, ks.inkling2_dspark, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (d.pipes[0..made]) |p| p.deinit();
            for (kernel_names, 0..) |n, i| {
                d.pipes[i] = try mtl.Pipeline.init(device, lib, n, false);
                made += 1;
            }
        }
        errdefer for (d.pipes) |p| p.deinit();
        if (anyQuantized(weights)) d.qmv = try Qmv.init(device);
        errdefer if (d.qmv) |*q| q.deinit();
        try d.checkQuantized();
        try dw.fold(weights, &model.checkpoint, device, model.queue, model, d.pipe(.fold_t));
        d.s = try Scratch.init(device, model.config.text.vocab, weights);
        errdefer d.s.deinit();
        try d.warm();
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        d.sync() catch |err| std.log.err("inkling2 drafter: an absorb failed on its way out: {s}", .{@errorName(err)});
        d.setResident(null);
        var it = d.lanes.valueIterator();
        while (it.next()) |l| d.dropLane(l.*);
        d.lanes.deinit(d.gpa);
        d.s.deinit();
        if (d.qmv) |*q| q.deinit();
        for (d.pipes) |p| p.deinit();
        d.gpa.destroy(d);
    }

    fn pipe(d: *const Drafter, k: Kernel) mtl.Pipeline {
        return d.pipes[@backingInt(k)];
    }

    /// Whether the kernels' shapes are this head's and the taps are the ones the target writes.
    fn supported(m: *const cfg.Mtp, text: *const cfg.Text) bool {
        const ok = m.layers == draft_layers and m.q_heads == heads and m.kv_heads == kv_heads and m.head_dim == head_dim and m.rank == rank and
            m.block == block and m.taps_n == taps and m.window == window + 1 and m.hidden == hidden and text.hidden == hidden and
            m.vocab == text.vocab and m.inter % 256 == 0;
        if (!ok) return false;
        return std.mem.eql(u32, m.taps[0 .. taps - 1], &fwd.tap_layers) and m.taps[taps - 1] == text.layers;
    }

    /// Whether any projection of the draft layers is quantized.
    fn anyQuantized(w: *const dw.Weights) bool {
        for (w.layers) |layer| inline for (proj_fields) |f| if (@field(layer, f) == .q8) return true;
        return false;
    }

    /// Builds the tiles' variants the layers' quantized shapes ask for, at a row, a block and a window of rows: a shape they
    /// cannot take is refused at the load, not in a round.
    fn checkQuantized(d: *Drafter) !void {
        const q: *Qmv = if (d.qmv) |*x| x else return;
        for (d.w.layers) |layer| inline for (proj_fields) |f| switch (@field(layer, f)) {
            .q8 => |x| for ([_]usize{ 1, block, absorb_rows }) |rows| try q.check(x, rows),
            .bf16 => {},
        };
    }

    /// Joins (or, with null, leaves) a residency set: the scratch and every lane's buffers, as lanes open and close.
    pub fn setResident(d: *Drafter, hook: ?mtl.ResidencySet.Follow) void {
        if (d.follow) |f| {
            mtl.resident.fields.apply(&d.s, f, false);
            var it = d.lanes.valueIterator();
            while (it.next()) |l| mtl.resident.fields.apply(l.*, f, false);
        }
        d.follow = hook;
        if (hook) |f| {
            mtl.resident.fields.apply(&d.s, f, true);
            var it = d.lanes.valueIterator();
            while (it.next()) |l| mtl.resident.fields.apply(l.*, f, true);
        }
    }

    /// Device bytes one stream's draft state takes: ten rings, the Markov feature, the state and the token row.
    pub fn laneBytes(d: *const Drafter) usize {
        _ = d;
        return 2 * draft_layers * ringBytes() + hidden * 2 + rank * 2 + block * 4;
    }

    /// Device bytes the head holds that no checkpoint shard counts: the scratch and `streams` streams' draft state.
    pub fn extraBytes(d: *const Drafter, streams: usize) usize {
        return d.scratchBytes() + streams * d.laneBytes();
    }

    /// Device bytes the whole head holds with `streams` streams: its file's shard and the fold, both shards of the checkpoint,
    /// and `extraBytes`.
    pub fn residentBytes(d: *const Drafter, streams: usize) usize {
        return d.w.bytes + d.w.fold_w.?.t.bytes + d.w.fold_b.?.bytes + d.extraBytes(streams);
    }

    /// Device bytes of the scratch (with the step logits' buffer, if it is made).
    pub fn scratchBytes(d: *const Drafter) usize {
        var total: usize = 0;
        inline for (@typeInfo(Scratch).@"struct".field_names) |name| {
            const v = @field(d.s, name);
            if (@TypeOf(v) == mtl.Buffer) total += v.length() else if (v) |buf| total += buf.length();
        }
        return total;
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .kind = "DSpark draft head", .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(ptr: *anyopaque) []const u32 {
        const d = cast(ptr);
        return d.mtp.taps[0..taps];
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        return .{ .depth = d.options.depth, .step_ms = d.step_ms, .fixed_block = true, .plain_guard = true, .stream_bytes = d.laneBytes(), .prior = &prior };
    }

    fn ringBytes() usize {
        return @as(usize, ring + ring_pad) * kv_width * 2;
    }

    fn newLane(d: *Drafter) !*Lane {
        const l = try d.gpa.create(Lane);
        errdefer d.gpa.destroy(l);
        var bufs: [2 * draft_layers + 3]mtl.Buffer = undefined;
        var made: usize = 0;
        errdefer for (bufs[0..made]) |x| x.deinit();
        const small = [3]usize{ hidden * 2, rank * 2, block * 4 };
        for (&bufs, 0..) |*x, i| {
            x.* = if (i < 2 * draft_layers) try d.device.buffer(ringBytes(), mtl.ResourceOptions.private) else try d.device.buffer(small[i - 2 * draft_layers], mtl.ResourceOptions.shared);
            made += 1;
        }
        l.* = .{ .k = undefined, .v = undefined, .f_rnn = bufs[2 * draft_layers], .state = bufs[2 * draft_layers + 1], .seq = bufs[2 * draft_layers + 2] };
        for (0..draft_layers) |i| {
            l.k[i] = bufs[2 * i];
            l.v[i] = bufs[2 * i + 1];
        }
        for ([_]mtl.Buffer{ l.f_rnn, l.state, l.seq }) |x| @memset(x.slice(u8, x.length()), 0);
        if (d.follow) |f| mtl.resident.fields.apply(l, f, true);
        return l;
    }

    fn dropLane(d: *Drafter, l: *Lane) void {
        if (d.follow) |f| mtl.resident.fields.apply(l, f, false);
        for (l.k) |x| x.deinit();
        for (l.v) |x| x.deinit();
        l.f_rnn.deinit();
        l.state.deinit();
        l.seq.deinit();
        d.gpa.destroy(l);
    }

    fn openFn(ptr: *anyopaque, s: *Stream) anyerror!void {
        const d = cast(ptr);
        if (d.lanes.contains(s)) return error.DraftStreamOpen;
        const l = try d.newLane();
        errdefer d.dropLane(l);
        try d.lanes.put(d.gpa, s, l);
    }

    fn releaseFn(ptr: *anyopaque, s: *Stream) void {
        const d = cast(ptr);
        d.sync() catch |err| std.log.err("inkling2 drafter: an absorb failed: {s}", .{@errorName(err)});
        if (d.lanes.fetchRemove(s)) |e| d.dropLane(e.value);
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(ptr);
        for (streams, out) |s, o| {
            const l = d.lanes.get(s) orelse return error.DraftStreamUnknown;
            if (o.len > l.held_len) return error.DraftHeldFewer;
            @memcpy(o, l.held[0..o.len]);
        }
    }

    // ---- encoding ----

    fn b(buf: mtl.Buffer) B {
        return .{ .buf = buf };
    }

    fn bo(buf: mtl.Buffer, off: usize) B {
        return .{ .buf = buf, .off = off };
    }

    fn t(x: dw.Tensor) B {
        return .{ .buf = x.buffer, .off = x.offset };
    }

    /// A kernel of the drafter's library: `args` (unless void) at buffer 0, then `bufs`.
    fn launch(d: *const Drafter, e: mtl.ComputeEncoder, k: Kernel, args: anytype, bufs: []const B, groups: [3]usize, threads: [3]usize) void {
        e.setPipeline(d.pipe(k));
        var at: usize = 0;
        if (@TypeOf(args) != void) {
            e.setValue(args, 0);
            at = 1;
        }
        for (bufs, at..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.dispatchGroups(mtl.Size.of(groups[0], groups[1], groups[2]), mtl.Size.of(threads[0], threads[1], threads[2]));
    }

    /// A kernel of the target's glue (forward.zig's Args at buffer 0, then `bufs`).
    fn glue(d: *const Drafter, e: mtl.ComputeEncoder, g: Glue, a: fwd.Args, bufs: []const B, grid: [3]usize, group: [3]usize, by_groups: bool) void {
        e.setPipeline(d.model.pipe(g));
        e.setValue(a, 0);
        for (bufs, 1..) |r, i| e.setBuffer(r.buf, r.off, i);
        const g3 = mtl.Size.of(grid[0], grid[1], grid[2]);
        const t3 = mtl.Size.of(group[0], group[1], group[2]);
        if (by_groups) e.dispatchGroups(g3, t3) else e.dispatchThreads(g3, t3);
    }

    /// nn.RMSNorm of `rows` bf16 rows of the hidden width: `in` to `out`.
    fn norm(d: *const Drafter, e: mtl.ComputeEncoder, w: dw.Tensor, in: B, out: B, rows: usize) void {
        d.glue(e, .norm, .{ .rows = @intCast(rows), .width = hidden, .stride = hidden, .eps = d.mtp.eps }, &.{ in, t(w), out }, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, true);
    }

    /// `rows` rows of `in` through the projection `p` into `out`: the target's bf16 matvec kernels, or core's tiles for a
    /// quantized one.
    fn mv(d: *Drafter, e: mtl.ComputeEncoder, p: dw.Proj, in: B, out: B, rows: usize) !void {
        switch (p) {
            .bf16 => |x| fwd.matvec(e, d.model, x, in, out, rows),
            .q8 => |x| try d.qmv.?.encode(e, x, in, out, rows),
        }
    }

    /// log2 of the rotary base, as mlx's rope takes it (the angle is the position times 2^(-(i / 64) * this)).
    fn rope(d: *const Drafter) f32 {
        return @floatCast(std.math.log2(@as(f64, d.mtp.rope_theta)));
    }

    /// The context rows [p0, p0 + n) from `src` (rows of eight taps) into every layer's ring; `newest`: the last of them is the
    /// stream's newest committed row, whose Markov feature the hold reads.
    fn encodeAbsorb(d: *Drafter, e: mtl.ComputeEncoder, l: *const Lane, src: B, p0: u64, n: usize, newest: bool) !void {
        const s = &d.s;
        const ls = d.w.layers;
        const rows: u32 = @intCast(n);
        d.launch(e, .mix, MixArgs{ .rows = rows, .width = hidden, .tap_stride = taps * hidden, .mix_stride = rows * hidden, .eps = d.mtp.eps }, &.{ src, b(d.w.alpha_buffer), t(ls[0].dflare), t(ls[1].dflare), t(ls[2].dflare), t(ls[3].dflare), t(ls[4].dflare), t(d.w.markov.dflare), b(s.mixed) }, .{ n, mixes, 1 }, .{ 288, 1, 1 });
        for (ls, 0..) |*layer, i| {
            const mixed = bo(s.mixed, i * n * hidden * 2);
            try d.mv(e, layer.k_ctx, mixed, b(s.kc), n);
            try d.mv(e, layer.v_ctx, mixed, b(s.vc), n);
            const a = PrepArgs{ .rows = rows, .heads = heads, .kv = kv_heads, .pos = @intCast(p0), .ring = ring, .eps = d.mtp.eps, .base_log2 = d.rope() };
            d.launch(e, .ctx_prep, a, &.{ b(s.kc), b(s.vc), t(layer.k_norm_ctx), b(l.k[i]), b(l.v[i]) }, .{ n, 1, 1 }, .{ 256, 1, 1 });
        }
        if (newest) d.glue(e, .copy, .{ .rows = 1, .width = hidden }, &.{ bo(s.mixed, (5 * n + n - 1) * hidden * 2), b(l.f_rnn) }, .{ hidden, 1, 1 }, .{ 256, 1, 1 }, false);
    }

    /// The block for `it`: the pending token at the lane's end and seven mask rows through the five layers, the unembedding of
    /// rows 1 to 7, and `it.depth` Markov steps; the proposals land in the lane's token row after the pending token.
    fn encodeHold(d: *Drafter, e: mtl.ComputeEncoder, l: *const Lane, it: dr.Hold) !void {
        const m = d.model;
        const s = &d.s;
        const text = &m.config.text;
        const eps = d.mtp.eps;
        const end: u32 = @intCast(l.end);
        const win: u32 = @intCast(@min(@as(u64, window), l.end - l.from));
        const blocks = attnBlocks(end, win);
        const inter: usize = d.mtp.inter;
        d.glue(e, .embed, .{ .rows = 1, .width = hidden, .eps = text.eps }, &.{ b(l.seq), t(m.weights.embed.t), t(m.weights.embed_norm), b(s.h0) }, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, true);
        var prev = b(s.h0);
        for (d.w.layers, 0..) |*layer, i| {
            d.norm(e, layer.input, prev, b(s.x), block);
            try d.mv(e, layer.q, b(s.x), b(s.q), block);
            try d.mv(e, layer.k, b(s.x), b(s.k), block);
            try d.mv(e, layer.v, b(s.x), b(s.v), block);
            const pa = PrepArgs{ .rows = block, .heads = heads, .kv = kv_heads, .pos = end, .ring = ring, .eps = eps, .base_log2 = d.rope() };
            d.launch(e, .block_prep, pa, &.{ b(s.q), b(s.k), b(s.v), t(layer.q_norm), t(layer.k_norm), b(s.qn), b(s.kb), b(s.vb) }, .{ (heads + kv_heads + 7) / 8, block, 1 }, .{ 256, 1, 1 });
            const aa = AttnArgs{ .heads = heads, .kv = kv_heads, .end = end, .window = win, .ring = ring, .blocks = blocks, .scale = @floatCast(1.0 / @sqrt(@as(f64, head_dim))) };
            d.launch(e, .attn_part, aa, &.{ b(s.qn), b(l.k[i]), b(l.v[i]), b(s.kb), b(s.vb), b(s.part) }, .{ kv_heads, 1, blocks }, .{ 192, 1, 1 });
            d.launch(e, .attn_merge, aa, &.{ b(s.part), b(s.att) }, .{ heads, block, 1 }, .{ 32, 1, 1 });
            try d.mv(e, layer.o, b(s.att), b(s.y), block);
            const row_add = EwArgs{ .rows = block, .width = hidden };
            d.launch(e, .add, row_add, &.{ prev, b(s.y), b(s.h) }, .{ hidden / 256, block, 1 }, .{ 256, 1, 1 });
            d.norm(e, layer.post, b(s.h), b(s.x), block);
            try d.mv(e, layer.gate, b(s.x), b(s.gate), block);
            try d.mv(e, layer.up, b(s.x), b(s.up), block);
            d.launch(e, .swiglu, EwArgs{ .rows = block, .width = @intCast(inter) }, &.{ b(s.gate), b(s.up), b(s.act) }, .{ (inter + 255) / 256, block, 1 }, .{ 256, 1, 1 });
            try d.mv(e, layer.down, b(s.act), b(s.y), block);
            d.launch(e, .add, row_add, &.{ b(s.h), b(s.y), b(s.h) }, .{ hidden / 256, block, 1 }, .{ 256, 1, 1 });
            prev = b(s.h);
        }
        // rows 1 to 7 only: the first row's last output is not read
        d.norm(e, d.w.norm, bo(s.h, hidden * 2), b(s.hn), max_depth);
        fwd.matvec(e, m, m.weights.unembed, b(s.hn), b(s.base), max_depth);
        const vocab: usize = text.vocab;
        const groups = (vocab + pick_group - 1) / pick_group;
        const fold_w = d.w.fold_w.?.t;
        const fold_b = d.w.fold_b.?;
        const flags: u32 = if (s.capture != null) write_logits else 0;
        for (0..it.depth) |k| {
            const step: u32 = @intCast(k);
            d.launch(e, .rnn_proj, ProjArgs{ .n = 3 * rank, .at = step }, &.{ b(l.state), t(d.w.markov.w1.t), b(l.seq), bo(s.hn, k * hidden * 2), b(l.f_rnn), t(d.w.markov.rnn.t), t(d.w.markov.rnn_b), b(s.proj) }, .{ 3 * rank / 8, 1, 1 }, .{ 256, 1, 1 });
            d.launch(e, .gru, {}, &.{ b(s.proj), b(l.state), b(l.state), b(s.latent) }, .{ 1, 1, 1 }, .{ rank, 1, 1 });
            const out = if (s.capture) |c| bo(c, k * vocab * 2) else b(s.latent);
            d.launch(e, .fold_pick, FoldArgs{ .vocab = @intCast(vocab), .valid = text.unpadded_vocab, .flags = flags, .cap = text.logit_cap }, &.{ bo(s.base, k * vocab * 2), b(s.latent), t(fold_w), t(fold_b), b(s.best), b(s.best_id), out }, .{ groups, 1, 1 }, .{ 256, 1, 1 });
            d.launch(e, .pick, PickArgs{ .groups = @intCast(groups), .step = step }, &.{ b(s.best), b(s.best_id), b(l.seq) }, .{ 1, 1, 1 }, .{ 1024, 1, 1 });
        }
    }

    // ---- absorb and hold ----

    /// Waits for the last absorb's command buffer and reports its failure.
    pub fn sync(d: *Drafter) !void {
        const cb = d.pending orelse return;
        d.pending = null;
        defer cb.release();
        cb.wait();
        if (cb.failure()) |msg| return failed(msg);
        d.absorb_gpu = cb.gpuSeconds();
    }

    /// Rows [start, start + rows) replace the context from `start`; the rows of a prompt before its last window are skipped. The
    /// work is committed and not waited for: the hold that follows runs behind it on the queue, and the target's next forward
    /// writes its taps only after it.
    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        for (items) |it| {
            _ = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
            const f = it.features;
            if (f.space != .device or f.dtype != .bf16 or f.row_bytes != tap_row_bytes or f.rows == 0 or std.meta.activeTag(f.ready) != .none or it.rows != null) return error.DraftFeaturesUnsupported;
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try d.sync();
        var cb: ?mtl.CommandBuffer = null;
        var enc: mtl.ComputeEncoder = undefined;
        for (items) |it| {
            const l = d.lanes.get(it.stream).?;
            const f = it.features;
            if (it.start > l.end) l.from = it.start; // the rows before it were restored, not absorbed
            const src = mtl.Buffer{ .id = @ptrFromInt(f.buffer) };
            const end = it.start + f.rows;
            // only the last window of a long prompt is ever attended
            const keep_from = if (it.prompt) @max(it.start, @as(u64, it.stream.prompt_len -| window)) else it.start;
            var at: u64 = keep_from;
            while (at < end) {
                if (cb == null) {
                    cb = d.model.queue.commandBuffer();
                    enc = cb.?.compute(.serial);
                }
                const n: usize = @intCast(@min(@as(u64, absorb_rows), end - at));
                d.encodeAbsorb(enc, l, bo(src, @intCast(f.offset + (at - it.start) * tap_row_bytes)), at, n, at + n == end) catch |err| {
                    enc.end(); // an open encoder may not be dropped
                    return err;
                };
                at += n;
            }
            l.end = end;
            l.held_len = 0;
        }
        const one = cb orelse return;
        enc.end();
        one.commit();
        d.pending = one.retain();
    }

    /// Each stream's `depth` drafts after `pending` at `position`: a whole block each, all in one command buffer, behind the
    /// absorbs before it. A hold needs the context to end just before its position, as the absorbs left it.
    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        const vocab = d.model.config.text.vocab;
        for (items) |it| {
            const l = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
            if (it.depth == 0 or it.depth > max_depth or it.position == 0) return error.DraftDepth;
            if (l.end != it.position - 1) return error.DraftContextGap;
            if (it.pending >= vocab) return error.InvalidDraftToken;
        }
        if (items.len == 0) return;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const began = now();
        const cb = d.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (items) |it| {
            const l = d.lanes.get(it.stream).?;
            l.seq.slice(u32, block)[0] = it.pending;
            @memset(l.state.slice(u8, rank * 2), 0);
            d.encodeHold(e, l, it) catch |err| {
                e.end();
                return err;
            };
        }
        e.end();
        const committed = fwd.seconds();
        cb.commit();
        if (d.model.warm_wait) fwd.waitWarm(cb, d.expected, committed) else cb.wait();
        d.sync() catch |err| {
            std.log.err("inkling2 drafter: the absorb before a hold failed: {s}", .{@errorName(err)});
            return err;
        };
        if (cb.failure()) |msg| return failed(msg);
        d.hold_gpu = cb.gpuSeconds();
        d.expected = if (d.expected == 0 or d.hold_gpu < d.expected) d.hold_gpu else d.expected + (d.hold_gpu - d.expected) / 4;
        d.hold_ms = (now() - began) / @as(f64, @floatFromInt(items.len));
        for (items) |it| {
            const l = d.lanes.get(it.stream).?;
            const seq = l.seq.slice(u32, block);
            for (seq[1..][0..it.depth], l.held[0..it.depth]) |x, *y| {
                if (x >= vocab) return error.DraftNonfinite;
                y.* = x;
            }
            l.held_len = it.depth;
        }
    }

    // ---- replay support ----

    /// From the next hold on, each step's logits (the sum of base and fold bias, softcapped, the padded vocabulary at -inf) are
    /// kept as bf16 [7][vocab] for `captured`; the last hold's of a batch only.
    pub fn captureLogits(d: *Drafter) !void {
        if (d.s.capture != null) return;
        const buf = try d.device.buffer(max_depth * @as(usize, d.model.config.text.vocab) * 2, mtl.ResourceOptions.shared);
        d.s.capture = buf;
        if (d.follow) |f| f.join(f.ctx, buf);
    }

    /// The last hold's steps' logits, bf16 [7][vocab] (steps past its depth hold the hold before's).
    pub fn captured(d: *const Drafter) []const u16 {
        return d.s.capture.?.slice(u16, max_depth * @as(usize, d.model.config.text.vocab));
    }

    // ---- load-time timing ----

    /// The pipelines' first launch and a context of a full window, then a hold timed at the facts' depth: the median of three.
    fn warm(d: *Drafter) !void {
        const l = try d.newLane();
        defer d.dropLane(l);
        const rows = window + 77;
        const feat = try d.device.buffer(rows * tap_row_bytes, mtl.ResourceOptions.shared);
        defer feat.deinit();
        var prng = std.Random.DefaultPrng.init(0x5eed);
        const row = feat.slice(u16, tap_row_bytes / 2);
        for (row) |*x| x.* = mix.bf16Of(prng.random().float(f32) * 2 - 1);
        const all = feat.slice(u16, rows * tap_row_bytes / 2);
        for (1..rows) |r| @memcpy(all[r * row.len ..][0..row.len], row);
        const ids = [_]u32{ 1000, 2000, 3000, 4000 };
        var stream: Stream = try Stream.init(d.gpa, .{ .id = "warm", .prompt = &ids, .max_new = 1 });
        defer stream.deinit(d.gpa);
        try d.lanes.put(d.gpa, &stream, l);
        defer _ = d.lanes.remove(&stream);
        const feature = lanes.backend.Features{ .buffer = @intFromPtr(feat.id), .rows = rows, .row_bytes = tap_row_bytes, .space = .device, .dtype = .bf16 };
        try absorbFn(d, &.{.{ .stream = &stream, .features = feature, .start = 0, .follow = &.{}, .prompt = false }});
        var ms: [4]f64 = undefined;
        for (&ms) |*x| {
            try holdFn(d, &.{.{ .stream = &stream, .pending = ids[0], .position = rows + 1, .depth = d.options.depth }});
            x.* = d.hold_ms;
        }
        std.mem.sort(f64, ms[1..], {}, std.sort.asc(f64)); // the first hold takes the weights' first touch
        d.hold_ms = ms[2];
        d.step_ms = d.hold_ms / @as(f64, @floatFromInt(d.options.depth));
    }

    fn failed(msg: [*:0]const u8) anyerror {
        std.log.err("inkling2 drafter: command buffer failed: {s}", .{msg});
        return error.GpuFailed;
    }
};

fn now() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1e3 + @as(f64, @floatFromInt(ts.nsec)) / 1e6;
}

test "inkling2 dspark: the attention's blocks are the 256-key blocks that meet the window, then the block's own keys" {
    const cases = [_][2]u32{ .{ 0, 1023 }, .{ 1, 1023 }, .{ 255, 1023 }, .{ 256, 1023 }, .{ 1000, 1023 }, .{ 1023, 1023 }, .{ 1024, 1023 }, .{ 1100, 1023 }, .{ 1279, 1023 }, .{ 1280, 1023 }, .{ 5000, 1023 }, .{ 5001, 1023 }, .{ 40, 0 }, .{ 700, 37 }, .{ 700, 1 }, .{ 2048, 1023 } };
    for (cases) |c| {
        const end, const win = c;
        const lo = if (end > win) end - win else 0;
        var meet: u32 = 0;
        var b: u32 = 0;
        // block b holds positions [256 b, 256 b + 256): it counts when one of them is in [lo, end)
        while (lo < end and b * key_block < end) : (b += 1) {
            if ((b + 1) * key_block > lo) meet += 1;
        }
        try std.testing.expectEqual(meet + 1, attnBlocks(end, win));
        try std.testing.expect(attnBlocks(end, win) <= max_blocks);
    }
}

test "inkling2 dspark: the depth setting is a whole number from 1 to 7 that fixes every round's drafts, auto, or 4 when unset" {
    const t = std.testing;
    for ([_]?[]const u8{ null, "", "  \t" }) |text| try t.expectEqual(Depth{ .fixed = 4 }, try Depth.parse(text));
    // the drafter's own default is the setting's, and what the load asks of the depth rule when nothing is set
    try t.expectEqual(@as(u32, 4), (Options{}).depth);
    try t.expectEqual(Depth.default.most(), (Options{}).depth);
    for (1..max_depth + 1) |n| {
        var buf: [4]u8 = undefined;
        const want: u8 = @intCast(n);
        const d = try Depth.parse(try std.fmt.bufPrint(&buf, " {d}", .{n}));
        try t.expectEqual(Depth{ .fixed = want }, d);
        try t.expectEqual(@as(u32, want), d.most());
        try t.expectEqual(@as(?u8, want), d.fixedDepth());
    }
    for ([_][]const u8{ "auto", "AUTO", " Auto " }) |text| {
        const d = try Depth.parse(text);
        try t.expectEqual(Depth.auto, d);
        try t.expectEqual(@as(u32, max_depth), d.most());
        try t.expectEqual(@as(?u8, null), d.fixedDepth());
    }
    for ([_][]const u8{ "0", "8", "-1", "3.5", "five", "autos", "256", "1 2" }) |text| try t.expectError(error.BadDraftDepth, Depth.parse(text));
}
