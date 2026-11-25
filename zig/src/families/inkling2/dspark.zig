//! Inkling 2's DSpark draft head as a lane drafter over the target's taps. Each committed row's taps are mixed, normed,
//! and projected into the draft layers' context keys and values, kept in a ring of the newest 1,023 positions a layer.
//! A hold runs one block of eight rows (the pending token's embedding, then seven mask rows) through those layers, takes
//! the target's unembedding of rows 1 to 7 as base logits, and names proposals with the Markov GRU: each step folds
//! the previous token, a state and two features into a latent whose product with the folded unembedding
//! (dspark_weights.fold) joins the base logits before the softcap, the padded vocabulary's mask and the argmax.
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
const dense_prefill = @import("dense_prefill.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const snapshot = @import("snapshot.zig");
const Model = @import("model.zig").Model;
const Glue = @import("model.zig").Glue;

const dr = lanes.drafter;
const Stream = lanes.Stream;
const B = fwd.B;

// Fixed geometry shared by the kernels; layer count, width, q heads and tap count come from config.Mtp.
const kv_heads = 8;
const head_dim = 128;
const kv_width = kv_heads * head_dim;
const rank = 256;
const max_draft_layers = 8;
const default_draft_layers = 5;
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
/// Rows one absorb pass projects at most: the scratch holds this many rows' mixes, keys and values. A prompt chunk of any size
/// up to the target's prefill rows (8,192 at most) is absorbed in passes of this many, in order in one command buffer.
const absorb_rows = 512;
/// Ids of the vocabulary that groups of 256 ids' argmax partials cover.
const pick_group = 256;

/// The prior the depth rule starts from: how likely draft j is kept when drafts 1 to j - 1 were, over recorded greedy runs of
/// the reference head.
const prior = [max_depth]f64{ 0.857, 0.859, 0.861, 0.875, 0.901, 0.902, 0.905 };

/// The fields of a draft layer that are projections (dw.Proj).
const proj_fields = [_][]const u8{ "q", "k", "v", "k_ctx", "v_ctx", "o", "gate", "up", "down" };

const kernel_names = [_][:0]const u8{ "ikd_mix", "ikd_ctx_prep", "ikd_block_prep", "ikd_attn_part", "ikd_attn_merge", "ikd_swiglu", "ikd_add", "ikd_rnn_proj", "ikd_gru", "ikd_fold_pick", "ikd_fold_pick_batch", "ikd_pick", "ikd_fold_t" };
const Kernel = enum { mix, ctx_prep, block_prep, attn_part, attn_merge, swiglu, add, rnn_proj, gru, fold_pick, fold_pick_batch, pick, fold_t };

comptime {
    const members = @typeInfo(Kernel).@"enum".field_names;
    std.debug.assert(members.len == kernel_names.len);
    for (members, kernel_names) |member, name| std.debug.assert(std.mem.eql(u8, "ikd_" ++ member, name));
}

/// Maximum streams handled in one batched hold pass (matches IKD_MAX_BATCH in dspark.metal).
const max_batch = 8;

// The kernels' argument structs, field for field.
const MixArgs = extern struct { rows: u32, width: u32, tap_stride: u32, mix_stride: u32, eps: f32 };
const PrepArgs = extern struct { rows: u32, heads: u32, kv: u32, pos: u32, ring: u32, eps: f32, base_log2: f32 };
const AttnArgs = extern struct { heads: u32, kv: u32, end: u32, window: u32, ring: u32, blocks: u32, scale: f32 };
const EwArgs = extern struct { rows: u32, width: u32 };
const ProjArgs = extern struct { n: u32, at: u32, hidden: u32 };
const FoldArgs = extern struct { vocab: u32, valid: u32, flags: u32, cap: f32 };
const FoldBatchArgs = extern struct { vocab: u32, valid: u32, flags: u32, count: u32, base_stride: u32, cap: f32, slot_map: [max_batch]u32 };
const PickArgs = extern struct { groups: u32, step: u32, flags: u32 };

comptime {
    std.debug.assert(max_batch == 8 and @sizeOf(FoldBatchArgs) == 56);
}
/// ikd_fold_pick's flags: also write every id's logit, and (with ikd_pick's) the step's probability of the pick.
const write_logits: u32 = 1;
const write_mass: u32 = 2;

/// Blocks a (row, head) of the attention has when the block starts at position `end` and attends the newest `win` positions
/// before it: the 256-key blocks, anchored at absolute positions, that meet [end - win, end), then the block's own keys; just
/// those when nothing precedes it.
fn attnBlocks(end: u32, win: u32) u32 {
    const lo = if (end > win) end - win else 0;
    if (end == lo) return 1;
    return (end - 1) / key_block - lo / key_block + 2;
}

/// Metal source for dspark.metal specialized with `IKD_TAPS = taps_n`.
pub fn sourceFor(gpa: std.mem.Allocator, taps_n: u32) ![]u8 {
    return std.fmt.allocPrint(gpa, "#define IKD_TAPS {d}\n{s}", .{ taps_n, ks.inkling2_dspark });
}

/// The drafter's scratch: absorb's mixed rows and context keys and values, the block's activations, base logits and the
/// Markov step's working buffers. Every field is a buffer, so a residency hook joins them all.
const Scratch = struct {
    mixed: mtl.Buffer, // [layers + 1][rows][hidden]: the tap mixes of an absorb's rows
    kc: mtl.Buffer, // [rows][kv_width]: a layer's context keys and values
    vc: mtl.Buffer,
    h0: mtl.Buffer, // [8][hidden]: the block's input rows; rows 1 to 7 are mask_embed from the start
    h: mtl.Buffer, // [8][hidden]: the block's residual stream
    x: mtl.Buffer, // [8][hidden]: normed rows
    y: mtl.Buffer, // [8][hidden]: a branch's output
    q: mtl.Buffer, // [8][q_heads * head_dim]
    qn: mtl.Buffer, // [8][q_heads * head_dim]: the rotated queries
    att: mtl.Buffer, // [8][q_heads * head_dim]: the attention's output
    k: mtl.Buffer, // [8][kv_width]: the block's keys and values, as projected and as normed and rotated
    v: mtl.Buffer,
    kb: mtl.Buffer,
    vb: mtl.Buffer,
    gate: mtl.Buffer, // [8][inter]
    up: mtl.Buffer,
    act: mtl.Buffer,
    part: mtl.Buffer, // f32 [8][q_heads][max_blocks][130]: the attention's partials
    hn: mtl.Buffer, // [8][hidden]: the final norm of rows 1 to 7
    base: mtl.Buffer, // [7][vocab]: the unembedding of those rows
    proj: mtl.Buffer, // [3 * rank]: the Markov projection
    latent: mtl.Buffer, // [rank]
    best: mtl.Buffer, // f32 [groups]: each group of ids' best logit and its id
    best_id: mtl.Buffer, // u32 [groups]
    mass: mtl.Buffer, // f32 [groups]: each group's sum of exp(logit - its best), for the probability of the pick
    /// The steps' logits, when asked for: bf16 [7][vocab].
    capture: ?mtl.Buffer = null,

    /// The buffers made up front, in `bytes`' order.
    const made = [_][]const u8{ "mixed", "kc", "vc", "h0", "h", "x", "y", "q", "qn", "att", "k", "v", "kb", "vb", "gate", "up", "act", "part", "hn", "base", "proj", "latent", "best", "best_id", "mass" };

    /// Bytes of each buffer for the draft head's shape `m`.
    fn bytes(m: cfg.Mtp) [made.len]usize {
        const rows = block;
        const h: usize = m.hidden;
        const q_heads: usize = m.q_heads;
        const mixes: usize = @as(usize, m.layers) + 1;
        const inter: usize = m.inter;
        const vocab: usize = m.vocab;
        const groups = (vocab + pick_group - 1) / pick_group;
        return .{
            mixes * absorb_rows * h * 2,
            absorb_rows * kv_width * 2,
            absorb_rows * kv_width * 2,
            rows * h * 2,
            rows * h * 2,
            rows * h * 2,
            rows * h * 2,
            rows * q_heads * head_dim * 2,
            rows * q_heads * head_dim * 2,
            rows * q_heads * head_dim * 2,
            rows * kv_width * 2,
            rows * kv_width * 2,
            rows * kv_width * 2,
            rows * kv_width * 2,
            rows * inter * 2,
            rows * inter * 2,
            rows * inter * 2,
            rows * q_heads * max_blocks * part_slot * 4,
            rows * h * 2,
            max_depth * vocab * 2,
            3 * rank * 2,
            rank * 2,
            groups * 4,
            groups * 4,
            groups * 4,
        };
    }

    /// Every buffer made, and the block's input rows ready: row 0 is the pending token's embedding, each hold's to write, and
    /// rows 1 to 7 are the mask embedding.
    fn init(device: mtl.Device, m: cfg.Mtp, w: *const dw.Weights) !Scratch {
        const sz = bytes(m);
        const h: usize = m.hidden;
        var s: Scratch = undefined;
        s.capture = null;
        var done: [made.len]bool = @splat(false);
        errdefer inline for (made, 0..) |f, i| if (done[i]) @field(s, f).deinit();
        inline for (made, 0..) |f, i| {
            @field(s, f) = try device.buffer(@max(sz[i], 16), mtl.ResourceOptions.shared);
            done[i] = true;
        }
        const rows = s.h0.slice(u16, block * h);
        for (1..block) |r| @memcpy(rows[r * h ..][0..h], w.mask_embed.host(u16));
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

/// Lazily allocated scratch for batched multi-stream holds (up to `max_batch` streams). Allocated on the first
/// multi-stream hold without logit capture so single-stream workloads pay zero extra memory. Every field is a buffer,
/// so a residency hook joins them all.
const BatchScratch = struct {
    h0: mtl.Buffer, // [max_batch][8][hidden]: rows 1..7 initialized to mask_embed per stream slot
    h: mtl.Buffer, // [max_batch][8][hidden]
    x: mtl.Buffer, // [max_batch][8][hidden]
    y: mtl.Buffer, // [max_batch][8][hidden]
    q: mtl.Buffer, // [max_batch][8][q_heads * head_dim]
    qn: mtl.Buffer, // [max_batch][8][q_heads * head_dim]
    att: mtl.Buffer, // [max_batch][8][q_heads * head_dim]
    k: mtl.Buffer, // [max_batch][8][kv_width]
    v: mtl.Buffer,
    kb: mtl.Buffer,
    vb: mtl.Buffer,
    gate: mtl.Buffer, // [max_batch][8][inter]
    up: mtl.Buffer,
    act: mtl.Buffer,
    hn: mtl.Buffer, // [max_batch][max_depth][hidden]: final norm of rows 1..7 per stream
    base: mtl.Buffer, // [max_batch][max_depth][vocab]: unembedding output
    latent: mtl.Buffer, // [max_batch][rank]
    best: mtl.Buffer, // f32 [max_batch][groups]
    best_id: mtl.Buffer, // u32 [max_batch][groups]
    mass: mtl.Buffer, // f32 [max_batch][groups]

    const made = [_][]const u8{ "h0", "h", "x", "y", "q", "qn", "att", "k", "v", "kb", "vb", "gate", "up", "act", "hn", "base", "latent", "best", "best_id", "mass" };

    fn bytes(m: cfg.Mtp) [made.len]usize {
        const rows = max_batch * block;
        const h: usize = m.hidden;
        const q_heads: usize = m.q_heads;
        const inter: usize = m.inter;
        const vocab: usize = m.vocab;
        const groups = (vocab + pick_group - 1) / pick_group;
        return .{
            rows * h * 2,
            rows * h * 2,
            rows * h * 2,
            rows * h * 2,
            rows * q_heads * head_dim * 2,
            rows * q_heads * head_dim * 2,
            rows * q_heads * head_dim * 2,
            rows * kv_width * 2,
            rows * kv_width * 2,
            rows * kv_width * 2,
            rows * kv_width * 2,
            rows * inter * 2,
            rows * inter * 2,
            rows * inter * 2,
            max_batch * max_depth * h * 2,
            max_batch * max_depth * vocab * 2,
            max_batch * rank * 2,
            max_batch * groups * 4,
            max_batch * groups * 4,
            max_batch * groups * 4,
        };
    }

    fn init(device: mtl.Device, m: cfg.Mtp, w: *const dw.Weights) !BatchScratch {
        const sz = bytes(m);
        const h: usize = m.hidden;
        var bs: BatchScratch = undefined;
        var done: [made.len]bool = @splat(false);
        errdefer inline for (made, 0..) |f, i| if (done[i]) @field(bs, f).deinit();
        inline for (made, 0..) |f, i| {
            @field(bs, f) = try device.buffer(@max(sz[i], 16), mtl.ResourceOptions.shared);
            done[i] = true;
        }
        const rows = bs.h0.slice(u16, max_batch * block * h);
        const mask = w.mask_embed.host(u16);
        for (0..max_batch) |si| {
            for (1..block) |r| @memcpy(rows[(si * block + r) * h ..][0..h], mask);
        }
        return bs;
    }

    fn deinit(bs: *BatchScratch) void {
        inline for (made) |f| @field(bs.*, f).deinit();
    }
};

comptime {
    std.debug.assert(@typeInfo(BatchScratch).@"struct".field_names.len == BatchScratch.made.len);
}

/// One stream's draft state: each layer's context ring (slots 0..mtp.layers), the Markov feature of its newest absorbed
/// row, the GRU's state, the hold's token row (the pending token, then its proposals) and each proposal's probability.
const Lane = struct {
    k: [max_draft_layers]mtl.Buffer,
    v: [max_draft_layers]mtl.Buffer,
    f_rnn: mtl.Buffer,
    state: mtl.Buffer,
    seq: mtl.Buffer,
    prob: mtl.Buffer, // f32 [8]: step k's at k, written when the drafter keeps probabilities
    end: u64 = 0, // context rows absorbed: positions [from, end)
    from: u64 = 0, // the first position held (a resumed prompt state's drafter context is not kept)
    held: [max_depth]u32 = @splat(0),
    held_len: u32 = 0,
};

/// How deep a served round drafts (INKLING_DRAFT_DEPTH): a whole number from 1 to `max_depth` holds every round at that depth,
/// `auto` leaves it to the lane core's cost rule, which chooses among 1 to `max_depth` by what the drafts and the window
/// forwards cost. Unset it is `default` with the confidence floor off, and `max_depth` beside a floor (the served default: see
/// `Drafting`).
pub const Depth = union(enum) {
    fixed: u8,
    auto,

    /// What an unset or blank setting means with no confidence floor.
    pub const default: Depth = .{ .fixed = 4 };

    /// The setting's text: null or blank is the default, `auto` (any case) or a whole number from 1 to `max_depth`.
    pub fn parse(text: ?[]const u8) error{BadDraftDepth}!Depth {
        return parseFor(text, false);
    }

    /// `parse`, where an unset or blank setting beside a confidence floor (`confident`: the served default) is `max_depth`: the
    /// floor decides how many of a hold's drafts a round verifies, and a number then caps what is held.
    pub fn parseFor(text: ?[]const u8, confident: bool) error{BadDraftDepth}!Depth {
        const t = std.mem.trim(u8, text orelse "", " \t");
        if (t.len == 0) return if (confident) .{ .fixed = max_depth } else default;
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

/// The floor for a draft's own probability (INKLING_DRAFT_CONFIDENCE): a round verifies a hold's first draft, then each next one
/// only while its probability, the top-1 probability of its step's logits, is at least the floor, and stops at the first that is
/// not. It is each draft's own probability, not the product of the drafts' before it. A served round has the floor `default`
/// unless the setting names another or says `off`, which verifies every draft a round holds.
pub const Confidence = struct {
    /// What an unset or blank setting means.
    pub const default: f64 = 0.7;

    /// The setting's text: null or blank is `default`, `off` (any case) is no floor (null), else a number above 0 and below 1.
    pub fn parse(text: ?[]const u8) error{BadDraftConfidence}!?f64 {
        const t = std.mem.trim(u8, text orelse "", " \t");
        if (t.len == 0) return default;
        if (std.ascii.eqlIgnoreCase(t, "off")) return null;
        const x = std.fmt.parseFloat(f64, t) catch return error.BadDraftConfidence;
        if (!(x > 0 and x < 1)) return error.BadDraftConfidence;
        return x;
    }
};

/// What a served round drafts: INKLING_DRAFT_DEPTH and INKLING_DRAFT_CONFIDENCE read together. By default, with the floor of
/// `Confidence.default`, every round holds the deepest chain `depth` allows (7 when INKLING_DRAFT_DEPTH is unset, else the number
/// it gives, which caps it) and verifies as many of its drafts as the floor lets through; `auto` then holds 7 for a greedy stream.
/// With the floor `off`, `depth` drafts are held and verified every round (4 when unset).
pub const Drafting = struct {
    depth: Depth,
    confidence: ?f64 = null,

    pub fn parse(depth_text: ?[]const u8, confidence_text: ?[]const u8) error{ BadDraftDepth, BadDraftConfidence }!Drafting {
        const floor = try Confidence.parse(confidence_text);
        return .{ .depth = try Depth.parseFor(depth_text, floor != null), .confidence = floor };
    }

    /// The drafter's options for this setting.
    pub fn options(s: Drafting) Options {
        return .{ .depth = s.depth.most(), .confidence = s.confidence };
    }

    /// The log's words for how a round drafts: the drafts held and the floor that lets some through, or that every one is verified.
    pub fn describe(s: Drafting, a: std.mem.Allocator) std.mem.Allocator.Error![]u8 {
        const floor = s.confidence orelse return switch (s.depth) {
            .fixed => |n| std.fmt.allocPrint(a, "{d} drafts a round, every draft verified", .{n}),
            .auto => std.fmt.allocPrint(a, "1 to {d} drafts a round chosen by the cost rule, every draft verified", .{s.depth.most()}),
        };
        return std.fmt.allocPrint(a, "up to {d} drafts a round, the first verified and each next one while its own probability is at least {d}", .{ s.depth.most(), floor });
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
    /// The floor above 0 and below 1 for a draft's own probability (see `Confidence`): `facts` asks the lane core to verify only
    /// the drafts it lets through, and the drafter keeps each draft's probability for `chances`. Null (no floor unless the caller
    /// gives one; `Drafting` is where the served default is) verifies every draft a hold makes.
    confidence: ?f64 = null,
    /// Keep each draft's probability for `chances` without a floor (the replay records them); `confidence` implies it.
    probabilities: bool = false,
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
    /// Lazily allocated scratch for batched multi-stream holds; null while only single-stream holds run.
    batch: ?BatchScratch = null,
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
    /// The host's clock at the last absorb's commit and the GPU's end of the last settled one: floors for a command
    /// buffer Metal reports no start for.
    absorbed_at: f64 = 0,
    absorb_end: f64 = 0,
    /// Expected GPU seconds per hold by stream count (0: unset), updated like `forward.waitWarm`.
    expected: [st.max_streams + 1]f64 = @splat(0),
    /// Prompt snapshot saved at a `prefill_step` mark before `Drafted.prefillStep` absorbs it; filled on absorb.
    deferred_keep: ?struct { stream: *Stream, snap: *snapshot.Snap } = null,

    /// The draft head over `model`'s taps with `weights` (the model's `draft`): its kernels, scratch and, when the head is not
    /// yet folded, the fold; then one hold timed for `facts`.
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, model: *Model, weights: *dw.Weights, options: Options) !*Drafter {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const mtp = model.config.mtp orelse return error.NoDraftHead;
        if (!supported(&mtp, &model.config.text)) return error.UnsupportedDSpark;
        if (options.depth == 0 or options.depth > max_depth) return error.DraftDepth;
        if (options.confidence) |c| if (!(c > 0 and c < 1)) return error.DraftConfidence;
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .device = device, .model = model, .w = weights, .mtp = mtp, .options = options, .pipes = undefined, .s = undefined };
        {
            const src = try sourceFor(gpa, mtp.taps_n);
            defer gpa.free(src);
            const lib = try mtl.Library.fromSource(device, src, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
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
        d.s = try Scratch.init(device, mtp, weights);
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
        if (d.batch) |*bs| bs.deinit();
        d.s.deinit();
        if (d.qmv) |*q| q.deinit();
        for (d.pipes) |p| p.deinit();
        d.gpa.destroy(d);
    }

    fn pipe(d: *const Drafter, k: Kernel) mtl.Pipeline {
        return d.pipes[@backingInt(k)];
    }

    /// Whether the kernels support this head's shape and its final tap matches the backbone's virtual layer count.
    pub fn supported(m: *const cfg.Mtp, text: *const cfg.Text) bool {
        if (m.kv_heads != kv_heads or m.q_heads % kv_heads != 0) return false;
        const gqa = m.q_heads / kv_heads;
        if (gqa != 3 and gqa != 6) return false;
        const ok = m.layers >= 1 and m.layers <= max_draft_layers and m.head_dim == head_dim and m.rank == rank and
            m.block == block and m.taps_n >= 1 and m.taps_n <= 16 and m.window == window + 1 and
            m.hidden > 0 and m.hidden == text.hidden and m.hidden % 256 == 0 and m.vocab == text.vocab and
            m.inter > 0 and m.inter % 256 == 0;
        if (!ok) return false;
        return m.taps[m.taps_n - 1] == text.virtualLayers();
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

    fn applyLaneResident(d: *const Drafter, l: *const Lane, f: mtl.ResidencySet.Follow, join: bool) void {
        const call = if (join) f.join else f.leave;
        for (l.k[0..d.mtp.layers]) |buf| call(f.ctx, buf);
        for (l.v[0..d.mtp.layers]) |buf| call(f.ctx, buf);
        call(f.ctx, l.f_rnn);
        call(f.ctx, l.state);
        call(f.ctx, l.seq);
        call(f.ctx, l.prob);
    }

    /// Joins (or, with null, leaves) a residency set: the scratch and every lane's buffers, as lanes open and close.
    pub fn setResident(d: *Drafter, hook: ?mtl.ResidencySet.Follow) void {
        if (d.follow) |f| {
            mtl.resident.fields.apply(&d.s, f, false);
            if (d.batch) |*bs| mtl.resident.fields.apply(bs, f, false);
            var it = d.lanes.valueIterator();
            while (it.next()) |l| d.applyLaneResident(l.*, f, false);
        }
        d.follow = hook;
        if (hook) |f| {
            mtl.resident.fields.apply(&d.s, f, true);
            if (d.batch) |*bs| mtl.resident.fields.apply(bs, f, true);
            var it = d.lanes.valueIterator();
            while (it.next()) |l| d.applyLaneResident(l.*, f, true);
        }
    }

    fn ensureBatch(d: *Drafter) !void {
        if (d.batch != null) return;
        d.batch = try BatchScratch.init(d.device, d.mtp, d.w);
        if (d.follow) |f| mtl.resident.fields.apply(&d.batch.?, f, true);
    }

    /// Device bytes one stream's draft state takes: layer rings, Markov feature, state, token row and probabilities.
    pub fn laneBytes(d: *const Drafter) usize {
        return 2 * @as(usize, d.mtp.layers) * ringBytes() + @as(usize, d.mtp.hidden) * 2 + rank * 2 + block * 4 + block * 4;
    }

    /// Bytes of one row's taps: bf16 [taps_n][hidden].
    pub fn tapRowBytes(d: *const Drafter) usize {
        return @as(usize, d.mtp.taps_n) * @as(usize, d.mtp.hidden) * 2;
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

    /// Device bytes of the scratch (with the step logits' buffer and batch scratch, if made).
    pub fn scratchBytes(d: *const Drafter) usize {
        var total: usize = 0;
        inline for (@typeInfo(Scratch).@"struct".field_names) |name| {
            const v = @field(d.s, name);
            if (@TypeOf(v) == mtl.Buffer) total += v.length() else if (v) |buf| total += buf.length();
        }
        if (d.batch) |bs| inline for (BatchScratch.made) |name| {
            total += @field(bs, name).length();
        };
        return total;
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .kind = "DSpark draft head", .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn, .chances = chancesFn } };
    }

    /// Whether a hold keeps each step's probability of its pick.
    fn keepsProbabilities(d: *const Drafter) bool {
        return d.options.probabilities or d.options.confidence != null;
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(ptr: *anyopaque) []const u32 {
        const d = cast(ptr);
        return d.mtp.taps[0..d.mtp.taps_n];
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        return .{ .depth = d.options.depth, .step_ms = d.step_ms, .fixed_block = true, .plain_guard = true, .stream_bytes = d.laneBytes(), .prior = &prior, .chances = d.options.confidence != null, .confidence_floor = d.options.confidence };
    }

    fn ringBytes() usize {
        return @as(usize, ring + ring_pad) * kv_width * 2;
    }

    /// Device bytes a kept draft context takes beside a prompt state for head `m`: each draft layer's K and V rings.
    pub fn keptBytesFor(m: cfg.Mtp) u64 {
        return 2 * @as(u64, m.layers) * ringBytes();
    }

    /// Device bytes a kept draft context takes beside a prompt state (5-layer default).
    pub fn keptBytes() u64 {
        return 2 * default_draft_layers * ringBytes();
    }

    /// Copies `s`'s context rings into `snap` at `snap.at`, deferring if `prefill_step` has not absorbed yet.
    pub fn keepLane(d: *Drafter, s: *Stream, snap: *snapshot.Snap) !void {
        const l = d.lanes.get(s) orelse {
            d.deferred_keep = .{ .stream = s, .snap = snap };
            return;
        };
        try d.sync(); // the chunk's absorb, left pending
        if (l.end < snap.at) {
            d.deferred_keep = .{ .stream = s, .snap = snap };
            return;
        }
        if (l.end != snap.at or l.end <= l.from) return;
        const buf = try d.device.buffer(@intCast(keptBytesFor(d.mtp)), mtl.ResourceOptions.private);
        errdefer buf.deinit();
        try d.copyLane(l, buf, true);
        snap.setDraft(buf, l.from);
    }

    /// Clear any deferred draft-ring capture pointing at `snap` before it is freed.
    pub fn dropKeep(d: *Drafter, snap: *snapshot.Snap) void {
        if (d.deferred_keep) |dk| if (dk.snap == snap) {
            d.deferred_keep = null;
        };
    }

    /// Restore the lane's rings from the kept prompt state its stream resumed at `start`, returning true if restored.
    fn resumeLane(d: *Drafter, s: *Stream, l: *Lane, start: u64) !bool {
        const saved = s.reuse.saved orelse return false;
        const snap: *const snapshot.Snap = @ptrCast(@alignCast(saved));
        if (snap.freed or s.cached != start or snap.at != start or l.end != 0) return false;
        const buf = snap.draft orelse return false;
        try d.copyLane(l, buf, false);
        l.from = snap.draft_from;
        l.end = start;
        return true;
    }

    /// A lane's key and value rings and a kept buffer, one blit pass, waited on (`out`: lane to buffer).
    fn copyLane(d: *Drafter, l: *const Lane, buf: mtl.Buffer, out: bool) !void {
        const n = ringBytes();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.model.queue.commandBuffer();
        const e = cb.blit();
        var at: usize = 0;
        for (0..d.mtp.layers) |i| for ([_]mtl.Buffer{ l.k[i], l.v[i] }) |live| {
            if (out) e.copy(live, 0, buf, at, n) else e.copy(buf, at, live, 0, n);
            at += n;
        };
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| return failed(msg);
    }

    fn newLane(d: *Drafter) !*Lane {
        const n_layers: usize = d.mtp.layers;
        const h: usize = d.mtp.hidden;
        const l = try d.gpa.create(Lane);
        errdefer d.gpa.destroy(l);
        var bufs: [2 * max_draft_layers + 4]mtl.Buffer = undefined;
        const total = 2 * n_layers + 4;
        var made: usize = 0;
        errdefer for (bufs[0..made]) |x| x.deinit();
        const small = [4]usize{ h * 2, rank * 2, block * 4, block * 4 };
        for (bufs[0..total], 0..) |*x, i| {
            x.* = if (i < 2 * n_layers) try d.device.buffer(ringBytes(), mtl.ResourceOptions.private) else try d.device.buffer(small[i - 2 * n_layers], mtl.ResourceOptions.shared);
            made += 1;
        }
        l.* = .{
            .k = undefined,
            .v = undefined,
            .f_rnn = bufs[2 * n_layers],
            .state = bufs[2 * n_layers + 1],
            .seq = bufs[2 * n_layers + 2],
            .prob = bufs[2 * n_layers + 3],
        };
        for (0..n_layers) |i| {
            l.k[i] = bufs[2 * i];
            l.v[i] = bufs[2 * i + 1];
        }
        for ([_]mtl.Buffer{ l.f_rnn, l.state, l.seq, l.prob }) |x| @memset(x.slice(u8, x.length()), 0);
        if (d.follow) |f| d.applyLaneResident(l, f, true);
        return l;
    }

    fn dropLane(d: *Drafter, l: *Lane) void {
        if (d.follow) |f| d.applyLaneResident(l, f, false);
        for (l.k[0..d.mtp.layers]) |x| x.deinit();
        for (l.v[0..d.mtp.layers]) |x| x.deinit();
        l.f_rnn.deinit();
        l.state.deinit();
        l.seq.deinit();
        l.prob.deinit();
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
        if (d.deferred_keep) |dk| if (dk.stream == s) {
            d.deferred_keep = null;
        };
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

    /// Each held draft's own probability under its step's logits, for the first `out.len` of the hold's: what a confidence floor is
    /// compared with (the lane core compares each chance with it from the second on and stops at the first that falls short), so
    /// not the running product of the drafts' before it. False when this drafter keeps no probabilities.
    fn chancesFn(ptr: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
        const d = cast(ptr);
        if (!d.keepsProbabilities()) return false;
        const l = d.lanes.get(s) orelse return error.DraftStreamUnknown;
        if (out.len > l.held_len) return error.DraftHeldFewer;
        try chancesOf(l.prob.slice(f32, max_depth)[0..out.len], out);
        return true;
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
        const h = d.mtp.hidden;
        d.glue(e, .norm, .{ .rows = @intCast(rows), .width = h, .stride = h, .eps = d.mtp.eps }, &.{ in, t(w), out }, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, true);
    }

    /// `rows` rows of `in` through the projection `p` into `out`: the target's bf16 matvec kernels, or core's tiles for a
    /// quantized one.
    fn mv(d: *Drafter, e: mtl.ComputeEncoder, p: dw.Proj, in: B, out: B, rows: usize) !void {
        switch (p) {
            .bf16 => |x| fwd.matvec(e, d.model, x, in, out, rows),
            .q8 => |x| try d.qmv.?.encode(e, x, in, out, rows),
        }
    }

    /// Tensor-unit GEMM for bf16 dense projections when `d.model.chunk != null`.
    fn gemm(d: *const Drafter, e: mtl.ComputeEncoder, w: wts.Dense, in: B, out: B, rows: usize) void {
        dense_prefill.gemmRows(
            e,
            &d.model.chunk.?.dense,
            .{ .buf = in.buf, .off = in.off },
            .{ .buf = w.t.buffer, .off = w.t.offset, .n = w.n, .k = w.k },
            .{ .buf = out.buf, .off = out.off },
            .bf16,
            rows,
            w.n,
        );
    }

    /// Batched projection over `rows` rows across concurrent streams: uses tensor-unit GEMM for bf16 when available
    /// and `rows >= 16`, else `mv`.
    fn mvBatch(d: *Drafter, e: mtl.ComputeEncoder, p: dw.Proj, in: B, out: B, rows: usize) !void {
        switch (p) {
            .bf16 => |x| {
                if (d.model.chunk != null and rows >= 16) {
                    d.gemm(e, x, in, out, rows);
                } else {
                    fwd.matvec(e, d.model, x, in, out, rows);
                }
            },
            .q8 => |x| try d.qmv.?.encode(e, x, in, out, rows),
        }
    }

    /// log2 of the rotary base, as mlx's rope takes it (the angle is the position times 2^(-(i / 64) * this)).
    fn rope(d: *const Drafter) f32 {
        return @floatCast(std.math.log2(@as(f64, d.mtp.rope_theta)));
    }

    /// Context rows [p0, p0 + n) from `src` (rows of taps) into each layer's ring; `newest`: the last of them is the
    /// stream's newest committed row, whose Markov feature the hold reads.
    fn encodeAbsorb(d: *Drafter, e: mtl.ComputeEncoder, l: *const Lane, src: B, p0: u64, n: usize, newest: bool) !void {
        const s = &d.s;
        const ls = d.w.layers;
        const h: usize = d.mtp.hidden;
        const h32: u32 = d.mtp.hidden;
        const q_heads: u32 = d.mtp.q_heads;
        const taps_n: u32 = d.mtp.taps_n;
        const mixes: usize = @as(usize, d.mtp.layers) + 1;
        const rows: u32 = @intCast(n);
        const mix_threads = std.mem.alignForward(usize, (h + 15) / 16, 32);
        d.launch(e, .mix, MixArgs{ .rows = rows, .width = h32, .tap_stride = taps_n * h32, .mix_stride = rows * h32, .eps = d.mtp.eps }, &.{ src, b(d.w.alpha_buffer), b(d.w.dflare_norms), b(s.mixed) }, .{ n, mixes, 1 }, .{ mix_threads, 1, 1 });
        for (ls, 0..) |*layer, i| {
            const mixed = bo(s.mixed, i * n * h * 2);
            try d.mv(e, layer.k_ctx, mixed, b(s.kc), n);
            try d.mv(e, layer.v_ctx, mixed, b(s.vc), n);
            const a = PrepArgs{ .rows = rows, .heads = q_heads, .kv = kv_heads, .pos = @intCast(p0), .ring = ring, .eps = d.mtp.eps, .base_log2 = d.rope() };
            d.launch(e, .ctx_prep, a, &.{ b(s.kc), b(s.vc), t(layer.k_norm_ctx), b(l.k[i]), b(l.v[i]) }, .{ n, 1, 1 }, .{ 256, 1, 1 });
        }
        if (newest) d.glue(e, .copy, .{ .rows = 1, .width = h32 }, &.{ bo(s.mixed, (ls.len * n + n - 1) * h * 2), b(l.f_rnn) }, .{ h, 1, 1 }, .{ 256, 1, 1 }, false);
    }

    /// Block for `it`: pending token at lane's end and seven mask rows through draft layers, unembedding of rows 1..7,
    /// and `it.depth` Markov steps; proposals land in the lane's token row after the pending token.
    fn encodeHold(d: *Drafter, e: mtl.ComputeEncoder, l: *const Lane, it: dr.Hold) !void {
        const m = d.model;
        const s = &d.s;
        const text = &m.config.text;
        const h: usize = d.mtp.hidden;
        const h32: u32 = d.mtp.hidden;
        const q_heads: usize = d.mtp.q_heads;
        const q_heads32: u32 = d.mtp.q_heads;
        const gqa: usize = q_heads / kv_heads;
        const eps = d.mtp.eps;
        const end: u32 = @intCast(l.end);
        const win: u32 = @intCast(@min(@as(u64, window), l.end - l.from));
        const blocks = attnBlocks(end, win);
        const inter: usize = d.mtp.inter;
        d.glue(e, .embed, .{ .rows = 1, .width = h32, .eps = text.eps }, &.{ b(l.seq), t(m.weights.embed.t), t(m.weights.embed_norm), b(s.h0) }, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, true);
        var prev = b(s.h0);
        for (d.w.layers, 0..) |*layer, i| {
            d.norm(e, layer.input, prev, b(s.x), block);
            try d.mv(e, layer.q, b(s.x), b(s.q), block);
            try d.mv(e, layer.k, b(s.x), b(s.k), block);
            try d.mv(e, layer.v, b(s.x), b(s.v), block);
            const pa = PrepArgs{ .rows = block, .heads = q_heads32, .kv = kv_heads, .pos = end, .ring = ring, .eps = eps, .base_log2 = d.rope() };
            d.launch(e, .block_prep, pa, &.{ b(s.q), b(s.k), b(s.v), t(layer.q_norm), t(layer.k_norm), b(s.qn), b(s.kb), b(s.vb) }, .{ (q_heads + kv_heads + 7) / 8, block, 1 }, .{ 256, 1, 1 });
            const aa = AttnArgs{ .heads = q_heads32, .kv = kv_heads, .end = end, .window = win, .ring = ring, .blocks = blocks, .scale = @floatCast(1.0 / @sqrt(@as(f64, head_dim))) };
            d.launch(e, .attn_part, aa, &.{ b(s.qn), b(l.k[i]), b(l.v[i]), b(s.kb), b(s.vb), b(s.part) }, .{ kv_heads, 1, blocks }, .{ gqa * 32, 1, 1 });
            d.launch(e, .attn_merge, aa, &.{ b(s.part), b(s.att) }, .{ q_heads, block, 1 }, .{ 32, 1, 1 });
            try d.mv(e, layer.o, b(s.att), b(s.y), block);
            const row_add = EwArgs{ .rows = block, .width = h32 };
            d.launch(e, .add, row_add, &.{ prev, b(s.y), b(s.h) }, .{ h / 256, block, 1 }, .{ 256, 1, 1 });
            d.norm(e, layer.post, b(s.h), b(s.x), block);
            try d.mv(e, layer.gate, b(s.x), b(s.gate), block);
            try d.mv(e, layer.up, b(s.x), b(s.up), block);
            d.launch(e, .swiglu, EwArgs{ .rows = block, .width = @intCast(inter) }, &.{ b(s.gate), b(s.up), b(s.act) }, .{ (inter + 255) / 256, block, 1 }, .{ 256, 1, 1 });
            try d.mv(e, layer.down, b(s.act), b(s.y), block);
            d.launch(e, .add, row_add, &.{ b(s.h), b(s.y), b(s.h) }, .{ h / 256, block, 1 }, .{ 256, 1, 1 });
            prev = b(s.h);
        }
        // rows 1 to 7 only: the first row's last output is not read
        d.norm(e, d.w.norm, bo(s.h, h * 2), b(s.hn), max_depth);
        fwd.matvec(e, m, m.weights.unembed, b(s.hn), b(s.base), max_depth);
        const vocab: usize = text.vocab;
        const groups = (vocab + pick_group - 1) / pick_group;
        const fold_w = d.w.fold_w.?.t;
        const fold_b = d.w.fold_b.?;
        const mass: u32 = if (d.keepsProbabilities()) write_mass else 0;
        const flags: u32 = if (s.capture != null) write_logits | mass else mass;
        for (0..it.depth) |k| {
            const step: u32 = @intCast(k);
            d.launch(e, .rnn_proj, ProjArgs{ .n = 3 * rank, .at = step, .hidden = h32 }, &.{ b(l.state), t(d.w.markov.w1.t), b(l.seq), bo(s.hn, k * h * 2), b(l.f_rnn), t(d.w.markov.rnn.t), t(d.w.markov.rnn_b), b(s.proj) }, .{ 3 * rank / 8, 1, 1 }, .{ 256, 1, 1 });
            d.launch(e, .gru, {}, &.{ b(s.proj), b(l.state), b(l.state), b(s.latent) }, .{ 1, 1, 1 }, .{ rank, 1, 1 });
            const out = if (s.capture) |c| bo(c, k * vocab * 2) else b(s.latent);
            d.launch(e, .fold_pick, FoldArgs{ .vocab = @intCast(vocab), .valid = text.unpadded_vocab, .flags = flags, .cap = text.logit_cap }, &.{ bo(s.base, k * vocab * 2), b(s.latent), t(fold_w), t(fold_b), b(s.best), b(s.best_id), out, b(s.mass) }, .{ groups, 1, 1 }, .{ 256, 1, 1 });
            d.launch(e, .pick, PickArgs{ .groups = @intCast(groups), .step = step, .flags = mass }, &.{ b(s.best), b(s.best_id), b(l.seq), b(s.mass), b(l.prob) }, .{ 1, 1, 1 }, .{ 1024, 1, 1 });
        }
    }

    /// Batched hold for `items` (`2 <= items.len <= max_batch`, no logit capture): shares draft-layer projections,
    /// target unembedding, and fold-pick weight reads across all concurrent streams.
    fn encodeHoldBatch(d: *Drafter, e: mtl.ComputeEncoder, items: []const dr.Hold) !void {
        std.debug.assert(items.len >= 2 and items.len <= max_batch and d.s.capture == null);
        const c = items.len;
        const bs = &d.batch.?;
        const m = d.model;
        const text = &m.config.text;
        const h: usize = d.mtp.hidden;
        const h32: u32 = d.mtp.hidden;
        const q_heads: usize = d.mtp.q_heads;
        const q_heads32: u32 = d.mtp.q_heads;
        const gqa: usize = q_heads / kv_heads;
        const eps = d.mtp.eps;
        const inter: usize = d.mtp.inter;
        const total_rows = c * block;
        const total_rows32: u32 = @intCast(total_rows);
        var lanes_buf: [max_batch]*const Lane = undefined;
        for (items, 0..) |it, si| {
            const l = d.lanes.get(it.stream).?;
            lanes_buf[si] = l;
            d.glue(e, .embed, .{ .rows = 1, .width = h32, .eps = text.eps }, &.{ b(l.seq), t(m.weights.embed.t), t(m.weights.embed_norm), bo(bs.h0, si * block * h * 2) }, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, true);
        }
        const q_stride = block * q_heads * head_dim * 2;
        const kv_stride = block * kv_width * 2;
        const attn_scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, head_dim)));
        const row_add = EwArgs{ .rows = total_rows32, .width = h32 };
        var prev = b(bs.h0);
        for (d.w.layers, 0..) |*layer, i| {
            d.norm(e, layer.input, prev, b(bs.x), total_rows);
            try d.mvBatch(e, layer.q, b(bs.x), b(bs.q), total_rows);
            try d.mvBatch(e, layer.k, b(bs.x), b(bs.k), total_rows);
            try d.mvBatch(e, layer.v, b(bs.x), b(bs.v), total_rows);
            for (lanes_buf[0..c], 0..) |l, si| {
                const end: u32 = @intCast(l.end);
                const win: u32 = @intCast(@min(@as(u64, window), l.end - l.from));
                const blocks = attnBlocks(end, win);
                const q_off = si * q_stride;
                const kv_off = si * kv_stride;
                const pa = PrepArgs{ .rows = block, .heads = q_heads32, .kv = kv_heads, .pos = end, .ring = ring, .eps = eps, .base_log2 = d.rope() };
                d.launch(e, .block_prep, pa, &.{ bo(bs.q, q_off), bo(bs.k, kv_off), bo(bs.v, kv_off), t(layer.q_norm), t(layer.k_norm), bo(bs.qn, q_off), bo(bs.kb, kv_off), bo(bs.vb, kv_off) }, .{ (q_heads + kv_heads + 7) / 8, block, 1 }, .{ 256, 1, 1 });
                const aa = AttnArgs{ .heads = q_heads32, .kv = kv_heads, .end = end, .window = win, .ring = ring, .blocks = blocks, .scale = attn_scale };
                d.launch(e, .attn_part, aa, &.{ bo(bs.qn, q_off), b(l.k[i]), b(l.v[i]), bo(bs.kb, kv_off), bo(bs.vb, kv_off), b(d.s.part) }, .{ kv_heads, 1, blocks }, .{ gqa * 32, 1, 1 });
                d.launch(e, .attn_merge, aa, &.{ b(d.s.part), bo(bs.att, q_off) }, .{ q_heads, block, 1 }, .{ 32, 1, 1 });
            }
            try d.mvBatch(e, layer.o, b(bs.att), b(bs.y), total_rows);
            d.launch(e, .add, row_add, &.{ prev, b(bs.y), b(bs.h) }, .{ h / 256, total_rows, 1 }, .{ 256, 1, 1 });
            d.norm(e, layer.post, b(bs.h), b(bs.x), total_rows);
            try d.mvBatch(e, layer.gate, b(bs.x), b(bs.gate), total_rows);
            try d.mvBatch(e, layer.up, b(bs.x), b(bs.up), total_rows);
            d.launch(e, .swiglu, EwArgs{ .rows = total_rows32, .width = @intCast(inter) }, &.{ b(bs.gate), b(bs.up), b(bs.act) }, .{ (inter + 255) / 256, total_rows, 1 }, .{ 256, 1, 1 });
            try d.mvBatch(e, layer.down, b(bs.act), b(bs.y), total_rows);
            d.launch(e, .add, row_add, &.{ b(bs.h), b(bs.y), b(bs.h) }, .{ h / 256, total_rows, 1 }, .{ 256, 1, 1 });
            prev = b(bs.h);
        }
        // Final norm of rows 1..7 per stream into contiguous [c][max_depth][h], then one batched unembed.
        for (0..c) |si| {
            d.norm(e, d.w.norm, bo(bs.h, (si * block + 1) * h * 2), bo(bs.hn, si * max_depth * h * 2), max_depth);
        }
        const base_rows = c * max_depth;
        if (m.chunk != null) {
            d.gemm(e, m.weights.unembed, b(bs.hn), b(bs.base), base_rows);
        } else {
            fwd.matvec(e, m, m.weights.unembed, b(bs.hn), b(bs.base), base_rows);
        }
        const vocab: usize = text.vocab;
        const groups = (vocab + pick_group - 1) / pick_group;
        const fold_w = d.w.fold_w.?.t;
        const fold_b = d.w.fold_b.?;
        const mass: u32 = if (d.keepsProbabilities()) write_mass else 0;
        var max_it_depth: u32 = 0;
        for (items) |it| max_it_depth = @max(max_it_depth, it.depth);
        for (0..max_it_depth) |k| {
            const step: u32 = @intCast(k);
            var act_si: [max_batch]u32 = undefined;
            var na: usize = 0;
            for (items, 0..) |it, si| {
                if (step < it.depth) {
                    act_si[na] = @intCast(si);
                    na += 1;
                }
            }
            for (0..na) |ai| {
                const si: usize = act_si[ai];
                const l = lanes_buf[si];
                d.launch(e, .rnn_proj, ProjArgs{ .n = 3 * rank, .at = step, .hidden = h32 }, &.{ b(l.state), t(d.w.markov.w1.t), b(l.seq), bo(bs.hn, (si * max_depth + k) * h * 2), b(l.f_rnn), t(d.w.markov.rnn.t), t(d.w.markov.rnn_b), b(d.s.proj) }, .{ 3 * rank / 8, 1, 1 }, .{ 256, 1, 1 });
                d.launch(e, .gru, {}, &.{ b(d.s.proj), b(l.state), b(l.state), bo(bs.latent, ai * rank * 2) }, .{ 1, 1, 1 }, .{ rank, 1, 1 });
            }
            if (na == 1) {
                const si: usize = act_si[0];
                d.launch(e, .fold_pick, FoldArgs{ .vocab = @intCast(vocab), .valid = text.unpadded_vocab, .flags = mass, .cap = text.logit_cap }, &.{ bo(bs.base, (si * max_depth + k) * vocab * 2), b(bs.latent), t(fold_w), t(fold_b), b(bs.best), b(bs.best_id), b(bs.latent), b(bs.mass) }, .{ groups, 1, 1 }, .{ 256, 1, 1 });
            } else {
                var fa: FoldBatchArgs = .{
                    .vocab = @intCast(vocab),
                    .valid = text.unpadded_vocab,
                    .flags = mass,
                    .count = @intCast(na),
                    .base_stride = @intCast(max_depth * vocab),
                    .cap = text.logit_cap,
                    .slot_map = @splat(0),
                };
                for (0..na) |ai| fa.slot_map[ai] = act_si[ai];
                d.launch(e, .fold_pick_batch, fa, &.{ bo(bs.base, k * vocab * 2), b(bs.latent), t(fold_w), t(fold_b), b(bs.best), b(bs.best_id), b(bs.mass) }, .{ groups, 1, 1 }, .{ 256, 1, 1 });
            }
            for (0..na) |ai| {
                const l = lanes_buf[act_si[ai]];
                d.launch(e, .pick, PickArgs{ .groups = @intCast(groups), .step = step, .flags = mass }, &.{ bo(bs.best, ai * groups * 4), bo(bs.best_id, ai * groups * 4), b(l.seq), bo(bs.mass, ai * groups * 4), b(l.prob) }, .{ 1, 1, 1 }, .{ 1024, 1, 1 });
            }
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
        d.absorb_end = cb.gpuEnd();
        d.absorb_gpu = d.absorb_end - cb.gpuStartOr(d.absorbed_at);
    }

    /// Rows [start, start + rows) replace the context from `start`; the rows of a prompt before its last window are skipped. The
    /// work is committed and not waited for: the hold that follows runs behind it on the queue, and the target's next forward
    /// writes its taps only after it.
    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        const row_bytes = d.tapRowBytes();
        for (items) |it| {
            _ = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
            const f = it.features;
            if (f.space != .device or f.dtype != .bf16 or f.row_bytes != row_bytes or f.rows == 0 or std.meta.activeTag(f.ready) != .none or it.rows != null) return error.DraftFeaturesUnsupported;
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try d.sync();
        for (items) |it| {
            const l = d.lanes.get(it.stream).?;
            if (it.start > l.end) {
                const need_prior = !it.prompt or (@as(u64, it.stream.prompt_len -| window) < it.start);
                const restored = if (need_prior) try d.resumeLane(it.stream, l, it.start) else false;
                if (!restored) {
                    l.from = it.start;
                    l.end = it.start;
                }
            }
        }
        var cb: ?mtl.CommandBuffer = null;
        var enc: mtl.ComputeEncoder = undefined;
        for (items) |it| {
            const l = d.lanes.get(it.stream).?;
            const f = it.features;
            const src = mtl.Buffer{ .id = @ptrFromInt(f.buffer) };
            const end = it.start + f.rows;
            if (it.start < l.from) l.from = it.start;
            // only the last window of a long prompt is ever attended
            const keep_from = if (it.prompt) @max(it.start, @as(u64, it.stream.prompt_len -| window)) else it.start;
            if (keep_from >= end) {
                l.from = end;
                l.end = end;
                l.held_len = 0;
                continue;
            }
            if (keep_from > l.end) l.from = keep_from;
            var at: u64 = keep_from;
            while (at < end) {
                if (cb == null) {
                    cb = d.model.queue.commandBuffer();
                    enc = cb.?.compute(.serial);
                }
                const n: usize = @intCast(@min(@as(u64, absorb_rows), end - at));
                d.encodeAbsorb(enc, l, bo(src, @intCast(f.offset + (at - it.start) * @as(u64, row_bytes))), at, n, at + n == end) catch |err| {
                    enc.end(); // an open encoder may not be dropped
                    return err;
                };
                at += n;
            }
            l.end = end;
            l.held_len = 0;
        }
        if (cb) |one| {
            enc.end();
            d.absorbed_at = fwd.seconds();
            one.commit();
            d.pending = one.retain();
        }
        if (d.deferred_keep) |dk| {
            for (items) |it| if (it.stream == dk.stream) {
                const l = d.lanes.get(dk.stream).?;
                if (l.end >= dk.snap.at) {
                    d.deferred_keep = null;
                    if (l.end == dk.snap.at) try d.keepLane(dk.stream, dk.snap);
                }
                break;
            };
        }
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
        const use_batch = items.len > 1 and d.s.capture == null;
        if (use_batch) try d.ensureBatch();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const began = now();
        const cb = d.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (items) |it| {
            const l = d.lanes.get(it.stream).?;
            l.seq.slice(u32, block)[0] = it.pending;
            @memset(l.state.slice(u8, rank * 2), 0);
        }
        if (use_batch) {
            var at: usize = 0;
            while (at < items.len) {
                const n = @min(items.len - at, max_batch);
                const chunk = items[at..][0..n];
                const res = if (chunk.len == 1)
                    d.encodeHold(e, d.lanes.get(chunk[0].stream).?, chunk[0])
                else
                    d.encodeHoldBatch(e, chunk);
                res catch |err| {
                    e.end();
                    return err;
                };
                at += n;
            }
        } else {
            for (items) |it| {
                const l = d.lanes.get(it.stream).?;
                d.encodeHold(e, l, it) catch |err| {
                    e.end();
                    return err;
                };
            }
        }
        e.end();
        const committed = fwd.seconds();
        cb.commit();
        const slot = @min(items.len, st.max_streams);
        if (d.model.warm_wait) fwd.waitWarm(cb, d.expected[slot], committed) else cb.wait();
        d.sync() catch |err| {
            std.log.err("inkling2 drafter: the absorb before a hold failed: {s}", .{@errorName(err)});
            return err;
        };
        if (cb.failure()) |msg| return failed(msg);
        // a hold queued behind an absorb starts no earlier than the absorb's end; without a reported start its span is
        // counted from there (or from its commit), never from the clock's zero
        d.hold_gpu = cb.gpuEnd() - cb.gpuStartOr(@max(committed, d.absorb_end));
        const old = d.expected[slot];
        d.expected[slot] = if (old == 0 or d.hold_gpu < old) d.hold_gpu else old + (d.hold_gpu - old) / 4;
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
        const row_bytes = d.tapRowBytes();
        const feat = try d.device.buffer(rows * row_bytes, mtl.ResourceOptions.shared);
        defer feat.deinit();
        var prng = std.Random.DefaultPrng.init(0x5eed);
        const row = feat.slice(u16, row_bytes / 2);
        for (row) |*x| x.* = mix.bf16Of(prng.random().float(f32) * 2 - 1);
        const all = feat.slice(u16, rows * row_bytes / 2);
        for (1..rows) |r| @memcpy(all[r * row.len ..][0..row.len], row);
        const ids = [_]u32{ 1000, 2000, 3000, 4000 };
        var stream: Stream = try Stream.init(d.gpa, .{ .id = "warm", .prompt = &ids, .max_new = 1 });
        defer stream.deinit(d.gpa);
        try d.lanes.put(d.gpa, &stream, l);
        defer _ = d.lanes.remove(&stream);
        const feature = lanes.backend.Features{ .buffer = @intFromPtr(feat.id), .rows = rows, .row_bytes = @intCast(row_bytes), .space = .device, .dtype = .bf16 };
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

/// `raw`'s probabilities as the lane core's chances: each is above 0 and at most 1, or the hold went wrong (a NaN among them).
fn chancesOf(raw: []const f32, out: []f64) error{BadDraftProbability}!void {
    for (raw, out) |p, *o| {
        if (!(p > 0 and p <= 1)) return error.BadDraftProbability;
        o.* = p;
    }
}

test "inkling2 dspark: a hold's probabilities are the chances the lane core compares with the floor, each above 0 and at most 1" {
    const t = std.testing;
    const raw = [_]f32{ 1, 0.9990234375, 0.5, 0.123456789, 4.9987e-6, 0.7 };
    var out: [raw.len]f64 = undefined;
    try chancesOf(&raw, &out);
    // each is the f32 value itself, as an f64: the draft's own probability, not a running product
    for (raw, out) |x, y| try t.expectEqual(@as(f64, x), y);
    try chancesOf(raw[0..0], out[0..0]);
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    for ([_]f32{ 0, -0.5, 1.0000001, 2, nan, inf, -inf }) |bad| {
        var one = [_]f32{ 0.8, bad, 0.6 };
        try t.expectError(error.BadDraftProbability, chancesOf(&one, out[0..3]));
    }
}

test "inkling2 dspark: the confidence floor is 0.7 when unset, none when off, else a number above 0 and below 1" {
    const t = std.testing;
    try t.expectEqual(@as(f64, 0.7), Confidence.default);
    for ([_]?[]const u8{ null, "", " \t" }) |text| try t.expectEqual(@as(?f64, 0.7), try Confidence.parse(text));
    for ([_][]const u8{ "off", "OFF", " Off ", "oFf\t" }) |text| try t.expectEqual(@as(?f64, null), try Confidence.parse(text));
    for ([_][]const u8{ "0.7", " 0.7 ", "7e-1", "0.5", "0.999", "1e-3" }) |text| {
        const want = try std.fmt.parseFloat(f64, std.mem.trim(u8, text, " "));
        try t.expectEqual(@as(?f64, want), try Confidence.parse(text));
    }
    for ([_][]const u8{ "0", "1", "-0.5", "1.5", "70", "nan", "inf", "-inf", "half", "0.7 0.8", "0,7", "%", "of", "offf", "on", "no", "none", "false" }) |text| try t.expectError(error.BadDraftConfidence, Confidence.parse(text));
}

test "inkling2 dspark: a served round holds the deepest chain and verifies by a floor of 0.7, or with the floor off every draft of a fixed depth" {
    const t = std.testing;
    // nothing set: the floor, and all seven drafts held
    for ([_]?[]const u8{ null, "", "  " }) |text| {
        const d = try Drafting.parse(text, null);
        try t.expectEqual(Depth{ .fixed = max_depth }, d.depth);
        try t.expectEqual(@as(?f64, Confidence.default), d.confidence);
        try t.expectEqual(Options{ .depth = max_depth, .confidence = 0.7 }, d.options());
        try t.expectEqual(d, try Drafting.parse(text, ""));
    }
    // a number caps what is held, the floor may be another, and auto holds seven for a greedy stream
    try t.expectEqual(Options{ .depth = 5, .confidence = 0.7 }, (try Drafting.parse("5", null)).options());
    try t.expectEqual(Options{ .depth = 5, .confidence = 0.9 }, (try Drafting.parse("5", "0.9")).options());
    try t.expectEqual(Options{ .depth = max_depth, .confidence = 0.5 }, (try Drafting.parse(null, "0.5")).options());
    const auto = try Drafting.parse("auto", null);
    try t.expectEqual(Depth.auto, auto.depth);
    try t.expectEqual(Options{ .depth = max_depth, .confidence = 0.7 }, auto.options());
    // off: no floor, the depth setting reads as it always has (4 when unset), and nothing is kept
    for ([_]?[]const u8{ null, "", "  " }) |text| {
        const d = try Drafting.parse(text, "off");
        try t.expectEqual(Depth{ .fixed = 4 }, d.depth);
        try t.expectEqual(@as(?f64, null), d.confidence);
        try t.expectEqual(Options{ .depth = 4 }, d.options());
    }
    try t.expectEqual(Options{ .depth = 6 }, (try Drafting.parse("6", "OFF")).options());
    const off_auto = try Drafting.parse("auto", " off ");
    try t.expectEqual(Depth.auto, off_auto.depth);
    try t.expectEqual(Options{ .depth = max_depth }, off_auto.options());
    // each text is refused by its own name
    try t.expectError(error.BadDraftDepth, Drafting.parse("8", null));
    try t.expectError(error.BadDraftDepth, Drafting.parse("0", "off"));
    try t.expectError(error.BadDraftConfidence, Drafting.parse("4", "1"));
    try t.expectError(error.BadDraftConfidence, Drafting.parse(null, "0"));
    try t.expectError(error.BadDraftConfidence, Drafting.parse(null, "of"));
    // the library's own default is no floor: only the settings' reading has one
    try t.expectEqual(@as(?f64, null), (Options{}).confidence);
}

test "inkling2 dspark: the startup log names the floor, or says every draft is verified" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const floor = "the first verified and each next one while its own probability is at least";
    const cases = [_]struct { depth: ?[]const u8, confidence: ?[]const u8, text: []const u8 }{
        .{ .depth = null, .confidence = null, .text = "up to 7 drafts a round, " ++ floor ++ " 0.7" },
        .{ .depth = "5", .confidence = "0.9", .text = "up to 5 drafts a round, " ++ floor ++ " 0.9" },
        .{ .depth = "auto", .confidence = null, .text = "up to 7 drafts a round, " ++ floor ++ " 0.7" },
        .{ .depth = null, .confidence = "off", .text = "4 drafts a round, every draft verified" },
        .{ .depth = "6", .confidence = "OFF", .text = "6 drafts a round, every draft verified" },
        .{ .depth = "auto", .confidence = "off", .text = "1 to 7 drafts a round chosen by the cost rule, every draft verified" },
    };
    for (cases) |c| try std.testing.expectEqualStrings(c.text, try (try Drafting.parse(c.depth, c.confidence)).describe(a));
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
