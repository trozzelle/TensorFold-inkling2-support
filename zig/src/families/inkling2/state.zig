//! A stream's caches (each layer's keys and values, the convolutions' input rings, the last logits) and a forward's
//! scratch. Global layers keep every position; sliding layers keep a ring of their window, a prompt chunk and one
//! 64-key block more, so old rows are evicted as the sequence advances.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const exl3 = @import("../../core/exl3_projection.zig");
const attention_prefill = @import("attention_prefill.zig");
const experts_prefill = @import("experts_prefill.zig");

/// Rows of a verify window at most (the pending token and its copy drafts).
pub const window_rows = 16;
/// Total rows across all concurrent streams in one shared decode round.
pub const batch_rows = 64;
/// Maximum concurrent streams in one shared decode round.
pub const max_streams = 8;
/// Maximum 16-row slabs of stacked attention-input matvecs per layer in one shared decode round.
pub const max_mvs_slabs = batch_rows / window_rows;
/// Rows of a prompt chunk at most when a load does not ask for others (`model.Options.prefill_rows`): a chunk reads every weight
/// once, so more rows share that read, and the scratch grows with them. A row's bits do not depend on the chunk it is in, so the
/// rows a chunk holds change no bits.
pub const default_prefill_rows = 4096;
/// Rows of a per-operation prompt chunk at most when a load names none: the first release's (0.57 GiB, not 4.0).
pub const per_op_prefill_rows = 512;
/// A load's chunk rows are a multiple of `prefill_unit` from `prefill_unit` to `max_prefill_rows`.
pub const prefill_unit = 64;
pub const max_prefill_rows = 8192;

/// Whether `rows` is a prompt chunk size a load may take.
pub fn validPrefillRows(rows: usize) bool {
    return rows >= prefill_unit and rows <= max_prefill_rows and rows % prefill_unit == 0;
}

/// The rows INKLING_PREFILL_ROWS names: null when blank (see `defaultPrefillRows`), else one `validPrefillRows` takes.
pub fn parsePrefillRows(text: ?[]const u8) error{BadPrefillRows}!?u32 {
    const t = std.mem.trim(u8, text orelse "", " \t");
    if (t.len == 0) return null;
    const n = std.fmt.parseInt(u32, t, 10) catch return error.BadPrefillRows;
    return if (validPrefillRows(n)) n else error.BadPrefillRows;
}

/// Rows of a prompt chunk at most for a load that names none, by the path its chunks take.
pub fn defaultPrefillRows(path: PromptPath) u32 {
    return if (path == .per_op) per_op_prefill_rows else default_prefill_rows;
}

/// How a load runs the rows of a prompt chunk, which decides what its scratch holds.
pub const PromptPath = enum {
    /// On the tensor units: the GEMM and attention kernels, and the experts' chain on every layer.
    chunk,
    /// The same GEMM and attention, with the per-row grouped projections for the experts of a layer the chain does not take (the
    /// chain takes K2 and K3 experts only): the scratch holds those projections' buffers for a whole chunk.
    chunk_rows,
    /// A GPU without tensor units: the per-operation kernels (matvec slabs, a relative-bias table and one-row attention, the
    /// per-row grouped expert projections), with a depth-mix buffer and a table of their own.
    per_op,
};

/// What a scratch is made for: the rows of a prompt chunk at most, and the path they take.
pub const Chunking = struct { rows: usize, path: PromptPath = .chunk };

/// What INKLING_PROMPT_PATH asks of a load, to compare the paths and to test the fallbacks on a GPU and a pack that need none:
/// `per_op` runs prompt chunks on the per-operation kernels whatever the GPU; `rows` (every layer) or `rows:3,7` (those layers)
/// sends layers' experts through the per-row grouped projections as if the chain could not take them. Blank and `auto` ask nothing.
pub const PathRequest = struct {
    per_op: bool = false,
    /// Bit i: layer i's experts take the per-row projections.
    rows: u128 = 0,
};

comptime {
    std.debug.assert(cfg.max_layers <= 128);
}

pub fn parsePromptPath(text: ?[]const u8) error{BadPromptPath}!PathRequest {
    const t = std.mem.trim(u8, text orelse "", " \t");
    if (t.len == 0 or std.mem.eql(u8, t, "auto")) return .{};
    if (std.mem.eql(u8, t, "per_op")) return .{ .per_op = true };
    if (std.mem.eql(u8, t, "rows")) return .{ .rows = ~@as(u128, 0) };
    if (!std.mem.startsWith(u8, t, "rows:")) return error.BadPromptPath;
    var out: PathRequest = .{};
    var it = std.mem.splitScalar(u8, t["rows:".len..], ',');
    while (it.next()) |part| {
        const layer = std.fmt.parseInt(u32, std.mem.trim(u8, part, " \t"), 10) catch return error.BadPromptPath;
        if (layer >= cfg.max_layers) return error.BadPromptPath;
        out.rows |= @as(u128, 1) << @intCast(layer);
    }
    return out;
}

/// Positions each convolution's input ring holds: three taps back plus a window's rows, so a rejected window rolls
/// back by moving the cache length alone.
pub const ring = 32;
/// Keys in one block of the decode path's split attention; blocks are anchored at absolute positions.
pub const key_block = 256;
/// Floats one (row, q head, block) partial of the split attention holds: the block's maximum and weight sum, then its
/// 128 values.
pub const part_slot = 130;
/// Taps in a row of the drafter's features: the hidden state entering each of seven layers (forward.zig's `tap_layers`),
/// then the finals.
pub const feature_taps = 8;

/// Rows of the grouped expert tiles for `slots` picks: each expert's picks padded to whole 8-row tiles.
pub fn tileRows(slots: usize, experts: usize) usize {
    return std.mem.alignForward(usize, slots + 7 * @min(experts, slots), 8);
}

/// Block states a row holds at most: one a finished block of n_layers_per_attn_res_block layers.
pub fn maxBlocks(c: *const cfg.Text) usize {
    return (c.layers + c.block - 1) / c.block;
}

fn kvWidth(c: *const cfg.Text) usize {
    return @as(usize, c.local.kv) * c.local.dim;
}

/// Bytes of one position's keys and values in one layer (bf16).
pub fn kvBytes(c: *const cfg.Text) usize {
    return kvWidth(c) * 2 * 2;
}

/// The four convolution inputs' widths: k, v, attention output, MLP output.
pub fn convWidths(c: *const cfg.Text) [4]usize {
    return .{ kvWidth(c), kvWidth(c), c.hidden, c.hidden };
}

/// Bytes of one layer's convolution rings (bf16).
pub fn convBytes(c: *const cfg.Text) usize {
    var w: usize = 0;
    for (convWidths(c)) |x| w += x;
    return ring * w * 2;
}

/// Positions a cache's keys and values hold for a context of `capacity`: a multiple of the chunk attention's key block, which
/// reads whole blocks of keys and values, so the last block of a context never reaches past a layer's keys into its values or
/// past the allocation.
pub fn cachePositions(capacity: usize) usize {
    return std.mem.alignForward(usize, capacity, attention_prefill.key_block);
}

/// Slots of a sliding layer's ring for a load whose prompt chunks hold `rows` rows at most: its window, a chunk (a chunk's
/// keys are all written before its attention reads) and one more 64-key block (the chunk attention reads whole blocks, so up
/// to 63 keys below a window), to a multiple of the decode attention's block: no block of either kernel wraps.
pub fn slidingRing(c: *const cfg.Text, rows: usize) usize {
    return std.mem.alignForward(usize, @as(usize, c.window) + rows + attention_prefill.key_block, key_block);
}

/// Slots layer `i` holds for a stream of `capacity` positions: every position on a global layer; on a sliding layer its
/// ring, or every position where the stream is shorter than the ring.
pub fn layerSlots(c: *const cfg.Text, i: usize, capacity: usize, rows: usize) usize {
    const full = cachePositions(capacity);
    return if (cfg.isGlobal(@intCast(i))) full else @min(full, slidingRing(c, rows));
}

/// Bytes a stream of `capacity` positions holds over the first `layers` layers, for prompt chunks of `rows` rows.
pub fn cacheBytes(c: *const cfg.Text, capacity: usize, layers: usize, rows: usize) usize {
    var total: usize = layers * convBytes(c) + @as(usize, c.vocab) * 2;
    for (0..layers) |i| total += layerSlots(c, i, capacity, rows) * kvBytes(c);
    return total;
}

/// A stream's latest tapped rows for its drafter: cache positions [start, start + rows), a row `feature_taps` hidden rows
/// wide, from byte `off` of `buf`, valid while `gen` matches the backend's tap generation.
pub const Tap = struct { buf: ?mtl.Buffer = null, off: usize = 0, start: usize = 0, rows: usize = 0, gen: u64 = 0 };

pub const Cache = struct {
    kv: mtl.Buffer,
    conv: mtl.Buffer,
    logits: mtl.Buffer, // the prompt's last row's bf16 logits, kept for the first draw
    /// Positions the stream's context may reach (`len` stops there); each layer holds `slots[l]` slots.
    capacity: usize,
    layers: usize,
    len: usize = 0,
    layer_at: [cfg.max_layers]usize,
    slots: [cfg.max_layers]u32,
    row: usize,
    rows: usize,
    conv_layer: usize,
    widths: [4]usize,
    follow: ?mtl.ResidencySet.Follow = null,
    tap: Tap = .{},
    /// The prompt's last tapped row, kept for the first draft (the prompt chunk's buffer serves the next one).
    last_tap: ?mtl.Buffer = null,
    /// Absolute position stored in `last_tap` (null when none).
    last_tap_pos: ?usize = null,

    pub fn init(device: mtl.Device, c: *const cfg.Text, capacity: usize, layers: usize, rows: usize) !Cache {
        const kv_row = kvBytes(c);
        var out = Cache{
            .kv = undefined,
            .conv = undefined,
            .logits = undefined,
            .capacity = capacity,
            .layers = layers,
            .layer_at = @splat(0),
            .slots = @splat(0),
            .row = kv_row / 2,
            .rows = rows,
            .conv_layer = convBytes(c),
            .widths = convWidths(c),
        };
        var kv_total: usize = 0;
        for (0..layers) |i| {
            const s = layerSlots(c, i, capacity, rows);
            out.layer_at[i] = kv_total;
            out.slots[i] = @intCast(s);
            kv_total += s * kv_row;
        }
        out.kv = try device.buffer(@max(kv_total, 16), mtl.ResourceOptions.private);
        errdefer out.kv.deinit();
        out.conv = try device.buffer(@max(out.conv_layer * layers, 16), mtl.ResourceOptions.private);
        errdefer out.conv.deinit();
        out.logits = try device.buffer(@as(usize, c.vocab) * 2, mtl.ResourceOptions.shared);
        return out;
    }

    /// Byte offsets in `kv`: layer l's keys [slots[l], kv, dim], then its values; position p lives at slot p % ring(l).
    pub fn keys(self: *const Cache, l: usize) usize {
        return self.layer_at[l];
    }
    pub fn values(self: *const Cache, l: usize) usize {
        return self.layer_at[l] + @as(usize, self.slots[l]) * self.row;
    }
    pub fn ring(self: *const Cache, l: usize) u32 {
        return self.slots[l];
    }

    /// Byte offset in `conv` of layer l's ring `which` (0 k, 1 v, 2 attention output, 3 MLP output): [ring, width].
    pub fn ringAt(self: *const Cache, l: usize, which: usize) usize {
        var at = l * self.conv_layer;
        for (self.widths[0..which]) |w| at += @import("state.zig").ring * w * 2;
        return at;
    }

    pub fn setResident(self: *Cache, hook: ?mtl.ResidencySet.Follow) void {
        if (self.follow) |f| mtl.resident.fields.apply(self, f, false);
        self.follow = hook;
        if (hook) |f| mtl.resident.fields.apply(self, f, true);
    }

    pub fn deinit(self: *Cache) void {
        self.setResident(null);
        if (self.last_tap) |x| x.deinit();
        self.kv.deinit();
        self.conv.deinit();
        self.logits.deinit();
    }
};

/// Bytes of the split attention's partials for a decode window over caches of up to `capacity` positions: one slot a
/// row, q head and 256-key block.
pub fn partBytes(c: *const cfg.Text, capacity: usize) usize {
    return window_rows * @as(usize, c.local.q) * ((capacity + key_block - 1) / key_block) * part_slot * 4;
}

/// The buffers of a `Scratch`, by name. Each holds `prefill_rows` rows, but for the decode path's (a verify window's rows: `logits`,
/// `segs`, `planes`, `sums`, `ftaps`), `part` (the context's) and the per-row expert path's (`texp`, `tslot`, `ttok`, `gu`, `act`,
/// `ey`, `exl3`), which hold a window's rows unless the load sends prompt chunks through that path (`PromptPath`).
///
/// A chunk on the tensor units shares bytes between buffers whose lifetimes within a prompt layer do not overlap (each is written
/// in full before it is read, within the layer, and none carries a value into the next): `stage` is the attention input's depth mix
/// (f32 [rows][3][hidden], written by the depth kernel and read by the norms after it), the chunk attention's relative-bias table
/// (written and read by the attention) and the experts' chain (`experts_prefill.Xp`: sort tables, window lists and planes, written
/// and read by the experts). A layer runs them in that order, each finished before the next starts (one serial encoder), and each
/// starts at the front of the buffer. The per-operation path has no `stage`: its depth mix is `ai` and its table `pb`, and they are
/// empty on the other paths.
pub const fields = [_][]const u8{ "ids", "h", "nr", "blocks", "ai", "xq", "xk", "xv", "q", "k", "v", "r", "gate", "sink", "dyn", "kc", "vc", "qn", "reln", "pb", "att", "o", "ya", "res", "ns", "mi", "xm", "logits32", "eids", "ew", "texp", "tslot", "ttok", "gu", "act", "ey", "moe", "ym", "exl3", "fin", "finals", "logits", "segs", "part", "planes", "sums", "ftaps", "ptaps", "stage" };

/// The segment tables of the stacked attention-input matvecs: one `seg_slot` bytes of up to `max_segs` entries a layer.
pub const max_segs = 6;
pub const seg_slot = 256;

/// Queries the attention input's depth mix holds a row: one for a local layer, three (q, k, v) for a global one.
const max_queries = 3;

fn named(comptime a: []const u8, comptime b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// Rows the per-row expert path (`ik_tiles` and the grouped projections) holds at once: a shared decode round's, or a chunk's on a path
/// that sends the chunk through it.
fn genericRows(ch: Chunking) usize {
    return if (ch.path == .chunk) batch_rows else ch.rows;
}

/// Bytes of the Scratch buffer `name` for a load's `ch` and caches of up to `capacity` positions; `ex` is the per-row expert path's
/// scratch for `genericRows`.
fn bytesOf(comptime name: []const u8, c: *const cfg.Text, capacity: usize, ch: Chunking, ex: usize) usize {
    const R: usize = ch.rows;
    const H: usize = c.hidden;
    const qh: usize = c.local.q;
    const qw: usize = qh * c.local.dim;
    const kvw = kvWidth(c);
    const I: usize = c.expert_inter;
    const picks = R * c.topk; // a chunk's expert picks
    const win_slots = batch_rows * c.topk; // a shared decode round's
    const gen_slots = genericRows(ch) * c.topk; // the per-row expert path's
    const tiles = tileRows(gen_slots, c.experts);
    const per_op = ch.path == .per_op;
    if (comptime named(name, "ids")) return R * 4;
    if (comptime named(name, "h")) return R * H * 2;
    if (comptime named(name, "nr")) return R * H * 4;
    if (comptime named(name, "blocks")) return maxBlocks(c) * R * H * 4;
    if (comptime named(name, "ai")) return if (per_op) R * max_queries * H * 4 else 0;
    if (comptime named(name, "xq")) return @max(R, max_queries * batch_rows) * H * 2;
    if (comptime (named(name, "xk") or named(name, "xv"))) return R * H * 2;
    if (comptime named(name, "q")) return R * qw * 2;
    if (comptime (named(name, "k") or named(name, "v"))) return R * kvw * 2;
    if (comptime named(name, "r")) return R * qh * c.d_rel * 2;
    if (comptime named(name, "gate")) return R * qw * 2;
    if (comptime named(name, "sink")) return R * qh * 2;
    if (comptime named(name, "dyn")) return R * c.local.kv * 4 * 2;
    if (comptime (named(name, "kc") or named(name, "vc"))) return R * kvw * 2;
    if (comptime named(name, "qn")) return R * qw * 2;
    if (comptime named(name, "reln")) return R * qh * c.d_rel * 2;
    if (comptime named(name, "pb")) return if (per_op) R * qh * c.window * 2 else 0;
    if (comptime named(name, "att")) return R * qw * 2;
    if (comptime (named(name, "o") or named(name, "ya"))) return R * H * 2;
    if (comptime (named(name, "res") or named(name, "ns") or named(name, "mi"))) return R * H * 4;
    if (comptime named(name, "xm")) return R * H * 2;
    if (comptime named(name, "logits32")) return R * c.experts * 4;
    if (comptime (named(name, "eids") or named(name, "ew"))) return picks * 4;
    if (comptime (named(name, "texp") or named(name, "tslot") or named(name, "ttok"))) return tiles * 4;
    if (comptime named(name, "gu")) return gen_slots * 2 * I * 2;
    if (comptime named(name, "act")) return gen_slots * I * 2;
    if (comptime named(name, "ey")) return gen_slots * H * 2;
    if (comptime (named(name, "moe") or named(name, "ym"))) return R * H * 2;
    if (comptime named(name, "exl3")) return ex;
    if (comptime named(name, "fin")) return R * H * 4;
    if (comptime named(name, "finals")) return (R + 7) * H * 2;
    if (comptime named(name, "logits")) return batch_rows * @as(usize, c.vocab) * 2;
    if (comptime named(name, "segs")) return @as(usize, c.layers) * max_mvs_slabs * seg_slot;
    if (comptime named(name, "part")) return partBytes(c, capacity);
    if (comptime named(name, "planes")) return win_slots * 2 * H * 4; // the decode chain's w13 output per slot as two f32 K halves
    if (comptime named(name, "sums")) return win_slots * H * 4; // its w2 output per slot before the output rotation, f32
    if (comptime named(name, "ftaps")) return batch_rows * feature_taps * H * 2; // a shared decode round's rows of the drafter's taps
    if (comptime named(name, "ptaps")) return R * feature_taps * H * 2; // a prompt chunk's
    if (comptime named(name, "stage")) return if (per_op) 0 else @max(experts_prefill.xpBytes(R), @max(attention_prefill.tableBytes(R), R * max_queries * H * 4));
    @compileError("a scratch buffer without a size: " ++ name);
}

/// Bytes of each Scratch buffer, in `fields` order, for a load's `ch` and caches of up to `capacity` positions.
fn sizes(c: *const cfg.Text, capacity: usize, ch: Chunking) ![fields.len]usize {
    @setEvalBranchQuota(20000);
    if (!validPrefillRows(ch.rows)) return error.InvalidInkling2PrefillRows;
    const H: usize = c.hidden;
    const I: usize = c.expert_inter;
    const tiles = tileRows(genericRows(ch) * c.topk, c.experts);
    const ex = @max(try exl3.scratchBytes(H, 2 * I, tiles), try exl3.scratchBytes(I, H, tiles));
    var out: [fields.len]usize = undefined;
    inline for (fields, 0..) |f, i| out[i] = bytesOf(f, c, capacity, ch, ex);
    return out;
}

/// Bytes of a Scratch for caches of up to `capacity` positions and a load's `ch`; capacity 0 counts what does not grow with the
/// context.
pub fn scratchBytes(c: *const cfg.Text, capacity: usize, ch: Chunking) !usize {
    var total: usize = 0;
    for (try sizes(c, capacity, ch)) |n| total += n;
    return total;
}

pub const Scratch = struct {
    ids: mtl.Buffer,
    h: mtl.Buffer,
    nr: mtl.Buffer,
    blocks: mtl.Buffer,
    ai: mtl.Buffer,
    xq: mtl.Buffer,
    xk: mtl.Buffer,
    xv: mtl.Buffer,
    q: mtl.Buffer,
    k: mtl.Buffer,
    v: mtl.Buffer,
    r: mtl.Buffer,
    gate: mtl.Buffer,
    sink: mtl.Buffer,
    dyn: mtl.Buffer,
    kc: mtl.Buffer,
    vc: mtl.Buffer,
    qn: mtl.Buffer,
    reln: mtl.Buffer,
    pb: mtl.Buffer,
    att: mtl.Buffer,
    o: mtl.Buffer,
    ya: mtl.Buffer,
    res: mtl.Buffer,
    ns: mtl.Buffer,
    mi: mtl.Buffer,
    xm: mtl.Buffer,
    logits32: mtl.Buffer,
    eids: mtl.Buffer,
    ew: mtl.Buffer,
    texp: mtl.Buffer,
    tslot: mtl.Buffer,
    ttok: mtl.Buffer,
    gu: mtl.Buffer,
    act: mtl.Buffer,
    ey: mtl.Buffer,
    moe: mtl.Buffer,
    ym: mtl.Buffer,
    exl3: mtl.Buffer,
    fin: mtl.Buffer,
    finals: mtl.Buffer,
    logits: mtl.Buffer,
    segs: mtl.Buffer,
    part: mtl.Buffer,
    planes: mtl.Buffer,
    sums: mtl.Buffer,
    /// The drafter's taps, [row][feature_taps][hidden] bf16: a decode window's rows and a prompt chunk's.
    ftaps: mtl.Buffer,
    ptaps: mtl.Buffer,
    /// A prompt layer's relative-bias table, attention-input depth mix and experts' tables and planes on the tensor units, one
    /// stage at a time (see `fields`); empty on the per-operation path.
    stage: mtl.Buffer,
    /// Positions of the longest cache this scratch serves: the split attention's partials hold that many keys' blocks.
    capacity: usize,
    /// Rows of a prompt chunk at most: every per-row buffer holds that many.
    prefill_rows: usize,
    /// The path the buffers were made for: a forward's chunks take the path the model does.
    path: PromptPath,
    /// The GPU seconds a decode step of each row count is expected to take (0: none yet), for the next one's wait: a
    /// shorter step replaces it at once, a longer one moves it a quarter of the way.
    expected: [batch_rows + 1]f64,

    pub fn init(device: mtl.Device, c: *const cfg.Text, capacity: usize, ch: Chunking) !Scratch {
        @setEvalBranchQuota(10000);
        const sz = try sizes(c, capacity, ch);
        var s: Scratch = undefined;
        s.capacity = capacity;
        s.prefill_rows = ch.rows;
        s.path = ch.path;
        s.expected = @splat(0);
        var made: [fields.len]bool = @splat(false);
        errdefer inline for (fields, 0..) |f, i| if (made[i]) @field(s, f).deinit();
        inline for (fields, 0..) |f, i| {
            @field(s, f) = try device.buffer(@max(sz[i], 16), mtl.ResourceOptions.shared);
            made[i] = true;
        }
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (fields) |f| @field(s, f).deinit();
    }

    /// The experts' chain's tables and planes over the front of `stage`, for a chunk of up to `prefill_rows` rows.
    pub fn xp(s: *const Scratch) experts_prefill.Xp {
        return experts_prefill.Xp.init(s.stage, 0, s.prefill_rows);
    }
};

/// Inkling 2 Small's text config, as the size checks need it.
pub const small = cfg.Text{ .hidden = 4608, .layers = 48, .vocab = 201024, .unpadded_vocab = 200058, .local = .{ .q = 48, .kv = 8, .dim = 128 }, .global = .{ .q = 48, .kv = 8, .dim = 128 }, .window = 1024, .d_rel = 16, .rel_extent = 1024, .log_floor = 8192, .log_alpha = 0.1, .eps = 1e-6, .qk_clip = 22.627, .rel_clip = 8, .block = 6, .experts = 384, .topk = 8, .expert_inter = 2304, .route_scale = 8, .gate_cap = 4, .up_cap = 25, .logit_cap = 30, .max_context = 1048576 };

test "inkling2 state: tiles pad each expert to 8 rows; a cache's bytes; the split attention's partials" {
    try std.testing.expectEqual(@as(usize, 64), tileRows(8, 384));
    try std.testing.expectEqual(@as(usize, 4096 + 7 * 384), tileRows(4096, 384));
    const c = small;
    try std.testing.expectEqual(@as(usize, 8), maxBlocks(&c));
    // 100 positions are held as 128, whole blocks of 64
    try std.testing.expectEqual(@as(usize, 48 * (128 * 4096 + 32 * 11264 * 2) + 201024 * 2), cacheBytes(&c, 100, 48, default_prefill_rows));
    // 16 rows x 48 heads x one slot a 256-key block: 32k positions are 128 blocks, 51 MB
    try std.testing.expectEqual(@as(usize, 51_118_080), partBytes(&c, 32768));
    try std.testing.expectEqual(partBytes(&c, 256), partBytes(&c, 1));
    try std.testing.expectEqual(2 * partBytes(&c, 256), partBytes(&c, 257));
    try std.testing.expectEqual(@as(usize, 0), partBytes(&c, 0));
}

test "inkling2 state: a cache holds whole blocks of 64 positions, so a chunk's last key block never reaches past its layer" {
    for ([_]usize{ 1, 63, 64 }) |capacity| try std.testing.expectEqual(@as(usize, 64), cachePositions(capacity));
    for ([_]usize{ 65, 100, 128 }) |capacity| try std.testing.expectEqual(@as(usize, 128), cachePositions(capacity));
    try std.testing.expectEqual(@as(usize, 8256), cachePositions(8193));
    try std.testing.expectEqual(cacheBytes(&small, 65, 48, default_prefill_rows), cacheBytes(&small, 128, 48, default_prefill_rows));
    try std.testing.expect(cacheBytes(&small, 64, 48, default_prefill_rows) < cacheBytes(&small, 65, 48, default_prefill_rows));
    // every context's last position lies inside the positions held
    for ([_]usize{ 1, 63, 64, 65, 4095, 4096, 8193 }) |capacity| try std.testing.expect(cachePositions(capacity) >= capacity and cachePositions(capacity) % 64 == 0);
}

test "inkling2 state: a sliding layer holds its window, a chunk and a block more; a global layer every position" {
    try std.testing.expectEqual(@as(usize, 5376), slidingRing(&small, 4096));
    try std.testing.expectEqual(@as(usize, 1792), slidingRing(&small, 512));
    try std.testing.expectEqual(@as(usize, 1280), slidingRing(&small, 64));
    try std.testing.expectEqual(@as(usize, 9472), slidingRing(&small, 8192));
    // every chunk size a load may take: a chunk never overwrites a key its own first tile reads
    var rows: usize = prefill_unit;
    while (rows <= max_prefill_rows) : (rows += prefill_unit) {
        const r = slidingRing(&small, rows);
        try std.testing.expect(r % key_block == 0 and r >= @as(usize, small.window) + rows + 63);
    }
    // layer 5 is global, layer 0 sliding
    try std.testing.expectEqual(@as(usize, 131072), layerSlots(&small, 5, 131072, 4096));
    try std.testing.expectEqual(@as(usize, 5376), layerSlots(&small, 0, 131072, 4096));
    // a stream shorter than the ring holds its positions, as before
    try std.testing.expectEqual(@as(usize, 2048), layerSlots(&small, 0, 2000, 4096));
    // 8 global layers at 4 KiB a position, 40 rings, the convolution rings and the kept logits
    const want = 8 * 131072 * 4096 + 40 * 5376 * 4096 + 48 * convBytes(&small) + 201024 * 2;
    try std.testing.expectEqual(@as(usize, want), cacheBytes(&small, 131072, 48, 4096));
    try std.testing.expect(cacheBytes(&small, 131072, 48, 4096) < 5 << 30);
}

test "inkling2 state: the drafter's taps are eight hidden rows a row, a shared decode batch's and a prompt chunk's" {
    const row: usize = feature_taps * 4608 * 2;
    const bytes = try sizes(&small, 256, .{ .rows = default_prefill_rows });
    inline for (fields, 0..) |f, i| {
        if (comptime std.mem.eql(u8, f, "ftaps")) try std.testing.expectEqual(batch_rows * row, bytes[i]);
        if (comptime std.mem.eql(u8, f, "ptaps")) try std.testing.expectEqual(default_prefill_rows * row, bytes[i]);
    }
    try std.testing.expectEqual(@as(usize, 8), feature_taps);
}

test "inkling2 state: batch scratch holds 64 decode rows while keeping split partials at 16" {
    const c = small;
    try std.testing.expectEqual(@as(usize, 64), batch_rows);
    try std.testing.expectEqual(@as(usize, 8), max_streams);
    // xq must hold 3 * batch_rows even at minimum prefill_rows = 64
    const sz64 = try sizes(&c, 4096, .{ .rows = 64, .path = .chunk });
    inline for (fields, 0..) |f, i| {
        if (comptime named(f, "xq")) try std.testing.expect(sz64[i] >= 3 * batch_rows * @as(usize, c.hidden) * 2);
        if (comptime named(f, "segs")) try std.testing.expectEqual(@as(usize, c.layers) * max_mvs_slabs * seg_slot, sz64[i]);
    }
    // part stays sized to window_rows (16), not batch_rows (64)
    try std.testing.expectEqual(window_rows * @as(usize, c.local.q) * ((4096 + key_block - 1) / key_block) * part_slot * 4, partBytes(&c, 4096));
}

test "inkling2 state: a prompt chunk's rows are a multiple of 64 from 64 to 8192, and 4096 (512 per-operation) unless the setting names others" {
    for ([_]usize{ 64, 128, 512, 4096, 8192 }) |rows| try std.testing.expect(validPrefillRows(rows));
    for ([_]usize{ 0, 1, 63, 65, 100, 8192 + 64, 1 << 20 }) |rows| try std.testing.expect(!validPrefillRows(rows));
    try std.testing.expectEqual(@as(u32, 4096), defaultPrefillRows(.chunk));
    try std.testing.expectEqual(@as(u32, 4096), defaultPrefillRows(.chunk_rows));
    try std.testing.expectEqual(@as(u32, 512), defaultPrefillRows(.per_op));
    try std.testing.expectEqual(@as(?u32, null), try parsePrefillRows(null));
    try std.testing.expectEqual(@as(?u32, null), try parsePrefillRows(" \t"));
    try std.testing.expectEqual(@as(?u32, 2048), try parsePrefillRows("2048"));
    try std.testing.expectEqual(@as(?u32, 64), try parsePrefillRows(" 64 "));
    for ([_][]const u8{ "0", "63", "100", "8256", "-64", "lots", "2048k", "1e3", "0x40" }) |bad| try std.testing.expectError(error.BadPrefillRows, parsePrefillRows(bad));
}

test "inkling2 state: INKLING_PROMPT_PATH names the per-operation path, or the layers whose experts take the per-row projections" {
    try std.testing.expectEqual(PathRequest{}, try parsePromptPath(null));
    try std.testing.expectEqual(PathRequest{}, try parsePromptPath("  "));
    try std.testing.expectEqual(PathRequest{}, try parsePromptPath("auto"));
    try std.testing.expectEqual(PathRequest{ .per_op = true }, try parsePromptPath(" per_op "));
    try std.testing.expectEqual(PathRequest{ .rows = ~@as(u128, 0) }, try parsePromptPath("rows"));
    try std.testing.expectEqual(PathRequest{ .rows = 1 }, try parsePromptPath("rows:0"));
    try std.testing.expectEqual(PathRequest{ .rows = (1 << 3) | (1 << 7) }, try parsePromptPath("rows:3, 7"));
    try std.testing.expectEqual(PathRequest{ .rows = @as(u128, 1) << 127 }, try parsePromptPath("rows:127"));
    for ([_][]const u8{ "chunk", "rows:", "rows:x", "rows:128", "rows:-1", "rows:3,,7", "per_op:1", "ROWS" }) |bad| try std.testing.expectError(error.BadPromptPath, parsePromptPath(bad));
}

test "inkling2 state: the scratch's per-row buffers follow the prompt chunk's rows, the decode path's and the context's do not" {
    @setEvalBranchQuota(100000);
    const row: usize = feature_taps * 4608 * 2;
    var decode_only: [fields.len]usize = undefined;
    for ([_]usize{ 64, 512, 2048, 8192 }, 0..) |rows, round| {
        const bytes = try sizes(&small, 256, .{ .rows = rows });
        inline for (fields, 0..) |f, i| {
            if (comptime named(f, "ids")) try std.testing.expectEqual(rows * 4, bytes[i]);
            if (comptime named(f, "h")) try std.testing.expectEqual(rows * 4608 * 2, bytes[i]);
            if (comptime named(f, "blocks")) try std.testing.expectEqual(maxBlocks(&small) * rows * 4608 * 4, bytes[i]);
            if (comptime named(f, "ptaps")) try std.testing.expectEqual(rows * row, bytes[i]);
            if (comptime named(f, "ftaps")) try std.testing.expectEqual(batch_rows * row, bytes[i]);
            if (comptime named(f, "part")) try std.testing.expectEqual(partBytes(&small, 256), bytes[i]);
            // a chunk's picks follow its rows
            if (comptime (named(f, "eids") or named(f, "ew"))) try std.testing.expectEqual(rows * 8 * 4, bytes[i]);
            // the per-row expert path's buffers are a shared batch's on the chunk path, whatever the chunk
            if (comptime (named(f, "gu") or named(f, "act") or named(f, "ey") or named(f, "texp") or named(f, "tslot") or named(f, "ttok") or named(f, "exl3"))) {
                if (round == 0) decode_only[i] = bytes[i] else try std.testing.expectEqual(decode_only[i], bytes[i]);
            }
        }
    }
    // the default chunk's scratch, with the experts' chain, the attention's table and the depth mix sharing `stage`
    const b4096 = try scratchBytes(&small, 0, .{ .rows = default_prefill_rows });
    const b512 = try scratchBytes(&small, 0, .{ .rows = 512 });
    const b2048 = try scratchBytes(&small, 0, .{ .rows = 2048 });
    try std.testing.expect(b4096 > 2_868_139_264 and b4096 < 3_000_000_000);
    try std.testing.expect(b512 > 398_834_944 and b512 < 520_000_000);
    try std.testing.expect(b2048 > b512);
    for ([_]usize{ 0, 63, 100, 8192 + 64 }) |rows| {
        try std.testing.expectError(error.InvalidInkling2PrefillRows, sizes(&small, 256, .{ .rows = rows }));
        try std.testing.expectError(error.InvalidInkling2PrefillRows, scratchBytes(&small, 0, .{ .rows = rows }));
    }
}

test "inkling2 state: the per-row expert path's buffers hold a chunk's rows where a path sends the chunk through it, and the per-operation path keeps its own table" {
    @setEvalBranchQuota(100000);
    for ([_]usize{ 64, 512, 4096 }) |rows| {
        for ([_]PromptPath{ .chunk, .chunk_rows, .per_op }) |path| {
            const bytes = try sizes(&small, 0, .{ .rows = rows, .path = path });
            const gen: usize = if (path == .chunk) batch_rows else rows;
            inline for (fields, 0..) |f, i| {
                if (comptime named(f, "gu")) try std.testing.expectEqual(gen * 8 * 2 * 2304 * 2, bytes[i]);
                if (comptime named(f, "act")) try std.testing.expectEqual(gen * 8 * 2304 * 2, bytes[i]);
                if (comptime named(f, "ey")) try std.testing.expectEqual(gen * 8 * 4608 * 2, bytes[i]);
                if (comptime (named(f, "texp") or named(f, "tslot") or named(f, "ttok"))) try std.testing.expectEqual(tileRows(gen * 8, 384) * 4, bytes[i]);
                // the per-operation path's depth mix and relative-bias table, and no stage; the others the reverse
                if (comptime named(f, "ai")) try std.testing.expectEqual(if (path == .per_op) rows * 3 * 4608 * 4 else 0, bytes[i]);
                if (comptime named(f, "pb")) try std.testing.expectEqual(if (path == .per_op) rows * 48 * 1024 * 2 else 0, bytes[i]);
                if (comptime named(f, "stage")) try std.testing.expectEqual(if (path == .per_op) 0 else experts_prefill.xpBytes(rows), bytes[i]);
            }
        }
    }
    // per_op and chunk_rows grow by the shared decode scratch delta (+47 MiB)
    try std.testing.expect(try scratchBytes(&small, 0, .{ .rows = 512, .path = .per_op }) > 614_881_792);
    try std.testing.expect(try scratchBytes(&small, 0, .{ .rows = 4096, .path = .per_op }) > 4_295_234_048);
    try std.testing.expect(try scratchBytes(&small, 0, .{ .rows = 512, .path = .chunk_rows }) > 649_579_264);
    try std.testing.expect(try scratchBytes(&small, 0, .{ .rows = 4096, .path = .chunk_rows }) > 4_572_553_984);
}

test "inkling2 state: the stage holds the experts' chain, a whole chunk's relative-bias table and the depth mix, one after the other" {
    @setEvalBranchQuota(100000);
    for ([_]usize{ 64, 512, 2048, 4096, 8192 }) |rows| {
        const bytes = try sizes(&small, 0, .{ .rows = rows });
        inline for (fields, 0..) |f, i| if (comptime named(f, "stage")) {
            try std.testing.expect(bytes[i] >= experts_prefill.xpBytes(rows));
            try std.testing.expect(bytes[i] >= attention_prefill.tableBytes(rows));
            try std.testing.expect(bytes[i] >= rows * 3 * 4608 * 4);
            // the chain is the largest of the three, so the stage costs the chain's bytes: 906 MB at 4,096 rows
            try std.testing.expectEqual(experts_prefill.xpBytes(rows), bytes[i]);
        };
    }
    try std.testing.expectEqual(@as(usize, 906_465_536), experts_prefill.xpBytes(4096));
}

test "inkling2 state: a scratch made for some rows holds buffers for them and says how many and for what path" {
    const device = mtl.Device.init() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer device.deinit();
    var few = try Scratch.init(device, &small, 256, .{ .rows = 64 });
    defer few.deinit();
    var more = try Scratch.init(device, &small, 256, .{ .rows = 128 });
    defer more.deinit();
    try std.testing.expectEqual(@as(usize, 64), few.prefill_rows);
    try std.testing.expectEqual(@as(usize, 128), more.prefill_rows);
    try std.testing.expectEqual(@as(usize, 64 * feature_taps * 4608 * 2), few.ptaps.length());
    try std.testing.expectEqual(@as(usize, 128 * feature_taps * 4608 * 2), more.ptaps.length());
    try std.testing.expectError(error.InvalidInkling2PrefillRows, Scratch.init(device, &small, 256, .{ .rows = 100 }));
    // the stage holds the experts' chain, whose views start at its front and fit
    try std.testing.expectEqual(experts_prefill.xpBytes(64), few.stage.length());
    try std.testing.expectEqual(experts_prefill.xpBytes(128), more.stage.length());
    const views = more.xp();
    try std.testing.expectEqual(@as(usize, 128), views.rows);
    try std.testing.expect(views.counts.buf.id == more.stage.id and views.counts.off == 0);
    // the per-operation path has the depth mix and the table of its own and no stage; the chunk path has the stage alone
    var per_op = try Scratch.init(device, &small, 256, .{ .rows = 64, .path = .per_op });
    defer per_op.deinit();
    try std.testing.expectEqual(PromptPath.per_op, per_op.path);
    try std.testing.expectEqual(PromptPath.chunk, few.path);
    try std.testing.expectEqual(@as(usize, 64 * 3 * 4608 * 4), per_op.ai.length());
    try std.testing.expectEqual(@as(usize, 64 * 48 * 1024 * 2), per_op.pb.length());
    try std.testing.expectEqual(@as(usize, 16), per_op.stage.length());
    try std.testing.expectEqual(@as(usize, 16), few.ai.length());
    try std.testing.expectEqual(@as(usize, 16), few.pb.length());
}

test "inkling2 state: a cache's layers hold whole blocks of 64 positions, the values a half after the keys" {
    const device = mtl.Device.init() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer device.deinit();
    var cache = try Cache.init(device, &small, 100, 2, default_prefill_rows);
    defer cache.deinit();
    try std.testing.expectEqual(@as(usize, 100), cache.capacity);
    try std.testing.expectEqual(@as(u32, 128), cache.ring(0));
    try std.testing.expectEqual(2 * 128 * kvBytes(&small), cache.kv.length());
    try std.testing.expectEqual(@as(usize, cache.slots[0]) * kvBytes(&small), cache.keys(1) - cache.keys(0));
    try std.testing.expectEqual(@as(usize, cache.slots[1]) * cache.row, cache.values(1) - cache.keys(1));
    // the last position of the context is inside the keys, and the block it ends in is whole
    try std.testing.expect(cache.values(0) >= cachePositions(cache.capacity) * kvBytes(&small) / 2);
}
