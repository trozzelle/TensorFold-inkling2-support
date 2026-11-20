//! A stream's caches (each layer's keys and values by position, the convolutions' input rings, the last logits) and a
//! forward's scratch. Every layer keeps its whole context: sliding layers mask rather than evict.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const exl3 = @import("../../core/exl3_projection.zig");

/// Rows of a verify window at most (the pending token and its copy drafts).
pub const window_rows = 16;
/// Rows of a prompt chunk at most; every kernel keeps rows independent, so chunks change no bits.
pub const prompt_rows = 512;
/// Positions each convolution's input ring holds: three taps back plus a window's rows, so a rejected window rolls
/// back by moving the cache length alone.
pub const ring = 32;

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

/// Bytes a stream of `capacity` positions holds over `layers` layers.
pub fn cacheBytes(c: *const cfg.Text, capacity: usize, layers: usize) usize {
    return layers * (capacity * kvBytes(c) + convBytes(c)) + @as(usize, c.vocab) * 2;
}

pub const Cache = struct {
    kv: mtl.Buffer,
    conv: mtl.Buffer,
    logits: mtl.Buffer, // the last row's bf16 logits, the next first draw's
    capacity: usize,
    layers: usize,
    len: usize = 0,
    kv_layer: usize,
    conv_layer: usize,
    widths: [4]usize,
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn init(device: mtl.Device, c: *const cfg.Text, capacity: usize, layers: usize) !Cache {
        var out = Cache{ .kv = undefined, .conv = undefined, .logits = undefined, .capacity = capacity, .layers = layers, .kv_layer = capacity * kvBytes(c), .conv_layer = convBytes(c), .widths = convWidths(c) };
        out.kv = try device.buffer(@max(out.kv_layer * layers, 16), mtl.ResourceOptions.private);
        errdefer out.kv.deinit();
        out.conv = try device.buffer(out.conv_layer * layers, mtl.ResourceOptions.private);
        errdefer out.conv.deinit();
        out.logits = try device.buffer(@as(usize, c.vocab) * 2, mtl.ResourceOptions.shared);
        return out;
    }

    /// Byte offsets in `kv`: layer l's keys [capacity, kv, dim], then its values.
    pub fn keys(self: *const Cache, l: usize) usize {
        return l * self.kv_layer;
    }
    pub fn values(self: *const Cache, l: usize) usize {
        return l * self.kv_layer + self.kv_layer / 2;
    }

    /// Byte offset in `conv` of layer l's ring `which` (0 k, 1 v, 2 attention output, 3 MLP output): [ring, width].
    pub fn ringAt(self: *const Cache, l: usize, which: usize) usize {
        var at = l * self.conv_layer;
        for (self.widths[0..which]) |w| at += ring * w * 2;
        return at;
    }

    pub fn setResident(self: *Cache, hook: ?mtl.ResidencySet.Follow) void {
        if (self.follow) |f| mtl.resident.fields.apply(self, f, false);
        self.follow = hook;
        if (hook) |f| mtl.resident.fields.apply(self, f, true);
    }

    pub fn deinit(self: *Cache) void {
        self.setResident(null);
        self.kv.deinit();
        self.conv.deinit();
        self.logits.deinit();
    }
};

pub const fields = [_][]const u8{ "ids", "h", "nr", "blocks", "ai", "xq", "xk", "xv", "q", "k", "v", "r", "gate", "sink", "dyn", "kc", "vc", "qn", "reln", "pb", "att", "o", "ya", "res", "ns", "mi", "xm", "logits32", "eids", "ew", "texp", "tslot", "ttok", "gu", "act", "ey", "moe", "ym", "exl3", "fin", "finals", "logits" };

/// Bytes of each Scratch buffer, in `fields` order, for prompt_rows rows.
fn sizes(c: *const cfg.Text) ![fields.len]usize {
    const R: usize = prompt_rows;
    const H: usize = c.hidden;
    const qh: usize = c.local.q;
    const qw: usize = qh * c.local.dim;
    const kvw = kvWidth(c);
    const S: usize = R * c.topk;
    const T = tileRows(S, c.experts);
    const I: usize = c.expert_inter;
    const ex = @max(try exl3.scratchBytes(H, 2 * I, T), try exl3.scratchBytes(I, H, T));
    return .{
        R * 4,                R * H * 2,                 R * H * 4,              maxBlocks(c) * R * H * 4, R * 3 * H * 4,        R * H * 2,
        R * H * 2,            R * H * 2,                 R * qw * 2,             R * kvw * 2,              R * kvw * 2,          R * qh * c.d_rel * 2,
        R * qw * 2,           R * qh * 2,                R * c.local.kv * 4 * 2, R * kvw * 2,              R * kvw * 2,          R * qw * 2,
        R * qh * c.d_rel * 2, R * qh * c.window * 2,     R * qw * 2,             R * H * 2,                R * H * 2,            R * H * 4,
        R * H * 4,            R * H * 4,                 R * H * 2,              R * c.experts * 4,        S * 4,                S * 4,
        T * 4,                T * 4,                     T * 4,                  S * 2 * I * 2,            S * I * 2,            S * H * 2,
        R * H * 2,            R * H * 2,                 ex,                     R * H * 4,                R * H * 2,            window_rows * @as(usize, c.vocab) * 2,
    };
}

pub fn scratchBytes(c: *const cfg.Text) !usize {
    var total: usize = 0;
    for (try sizes(c)) |n| total += n;
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

    pub fn init(device: mtl.Device, c: *const cfg.Text) !Scratch {
        @setEvalBranchQuota(10000);
        const sz = try sizes(c);
        var s: Scratch = undefined;
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
};

test "inkling2 state: tiles pad each expert to 8 rows; a cache's bytes" {
    try std.testing.expectEqual(@as(usize, 64), tileRows(8, 384));
    try std.testing.expectEqual(@as(usize, 4096 + 7 * 384), tileRows(4096, 384));
    const c = cfg.Text{ .hidden = 4608, .layers = 48, .vocab = 201024, .unpadded_vocab = 200058, .local = .{ .q = 48, .kv = 8, .dim = 128 }, .global = .{ .q = 48, .kv = 8, .dim = 128 }, .window = 1024, .d_rel = 16, .rel_extent = 1024, .log_floor = 8192, .log_alpha = 0.1, .eps = 1e-6, .qk_clip = 22.627, .rel_clip = 8, .block = 6, .experts = 384, .topk = 8, .expert_inter = 2304, .route_scale = 8, .gate_cap = 4, .up_cap = 25, .logit_cap = 30, .max_context = 1048576 };
    try std.testing.expectEqual(@as(usize, 8), maxBlocks(&c));
    try std.testing.expectEqual(@as(usize, 48 * (100 * 4096 + 32 * 11264 * 2) + 201024 * 2), cacheBytes(&c, 100, 48));
}
