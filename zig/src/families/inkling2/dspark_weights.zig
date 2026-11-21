//! The DSpark draft head's tensors (mtp.safetensors: 84 bf16 tensors under unprefixed names) as views into the
//! checkpoint's buffers, and the mix weights derived from fuse_weight.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const config = @import("config.zig");
const wts = @import("weights.zig");
const mix = @import("dspark_alpha.zig");

pub const Tensor = wts.Tensor;
pub const Dense = wts.Dense;

/// The draft head's file in a model folder (tfpack links it from the source checkpoint).
pub const file_name = "mtp.safetensors";

/// Most draft layers (each has a mix row, and the Markov head one more) and most taps a mix row spans.
/// One draft layer. Norms are [hidden] (`dflare` the tap mix's) or [head_dim] (`q_norm`, `k_norm` for block rows,
/// `k_norm_ctx` for context rows); `k` and `v` project block rows, `k_ctx` and `v_ctx` context rows.
pub const Layer = struct {
    dflare: Tensor,
    input: Tensor,
    post: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    k_norm_ctx: Tensor,
    q: Dense, // [q_heads * head_dim, hidden]
    k: Dense, // [kv_heads * head_dim, hidden]
    v: Dense,
    k_ctx: Dense,
    v_ctx: Dense,
    o: Dense, // [hidden, q_heads * head_dim]
    gate: Dense, // [inter, hidden]
    up: Dense,
    down: Dense, // [hidden, inter]
};

/// The Markov head: a GRU over [state, w1 row of the previous token, draft hidden, f_rnn] whose latent is folded into
/// vocabulary logits through `hidden_out` and the unembedding.
pub const Markov = struct {
    dflare: Tensor, // [hidden]
    w1: Dense, // [vocab, rank]: a row per token
    rnn: Dense, // [3 * rank, 2 * rank + 2 * hidden]: update gate, candidate and latent
    rnn_b: Tensor, // [3 * rank]
    hidden_out: Dense, // [hidden, rank]
    hidden_out_b: Tensor, // [hidden]
};

pub const Weights = struct {
    layers: []Layer,
    /// softmax(fuse_weight, -1) rounded to bf16 (bits), row `i` for draft layer `i` and row `layers.len` for the
    /// Markov head, over the first `taps` columns.
    alpha: [mix.max_rows][mix.max_taps]u16,
    /// The same bits, [rows][taps] packed, in a buffer for the kernels.
    alpha_buffer: mtl.Buffer,
    rows: u32,
    taps: u32,
    mask_embed: Tensor, // [hidden]
    norm: Tensor, // [hidden]
    markov: Markov,
    /// unembed . hidden_out: fold_w [vocab, rank] and fold_b [vocab] (bf16), filled by the fold pass after load.
    fold_w: ?Dense = null,
    fold_b: ?Tensor = null,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        w.alpha_buffer.deinit();
        gpa.free(w.layers);
    }
};

fn layerTensors(c: *const ck.Checkpoint, m: config.Mtp, i: usize) !Layer {
    const h: usize = m.hidden;
    const d: usize = m.head_dim;
    const qw: usize = @as(usize, m.q_heads) * d;
    const kvw: usize = @as(usize, m.kv_heads) * d;
    const inter: usize = m.inter;
    var b: [96]u8 = undefined;
    const N = struct {
        fn at(buf: []u8, k: usize, comptime suffix: []const u8) ![]const u8 {
            return std.fmt.bufPrint(buf, "layers.{d}." ++ suffix, .{k});
        }
    };
    return .{
        .dflare = try wts.tensor(c, try N.at(&b, i, "dflare_norm.weight"), .bf16, &.{h}),
        .input = try wts.tensor(c, try N.at(&b, i, "input_layernorm.weight"), .bf16, &.{h}),
        .post = try wts.tensor(c, try N.at(&b, i, "post_attention_layernorm.weight"), .bf16, &.{h}),
        .q_norm = try wts.tensor(c, try N.at(&b, i, "self_attn.q_norm.weight"), .bf16, &.{d}),
        .k_norm = try wts.tensor(c, try N.at(&b, i, "self_attn.k_norm.weight"), .bf16, &.{d}),
        .k_norm_ctx = try wts.tensor(c, try N.at(&b, i, "self_attn.k_norm_ctx.weight"), .bf16, &.{d}),
        .q = try wts.dense(c, try N.at(&b, i, "self_attn.q_proj.weight"), qw, h),
        .k = try wts.dense(c, try N.at(&b, i, "self_attn.k_proj.weight"), kvw, h),
        .v = try wts.dense(c, try N.at(&b, i, "self_attn.v_proj.weight"), kvw, h),
        .k_ctx = try wts.dense(c, try N.at(&b, i, "self_attn.k_proj_ctx.weight"), kvw, h),
        .v_ctx = try wts.dense(c, try N.at(&b, i, "self_attn.v_proj_ctx.weight"), kvw, h),
        .o = try wts.dense(c, try N.at(&b, i, "self_attn.o_proj.weight"), h, qw),
        .gate = try wts.dense(c, try N.at(&b, i, "mlp.gate_proj.weight"), inter, h),
        .up = try wts.dense(c, try N.at(&b, i, "mlp.up_proj.weight"), inter, h),
        .down = try wts.dense(c, try N.at(&b, i, "mlp.down_proj.weight"), h, inter),
    };
}

/// The draft head in `dir`'s mtp.safetensors, added to `c` (names unprefixed, so the draft's `layers.N.` and `norm.weight`
/// stay apart from the backbone's `model.llm.` ones). Null when the folder has no such file (the draft stays off).
/// Every tensor is checked for its name, bf16 and shape against `m`; the mix rows are fuse_weight's.
pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, c: *ck.Checkpoint, dir: []const u8, m: config.Mtp) !?Weights {
    const path = try std.fs.path.joinZ(gpa, &.{ dir, file_name });
    defer gpa.free(path);
    std.Io.Dir.cwd().access(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => {
            std.log.info("inkling2: no {s} in the model folder; the draft head stays off", .{file_name});
            return null;
        },
        else => return err,
    };
    const rows: usize = m.layers + 1;
    const taps: usize = m.taps_n;
    const h: usize = m.hidden;
    const rank: usize = m.rank;
    try c.addFile(device, path, "");
    const fuse = try wts.tensor(c, "fuse_weight", .bf16, &.{ rows, taps });
    const markov: Markov = .{
        .dflare = try wts.tensor(c, "markov_head.dflare_norm.weight", .bf16, &.{h}),
        .w1 = try wts.dense(c, "markov_head.markov_w1.weight", m.vocab, rank),
        .rnn = try wts.dense(c, "markov_head.rnn_proj.weight", 3 * rank, 2 * rank + 2 * h),
        .rnn_b = try wts.tensor(c, "markov_head.rnn_proj.bias", .bf16, &.{3 * rank}),
        .hidden_out = try wts.dense(c, "markov_head.hidden_out.weight", h, rank),
        .hidden_out_b = try wts.tensor(c, "markov_head.hidden_out.bias", .bf16, &.{h}),
    };
    const mask_embed = try wts.tensor(c, "mask_embed", .bf16, &.{h});
    const norm = try wts.tensor(c, "norm.weight", .bf16, &.{h});
    const layers = try gpa.alloc(Layer, m.layers);
    errdefer gpa.free(layers);
    for (layers, 0..) |*l, i| l.* = try layerTensors(c, m, i);
    var w: Weights = .{
        .layers = layers,
        .alpha = std.mem.zeroes([mix.max_rows][mix.max_taps]u16),
        .alpha_buffer = try device.buffer(rows * taps * @sizeOf(u16), mtl.ResourceOptions.shared),
        .rows = @intCast(rows),
        .taps = @intCast(taps),
        .mask_embed = mask_embed,
        .norm = norm,
        .markov = markov,
    };
    mix.softmaxRows(fuse.host(u16), rows, taps, &w.alpha);
    const packed_alpha = w.alpha_buffer.slice(u16, rows * taps);
    for (0..rows) |r| @memcpy(packed_alpha[r * taps ..][0..taps], w.alpha[r][0..taps]);
    return w;
}

