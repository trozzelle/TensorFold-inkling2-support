//! The DSpark draft head's tensors (mtp.safetensors: 84 bf16 tensors under unprefixed names) as views into the
//! checkpoint's buffers, and the mix weights derived from fuse_weight. The draft layers' nine projections are either those views
//! (bf16) or, by default, quantized at load to MLX's affine 8-bit form (dspark_quant.zig): they then stay in the file, are read
//! through a host buffer one at a time, and only their codes, scales and biases reach the GPU.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const config = @import("config.zig");
const wts = @import("weights.zig");
const mix = @import("dspark_alpha.zig");
const quant = @import("dspark_quant.zig");
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;

pub const Tensor = wts.Tensor;
pub const Dense = wts.Dense;

/// The draft head's file in a model folder (tfpack links it from the source checkpoint).
pub const file_name = "mtp.safetensors";

/// The width of the draft layers' projection weights: oMLX's affine 8-bit form (the default), or the file's bf16.
pub const Bits = enum(u8) {
    eight = 8,
    sixteen = 16,

    /// What an unset or blank INKLING_DRAFT_BITS means.
    pub const default: Bits = .eight;

    /// The setting's text: null or blank is the default, else `8` or `16`.
    pub fn parse(text: ?[]const u8) error{BadDraftBits}!Bits {
        const t = std.mem.trim(u8, text orelse "", " \t");
        if (t.len == 0) return default;
        if (std.mem.eql(u8, t, "8")) return .eight;
        if (std.mem.eql(u8, t, "16")) return .sixteen;
        return error.BadDraftBits;
    }

    /// How a log names the layers' weights.
    pub fn label(b: Bits) []const u8 {
        return switch (b) {
            .eight => "8-bit",
            .sixteen => "bf16",
        };
    }
};

/// Bytes the parts of a quantized projection start at, so every kernel reads its codes and tables aligned.
const part_alignment = 256;

/// A projection [n, k] quantized to MLX's affine 8-bit form, in one buffer: codes u8 [n][k], then bf16 scales [n][k / 64], then
/// bf16 biases [n][k / 64], each part at a multiple of `part_alignment` bytes (the codes at 0).
pub const Quant = struct {
    buf: mtl.Buffer,
    scales: usize, // byte offsets
    biases: usize,
    n: u32,
    k: u32,

    /// Bytes of the buffer for an [n, k] projection.
    pub fn bytes(n: usize, k: usize) usize {
        return layout(n, k)[2];
    }

    /// The scale part's offset, the bias part's, and the buffer's size.
    fn layout(n: usize, k: usize) [3]usize {
        const table = n * (k / quant.group) * 2;
        const scales = std.mem.alignForward(usize, n * k, part_alignment);
        const biases = scales + std.mem.alignForward(usize, table, part_alignment);
        return .{ scales, biases, biases + table };
    }
};

/// One projection of a draft layer, as the drafter's kernels take it: bf16 views into the checkpoint, or quantized.
pub const Proj = union(enum) {
    bf16: Dense,
    q8: Quant,

    pub fn n(p: Proj) u32 {
        return switch (p) {
            inline else => |x| x.n,
        };
    }

    pub fn k(p: Proj) u32 {
        return switch (p) {
            inline else => |x| x.k,
        };
    }
};

/// One draft layer. Norms are [hidden] (`dflare` the tap mix's) or [head_dim] (`q_norm`, `k_norm` for block rows,
/// `k_norm_ctx` for context rows); `k` and `v` project block rows, `k_ctx` and `v_ctx` context rows.
pub const Layer = struct {
    dflare: Tensor,
    input: Tensor,
    post: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    k_norm_ctx: Tensor,
    q: Proj, // [q_heads * head_dim, hidden]
    k: Proj, // [kv_heads * head_dim, hidden]
    v: Proj,
    k_ctx: Proj,
    v_ctx: Proj,
    o: Proj, // [hidden, q_heads * head_dim]
    gate: Proj, // [inter, hidden]
    up: Proj,
    down: Proj, // [hidden, inter]
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
    /// The width of the layers' projections: the file's bf16, or quantized at load.
    bits: Bits,
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
    /// Bytes the draft file takes among the checkpoint's shards, the layers' quantized buffers among them (the fold's are
    /// counted apart: `foldBytes`).
    bytes: usize = 0,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        w.alpha_buffer.deinit();
        gpa.free(w.layers);
    }
};

/// The tensors a quantizing load leaves in the file: the draft layers' projection weights (rank 2 under `layers.`; the norms
/// are rank 1).
fn isProjection(name: []const u8, rank: usize) bool {
    return rank == 2 and std.mem.startsWith(u8, name, "layers.");
}

/// One slice of rows `quantize` hands a core.
const Job = struct {
    src: []const u16,
    k: usize,
    codes: []u8,
    scales: []u16,
    biases: []u16,
    bad: ?usize = null, // the first group quantizeRows refused

    fn run(j: *Job) void {
        j.bad = quant.quantizeRows(j.src, j.k, j.codes, j.scales, j.biases);
    }
};

/// MLX's affine 8-bit quantization (dspark_quant.zig) of the bf16 values `src`, [n][k] row-major with k a multiple of 64, into a
/// new buffer: the rows go to every core. The caller owns the buffer (the checkpoint, through `own`).
pub fn quantize(gpa: std.mem.Allocator, device: mtl.Device, src: []const u16, n: usize, k: usize) !Quant {
    if (k == 0 or k % quant.group != 0 or src.len != n * k or n == 0 or n > std.math.maxInt(u32) or k > std.math.maxInt(u32)) return error.UnexpectedInkling2Tensor;
    const at = Quant.layout(n, k);
    const groups = k / quant.group;
    const buf = try device.buffer(at[2], mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer buf.deinit();
    const mem = buf.contents();
    const codes = mem[0 .. n * k];
    const scales = @as([*]u16, @ptrCast(@alignCast(mem + at[0])))[0 .. n * groups];
    const biases = @as([*]u16, @ptrCast(@alignCast(mem + at[1])))[0 .. n * groups];
    // one slice of rows a core; a small projection stays on this thread
    const cpus = std.Thread.getCpuCount() catch 4;
    const parts: usize = @max(1, @min(@min(cpus, 32), n * k / (1 << 20), n));
    const jobs = try gpa.alloc(Job, parts);
    defer gpa.free(jobs);
    const threads = try gpa.alloc(?std.Thread, parts);
    defer gpa.free(threads);
    const per = (n + parts - 1) / parts;
    for (jobs, threads, 0..) |*j, *t, i| {
        const r0 = @min(i * per, n);
        const r1 = @min(r0 + per, n);
        j.* = .{ .src = src[r0 * k .. r1 * k], .k = k, .codes = codes[r0 * k .. r1 * k], .scales = scales[r0 * groups .. r1 * groups], .biases = biases[r0 * groups .. r1 * groups] };
        t.* = if (i + 1 == parts) null else std.Thread.spawn(.{}, Job.run, .{j}) catch null;
        if (t.* == null) j.run();
    }
    for (threads) |t| if (t) |th| th.join();
    for (jobs, 0..) |j, i| if (j.bad) |g| {
        std.log.warn("inkling2: a draft weight is not finite (row {d}, group {d})", .{ i * per + g / groups, g % groups });
        return error.NonFiniteDraftWeight;
    };
    return .{ .buf = buf, .scales = at[0], .biases = at[1], .n = @intCast(n), .k = @intCast(k) };
}

/// Where a layer's projections come from: the checkpoint's bf16 views, or each one read from the file into `scratch`, quantized,
/// and its buffer handed to the checkpoint.
const Projections = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    c: *ck.Checkpoint,
    bits: Bits,
    scratch: []u16 = &.{}, // a deferred projection's bf16 values as read, room for the largest so far

    fn deinit(p: *Projections) void {
        p.gpa.free(p.scratch);
    }

    /// The [n, k] projection `name`.
    fn at(p: *Projections, name: []const u8, n: usize, k: usize) !Proj {
        if (p.bits == .sixteen) return .{ .bf16 = try wts.dense(p.c, name, n, k) };
        const d = p.c.deferred.get(name) orelse {
            std.log.err("inkling2: {s} is not a tensor of {s}", .{ name, file_name });
            return error.UnexpectedInkling2Tensor;
        };
        if (d.dtype != .bf16 or d.rank != 2 or d.shape[0] != n or d.shape[1] != k or d.bytes != n * k * 2) {
            std.log.err("inkling2: {s} is not bf16[{d}, {d}]", .{ name, n, k });
            return error.UnexpectedInkling2Tensor;
        }
        if (p.scratch.len < n * k) {
            p.gpa.free(p.scratch);
            p.scratch = &.{};
            p.scratch = try p.gpa.alloc(u16, n * k);
        }
        const src = p.scratch[0 .. n * k];
        try p.c.read(name, 0, std.mem.sliceAsBytes(src));
        const q = try quantize(p.gpa, p.device, src, n, k);
        p.c.own(q.buf, Quant.bytes(n, k)) catch |err| {
            q.buf.deinit();
            return err;
        };
        return .{ .q8 = q };
    }
};

fn layerTensors(c: *const ck.Checkpoint, p: *Projections, m: config.Mtp, i: usize) !Layer {
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
        .q = try p.at(try N.at(&b, i, "self_attn.q_proj.weight"), qw, h),
        .k = try p.at(try N.at(&b, i, "self_attn.k_proj.weight"), kvw, h),
        .v = try p.at(try N.at(&b, i, "self_attn.v_proj.weight"), kvw, h),
        .k_ctx = try p.at(try N.at(&b, i, "self_attn.k_proj_ctx.weight"), kvw, h),
        .v_ctx = try p.at(try N.at(&b, i, "self_attn.v_proj_ctx.weight"), kvw, h),
        .o = try p.at(try N.at(&b, i, "self_attn.o_proj.weight"), h, qw),
        .gate = try p.at(try N.at(&b, i, "mlp.gate_proj.weight"), inter, h),
        .up = try p.at(try N.at(&b, i, "mlp.up_proj.weight"), inter, h),
        .down = try p.at(try N.at(&b, i, "mlp.down_proj.weight"), h, inter),
    };
}

/// The draft head in `dir`'s mtp.safetensors, added to `c` (names unprefixed, so the draft's `layers.N.` and `norm.weight`
/// stay apart from the backbone's `model.llm.` ones). Null when the folder has no such file (the draft stays off).
/// Every tensor is checked for its name, bf16 and shape against `m`; the mix rows are fuse_weight's. With `bits` eight, the
/// layers' nine projections each stay in the file until they are read, quantized (quantize) and given to `c` as buffers of their
/// own, so the bf16 values are never resident; the file's other tensors are one shard, as at bf16.
pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, c: *ck.Checkpoint, dir: []const u8, m: config.Mtp, bits: Bits) !?Weights {
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
    const before = c.residentBytes();
    if (bits == .eight) try c.addFileSplit(device, path, "", null, isProjection) else try c.addFile(device, path, "");
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
    var projections: Projections = .{ .gpa = gpa, .device = device, .c = c, .bits = bits };
    defer projections.deinit();
    for (layers, 0..) |*l, i| l.* = try layerTensors(c, &projections, m, i);
    if (bits == .eight) c.closeFiles();
    var w: Weights = .{
        .layers = layers,
        .bits = bits,
        .alpha = std.mem.zeroes([mix.max_rows][mix.max_taps]u16),
        .alpha_buffer = try device.buffer(rows * taps * @sizeOf(u16), mtl.ResourceOptions.shared),
        .rows = @intCast(rows),
        .taps = @intCast(taps),
        .mask_embed = mask_embed,
        .norm = norm,
        .markov = markov,
        .bytes = c.residentBytes() - before,
    };
    mix.softmaxRows(fuse.host(u16), rows, taps, &w.alpha);
    const packed_alpha = w.alpha_buffer.slice(u16, rows * taps);
    for (0..rows) |r| @memcpy(packed_alpha[r * taps ..][0..taps], w.alpha[r][0..taps]);
    return w;
}

/// Bytes of the fold's two tensors: fold_w [vocab, rank] and fold_b [vocab], bf16.
pub fn foldBytes(vocab: usize, rank: usize) usize {
    return vocab * rank * 2 + vocab * 2;
}

/// The fold, once, on the GPU: the unembedding applied to hidden_out, so a Markov step's latent reaches the vocabulary as
/// `bf16(bf16(latent . fold_w[v]) + fold_b[v])` instead of through hidden_out and then the unembedding. Each fold value is
/// the f32 sum over the hidden width rounded to bf16 once. The unembedding's bf16 matvec takes hidden_out's `rank` columns
/// and its bias as `rank + 1` input rows, and `transpose` (ikd_fold_t) lays its output [rank + 1][vocab] out by token. The
/// two tensors live in one buffer the checkpoint owns, named `markov_head.fold_w` and `markov_head.fold_b`; a head already
/// folded is left as it is.
pub fn fold(w: *Weights, c: *ck.Checkpoint, device: mtl.Device, queue: mtl.Queue, model: *const Model, transpose: mtl.Pipeline) !void {
    if (w.fold_w != null) return;
    const unembed = model.weights.unembed;
    const hidden: usize = unembed.k;
    const vocab: usize = unembed.n;
    const rank: usize = w.markov.hidden_out.k;
    if (w.markov.hidden_out.n != hidden or w.markov.w1.n != vocab or w.markov.w1.k != rank) return error.UnexpectedInkling2Tensor;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const rows = rank + 1;
    const input = try device.buffer(rows * hidden * 2, mtl.ResourceOptions.shared);
    defer input.deinit();
    // input row `col` is hidden_out's column `col` ([hidden, rank] on disk); the last row is its bias
    const x = input.slice(u16, rows * hidden);
    const weight = w.markov.hidden_out.t.host(u16);
    for (0..rank) |col| for (0..hidden) |h| {
        x[col * hidden + h] = weight[h * rank + col];
    };
    @memcpy(x[rank * hidden ..][0..hidden], w.markov.hidden_out_b.host(u16));
    const y = try device.buffer(rows * vocab * 2, mtl.ResourceOptions.shared);
    defer y.deinit();
    const w_bytes = vocab * rank * 2;
    const total = foldBytes(vocab, rank);
    const out = try device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer out.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    fwd.matvec(e, model, unembed, .{ .buf = input }, .{ .buf = y }, rows);
    e.setPipeline(transpose);
    e.setValue(extern struct { vocab: u32 }{ .vocab = @intCast(vocab) }, 0);
    e.setBuffer(y, 0, 1);
    e.setBuffer(out, 0, 2);
    e.setBuffer(out, w_bytes, 3);
    e.dispatchGroups(mtl.Size.of((vocab + 31) / 32, 1, 1), mtl.Size.of(256, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2 draft fold: {s}", .{msg});
        return error.GpuFailed;
    }
    var fw: Tensor = .{ .buffer = out, .offset = 0, .bytes = w_bytes, .dtype = .bf16, .rank = 2 };
    fw.shape[0] = vocab;
    fw.shape[1] = rank;
    var fb: Tensor = .{ .buffer = out, .offset = w_bytes, .bytes = vocab * 2, .dtype = .bf16, .rank = 1 };
    fb.shape[0] = vocab;
    try c.set("markov_head.fold_w", fw);
    try c.set("markov_head.fold_b", fb);
    try c.own(out, total);
    w.fold_w = .{ .t = fw, .n = @intCast(vocab), .k = @intCast(rank) };
    w.fold_b = fb;
}
