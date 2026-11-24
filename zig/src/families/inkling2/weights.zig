//! Inkling 2's dense tensors and routed experts (EXL3 packs or native HF NVFP4/MXFP8 checkpoints) as views into Metal
//! buffers, plus looped-core controllers and cross-pass queries.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const exl3 = @import("../../core/exl3_projection.zig");
const config = @import("config.zig");
const plan = @import("plan.zig");
const pack = @import("expert_pack.zig");

pub const Tensor = ck.Tensor;
/// The checkpoint a weight load fills (for callers that build single-layer tables).
pub const Checkpoint = ck.Checkpoint;
/// A bf16 [n, k] matrix in torch's Linear layout, as core/bf16_matvec.metal reads it.
pub const Dense = struct { t: Tensor, n: u32, k: u32 };
/// A short convolution's taps: static [C, 1, 4] bf16, or the global layers' static [4, C] plus their projection.
pub const Conv = union(enum) { static: Tensor, dynamic: struct { taps: Tensor, proj: Dense } };

/// One ModelOpt NVFP4 routed-expert projection: packed E2M1 nibbles, E4M3 group-16 scales, and precomputed F32 global scales.
pub const Nvfp4Proj = struct {
    weight: Tensor,
    scale: Tensor,
    global_scale: mtl.Buffer,
    n: u32,
    k: u32,
};

/// One MXFP8 routed-expert projection: E4M3 weights and E8M0 group-32 exponent scales.
pub const Mxfp8Proj = struct {
    weight: Tensor,
    scale: Tensor,
    n: u32,
    k: u32,
};

/// One layer's routed experts in their stored format.
pub const Experts = union(enum) {
    exl3: struct {
        w13: exl3.Experts,
        w2: exl3.Experts,
        /// Whether every expert of w13 and w2 can take the decode chain and the prompt chunk's chain (`xdOk`: K2 or K3 experts, in the
        /// layout both chains address); otherwise the layer's experts take `exl3.encodeIndexed` in a decode window and in a prompt chunk.
        xd_ok: bool,
    },
    nvfp4: struct {
        w13: Nvfp4Proj,
        w2: Nvfp4Proj,
    },
    mxfp8: struct {
        w13: Mxfp8Proj,
        w2: Mxfp8Proj,
    },
};

/// One low-rank MessageGate controller (attn, k, v or mlp) on a looped-core layer.
pub const Controller = struct {
    down: Dense,
    forget: Dense,
    gain: Dense,
};

pub const Layer = struct {
    attn_query: ?Tensor, // f32 [H] (local) or [3, H] (global); none for layer 0
    mlp_query: Tensor, // f32 [H]
    in_norm: [3]?Tensor, // attn_norm (local), or the q, k and v input norms (global)
    q: Dense,
    k: Dense,
    v: Dense,
    r: Dense,
    o: Dense,
    gate: Dense,
    sink: Dense,
    q_scale: Tensor,
    rel_scale: Tensor,
    rel_proj: Tensor,
    k_conv: Conv,
    v_conv: Conv,
    attn_conv: Tensor,
    mlp_conv: Tensor,
    mlp_norm: Tensor,
    router: Dense,
    bias: Tensor,
    experts: Experts,
    controllers: ?[4]Controller = null,
};

pub const Weights = struct {
    embed: Dense,
    unembed: Dense,
    embed_norm: Tensor,
    norm: Tensor,
    out_query: Tensor,
    message_queries: ?mtl.Buffer = null, // f32 [core_len, 4, H] interleaved (q, k, v, mlp) cross-pass queries
    layers: []Layer,
    tables: []mtl.Buffer, // a layer's EXL3 descriptors: w13's then w2's
    expert_buffers: []mtl.Buffer, // the EXL3 expert files' buffers, used each forward (descriptors read them indirectly)
    global_scales: []mtl.Buffer = &.{}, // precomputed f32 [E] NVFP4 global scales (two per NVFP4 layer)
    widths: [5]u32 = @splat(0), // EXL3 expert projections by bits

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        if (w.message_queries) |b| b.deinit();
        for (w.global_scales) |b| b.deinit();
        gpa.free(w.global_scales);
        for (w.tables) |t| t.deinit();
        gpa.free(w.tables);
        gpa.free(w.layers);
        gpa.free(w.expert_buffers);
    }
};

pub fn tensor(c: *const ck.Checkpoint, name: []const u8, dtype: ck.DType, dims: []const usize) !Tensor {
    const x = try c.get(name);
    var ok = x.dtype == dtype and x.rank == dims.len;
    if (ok) for (dims, x.shape[0..x.rank]) |want, got| {
        if (want != got) ok = false;
    };
    if (!ok) {
        std.log.err("inkling2: {s} is not {s}{any}", .{ name, @tagName(dtype), dims });
        return error.UnexpectedInkling2Tensor;
    }
    return x;
}

pub fn dense(c: *const ck.Checkpoint, name: []const u8, n: usize, k: usize) !Dense {
    return .{ .t = try tensor(c, name, .bf16, &.{ n, k }), .n = @intCast(n), .k = @intCast(k) };
}

fn fmt(buf: []u8, comptime f: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, f, args);
}

/// One expert projection: trellis int16 [k/16, n/16, 16 K] (K 2 to 4), suh f16 [k], svh f16 [n], the mul1 marker.
fn matrix(c: *const ck.Checkpoint, base: []const u8, k: usize, n: usize, widths: *[5]u32) !exl3.Matrix {
    var b: [160]u8 = undefined;
    const tr = try c.get(try fmt(&b, "{s}.trellis", .{base}));
    if (tr.dtype != .i16 or tr.rank != 3 or tr.shape[0] != k / 16 or tr.shape[1] != n / 16 or tr.shape[2] % 16 != 0) {
        std.log.err("inkling2: {s}.trellis is not int16 [{d}, {d}, 16 K]", .{ base, k / 16, n / 16 });
        return error.UnexpectedInkling2Tensor;
    }
    const bits = tr.shape[2] / 16;
    if (bits < 2 or bits > 4) return error.UnsupportedInkling2Width;
    widths[bits] += 1;
    const suh = try tensor(c, try fmt(&b, "{s}.suh", .{base}), .f16, &.{k});
    const svh = try tensor(c, try fmt(&b, "{s}.svh", .{base}), .f16, &.{n});
    const mul = try c.get(try fmt(&b, "{s}.mul1", .{base}));
    if (mul.dtype != .i32 or mul.count() != 1 or @as(u32, @bitCast(mul.host(i32)[0])) != 0x83dcd12d) return error.NotAMul1Inkling2Pack;
    const m = exl3.Matrix{ .trellis = .{ .buffer = tr.buffer, .offset = tr.offset }, .suh = .{ .buffer = suh.buffer, .offset = suh.offset }, .svh = .{ .buffer = svh.buffer, .offset = svh.offset }, .k = k, .n = n, .width = .{ .halves = @intCast(bits * 2) }, .codebook = .mul1 };
    try m.validate();
    return m;
}

/// Layer `layer`'s experts from its pack files into `c`, as one table of 2 * E descriptors: w13's, then w2's.
pub fn expertTable(gpa: std.mem.Allocator, device: mtl.Device, c: *ck.Checkpoint, dir: []const u8, batches: []const pack.Batch, layer: u32, H: usize, I: usize, E: usize, widths: *[5]u32) !mtl.Buffer {
    const table = try device.buffer(2 * E * @sizeOf(exl3.ExpertDescriptor), mtl.ResourceOptions.shared);
    errdefer table.deinit();
    const desc = table.slice(exl3.ExpertDescriptor, 2 * E);
    for (batches) |batch| {
        const file = try std.fs.path.joinZ(gpa, &.{ dir, "experts", batch.file });
        defer gpa.free(file);
        var fb: [32]u8 = undefined;
        const fp = try fmt(&fb, "x{d}.{d}.", .{ layer, batch.start });
        try c.addFile(device, file, fp);
        for (batch.start..batch.end) |e| {
            var eb: [96]u8 = undefined;
            // kortexa files name experts by their index in the file; accept the layer index as well
            const local = try fmt(&eb, "{s}experts.{d}.w13.trellis", .{ fp, e - batch.start });
            const idx = if (c.tensors.contains(local)) e - batch.start else e;
            var mb: [96]u8 = undefined;
            desc[e] = try exl3.descriptor(try matrix(c, try fmt(&mb, "{s}experts.{d}.w13", .{ fp, idx }), H, 2 * I, widths));
            desc[E + e] = try exl3.descriptor(try matrix(c, try fmt(&mb, "{s}experts.{d}.w2", .{ fp, idx }), I, H, widths));
        }
    }
    return table;
}

/// Whether a layer's descriptor table (`count` w13 descriptors, then `count` w2 ones) can take the decode chain
/// (`ik_xd_*` in kernels/metal/inkling2/xd.metal, written for a hidden width of 4608 and an expert width of 2304): every
/// expert at 2 or 3 bits, the mul1 codebook, f16 scales, no bias, and tiles stored back to back at 4-byte alignment,
/// which is how the chain's kernels address them.
pub fn xdOk(table: []const exl3.ExpertDescriptor, count: usize, hidden: usize, inner: usize) bool {
    if (hidden != 4608 or inner != 2304 or table.len != 2 * count) return false;
    for (table, 0..) |d, i| {
        const n: u64 = if (i < count) 2 * inner else hidden;
        const tile: u64 = 16 * @as(u64, d.halves); // bytes: 8 * halves int16 words
        if (d.halves != 4 and d.halves != 6) return false;
        if (d.codebook != @backingInt(exl3.format.Codebook.mul1)) return false;
        if (d.suh_signs != 0 or d.svh_signs != 0 or d.has_bias != 0) return false;
        if (d.tile_n_stride != tile or d.tile_k_stride != n / 16 * tile or d.trellis % 4 != 0) return false;
    }
    return true;
}

/// The tensors of the layers and of their attention-residual queries (the file's names sort them between the embedding and
/// the unembedding, so they cannot be left out as a range).
fn perLayer(name: []const u8, rank: usize) bool {
    _ = rank;
    return std.mem.startsWith(u8, name, "model.llm.layers.") or std.mem.startsWith(u8, name, "model.llm.attn_res.attn_queries.") or
        std.mem.startsWith(u8, name, "model.llm.attn_res.mlp_queries.");
}

fn skipNonBackbone(name: []const u8, rank: usize) bool {
    _ = rank;
    return std.mem.startsWith(u8, name, "model.visual.");
}

fn skipNonBackboneOrLayer(name: []const u8, rank: usize) bool {
    return skipNonBackbone(name, rank) or perLayer(name, rank);
}

fn interleaveMessageQueries(qm: []const f32, kv: []const f32, core_len: usize, H: usize, out: []f32) void {
    for (0..core_len) |c| {
        const dst = c * 4 * H;
        @memcpy(out[dst + 0 * H ..][0..H], qm[(2 * c + 0) * H ..][0..H]); // q
        @memcpy(out[dst + 1 * H ..][0..H], kv[(2 * c + 0) * H ..][0..H]); // k
        @memcpy(out[dst + 2 * H ..][0..H], kv[(2 * c + 1) * H ..][0..H]); // v
        @memcpy(out[dst + 3 * H ..][0..H], qm[(2 * c + 1) * H ..][0..H]); // mlp
    }
}

fn computeGlobalScale(amax: []const f32, denom: f32, out: []f32) !void {
    if (!std.math.isFinite(denom) or denom <= 0) return error.UnexpectedInkling2Tensor;
    for (amax, out) |a, *dst| {
        if (!std.math.isFinite(a) or a < 0) return error.UnexpectedInkling2Tensor;
        dst.* = a / denom;
    }
}

/// Shared dense attention, norm, convolution, and router tensors for physical layer `i`.
fn loadLayerDense(c: *const ck.Checkpoint, t: *const config.Text, i: u32, l: *Layer) !void {
    const H: usize = t.hidden;
    const E: usize = t.experts;
    const hd = t.heads(i);
    const qw: usize = @as(usize, hd.q) * hd.dim;
    const kvw: usize = @as(usize, hd.kv) * hd.dim;
    const global = config.isGlobal(i);
    var b: [128]u8 = undefined;
    const pre = try fmt(&b, "model.llm.layers.{d}.", .{i});
    var nb: [192]u8 = undefined;
    const N = struct {
        fn n(buf: []u8, prefix: []const u8, s: []const u8) ![]const u8 {
            return fmt(buf, "{s}{s}", .{ prefix, s });
        }
    };
    l.attn_query = if (i == 0) null else try tensor(c, try fmt(&nb, "model.llm.attn_res.attn_queries.{d}", .{i - 1}), .f32, if (global) &.{ 3, H } else &.{H});
    l.mlp_query = try tensor(c, try fmt(&nb, "model.llm.attn_res.mlp_queries.{d}", .{i}), .f32, &.{H});
    l.in_norm = .{ null, null, null };
    if (global) {
        for ([_][]const u8{ "q", "k", "v" }, 0..) |x, j| l.in_norm[j] = try tensor(c, try fmt(&nb, "{s}{s}_input_norm.weight", .{ pre, x }), .bf16, &.{H});
    } else l.in_norm[0] = try tensor(c, try N.n(&nb, pre, "attn_norm.weight"), .bf16, &.{H});
    l.q = try dense(c, try N.n(&nb, pre, "attn.wq_du.weight"), qw, H);
    l.k = try dense(c, try N.n(&nb, pre, "attn.wk_dv.weight"), kvw, H);
    l.v = try dense(c, try N.n(&nb, pre, "attn.wv_dv.weight"), kvw, H);
    l.r = try dense(c, try N.n(&nb, pre, "attn.wr_du.weight"), @as(usize, hd.q) * t.d_rel, H);
    l.o = try dense(c, try N.n(&nb, pre, "attn.wo_ud.weight"), H, qw);
    l.gate = try dense(c, try N.n(&nb, pre, "attn.gate_attn.weight"), qw, H);
    l.sink = try dense(c, try N.n(&nb, pre, "attn.attn_sink_proj.weight"), hd.q, H);
    l.q_scale = try tensor(c, try N.n(&nb, pre, "attn.q_l2_clip.scale"), .bf16, &.{ hd.q, hd.dim });
    l.rel_scale = try tensor(c, try N.n(&nb, pre, "attn.rel_l2_clip.scale"), .bf16, &.{ hd.q, t.d_rel });
    l.rel_proj = try tensor(c, try N.n(&nb, pre, "attn.rel_logits_proj.proj"), .bf16, &.{ t.d_rel, t.extent(i) });
    inline for (.{ "k", "v" }, 0..) |x, j| {
        const cv: Conv = if (global) .{ .dynamic = .{
            .taps = try tensor(c, try N.n(&nb, pre, "attn." ++ x ++ "_sconv.static_w"), .bf16, &.{ config.conv_taps, kvw }),
            .proj = try dense(c, try N.n(&nb, pre, "attn." ++ x ++ "_sconv.weight_proj"), @as(usize, hd.kv) * config.conv_taps, H),
        } } else .{ .static = try tensor(c, try N.n(&nb, pre, "attn." ++ x ++ "_sconv.weight"), .bf16, &.{ kvw, 1, config.conv_taps }) };
        if (j == 0) l.k_conv = cv else l.v_conv = cv;
    }
    l.attn_conv = try tensor(c, try N.n(&nb, pre, "attn_sconv.weight"), .bf16, &.{ H, 1, config.conv_taps });
    l.mlp_conv = try tensor(c, try N.n(&nb, pre, "mlp_sconv.weight"), .bf16, &.{ H, 1, config.conv_taps });
    l.mlp_norm = try tensor(c, try N.n(&nb, pre, "mlp_norm.weight"), .bf16, &.{H});
    l.router = try dense(c, try N.n(&nb, pre, "mlp.gate.weight"), E, H);
    l.bias = try tensor(c, try N.n(&nb, pre, "mlp.gate.bias"), .f32, &.{E});
    l.controllers = null;
}

/// The dense file and the first `layers` layers' expert files into `c`, then every view the forward reads. With no layers the
/// file's per-layer tensors stay unread in it: the embedding, the norms and the unembedding are all that load.
pub fn load(gpa: std.mem.Allocator, device: mtl.Device, c: *ck.Checkpoint, t: *const config.Text, p: *const plan.Plan, ex: *const pack.Pack, dir: []const u8, layers: u32) !Weights {
    _ = p;
    const path = try std.fs.path.joinZ(gpa, &.{ dir, "model-dense.safetensors" });
    defer gpa.free(path);
    if (layers == 0) {
        try c.addFileSplit(device, path, "", null, perLayer);
        c.closeFiles();
    } else try c.addFile(device, path, "");
    const first = c.shards.items.len;
    const H: usize = t.hidden;
    const I: usize = t.expert_inter;
    const E: usize = t.experts;
    var w = Weights{
        .embed = try dense(c, "model.llm.embed.weight", t.vocab, H),
        .unembed = try dense(c, "model.llm.unembed.weight", t.vocab, H),
        .embed_norm = try tensor(c, "model.llm.embed_norm.weight", .bf16, &.{H}),
        .norm = try tensor(c, "model.llm.norm.weight", .bf16, &.{H}),
        .out_query = try tensor(c, "model.llm.attn_res.output_query", .f32, &.{H}),
        .layers = try gpa.alloc(Layer, layers),
        .tables = try gpa.alloc(mtl.Buffer, layers),
        .expert_buffers = &.{},
    };
    var tables_made: usize = 0;
    errdefer {
        for (w.tables[0..tables_made]) |x| x.deinit();
        gpa.free(w.tables);
        gpa.free(w.layers);
    }
    for (w.layers, 0..) |*l, li| {
        const i: u32 = @intCast(li);
        try loadLayerDense(c, t, i, l);
        // the experts: each file under its own prefix, each expert's two projections into this layer's table
        const table = try expertTable(gpa, device, c, dir, ex.layers[li], i, H, I, E, &w.widths);
        w.tables[li] = table;
        tables_made += 1;
        l.experts = .{ .exl3 = .{
            .w13 = .{ .descriptors = .{ .buffer = table }, .count = E, .k = H, .n = 2 * I },
            .w2 = .{ .descriptors = .{ .buffer = table, .offset = E * @sizeOf(exl3.ExpertDescriptor) }, .count = E, .k = I, .n = H },
            .xd_ok = xdOk(table.slice(exl3.ExpertDescriptor, 2 * E), E, H, I),
        } };
    }
    const bufs = try gpa.alloc(mtl.Buffer, c.shards.items.len - first);
    for (c.shards.items[first..], bufs) |sh, *dst| dst.* = sh.buffer;
    w.expert_buffers = bufs;
    return w;
}

fn loadNvfp4Proj(device: mtl.Device, c: *const ck.Checkpoint, layer: u32, comptime which: []const u8, n: usize, k: usize, e: usize) !Nvfp4Proj {
    var b: [160]u8 = undefined;
    const weight = try tensor(c, try fmt(&b, "model.llm.layers.{d}.mlp.experts." ++ which ++ "_weight", .{layer}), .u8, &.{ e, n, k / 2 });
    const scale = try tensor(c, try fmt(&b, "model.llm.layers.{d}.mlp.experts." ++ which ++ "_weight_scale", .{layer}), .f8_e4m3, &.{ e, n, k / 16 });
    const amax = try tensor(c, try fmt(&b, "model.llm.layers.{d}.mlp.experts." ++ which ++ "_weight_amax", .{layer}), .f32, &.{e});
    const denom = try tensor(c, try fmt(&b, "model.llm.layers.{d}.mlp.experts." ++ which ++ "_weight_outer_scale_denominator", .{layer}), .f32, &.{1});
    const gs = try device.buffer(e * @sizeOf(f32), mtl.ResourceOptions.shared);
    errdefer gs.deinit();
    try computeGlobalScale(amax.host(f32), denom.host(f32)[0], gs.slice(f32, e));
    return .{ .weight = weight, .scale = scale, .global_scale = gs, .n = @intCast(n), .k = @intCast(k) };
}

fn loadMxfp8Proj(c: *const ck.Checkpoint, layer: u32, comptime which: []const u8, n: usize, k: usize, e: usize) !Mxfp8Proj {
    var b: [160]u8 = undefined;
    const weight = try tensor(c, try fmt(&b, "model.llm.layers.{d}.mlp.experts." ++ which ++ "_weight", .{layer}), .f8_e4m3, &.{ e, n, k });
    const scale = try tensor(c, try fmt(&b, "model.llm.layers.{d}.mlp.experts." ++ which ++ "_weight_scale", .{layer}), .u8, &.{ e, n, k / 32 });
    return .{ .weight = weight, .scale = scale, .n = @intCast(n), .k = @intCast(k) };
}

fn lessThanStr(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

fn safeShardName(name: []const u8) bool {
    if (name.len == 0 or std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) return false;
    for (name) |ch| if (ch == '/' or ch == '\\' or ch == 0) return false;
    return true;
}

/// Discover safetensors shard filenames from `model.safetensors.index.json`, falling back to
/// `model-00001-of-00001.safetensors` or `model.safetensors`.
fn discoverShards(arena: std.mem.Allocator, io: std.Io, dir: []const u8) ![]const []const u8 {
    const index_path = try std.fs.path.join(arena, &.{ dir, "model.safetensors.index.json" });
    if (std.Io.Dir.cwd().readFileAlloc(io, index_path, arena, .limited(64 << 20))) |text| {
        const tree = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch return error.BadSafetensors;
        if (tree != .object) return error.BadSafetensors;
        const wm_val = tree.object.get("weight_map") orelse return error.BadSafetensors;
        if (wm_val != .object) return error.BadSafetensors;
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        var names: std.ArrayList([]const u8) = .empty;
        var it = wm_val.object.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.* != .string) return error.BadSafetensors;
            const fname = entry.value_ptr.string;
            if (!safeShardName(fname)) return error.BadSafetensors;
            const gop = try seen.getOrPut(arena, fname);
            if (!gop.found_existing) try names.append(arena, fname);
        }
        if (names.items.len == 0) return error.BadSafetensors;
        std.mem.sort([]const u8, names.items, {}, lessThanStr);
        return names.items;
    } else |err| switch (err) {
        error.FileNotFound => {
            const single = try std.fs.path.join(arena, &.{ dir, "model-00001-of-00001.safetensors" });
            const chosen: []const u8 = if (std.Io.Dir.cwd().access(io, single, .{})) |_|
                "model-00001-of-00001.safetensors"
            else |_|
                "model.safetensors";
            const out = try arena.alloc([]const u8, 1);
            out[0] = chosen;
            return out;
        },
        else => return err,
    }
}

/// Direct Hugging Face checkpoint loader for native NVFP4/MXFP8 Inkling 2 checkpoints, skipping vision tensors.
pub fn loadNative(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, c: *ck.Checkpoint, t: *const config.Text, prec_out: []plan.Precision, dir: []const u8, phys_layers: u32) !Weights {
    if (prec_out.len < t.layers) return error.PrecisionCount;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const shards = try discoverShards(arena.allocator(), io, dir);
    const filter: ck.Later = if (phys_layers == 0) skipNonBackboneOrLayer else skipNonBackbone;
    for (shards) |fname| {
        const shard_path = try std.fs.path.joinZ(arena.allocator(), &.{ dir, fname });
        try c.addFileSplit(device, shard_path, "", null, filter);
    }
    c.closeFiles();

    const H: usize = t.hidden;
    const I: usize = t.expert_inter;
    const E: usize = t.experts;
    var w = Weights{
        .embed = try dense(c, "model.llm.embed.weight", t.vocab, H),
        .unembed = try dense(c, "model.llm.unembed.weight", t.vocab, H),
        .embed_norm = try tensor(c, "model.llm.embed_norm.weight", .bf16, &.{H}),
        .norm = try tensor(c, "model.llm.norm.weight", .bf16, &.{H}),
        .out_query = try tensor(c, "model.llm.attn_res.output_query", .f32, &.{H}),
        .layers = try gpa.alloc(Layer, phys_layers),
        .tables = &.{},
        .expert_buffers = &.{},
    };
    var scales: std.ArrayList(mtl.Buffer) = .empty;
    errdefer {
        if (w.message_queries) |mq| mq.deinit();
        for (scales.items) |gs| gs.deinit();
        scales.deinit(gpa);
        gpa.free(w.layers);
    }

    if (t.loop) |lp| {
        const core_len: usize = lp.core_end - lp.core_start;
        const qm = try tensor(c, "model.llm.middle_cycle_cross_pass_queries", .f32, &.{ 2 * core_len, H });
        const kv = try tensor(c, "model.llm.middle_cycle_cross_pass_kv_queries", .f32, &.{ 2 * core_len, H });
        const mq = try device.buffer(core_len * 4 * H * @sizeOf(f32), mtl.ResourceOptions.shared);
        w.message_queries = mq;
        interleaveMessageQueries(qm.host(f32), kv.host(f32), core_len, H, mq.slice(f32, core_len * 4 * H));
    }

    for (w.layers, 0..) |*l, li| {
        const i: u32 = @intCast(li);
        try loadLayerDense(c, t, i, l);
        var nb: [160]u8 = undefined;
        if (t.loop) |lp| if (i >= lp.core_start and i < lp.core_end) {
            const rank: usize = lp.gate_rank;
            var ctrls: [4]Controller = undefined;
            inline for (.{ "attn", "k", "v", "mlp" }, 0..) |kind, ci| {
                ctrls[ci] = .{
                    .down = try dense(c, try fmt(&nb, "model.llm.layers.{d}.controllers." ++ kind ++ ".down_weight", .{i}), rank, 2 * H),
                    .forget = try dense(c, try fmt(&nb, "model.llm.layers.{d}.controllers." ++ kind ++ ".forget_weight", .{i}), H, rank),
                    .gain = try dense(c, try fmt(&nb, "model.llm.layers.{d}.controllers." ++ kind ++ ".gain_weight", .{i}), H, rank),
                };
            }
            l.controllers = ctrls;
        };
        const w13_raw = try c.get(try fmt(&nb, "model.llm.layers.{d}.mlp.experts.w13_weight", .{i}));
        switch (w13_raw.dtype) {
            .u8 => {
                prec_out[i] = .nvfp4;
                const w13 = try loadNvfp4Proj(device, c, i, "w13", 2 * I, H, E);
                scales.append(gpa, w13.global_scale) catch |err| {
                    w13.global_scale.deinit();
                    return err;
                };
                const w2 = try loadNvfp4Proj(device, c, i, "w2", H, I, E);
                scales.append(gpa, w2.global_scale) catch |err| {
                    w2.global_scale.deinit();
                    return err;
                };
                l.experts = .{ .nvfp4 = .{ .w13 = w13, .w2 = w2 } };
            },
            .f8_e4m3 => {
                prec_out[i] = .mxfp8;
                const w13 = try loadMxfp8Proj(c, i, "w13", 2 * I, H, E);
                const w2 = try loadMxfp8Proj(c, i, "w2", H, I, E);
                l.experts = .{ .mxfp8 = .{ .w13 = w13, .w2 = w2 } };
            },
            else => return error.ExpertFormat,
        }
    }
    for (phys_layers..t.layers) |i| prec_out[i] = .nvfp4;
    w.global_scales = try scales.toOwnedSlice(gpa);
    return w;
}

test "inkling2 weights: skipNonBackbone skips vision tensors and keeps backbone tensors" {
    try std.testing.expect(skipNonBackbone("model.visual.backbone.layers.0.ln_1.weight", 1));
    try std.testing.expect(skipNonBackbone("model.visual.proj_out.weight", 2));
    try std.testing.expect(!skipNonBackbone("model.llm.embed.weight", 2));
    try std.testing.expect(!skipNonBackbone("model.llm.layers.6.controllers.attn.down_weight", 2));
    try std.testing.expect(!skipNonBackbone("model.llm.middle_cycle_cross_pass_queries", 2));
}

test "inkling2 weights: interleaveMessageQueries orders (q, k, v, mlp) per core layer" {
    const qm = [_]f32{ 10, 11, 20, 21, 30, 31, 40, 41 }; // 2 core layers x [q, mlp] x H=2
    const kv = [_]f32{ 12, 13, 14, 15, 32, 33, 34, 35 }; // 2 core layers x [k, v] x H=2
    var out: [16]f32 = undefined;
    interleaveMessageQueries(&qm, &kv, 2, 2, &out);
    try std.testing.expectEqualSlices(f32, &.{
        10, 11, 12, 13, 14, 15, 20, 21,
        30, 31, 32, 33, 34, 35, 40, 41,
    }, &out);
}

test "inkling2 weights: computeGlobalScale divides amax by denominator and refuses bad denominators" {
    const amax = [_]f32{ 2688.0, 1344.0 };
    var out: [2]f32 = undefined;
    try computeGlobalScale(&amax, 2688.0, &out);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 0.5 }, &out);
    try std.testing.expectError(error.UnexpectedInkling2Tensor, computeGlobalScale(&amax, 0.0, &out));
    try std.testing.expectError(error.UnexpectedInkling2Tensor, computeGlobalScale(&amax, std.math.nan(f32), &out));
}
