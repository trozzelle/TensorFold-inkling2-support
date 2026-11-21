//! Inkling 2's dense tensors (model-dense.safetensors: the source's bytes) and its EXL3 experts (experts/), as views
//! into Metal buffers; each layer's experts as one descriptor table per projection.
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
    w13: exl3.Experts,
    w2: exl3.Experts,
    /// Whether every expert of w13 and w2 can take the decode chain (`xdOk`); otherwise the decode path keeps
    /// `exl3.encodeIndexed` for the experts.
    xd_ok: bool,
};

pub const Weights = struct {
    embed: Dense,
    unembed: Dense,
    embed_norm: Tensor,
    norm: Tensor,
    out_query: Tensor,
    layers: []Layer,
    tables: []mtl.Buffer, // a layer's descriptors: w13's then w2's
    expert_buffers: []mtl.Buffer, // the expert files' buffers, used each forward (descriptors read them indirectly)
    widths: [5]u32 = @splat(0), // expert projections by bits

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.tables) |t| t.deinit();
        gpa.free(w.tables);
        gpa.free(w.layers);
        gpa.free(w.expert_buffers);
    }
};

fn tensor(c: *const ck.Checkpoint, name: []const u8, dtype: ck.DType, dims: []const usize) !Tensor {
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

fn dense(c: *const ck.Checkpoint, name: []const u8, n: usize, k: usize) !Dense {
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

/// The dense file and the first `layers` layers' expert files into `c`, then every view the forward reads.
pub fn load(gpa: std.mem.Allocator, device: mtl.Device, c: *ck.Checkpoint, t: *const config.Text, p: *const plan.Plan, ex: *const pack.Pack, dir: []const u8, layers: u32) !Weights {
    const path = try std.fs.path.joinZ(gpa, &.{ dir, "model-dense.safetensors" });
    defer gpa.free(path);
    try c.addFile(device, path, "");
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
        const hd = p.at(i).heads;
        const qw: usize = @as(usize, hd.q) * hd.dim;
        const kvw: usize = @as(usize, hd.kv) * hd.dim;
        const global = p.at(i).kind == .global;
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
        l.rel_proj = try tensor(c, try N.n(&nb, pre, "attn.rel_logits_proj.proj"), .bf16, &.{ t.d_rel, p.at(i).extent });
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
        // the experts: each file under its own prefix, each expert's two projections into this layer's table
        const table = try expertTable(gpa, device, c, dir, ex.layers[li], i, H, I, E, &w.widths);
        w.tables[li] = table;
        tables_made += 1;
        l.w13 = .{ .descriptors = .{ .buffer = table }, .count = E, .k = H, .n = 2 * I };
        l.w2 = .{ .descriptors = .{ .buffer = table, .offset = E * @sizeOf(exl3.ExpertDescriptor) }, .count = E, .k = I, .n = H };
        l.xd_ok = xdOk(table.slice(exl3.ExpertDescriptor, 2 * E), E, H, I);
    }
    const bufs = try gpa.alloc(mtl.Buffer, c.shards.items.len - first);
    for (c.shards.items[first..], bufs) |sh, *dst| dst.* = sh.buffer;
    w.expert_buffers = bufs;
    return w;
}
