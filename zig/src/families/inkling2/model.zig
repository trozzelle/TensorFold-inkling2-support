//! A native Inkling 2 model (EXL3 or native NVFP4/MXFP8 experts): config, plan, checkpoint buffers, and pipelines.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const ck = @import("../../core/checkpoint_metal.zig");
const exl3 = @import("../../core/exl3_projection.zig");
const cfg = @import("config.zig");
const plan = @import("plan.zig");
const pack = @import("expert_pack.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const dspark_weights = @import("dspark_weights.zig");
const dense_prefill = @import("dense_prefill.zig");
const attention_prefill = @import("attention_prefill.zig");
const experts_prefill = @import("experts_prefill.zig");
const experts_fp_prefill = @import("experts_fp_prefill.zig");

pub const glue_names = [_][:0]const u8{ "ik_embed", "ik_rms", "ik_norm", "ik_depth", "ik_resid", "ik_out", "ik_copy", "ik_softcap", "ik_conv", "ik_conv_dyn", "ik_ring", "ik_prep", "ik_relbias", "ik_attn", "ik_router", "ik_route", "ik_tiles", "ik_swiglu", "ik_combine", "ik_matvec_1", "ik_matvec_2", "ik_matvec_4", "ik_matvec_8", "ik_matvec_16", "ik_matvec_32", "ik_mvs_8", "ik_mvs_16", "ik_rdn", "ik_rdn3", "ik_rrdn", "ik_attn_part", "ik_attn_part_w", "ik_attn_merge", "ik_xd_pair", "ik_xd_mid", "ik_xd_down", "ik_xd_finish", "ik_conv_ring", "ik_conv_dyn_ring", "ik_conv_out", "ik_tap", "ik_nop", "ik_copy_f32", "ik_msg_depth", "ik_msggate_prep", "ik_msggate_apply_norm", "ik_msggate_prep3", "ik_msggate_apply_norm3", "ik_attn_part_g3", "ik_attn_part_w_g3", "ik_nvfp4_w13_swiglu", "ik_nvfp4_w2", "ik_mxfp8_w13_swiglu", "ik_mxfp8_w2" };
pub const Glue = enum { embed, rms, norm, depth, resid, out, copy, softcap, conv, conv_dyn, ring, prep, relbias, attn, router, route, tiles, swiglu, combine, matvec_1, matvec_2, matvec_4, matvec_8, matvec_16, matvec_32, mvs_8, mvs_16, rdn, rdn3, rrdn, attn_part, attn_part_w, attn_merge, xd_pair, xd_mid, xd_down, xd_finish, conv_ring, conv_dyn_ring, conv_out, tap, nop, copy_f32, msg_depth, msggate_prep, msggate_apply_norm, msggate_prep3, msggate_apply_norm3, attn_part_g3, attn_part_w_g3, nvfp4_w13_swiglu, nvfp4_w2, mxfp8_w13_swiglu, mxfp8_w2 };

comptime {
    // `pipe` indexes the pipelines by the enum's order: each member is its kernel's name without the ik_ prefix
    const members = @typeInfo(Glue).@"enum".field_names;
    std.debug.assert(members.len == glue_names.len);
    for (members, glue_names) |member, name| std.debug.assert(std.mem.eql(u8, "ik_" ++ member, name));
}

/// `layers`: load only the first layers (the per-layer checks on a Mac that cannot hold the model); 0 loads none, and no
/// experts: the embedding, the final norm and the unembedding alone, enough to run the draft head over recorded taps.
/// `drafts`: also load the DSpark draft head from the folder's mtp.safetensors, when the config has one and the file is there.
/// `draft_bits`: draft layer projection width (INKLING_DRAFT_BITS); null picks `dspark_weights.Bits.defaultFor(mtp)`.
/// `prefill_rows`: rows of a prompt chunk at most (INKLING_PREFILL_ROWS), a multiple of 64 from 64 to 8192: a scratch is made for
/// that many rows and a prompt is fed in pieces of them; null takes `state.defaultPrefillRows` for the load's path.
pub const Options = struct { layers: ?u32 = null, drafts: bool = false, draft_bits: ?dspark_weights.Bits = null, prefill_rows: ?u32 = null };

/// The kernels of the chunk path, which runs a prompt chunk's rows on the tensor units: the dense projections' GEMM, the attention
/// and the experts' chain. A load has them on a GPU with tensor units, for the shapes they are written for (`chunkGeometry`).
pub const Chunk = struct {
    dense: dense_prefill.Pipes,
    attention: attention_prefill.Pipes,
    experts: union(enum) {
        exl3: experts_prefill.Pipes,
        fp: experts_fp_prefill.Pipes,
    },

    /// Compiles default 48/8 EXL3 chunk kernels for `device`, which has tensor units (`error.NeedsTensorUnits` otherwise).
    pub fn init(device: mtl.Device, gpa: std.mem.Allocator) !Chunk {
        return initFor(device, gpa, attention_prefill.heads, attention_prefill.kv_heads, .exl3);
    }

    /// Compiles chunk kernels for `device`, parameterized by attention head geometry and expert storage format.
    pub fn initFor(device: mtl.Device, gpa: std.mem.Allocator, q_heads: u32, kv_heads: u32, quant: cfg.Quant) !Chunk {
        if (quant == .exl3 and !experts_prefill.available(device)) return error.NeedsTensorUnits;
        if (quant != .exl3 and !experts_fp_prefill.available(device)) return error.NeedsTensorUnits;
        var dense = try dense_prefill.Pipes.init(device, gpa);
        errdefer dense.deinit();
        var attention = try attention_prefill.Pipes.initFor(device, gpa, q_heads, kv_heads);
        errdefer attention.deinit();
        if (quant == .exl3) {
            return .{ .dense = dense, .attention = attention, .experts = .{ .exl3 = try experts_prefill.Pipes.init(device, gpa) } };
        }
        return .{ .dense = dense, .attention = attention, .experts = .{ .fp = try experts_fp_prefill.Pipes.init(device, gpa) } };
    }

    pub fn deinit(c: *Chunk) void {
        switch (c.experts) {
            inline else => |*e| e.deinit(),
        }
        c.attention.deinit();
        c.dense.deinit();
    }
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    plan: plan.Plan,
    layers: u32,
    checkpoint: ck.Checkpoint,
    weights: wts.Weights,
    glue: [glue_names.len]mtl.Pipeline,
    exl3: ?exl3.Pipelines = null,
    /// The chunk path's kernels; null where prompt chunks run on the per-operation kernels (a GPU without tensor units, shapes the
    /// chunk kernels are not written for, or INKLING_PROMPT_PATH=per_op).
    chunk: ?Chunk,
    /// The draft head's tensors in `checkpoint`, when `Options.drafts` asked for them and they exist.
    draft: ?dspark_weights.Weights,
    /// Whether decode steps wait warm (forward.zig's `waitWarm`); INKLING_WARM_WAIT=0 turns it off.
    warm_wait: bool,
    /// Layers a decode step commits in a command buffer of their own, before the rest are encoded (0: the step is one
    /// command buffer); INKLING_FIRST_CHUNK.
    first_chunk: u32,
    /// Rows of a prompt chunk at most (`Options.prefill_rows`): the load's scratch and budget are sized for them.
    prefill_rows: u32,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, opts: Options) !*Model {
        if (opts.prefill_rows) |rows| if (!st.validPrefillRows(rows)) return error.InvalidInkling2PrefillRows;
        // INKLING_PROMPT_PATH, read once: names a path to compare or to test the fallbacks with; unset it asks nothing
        const request = st.parsePromptPath(if (std.c.getenv("INKLING_PROMPT_PATH")) |v| std.mem.span(v) else null) catch {
            std.log.err("inkling2: INKLING_PROMPT_PATH is not a prompt path: give per_op, rows, rows:LAYER,LAYER or auto", .{});
            return error.InvalidInkling2PromptPath;
        };
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.warm_wait = if (std.c.getenv("INKLING_WARM_WAIT")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        m.first_chunk = if (std.c.getenv("INKLING_FIRST_CHUNK")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch 4 else 4;
        const cpath = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(cpath);
        const json = try std.Io.Dir.cwd().readFileAlloc(io, cpath, gpa, .limited(1 << 22));
        defer gpa.free(json);
        var why: cfg.Why = .{};
        m.config = cfg.parse(gpa, json, &why) catch |err| {
            std.log.err("inkling2: {s}", .{why.text()});
            return err;
        };
        if (m.config.quant != .exl3 and m.config.quant != .nvfp4) return error.UnsupportedInkling2Quant;
        const t = &m.config.text;
        try geometry(t);
        var prec: [cfg.max_layers]plan.Precision = @splat(if (m.config.quant == .exl3) .exl3 else .nvfp4);
        m.plan = try plan.build(t, prec[0..t.layers]);
        const vtotal = t.virtualLayers();
        m.layers = opts.layers orelse vtotal;
        if (m.layers > vtotal) return error.InvalidInkling2Layers;
        var phys_layers: u32 = 0;
        for (0..m.layers) |vi| phys_layers = @max(phys_layers, m.plan.at(@intCast(vi)).physical + 1);
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        // prompt chunks run on the tensor units where the GPU has them and the shapes fit; elsewhere on the per-operation kernels
        const reason: ?[]const u8 = if (request.per_op)
            "INKLING_PROMPT_PATH asks for them"
        else if (!m.device.tensorUnits())
            "this GPU has no tensor units"
        else if (!chunkGeometry(t, m.config.quant))
            "this config's shapes are not the tensor-unit kernels'"
        else
            null;
        if (reason) |r| {
            m.chunk = null;
            std.log.info("inkling2: prompt chunks run on the per-operation kernels: {s}", .{r});
        } else m.chunk = try Chunk.initFor(m.device, gpa, t.local.q, t.local.kv, m.config.quant);
        errdefer if (m.chunk) |*chunk| chunk.deinit();
        m.prefill_rows = opts.prefill_rows orelse st.defaultPrefillRows(if (m.chunk == null) .per_op else .chunk);
        m.checkpoint = ck.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        if (m.config.quant == .exl3) {
            var p = if (phys_layers == 0) pack.Pack{ .arena = std.heap.ArenaAllocator.init(gpa), .layers = &.{} } else try pack.Pack.read(gpa, io, dir, t.layers, t.experts);
            defer p.deinit();
            m.weights = try wts.load(gpa, m.device, &m.checkpoint, t, &m.plan, &p, dir, phys_layers);
        } else {
            m.weights = try wts.loadNative(gpa, io, m.device, &m.checkpoint, t, prec[0..t.layers], dir, phys_layers);
            m.plan = try plan.build(t, prec[0..t.layers]);
        }
        errdefer m.weights.deinit(gpa);
        // a layer the chain cannot take (see `weights.xdOk`), or that INKLING_PROMPT_PATH names, runs its experts through the per-row
        // grouped projections in a chunk, and in a decode window too: `xd_ok` is what both paths read
        for (m.weights.layers, 0..) |*l, i| if ((request.rows >> @intCast(i)) & 1 != 0) switch (l.experts) {
            .exl3 => |*ex| ex.xd_ok = false,
            else => {},
        };
        if (m.chunk != null) {
            var per_row: usize = 0;
            for (m.weights.layers) |l| switch (l.experts) {
                .exl3 => |ex| per_row += @intFromBool(!ex.xd_ok),
                else => {},
            };
            if (per_row > 0) std.log.info("inkling2: prompt chunks take the per-row expert projections on {d} of {d} layers: {s}", .{ per_row, m.weights.layers.len, if (request.rows != 0) "INKLING_PROMPT_PATH asks for them, for decode windows too" else "their experts are not all K2 or K3, which the tensor-unit chain needs" });
        }
        {
            const lib = try mtl.Library.fromSource(m.device, ks.inkling2_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (m.glue[0..made]) |x| x.deinit();
            for (glue_names, 0..) |n, i| {
                m.glue[i] = try mtl.Pipeline.init(m.device, lib, n, false);
                made += 1;
            }
        }
        errdefer for (m.glue) |x| x.deinit();
        m.exl3 = if (m.config.quant == .exl3) try exl3.Pipelines.load(m.device) else null;
        errdefer if (m.exl3) |*ex| ex.deinit();
        m.draft = null;
        if (opts.drafts) {
            if (m.config.mtp) |mtp| {
                const bits = opts.draft_bits orelse dspark_weights.Bits.defaultFor(mtp);
                m.draft = try dspark_weights.load(gpa, io, m.device, &m.checkpoint, dir, mtp, bits);
            } else std.log.info("inkling2: the config has no mtp_config; the draft head stays off", .{});
        }
        return m;
    }

    pub fn pipe(m: *const Model, g: Glue) mtl.Pipeline {
        return m.glue[@backingInt(g)];
    }

    /// The path this load's prompt chunks take, which decides the scratch they need: the chunk kernels on every layer, the chunk
    /// kernels with the per-row expert projections where a layer's experts need them, or the per-operation kernels.
    pub fn promptPath(m: *const Model) st.PromptPath {
        if (m.chunk == null) return .per_op;
        for (m.weights.layers) |l| switch (l.experts) {
            .exl3 => |ex| if (!ex.xd_ok) return .chunk_rows,
            .nvfp4, .mxfp8 => {},
        };
        return .chunk;
    }

    /// What a scratch for this load's prompt chunks is made for.
    pub fn chunking(m: *const Model) st.Chunking {
        return .{
            .rows = m.prefill_rows,
            .path = m.promptPath(),
            .quant = m.config.quant,
            .taps = if (m.config.mtp) |mtp| mtp.taps_n else st.feature_taps,
        };
    }

    pub fn deinit(m: *Model) void {
        if (m.draft) |*w| w.deinit(m.gpa);
        if (m.chunk) |*chunk| chunk.deinit();
        if (m.exl3) |*ex| ex.deinit();
        for (m.glue) |x| x.deinit();
        m.weights.deinit(m.gpa);
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};

/// The shapes the kernels are written for: both layer kinds with GQA=6 (48/8) or GQA=3 (24/8) heads of 128 dims,
/// d_rel 16, top-8, one relative extent for both kinds.
fn geometry(t: *const cfg.Text) !void {
    const gqa = t.local.q / t.local.kv;
    const ok = t.local.q == t.global.q and t.local.kv == t.global.kv and
        (gqa == 6 or gqa == 3) and t.local.q == gqa * t.local.kv and
        t.local.dim == 128 and t.global.dim == 128 and t.d_rel == 16 and t.topk == 8 and t.experts <= 384 and
        t.window == t.rel_extent and t.hidden % 128 == 0 and t.expert_inter % 128 == 0 and st.maxBlocks(t) <= 9;
    if (!ok) return error.UnsupportedInkling2Geometry;
}

/// Whether the tensor-unit chunk kernels support this config's attention, dense, and routed-expert geometry.
fn chunkGeometry(t: *const cfg.Text, quant: cfg.Quant) bool {
    const a = attention_prefill;
    const heads_ok = (t.local.q == 48 or t.local.q == 24) and t.local.kv == a.kv_heads and t.local.dim == a.dim and
        t.global.q == t.local.q and t.global.kv == t.local.kv and t.global.dim == t.local.dim;
    const base_ok = heads_ok and t.d_rel == a.rel_dims and t.window == a.extent and t.rel_extent == a.extent and
        t.hidden % 256 == 0 and (t.local.q * t.local.dim) % 256 == 0 and t.expert_inter % 128 == 0 and t.experts > 0;
    if (!base_ok) return false;
    if (quant == .exl3) return t.hidden == experts_prefill.hidden and t.expert_inter == experts_prefill.inner;
    return 2 * t.expert_inter <= t.hidden;
}
