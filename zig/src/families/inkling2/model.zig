//! A native Inkling 2 model with EXL3 experts: config, plan, the checkpoint in Metal buffers and every pipeline.
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

pub const glue_names = [_][:0]const u8{ "ik_embed", "ik_rms", "ik_norm", "ik_depth", "ik_resid", "ik_out", "ik_copy", "ik_softcap", "ik_conv", "ik_conv_dyn", "ik_ring", "ik_prep", "ik_relbias", "ik_attn", "ik_router", "ik_route", "ik_tiles", "ik_swiglu", "ik_combine", "ik_matvec_1", "ik_matvec_2", "ik_matvec_4", "ik_matvec_8", "ik_matvec_16", "ik_matvec_32", "ik_mvs_8", "ik_mvs_16", "ik_rdn", "ik_rdn3", "ik_rrdn", "ik_attn_part", "ik_attn_merge", "ik_xd_pair", "ik_xd_mid", "ik_xd_down", "ik_xd_finish", "ik_conv_ring", "ik_conv_dyn_ring", "ik_conv_out", "ik_tap" };
pub const Glue = enum { embed, rms, norm, depth, resid, out, copy, softcap, conv, conv_dyn, ring, prep, relbias, attn, router, route, tiles, swiglu, combine, matvec_1, matvec_2, matvec_4, matvec_8, matvec_16, matvec_32, mvs_8, mvs_16, rdn, rdn3, rrdn, attn_part, attn_merge, xd_pair, xd_mid, xd_down, xd_finish, conv_ring, conv_dyn_ring, conv_out, tap };

comptime {
    // `pipe` indexes the pipelines by the enum's order: each member is its kernel's name without the ik_ prefix
    const members = @typeInfo(Glue).@"enum".field_names;
    std.debug.assert(members.len == glue_names.len);
    for (members, glue_names) |member, name| std.debug.assert(std.mem.eql(u8, "ik_" ++ member, name));
}

/// `layers`: load only the first layers (the per-layer checks on a Mac that cannot hold the model).
pub const Options = struct { layers: ?u32 = null };

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
    exl3: exl3.Pipelines,
    /// Whether decode steps wait warm (forward.zig's `waitWarm`); INKLING_WARM_WAIT=0 turns it off.
    warm_wait: bool,
    /// Layers a decode step commits in a command buffer of their own, before the rest are encoded (0: the step is one
    /// command buffer); INKLING_FIRST_CHUNK.
    first_chunk: u32,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, opts: Options) !*Model {
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
        if (m.config.quant != .exl3) return error.NotAnInkling2Exl3Pack;
        const t = &m.config.text;
        try geometry(t);
        var prec: [cfg.max_layers]plan.Precision = @splat(.exl3);
        m.plan = try plan.build(t, prec[0..t.layers]);
        m.layers = opts.layers orelse t.layers;
        if (m.layers == 0 or m.layers > t.layers) return error.InvalidInkling2Layers;
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.checkpoint = ck.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        var p = try pack.Pack.read(gpa, io, dir, t.layers, t.experts);
        defer p.deinit();
        m.weights = try wts.load(gpa, m.device, &m.checkpoint, t, &m.plan, &p, dir, m.layers);
        errdefer m.weights.deinit(gpa);
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
        m.exl3 = try exl3.Pipelines.load(m.device);
        return m;
    }

    pub fn pipe(m: *const Model, g: Glue) mtl.Pipeline {
        return m.glue[@backingInt(g)];
    }

    pub fn deinit(m: *Model) void {
        m.exl3.deinit();
        for (m.glue) |x| x.deinit();
        m.weights.deinit(m.gpa);
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};

/// The shapes the kernels are written for: both layer kinds 48 x 128 queries over 8 x 128 keys (six query heads to a key
/// head), d_rel 16, top-8, one relative extent for both kinds.
fn geometry(t: *const cfg.Text) !void {
    const ok = t.local.q == t.global.q and t.local.kv == t.global.kv and t.local.q == 6 * t.local.kv and
        t.local.dim == 128 and t.global.dim == 128 and t.d_rel == 16 and t.topk == 8 and t.experts <= 384 and
        t.window == t.rel_extent and t.hidden % 128 == 0 and t.expert_inter % 128 == 0 and st.maxBlocks(t) <= 9;
    if (!ok) return error.UnsupportedInkling2Geometry;
}
