//! The layer plan: which layers slide or see everything, where attention-residual blocks end, what layers sum over.
const std = @import("std");
const config = @import("config.zig");

/// A routed-expert layer's stored format: ModelOpt NVFP4 (groups of 16), MXFP8 (groups of 32), or an EXL3 pack's trellis.
pub const Precision = enum { nvfp4, mxfp8, exl3 };

pub const Kind = enum { local, global };

pub const Layer = struct {
    kind: Kind,
    heads: config.Heads,
    extent: u32, // backward tokens the relative bias reaches
    boundary: bool, // the layer's input joins the finished blocks after its attention, before its MLP input is formed
    blocks_in: u32, // finished block states when the layer starts
    attn_queries: u8, // learned queries of the attention input: none for layer 0, one per q/k/v for a global layer
    attn_sources: u32, // states the attention input attends over: finished blocks plus the running partial one
    mlp_sources: u32, // the same for the MLP input; a boundary layer first adds its own input as a block
    precision: Precision,
    physical: u32,
    pass: u8,
    core_idx: u16,
    save_prelude: bool,
    reset_from_prelude: bool,
};

pub const Plan = struct {
    layers: [config.max_layers]Layer,
    n: u32,
    final_sources: u32, // the output query's states: every finished block and the last layer's output
    max_blocks: u32,
    pre_blocks: u32,
    bank_blocks: u32,
    core_len: u32,

    pub fn at(p: *const Plan, i: u32) *const Layer {
        return &p.layers[i];
    }

    pub fn count(p: *const Plan, kind: Kind) u32 {
        var k: u32 = 0;
        for (p.layers[0..p.n]) |l| k += @intFromBool(l.kind == kind);
        return k;
    }
};

fn makeLayer(c: *const config.Text, phys: u32, prec: Precision, pass: u8, core_idx: u16, save_prelude: bool, reset_from_prelude: bool) Layer {
    const boundary = phys % c.block == 0;
    const blocks_in = (phys + c.block - 1) / c.block;
    const global = config.isGlobal(phys);
    return .{
        .kind = if (global) .global else .local,
        .heads = c.heads(phys),
        .extent = c.extent(phys),
        .boundary = boundary,
        .blocks_in = blocks_in,
        .attn_queries = if (phys == 0) 0 else if (global) 3 else 1,
        .attn_sources = blocks_in + 1,
        .mlp_sources = blocks_in + 1 + @intFromBool(boundary),
        .precision = prec,
        .physical = phys,
        .pass = pass,
        .core_idx = core_idx,
        .save_prelude = save_prelude,
        .reset_from_prelude = reset_from_prelude,
    };
}

/// The plan for a parsed config and each physical layer's expert precision (`prec` holds one entry per physical layer).
pub fn build(c: *const config.Text, prec: []const Precision) error{PrecisionCount}!Plan {
    if (prec.len != c.layers) return error.PrecisionCount;
    var p: Plan = undefined;
    p.n = c.virtualLayers();
    p.max_blocks = (c.layers + c.block - 1) / c.block;
    p.final_sources = p.max_blocks + 1;
    if (c.loop) |lp| {
        const core_len = lp.core_end - lp.core_start;
        p.pre_blocks = lp.core_start / c.block;
        p.bank_blocks = core_len / c.block - 1;
        p.core_len = core_len;
        var v: u32 = 0;
        for (0..lp.core_start) |k| {
            const phys: u32 = @intCast(k);
            p.layers[v] = makeLayer(c, phys, prec[phys], 0, 0, false, false);
            v += 1;
        }
        for (0..lp.num_passes) |pass_u| {
            const pass: u8 = @intCast(pass_u);
            for (0..core_len) |ci| {
                const phys: u32 = lp.core_start + @as(u32, @intCast(ci));
                p.layers[v] = makeLayer(c, phys, prec[phys], pass, @intCast(ci), pass == 0 and ci == 0, pass > 0 and ci == 0);
                v += 1;
            }
        }
        for (lp.core_end..c.layers) |k| {
            const phys: u32 = @intCast(k);
            p.layers[v] = makeLayer(c, phys, prec[phys], 0, 0, false, false);
            v += 1;
        }
        std.debug.assert(v == p.n);
    } else {
        p.pre_blocks = 0;
        p.bank_blocks = 0;
        p.core_len = 0;
        for (0..c.layers) |k| {
            const i: u32 = @intCast(k);
            p.layers[i] = makeLayer(c, i, prec[i], 0, 0, false, false);
        }
    }
    return p;
}
