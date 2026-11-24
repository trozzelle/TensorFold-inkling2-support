//! Routed-expert prefill chain on M5 tensor units for native NVFP4 and MXFP8 Inkling 2 checkpoints
//! (kernels/metal/inkling2/xp_fp.metal): picks grouped by expert into 32-row windows on the GPU, bf16 input gather,
//! grouped tensor-unit GEMM with exact in-register bf16 dequantization (f32 accumulation), SoftcappedSwiGLU, and
//! pick-ordered weighted combine. Every row is independent of neighbouring rows and chunk cuts.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const weights = @import("weights.zig");

pub const topk = 8;
/// Consecutive sorted slots of one expert that share a dequantization pass on the tensor units.
pub const window = 32;

/// A buffer and the byte offset of a view in it.
pub const B = struct { buf: mtl.Buffer, off: usize = 0 };
/// ik_route's output: u32 expert ids (16-byte aligned) and f32 weights, [rows][topk] each.
pub const Picks = struct { ids: B, weights: B };
/// SoftcappedSwiGLU caps on gate and up halves.
pub const Caps = struct { gate: f32, up: f32 };
/// One layer's native NVFP4 expert projections and routed expert count.
pub const Nvfp4Layer = struct { w13: weights.Nvfp4Proj, w2: weights.Nvfp4Proj, experts: usize };
/// One layer's native MXFP8 expert projections and routed expert count.
pub const Mxfp8Layer = struct { w13: weights.Mxfp8Proj, w2: weights.Mxfp8Proj, experts: usize };

/// ik_xpf_args in xp_fp.metal, field for field.
const Args = extern struct {
    rows: u32,
    slots: u32,
    experts: u32,
    topk: u32,
    hidden: u32,
    inner: u32,
    n: u32,
    k: u32,
    cap: f32,
    thr: f32,
};

/// ik_xpf_win: a window's first sorted position, its live rows, and its expert.
const Win = extern struct { start: u32, nlive: u32, eid: u32, pad: u32 };

comptime {
    std.debug.assert(@sizeOf(Args) == 40 and @sizeOf(Win) == 16);
}

/// Whether this GPU runs the chain (M5 tensor-unit operand layout).
pub fn available(device: mtl.Device) bool {
    return device.tensorUnits();
}

/// Kernel source with nax.h inlined for M5 tensor units and the window size defined.
pub fn source(a: std.mem.Allocator) ![]u8 {
    const body = try frags.precision.source(a, true, ks.inkling2_xp_fp, ks.nax, false);
    defer a.free(body);
    return std.fmt.allocPrint(a, "#define IK_XPF_WINDOW {d}\n{s}", .{ window, body });
}

const kernels = [_]struct { name: [:0]const u8, threads: usize }{
    .{ .name = "ik_xpf_count", .threads = 256 },
    .{ .name = "ik_xpf_place", .threads = 256 },
    .{ .name = "ik_xpf_gather", .threads = 128 },
    .{ .name = "ik_xpf_gemm_nvfp4", .threads = 128 },
    .{ .name = "ik_xpf_gemm_mxfp8", .threads = 128 },
    .{ .name = "ik_xpf_mid", .threads = 128 },
    .{ .name = "ik_xpf_finish", .threads = 128 },
};

pub const Pipes = struct {
    lib: mtl.Library,
    count: mtl.Pipeline,
    place: mtl.Pipeline,
    gather: mtl.Pipeline,
    gemm_nvfp4: mtl.Pipeline,
    gemm_mxfp8: mtl.Pipeline,
    mid: mtl.Pipeline,
    finish: mtl.Pipeline,

    pub fn init(device: mtl.Device, a: std.mem.Allocator) !Pipes {
        if (!available(device)) return error.NeedsTensorUnits;
        const text = try source(a);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, .{
            .language = mtl.CompileOptions.mlx().language,
            .math_functions = .precise,
        });
        errdefer lib.deinit();
        var made: [kernels.len]mtl.Pipeline = undefined;
        var n: usize = 0;
        errdefer for (made[0..n]) |x| x.deinit();
        for (kernels) |k| {
            const pipe = try mtl.Pipeline.init(device, lib, k.name, false);
            made[n] = pipe;
            n += 1;
            if (pipe.simdWidth() != 32 or pipe.maxThreads() < k.threads) return error.UnsupportedSimd;
        }
        return .{
            .lib = lib,
            .count = made[0],
            .place = made[1],
            .gather = made[2],
            .gemm_nvfp4 = made[3],
            .gemm_mxfp8 = made[4],
            .mid = made[5],
            .finish = made[6],
        };
    }

    pub fn deinit(p: *Pipes) void {
        for ([_]mtl.Pipeline{ p.count, p.place, p.gather, p.gemm_nvfp4, p.gemm_mxfp8, p.mid, p.finish }) |x| x.deinit();
        p.lib.deinit();
    }
};

fn slotsOf(rows: usize) usize {
    return rows * topk;
}

/// Upper bound on sorted windows for `rows` rows and `experts` routed experts.
pub fn winCap(rows: usize, experts: usize) usize {
    return (slotsOf(rows) + window - 1) / window + experts;
}

/// Scratch region sizes in bytes, in `Xp`'s layout order.
fn regions(rows: usize, hidden: usize, experts: usize) [8]usize {
    const slots = slotsOf(rows);
    return .{
        experts * 4, // counts
        slots * 4, // order
        slots * 4, // seid
        slots * 4, // inverse
        winCap(rows, experts) * @sizeOf(Win), // wins
        2 * 16, // dargs: w13 (offset 0) and w2 (offset 16)
        slots * hidden * 2, // prep: bf16 [slots, hidden], reused by w2 input [slots, inner]
        slots * hidden * 4, // plane: f32 [slots, hidden], reused by w13 and w2 outputs
    };
}

const region_align = 256;

/// Scratch bytes needed for chunks up to `rows` rows of width `hidden` across `experts` experts.
pub fn xpBytes(rows: usize, hidden: usize, experts: usize) usize {
    var total: usize = 0;
    for (regions(rows, hidden, experts)) |r| total = std.mem.alignForward(usize, total, region_align) + r;
    return std.mem.alignForward(usize, total, region_align);
}

/// Views into a single scratch buffer for sorted-order tables, window list, and staging planes.
pub const Xp = struct {
    rows: usize,
    hidden: usize,
    experts: usize,
    counts: B,
    order: B,
    seid: B,
    inverse: B,
    wins: B,
    dargs: B,
    prep: B,
    plane: B,

    /// Initialize views at 256-byte-aligned `off` in `buf`, which must hold at least `xpBytes(rows, hidden, experts)`.
    pub fn init(buf: mtl.Buffer, off: usize, rows: usize, hidden: usize, experts: usize) Xp {
        std.debug.assert(off % region_align == 0 and off + xpBytes(rows, hidden, experts) <= buf.length());
        var at: usize = off;
        var views: [8]B = undefined;
        for (regions(rows, hidden, experts), &views) |r, *v| {
            at = std.mem.alignForward(usize, at, region_align);
            v.* = .{ .buf = buf, .off = at };
            at += r;
        }
        return .{
            .rows = rows,
            .hidden = hidden,
            .experts = experts,
            .counts = views[0],
            .order = views[1],
            .seid = views[2],
            .inverse = views[3],
            .wins = views[4],
            .dargs = views[5],
            .prep = views[6],
            .plane = views[7],
        };
    }
};

fn bind(e: mtl.ComputeEncoder, v: B, index: usize) void {
    e.setBuffer(v.buf, v.off, index);
}

fn checkShape(w13_n: u32, w13_k: u32, w2_n: u32, w2_k: u32, experts: usize, s: Xp, n: usize, ids_off: usize) void {
    std.debug.assert(n >= 1 and n <= s.rows);
    std.debug.assert(experts >= 1 and experts <= s.experts);
    std.debug.assert(w13_k == w2_n and w13_n == 2 * w2_k);
    std.debug.assert(w13_k <= s.hidden and w13_n <= s.hidden);
    std.debug.assert(w13_k % 128 == 0 and w2_k % 128 == 0 and w13_n % 128 == 0);
    std.debug.assert(ids_off % 16 == 0);
}

fn makeArgs(n: usize, experts: usize, hidden: u32, inner: u32, proj_n: u32, proj_k: u32, caps: Caps) Args {
    return .{
        .rows = @intCast(n),
        .slots = @intCast(slotsOf(n)),
        .experts = @intCast(experts),
        .topk = topk,
        .hidden = hidden,
        .inner = inner,
        .n = proj_n,
        .k = proj_k,
        .cap = caps.gate,
        .thr = caps.up,
    };
}

fn encodeGroupAndGather(e: mtl.ComputeEncoder, p: *const Pipes, a: Args, x: B, ids: B, s: Xp) void {
    e.setPipeline(p.count);
    e.setValue(a, 0);
    bind(e, ids, 1);
    bind(e, s.counts, 2);
    e.dispatchGroups(mtl.Size.of(a.experts, 1, 1), mtl.Size.of(256, 1, 1));

    e.setPipeline(p.place);
    e.setValue(a, 0);
    bind(e, ids, 1);
    bind(e, s.counts, 2);
    bind(e, s.order, 3);
    bind(e, s.seid, 4);
    bind(e, s.inverse, 5);
    bind(e, s.wins, 6);
    bind(e, s.dargs, 7);
    e.dispatchGroups(mtl.Size.of(a.experts + 1, 1, 1), mtl.Size.of(256, 1, 1));

    e.setPipeline(p.gather);
    e.setValue(a, 0);
    bind(e, x, 1);
    bind(e, s.order, 2);
    bind(e, s.seid, 3);
    bind(e, s.prep, 4);
    e.dispatchGroups(mtl.Size.of(a.hidden / 128, a.slots, 1), mtl.Size.of(128, 1, 1));
}

fn encodeMid(e: mtl.ComputeEncoder, p: *const Pipes, a: Args, s: Xp) void {
    e.setPipeline(p.mid);
    e.setValue(a, 0);
    bind(e, s.plane, 1);
    bind(e, s.seid, 2);
    bind(e, s.prep, 3);
    e.dispatchGroups(mtl.Size.of(a.inner / 128, a.slots, 1), mtl.Size.of(128, 1, 1));
}

fn encodeFinish(e: mtl.ComputeEncoder, p: *const Pipes, a: Args, picks: Picks, out: B, s: Xp) void {
    e.setPipeline(p.finish);
    e.setValue(a, 0);
    bind(e, s.plane, 1);
    bind(e, picks.ids, 2);
    bind(e, s.inverse, 3);
    bind(e, picks.weights, 4);
    bind(e, out, 5);
    e.dispatchGroups(mtl.Size.of(a.hidden / 128, a.rows, 1), mtl.Size.of(128, 1, 1));
}

/// Encode a full NVFP4 routed-expert layer for `n` rows of `x` (bf16 [n, hidden]) into `out` (bf16 [n, hidden]).
pub fn encodeNvfp4(
    e: mtl.ComputeEncoder,
    p: *const Pipes,
    l: Nvfp4Layer,
    x: B,
    picks: Picks,
    out: B,
    s: Xp,
    n: usize,
    caps: Caps,
) void {
    checkShape(l.w13.n, l.w13.k, l.w2.n, l.w2.k, l.experts, s, n, picks.ids.off);
    const hidden = l.w13.k;
    const inner = l.w2.k;
    const a13 = makeArgs(n, l.experts, hidden, inner, l.w13.n, l.w13.k, caps);
    const a2 = makeArgs(n, l.experts, hidden, inner, l.w2.n, l.w2.k, caps);

    encodeGroupAndGather(e, p, a13, x, picks.ids, s);

    e.setPipeline(p.gemm_nvfp4);
    e.setValue(a13, 0);
    bind(e, s.prep, 1);
    bind(e, s.wins, 2);
    e.setBuffer(l.w13.weight.buffer, l.w13.weight.offset, 3);
    e.setBuffer(l.w13.scale.buffer, l.w13.scale.offset, 4);
    e.setBuffer(l.w13.global_scale, 0, 5);
    bind(e, s.plane, 6);
    e.dispatchIndirect(s.dargs.buf, s.dargs.off, mtl.Size.of(128, 1, 1));

    encodeMid(e, p, a13, s);

    e.setPipeline(p.gemm_nvfp4);
    e.setValue(a2, 0);
    bind(e, s.prep, 1);
    bind(e, s.wins, 2);
    e.setBuffer(l.w2.weight.buffer, l.w2.weight.offset, 3);
    e.setBuffer(l.w2.scale.buffer, l.w2.scale.offset, 4);
    e.setBuffer(l.w2.global_scale, 0, 5);
    bind(e, s.plane, 6);
    e.dispatchIndirect(s.dargs.buf, s.dargs.off + 16, mtl.Size.of(128, 1, 1));

    encodeFinish(e, p, a2, picks, out, s);
}

/// Encode a full MXFP8 routed-expert layer for `n` rows of `x` (bf16 [n, hidden]) into `out` (bf16 [n, hidden]).
pub fn encodeMxfp8(
    e: mtl.ComputeEncoder,
    p: *const Pipes,
    l: Mxfp8Layer,
    x: B,
    picks: Picks,
    out: B,
    s: Xp,
    n: usize,
    caps: Caps,
) void {
    checkShape(l.w13.n, l.w13.k, l.w2.n, l.w2.k, l.experts, s, n, picks.ids.off);
    const hidden = l.w13.k;
    const inner = l.w2.k;
    const a13 = makeArgs(n, l.experts, hidden, inner, l.w13.n, l.w13.k, caps);
    const a2 = makeArgs(n, l.experts, hidden, inner, l.w2.n, l.w2.k, caps);

    encodeGroupAndGather(e, p, a13, x, picks.ids, s);

    e.setPipeline(p.gemm_mxfp8);
    e.setValue(a13, 0);
    bind(e, s.prep, 1);
    bind(e, s.wins, 2);
    e.setBuffer(l.w13.weight.buffer, l.w13.weight.offset, 3);
    e.setBuffer(l.w13.scale.buffer, l.w13.scale.offset, 4);
    bind(e, s.plane, 5);
    e.dispatchIndirect(s.dargs.buf, s.dargs.off, mtl.Size.of(128, 1, 1));

    encodeMid(e, p, a13, s);

    e.setPipeline(p.gemm_mxfp8);
    e.setValue(a2, 0);
    bind(e, s.prep, 1);
    bind(e, s.wins, 2);
    e.setBuffer(l.w2.weight.buffer, l.w2.weight.offset, 3);
    e.setBuffer(l.w2.scale.buffer, l.w2.scale.offset, 4);
    bind(e, s.plane, 5);
    e.dispatchIndirect(s.dargs.buf, s.dargs.off + 16, mtl.Size.of(128, 1, 1));

    encodeFinish(e, p, a2, picks, out, s);
}

test "xp_fp scratch: regions are 256-byte aligned, planes dominate, and sizes scale cleanly" {
    const rows: usize = 4096;
    const hidden: usize = 2304;
    const experts: usize = 128;
    const r = regions(rows, hidden, experts);
    var at: usize = 0;
    for (r) |sz| {
        at = std.mem.alignForward(usize, at, region_align);
        try std.testing.expectEqual(@as(usize, 0), at % region_align);
        at += sz;
    }
    const total = xpBytes(rows, hidden, experts);
    try std.testing.expectEqual(std.mem.alignForward(usize, at, region_align), total);
    try std.testing.expectEqual(@as(usize, 0), total % region_align);
    try std.testing.expectEqual(@as(usize, slotsOf(rows) * hidden * 2), r[6]);
    try std.testing.expectEqual(@as(usize, slotsOf(rows) * hidden * 4), r[7]);
    try std.testing.expect(total > r[6] + r[7] and total < r[6] + r[7] + (1 << 20));
    try std.testing.expect(xpBytes(64, hidden, experts) < xpBytes(128, hidden, experts));
    try std.testing.expect(xpBytes(128, hidden, 64) < xpBytes(128, hidden, 128));
}
