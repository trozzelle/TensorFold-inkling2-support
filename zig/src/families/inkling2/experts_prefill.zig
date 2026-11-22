//! Inkling 2's routed experts for a prompt chunk on the tensor units (kernels/metal/inkling2/xp.metal): the picks grouped
//! by expert on the GPU, one rotated f16 input plane, a grouped tensor-unit GEMM per rate and projection over the
//! experts' decoded trellis tiles (f32 sums), the capped SwiGLU, and the weighted sum at the decode chain's rounding
//! points. Any number of rows from 1 to the scratch's capacity runs, and a row's bits do not depend on the chunk it is in:
//! no split-K, no cross-row reduction, a row's sums in one fixed order whichever rows share its window.
//!
//! The kernels reach trellis and scales through the layer's descriptor tables, so the caller declares the weights'
//! buffers on the encoder (`useResource`) as for the decode chain, and the tables must pass `weights.xdOk` (K2 or K3
//! experts, mul1, f16 scales, tiles back to back). They need a GPU with tensor units (`available`).
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const exl3 = @import("../../core/exl3_projection.zig");

pub const hidden = 4608;
pub const inner = 2304;
pub const topk = 8;
/// Rows of a window: up to this many consecutive sorted slots of one expert share a decode of its tiles.
pub const window = 32;
/// Experts a layer has at most (the count and window kernels hold one threadgroup an expert).
pub const max_experts = 512;

/// A buffer and the byte offset of a view in it.
pub const B = struct { buf: mtl.Buffer, off: usize = 0 };
/// ik_route's output: u32 expert ids and f32 weights, [rows][8] each, in pick order. The ids are read as 16-byte vectors.
pub const Picks = struct { ids: B, weights: B };
/// The capped SwiGLU's caps on the gate and on the up half.
pub const Caps = struct { gate: f32, up: f32 };
/// A layer's expert tables: w13 [4608 -> 4608] and w2 [2304 -> 4608] descriptors (`Weights.Layer.w13` and `.w2`).
pub const Layer = struct { w13: exl3.Experts, w2: exl3.Experts };
pub const Proj = enum { w13, w2 };

/// ik_xp_args in xp.metal, field for field.
const Args = extern struct { rows: u32, slots: u32, experts: u32, win_cap: u32, gx: u32, cap: f32, thr: f32 };
/// ik_xp_win: a window's first sorted position, its live rows, its expert.
const Win = extern struct { start: u32, nlive: u32, eid: u32, pad: u32 };

comptime {
    std.debug.assert(@sizeOf(Args) == 28 and @sizeOf(Win) == 16);
}

/// Whether this GPU runs the chain: the weight fragments' layout is the tensor units' (nax.h's M5 layout).
pub fn available(device: mtl.Device) bool {
    return device.tensorUnits();
}

/// The kernels' text with nax.h inlined for tensor units and the window's rows defined.
pub fn source(a: std.mem.Allocator) ![]u8 {
    const body = try frags.precision.source(a, true, ks.inkling2_xp, ks.nax, false);
    defer a.free(body);
    return std.fmt.allocPrint(a, "#define IK_XP_WINDOW {d}\n{s}", .{ window, body });
}

/// The kernels in `Pipes`' field order, with the threads a threadgroup of each must be able to hold.
const kernels = [_]struct { name: [:0]const u8, threads: usize }{
    .{ .name = "ik_xp_count", .threads = 256 },
    .{ .name = "ik_xp_place", .threads = 256 },
    .{ .name = "ik_xp_prepare", .threads = 128 },
    .{ .name = "ik_xp_gemm13_k2", .threads = 128 },
    .{ .name = "ik_xp_gemm13_k3", .threads = 128 },
    .{ .name = "ik_xp_gemm2_k2", .threads = 128 },
    .{ .name = "ik_xp_gemm2_k3", .threads = 128 },
    .{ .name = "ik_xp_mid", .threads = 192 },
    .{ .name = "ik_xp_mid_dbg", .threads = 192 },
    .{ .name = "ik_xp_finish", .threads = 256 },
    .{ .name = "ik_xp_finish_dbg", .threads = 256 },
};

pub const Pipes = struct {
    lib: mtl.Library,
    count: mtl.Pipeline,
    place: mtl.Pipeline,
    prepare: mtl.Pipeline,
    /// [projection][rate]: w13 or w2, K2 or K3.
    gemm: [2][2]mtl.Pipeline,
    mid: mtl.Pipeline,
    mid_dbg: mtl.Pipeline,
    finish: mtl.Pipeline,
    finish_dbg: mtl.Pipeline,

    pub fn init(device: mtl.Device, a: std.mem.Allocator) !Pipes {
        if (!available(device)) return error.NeedsTensorUnits;
        const text = try source(a);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
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
            .prepare = made[2],
            .gemm = .{ .{ made[3], made[4] }, .{ made[5], made[6] } },
            .mid = made[7],
            .mid_dbg = made[8],
            .finish = made[9],
            .finish_dbg = made[10],
        };
    }

    pub fn deinit(p: *Pipes) void {
        for ([_]mtl.Pipeline{ p.count, p.place, p.prepare, p.gemm[0][0], p.gemm[0][1], p.gemm[1][0], p.gemm[1][1], p.mid, p.mid_dbg, p.finish, p.finish_dbg }) |x| x.deinit();
        p.lib.deinit();
    }
};

fn slotsOf(rows: usize) usize {
    return rows * topk;
}

/// Windows one list holds at most: one for every `window` slots, and a partial one for each expert.
fn winCap(rows: usize) usize {
    return (slotsOf(rows) + window - 1) / window + max_experts;
}

/// The scratch's regions for up to `rows` rows, in bytes, in `Xp`'s order.
fn regions(rows: usize) [9]usize {
    const slots = slotsOf(rows);
    return .{
        max_experts * 4, // counts
        max_experts * 4, // classes
        slots * 4, // order
        slots * 4, // seid
        slots * 4, // inverse
        4 * winCap(rows) * @sizeOf(Win), // wins: lists w13 K2, w13 K3, w2 K2, w2 K3
        4 * 16, // dargs: each list's indirect dispatch
        slots * hidden * 2, // prep: f16 rotated rows [slots][4608]; the down projection's [slots][2304] reuse it after w13 has read it
        slots * hidden * 4, // plane: f32 sums [slots][4608], w13's and then w2's
    };
}

const region_align = 256;

/// Bytes of scratch for chunks of up to `rows` rows: about 906 MB at 4,096 rows.
pub fn xpBytes(rows: usize) usize {
    var total: usize = 0;
    for (regions(rows)) |r| total = std.mem.alignForward(usize, total, region_align) + r;
    return std.mem.alignForward(usize, total, region_align);
}

/// The scratch's views into one buffer: sorted-order tables, the window lists, and the two planes. Every region is
/// written before it is read within a chunk, so nothing carries over between chunks or layers.
pub const Xp = struct {
    rows: usize,
    counts: B,
    classes: B,
    /// order[pos]: the slot at sorted position pos; seid[pos]: its expert; inverse[slot]: its position.
    order: B,
    seid: B,
    inverse: B,
    /// Window records and indirect dispatch arguments (12 bytes in 16) of the four lists.
    wins: B,
    dargs: B,
    prep: B,
    plane: B,

    /// Views at `off` in `buf` (aligned to 256), which holds at least `xpBytes(rows)` from there.
    pub fn init(buf: mtl.Buffer, off: usize, rows: usize) Xp {
        std.debug.assert(off % region_align == 0 and off + xpBytes(rows) <= buf.length());
        var at: usize = off;
        var views: [9]B = undefined;
        for (regions(rows), &views) |r, *v| {
            at = std.mem.alignForward(usize, at, region_align);
            v.* = .{ .buf = buf, .off = at };
            at += r;
        }
        return .{ .rows = rows, .counts = views[0], .classes = views[1], .order = views[2], .seid = views[3], .inverse = views[4], .wins = views[5], .dargs = views[6], .prep = views[7], .plane = views[8] };
    }
};

fn args(l: Layer, s: Xp, n: usize, caps: Caps) Args {
    return .{ .rows = @intCast(n), .slots = @intCast(slotsOf(n)), .experts = @intCast(l.w13.count), .win_cap = @intCast(winCap(s.rows)), .gx = hidden / 128, .cap = caps.gate, .thr = caps.up };
}

/// Every descriptor of both tables is a K2 or K3 expert, which the kernels' rate classes need (the tables are host
/// memory: shared buffers).
fn ratesOk(l: Layer) bool {
    for ([_]exl3.Experts{ l.w13, l.w2 }) |w| {
        const size = @sizeOf(exl3.ExpertDescriptor);
        const bytes = w.descriptors.buffer.slice(u8, w.descriptors.offset + w.count * size)[w.descriptors.offset..];
        const desc: []const exl3.ExpertDescriptor = @alignCast(std.mem.bytesAsSlice(exl3.ExpertDescriptor, bytes));
        for (desc) |d| if (d.halves != 4 and d.halves != 6) return false;
    }
    return true;
}

fn check(l: Layer, s: Xp, n: usize) void {
    std.debug.assert(n >= 1 and n <= s.rows);
    std.debug.assert(l.w13.count == l.w2.count and l.w13.count >= 1 and l.w13.count <= max_experts);
    std.debug.assert(l.w13.k == hidden and l.w13.n == 2 * inner and l.w2.k == inner and l.w2.n == hidden);
    std.debug.assert(ratesOk(l));
}

fn bind(e: mtl.ComputeEncoder, v: B, index: usize) void {
    e.setBuffer(v.buf, v.off, index);
}

fn table(e: mtl.ComputeEncoder, w: exl3.Experts, index: usize) void {
    e.setBuffer(w.descriptors.buffer, w.descriptors.offset, index);
}

/// The picks' counting sort by expert and the window lists: two dispatches, no host sync. `ids` is u32 [n][8].
pub fn group(e: mtl.ComputeEncoder, p: *const Pipes, l: Layer, ids: B, s: Xp, n: usize) void {
    check(l, s, n);
    std.debug.assert(ids.off % 16 == 0);
    const a = args(l, s, n, .{ .gate = 0, .up = 0 });
    const experts = l.w13.count;
    e.setPipeline(p.count);
    e.setValue(a, 0);
    bind(e, ids, 1);
    table(e, l.w13, 2);
    table(e, l.w2, 3);
    bind(e, s.counts, 4);
    bind(e, s.classes, 5);
    e.dispatchGroups(mtl.Size.of(experts, 1, 1), mtl.Size.of(256, 1, 1));
    e.setPipeline(p.place);
    e.setValue(a, 0);
    bind(e, ids, 1);
    bind(e, s.counts, 2);
    bind(e, s.classes, 3);
    bind(e, s.order, 4);
    bind(e, s.seid, 5);
    bind(e, s.inverse, 6);
    bind(e, s.wins, 7);
    bind(e, s.dargs, 8);
    e.dispatchGroups(mtl.Size.of(experts + 1, 1, 1), mtl.Size.of(256, 1, 1));
}

/// w13's rotated f16 input of every sorted slot into `s.prep`; `x` is bf16 [n][4608].
pub fn prepare(e: mtl.ComputeEncoder, p: *const Pipes, l: Layer, x: B, s: Xp, n: usize) void {
    check(l, s, n);
    e.setPipeline(p.prepare);
    e.setValue(args(l, s, n, .{ .gate = 0, .up = 0 }), 0);
    bind(e, x, 1);
    bind(e, s.order, 2);
    bind(e, s.seid, 3);
    table(e, l.w13, 4);
    bind(e, s.prep, 5);
    e.dispatchGroups(mtl.Size.of(hidden / 512, slotsOf(n), 1), mtl.Size.of(128, 1, 1));
}

/// `which`'s grouped GEMM: `s.prep`'s sorted rows by each window's expert, f32 sums into `s.plane` (w13: [slots][4608]
/// from [slots][4608]; w2: [slots][4608] from [slots][2304]). One dispatch a rate, over that rate's windows.
pub fn gemm(e: mtl.ComputeEncoder, p: *const Pipes, which: Proj, l: Layer, s: Xp, n: usize) void {
    check(l, s, n);
    const a = args(l, s, n, .{ .gate = 0, .up = 0 });
    const w = if (which == .w13) l.w13 else l.w2;
    const first: usize = if (which == .w13) 0 else 2;
    for (0..2) |rate| {
        const list = first + rate;
        e.setPipeline(p.gemm[@intFromEnum(which)][rate]);
        e.setValue(a, 0);
        bind(e, s.prep, 1);
        e.setBuffer(s.wins.buf, s.wins.off + list * winCap(s.rows) * @sizeOf(Win), 2);
        table(e, w, 3);
        bind(e, s.plane, 4);
        e.dispatchIndirect(s.dargs.buf, s.dargs.off + list * 16, mtl.Size.of(128, 1, 1));
    }
}

/// gu and act of the sorted slots, bf16 [slots][4608] and [slots][2304], for the checks.
pub const MidDbg = struct { gu: B, act: B };

/// The plane's w13 sums to the capped SwiGLU's acts and the down projection's rotated f16 input, into `s.prep`.
pub fn mid(e: mtl.ComputeEncoder, p: *const Pipes, l: Layer, s: Xp, n: usize, caps: Caps, dbg: ?MidDbg) void {
    check(l, s, n);
    e.setPipeline(if (dbg != null) p.mid_dbg else p.mid);
    e.setValue(args(l, s, n, caps), 0);
    bind(e, s.plane, 1);
    bind(e, s.seid, 2);
    table(e, l.w13, 3);
    table(e, l.w2, 4);
    bind(e, s.prep, 5);
    if (dbg) |d| {
        bind(e, d.act, 6);
        bind(e, d.gu, 7);
    }
    e.dispatchGroups(mtl.Size.of(inner / 128 / 6, slotsOf(n), 1), mtl.Size.of(192, 1, 1));
}

/// The output rotation of each pick's w2 sums and the pick-ordered weighted sum into `out` (bf16 [n][4608]); `ey`, when
/// given, receives each slot's rounded output, bf16 [slots][4608].
pub fn finish(e: mtl.ComputeEncoder, p: *const Pipes, l: Layer, picks: Picks, out: B, s: Xp, n: usize, ey: ?B) void {
    check(l, s, n);
    e.setPipeline(if (ey != null) p.finish_dbg else p.finish);
    e.setValue(args(l, s, n, .{ .gate = 0, .up = 0 }), 0);
    bind(e, s.plane, 1);
    bind(e, picks.ids, 2);
    bind(e, s.inverse, 3);
    table(e, l.w2, 4);
    bind(e, picks.weights, 5);
    bind(e, out, 6);
    if (ey) |v| bind(e, v, 7);
    e.dispatchGroups(mtl.Size.of(hidden / 128, n, 1), mtl.Size.of(256, 1, 1));
}

/// A layer's routed experts for the `n` rows of `x` (bf16 [n][4608]) with picks `picks`: the weighted sums into `out`
/// (bf16 [n][4608], the MoE output before the MLP convolution). `s` is `Xp.init` for at least `n` rows. The caller
/// declares the weights' buffers (trellis, scales, and the descriptor tables).
pub fn encode(e: mtl.ComputeEncoder, p: *const Pipes, l: Layer, x: B, picks: Picks, out: B, s: Xp, n: usize, caps: Caps) void {
    group(e, p, l, picks.ids, s, n);
    prepare(e, p, l, x, s, n);
    gemm(e, p, .w13, l, s, n);
    mid(e, p, l, s, n, caps, null);
    gemm(e, p, .w2, l, s, n);
    finish(e, p, l, picks, out, s, n, null);
}

test "xp scratch: regions are aligned, planes dominate, the down input fits where w13's rotated rows were" {
    const rows = 4096;
    const r = regions(rows);
    try std.testing.expect(r[7] >= slotsOf(rows) * inner * 2);
    try std.testing.expectEqual(@as(usize, slotsOf(rows) * hidden * 4), r[8]);
    const total = xpBytes(rows);
    try std.testing.expect(total % region_align == 0);
    try std.testing.expect(total > r[7] + r[8] and total < r[7] + r[8] + (4 << 20));
    try std.testing.expect(xpBytes(64) < xpBytes(128));
}
