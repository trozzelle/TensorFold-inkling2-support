//! Inkling 2 kernel checks: the DSpark drafter's kernels (ikd_*, their own library) against FP64 references at the same rounding
//! points, on random data at the real shapes: the tap mix, the context and block preparation (norm and rotary embedding),
//! block attention over the ring and the block's own keys (and its time at 1,031 keys), the elementwise kernels, the Markov
//! GRU and its projection, the fold's pick (a planted tie, planted ids past the unpadded vocabulary) and the fold's transpose.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const h = @import("harness.zig");
const tf = @import("tensorfold");
const dws = tf.inkling2.dspark_weights;
const Qmv = tf.inkling2.dspark_qmv.Qmv;

const H = 4608;
const D = 128;
const Hq = 48;
const KV = 8;
const GROUP = Hq / KV;
const ROWS = 8; // the block's rows
const RING = 1031; // ring slots
const PAD = 7; // a ring holds RING + PAD slots: the first PAD are written again past the last
const SLOTS = RING + PAD;
const WINDOW = 1023; // newest ring positions attended
const KEYS = 256; // keys in one attention block
const SLOT = D + 2; // floats in one (row, head, block) partial: m, l, acc[D]
const MAX_BLOCKS = 7; // the deepest ring's blocks (5 of 256 keys) plus the block's own keys, and a spare
const TAPS = 8;
const MIXES = 6;
const RANK = 256;
const V = 201024;
const VALID = 200058;
const IN = 2 * RANK + 2 * H; // the Markov projection's inputs
const THETA_LOG2: f32 = @floatCast(std.math.log2(@as(f64, 1e6)));
const NAN16: u16 = 0xffff;

// ---- the kernels' argument structs, field for field ----
const MixArgs = extern struct { rows: u32, width: u32, tap_stride: u32, mix_stride: u32, eps: f32 };
const PrepArgs = extern struct { rows: u32, heads: u32, kv: u32, pos: u32, ring: u32, eps: f32, base_log2: f32 };
const AttnArgs = extern struct { heads: u32, kv: u32, end: u32, window: u32, ring: u32, blocks: u32, scale: f32 };
const EwArgs = extern struct { rows: u32, width: u32 };
const ProjArgs = extern struct { n: u32, at: u32 };
const FoldArgs = extern struct { vocab: u32, valid: u32, flags: u32, cap: f32 };
const PickArgs = extern struct { groups: u32, step: u32 };
const TArgs = extern struct { vocab: u32 };

const Ctx = struct {
    g: *const h.Gpu,
    lib: mtl.Library,

    fn go(c: Ctx, name: [:0]const u8, slots: []const h.Slot, groups: [3]usize, threads: [3]usize) !void {
        try c.g.dispatchIn(c.lib, name, slots, &.{}, groups, threads);
    }

    /// The least GPU time, in microseconds a dispatch, of a few trials of `reps` dependent dispatches of `name` in one command
    /// buffer, printed as `what`: informational, the kernels' cost at their decode shapes.
    fn time(c: Ctx, what: []const u8, name: [:0]const u8, slots: []const h.Slot, groups: [3]usize, threads: [3]usize, reps: usize) !void {
        const p = try mtl.Pipeline.init(c.g.device, c.lib, name, false);
        defer p.deinit();
        var best: f64 = 1e9;
        for (0..5) |_| {
            const cb = c.g.queue.commandBuffer();
            const e = cb.compute(.serial);
            for (0..reps) |_| {
                e.setPipeline(p);
                for (slots, 0..) |s, i| switch (s) {
                    .buf => |b| e.setBuffer(b.buf, b.off, i),
                    .bytes => |v| e.setBytes(v, i),
                };
                e.dispatchGroups(mtl.Size.of(groups[0], groups[1], groups[2]), mtl.Size.of(threads[0], threads[1], threads[2]));
            }
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.log.err("{s} timing: {s}", .{ name, msg });
                return error.GpuFailed;
            }
            best = @min(best, cb.gpuSeconds());
        }
        std.debug.print("time {s}: {d:.2} us\n", .{ what, best / @as(f64, @floatFromInt(reps)) * 1e6 });
    }
};

fn bytes(a: anytype) h.Slot {
    return .{ .bytes = std.mem.asBytes(a) };
}
fn bs(x: mtl.Buffer) h.Slot {
    return .{ .buf = .{ .buf = x } };
}
fn bo(x: mtl.Buffer, off: usize) h.Slot {
    return .{ .buf = .{ .buf = x, .off = off } };
}

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    const lib = try g.library(ks.inkling2_dspark);
    defer lib.deinit();
    const c: Ctx = .{ .g = g, .lib = lib };
    try caseMix(c, r);
    try caseCtxPrep(c, r);
    try caseBlockPrep(c, r);
    try caseAttn(c, r);
    try caseElementwise(c, r);
    try caseProj(c, r);
    try caseGru(c, r);
    try caseFold(c, r);
    try caseFoldT(c, r);
    try caseQmv(c, r);
    try timeAttn(c, r);
}

// ---- reference pieces ----

/// A value whose f32 rounding lies within a few f32 steps of a bf16 tie: the kernel's f32 and the reference's f64 may round it
/// to different bf16 values, so outputs that depend on it are not judged.
fn ambiguous(x: f64) bool {
    const f: f32 = @floatCast(x);
    if (!std.math.isFinite(f) or f == 0) return false;
    const low: i32 = @intCast(@as(u32, @bitCast(f)) & 0xffff);
    return @abs(low - 0x8000) <= 24;
}

/// A bf16 output equal to its reference's rounding, counted when it is not: a kernel that rounds once too many or too few
/// differs from the reference in a large share of its outputs.
fn pinned(c: *h.Check, off: *usize, got: u16, want: f64, steps: u32) void {
    c.b16n(got, want, steps);
    if (got != h.bf(want)) off.* += 1;
}

/// `pinned` with an absolute allowance for a sum that cancels toward zero: its f32 error is that of its terms, not of its value.
fn pinnedAbs(c: *h.Check, off: *usize, got: u16, want: f64, abs: f64) void {
    close(c, got, want, 1, abs);
    if (got != h.bf(want)) off.* += 1;
}

/// Check.b16a, failing on a NaN output (the comparison against the tolerance is false for a NaN, so it would pass).
fn close(c: *h.Check, got: u16, want: f64, steps: u32, abs: f64) void {
    if (std.math.isNan(h.fb(got))) {
        c.n += 1;
        c.bad += 1;
        if (c.first == null) c.first = .{ std.math.nan(f64), want };
        return;
    }
    c.b16a(got, want, steps, abs);
}

/// Check.f, failing on a NaN output.
fn closeF(c: *h.Check, got: f32, want: f64, rel: f64) void {
    if (std.math.isNan(got)) {
        c.n += 1;
        c.bad += 1;
        return;
    }
    c.f(got, want, rel);
}

/// The share of outputs that may differ from the reference's roundings (an f32 sum and an f64 one can split a rounding): up to
/// 0.2% and a few more, where a rounding point wrong by one differs in about a quarter of them.
fn pin(c: *h.Check, off: usize, of: usize) void {
    c.n += 1;
    if (off > of / 500 + 8) {
        c.bad += 1;
        std.debug.print("  {s}: {d} of {d} outputs differ from the reference's roundings\n", .{ c.name, off, of });
    }
}

fn softmaxBf16(r: std.Random, out: []u16) void {
    var e: [TAPS]f64 = undefined;
    var z: f64 = 0;
    for (&e) |*x| {
        x.* = @exp(r.float(f64) * 3 - 1.5);
        z += x.*;
    }
    for (out, e) |*o, x| o.* = h.bf(x / z);
}

// ---- ikd_mix ----

/// The tap mix against FP64: the f32 sum over the taps in index order, the f32 norm, one bf16 rounding; a NaN tap of one row
/// stays in that row; the result does not depend on the threads a group uses.
fn caseMix(c: Ctx, r: std.Random) !void {
    const g = c.g;
    const rows = 3;
    const taps = try g.buf(rows * TAPS * H * 2);
    const alpha = try g.buf(MIXES * TAPS * 2);
    const out = try g.buf(MIXES * rows * H * 2);
    var wb: [MIXES]mtl.Buffer = undefined;
    for (&wb) |*w| w.* = try g.buf(H * 2);
    defer {
        for ([_]mtl.Buffer{ taps, alpha, out }) |b| b.deinit();
        for (wb) |b| b.deinit();
    }
    const tv = h.fill16(taps, rows * TAPS * H, r, 3.0);
    const av = alpha.slice(u16, MIXES * TAPS);
    for (0..MIXES) |m| softmaxBf16(r, av[m * TAPS ..][0..TAPS]);
    var wv: [MIXES][]u16 = undefined;
    for (&wv, wb) |*s, w| s.* = h.fill16(w, H, r, 1.5);
    const a: MixArgs = .{ .rows = rows, .width = H, .tap_stride = TAPS * H, .mix_stride = rows * H, .eps = 1e-6 };
    const slots = [_]h.Slot{ bytes(&a), bs(taps), bs(alpha), bs(wb[0]), bs(wb[1]), bs(wb[2]), bs(wb[3]), bs(wb[4]), bs(wb[5]), bs(out) };
    var ch: h.Check = .{ .name = "mix" };
    var off: usize = 0;
    const saved = try std.heap.page_allocator.alloc(u8, MIXES * rows * H * 2);
    defer std.heap.page_allocator.free(saved);
    for ([_]usize{ 288, 576 }) |nt| {
        @memset(out.slice(u8, MIXES * rows * H * 2), 0xff);
        try c.go("ikd_mix", &slots, .{ rows, MIXES, 1 }, .{ nt, 1, 1 });
        const ys = out.slice(u16, MIXES * rows * H);
        for (0..MIXES) |m| for (0..rows) |t| {
            var x: [H]f64 = undefined;
            var mag: [H]f64 = undefined;
            var ss: f64 = 0;
            for (0..H) |i| {
                var s: f64 = 0;
                var a_s: f64 = 0;
                for (0..TAPS) |j| {
                    const term = h.fb(av[m * TAPS + j]) * h.fb(tv[(t * TAPS + j) * H + i]);
                    s += term;
                    a_s += @abs(term);
                }
                x[i] = s;
                mag[i] = a_s;
                ss += s * s;
            }
            const inv = 1 / @sqrt(ss / H + 1e-6);
            for (0..H) |i| {
                const w = h.fb(wv[m][i]);
                pinnedAbs(&ch, &off, ys[(m * rows + t) * H + i], w * (x[i] * inv), @abs(w) * inv * mag[i] * 4e-7);
            }
        };
        if (nt == 288) @memcpy(saved, out.slice(u8, MIXES * rows * H * 2));
    }
    pin(&ch, off, 2 * MIXES * rows * H);
    // a NaN in row 1's taps stays in row 1
    tv[(1 * TAPS + 3) * H + 17] = 0x7fc0;
    try c.go("ikd_mix", &slots, .{ rows, MIXES, 1 }, .{ 288, 1, 1 });
    for (0..MIXES) |m| for ([_]usize{ 0, 2 }) |t| {
        const at = (m * rows + t) * H * 2;
        ch.same(out.slice(u8, MIXES * rows * H * 2)[at..][0 .. H * 2], saved[at..][0 .. H * 2]);
    };
    try ch.done();
    try c.time("ikd_mix, 3 rows of 6 mixes", "ikd_mix", &slots, .{ rows, MIXES, 1 }, .{ 288, 1, 1 }, 100);
}

// ---- ikd_ctx_prep, ikd_block_prep ----

/// One head's reference: the weighted RMSNorm of the bf16 values (the normalized value rounds to bf16 before the weight, the
/// product rounds again), then the rotation at `pos` over pairs (i, i + 64). `skip[d]` marks outputs that depend on a
/// normalized value within an f32 step of a bf16 tie. Returns the largest magnitude among the normalized values.
fn headRef(x: []const u16, w: []const u16, pos: usize, out: *[D]f64, skip: *[D]bool) f64 {
    var ss: f64 = 0;
    for (x) |v| ss += h.fb(v) * h.fb(v);
    const inv = 1 / @sqrt(ss / D + 1e-6);
    var n: [D]f64 = undefined;
    var amb: [D]bool = undefined;
    var big: f64 = 0;
    for (0..D) |d| {
        const u = h.fb(x[d]) * inv;
        amb[d] = ambiguous(u);
        n[d] = h.rb(h.fb(w[d]) * h.rb(u));
        big = @max(big, @abs(n[d]));
    }
    for (0..D / 2) |i| {
        const freq = std.math.pow(f64, 1e6, -@as(f64, @floatFromInt(i)) / 64.0);
        const th = @as(f64, @floatFromInt(pos)) * freq;
        out[i] = n[i] * @cos(th) - n[i + 64] * @sin(th);
        out[i + 64] = n[i] * @sin(th) + n[i + 64] * @cos(th);
        skip[i] = amb[i] or amb[i + 64];
        skip[i + 64] = skip[i];
    }
    return big;
}

/// The rotation's tolerance: the angle pos * freq is formed in f32, and the fast sine and cosine add their own error, so the
/// rotated values carry an absolute error that grows with the position (and the normalized values' size).
fn ropeTol(pos: usize, big: f64) f64 {
    return big * (2e-5 + 4e-7 * @as(f64, @floatFromInt(pos)));
}

fn prepArgs(rows: usize, pos: usize) PrepArgs {
    return .{ .rows = @intCast(rows), .heads = Hq, .kv = KV, .pos = @intCast(pos), .ring = RING, .eps = 1e-6, .base_log2 = THETA_LOG2 };
}

/// Context rows into the ring: k_norm_ctx and the rotation at the row's absolute position, v copied, each in slot
/// position % 1031 (a slot below 7 also in the slot 1031 past it), no other slot touched. Row 0 of the first run sits at
/// position 0, where the rotation is the identity: its keys must equal the norm's roundings. Positions at the ring's wrap, at
/// a large offset, and a prompt's worth of rows.
fn caseCtxPrep(c: Ctx, r: std.Random) !void {
    const g = c.g;
    const max_rows = WINDOW;
    const kc = try g.buf(max_rows * KV * D * 2);
    const vc = try g.buf(max_rows * KV * D * 2);
    const w = try g.buf(D * 2);
    const rk = try g.buf(SLOTS * KV * D * 2);
    const rv = try g.buf(SLOTS * KV * D * 2);
    defer for ([_]mtl.Buffer{ kc, vc, w, rk, rv }) |b| b.deinit();
    const kv = h.fill16(kc, max_rows * KV * D, r, 2.5);
    const vv = h.fill16(vc, max_rows * KV * D, r, 2.5);
    const wv = h.fill16(w, D, r, 1.5);
    var ch: h.Check = .{ .name = "ctx_prep" };
    var off: usize = 0;
    var exact: usize = 0;
    for ([_][2]usize{ .{ 0, 5 }, .{ 1029, 5 }, .{ 5000, 8 }, .{ 1030, 1 }, .{ 3000, max_rows } }) |cs| {
        const pos0, const rows = cs;
        @memset(rk.slice(u16, SLOTS * KV * D), NAN16);
        @memset(rv.slice(u16, SLOTS * KV * D), NAN16);
        const a = prepArgs(rows, pos0);
        try c.go("ikd_ctx_prep", &.{ bytes(&a), bs(kc), bs(vc), bs(w), bs(rk), bs(rv) }, .{ rows, 1, 1 }, .{ 256, 1, 1 });
        const ks_ = rk.slice(u16, SLOTS * KV * D);
        const vs = rv.slice(u16, SLOTS * KV * D);
        for (0..rows) |t| for (0..KV) |hd| {
            const slot = (pos0 + t) % RING;
            var want: [D]f64 = undefined;
            var skip: [D]bool = undefined;
            const big = headRef(kv[(t * KV + hd) * D ..][0..D], wv, pos0 + t, &want, &skip);
            const tol = ropeTol(pos0 + t, big);
            for (0..D) |d| {
                const got = ks_[(slot * KV + hd) * D + d];
                if (!skip[d]) {
                    if (pos0 + t == 0) {
                        pinned(&ch, &off, got, want[d], 1);
                        exact += 1;
                    } else close(&ch, got, want[d], 1, tol);
                }
                ch.b16n(vs[(slot * KV + hd) * D + d], h.fb(vv[(t * KV + hd) * D + d]), 0);
                if (slot < PAD) {
                    ch.same(std.mem.asBytes(&ks_[((slot + RING) * KV + hd) * D + d]), std.mem.asBytes(&ks_[(slot * KV + hd) * D + d]));
                    ch.same(std.mem.asBytes(&vs[((slot + RING) * KV + hd) * D + d]), std.mem.asBytes(&vs[(slot * KV + hd) * D + d]));
                }
            }
        };
        // every other slot is untouched
        for (0..SLOTS) |slot| {
            const first = if (slot >= RING) slot - RING else slot;
            const used = (first + RING * 8 - pos0 % RING) % RING < rows;
            if (used) continue;
            ch.n += 1;
            for (ks_[slot * KV * D ..][0 .. KV * D]) |x| if (x != NAN16) {
                ch.bad += 1;
                break;
            };
            for (vs[slot * KV * D ..][0 .. KV * D]) |x| if (x != NAN16) {
                ch.bad += 1;
                break;
            };
        }
    }
    pin(&ch, off, exact);
    try ch.done();
    const ta = prepArgs(4, 5000);
    try c.time("ikd_ctx_prep, 4 rows", "ikd_ctx_prep", &.{ bytes(&ta), bs(kc), bs(vc), bs(w), bs(rk), bs(rv) }, .{ 4, 1, 1 }, .{ 256, 1, 1 }, 100);
}

/// The block's rows at positions P + r: q_norm and k_norm, the rotation, v copied. Row 0 of the first run sits at position 0,
/// where the rotation is the identity: its q and k must equal the norm's roundings.
fn caseBlockPrep(c: Ctx, r: std.Random) !void {
    const g = c.g;
    const q = try g.buf(ROWS * Hq * D * 2);
    const k = try g.buf(ROWS * KV * D * 2);
    const v = try g.buf(ROWS * KV * D * 2);
    const qw = try g.buf(D * 2);
    const kw = try g.buf(D * 2);
    const qn = try g.buf(ROWS * Hq * D * 2);
    const kb = try g.buf(ROWS * KV * D * 2);
    const vb = try g.buf(ROWS * KV * D * 2);
    defer for ([_]mtl.Buffer{ q, k, v, qw, kw, qn, kb, vb }) |b| b.deinit();
    const qv = h.fill16(q, ROWS * Hq * D, r, 2.5);
    const kv = h.fill16(k, ROWS * KV * D, r, 2.5);
    const vv = h.fill16(v, ROWS * KV * D, r, 2.5);
    const qwv = h.fill16(qw, D, r, 1.5);
    const kwv = h.fill16(kw, D, r, 1.5);
    var ch: h.Check = .{ .name = "block_prep" };
    var off: usize = 0;
    var exact: usize = 0;
    for ([_]usize{ 0, 1023, 5000 }) |pos| {
        @memset(qn.slice(u16, ROWS * Hq * D), NAN16);
        @memset(kb.slice(u16, ROWS * KV * D), NAN16);
        @memset(vb.slice(u16, ROWS * KV * D), NAN16);
        const a = prepArgs(ROWS, pos);
        try c.go("ikd_block_prep", &.{ bytes(&a), bs(q), bs(k), bs(v), bs(qw), bs(kw), bs(qn), bs(kb), bs(vb) }, .{ (Hq + KV + 7) / 8, ROWS, 1 }, .{ 256, 1, 1 });
        const qs = qn.slice(u16, ROWS * Hq * D);
        const ksl = kb.slice(u16, ROWS * KV * D);
        const vs = vb.slice(u16, ROWS * KV * D);
        for (0..ROWS) |t| {
            var want: [D]f64 = undefined;
            var skip: [D]bool = undefined;
            for (0..Hq) |hd| {
                const big = headRef(qv[(t * Hq + hd) * D ..][0..D], qwv, pos + t, &want, &skip);
                for (0..D) |d| if (!skip[d]) {
                    const got = qs[(t * Hq + hd) * D + d];
                    if (pos + t == 0) {
                        pinned(&ch, &off, got, want[d], 1);
                        exact += 1;
                    } else close(&ch, got, want[d], 1, ropeTol(pos + t, big));
                };
            }
            for (0..KV) |hd| {
                const big = headRef(kv[(t * KV + hd) * D ..][0..D], kwv, pos + t, &want, &skip);
                for (0..D) |d| {
                    if (!skip[d]) {
                        const got = ksl[(t * KV + hd) * D + d];
                        if (pos + t == 0) {
                            pinned(&ch, &off, got, want[d], 1);
                            exact += 1;
                        } else close(&ch, got, want[d], 1, ropeTol(pos + t, big));
                    }
                    ch.b16n(vs[(t * KV + hd) * D + d], h.fb(vv[(t * KV + hd) * D + d]), 0);
                }
            }
        }
    }
    pin(&ch, off, exact);
    try ch.done();
    const ta = prepArgs(ROWS, 5000);
    try c.time("ikd_block_prep, 8 rows", "ikd_block_prep", &.{ bytes(&ta), bs(q), bs(k), bs(v), bs(qw), bs(kw), bs(qn), bs(kb), bs(vb) }, .{ (Hq + KV + 7) / 8, ROWS, 1 }, .{ 256, 1, 1 }, 100);
}

// ---- ikd_attn_part, ikd_attn_merge ----

/// The ring blocks' first index and the blocks per head (the ring's, then the block's own keys) for a block at `end`.
fn blocksOf(end: u32) struct { lo: u32, n: u32 } {
    const lo: u32 = if (end > WINDOW) end - WINDOW else 0;
    if (end == lo) return .{ .lo = lo, .n = 1 };
    return .{ .lo = lo, .n = (end - 1) / KEYS - lo / KEYS + 2 };
}

fn attnArgs(end: u32) AttnArgs {
    return .{ .heads = Hq, .kv = KV, .end = end, .window = WINDOW, .ring = RING, .blocks = blocksOf(end).n, .scale = @floatCast(1 / @sqrt(@as(f64, D))) };
}

/// A ring's host view with its first PAD slots copied past the last, as the context kernel keeps them.
fn mirrored(s: []u16) []u16 {
    @memcpy(s[RING * KV * D ..][0 .. PAD * KV * D], s[0 .. PAD * KV * D]);
    return s;
}

const Attn = struct {
    q: mtl.Buffer,
    rk: mtl.Buffer,
    rv: mtl.Buffer,
    bk: mtl.Buffer,
    bv: mtl.Buffer,
    part: mtl.Buffer,
    att: mtl.Buffer,
    qv: []u16,
    rkv: []u16,
    rvv: []u16,
    bkv: []u16,
    bvv: []u16,
    orig_k: []u16, // the rings as filled: `stale` restores the attended slots from them
    orig_v: []u16,

    /// Scores spread over several units, so blocks differ in their maxima and the merge's rescaling matters.
    fn init(g: *const h.Gpu, r: std.Random) !Attn {
        const q = try g.buf(ROWS * Hq * D * 2);
        const rk = try g.buf(SLOTS * KV * D * 2);
        const rv = try g.buf(SLOTS * KV * D * 2);
        const bk = try g.buf(ROWS * KV * D * 2);
        const bv = try g.buf(ROWS * KV * D * 2);
        const rkv = mirrored(h.fill16(rk, SLOTS * KV * D, r, 2.5));
        const rvv = mirrored(h.fill16(rv, SLOTS * KV * D, r, 1.0));
        return .{
            .q = q,
            .rk = rk,
            .rv = rv,
            .bk = bk,
            .bv = bv,
            .part = try g.buf(ROWS * Hq * MAX_BLOCKS * SLOT * 4),
            .att = try g.buf(ROWS * Hq * D * 2),
            .qv = h.fill16(q, ROWS * Hq * D, r, 2.5),
            .rkv = rkv,
            .rvv = rvv,
            .bkv = h.fill16(bk, ROWS * KV * D, r, 2.5),
            .bvv = h.fill16(bv, ROWS * KV * D, r, 1.0),
            .orig_k = try std.heap.page_allocator.dupe(u16, rkv),
            .orig_v = try std.heap.page_allocator.dupe(u16, rvv),
        };
    }

    fn deinit(x: Attn) void {
        for ([_]mtl.Buffer{ x.q, x.rk, x.rv, x.bk, x.bv, x.part, x.att }) |b| b.deinit();
        std.heap.page_allocator.free(x.orig_k);
        std.heap.page_allocator.free(x.orig_v);
    }

    /// The rings hold the attended slots of a block at `end` as filled; every other slot's keys are the largest finite bf16
    /// (their scores overflow) and its values NaN, mirrored slots included: the kernel reads whole tiles of 8 slots, so keys and
    /// values past the attended ones must not reach the result.
    fn stale(x: Attn, end: u32) void {
        var live: [RING]bool = @splat(false);
        for (blocksOf(end).lo..end) |p| live[p % RING] = true;
        for (0..RING) |s| for ([_][]u16{ x.rkv, x.rvv }, [_][]const u16{ x.orig_k, x.orig_v }, [_]u16{ 0x7f7f, NAN16 }) |dst, src, poison| {
            const row = s * KV * D;
            if (live[s]) @memcpy(dst[row..][0 .. KV * D], src[row..][0 .. KV * D]) else @memset(dst[row..][0 .. KV * D], poison);
        };
        _ = mirrored(x.rkv);
        _ = mirrored(x.rvv);
    }

    /// The partials and the output start as NaN, so anything the kernels leave unwritten shows.
    fn attend(x: Attn, c: Ctx, end: u32) !void {
        const a = attnArgs(end);
        @memset(x.part.slice(f32, ROWS * Hq * a.blocks * SLOT), std.math.nan(f32));
        @memset(x.att.slice(u16, ROWS * Hq * D), NAN16);
        try c.go("ikd_attn_part", &.{ bytes(&a), bs(x.q), bs(x.rk), bs(x.rv), bs(x.bk), bs(x.bv), bs(x.part) }, .{ KV, 1, a.blocks }, .{ 192, 1, 1 });
        try c.go("ikd_attn_merge", &.{ bytes(&a), bs(x.part), bs(x.att) }, .{ Hq, ROWS, 1 }, .{ 32, 1, 1 });
    }
};

/// The block's attention against an FP64 softmax over the attended ring positions [max(P - 1023, 0), P) (slot position % 1031)
/// and the block's 8 keys, which every row sees: rings of 0, 255, 256, 1,000 and 1,023 keys, and past the ring's first laps,
/// where the window starts inside a block, the slots wrap (inside a tile of 8, at several alignments) and the block's first
/// key sits at any slot. The same run twice gives the same bytes.
fn caseAttn(c: Ctx, r: std.Random) !void {
    const x = try Attn.init(c.g, r);
    defer x.deinit();
    const alloc = std.heap.page_allocator;
    const kf = try alloc.alloc(f64, RING * KV * D);
    const vf = try alloc.alloc(f64, RING * KV * D);
    const qf = try alloc.alloc(f64, ROWS * Hq * D);
    const bkf = try alloc.alloc(f64, ROWS * KV * D);
    const bvf = try alloc.alloc(f64, ROWS * KV * D);
    defer for ([_][]f64{ kf, vf, qf, bkf, bvf }) |s| alloc.free(s);
    for (kf, x.rkv[0 .. RING * KV * D]) |*o, v| o.* = h.fb(v);
    for (vf, x.rvv[0 .. RING * KV * D]) |*o, v| o.* = h.fb(v);
    for (qf, x.qv) |*o, v| o.* = h.fb(v);
    for (bkf, x.bkv) |*o, v| o.* = h.fb(v);
    for (bvf, x.bvv) |*o, v| o.* = h.fb(v);
    const scale = 1 / @sqrt(@as(f64, D));
    var scores: [RING + ROWS]f64 = undefined;
    var ch: h.Check = .{ .name = "attention" };
    const first = try alloc.alloc(u8, ROWS * Hq * D * 2);
    defer alloc.free(first);
    for ([_]u32{ 0, 1, 7, 8, 63, 64, 65, 255, 256, 257, 1000, 1023, 1024, 1500, 2062, 3000, 3500, 4100, 4200, 5000, 5001, 6000, 7000, 9000 }) |end| {
        const bad = ch.bad;
        x.stale(end);
        try x.attend(c, end);
        @memcpy(first, x.att.slice(u8, ROWS * Hq * D * 2));
        const att = x.att.slice(u16, ROWS * Hq * D);
        const lo = blocksOf(end).lo;
        for (0..ROWS) |t| for (0..Hq) |hd| {
            const g = hd / GROUP;
            const qh = qf[(t * Hq + hd) * D ..][0..D];
            var n: usize = 0;
            var m: f64 = -std.math.inf(f64);
            for (lo..end) |p| {
                const kr = kf[((p % RING) * KV + g) * D ..][0..D];
                var s: f64 = 0;
                for (qh, kr) |qd, kd| s += qd * kd;
                scores[n] = s * scale;
                m = @max(m, scores[n]);
                n += 1;
            }
            for (0..ROWS) |j| {
                const kr = bkf[(j * KV + g) * D ..][0..D];
                var s: f64 = 0;
                for (qh, kr) |qd, kd| s += qd * kd;
                scores[n] = s * scale;
                m = @max(m, scores[n]);
                n += 1;
            }
            var z: f64 = 0;
            var o: [D]f64 = @splat(0);
            n = 0;
            for (lo..end) |p| {
                const w = @exp(scores[n] - m);
                n += 1;
                z += w;
                const vr = vf[((p % RING) * KV + g) * D ..][0..D];
                for (&o, vr) |*od, vd| od.* += w * vd;
            }
            for (0..ROWS) |j| {
                const w = @exp(scores[n] - m);
                n += 1;
                z += w;
                const vr = bvf[(j * KV + g) * D ..][0..D];
                for (&o, vr) |*od, vd| od.* += w * vd;
            }
            // two rounding points (the output, then nothing: bf16 of the quotient), and outputs near zero keep the f32 sums' absolute error
            for (0..D) |d| close(&ch, att[(t * Hq + hd) * D + d], o[d] / z, 2, 1e-6);
        };
        try x.attend(c, end);
        ch.same(x.att.slice(u8, ROWS * Hq * D * 2), first);
        if (ch.bad != bad) std.debug.print("  failing: block at {d}\n", .{end});
    }
    try ch.done();
}

/// The attention kernels' time at the drafter's real shape, a full ring of 1,023 keys plus the block's 8: the least GPU time of
/// a few trials of `reps` dependent runs in one command buffer, microseconds a run of the partial kernel and the merge
/// (`part`, `merge` choose which run). Prints the pair's time a layer, and each kernel's.
fn timeAttn(c: Ctx, r: std.Random) !void {
    const x = try Attn.init(c.g, r);
    defer x.deinit();
    const a = attnArgs(5000);
    const both = try timeKernels(c, x, a, true, true);
    const part = try timeKernels(c, x, a, true, false);
    const merge = try timeKernels(c, x, a, false, true);
    std.debug.print("ikd_attn_part + ikd_attn_merge, {d} keys, {d} rows: {d:.2} us a layer (part {d:.2}, merge {d:.2})\n", .{ WINDOW + ROWS, ROWS, both, part, merge });
    // a bound against a lost parallelism or a serial fallback (the serial per-key loop takes about 800 us a layer at this length),
    // loose enough for a busy GPU
    var ch: h.Check = .{ .name = "attention time (us a layer)", .n = 1, .worst = both };
    if (both > 100) ch.bad = 1;
    try ch.done();
}

fn timeKernels(c: Ctx, x: Attn, a: AttnArgs, part: bool, merge: bool) !f64 {
    const reps = 200;
    const pp = try mtl.Pipeline.init(c.g.device, c.lib, "ikd_attn_part", false);
    defer pp.deinit();
    const pm = try mtl.Pipeline.init(c.g.device, c.lib, "ikd_attn_merge", false);
    defer pm.deinit();
    var best: f64 = 1e9;
    for (0..6) |_| {
        const cb = c.g.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..reps) |_| {
            if (part) {
                e.setPipeline(pp);
                e.setValue(a, 0);
                for ([_]mtl.Buffer{ x.q, x.rk, x.rv, x.bk, x.bv, x.part }, 1..) |b, i| e.setBuffer(b, 0, i);
                e.dispatchGroups(mtl.Size.of(KV, 1, a.blocks), mtl.Size.of(192, 1, 1));
            }
            if (merge) {
                e.setPipeline(pm);
                e.setValue(a, 0);
                e.setBuffer(x.part, 0, 1);
                e.setBuffer(x.att, 0, 2);
                e.dispatchGroups(mtl.Size.of(Hq, ROWS, 1), mtl.Size.of(32, 1, 1));
            }
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("attention timing: {s}", .{msg});
            return error.GpuFailed;
        }
        best = @min(best, cb.gpuSeconds());
    }
    return best / reps * 1e6;
}

// ---- elementwise ----

/// ikd_swiglu against the reference as mlx computes it in bf16 (every op of silu rounds to bf16, then the product with up),
/// ikd_add against bf16(a + b), exactly.
fn caseElementwise(c: Ctx, r: std.Random) !void {
    const g = c.g;
    const rows = 8;
    const width = 18432;
    const gt = try g.buf(rows * width * 2);
    const up = try g.buf(rows * width * 2);
    const y = try g.buf(rows * width * 2);
    defer for ([_]mtl.Buffer{ gt, up, y }) |b| b.deinit();
    const gv = h.fill16(gt, rows * width, r, 9.0);
    const uv = h.fill16(up, rows * width, r, 2.0);
    // saturated and special gates: large positive and negative, zero
    gv[0] = h.bf(0);
    gv[1] = h.bf(-0.0);
    gv[2] = h.bf(200);
    gv[3] = h.bf(-200);
    const a: EwArgs = .{ .rows = rows, .width = width };
    var ch: h.Check = .{ .name = "swiglu" };
    var off: usize = 0;
    try c.go("ikd_swiglu", &.{ bytes(&a), bs(gt), bs(up), bs(y) }, .{ (width + 255) / 256, rows, 1 }, .{ 256, 1, 1 });
    for (y.slice(u16, rows * width), gv, uv) |got, gg, uu| {
        const x = h.fb(gg);
        const e = @exp(@abs(x));
        const d = h.rb(1 + h.rb(e));
        if (ambiguous(e) or ambiguous(1 / d)) {
            ch.n += 1;
            continue;
        }
        const inv = h.rb(1 / d);
        const sig = if (x < 0) inv else h.rb(1 - inv);
        pinned(&ch, &off, got, h.rb(x * sig) * h.fb(uu), 1);
    }
    pin(&ch, off, rows * width);
    try ch.done();
    try c.time("ikd_swiglu, 8 rows of 18432", "ikd_swiglu", &.{ bytes(&a), bs(gt), bs(up), bs(y) }, .{ (width + 255) / 256, rows, 1 }, .{ 256, 1, 1 }, 100);
    ch = .{ .name = "add" };
    try c.go("ikd_add", &.{ bytes(&a), bs(gt), bs(up), bs(y) }, .{ (width + 255) / 256, rows, 1 }, .{ 256, 1, 1 });
    for (y.slice(u16, rows * width), gv, uv) |got, gg, uu| ch.b16n(got, h.fb(gg) + h.fb(uu), 0);
    // in place: y is the first operand
    const first = try std.heap.page_allocator.dupe(u16, y.slice(u16, rows * width));
    defer std.heap.page_allocator.free(first);
    @memcpy(y.slice(u16, rows * width), gv);
    try c.go("ikd_add", &.{ bytes(&a), bs(y), bs(up), bs(y) }, .{ (width + 255) / 256, rows, 1 }, .{ 256, 1, 1 });
    ch.same(std.mem.sliceAsBytes(y.slice(u16, rows * width)), std.mem.sliceAsBytes(first));
    try ch.done();
    const ra: EwArgs = .{ .rows = rows, .width = H };
    try c.time("ikd_add, 8 rows of 4608", "ikd_add", &.{ bytes(&ra), bs(gt), bs(up), bs(y) }, .{ H / 256, rows, 1 }, .{ 256, 1, 1 }, 100);
}

// ---- ikd_rnn_proj, ikd_gru ----

/// The Markov projection against FP64 over its four-part input, one rounding of the sum and the bias. The weight and the
/// embedding table sit 8 bytes into their buffers, as checkpoint tensors do.
fn caseProj(c: Ctx, r: std.Random) !void {
    const g = c.g;
    const n = 3 * RANK;
    const tokens = 40;
    const w = try g.buf(n * IN * 2 + 16);
    const bias = try g.buf(n * 2);
    const table = try g.buf(tokens * RANK * 2 + 16);
    const state = try g.buf(RANK * 2);
    const hid = try g.buf(H * 2);
    const frnn = try g.buf(H * 2);
    const seq = try g.buf(8 * 4);
    const proj = try g.buf(n * 2);
    defer for ([_]mtl.Buffer{ w, bias, table, state, hid, frnn, seq, proj }) |b| b.deinit();
    const wv = h.fill16(w, n * IN + 8, r, 0.15)[4..];
    const bv = h.fill16(bias, n, r, 1.0);
    const tv = h.fill16(table, tokens * RANK + 8, r, 1.0)[4..];
    const sv = h.fill16(state, RANK, r, 1.0);
    const hv = h.fill16(hid, H, r, 2.0);
    const fv = h.fill16(frnn, H, r, 2.0);
    const sq = seq.slice(u32, 8);
    @memset(sq, 0);
    sq[3] = 37;
    const a: ProjArgs = .{ .n = n, .at = 3 };
    try c.go("ikd_rnn_proj", &.{ bytes(&a), bs(state), bo(table, 8), bs(seq), bs(hid), bs(frnn), bo(w, 8), bs(bias), bs(proj) }, .{ n / 8, 1, 1 }, .{ 256, 1, 1 });
    var ch: h.Check = .{ .name = "rnn_proj" };
    var off: usize = 0;
    var x: [IN]f64 = undefined;
    for (0..RANK) |i| x[i] = h.fb(sv[i]);
    for (0..RANK) |i| x[RANK + i] = h.fb(tv[37 * RANK + i]);
    for (0..H) |i| x[2 * RANK + i] = h.fb(hv[i]);
    for (0..H) |i| x[2 * RANK + H + i] = h.fb(fv[i]);
    for (0..n) |j| {
        var s: f64 = 0;
        var mag: f64 = @abs(h.fb(bv[j]));
        for (0..IN) |k| {
            const term = h.fb(wv[j * IN + k]) * x[k];
            s += term;
            mag += @abs(term);
        }
        pinnedAbs(&ch, &off, proj.slice(u16, n)[j], s + h.fb(bv[j]), mag * 4e-6);
    }
    pin(&ch, off, n);
    try ch.done();
    try c.time("ikd_rnn_proj", "ikd_rnn_proj", &.{ bytes(&a), bs(state), bo(table, 8), bs(seq), bs(hid), bs(frnn), bo(w, 8), bs(bias), bs(proj) }, .{ n / 8, 1, 1 }, .{ 256, 1, 1 }, 100);
}

/// One GRU step: sigmoid and tanh in f32, every product and sum rounded to bf16; the state may be updated in place.
fn caseGru(c: Ctx, r: std.Random) !void {
    const g = c.g;
    const proj = try g.buf(3 * RANK * 2);
    const state = try g.buf(RANK * 2);
    const next = try g.buf(RANK * 2);
    const latent = try g.buf(RANK * 2);
    defer for ([_]mtl.Buffer{ proj, state, next, latent }) |b| b.deinit();
    var ch: h.Check = .{ .name = "gru" };
    var off: usize = 0;
    var total: usize = 0;
    for (0..8) |trial| {
        const pv = h.fill16(proj, 3 * RANK, r, 6.0);
        const sv = h.fill16(state, RANK, r, 1.0);
        const gate_scale: f64 = if (trial == 0) 4.0 else 1.0; // a saturated gate in one trial
        for (0..RANK) |i| pv[i] = h.bf(h.fb(pv[i]) * gate_scale);
        try c.go("ikd_gru", &.{ bs(proj), bs(state), bs(next), bs(latent) }, .{ 1, 1, 1 }, .{ 256, 1, 1 });
        var want: [RANK]f64 = undefined;
        var skip: [RANK]bool = undefined;
        var wlat: [RANK]f64 = undefined;
        for (0..RANK) |i| {
            const gp = h.fb(pv[i]);
            const sig = 1 / (1 + @exp(-gp));
            const gate = h.rb(sig);
            const kept = h.rb(gate * h.fb(sv[i]));
            const cand_raw = std.math.tanh(h.fb(pv[RANK + i]));
            const blend = h.rb(h.rb(1 - gate) * h.rb(cand_raw));
            want[i] = kept + blend;
            skip[i] = ambiguous(sig) or ambiguous(cand_raw);
            wlat[i] = std.math.tanh(h.fb(pv[2 * RANK + i]));
        }
        const ns = next.slice(u16, RANK);
        const ls = latent.slice(u16, RANK);
        for (0..RANK) |i| {
            total += 2;
            if (!skip[i]) pinned(&ch, &off, ns[i], want[i], 1) else ch.n += 1;
            pinned(&ch, &off, ls[i], wlat[i], 1);
        }
        // in place
        const saved = try std.heap.page_allocator.dupe(u16, ns);
        defer std.heap.page_allocator.free(saved);
        try c.go("ikd_gru", &.{ bs(proj), bs(state), bs(state), bs(latent) }, .{ 1, 1, 1 }, .{ 256, 1, 1 });
        ch.same(std.mem.sliceAsBytes(state.slice(u16, RANK)), std.mem.sliceAsBytes(saved));
    }
    pin(&ch, off, total);
    try ch.done();
    try c.time("ikd_gru", "ikd_gru", &.{ bs(proj), bs(state), bs(next), bs(latent) }, .{ 1, 1, 1 }, .{ 256, 1, 1 }, 200);
}

// ---- ikd_fold_pick, ikd_pick ----

/// Values on a coarse grid (k / den, |k| <= steps), so the fold's f32 dot is exact in any order and the kernel's and the
/// reference's roundings agree.
fn fillGrid(b: mtl.Buffer, n: usize, r: std.Random, steps: i32, den: f64) []u16 {
    const s = b.slice(u16, n);
    for (s) |*x| x.* = h.bf(@as(f64, @floatFromInt(r.intRangeAtMost(i32, -steps, steps))) / den);
    return s;
}

const Fold = struct {
    base: mtl.Buffer,
    latent: mtl.Buffer,
    fw: mtl.Buffer,
    fb: mtl.Buffer,
    best: mtl.Buffer,
    best_id: mtl.Buffer,
    seq: mtl.Buffer,
    logits: mtl.Buffer,
    bv: []u16,
    lv: []u16,
    wv: []u16,
    biasv: []u16,
    groups: usize,

    fn init(g: *const h.Gpu, r: std.Random) !Fold {
        const groups = (V + 255) / 256;
        const base = try g.buf(2 * V * 2);
        const latent = try g.buf(RANK * 2);
        const fw = try g.buf(V * RANK * 2);
        const fb = try g.buf(V * 2);
        return .{
            .base = base,
            .latent = latent,
            .fw = fw,
            .fb = fb,
            .best = try g.buf(groups * 4),
            .best_id = try g.buf(groups * 4),
            .seq = try g.buf(8 * 4),
            .logits = try g.buf(V * 2),
            .bv = h.fill16(base, 2 * V, r, 12.0),
            .lv = fillGrid(latent, RANK, r, 16, 16),
            .wv = fillGrid(fw, V * RANK, r, 127, 1024),
            .biasv = h.fill16(fb, V, r, 0.5),
            .groups = groups,
        };
    }

    fn deinit(x: Fold) void {
        for ([_]mtl.Buffer{ x.base, x.latent, x.fw, x.fb, x.best, x.best_id, x.seq, x.logits }) |b| b.deinit();
    }

    /// The pair of kernels at `step` (its base row); returns nothing, the proposal lands in seq[step + 1].
    fn pick(x: Fold, c: Ctx, step: u32) !void {
        const fa: FoldArgs = .{ .vocab = V, .valid = VALID, .flags = 1, .cap = 30.0 };
        try c.go("ikd_fold_pick", &.{ bytes(&fa), bo(x.base, step * V * 2), bs(x.latent), bs(x.fw), bs(x.fb), bs(x.best), bs(x.best_id), bs(x.logits) }, .{ x.groups, 1, 1 }, .{ 256, 1, 1 });
        const pa: PickArgs = .{ .groups = @intCast(x.groups), .step = step };
        try c.go("ikd_pick", &.{ bytes(&pa), bs(x.best), bs(x.best_id), bs(x.seq) }, .{ 1, 1, 1 }, .{ 1024, 1, 1 });
    }

    /// The step's logit of every id in FP64 at the kernel's rounding points, -inf past the unpadded vocabulary.
    fn logit(x: Fold, step: u32, v: usize) f64 {
        if (v >= VALID) return -std.math.inf(f64);
        var dot: f64 = 0;
        for (0..RANK) |cc| dot += h.fb(x.lv[cc]) * h.fb(x.wv[v * RANK + cc]);
        const bias = h.rb(h.rb(dot) + h.fb(x.biasv[v]));
        return 30 * std.math.tanh((h.fb(x.bv[step * V + v]) + bias) / 30);
    }
};

/// The fold's pick against FP64: every group's best and the proposal on random data (within a near tie of the reference's),
/// every logit within a bf16 step; then planted cases that must be exact: a tie across groups and one within a group (the lower
/// id wins), and ids past 200,058 with the largest values (masked), the last valid id the winner.
fn caseFold(c: Ctx, r: std.Random) !void {
    const x = try Fold.init(c.g, r);
    defer x.deinit();
    const alloc = std.heap.page_allocator;
    const ref = try alloc.alloc(f64, V);
    defer alloc.free(ref);
    var ch: h.Check = .{ .name = "fold pick" };
    var off: usize = 0;
    const near = 5e-6;
    for ([_]u32{ 0, 1 }) |step| {
        try x.pick(c, step);
        var top: f64 = -std.math.inf(f64);
        var top_id: usize = 0;
        for (ref, 0..) |*o, v| {
            o.* = x.logit(step, v);
            if (o.* > top) {
                top = o.*;
                top_id = v;
            }
        }
        const lg = x.logits.slice(u16, V);
        for (ref, lg) |want, got| pinned(&ch, &off, got, want, 1);
        const best = x.best.slice(f32, x.groups);
        const ids = x.best_id.slice(u32, x.groups);
        for (0..x.groups) |gi| {
            var gtop: f64 = -std.math.inf(f64);
            for (ref[gi * 256 ..][0..@min(256, V - gi * 256)]) |o| gtop = @max(gtop, o);
            const id = ids[gi];
            ch.n += 1;
            if (id < gi * 256 or id >= @min(gi * 256 + 256, V) or ref[id] < gtop - near) {
                ch.bad += 1;
                std.debug.print("  group {d}: best id {d} ({e}), group top {e}\n", .{ gi, id, if (id < V) ref[id] else 0, gtop });
            }
            closeF(&ch, best[gi], ref[id], 1e-5);
        }
        const got = x.seq.slice(u32, 8)[step + 1];
        ch.n += 1;
        if (got >= V or ref[got] < top - near) {
            ch.bad += 1;
            std.debug.print("  step {d}: proposal {d} ({e}), reference {d} ({e})\n", .{ step, got, if (got < V) ref[got] else 0, top_id, top });
        }
    }
    pin(&ch, off, 2 * V);
    // planted ties and masked ids
    const bv = x.bv[0..V];
    const tie_a = 5;
    const tie_b = 150000;
    const hi = [_]usize{ VALID, VALID + 1, 200100, 200500, V - 1 };
    for (hi) |v| bv[v] = h.bf(30);
    for ([_][2]usize{ .{ tie_a, tie_b }, .{ 1000, 1001 }, .{ 70000, 70255 }, .{ 70256, 199999 } }) |tie| {
        const lo_id, const hi_id = tie;
        for (bv) |*b| b.* = h.bf(@mod(h.fb(b.*), 6.0)); // everything else low
        bv[lo_id] = h.bf(25);
        bv[hi_id] = h.bf(25);
        for (hi) |v| bv[v] = h.bf(30);
        @memcpy(x.wv[hi_id * RANK ..][0..RANK], x.wv[lo_id * RANK ..][0..RANK]);
        x.biasv[hi_id] = x.biasv[lo_id];
        try x.pick(c, 0);
        const got = x.seq.slice(u32, 8)[1];
        ch.n += 1;
        if (got != lo_id) {
            ch.bad += 1;
            std.debug.print("  tie {d} and {d}: proposal {d}\n", .{ lo_id, hi_id, got });
        }
        const lg = x.logits.slice(u16, V);
        ch.same(std.mem.asBytes(&lg[lo_id]), std.mem.asBytes(&lg[hi_id]));
        for (hi) |v| ch.b16n(lg[v], -std.math.inf(f64), 0);
    }
    // the last valid id wins over the larger ids past the vocabulary
    for (bv) |*b| b.* = h.bf(-8);
    bv[VALID - 1] = h.bf(20);
    for (hi) |v| bv[v] = h.bf(30);
    try x.pick(c, 0);
    ch.n += 1;
    if (x.seq.slice(u32, 8)[1] != VALID - 1) ch.bad += 1;
    try ch.done();
    const fa: FoldArgs = .{ .vocab = V, .valid = VALID, .flags = 0, .cap = 30.0 };
    try c.time("ikd_fold_pick, 201024 ids", "ikd_fold_pick", &.{ bytes(&fa), bo(x.base, 0), bs(x.latent), bs(x.fw), bs(x.fb), bs(x.best), bs(x.best_id), bs(x.logits) }, .{ x.groups, 1, 1 }, .{ 256, 1, 1 }, 50);
    const pa: PickArgs = .{ .groups = @intCast(x.groups), .step = 0 };
    try c.time("ikd_pick, 786 groups", "ikd_pick", &.{ bytes(&pa), bs(x.best), bs(x.best_id), bs(x.seq) }, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, 200);
}

// ---- ikd_fold_t ----

/// The fold's matvec output [257][V] to fold_w [V][256] and fold_b [V]: a copy, equal bit for bit, at the real vocabulary and
/// at one that does not fill the last tile.
fn caseFoldT(c: Ctx, r: std.Random) !void {
    const g = c.g;
    var ch: h.Check = .{ .name = "fold transpose" };
    for ([_]usize{ V, 1001 }) |vocab| {
        const y = try g.buf((RANK + 1) * vocab * 2);
        const fw = try g.buf(vocab * RANK * 2 + 64);
        const fb = try g.buf(vocab * 2 + 64);
        defer for ([_]mtl.Buffer{ y, fw, fb }) |b| b.deinit();
        const yv = h.fill16(y, (RANK + 1) * vocab, r, 4.0);
        @memset(fw.slice(u16, vocab * RANK + 32), NAN16);
        @memset(fb.slice(u16, vocab + 32), NAN16);
        const a: TArgs = .{ .vocab = @intCast(vocab) };
        try c.go("ikd_fold_t", &.{ bytes(&a), bs(y), bs(fw), bs(fb) }, .{ (vocab + 31) / 32, 1, 1 }, .{ 256, 1, 1 });
        const ws = fw.slice(u16, vocab * RANK + 32);
        const bsl = fb.slice(u16, vocab + 32);
        for (0..vocab) |v| {
            for (0..RANK) |cc| {
                ch.n += 1;
                if (ws[v * RANK + cc] != yv[cc * vocab + v]) ch.bad += 1;
            }
            ch.n += 1;
            if (bsl[v] != yv[RANK * vocab + v]) ch.bad += 1;
        }
        for (ws[vocab * RANK ..]) |w| {
            ch.n += 1;
            if (w != NAN16) ch.bad += 1;
        }
        for (bsl[vocab..]) |w| {
            ch.n += 1;
            if (w != NAN16) ch.bad += 1;
        }
        if (vocab == V) try c.time("ikd_fold_t, 257 x 201024", "ikd_fold_t", &.{ bytes(&a), bs(y), bs(fw), bs(fb) }, .{ (vocab + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, 10);
    }
    try ch.done();
}

// ---- the 8-bit projections' tiles ----

/// What `Qmv` encodes for the draft layers' quantized projections (core's row-exact tiles) against FP64 over the dequantized weights
/// `scale * code + bias` of the host quantizer, at the five shapes of a draft layer and 1 to 33 rows: f32 sums and one rounding to
/// bf16, so an output is within a bf16 step of its reference (or within f32's error of the terms' size, where the sum cancels toward
/// zero); and a row's bits are the same alone, among 3 rows, 8, 17 and 33. A zero, an all-positive and an all-negative group of the
/// weights are in the first rows. Then each shape's time at 8 rows, against the bf16 matvec of the same shape.
fn caseQmv(c: Ctx, r: std.Random) !void {
    const g = c.g;
    const gpa = std.heap.page_allocator;
    var qmv = try Qmv.init(g.device);
    defer qmv.deinit();
    const shapes = [_][2]usize{ .{ 6144, H }, .{ 1024, H }, .{ H, 6144 }, .{ 18432, H }, .{ H, 18432 } };
    const variants = [_]usize{ 33, 8, 1, 3, 9, 16, 17 };
    const most = variants[0];
    var ch: h.Check = .{ .name = "qmv (8-bit tiles)" };
    var off: usize = 0;
    var outputs: usize = 0;
    for (shapes) |shape| {
        const n, const k = shape;
        const groups = k / 64;
        const w = try g.buf(n * k * 2);
        const x = try g.buf(most * k * 2);
        const y = try g.buf(most * n * 2);
        defer for ([_]mtl.Buffer{ w, x, y }) |b| b.deinit();
        const wv = h.fill16(w, n * k, r, 0.1);
        for (0..64) |i| {
            wv[0 * k + 0 * 64 + i] = 0;
            wv[1 * k + 1 * 64 + i] = h.bf(0.01 + @as(f64, @floatFromInt(i)) * 0.001);
            wv[2 * k + 2 * 64 + i] = h.bf(-0.01 - @as(f64, @floatFromInt(i)) * 0.001);
        }
        const xv = h.fill16(x, most * k, r, 2.0);
        const q = try dws.quantize(gpa, g.device, wv, n, k);
        defer q.buf.deinit();
        const mem = q.buf.contents();
        const codes = mem[0 .. n * k];
        const scales = @as([*]const u16, @ptrCast(@alignCast(mem + q.scales)))[0 .. n * groups];
        const biases = @as([*]const u16, @ptrCast(@alignCast(mem + q.biases)))[0 .. n * groups];
        // the reference over a spread of columns, every row
        const step = @max(1, n / 400);
        var cols: std.ArrayList(usize) = .empty;
        defer cols.deinit(gpa);
        for (0..n) |j| if (j % step == 0 or j == n - 1) try cols.append(gpa, j);
        const want = try gpa.alloc(f64, most * cols.items.len);
        defer gpa.free(want);
        const mag = try gpa.alloc(f64, most * cols.items.len);
        defer gpa.free(mag);
        for (cols.items, 0..) |j, ci| for (0..most) |row| {
            var sum: f64 = 0;
            var size: f64 = 0;
            for (0..k) |kk| {
                const gi = j * groups + kk / 64;
                const dq = h.fb(scales[gi]) * @as(f64, @floatFromInt(codes[j * k + kk])) + h.fb(biases[gi]);
                const term = h.fb(xv[row * k + kk]) * dq;
                sum += term;
                size += @abs(term);
            }
            want[row * cols.items.len + ci] = sum;
            mag[row * cols.items.len + ci] = size;
        };
        const batch = try gpa.alloc(u16, most * n);
        defer gpa.free(batch);
        for (variants) |rows| {
            @memset(y.slice(u16, most * n), NAN16);
            const cb = g.queue.commandBuffer();
            const e = cb.compute(.serial);
            try qmv.encode(e, q, .{ .buf = x }, .{ .buf = y }, rows);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.log.err("qmv: {s}", .{msg});
                return error.GpuFailed;
            }
            const got = y.slice(u16, most * n);
            if (rows == most) {
                @memcpy(batch, got);
                for (cols.items, 0..) |j, ci| for (0..rows) |row| {
                    pinnedAbs(&ch, &off, got[row * n + j], want[row * cols.items.len + ci], mag[row * cols.items.len + ci] * 4e-6);
                    outputs += 1;
                };
            }
            // every row's bits are the batch's, and the rows past the call are untouched
            ch.same(std.mem.sliceAsBytes(got[0 .. rows * n]), std.mem.sliceAsBytes(batch[0 .. rows * n]));
            for (got[rows * n ..]) |v| ch.same(std.mem.asBytes(&v), std.mem.asBytes(&NAN16));
        }
    }
    pin(&ch, off, outputs);
    try ch.done();
    for (shapes) |shape| try timeQmv(c, &qmv, shape[0], shape[1], r);
}

/// One matvec of 8 rows at [n, k], the least GPU time of a few trials of dependent dispatches over six copies of the weights (so
/// the weights come from memory, not cache): the 8-bit tiles, and the bf16 `ik_matvec_8` the layer runs without them.
fn timeQmv(c: Ctx, qmv: *Qmv, n: usize, k: usize, r: std.Random) !void {
    const g = c.g;
    const gpa = std.heap.page_allocator;
    const copies = 6;
    const reps = 36;
    const rows = 8;
    const x = try g.buf(rows * k * 2);
    const y = try g.buf(rows * n * 2);
    defer for ([_]mtl.Buffer{ x, y }) |b| b.deinit();
    _ = h.fill16(x, rows * k, r, 2.0);
    var wb: [copies]mtl.Buffer = undefined;
    var wq: [copies]dws.Quant = undefined;
    for (0..copies) |i| {
        wb[i] = try g.buf(n * k * 2);
        const wv = h.fill16(wb[i], n * k, r, 0.1);
        wq[i] = try dws.quantize(gpa, g.device, wv, n, k);
    }
    defer for (0..copies) |i| {
        wb[i].deinit();
        wq[i].buf.deinit();
    };
    var q8: f64 = 1e9;
    var bf: f64 = 1e9;
    const p = try mtl.Pipeline.init(g.device, g.lib, "ik_matvec_8", false);
    defer p.deinit();
    const MvArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };
    for (0..5) |_| {
        for ([_]bool{ true, false }) |quant| {
            const cb = g.queue.commandBuffer();
            const e = cb.compute(.serial);
            for (0..reps) |i| {
                if (quant) {
                    try qmv.encode(e, wq[i % copies], .{ .buf = x }, .{ .buf = y }, rows);
                } else {
                    e.setPipeline(p);
                    e.setBuffer(x, 0, 0);
                    e.setBuffer(wb[i % copies], 0, 1);
                    e.setValue(MvArgs{ .rows = rows, .n = @intCast(n), .k = @intCast(k), .ldy = @intCast(n) }, 2);
                    e.setBuffer(y, 0, 3);
                    e.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                }
            }
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.log.err("qmv timing: {s}", .{msg});
                return error.GpuFailed;
            }
            if (quant) q8 = @min(q8, cb.gpuSeconds()) else bf = @min(bf, cb.gpuSeconds());
        }
    }
    std.debug.print("time qmv {d} x {d}, 8 rows: 8-bit {d:.2} us, bf16 matvec {d:.2} us\n", .{ n, k, q8 / reps * 1e6, bf / reps * 1e6 });
}
