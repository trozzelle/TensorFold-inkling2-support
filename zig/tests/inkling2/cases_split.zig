//! Inkling 2 kernel checks: split-K decode attention against an FP64 reference, a window's rows against the same rows
//! run one at a time, a row's blocks against the same row with empty blocks around them, and its relative bias formed
//! in the kernel against the table ik_relbias writes.
const std = @import("std");
const mtl = @import("metal");
const h = @import("harness.zig");

const Hq = 48;
const KV = 8;
const D = 128;
const E = 1024; // relative-bias extent, also the local layers' window
const REL = 16; // dims of the relative state
const KEYS = 256; // keys in one block
const PART_SIMDGROUPS = 24; // the partial kernel's threadgroup: 8 key groups of 32 keys, 3 pairs of q heads
const SLOT = D + 2; // floats in one (row, head, block) partial: m, l, acc[D]
const ROWS = 16; // a decode window's rows
const CAP = 3200; // cached keys; the deepest window's last row is at 3078
const MAX_BLOCKS = (CAP + KEYS - 1) / KEYS + 3; // the cache's blocks and three empty ones
const OUT_ROW = Hq * D * 2; // bytes of one row's attention output

/// Random inputs for ROWS rows over a CAP-key cache, and the buffers the two kernels write. Scores spread over several
/// units, so blocks differ in their maxima and the merge's rescaling matters. `reln` and `proj` are the relative state
/// and projection the kernel forms its bias from; `pb` holds the bias table of the same, formed on the host as
/// ik_relbias forms it (and read back through `pbv`).
const Inputs = struct {
    qn: mtl.Buffer,
    keys: mtl.Buffer,
    vals: mtl.Buffer,
    reln: mtl.Buffer,
    proj: mtl.Buffer,
    pb: mtl.Buffer,
    sink: mtl.Buffer,
    gate: mtl.Buffer,
    att: mtl.Buffer,
    part: mtl.Buffer,
    qv: []u16,
    kv: []u16,
    vv: []u16,
    relv: []u16,
    projv: []u16,
    pbv: []u16,
    sv: []u16,
    gv: []u16,

    fn init(g: *const h.Gpu, r: std.Random) !Inputs {
        const qn = try g.buf(ROWS * Hq * D * 2);
        const keys = try g.buf(CAP * KV * D * 2);
        const vals = try g.buf(CAP * KV * D * 2);
        const reln = try g.buf(ROWS * Hq * REL * 2);
        const proj = try g.buf(REL * E * 2);
        const pb = try g.buf(ROWS * Hq * E * 2);
        const sink = try g.buf(ROWS * Hq * 2);
        const gate = try g.buf(ROWS * Hq * D * 2);
        const att = try g.buf(ROWS * OUT_ROW);
        const part = try g.buf(ROWS * Hq * MAX_BLOCKS * SLOT * 4);
        const x: Inputs = .{
            .qn = qn,
            .keys = keys,
            .vals = vals,
            .reln = reln,
            .proj = proj,
            .pb = pb,
            .sink = sink,
            .gate = gate,
            .att = att,
            .part = part,
            .qv = h.fill16(qn, ROWS * Hq * D, r, 2.5),
            .kv = h.fill16(keys, CAP * KV * D, r, 2.5),
            .vv = h.fill16(vals, CAP * KV * D, r, 1.0),
            .relv = h.fill16(reln, ROWS * Hq * REL, r, 1.0),
            .projv = h.fill16(proj, REL * E, r, 1.5),
            .pbv = pb.slice(u16, ROWS * Hq * E),
            .sv = h.fill16(sink, ROWS * Hq, r, 2.0),
            .gv = h.fill16(gate, ROWS * Hq * D, r, 3.0),
        };
        x.formBias();
        return x;
    }

    /// The bias table of the current relative state and projection: bf16 of the 16 products (each exact in f32) summed
    /// in index order in f32.
    fn formBias(x: Inputs) void {
        for (0..ROWS * Hq) |rh| for (0..E) |d| {
            var s: f32 = 0;
            for (0..REL) |i| s += @as(f32, @floatCast(h.fb(x.relv[rh * REL + i]))) * @as(f32, @floatCast(h.fb(x.projv[i * E + d])));
            x.pbv[rh * E + d] = h.bf(s);
        };
    }

    fn deinit(x: Inputs) void {
        for ([_]mtl.Buffer{ x.qn, x.keys, x.vals, x.reln, x.proj, x.pb, x.sink, x.gate, x.att, x.part }) |b| b.deinit();
    }

    /// The attention bytes of rows `first` .. `first + n`.
    fn out(x: Inputs, first: usize, n: usize) []u8 {
        return x.att.slice(u8, ROWS * OUT_ROW)[first * OUT_ROW ..][0 .. n * OUT_ROW];
    }
};

/// The first block and the block count that cover rows pos .. pos + rows - 1: from the block of the first row's
/// earliest visible key to the block of the last row's position.
fn blocksOf(pos: u32, rows: u32, window: u32) struct { f0: u32, n: u32 } {
    const lo = if (window > 0 and pos + 1 > window) pos + 1 - window else 0;
    const f0 = lo / KEYS;
    return .{ .f0 = f0, .n = (pos + rows - 1) / KEYS - f0 + 1 };
}

fn argsOf(rows: u32, pos: u32, window: u32, f0: u32, n: u32, ring: u32) h.Args {
    return .{ .rows = rows, .heads = Hq, .kv = KV, .dim = D, .pos = pos, .window = window, .extent = E, .ring = ring, .blocks = n, .queries = f0, .scale = @floatCast(1 / @sqrt(@as(f64, D))) };
}

/// The partial kernel alone over `rows` rows that start at input row `row` and position `pos`, with blocks f0 .. f0 + n:
/// `ik_attn_part`, which forms the relative bias from the rows' state and the projection, or `ik_attn_part_table`,
/// which reads the bias table `table` (rows from `row`). The partials and the output start as NaN, so anything the
/// kernels leave unwritten shows.
fn runPart(g: *const h.Gpu, x: Inputs, name: [:0]const u8, table: mtl.Buffer, row: usize, rows: u32, pos: u32, window: u32, f0: u32, n: u32) !void {
    @memset(x.part.slice(f32, rows * Hq * n * SLOT), std.math.nan(f32));
    @memset(x.out(row, rows), 0xff);
    const q: h.Bind = .{ .buf = x.qn, .off = row * Hq * D * 2 };
    const a = argsOf(rows, pos, window, f0, n, CAP);
    const grid: [3]usize = .{ KV, rows, n };
    const group: [3]usize = .{ 32, PART_SIMDGROUPS, 1 };
    if (std.mem.eql(u8, name, "ik_attn_part")) {
        try g.run(name, a, &.{ q, .{ .buf = x.keys }, .{ .buf = x.vals }, .{ .buf = x.reln, .off = row * Hq * REL * 2 }, .{ .buf = x.proj }, .{ .buf = x.part } }, grid, group, true);
    } else {
        try g.run(name, a, &.{ q, .{ .buf = x.keys }, .{ .buf = x.vals }, .{ .buf = table, .off = row * Hq * E * 2 }, .{ .buf = x.part } }, grid, group, true);
    }
}

/// ik_attn_part, then ik_attn_merge, over the same rows and blocks.
fn attend(g: *const h.Gpu, x: Inputs, row: usize, rows: u32, pos: u32, window: u32, f0: u32, n: u32) !void {
    try runPart(g, x, "ik_attn_part", x.pb, row, rows, pos, window, f0, n);
    const a = argsOf(rows, pos, window, f0, n, CAP);
    try g.run("ik_attn_merge", a, &.{ .{ .buf = x.part }, .{ .buf = x.sink, .off = row * Hq * 2 }, .{ .buf = x.gate, .off = row * Hq * D * 2 }, .{ .buf = x.att, .off = row * OUT_ROW } }, .{ Hq, rows, 1 }, .{ 32, 1, 1 }, true);
}

/// A run's output must be all written and finite, so equal bytes below cannot be equal leftovers.
fn finite(c: *h.Check, bytes: []const u8) void {
    c.n += 1;
    for (std.mem.bytesAsSlice(u16, bytes)) |v| {
        if (std.math.isNan(h.fb(v))) {
            c.bad += 1;
            return;
        }
    }
}

/// FP64 softmax over the sink logit and the visible keys, with the relative bias and the output gate, as the one-pass
/// attention case: positions across block edges and the bias extent, for one row and for a full window.
fn caseSplitRef(g: *const h.Gpu, r: std.Random) !void {
    const x = try Inputs.init(g, r);
    defer x.deinit();
    const alloc = std.heap.page_allocator;
    const qf = try alloc.alloc(f64, x.qv.len);
    const kf = try alloc.alloc(f64, x.kv.len);
    const vf = try alloc.alloc(f64, x.vv.len);
    defer for ([_][]f64{ qf, kf, vf }) |s| alloc.free(s);
    for (qf, x.qv) |*o, v| o.* = h.fb(v);
    for (kf, x.kv) |*o, v| o.* = h.fb(v);
    for (vf, x.vv) |*o, v| o.* = h.fb(v);
    const scale = 1 / @sqrt(@as(f64, D));
    var logits: [CAP]f64 = undefined;
    for ([_]u32{ 1024, 0 }) |window| {
        var c: h.Check = .{ .name = if (window > 0) "split attention local" else "split attention global" };
        for ([_]u32{ 0, 255, 256, 700, 1500, 3000 }) |pos| for ([_]u32{ 1, ROWS }) |rows| {
            const bad = c.bad;
            const grid = blocksOf(pos, rows, window);
            try attend(g, x, 0, rows, pos, window, grid.f0, grid.n);
            const att = x.att.slice(u16, ROWS * Hq * D);
            for (0..rows) |t| for (0..Hq) |hd| {
                const p = pos + t;
                const lo = if (window > 0 and p + 1 > window) p + 1 - window else 0;
                const kvh = hd / (Hq / KV);
                const qh = qf[(t * Hq + hd) * D ..][0..D];
                const sink = h.fb(x.sv[t * Hq + hd]);
                var m = sink;
                for (lo..p + 1) |j| {
                    const kr = kf[(j * KV + kvh) * D ..][0..D];
                    var s: f64 = 0;
                    for (qh, kr) |qd, kd| s += qd * kd;
                    s *= scale;
                    if (p - j < E) s += h.fb(x.pbv[(t * Hq + hd) * E + p - j]);
                    logits[j - lo] = s;
                    m = @max(m, s);
                }
                var z = @exp(sink - m);
                var o: [D]f64 = @splat(0);
                for (lo..p + 1) |j| {
                    const w = @exp(logits[j - lo] - m);
                    z += w;
                    const vr = vf[(j * KV + kvh) * D ..][0..D];
                    for (&o, vr) |*od, vd| od.* += w * vd;
                }
                for (0..D) |d| {
                    const gt = h.fb(x.gv[(t * Hq + hd) * D + d]);
                    // two rounding points (o, then o times the gate), as the one-pass attention's check: a flip at the
                    // first can move the second by two steps, and outputs near zero keep the f32 sums' absolute error
                    c.b16a(att[(t * Hq + hd) * D + d], h.rb(o[d] / z) / (1 + @exp(-gt)), 2, 1e-6);
                }
            };
            if (c.bad != bad) std.debug.print("  failing: pos {d}, {d} rows\n", .{ pos, rows });
        };
        try c.done();
    }
}

/// A window's rows equal the same rows run one at a time (each with its own first block and block count) in bytes, over
/// windows that start and end near block edges and the bias extent; a NaN in row 1's query, or in its relative state,
/// stays in row 1.
fn caseSplitWindow(g: *const h.Gpu, r: std.Random) !void {
    const x = try Inputs.init(g, r);
    defer x.deinit();
    const saved = try std.heap.page_allocator.alloc(u8, ROWS * OUT_ROW);
    defer std.heap.page_allocator.free(saved);
    for ([_]u32{ 1024, 0 }) |window| {
        var c: h.Check = .{ .name = if (window > 0) "split window local" else "split window global" };
        for ([_]u32{ 0, 250, 1020, 1270, 2040, 3063 }) |pos| {
            const bad = c.bad;
            const grid = blocksOf(pos, ROWS, window);
            try attend(g, x, 0, ROWS, pos, window, grid.f0, grid.n);
            @memcpy(saved, x.out(0, ROWS));
            finite(&c, saved);
            for (0..ROWS) |t| {
                const one = blocksOf(pos + @as(u32, @intCast(t)), 1, window);
                try attend(g, x, t, 1, pos + @as(u32, @intCast(t)), window, one.f0, one.n);
                finite(&c, x.out(t, 1));
                c.same(x.out(t, 1), saved[t * OUT_ROW ..][0..OUT_ROW]);
            }
            const nan_at = Hq * D;
            const kept = x.qv[nan_at];
            x.qv[nan_at] = 0x7fc0;
            try attend(g, x, 0, ROWS, pos, window, grid.f0, grid.n);
            x.qv[nan_at] = kept;
            for (0..ROWS) |t| if (t != 1) c.same(x.out(t, 1), saved[t * OUT_ROW ..][0..OUT_ROW]);
            const rel_at = Hq * REL + 3 * REL; // head 3 of row 1
            const rel_kept = x.relv[rel_at];
            x.relv[rel_at] = 0x7fc0;
            try attend(g, x, 0, ROWS, pos, window, grid.f0, grid.n);
            x.relv[rel_at] = rel_kept;
            for (0..ROWS) |t| if (t != 1) c.same(x.out(t, 1), saved[t * OUT_ROW ..][0..OUT_ROW]);
            if (c.bad != bad) std.debug.print("  failing: pos {d}\n", .{pos});
        }
        try c.done();
    }
}

/// Blocks a row does not see change nothing: the same row with three empty blocks before its own, after them, or both
/// (the partials start as NaN, so an empty block must write its own marker) gives the plain run's bytes.
fn caseSplitBlocks(g: *const h.Gpu, r: std.Random) !void {
    const x = try Inputs.init(g, r);
    defer x.deinit();
    var plain: [OUT_ROW]u8 = undefined;
    for ([_]u32{ 1024, 0 }) |window| {
        var c: h.Check = .{ .name = if (window > 0) "split blocks local" else "split blocks global" };
        for ([_]u32{ 700, 3000 }) |pos| {
            const grid = blocksOf(pos, 1, window);
            try attend(g, x, 0, 1, pos, window, grid.f0, grid.n);
            @memcpy(&plain, x.out(0, 1));
            finite(&c, &plain);
            for ([_][2]u32{ .{ 3, 0 }, .{ 0, 3 }, .{ 3, 3 } }) |extra| {
                if (grid.f0 < extra[0]) continue; // a global row starts at block 0: nothing lies before it
                try attend(g, x, 0, 1, pos, window, grid.f0 - extra[0], grid.n + extra[0] + extra[1]);
                c.same(x.out(0, 1), &plain);
            }
        }
        try c.done();
    }
}

/// A relative state of +-1 and a projection whose every column holds a large pair of opposite signs among small terms:
/// for the heads whose two state values agree the pair cancels, and wherever it sits in the f32 sum it absorbs the small
/// terms' low bits, so the sum's order shows in the bf16 biases (about a tenth of them change when it does).
fn cancelling(x: Inputs, r: std.Random) void {
    for (x.relv) |*v| v.* = if (r.boolean()) 0x3f80 else 0xbf80;
    for (0..E) |d| {
        for (0..REL) |i| x.projv[i * E + d] = h.bf((r.float(f64) * 2 - 1) / 16);
        const up = r.uintLessThan(usize, REL);
        const down = (up + 1 + r.uintLessThan(usize, REL - 1)) % REL;
        const big = std.math.pow(f64, 2, @floatFromInt(r.intRangeAtMost(i32, 8, 12)));
        x.projv[up * E + d] = h.bf(big);
        x.projv[down * E + d] = h.bf(-big);
    }
    x.formBias();
}

/// The relative bias formed in the kernel is the table ik_relbias writes: the partials of ik_attn_part equal those of the
/// same body reading ik_relbias's output on the GPU (and the host's table, which ik_relbias must equal), for windows
/// that put some keys beyond the bias extent and some within it, at one row and at a window's rows.
fn caseSplitBias(g: *const h.Gpu, r: std.Random) !void {
    const x = try Inputs.init(g, r);
    defer x.deinit();
    cancelling(x, r);
    const gpu_pb = try g.buf(ROWS * Hq * E * 2);
    defer gpu_pb.deinit();
    const alloc = std.heap.page_allocator;
    const saved = try alloc.alloc(u8, ROWS * Hq * MAX_BLOCKS * SLOT * 4);
    defer alloc.free(saved);
    var c: h.Check = .{ .name = "split relative bias" };
    try g.run("ik_relbias", .{ .rows = ROWS, .heads = Hq, .extent = E, .dim = REL }, &.{ .{ .buf = x.reln }, .{ .buf = x.proj }, .{ .buf = gpu_pb } }, .{ E, Hq, ROWS }, .{ 256, 1, 1 }, false);
    c.same(gpu_pb.slice(u8, ROWS * Hq * E * 2), x.pb.slice(u8, ROWS * Hq * E * 2));
    for ([_]u32{ 1024, 0 }) |window| for ([_]u32{ 0, 255, 700, 1020, 1500, 3000 }) |pos| for ([_]u32{ 1, 5, ROWS }) |rows| {
        const grid = blocksOf(pos, rows, window);
        const bytes = rows * Hq * grid.n * SLOT * 4;
        try runPart(g, x, "ik_attn_part_table", gpu_pb, 0, rows, pos, window, grid.f0, grid.n);
        @memcpy(saved[0..bytes], x.part.slice(u8, bytes));
        try runPart(g, x, "ik_attn_part", gpu_pb, 0, rows, pos, window, grid.f0, grid.n);
        c.same(x.part.slice(u8, bytes), saved[0..bytes]);
    };
    try c.done();
}

/// A ring gives a linear cache's bytes across wraps, local and global, with non-live ring slots poisoned before each chunk.
fn caseSplitRing(g: *const h.Gpu, r: std.Random) !void {
    const R_small: usize = 1280;
    const T: usize = 3 * R_small + 100;
    const lin_cap: usize = std.mem.alignForward(usize, T, 64);
    const max_blocks: usize = (lin_cap + KEYS - 1) / KEYS;
    const nan: u16 = 0x7fc0;
    const q = try g.buf(ROWS * Hq * D * 2);
    const k = try g.buf(ROWS * KV * D * 2);
    const v = try g.buf(ROWS * KV * D * 2);
    const rr = try g.buf(ROWS * Hq * REL * 2);
    const qs = try g.buf(Hq * D * 2);
    const rs = try g.buf(Hq * REL * 2);
    const qn = try g.buf(ROWS * Hq * D * 2);
    const reln = try g.buf(ROWS * Hq * REL * 2);
    const proj = try g.buf(REL * E * 2);
    const sink = try g.buf(ROWS * Hq * 2);
    const gate = try g.buf(ROWS * Hq * D * 2);
    const part = try g.buf(ROWS * Hq * max_blocks * SLOT * 4);
    const out_lin = try g.buf(ROWS * OUT_ROW);
    const out_ring = try g.buf(ROWS * OUT_ROW);
    const lin_k = try g.buf(lin_cap * KV * D * 2);
    const lin_v = try g.buf(lin_cap * KV * D * 2);
    const ring_k = try g.buf(lin_cap * KV * D * 2);
    const ring_v = try g.buf(lin_cap * KV * D * 2);
    defer for ([_]mtl.Buffer{ q, k, v, rr, qs, rs, qn, reln, proj, sink, gate, part, out_lin, out_ring, lin_k, lin_v, ring_k, ring_v }) |b| b.deinit();
    _ = h.fill16(qs, Hq * D, r, 1.5);
    _ = h.fill16(rs, Hq * REL, r, 1.5);
    _ = h.fill16(proj, REL * E, r, 1.5);
    const chunks = [_]usize{ 1, 2, 5, 8, 16 };
    for ([_]bool{ false, true }) |global| {
        const ring_slots: usize = if (global) lin_cap else R_small;
        const window: u32 = if (global) 0 else E;
        var c: h.Check = .{ .name = if (global) "split ring global" else "split ring local" };
        var pos: usize = 0;
        var ci: usize = 0;
        while (pos < T) : (ci += 1) {
            const n = @min(chunks[ci % chunks.len], T - pos);
            _ = h.fill16(q, n * Hq * D, r, 2.5);
            _ = h.fill16(k, n * KV * D, r, 2.5);
            _ = h.fill16(v, n * KV * D, r, 1.0);
            _ = h.fill16(rr, n * Hq * REL, r, 1.0);
            _ = h.fill16(sink, n * Hq, r, 2.0);
            _ = h.fill16(gate, n * Hq * D, r, 3.0);
            var poisoned: [R_small]bool = @splat(false);
            if (!global) {
                const live_lo = std.mem.alignBackward(usize, pos -| (E - 1), 64);
                var keep: [R_small]bool = @splat(false);
                for (live_lo..pos) |p| keep[p % R_small] = true;
                const rk = ring_k.slice(u16, R_small * KV * D);
                const rv = ring_v.slice(u16, R_small * KV * D);
                for (0..R_small) |s| if (!keep[s]) {
                    poisoned[s] = true;
                    @memset(rk[s * KV * D ..][0 .. KV * D], nan);
                    @memset(rv[s * KV * D ..][0 .. KV * D], nan);
                };
            }
            const prep_base: h.Args = .{ .rows = @intCast(n), .heads = Hq, .kv = KV, .dim = D, .pos = @intCast(pos), .flags = if (global) 2 else 0, .eps = 1e-6, .cap = 22.627416997969522, .thr = 8, .alpha = 0.1, .floor_n = 8192 };
            var prep_lin = prep_base;
            prep_lin.ring = @intCast(lin_cap);
            try g.run("ik_prep", prep_lin, &.{ .{ .buf = q }, .{ .buf = k }, .{ .buf = v }, .{ .buf = rr }, .{ .buf = qs }, .{ .buf = rs }, .{ .buf = qn }, .{ .buf = lin_k }, .{ .buf = lin_v }, .{ .buf = reln } }, .{ 2 * Hq + 2 * KV, n, 1 }, .{ 32, 1, 1 }, true);
            var prep_ring = prep_base;
            prep_ring.ring = @intCast(ring_slots);
            try g.run("ik_prep", prep_ring, &.{ .{ .buf = q }, .{ .buf = k }, .{ .buf = v }, .{ .buf = rr }, .{ .buf = qs }, .{ .buf = rs }, .{ .buf = qn }, .{ .buf = ring_k }, .{ .buf = ring_v }, .{ .buf = reln } }, .{ 2 * Hq + 2 * KV, n, 1 }, .{ 32, 1, 1 }, true);
            const grid = blocksOf(@intCast(pos), @intCast(n), window);
            const a_lin = argsOf(@intCast(n), @intCast(pos), window, grid.f0, grid.n, @intCast(lin_cap));
            try g.run("ik_attn_part", a_lin, &.{ .{ .buf = qn }, .{ .buf = lin_k }, .{ .buf = lin_v }, .{ .buf = reln }, .{ .buf = proj }, .{ .buf = part } }, .{ KV, n, grid.n }, .{ 32, PART_SIMDGROUPS, 1 }, true);
            try g.run("ik_attn_merge", a_lin, &.{ .{ .buf = part }, .{ .buf = sink }, .{ .buf = gate }, .{ .buf = out_lin } }, .{ Hq, n, 1 }, .{ 32, 1, 1 }, true);
            const a_ring = argsOf(@intCast(n), @intCast(pos), window, grid.f0, grid.n, @intCast(ring_slots));
            try g.run("ik_attn_part", a_ring, &.{ .{ .buf = qn }, .{ .buf = ring_k }, .{ .buf = ring_v }, .{ .buf = reln }, .{ .buf = proj }, .{ .buf = part } }, .{ KV, n, grid.n }, .{ 32, PART_SIMDGROUPS, 1 }, true);
            try g.run("ik_attn_merge", a_ring, &.{ .{ .buf = part }, .{ .buf = sink }, .{ .buf = gate }, .{ .buf = out_ring } }, .{ Hq, n, 1 }, .{ 32, 1, 1 }, true);
            const bytes = n * OUT_ROW;
            finite(&c, out_ring.slice(u8, bytes));
            c.same(out_ring.slice(u8, bytes), out_lin.slice(u8, bytes));
            if (!global) {
                var written: [R_small]bool = @splat(false);
                for (pos..pos + n) |p| written[p % R_small] = true;
                const rk = ring_k.slice(u16, R_small * KV * D);
                const rv = ring_v.slice(u16, R_small * KV * D);
                for (0..R_small) |s| if (poisoned[s] and !written[s]) {
                    c.n += 1;
                    for (rk[s * KV * D ..][0 .. KV * D], rv[s * KV * D ..][0 .. KV * D]) |kv_x, vv_x| {
                        if (kv_x != nan or vv_x != nan) {
                            c.bad += 1;
                            break;
                        }
                    }
                };
            }
            pos += n;
        }
        try c.done();
    }
}

/// `ik_attn_part_w` over global windows of 2..16 rows: every row's partial bytes and merged output equal `ik_attn_part` +
/// `ik_attn_merge` run a row at a time, keys and values past the window's last row are NaN without affecting any output,
/// and a NaN query or relative-state row stays confined to its own row.
fn caseSplitWindowShared(g: *const h.Gpu, r: std.Random) !void {
    const cap: usize = 40064;
    const max_blocks: usize = (cap + KEYS - 1) / KEYS + 2;
    const nan: u16 = 0x7fc0;
    const alloc = std.heap.page_allocator;
    const qn = try g.buf(ROWS * Hq * D * 2);
    const keys = try g.buf(cap * KV * D * 2);
    const vals = try g.buf(cap * KV * D * 2);
    const reln = try g.buf(ROWS * Hq * REL * 2);
    const proj = try g.buf(REL * E * 2);
    const sink = try g.buf(ROWS * Hq * 2);
    const gate = try g.buf(ROWS * Hq * D * 2);
    const part = try g.buf(ROWS * Hq * max_blocks * SLOT * 4);
    const att = try g.buf(ROWS * OUT_ROW);
    defer for ([_]mtl.Buffer{ qn, keys, vals, reln, proj, sink, gate, part, att }) |b| b.deinit();
    const qv = h.fill16(qn, ROWS * Hq * D, r, 2.5);
    const kv = h.fill16(keys, cap * KV * D, r, 2.5);
    const vv = h.fill16(vals, cap * KV * D, r, 1.0);
    const relv = h.fill16(reln, ROWS * Hq * REL, r, 1.0);
    _ = h.fill16(proj, REL * E, r, 1.5);
    _ = h.fill16(sink, ROWS * Hq, r, 2.0);
    _ = h.fill16(gate, ROWS * Hq * D, r, 3.0);
    const ref_part = try alloc.alloc(u8, ROWS * Hq * max_blocks * SLOT * 4);
    const ref_att = try alloc.alloc(u8, ROWS * OUT_ROW);
    const saved_k = try alloc.alloc(u16, KEYS * KV * D);
    const saved_v = try alloc.alloc(u16, KEYS * KV * D);
    defer for ([_][]u8{ ref_part, ref_att }) |b| alloc.free(b);
    defer for ([_][]u16{ saved_k, saved_v }) |b| alloc.free(b);

    var c: h.Check = .{ .name = "split window shared global" };
    for ([_]u32{ 250, 300, 9000, 9210, 40000 }) |pos| {
        for (2..ROWS + 1) |rows_u| {
            const rows: u32 = @intCast(rows_u);
            const bad = c.bad;
            const grid = blocksOf(pos, rows, 0);
            const nblocks: u32 = if (pos == 300 and rows == 6) grid.n + 2 else grid.n;
            const row_part_bytes: usize = Hq * nblocks * SLOT * 4;
            const part_bytes: usize = rows_u * row_part_bytes;
            const att_bytes: usize = rows_u * OUT_ROW;

            // Poison keys and values past the window's last row so any out-of-window load shows as NaN.
            const tail0: usize = pos + rows_u;
            const tail_n: usize = @min(KEYS, cap - tail0);
            const ktail = kv[tail0 * KV * D ..][0 .. tail_n * KV * D];
            const vtail = vv[tail0 * KV * D ..][0 .. tail_n * KV * D];
            @memcpy(saved_k[0 .. tail_n * KV * D], ktail);
            @memcpy(saved_v[0 .. tail_n * KV * D], vtail);
            @memset(ktail, nan);
            @memset(vtail, nan);
            defer {
                @memcpy(ktail, saved_k[0 .. tail_n * KV * D]);
                @memcpy(vtail, saved_v[0 .. tail_n * KV * D]);
            }

            // Reference: ik_attn_part + ik_attn_merge one row at a time with the same block grid.
            for (0..rows_u) |t| {
                const a1 = argsOf(1, pos + @as(u32, @intCast(t)), 0, grid.f0, nblocks, @intCast(cap));
                @memset(part.slice(f32, Hq * nblocks * SLOT), std.math.nan(f32));
                @memset(att.slice(u8, OUT_ROW), 0xff);
                try g.run("ik_attn_part", a1, &.{ .{ .buf = qn, .off = t * Hq * D * 2 }, .{ .buf = keys }, .{ .buf = vals }, .{ .buf = reln, .off = t * Hq * REL * 2 }, .{ .buf = proj }, .{ .buf = part } }, .{ KV, 1, nblocks }, .{ 32, PART_SIMDGROUPS, 1 }, true);
                @memcpy(ref_part[t * row_part_bytes ..][0..row_part_bytes], part.slice(u8, row_part_bytes));
                try g.run("ik_attn_merge", a1, &.{ .{ .buf = part }, .{ .buf = sink, .off = t * Hq * 2 }, .{ .buf = gate, .off = t * Hq * D * 2 }, .{ .buf = att } }, .{ Hq, 1, 1 }, .{ 32, 1, 1 }, true);
                @memcpy(ref_att[t * OUT_ROW ..][0..OUT_ROW], att.slice(u8, OUT_ROW));
            }
            finite(&c, ref_att[0..att_bytes]);

            // Shared window kernel: grid (KV, 1, nblocks).
            const aw = argsOf(rows, pos, 0, grid.f0, nblocks, @intCast(cap));
            @memset(part.slice(f32, rows_u * Hq * nblocks * SLOT), std.math.nan(f32));
            @memset(att.slice(u8, att_bytes), 0xff);
            try g.run("ik_attn_part_w", aw, &.{ .{ .buf = qn }, .{ .buf = keys }, .{ .buf = vals }, .{ .buf = reln }, .{ .buf = proj }, .{ .buf = part } }, .{ KV, 1, nblocks }, .{ 32, 6, 1 }, true);
            c.same(part.slice(u8, part_bytes), ref_part[0..part_bytes]);
            try g.run("ik_attn_merge", aw, &.{ .{ .buf = part }, .{ .buf = sink }, .{ .buf = gate }, .{ .buf = att } }, .{ Hq, rows, 1 }, .{ 32, 1, 1 }, true);
            finite(&c, att.slice(u8, att_bytes));
            c.same(att.slice(u8, att_bytes), ref_att[0..att_bytes]);

            // NaN query and relative-state containment on full 16-row windows (covering both row tiles).
            if (rows == ROWS and (pos == 300 or pos == 9210)) {
                for ([_]usize{ 1, 9 }) |bad_row| {
                    const q_at = bad_row * Hq * D;
                    const kept_q = qv[q_at];
                    qv[q_at] = nan;
                    @memset(part.slice(f32, rows_u * Hq * nblocks * SLOT), std.math.nan(f32));
                    try g.run("ik_attn_part_w", aw, &.{ .{ .buf = qn }, .{ .buf = keys }, .{ .buf = vals }, .{ .buf = reln }, .{ .buf = proj }, .{ .buf = part } }, .{ KV, 1, nblocks }, .{ 32, 6, 1 }, true);
                    try g.run("ik_attn_merge", aw, &.{ .{ .buf = part }, .{ .buf = sink }, .{ .buf = gate }, .{ .buf = att } }, .{ Hq, rows, 1 }, .{ 32, 1, 1 }, true);
                    qv[q_at] = kept_q;
                    for (0..rows_u) |t| if (t != bad_row) c.same(att.slice(u8, att_bytes)[t * OUT_ROW ..][0..OUT_ROW], ref_att[t * OUT_ROW ..][0..OUT_ROW]);
                }
                const rel_at = Hq * REL + 3 * REL;
                const kept_r = relv[rel_at];
                relv[rel_at] = nan;
                @memset(part.slice(f32, rows_u * Hq * nblocks * SLOT), std.math.nan(f32));
                try g.run("ik_attn_part_w", aw, &.{ .{ .buf = qn }, .{ .buf = keys }, .{ .buf = vals }, .{ .buf = reln }, .{ .buf = proj }, .{ .buf = part } }, .{ KV, 1, nblocks }, .{ 32, 6, 1 }, true);
                try g.run("ik_attn_merge", aw, &.{ .{ .buf = part }, .{ .buf = sink }, .{ .buf = gate }, .{ .buf = att } }, .{ Hq, rows, 1 }, .{ 32, 1, 1 }, true);
                relv[rel_at] = kept_r;
                for (0..rows_u) |t| if (t != 1) c.same(att.slice(u8, att_bytes)[t * OUT_ROW ..][0..OUT_ROW], ref_att[t * OUT_ROW ..][0..OUT_ROW]);
            }
            if (c.bad != bad) std.debug.print("  failing: pos {d}, {d} rows\n", .{ pos, rows });
        }
    }
    try c.done();
}

/// G=3 (24 q heads, 8 kv heads, 128 dims) verification: FP64 reference for sliding and global attention, 1-row vs
/// 16-row window bit-exactness, `ik_attn_part_w_g3 == ik_attn_part_g3` across rows = 1, 4, 8, 16 (partials and merged
/// outputs), and NaN containment for both queries and relative states.
fn caseG3(g: *const h.Gpu, r: std.Random) !void {
    const Hq3 = 24;
    const OUT_ROW3 = Hq3 * D * 2;
    const nan: u16 = 0x7fc0;
    const alloc = std.heap.page_allocator;

    const qn = try g.buf(ROWS * Hq3 * D * 2);
    const keys = try g.buf(CAP * KV * D * 2);
    const vals = try g.buf(CAP * KV * D * 2);
    const reln = try g.buf(ROWS * Hq3 * REL * 2);
    const proj = try g.buf(REL * E * 2);
    const sink = try g.buf(ROWS * Hq3 * 2);
    const gate = try g.buf(ROWS * Hq3 * D * 2);
    const att = try g.buf(ROWS * OUT_ROW3);
    const part = try g.buf(ROWS * Hq3 * MAX_BLOCKS * SLOT * 4);
    defer for ([_]mtl.Buffer{ qn, keys, vals, reln, proj, sink, gate, att, part }) |b| b.deinit();

    const qv = h.fill16(qn, ROWS * Hq3 * D, r, 2.5);
    const kv = h.fill16(keys, CAP * KV * D, r, 2.5);
    const vv = h.fill16(vals, CAP * KV * D, r, 1.0);
    const relv = h.fill16(reln, ROWS * Hq3 * REL, r, 1.0);
    const projv = h.fill16(proj, REL * E, r, 1.5);
    const sv = h.fill16(sink, ROWS * Hq3, r, 2.0);
    const gv = h.fill16(gate, ROWS * Hq3 * D, r, 3.0);

    const pbv = try alloc.alloc(u16, ROWS * Hq3 * E);
    defer alloc.free(pbv);
    for (0..ROWS * Hq3) |rh| for (0..E) |d| {
        var s: f32 = 0;
        for (0..REL) |i| s += @as(f32, @floatCast(h.fb(relv[rh * REL + i]))) * @as(f32, @floatCast(h.fb(projv[i * E + d])));
        pbv[rh * E + d] = h.bf(s);
    };

    const args3 = struct {
        fn make(rows: u32, pos: u32, window: u32, f0: u32, n: u32) h.Args {
            return .{
                .rows = rows,
                .heads = Hq3,
                .kv = KV,
                .dim = D,
                .pos = pos,
                .window = window,
                .extent = E,
                .ring = CAP,
                .blocks = n,
                .queries = f0,
                .scale = @floatCast(1 / @sqrt(@as(f64, D))),
            };
        }
    }.make;

    const runG3 = struct {
        fn call(gpu: *const h.Gpu, qn_b: mtl.Buffer, keys_b: mtl.Buffer, vals_b: mtl.Buffer, reln_b: mtl.Buffer, proj_b: mtl.Buffer, sink_b: mtl.Buffer, gate_b: mtl.Buffer, part_b: mtl.Buffer, att_b: mtl.Buffer, row: usize, rows: u32, pos: u32, window: u32, f0: u32, n: u32) !void {
            @memset(part_b.slice(f32, rows * Hq3 * n * SLOT), std.math.nan(f32));
            @memset(att_b.slice(u8, ROWS * OUT_ROW3)[row * OUT_ROW3 ..][0 .. rows * OUT_ROW3], 0xff);
            const a = args3(rows, pos, window, f0, n);
            try gpu.run("ik_attn_part_g3", a, &.{
                .{ .buf = qn_b, .off = row * Hq3 * D * 2 },
                .{ .buf = keys_b },
                .{ .buf = vals_b },
                .{ .buf = reln_b, .off = row * Hq3 * REL * 2 },
                .{ .buf = proj_b },
                .{ .buf = part_b },
            }, .{ KV, rows, n }, .{ 32, PART_SIMDGROUPS, 1 }, true);
            try gpu.run("ik_attn_merge", a, &.{
                .{ .buf = part_b },
                .{ .buf = sink_b, .off = row * Hq3 * 2 },
                .{ .buf = gate_b, .off = row * Hq3 * D * 2 },
                .{ .buf = att_b, .off = row * OUT_ROW3 },
            }, .{ Hq3, rows, 1 }, .{ 32, 1, 1 }, true);
        }
    }.call;

    // 1. FP64 reference (sliding and global)
    const qf = try alloc.alloc(f64, qv.len);
    const kf = try alloc.alloc(f64, kv.len);
    const vf = try alloc.alloc(f64, vv.len);
    defer for ([_][]f64{ qf, kf, vf }) |s| alloc.free(s);
    for (qf, qv) |*o, v| o.* = h.fb(v);
    for (kf, kv) |*o, v| o.* = h.fb(v);
    for (vf, vv) |*o, v| o.* = h.fb(v);
    const scale = 1 / @sqrt(@as(f64, D));
    var logits: [CAP]f64 = undefined;

    for ([_]u32{ 1024, 0 }) |window| {
        var c: h.Check = .{ .name = if (window > 0) "split g3 local" else "split g3 global" };
        var co: h.Check = .{ .name = if (window > 0) "split g3 part local" else "split g3 part global" };
        for ([_]u32{ 0, 255, 1500, 2700 }) |pos| for ([_]u32{ 1, ROWS }) |rows| {
            const bad = c.bad + co.bad;
            const grid = blocksOf(pos, rows, window);
            try runG3(g, qn, keys, vals, reln, proj, sink, gate, part, att, 0, rows, pos, window, grid.f0, grid.n);
            const att16 = att.slice(u16, ROWS * Hq3 * D);
            const part32 = part.slice(f32, rows * Hq3 * grid.n * SLOT);
            for (0..rows) |t| for (0..Hq3) |hd| {
                const p = pos + t;
                const lo = if (window > 0 and p + 1 > window) p + 1 - window else 0;
                const kvh = hd / (Hq3 / KV);
                const qh = qf[(t * Hq3 + hd) * D ..][0..D];
                const sink_v = h.fb(sv[t * Hq3 + hd]);
                var m = sink_v;
                for (lo..p + 1) |j| {
                    const kr = kf[(j * KV + kvh) * D ..][0..D];
                    var s: f64 = 0;
                    for (qh, kr) |qd, kd| s += qd * kd;
                    s *= scale;
                    if (p - j < E) s += h.fb(pbv[(t * Hq3 + hd) * E + p - j]);
                    logits[j - lo] = s;
                    m = @max(m, s);
                }
                var z = @exp(sink_v - m);
                var o: [D]f64 = @splat(0);
                for (lo..p + 1) |j| {
                    const w = @exp(logits[j - lo] - m);
                    z += w;
                    const vr = vf[(j * KV + kvh) * D ..][0..D];
                    for (&o, vr) |*od, vd| od.* += w * vd;
                }
                // Fold partials from ik_attn_part_g3 as ik_attn_merge does to check the pre-gate bf16 output at <= 1 ULP.
                var pm: f32 = @floatCast(sink_v);
                var pl: f32 = 1.0;
                var pacc: [D]f32 = @splat(0.0);
                const head_part = part32[(t * Hq3 + hd) * grid.n * SLOT ..][0 .. grid.n * SLOT];
                for (0..grid.n) |b| {
                    const slot = head_part[b * SLOT ..][0..SLOT];
                    const mb = slot[0];
                    const lb = slot[1];
                    if (lb == 0.0) continue;
                    const mn = @max(pm, mb);
                    const f = @exp(pm - mn);
                    const fb_w = @exp(mb - mn);
                    pl = @mulAdd(f32, pl, f, lb * fb_w);
                    for (0..D) |d| pacc[d] = @mulAdd(f32, pacc[d], f, slot[2 + d] * fb_w);
                    pm = mn;
                }
                for (0..D) |d| {
                    co.b16a(h.bf(pacc[d] / pl), o[d] / z, 1, 1e-6);
                    const gt = h.fb(gv[(t * Hq3 + hd) * D + d]);
                    // two rounding points (o, then o times the gate), as caseSplitRef
                    c.b16a(att16[(t * Hq3 + hd) * D + d], h.rb(o[d] / z) / (1 + @exp(-gt)), 2, 1e-6);
                }
            };
            if (c.bad + co.bad != bad) std.debug.print("  failing: pos {d}, {d} rows\n", .{ pos, rows });
        };
        try co.done();
        try c.done();
    }

    // 2. 1-row vs 16-row window bit-exactness and NaN containment for ik_attn_part_g3
    const saved = try alloc.alloc(u8, ROWS * OUT_ROW3);
    defer alloc.free(saved);
    for ([_]u32{ 1024, 0 }) |window| {
        var c: h.Check = .{ .name = if (window > 0) "split g3 window local" else "split g3 window global" };
        for ([_]u32{ 0, 250, 1020, 1500, 2700 }) |pos| {
            const bad = c.bad;
            const grid = blocksOf(pos, ROWS, window);
            try runG3(g, qn, keys, vals, reln, proj, sink, gate, part, att, 0, ROWS, pos, window, grid.f0, grid.n);
            @memcpy(saved, att.slice(u8, ROWS * OUT_ROW3));
            finite(&c, saved);
            for (0..ROWS) |t| {
                const one = blocksOf(pos + @as(u32, @intCast(t)), 1, window);
                try runG3(g, qn, keys, vals, reln, proj, sink, gate, part, att, t, 1, pos + @as(u32, @intCast(t)), window, one.f0, one.n);
                const row_out = att.slice(u8, ROWS * OUT_ROW3)[t * OUT_ROW3 ..][0..OUT_ROW3];
                finite(&c, row_out);
                c.same(row_out, saved[t * OUT_ROW3 ..][0..OUT_ROW3]);
            }
            // Extra empty blocks before/after must not alter row 0.
            const one0 = blocksOf(pos, 1, window);
            for ([_][2]u32{ .{ 2, 0 }, .{ 0, 2 }, .{ 2, 2 } }) |extra| {
                if (one0.f0 < extra[0]) continue;
                try runG3(g, qn, keys, vals, reln, proj, sink, gate, part, att, 0, 1, pos, window, one0.f0 - extra[0], one0.n + extra[0] + extra[1]);
                c.same(att.slice(u8, OUT_ROW3), saved[0..OUT_ROW3]);
            }
            // NaN in row 1's query and relative state must stay confined to row 1.
            const q_at = Hq3 * D;
            const kept_q = qv[q_at];
            qv[q_at] = nan;
            try runG3(g, qn, keys, vals, reln, proj, sink, gate, part, att, 0, ROWS, pos, window, grid.f0, grid.n);
            qv[q_at] = kept_q;
            for (0..ROWS) |t| if (t != 1) c.same(att.slice(u8, ROWS * OUT_ROW3)[t * OUT_ROW3 ..][0..OUT_ROW3], saved[t * OUT_ROW3 ..][0..OUT_ROW3]);
            const rel_at = Hq3 * REL + 2 * REL;
            const kept_r = relv[rel_at];
            relv[rel_at] = nan;
            try runG3(g, qn, keys, vals, reln, proj, sink, gate, part, att, 0, ROWS, pos, window, grid.f0, grid.n);
            relv[rel_at] = kept_r;
            for (0..ROWS) |t| if (t != 1) c.same(att.slice(u8, ROWS * OUT_ROW3)[t * OUT_ROW3 ..][0..OUT_ROW3], saved[t * OUT_ROW3 ..][0..OUT_ROW3]);
            if (c.bad != bad) std.debug.print("  failing: pos {d}\n", .{pos});
        }
        try c.done();
    }

    // 3. ik_attn_part_w_g3 == ik_attn_part_g3 bit-exactness across rows = 1, 4, 8, 16, tail NaN poisoning, and NaN containment
    const ref_part = try alloc.alloc(u8, ROWS * Hq3 * MAX_BLOCKS * SLOT * 4);
    const ref_att = try alloc.alloc(u8, ROWS * OUT_ROW3);
    const saved_k = try alloc.alloc(u16, KEYS * KV * D);
    const saved_v = try alloc.alloc(u16, KEYS * KV * D);
    defer for ([_][]u8{ ref_part, ref_att }) |b| alloc.free(b);
    defer for ([_][]u16{ saved_k, saved_v }) |b| alloc.free(b);

    var cw: h.Check = .{ .name = "split g3 window shared global" };
    for ([_]u32{ 250, 300, 1500, 2700 }) |pos| {
        for ([_]u32{ 1, 4, 8, 16 }) |rows| {
            const rows_u: usize = rows;
            const bad = cw.bad;
            const grid = blocksOf(pos, rows, 0);
            const nblocks: u32 = if (pos == 300 and rows == 8) grid.n + 2 else grid.n;
            const part_bytes: usize = rows_u * Hq3 * nblocks * SLOT * 4;
            const att_bytes: usize = rows_u * OUT_ROW3;

            const tail0: usize = pos + rows_u;
            const tail_n: usize = @min(KEYS, CAP - tail0);
            const ktail = kv[tail0 * KV * D ..][0 .. tail_n * KV * D];
            const vtail = vv[tail0 * KV * D ..][0 .. tail_n * KV * D];
            @memcpy(saved_k[0 .. tail_n * KV * D], ktail);
            @memcpy(saved_v[0 .. tail_n * KV * D], vtail);
            @memset(ktail, nan);
            @memset(vtail, nan);
            defer {
                @memcpy(ktail, saved_k[0 .. tail_n * KV * D]);
                @memcpy(vtail, saved_v[0 .. tail_n * KV * D]);
            }

            // Reference: ik_attn_part_g3 + ik_attn_merge
            try runG3(g, qn, keys, vals, reln, proj, sink, gate, part, att, 0, rows, pos, 0, grid.f0, nblocks);
            @memcpy(ref_part[0..part_bytes], part.slice(u8, part_bytes));
            @memcpy(ref_att[0..att_bytes], att.slice(u8, att_bytes));
            finite(&cw, ref_att[0..att_bytes]);

            // Shared-KV global window kernel: grid (KV, 1, nblocks), group (32, 3, 1)
            const aw = args3(rows, pos, 0, grid.f0, nblocks);
            @memset(part.slice(f32, rows_u * Hq3 * nblocks * SLOT), std.math.nan(f32));
            @memset(att.slice(u8, att_bytes), 0xff);
            try g.run("ik_attn_part_w_g3", aw, &.{
                .{ .buf = qn },
                .{ .buf = keys },
                .{ .buf = vals },
                .{ .buf = reln },
                .{ .buf = proj },
                .{ .buf = part },
            }, .{ KV, 1, nblocks }, .{ 32, 3, 1 }, true);
            cw.same(part.slice(u8, part_bytes), ref_part[0..part_bytes]);
            try g.run("ik_attn_merge", aw, &.{
                .{ .buf = part },
                .{ .buf = sink },
                .{ .buf = gate },
                .{ .buf = att },
            }, .{ Hq3, rows, 1 }, .{ 32, 1, 1 }, true);
            finite(&cw, att.slice(u8, att_bytes));
            cw.same(att.slice(u8, att_bytes), ref_att[0..att_bytes]);

            // NaN query and relative-state containment on full 16-row windows (covering multiple row tiles).
            if (rows == ROWS and (pos == 300 or pos == 2700)) {
                for ([_]usize{ 1, 9 }) |bad_row| {
                    const q_at = bad_row * Hq3 * D;
                    const kept_q = qv[q_at];
                    qv[q_at] = nan;
                    @memset(part.slice(f32, rows_u * Hq3 * nblocks * SLOT), std.math.nan(f32));
                    try g.run("ik_attn_part_w_g3", aw, &.{
                        .{ .buf = qn },
                        .{ .buf = keys },
                        .{ .buf = vals },
                        .{ .buf = reln },
                        .{ .buf = proj },
                        .{ .buf = part },
                    }, .{ KV, 1, nblocks }, .{ 32, 3, 1 }, true);
                    try g.run("ik_attn_merge", aw, &.{
                        .{ .buf = part },
                        .{ .buf = sink },
                        .{ .buf = gate },
                        .{ .buf = att },
                    }, .{ Hq3, rows, 1 }, .{ 32, 1, 1 }, true);
                    qv[q_at] = kept_q;
                    for (0..rows_u) |t| if (t != bad_row) cw.same(att.slice(u8, att_bytes)[t * OUT_ROW3 ..][0..OUT_ROW3], ref_att[t * OUT_ROW3 ..][0..OUT_ROW3]);
                }
                const rel_at = Hq3 * REL + 2 * REL;
                const kept_r = relv[rel_at];
                relv[rel_at] = nan;
                @memset(part.slice(f32, rows_u * Hq3 * nblocks * SLOT), std.math.nan(f32));
                try g.run("ik_attn_part_w_g3", aw, &.{
                    .{ .buf = qn },
                    .{ .buf = keys },
                    .{ .buf = vals },
                    .{ .buf = reln },
                    .{ .buf = proj },
                    .{ .buf = part },
                }, .{ KV, 1, nblocks }, .{ 32, 3, 1 }, true);
                try g.run("ik_attn_merge", aw, &.{
                    .{ .buf = part },
                    .{ .buf = sink },
                    .{ .buf = gate },
                    .{ .buf = att },
                }, .{ Hq3, rows, 1 }, .{ 32, 1, 1 }, true);
                relv[rel_at] = kept_r;
                for (0..rows_u) |t| if (t != 1) cw.same(att.slice(u8, att_bytes)[t * OUT_ROW3 ..][0..OUT_ROW3], ref_att[t * OUT_ROW3 ..][0..OUT_ROW3]);
            }
            if (cw.bad != bad) std.debug.print("  failing: pos {d}, {d} rows\n", .{ pos, rows });
        }
    }
    try cw.done();
}

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    try caseSplitRef(g, r);
    try caseSplitWindow(g, r);
    try caseSplitBlocks(g, r);
    try caseSplitBias(g, r);
    try caseSplitRing(g, r);
    try caseSplitWindowShared(g, r);
    try caseG3(g, r);
}
