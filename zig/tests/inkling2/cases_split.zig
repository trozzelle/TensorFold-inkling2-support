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

fn argsOf(rows: u32, pos: u32, window: u32, f0: u32, n: u32) h.Args {
    return .{ .rows = rows, .heads = Hq, .kv = KV, .dim = D, .pos = pos, .window = window, .extent = E, .blocks = n, .queries = f0, .scale = @floatCast(1 / @sqrt(@as(f64, D))) };
}

/// The partial kernel alone over `rows` rows that start at input row `row` and position `pos`, with blocks f0 .. f0 + n:
/// `ik_attn_part`, which forms the relative bias from the rows' state and the projection, or `ik_attn_part_table`,
/// which reads the bias table `table` (rows from `row`). The partials and the output start as NaN, so anything the
/// kernels leave unwritten shows.
fn runPart(g: *const h.Gpu, x: Inputs, name: [:0]const u8, table: mtl.Buffer, row: usize, rows: u32, pos: u32, window: u32, f0: u32, n: u32) !void {
    @memset(x.part.slice(f32, rows * Hq * n * SLOT), std.math.nan(f32));
    @memset(x.out(row, rows), 0xff);
    const q: h.Bind = .{ .buf = x.qn, .off = row * Hq * D * 2 };
    const a = argsOf(rows, pos, window, f0, n);
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
    const a = argsOf(rows, pos, window, f0, n);
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

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    try caseSplitRef(g, r);
    try caseSplitWindow(g, r);
    try caseSplitBlocks(g, r);
    try caseSplitBias(g, r);
}
