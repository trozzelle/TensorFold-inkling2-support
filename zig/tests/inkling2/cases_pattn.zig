//! Inkling 2's prompt attention (ik_pattn) against an FP64 reference at the output's rounding points, the same rows run in
//! one call and in calls cut anywhere (bit for bit, with everything past a call's rows poisoned), and the two ways of
//! supplying the relative bias (read from the table ik_pattn_bias builds, formed in the kernel) against each other.
//! The dispatch under test is the engine's (`attention_prefill.encodeChunk`); the formed-bias kernel and the tensor-unit table
//! builder are test-only kernels, appended to pattn.metal here and in the bench.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const tf = @import("tensorfold");
const ap = tf.inkling2.attention_prefill;
const h = @import("harness.zig");

pub const Hq = ap.heads;
pub const KV = ap.kv_heads;
pub const D = ap.dim;
pub const E = ap.extent; // relative-bias extent, also the local layers' window
pub const REL = ap.rel_dims; // dims of the relative state
const QROW = Hq * D; // elements in a row of q, gate and the output
const POISON: u16 = 0x7fc0; // bf16 NaN
const REACH = 256; // cache rows past a call's last position that a call poisons (a block reaches 63)
/// Rows of the relative-bias table at a time in the cases: a piece's table is `tableBytes(PIECE)`, 100 MB.
const PIECE = 1024;
const tableBytes = ap.tableBytes;

/// ik_pattn's arguments: the call's rows, the first row's position, 1 on a global layer.
pub const Args = ap.Args;

/// How a call gets the relative bias: read from the table ik_pattn_bias builds (ik_pattn), or formed in the kernel from the
/// rows' relative state and the projection (a test-only kernel that runs ik_pattn's body).
pub const Form = enum { table, formed };

/// Binds of a call at its first row: q [rows][48][128], the layer's keys and values [capacity][8][128], sink [rows][48],
/// gate and out [rows][48][128] (all bf16).
pub const Io = ap.Io;

/// The engine's pipelines (pattn.metal alone: what `encodeChunk` runs) and the test-only kernels' library: pattn.metal with
/// ref_pattn.metal appended, nax.h inlined, and the two kernels of ref_pattn.metal.
pub const Pipes = struct {
    engine: ap.Pipes,
    lib: mtl.Library,
    formed: mtl.Pipeline,
    bias_mma: mtl.Pipeline,

    pub fn init(device: mtl.Device, a: std.mem.Allocator) !Pipes {
        return initUnits(device, a, device.tensorUnits());
    }

    /// The kernels on tensor units or on simdgroup matrices (the text older GPUs run: a GPU with tensor units compiles it but
    /// cannot run it, since its fragment layout is theirs).
    pub fn initUnits(device: mtl.Device, a: std.mem.Allocator, units: bool) !Pipes {
        var engine = try ap.Pipes.initUnits(device, a, units);
        errdefer engine.deinit();
        const text = try std.mem.concat(a, u8, &.{ ks.inkling2_pattn, @embedFile("ref_pattn.metal") });
        defer a.free(text);
        const src = try tf.frags.precision.source(a, units, text, ks.nax, false);
        defer a.free(src);
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        errdefer lib.deinit();
        const formed = try mtl.Pipeline.init(device, lib, "ik_pattn_formed", false);
        errdefer formed.deinit();
        const bias_mma = try mtl.Pipeline.init(device, lib, "ik_pattn_bias_mma", false);
        return .{ .engine = engine, .lib = lib, .formed = formed, .bias_mma = bias_mma };
    }

    pub fn deinit(p: *Pipes) void {
        p.engine.deinit();
        p.formed.deinit();
        p.bias_mma.deinit();
        p.lib.deinit();
    }
};

/// One call of the formed-bias kernel (ik_pattn's body, the bias computed in the kernel): a threadgroup of 64 threads for each
/// 32 rows of each of the 48 heads. `reln`: the rows' relative state [rows][48][16] bf16; `projt`: the projection as f32 columns
/// [1024][16].
pub fn encodeFormed(e: mtl.ComputeEncoder, pipe: mtl.Pipeline, a: Args, io: Io, reln: ap.B, projt: ap.B) void {
    e.setPipeline(pipe);
    e.setValue(a, 0);
    const bs = [_]ap.B{ io.q, io.k, io.v, io.sink, reln, io.gate, io.out };
    for (bs, 1..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setBuffer(projt.buf, projt.off, 8);
    e.dispatchGroups(mtl.Size.of((a.rows + 31) / 32, Hq, 1), mtl.Size.of(64, 1, 1));
}

/// The relative-bias table of `a.rows` rows built on the tensor units (whose bits differ from the engine's scalar builder),
/// from the rows' relative state `reln` [rows][48][16] and the projection `proj` [16][1024], into `table`.
pub fn encodeBiasMma(e: mtl.ComputeEncoder, pipe: mtl.Pipeline, a: Args, reln: ap.B, proj: ap.B, table: ap.B) void {
    e.setPipeline(pipe);
    e.setValue(a, 0);
    for ([_]ap.B{ reln, proj, table }, 1..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.dispatchGroups(mtl.Size.of(a.rows, Hq / 16, 1), mtl.Size.of(32, 1, 1));
}

fn finish(cb: mtl.CommandBuffer, what: []const u8) !void {
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("{s}: {s}", .{ what, msg });
        return error.GpuFailed;
    }
}

fn pow2(x: i32) f64 {
    return std.math.pow(f64, 2, @floatFromInt(x));
}

/// A sequence's inputs: `n` positions of q, sink, relative state and gate (rows), keys and values (cache), the projection;
/// `pad` NaN rows after the rows, a cache of `cap` positions (a multiple of 64) with NaN beyond n; and the buffers a call
/// writes.
pub const Seq = struct {
    n: usize,
    cap: usize,
    pad: usize,
    q: mtl.Buffer,
    sink: mtl.Buffer,
    reln: mtl.Buffer,
    gate: mtl.Buffer,
    out: mtl.Buffer,
    keys: mtl.Buffer,
    vals: mtl.Buffer,
    proj: mtl.Buffer, // the weights' projection [16][1024] bf16
    projt: mtl.Buffer, // the same as f32 columns [1024][16], what the formed-bias kernel reads
    qv: []u16,
    sv: []u16,
    rv: []u16,
    gv: []u16,
    kv: []u16,
    vv: []u16,
    pv: []u16,
    saved: [6][]u16, // what a call's poison replaced: q, sink, reln, gate rows, key and value rows

    /// Draws every input from `r`: q and k within +-3 (scores spread over a few units), v within +-1.7 (unit RMS), sinks
    /// within +-2, gates +-3, relative state +-2 and projection +-1.
    pub fn init(g: *const h.Gpu, r: std.Random, n: usize, pad: usize) !Seq {
        const rows = n + pad;
        const cap = (n + 63) / 64 * 64 + 64;
        var s: Seq = undefined;
        s.n = n;
        s.cap = cap;
        s.pad = pad;
        s.q = try g.buf(rows * QROW * 2);
        s.sink = try g.buf(rows * Hq * 2);
        s.reln = try g.buf(rows * Hq * REL * 2);
        s.gate = try g.buf(rows * QROW * 2);
        s.out = try g.buf(rows * QROW * 2);
        s.keys = try g.buf(cap * KV * D * 2);
        s.vals = try g.buf(cap * KV * D * 2);
        s.proj = try g.buf(REL * E * 2);
        s.projt = try g.buf(E * REL * 4);
        s.qv = h.fill16(s.q, rows * QROW, r, 3);
        s.sv = h.fill16(s.sink, rows * Hq, r, 2);
        s.rv = h.fill16(s.reln, rows * Hq * REL, r, 2);
        s.gv = h.fill16(s.gate, rows * QROW, r, 3);
        s.kv = h.fill16(s.keys, cap * KV * D, r, 3);
        s.vv = h.fill16(s.vals, cap * KV * D, r, 1.7);
        s.pv = h.fill16(s.proj, REL * E, r, 1);
        @memset(s.qv[n * QROW ..], POISON);
        @memset(s.sv[n * Hq ..], POISON);
        @memset(s.rv[n * Hq * REL ..], POISON);
        @memset(s.gv[n * QROW ..], POISON);
        @memset(s.kv[n * KV * D ..], POISON);
        @memset(s.vv[n * KV * D ..], POISON);
        const alloc = std.heap.page_allocator;
        const sizes = [6]usize{ pad * QROW, pad * Hq, pad * Hq * REL, pad * QROW, REACH * KV * D, REACH * KV * D };
        for (&s.saved, sizes) |*sv, len| sv.* = try alloc.alloc(u16, len);
        return s;
    }

    pub fn deinit(s: *Seq) void {
        for ([_]mtl.Buffer{ s.q, s.sink, s.reln, s.gate, s.out, s.keys, s.vals, s.proj, s.projt }) |b| b.deinit();
        for (s.saved) |sv| std.heap.page_allocator.free(sv);
    }

    fn views(s: *const Seq) [6][]u16 {
        return .{ s.qv, s.sv, s.rv, s.gv, s.kv, s.vv };
    }

    /// The row widths of `views`, in elements.
    const widths = [6]usize{ QROW, Hq, Hq * REL, QROW, KV * D, KV * D };

    /// What a call of rows .. `end` must not read, as NaN: the `pad` rows after `end` of the per-row inputs, and the
    /// cache's REACH rows from `end` on. `restore` undoes it.
    fn poison(s: *Seq, end: usize) void {
        for (s.views(), s.saved, widths, 0..) |v, saved, w, i| {
            const total = if (i < 4) s.n + s.pad else s.cap;
            const len = @min(saved.len / w, total - end);
            @memcpy(saved[0 .. len * w], v[end * w ..][0 .. len * w]);
            @memset(v[end * w ..][0 .. len * w], POISON);
        }
    }

    fn restore(s: *Seq, end: usize) void {
        for (s.views(), s.saved, widths, 0..) |v, saved, w, i| {
            const total = if (i < 4) s.n + s.pad else s.cap;
            const len = @min(saved.len / w, total - end);
            @memcpy(v[end * w ..][0 .. len * w], saved[0 .. len * w]);
        }
    }

    /// The bias bits at (row p, head hd, distance d): bf16 of the 16 products (each exact in f32) summed in index order in
    /// f32, as ik_relbias forms them.
    pub fn biasBits(s: *const Seq, p: usize, hd: usize, d: usize) u16 {
        var acc: f32 = 0;
        for (0..REL) |i| acc += @as(f32, @floatCast(h.fb(s.rv[(p * Hq + hd) * REL + i]))) * @as(f32, @floatCast(h.fb(s.pv[i * E + d])));
        return h.bf(acc);
    }

    /// The projection as f32 columns [distance][REL], the layout the formed-bias kernel reads.
    fn syncProjt(s: *const Seq) void {
        const t = s.projt.slice(f32, E * REL);
        for (0..E) |d| for (0..REL) |i| {
            t[d * REL + i] = @floatCast(h.fb(s.pv[i * E + d]));
        };
    }

    /// The call's Io: rows p0 ..
    fn io(s: *const Seq, p0: usize) Io {
        return .{
            .q = .{ .buf = s.q, .off = p0 * QROW * 2 },
            .k = .{ .buf = s.keys },
            .v = .{ .buf = s.vals },
            .sink = .{ .buf = s.sink, .off = p0 * Hq * 2 },
            .gate = .{ .buf = s.gate, .off = p0 * QROW * 2 },
            .out = .{ .buf = s.out, .off = p0 * QROW * 2 },
        };
    }

    /// The call's relative state at row p0.
    fn relnAt(s: *const Seq, p0: usize) ap.B {
        return .{ .buf = s.reln, .off = p0 * Hq * REL * 2 };
    }

    /// The call's output rows as bytes (row 0 is the call's first).
    fn rowsOf(s: *const Seq, p0: usize, n: usize) []u8 {
        return s.out.slice(u8, (s.n + s.pad) * QROW * 2)[p0 * QROW * 2 ..][0 .. n * QROW * 2];
    }
};

/// Rows p0 .. p0 + n of `s` in one call on `form` with everything past them poisoned: the output starts as 0xff bytes
/// (NaN) so a row the call leaves unwritten shows, and the rows after the call's must still hold them. The .table form
/// runs `encodeChunk` in pieces of `piece` rows through `table` (`tableBytes(piece)`).
pub fn call(g: *const h.Gpu, p: *const Pipes, form: Form, s: *Seq, p0: usize, n: usize, global: bool, table: ?mtl.Buffer, piece: usize) !void {
    const end = p0 + n;
    s.poison(end);
    defer s.restore(end);
    const after = @min(s.pad, s.n + s.pad - end);
    @memset(s.rowsOf(p0, n + after), 0xff);
    s.syncProjt();
    const a: Args = .{ .rows = @intCast(n), .p0 = @intCast(p0), .global = @intFromBool(global), .ring = @intCast(s.cap) };
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    switch (form) {
        .table => ap.encodeChunk(e, &p.engine, a, s.io(p0), s.relnAt(p0), .{ .buf = s.proj }, .{ .buf = table.? }, piece),
        .formed => encodeFormed(e, p.formed, a, s.io(p0), s.relnAt(p0), .{ .buf = s.projt }),
    }
    e.end();
    try finish(cb, "ik_pattn");
    for (s.rowsOf(p0 + n, after)) |b| if (b != 0xff) return error.WroteBeyondRows;
}

/// FP64 attention of row p, head hd: the output gate included, before the final rounding. `scratch` holds p + 1 logits.
fn refRow(s: *const Seq, p: usize, hd: usize, global: bool, scratch: []f64, o: *[D]f64, mag: *[D]f64) void {
    const scale = 1 / @sqrt(@as(f64, D));
    const first = if (!global and p + 1 > E) p + 1 - E else 0;
    const kvh = hd / (Hq / KV);
    const sink = h.fb(s.sv[p * Hq + hd]);
    var m = sink;
    for (first..p + 1) |j| {
        var x: f64 = 0;
        for (0..D) |d| x += h.fb(s.qv[(p * Hq + hd) * D + d]) * h.fb(s.kv[(j * KV + kvh) * D + d]);
        x *= scale;
        if (p - j < E) x += h.fb(s.biasBits(p, hd, p - j));
        scratch[j] = x;
        m = @max(m, x);
    }
    var z = @exp(sink - m);
    for (first..p + 1) |j| z += @exp(scratch[j] - m);
    o.* = @splat(0);
    mag.* = @splat(0);
    for (first..p + 1) |j| {
        const w = @exp(scratch[j] - m) / z;
        for (o, mag, 0..) |*od, *md, d| {
            const v = h.fb(s.vv[(j * KV + kvh) * D + d]);
            od.* += w * v;
            md.* += w * @abs(v);
        }
    }
}

/// Checks the call's rows `rows` (call row numbers) and heads `heads` against FP64: two bf16 steps, as the one-row
/// attention's check (two rounding points), or within 2^-10 of the weighted values' magnitudes: the tensor op takes the
/// probabilities at ten fraction bits, so an output that cancels toward zero keeps an error of that size.
fn checkRef(c: *h.Check, s: *const Seq, p0: usize, global: bool, rows: []const usize, heads: []const usize) !void {
    const scratch = try std.heap.page_allocator.alloc(f64, s.n + 1);
    defer std.heap.page_allocator.free(scratch);
    const out = s.out.slice(u16, (s.n + s.pad) * QROW);
    var o: [D]f64 = undefined;
    var mag: [D]f64 = undefined;
    for (rows) |t| for (heads) |hd| {
        const p = p0 + t;
        refRow(s, p, hd, global, scratch, &o, &mag);
        for (0..D) |d| {
            const gt = h.fb(s.gv[(p * Hq + hd) * D + d]);
            const gate = 1 / (1 + @exp(-gt));
            c.b16a(out[(p * Hq + hd) * D + d], h.rb(o[d]) * gate, 2, 1e-6 + mag[d] * gate / 1024);
        }
    };
}

fn rowList(comptime lo: usize, comptime hi: usize) [hi - lo]usize {
    var a: [hi - lo]usize = undefined;
    for (&a, lo..) |*x, i| x.* = i;
    return a;
}

/// Heads that straddle the key-head groups (5|6, 11|12) and end them (47), with dominant sinks on 5 and 6 and negligible
/// ones on 11 and 12.
const heads_ref = [_]usize{ 0, 5, 6, 11, 12, 29, 47 };

fn setSinks(s: *Seq) void {
    for (0..s.n) |t| {
        s.sv[t * Hq + 5] = h.bf(25);
        s.sv[t * Hq + 6] = h.bf(25);
        s.sv[t * Hq + 11] = h.bf(-25);
        s.sv[t * Hq + 12] = h.bf(-25);
    }
}

/// FP64 cases, local and global layers: the sequence's first rows (a row with one key, the window's growth), rows across
/// the window and bias-extent edges (positions 1,023, 1,024, 1,025) and through a tile whose last rows have a whole block
/// behind their window, and a deep global call; the table form and the formed form.
fn caseRef(g: *const h.Gpu, pipes: *const Pipes, r: std.Random) !void {
    const Run = struct { p0: usize, rows: usize, heads: []const usize, row_list: []const usize };
    const all_heads = comptime rowList(0, Hq);
    const first_rows = comptime rowList(0, 72);
    const edge_rows = comptime rowList(0, 100);
    const deep_rows = [_]usize{ 0, 1, 5, 31, 32, 40, 63, 64, 69 };
    const table = try g.buf(tableBytes(100));
    defer table.deinit();
    for ([_]Form{ .table, .formed }) |form| for ([_]bool{ false, true }) |global| {
        const runs = [_]Run{
            .{ .p0 = 0, .rows = 72, .heads = &heads_ref, .row_list = &first_rows },
            .{ .p0 = 1000, .rows = 100, .heads = &all_heads, .row_list = &edge_rows },
            .{ .p0 = 3000, .rows = 70, .heads = &heads_ref, .row_list = &deep_rows },
        };
        var c: h.Check = .{ .name = "" };
        var name_buf: [64]u8 = undefined;
        c.name = try std.fmt.bufPrint(&name_buf, "pattn {s} {s}", .{ @tagName(form), if (global) "global" else "local" });
        for (runs) |rc| {
            var s = try Seq.init(g, r, rc.p0 + rc.rows, 64);
            defer s.deinit();
            setSinks(&s);
            try call(g, pipes, form, &s, rc.p0, rc.rows, global, table, PIECE);
            const bad = c.bad;
            try checkRef(&c, &s, rc.p0, global, rc.row_list, rc.heads);
            if (c.bad != bad) std.debug.print("  failing: p0 {d}, {d} rows\n", .{ rc.p0, rc.rows });
        }
        try c.done();
    };
}

/// A relative state of +-1 and a projection whose every column holds a large pair of opposite signs among small terms: for
/// the heads whose two state values agree the pair cancels, and wherever it sits in the f32 sum it absorbs the small
/// terms' low bits, so the sum's order shows in the bf16 biases (about a tenth of them change when it does).
fn cancelling(s: *Seq, r: std.Random) void {
    for (s.rv) |*v| v.* = if (r.boolean()) 0x3f80 else 0xbf80;
    for (0..E) |d| {
        for (0..REL) |i| s.pv[i * E + d] = h.bf((r.float(f64) * 2 - 1) / 16);
        const up = r.uintLessThan(usize, REL);
        const down_i = (up + 1 + r.uintLessThan(usize, REL - 1)) % REL;
        const big = pow2(r.intRangeAtMost(i32, 8, 12));
        s.pv[up * E + d] = h.bf(big);
        s.pv[down_i * E + d] = h.bf(-big);
    }
}

/// The bias bits: the scalar table builder's table equals the host's sums for every row, head and distance; the kernel
/// that forms the bias gives the bytes of the one that reads the table, over windows across the extent edge on both
/// layer kinds; the tensor-unit table's differences from the scalar one are counted (printed, not failed: its sums are
/// the unit's own).
fn caseBias(g: *const h.Gpu, pipes: *const Pipes, r: std.Random) !void {
    const rows = 130;
    var s = try Seq.init(g, r, 1024 + rows, 64);
    defer s.deinit();
    const table = try g.buf(tableBytes(rows));
    defer table.deinit();
    const tv = table.slice(u16, rows * Hq * E);
    const saved = try std.heap.page_allocator.alloc(u8, rows * QROW * 2);
    defer std.heap.page_allocator.free(saved);
    for ([_]bool{ false, true }) |adversarial| {
        if (adversarial) cancelling(&s, r);
        var c: h.Check = .{ .name = if (adversarial) "pattn bias bits (cancelling sums)" else "pattn bias bits" };
        const p0 = 1024 - 40;
        const a: Args = .{ .rows = rows, .p0 = p0, .global = 0, .ring = @intCast(s.cap) };
        for ([_]bool{ false, true }) |mma| {
            @memset(tv, 0xffff);
            const cb = g.queue.commandBuffer();
            const e = cb.compute(.serial);
            if (mma) encodeBiasMma(e, pipes.bias_mma, a, s.relnAt(p0), .{ .buf = s.proj }, .{ .buf = table }) else ap.encodeBias(e, pipes.engine.bias, a, s.relnAt(p0), .{ .buf = s.proj }, .{ .buf = table });
            e.end();
            try finish(cb, "ik_pattn_bias");
            var differ: usize = 0;
            for (0..rows) |t| for (0..Hq) |hd| for (0..E) |d| {
                const got = tv[(t * Hq + hd) * E + ((p0 + t - d) & (E - 1))];
                if (got != s.biasBits(p0 + t, hd, d)) differ += 1;
            };
            if (mma) {
                std.debug.print("  tensor-unit table ({s}): {d} of {d} bf16 biases differ from the index-order sums\n", .{ if (adversarial) "cancelling" else "uniform", differ, rows * Hq * E });
            } else {
                c.n += 1;
                c.bad += @intFromBool(differ != 0);
                if (differ != 0) std.debug.print("  scalar table: {d} biases differ from the host's\n", .{differ});
            }
        }
        for ([_]bool{ false, true }) |global| {
            try call(g, pipes, .formed, &s, p0, rows, global, null, PIECE);
            @memcpy(saved, s.rowsOf(p0, rows));
            try call(g, pipes, .table, &s, p0, rows, global, table, PIECE);
            c.same(s.rowsOf(p0, rows), saved);
        }
        try c.done();
    }
}

/// Rows of the sequence, the baseline's, against calls cut anywhere.
fn partitions(g: *const h.Gpu, pipes: *const Pipes, form: Form, s: *Seq, base_p0: usize, base_rows: usize, global: bool, table: ?mtl.Buffer, c: *h.Check, saved: []u8) !void {
    try call(g, pipes, form, s, base_p0, base_rows, global, table, PIECE);
    @memcpy(saved[0 .. base_rows * QROW * 2], s.rowsOf(base_p0, base_rows));
    const Cut = struct {
        g: *const h.Gpu,
        pipes: *const Pipes,
        form: Form,
        s: *Seq,
        base_p0: usize,
        global: bool,
        table: ?mtl.Buffer,
        c: *h.Check,
        saved: []u8,
        fn one(x: @This(), p0: usize, n: usize) !void {
            try call(x.g, x.pipes, x.form, x.s, x.base_p0 + p0, n, x.global, x.table, PIECE);
            const got = x.s.rowsOf(x.base_p0 + p0, n);
            x.c.n += 1;
            const row_bytes = QROW * 2;
            for (0..n) |t| {
                if (!std.mem.eql(u8, got[t * row_bytes ..][0..row_bytes], x.saved[(p0 + t) * row_bytes ..][0..row_bytes])) {
                    x.c.bad += 1;
                    std.debug.print("  differs: call at row {d} of {d}, call row {d} (position {d})\n", .{ p0, n, t, x.base_p0 + p0 + t });
                    return;
                }
            }
        }
    };
    const x: Cut = .{ .g = g, .pipes = pipes, .form = form, .s = s, .base_p0 = base_p0, .global = global, .table = table, .c = c, .saved = saved };
    // every p0 mod 64: the calls' first row at each place in its tile, the rows' ends unchanged
    for (1..64) |res| try x.one(res, base_rows - res);
    // cuts, as head and tail calls: inside a 16-key group and a 64-key block (37), a key group's end (4,095), mid-chunk
    // (2,050), and the window and extent edges
    for ([_]usize{ 4095, 37, 2050, 1024, 1023, 1025 }) |cut| {
        if (cut >= base_rows) continue;
        try x.one(0, cut);
        try x.one(cut, base_rows - cut);
    }
    // 1-31-row tails at the end and heads at the start
    for (1..32) |n| {
        try x.one(base_rows - n, n);
        try x.one(0, n);
    }
    // a run of consecutive small calls, then a few random pieces
    var at: usize = 0;
    for ([_]usize{ 5, 7, 3, 11, 13, 2, 19, 31, 1, 33 }) |n| {
        try x.one(at, n);
        at += n;
    }
    var prng = std.Random.DefaultPrng.init(0x9a77 + base_p0);
    const rr = prng.random();
    for (0..24) |_| {
        const n = rr.intRangeAtMost(usize, 1, 100);
        try x.one(rr.uintLessThan(usize, base_rows - n), n);
    }
}

/// The same rows in one 4,096-row call and in calls cut every way (see `partitions`), on local and global layers, with the
/// sequence starting at position 0 and (the table form only: fewer cuts matter there) 4,096.
fn casePartition(g: *const h.Gpu, pipes: *const Pipes, r: std.Random) !void {
    const rows = 4096;
    const saved = try std.heap.page_allocator.alloc(u8, rows * QROW * 2);
    defer std.heap.page_allocator.free(saved);
    const table = try g.buf(tableBytes(PIECE));
    defer table.deinit();
    for ([_]usize{ 0, 4096 }) |base| {
        var s = try Seq.init(g, r, base + rows, 64);
        defer s.deinit();
        setSinks(&s);
        for ([_]Form{ .table, .formed }) |form| for ([_]bool{ false, true }) |global| {
            if (form == .formed and base != 0) continue;
            var c: h.Check = .{ .name = "" };
            var name_buf: [64]u8 = undefined;
            c.name = try std.fmt.bufPrint(&name_buf, "pattn partition {s} {s} from {d}", .{ @tagName(form), if (global) "global" else "local", base });
            try partitions(g, pipes, form, &s, base, rows, global, table, &c, saved);
            try c.done();
        };
    }
}

/// A call's bytes do not depend on the piece the table is built in: one piece, 512, 96 and 32 rows, local and global.
fn casePieces(g: *const h.Gpu, pipes: *const Pipes, r: std.Random) !void {
    const rows = 1536;
    var s = try Seq.init(g, r, 1000 + rows, 64);
    defer s.deinit();
    const table = try g.buf(tableBytes(rows));
    defer table.deinit();
    const saved = try std.heap.page_allocator.alloc(u8, rows * QROW * 2);
    defer std.heap.page_allocator.free(saved);
    for ([_]bool{ false, true }) |global| {
        var c: h.Check = .{ .name = if (global) "pattn pieces global" else "pattn pieces local" };
        try call(g, pipes, .table, &s, 1000, rows, global, table, rows);
        @memcpy(saved, s.rowsOf(1000, rows));
        for ([_]usize{ 512, 96, 32 }) |piece| {
            try call(g, pipes, .table, &s, 1000, rows, global, table, piece);
            c.same(s.rowsOf(1000, rows), saved);
        }
        try c.done();
    }
}

/// A ring gives a linear cache's bytes across wraps, local and global, with non-live ring slots poisoned before each chunk.
fn caseRing(g: *const h.Gpu, pipes: *const Pipes, r: std.Random) !void {
    const R_small: usize = 1280;
    const T: usize = 3 * R_small + 100;
    const lin_cap: usize = std.mem.alignForward(usize, T, 64);
    const max_chunk: usize = 64;
    const q = try g.buf(max_chunk * QROW * 2);
    const k = try g.buf(max_chunk * KV * D * 2);
    const v = try g.buf(max_chunk * KV * D * 2);
    const rr = try g.buf(max_chunk * Hq * REL * 2);
    const qs = try g.buf(Hq * D * 2);
    const rs = try g.buf(Hq * REL * 2);
    const qn = try g.buf(max_chunk * QROW * 2);
    const reln = try g.buf(max_chunk * Hq * REL * 2);
    const proj = try g.buf(REL * E * 2);
    const sink = try g.buf(max_chunk * Hq * 2);
    const gate = try g.buf(max_chunk * QROW * 2);
    const table = try g.buf(tableBytes(max_chunk));
    const out_lin = try g.buf(max_chunk * QROW * 2);
    const out_ring = try g.buf(max_chunk * QROW * 2);
    const lin_k = try g.buf(lin_cap * KV * D * 2);
    const lin_v = try g.buf(lin_cap * KV * D * 2);
    const ring_k = try g.buf(lin_cap * KV * D * 2);
    const ring_v = try g.buf(lin_cap * KV * D * 2);
    defer for ([_]mtl.Buffer{ q, k, v, rr, qs, rs, qn, reln, proj, sink, gate, table, out_lin, out_ring, lin_k, lin_v, ring_k, ring_v }) |b| b.deinit();
    _ = h.fill16(qs, Hq * D, r, 1.5);
    _ = h.fill16(rs, Hq * REL, r, 1.5);
    _ = h.fill16(proj, REL * E, r, 1.0);
    const chunks = [_]usize{ 64, 37, 64, 1, 64, 31 };
    for ([_]bool{ false, true }) |global| {
        const ring_slots: usize = if (global) lin_cap else R_small;
        var c: h.Check = .{ .name = if (global) "pattn ring global" else "pattn ring local" };
        var pos: usize = 0;
        var ci: usize = 0;
        while (pos < T) : (ci += 1) {
            const n = @min(chunks[ci % chunks.len], T - pos);
            _ = h.fill16(q, n * QROW, r, 3);
            _ = h.fill16(k, n * KV * D, r, 3);
            _ = h.fill16(v, n * KV * D, r, 1.7);
            _ = h.fill16(rr, n * Hq * REL, r, 2);
            _ = h.fill16(sink, n * Hq, r, 2);
            _ = h.fill16(gate, n * QROW, r, 3);
            var poisoned: [R_small]bool = @splat(false);
            if (!global) {
                const live_lo = std.mem.alignBackward(usize, pos -| (E - 1), 64);
                var keep: [R_small]bool = @splat(false);
                for (live_lo..pos) |p| keep[p % R_small] = true;
                const rk = ring_k.slice(u16, R_small * KV * D);
                const rv = ring_v.slice(u16, R_small * KV * D);
                for (0..R_small) |s| if (!keep[s]) {
                    poisoned[s] = true;
                    @memset(rk[s * KV * D ..][0 .. KV * D], POISON);
                    @memset(rv[s * KV * D ..][0 .. KV * D], POISON);
                };
            }
            const prep_base: h.Args = .{ .rows = @intCast(n), .heads = Hq, .kv = KV, .dim = D, .pos = @intCast(pos), .flags = if (global) 2 else 0, .eps = 1e-6, .cap = 22.627416997969522, .thr = 8, .alpha = 0.1, .floor_n = 8192 };
            var prep_lin = prep_base;
            prep_lin.ring = @intCast(lin_cap);
            try g.run("ik_prep", prep_lin, &.{ .{ .buf = q }, .{ .buf = k }, .{ .buf = v }, .{ .buf = rr }, .{ .buf = qs }, .{ .buf = rs }, .{ .buf = qn }, .{ .buf = lin_k }, .{ .buf = lin_v }, .{ .buf = reln } }, .{ 2 * Hq + 2 * KV, n, 1 }, .{ 32, 1, 1 }, true);
            var prep_ring = prep_base;
            prep_ring.ring = @intCast(ring_slots);
            try g.run("ik_prep", prep_ring, &.{ .{ .buf = q }, .{ .buf = k }, .{ .buf = v }, .{ .buf = rr }, .{ .buf = qs }, .{ .buf = rs }, .{ .buf = qn }, .{ .buf = ring_k }, .{ .buf = ring_v }, .{ .buf = reln } }, .{ 2 * Hq + 2 * KV, n, 1 }, .{ 32, 1, 1 }, true);
            const cb1 = g.queue.commandBuffer();
            const e1 = cb1.compute(.serial);
            ap.encodeChunk(e1, &pipes.engine, .{ .rows = @intCast(n), .p0 = @intCast(pos), .global = @intFromBool(global), .ring = @intCast(lin_cap) }, .{ .q = .{ .buf = qn }, .k = .{ .buf = lin_k }, .v = .{ .buf = lin_v }, .sink = .{ .buf = sink }, .gate = .{ .buf = gate }, .out = .{ .buf = out_lin } }, .{ .buf = reln }, .{ .buf = proj }, .{ .buf = table }, max_chunk);
            e1.end();
            try finish(cb1, "ik_pattn linear");
            const cb2 = g.queue.commandBuffer();
            const e2 = cb2.compute(.serial);
            ap.encodeChunk(e2, &pipes.engine, .{ .rows = @intCast(n), .p0 = @intCast(pos), .global = @intFromBool(global), .ring = @intCast(ring_slots) }, .{ .q = .{ .buf = qn }, .k = .{ .buf = ring_k }, .v = .{ .buf = ring_v }, .sink = .{ .buf = sink }, .gate = .{ .buf = gate }, .out = .{ .buf = out_ring } }, .{ .buf = reln }, .{ .buf = proj }, .{ .buf = table }, max_chunk);
            e2.end();
            try finish(cb2, "ik_pattn ring");
            const bytes = n * QROW * 2;
            c.same(out_ring.slice(u8, bytes), out_lin.slice(u8, bytes));
            for (out_ring.slice(u16, n * QROW)) |x| {
                if (std.math.isNan(h.fb(x))) {
                    c.bad += 1;
                    break;
                }
            }
            if (!global) {
                var written: [R_small]bool = @splat(false);
                for (pos..pos + n) |p| written[p % R_small] = true;
                const rk = ring_k.slice(u16, R_small * KV * D);
                const rv = ring_v.slice(u16, R_small * KV * D);
                for (0..R_small) |s| if (poisoned[s] and !written[s]) {
                    c.n += 1;
                    for (rk[s * KV * D ..][0 .. KV * D], rv[s * KV * D ..][0 .. KV * D]) |kv_x, vv_x| {
                        if (kv_x != POISON or vv_x != POISON) {
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

/// 24 query heads over 8 key heads (G = 3): FP64 reference parity (local and global), exact bias bits, piece invariance,
/// chunk-cut invariance, and NaN containment past each call's rows.
fn caseG3(g: *const h.Gpu, r: std.Random) !void {
    const hq: usize = 24;
    const kv: usize = 8;
    const qrow = hq * D;
    const alloc = std.heap.page_allocator;
    var pipes = try ap.Pipes.initFor(g.device, alloc, hq, kv);
    defer pipes.deinit();
    if (g.device.tensorUnits()) {
        var older = try ap.Pipes.initUnitsFor(g.device, alloc, false, hq, kv);
        older.deinit();
    }

    const SeqG3 = struct {
        n: usize,
        cap: usize,
        pad: usize,
        q: mtl.Buffer,
        sink: mtl.Buffer,
        reln: mtl.Buffer,
        gate: mtl.Buffer,
        out: mtl.Buffer,
        keys: mtl.Buffer,
        vals: mtl.Buffer,
        proj: mtl.Buffer,
        qv: []u16,
        sv: []u16,
        rv: []u16,
        gv: []u16,
        kv_buf: []u16,
        vv: []u16,
        pv: []u16,
        saved: [6][]u16,

        const widths = [6]usize{ qrow, hq, hq * REL, qrow, kv * D, kv * D };

        fn init(gpu: *const h.Gpu, rng: std.Random, n: usize, pad: usize) !@This() {
            const rows = n + pad;
            const cap = (n + 63) / 64 * 64 + 64;
            var s: @This() = undefined;
            s.n = n;
            s.cap = cap;
            s.pad = pad;
            s.q = try gpu.buf(rows * qrow * 2);
            s.sink = try gpu.buf(rows * hq * 2);
            s.reln = try gpu.buf(rows * hq * REL * 2);
            s.gate = try gpu.buf(rows * qrow * 2);
            s.out = try gpu.buf(rows * qrow * 2);
            s.keys = try gpu.buf(cap * kv * D * 2);
            s.vals = try gpu.buf(cap * kv * D * 2);
            s.proj = try gpu.buf(REL * E * 2);
            s.qv = h.fill16(s.q, rows * qrow, rng, 3);
            s.sv = h.fill16(s.sink, rows * hq, rng, 2);
            s.rv = h.fill16(s.reln, rows * hq * REL, rng, 2);
            s.gv = h.fill16(s.gate, rows * qrow, rng, 3);
            s.kv_buf = h.fill16(s.keys, cap * kv * D, rng, 3);
            s.vv = h.fill16(s.vals, cap * kv * D, rng, 1.7);
            s.pv = h.fill16(s.proj, REL * E, rng, 1);
            @memset(s.qv[n * qrow ..], POISON);
            @memset(s.sv[n * hq ..], POISON);
            @memset(s.rv[n * hq * REL ..], POISON);
            @memset(s.gv[n * qrow ..], POISON);
            @memset(s.kv_buf[n * kv * D ..], POISON);
            @memset(s.vv[n * kv * D ..], POISON);
            const sizes = [6]usize{ pad * qrow, pad * hq, pad * hq * REL, pad * qrow, REACH * kv * D, REACH * kv * D };
            for (&s.saved, sizes) |*sv, len| sv.* = try alloc.alloc(u16, len);
            return s;
        }

        fn deinit(s: *@This()) void {
            for ([_]mtl.Buffer{ s.q, s.sink, s.reln, s.gate, s.out, s.keys, s.vals, s.proj }) |b| b.deinit();
            for (s.saved) |sv| alloc.free(sv);
        }

        fn views(s: *const @This()) [6][]u16 {
            return .{ s.qv, s.sv, s.rv, s.gv, s.kv_buf, s.vv };
        }

        fn poison(s: *@This(), end: usize) void {
            for (s.views(), s.saved, widths, 0..) |v, saved, w, i| {
                const total = if (i < 4) s.n + s.pad else s.cap;
                const len = @min(saved.len / w, total - end);
                @memcpy(saved[0 .. len * w], v[end * w ..][0 .. len * w]);
                @memset(v[end * w ..][0 .. len * w], POISON);
            }
        }

        fn restore(s: *@This(), end: usize) void {
            for (s.views(), s.saved, widths, 0..) |v, saved, w, i| {
                const total = if (i < 4) s.n + s.pad else s.cap;
                const len = @min(saved.len / w, total - end);
                @memcpy(v[end * w ..][0 .. len * w], saved[0 .. len * w]);
            }
        }

        fn biasBits(s: *const @This(), p: usize, hd: usize, d: usize) u16 {
            var acc: f32 = 0;
            for (0..REL) |i| acc += @as(f32, @floatCast(h.fb(s.rv[(p * hq + hd) * REL + i]))) * @as(f32, @floatCast(h.fb(s.pv[i * E + d])));
            return h.bf(acc);
        }

        fn io(s: *const @This(), p0: usize) Io {
            return .{
                .q = .{ .buf = s.q, .off = p0 * qrow * 2 },
                .k = .{ .buf = s.keys },
                .v = .{ .buf = s.vals },
                .sink = .{ .buf = s.sink, .off = p0 * hq * 2 },
                .gate = .{ .buf = s.gate, .off = p0 * qrow * 2 },
                .out = .{ .buf = s.out, .off = p0 * qrow * 2 },
            };
        }

        fn relnAt(s: *const @This(), p0: usize) ap.B {
            return .{ .buf = s.reln, .off = p0 * hq * REL * 2 };
        }

        fn rowsOf(s: *const @This(), p0: usize, n: usize) []u8 {
            return s.out.slice(u8, (s.n + s.pad) * qrow * 2)[p0 * qrow * 2 ..][0 .. n * qrow * 2];
        }

        fn callG3(s: *@This(), gpu: *const h.Gpu, p: *const ap.Pipes, p0: usize, n: usize, global: bool, table: mtl.Buffer, piece: usize) !void {
            const end = p0 + n;
            s.poison(end);
            defer s.restore(end);
            const after = @min(s.pad, s.n + s.pad - end);
            @memset(s.rowsOf(p0, n + after), 0xff);
            const a: Args = .{ .rows = @intCast(n), .p0 = @intCast(p0), .global = @intFromBool(global), .ring = @intCast(s.cap) };
            const cb = gpu.queue.commandBuffer();
            const e = cb.compute(.serial);
            ap.encodeChunk(e, p, a, s.io(p0), s.relnAt(p0), .{ .buf = s.proj }, .{ .buf = table }, piece);
            e.end();
            try finish(cb, "ik_pattn g3");
            for (s.rowsOf(p0 + n, after)) |b| if (b != 0xff) return error.WroteBeyondRows;
            const out_u16 = s.out.slice(u16, (s.n + s.pad) * qrow)[p0 * qrow ..][0 .. n * qrow];
            for (out_u16) |x| if (std.math.isNan(h.fb(x))) return error.NanInOutput;
        }
    };

    // 1. FP64 scalar reference parity (sliding and global) with NaN poisoning past every call
    const heads_g3 = [_]usize{ 0, 2, 3, 5, 6, 11, 12, 23 };
    const all_heads_g3 = comptime rowList(0, hq);
    const first_rows = comptime rowList(0, 72);
    const edge_rows = comptime rowList(0, 100);
    const deep_rows = [_]usize{ 0, 1, 5, 31, 32, 40, 63, 64, 69 };
    const Run = struct { p0: usize, rows: usize, heads: []const usize, row_list: []const usize };
    const runs = [_]Run{
        .{ .p0 = 0, .rows = 72, .heads = &heads_g3, .row_list = &first_rows },
        .{ .p0 = 1000, .rows = 100, .heads = &all_heads_g3, .row_list = &edge_rows },
        .{ .p0 = 3000, .rows = 70, .heads = &heads_g3, .row_list = &deep_rows },
    };
    const ref_table = try g.buf(ap.tableBytesFor(100, hq));
    defer ref_table.deinit();
    const scratch = try alloc.alloc(f64, 3071);
    defer alloc.free(scratch);
    const scale = 1 / @sqrt(@as(f64, D));
    for ([_]bool{ false, true }) |global| {
        var c: h.Check = .{ .name = if (global) "pattn g3 (24/8) global" else "pattn g3 (24/8) local" };
        for (runs) |rc| {
            var s = try SeqG3.init(g, r, rc.p0 + rc.rows, 64);
            defer s.deinit();
            for (0..s.n) |t| {
                s.sv[t * hq + 2] = h.bf(25);
                s.sv[t * hq + 3] = h.bf(25);
                s.sv[t * hq + 5] = h.bf(-25);
                s.sv[t * hq + 6] = h.bf(-25);
            }
            try s.callG3(g, &pipes, rc.p0, rc.rows, global, ref_table, PIECE);
            const out = s.out.slice(u16, (s.n + s.pad) * qrow);
            const bad = c.bad;
            for (rc.row_list) |t| for (rc.heads) |hd| {
                const p = rc.p0 + t;
                const first = if (!global and p + 1 > E) p + 1 - E else 0;
                const kvh = hd / (hq / kv);
                const sink = h.fb(s.sv[p * hq + hd]);
                var m = sink;
                for (first..p + 1) |j| {
                    var x: f64 = 0;
                    for (0..D) |d| x += h.fb(s.qv[(p * hq + hd) * D + d]) * h.fb(s.kv_buf[(j * kv + kvh) * D + d]);
                    x *= scale;
                    if (p - j < E) x += h.fb(s.biasBits(p, hd, p - j));
                    scratch[j] = x;
                    m = @max(m, x);
                }
                var z = @exp(sink - m);
                for (first..p + 1) |j| z += @exp(scratch[j] - m);
                var o: [D]f64 = @splat(0);
                var mag: [D]f64 = @splat(0);
                for (first..p + 1) |j| {
                    const w = @exp(scratch[j] - m) / z;
                    for (&o, &mag, 0..) |*od, *md, d| {
                        const v = h.fb(s.vv[(j * kv + kvh) * D + d]);
                        od.* += w * v;
                        md.* += w * @abs(v);
                    }
                }
                for (0..D) |d| {
                    const gt = h.fb(s.gv[(p * hq + hd) * D + d]);
                    const gate = 1 / (1 + @exp(-gt));
                    c.b16a(out[(p * hq + hd) * D + d], h.rb(o[d]) * gate, 2, 1e-6 + mag[d] * gate / 1024);
                }
            };
            if (c.bad != bad) std.debug.print("  failing g3: p0 {d}, {d} rows\n", .{ rc.p0, rc.rows });
        }
        try c.done();
    }

    // 2. Bias table bits for 24 heads against host index-order sums
    {
        const rows: usize = 96;
        const p0: usize = 1024 - 32;
        var s = try SeqG3.init(g, r, p0 + rows, 64);
        defer s.deinit();
        const table = try g.buf(ap.tableBytesFor(rows, hq));
        defer table.deinit();
        const tv = table.slice(u16, rows * hq * E);
        @memset(tv, 0xffff);
        const a: Args = .{ .rows = rows, .p0 = p0, .global = 0, .ring = @intCast(s.cap) };
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);
        ap.encodeBiasFor(e, pipes.bias, hq, a, s.relnAt(p0), .{ .buf = s.proj }, .{ .buf = table });
        e.end();
        try finish(cb, "ik_pattn_bias g3");
        var c: h.Check = .{ .name = "pattn g3 (24/8) bias bits" };
        var differ: usize = 0;
        for (0..rows) |t| for (0..hq) |hd| for (0..E) |d| {
            const got = tv[(t * hq + hd) * E + ((p0 + t - d) & (E - 1))];
            if (got != s.biasBits(p0 + t, hd, d)) differ += 1;
        };
        c.n += 1;
        c.bad += @intFromBool(differ != 0);
        try c.done();
    }

    // 3. Chunk-cut and piece invariance (sliding and global) with NaN poisoning past every sub-call
    const cut_rows: usize = 1200;
    const saved = try alloc.alloc(u8, cut_rows * qrow * 2);
    defer alloc.free(saved);
    const cut_table = try g.buf(ap.tableBytesFor(512, hq));
    defer cut_table.deinit();
    var cs = try SeqG3.init(g, r, cut_rows, 64);
    defer cs.deinit();
    for (0..cs.n) |t| {
        cs.sv[t * hq + 2] = h.bf(25);
        cs.sv[t * hq + 3] = h.bf(25);
    }
    const row_bytes = qrow * 2;
    for ([_]bool{ false, true }) |global| {
        var c: h.Check = .{ .name = if (global) "pattn g3 (24/8) partition global" else "pattn g3 (24/8) partition local" };
        try cs.callG3(g, &pipes, 0, cut_rows, global, cut_table, 512);
        @memcpy(saved, cs.rowsOf(0, cut_rows));
        for ([_]usize{ 96, 32 }) |piece| {
            try cs.callG3(g, &pipes, 0, cut_rows, global, cut_table, piece);
            c.same(cs.rowsOf(0, cut_rows), saved);
        }
        const Sub = struct { p0: usize, n: usize };
        const subs = [_]Sub{
            .{ .p0 = 1, .n = cut_rows - 1 },
            .{ .p0 = 7, .n = cut_rows - 7 },
            .{ .p0 = 31, .n = cut_rows - 31 },
            .{ .p0 = 32, .n = cut_rows - 32 },
            .{ .p0 = 37, .n = cut_rows - 37 },
            .{ .p0 = 63, .n = cut_rows - 63 },
            .{ .p0 = 0, .n = 37 },
            .{ .p0 = 0, .n = 1023 },
            .{ .p0 = 1023, .n = cut_rows - 1023 },
            .{ .p0 = 0, .n = 1024 },
            .{ .p0 = 1024, .n = cut_rows - 1024 },
            .{ .p0 = 0, .n = 1025 },
            .{ .p0 = 1025, .n = cut_rows - 1025 },
            .{ .p0 = cut_rows - 1, .n = 1 },
            .{ .p0 = cut_rows - 13, .n = 13 },
            .{ .p0 = cut_rows - 31, .n = 31 },
            .{ .p0 = 0, .n = 33 },
        };
        for (subs) |sub| {
            try cs.callG3(g, &pipes, sub.p0, sub.n, global, cut_table, 512);
            c.same(cs.rowsOf(sub.p0, sub.n), saved[sub.p0 * row_bytes ..][0 .. sub.n * row_bytes]);
        }
        var at: usize = 0;
        for ([_]usize{ 5, 7, 3, 11, 13, 2, 19, 31, 1, 33 }) |n| {
            try cs.callG3(g, &pipes, at, n, global, cut_table, 512);
            c.same(cs.rowsOf(at, n), saved[at * row_bytes ..][0 .. n * row_bytes]);
            at += n;
        }
        try c.done();
    }
}

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    var pipes = try Pipes.init(g.device, std.heap.page_allocator);
    defer pipes.deinit();
    // the text older GPUs run (simdgroup matrices) builds here too; its fragment layout is theirs, so only this GPU's runs
    if (g.device.tensorUnits()) {
        var older = try Pipes.initUnits(g.device, std.heap.page_allocator, false);
        older.deinit();
    }
    try caseRef(g, &pipes, r);
    try caseBias(g, &pipes, r);
    try casePieces(g, &pipes, r);
    try casePartition(g, &pipes, r);
    try caseRing(g, &pipes, r);
    try caseG3(g, r);
}
