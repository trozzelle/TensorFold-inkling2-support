//! Inkling 2's prompt attention (ik_pattn) against an FP64 reference at the output's rounding points, the same rows run in
//! one call and in calls cut anywhere (bit for bit, with everything past a call's rows poisoned), and the two ways of
//! supplying the relative bias (read from the table ik_pattn_bias builds, formed in the kernel) against each other.
//! Also the dispatch the engine runs for a chunk's attention (`encodeChunk`), which the benches use.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const tf = @import("tensorfold");
const h = @import("harness.zig");

pub const Hq = 48;
pub const KV = 8;
pub const D = 128;
pub const E = 1024; // relative-bias extent, also the local layers' window
pub const REL = 16; // dims of the relative state
const QROW = Hq * D; // elements in a row of q, gate and the output
const POISON: u16 = 0x7fc0; // bf16 NaN
const REACH = 256; // cache rows past a call's last position that a call poisons (a block reaches 63)
/// Rows of the relative-bias table at a time in the cases: a piece's table is `tableBytes(PIECE)`, 100 MB.
const PIECE = 1024;

/// ik_pattn's arguments: the call's rows, the first row's position, 1 on a global layer.
pub const Args = extern struct { rows: u32, p0: u32, global: u32 };

/// How a call gets the relative bias: read from the table ik_pattn_bias builds (ik_pattn), or formed in the kernel from the
/// rows' relative state and the projection (a test-only kernel that runs ik_pattn's body).
pub const Form = enum { table, formed };

/// Binds of a call at its first row: q [rows][48][128], the layer's keys and values [capacity][8][128], sink [rows][48],
/// gate and out [rows][48][128] (all bf16).
pub const Io = struct { q: h.Bind, k: h.Bind, v: h.Bind, sink: h.Bind, gate: h.Bind, out: h.Bind };

/// The pattn library (pattn.metal with the test-only kernels, nax.h inlined) and its pipelines.
pub const Pipes = struct {
    lib: mtl.Library,
    attn: mtl.Pipeline,
    bias: mtl.Pipeline,
    formed: mtl.Pipeline,
    bias_mma: mtl.Pipeline,

    pub fn init(device: mtl.Device, a: std.mem.Allocator) !Pipes {
        return initWith(device, a, "");
    }

    /// `prefix` goes before the source (a `#define TF_SIMD_FRAGS 1` builds the text older GPUs run).
    pub fn initWith(device: mtl.Device, a: std.mem.Allocator, prefix: []const u8) !Pipes {
        const text = try std.mem.concat(a, u8, &.{ ks.inkling2_pattn, @embedFile("ref_pattn.metal") });
        defer a.free(text);
        const body = try tf.frags.source(device, a, text);
        defer a.free(body);
        const src = try std.mem.concat(a, u8, &.{ prefix, body });
        defer a.free(src);
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        errdefer lib.deinit();
        var p: Pipes = undefined;
        p.lib = lib;
        p.attn = try mtl.Pipeline.init(device, lib, "ik_pattn", false);
        errdefer p.attn.deinit();
        p.bias = try mtl.Pipeline.init(device, lib, "ik_pattn_bias", false);
        errdefer p.bias.deinit();
        p.formed = try mtl.Pipeline.init(device, lib, "ik_pattn_formed", false);
        errdefer p.formed.deinit();
        p.bias_mma = try mtl.Pipeline.init(device, lib, "ik_pattn_bias_mma", false);
        return p;
    }

    pub fn deinit(p: *Pipes) void {
        p.attn.deinit();
        p.bias.deinit();
        p.formed.deinit();
        p.bias_mma.deinit();
        p.lib.deinit();
    }
};

/// Bytes of the relative-bias table for `rows` rows: a bf16 for each of a row's 48 heads' 1,024 slots.
pub fn tableBytes(rows: usize) usize {
    return rows * Hq * E * 2;
}

/// One call of ik_pattn (or of the formed-bias kernel): a threadgroup of 64 threads for each 32 rows of each of the 48
/// heads. `rel`: the table at the call's first row, or for the formed kernel the rows' relative state [rows][48][16]
/// bf16 and (`projt`) the projection as f32 columns [1024][16].
pub fn encodeAttn(e: mtl.ComputeEncoder, pipe: mtl.Pipeline, a: Args, io: Io, rel: h.Bind, projt: ?h.Bind) void {
    e.setPipeline(pipe);
    e.setValue(a, 0);
    const bs = [_]h.Bind{ io.q, io.k, io.v, io.sink, rel, io.gate, io.out };
    for (bs, 1..) |b, i| e.setBuffer(b.buf, b.off, i);
    if (projt) |t| e.setBuffer(t.buf, t.off, 8);
    e.dispatchGroups(mtl.Size.of((a.rows + 31) / 32, Hq, 1), mtl.Size.of(64, 1, 1));
}

/// The relative-bias table of `a.rows` rows (ik_pattn_bias; `mma`: the tensor-unit builder, whose bits differ): from the
/// rows' relative state `reln` [rows][48][16] and the weights' projection `proj` [16][1024], into `table`.
pub fn encodeBias(e: mtl.ComputeEncoder, p: *const Pipes, mma: bool, a: Args, reln: h.Bind, proj: h.Bind, table: h.Bind) void {
    e.setPipeline(if (mma) p.bias_mma else p.bias);
    e.setValue(a, 0);
    for ([_]h.Bind{ reln, proj, table }, 1..) |b, i| e.setBuffer(b.buf, b.off, i);
    if (mma) {
        e.dispatchGroups(mtl.Size.of(a.rows, Hq / 16, 1), mtl.Size.of(32, 1, 1));
    } else {
        e.dispatchThreads(mtl.Size.of(E / 4, Hq / 4, a.rows), mtl.Size.of(64, 1, 1));
    }
}

fn down(b: h.Bind, row_bytes: usize, rows: usize) h.Bind {
    return .{ .buf = b.buf, .off = b.off + rows * row_bytes };
}

/// The engine's dispatch for a chunk's attention on one layer: the `a.rows` rows in pieces of `piece` rows (a multiple of
/// 32), each piece's relative-bias table built into `table` (`tableBytes(piece)`) and read by ik_pattn. Every bind is at
/// the chunk's first row. A row's bytes do not depend on `piece` or on `a.p0`.
pub fn encodeChunk(e: mtl.ComputeEncoder, p: *const Pipes, a: Args, io: Io, reln: h.Bind, proj: h.Bind, table: h.Bind, piece: usize) void {
    var at: usize = 0;
    while (at < a.rows) : (at += piece) {
        const sub: Args = .{ .rows = @intCast(@min(piece, a.rows - at)), .p0 = a.p0 + @as(u32, @intCast(at)), .global = a.global };
        encodeBias(e, p, false, sub, down(reln, Hq * REL * 2, at), proj, table);
        const part: Io = .{ .q = down(io.q, QROW * 2, at), .k = io.k, .v = io.v, .sink = down(io.sink, Hq * 2, at), .gate = down(io.gate, QROW * 2, at), .out = down(io.out, QROW * 2, at) };
        encodeAttn(e, p.attn, sub, part, table, null);
    }
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
    fn relnAt(s: *const Seq, p0: usize) h.Bind {
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
    const a: Args = .{ .rows = @intCast(n), .p0 = @intCast(p0), .global = @intFromBool(global) };
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    switch (form) {
        .table => encodeChunk(e, p, a, s.io(p0), s.relnAt(p0), .{ .buf = s.proj }, .{ .buf = table.? }, piece),
        .formed => encodeAttn(e, p.formed, a, s.io(p0), s.relnAt(p0), .{ .buf = s.projt }),
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
        const a: Args = .{ .rows = rows, .p0 = p0, .global = 0 };
        for ([_]bool{ false, true }) |mma| {
            @memset(tv, 0xffff);
            const cb = g.queue.commandBuffer();
            const e = cb.compute(.serial);
            encodeBias(e, pipes, mma, a, s.relnAt(p0), .{ .buf = s.proj }, .{ .buf = table });
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

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    var pipes = try Pipes.init(g.device, std.heap.page_allocator);
    defer pipes.deinit();
    // the text older GPUs run (simdgroup matrices) builds here too; its fragment layout is theirs, so only this GPU's runs
    if (g.device.tensorUnits()) {
        var older = try Pipes.initWith(g.device, std.heap.page_allocator, "#define TF_SIMD_FRAGS 1\n");
        older.deinit();
    }
    try caseRef(g, &pipes, r);
    try caseBias(g, &pipes, r);
    try casePieces(g, &pipes, r);
    try casePartition(g, &pipes, r);
}
