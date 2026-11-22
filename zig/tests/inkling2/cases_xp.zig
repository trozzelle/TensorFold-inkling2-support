//! Inkling 2 kernel checks: the expert chain for a prompt chunk (ik_xp_*, experts_prefill.zig): the counting sort and its
//! windows, each stage against an FP64 reference at the reference's rounding points, the grouped tensor-unit GEMM
//! against the decode port's error, and the chain's invariances: every row alone equals the row in a batch, a prompt cut
//! anywhere gives the same bits, and the neighbours that share a window do not matter. INKLING2_PACK names a model
//! folder whose layer 0 experts the same checks then run on (the one-layer pack, or any full one).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const h = @import("harness.zig");
const xp = tf.inkling2.experts_prefill;
const exl3 = tf.exl3_projection;
const format = exl3.format;

const H = 4608;
const I = 2304;
const TOPK = 8;
const had_scale: f64 = 1.0 / @sqrt(128.0);
const nan_bf16: u16 = 0x7fc0;
const caps: xp.Caps = .{ .gate = 4, .up = 25 };
/// The rows the checks' scratch holds: the largest chunk.
const rows_max = 4096;
const S_max = rows_max * TOPK;
/// The prefill GEMM's relative RMS error against the exact math, in units of the decode kernel's: the f16 rounding of
/// the rotated input is the error both share.
const gemm_ratio: f64 = 1.05;
/// The prefill GEMM's relative RMS error against the exact sums of the f16 rows it multiplies: its f32 accumulation, one
/// chain of 16-deep tensor-unit steps (6e-7 measured; the decode kernel's 32 short chains give 1.2e-7).
const accumulation_bound: f64 = 2e-6;

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    const gpa = std.heap.page_allocator;
    if (!xp.available(g.device)) {
        std.debug.print("SKIP xp: no tensor units on this GPU\n", .{});
        return;
    }
    var pipes = try xp.Pipes.init(g.device, gpa);
    defer pipes.deinit();
    const scratch = try g.buf(xp.xpBytes(rows_max));
    defer scratch.deinit();
    const s = xp.Xp.init(scratch, 0, rows_max);
    const lut = try mul1Table(gpa);
    defer gpa.free(lut);
    var io = try Io.init(g);
    defer io.deinit();
    var dec = try Decode.init(g);
    defer dec.deinit();

    try caseGroup(g, gpa, &pipes, s, r);

    // synthetic experts: a layer of 24 table entries cycling six experts at K3, K2, K3, K2, K3, K3
    var experts: [6]Expert = undefined;
    const bits = [_]u32{ 3, 2, 3, 2, 3, 3 };
    var made: usize = 0;
    defer for (experts[0..made]) |e| e.deinit();
    for (&experts, bits) |*e, b| {
        e.* = try Expert.init(g, r, b);
        made += 1;
    }
    var lay = try Lay.synthetic(g, gpa, &experts, 24);
    defer lay.deinit(gpa);
    try suite(.{ .g = g, .gpa = gpa, .p = &pipes, .s = s, .lay = &lay, .lut = lut, .io = &io, .dec = &dec, .r = r, .hot = 1.2 });

    // a real layer's experts: INKLING2_XP_LAYER (0) of the model folder INKLING2_PACK, else the one-layer pack beside the fork
    const dir = if (std.c.getenv("INKLING2_PACK")) |v| std.mem.span(v) else "../inkling2/work/tf-pack-l0";
    const layer: u32 = if (std.c.getenv("INKLING2_XP_LAYER")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 0;
    var real = Lay.real(g, gpa, dir, layer) catch |err| {
        std.debug.print("SKIP xp on a real layer: {s} ({s}, layer {d})\n", .{ @errorName(err), dir, layer });
        return;
    };
    defer real.deinit(gpa);
    std.debug.print("xp on layer {d} of {s}:\n", .{ layer, dir });
    try suite(.{ .g = g, .gpa = gpa, .p = &pipes, .s = s, .lay = &real, .lut = lut, .io = &io, .dec = &dec, .r = r, .hot = 2.2 });
}

/// Every stage and invariance case on one layer.
fn suite(t: Ctx) !void {
    try casePrepare(t);
    try caseGemm(t, .w13);
    try caseGemm(t, .w2);
    try caseMid(t);
    try caseFinish(t);
    try caseChain(t);
    try caseRows(t);
    try caseSameRoute(t);
    try caseWindows(t);
    try casePartition(t);
}

// ---- views and sessions ----

/// `n` elements of a scratch view.
fn view(comptime T: type, b: xp.B, n: usize) []T {
    const bytes = b.buf.slice(u8, b.off + n * @sizeOf(T))[b.off..];
    return @alignCast(std.mem.bytesAsSlice(T, bytes));
}

fn poison(b: mtl.Buffer) void {
    @memset(b.slice(u8, b.length()), 0xff);
}

/// One command buffer of encoded work, waited, its failure reported.
const Session = struct {
    cb: mtl.CommandBuffer,
    e: mtl.ComputeEncoder,

    fn open(g: *const h.Gpu, uses: []const mtl.Buffer) Session {
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (uses) |b| e.useResource(b, mtl.ResourceUsage.read);
        return .{ .cb = cb, .e = e };
    }

    fn close(s: Session) !void {
        s.e.end();
        s.cb.commit();
        s.cb.wait();
        if (s.cb.failure()) |msg| {
            std.log.err("ik_xp: {s}", .{msg});
            return error.GpuFailed;
        }
    }
};

/// The chain's inputs and output buffers for up to `rows_max` rows.
const Io = struct {
    x: mtl.Buffer,
    ids: mtl.Buffer,
    ew: mtl.Buffer,
    out: mtl.Buffer,

    fn init(g: *const h.Gpu) !Io {
        return .{
            .x = try g.buf(rows_max * H * 2),
            .ids = try g.buf(S_max * 4),
            .ew = try g.buf(S_max * 4),
            .out = try g.buf(rows_max * H * 2),
        };
    }

    fn deinit(io: Io) void {
        for ([_]mtl.Buffer{ io.x, io.ids, io.ew, io.out }) |b| b.deinit();
    }

    /// Random rows: x in [-0.1, 0.1] (gu then spans the SwiGLU's caps), router weights in [0.05, 1.05].
    fn fill(io: Io, r: std.Random, rows: usize) void {
        _ = h.fill16(io.x, rows * H, r, 0.1);
        for (io.ew.slice(f32, rows * TOPK)) |*w| w.* = 0.05 + r.float(f32);
    }

    fn picks(io: Io, row0: usize) xp.Picks {
        return .{ .ids = .{ .buf = io.ids, .off = row0 * TOPK * 4 }, .weights = .{ .buf = io.ew, .off = row0 * TOPK * 4 } };
    }

    fn xAt(io: Io, row0: usize) xp.B {
        return .{ .buf = io.x, .off = row0 * H * 2 };
    }

    fn outAt(io: Io, row0: usize) xp.B {
        return .{ .buf = io.out, .off = row0 * H * 2 };
    }
};

// ---- picks ----

/// `n` rows of 8 distinct expert ids each, drawn from a skewed popularity over `experts`: the expert at rank k of the
/// permutation `perm` has weight 1 / (k + 1)^skew.
fn skewedPicks(r: std.Random, gpa: std.mem.Allocator, ids: []u32, n: usize, experts: usize, skew: f64, perm: []const u32) !void {
    std.debug.assert(experts >= TOPK);
    const cdf = try gpa.alloc(f64, experts);
    defer gpa.free(cdf);
    var total: f64 = 0;
    for (cdf, 0..) |*c, k| {
        total += 1.0 / std.math.pow(f64, @as(f64, @floatFromInt(k + 1)), skew);
        c.* = total;
    }
    for (0..n) |row| {
        var k: usize = 0;
        while (k < TOPK) {
            const u = r.float(f64) * total;
            var lo: usize = 0;
            var hi: usize = experts - 1;
            while (lo < hi) {
                const mid = (lo + hi) / 2;
                if (cdf[mid] < u) lo = mid + 1 else hi = mid;
            }
            const pick = perm[lo];
            if (std.mem.indexOfScalar(u32, ids[row * TOPK ..][0..k], pick) != null) continue;
            ids[row * TOPK + k] = pick;
            k += 1;
        }
    }
}

fn shuffled(r: std.Random, gpa: std.mem.Allocator, count: usize) ![]u32 {
    const perm = try gpa.alloc(u32, count);
    for (perm, 0..) |*p, i| p.* = @intCast(i);
    r.shuffle(u32, perm);
    return perm;
}

// ---- host references ----

/// The f16 weight of each 16-bit state under the mul1 codebook: 1024 + the four bytes of state * 0x83DCD12D summed (the
/// half 0x6400 + sum), then fma(h, half 0x1EEE, half 0xC931) in f16. The product and sum are exact in f64 (an integer
/// times 2^-18 below 2^23 in magnitude), so one rounding to f16 is the f16 fma.
fn mul1Table(gpa: std.mem.Allocator) ![]f32 {
    const lut = try gpa.alloc(f32, 65536);
    const scale: f16 = @bitCast(@as(u16, 0x1EEE));
    const bias: f16 = @bitCast(@as(u16, 0xC931));
    for (lut, 0..) |*v, state| {
        const x: u32 = @as(u32, @intCast(state)) *% 0x83DCD12D;
        const sum: u16 = @intCast((x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24));
        const half: f16 = @bitCast(@as(u16, 0x6400) + sum);
        const exact = @as(f64, half) * @as(f64, scale) + @as(f64, bias);
        v.* = @floatCast(@as(f16, @floatCast(exact)));
    }
    return lut;
}

/// One 16x16 tile's weights d[input row * 16 + output column] from the tile's u32 words: weight p's codeword is the
/// 16-bit window ending at bit (p + 1) * bits of the tile's circular MSB-first stream, and the two mma fragments' lane
/// order places it (stream index 8 * lane + j: row 2 * (lane % 4) + {0, 1, 8, 9}[j % 4], column lane / 4 + 8 * (j / 4)).
fn decodeTile(words: []const u32, bits: u32, lut: []const f32, d: *[256]f32) void {
    const ring: u32 = 256 * bits;
    for (0..256) |p| {
        const end: u32 = @intCast((p + 1) * bits);
        const start = (end + ring - 16) % ring;
        const wi = start / 32;
        const off: u6 = @intCast(start % 32);
        const hi: u64 = words[wi];
        const lo: u64 = words[(wi + 1) % words.len];
        const cw: u16 = @truncate(((hi << 32) | lo) >> (48 - off));
        const lane = p / 8;
        const j = p % 8;
        const row = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1);
        const col = lane / 4 + 8 * (j >> 2);
        d[row * 16 + col] = lut[cw];
    }
}

/// The natural-order 128-point Hadamard of each 128-block of `v`, scaled by 1/sqrt(128).
fn had128(v: []f64) void {
    var b: usize = 0;
    while (b < v.len) : (b += 128) {
        var len: usize = 1;
        while (len < 128) : (len *= 2) {
            var i: usize = 0;
            while (i < 128) : (i += 2 * len) {
                for (i..i + len) |j| {
                    const lo = v[b + j];
                    const hi = v[b + j + len];
                    v[b + j] = lo + hi;
                    v[b + j + len] = lo - hi;
                }
            }
        }
        for (v[b..][0..128]) |*e| e.* *= had_scale;
    }
}

/// out[i] = in[i] . W for the rows of one expert's projection (k inputs, n outputs; W the trellis' decoded weights,
/// one tile at a time), summed in f64: the exact value the kernels' f32 sums approximate.
fn gemmRef(lut: []const f32, words: []const u32, bits: u32, k: usize, n: usize, ins: []const []const f64, outs: []const []f64) void {
    for (outs) |o| @memset(o, 0);
    const per_tile = 8 * bits;
    var tile: [256]f32 = undefined;
    for (0..k / 16) |ti| for (0..n / 16) |tj| {
        decodeTile(words[(ti * (n / 16) + tj) * per_tile ..][0..per_tile], bits, lut, &tile);
        for (ins, outs) |in, out| {
            const v = in[ti * 16 ..][0..16];
            const a = out[tj * 16 ..][0..16];
            for (0..16) |rr| {
                for (0..16) |cc| a[cc] += v[rr] * tile[rr * 16 + cc];
            }
        }
    };
}

fn rms(v: []const f64) f64 {
    var s: f64 = 0;
    for (v) |x| s += x * x;
    return @sqrt(s / @as(f64, @floatFromInt(@max(v.len, 1))));
}

fn f16v(u: u16) f64 {
    return @as(f64, @as(f16, @bitCast(u)));
}

/// An f16 output within one f16 step (at least one subnormal step) of the reference, plus 5e-7 of the row's magnitude:
/// the f32 butterfly's rounding against an exact sum, which cancels toward zero in some outputs and keeps its absolute error.
fn near16h(c: *h.Check, got: u16, want: f64, mag: f64) void {
    c.n += 1;
    const g = f16v(got);
    const step: f64 = if (want == 0) 0x1p-24 else @max(@exp2(@floor(@log2(@abs(want))) - 10), 0x1p-24);
    const slack = step + 5e-7 * mag;
    c.worst = @max(c.worst, @abs(g - want) / slack);
    if (!(@abs(g - want) <= slack)) {
        c.bad += 1;
        if (c.first == null) c.first = .{ g, want };
    }
}

/// A bf16 output within one bf16 step of the reference plus `rel` of max(|want|, mag): the chain's f16 input rounding
/// and the sums' order leave this much against the exact FP64 value. `worst` is the error beyond the step, relative
/// to max(|want|, mag).
fn near16(c: *h.Check, got: u16, want: f64, rel: f64, mag: f64) void {
    c.n += 1;
    const g = h.fb(got);
    const step: f64 = if (want == 0) 0 else @exp2(@floor(@log2(@abs(want))) - 7);
    const scale = @max(@abs(want), mag);
    c.worst = @max(c.worst, @max(@abs(g - want) - step, 0) / scale);
    if (!(@abs(g - want) <= step + rel * scale)) {
        c.bad += 1;
        if (c.first == null) c.first = .{ g, want };
    }
}

// ---- experts ----

/// One expert at `bits` of both projections, with random trellis words and scales in [0.5, 1.5]: w13's trellis is
/// [288][288][16 bits] int16 and w2's [144][288][16 bits], as the checkpoint stores them.
const Expert = struct {
    bits: u32,
    t13: mtl.Buffer,
    s13: mtl.Buffer,
    v13: mtl.Buffer,
    t2: mtl.Buffer,
    s2: mtl.Buffer,
    v2: mtl.Buffer,

    fn scales(b: mtl.Buffer, count: usize, r: std.Random) void {
        for (b.slice(f16, count)) |*s| s.* = @floatCast(0.5 + r.float(f32));
    }

    fn init(g: *const h.Gpu, r: std.Random, bits: u32) !Expert {
        const tile = 32 * bits; // bytes
        const e: Expert = .{
            .bits = bits,
            .t13 = try g.buf((H / 16) * (2 * I / 16) * tile),
            .s13 = try g.buf(H * 2),
            .v13 = try g.buf(2 * I * 2),
            .t2 = try g.buf((I / 16) * (H / 16) * tile),
            .s2 = try g.buf(I * 2),
            .v2 = try g.buf(H * 2),
        };
        r.bytes(e.t13.slice(u8, e.t13.length()));
        r.bytes(e.t2.slice(u8, e.t2.length()));
        scales(e.s13, H, r);
        scales(e.v13, 2 * I, r);
        scales(e.s2, I, r);
        scales(e.v2, H, r);
        return e;
    }

    fn deinit(e: Expert) void {
        for ([_]mtl.Buffer{ e.t13, e.s13, e.v13, e.t2, e.s2, e.v2 }) |b| b.deinit();
    }

    fn buffers(e: *const Expert) [6]mtl.Buffer {
        return .{ e.t13, e.s13, e.v13, e.t2, e.s2, e.v2 };
    }
};

/// What the references read of one table entry's expert: its rate and the host views of its tensors.
const Ref = struct {
    bits: u32,
    t13: []const u32,
    s13: []const f16,
    v13: []const f16,
    t2: []const u32,
    s2: []const f16,
    v2: []const f16,
};

fn words32(b: mtl.Buffer) []const u32 {
    return b.slice(u32, b.length() / 4);
}

/// A layer's descriptor table (w13's entries, then w2's), the buffers its descriptors point at, and the references' views.
const Lay = struct {
    count: usize,
    table: mtl.Buffer,
    /// Every buffer the kernels reach through the table, the table included.
    uses: []mtl.Buffer,
    refs: []Ref,
    layer: xp.Layer,
    /// Real layers own a checkpoint; synthetic ones borrow their experts.
    checkpoint: ?*tf.inkling2.weights.Checkpoint = null,

    /// `count` entries cycling `experts`.
    fn synthetic(g: *const h.Gpu, gpa: std.mem.Allocator, experts: []const Expert, count: usize) !Lay {
        const table = try g.buf(2 * count * @sizeOf(exl3.ExpertDescriptor));
        errdefer table.deinit();
        const desc = table.slice(exl3.ExpertDescriptor, 2 * count);
        const refs = try gpa.alloc(Ref, count);
        errdefer gpa.free(refs);
        const uses = try gpa.alloc(mtl.Buffer, 1 + 6 * experts.len);
        errdefer gpa.free(uses);
        uses[0] = table;
        for (experts, 0..) |*ex, i| @memcpy(uses[1 + 6 * i ..][0..6], &ex.buffers());
        for (0..count) |e| {
            const ex = &experts[e % experts.len];
            const width: format.Width = .{ .halves = @intCast(2 * ex.bits) };
            desc[e] = try exl3.descriptor(.{ .trellis = .{ .buffer = ex.t13 }, .suh = .{ .buffer = ex.s13 }, .svh = .{ .buffer = ex.v13 }, .k = H, .n = 2 * I, .width = width, .codebook = .mul1 });
            desc[count + e] = try exl3.descriptor(.{ .trellis = .{ .buffer = ex.t2 }, .suh = .{ .buffer = ex.s2 }, .svh = .{ .buffer = ex.v2 }, .k = I, .n = H, .width = width, .codebook = .mul1 });
            refs[e] = .{ .bits = ex.bits, .t13 = words32(ex.t13), .s13 = ex.s13.slice(f16, H), .v13 = ex.v13.slice(f16, 2 * I), .t2 = words32(ex.t2), .s2 = ex.s2.slice(f16, I), .v2 = ex.v2.slice(f16, H) };
        }
        return .{ .count = count, .table = table, .uses = uses, .refs = refs, .layer = layerOf(table, count) };
    }

    /// Layer `layer`'s 384 experts of the model folder `dir`, as the checkpoint loads them; the references read the
    /// loaded files' memory.
    fn real(g: *const h.Gpu, gpa: std.mem.Allocator, dir: []const u8, layer: u32) !Lay {
        const E = 384;
        const weights = tf.inkling2.weights;
        var threaded: std.Io.Threaded = .init_single_threaded;
        var pack = try tf.inkling2.expert_pack.Pack.read(gpa, threaded.io(), dir, layer + 1, E);
        defer pack.deinit();
        const ck = try gpa.create(weights.Checkpoint);
        ck.* = weights.Checkpoint.init(gpa);
        errdefer {
            ck.deinit();
            gpa.destroy(ck);
        }
        var widths: [5]u32 = @splat(0);
        const table = try weights.expertTable(gpa, g.device, ck, dir, pack.layers[layer], layer, H, I, E, &widths);
        errdefer table.deinit();
        const desc = table.slice(exl3.ExpertDescriptor, 2 * E);
        if (!weights.xdOk(desc, E, H, I)) return error.LayerNotXd;
        const refs = try gpa.alloc(Ref, E);
        errdefer gpa.free(refs);
        const uses = try gpa.alloc(mtl.Buffer, 1 + ck.shards.items.len);
        errdefer gpa.free(uses);
        uses[0] = table;
        for (ck.shards.items, uses[1..]) |sh, *u| u.* = sh.buffer;
        for (refs, 0..) |*ref, e| {
            const a = desc[e];
            const b = desc[E + e];
            const bits = a.halves / 2;
            ref.* = .{
                .bits = bits,
                .t13 = try hostView(u32, ck, a.trellis, (H / 16) * (2 * I / 16) * 8 * bits),
                .s13 = try hostView(f16, ck, a.suh, H),
                .v13 = try hostView(f16, ck, a.svh, 2 * I),
                .t2 = try hostView(u32, ck, b.trellis, (I / 16) * (H / 16) * 8 * bits),
                .s2 = try hostView(f16, ck, b.suh, I),
                .v2 = try hostView(f16, ck, b.svh, H),
            };
        }
        return .{ .count = E, .table = table, .uses = uses, .refs = refs, .layer = layerOf(table, E), .checkpoint = ck };
    }

    fn deinit(l: *Lay, gpa: std.mem.Allocator) void {
        gpa.free(l.refs);
        gpa.free(l.uses);
        l.table.deinit();
        if (l.checkpoint) |ck| {
            ck.deinit();
            gpa.destroy(ck);
        }
    }
};

/// `count` elements of type T at GPU address `addr` inside one of the checkpoint's files.
fn hostView(comptime T: type, ck: *const tf.inkling2.weights.Checkpoint, addr: u64, count: usize) ![]const T {
    const bytes = count * @sizeOf(T);
    for (ck.shards.items) |sh| {
        const base = sh.buffer.gpuAddress();
        const len = sh.buffer.length();
        if (addr >= base and addr + bytes <= base + len) {
            const at: usize = @intCast(addr - base);
            return @alignCast(std.mem.bytesAsSlice(T, sh.buffer.slice(u8, len)[at..][0..bytes]));
        }
    }
    return error.NotInCheckpoint;
}

fn layerOf(table: mtl.Buffer, count: usize) xp.Layer {
    return .{
        .w13 = .{ .descriptors = .{ .buffer = table }, .count = count, .k = H, .n = 2 * I },
        .w2 = .{ .descriptors = .{ .buffer = table, .offset = count * @sizeOf(exl3.ExpertDescriptor) }, .count = count, .k = I, .n = H },
    };
}

/// Everything a case needs.
const Ctx = struct {
    g: *const h.Gpu,
    gpa: std.mem.Allocator,
    p: *const xp.Pipes,
    s: xp.Xp,
    lay: *const Lay,
    lut: []const f32,
    io: *const Io,
    dec: *const Decode,
    r: std.Random,
    /// The popularity skew of the GEMM cases' picks: steep on a layer of many experts, so a few dozen of them carry
    /// the slots the references decode.
    hot: f64,

    /// The checks' own picks for rows [0, n): skewed over the layer's table, hot experts shuffled.
    fn picks(t: Ctx, n: usize, skew: f64) !void {
        const perm = try shuffled(t.r, t.gpa, t.lay.count);
        defer t.gpa.free(perm);
        try skewedPicks(t.r, t.gpa, t.io.ids.slice(u32, n * TOPK), n, t.lay.count, skew, perm);
    }

    fn open(t: Ctx) Session {
        return Session.open(t.g, t.lay.uses);
    }

    fn encode(t: Ctx, ses: Session, x: xp.B, route: xp.Picks, out: xp.B, n: usize) void {
        xp.encode(ses.e, t.p, t.lay.layer, x, route, out, t.s, n, caps);
    }
};

// ---- the decode port, as the reference's error ----

/// The decode chain's pipelines and buffers for up to 16 rows.
const Decode = struct {
    pair: mtl.Pipeline,
    mid: mtl.Pipeline,
    down: mtl.Pipeline,
    finish: mtl.Pipeline,
    planes: mtl.Buffer,
    act: mtl.Buffer,
    sums: mtl.Buffer,
    moe: mtl.Buffer,

    const slots = 16 * TOPK;

    fn init(g: *const h.Gpu) !Decode {
        var d: Decode = undefined;
        d.pair = try mtl.Pipeline.init(g.device, g.lib, "ik_xd_pair", false);
        d.mid = try mtl.Pipeline.init(g.device, g.lib, "ik_xd_mid", false);
        d.down = try mtl.Pipeline.init(g.device, g.lib, "ik_xd_down", false);
        d.finish = try mtl.Pipeline.init(g.device, g.lib, "ik_xd_finish", false);
        d.planes = try g.buf(slots * 2 * H * 4);
        d.act = try g.buf(slots * I * 2);
        d.sums = try g.buf(slots * H * 4);
        d.moe = try g.buf(16 * H * 2);
        return d;
    }

    fn deinit(d: Decode) void {
        for ([_]mtl.Pipeline{ d.pair, d.mid, d.down, d.finish }) |p| p.deinit();
        for ([_]mtl.Buffer{ d.planes, d.act, d.sums, d.moe }) |b| b.deinit();
    }

    const Stages = struct { pair: bool = false, mid: bool = false, down: bool = false, finish: bool = false };

    /// The chosen stages of the decode chain over `rows` (at most 16) rows of x, ids and ew (from byte offsets of their
    /// own), with `lay`'s table.
    fn run(d: Decode, t: Ctx, x: xp.B, ids: xp.B, ew: xp.B, rows: usize, st: Stages) !void {
        std.debug.assert(rows <= 16);
        const S = rows * TOPK;
        const ses = t.open();
        const e = ses.e;
        const a = h.Args{ .rows = @intCast(S), .cap = caps.gate, .thr = caps.up };
        const w2 = t.lay.count * @sizeOf(exl3.ExpertDescriptor);
        if (st.pair) {
            e.setPipeline(d.pair);
            e.setValue(a, 0);
            e.setBuffer(x.buf, x.off, 1);
            e.setBuffer(ids.buf, ids.off, 2);
            e.setBuffer(t.lay.table, 0, 3);
            e.setBuffer(d.planes, 0, 4);
            e.dispatchGroups(mtl.Size.of(144, S, 2), mtl.Size.of(128, 1, 1));
        }
        if (st.mid) {
            e.setPipeline(d.mid);
            e.setValue(a, 0);
            e.setBuffer(d.planes, 0, 1);
            e.setBuffer(ids.buf, ids.off, 2);
            e.setBuffer(t.lay.table, 0, 3);
            e.setBuffer(d.act, 0, 4);
            e.dispatchGroups(mtl.Size.of(36, S, 1), mtl.Size.of(32, 1, 1));
        }
        if (st.down) {
            e.setPipeline(d.down);
            e.setValue(a, 0);
            e.setBuffer(d.act, 0, 1);
            e.setBuffer(ids.buf, ids.off, 2);
            e.setBuffer(t.lay.table, w2, 3);
            e.setBuffer(d.sums, 0, 4);
            e.dispatchGroups(mtl.Size.of(144, S, 1), mtl.Size.of(128, 1, 1));
        }
        if (st.finish) {
            e.setPipeline(d.finish);
            e.setValue(a, 0);
            e.setBuffer(d.sums, 0, 1);
            e.setBuffer(ids.buf, ids.off, 2);
            e.setBuffer(t.lay.table, w2, 3);
            e.setBuffer(ew.buf, ew.off, 4);
            e.setBuffer(d.moe, 0, 5);
            e.dispatchGroups(mtl.Size.of(36, rows, 1), mtl.Size.of(256, 1, 1));
        }
        try ses.close();
    }
};

// ---- the counting sort ----

/// A descriptor table that carries only the rates (the grouping kernels read nothing else): expert e's w13 and w2 at
/// the given halves (4 for K2, 6 for K3).
fn rateTable(g: *const h.Gpu, halves13: []const u32, halves2: []const u32) !mtl.Buffer {
    const count = halves13.len;
    const buf = try g.buf(2 * count * @sizeOf(exl3.ExpertDescriptor));
    const desc = buf.slice(exl3.ExpertDescriptor, 2 * count);
    @memset(std.mem.sliceAsBytes(desc), 0);
    for (halves13, 0..) |v, e| desc[e].halves = v;
    for (halves2, 0..) |v, e| desc[count + e].halves = v;
    return buf;
}

const Win = extern struct { start: u32, nlive: u32, eid: u32, pad: u32 };

/// Random skewed picks over layers of several sizes and rate mixes, at several row counts: the sort is stable by
/// expert, every slot appears once, the windows are single-expert and at most 32 rows, and the kernels write nothing
/// past the slots.
fn caseGroup(g: *const h.Gpu, gpa: std.mem.Allocator, p: *const xp.Pipes, s: xp.Xp, r: std.Random) !void {
    var c: h.Check = .{ .name = "xp grouping: stable sort by expert, windows of one expert" };
    const ids_buf = try g.buf(S_max * 4);
    defer ids_buf.deinit();
    const sizes = [_]usize{ 1, 2, 7, 24, 384 };
    for (sizes) |count| {
        const h13 = try gpa.alloc(u32, count);
        defer gpa.free(h13);
        const h2 = try gpa.alloc(u32, count);
        defer gpa.free(h2);
        // a mixed layer, a K3 layer, and a layer whose two projections' rates differ
        const mode = count % 3;
        for (h13, h2, 0..) |*a, *b, e| {
            a.* = if (mode == 1) 6 else if (r.boolean()) 4 else 6;
            b.* = if (mode == 1) 6 else if (mode == 2) (if (e % 2 == 0) 4 else 6) else a.*;
        }
        const table = try rateTable(g, h13, h2);
        defer table.deinit();
        const layer = layerOf(table, count);
        const perm = try shuffled(r, gpa, count);
        defer gpa.free(perm);
        for ([_]usize{ 1, 3, 31, 32, 33, 100, 511, 4096 }) |n| {
            const S = n * TOPK;
            const ids = ids_buf.slice(u32, S);
            if (count < TOPK) {
                // fewer experts than picks: repeats are allowed
                for (ids) |*v| v.* = @intCast(r.uintLessThan(usize, count));
            } else try skewedPicks(r, gpa, ids, n, count, if (n % 2 == 0) 1.3 else 0.4, perm);
            // one pick outside the table: it is not placed, and nothing else moves
            if (n == 33) ids[5] = @intCast(count);
            poison(s.counts.buf);
            const ses = Session.open(g, &.{});
            xp.group(ses.e, p, layer, .{ .buf = ids_buf }, s, n);
            try ses.close();
            try checkGroup(&c, gpa, s, ids, h13, h2, count);
        }
    }
    try c.done();
}

fn checkGroup(c: *h.Check, gpa: std.mem.Allocator, s: xp.Xp, ids: []const u32, h13: []const u32, h2: []const u32, count: usize) !void {
    const S = ids.len;
    const cap = (S_max + xp.window - 1) / xp.window + xp.max_experts;
    // the host's sort
    const counts = try gpa.alloc(u32, count);
    defer gpa.free(counts);
    @memset(counts, 0);
    for (ids) |e| if (e < count) {
        counts[e] += 1;
    };
    const offs = try gpa.alloc(u32, count + 1);
    defer gpa.free(offs);
    offs[0] = 0;
    for (counts, 0..) |n, e| offs[e + 1] = offs[e] + n;
    const placed = offs[count];
    const order = try gpa.alloc(u32, placed);
    defer gpa.free(order);
    const inverse = try gpa.alloc(u32, S);
    defer gpa.free(inverse);
    @memset(inverse, 0xffffffff);
    const fill = try gpa.alloc(u32, count);
    defer gpa.free(fill);
    @memset(fill, 0);
    for (ids, 0..) |e, slot| if (e < count) {
        const pos = offs[e] + fill[e];
        fill[e] += 1;
        order[pos] = @intCast(slot);
        inverse[slot] = pos;
    };
    // per-expert counts and rate classes
    c.same(std.mem.sliceAsBytes(view(u32, s.counts, count)), std.mem.sliceAsBytes(counts));
    var classes: [xp.max_experts]u32 = undefined;
    for (0..count) |e| classes[e] = @as(u32, @intFromBool(h13[e] == 6)) | (@as(u32, @intFromBool(h2[e] == 6)) << 1);
    c.same(std.mem.sliceAsBytes(view(u32, s.classes, count)), std.mem.sliceAsBytes(classes[0..count]));
    // the sorted order, the experts at each position (the unplaced ones after them are marked) and the inverse
    c.same(std.mem.sliceAsBytes(view(u32, s.order, placed)), std.mem.sliceAsBytes(order));
    const got_inv = view(u32, s.inverse, S);
    const got_seid = view(u32, s.seid, S);
    for (ids, 0..) |e, slot| {
        if (e >= count) continue;
        c.n += 1;
        if (got_inv[slot] != inverse[slot]) c.bad += 1;
    }
    for (0..placed) |pos| {
        c.n += 1;
        if (got_seid[pos] != ids[order[pos]]) c.bad += 1;
    }
    for (placed..S) |pos| {
        c.n += 1;
        if (got_seid[pos] != 0xffffffff) c.bad += 1;
    }
    // the windows of each list: expert after expert, one expert each, at most 32 rows, covering the expert's slots
    var lens = [4]usize{ 0, 0, 0, 0 };
    var covered = [4]u32{ 0, 0, 0, 0 };
    const wins = view(Win, s.wins, 4 * cap);
    const dargs = view(u32, s.dargs, 16);
    for (0..count) |e| {
        if (counts[e] == 0) continue;
        const lists = [2]usize{ if (h13[e] == 6) 1 else 0, 2 + @as(usize, if (h2[e] == 6) 1 else 0) };
        for (lists) |l| {
            var off: u32 = 0;
            while (off < counts[e]) : (off += xp.window) {
                const w = wins[l * cap + lens[l]];
                lens[l] += 1;
                c.n += 1;
                if (w.eid != e or w.start != offs[e] + off or w.nlive != @min(@as(u32, xp.window), counts[e] - off)) c.bad += 1;
                covered[l] += w.nlive;
            }
        }
    }
    for (0..4) |l| {
        c.n += 1;
        if (dargs[l * 4] != H / 128 or dargs[l * 4 + 1] != lens[l] or dargs[l * 4 + 2] != 1) c.bad += 1;
    }
    c.n += 1;
    if (covered[0] + covered[1] != placed or covered[2] + covered[3] != placed) c.bad += 1;
}

// ---- scratch poisoning ----

/// The scratch's regions for `rows` rows set to 0xff bytes: every value read before the chain writes it is NaN or an
/// out-of-range index, so a stage that reads what it should not shows.
fn poisonUpTo(s: xp.Xp, rows: usize) void {
    const slots = rows * TOPK;
    const regions = [_]struct { b: xp.B, bytes: usize }{
        .{ .b = s.counts, .bytes = xp.max_experts * 4 },
        .{ .b = s.classes, .bytes = xp.max_experts * 4 },
        .{ .b = s.order, .bytes = slots * 4 },
        .{ .b = s.seid, .bytes = slots * 4 },
        .{ .b = s.inverse, .bytes = slots * 4 },
        .{ .b = s.wins, .bytes = 4 * ((slots + xp.window - 1) / xp.window + xp.max_experts) * @sizeOf(Win) },
        .{ .b = s.dargs, .bytes = 64 },
        .{ .b = s.prep, .bytes = slots * H * 2 },
        .{ .b = s.plane, .bytes = slots * H * 4 },
    };
    for (regions) |r| @memset(r.b.buf.slice(u8, r.b.off + r.bytes)[r.b.off..], 0xff);
}

// ---- stage cases ----

const psc: f32 = 0.08838834764831845;

/// ik_xd_h128 on the host: the 128-point Hadamard of one block in the kernels' butterfly order (the five lane stages,
/// then the in-lane ones), in f32, unscaled.
fn h128f32(v: *[128]f32) void {
    var cur = v.*;
    var bit: usize = 1;
    while (bit <= 16) : (bit <<= 1) {
        var next: [128]f32 = undefined;
        for (0..32) |lane| for (0..4) |q| {
            const me = cur[lane + 32 * q];
            const other = cur[(lane ^ bit) + 32 * q];
            next[lane + 32 * q] = if ((lane & bit) == 0) me + other else other - me;
        };
        cur = next;
    }
    for (0..32) |lane| {
        const s0 = cur[lane] + cur[lane + 32];
        const s1 = cur[lane] - cur[lane + 32];
        const s2 = cur[lane + 64] + cur[lane + 96];
        const s3 = cur[lane + 64] - cur[lane + 96];
        v[lane] = s0 + s2;
        v[lane + 32] = s1 + s3;
        v[lane + 64] = s0 - s2;
        v[lane + 96] = s1 - s3;
    }
}

/// The kernels' rotated f16 input of a row: each 128-block of `prods` through the butterfly, times the scale, rounded to f16.
fn rotateF16(out: []u16, prods: []const f32) void {
    var b: [128]f32 = undefined;
    var at: usize = 0;
    while (at < prods.len) : (at += 128) {
        @memcpy(&b, prods[at..][0..128]);
        h128f32(&b);
        for (out[at..][0..128], b) |*o, v| {
            const q: f16 = @floatCast(v * psc);
            o.* = @bitCast(q);
        }
    }
}

fn bf16f32(u: u16) f32 {
    return @bitCast(@as(u32, u) << 16);
}

/// w13's rotated f16 input of every sorted slot: x * suh, the 128-block Hadamard, the scale, one f16 rounding. Equal
/// bit for bit to the decode kernels' own rotation (replayed on the host in f32), and within one f16 step of FP64.
fn casePrepare(t: Ctx) !void {
    const n = 40;
    const S = n * TOPK;
    t.io.fill(t.r, n);
    try t.picks(n, 1.0);
    poisonUpTo(t.s, n);
    const ses = t.open();
    xp.group(ses.e, t.p, t.lay.layer, t.io.picks(0).ids, t.s, n);
    xp.prepare(ses.e, t.p, t.lay.layer, t.io.xAt(0), t.s, n);
    try ses.close();
    var ce: h.Check = .{ .name = "xp prepare equals the decode kernels' f32 rotation, bit for bit" };
    var cd: h.Check = .{ .name = "xp prepare against FP64: x suh, Hadamard, f16" };
    const order = view(u32, t.s.order, S);
    const ids = t.io.ids.slice(u32, S);
    const x = t.io.x.slice(u16, n * H);
    const prep = view(u16, t.s.prep, S * H);
    var prods: [H]f32 = undefined;
    var row: [H]u16 = undefined;
    var v: [H]f64 = undefined;
    for (0..S) |pos| {
        const slot = order[pos];
        const ref = t.lay.refs[ids[slot]];
        for (&prods, 0..) |*p, i| p.* = bf16f32(x[slot / TOPK * H + i]) * @as(f32, ref.s13[i]);
        rotateF16(&row, &prods);
        ce.same(std.mem.sliceAsBytes(prep[pos * H ..][0..H]), std.mem.sliceAsBytes(&row));
        for (&v, prods) |*e, p| e.* = p; // bf16 times f16 is exact in f32
        had128(&v);
        const mag = rms(&v);
        for (0..H) |i| near16h(&cd, prep[pos * H + i], v[i], mag);
    }
    try ce.done();
    try cd.done();
}

/// Relative RMS of `got` against `want`.
fn relRms(got: []const f64, want: []const f64) f64 {
    var dq: f64 = 0;
    var sq: f64 = 0;
    for (got, want) |a, b| {
        dq += (a - b) * (a - b);
        sq += b * b;
    }
    return @sqrt(dq / @max(sq, 1e-300));
}

/// One projection's grouped GEMM on slots across every window of a skewed batch, against the decode port's error on
/// the same slots. Two references: the exact value of the operation (the unrounded rotated input, FP64 sums), where the
/// f16 rounding of the rotated input is the error both kernels share and the prefill kernel's f32 sums land within 1.05 of
/// the decode kernel's relative RMS error; and the exact sums of the f16 rows the kernels multiply, which isolate the
/// accumulation: one chain of 16-deep tensor-unit steps against the decode kernel's 32 short f32 chains. w13 reads the
/// kernel's own prepared rows, w2 rotated random acts (the host's f32 replay of the rotation the decode kernel does).
fn caseGemm(t: Ctx, which: xp.Proj) !void {
    const n = 96;
    const S = n * TOPK;
    const w13 = which == .w13;
    const K: usize = if (w13) H else I;
    const N: usize = H;
    t.io.fill(t.r, n);
    try t.picks(n, t.hot);
    const ids = t.io.ids.slice(u32, S);
    poisonUpTo(t.s, n);
    const acts = try t.gpa.alloc(u16, S * I);
    defer t.gpa.free(acts);
    for (acts) |*a| a.* = h.bf((t.r.float(f64) * 2 - 1) * 0.7);
    {
        const ses = t.open();
        xp.group(ses.e, t.p, t.lay.layer, t.io.picks(0).ids, t.s, n);
        if (w13) {
            xp.prepare(ses.e, t.p, t.lay.layer, t.io.xAt(0), t.s, n);
            xp.gemm(ses.e, t.p, .w13, t.lay.layer, t.s, n);
        }
        try ses.close();
    }
    const inverse = view(u32, t.s.inverse, S);
    // the rotated rows of each slot: exact (FP64, unrounded) and rounded (the f16 values the kernels multiply)
    const exact = try t.gpa.alloc(f64, S * K);
    defer t.gpa.free(exact);
    const rounded = try t.gpa.alloc(f64, S * K);
    defer t.gpa.free(rounded);
    var prods: [H]f32 = undefined;
    var row: [H]u16 = undefined;
    const prep = view(u16, t.s.prep, S * H);
    const x = t.io.x.slice(u16, n * H);
    for (0..S) |slot| {
        const ref = t.lay.refs[ids[slot]];
        const e = exact[slot * K ..][0..K];
        if (w13) {
            for (prods[0..K], 0..) |*p, i| p.* = bf16f32(x[slot / TOPK * H + i]) * @as(f32, ref.s13[i]);
            for (rounded[slot * K ..][0..K], prep[inverse[slot] * K ..][0..K]) |*d, q| d.* = f16v(q);
        } else {
            for (prods[0..K], 0..) |*p, i| p.* = bf16f32(acts[slot * I + i]) * @as(f32, ref.s2[i]);
            rotateF16(row[0..K], prods[0..K]);
            for (rounded[slot * K ..][0..K], row[0..K], prep[inverse[slot] * K ..][0..K]) |*d, q, *dst| {
                d.* = f16v(q);
                dst.* = q;
            }
        }
        for (e, prods[0..K]) |*d, p| d.* = p;
        had128(e);
    }
    if (!w13) {
        const ses = t.open();
        xp.gemm(ses.e, t.p, .w2, t.lay.layer, t.s, n);
        try ses.close();
    }
    // the decode port on the last 16 rows (the last slots of each expert: the ends of its windows)
    const rows0 = n - 16;
    const ids_at: xp.B = .{ .buf = t.io.ids, .off = rows0 * TOPK * 4 };
    const ew_at: xp.B = .{ .buf = t.io.ew, .off = rows0 * TOPK * 4 };
    if (w13) {
        try t.dec.run(t, t.io.xAt(rows0), ids_at, ew_at, 16, .{ .pair = true });
    } else {
        @memcpy(t.dec.act.slice(u16, Decode.slots * I), acts[rows0 * TOPK * I ..][0 .. Decode.slots * I]);
        try t.dec.run(t, t.io.xAt(rows0), ids_at, ew_at, 16, .{ .down = true });
    }
    // exact sums for set A (the decode port's slots) and set B (random slots across the batch)
    const nb: usize = if (t.lay.count > 64) 48 else 96;
    const picked = try t.gpa.alloc(usize, Decode.slots + nb);
    defer t.gpa.free(picked);
    for (picked[0..Decode.slots], 0..) |*s, i| s.* = rows0 * TOPK + i;
    for (picked[Decode.slots..]) |*s| s.* = t.r.uintLessThan(usize, S);
    const want_x = try t.gpa.alloc(f64, picked.len * N);
    defer t.gpa.free(want_x);
    const want_r = try t.gpa.alloc(f64, picked.len * N);
    defer t.gpa.free(want_r);
    const ins = try t.gpa.alloc([]const f64, 2 * picked.len);
    defer t.gpa.free(ins);
    const outs = try t.gpa.alloc([]f64, 2 * picked.len);
    defer t.gpa.free(outs);
    // each expert's tiles are decoded once for both references' rows
    for (0..t.lay.count) |e| {
        const ref = t.lay.refs[e];
        var m: usize = 0;
        for (picked, 0..) |slot, i| if (ids[slot] == e) {
            ins[m] = exact[slot * K ..][0..K];
            outs[m] = want_x[i * N ..][0..N];
            ins[m + 1] = rounded[slot * K ..][0..K];
            outs[m + 1] = want_r[i * N ..][0..N];
            m += 2;
        };
        if (m > 0) gemmRef(t.lut, if (w13) ref.t13 else ref.t2, ref.bits, K, N, ins[0..m], outs[0..m]);
    }
    const plane = view(f32, t.s.plane, S * N);
    const dsum = if (w13) t.dec.planes.slice(f32, Decode.slots * 2 * N) else t.dec.sums.slice(f32, Decode.slots * N);
    const got_pre = try t.gpa.alloc(f64, picked.len * N);
    defer t.gpa.free(got_pre);
    const got_dec = try t.gpa.alloc(f64, Decode.slots * N);
    defer t.gpa.free(got_dec);
    var worst_abs: f64 = 0;
    for (picked, 0..) |slot, i| for (0..N) |col| {
        got_pre[i * N + col] = plane[inverse[slot] * N + col];
        worst_abs = @max(worst_abs, @abs(got_pre[i * N + col] - want_r[i * N + col]));
    };
    for (0..Decode.slots) |i| for (0..N) |col| {
        // the decode chain's mid adds w13's two K halves in this order
        const v: f32 = if (w13) (0.0 + dsum[(i * 2) * N + col]) + dsum[(i * 2 + 1) * N + col] else dsum[i * N + col];
        got_dec[i * N + col] = v;
    };
    const a_n = Decode.slots * N;
    const x_pre_a = relRms(got_pre[0..a_n], want_x[0..a_n]);
    const x_dec_a = relRms(got_dec, want_x[0..a_n]);
    const x_pre_b = relRms(got_pre[a_n..], want_x[a_n..]);
    const r_pre_a = relRms(got_pre[0..a_n], want_r[0..a_n]);
    const r_dec_a = relRms(got_dec, want_r[0..a_n]);
    const r_pre_b = relRms(got_pre[a_n..], want_r[a_n..]);
    const mag = rms(want_r);
    var c: h.Check = .{ .name = if (w13) "xp w13 GEMM f32 sums against FP64, within the decode port's error" else "xp w2 GEMM f32 sums against FP64, within the decode port's error" };
    c.n += 1;
    c.worst = x_pre_a / x_dec_a;
    if (!(x_pre_a <= gemm_ratio * x_dec_a)) c.bad += 1;
    c.n += 1;
    if (!(x_pre_b <= 1.25 * x_dec_a)) c.bad += 1;
    c.n += 1;
    if (!(r_pre_a <= accumulation_bound and r_pre_b <= accumulation_bound)) c.bad += 1;
    c.n += 1;
    if (!(worst_abs <= 1e-4 * mag)) c.bad += 1;
    std.debug.print("  {s} vs exact math: rel RMS prefill {e:.3} (random slots {e:.3}), decode {e:.3}\n", .{ if (w13) "w13" else "w2", x_pre_a, x_pre_b, x_dec_a });
    std.debug.print("  {s} accumulation only (the same f16 rows): prefill {e:.3} (random slots {e:.3}), decode {e:.3}; worst |error| {e:.3} of rms {e:.3}\n", .{ if (w13) "w13" else "w2", r_pre_a, r_pre_b, r_dec_a, worst_abs, mag });
    try c.done();
}

/// gu, act and the down projection's rotated input from the w13 plane: the Hadamard, scale and svh, gu rounded to
/// bf16, the capped SwiGLU on adjacent pairs rounded to bf16, then act * suh2, the Hadamard, the scale and f16.
fn caseMid(t: Ctx) !void {
    const n = 40;
    const S = n * TOPK;
    t.io.fill(t.r, n);
    try t.picks(n, 1.0);
    poisonUpTo(t.s, n);
    {
        const ses = t.open();
        xp.group(ses.e, t.p, t.lay.layer, t.io.picks(0).ids, t.s, n);
        try ses.close();
    }
    const plane = view(f32, t.s.plane, S * H);
    for (plane) |*v| v.* = @floatCast((t.r.float(f64) * 2 - 1) * 6);
    const gu = try t.g.buf(S * H * 2);
    defer gu.deinit();
    const act = try t.g.buf(S * I * 2);
    defer act.deinit();
    {
        const ses = t.open();
        xp.mid(ses.e, t.p, t.lay.layer, t.s, n, caps, .{ .gu = .{ .buf = gu }, .act = .{ .buf = act } });
        try ses.close();
    }
    const seid = view(u32, t.s.seid, S);
    const gus = gu.slice(u16, S * H);
    const acts = act.slice(u16, S * I);
    const prep = view(u16, t.s.prep, S * I);
    var cg: h.Check = .{ .name = "xp mid gu against FP64" };
    var cb: h.Check = .{ .name = "xp mid gu equals the decode kernels' f32 chain, bit for bit" };
    var ca: h.Check = .{ .name = "xp mid act against FP64 on the kernel's gu" };
    var cp: h.Check = .{ .name = "xp mid down input equals the decode kernels' rotation of the kernel's act, bit for bit" };
    var v: [H]f64 = undefined;
    var blk: [128]f32 = undefined;
    var gu_host: [H]u16 = undefined;
    var prods: [I]f32 = undefined;
    var row: [I]u16 = undefined;
    for (0..S) |pos| {
        const ref = t.lay.refs[seid[pos]];
        for (&v, plane[pos * H ..][0..H]) |*e, p| e.* = p;
        had128(&v);
        for (&v, 0..) |*e, i| e.* *= @as(f64, ref.v13[i]);
        const mag = rms(&v);
        for (0..H) |i| near16(&cg, gus[pos * H + i], v[i], 2e-3, mag);
        for (0..H / 128) |b| {
            @memcpy(&blk, plane[pos * H + b * 128 ..][0..128]);
            h128f32(&blk);
            for (blk, 0..) |s, i| gu_host[b * 128 + i] = h.bf(@as(f64, s * psc * @as(f32, ref.v13[b * 128 + i])));
        }
        cb.same(std.mem.sliceAsBytes(gus[pos * H ..][0..H]), std.mem.sliceAsBytes(&gu_host));
        for (0..I) |j| {
            const gg = h.fb(gus[pos * H + 2 * j]);
            const u = h.fb(gus[pos * H + 2 * j + 1]);
            ca.b16(acts[pos * I + j], 4 * std.math.tanh(gg / 4) * (1 / (1 + @exp(-gg))) * (25 * std.math.tanh(u / 25)));
        }
        for (&prods, 0..) |*p, i| p.* = bf16f32(acts[pos * I + i]) * @as(f32, ref.s2[i]);
        rotateF16(&row, &prods);
        cp.same(std.mem.sliceAsBytes(prep[pos * I ..][0..I]), std.mem.sliceAsBytes(&row));
    }
    try cg.done();
    try cb.done();
    try ca.done();
    try cp.done();
}

/// Each pick's w2 sums (rows of the plane at its sorted position) through the output Hadamard to ey at bf16, and the
/// pick-ordered f32 sum of ey * w rounded once: against FP64, and the sum's f32 chain bit for bit.
fn caseFinish(t: Ctx) !void {
    const n = 40;
    const S = n * TOPK;
    t.io.fill(t.r, n);
    try t.picks(n, 1.0);
    poisonUpTo(t.s, n);
    {
        const ses = t.open();
        xp.group(ses.e, t.p, t.lay.layer, t.io.picks(0).ids, t.s, n);
        try ses.close();
    }
    const plane = view(f32, t.s.plane, S * H);
    for (plane) |*v| v.* = @floatCast((t.r.float(f64) * 2 - 1) * 2);
    const ey = try t.g.buf(S * H * 2);
    defer ey.deinit();
    @memset(t.io.out.slice(u8, n * H * 2), 0xab);
    {
        const ses = t.open();
        xp.finish(ses.e, t.p, t.lay.layer, t.io.picks(0), t.io.outAt(0), t.s, n, .{ .buf = ey });
        try ses.close();
    }
    const ids = t.io.ids.slice(u32, S);
    const ew = t.io.ew.slice(f32, S);
    const inverse = view(u32, t.s.inverse, S);
    const eys = ey.slice(u16, S * H);
    const outs = t.io.out.slice(u16, n * H);
    var ce: h.Check = .{ .name = "xp finish per-pick ey against FP64: output Hadamard, svh2, bf16" };
    var cm: h.Check = .{ .name = "xp finish weighted sum against FP64 on the kernel's ey" };
    var cx: h.Check = .{ .name = "xp finish weighted sum equals the pick-ordered f32 fma chain, bit for bit" };
    var v: [H]f64 = undefined;
    for (0..S) |slot| {
        const ref = t.lay.refs[ids[slot]];
        for (&v, plane[inverse[slot] * H ..][0..H]) |*e, p| e.* = p;
        had128(&v);
        for (&v, 0..) |*e, i| e.* *= @as(f64, ref.v2[i]);
        const mag = rms(&v);
        for (0..H) |i| near16(&ce, eys[slot * H + i], v[i], 2e-3, mag);
    }
    for (0..n) |row| {
        var want: [H]f64 = undefined;
        for (&want, 0..) |*w, i| {
            w.* = 0;
            for (0..TOPK) |k| w.* += h.fb(eys[(row * TOPK + k) * H + i]) * ew[row * TOPK + k];
        }
        const mag = rms(&want);
        for (0..H) |i| {
            cm.b16a(outs[row * H + i], want[i], 2, 1e-5 * mag);
            var a: f32 = 0;
            for (0..TOPK) |k| a = @mulAdd(f32, @as(f32, @floatCast(h.fb(eys[(row * TOPK + k) * H + i]))), ew[row * TOPK + k], a);
            cx.n += 1;
            if (outs[row * H + i] != h.bf(a)) cx.bad += 1;
        }
    }
    try ce.done();
    try cm.done();
    try cx.done();
}

/// The whole chain on 16 rows against the decode chain on the same rows, picks and weights: the two differ only in the
/// order of f32 sums, which moves a few values across bf16 rounding boundaries (gu, act and ey each round), so the
/// outputs agree to the size of two bf16 roundings (relative RMS under 2e-3; one rounding is 1.1e-3) and no output
/// moves by more than a tenth of its row's magnitude (a sum that cancels toward zero keeps the absolute error of its
/// terms); a wrong weight, rotation or pick moves them by the whole of it.
fn caseChain(t: Ctx) !void {
    const n = 16;
    t.io.fill(t.r, n);
    try t.picks(n, 1.0);
    poisonUpTo(t.s, n);
    @memset(t.io.out.slice(u8, n * H * 2), 0xab);
    const ses = t.open();
    t.encode(ses, t.io.xAt(0), t.io.picks(0), t.io.outAt(0), n);
    try ses.close();
    try t.dec.run(t, t.io.xAt(0), .{ .buf = t.io.ids }, .{ .buf = t.io.ew }, 16, .{ .pair = true, .mid = true, .down = true, .finish = true });
    const got = t.io.out.slice(u16, n * H);
    const want = t.dec.moe.slice(u16, n * H);
    var a: [n * H]f64 = undefined;
    var b: [n * H]f64 = undefined;
    for (&a, got) |*d, w| d.* = h.fb(w);
    for (&b, want) |*d, w| d.* = h.fb(w);
    var differ: usize = 0;
    var worst: f64 = 0;
    const mag = rms(&b);
    for (got, want, a, b) |x, y, p, q| if (x != y) {
        differ += 1;
        worst = @max(worst, @abs(p - q) / mag);
    };
    const rel = relRms(&a, &b);
    var c: h.Check = .{ .name = "xp chain output against the decode chain's" };
    c.n += 1;
    c.worst = rel;
    if (!(rel <= 2e-3)) c.bad += 1;
    c.n += 1;
    if (!(worst <= 0.1)) c.bad += 1;
    std.debug.print("  chain vs decode: {d} of {d} outputs differ, rel RMS {e:.3}, largest difference {e:.3} of the rms\n", .{ differ, n * H, rel, worst });
    try c.done();
}

/// Every row of a batch equals the same row run alone, a non-finite input row stays in its own output row, and a pick
/// outside the table does too.
fn caseRows(t: Ctx) !void {
    const n = 40;
    t.io.fill(t.r, n);
    try t.picks(n, 1.4);
    poisonUpTo(t.s, n);
    const ses = t.open();
    t.encode(ses, t.io.xAt(0), t.io.picks(0), t.io.outAt(0), n);
    try ses.close();
    const base = try t.gpa.dupe(u16, t.io.out.slice(u16, n * H));
    defer t.gpa.free(base);
    var c: h.Check = .{ .name = "xp every row alone equals the row in a batch" };
    const solo = try t.g.buf(H * 2);
    defer solo.deinit();
    for (0..n) |row| {
        poisonUpTo(t.s, 4);
        @memset(solo.slice(u8, H * 2), 0xab);
        const s1 = t.open();
        t.encode(s1, t.io.xAt(row), t.io.picks(row), .{ .buf = solo }, 1);
        try s1.close();
        c.same(std.mem.sliceAsBytes(solo.slice(u16, H)), std.mem.sliceAsBytes(base[row * H ..][0..H]));
    }
    try c.done();

    var cn: h.Check = .{ .name = "xp non-finite row and a pick outside the table contained" };
    const xs = t.io.x.slice(u16, n * H);
    const kept = xs[H + 5];
    xs[H + 5] = nan_bf16;
    poisonUpTo(t.s, n);
    const s2 = t.open();
    t.encode(s2, t.io.xAt(0), t.io.picks(0), t.io.outAt(0), n);
    try s2.close();
    xs[H + 5] = kept;
    try allButRow(&cn, t, base, 1, n);
    cn.n += 1;
    for (t.io.out.slice(u16, n * H)[H..][0..H]) |v| if (!std.math.isNan(h.fb(v))) {
        cn.bad += 1;
        break;
    };
    const ids = t.io.ids.slice(u32, n * TOPK);
    const was = ids[2 * TOPK + 3];
    ids[2 * TOPK + 3] = @intCast(t.lay.count);
    poisonUpTo(t.s, n);
    const s3 = t.open();
    t.encode(s3, t.io.xAt(0), t.io.picks(0), t.io.outAt(0), n);
    try s3.close();
    ids[2 * TOPK + 3] = was;
    try allButRow(&cn, t, base, 2, n);
    cn.n += 1;
    for (t.io.out.slice(u16, n * H)[2 * H ..][0..H]) |v| if (!std.math.isNan(h.fb(v))) {
        cn.bad += 1;
        break;
    };
    try cn.done();
}

/// Every row picks the same eight experts, so each expert's slots fill whole windows (32, 32 and a part) and a row's
/// place in them moves with its index: rows at the window edges run alone give the batch's bits.
fn caseSameRoute(t: Ctx) !void {
    const n = 70;
    t.io.fill(t.r, n);
    const ids = t.io.ids.slice(u32, n * TOPK);
    const perm = try shuffled(t.r, t.gpa, t.lay.count);
    defer t.gpa.free(perm);
    for (0..n) |row| for (0..TOPK) |k| {
        ids[row * TOPK + k] = perm[(k + row) % TOPK];
    };
    poisonUpTo(t.s, n);
    const ses = t.open();
    t.encode(ses, t.io.xAt(0), t.io.picks(0), t.io.outAt(0), n);
    try ses.close();
    const base = try t.gpa.dupe(u16, t.io.out.slice(u16, n * H));
    defer t.gpa.free(base);
    var c: h.Check = .{ .name = "xp rows of one route at the edges of full windows equal the same rows alone" };
    const solo = try t.g.buf(H * 2);
    defer solo.deinit();
    for ([_]usize{ 0, 1, 15, 16, 31, 32, 33, 63, 64, 69 }) |row| {
        poisonUpTo(t.s, 4);
        @memset(solo.slice(u8, H * 2), 0xab);
        const s1 = t.open();
        t.encode(s1, t.io.xAt(row), t.io.picks(row), .{ .buf = solo }, 1);
        try s1.close();
        c.same(std.mem.sliceAsBytes(solo.slice(u16, H)), std.mem.sliceAsBytes(base[row * H ..][0..H]));
    }
    try c.done();
}

/// Every output row of the chain's `out` but `skip` equal to the same row of `base`.
fn allButRow(c: *h.Check, t: Ctx, base: []const u16, skip: usize, n: usize) !void {
    const out = t.io.out.slice(u16, n * H);
    for (0..n) |row| {
        if (row == skip) continue;
        c.same(std.mem.sliceAsBytes(out[row * H ..][0..H]), std.mem.sliceAsBytes(base[row * H ..][0..H]));
    }
}

/// The same rows with the same picks, preceded or followed by other rows that route differently, share windows with
/// different neighbours and sit at other places in them: their outputs are the same bits.
fn caseWindows(t: Ctx) !void {
    const n0 = 300;
    t.io.fill(t.r, n0);
    try t.picks(n0, 1.2);
    const bx = try t.gpa.dupe(u16, t.io.x.slice(u16, n0 * H));
    defer t.gpa.free(bx);
    const bi = try t.gpa.dupe(u32, t.io.ids.slice(u32, n0 * TOPK));
    defer t.gpa.free(bi);
    const bw = try t.gpa.dupe(f32, t.io.ew.slice(f32, n0 * TOPK));
    defer t.gpa.free(bw);
    poisonUpTo(t.s, n0);
    const ses = t.open();
    t.encode(ses, t.io.xAt(0), t.io.picks(0), t.io.outAt(0), n0);
    try ses.close();
    const base = try t.gpa.dupe(u16, t.io.out.slice(u16, n0 * H));
    defer t.gpa.free(base);
    var c: h.Check = .{ .name = "xp rows share windows with other neighbours: same bits" };
    // rows before (the target rows shift by any number of positions in each expert's windows) and rows after (they
    // fill the last windows differently)
    for ([_]struct { before: usize, after: usize, skew: f64 }{
        .{ .before = 1, .after = 0, .skew = 0.2 },
        .{ .before = 5, .after = 0, .skew = 2.0 },
        .{ .before = 17, .after = 0, .skew = 0.2 },
        .{ .before = 100, .after = 0, .skew = 1.2 },
        .{ .before = 0, .after = 77, .skew = 2.0 },
        .{ .before = 33, .after = 200, .skew = 0.8 },
    }) |w| {
        const total = w.before + n0 + w.after;
        t.io.fill(t.r, total);
        try t.picks(total, w.skew);
        @memcpy(t.io.x.slice(u16, total * H)[w.before * H ..][0 .. n0 * H], bx);
        @memcpy(t.io.ids.slice(u32, total * TOPK)[w.before * TOPK ..][0 .. n0 * TOPK], bi);
        @memcpy(t.io.ew.slice(f32, total * TOPK)[w.before * TOPK ..][0 .. n0 * TOPK], bw);
        poisonUpTo(t.s, total);
        const s2 = t.open();
        t.encode(s2, t.io.xAt(0), t.io.picks(0), t.io.outAt(0), total);
        try s2.close();
        c.same(std.mem.sliceAsBytes(t.io.out.slice(u16, total * H)[w.before * H ..][0 .. n0 * H]), std.mem.sliceAsBytes(base));
    }
    try c.done();
}

/// One run of a full chunk against the same rows cut into pieces, every piece its own run on a scratch poisoned before
/// it: `out` is bit-identical row for row.
fn casePartition(t: Ctx) !void {
    const n = rows_max;
    t.io.fill(t.r, n);
    try t.picks(n, 1.1);
    poisonUpTo(t.s, n);
    const ses = t.open();
    t.encode(ses, t.io.xAt(0), t.io.picks(0), t.io.outAt(0), n);
    try ses.close();
    const ref = try t.gpa.dupe(u8, t.io.out.slice(u8, n * H * 2));
    defer t.gpa.free(ref);
    var c: h.Check = .{ .name = "xp a 4,096-row chunk cut anywhere gives the same bits" };
    var tails: [31][2]usize = undefined;
    for (&tails, 1..) |*cut, k| cut.* = .{ n - k, k };
    try partition(t, &c, ref, &.{ 4095, 1 });
    try partition(t, &c, ref, &.{ 2048, 2048 });
    try partition(t, &c, ref, &.{ 37, 4059 });
    try partition(t, &c, ref, &.{ 1000, 1000, 2096 });
    for (tails) |cut| try partition(t, &c, ref, &cut);
    try c.done();
}

fn partition(t: Ctx, c: *h.Check, ref: []const u8, cuts: []const usize) !void {
    @memset(t.io.out.slice(u8, rows_max * H * 2), 0xff);
    var row0: usize = 0;
    for (cuts) |len| {
        poisonUpTo(t.s, rows_max);
        const ses = t.open();
        t.encode(ses, t.io.xAt(row0), t.io.picks(row0), t.io.outAt(row0), len);
        try ses.close();
        row0 += len;
    }
    std.debug.assert(row0 == rows_max);
    c.same(t.io.out.slice(u8, rows_max * H * 2), ref);
}
