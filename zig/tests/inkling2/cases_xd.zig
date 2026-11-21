//! Inkling 2 kernel checks: the EXL3 decode chain (ik_xd_*) against an FP64 reference rebuilt from the trellis bytes,
//! windows of rows against the same rows one at a time, a non-finite slot confined to its own outputs, and layers of
//! mixed rates.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const h = @import("harness.zig");
const exl3 = tf.exl3_projection;
const format = exl3.format;

const H = 4608;
const I = 2304;
const TOPK = 8;
const MAX_ROWS = 16;
const had_scale: f64 = 1.0 / @sqrt(128.0);
const nan_bf16: u16 = 0x7fc0;

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    const gpa = std.heap.page_allocator;
    const lut = try mul1Table(gpa);
    defer gpa.free(lut);
    try caseDecode(gpa, lut, r);
    try caseXdDecode(g, lut);
    var fx = try Fixture.init(g, gpa, r);
    defer fx.deinit(gpa);
    try caseXdRef(g, gpa, lut, &fx, r);
    try caseXdWindow(g, gpa, &fx, r);
    try caseXdMixed(g, gpa, &fx, r);
}

/// Every 16-bit state through the kernels' codebook decode: the f16 value the reference table holds, bit for bit.
fn caseXdDecode(g: *const h.Gpu, lut: []const f32) !void {
    const states = try g.buf(65536 * 2);
    defer states.deinit();
    const out = try g.buf(65536 * 2);
    defer out.deinit();
    for (states.slice(u16, 65536), 0..) |*s, i| s.* = @intCast(i);
    try g.run("ik_xd_decode", .{}, &.{ .{ .buf = states }, .{ .buf = out } }, .{ 32768, 1, 1 }, .{ 256, 1, 1 }, false);
    var c: h.Check = .{ .name = "xd codebook decode of every state" };
    for (out.slice(u16, 65536), lut) |o, want| {
        c.n += 1;
        if (@as(f32, @floatCast(@as(f16, @bitCast(o)))) != want) c.bad += 1;
    }
    try c.done();
}

// ---- host reference ----

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

/// y = svh .* H((H(x .* suh)) . Wq) for each input of one expert's projection (k inputs, n outputs; Wq the trellis'
/// decoded weights, one tile at a time): the checkpoint format's W = diag(suh) H Wq H diag(svh) without building W.
fn project(gpa: std.mem.Allocator, lut: []const f32, words: []const u32, bits: u32, k: usize, n: usize, suh: []const f16, svh: []const f16, xs: []const []const f64, ys: []const []f64) !void {
    const prep = try gpa.alloc(f64, xs.len * k);
    defer gpa.free(prep);
    const acc = try gpa.alloc(f64, xs.len * n);
    defer gpa.free(acc);
    @memset(acc, 0);
    for (xs, 0..) |x, i| {
        const v = prep[i * k ..][0..k];
        for (v, x, suh) |*d, xv, sv| d.* = xv * sv;
        had128(v);
    }
    const per_tile = 8 * bits;
    var tile: [256]f32 = undefined;
    for (0..k / 16) |ti| {
        for (0..n / 16) |tj| {
            decodeTile(words[(ti * (n / 16) + tj) * per_tile ..][0..per_tile], bits, lut, &tile);
            for (0..xs.len) |i| {
                const v = prep[i * k + ti * 16 ..][0..16];
                const a = acc[i * n + tj * 16 ..][0..16];
                for (0..16) |rr| {
                    for (0..16) |cc| a[cc] += v[rr] * tile[rr * 16 + cc];
                }
            }
        }
    }
    for (0..xs.len) |i| {
        const a = acc[i * n ..][0..n];
        had128(a);
        for (a, svh, ys[i]) |av, sv, *out| out.* = av * sv;
    }
}

fn rms(v: []const f64) f64 {
    var s: f64 = 0;
    for (v) |x| s += x * x;
    return @sqrt(s / @as(f64, @floatFromInt(@max(v.len, 1))));
}

fn near64(c: *h.Check, got: f64, want: f64, tol: f64, mag: f64) void {
    c.n += 1;
    const e = @abs(got - want) / @max(@abs(want), mag);
    c.worst = @max(c.worst, e);
    if (!(e <= tol)) {
        c.bad += 1;
        if (c.first == null) c.first = .{ got, want };
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

fn words32(b: mtl.Buffer) []u32 {
    return b.slice(u32, b.length() / 4);
}

/// The reference against the checkpoint format's own decoder: every state's value, and a 256 x 256 projection of each
/// width as y = x . W with W = format.dequantize's dense matrix.
fn caseDecode(gpa: std.mem.Allocator, lut: []const f32, r: std.Random) !void {
    var c: h.Check = .{ .name = "xd reference against the format decoder" };
    for (lut, 0..) |v, state| {
        c.n += 1;
        if (v != @as(f32, @floatCast(format.codebookValue(.mul1, @intCast(state))))) c.bad += 1;
    }
    const k = 256;
    const n = 256;
    for ([_]u32{ 2, 3 }) |bits| {
        const words = try gpa.alloc(u32, (k / 16) * (n / 16) * 8 * bits);
        defer gpa.free(words);
        for (words) |*w| w.* = r.int(u32);
        var suh: [k]f16 = undefined;
        var svh: [n]f16 = undefined;
        for (&suh) |*s| s.* = @floatCast(0.5 + r.float(f32));
        for (&svh) |*s| s.* = @floatCast(0.5 + r.float(f32));
        const dense = try gpa.alloc(f64, k * n);
        defer gpa.free(dense);
        const scratch = try gpa.alloc(f64, 2 * k * n);
        defer gpa.free(scratch);
        format.dequantize(.{ .halves = @intCast(2 * bits) }, .mul1, std.mem.bytesAsSlice(i16, std.mem.sliceAsBytes(words)), k / 16, n / 16, &suh, &svh, dense, scratch);
        var x: [k]f64 = undefined;
        for (&x) |*e| e.* = r.float(f64) * 2 - 1;
        var y: [n]f64 = undefined;
        try project(gpa, lut, words, bits, k, n, &suh, &svh, &.{&x}, &.{&y});
        var want: [n]f64 = undefined;
        for (&want, 0..) |*w, col| {
            w.* = 0;
            for (0..k) |row| w.* += x[row] * dense[row * n + col];
        }
        const mag = rms(&want);
        for (y, want) |got, w| near64(&c, got, w, 1e-9, mag);
    }
    try c.done();
}

// ---- fixture ----

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
};

/// A layer's descriptor table over `experts` (w13's then w2's), and every buffer the descriptors point at.
const Table = struct {
    buf: mtl.Buffer,
    count: usize,
    weights: []mtl.Buffer,

    fn init(g: *const h.Gpu, gpa: std.mem.Allocator, experts: []const *const Expert) !Table {
        const count = experts.len;
        const buf = try g.buf(2 * count * @sizeOf(exl3.ExpertDescriptor));
        const desc = buf.slice(exl3.ExpertDescriptor, 2 * count);
        const weights = try gpa.alloc(mtl.Buffer, 6 * count);
        for (experts, 0..) |ex, i| {
            const width: format.Width = .{ .halves = @intCast(2 * ex.bits) };
            desc[i] = try exl3.descriptor(.{ .trellis = .{ .buffer = ex.t13 }, .suh = .{ .buffer = ex.s13 }, .svh = .{ .buffer = ex.v13 }, .k = H, .n = 2 * I, .width = width, .codebook = .mul1 });
            desc[count + i] = try exl3.descriptor(.{ .trellis = .{ .buffer = ex.t2 }, .suh = .{ .buffer = ex.s2 }, .svh = .{ .buffer = ex.v2 }, .k = I, .n = H, .width = width, .codebook = .mul1 });
            @memcpy(weights[6 * i ..][0..6], &[_]mtl.Buffer{ ex.t13, ex.s13, ex.v13, ex.t2, ex.s2, ex.v2 });
        }
        return .{ .buf = buf, .count = count, .weights = weights };
    }

    fn deinit(t: Table, gpa: std.mem.Allocator) void {
        t.buf.deinit();
        gpa.free(t.weights);
    }

    fn descriptors(t: Table) []exl3.ExpertDescriptor {
        return t.buf.slice(exl3.ExpertDescriptor, 2 * t.count);
    }
};

/// The chain's inputs and outputs for up to 16 rows (128 slots).
const Bufs = struct {
    x: mtl.Buffer,
    eids: mtl.Buffer,
    ew: mtl.Buffer,
    planes: mtl.Buffer,
    act: mtl.Buffer,
    sums: mtl.Buffer,
    moe: mtl.Buffer,
    gu: mtl.Buffer,
    ey: mtl.Buffer,

    const slots = MAX_ROWS * TOPK;

    fn init(g: *const h.Gpu) !Bufs {
        return .{
            .x = try g.buf(MAX_ROWS * H * 2),
            .eids = try g.buf(slots * 4),
            .ew = try g.buf(slots * 4),
            .planes = try g.buf(slots * 2 * H * 4),
            .act = try g.buf(slots * I * 2),
            .sums = try g.buf(slots * H * 4),
            .moe = try g.buf(MAX_ROWS * H * 2),
            .gu = try g.buf(slots * H * 2),
            .ey = try g.buf(slots * H * 2),
        };
    }

    fn deinit(b: Bufs) void {
        for ([_]mtl.Buffer{ b.x, b.eids, b.ew, b.planes, b.act, b.sums, b.moe, b.gu, b.ey }) |x| x.deinit();
    }

    /// Every output byte `byte`, so an unwritten element shows.
    fn fillOutputs(b: Bufs, byte: u8) void {
        for ([_]mtl.Buffer{ b.planes, b.act, b.sums, b.moe, b.gu, b.ey }) |x| @memset(x.slice(u8, x.length()), byte);
    }
};

const Chain = struct {
    pair: mtl.Pipeline,
    mid: mtl.Pipeline,
    down: mtl.Pipeline,
    finish: mtl.Pipeline,
    mid_dbg: mtl.Pipeline,
    finish_dbg: mtl.Pipeline,

    const names = .{ "pair", "mid", "down", "finish", "mid_dbg", "finish_dbg" };

    fn init(g: *const h.Gpu) !Chain {
        var c: Chain = undefined;
        inline for (names) |name| {
            @field(c, name) = try mtl.Pipeline.init(g.device, g.lib, "ik_xd_" ++ name, false);
            if (@field(c, name).simdWidth() != 32 or @field(c, name).maxThreads() < 256) return error.UnsupportedSimd;
        }
        return c;
    }

    fn deinit(c: Chain) void {
        inline for (names) |name| @field(c, name).deinit();
    }
};

/// Which dispatches of the chain to encode; `dbg` takes the mid and finish kernels that also store gu and ey.
const Stages = struct { pair: bool = true, mid: bool = true, down: bool = true, finish: bool = true, dbg: bool = false };

const Fixture = struct {
    experts: [6]Expert,
    /// Experts 0..3 at K3, K2, K3, K2.
    mixed: Table,
    /// Experts 0, 4, 2, 5, all at K3: the mixed layer's K3 experts in place, its K2 ones replaced.
    pure: Table,
    bufs: Bufs,
    chain: Chain,

    fn init(g: *const h.Gpu, gpa: std.mem.Allocator, r: std.Random) !Fixture {
        var fx: Fixture = undefined;
        const bits = [_]u32{ 3, 2, 3, 2, 3, 3 };
        for (&fx.experts, bits) |*e, b| e.* = try Expert.init(g, r, b);
        const ex = &fx.experts;
        fx.mixed = try Table.init(g, gpa, &.{ &ex[0], &ex[1], &ex[2], &ex[3] });
        fx.pure = try Table.init(g, gpa, &.{ &ex[0], &ex[4], &ex[2], &ex[5] });
        fx.bufs = try Bufs.init(g);
        fx.chain = try Chain.init(g);
        return fx;
    }

    fn deinit(fx: *Fixture, gpa: std.mem.Allocator) void {
        fx.chain.deinit();
        fx.bufs.deinit();
        fx.pure.deinit(gpa);
        fx.mixed.deinit(gpa);
        for (fx.experts) |e| e.deinit();
    }

    /// Rows `row0 ..` of the buffers through the chain: `rows` rows in one command buffer, one dispatch a stage.
    fn run(fx: *const Fixture, g: *const h.Gpu, t: Table, row0: usize, rows: usize, st: Stages) !void {
        const b = fx.bufs;
        const c = fx.chain;
        const S = rows * TOPK;
        const s0 = row0 * TOPK;
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.useResource(t.buf, mtl.ResourceUsage.read);
        for (t.weights) |w| e.useResource(w, mtl.ResourceUsage.read);
        const args = h.Args{ .rows = @intCast(S), .cap = 4, .thr = 25 };
        const w2 = t.count * @sizeOf(exl3.ExpertDescriptor);
        if (st.pair) {
            e.setPipeline(c.pair);
            e.setValue(args, 0);
            e.setBuffer(b.x, row0 * H * 2, 1);
            e.setBuffer(b.eids, s0 * 4, 2);
            e.setBuffer(t.buf, 0, 3);
            e.setBuffer(b.planes, s0 * 2 * H * 4, 4);
            e.dispatchGroups(mtl.Size.of(144, S, 2), mtl.Size.of(128, 1, 1));
        }
        if (st.mid) {
            e.setPipeline(if (st.dbg) c.mid_dbg else c.mid);
            e.setValue(args, 0);
            e.setBuffer(b.planes, s0 * 2 * H * 4, 1);
            e.setBuffer(b.eids, s0 * 4, 2);
            e.setBuffer(t.buf, 0, 3);
            e.setBuffer(b.act, s0 * I * 2, 4);
            if (st.dbg) e.setBuffer(b.gu, s0 * H * 2, 5);
            e.dispatchGroups(mtl.Size.of(36, S, 1), mtl.Size.of(32, 1, 1));
        }
        if (st.down) {
            e.setPipeline(c.down);
            e.setValue(args, 0);
            e.setBuffer(b.act, s0 * I * 2, 1);
            e.setBuffer(b.eids, s0 * 4, 2);
            e.setBuffer(t.buf, w2, 3);
            e.setBuffer(b.sums, s0 * H * 4, 4);
            e.dispatchGroups(mtl.Size.of(144, S, 1), mtl.Size.of(128, 1, 1));
        }
        if (st.finish) {
            e.setPipeline(if (st.dbg) c.finish_dbg else c.finish);
            e.setValue(args, 0);
            e.setBuffer(b.sums, s0 * H * 4, 1);
            e.setBuffer(b.eids, s0 * 4, 2);
            e.setBuffer(t.buf, w2, 3);
            e.setBuffer(b.ew, s0 * 4, 4);
            e.setBuffer(b.moe, row0 * H * 2, 5);
            if (st.dbg) e.setBuffer(b.ey, s0 * H * 2, 6);
            e.dispatchGroups(mtl.Size.of(36, rows, 1), mtl.Size.of(256, 1, 1));
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("ik_xd chain: {s}", .{msg});
            return error.GpuFailed;
        }
    }

    /// Random inputs for `rows` rows: x in [-0.1, 0.1] (gu then spans the SwiGLU's caps), eids over the table's four
    /// experts with every expert in the first row's picks, router weights in [0.05, 1.05].
    fn inputs(fx: *const Fixture, r: std.Random, rows: usize) void {
        _ = h.fill16(fx.bufs.x, rows * H, r, 0.1);
        const ids = fx.bufs.eids.slice(u32, rows * TOPK);
        for (ids) |*e| e.* = r.uintLessThan(u32, 4);
        for (ids[0..4], 0..) |*e, i| e.* = @intCast(i);
        for (fx.bufs.ew.slice(f32, rows * TOPK)) |*w| w.* = 0.05 + r.float(f32);
    }
};

// ---- cases ----

fn caseXdRef(g: *const h.Gpu, gpa: std.mem.Allocator, lut: []const f32, fx: *const Fixture, r: std.Random) !void {
    const rows = 2;
    const S = rows * TOPK;
    fx.inputs(r, rows);
    try fx.run(g, fx.mixed, 0, rows, .{ .dbg = true });
    const b = fx.bufs;
    const ids = b.eids.slice(u32, S);
    const ws = b.ew.slice(f32, S);
    const xs = b.x.slice(u16, rows * H);
    const gus = b.gu.slice(u16, S * H);
    const acts = b.act.slice(u16, S * I);
    const eys = b.ey.slice(u16, S * H);
    const moes = b.moe.slice(u16, rows * H);

    // the layer's descriptor table is a valid input of the chain; a K4 expert, another codebook, a bias or a tile
    // stride other than back to back is not
    {
        var c: h.Check = .{ .name = "xd chain eligibility" };
        var ds: [8]exl3.ExpertDescriptor = undefined;
        @memcpy(&ds, fx.mixed.descriptors());
        c.n += 1;
        if (!tf.inkling2.weights.xdOk(&ds, fx.mixed.count, H, I)) c.bad += 1;
        for ([_]usize{ 1, 2, 5, 7 }, 0..) |at, kind| {
            const kept = ds[at];
            switch (kind) {
                0 => ds[at].halves = 8,
                1 => ds[at].codebook = 0,
                2 => ds[at].has_bias = 1,
                else => ds[at].tile_k_stride += 2,
            }
            c.n += 1;
            if (tf.inkling2.weights.xdOk(&ds, fx.mixed.count, H, I)) c.bad += 1;
            ds[at] = kept;
        }
        try c.done();
    }

    // each stage against FP64 on the kernels' own upstream outputs
    var xf: [rows][H]f64 = undefined;
    for (&xf, 0..) |*row, t| for (row, 0..) |*v, i| {
        v.* = h.fb(xs[t * H + i]);
    };
    var gu_ref: [S][H]f64 = undefined;
    var ey_ref: [S][H]f64 = undefined;
    var act_in: [S][I]f64 = undefined;
    for (&act_in, 0..) |*row, s| for (row, 0..) |*v, i| {
        v.* = h.fb(acts[s * I + i]);
    };
    for (0..4) |e| {
        var in13: [S][]const f64 = undefined;
        var out13: [S][]f64 = undefined;
        var in2: [S][]const f64 = undefined;
        var out2: [S][]f64 = undefined;
        var n: usize = 0;
        for (0..S) |s| if (ids[s] == e) {
            in13[n] = &xf[s / TOPK];
            out13[n] = &gu_ref[s];
            in2[n] = &act_in[s];
            out2[n] = &ey_ref[s];
            n += 1;
        };
        if (n == 0) continue;
        const ex = &fx.experts[e];
        try project(gpa, lut, words32(ex.t13), ex.bits, H, 2 * I, ex.s13.slice(f16, H), ex.v13.slice(f16, 2 * I), in13[0..n], out13[0..n]);
        try project(gpa, lut, words32(ex.t2), ex.bits, I, H, ex.s2.slice(f16, I), ex.v2.slice(f16, H), in2[0..n], out2[0..n]);
    }

    var cg: h.Check = .{ .name = "xd gu against FP64" };
    var ca: h.Check = .{ .name = "xd act against FP64 on the kernel's gu" };
    var ce: h.Check = .{ .name = "xd per-pick ey against FP64 on the kernel's act" };
    var cm: h.Check = .{ .name = "xd moe against FP64 on the kernel's ey" };
    for (0..S) |s| {
        const mag_gu = rms(&gu_ref[s]);
        for (0..H) |i| near16(&cg, gus[s * H + i], gu_ref[s][i], 2e-3, mag_gu);
        for (0..I) |i| {
            const gg = h.fb(gus[s * H + 2 * i]);
            const u = h.fb(gus[s * H + 2 * i + 1]);
            ca.b16(acts[s * I + i], 4 * std.math.tanh(gg / 4) * (1 / (1 + @exp(-gg))) * (25 * std.math.tanh(u / 25)));
        }
        const mag_ey = rms(&ey_ref[s]);
        for (0..H) |i| near16(&ce, eys[s * H + i], ey_ref[s][i], 2e-3, mag_ey);
    }
    for (0..rows) |t| {
        var want: [H]f64 = undefined;
        for (&want, 0..) |*w, i| {
            w.* = 0;
            for (0..TOPK) |k| w.* += h.fb(eys[(t * TOPK + k) * H + i]) * ws[t * TOPK + k];
        }
        const mag = rms(&want);
        for (0..H) |i| cm.b16a(moes[t * H + i], want[i], 2, 1e-5 * mag);
    }
    try cg.done();
    try ca.done();
    try ce.done();
    try cm.done();

    // the one-kernel-per-op path's glue on the same values: SwiGLU of gu and the weighted sum of ey are the same bytes
    var cq: h.Check = .{ .name = "xd act and moe equal ik_swiglu and ik_combine on the same values" };
    const act2 = try g.buf(S * I * 2);
    defer act2.deinit();
    const moe2 = try g.buf(rows * H * 2);
    defer moe2.deinit();
    try g.run("ik_swiglu", .{ .rows = S, .width = I, .cap = 4, .thr = 25 }, &.{ .{ .buf = b.gu }, .{ .buf = act2 } }, .{ I, S, 1 }, .{ 256, 1, 1 }, false);
    try g.run("ik_combine", .{ .rows = rows, .width = H, .queries = TOPK }, &.{ .{ .buf = b.ey }, .{ .buf = b.ew }, .{ .buf = moe2 } }, .{ H, rows, 1 }, .{ 256, 1, 1 }, false);
    cq.same(act2.slice(u8, S * I * 2), b.act.slice(u8, S * I * 2));
    cq.same(moe2.slice(u8, rows * H * 2), b.moe.slice(u8, rows * H * 2));
    try cq.done();

    // the checks' kernels and the forward's are one body: the same act and moe bytes
    var cs: h.Check = .{ .name = "xd check kernels equal the forward's" };
    const act_dbg = try gpa.dupe(u8, b.act.slice(u8, S * I * 2));
    defer gpa.free(act_dbg);
    const moe_dbg = try gpa.dupe(u8, b.moe.slice(u8, rows * H * 2));
    defer gpa.free(moe_dbg);
    b.fillOutputs(0xab);
    try fx.run(g, fx.mixed, 0, rows, .{});
    cs.same(b.act.slice(u8, S * I * 2), act_dbg);
    cs.same(b.moe.slice(u8, rows * H * 2), moe_dbg);
    try cs.done();
}

/// The four outputs of rows 0..16, each row's bytes.
const Snap = struct {
    planes: []u8,
    act: []u8,
    sums: []u8,
    moe: []u8,

    const row_planes = TOPK * 2 * H * 4;
    const row_act = TOPK * I * 2;
    const row_sums = TOPK * H * 4;
    const row_moe = H * 2;

    fn take(gpa: std.mem.Allocator, b: Bufs) !Snap {
        return .{
            .planes = try gpa.dupe(u8, b.planes.slice(u8, MAX_ROWS * row_planes)),
            .act = try gpa.dupe(u8, b.act.slice(u8, MAX_ROWS * row_act)),
            .sums = try gpa.dupe(u8, b.sums.slice(u8, MAX_ROWS * row_sums)),
            .moe = try gpa.dupe(u8, b.moe.slice(u8, MAX_ROWS * row_moe)),
        };
    }

    fn deinit(s: Snap, gpa: std.mem.Allocator) void {
        gpa.free(s.planes);
        gpa.free(s.act);
        gpa.free(s.sums);
        gpa.free(s.moe);
    }

    /// Row `t` of the buffers equal to row `t` of the snapshot.
    fn rowSame(s: Snap, c: *h.Check, b: Bufs, t: usize) void {
        c.same(b.planes.slice(u8, MAX_ROWS * row_planes)[t * row_planes ..][0..row_planes], s.planes[t * row_planes ..][0..row_planes]);
        c.same(b.act.slice(u8, MAX_ROWS * row_act)[t * row_act ..][0..row_act], s.act[t * row_act ..][0..row_act]);
        c.same(b.sums.slice(u8, MAX_ROWS * row_sums)[t * row_sums ..][0..row_sums], s.sums[t * row_sums ..][0..row_sums]);
        c.same(b.moe.slice(u8, MAX_ROWS * row_moe)[t * row_moe ..][0..row_moe], s.moe[t * row_moe ..][0..row_moe]);
    }
};

fn allBytes(c: *h.Check, bytes: []const u8, byte: u8) void {
    c.n += 1;
    for (bytes) |x| if (x != byte) {
        c.bad += 1;
        return;
    };
}

fn caseXdWindow(g: *const h.Gpu, gpa: std.mem.Allocator, fx: *const Fixture, r: std.Random) !void {
    const b = fx.bufs;
    fx.inputs(r, MAX_ROWS);
    var c: h.Check = .{ .name = "xd windows of 1..16 rows against single rows" };
    // each row alone, into its own place in the buffers
    for (0..MAX_ROWS) |t| try fx.run(g, fx.mixed, t, 1, .{});
    const solo = try Snap.take(gpa, b);
    defer solo.deinit(gpa);
    for (1..MAX_ROWS + 1) |w| {
        b.fillOutputs(0xab);
        try fx.run(g, fx.mixed, 0, w, .{});
        for (0..w) |t| solo.rowSame(&c, b, t);
        // nothing past the window is written
        allBytes(&c, b.planes.slice(u8, MAX_ROWS * Snap.row_planes)[w * Snap.row_planes ..], 0xab);
        allBytes(&c, b.act.slice(u8, MAX_ROWS * Snap.row_act)[w * Snap.row_act ..], 0xab);
        allBytes(&c, b.sums.slice(u8, MAX_ROWS * Snap.row_sums)[w * Snap.row_sums ..], 0xab);
        allBytes(&c, b.moe.slice(u8, MAX_ROWS * Snap.row_moe)[w * Snap.row_moe ..], 0xab);
    }
    try c.done();

    // a non-finite input row stays in its own slots and row
    var cn: h.Check = .{ .name = "xd non-finite row contained" };
    const xs = b.x.slice(u16, MAX_ROWS * H);
    const kept = xs[H + 5];
    xs[H + 5] = nan_bf16;
    b.fillOutputs(0xab);
    try fx.run(g, fx.mixed, 0, MAX_ROWS, .{});
    for (0..MAX_ROWS) |t| if (t != 1) solo.rowSame(&cn, b, t);
    cn.n += 1;
    var nan_out = false;
    for (b.moe.slice(u16, MAX_ROWS * H)[H..][0..H]) |v| nan_out = nan_out or std.math.isNan(h.fb(v));
    if (!nan_out) cn.bad += 1;
    xs[H + 5] = kept;

    // one non-finite slot's planes reach only that slot's act and sums and its row's moe
    const poisoned = 37;
    b.fillOutputs(0xab);
    try fx.run(g, fx.mixed, 0, MAX_ROWS, .{});
    @memset(b.planes.slice(f32, MAX_ROWS * TOPK * 2 * H)[poisoned * 2 * H ..][0 .. 2 * H], std.math.nan(f32));
    try fx.run(g, fx.mixed, 0, MAX_ROWS, .{ .pair = false });
    const act = b.act.slice(u8, MAX_ROWS * Snap.row_act);
    const sums = b.sums.slice(u8, MAX_ROWS * Snap.row_sums);
    const slot_act = Snap.row_act / TOPK;
    const slot_sums = Snap.row_sums / TOPK;
    for (0..MAX_ROWS * TOPK) |s| {
        if (s == poisoned) continue;
        cn.same(act[s * slot_act ..][0..slot_act], solo.act[s * slot_act ..][0..slot_act]);
        cn.same(sums[s * slot_sums ..][0..slot_sums], solo.sums[s * slot_sums ..][0..slot_sums]);
    }
    for (0..MAX_ROWS) |t| {
        if (t == poisoned / TOPK) continue;
        cn.same(b.moe.slice(u8, MAX_ROWS * Snap.row_moe)[t * Snap.row_moe ..][0..Snap.row_moe], solo.moe[t * Snap.row_moe ..][0..Snap.row_moe]);
    }
    cn.n += 1;
    if (!std.math.isNan(h.fb(b.act.slice(u16, MAX_ROWS * TOPK * I)[poisoned * I]))) cn.bad += 1;
    try cn.done();
}

/// A K3 expert's slots equal bytes in a layer of only K3 experts and in a layer that also has K2 experts.
fn caseXdMixed(g: *const h.Gpu, gpa: std.mem.Allocator, fx: *const Fixture, r: std.Random) !void {
    const rows = 4;
    const b = fx.bufs;
    fx.inputs(r, rows);
    const ids = b.eids.slice(u32, rows * TOPK);
    for (ids[0..TOPK], 0..) |*e, i| e.* = if (i % 2 == 0) 0 else 2; // row 0 picks only the experts both layers share
    b.fillOutputs(0xab);
    try fx.run(g, fx.pure, 0, rows, .{});
    const pure = try Snap.take(gpa, b);
    defer pure.deinit(gpa);
    b.fillOutputs(0xab);
    try fx.run(g, fx.mixed, 0, rows, .{});
    var c: h.Check = .{ .name = "xd K3 slots equal in mixed and K3-only layers" };
    const slot_planes = Snap.row_planes / TOPK;
    const slot_act = Snap.row_act / TOPK;
    const slot_sums = Snap.row_sums / TOPK;
    var shared: usize = 0;
    for (0..rows * TOPK) |s| {
        if (ids[s] != 0 and ids[s] != 2) continue;
        shared += 1;
        c.same(b.planes.slice(u8, rows * Snap.row_planes)[s * slot_planes ..][0..slot_planes], pure.planes[s * slot_planes ..][0..slot_planes]);
        c.same(b.act.slice(u8, rows * Snap.row_act)[s * slot_act ..][0..slot_act], pure.act[s * slot_act ..][0..slot_act]);
        c.same(b.sums.slice(u8, rows * Snap.row_sums)[s * slot_sums ..][0..slot_sums], pure.sums[s * slot_sums ..][0..slot_sums]);
    }
    c.same(b.moe.slice(u8, Snap.row_moe), pure.moe[0..Snap.row_moe]);
    c.n += 1;
    if (shared <= TOPK) c.bad += 1; // the other rows must share some too
    try c.done();
}
