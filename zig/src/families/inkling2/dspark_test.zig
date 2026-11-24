//! The draft head's drafter on this Mac's GPU, over a model built by hand: the real hidden width and head shapes (the kernels' own)
//! at a vocabulary of a few thousand ids, random weights that every draft layer shares but for its norms, and random taps. The
//! checks are the drafter's own bookkeeping and the fold: the fold against a host sum, a context absorbed in any chunking or
//! with a prompt's old rows skipped holding the same block, and core's depth sweep (a hold at every depth holds the first drafts
//! of a block held at the deepest).
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const ks = @import("kernel_sources");
const ck = @import("../../core/checkpoint_metal.zig");
const config = @import("config.zig");
const config_test = @import("config_test.zig");
const dw = @import("dspark_weights.zig");
const mix = @import("dspark_alpha.zig");
const wts = @import("weights.zig");
const Model = @import("model.zig").Model;
const glue_names = @import("model.zig").glue_names;
const dspark = @import("dspark.zig");
const backend = @import("backend.zig");
const fwd = @import("forward.zig");
const budget = @import("budget.zig");
const state = @import("state.zig");

const t = std.testing;
const gpa = t.allocator;
const sweep = lanes.drafter_depth_sweep;
const be = lanes.backend;
const dr = lanes.drafter;

const hidden = 4608;
const vocab = 4096;
const valid = 4000;
const row_bytes = 8 * hidden * 2;

/// `n` bf16 values of the buffer from `at`, uniform in ±scale: a random block of up to 2^20 values, tiled.
fn fill(b: mtl.Buffer, at: usize, n: usize, scale: f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const s = b.slice(u16, (at + n));
    const block: usize = @min(n, 1 << 20);
    for (s[at..][0..block]) |*x| x.* = mix.bf16Of((r.float(f32) * 2 - 1) * scale);
    var done = block;
    while (done < n) : (done += block) @memcpy(s[at + done ..][0..@min(block, n - done)], s[at..][0..@min(block, n - done)]);
}

const Fixture = struct {
    device: mtl.Device,
    model: *Model,
    pipes: [glue_names.len]mtl.Pipeline,
    buffers: std.ArrayList(mtl.Buffer) = .empty,
    seed: u64 = 1,
    bits: dw.Bits = .sixteen, // the width of the layers' projections

    /// A tensor of `dims` at 8 bytes into its own buffer (as a checkpoint's tensors sit), random in ±scale.
    fn tensor(f: *Fixture, dims: []const usize, scale: f32) !wts.Tensor {
        var n: usize = 1;
        for (dims) |d| n *= d;
        const buf = try f.device.buffer(n * 2 + 16, mtl.ResourceOptions.shared);
        try f.buffers.append(gpa, buf);
        f.seed += 1;
        fill(buf, 4, n, scale, f.seed);
        var out: wts.Tensor = .{ .buffer = buf, .offset = 8, .bytes = n * 2, .dtype = .bf16, .rank = dims.len };
        for (dims, 0..) |d, i| out.shape[i] = d;
        return out;
    }

    fn norm(f: *Fixture, n: usize) !wts.Tensor {
        const x = try f.tensor(&.{n}, 0.2);
        for (x.buffer.slice(u16, n + 8)[4..][0..n]) |*v| v.* = mix.bf16Of(1.0 + mix.f32Of(v.*));
        return x;
    }

    fn dense(f: *Fixture, n: usize, k: usize, scale: f32) !wts.Dense {
        return .{ .t = try f.tensor(&.{ n, k }, scale), .n = @intCast(n), .k = @intCast(k) };
    }

    /// A draft layer's projection: the random bf16 matrix, or at 8 bits its quantization (the buffer joins `buffers`).
    fn proj(f: *Fixture, n: usize, k: usize, scale: f32) !dw.Proj {
        const d = try f.dense(n, k, scale);
        if (f.bits == .sixteen) return .{ .bf16 = d };
        const q = try dw.quantize(gpa, f.device, d.t.host(u16), n, k);
        errdefer q.buf.deinit();
        try f.buffers.append(gpa, q.buf);
        return .{ .q8 = q };
    }

    fn init(a: std.mem.Allocator) !*Fixture {
        return make(a, .sixteen);
    }

    fn make(a: std.mem.Allocator, bits: dw.Bits) !*Fixture {
        return makeFor(a, config_test.real, bits);
    }

    fn makeFor(a: std.mem.Allocator, config_json: []const u8, bits: dw.Bits) !*Fixture {
        const device = try mtl.Device.init();
        errdefer device.deinit();
        const f = try gpa.create(Fixture);
        errdefer gpa.destroy(f);
        f.* = .{ .device = device, .model = try gpa.create(Model), .pipes = undefined, .bits = bits };
        errdefer gpa.destroy(f.model);
        errdefer f.free();
        var why: config.Why = .{};
        var cfg = try config.parse(a, config_json, &why);
        cfg.text.vocab = vocab;
        cfg.text.unpadded_vocab = valid;
        cfg.mtp.?.vocab = vocab;
        const m = cfg.mtp.?;
        const h: usize = m.hidden;
        const lib = try mtl.Library.fromSource(device, ks.inkling2_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        defer lib.deinit();
        var made: usize = 0;
        errdefer for (f.pipes[0..made]) |p| p.deinit();
        for (glue_names, 0..) |name, i| {
            f.pipes[i] = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        const qw = m.q_heads * m.head_dim;
        const kvw = m.kv_heads * m.head_dim;
        const shared: dw.Layer = .{
            .dflare = try f.norm(h),
            .input = try f.norm(h),
            .post = try f.norm(h),
            .q_norm = try f.norm(m.head_dim),
            .k_norm = try f.norm(m.head_dim),
            .k_norm_ctx = try f.norm(m.head_dim),
            .q = try f.proj(qw, h, 0.02),
            .k = try f.proj(kvw, h, 0.02),
            .v = try f.proj(kvw, h, 0.02),
            .k_ctx = try f.proj(kvw, h, 0.02),
            .v_ctx = try f.proj(kvw, h, 0.02),
            .o = try f.proj(h, qw, 0.02),
            .gate = try f.proj(m.inter, h, 0.02),
            .up = try f.proj(m.inter, h, 0.02),
            .down = try f.proj(h, m.inter, 0.02),
        };
        const layers = try gpa.alloc(dw.Layer, m.layers);
        errdefer gpa.free(layers);
        for (layers) |*l| {
            l.* = shared;
            l.dflare = try f.norm(h);
            l.input = try f.norm(h);
            l.post = try f.norm(h);
        }
        const rows_n: usize = @as(usize, m.layers) + 1;
        const alpha_buffer = try device.buffer(rows_n * m.taps_n * 2, mtl.ResourceOptions.shared);
        errdefer alpha_buffer.deinit();
        const dflare_norms = try device.buffer(rows_n * h * 2, mtl.ResourceOptions.shared);
        errdefer dflare_norms.deinit();
        const markov_dflare = try f.norm(h);
        const packed_norms = dflare_norms.slice(u16, rows_n * h);
        for (layers, 0..) |l, i| @memcpy(packed_norms[i * h ..][0..h], l.dflare.host(u16));
        @memcpy(packed_norms[layers.len * h ..][0..h], markov_dflare.host(u16));
        var w: dw.Weights = .{
            .layers = layers,
            .bits = bits,
            .alpha = std.mem.zeroes([mix.max_rows][mix.max_taps]u16),
            .alpha_buffer = alpha_buffer,
            .dflare_norms = dflare_norms,
            .rows = m.layers + 1,
            .taps = m.taps_n,
            .mask_embed = try f.tensor(&.{h}, 1.0),
            .norm = try f.norm(h),
            .markov = .{
                .dflare = markov_dflare,
                .w1 = try f.dense(vocab, m.rank, 1.0),
                .rnn = try f.dense(3 * m.rank, 2 * m.rank + 2 * h, 0.05),
                .rnn_b = try f.tensor(&.{3 * m.rank}, 0.5),
                .hidden_out = try f.dense(h, m.rank, 0.05),
                .hidden_out_b = try f.tensor(&.{h}, 0.5),
            },
        };
        errdefer w.deinit(gpa);
        var fuse: [(mix.max_rows) * mix.max_taps]u16 = undefined;
        var prng = std.Random.DefaultPrng.init(0xf05e);
        for (fuse[0 .. w.rows * w.taps]) |*x| x.* = mix.bf16Of((prng.random().float(f32) * 2 - 1) * 2);
        mix.softmaxRows(fuse[0 .. w.rows * w.taps], w.rows, w.taps, &w.alpha);
        const packed_alpha = w.alpha_buffer.slice(u16, w.rows * w.taps);
        for (0..w.rows) |r| @memcpy(packed_alpha[r * w.taps ..][0..w.taps], w.alpha[r][0..w.taps]);
        f.model.* = .{
            .gpa = gpa,
            .device = device,
            .queue = try device.queue(),
            .config = cfg,
            .plan = undefined,
            .layers = 0,
            .checkpoint = ck.Checkpoint.init(gpa),
            .weights = .{
                .embed = try f.dense(vocab, h, 0.5),
                .unembed = try f.dense(vocab, h, 0.5),
                .embed_norm = try f.norm(h),
                .norm = try f.norm(h),
                .out_query = try f.tensor(&.{h}, 1.0),
                .layers = &.{},
                .tables = &.{},
                .expert_buffers = &.{},
            },
            .glue = f.pipes,
            .exl3 = null,
            .chunk = null, // no chunk kernels: prompt chunks take the per-operation path
            .draft = w,
            .warm_wait = true,
            .first_chunk = 0,
            .prefill_rows = state.default_prefill_rows,
        };
        return f;
    }

    fn free(f: *Fixture) void {
        for (f.buffers.items) |b| b.deinit();
        f.buffers.deinit(gpa);
    }

    fn deinit(f: *Fixture) void {
        f.model.draft.?.deinit(gpa);
        f.model.checkpoint.deinit();
        f.model.queue.deinit();
        for (f.pipes) |p| p.deinit();
        f.free();
        gpa.destroy(f.model);
        f.device.deinit();
        gpa.destroy(f);
    }

    fn drafter(f: *Fixture, depth: u32) !*dspark.Drafter {
        return f.drafterWith(.{ .depth = depth });
    }

    fn drafterWith(f: *Fixture, options: dspark.Options) !*dspark.Drafter {
        return dspark.Drafter.init(gpa, f.device, f.model, &f.model.draft.?, options);
    }
};

/// `rows` rows of taps, random bf16 of the scale hidden states have.
fn tapsBufferFor(device: mtl.Device, rows: usize, rb: usize, seed: u64) !mtl.Buffer {
    const b = try device.buffer(rows * rb, mtl.ResourceOptions.shared);
    fill(b, 0, rows * rb / 2, 3.0, seed);
    return b;
}

fn tapsBuffer(device: mtl.Device, rows: usize, seed: u64) !mtl.Buffer {
    return tapsBufferFor(device, rows, row_bytes, seed);
}

fn featuresFor(taps: mtl.Buffer, start: usize, n: usize, rb: usize) be.Features {
    return .{ .buffer = @intFromPtr(taps.id), .offset = start * rb, .rows = @intCast(n), .row_bytes = @intCast(rb), .space = .device, .dtype = .bf16 };
}

fn features(taps: mtl.Buffer, start: usize, n: usize) be.Features {
    return featuresFor(taps, start, n, row_bytes);
}

fn tokenAt(i: u64) u32 {
    return @intCast((i * 13 + 7) % 997 + 5);
}

test "inkling2 dspark: the fold is the unembedding applied to hidden_out, each sum rounded to bf16 once" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.init(arena.allocator()) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const d = try f.drafter(5);
    defer d.deinit();
    const w = &f.model.draft.?;
    const fw = w.fold_w.?;
    const fb = w.fold_b.?;
    try t.expectEqual(@as(u32, vocab), fw.n);
    try t.expectEqual(@as(u32, 256), fw.k);
    const unembed = f.model.weights.unembed.t.host(u16);
    const ho = w.markov.hidden_out.t.host(u16);
    const bias = w.markov.hidden_out_b.host(u16);
    const got_w = fw.t.host(u16);
    const got_b = fb.host(u16);
    var worst: u32 = 0;
    for ([_]usize{ 0, 1, 7, 255, 256, 1000, 2047, 3999, 4000, vocab - 1 }) |v| {
        for (0..257) |c| {
            var sum: f64 = 0;
            for (0..hidden) |h| {
                const u = mix.f32Of(unembed[v * hidden + h]);
                const x = if (c < 256) mix.f32Of(ho[h * 256 + c]) else mix.f32Of(bias[h]);
                sum += @as(f64, u) * x;
            }
            const want = mix.bf16Of(@floatCast(sum));
            const got = if (c < 256) got_w[v * 256 + c] else got_b[v];
            const key = struct {
                fn of(x: u16) i32 {
                    return if (x & 0x8000 != 0) -@as(i32, x & 0x7fff) else @as(i32, x);
                }
            };
            worst = @max(worst, @abs(key.of(got) - key.of(want)));
        }
    }
    // an f32 sum and an f64 one differ at most by a rounding that sits between two bf16 values
    try t.expect(worst <= 1);
    // the checkpoint holds the folded tensors and counts their bytes
    try t.expect(f.model.checkpoint.has("markov_head.fold_w") and f.model.checkpoint.has("markov_head.fold_b"));
    try t.expectEqual(dw.foldBytes(vocab, 256), f.model.checkpoint.residentBytes());
    // folding twice keeps the first (the pipeline is not used)
    try dw.fold(w, &f.model.checkpoint, f.device, f.model.queue, f.model, f.pipes[0]);
    try t.expectEqual(dw.foldBytes(vocab, 256), f.model.checkpoint.residentBytes());
}

/// One hold of `depth` over the context `taps` as `chunks` absorb it (rows [0, n) in order, `chunk` rows a call, as a prompt's
/// when `prompt`), the proposals and the steps' logits.
fn held(d: *dspark.Drafter, s: *lanes.Stream, taps: mtl.Buffer, n: usize, chunk: usize, prompt: bool, depth: u32, out: []u32, logits: []u16) !void {
    const dv = d.drafter();
    const rb = d.tapRowBytes();
    try dv.open(s);
    defer dv.release(s);
    var at: usize = 0;
    while (at < n) : (at += chunk) {
        const rows = @min(chunk, n - at);
        try dv.absorb(&.{.{ .stream = s, .features = featuresFor(taps, at, rows, rb), .start = at, .follow = &.{}, .prompt = prompt }});
    }
    try dv.hold(&.{.{ .stream = s, .pending = tokenAt(n), .position = n + 1, .depth = depth }});
    try dv.held(&.{s}, &.{out[0..depth]});
    @memcpy(logits, d.captured());
}

/// A context absorbed in any chunking, or with a long prompt's old rows skipped, holds the same block, over a fixture whose layers
/// are `bits` wide: a row's bits through the projections do not depend on the rows beside it.
fn chunkingHolds(bits: dw.Bits) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.make(arena.allocator(), bits) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const d = try f.drafter(7);
    defer d.deinit();
    try d.captureLogits();
    const n = 1500; // past a window: the ring has wrapped, and a prompt keeps its last 1,023 rows
    const taps = try tapsBuffer(f.device, n + 1, 7);
    defer taps.deinit();
    const prompt = try gpa.alloc(u32, n);
    defer gpa.free(prompt);
    for (prompt, 0..) |*x, i| x.* = tokenAt(i);
    var s = try lanes.Stream.init(gpa, .{ .id = "chunks", .prompt = prompt, .max_new = 8 });
    defer s.deinit(gpa);
    const a_logits = try gpa.alloc(u16, 7 * vocab);
    defer gpa.free(a_logits);
    const b_logits = try gpa.alloc(u16, 7 * vocab);
    defer gpa.free(b_logits);
    var a_props: [7]u32 = undefined;
    var b_props: [7]u32 = undefined;
    // the whole context in one call (three passes inside), then 37 rows a call, then as a prompt's 512-row chunks: that pass
    // skips rows 0 to 476, outside every window the hold attends; and as one prompt chunk of the whole 1,500 rows (past the
    // drafter's 512-row passes, as a larger prefill chunk is), which keeps the last 1,023 of them in two passes
    try held(d, &s, taps, n, n, false, 7, &a_props, a_logits);
    for (a_props) |x| try t.expect(x < valid);
    for ([_]struct { chunk: usize, prompt: bool }{ .{ .chunk = 37, .prompt = false }, .{ .chunk = 512, .prompt = true }, .{ .chunk = n, .prompt = true } }) |c| {
        try held(d, &s, taps, n, c.chunk, c.prompt, 7, &b_props, b_logits);
        try t.expectEqualSlices(u32, &a_props, &b_props);
        try t.expectEqualSlices(u16, a_logits, b_logits);
    }
}

test "inkling2 dspark: a context absorbed in any chunking, or with a long prompt's old rows skipped, holds the same block" {
    try chunkingHolds(.sixteen);
}

test "inkling2 dspark: with 8-bit layers a context absorbed in any chunking, or with a long prompt's old rows skipped, holds the same block" {
    try chunkingHolds(.eight);
}

/// The relative RMS difference of two bf16 tables of the seven steps' logits over the ids the vocabulary keeps (the padded ids are
/// -inf in both).
fn logitsDiff(a: []const u16, b: []const u16) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (0..7) |step| for (0..valid) |v| {
        const x = mix.f32Of(a[step * vocab + v]);
        const y = mix.f32Of(b[step * vocab + v]);
        num += (x - y) * (x - y);
        den += y * y;
    };
    return @sqrt(num / den);
}

test "inkling2 dspark: 8-bit layers hold logits close to the bf16 layers' on the same weights and taps" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var logits: [2]?[]u16 = .{ null, null };
    defer for (logits) |l| if (l) |x| gpa.free(x);
    for ([_]dw.Bits{ .sixteen, .eight }, 0..) |bits, i| {
        // the same seeds a fixture, so the 8-bit layers are the quantization of the bf16 ones
        const f = Fixture.make(arena.allocator(), bits) catch |err| switch (err) {
            error.NoDevice => return error.SkipZigTest,
            else => return err,
        };
        defer f.deinit();
        const d = try f.drafter(7);
        defer d.deinit();
        try d.captureLogits();
        const n = 300;
        const taps = try tapsBuffer(f.device, n + 1, 7);
        defer taps.deinit();
        var prompt: [n]u32 = undefined;
        for (&prompt, 0..) |*x, j| x.* = tokenAt(j);
        var s = try lanes.Stream.init(gpa, .{ .id = "bits", .prompt = &prompt, .max_new = 8 });
        defer s.deinit(gpa);
        const out = try gpa.alloc(u16, 7 * vocab);
        logits[i] = out;
        var props: [7]u32 = undefined;
        try held(d, &s, taps, n, 64, true, 7, &props, out);
    }
    const diff = logitsDiff(logits[1].?, logits[0].?);
    // a kernel that misapplies a scale or a bias is off by the logits' size; the quantization's own noise is under 2% of their RMS
    // here (1.7% measured on this fixture), and not zero
    try t.expect(diff > 0 and diff < 0.05);
}

/// Core's depth sweep over the drafter, a second one held at the deepest as its reference.
const Sweep = struct {
    taps: mtl.Buffer,
    a: *dspark.Drafter,
    b: *dspark.Drafter,
    stream: lanes.Stream = undefined,
    prompt: []u32 = &.{},

    fn harness(h: *Sweep) sweep.Harness {
        return .{ .ctx = h, .begin = begin, .verify = verify, .end = end };
    }

    fn self(ctx: *anyopaque) *Sweep {
        return @ptrCast(@alignCast(ctx));
    }

    fn rows(h: *Sweep, start: u64, n: u64, prompt: bool) dr.Absorb {
        return .{ .stream = &h.stream, .features = featuresFor(h.taps, @intCast(start), @intCast(n), h.a.tapRowBytes()), .start = start, .follow = &.{}, .prompt = prompt };
    }

    fn begin(ctx: *anyopaque, n: usize) anyerror!sweep.Case {
        const h = self(ctx);
        h.prompt = try gpa.alloc(u32, n);
        for (h.prompt, 0..) |*x, i| x.* = tokenAt(i);
        h.stream = try lanes.Stream.init(gpa, .{ .id = "sweep", .prompt = h.prompt, .max_new = 64 });
        const first: sweep.Step = .{ .absorb = h.rows(0, n, true), .next = .{ .pending = tokenAt(n), .position = n + 1 } };
        return .{ .drafter = h.a.drafter(), .stream = &h.stream, .prompt = first, .deeper = h.b.drafter() };
    }

    /// The target keeps the pending row and a position-keyed count of the drafts.
    fn verify(ctx: *anyopaque, r: sweep.Round, drafts: []const u32) anyerror!sweep.Step {
        const h = self(ctx);
        const k = (r.position *% 2654435761) % (drafts.len + 1);
        return .{ .absorb = h.rows(r.position - 1, 1 + k, false), .next = .{ .pending = tokenAt(r.position + k), .position = r.position + 1 + k } };
    }

    fn end(ctx: *anyopaque) void {
        const h = self(ctx);
        h.stream.deinit(gpa);
        gpa.free(h.prompt);
    }
};

/// A hold at every depth 1 to 7 holds exactly the asked drafts, the first of the deepest block's, over a fixture of `bits`.
fn depthSweep(bits: dw.Bits) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.make(arena.allocator(), bits) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const a = try f.drafter(5);
    defer a.deinit();
    const b = try f.drafter(7);
    defer b.deinit();
    const taps = try tapsBuffer(f.device, 400, 11);
    defer taps.deinit();
    var h: Sweep = .{ .taps = taps, .a = a, .b = b };
    const r = try sweep.sweep(gpa, h.harness(), .{ .block = 7, .prompts = &.{ 6, 40 }, .rounds = 12 });
    // two prompts x (depths 1..7 + one zig-zag run) x 12 rounds; drafts 12 x 28 fixed + 48 zig-zag a prompt
    try t.expectEqual(sweep.Report{ .runs = 16, .rounds = 192, .drafts = 2 * (12 * 28 + 48) }, r);
}

test "inkling2 dspark: a hold at every depth 1 to 7 holds exactly the asked drafts, the first of the deepest block's" {
    try depthSweep(.sixteen);
}

test "inkling2 dspark: with 8-bit layers a hold at every depth 1 to 7 holds exactly the asked drafts, the first of the deepest block's" {
    try depthSweep(.eight);
}

test "inkling2 dspark: facts offer the asked depth, a fixed block, the lane's bytes and the taps the target writes" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.init(arena.allocator()) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const d = try f.drafter(3);
    defer d.deinit();
    const dv = d.drafter();
    const facts = dv.facts();
    try t.expectEqual(@as(u32, 3), facts.depth);
    try t.expect(facts.fixed_block and facts.plain_guard and facts.step_ms > 0);
    try t.expectEqual(@as(usize, 2 * 5 * (1031 + 7) * 1024 * 2 + 4608 * 2 + 256 * 2 + 8 * 4 + 8 * 4), facts.stream_bytes);
    try t.expectEqualSlices(u32, &.{ 5, 11, 17, 23, 29, 35, 41, 48 }, dv.taps());
    try t.expectEqual(@as(usize, 7), facts.prior.len);
    // depth 0 and 8 are not drafts of a block of eight rows
    try t.expectError(error.DraftDepth, f.drafter(0));
    try t.expectError(error.DraftDepth, f.drafter(8));
    // a hold needs the context to end just before its position, and an opened stream
    var s = try lanes.Stream.init(gpa, .{ .id = "facts", .prompt = &.{ 5, 6, 7 }, .max_new = 1 });
    defer s.deinit(gpa);
    try t.expectError(error.DraftStreamUnknown, dv.hold(&.{.{ .stream = &s, .pending = 9, .position = 4, .depth = 1 }}));
    try dv.open(&s);
    defer dv.release(&s);
    try t.expectError(error.DraftStreamOpen, dv.open(&s));
    try t.expectError(error.DraftContextGap, dv.hold(&.{.{ .stream = &s, .pending = 9, .position = 4, .depth = 1 }}));
    const taps = try tapsBuffer(f.device, 3, 5);
    defer taps.deinit();
    try dv.absorb(&.{.{ .stream = &s, .features = features(taps, 0, 3), .start = 0, .follow = &.{}, .prompt = true }});
    try t.expectError(error.DraftContextGap, dv.hold(&.{.{ .stream = &s, .pending = 9, .position = 5, .depth = 1 }}));
    try t.expectError(error.DraftDepth, dv.hold(&.{.{ .stream = &s, .pending = 9, .position = 4, .depth = 8 }}));
    try t.expectError(error.InvalidDraftToken, dv.hold(&.{.{ .stream = &s, .pending = vocab, .position = 4, .depth = 1 }}));
    var out: [2]u32 = undefined;
    try t.expectError(error.DraftHeldFewer, dv.held(&.{&s}, &.{out[0..1]}));
    try dv.hold(&.{.{ .stream = &s, .pending = 9, .position = 4, .depth = 2 }});
    try dv.held(&.{&s}, &.{out[0..2]});
    try t.expect(out[0] < valid and out[1] < valid);
    // an absorb drops what was held
    try dv.absorb(&.{.{ .stream = &s, .features = features(taps, 2, 1), .start = 2, .follow = &.{}, .prompt = false }});
    try t.expectError(error.DraftHeldFewer, dv.held(&.{&s}, &.{out[0..1]}));
}

/// Two streams absorbed and held in one round each hold what they hold alone, over a fixture of `bits`.
fn twoStreams(bits: dw.Bits) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.make(arena.allocator(), bits) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const d = try f.drafter(7);
    defer d.deinit();
    try d.captureLogits();
    const dv = d.drafter();
    const ns = [2]usize{ 300, 1200 };
    var taps: [2]mtl.Buffer = undefined;
    var prompts: [2][]u32 = undefined;
    var streams: [2]lanes.Stream = undefined;
    for (0..2) |i| {
        taps[i] = try tapsBuffer(f.device, ns[i] + 1, 21 + i);
        prompts[i] = try gpa.alloc(u32, ns[i]);
        for (prompts[i], 0..) |*x, j| x.* = tokenAt(j + 5 * i);
        streams[i] = try lanes.Stream.init(gpa, .{ .id = "pair", .prompt = prompts[i], .max_new = 4 });
    }
    defer for (0..2) |i| {
        taps[i].deinit();
        gpa.free(prompts[i]);
        streams[i].deinit(gpa);
    };
    const logits = try gpa.alloc(u16, 7 * vocab);
    defer gpa.free(logits);
    var alone: [2][7]u32 = undefined;
    var last: [7 * vocab]u16 = undefined;
    for (0..2) |i| try held(d, &streams[i], taps[i], ns[i], 512, true, 7, &alone[i], logits);
    @memcpy(&last, logits);
    // both in one round
    for (&streams) |*s| try dv.open(s);
    defer for (&streams) |*s| dv.release(s);
    try dv.absorb(&.{
        .{ .stream = &streams[0], .features = features(taps[0], 0, ns[0]), .start = 0, .follow = &.{}, .prompt = true },
        .{ .stream = &streams[1], .features = features(taps[1], 0, ns[1]), .start = 0, .follow = &.{}, .prompt = true },
    });
    try dv.hold(&.{
        .{ .stream = &streams[0], .pending = tokenAt(ns[0]), .position = ns[0] + 1, .depth = 7 },
        .{ .stream = &streams[1], .pending = tokenAt(ns[1]), .position = ns[1] + 1, .depth = 4 },
    });
    var both: [2][7]u32 = undefined;
    try dv.held(&.{ &streams[0], &streams[1] }, &.{ both[0][0..7], both[1][0..4] });
    try t.expectEqualSlices(u32, &alone[0], &both[0]);
    try t.expectEqualSlices(u32, alone[1][0..4], both[1][0..4]);
    // the logits kept are the last hold's of the round, its four steps
    try t.expectEqualSlices(u16, last[0 .. 4 * vocab], d.captured()[0 .. 4 * vocab]);
}

test "inkling2 dspark: two streams absorbed and held in one round each hold what they hold alone" {
    try twoStreams(.sixteen);
}

test "inkling2 dspark: with 8-bit layers two streams absorbed and held in one round each hold what they hold alone" {
    try twoStreams(.eight);
}

/// A residency hook that counts the buffers joined and left, and their bytes.
const Counter = struct {
    joined: usize = 0,
    left: usize = 0,
    joined_bytes: usize = 0,
    left_bytes: usize = 0,

    fn join(ctx: *anyopaque, buf: mtl.Buffer) void {
        const c: *Counter = @ptrCast(@alignCast(ctx));
        c.joined += 1;
        c.joined_bytes += buf.length();
    }

    fn leave(ctx: *anyopaque, buf: mtl.Buffer) void {
        const c: *Counter = @ptrCast(@alignCast(ctx));
        c.left += 1;
        c.left_bytes += buf.length();
    }

    fn follow(c: *Counter) mtl.ResidencySet.Follow {
        return .{ .ctx = c, .join = join, .leave = leave };
    }
};

test "inkling2 dspark: a residency hook gets the scratch, each lane's buffers as lanes open and close, and all of them back" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.init(arena.allocator()) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const d = try f.drafter(5);
    defer d.deinit();
    var c: Counter = .{};
    d.setResident(c.follow());
    const scratch = c.joined;
    try t.expect(scratch >= 24);
    const dv = d.drafter();
    var s = try lanes.Stream.init(gpa, .{ .id = "resident", .prompt = &.{ 5, 6 }, .max_new = 1 });
    defer s.deinit(gpa);
    try dv.open(&s);
    try t.expectEqual(scratch + 14, c.joined); // ten rings, the Markov feature, the state, the token row and the probabilities
    try d.captureLogits();
    try t.expectEqual(scratch + 15, c.joined);
    dv.release(&s);
    try t.expectEqual(@as(usize, 14), c.left);
    d.setResident(null);
    try t.expectEqual(c.joined, c.left);
}

test "inkling2 dspark: the bytes the load counts for the head are the buffers the head makes" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.init(arena.allocator()) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const w = &f.model.draft.?;
    const d = try f.drafter(5);
    defer d.deinit();
    // the fold joined the checkpoint's shards when the head was made
    const fold = dw.foldBytes(vocab, 256);
    try t.expectEqual(fold, w.fold_w.?.t.bytes + w.fold_b.?.bytes);
    try t.expectEqual(fold, f.model.checkpoint.residentBytes());
    // a residency hook is given the scratch, then a stream's buffers: their bytes are the ones counted
    var c: Counter = .{};
    d.setResident(c.follow());
    try t.expectEqual(d.scratchBytes(), c.joined_bytes);
    var s = try lanes.Stream.init(gpa, .{ .id = "bytes", .prompt = &.{ 5, 6 }, .max_new = 1 });
    defer s.deinit(gpa);
    const dv = d.drafter();
    try dv.open(&s);
    try t.expectEqual(d.laneBytes(), c.joined_bytes - d.scratchBytes());
    try t.expectEqual(dv.facts().stream_bytes, d.laneBytes());
    try t.expectEqual(d.scratchBytes() + d.laneBytes(), d.extraBytes(1));
    try t.expectEqual(d.scratchBytes() + 3 * d.laneBytes(), d.extraBytes(3));
    // the whole head: the draft file's shard (as the loader records it), the fold, the scratch and the streams
    w.bytes = 12345;
    try t.expectEqual(12345 + fold + d.extraBytes(2), d.residentBytes(2));
    dv.release(&s);
    d.setResident(null);
    try t.expectEqual(c.joined_bytes, c.left_bytes);
}

test "inkling2 dspark: the load's budget takes the head's file, fold, scratch and stream off the caches' room" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.init(arena.allocator()) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const w = &f.model.draft.?;
    // a load without the head: the backbone's shards and the scratch
    const dense = 5 << 20;
    try f.model.checkpoint.own(try f.device.buffer(dense, mtl.ResourceOptions.shared), dense);
    const plain = try budget.held(f.model, 0);
    try t.expectEqual(dense + try state.scratchBytes(&f.model.config.text, 0, f.model.chunking()), plain);
    // with drafts: the draft file joins the shards (as `Model.load` leaves it), then the head is made and its fold joins them
    const file = 3 << 20;
    try f.model.checkpoint.own(try f.device.buffer(file, mtl.ResourceOptions.shared), file);
    w.bytes = file;
    const d = try f.drafter(5);
    defer d.deinit();
    const drafted = try budget.held(f.model, d.extraBytes(1));
    try t.expectEqual(plain + d.residentBytes(1), drafted);
    // under a 236 GB limit on a 256 GiB Mac the caches' room falls by the head's resident bytes, no more and no less
    const gib: u64 = 1 << 30;
    const base = 186 * gib; // the rest of the backbone's shards
    const room_plain = try budget.caches(256 * gib, 240 * gib, 236, base + plain);
    const room_drafted = try budget.caches(256 * gib, 240 * gib, 236, base + drafted);
    try t.expect(room_plain > 20 * gib);
    try t.expectEqual(d.residentBytes(1), room_plain - room_drafted);
}

test "inkling2 prefill rows: the model's setting sizes the scratch and the load's budget, and a chunk past the scratch's rows is refused" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.init(arena.allocator()) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const c = &f.model.config.text;
    var cache = try state.Cache.init(f.device, c, 256, c.layers, state.default_prefill_rows);
    defer cache.deinit();
    var ids: [129]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(5 + i);
    // a scratch of 64 rows takes a chunk of 64 and refuses one of 65; a scratch of 128 takes 128 and refuses 129
    const prompt: fwd.Out = .{ .logits = .none, .prompt = true };
    var few = try state.Scratch.init(f.device, c, 256, .{ .rows = 64, .path = f.model.promptPath() });
    defer few.deinit();
    try fwd.run(f.model, &few, &cache, ids[0..64], prompt);
    try t.expectError(error.Inkling2RowsExceeded, fwd.run(f.model, &few, &cache, ids[0..65], prompt));
    var more = try state.Scratch.init(f.device, c, 256, .{ .rows = 128, .path = f.model.promptPath() });
    defer more.deinit();
    try fwd.run(f.model, &more, &cache, ids[0..128], prompt);
    try t.expectError(error.Inkling2RowsExceeded, fwd.run(f.model, &more, &cache, ids[0..129], prompt));
    // the model's rows size the backend's scratch, and the load's budget counts that scratch
    f.model.prefill_rows = 256;
    const metal = try backend.Metal.init(gpa, f.model, .{ .capacity = 64, .budget = 1 << 30 });
    defer metal.deinit();
    try t.expectEqual(@as(usize, 256), metal.scratch.prefill_rows);
    try t.expectEqual(try state.scratchBytes(c, 0, f.model.chunking()), try budget.held(f.model, 0) - f.model.checkpoint.residentBytes());
    f.model.prefill_rows = 128;
    try t.expectEqual(try state.scratchBytes(c, 0, f.model.chunking()), try budget.held(f.model, 0) - f.model.checkpoint.residentBytes());
}

test "inkling2 prefill commits: a chunk's layers go to the GPU in command buffers of bounded rows times layers" {
    // a 4,096-row chunk takes a layer a buffer, and so does anything bigger; a smaller one takes more layers, a small one all
    for ([_]usize{ 8192, 4096, 2048 }) |rows| try t.expectEqual(@as(usize, 1), fwd.commitEvery(rows, 48));
    try t.expectEqual(@as(usize, 3), fwd.commitEvery(1024, 48));
    try t.expectEqual(@as(usize, 6), fwd.commitEvery(512, 48));
    try t.expectEqual(@as(usize, 24), fwd.commitEvery(128, 48));
    for ([_]usize{ 64, 16, 1 }) |rows| try t.expect(fwd.commitEvery(rows, 48) >= 48);
    // never more command buffers than a queue should hold, whatever the rows, layers and repeats
    for ([_]usize{ 1, 64, 512, 4096, 8192 }) |rows| {
        for ([_]usize{ 1, 2, 47, 48, 49, 96, 500, 2047 }) |total| {
            const every = fwd.commitEvery(rows, total);
            try t.expect(every >= 1);
            try t.expect((total + every - 1) / every <= fwd.max_buffers);
        }
    }
}

test "inkling2 prefill commits: a forward reports the command buffers it used and the longest one's GPU time" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.init(arena.allocator()) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const c = &f.model.config.text;
    var cache = try state.Cache.init(f.device, c, 256, c.layers, state.default_prefill_rows);
    defer cache.deinit();
    var scratch = try state.Scratch.init(f.device, c, 256, .{ .rows = 64, .path = f.model.promptPath() });
    defer scratch.deinit();
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(5 + i);
    // the fixture has no layers: the embedding is the whole forward, one command buffer
    var tm: fwd.Timing = .{};
    try fwd.run(f.model, &scratch, &cache, &ids, .{ .logits = .none, .prompt = true, .timing = &tm });
    try t.expectEqual(@as(u32, 1), tm.buffers);
    try t.expect(tm.gpu > 0);
    try t.expectEqual(tm.gpu, tm.longest);
    // the forward's command buffers ran between its first commit and the host's wake: a buffer without a reported start
    // must not count the clock's whole uptime
    try t.expect(tm.gpu <= tm.woke - tm.committed);
    // the buffers and the GPU time add up over forwards; the longest is the longest of them all
    var again = tm;
    try fwd.run(f.model, &scratch, &cache, &ids, .{ .logits = .none, .prompt = true, .timing = &again });
    try t.expectEqual(@as(u32, 2), again.buffers);
    try t.expect(again.gpu > tm.gpu and again.longest >= tm.longest and again.longest <= again.gpu);
}

test "inkling2 prompt path: the chunk kernels where the load has them, the per-row expert projections for a layer the chain cannot take, the per-operation kernels without" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.init(arena.allocator()) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const c = &f.model.config.text;
    var cache = try state.Cache.init(f.device, c, 256, c.layers, state.default_prefill_rows);
    defer cache.deinit();
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(5 + i);
    const prompt: fwd.Out = .{ .logits = .none, .prompt = true };
    // the fixture has no chunk kernels: prompt chunks take the per-operation path, whatever its experts
    try t.expectEqual(state.PromptPath.per_op, f.model.promptPath());
    var per_op = try state.Scratch.init(f.device, c, 256, .{ .rows = 64, .path = f.model.promptPath() });
    defer per_op.deinit();
    try fwd.run(f.model, &per_op, &cache, &ids, prompt);
    // a scratch made for another path is refused before any work; a decode window's forward reads no chunk buffer and is not
    var chunk = try state.Scratch.init(f.device, c, 256, .{ .rows = 64, .path = .chunk });
    defer chunk.deinit();
    try t.expectError(error.Inkling2ScratchForAnotherPromptPath, fwd.run(f.model, &chunk, &cache, &ids, prompt));
    try fwd.run(f.model, &chunk, &cache, ids[0..8], .{ .logits = .none });
    // a load with the chunk kernels takes the chunk path, and the per-row expert projections once a layer needs them
    if (!f.device.tensorUnits()) return;
    f.model.chunk = try @import("model.zig").Chunk.init(f.device, gpa);
    defer {
        f.model.chunk.?.deinit();
        f.model.chunk = null;
    }
    try t.expectEqual(state.PromptPath.chunk, f.model.promptPath());
    try fwd.run(f.model, &chunk, &cache, &ids, prompt);
    try t.expectError(error.Inkling2ScratchForAnotherPromptPath, fwd.run(f.model, &per_op, &cache, &ids, prompt));
    var layer: wts.Layer = undefined;
    layer.experts = .{ .exl3 = .{ .w13 = undefined, .w2 = undefined, .xd_ok = false } };
    f.model.weights.layers = (&layer)[0..1];
    defer f.model.weights.layers = &.{};
    try t.expectEqual(state.PromptPath.chunk_rows, f.model.promptPath());
    try t.expectError(error.Inkling2ScratchForAnotherPromptPath, fwd.run(f.model, &chunk, &cache, &ids, prompt));
    var rows = try state.Scratch.init(f.device, c, 256, .{ .rows = 64, .path = .chunk_rows });
    defer rows.deinit();
    try fwd.run(f.model, &rows, &cache, &ids, prompt);
    layer.experts.exl3.xd_ok = true;
    try t.expectEqual(state.PromptPath.chunk, f.model.promptPath());
    try t.expectError(error.Inkling2ScratchForAnotherPromptPath, fwd.run(f.model, &rows, &cache, &ids, prompt));
}

test "inkling2 dspark: a drafted load's facts fix the depth or leave it to the rule, with the timed window costs and the logprob rows kept" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.init(arena.allocator()) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const metal = try backend.Metal.init(gpa, f.model, .{ .capacity = 64, .budget = 1 << 30 });
    defer metal.deinit();
    // a load without drafts: no costs timed, no head, no fixed depth
    const plain = metal.facts();
    try t.expect(!plain.mtp and !plain.speculate and plain.fixed_depth == null and plain.window_costs.len == 0);
    try t.expect(!metal.tapped);
    for (&metal.costs, 1..) |*cost, rows| cost.* = .{ .width = @intCast(rows), .ms = 5.0 + 2.0 * @as(f64, @floatFromInt(rows)) };
    metal.timed = backend.measured_widths;
    for ([_]dspark.Depth{ .{ .fixed = 3 }, .{ .fixed = 7 }, .auto }) |depth| {
        const d = try f.drafter(depth.most());
        defer d.deinit();
        var wrap = try dspark.wrap(gpa, metal.backend(), d);
        defer wrap.deinit();
        // the target writes the drafter's taps, and the wrapped backend still gives a first token's logprob row
        try t.expect(metal.tapped);
        try t.expect(wrap.backend().vtable.first_row != null);
        const facts = dspark.laneFacts(&wrap, metal.facts(), depth);
        try t.expectEqual(depth.fixedDepth(), facts.fixed_depth);
        try t.expect(facts.mtp and facts.speculate and facts.fixed_draft_block and facts.plain_guard);
        try t.expectEqual(depth.most(), facts.drafts);
        try t.expectEqual(@as(usize, backend.measured_widths), facts.window_costs.len);
        var cfg = try lanes.Config.init(gpa, facts, state.window_rows, state.window_rows - 1);
        defer cfg.deinit(gpa);
        try t.expect(cfg.family_mtp);
        // no floor and no chances: the lane core prices by its own rates, as before
        try t.expect(facts.draft_confidence == null and !facts.draft_probabilities and cfg.draft_confidence == null and !cfg.node_probabilities);
        try t.expectEqual(depth.fixedDepth(), cfg.fixed_depth);
        try t.expectEqual(depth.most(), cfg.most_drafts);
        try t.expectEqual(@as(?f64, 9.0), cfg.family_costs.get(2));
        try t.expectEqual(@as(?f64, 21.0), cfg.family_costs.get(8));
        try t.expectEqual(@as(?f64, null), cfg.family_costs.get(9));
    }
}

/// A stream opened on `d` with `n` rows of `taps` absorbed and a hold of `depth` after them, left open for the caller to release.
fn holdOpen(d: *dspark.Drafter, s: *lanes.Stream, taps: mtl.Buffer, n: usize, depth: u32) !void {
    const dv = d.drafter();
    try dv.open(s);
    errdefer dv.release(s);
    try dv.absorb(&.{.{ .stream = s, .features = features(taps, 0, n), .start = 0, .follow = &.{}, .prompt = true }});
    try dv.hold(&.{.{ .stream = s, .pending = tokenAt(n), .position = n + 1, .depth = depth }});
}

/// The top-1 probability of each of a hold's seven steps, summed on the host over `logits` (bf16 [7][vocab], the padded ids -inf).
fn hostProbs(logits: []const u16, out: *[7]f64) void {
    for (out, 0..) |*o, k| {
        const row = logits[k * vocab ..][0..vocab];
        var top: f64 = -std.math.inf(f64);
        for (row) |b| top = @max(top, mix.f32Of(b));
        var sum: f64 = 0;
        for (row) |b| sum += @exp(@as(f64, mix.f32Of(b)) - top);
        o.* = 1 / sum;
    }
}

test "inkling2 dspark: a hold that keeps probabilities gives each held draft's own, from the same hold as one that keeps none" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.init(arena.allocator()) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const plain = try f.drafter(7);
    defer plain.deinit();
    const kept = try f.drafterWith(.{ .depth = 7, .probabilities = true });
    defer kept.deinit();
    try plain.captureLogits();
    try kept.captureLogits();
    const n = 300;
    const taps = try tapsBuffer(f.device, n + 1, 31);
    defer taps.deinit();
    const prompt = try gpa.alloc(u32, n);
    defer gpa.free(prompt);
    for (prompt, 0..) |*x, i| x.* = tokenAt(i);
    var s = try lanes.Stream.init(gpa, .{ .id = "probs", .prompt = prompt, .max_new = 8 });
    defer s.deinit(gpa);
    // a drafter that keeps none has none to give
    var out: [7]f64 = undefined;
    var plain_props: [7]u32 = undefined;
    try holdOpen(plain, &s, taps, n, 7);
    try t.expect(!try plain.drafter().chances(&s, &out));
    try plain.drafter().held(&.{&s}, &.{&plain_props});
    const plain_logits = try gpa.dupe(u16, plain.captured());
    defer gpa.free(plain_logits);
    plain.drafter().release(&s);
    // one that keeps them holds the same drafts from the same logits
    var props: [7]u32 = undefined;
    try holdOpen(kept, &s, taps, n, 7);
    defer kept.drafter().release(&s);
    try kept.drafter().held(&.{&s}, &.{&props});
    try t.expectEqualSlices(u32, &plain_props, &props);
    try t.expectEqualSlices(u16, plain_logits, kept.captured());
    try t.expect(try kept.drafter().chances(&s, &out));
    var want: [7]f64 = undefined;
    hostProbs(kept.captured(), &want);
    var spread: f64 = 0;
    for (out, want) |got, p| {
        try t.expect(got > 1.0 / @as(f64, valid) and got <= 1);
        // the host sums logits rounded to bf16 (up to 6% in an exponent near the softcap), the drafter's are the f32 ones
        try t.expectApproxEqRel(p, got, 0.15);
        spread = @max(spread, @abs(got - out[0]));
    }
    try t.expect(spread > 0); // each step has its own
    // the first drafts of the hold, asked for alone, have the same values; more than were held are refused
    var few: [3]f64 = undefined;
    try t.expect(try kept.drafter().chances(&s, &few));
    try t.expectEqualSlices(f64, out[0..3], &few);
    var more: [8]f64 = undefined;
    try t.expectError(error.DraftHeldFewer, kept.drafter().chances(&s, &more));
    var unopened = try lanes.Stream.init(gpa, .{ .id = "unopened", .prompt = &.{ 5, 6 }, .max_new = 1 });
    defer unopened.deinit(gpa);
    try t.expectError(error.DraftStreamUnknown, kept.drafter().chances(&unopened, &few));
    // an absorb drops the hold, and its probabilities with it
    try kept.drafter().absorb(&.{.{ .stream = &s, .features = features(taps, n, 1), .start = n, .follow = &.{}, .prompt = false }});
    try t.expectError(error.DraftHeldFewer, kept.drafter().chances(&s, &few));
}

test "inkling2 dspark: each stream of a round has the probabilities of its hold alone, and a shallower hold the deeper one's first" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.init(arena.allocator()) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const d = try f.drafterWith(.{ .depth = 7, .probabilities = true });
    defer d.deinit();
    const dv = d.drafter();
    const ns = [2]usize{ 300, 1200 };
    var taps: [2]mtl.Buffer = undefined;
    var prompts: [2][]u32 = undefined;
    var streams: [2]lanes.Stream = undefined;
    for (0..2) |i| {
        taps[i] = try tapsBuffer(f.device, ns[i] + 1, 41 + i);
        prompts[i] = try gpa.alloc(u32, ns[i]);
        for (prompts[i], 0..) |*x, j| x.* = tokenAt(j + 3 * i);
        streams[i] = try lanes.Stream.init(gpa, .{ .id = "pair", .prompt = prompts[i], .max_new = 4 });
    }
    defer for (0..2) |i| {
        taps[i].deinit();
        gpa.free(prompts[i]);
        streams[i].deinit(gpa);
    };
    var alone: [2][7]f64 = undefined;
    for (0..2) |i| {
        try holdOpen(d, &streams[i], taps[i], ns[i], 7);
        defer dv.release(&streams[i]);
        try t.expect(try dv.chances(&streams[i], &alone[i]));
    }
    try t.expect(!std.mem.eql(f64, &alone[0], &alone[1]));
    // both in one round, the second asking for four drafts: the probabilities are those of each hold alone
    for (&streams) |*s| try dv.open(s);
    defer for (&streams) |*s| dv.release(s);
    try dv.absorb(&.{
        .{ .stream = &streams[0], .features = features(taps[0], 0, ns[0]), .start = 0, .follow = &.{}, .prompt = true },
        .{ .stream = &streams[1], .features = features(taps[1], 0, ns[1]), .start = 0, .follow = &.{}, .prompt = true },
    });
    try dv.hold(&.{
        .{ .stream = &streams[0], .pending = tokenAt(ns[0]), .position = ns[0] + 1, .depth = 7 },
        .{ .stream = &streams[1], .pending = tokenAt(ns[1]), .position = ns[1] + 1, .depth = 4 },
    });
    var both: [2][7]f64 = undefined;
    try t.expect(try dv.chances(&streams[0], &both[0]));
    try t.expect(try dv.chances(&streams[1], both[1][0..4]));
    try t.expectEqualSlices(f64, &alone[0], &both[0]);
    try t.expectEqualSlices(f64, alone[1][0..4], both[1][0..4]);
    var deeper: [5]f64 = undefined;
    try t.expectError(error.DraftHeldFewer, dv.chances(&streams[1], &deeper));
}

test "inkling2 dspark: the drafting settings give the lane core a floor of 0.7 and the deepest chain, or with the floor off every draft of a fixed depth" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.init(arena.allocator()) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const metal = try backend.Metal.init(gpa, f.model, .{ .capacity = 64, .budget = 1 << 30 });
    defer metal.deinit();
    for (&metal.costs, 1..) |*cost, rows| cost.* = .{ .width = @intCast(rows), .ms = 5.0 + 2.0 * @as(f64, @floatFromInt(rows)) };
    metal.timed = backend.measured_widths;
    const Case = struct { depth: ?[]const u8, floor: ?[]const u8, want: ?f64, held: u32, fixed: ?u8 };
    for ([_]Case{
        .{ .depth = null, .floor = null, .want = 0.7, .held = 7, .fixed = 7 }, // nothing set: the default floor, the deepest chain
        .{ .depth = null, .floor = "0.9", .want = 0.9, .held = 7, .fixed = 7 },
        .{ .depth = "5", .floor = null, .want = 0.7, .held = 5, .fixed = 5 }, // a depth caps what is held
        .{ .depth = "auto", .floor = "0.5", .want = 0.5, .held = 7, .fixed = null },
        .{ .depth = null, .floor = "off", .want = null, .held = 4, .fixed = 4 }, // off: every draft of the depth is verified
        .{ .depth = "6", .floor = "OFF", .want = null, .held = 6, .fixed = 6 },
        .{ .depth = "auto", .floor = "off", .want = null, .held = 7, .fixed = null },
    }) |c| {
        const drafting = try dspark.Drafting.parse(c.depth, c.floor);
        try t.expectEqual(c.want, drafting.confidence);
        const d = try f.drafterWith(drafting.options());
        defer d.deinit();
        const own = d.drafter().facts();
        try t.expectEqual(c.want != null, own.chances);
        try t.expectEqual(c.want, own.confidence_floor);
        try t.expectEqual(c.held, own.depth);
        var wrap = try dspark.wrap(gpa, metal.backend(), d);
        defer wrap.deinit();
        const facts = dspark.laneFacts(&wrap, metal.facts(), drafting.depth);
        // with a floor the core compares each draft's chance with it, and prices no rows by chances or grants; without one it
        // prices by its own rates, as it always has
        try t.expectEqual(c.want, facts.draft_confidence);
        try t.expect(!facts.draft_probabilities and !facts.granted_depth);
        try t.expectEqual(c.fixed, facts.fixed_depth);
        try t.expectEqual(c.held, facts.drafts);
        var cfg = try lanes.Config.init(gpa, facts, state.window_rows, state.window_rows - 1);
        defer cfg.deinit(gpa);
        try t.expectEqual(c.want, cfg.draft_confidence);
        try t.expectEqual(c.want != null, cfg.node_probabilities);
        try t.expect(!cfg.granted_depth);
        try t.expectEqual(c.held, cfg.most_drafts);
        try t.expectEqual(c.fixed, cfg.fixed_depth);
    }
    // the library's own default offers no floor: only the settings' reading has one
    const d = try f.drafter(4);
    defer d.deinit();
    const own = d.drafter().facts();
    try t.expect(!own.chances and own.confidence_floor == null);
    // a floor outside (0, 1) is not a floor
    for ([_]f64{ 0, 1, -0.1, 1.5, std.math.nan(f64) }) |bad| try t.expectError(error.DraftConfidence, f.drafterWith(.{ .depth = 4, .confidence = bad }));
}

test "inkling2 dspark: kept draft rings restore full-window proposals across a late prompt resume" {
    const snapshot = @import("snapshot.zig");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.init(arena.allocator()) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const d = try f.drafter(7);
    defer d.deinit();
    const dv = d.drafter();
    const total: usize = 1204;
    const restore_at: usize = 1200;
    const taps = try f.device.buffer(total * row_bytes, mtl.ResourceOptions.shared);
    defer taps.deinit();
    fill(taps, 0, total * 8 * hidden, 0.5, 4242);
    const prompt = try gpa.alloc(u32, total);
    defer gpa.free(prompt);
    for (prompt, 0..) |*x, i| x.* = tokenAt(i + 11);

    // Reference: absorb the full prompt, then hold at `total`
    var s_full = try lanes.Stream.init(gpa, .{ .id = "full", .prompt = prompt, .max_new = 4 });
    defer s_full.deinit(gpa);
    try dv.open(&s_full);
    defer dv.release(&s_full);
    try dv.absorb(&.{.{ .stream = &s_full, .features = features(taps, 0, total), .start = 0, .follow = &.{}, .prompt = true }});
    try dv.hold(&.{.{ .stream = &s_full, .pending = tokenAt(total + 11), .position = total + 1, .depth = 7 }});
    var want: [7]u32 = undefined;
    try dv.held(&.{&s_full}, &.{&want});

    // Save at `restore_at` (with draft rings), then resume on a fresh stream for the last 4 rows
    var s_save = try lanes.Stream.init(gpa, .{ .id = "save", .prompt = prompt, .max_new = 4 });
    defer s_save.deinit(gpa);
    try dv.open(&s_save);
    defer dv.release(&s_save);
    try dv.absorb(&.{.{ .stream = &s_save, .features = features(taps, 0, restore_at), .start = 0, .follow = &.{}, .prompt = true }});
    var snap = snapshot.Snap{
        .buf = try f.device.buffer(16, mtl.ResourceOptions.private),
        .at = restore_at,
        .bytes = 16,
    };
    defer snap.free();
    try d.keepLane(&s_save, &snap);
    try t.expect(snap.draft != null);
    try t.expectEqual(@as(u64, total - 1023), snap.draft_from);

    var s_res = try lanes.Stream.init(gpa, .{ .id = "res", .prompt = prompt, .max_new = 4 });
    defer s_res.deinit(gpa);
    s_res.reuse.saved = &snap;
    s_res.cached = restore_at;
    try dv.open(&s_res);
    defer dv.release(&s_res);
    try dv.absorb(&.{.{ .stream = &s_res, .features = features(taps, restore_at, total - restore_at), .start = restore_at, .follow = &.{}, .prompt = true }});
    try dv.hold(&.{.{ .stream = &s_res, .pending = tokenAt(total + 11), .position = total + 1, .depth = 7 }});
    var got: [7]u32 = undefined;
    try dv.held(&.{&s_res}, &.{&got});
    try t.expectEqualSlices(u32, &want, &got);
}

test "inkling2 dspark mini: facts, taps, laneBytes, keptBytesFor, residency hook and supported checks" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const f = Fixture.makeFor(a, config_test.mini, .sixteen) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const m = f.model.config.mtp.?;
    try t.expect(dspark.Drafter.supported(&m, &f.model.config.text));
    // Refusals: unsupported GQA, layer count, width alignment, final tap mismatch
    var bad_gqa = m;
    bad_gqa.q_heads = 32;
    try t.expect(!dspark.Drafter.supported(&bad_gqa, &f.model.config.text));
    var bad_layers = m;
    bad_layers.layers = 9;
    try t.expect(!dspark.Drafter.supported(&bad_layers, &f.model.config.text));
    var bad_hidden = m;
    var bad_text = f.model.config.text;
    bad_hidden.hidden = 2400;
    bad_text.hidden = 2400;
    try t.expect(!dspark.Drafter.supported(&bad_hidden, &bad_text));
    var bad_tap = m;
    bad_tap.taps[bad_tap.taps_n - 1] = 24;
    try t.expect(!dspark.Drafter.supported(&bad_tap, &f.model.config.text));

    const d = try f.drafter(7);
    defer d.deinit();
    const dv = d.drafter();
    try t.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, dv.taps());
    try t.expectEqual(@as(usize, 4 * 2304 * 2), d.tapRowBytes());
    const ring_bytes = (1031 + 7) * 1024 * 2;
    try t.expectEqual(@as(usize, 2 * 3 * ring_bytes + 2304 * 2 + 256 * 2 + 8 * 4 + 8 * 4), d.laneBytes());
    try t.expectEqual(@as(u64, 2 * 3 * ring_bytes), dspark.Drafter.keptBytesFor(m));
    // Residency hook only touches the 3 active layer pairs (6 rings + f_rnn + state + seq + prob = 10 buffers)
    var c: Counter = .{};
    d.setResident(c.follow());
    const scratch = c.joined;
    try t.expectEqual(d.scratchBytes(), c.joined_bytes);
    var s = try lanes.Stream.init(gpa, .{ .id = "mini-res", .prompt = &.{ 5, 6 }, .max_new = 1 });
    defer s.deinit(gpa);
    try dv.open(&s);
    try t.expectEqual(scratch + 10, c.joined);
    try t.expectEqual(d.laneBytes(), c.joined_bytes - d.scratchBytes());
    dv.release(&s);
    try t.expectEqual(@as(usize, 10), c.left);
    d.setResident(null);
    try t.expectEqual(c.joined, c.left);
    try t.expectEqual(c.joined_bytes, c.left_bytes);
}

test "inkling2 dspark mini: bf16 and 8-bit contexts absorbed in any chunking hold the same block" {
    for ([_]dw.Bits{ .sixteen, .eight }) |bits| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const f = Fixture.makeFor(arena.allocator(), config_test.mini, bits) catch |err| switch (err) {
            error.NoDevice => return error.SkipZigTest,
            else => return err,
        };
        defer f.deinit();
        const d = try f.drafter(7);
        defer d.deinit();
        try d.captureLogits();
        const n = 1500;
        const taps = try tapsBufferFor(f.device, n + 1, d.tapRowBytes(), 17);
        defer taps.deinit();
        const prompt = try gpa.alloc(u32, n);
        defer gpa.free(prompt);
        for (prompt, 0..) |*x, i| x.* = tokenAt(i);
        var s = try lanes.Stream.init(gpa, .{ .id = "mini-chunks", .prompt = prompt, .max_new = 8 });
        defer s.deinit(gpa);
        const a_logits = try gpa.alloc(u16, 7 * vocab);
        defer gpa.free(a_logits);
        const b_logits = try gpa.alloc(u16, 7 * vocab);
        defer gpa.free(b_logits);
        var a_props: [7]u32 = undefined;
        var b_props: [7]u32 = undefined;
        try held(d, &s, taps, n, n, false, 7, &a_props, a_logits);
        for (a_props) |x| try t.expect(x < valid);
        for ([_]struct { chunk: usize, prompt: bool }{ .{ .chunk = 37, .prompt = false }, .{ .chunk = 512, .prompt = true }, .{ .chunk = n, .prompt = true } }) |c| {
            try held(d, &s, taps, n, c.chunk, c.prompt, 7, &b_props, b_logits);
            try t.expectEqualSlices(u32, &a_props, &b_props);
            try t.expectEqualSlices(u16, a_logits, b_logits);
        }
    }
}

test "inkling2 dspark mini: depth sweep 1 to 7 and kept draft rings restore full-window proposals" {
    const snapshot = @import("snapshot.zig");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.makeFor(arena.allocator(), config_test.mini, .sixteen) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const a_dr = try f.drafter(5);
    defer a_dr.deinit();
    const b_dr = try f.drafter(7);
    defer b_dr.deinit();
    const rb = a_dr.tapRowBytes();
    const sweep_taps = try tapsBufferFor(f.device, 400, rb, 23);
    defer sweep_taps.deinit();
    var h: Sweep = .{ .taps = sweep_taps, .a = a_dr, .b = b_dr };
    const r = try sweep.sweep(gpa, h.harness(), .{ .block = 7, .prompts = &.{ 6, 40 }, .rounds = 8 });
    try t.expectEqual(sweep.Report{ .runs = 16, .rounds = 128, .drafts = 2 * (8 * 28 + 32) }, r);

    // Late prompt resume with kept 3-layer draft rings
    const dv = b_dr.drafter();
    const total: usize = 1204;
    const restore_at: usize = 1200;
    const taps = try tapsBufferFor(f.device, total, rb, 99);
    defer taps.deinit();
    const prompt = try gpa.alloc(u32, total);
    defer gpa.free(prompt);
    for (prompt, 0..) |*x, i| x.* = tokenAt(i + 13);
    var s_full = try lanes.Stream.init(gpa, .{ .id = "mini-full", .prompt = prompt, .max_new = 4 });
    defer s_full.deinit(gpa);
    try dv.open(&s_full);
    defer dv.release(&s_full);
    try dv.absorb(&.{.{ .stream = &s_full, .features = featuresFor(taps, 0, total, rb), .start = 0, .follow = &.{}, .prompt = true }});
    try dv.hold(&.{.{ .stream = &s_full, .pending = tokenAt(total + 13), .position = total + 1, .depth = 7 }});
    var want: [7]u32 = undefined;
    try dv.held(&.{&s_full}, &.{&want});

    var s_save = try lanes.Stream.init(gpa, .{ .id = "mini-save", .prompt = prompt, .max_new = 4 });
    defer s_save.deinit(gpa);
    try dv.open(&s_save);
    defer dv.release(&s_save);
    try dv.absorb(&.{.{ .stream = &s_save, .features = featuresFor(taps, 0, restore_at, rb), .start = 0, .follow = &.{}, .prompt = true }});
    var snap = snapshot.Snap{
        .buf = try f.device.buffer(16, mtl.ResourceOptions.private),
        .at = restore_at,
        .bytes = 16,
    };
    defer snap.free();
    try b_dr.keepLane(&s_save, &snap);
    try t.expect(snap.draft != null);
    try t.expectEqual(dspark.Drafter.keptBytesFor(f.model.config.mtp.?), snap.draft.?.length());

    var s_res = try lanes.Stream.init(gpa, .{ .id = "mini-res", .prompt = prompt, .max_new = 4 });
    defer s_res.deinit(gpa);
    s_res.reuse.saved = &snap;
    s_res.cached = restore_at;
    try dv.open(&s_res);
    defer dv.release(&s_res);
    try dv.absorb(&.{.{ .stream = &s_res, .features = featuresFor(taps, restore_at, total - restore_at, rb), .start = restore_at, .follow = &.{}, .prompt = true }});
    try dv.hold(&.{.{ .stream = &s_res, .pending = tokenAt(total + 13), .position = total + 1, .depth = 7 }});
    var got: [7]u32 = undefined;
    try dv.held(&.{&s_res}, &.{&got});
    try t.expectEqualSlices(u32, &want, &got);
}

/// Multi-stream batched hold (`count` in 2..8, mixed context lengths and depths, probabilities on) produces
/// bit-identical proposals and probabilities to solo holds when `chunk == null`, and allocates no batch scratch
/// until a multi-stream hold actually runs.
fn batchHoldMatchesSolo(config_json: []const u8, bits: dw.Bits, counts: []const usize) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.makeFor(arena.allocator(), config_json, bits) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer f.deinit();
    const d = try f.drafterWith(.{ .depth = 7, .probabilities = true });
    defer d.deinit();
    const dv = d.drafter();
    const rb = d.tapRowBytes();
    // Single-stream warm-up never allocates BatchScratch.
    try t.expect(d.batch == null);
    const max_c = 8;
    const ns = [max_c]usize{ 12, 300, 1150, 64, 512, 1024, 255, 1300 };
    const depths = [max_c]u32{ 7, 4, 6, 1, 5, 3, 7, 2 };
    var taps: [max_c]mtl.Buffer = undefined;
    var prompts: [max_c][]u32 = undefined;
    var streams: [max_c]lanes.Stream = undefined;
    for (0..max_c) |i| {
        taps[i] = try tapsBufferFor(f.device, ns[i] + 1, rb, 101 + i);
        prompts[i] = try gpa.alloc(u32, ns[i]);
        for (prompts[i], 0..) |*x, j| x.* = tokenAt(j + 7 * i);
        streams[i] = try lanes.Stream.init(gpa, .{ .id = "batch", .prompt = prompts[i], .max_new = 8 });
    }
    defer for (0..max_c) |i| {
        taps[i].deinit();
        gpa.free(prompts[i]);
        streams[i].deinit(gpa);
    };
    var alone_props: [max_c][7]u32 = undefined;
    var alone_probs: [max_c][7]f64 = undefined;
    for (0..max_c) |i| {
        try dv.open(&streams[i]);
        defer dv.release(&streams[i]);
        try dv.absorb(&.{.{ .stream = &streams[i], .features = featuresFor(taps[i], 0, ns[i], rb), .start = 0, .follow = &.{}, .prompt = true }});
        try dv.hold(&.{.{ .stream = &streams[i], .pending = tokenAt(ns[i] + i), .position = ns[i] + 1, .depth = depths[i] }});
        try dv.held(&.{&streams[i]}, &.{alone_props[i][0..depths[i]]});
        try t.expect(try dv.chances(&streams[i], alone_probs[i][0..depths[i]]));
    }
    // Solo holds alone still allocate zero batch scratch.
    try t.expect(d.batch == null);
    var res_c: Counter = .{};
    d.setResident(res_c.follow());
    defer {
        d.setResident(null);
        std.debug.assert(res_c.joined == res_c.left and res_c.joined_bytes == res_c.left_bytes);
    }
    const base_scratch = d.scratchBytes();
    for (counts) |count| {
        var absorbs: [max_c]dr.Absorb = undefined;
        var holds: [max_c]dr.Hold = undefined;
        for (0..count) |i| {
            try dv.open(&streams[i]);
            absorbs[i] = .{ .stream = &streams[i], .features = featuresFor(taps[i], 0, ns[i], rb), .start = 0, .follow = &.{}, .prompt = true };
            holds[i] = .{ .stream = &streams[i], .pending = tokenAt(ns[i] + i), .position = ns[i] + 1, .depth = depths[i] };
        }
        defer for (0..count) |i| dv.release(&streams[i]);
        try dv.absorb(absorbs[0..count]);
        try dv.hold(holds[0..count]);
        try t.expect(d.batch != null);
        try t.expect(d.scratchBytes() > base_scratch);
        for (0..count) |i| {
            var got_props: [7]u32 = undefined;
            var got_probs: [7]f64 = undefined;
            try dv.held(&.{&streams[i]}, &.{got_props[0..depths[i]]});
            try t.expect(try dv.chances(&streams[i], got_probs[0..depths[i]]));
            try t.expectEqualSlices(u32, alone_props[i][0..depths[i]], got_props[0..depths[i]]);
            try t.expectEqualSlices(f64, alone_probs[i][0..depths[i]], got_probs[0..depths[i]]);
        }
    }
}

test "inkling2 dspark: batched multi-stream hold matches solo holds bit-for-bit across bf16 and 8-bit" {
    for ([_]dw.Bits{ .sixteen, .eight }) |bits| {
        try batchHoldMatchesSolo(config_test.real, bits, &.{ 2, 4, 8 });
    }
}

test "inkling2 dspark mini: batched multi-stream hold matches solo holds bit-for-bit across bf16 and 8-bit" {
    for ([_]dw.Bits{ .sixteen, .eight }) |bits| {
        try batchHoldMatchesSolo(config_test.mini, bits, &.{ 2, 4, 8 });
    }
}

test "inkling2 dspark: batched multi-stream hold with tensor-unit GEMM is valid and batch-size invariant" {
    const Chunk = @import("model.zig").Chunk;
    for ([_][]const u8{ config_test.mini, config_test.real }) |cfg_json| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const f = Fixture.makeFor(arena.allocator(), cfg_json, .sixteen) catch |err| switch (err) {
            error.NoDevice => return error.SkipZigTest,
            else => return err,
        };
        defer f.deinit();
        if (!f.device.tensorUnits()) return;
        f.model.chunk = try Chunk.initFor(f.device, gpa, f.model.config.text.local.q, f.model.config.text.local.kv, f.model.config.quant);
        defer {
            f.model.chunk.?.deinit();
            f.model.chunk = null;
        }
        const d = try f.drafterWith(.{ .depth = 7, .probabilities = true });
        defer d.deinit();
        const dv = d.drafter();
        const rb = d.tapRowBytes();
        const max_c = 8;
        const ns = [max_c]usize{ 24, 300, 1100, 80, 500, 1024, 255, 1250 };
        const depths = [max_c]u32{ 7, 4, 6, 2, 5, 3, 7, 1 };
        var taps: [max_c]mtl.Buffer = undefined;
        var prompts: [max_c][]u32 = undefined;
        var streams: [max_c]lanes.Stream = undefined;
        for (0..max_c) |i| {
            taps[i] = try tapsBufferFor(f.device, ns[i] + 1, rb, 201 + i);
            prompts[i] = try gpa.alloc(u32, ns[i]);
            for (prompts[i], 0..) |*x, j| x.* = tokenAt(j + 11 * i);
            streams[i] = try lanes.Stream.init(gpa, .{ .id = "gemm-batch", .prompt = prompts[i], .max_new = 8 });
        }
        defer for (0..max_c) |i| {
            taps[i].deinit();
            gpa.free(prompts[i]);
            streams[i].deinit(gpa);
        };
        var solo_probs: [4][7]f64 = undefined;
        for (0..4) |i| {
            try dv.open(&streams[i]);
            defer dv.release(&streams[i]);
            try dv.absorb(&.{.{ .stream = &streams[i], .features = featuresFor(taps[i], 0, ns[i], rb), .start = 0, .follow = &.{}, .prompt = true }});
            try dv.hold(&.{.{ .stream = &streams[i], .pending = tokenAt(ns[i] + i), .position = ns[i] + 1, .depth = depths[i] }});
            try t.expect(try dv.chances(&streams[i], solo_probs[i][0..depths[i]]));
        }
        // Run C = 4 and C = 8 on the tensor-unit GEMM path: first 4 streams must be bit-identical across batch sizes,
        // and all tokens/probabilities must be valid and close to solo matvec probabilities.
        var props4: [4][7]u32 = undefined;
        var probs4: [4][7]f64 = undefined;
        {
            var absorbs: [4]dr.Absorb = undefined;
            var holds: [4]dr.Hold = undefined;
            for (0..4) |i| {
                try dv.open(&streams[i]);
                absorbs[i] = .{ .stream = &streams[i], .features = featuresFor(taps[i], 0, ns[i], rb), .start = 0, .follow = &.{}, .prompt = true };
                holds[i] = .{ .stream = &streams[i], .pending = tokenAt(ns[i] + i), .position = ns[i] + 1, .depth = depths[i] };
            }
            defer for (0..4) |i| dv.release(&streams[i]);
            try dv.absorb(&absorbs);
            try dv.hold(&holds);
            for (0..4) |i| {
                try dv.held(&.{&streams[i]}, &.{props4[i][0..depths[i]]});
                try t.expect(try dv.chances(&streams[i], probs4[i][0..depths[i]]));
                for (props4[i][0..depths[i]]) |tok| try t.expect(tok < valid);
                for (probs4[i][0..depths[i]], solo_probs[i][0..depths[i]]) |got, want| {
                    try t.expect(got > 0 and got <= 1);
                    try t.expectApproxEqRel(want, got, 0.05);
                }
            }
        }
        {
            var absorbs: [max_c]dr.Absorb = undefined;
            var holds: [max_c]dr.Hold = undefined;
            for (0..max_c) |i| {
                try dv.open(&streams[i]);
                absorbs[i] = .{ .stream = &streams[i], .features = featuresFor(taps[i], 0, ns[i], rb), .start = 0, .follow = &.{}, .prompt = true };
                holds[i] = .{ .stream = &streams[i], .pending = tokenAt(ns[i] + i), .position = ns[i] + 1, .depth = depths[i] };
            }
            defer for (0..max_c) |i| dv.release(&streams[i]);
            try dv.absorb(&absorbs);
            try dv.hold(&holds);
            for (0..max_c) |i| {
                var got_props: [7]u32 = undefined;
                var got_probs: [7]f64 = undefined;
                try dv.held(&.{&streams[i]}, &.{got_props[0..depths[i]]});
                try t.expect(try dv.chances(&streams[i], got_probs[0..depths[i]]));
                for (got_props[0..depths[i]]) |tok| try t.expect(tok < valid);
                for (got_probs[0..depths[i]]) |p| try t.expect(p > 0 and p <= 1);
                if (i < 4) {
                    try t.expectEqualSlices(u32, props4[i][0..depths[i]], got_props[0..depths[i]]);
                    try t.expectEqualSlices(f64, probs4[i][0..depths[i]], got_probs[0..depths[i]]);
                }
            }
        }
    }
}
