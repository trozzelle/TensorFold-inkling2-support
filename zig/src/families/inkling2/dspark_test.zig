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

    fn init(a: std.mem.Allocator) !*Fixture {
        const device = try mtl.Device.init();
        errdefer device.deinit();
        const f = try gpa.create(Fixture);
        errdefer gpa.destroy(f);
        f.* = .{ .device = device, .model = try gpa.create(Model), .pipes = undefined };
        errdefer gpa.destroy(f.model);
        errdefer f.free();
        var why: config.Why = .{};
        var cfg = try config.parse(a, config_test.real, &why);
        cfg.text.vocab = vocab;
        cfg.text.unpadded_vocab = valid;
        cfg.mtp.?.vocab = vocab;
        const m = cfg.mtp.?;
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
            .dflare = try f.norm(hidden),
            .input = try f.norm(hidden),
            .post = try f.norm(hidden),
            .q_norm = try f.norm(m.head_dim),
            .k_norm = try f.norm(m.head_dim),
            .k_norm_ctx = try f.norm(m.head_dim),
            .q = try f.dense(qw, hidden, 0.02),
            .k = try f.dense(kvw, hidden, 0.02),
            .v = try f.dense(kvw, hidden, 0.02),
            .k_ctx = try f.dense(kvw, hidden, 0.02),
            .v_ctx = try f.dense(kvw, hidden, 0.02),
            .o = try f.dense(hidden, qw, 0.02),
            .gate = try f.dense(m.inter, hidden, 0.02),
            .up = try f.dense(m.inter, hidden, 0.02),
            .down = try f.dense(hidden, m.inter, 0.02),
        };
        const layers = try gpa.alloc(dw.Layer, m.layers);
        errdefer gpa.free(layers);
        for (layers) |*l| {
            l.* = shared;
            l.dflare = try f.norm(hidden);
            l.input = try f.norm(hidden);
            l.post = try f.norm(hidden);
        }
        var w: dw.Weights = .{
            .layers = layers,
            .alpha = std.mem.zeroes([mix.max_rows][mix.max_taps]u16),
            .alpha_buffer = try device.buffer((m.layers + 1) * m.taps_n * 2, mtl.ResourceOptions.shared),
            .rows = m.layers + 1,
            .taps = m.taps_n,
            .mask_embed = try f.tensor(&.{hidden}, 1.0),
            .norm = try f.norm(hidden),
            .markov = .{
                .dflare = try f.norm(hidden),
                .w1 = try f.dense(vocab, m.rank, 1.0),
                .rnn = try f.dense(3 * m.rank, 2 * m.rank + 2 * hidden, 0.05),
                .rnn_b = try f.tensor(&.{3 * m.rank}, 0.5),
                .hidden_out = try f.dense(hidden, m.rank, 0.05),
                .hidden_out_b = try f.tensor(&.{hidden}, 0.5),
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
                .embed = try f.dense(vocab, hidden, 0.5),
                .unembed = try f.dense(vocab, hidden, 0.5),
                .embed_norm = try f.norm(hidden),
                .norm = try f.norm(hidden),
                .out_query = try f.tensor(&.{hidden}, 1.0),
                .layers = &.{},
                .tables = &.{},
                .expert_buffers = &.{},
            },
            .glue = f.pipes,
            .exl3 = undefined,
            .draft = w,
            .warm_wait = true,
            .first_chunk = 0,
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
        return dspark.Drafter.init(gpa, f.device, f.model, &f.model.draft.?, .{ .depth = depth });
    }
};

/// `rows` rows of taps, random bf16 of the scale hidden states have.
fn tapsBuffer(device: mtl.Device, rows: usize, seed: u64) !mtl.Buffer {
    const b = try device.buffer(rows * row_bytes, mtl.ResourceOptions.shared);
    fill(b, 0, rows * row_bytes / 2, 3.0, seed);
    return b;
}

fn features(taps: mtl.Buffer, start: usize, n: usize) be.Features {
    return .{ .buffer = @intFromPtr(taps.id), .offset = start * row_bytes, .rows = @intCast(n), .row_bytes = row_bytes, .space = .device, .dtype = .bf16 };
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
    try dv.open(s);
    defer dv.release(s);
    var at: usize = 0;
    while (at < n) : (at += chunk) {
        const rows = @min(chunk, n - at);
        try dv.absorb(&.{.{ .stream = s, .features = features(taps, at, rows), .start = at, .follow = &.{}, .prompt = prompt }});
    }
    try dv.hold(&.{.{ .stream = s, .pending = tokenAt(n), .position = n + 1, .depth = depth }});
    try dv.held(&.{s}, &.{out[0..depth]});
    @memcpy(logits, d.captured());
}

test "inkling2 dspark: a context absorbed in any chunking, or with a long prompt's old rows skipped, holds the same block" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.init(arena.allocator()) catch |err| switch (err) {
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
    // skips rows 0 to 476, outside every window the hold attends
    try held(d, &s, taps, n, n, false, 7, &a_props, a_logits);
    for (a_props) |x| try t.expect(x < valid);
    for ([_]struct { chunk: usize, prompt: bool }{ .{ .chunk = 37, .prompt = false }, .{ .chunk = 512, .prompt = true } }) |c| {
        try held(d, &s, taps, n, c.chunk, c.prompt, 7, &b_props, b_logits);
        try t.expectEqualSlices(u32, &a_props, &b_props);
        try t.expectEqualSlices(u16, a_logits, b_logits);
    }
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
        return .{ .stream = &h.stream, .features = features(h.taps, @intCast(start), @intCast(n)), .start = start, .follow = &.{}, .prompt = prompt };
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

test "inkling2 dspark: a hold at every depth 1 to 7 holds exactly the asked drafts, the first of the deepest block's" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.init(arena.allocator()) catch |err| switch (err) {
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
    try t.expectEqual(@as(usize, 2 * 5 * (1031 + 7) * 1024 * 2 + 4608 * 2 + 256 * 2 + 8 * 4), facts.stream_bytes);
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

test "inkling2 dspark: two streams absorbed and held in one round each hold what they hold alone" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const f = Fixture.init(arena.allocator()) catch |err| switch (err) {
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

/// A residency hook that counts the buffers joined and left.
const Counter = struct {
    joined: usize = 0,
    left: usize = 0,

    fn join(ctx: *anyopaque, _: mtl.Buffer) void {
        const c: *Counter = @ptrCast(@alignCast(ctx));
        c.joined += 1;
    }

    fn leave(ctx: *anyopaque, _: mtl.Buffer) void {
        const c: *Counter = @ptrCast(@alignCast(ctx));
        c.left += 1;
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
    try t.expectEqual(scratch + 13, c.joined); // ten rings, the Markov feature, the state and the token row
    try d.captureLogits();
    try t.expectEqual(scratch + 14, c.joined);
    dv.release(&s);
    try t.expectEqual(@as(usize, 13), c.left);
    d.setResident(null);
    try t.expectEqual(c.joined, c.left);
}
