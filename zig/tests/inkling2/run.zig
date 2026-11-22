//! Inkling 2 (EXL3 experts) on real weights: load check, layer dumps, teacher-forced finals, exactness, greedy, decode and prompt-path profiles (usage).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ik = tf.inkling2;
const lanes = tf.lanes;
const settings = @import("settings.zig");

const usage =
    \\tf-inkling-run <model> check [layers]                    load the model (or its first layers) and report it
    \\tf-inkling-run <model> layers <ids.json> <out.bin> <L> [--window] [--chunk N]   the first L layers on ids, fed as prompt chunks of the prefill rows
    \\                                                         (--chunk N sets them; the chunk path's arithmetic): bf16 [L, T, H]
    \\                                                         --window: the first 16 rows as a prompt chunk, the rest in 16-row windows through the decode path
    \\tf-inkling-run <model> finals <index.txt> <out_dir> [--window] [--chunk N]   each listed ids file in prompt chunks of the prefill rows (--chunk N
    \\                                                         sets them): seq-NNNNN.bin with bf16 finals [T, H] (out_dir must exist)
    \\                                                         --window: as layers
    \\tf-inkling-run <model> exact <ids.json> <at> [layers]    ids[at..] (ids at most the prefill rows) after ids[..at]: one-row steps
    \\                                                         against 16-row windows, bit for bit
    \\tf-inkling-run <model> greedy <ids.json> <n> [--drafts]   the prompt, then n one-row greedy steps; tokens, tok/s
    \\                                                         --drafts: the same n tokens through the lane engine with the draft head in
    \\                                                         front of the target, as served. INKLING_DRAFT_CONFIDENCE: a probability above 0
    \\                                                         and below 1, 0.7 when unset; a round holds 7 drafts (INKLING_DRAFT_DEPTH, when
    \\                                                         given, caps them) and verifies the first and each next one while its own
    \\                                                         probability is at least that. off verifies every draft held; INKLING_DRAFT_DEPTH
    \\                                                         is then 1 to 7 or auto, 4 when unset. INKLING_DRAFT_BITS: 8 or 16, 8 when unset.
    \\                                                         Needs mtp.safetensors in the folder; also prints rounds, drafts, accepted
    \\tf-inkling-run <model> taps <index.txt> <out_dir> <steps>   each prompt, then steps greedy tokens: taps, picks, MoE inputs
    \\tf-inkling-run <model> ftaps <ids.json> <out.bin> [at [window]]   every row's serving taps: bf16 [T, 8, H], the inputs of layers
    \\                                                         5, 11, .. 41, then the finals. ids[..at] (default: all) go through in
    \\                                                         prompt chunks, the rest in decode windows of at most `window` rows (16)
    \\tf-inkling-run <model> dspark <taps_dir> <seqs> <out_dir> [--depth d] [--bits 8|16]   the draft head over recorded taps (taps mode's files), from the
    \\                                                         embedding, the unembedding and mtp.safetensors alone: <seqs> is N, A..B (inclusive)
    \\                                                         or a comma list; per sequence out_dir gets seq-NNNNN.props.bin (int32 [T][7],
    \\                                                         -1 where no hold ran), seq-NNNNN.probs.bin (f32 [T][7]: each proposal's top-1
    \\                                                         probability under its step's logits, -1 where no hold ran) and
    \\                                                         seq-NNNNN.logits.bin (bf16 [16][7][vocab]: the first 16
    \\                                                         holds' step logits after the softcap and the padded vocabulary's mask);
    \\                                                         --depth: drafts a hold makes (7 when not given); --bits: the draft layers' weights,
    \\                                                         quantized to 8 bits at load (8 when not given) or bf16
    \\tf-inkling-run <model> profile <layers> <pos> <rows> <steps> [repeat]   per-class GPU time of decode steps
    \\                                                         (the layers run repeat times a step, default 48/layers)
    \\tf-inkling-run <model> wall <layers> <pos> <rows> <steps> [repeat]   decode steps each followed by the greedy draw: median
    \\                                                         wall time and its parts (encode, commit to GPU start, GPU, GPU end to wake, draw)
    \\tf-inkling-run <model> prefill <ids.json> [--chunk N] [--layers L] [--repeat R] [--spans] [--drafts]   the prompt in chunks of N rows through the serving
    \\                                                         prompt path: one JSON line with prompt_ms, tok_s, the GPU ms by class (dense, norm,
    \\                                                         attention_prepare, attention, route, experts, activation, final, taps, absorb), the command
    \\                                                         buffers used and the longest one's GPU ms, and a digest of the last row's hidden state
    \\                                                         (the same at any chunk size: rows are independent)
    \\                                                         --chunk: the prefill rows (below); --layers: the first L layers alone, no final stage or
    \\                                                         logits; --repeat: the loaded layers run R times a chunk, as a deeper model's (the digest then differs);
    \\                                                         --spans: also a line a chunk with its position, for attention's growth with it;
    \\                                                         --drafts: as served with the draft head (INKLING_DRAFT_* as greedy --drafts reads them):
    \\                                                         every chunk taps its rows and runs the final stage on all of them, and the head absorbs
    \\                                                         the prompt's rows but the last
    \\tf-inkling-run <model> partition <ids.json> <cut>... [--layers L] [--drafts]   the prompt twice through the serving prompt path: in chunks of
    \\                                                         the prefill rows, and in chunks of the lengths given (the cuts, then the prefill rows for
    \\                                                         the rest), each leg with its scratch and its cache poisoned with NaN past the rows in use,
    \\                                                         then compares the legs bit for bit: every layer's keys and values, the convolution rings,
    \\                                                         every row's last layer output and expert picks and, on the whole model, the taps, the last
    \\                                                         row's finals and logits and, with --drafts, the first hold's proposals. Exits non-zero
    \\                                                         on a difference or a NaN in either leg's results
    \\INKLING_PREFILL_ROWS: the prefill rows, the rows of a prompt chunk at most, as the server reads it: a multiple of 64 from 64 to 8192, 4096 when unset
    \\INKLING_PROMPT_PATH: the path prompt chunks take, read by every mode that loads a model: per_op (the per-operation kernels, as on a GPU without tensor units), rows or rows:3,7 (the per-row expert projections on every layer or on those layers, as for experts the tensor-unit chain cannot take), auto or unset (what the GPU and the pack allow)
    \\
;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("{s}", .{usage});
        return error.ExpectedModelAndMode;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    if (std.mem.eql(u8, args[2], "check")) return check(init, args);
    if (std.mem.eql(u8, args[2], "layers")) return layersMode(init, args);
    if (std.mem.eql(u8, args[2], "taps")) return tapsMode(init, args);
    if (std.mem.eql(u8, args[2], "ftaps")) return ftapsMode(init, args);
    if (std.mem.eql(u8, args[2], "dspark")) return @import("replay.zig").run(init, args);
    if (std.mem.eql(u8, args[2], "finals")) return finalsMode(init, args);
    if (std.mem.eql(u8, args[2], "exact")) return exactMode(init, args);
    if (std.mem.eql(u8, args[2], "greedy")) return greedyMode(init, args);
    if (std.mem.eql(u8, args[2], "profile")) return profileMode(init, args);
    if (std.mem.eql(u8, args[2], "wall")) return wallMode(init, args);
    if (std.mem.eql(u8, args[2], "prefill")) return prefillMode(init, args);
    if (std.mem.eql(u8, args[2], "partition")) return partitionMode(init, args);
    std.debug.print("{s}", .{usage});
    return error.UnknownMode;
}

fn check(init: std.process.Init, args: []const [:0]const u8) !void {
    const layers: ?u32 = if (args.len > 3) try std.fmt.parseInt(u32, args[3], 10) else null;
    const t0 = now();
    const m = try ik.Model.load(init.gpa, init.io, args[1], .{ .layers = layers });
    defer m.deinit();
    var bytes: usize = 0;
    for (m.checkpoint.shards.items) |sh| bytes += sh.bytes;
    const w = m.weights.widths;
    std.debug.print("{{\"layers\": {d}, \"expert_files\": {d}, \"k2\": {d}, \"k3\": {d}, \"k4\": {d}, \"gib\": {d:.2}, \"seconds\": {d:.1}}}\n", .{ m.layers, m.weights.expert_buffers.len, w[2], w[3], w[4], @as(f64, @floatFromInt(bytes)) / (1 << 30), @as(f64, @floatFromInt(now() - t0)) / 1e9 });
}

fn readIds(init: std.process.Init, path: []const u8) !std.json.Parsed([]u32) {
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, path, init.gpa, .limited(64 << 20));
    defer init.gpa.free(text);
    return std.json.parseFromSlice([]u32, init.gpa, text, .{});
}

/// The load options every mode shares: the first `layers` layers (all when null) and the prefill rows INKLING_PREFILL_ROWS names.
fn options(layers: ?u32) !ik.model.Options {
    return chunkOptions(layers, null);
}

/// The options with `chunk` as the prefill rows when given.
fn chunkOptions(layers: ?u32, chunk: ?u32) !ik.model.Options {
    return .{ .layers = layers, .prefill_rows = chunk orelse try settings.prefillRows() };
}

/// ids at the cache's end in chunks of the model's prefill rows; finals rows copied into `finals` when given; logits of the last.
fn feed(m: *const ik.Model, s: *ik.state.Scratch, cache: *ik.state.Cache, ids: []const u32, finals: ?[]u16, logits: bool) !void {
    const H: usize = m.config.text.hidden;
    var at: usize = 0;
    while (at < ids.len) {
        const n = @min(ids.len - at, m.prefill_rows);
        const last = at + n == ids.len;
        try ik.forward.run(m, s, cache, ids[at..][0..n], .{ .finals = finals != null, .logits = if (logits and last) .last else .none, .prompt = true });
        if (finals) |f| @memcpy(f[at * H ..][0 .. n * H], s.finals.slice(u16, n * H));
        cache.len += n;
        at += n;
    }
}

/// Where a recorded run keeps what it records: finals rows into `finals` (T rows), and each loaded layer's output
/// rows into `layers` ([L][T][H]), the latter through `scratch`, a dump buffer of one piece.
const Record = struct { finals: ?[]u16 = null, layers: ?[]u16 = null, scratch: ?mtl.Buffer = null };

/// How a recorded run cuts its ids: `chunks` feeds prompt chunks of the model's prefill rows, `windows` the way a
/// decode-time client sends them: the first window_rows rows as a prompt chunk, every later row in windows of window_rows
/// through the decode path.
const Pieces = enum { chunks, windows };

/// ids at the cache's end in pieces (the last one possibly shorter), what `rec` names recorded after each.
fn feedRecorded(m: *const ik.Model, s: *ik.state.Scratch, cache: *ik.state.Cache, ids: []const u32, rec: Record, pieces: Pieces) !void {
    const H: usize = m.config.text.hidden;
    const T = ids.len;
    var at: usize = 0;
    while (at < T) {
        const n = @min(T - at, if (pieces == .windows) ik.state.window_rows else m.prefill_rows);
        try ik.forward.run(m, s, cache, ids[at..][0..n], .{ .finals = rec.finals != null, .logits = .none, .dump = rec.scratch, .prompt = pieces == .chunks or at == 0 });
        if (rec.finals) |f| @memcpy(f[at * H ..][0 .. n * H], s.finals.slice(u16, n * H));
        if (rec.layers) |d| {
            const w = rec.scratch.?.slice(u16, m.layers * n * H);
            for (0..m.layers) |l| @memcpy(d[(l * T + at) * H ..][0 .. n * H], w[l * n * H ..][0 .. n * H]);
        }
        cache.len += n;
        at += n;
    }
}

/// What `layers` and `finals` take after their fixed arguments.
const FeedArgs = struct { window: bool = false, chunk: ?u32 = null };

fn feedArgs(args: []const [:0]const u8) !FeedArgs {
    var p: FeedArgs = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--window")) {
            p.window = true;
        } else if (std.mem.eql(u8, args[i], "--chunk")) {
            p.chunk = try optionValue(args, &i);
        } else return error.UnknownOption;
    }
    if (p.window and p.chunk != null) return error.ChunkSizeIsForTheChunkPath;
    return p;
}

fn layersMode(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len < 6) return error.ExpectedIdsOutLayers;
    const opt = try feedArgs(args[6..]);
    const L = try std.fmt.parseInt(u32, args[5], 10);
    const ids = try readIds(init, args[3]);
    defer ids.deinit();
    const m = try ik.Model.load(init.gpa, init.io, args[1], try chunkOptions(L, opt.chunk));
    defer m.deinit();
    const c = &m.config.text;
    const n = ids.value.len;
    const capacity = n + 1;
    var cache = try ik.state.Cache.init(m.device, c, capacity, L, m.prefill_rows);
    defer cache.deinit();
    var s = try ik.state.Scratch.init(m.device, c, capacity, m.chunking());
    defer s.deinit();
    const out = try init.gpa.alloc(u16, @as(usize, L) * n * c.hidden);
    defer init.gpa.free(out);
    const rows = @min(n, if (opt.window) ik.state.window_rows else m.prefill_rows);
    const scratch = try m.device.buffer(@as(usize, L) * rows * c.hidden * 2, mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const t0 = now();
    try feedRecorded(m, &s, &cache, ids.value, .{ .layers = out, .scratch = scratch }, if (opt.window) .windows else .chunks);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = std.mem.sliceAsBytes(out) });
    const seconds = @as(f64, @floatFromInt(now() - t0)) / 1e9;
    if (opt.window) {
        std.debug.print("{{\"layers\": {d}, \"tokens\": {d}, \"seconds\": {d:.2}, \"window\": true}}\n", .{ L, n, seconds });
    } else {
        std.debug.print("{{\"layers\": {d}, \"tokens\": {d}, \"seconds\": {d:.2}, \"chunk\": {d}}}\n", .{ L, n, seconds, m.prefill_rows });
    }
}

fn finalsMode(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len < 5) return error.ExpectedIndexOutDir;
    const opt = try feedArgs(args[5..]);
    const index = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], init.gpa, .limited(1 << 20));
    defer init.gpa.free(index);
    var paths: std.ArrayList([]const u8) = .empty;
    defer paths.deinit(init.gpa);
    var longest: usize = 0;
    var it = std.mem.tokenizeAny(u8, index, "\r\n");
    while (it.next()) |p| {
        try paths.append(init.gpa, p);
        const ids = try readIds(init, p);
        defer ids.deinit();
        longest = @max(longest, ids.value.len);
    }
    const m = try ik.Model.load(init.gpa, init.io, args[1], try chunkOptions(null, opt.chunk));
    defer m.deinit();
    const c = &m.config.text;
    const capacity = longest + 1;
    var cache = try ik.state.Cache.init(m.device, c, capacity, c.layers, m.prefill_rows);
    defer cache.deinit();
    var s = try ik.state.Scratch.init(m.device, c, capacity, m.chunking());
    defer s.deinit();
    const finals = try init.gpa.alloc(u16, longest * c.hidden);
    defer init.gpa.free(finals);
    for (paths.items, 0..) |p, i| {
        const ids = try readIds(init, p);
        defer ids.deinit();
        const t0 = now();
        cache.len = 0;
        if (opt.window) try feedRecorded(m, &s, &cache, ids.value, .{ .finals = finals }, .windows) else try feed(m, &s, &cache, ids.value, finals, false);
        const out = try std.fmt.allocPrint(init.gpa, "{s}/seq-{d:0>5}.bin", .{ args[4], i });
        defer init.gpa.free(out);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out, .data = std.mem.sliceAsBytes(finals[0 .. ids.value.len * c.hidden]) });
        std.debug.print("{{\"seq\": {d}, \"tokens\": {d}, \"seconds\": {d:.1}}}\n", .{ i, ids.value.len, @as(f64, @floatFromInt(now() - t0)) / 1e9 });
    }
}

/// ids[at..] two ways after the prompt ids[..at]: one-row steps and 16-row windows. The last loaded layer's output rows
/// (and the finals, on the whole model) must agree bit for bit.
fn exactMode(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len < 5) return error.ExpectedIdsAt;
    const at = try std.fmt.parseInt(usize, args[4], 10);
    const L: ?u32 = if (args.len > 5) try std.fmt.parseInt(u32, args[5], 10) else null;
    const ids = try readIds(init, args[3]);
    defer ids.deinit();
    const all = ids.value;
    if (at == 0 or at >= all.len) return error.ExpectedAtInsideIds;
    const m = try ik.Model.load(init.gpa, init.io, args[1], try options(L));
    defer m.deinit();
    if (all.len > m.prefill_rows) return error.IdsExceedThePrefillRows;
    const c = &m.config.text;
    const H: usize = c.hidden;
    const full = m.layers == c.layers;
    const cont = all.len - at;
    const capacity = all.len + 1;
    var s = try ik.state.Scratch.init(m.device, c, capacity, m.chunking());
    defer s.deinit();
    var outs: [2][]u16 = undefined;
    for (&outs) |*o| o.* = try init.gpa.alloc(u16, cont * H * 2);
    defer for (outs) |o| init.gpa.free(o);
    for (0..2) |way| {
        var cache = try ik.state.Cache.init(m.device, c, capacity, m.layers, m.prefill_rows);
        defer cache.deinit();
        const rows: usize = if (way == 0) 1 else ik.state.window_rows;
        try ik.forward.run(m, &s, &cache, all[0..at], .{ .finals = full, .logits = .none, .prompt = true });
        cache.len = at;
        var i: usize = at;
        while (i < all.len) {
            const n = @min(all.len - i, rows);
            try ik.forward.run(m, &s, &cache, all[i..][0..n], .{ .finals = full, .logits = .none });
            for (0..n) |r| {
                const k = i + r - at;
                @memcpy(outs[way][k * H ..][0..H], s.h.slice(u16, n * H)[r * H ..][0..H]);
                if (full) @memcpy(outs[way][(cont + k) * H ..][0..H], s.finals.slice(u16, n * H)[r * H ..][0..H]);
            }
            cache.len += n;
            i += n;
        }
    }
    var differ: usize = 0;
    for (0..cont) |k| {
        const span: usize = if (full) 2 else 1;
        for (0..span) |part| {
            const off = (part * cont + k) * H;
            if (!std.mem.eql(u16, outs[0][off..][0..H], outs[1][off..][0..H])) {
                differ += 1;
                if (differ <= 4) std.debug.print("row {d} ({s}) differs between steps and windows\n", .{ at + k, if (part == 0) "layer output" else "finals" });
            }
        }
    }
    std.debug.print("{{\"exact_rows\": {d}, \"differ\": {d}, \"layers\": {d}}}\n", .{ cont, differ, m.layers });
    if (differ != 0) return error.Inkling2NotRowExact;
}

fn greedyMode(init: std.process.Init, args: []const [:0]const u8) !void {
    const drafts = args.len == 6 and std.mem.eql(u8, args[5], "--drafts");
    if (args.len != 5 and !drafts) return error.ExpectedIdsStepsAndMaybeDrafts;
    const steps = try std.fmt.parseInt(usize, args[4], 10);
    const ids = try readIds(init, args[3]);
    defer ids.deinit();
    if (drafts) return @import("drafted.zig").greedy(init, args[1], ids.value, steps);
    const m = try ik.Model.load(init.gpa, init.io, args[1], try options(null));
    defer m.deinit();
    const c = &m.config.text;
    const capacity = ids.value.len + steps + 1;
    var cache = try ik.state.Cache.init(m.device, c, capacity, c.layers, m.prefill_rows);
    defer cache.deinit();
    var s = try ik.state.Scratch.init(m.device, c, capacity, m.chunking());
    defer s.deinit();
    var t0 = now();
    try feed(m, &s, &cache, ids.value, null, true);
    const prompt_ms = @as(f64, @floatFromInt(now() - t0)) / 1e6;
    const picks = try init.gpa.alloc(u32, steps);
    defer init.gpa.free(picks);
    var tm: ik.forward.Timing = .{};
    t0 = now();
    for (picks) |*p| {
        p.* = ik.greedy.argmax(s.logits.slice(u16, c.vocab));
        try ik.forward.run(m, &s, &cache, &.{p.*}, .{ .timing = &tm });
        cache.len += 1;
    }
    const secs = @as(f64, @floatFromInt(now() - t0)) / 1e9;
    var w: std.Io.Writer.Allocating = .init(init.gpa);
    defer w.deinit();
    try std.json.Stringify.value(picks, .{}, &w.writer);
    std.debug.print("{s}\n{{\"prompt_tokens\": {d}, \"prompt_ms\": {d:.0}, \"steps\": {d}, \"tok_s\": {d:.2}, \"gpu_ms\": {d:.1}, \"encode_ms\": {d:.1}}}\n", .{ w.written(), ids.value.len, prompt_ms, steps, @as(f64, @floatFromInt(steps)) / secs, tm.gpu * 1e3 / @as(f64, @floatFromInt(steps)), tm.encode * 1e3 / @as(f64, @floatFromInt(steps)) });
}

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Per listed prompt: the prompt in chunks, then greedy steps; every fed row's taps (layers 5..41 and the finals),
/// every layer's expert picks and the MoE inputs of layers 0 and 30, written beside the tokens.
fn tapsMode(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len != 6) return error.ExpectedIndexOutDirSteps;
    const steps = try std.fmt.parseInt(usize, args[5], 10);
    if (steps < 2) return error.AtLeastTwoSteps;
    const index = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], init.gpa, .limited(1 << 20));
    defer init.gpa.free(index);
    const m = try ik.Model.load(init.gpa, init.io, args[1], try options(null));
    defer m.deinit();
    const c = &m.config.text;
    const H: usize = c.hidden;
    const L: usize = c.layers;
    const R: usize = m.prefill_rows;
    // the longest cache a prompt and its steps fill
    var longest: usize = 0;
    var scan = std.mem.tokenizeAny(u8, index, "\r\n");
    while (scan.next()) |path| {
        const ids = try readIds(init, path);
        defer ids.deinit();
        longest = @max(longest, ids.value.len + steps);
    }
    var s = try ik.state.Scratch.init(m.device, c, longest, m.chunking());
    defer s.deinit();
    const tp: ik.forward.Taps = .{
        .h = try m.device.buffer(ik.forward.tap_layers.len * R * H * 2, mtl.ResourceOptions.shared),
        .eids = try m.device.buffer(L * R * 32, mtl.ResourceOptions.shared),
        .xm = try m.device.buffer(ik.forward.xm_layers.len * R * H * 2, mtl.ResourceOptions.shared),
    };
    defer tp.h.deinit();
    defer tp.eids.deinit();
    defer tp.xm.?.deinit();
    var it = std.mem.tokenizeAny(u8, index, "\r\n");
    var seq: usize = 0;
    while (it.next()) |path| : (seq += 1) {
        const ids = try readIds(init, path);
        defer ids.deinit();
        const P = ids.value.len;
        const T = P + steps - 1;
        const taps = try init.gpa.alloc(u16, T * 8 * H);
        defer init.gpa.free(taps);
        const eids = try init.gpa.alloc(u32, T * L * 8);
        defer init.gpa.free(eids);
        const xm = try init.gpa.alloc(u16, T * 2 * H);
        defer init.gpa.free(xm);
        var tokens: std.ArrayList(u32) = .empty;
        defer tokens.deinit(init.gpa);
        try tokens.appendSlice(init.gpa, ids.value);
        var cache = try ik.state.Cache.init(m.device, c, T + 1, L, m.prefill_rows);
        defer cache.deinit();
        const t0 = now();
        var row: usize = 0;
        var at: usize = 0;
        while (at < P) {
            const n = @min(P - at, R);
            try ik.forward.run(m, &s, &cache, ids.value[at..][0..n], .{ .finals = true, .logits = if (at + n == P) .last else .none, .taps = tp, .prompt = true });
            gather(&s, tp, n, row, H, L, taps, eids, xm);
            cache.len += n;
            row += n;
            at += n;
        }
        for (0..steps) |k| {
            const pick = ik.greedy.argmax(s.logits.slice(u16, c.vocab));
            try tokens.append(init.gpa, pick);
            if (k + 1 == steps) break;
            try ik.forward.run(m, &s, &cache, &.{pick}, .{ .finals = true, .taps = tp });
            gather(&s, tp, 1, row, H, L, taps, eids, xm);
            cache.len += 1;
            row += 1;
        }
        std.debug.assert(row == T);
        try writeSeq(init, args[4], seq, "taps.bin", std.mem.sliceAsBytes(taps));
        try writeSeq(init, args[4], seq, "eids.bin", std.mem.sliceAsBytes(eids));
        try writeSeq(init, args[4], seq, "xm.bin", std.mem.sliceAsBytes(xm));
        var w: std.Io.Writer.Allocating = .init(init.gpa);
        defer w.deinit();
        try std.json.Stringify.value(.{ .prompt = P, .tokens = tokens.items }, .{}, &w.writer);
        try writeSeq(init, args[4], seq, "tokens.json", w.written());
        std.debug.print("{{\"seq\": {d}, \"prompt\": {d}, \"rows\": {d}, \"seconds\": {d:.1}}}\n", .{ seq, P, T, @as(f64, @floatFromInt(now() - t0)) / 1e9 });
    }
}

/// The `n` rows just run, from the capture buffers and s.finals into the host arrays at `row`.
fn gather(s: *ik.state.Scratch, tp: ik.forward.Taps, n: usize, row: usize, H: usize, L: usize, taps: []u16, eids: []u32, xm: []u16) void {
    const th = tp.h.slice(u16, ik.forward.tap_layers.len * n * H);
    const fin = s.finals.slice(u16, n * H);
    const te = tp.eids.slice(u32, L * n * 8);
    const tx = tp.xm.?.slice(u16, ik.forward.xm_layers.len * n * H);
    for (0..n) |r| {
        const dst = taps[(row + r) * 8 * H ..][0 .. 8 * H];
        for (0..ik.forward.tap_layers.len) |k| @memcpy(dst[k * H ..][0..H], th[(k * n + r) * H ..][0..H]);
        @memcpy(dst[7 * H ..][0..H], fin[r * H ..][0..H]);
        for (0..L) |l| @memcpy(eids[((row + r) * L + l) * 8 ..][0..8], te[(l * n + r) * 8 ..][0..8]);
        for (0..ik.forward.xm_layers.len) |k| @memcpy(xm[((row + r) * 2 + k) * H ..][0..H], tx[(k * n + r) * H ..][0..H]);
    }
}

fn writeSeq(init: std.process.Init, dir: []const u8, seq: usize, suffix: []const u8, data: []const u8) !void {
    const path = try std.fmt.allocPrint(init.gpa, "{s}/seq-{d:0>5}.{s}", .{ dir, seq, suffix });
    defer init.gpa.free(path);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = data });
}

/// ids[..at] in prompt chunks, then the rest in decode windows of at most `window` rows, every forward writing its rows' serving
/// taps (what the drafter reads): bf16 [T][8][H] to the file, the layout of `taps` mode's taps.bin.
fn ftapsMode(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len < 5 or args.len > 7) return error.ExpectedIdsOutAtWindow;
    const ids = try readIds(init, args[3]);
    defer ids.deinit();
    const T = ids.value.len;
    const at = if (args.len > 5) try std.fmt.parseInt(usize, args[5], 10) else T;
    const window = if (args.len > 6) try std.fmt.parseInt(usize, args[6], 10) else ik.state.window_rows;
    if (T == 0 or at == 0 or at > T or window == 0 or window > ik.state.window_rows) return error.ExpectedAtInsideIdsAndWindowOneTo16;
    const m = try ik.Model.load(init.gpa, init.io, args[1], try options(null));
    defer m.deinit();
    const c = &m.config.text;
    const row: usize = ik.state.feature_taps * @as(usize, c.hidden);
    var s = try ik.state.Scratch.init(m.device, c, T + 1, m.chunking());
    defer s.deinit();
    var cache = try ik.state.Cache.init(m.device, c, T + 1, c.layers, m.prefill_rows);
    defer cache.deinit();
    const taps = try init.gpa.alloc(u16, T * row);
    defer init.gpa.free(taps);
    const t0 = now();
    var done: usize = 0;
    while (done < T) {
        const prompt = done < at;
        const n = if (prompt) @min(at - done, m.prefill_rows) else @min(T - done, window);
        const buf = if (prompt) s.ptaps else s.ftaps;
        try ik.forward.run(m, &s, &cache, ids.value[done..][0..n], .{ .logits = .none, .prompt = prompt, .feature_taps = .{ .buf = buf } });
        @memcpy(taps[done * row ..][0 .. n * row], buf.slice(u16, n * row));
        cache.len += n;
        done += n;
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = std.mem.sliceAsBytes(taps) });
    std.debug.print("{{\"rows\": {d}, \"prompt\": {d}, \"window\": {d}, \"seconds\": {d:.2}}}\n", .{ T, at, window, @as(f64, @floatFromInt(now() - t0)) / 1e9 });
}

const profile_classes = [_]ik.forward.counters.Class{ .dense, .norm, .attention_prepare, .attention, .route, .experts, .activation, .final };
const profile_warmup = 4;

/// Decode steps of `rows` rows at `pos` through the first `layers` layers, each step's layers repeated `repeat` times in
/// one command buffer: the mean per-class GPU time after the warm-up steps, and the same steps without spans.
fn profileMode(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len < 7 or args.len > 8) return error.ExpectedLayersPosRowsSteps;
    const L = try std.fmt.parseInt(u32, args[3], 10);
    const pos = try std.fmt.parseInt(usize, args[4], 10);
    const rows = try std.fmt.parseInt(usize, args[5], 10);
    const steps = try std.fmt.parseInt(usize, args[6], 10);
    const repeat: u32 = if (args.len > 7) try std.fmt.parseInt(u32, args[7], 10) else @max(1, 48 / L);
    if (L == 0 or rows == 0 or repeat == 0) return error.ExpectedPositiveLayersRowsRepeat;
    if (steps <= profile_warmup) return error.StepsMustExceedWarmup;
    const m = try ik.Model.load(init.gpa, init.io, args[1], try options(L));
    defer m.deinit();
    if (rows > m.prefill_rows) return error.RowsExceedThePrefillRows;
    const c = &m.config.text;
    const full = m.layers == c.layers;
    const capacity = pos + rows * steps + 1;
    var cache = try ik.state.Cache.init(m.device, c, capacity, m.layers, m.prefill_rows);
    defer cache.deinit();
    var s = try ik.state.Scratch.init(m.device, c, capacity, m.chunking());
    defer s.deinit();
    const zeros = try init.gpa.alloc(u32, @max(pos, rows));
    defer init.gpa.free(zeros);
    @memset(zeros, 0);
    try feed(m, &s, &cache, zeros[0..pos], null, false);
    const max_spans = ik.forward.spans_per_layer * @as(usize, m.layers) * repeat + 16;
    var rec = try ik.forward.counters.Recorder.init(init.gpa, m.device, .{ .enabled = true, .max_spans = max_spans });
    defer rec.deinit() catch {};
    const measured = steps - profile_warmup;
    var secs: [ik.forward.counters.class_count]f64 = @splat(0);
    var gpu: f64 = 0;
    for (0..steps) |step| {
        var tm: ik.forward.Timing = .{};
        try ik.forward.run(m, &s, &cache, zeros[0..rows], .{ .logits = if (full) .last else .none, .timing = &tm, .spans = &rec, .repeat = repeat });
        _ = try rec.readAfterCompletion();
        if (step >= profile_warmup) {
            gpu += tm.gpu;
            for (rec.aggregates(), 0..) |a, i| secs[i] += a.gpu_seconds;
        }
        cache.len += rows;
    }
    cache.len = pos;
    var plain: f64 = 0;
    for (0..steps) |step| {
        var tm: ik.forward.Timing = .{};
        try ik.forward.run(m, &s, &cache, zeros[0..rows], .{ .logits = if (full) .last else .none, .timing = &tm, .repeat = repeat });
        if (step >= profile_warmup) plain += tm.gpu;
        cache.len += rows;
    }
    const per_step = 1e6 / @as(f64, @floatFromInt(measured));
    const per_layer = per_step / @as(f64, @floatFromInt(@as(usize, m.layers) * repeat));
    var sum: f64 = 0;
    for (profile_classes) |k| {
        const t = secs[@backingInt(k)];
        sum += t;
        std.debug.print("{{\"class\": \"{s}\", \"us_per_layer\": {d:.1}, \"us_per_step\": {d:.1}}}\n", .{ @tagName(k), t * per_layer, t * per_step });
    }
    const ms = 1e3 / @as(f64, @floatFromInt(measured));
    std.debug.print("{{\"steps\": {d}, \"rows\": {d}, \"pos\": {d}, \"repeat\": {d}, \"gpu_ms_per_step\": {d:.3}, \"wall_gpu_ms_plain\": {d:.3}, \"class_sum_ms_per_step\": {d:.3}}}\n", .{ measured, rows, pos, repeat, gpu * ms, plain * ms, sum * ms });
}

fn median(v: []f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return v[v.len / 2];
}

/// Plain decode steps of `rows` rows at `pos` through the first `layers` layers (each step's layers repeated `repeat`
/// times), the greedy draw after each as the serving path takes it: the median over the steps after the warm-up of a
/// step's wall time and of its parts, host entry to commit, commit to the GPU's start, the GPU's run, its end to the
/// host's wake-up, and the rest to the draw's end. A partial model draws from a row of noise.
fn wallMode(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len < 7 or args.len > 8) return error.ExpectedLayersPosRowsSteps;
    const L = try std.fmt.parseInt(u32, args[3], 10);
    const pos = try std.fmt.parseInt(usize, args[4], 10);
    const rows = try std.fmt.parseInt(usize, args[5], 10);
    const steps = try std.fmt.parseInt(usize, args[6], 10);
    const repeat: u32 = if (args.len > 7) try std.fmt.parseInt(u32, args[7], 10) else @max(1, 48 / L);
    if (L == 0 or rows == 0 or rows > ik.state.window_rows or repeat == 0) return error.ExpectedPositiveLayersRowsRepeat;
    if (steps <= profile_warmup) return error.StepsMustExceedWarmup;
    const m = try ik.Model.load(init.gpa, init.io, args[1], try options(L));
    defer m.deinit();
    const c = &m.config.text;
    const full = m.layers == c.layers;
    const capacity = pos + rows * steps + 1;
    var cache = try ik.state.Cache.init(m.device, c, capacity, m.layers, m.prefill_rows);
    defer cache.deinit();
    var s = try ik.state.Scratch.init(m.device, c, capacity, m.chunking());
    defer s.deinit();
    const zeros = try init.gpa.alloc(u32, @max(pos, rows));
    defer init.gpa.free(zeros);
    @memset(zeros, 0);
    try feed(m, &s, &cache, zeros[0..pos], null, false);
    var prng = std.Random.DefaultPrng.init(0x57ee);
    for (s.logits.slice(u16, c.vocab)) |*w| w.* = prng.random().int(u16);
    const parts = [_][]const u8{ "wall", "encode", "commit_to_start", "gpu", "end_to_wake", "draw" };
    var samples: [parts.len][]f64 = undefined;
    for (&samples) |*v| v.* = try init.gpa.alloc(f64, steps - profile_warmup);
    defer for (samples) |v| init.gpa.free(v);
    for (0..steps) |step| {
        var tm: ik.forward.Timing = .{};
        const t0 = ik.forward.seconds();
        try ik.forward.run(m, &s, &cache, zeros[0..rows], .{ .logits = if (full) .last else .none, .timing = &tm, .repeat = repeat });
        const pick = ik.greedy.argmax(s.logits.slice(u16, c.vocab));
        const t1 = ik.forward.seconds();
        std.mem.doNotOptimizeAway(pick);
        cache.len += rows;
        if (step < profile_warmup) continue;
        const k = step - profile_warmup;
        const at = [_]f64{ t1 - t0, tm.committed - t0, tm.started - tm.committed, tm.ended - tm.started, tm.woke - tm.ended, t1 - tm.woke };
        for (&samples, at) |v, x| v[k] = x * 1e3;
    }
    var med: [parts.len]f64 = undefined;
    for (&med, samples) |*o, v| o.* = median(v);
    var mean: f64 = 0;
    for (samples[0]) |x| mean += x;
    mean /= @floatFromInt(samples[0].len);
    std.debug.print("{{\"steps\": {d}, \"rows\": {d}, \"pos\": {d}, \"repeat\": {d}", .{ steps - profile_warmup, rows, pos, repeat });
    for (parts, med) |p, x| std.debug.print(", \"{s}_ms\": {d:.3}", .{ p, x });
    const tenth = samples[0].len / 10;
    // the last step's hidden rows, so two configurations of the same steps can be compared bit for bit
    const digest = std.hash.Wyhash.hash(0, s.h.slice(u8, rows * c.hidden * 2));
    std.debug.print(", \"wall_mean_ms\": {d:.3}, \"wall_min_ms\": {d:.3}, \"wall_p10_ms\": {d:.3}, \"gpu_min_ms\": {d:.3}, \"gpu_p10_ms\": {d:.3}, \"tok_s\": {d:.2}, \"h_digest\": \"{x}\"}}\n", .{ mean, samples[0][0], samples[0][tenth], samples[3][0], samples[3][tenth], 1e3 / med[0], digest });
}

const prefill_classes = [_]ik.forward.counters.Class{ .dense, .norm, .attention_prepare, .attention, .route, .experts, .activation, .final, .taps, .absorb };
const class_count = ik.forward.counters.class_count;
const absorb_class = @backingInt(ik.forward.counters.Class.absorb);

/// {"dense": ms, ...} over `prefill_classes`, written at the current line's end, from GPU seconds by class.
fn printClasses(secs: *const [class_count]f64) void {
    std.debug.print("{{", .{});
    for (prefill_classes, 0..) |k, i| std.debug.print("{s}\"{s}\": {d:.2}", .{ if (i == 0) "" else ", ", @tagName(k), secs[@backingInt(k)] * 1e3 });
    std.debug.print("}}", .{});
}

/// What `prefill` takes after the ids file.
const PrefillArgs = struct { chunk: ?u32 = null, layers: ?u32 = null, repeat: u32 = 1, spans: bool = false, drafts: bool = false };

fn optionValue(args: []const [:0]const u8, at: *usize) !u32 {
    at.* += 1;
    if (at.* == args.len) return error.ExpectedAValueAfterTheOption;
    return std.fmt.parseInt(u32, args[at.*], 10);
}

fn prefillArgs(args: []const [:0]const u8) !PrefillArgs {
    var p: PrefillArgs = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--spans")) {
            p.spans = true;
        } else if (std.mem.eql(u8, args[i], "--drafts")) {
            p.drafts = true;
        } else if (std.mem.eql(u8, args[i], "--chunk")) {
            p.chunk = try optionValue(args, &i);
        } else if (std.mem.eql(u8, args[i], "--layers")) {
            p.layers = try optionValue(args, &i);
        } else if (std.mem.eql(u8, args[i], "--repeat")) {
            p.repeat = try optionValue(args, &i);
        } else return error.UnknownPrefillOption;
    }
    if (p.layers == 0 or p.repeat == 0) return error.ExpectedPositiveLayersRepeat;
    if (p.drafts and (p.layers != null or p.repeat != 1)) return error.DraftsNeedTheWholeModel;
    return p;
}

const Absorber = @import("absorber.zig").Absorber;

/// The prompt through the serving prompt path in two partitions, compared bit for bit (partition.zig).
fn partitionMode(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len < 5) return error.ExpectedIdsAndCuts;
    const ids = try readIds(init, args[3]);
    defer ids.deinit();
    return @import("partition.zig").run(init, args[1], ids.value, args[4..]);
}

/// The prompt in chunks of the prefill rows through the serving prompt path (`.prompt = true`, the last row's logits after the
/// last chunk), each forward's stage groups timed by class: one line of totals and, with --spans, a line a chunk. With --drafts
/// the model also loads the draft head and runs the path as the drafted lane backend does: every chunk taps its rows, so the
/// final stage runs on all of them, and the head absorbs each chunk's rows but the prompt's last (the class `absorb`: its own
/// command buffers' GPU time). A 16-row window runs first, as the server's warm-up does.
fn prefillMode(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len < 4) return error.ExpectedIds;
    const opt = try prefillArgs(args[4..]);
    const ids = try readIds(init, args[3]);
    defer ids.deinit();
    const T = ids.value.len;
    if (T == 0) return error.ExpectedAPrompt;
    const gpa = init.gpa;
    const draft: ?settings.Draft = if (opt.drafts) try settings.draft() else null;
    var load = try chunkOptions(opt.layers, opt.chunk);
    load.drafts = opt.drafts;
    load.draft_bits = if (draft) |d| d.bits else .default;
    const m = try ik.Model.load(gpa, init.io, args[1], load);
    defer m.deinit();
    const c = &m.config.text;
    const full = m.layers == c.layers;
    var cache = try ik.state.Cache.init(m.device, c, T + 1, m.layers, m.prefill_rows);
    defer cache.deinit();
    var s = try ik.state.Scratch.init(m.device, c, @max(T + 1, ik.state.window_rows), m.chunking());
    defer s.deinit();
    const absorber: ?Absorber = if (draft) |d| try Absorber.init(gpa, m, d, ids.value) else null;
    defer if (absorber) |a| a.deinit();
    const taps: ?ik.forward.B = if (absorber != null) .{ .buf = s.ptaps } else null;
    // the weights' first pass is slower than the next ones: the server takes it before a first request (backend.Metal.warm)
    {
        var warm = try ik.state.Cache.init(m.device, c, 32, m.layers, m.prefill_rows);
        defer warm.deinit();
        var warm_ids: [ik.state.window_rows]u32 = undefined;
        for (&warm_ids, 0..) |*x, i| x.* = @intCast(1000 + i * 37);
        try ik.forward.run(m, &s, &warm, &warm_ids, .{ .logits = if (full) .all else .none });
    }
    const max_spans = ik.forward.spans_per_layer * @as(usize, m.layers) * opt.repeat + 32;
    var rec = try ik.forward.counters.Recorder.init(gpa, m.device, .{ .enabled = true, .max_spans = max_spans });
    defer rec.deinit() catch {};
    var totals: [class_count]f64 = @splat(0);
    var gpu: f64 = 0;
    var encode: f64 = 0;
    var buffers: u32 = 0;
    var longest: f64 = 0;
    var last: usize = 0;
    var at: usize = 0;
    var chunk: usize = 0;
    const t0 = now();
    while (at < T) : (chunk += 1) {
        const n = @min(T - at, m.prefill_rows);
        var tm: ik.forward.Timing = .{};
        try ik.forward.run(m, &s, &cache, ids.value[at..][0..n], .{ .logits = if (at + n == T and full) .last else .none, .prompt = true, .feature_taps = taps, .timing = &tm, .spans = &rec, .repeat = opt.repeat });
        _ = try rec.readAfterCompletion();
        var secs: [class_count]f64 = undefined;
        for (rec.aggregates(), &secs) |a, *x| x.* = a.gpu_seconds;
        var chunk_gpu = tm.gpu;
        var chunk_buffers = tm.buffers;
        var chunk_longest = tm.longest;
        if (absorber) |a| {
            secs[absorb_class] = try a.take(s.ptaps, at, n);
            chunk_gpu += secs[absorb_class];
            // the head's absorb is a command buffer of its own, when it keeps any of the chunk's rows
            if (secs[absorb_class] > 0) {
                chunk_buffers += 1;
                chunk_longest = @max(chunk_longest, secs[absorb_class]);
            }
        }
        for (secs, &totals) |x, *y| y.* += x;
        gpu += chunk_gpu;
        encode += tm.encode;
        buffers += chunk_buffers;
        longest = @max(longest, chunk_longest);
        last = n;
        if (opt.spans) {
            std.debug.print("{{\"chunk\": {d}, \"pos\": {d}, \"rows\": {d}, \"gpu_ms\": {d:.1}, \"buffers\": {d}, \"max_buffer_ms\": {d:.1}, \"classes\": ", .{ chunk, at, n, chunk_gpu * 1e3, chunk_buffers, chunk_longest * 1e3 });
            printClasses(&secs);
            std.debug.print("}}\n", .{});
        }
        cache.len += n;
        at += n;
    }
    const prompt_s = @as(f64, @floatFromInt(now() - t0)) / 1e9;
    // the last row's hidden state after the last layer: a row's bits do not depend on the chunk it was fed in
    const row_bytes = @as(usize, c.hidden) * 2;
    const digest = std.hash.Wyhash.hash(0, s.h.slice(u8, last * row_bytes)[(last - 1) * row_bytes ..][0..row_bytes]);
    // the command buffers run one after another: their GPU times cannot add up to more than the prompt's wall time
    if (gpu > prompt_s * 1.02) {
        std.debug.print("the GPU time ({d:.1} ms) exceeds the prompt's wall time ({d:.1} ms): a command buffer's timestamps are wrong\n", .{ gpu * 1e3, prompt_s * 1e3 });
        return error.GpuTimeExceedsWallTime;
    }
    std.debug.print("{{\"prompt_tokens\": {d}, \"chunk\": {d}, \"layers\": {d}, \"repeat\": {d}, \"drafts\": {}, \"prompt_ms\": {d:.1}, \"tok_s\": {d:.2}, \"gpu_ms\": {d:.1}, \"encode_ms\": {d:.1}, \"buffers\": {d}, \"max_buffer_ms\": {d:.1}, \"h_digest\": \"{x}\", \"classes\": ", .{ T, m.prefill_rows, m.layers, opt.repeat, opt.drafts, prompt_s * 1e3, @as(f64, @floatFromInt(T)) / prompt_s, gpu * 1e3, encode * 1e3, buffers, longest * 1e3, digest });
    printClasses(&totals);
    std.debug.print("}}\n", .{});
}
