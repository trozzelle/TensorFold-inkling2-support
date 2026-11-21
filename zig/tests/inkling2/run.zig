//! Inkling 2 (EXL3 experts) on real weights: load check, layer dumps, teacher-forced finals, exactness, greedy (usage).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ik = tf.inkling2;

const usage =
    \\tf-inkling-run <model> check [layers]                    load the model (or its first layers) and report it
    \\tf-inkling-run <model> layers <ids.json> <out.bin> <L> [--window]   the first L layers on ids (at most 512): bf16 [L, T, H]
    \\tf-inkling-run <model> finals <index.txt> <out_dir> [--window]   each listed ids file in 512-row chunks: seq-NNNNN.bin
    \\                                                         with bf16 finals [T, H] (out_dir must exist)
    \\                                                         --window: the first 16 rows as a prompt chunk, the rest in
    \\                                                         16-row windows through the decode path
    \\tf-inkling-run <model> exact <ids.json> <at> [layers]    ids[at..] (ids at most 512) after ids[..at]: one-row steps
    \\                                                         against 16-row windows, bit for bit
    \\tf-inkling-run <model> greedy <ids.json> <n> [--drafts]   the prompt, then n one-row greedy steps; tokens, tok/s
    \\                                                         --drafts: the same n tokens through the lane engine with the draft head in
    \\                                                         front of the target, as served (INKLING_DRAFT_DEPTH: 1 to 7 or auto, 5 when
    \\                                                         unset); needs mtp.safetensors in the folder; also prints rounds, drafts, accepted
    \\tf-inkling-run <model> taps <index.txt> <out_dir> <steps>   each prompt, then steps greedy tokens: taps, picks, MoE inputs
    \\tf-inkling-run <model> ftaps <ids.json> <out.bin> [at [window]]   every row's serving taps: bf16 [T, 8, H], the inputs of layers
    \\                                                         5, 11, .. 41, then the finals. ids[..at] (default: all) go through in
    \\                                                         prompt chunks, the rest in decode windows of at most `window` rows (16)
    \\tf-inkling-run <model> dspark <taps_dir> <seqs> <out_dir> [--depth d]   the draft head over recorded taps (taps mode's files), from the
    \\                                                         embedding, the unembedding and mtp.safetensors alone: <seqs> is N, A..B (inclusive)
    \\                                                         or a comma list; per sequence out_dir gets seq-NNNNN.props.bin (int32 [T][7],
    \\                                                         -1 where no hold ran) and seq-NNNNN.logits.bin (bf16 [16][7][vocab]: the first 16
    \\                                                         holds' step logits after the softcap and the padded vocabulary's mask)
    \\tf-inkling-run <model> profile <layers> <pos> <rows> <steps> [repeat]   per-class GPU time of decode steps
    \\                                                         (the layers run repeat times a step, default 48/layers)
    \\tf-inkling-run <model> wall <layers> <pos> <rows> <steps> [repeat]   decode steps each followed by the greedy draw: median
    \\                                                         wall time and its parts (encode, commit to GPU start, GPU, GPU end to wake, draw)
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

/// ids at the cache's end in prompt_rows chunks; finals rows copied into `finals` when given; logits of the last.
fn feed(m: *const ik.Model, s: *ik.state.Scratch, cache: *ik.state.Cache, ids: []const u32, finals: ?[]u16, logits: bool) !void {
    const H: usize = m.config.text.hidden;
    var at: usize = 0;
    while (at < ids.len) {
        const n = @min(ids.len - at, ik.state.prompt_rows);
        const last = at + n == ids.len;
        try ik.forward.run(m, s, cache, ids[at..][0..n], .{ .finals = finals != null, .logits = if (logits and last) .last else .none, .prompt = true });
        if (finals) |f| @memcpy(f[at * H ..][0 .. n * H], s.finals.slice(u16, n * H));
        cache.len += n;
        at += n;
    }
}

/// Where a window-mode run keeps what it records: finals rows into `finals` (T rows), and each loaded layer's output
/// rows into `layers` ([L][T][H]), the latter through `scratch`, a dump buffer of one window.
const Record = struct { finals: ?[]u16 = null, layers: ?[]u16 = null, scratch: ?mtl.Buffer = null };

/// ids at the cache's end the way a decode-time client sends them: the first window_rows rows as a prompt chunk, every
/// later row in windows of window_rows through the decode path (the last one possibly shorter).
fn feedWindows(m: *const ik.Model, s: *ik.state.Scratch, cache: *ik.state.Cache, ids: []const u32, rec: Record) !void {
    const H: usize = m.config.text.hidden;
    const T = ids.len;
    var at: usize = 0;
    while (at < T) {
        const n = @min(T - at, ik.state.window_rows);
        try ik.forward.run(m, s, cache, ids[at..][0..n], .{ .finals = rec.finals != null, .logits = .none, .dump = rec.scratch, .prompt = at == 0 });
        if (rec.finals) |f| @memcpy(f[at * H ..][0 .. n * H], s.finals.slice(u16, n * H));
        if (rec.layers) |d| {
            const w = rec.scratch.?.slice(u16, m.layers * n * H);
            for (0..m.layers) |l| @memcpy(d[(l * T + at) * H ..][0 .. n * H], w[l * n * H ..][0 .. n * H]);
        }
        cache.len += n;
        at += n;
    }
}

fn layersMode(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len != 6 and !(args.len == 7 and std.mem.eql(u8, args[6], "--window"))) return error.ExpectedIdsOutLayers;
    const window = args.len == 7;
    const L = try std.fmt.parseInt(u32, args[5], 10);
    const ids = try readIds(init, args[3]);
    defer ids.deinit();
    if (ids.value.len > ik.state.prompt_rows) return error.AtMost512Tokens;
    const m = try ik.Model.load(init.gpa, init.io, args[1], .{ .layers = L });
    defer m.deinit();
    const c = &m.config.text;
    const capacity = ids.value.len + 1;
    var cache = try ik.state.Cache.init(m.device, c, capacity, L);
    defer cache.deinit();
    var s = try ik.state.Scratch.init(m.device, c, capacity);
    defer s.deinit();
    const n = ids.value.len;
    if (window) {
        const out = try init.gpa.alloc(u16, @as(usize, L) * n * c.hidden);
        defer init.gpa.free(out);
        const scratch = try m.device.buffer(@as(usize, L) * ik.state.window_rows * c.hidden * 2, mtl.ResourceOptions.shared);
        defer scratch.deinit();
        const t0 = now();
        try feedWindows(m, &s, &cache, ids.value, .{ .layers = out, .scratch = scratch });
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = std.mem.sliceAsBytes(out) });
        std.debug.print("{{\"layers\": {d}, \"tokens\": {d}, \"seconds\": {d:.2}, \"window\": true}}\n", .{ L, n, @as(f64, @floatFromInt(now() - t0)) / 1e9 });
        return;
    }
    const dump = try m.device.buffer(@as(usize, L) * n * c.hidden * 2, mtl.ResourceOptions.shared);
    defer dump.deinit();
    const t0 = now();
    try ik.forward.run(m, &s, &cache, ids.value, .{ .logits = .none, .dump = dump, .prompt = true });
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = dump.slice(u8, @as(usize, L) * n * c.hidden * 2) });
    std.debug.print("{{\"layers\": {d}, \"tokens\": {d}, \"seconds\": {d:.2}}}\n", .{ L, n, @as(f64, @floatFromInt(now() - t0)) / 1e9 });
}

fn finalsMode(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len != 5 and !(args.len == 6 and std.mem.eql(u8, args[5], "--window"))) return error.ExpectedIndexOutDir;
    const window = args.len == 6;
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
    const m = try ik.Model.load(init.gpa, init.io, args[1], .{});
    defer m.deinit();
    const c = &m.config.text;
    const capacity = longest + 1;
    var cache = try ik.state.Cache.init(m.device, c, capacity, c.layers);
    defer cache.deinit();
    var s = try ik.state.Scratch.init(m.device, c, capacity);
    defer s.deinit();
    const finals = try init.gpa.alloc(u16, longest * c.hidden);
    defer init.gpa.free(finals);
    for (paths.items, 0..) |p, i| {
        const ids = try readIds(init, p);
        defer ids.deinit();
        const t0 = now();
        cache.len = 0;
        if (window) try feedWindows(m, &s, &cache, ids.value, .{ .finals = finals }) else try feed(m, &s, &cache, ids.value, finals, false);
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
    if (all.len > ik.state.prompt_rows or at == 0 or at >= all.len) return error.ExpectedIdsAtMost512AndAtInside;
    const m = try ik.Model.load(init.gpa, init.io, args[1], .{ .layers = L });
    defer m.deinit();
    const c = &m.config.text;
    const H: usize = c.hidden;
    const full = m.layers == c.layers;
    const cont = all.len - at;
    const capacity = all.len + 1;
    var s = try ik.state.Scratch.init(m.device, c, capacity);
    defer s.deinit();
    var outs: [2][]u16 = undefined;
    for (&outs) |*o| o.* = try init.gpa.alloc(u16, cont * H * 2);
    defer for (outs) |o| init.gpa.free(o);
    for (0..2) |way| {
        var cache = try ik.state.Cache.init(m.device, c, capacity, m.layers);
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
    const m = try ik.Model.load(init.gpa, init.io, args[1], .{});
    defer m.deinit();
    const c = &m.config.text;
    const capacity = ids.value.len + steps + 1;
    var cache = try ik.state.Cache.init(m.device, c, capacity, c.layers);
    defer cache.deinit();
    var s = try ik.state.Scratch.init(m.device, c, capacity);
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
    const m = try ik.Model.load(init.gpa, init.io, args[1], .{});
    defer m.deinit();
    const c = &m.config.text;
    const H: usize = c.hidden;
    const L: usize = c.layers;
    const R: usize = ik.state.prompt_rows;
    // the longest cache a prompt and its steps fill
    var longest: usize = 0;
    var scan = std.mem.tokenizeAny(u8, index, "\r\n");
    while (scan.next()) |path| {
        const ids = try readIds(init, path);
        defer ids.deinit();
        longest = @max(longest, ids.value.len + steps);
    }
    var s = try ik.state.Scratch.init(m.device, c, longest);
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
        var cache = try ik.state.Cache.init(m.device, c, T + 1, L);
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
    const m = try ik.Model.load(init.gpa, init.io, args[1], .{});
    defer m.deinit();
    const c = &m.config.text;
    const row: usize = ik.state.feature_taps * @as(usize, c.hidden);
    var s = try ik.state.Scratch.init(m.device, c, T + 1);
    defer s.deinit();
    var cache = try ik.state.Cache.init(m.device, c, T + 1, c.layers);
    defer cache.deinit();
    const taps = try init.gpa.alloc(u16, T * row);
    defer init.gpa.free(taps);
    const t0 = now();
    var done: usize = 0;
    while (done < T) {
        const prompt = done < at;
        const n = if (prompt) @min(at - done, ik.state.prompt_rows) else @min(T - done, window);
        const buf = if (prompt) s.ptaps else s.ftaps;
        try ik.forward.run(m, &s, &cache, ids.value[done..][0..n], .{ .logits = .none, .prompt = prompt, .feature_taps = .{ .buf = buf } });
        @memcpy(taps[done * row ..][0 .. n * row], buf.slice(u16, n * row));
        cache.len += n;
        done += n;
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = std.mem.sliceAsBytes(taps) });
    std.debug.print("{{\"rows\": {d}, \"prompt\": {d}, \"window\": {d}, \"seconds\": {d:.2}}}\n", .{ T, at, window, @as(f64, @floatFromInt(now() - t0)) / 1e9 });
}

const profile_classes = [_]ik.forward.counters.Class{ .dense, .norm, .attention_prepare, .attention, .route, .experts, .activation };
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
    if (L == 0 or rows == 0 or rows > ik.state.prompt_rows or repeat == 0) return error.ExpectedPositiveLayersRowsRepeat;
    if (steps <= profile_warmup) return error.StepsMustExceedWarmup;
    const m = try ik.Model.load(init.gpa, init.io, args[1], .{ .layers = L });
    defer m.deinit();
    const c = &m.config.text;
    const full = m.layers == c.layers;
    const capacity = pos + rows * steps + 1;
    var cache = try ik.state.Cache.init(m.device, c, capacity, m.layers);
    defer cache.deinit();
    var s = try ik.state.Scratch.init(m.device, c, capacity);
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
    const m = try ik.Model.load(init.gpa, init.io, args[1], .{ .layers = L });
    defer m.deinit();
    const c = &m.config.text;
    const full = m.layers == c.layers;
    const capacity = pos + rows * steps + 1;
    var cache = try ik.state.Cache.init(m.device, c, capacity, m.layers);
    defer cache.deinit();
    var s = try ik.state.Scratch.init(m.device, c, capacity);
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
