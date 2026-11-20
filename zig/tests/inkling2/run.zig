//! Inkling 2 (EXL3 experts) on real weights: load check, layer dumps, teacher-forced finals, exactness, greedy (usage).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ik = tf.inkling2;

const usage =
    \\tf-inkling-run <model> check [layers]                    load the model (or its first layers) and report it
    \\tf-inkling-run <model> layers <ids.json> <out.bin> <L>   the first L layers on ids (at most 512): bf16 [L, T, H]
    \\tf-inkling-run <model> finals <index.txt> <out_dir>      each listed ids file in 512-row chunks: seq-NNNNN.bin
    \\                                                         with bf16 finals [T, H] (out_dir must exist)
    \\tf-inkling-run <model> exact <ids.json> <at> [layers]    ids[at..] (ids at most 512) as one-row steps, 16-row
    \\                                                         windows and inside one chunk: outputs bit for bit
    \\tf-inkling-run <model> greedy <ids.json> <n>             the prompt, then n one-row greedy steps; tokens, tok/s
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
    if (std.mem.eql(u8, args[2], "finals")) return finalsMode(init, args);
    if (std.mem.eql(u8, args[2], "exact")) return exactMode(init, args);
    if (std.mem.eql(u8, args[2], "greedy")) return greedyMode(init, args);
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
        try ik.forward.run(m, s, cache, ids[at..][0..n], .{ .finals = finals != null, .logits = if (logits and last) .last else .none });
        if (finals) |f| @memcpy(f[at * H ..][0 .. n * H], s.finals.slice(u16, n * H));
        cache.len += n;
        at += n;
    }
}

fn layersMode(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len != 6) return error.ExpectedIdsOutLayers;
    const L = try std.fmt.parseInt(u32, args[5], 10);
    const ids = try readIds(init, args[3]);
    defer ids.deinit();
    if (ids.value.len > ik.state.prompt_rows) return error.AtMost512Tokens;
    const m = try ik.Model.load(init.gpa, init.io, args[1], .{ .layers = L });
    defer m.deinit();
    const c = &m.config.text;
    var cache = try ik.state.Cache.init(m.device, c, ids.value.len + 1, L);
    defer cache.deinit();
    var s = try ik.state.Scratch.init(m.device, c);
    defer s.deinit();
    const n = ids.value.len;
    const dump = try m.device.buffer(@as(usize, L) * n * c.hidden * 2, mtl.ResourceOptions.shared);
    defer dump.deinit();
    const t0 = now();
    try ik.forward.run(m, &s, &cache, ids.value, .{ .logits = .none, .dump = dump });
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = dump.slice(u8, @as(usize, L) * n * c.hidden * 2) });
    std.debug.print("{{\"layers\": {d}, \"tokens\": {d}, \"seconds\": {d:.2}}}\n", .{ L, n, @as(f64, @floatFromInt(now() - t0)) / 1e9 });
}

fn finalsMode(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len != 5) return error.ExpectedIndexOutDir;
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
    var cache = try ik.state.Cache.init(m.device, c, longest + 1, c.layers);
    defer cache.deinit();
    var s = try ik.state.Scratch.init(m.device, c);
    defer s.deinit();
    const finals = try init.gpa.alloc(u16, longest * c.hidden);
    defer init.gpa.free(finals);
    for (paths.items, 0..) |p, i| {
        const ids = try readIds(init, p);
        defer ids.deinit();
        const t0 = now();
        cache.len = 0;
        try feed(m, &s, &cache, ids.value, finals, false);
        const out = try std.fmt.allocPrint(init.gpa, "{s}/seq-{d:0>5}.bin", .{ args[4], i });
        defer init.gpa.free(out);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out, .data = std.mem.sliceAsBytes(finals[0 .. ids.value.len * c.hidden]) });
        std.debug.print("{{\"seq\": {d}, \"tokens\": {d}, \"seconds\": {d:.1}}}\n", .{ i, ids.value.len, @as(f64, @floatFromInt(now() - t0)) / 1e9 });
    }
}

/// ids[at..] three ways after ids[..at]: one-row steps, 16-row windows, and inside one chunk with the prompt. The last
/// loaded layer's output rows (and the finals, on the whole model) must agree bit for bit.
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
    var s = try ik.state.Scratch.init(m.device, c);
    defer s.deinit();
    var outs: [3][]u16 = undefined;
    for (&outs) |*o| o.* = try init.gpa.alloc(u16, cont * H * 2);
    defer for (outs) |o| init.gpa.free(o);
    for (0..3) |way| {
        var cache = try ik.state.Cache.init(m.device, c, all.len + 1, m.layers);
        defer cache.deinit();
        const rows: usize = switch (way) {
            0 => 1,
            1 => ik.state.window_rows,
            else => all.len,
        };
        if (way < 2) {
            try ik.forward.run(m, &s, &cache, all[0..at], .{ .finals = full, .logits = .none });
            cache.len = at;
        }
        var i: usize = if (way < 2) at else 0;
        while (i < all.len) {
            const n = @min(all.len - i, rows);
            try ik.forward.run(m, &s, &cache, all[i..][0..n], .{ .finals = full, .logits = .none });
            for (0..n) |r| if (i + r >= at) {
                const k = i + r - at;
                @memcpy(outs[way][k * H ..][0..H], s.h.slice(u16, n * H)[r * H ..][0..H]);
                if (full) @memcpy(outs[way][(cont + k) * H ..][0..H], s.finals.slice(u16, n * H)[r * H ..][0..H]);
            };
            cache.len += n;
            i += n;
        }
    }
    var differ: usize = 0;
    for (0..cont) |k| for (1..3) |way| {
        const span: usize = if (full) 2 else 1;
        for (0..span) |part| {
            const off = (part * cont + k) * H;
            if (!std.mem.eql(u16, outs[0][off..][0..H], outs[way][off..][0..H])) {
                differ += 1;
                if (differ <= 4) std.debug.print("row {d} ({s}) differs between steps and {s}\n", .{ at + k, if (part == 0) "layer output" else "finals", if (way == 1) "windows" else "the chunk" });
            }
        }
    };
    std.debug.print("{{\"exact_rows\": {d}, \"differ\": {d}, \"layers\": {d}}}\n", .{ cont, differ, m.layers });
    if (differ != 0) return error.Inkling2NotRowExact;
}

fn greedyMode(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len != 5) return error.ExpectedIdsSteps;
    const steps = try std.fmt.parseInt(usize, args[4], 10);
    const ids = try readIds(init, args[3]);
    defer ids.deinit();
    const m = try ik.Model.load(init.gpa, init.io, args[1], .{});
    defer m.deinit();
    const c = &m.config.text;
    var cache = try ik.state.Cache.init(m.device, c, ids.value.len + steps + 1, c.layers);
    defer cache.deinit();
    var s = try ik.state.Scratch.init(m.device, c);
    defer s.deinit();
    var t0 = now();
    try feed(m, &s, &cache, ids.value, null, true);
    const prompt_ms = @as(f64, @floatFromInt(now() - t0)) / 1e6;
    const picks = try init.gpa.alloc(u32, steps);
    defer init.gpa.free(picks);
    var tm: ik.forward.Timing = .{};
    t0 = now();
    for (picks) |*p| {
        p.* = argmax(s.logits.slice(u16, c.vocab));
        try ik.forward.run(m, &s, &cache, &.{p.*}, .{ .timing = &tm });
        cache.len += 1;
    }
    const secs = @as(f64, @floatFromInt(now() - t0)) / 1e9;
    var w: std.Io.Writer.Allocating = .init(init.gpa);
    defer w.deinit();
    try std.json.Stringify.value(picks, .{}, &w.writer);
    std.debug.print("{s}\n{{\"prompt_tokens\": {d}, \"prompt_ms\": {d:.0}, \"steps\": {d}, \"tok_s\": {d:.2}, \"gpu_ms\": {d:.1}, \"encode_ms\": {d:.1}}}\n", .{ w.written(), ids.value.len, prompt_ms, steps, @as(f64, @floatFromInt(steps)) / secs, tm.gpu * 1e3 / @as(f64, @floatFromInt(steps)), tm.encode * 1e3 / @as(f64, @floatFromInt(steps)) });
}

fn argmax(l: []const u16) u32 {
    var best: usize = 0;
    for (l, 0..) |x, i| if (logit(x) > logit(l[best])) {
        best = i;
    };
    return @intCast(best);
}

fn logit(word: u16) f32 {
    return @bitCast(@as(u32, word) << 16);
}

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
