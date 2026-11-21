//! tf-inkling-run's `dspark` mode: the DSpark draft head over recorded taps, as the reference drove it. For each listed sequence
//! (`seq-NNNNN.taps.bin`, bf16 [T][8][H], and `seq-NNNNN.tokens.json`, the prompt's length and the T + 1 tokens) the whole
//! prompt's rows are absorbed and a block is held after it (the anchor is the token after the prompt's last row); then, one
//! generated row at a time, the row is absorbed and a block held with the next token as anchor. It writes
//!   seq-NNNNN.props.bin   int32 [T][7]: the proposals made after committed row p in row p, -1 where no hold ran
//!   seq-NNNNN.logits.bin  bf16 [16][7][vocab]: the first 16 holds' steps' logits after the bias, the softcap and the padded
//!                         vocabulary's mask
//! The model loads without its layers (the embedding, the unembedding and the draft head: about 8 GB).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ik = tf.inkling2;
const lanes = tf.lanes;

const H = 4608;
const row_bytes = ik.state.feature_taps * H * 2;
/// Holds whose steps' logits are written, and the proposals of a hold (the block's rows after the first).
const saved_holds = 16;
const steps = ik.dspark.max_depth;

const Meta = struct { prompt: usize, tokens: []u32 };

/// '0..20' is inclusive; commas join: '0..3,7,9..10'.
fn parseSeqs(gpa: std.mem.Allocator, spec: []const u8) ![]usize {
    var out: std.ArrayList(usize) = .empty;
    errdefer out.deinit(gpa);
    var parts = std.mem.splitScalar(u8, spec, ',');
    while (parts.next()) |part| {
        const dots = std.mem.indexOf(u8, part, "..");
        const lo = try std.fmt.parseInt(usize, if (dots) |i| part[0..i] else part, 10);
        const hi = if (dots) |i| try std.fmt.parseInt(usize, part[i + 2 ..], 10) else lo;
        if (hi < lo) return error.EmptySeqRange;
        for (lo..hi + 1) |s| try out.append(gpa, s);
    }
    return out.toOwnedSlice(gpa);
}

fn now() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

pub fn run(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len != 6 and args.len != 8) return error.ExpectedTapsSeqsOutDepth;
    var depth: u32 = steps;
    if (args.len == 8) {
        if (!std.mem.eql(u8, args[6], "--depth")) return error.ExpectedTapsSeqsOutDepth;
        depth = try std.fmt.parseInt(u32, args[7], 10);
    }
    if (depth == 0 or depth > steps) return error.DepthOneToSeven;
    const seqs = try parseSeqs(init.gpa, args[4]);
    defer init.gpa.free(seqs);
    try std.Io.Dir.cwd().createDirPath(init.io, args[5]);
    const t0 = now();
    const m = try ik.Model.load(init.gpa, init.io, args[1], .{ .layers = 0, .drafts = true });
    defer m.deinit();
    const weights: *ik.dspark_weights.Weights = if (m.draft) |*w| w else return error.NoDraftHead;
    const loaded = now();
    const drafter = try ik.dspark.Drafter.init(init.gpa, m.device, m, weights, .{ .depth = depth });
    defer drafter.deinit();
    try drafter.captureLogits();
    std.debug.print("{{\"load_s\": {d:.1}, \"drafter_s\": {d:.1}, \"depth\": {d}, \"hold_ms\": {d:.3}, \"step_ms\": {d:.3}, \"resident_gib\": {d:.2}}}\n", .{ loaded - t0, now() - loaded, depth, drafter.hold_ms, drafter.step_ms, @as(f64, @floatFromInt(m.checkpoint.residentBytes() + drafter.scratchBytes())) / (1 << 30) });
    for (seqs) |seq| try replay(init, m, drafter, args[3], args[5], seq, depth);
}

fn path(gpa: std.mem.Allocator, dir: []const u8, seq: usize, suffix: []const u8) ![]u8 {
    return std.fmt.allocPrint(gpa, "{s}/seq-{d:0>5}.{s}", .{ dir, seq, suffix });
}

/// One sequence: the prompt, then every generated row, written as `props.bin` and `logits.bin` under `out_dir`.
fn replay(init: std.process.Init, m: *ik.Model, drafter: *ik.dspark.Drafter, taps_dir: []const u8, out_dir: []const u8, seq: usize, depth: u32) !void {
    const gpa = init.gpa;
    const began = now();
    const meta_path = try path(gpa, taps_dir, seq, "tokens.json");
    defer gpa.free(meta_path);
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, meta_path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    const meta = try std.json.parseFromSlice(Meta, gpa, text, .{ .ignore_unknown_fields = true });
    defer meta.deinit();
    const L = meta.value.prompt;
    const tokens = meta.value.tokens;
    if (L == 0 or tokens.len < L + 1) return error.BadTokensFile;
    const T = tokens.len - 1;
    const taps_path = try path(gpa, taps_dir, seq, "taps.bin");
    defer gpa.free(taps_path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, taps_path, gpa, .limited(8 << 30));
    defer gpa.free(bytes);
    if (bytes.len != T * row_bytes) {
        std.log.err("{s}: {d} bytes, {d} rows of {d} expected", .{ taps_path, bytes.len, T, row_bytes });
        return error.BadTapsFile;
    }
    const taps = try m.device.buffer(bytes.len, mtl.ResourceOptions.shared);
    defer taps.deinit();
    @memcpy(taps.slice(u8, bytes.len), bytes);
    const vocab: usize = m.config.text.vocab;
    const props = try gpa.alloc(i32, T * steps);
    defer gpa.free(props);
    @memset(props, -1);
    const logits = try gpa.alloc(u16, saved_holds * steps * vocab);
    defer gpa.free(logits);
    @memset(logits, 0);
    var stream = try lanes.Stream.init(gpa, .{ .id = "replay", .prompt = tokens[0..L], .max_new = 1 });
    defer stream.deinit(gpa);
    const dv = drafter.drafter();
    try dv.open(&stream);
    defer dv.release(&stream);
    var holds: usize = 0;
    var next: usize = 0; // the first row not yet absorbed
    var p: usize = L - 1;
    while (p < T) : (p += 1) {
        const rows = p + 1 - next;
        const f: lanes.backend.Features = .{ .buffer = @intFromPtr(taps.id), .offset = next * row_bytes, .rows = @intCast(rows), .row_bytes = row_bytes, .space = .device, .dtype = .bf16 };
        try dv.absorb(&.{.{ .stream = &stream, .features = f, .start = next, .follow = tokens[next + 1 ..][0..rows], .prompt = next == 0 }});
        try dv.hold(&.{.{ .stream = &stream, .pending = tokens[p + 1], .position = p + 2, .depth = depth }});
        var held: [steps]u32 = undefined;
        try dv.held(&.{&stream}, &.{held[0..depth]});
        for (held[0..depth], props[p * steps ..][0..depth]) |x, *y| y.* = @intCast(x);
        if (holds < saved_holds) @memcpy(logits[holds * steps * vocab ..][0 .. depth * vocab], drafter.captured()[0 .. depth * vocab]);
        holds += 1;
        next = p + 1;
    }
    const props_path = try path(gpa, out_dir, seq, "props.bin");
    defer gpa.free(props_path);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = props_path, .data = std.mem.sliceAsBytes(props) });
    const logits_path = try path(gpa, out_dir, seq, "logits.bin");
    defer gpa.free(logits_path);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = logits_path, .data = std.mem.sliceAsBytes(logits) });
    const secs = now() - began;
    std.debug.print("{{\"seq\": {d}, \"prompt\": {d}, \"rows\": {d}, \"holds\": {d}, \"seconds\": {d:.1}, \"ms_per_hold\": {d:.2}}}\n", .{ seq, L, T, holds, secs, secs * 1e3 / @as(f64, @floatFromInt(holds)) });
}
