//! tf-inkling-bench's `dspark` mode: the DSpark draft head's hold at depth 4, 5 and 7 and an absorb of one row, on a model folder's
//! real weights (the embedding, the unembedding and mtp.safetensors; no layers) over a full window of context rows of random taps.
//! Each figure is the median of nine runs after one that is not counted: a hold's wall milliseconds (the host's encode, the GPU and
//! the wake-up) and its command buffer's GPU milliseconds; an absorb's host milliseconds (encode and commit), its wall milliseconds
//! to completion and its GPU milliseconds; and a drafted round's absorb and hold at depth 4 and at depth 5 together. `--bits 8|16`
//! picks the draft layers' weights as INKLING_DRAFT_BITS does when serving (8 by default); the first line says which, and what the
//! draft head holds resident.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ik = tf.inkling2;
const lanes = tf.lanes;

const trials = 9;
/// Rows absorbed before timing: more than a window, so the attention reads every key it can.
const context = 1100;

fn now() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1e3 + @as(f64, @floatFromInt(ts.nsec)) / 1e6;
}

fn median(v: *[trials]f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return v[trials / 2];
}

pub fn run(init: std.process.Init, args: []const [:0]const u8) !void {
    // dspark <model>, then --bits 8|16 if given
    var bits: ?ik.dspark_weights.Bits = null;
    if (args.len == 5 and std.mem.eql(u8, args[3], "--bits")) {
        bits = try ik.dspark_weights.Bits.parse(args[4]);
    } else if (args.len != 3) return error.ExpectedModelAndMaybeBits;
    const m = try ik.Model.load(init.gpa, init.io, args[2], .{ .layers = 0, .drafts = true, .draft_bits = bits });
    defer m.deinit();
    const weights: *ik.dspark_weights.Weights = if (m.draft) |*w| w else return error.NoDraftHead;
    const drafter = try ik.dspark.Drafter.init(init.gpa, m.device, m, weights, .{ .depth = 7 });
    defer drafter.deinit();
    const row_bytes = drafter.tapRowBytes();
    std.debug.print("{{\"bits\": {d}, \"head_bytes\": {d}, \"head_gib\": {d:.3}, \"weights_bytes\": {d}, \"fold_bytes\": {d}, \"scratch_bytes\": {d}, \"lane_bytes\": {d}}}\n", .{ @backingInt(weights.bits), drafter.residentBytes(1), @as(f64, @floatFromInt(drafter.residentBytes(1))) / (1 << 30), weights.bytes, weights.fold_w.?.t.bytes + weights.fold_b.?.bytes, drafter.scratchBytes(), drafter.laneBytes() });
    const dv = drafter.drafter();
    const rows = context + 3 * (trials + 1) + 1;
    const taps = try m.device.buffer(rows * row_bytes, mtl.ResourceOptions.shared);
    defer taps.deinit();
    var prng = std.Random.DefaultPrng.init(0xd5a4c);
    const r = prng.random();
    for (taps.slice(u16, rows * row_bytes / 2)) |*x| {
        const f: f32 = (r.float(f32) * 2 - 1) * 4;
        x.* = @intCast(@as(u32, @bitCast(f)) >> 16);
    }
    const ids = [_]u32{ 1000, 2000, 3000, 4000 };
    var stream = try lanes.Stream.init(init.gpa, .{ .id = "bench", .prompt = &ids, .max_new = 1 });
    defer stream.deinit(init.gpa);
    try dv.open(&stream);
    defer dv.release(&stream);
    var next: usize = 0; // the first row not yet absorbed
    try absorb(dv, &stream, taps, row_bytes, &next, context, false);
    try drafter.sync();
    var wall: [trials]f64 = undefined;
    var gpu: [trials]f64 = undefined;
    for ([_]u32{ 4, 5, 7 }) |depth| {
        for (0..trials + 1) |i| {
            try dv.hold(&.{.{ .stream = &stream, .pending = ids[i % ids.len], .position = next + 1, .depth = depth }});
            if (i == 0) continue;
            wall[i - 1] = drafter.hold_ms;
            gpu[i - 1] = drafter.hold_gpu * 1e3;
        }
        std.debug.print("{{\"hold_depth\": {d}, \"wall_ms\": {d:.3}, \"gpu_ms\": {d:.3}}}\n", .{ depth, median(&wall), median(&gpu) });
    }
    var host: [trials]f64 = undefined;
    var round: [trials]f64 = undefined;
    for (0..trials + 1) |i| {
        const t0 = now();
        try absorb(dv, &stream, taps, row_bytes, &next, 1, false);
        const t1 = now();
        try drafter.sync();
        const t2 = now();
        if (i == 0) continue;
        host[i - 1] = t1 - t0;
        wall[i - 1] = t2 - t0;
        gpu[i - 1] = drafter.absorb_gpu * 1e3;
    }
    std.debug.print("{{\"absorb_rows\": 1, \"host_ms\": {d:.3}, \"wall_ms\": {d:.3}, \"gpu_ms\": {d:.3}}}\n", .{ median(&host), median(&wall), median(&gpu) });
    for ([_]u32{ 4, 5 }) |depth| {
        for (0..trials + 1) |i| {
            const t0 = now();
            try absorb(dv, &stream, taps, row_bytes, &next, 1, false);
            try dv.hold(&.{.{ .stream = &stream, .pending = ids[i % ids.len], .position = next + 1, .depth = depth }});
            if (i == 0) continue;
            round[i - 1] = now() - t0;
        }
        std.debug.print("{{\"round\": \"absorb 1 row, hold depth {d}\", \"wall_ms\": {d:.3}, \"trials\": {d}, \"context_rows\": {d}, \"load_hold_ms\": {d:.3}, \"load_step_ms\": {d:.3}}}\n", .{ depth, median(&round), trials, context, drafter.hold_ms, drafter.step_ms });
    }
}

/// `n` more rows of `taps` into the stream's lane.
fn absorb(dv: lanes.drafter.Drafter, stream: *lanes.Stream, taps: mtl.Buffer, row_bytes: usize, next: *usize, n: usize, prompt: bool) !void {
    const f: lanes.backend.Features = .{ .buffer = @intFromPtr(taps.id), .offset = next.* * row_bytes, .rows = @intCast(n), .row_bytes = @intCast(row_bytes), .space = .device, .dtype = .bf16 };
    try dv.absorb(&.{.{ .stream = stream, .features = f, .start = next.*, .follow = &.{}, .prompt = prompt }});
    next.* += n;
}
