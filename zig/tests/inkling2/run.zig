//! Inkling 2 (EXL3 experts) on real weights: load check, layer dumps, teacher-forced finals, exactness, greedy (usage).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ik = tf.inkling2;

const usage =
    \\tf-inkling-run <model> check [layers]                    load the model (or its first layers) and report it
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

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
