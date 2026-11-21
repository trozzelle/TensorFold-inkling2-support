//! What an Inkling 2 load may hold: weights, scratch and the stream's caches under a limit below the panic line.
const std = @import("std");
const cfg = @import("config.zig");
const st = @import("state.zig");
const Model = @import("model.zig").Model;

/// The default share of RAM: past about 80% wired, macOS stalls until its watchdog restarts the Mac.
pub const default_share = 0.75;
/// Kept free inside the limit for the command buffers, the keepalive and the host.
pub const margin: u64 = 2 << 30;

/// The limit in decimal GB that INKLING_LOAD_LIMIT_GB's text names (null: unset or blank).
pub fn parseLimit(text: ?[]const u8) error{BadLoadLimit}!?f64 {
    const t = std.mem.trim(u8, text orelse "", " \t");
    if (t.len == 0) return null;
    return std.fmt.parseFloat(f64, t) catch error.BadLoadLimit;
}

/// Bytes for caches: `ram` times the share (or `limit_gb` decimal GB, at most RAM and the working set), less the rest.
pub fn caches(ram: u64, working_set: u64, limit_gb: ?f64, resident: u64) error{BadLoadLimit}!u64 {
    var limit: u64 = @intFromFloat(@as(f64, @floatFromInt(ram)) * default_share);
    if (limit_gb) |gb| {
        if (!std.math.isFinite(gb) or gb <= 0) return error.BadLoadLimit;
        const most = @min(ram, working_set);
        const bytes = gb * 1e9;
        limit = if (bytes >= @as(f64, @floatFromInt(most))) most else @intFromFloat(bytes);
    }
    return if (limit > resident + margin) limit - resident - margin else 0;
}

/// Bytes a load holds that the caches' room is left after: the checkpoint's shards (once the draft head is made, its file and
/// its fold among them), the scratch that does not grow with the context, and `head`, the draft head's scratch and its stream's
/// state (0 without one).
pub fn held(model: *const Model, head: u64) !u64 {
    return model.checkpoint.residentBytes() + try st.scratchBytes(&model.config.text, 0, model.prefill_rows) + head;
}

/// Bytes a context of `capacity` positions takes besides the weights: a stream's caches, and the scratch that grows with
/// it (the split attention's partials, sized for a cache a position past the window).
fn contextBytes(c: *const cfg.Text, layers: usize, capacity: usize) u64 {
    return st.cacheBytes(c, capacity, layers) + st.partBytes(c, capacity + 1);
}

/// The longest context one stream's caches and the scratch that grows with it fit in `bytes`, at most `most`.
pub fn longest(c: *const cfg.Text, layers: usize, bytes: u64, most: usize) usize {
    if (contextBytes(c, layers, 1) > bytes) return 0;
    var lo: usize = 1;
    var hi: usize = most;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (contextBytes(c, layers, mid) <= bytes) lo = mid else hi = mid - 1;
    }
    return lo;
}

test "inkling2 budget: the longest context counts the partials that grow with it" {
    var c = std.mem.zeroes(cfg.Text);
    c.hidden = 4608;
    c.vocab = 201024;
    c.local = .{ .q = 48, .kv = 8, .dim = 128 };
    const layers = 48;
    const room: u64 = 40 << 30;
    const n = longest(&c, layers, room, 1 << 20);
    // it fits with its partials and one position more does not; the caches alone would have fit more
    try std.testing.expect(contextBytes(&c, layers, n) <= room);
    try std.testing.expect(contextBytes(&c, layers, n + 1) > room);
    const caches_only = (room - layers * st.convBytes(&c) - @as(u64, c.vocab) * 2) / (layers * st.kvBytes(&c));
    try std.testing.expect(n < caches_only);
    try std.testing.expectEqual(@as(usize, 0), longest(&c, layers, 0, 1000));
    try std.testing.expectEqual(@as(usize, 1000), longest(&c, layers, 1 << 40, 1000));
}

test "inkling2 budget: a share of RAM less the weights, a set limit capped at the working set, bad limits refused" {
    const gib: u64 = 1 << 30;
    try std.testing.expectEqual(@as(u64, 192 * gib - 179 * gib - margin), try caches(256 * gib, 240 * gib, null, 179 * gib));
    try std.testing.expectEqual(@as(u64, 0), try caches(256 * gib, 240 * gib, null, 191 * gib));
    // a limit below the working set is taken as decimal GB; one above it is capped at the working set
    try std.testing.expectEqual(@as(u64, 236_000_000_000 - 179 * gib - margin), try caches(256 * gib, 240 * gib, 236, 179 * gib));
    try std.testing.expectEqual(@as(u64, 240 * gib - 179 * gib - margin), try caches(256 * gib, 240 * gib, 300, 179 * gib));
    try std.testing.expectEqual(@as(u64, 0), try caches(256 * gib, 240 * gib, 100, 179 * gib));
    for ([_]f64{ 0, -1, std.math.nan(f64), std.math.inf(f64) }) |bad| try std.testing.expectError(error.BadLoadLimit, caches(256 * gib, 240 * gib, bad, 24 * gib));
    // the env var's text: unset or blank names no limit, a number is its GB, anything else is refused
    try std.testing.expectEqual(@as(?f64, null), try parseLimit(null));
    try std.testing.expectEqual(@as(?f64, null), try parseLimit(" \t"));
    try std.testing.expectEqual(@as(?f64, 236), try parseLimit(" 236 "));
    try std.testing.expectError(error.BadLoadLimit, parseLimit("lots"));
}
