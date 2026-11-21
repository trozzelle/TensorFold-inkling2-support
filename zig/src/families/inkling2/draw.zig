//! The host's draws for a verify window: each row's greedy argmax or keyed draw (lanes' gpu_full host twin) from its bf16
//! logits, and which rows of a window a round reads.
const std = @import("std");
const lanes = @import("lanes");
const greedy = @import("greedy.zig");

/// A row's draw that was not made. No token equals it, so a read past the rows a round needs cannot accept a draft.
pub const undrawn: u32 = std.math.maxInt(u32);

fn logit(word: u16) f32 {
    return @bitCast(@as(u32, word) << 16);
}

fn sampled(sampling: ?lanes.Sampling) ?lanes.Sampling {
    const t = sampling orelse return null;
    return if (t.temperature > 0) t else null;
}

/// A window as its draws read it.
pub const Rows = struct {
    drafts: []const u32, // the tokens after the pending one: row r + 1 holds drafts[r]
    positions: []const u64, // each row's keyed draw position
    masks: []const u32 = &.{}, // each row's allowed-token bits, rows x words (none: every token)
    sampling: ?lanes.Sampling = null,
    vocab: usize,
};

pub const Drawer = struct {
    gpa: std.mem.Allocator,
    /// A logits row as f32, for the keyed draw.
    values: []f32,

    /// Greedy argmax or the keyed draw at `position`, a grammar mask's tokens only.
    pub fn one(d: Drawer, l: []const u16, sampling: ?lanes.Sampling, position: u64, mask: []const u32) !u32 {
        const t = sampled(sampling) orelse {
            if (mask.len == 0) return greedy.argmax(l);
            var best: ?usize = null;
            for (l, 0..) |w, id| {
                if (!lanes.grammar.allows(mask, id)) continue;
                if (best == null or logit(w) > logit(l[best.?])) best = id;
            }
            return @intCast(best orelse 0);
        };
        for (l, d.values, 0..) |w, *v, id| {
            v.* = logit(w);
            if (mask.len > 0 and !lanes.grammar.allows(mask, id)) v.* = -std.math.inf(f32);
        }
        return lanes.gpu_full.draw(d.gpa, d.values, t, @intCast(position), null);
    }

    /// The draws of a window's rows into `out`, row r's logits being `all[r * vocab ..][0..vocab]`.
    /// A round reads them along the path it accepts, the drafts that equal their rows' draws and the draw of the first row
    /// whose draft does not (core's acceptPath), and a forced window's last row, which it commits whatever its rows drew.
    /// With `lazy` only those are made, in row order; the rows between hold `undrawn`. A draw is a function of its row's
    /// logits, the sampling and its position alone, so every row drawn holds the draw the full window gives.
    pub fn window(d: Drawer, all: []const u16, w: Rows, out: []u32, lazy: bool) !void {
        const words = (w.vocab + 31) / 32;
        var on_path = true; // every draft so far equals its row's draw
        for (out, 0..) |*o, r| {
            if (lazy and !on_path and r + 1 < out.len) {
                o.* = undrawn;
                continue;
            }
            const mask = if (w.masks.len >= (r + 1) * words) w.masks[r * words ..][0..words] else &.{};
            o.* = try d.one(all[r * w.vocab ..][0..w.vocab], w.sampling, w.positions[r], mask);
            if (r < w.drafts.len and o.* != w.drafts[r]) on_path = false;
        }
    }
};

const vocab = 64;
const rows = 6;

/// `rows` rows of logits whose largest entries are `picks`, the next largest the entry after each.
fn fakeLogits(picks: [rows]u32) [rows * vocab]u16 {
    var all: [rows * vocab]u16 = @splat(0xbf80); // -1.0
    for (picks, 0..) |p, r| {
        all[r * vocab + (p + 1) % vocab] = 0x3f80; // 1.0
        all[r * vocab + p] = 0x4000; // 2.0
    }
    return all;
}

fn positions() [rows]u64 {
    var p: [rows]u64 = undefined;
    for (&p, 0..) |*x, r| x.* = 100 + r;
    return p;
}

/// The accepted path's draws as core reads them: the path rows' draws, which the bonus ends.
fn accepted(gpa: std.mem.Allocator, pending: u32, drafts: []const u32, picks: []const u32) ![]u32 {
    var tokens: [rows]u32 = undefined;
    tokens[0] = pending;
    @memcpy(tokens[1..][0..drafts.len], drafts);
    const parents = try lanes.accept.rowParents(gpa, drafts.len + 1, null);
    defer gpa.free(parents);
    const path = try lanes.accept.acceptPath(gpa, tokens[0 .. drafts.len + 1], parents, picks);
    defer gpa.free(path);
    const out = try gpa.alloc(u32, path.len);
    for (out, path) |*o, r| o.* = picks[r];
    return out;
}

/// The drafts of a window whose rows draw `full`: the first `kept` right, the rest wrong.
fn draftsKeeping(full: [rows]u32, kept: usize) [rows - 1]u32 {
    var drafts: [rows - 1]u32 = undefined;
    for (&drafts, 0..) |*x, r| x.* = if (r < kept) full[r] else (full[r] + 1) % vocab;
    return drafts;
}

test "inkling2 draw: a window's draws stop after its first rejected draft and keep the last row's; the path and bonus are the full draw's" {
    const gpa = std.testing.allocator;
    var values: [vocab]f32 = undefined;
    const d: Drawer = .{ .gpa = gpa, .values = &values };
    const picks = [rows]u32{ 3, 17, 40, 8, 29, 55 };
    const all = fakeLogits(picks);
    const pos = positions();
    for (0..rows) |kept| {
        const drafts = draftsKeeping(picks, kept);
        const w: Rows = .{ .drafts = &drafts, .positions = &pos, .vocab = vocab };
        var full: [rows]u32 = undefined;
        var lazy: [rows]u32 = undefined;
        try d.window(&all, w, &full, false);
        try d.window(&all, w, &lazy, true);
        try std.testing.expectEqualSlices(u32, &picks, &full);
        // the rejected draft's row is the last the path reaches; the rows past it, but the last, are not drawn
        const stop = @min(kept, rows - 1);
        for (0..rows) |r| {
            const wanted = r <= stop or r == rows - 1;
            try std.testing.expectEqual(if (wanted) full[r] else undrawn, lazy[r]);
        }
        const want = try accepted(gpa, 1, &drafts, &full);
        defer gpa.free(want);
        const got = try accepted(gpa, 1, &drafts, &lazy);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u32, want, got);
        try std.testing.expectEqual(@min(kept, rows - 1) + 1, got.len);
    }
}

test "inkling2 draw: keyed draws are the full window's on the rows drawn, so the path and bonus are the same" {
    const gpa = std.testing.allocator;
    var values: [vocab]f32 = undefined;
    const d: Drawer = .{ .gpa = gpa, .values = &values };
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const random = prng.random();
    const pos = positions();
    for (0..8) |seed| {
        var all: [rows * vocab]u16 = undefined;
        for (&all) |*x| x.* = @intCast(@as(u32, @bitCast((random.float(f32) * 6 - 3))) >> 16);
        const sampling: lanes.Sampling = .{ .seed = seed, .temperature = 0.8, .top_k = 20, .top_p = 0.95 };
        const none: [rows - 1]u32 = @splat(0);
        var picks: [rows]u32 = undefined;
        try d.window(&all, .{ .drafts = &none, .positions = &pos, .sampling = sampling, .vocab = vocab }, &picks, false);
        for (0..rows) |kept| {
            const drafts = draftsKeeping(picks, kept);
            const w: Rows = .{ .drafts = &drafts, .positions = &pos, .sampling = sampling, .vocab = vocab };
            var full: [rows]u32 = undefined;
            var lazy: [rows]u32 = undefined;
            try d.window(&all, w, &full, false);
            try d.window(&all, w, &lazy, true);
            try std.testing.expectEqualSlices(u32, &picks, &full);
            for (0..rows) |r| try std.testing.expectEqual(if (r <= kept or r == rows - 1) full[r] else undrawn, lazy[r]);
            const want = try accepted(gpa, 1, &drafts, &full);
            defer gpa.free(want);
            const got = try accepted(gpa, 1, &drafts, &lazy);
            defer gpa.free(got);
            try std.testing.expectEqualSlices(u32, want, got);
        }
    }
}

test "inkling2 draw: each row keeps its own mask whichever rows are drawn" {
    const gpa = std.testing.allocator;
    var values: [vocab]f32 = undefined;
    const d: Drawer = .{ .gpa = gpa, .values = &values };
    const picks = [rows]u32{ 3, 17, 40, 8, 29, 55 };
    const all = fakeLogits(picks);
    // row r allows token 10 + r alone, whatever its logits say
    var masks: [rows * 2]u32 = @splat(0);
    for (0..rows) |r| masks[r * 2] |= @as(u32, 1) << @intCast(10 + r);
    const pos = positions();
    const drafts = [rows - 1]u32{ 10, 11, 50, 13, 14 };
    var lazy: [rows]u32 = undefined;
    try d.window(&all, .{ .drafts = &drafts, .positions = &pos, .masks = &masks, .vocab = vocab }, &lazy, true);
    try std.testing.expectEqualSlices(u32, &.{ 10, 11, 12, undrawn, undrawn, 15 }, &lazy);
}

test "inkling2 draw: a window of one row, or of two with a rejected draft, draws every row it has" {
    const gpa = std.testing.allocator;
    var values: [vocab]f32 = undefined;
    const d: Drawer = .{ .gpa = gpa, .values = &values };
    const picks = [rows]u32{ 3, 17, 40, 8, 29, 55 };
    const all = fakeLogits(picks);
    const pos = positions();
    var one: [1]u32 = undefined;
    try d.window(&all, .{ .drafts = &.{}, .positions = pos[0..1], .vocab = vocab }, &one, true);
    try std.testing.expectEqualSlices(u32, &.{3}, &one);
    var two: [2]u32 = undefined;
    try d.window(&all, .{ .drafts = &.{9}, .positions = pos[0..2], .vocab = vocab }, &two, true);
    try std.testing.expectEqualSlices(u32, &.{ 3, 17 }, &two);
}
