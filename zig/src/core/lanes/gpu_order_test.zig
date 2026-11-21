//! The sampler's partial order against the full sort it replaced: the same Plan and the same draw for the same key.
const std = @import("std");
const full = @import("gpu_full.zig");
const rule = @import("gpu_rule.zig");
const Sampling = @import("sampling.zig").Sampling;

const a = std.testing.allocator;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

/// A float cut to bf16's width: a bf16 logits row ties constantly.
fn bf16(x: f32) f32 {
    return @bitCast(@as(u32, @bitCast(x)) & 0xFFFF_0000);
}

const Shape = enum { gaussian, ties, flat, masked, near_flat, peaked };

/// A decode-like row: a soft-capped Gaussian bulk, a few levels (ties), one level, -inf holes, a bulk too flat for a
/// nucleus to end within 1,024 tokens, or a bulk with a few dominant tokens.
fn fill(row: []f32, shape: Shape, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (row) |*x| {
        const bulk = bf16(30.0 * std.math.tanh((r.floatNorm(f32) * 2.0 - 4.0) / 30.0));
        x.* = switch (shape) {
            .gaussian, .peaked => bulk,
            .ties => @floatFromInt(r.uintLessThan(u32, 12)),
            .flat => 1.5,
            .masked => if (r.uintLessThan(u8, 10) < 4) -std.math.inf(f32) else bulk,
            .near_flat => bf16(r.float(f32) * 0.5 - 0.25),
        };
    }
    if (shape == .peaked) for (0..6) |k| {
        row[r.uintLessThan(usize, row.len)] = 14.0 - 2.0 * @as(f32, @floatFromInt(k));
    };
}

fn mark(c: rule.Cand) full.Mark {
    return .{ .key = rule.key(c.v), .id = c.i };
}

/// `full.plan` as it was before the partial order: every candidate sorted, the order read where it is read.
fn fullSortPlan(logits: []const f32, s: Sampling) !full.Plan {
    const r = rule.rule(s);
    const inv_t = r.inv_t;
    const order = try a.alloc(rule.Cand, logits.len);
    defer a.free(order);
    var m = -std.math.inf(f32);
    for (order, logits, 0..) |*c, l, i| {
        c.* = .{ .v = l * inv_t, .i = @intCast(i) };
        m = @max(m, c.v);
    }
    std.mem.sort(rule.Cand, order, {}, rule.better);
    var p = full.Plan{ .inv_t = inv_t, .m = m, .lo = @max(m + r.min_log, m - full.never) };
    if (s.top_k != 0 and s.top_k < logits.len) p.top = mark(order[s.top_k - 1]);
    if (r.top_p > 0.0 and r.top_p < 1.0) p.cut = full.nucleus(logits, p, order, r.top_p);
    return p;
}

/// `rule.sample` as it was before the partial order: every candidate sorted.
fn fullSortSample(logits: []const f32, s: Sampling, position: u32, ids: ?[]const u32) !u32 {
    const r = rule.rule(s);
    const inv_t = r.inv_t;
    const kc = r.top_k;
    const cap: usize = if (kc == 0 or kc > rule.candidates) rule.candidates else kc;
    const all = try a.alloc(rule.Cand, logits.len);
    defer a.free(all);
    var m = -std.math.inf(f32);
    for (all, logits, 0..) |*c, l, i| {
        c.* = .{ .v = l * inv_t, .i = @intCast(i) };
        m = @max(m, c.v);
    }
    var lanes: [1024]f32 = @splat(0.0);
    for (all, 0..) |c, i| lanes[i % 1024] += @exp(c.v - m);
    const z = rule.lanesSum(&lanes);
    std.mem.sort(rule.Cand, all, {}, rule.better);
    const n: usize = @min(@min(all.len, rule.candidates), cap);
    var norm = z;
    if (kc != 0) {
        norm = 0.0;
        for (all[0..n]) |c| norm += @exp(c.v - m);
    }
    var keep: usize = n;
    if (r.top_p > 0.0 and r.top_p < 1.0) {
        var cum: f32 = 0.0;
        for (all[0..n], 0..) |c, j| {
            cum += @exp(c.v - m) / norm;
            if (cum >= r.top_p) {
                keep = j + 1;
                break;
            }
        }
    }
    if (r.min_log > -std.math.inf(f32)) {
        const floor = m + r.min_log;
        var j: usize = 0;
        while (j < keep and all[j].v >= floor) j += 1;
        keep = j;
    }
    var best: usize = 0;
    var best_score = -std.math.inf(f32);
    for (all[0..keep], 0..) |c, t| {
        const id = if (ids) |map| map[c.i] else c.i;
        const score = c.v - @log(-@log(rule.uniform24(s.seed, position, id)));
        if (t == 0 or score > best_score) {
            best = t;
            best_score = score;
        }
    }
    const pick = all[best].i;
    return if (ids) |map| map[pick] else pick;
}

fn bits(x: f32) u32 {
    return @bitCast(x);
}

fn samePlan(want: full.Plan, got: full.Plan) !void {
    try expectEqual(bits(want.inv_t), bits(got.inv_t));
    try expectEqual(bits(want.m), bits(got.m));
    try expectEqual(bits(want.lo), bits(got.lo));
    try expectEqual(want.top, got.top);
    try expectEqual(want.cut, got.cut);
}

/// One row under `s`: the plan and the draws at a few positions, as the full sort gave them.
fn same(row: []const f32, s: Sampling) !void {
    const want = try fullSortPlan(row, s);
    try samePlan(want, try full.plan(a, row, s));
    for ([_]u32{ 0, 1, 7, 4096, 66903620 }) |pos| {
        try expectEqual(full.race(row, want, s.seed, pos, null), try full.sample(a, row, s, pos, null));
        const old = if (full.fullVocabulary(s)) full.race(row, want, s.seed, pos, null) else try fullSortSample(row, s, pos, null);
        try expectEqual(old, try full.draw(a, row, s, pos, null));
    }
}

test "the first n of orderTop are the full sort's, ties, signed zeros, infinities and NaNs included" {
    const len = 5000;
    const all = try a.alloc(rule.Cand, len);
    defer a.free(all);
    const want = try a.alloc(rule.Cand, len);
    defer a.free(want);
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const r = prng.random();
    const palette = [_]f32{ 0.0, -0.0, 1.0, -1.0, 3.5, -3.5, std.math.inf(f32), -std.math.inf(f32), 1e-40, -1e-40, 2.0, std.math.nan(f32), -std.math.nan(f32) };
    for (0..6) |shape| for ([_]usize{ 0, 1, 2, 3, 20, 100, 1024, 1250, 1251, 5000 }) |n| {
        for (all, 0..) |*c, i| {
            const v: f32 = switch (shape) {
                0 => r.floatNorm(f32),
                1 => palette[r.uintLessThan(usize, palette.len)],
                2 => 1.0, // every key equal: the lowest ids
                3 => @floatFromInt(i), // ascending
                4 => -@as(f32, @floatFromInt(i)), // descending
                else => bf16(r.floatNorm(f32) * 4.0),
            };
            c.* = .{ .v = v, .i = @intCast(i) };
        }
        @memcpy(want, all);
        std.mem.sort(rule.Cand, want, {}, rule.better);
        rule.orderTop(all, n);
        for (all[0..n], want[0..n]) |got, expected| {
            try expectEqual(expected.i, got.i);
            try expectEqual(bits(expected.v), bits(got.v));
        }
    };
}

test "plans and draws equal the full sort's over ties, flat, masked and peaked rows, every top_k, a nucleus past 1,024" {
    var row: [4100]f32 = undefined;
    var small: [97]f32 = undefined;
    for (std.enums.values(Shape), 0..) |shape, k| {
        fill(&row, shape, 100 + k);
        fill(&small, shape, 200 + k);
        for ([_]u32{ 0, 1, 20, 1024, 1025, 5000 }) |top_k| for ([_]f64{ 1.0, 0.95, 0.5 }) |top_p| for ([_]f64{ 0.7, 1.5 }) |temperature| {
            // flat bulks end no nucleus within 1,024: a pair of cases is enough to run extend() there
            if ((shape == .flat or shape == .near_flat) and top_p < 1.0 and (top_k > 0 or temperature != 0.7)) continue;
            for ([_]f64{ 0.0, 0.05 }) |min_p| {
                const s: Sampling = .{ .seed = 7 + 31 * k + top_k, .temperature = temperature, .top_k = top_k, .top_p = top_p, .min_p = min_p };
                try same(&row, s);
                try same(&small, s);
            }
        };
    }
}

test "a top_k of 5,000 and 1,025 selects from rows long enough to cut well above the 1,024 the nucleus needs" {
    const row = try a.alloc(f32, 24_000);
    defer a.free(row);
    for ([_]Shape{ .gaussian, .ties, .masked }, 0..) |shape, k| {
        fill(row, shape, 300 + k);
        for ([_]u32{ 1025, 5000, 5999 }) |top_k| for ([_]f64{ 1.0, 0.9 }) |top_p| {
            try same(row, .{ .seed = 11 + k, .temperature = 0.9, .top_k = top_k, .top_p = top_p });
            try same(row, .{ .seed = 12 + k, .temperature = 1.2, .top_k = top_k, .top_p = top_p, .min_p = 0.02 });
        };
    }
}

test "draws that map columns to token ids equal the full sort's" {
    var row: [4100]f32 = undefined;
    var ids: [4100]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(2 * i + 1);
    fill(&row, .ties, 5);
    for ([_]u32{ 1, 20, 1024 }) |top_k| for ([_]f64{ 1.0, 0.9 }) |top_p| {
        const s: Sampling = .{ .seed = 3, .temperature = 0.8, .top_k = top_k, .top_p = top_p, .min_p = 0.01 };
        for (0..8) |pos| try expectEqual(try fullSortSample(&row, s, @intCast(pos), &ids), try rule.sample(a, &row, s, @intCast(pos), &ids));
    };
}

test "no order is read, and none made, with top_k off and top_p off" {
    var row: [4100]f32 = undefined;
    fill(&row, .gaussian, 9);
    const s: Sampling = .{ .seed = 1, .temperature = 0.7, .top_k = 0, .top_p = 1.0 };
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    const p = try full.plan(failing.allocator(), &row, s);
    try expect(!failing.has_induced_failure);
    try samePlan(try fullSortPlan(&row, s), p);
    try expectEqual(full.end, p.cut);
    try expectEqual(full.end, p.top);
}
