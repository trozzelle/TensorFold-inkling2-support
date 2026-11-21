//! The greedy draw over a bf16 logits row: the first index of its largest value, found with vectors.
const std = @import("std");

/// Words a vector step reads, and the independent running maxima that hide the max's latency.
const lane_words = 16;
const unroll = 4;
const Words = @Vector(lane_words, u16);
const Floats = @Vector(lane_words, f32);

/// The floats the words hold (a bf16 is the high half of an f32).
fn floats(words: Words) Floats {
    const wide: @Vector(lane_words, u32) = words;
    return @bitCast(wide << @splat(16));
}

/// `floats` with each NaN read as -inf, so a maximum over them ignores NaNs without relying on how a backend's `@max`
/// treats them.
fn numbers(words: Words) Floats {
    const f = floats(words);
    return @select(f32, f == f, f, @as(Floats, @splat(-std.math.inf(f32))));
}

/// The scalar rule the vectors reproduce: `best` starts at 0 and a word takes it only by being strictly larger, so
/// the first of equal maxima wins, a NaN never wins, and a NaN at 0 keeps 0.
pub fn reference(l: []const u16) u32 {
    if (l.len == 0) return 0;
    var best: usize = 0;
    for (l, 0..) |w, i| if (@as(f32, @bitCast(@as(u32, w) << 16)) > @as(f32, @bitCast(@as(u32, l[best]) << 16))) {
        best = i;
    };
    return @intCast(best);
}

fn isNan(word: u16) bool {
    return word & 0x7FFF > 0x7F80;
}

/// The first index of the largest value of `l`, as `reference` finds it: the maximum over the whole row first (a NaN
/// is no maximum), then the first word equal to it (+0 and -0 are equal).
pub fn argmax(l: []const u16) u32 {
    if (l.len == 0 or isNan(l[0])) return 0; // nothing is larger than a NaN
    const block = lane_words * unroll;
    var acc: [unroll]Floats = @splat(@splat(-std.math.inf(f32)));
    var i: usize = 0;
    while (i + block <= l.len) : (i += block) {
        inline for (0..unroll) |k| {
            const words: Words = l[i + k * lane_words ..][0..lane_words].*;
            acc[k] = @max(acc[k], numbers(words));
        }
    }
    var top: f32 = -std.math.inf(f32);
    inline for (acc) |a| top = @max(top, @reduce(.Max, a));
    for (l[i..]) |w| {
        const x: f32 = @bitCast(@as(u32, w) << 16);
        if (x > top) top = x; // a NaN is no larger
    }
    // the first word holding `top`; l[0] is no NaN, so the row's maximum is a number and some word equals it
    const at: Floats = @splat(top);
    i = 0;
    while (i + lane_words <= l.len) : (i += lane_words) {
        const words: Words = l[i..][0..lane_words].*;
        const hit = floats(words) == at;
        if (@reduce(.Or, hit)) {
            const bits: u16 = @bitCast(hit);
            return @intCast(i + @ctz(bits));
        }
    }
    for (l[i..], i..) |w, at_i| if (@as(f32, @bitCast(@as(u32, w) << 16)) == top) return @intCast(at_i);
    return 0;
}

test "vectors find what the scalar rule finds: first of equal maxima, NaNs ignored, a NaN first keeps 0, zeros equal" {
    var prng = std.Random.DefaultPrng.init(0x1d2c3b4a);
    const r = prng.random();
    const sizes = [_]usize{ 1, 2, 15, 16, 17, 63, 64, 65, 127, 128, 129, 1000, 4097, 201_024 };
    var row: [201_024]u16 = undefined;
    for (sizes) |n| {
        for (0..40) |round| {
            const l = row[0..n];
            // a handful of distinct values, so ties are common, with negatives, zeros of both signs and infinities
            const palette = [_]u16{ 0x0000, 0x8000, 0x3F80, 0xBF80, 0x4000, 0xC000, 0x7F80, 0xFF80, 0x4120, 0x0001, 0x8001 };
            for (l) |*w| w.* = if (round % 4 == 0) r.int(u16) else palette[r.uintLessThan(usize, if (round % 4 == 1) 4 else palette.len)];
            if (round % 8 == 3) for (0..4) |_| {
                l[r.uintLessThan(usize, n)] = 0x7FC0; // quiet NaNs anywhere
            };
            if (round % 8 == 7) l[0] = 0xFFC1; // a NaN at 0
            try std.testing.expectEqual(reference(l), argmax(l));
        }
    }
    // the maximum alone at the first word, the last word and across a vector boundary
    const l = row[0..201_024];
    @memset(l, 0xC0A0);
    for ([_]usize{ 0, 15, 16, 63, 64, 201_023 }) |at| {
        l[at] = 0x4001;
        try std.testing.expectEqual(@as(u32, @intCast(at)), argmax(l));
        l[at] = 0xC0A0;
    }
    // all NaN, all -inf, all +0 and -0 mixed: the first word
    @memset(l, 0x7FC0);
    try std.testing.expectEqual(@as(u32, 0), argmax(l));
    @memset(l, 0xFF80);
    try std.testing.expectEqual(@as(u32, 0), argmax(l));
    for (l, 0..) |*w, i| w.* = if (i % 3 == 0) 0x8000 else 0;
    try std.testing.expectEqual(@as(u32, 0), argmax(l));
    try std.testing.expectEqual(@as(u32, 0), argmax(&.{}));
}
