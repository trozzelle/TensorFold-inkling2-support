//! MLX's affine 8-bit quantizer at groups of 64 along a row, on the host (no device): what `mx.quantize(w, group_size=64, bits=8)`
//! makes of bf16 weights, so the draft layers quantized at load dequantize to the reference's quantized weights. Per group of 64
//! values `w`: the range is extended to include 0 on its upper side, the scale is that range over 255 (at least 1e-7), signed so
//! the larger-magnitude edge anchors the grid, and rounded so the edge is a whole number of steps; the codes are
//! round((w - bias) / scale), at most 255. The scale and bias are stored rounded to bf16, the codes from the unrounded ones, and a
//! weight reads back as `scale * code + bias`.
const std = @import("std");

/// Values of a row that share one scale and bias.
pub const group = 64;
/// The largest code: 2^8 - 1.
pub const bins = 255;
/// mlx's `eps`: the smallest scale before the edge's rounding.
const min_scale: f32 = 1e-7;
/// bf16's largest value, the start of the group minimum (Limits<bfloat16_t>::max).
const bf16_max: f32 = @bitCast(@as(u32, 0x7f7f) << 16);

pub fn f32Of(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// f32 to bf16, round to nearest even (the inputs are finite).
pub fn bf16Of(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// A group's scale and bias, bf16 bits.
pub const Group = struct { scale: u16, bias: u16 };

/// The codes of one group of `group` bf16 values and its scale and bias (mlx's affine_quantize), or null when a value, the
/// scale or the bias is not finite (a bf16 scale or bias can overflow only for values near f32's range).
pub fn quantizeGroup(x: *const [group]u16, codes: *[group]u8) ?Group {
    var v: [group]f32 = undefined;
    var lo: f32 = bf16_max;
    var hi: f32 = 0;
    for (x, &v) |b, *f| {
        f.* = f32Of(b);
        if (!std.math.isFinite(f.*)) return null;
        lo = @min(lo, f.*);
        hi = @max(hi, f.*);
    }
    var scale = @max((hi - lo) / bins, min_scale);
    const low_edge = @abs(lo) > @abs(hi);
    scale = if (low_edge) scale else -scale;
    const edge = if (low_edge) lo else hi;
    const q0 = @round(edge / scale);
    const at_zero = q0 == 0;
    scale = if (at_zero) scale else edge / q0;
    const bias: f32 = if (at_zero) 0 else edge;
    const out: Group = .{ .scale = bf16Of(scale), .bias = bf16Of(bias) };
    if (!std.math.isFinite(f32Of(out.scale)) or !std.math.isFinite(f32Of(out.bias))) return null;
    for (v, codes) |f, *c| c.* = @intFromFloat(@max(@min(@round((f - bias) / scale), bins), 0));
    return out;
}

/// Rows of `k` bf16 values (a multiple of `group`) quantized: codes [rows][k], scales and biases [rows][k / group], bf16 bits.
/// Returns the first group (counting from the first row's) that quantizeGroup refuses, else null.
pub fn quantizeRows(src: []const u16, k: usize, codes: []u8, scales: []u16, biases: []u16) ?usize {
    std.debug.assert(k % group == 0 and src.len % k == 0 and codes.len == src.len and scales.len == src.len / group and biases.len == scales.len);
    for (0..src.len / group) |g| {
        const q = quantizeGroup(src[g * group ..][0..group], codes[g * group ..][0..group]) orelse return g;
        scales[g] = q.scale;
        biases[g] = q.bias;
    }
    return null;
}

/// What a code reads back as: the f32 `scale * code + bias` of the stored bf16 scale and bias.
pub fn dequantize(code: u8, scale: u16, bias: u16) f32 {
    return @mulAdd(f32, f32Of(scale), @floatFromInt(code), f32Of(bias));
}

const t = std.testing;

/// mlx's affine_quantize written out for one group in f64 with the same decisions, as an independent check of the f32 path's
/// anchoring and clamping: the codes, the scale and the bias as reals, and whether the edge's step count sits so near a half
/// that f32 and f64 can round it to different grids (a symmetric group's sits exactly on one).
fn reference(w: []const f64) struct { scale: f64, bias: f64, codes: [group]u8, near_half: bool } {
    var lo: f64 = std.math.floatMax(f64);
    var hi: f64 = 0;
    for (w) |x| {
        lo = @min(lo, x);
        hi = @max(hi, x);
    }
    var scale = @max((hi - lo) / bins, 1e-7);
    const low = @abs(lo) > @abs(hi);
    scale = if (low) scale else -scale;
    const edge = if (low) lo else hi;
    const steps = edge / scale;
    const q0 = @round(steps);
    if (q0 != 0) scale = edge / q0;
    const bias: f64 = if (q0 == 0) 0 else edge;
    var codes: [group]u8 = undefined;
    for (w, &codes) |x, *c| c.* = @intFromFloat(@max(@min(@round((x - bias) / scale), bins), 0));
    return .{ .scale = scale, .bias = bias, .codes = codes, .near_half = @abs(@abs(steps) - @floor(@abs(steps)) - 0.5) < 1e-4 };
}

fn bf16Values(prng: *std.Random.DefaultPrng, x: *[group]u16, spread: f32, shift: f32) void {
    for (x) |*b| b.* = bf16Of((prng.random().float(f32) * 2 - 1) * spread + shift);
}

test "inkling2 draft quantizer: the edge with the larger magnitude anchors the grid and the codes span 0 to 255" {
    var prng = std.Random.DefaultPrng.init(0x8b17);
    var x: [group]u16 = undefined;
    var codes: [group]u8 = undefined;
    for (0..200) |trial| {
        bf16Values(&prng, &x, 0.05, if (trial % 3 == 0) 0 else if (trial % 3 == 1) 0.01 else -0.01);
        const q = quantizeGroup(&x, &codes).?;
        const scale = f32Of(q.scale);
        const bias = f32Of(q.bias);
        // the larger edge is a code of 0 (bias at the edge); the other end of the range is within a step of code 255
        var lo: f32 = 0;
        var hi: f32 = 0;
        for (x) |b| {
            lo = @min(lo, f32Of(b));
            hi = @max(hi, f32Of(b));
        }
        const edge = if (@abs(lo) > @abs(hi)) lo else hi;
        try t.expectApproxEqRel(edge, bias, 1.0 / 128.0);
        for (x, codes) |b, c| {
            const back = dequantize(c, q.scale, q.bias);
            // a value reads back within one and a half steps (the other edge clips by up to a step; bf16 scale and bias add a little)
            try t.expect(@abs(back - f32Of(b)) <= 1.5 * @abs(scale) + 2 * @abs(bias) / 128.0);
        }
    }
}

test "inkling2 draft quantizer: a group's f32 decisions equal the f64 reference's, rounding only the scale and bias" {
    var prng = std.Random.DefaultPrng.init(0x11d0);
    var x: [group]u16 = undefined;
    var codes: [group]u8 = undefined;
    var differ: usize = 0;
    const trials = 4000;
    for (0..trials) |trial| {
        const spread: f32 = @exp(prng.random().float(f32) * 8 - 6);
        bf16Values(&prng, &x, spread, if (trial % 4 == 0) spread * 0.5 else 0);
        const q = quantizeGroup(&x, &codes).?;
        var w: [group]f64 = undefined;
        for (x, &w) |b, *f| f.* = f32Of(b);
        const want = reference(&w);
        if (want.near_half) continue;
        // the f32 scale and the f64 one round to bf16 within a step of each other (a rounding that sits between two bf16 values)
        const ws = bf16Of(@floatCast(want.scale));
        const wb = bf16Of(@floatCast(want.bias));
        try t.expect(@abs(@as(i32, q.scale) - @as(i32, ws)) <= 1 and @abs(@as(i32, q.bias) - @as(i32, wb)) <= 1);
        if (q.scale != ws or q.bias != wb) continue;
        // codes come from the unrounded scale and bias: an f32 quotient within an ulp of a half can round the other way
        for (codes, want.codes) |c, r| {
            if (c != r) {
                differ += 1;
                try t.expect(@abs(@as(i32, c) - @as(i32, r)) == 1);
            }
        }
    }
    try t.expect(differ * 2000 < trials * group);
}

test "inkling2 draft quantizer: an all-negative group extends its range to 0, an all-positive one does not" {
    var codes: [group]u8 = undefined;
    var x: [group]u16 = undefined;
    // negative: [-2, -1], so the range is [-2, 0]: scale 2/255 positive, edge -2 at code 0, and -1 near code 127.5
    for (&x, 0..) |*b, i| b.* = bf16Of(-2 + @as(f32, @floatFromInt(i)) / 63);
    var q = quantizeGroup(&x, &codes).?;
    try t.expectEqual(bf16Of(-2), q.bias);
    try t.expect(f32Of(q.scale) > 0 and @abs(f32Of(q.scale) - 2.0 / 255.0) < 2e-4);
    try t.expectEqual(@as(u8, 0), codes[0]);
    try t.expect(codes[63] > 120 and codes[63] < 135);
    // positive: [1, 2] keeps its own minimum: the larger edge is 2 (scale negative), the range 1/255
    for (&x, 0..) |*b, i| b.* = bf16Of(1 + @as(f32, @floatFromInt(i)) / 63);
    q = quantizeGroup(&x, &codes).?;
    try t.expectEqual(bf16Of(2), q.bias);
    try t.expect(f32Of(q.scale) < 0);
    try t.expectEqual(@as(u8, 0), codes[63]);
    try t.expectEqual(@as(u8, 255), codes[0]);
}

test "inkling2 draft quantizer: a zero group and a tiny one keep the minimum scale and read back as they were" {
    var codes: [group]u8 = undefined;
    var x: [group]u16 = @splat(0);
    var q = quantizeGroup(&x, &codes).?;
    try t.expectEqual(bf16Of(-1e-7), q.scale);
    try t.expectEqual(@as(u16, 0), q.bias);
    for (codes) |c| try t.expectEqual(@as(u8, 0), c);
    // all equal and positive: a range of 0 reads back exactly through the bias
    x = @splat(bf16Of(0.5));
    q = quantizeGroup(&x, &codes).?;
    for (codes) |c| try t.expectEqual(@as(f32, 0.5), dequantize(c, q.scale, q.bias));
    // a group of tiny values stays near zero
    for (&x, 0..) |*b, i| b.* = bf16Of((@as(f32, @floatFromInt(i)) - 31) * 1e-9);
    q = quantizeGroup(&x, &codes).?;
    for (x, codes) |b, c| try t.expect(@abs(dequantize(c, q.scale, q.bias) - f32Of(b)) < 2e-7);
}

test "inkling2 draft quantizer: a non-finite value refuses its group, rows report the first refused group" {
    var x: [3 * group]u16 = undefined;
    var prng = std.Random.DefaultPrng.init(3);
    for (&x) |*b| b.* = bf16Of(prng.random().float(f32) - 0.5);
    var codes: [3 * group]u8 = undefined;
    var scales: [3]u16 = undefined;
    var biases: [3]u16 = undefined;
    try t.expectEqual(@as(?usize, null), quantizeRows(&x, 3 * group, &codes, &scales, &biases));
    x[group + 5] = 0x7f80; // +inf
    try t.expectEqual(@as(?usize, 1), quantizeRows(&x, 3 * group, &codes, &scales, &biases));
    x[group + 5] = 0x7fc0; // NaN
    try t.expectEqual(@as(?usize, 1), quantizeRows(&x, 3 * group, &codes, &scales, &biases));
}
