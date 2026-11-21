//! The draft head's tap mix weights: softmax(fuse_weight, -1) in f32, rounded to bf16 once (host-only, no device).
const std = @import("std");
const config = @import("config.zig");

/// Most draft layers (each has a mix row, and the Markov head one more) and most taps a mix row spans.
pub const max_rows = @typeInfo(@FieldType(config.Mtp, "local")).array.len + 1;
pub const max_taps = @typeInfo(@FieldType(config.Mtp, "taps")).array.len;

pub fn f32Of(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// f32 to bf16, round to nearest even (the inputs are finite).
pub fn bf16Of(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// softmax over each of `rows` rows of `cols` bf16 values in f32, the results rounded to bf16 once.
pub fn softmaxRows(fuse: []const u16, rows: usize, cols: usize, out: *[max_rows][max_taps]u16) void {
    std.debug.assert(fuse.len == rows * cols and rows <= max_rows and cols <= max_taps);
    for (0..rows) |r| {
        const x = fuse[r * cols ..][0..cols];
        var top = f32Of(x[0]);
        for (x[1..]) |v| top = @max(top, f32Of(v));
        var e: [max_taps]f32 = undefined;
        var sum: f32 = 0;
        for (x, 0..) |v, j| {
            e[j] = @exp(f32Of(v) - top);
            sum += e[j];
        }
        for (0..cols) |j| out[r][j] = bf16Of(e[j] / sum);
    }
}

const t = std.testing;

test "inkling2 draft weights: equal logits share the mass and a dominant one takes it" {
    var got: [max_rows][max_taps]u16 = undefined;
    const x = [_]u16{ bf16Of(1), bf16Of(1), bf16Of(1), bf16Of(1), bf16Of(40), bf16Of(-40) }; // row 1: 1, 40, -40
    softmaxRows(&x, 2, 3, &got);
    for (got[0][0..3]) |v| try t.expectEqual(bf16Of(1.0 / 3.0), v);
    try t.expectEqual(bf16Of(1), got[1][1]);
    try t.expect(f32Of(got[1][0]) < 1e-10 and f32Of(got[1][2]) < 1e-10);
}
