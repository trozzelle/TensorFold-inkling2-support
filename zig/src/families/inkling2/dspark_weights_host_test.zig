//! The draft head's mix weights at the tiny header fixture's shapes (host only, no device).
const std = @import("std");
const config = @import("config.zig");
const config_test = @import("config_test.zig");
const names_test = @import("names_test.zig");
const mix = @import("dspark_alpha.zig");

const t = std.testing;
const f32Of = mix.f32Of;
const bf16Of = mix.bf16Of;
const max_rows = mix.max_rows;
const max_taps = mix.max_taps;
const softmaxRows = mix.softmaxRows;

fn hostSoftmaxBf16(x: []const u16, j: usize) u16 {
    var top: f64 = -std.math.inf(f64);
    for (x) |v| top = @max(top, f32Of(v));
    var sum: f64 = 0;
    for (x) |v| sum += @exp(f32Of(v) - top);
    return bf16Of(@floatCast(@exp(f32Of(x[j]) - top) / sum));
}

test "inkling2 draft weights: alpha is the bf16 of a softmax over each fuse_weight row, at the tiny header's shape" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: config.Why = .{};
    const c = try config.parse(a, config_test.tiny, &why);
    const m = c.mtp.?;
    const header = try names_test.unfold(a, @embedFile("fixtures/tiny_mtp_header.txt"), "inkling2-tiny-mtp");
    const e = header.get("fuse_weight").?;
    try t.expectEqual(@as(usize, 2), e.rank);
    try t.expectEqual(@as(usize, m.layers + 1), e.shape[0]);
    try t.expectEqual(@as(usize, m.taps_n), e.shape[1]);
    // the tiny head has two taps; also run the real head's eight
    for ([_]usize{ e.shape[1], 8 }) |cols| {
        const rows = e.shape[0];
        const fuse = try a.alloc(u16, rows * cols);
        var prng = std.Random.DefaultPrng.init(0x5eed);
        for (fuse) |*v| v.* = bf16Of((prng.random().float(f32) - 0.5) * 6);
        var got: [max_rows][max_taps]u16 = undefined;
        softmaxRows(fuse, rows, cols, &got);
        for (0..rows) |r| {
            var sum: f32 = 0;
            for (0..cols) |j| {
                try t.expectEqual(hostSoftmaxBf16(fuse[r * cols ..][0..cols], j), got[r][j]);
                sum += f32Of(got[r][j]);
            }
            try t.expectApproxEqAbs(@as(f32, 1), sum, 0.02);
        }
    }
}

test "inkling2 draft weights: alpha softmax matches host at Mini's 4-row 4-tap shape" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: config.Why = .{};
    const c = try config.parse(a, config_test.mini, &why);
    const m = c.mtp.?;
    const rows: usize = m.layers + 1;
    const cols: usize = m.taps_n;
    try t.expectEqual(@as(usize, 4), rows);
    try t.expectEqual(@as(usize, 4), cols);
    const fuse = try a.alloc(u16, rows * cols);
    var prng = std.Random.DefaultPrng.init(0x5eed2);
    for (fuse) |*v| v.* = bf16Of((prng.random().float(f32) - 0.5) * 6);
    var got: [max_rows][max_taps]u16 = undefined;
    softmaxRows(fuse, rows, cols, &got);
    for (0..rows) |r| {
        var sum: f32 = 0;
        for (0..cols) |j| {
            try t.expectEqual(hostSoftmaxBf16(fuse[r * cols ..][0..cols], j), got[r][j]);
            sum += f32Of(got[r][j]);
        }
        try t.expectApproxEqAbs(@as(f32, 1), sum, 0.02);
    }
}
