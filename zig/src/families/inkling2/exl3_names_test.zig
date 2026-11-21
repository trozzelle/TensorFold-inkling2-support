//! An EXL3 config: quant_method exl3 with the pack's format; its layers expect the router and no source experts.
const std = @import("std");
const config = @import("config.zig");
const names = @import("names.zig");
const plan = @import("plan.zig");

const fixture = @embedFile("fixtures/tiny_config.json");

fn withQuant(a: std.mem.Allocator, quant: []const u8) ![]u8 {
    var v = try std.json.parseFromSliceLeaky(std.json.Value, a, fixture, .{});
    const q = try std.json.parseFromSliceLeaky(std.json.Value, a, quant, .{});
    try v.object.put(a, "quantization_config", q);
    return std.json.Stringify.valueAlloc(a, v, .{});
}

const Collect = struct {
    a: std.mem.Allocator,
    list: std.ArrayList([]const u8) = .empty,
    pub fn put(k: *Collect, name: []const u8, _: anytype, _: []const u64) !void {
        try k.list.append(k.a, try k.a.dupe(u8, name));
    }
};

test "inkling2 exl3: the pack's format is admitted, another EXL3 format is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: config.Why = .{};
    const c = try config.parse(a, try withQuant(a, "{\"quant_method\": \"exl3\", \"format\": \"kortexa-inkling2-exl3-v1\"}"), &why);
    try std.testing.expectEqual(config.Quant.exl3, c.quant);
    try std.testing.expectError(error.Quant, config.parse(a, try withQuant(a, "{\"quant_method\": \"exl3\", \"format\": \"x\"}"), &why));
}

test "inkling2 exl3: layers expect the router weight and bias, no source experts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: config.Why = .{};
    const c = try config.parse(a, try withQuant(a, "{\"quant_method\": \"exl3\", \"format\": \"kortexa-inkling2-exl3-v1\"}"), &why);
    var prec: [config.max_layers]plan.Precision = @splat(.exl3);
    var k: Collect = .{ .a = a };
    try names.all(&c, prec[0..c.text.layers], .model, &k);
    var router: usize = 0;
    for (k.list.items) |n| {
        try std.testing.expect(std.mem.indexOf(u8, n, "mlp.experts.") == null);
        if (std.mem.endsWith(u8, n, "mlp.gate.weight")) router += 1;
    }
    try std.testing.expectEqual(@as(usize, c.text.layers), router);
}
