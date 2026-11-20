//! The EXL3 pack's checkpoint.json (format kortexa-inkling2-exl3-v1): each layer's expert files in expert order.
const std = @import("std");
const config = @import("config.zig");

/// Experts [start, end) of a layer, in one safetensors file under experts/.
pub const Batch = struct { file: []const u8, start: u32, end: u32 };

fn num(v: std.json.Value) !u32 {
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.BadInkling2Pack;
    return @intCast(v.integer);
}

/// Each of `layers` layers' batches covering experts 0..experts once, in order; strings live in `arena`.
pub fn parse(arena: std.mem.Allocator, json: []const u8, layers: u32, experts: u32) ![]const []const Batch {
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena, json, .{}) catch return error.BadInkling2Pack;
    if (root != .object) return error.BadInkling2Pack;
    const o = root.object;
    const format = o.get("format") orelse return error.NotAnInkling2Pack;
    if (format != .string or !std.mem.eql(u8, format.string, config.pack_format)) return error.NotAnInkling2Pack;
    const status = o.get("status") orelse return error.IncompleteInkling2Pack;
    if (status != .string or !std.mem.eql(u8, status.string, "complete")) return error.IncompleteInkling2Pack;
    const lv = o.get("layers") orelse return error.MissingInkling2PackLayer;
    if (lv != .object) return error.BadInkling2Pack;
    const out = try arena.alloc([]const Batch, layers);
    for (out, 0..) |*slot, l| {
        var kb: [12]u8 = undefined;
        const entry = lv.object.get(try std.fmt.bufPrint(&kb, "{d}", .{l})) orelse return error.MissingInkling2PackLayer;
        if (entry != .object) return error.BadInkling2Pack;
        const bv = entry.object.get("batches") orelse return error.BadInkling2Pack;
        if (bv != .array or bv.array.items.len == 0) return error.BadInkling2Pack;
        const list = try arena.alloc(Batch, bv.array.items.len);
        var next: u32 = 0;
        for (bv.array.items, list) |b, *dst| {
            if (b != .object) return error.BadInkling2Pack;
            const f = b.object.get("file") orelse return error.BadInkling2Pack;
            if (f != .string) return error.BadInkling2Pack;
            dst.* = .{ .file = f.string, .start = try num(b.object.get("start") orelse return error.BadInkling2Pack), .end = try num(b.object.get("end") orelse return error.BadInkling2Pack) };
            if (dst.start != next or dst.end <= dst.start) return error.Inkling2PackGap;
            next = dst.end;
        }
        if (next != experts) return error.Inkling2PackGap;
        slot.* = list;
    }
    return out;
}

/// dir/experts/checkpoint.json read and admitted.
pub const Pack = struct {
    arena: std.heap.ArenaAllocator,
    layers: []const []const Batch,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, layers: u32, experts: u32) !Pack {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const path = try std.fs.path.join(a, &.{ dir, "experts", "checkpoint.json" });
        const json = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
        return .{ .arena = arena, .layers = try parse(a, json, layers, experts) };
    }

    pub fn deinit(p: *Pack) void {
        p.arena.deinit();
    }
};
