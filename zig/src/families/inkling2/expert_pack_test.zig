//! The EXL3 pack's checkpoint.json admission: format, status, every layer, experts covered once in order.
const std = @import("std");
const pack = @import("expert_pack.zig");

const good =
    \\{"format": "kortexa-inkling2-exl3-v1", "status": "complete", "layers": {
    \\ "0": {"K": 3, "batches": [{"start": 0, "end": 8, "file": "layers/00/experts-000-007.safetensors"},
    \\                           {"start": 8, "end": 16, "file": "layers/00/experts-008-015.safetensors"}]},
    \\ "1": {"K": 2, "batches": [{"start": 0, "end": 16, "file": "layers/01/experts-000-015.safetensors"}]}}}
;

test "inkling2 pack: a complete pack lists each layer's files in expert order" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const layers = try pack.parse(arena.allocator(), good, 2, 16);
    try std.testing.expectEqual(@as(usize, 2), layers[0].len);
    try std.testing.expectEqual(@as(u32, 8), layers[0][1].start);
    try std.testing.expectEqualStrings("layers/01/experts-000-015.safetensors", layers[1][0].file);
}

test "inkling2 pack: wrong format, incomplete status, a missing layer and a gap are refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { json: []const u8, err: anyerror }{
        .{ .json = "{\"format\": \"other\", \"status\": \"complete\", \"layers\": {}}", .err = error.NotAnInkling2Pack },
        .{ .json = "{\"format\": \"kortexa-inkling2-exl3-v1\", \"status\": \"partial\", \"layers\": {}}", .err = error.IncompleteInkling2Pack },
        .{ .json = "{\"format\": \"kortexa-inkling2-exl3-v1\", \"status\": \"complete\", \"layers\": {\"0\": {\"batches\": [{\"start\": 0, \"end\": 16, \"file\": \"f\"}]}}}", .err = error.MissingInkling2PackLayer },
        .{ .json = "{\"format\": \"kortexa-inkling2-exl3-v1\", \"status\": \"complete\", \"layers\": {\"0\": {\"batches\": [{\"start\": 0, \"end\": 8, \"file\": \"f\"}, {\"start\": 9, \"end\": 16, \"file\": \"g\"}]}, \"1\": {\"batches\": [{\"start\": 0, \"end\": 16, \"file\": \"h\"}]}}}", .err = error.Inkling2PackGap },
    };
    for (cases) |c| try std.testing.expectError(c.err, pack.parse(a, c.json, 2, 16));
}
