//! Inkling 2 kernel checks: moe (filled by its kernel task).
const std = @import("std");
const h = @import("harness.zig");

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    _ = g;
    _ = r;
}
