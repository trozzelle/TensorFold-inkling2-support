//! Inkling 2's Metal glue against FP64 references at the same rounding points, on synthetic rows: values within one
//! bf16 step, every row independent of the rows beside it, a non-finite row confined to its own outputs.
const std = @import("std");
const mtl = @import("metal");
const h = @import("harness.zig");

/// tf-inkling-kernels [glue|conv|attn|moe|stack|fused|split|xd ...]: the named case groups, every group by default.
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const g = try h.Gpu.init();
    var prng = std.Random.DefaultPrng.init(0x1c11);
    const r = prng.random();
    const groups = .{ .{ "glue", @import("cases_glue.zig") }, .{ "conv", @import("cases_conv.zig") }, .{ "attn", @import("cases_attn.zig") }, .{ "moe", @import("cases_moe.zig") }, .{ "stack", @import("cases_stack.zig") }, .{ "fused", @import("cases_fused.zig") }, .{ "split", @import("cases_split.zig") }, .{ "xd", @import("cases_xd.zig") } };
    inline for (groups) |grp| {
        var on = args.len < 2;
        for (args[1..]) |a| on = on or std.mem.eql(u8, a, grp[0]);
        if (on) try grp[1].run(&g, r);
    }
    std.debug.print("all cases pass\n", .{});
}
