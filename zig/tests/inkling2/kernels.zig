//! Inkling 2's Metal glue against FP64 references at the same rounding points, on synthetic rows: values within one
//! bf16 step, every row independent of the rows beside it, a non-finite row confined to its own outputs.
const std = @import("std");
const mtl = @import("metal");
const h = @import("harness.zig");

/// tf-inkling-kernels [seed=N] [glue|conv|attn|moe|stack|fused|split|xd|dspark|gemm|pattn|xp ...]: the named case groups, every group by
/// default, on inputs drawn from seed N (0x1c11 by default); `time` instead times the fused norm kernels.
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const g = try h.Gpu.init();
    var seed: u64 = 0x1c11;
    for (args[1..]) |a| if (std.mem.startsWith(u8, a, "seed=")) {
        seed = try std.fmt.parseInt(u64, a[5..], 0);
    };
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (args[1..]) |a| if (std.mem.eql(u8, a, "time")) return @import("time_norms.zig").run(&g, r);
    const groups = .{ .{ "glue", @import("cases_glue.zig") }, .{ "conv", @import("cases_conv.zig") }, .{ "attn", @import("cases_attn.zig") }, .{ "moe", @import("cases_moe.zig") }, .{ "stack", @import("cases_stack.zig") }, .{ "fused", @import("cases_fused.zig") }, .{ "split", @import("cases_split.zig") }, .{ "xd", @import("cases_xd.zig") }, .{ "dspark", @import("cases_dspark.zig") }, .{ "gemm", @import("cases_gemm.zig") }, .{ "pattn", @import("cases_pattn.zig") }, .{ "xp", @import("cases_xp.zig") }, .{ "controllers", @import("cases_controllers.zig") } };
    inline for (groups) |grp| {
        var on = true;
        for (args[1..]) |a| if (!std.mem.startsWith(u8, a, "seed=")) {
            on = false;
        };
        for (args[1..]) |a| on = on or std.mem.eql(u8, a, grp[0]);
        if (on) try grp[1].run(&g, r);
    }
    std.debug.print("all cases pass\n", .{});
}
