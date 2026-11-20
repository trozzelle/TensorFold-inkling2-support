//! Inkling 2 Small with EXL3 experts served by the lane host: one stream, plain and copy-drafted rounds.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const fl = tf.inkling2;
const lanes = tf.lanes;
const Resident = @import("resident.zig").Resident;

const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *fl.Model,
    metal: *fl.backend.Metal,
    warm: mtl.keepalive.Target = undefined, // the residency set's, or the bare queue's, for the lane host's idle ticker
    resident: ?Resident = null, // weights and scratch kept wired between requests, when they fit the 70% limit
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        h.core.deinit();
        h.config.deinit(h.gpa);
        fl.residency.drop(h);
        h.metal.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (try o.slideRefusal(a, "Macs")) |why| {
        problem.* = why;
        return null;
    }
    if (o.vision != null) {
        problem.* = "--vision: the Inkling 2 EXL3 engine reads text only; serve without --vision";
        return null;
    }
    if (o.lanes_fixed and o.lanes > 1) {
        problem.* = "--parallel: the Inkling 2 EXL3 engine serves one stream; give --parallel 1";
        return null;
    }
    if (o.drafts) std.log.info("inkling2: no draft head yet; serving plain and copy-drafted rounds", .{});
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = fl.Model.load(gpa, io, o.dir, .{}) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Inkling 2 EXL3 engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    // refusals that return null free what they hold themselves: errdefer runs only for errors
    var ram: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &ram, &len, null, 0) != 0) ram = 0;
    // --load-limit-gb, else INKLING_LOAD_LIMIT_GB, else 75% of RAM
    const env_text: ?[]const u8 = if (std.c.getenv("INKLING_LOAD_LIMIT_GB")) |v| std.mem.span(v) else null;
    const limit_gb: ?f64 = o.load_limit_gb orelse fl.budget.parseLimit(env_text) catch {
        model.deinit();
        problem.* = "INKLING_LOAD_LIMIT_GB is not a load limit: give a number of GB above 0 (or --load-limit-gb), or unset it for 75% of RAM";
        return null;
    };
    const c = &model.config.text;
    const weights: u64 = model.checkpoint.residentBytes();
    const scratch: u64 = try fl.state.scratchBytes(c);
    const budget = fl.budget.caches(ram, model.device.maxWorkingSet(), limit_gb, weights + scratch) catch {
        model.deinit();
        problem.* = "--load-limit-gb is not a load limit: give a number of GB above 0, or leave it out for 75% of RAM";
        return null;
    };
    const fits = fl.budget.longest(c, c.layers, budget, c.max_context);
    const window: i64 = o.context orelse @intCast(fits);
    if (window < 1 or window > fits) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: this Mac's Inkling 2 caches fit {d} tokens under its load limit (75% of RAM, or --load-limit-gb, or INKLING_LOAD_LIMIT_GB)", .{ window, fits });
        model.deinit(); // after the message, so a failed allocation leaves the model to the errdefer
        return null;
    }
    std.log.info("inkling2: {d:.1} GiB of weights, {d:.1} GiB for caches, context up to {d} tokens", .{ @as(f64, @floatFromInt(weights)) / (1 << 30), @as(f64, @floatFromInt(budget)) / (1 << 30), window });
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.io = io;
    h.model = model;
    h.resident = null;
    h.metal = try fl.backend.Metal.init(gpa, model, .{ .capacity = @intCast(window), .budget = budget });
    errdefer h.metal.deinit();
    try h.metal.warm();
    h.config = try lanes.Config.init(gpa, h.metal.facts(), fl.state.window_rows, fl.state.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = 1, .context_window = @intCast(window), .prefill_step = fl.state.prompt_rows, .structures = true, .logprobs = true });
    try fl.residency.hold(h);
    errdefer fl.residency.drop(h);
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}
