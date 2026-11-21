//! Inkling 2 Small with EXL3 experts served by the lane host: one stream, plain and copy-drafted rounds, and the DSpark draft
//! head's drafts when the model folder has its weights.
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
    draft: ?*fl.dspark.Drafter = null, // the DSpark draft head, when drafts are on and the model folder has it
    wrap: ?lanes.drafted.Drafted = null, // the head in front of `metal`
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
        if (h.wrap) |*w| w.deinit();
        if (h.draft) |d| d.deinit();
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
    // how a round drafts and how wide the draft layers' weights are: only a load with drafts reads them, once
    var drafting: fl.dspark.Drafting = .{ .depth = .default };
    var bits: fl.dspark_weights.Bits = .default;
    if (o.drafts) {
        drafting = fl.dspark.Drafting.parse(if (std.c.getenv("INKLING_DRAFT_DEPTH")) |v| std.mem.span(v) else null, if (std.c.getenv("INKLING_DRAFT_CONFIDENCE")) |v| std.mem.span(v) else null) catch |err| {
            problem.* = switch (err) {
                error.BadDraftDepth => "INKLING_DRAFT_DEPTH is not a draft depth: give a whole number from 1 to 7, or auto; unset it for 7 (4 with INKLING_DRAFT_CONFIDENCE=off)",
                error.BadDraftConfidence => "INKLING_DRAFT_CONFIDENCE is not a draft confidence: give a probability above 0 and below 1, or off to verify every draft a round holds; unset for 0.7",
            };
            return null;
        };
        bits = fl.dspark_weights.Bits.parse(if (std.c.getenv("INKLING_DRAFT_BITS")) |v| std.mem.span(v) else null) catch {
            problem.* = "INKLING_DRAFT_BITS is not a draft weight width: give 8 (quantized at load) or 16 (bf16); unset it for 8";
            return null;
        };
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    // with drafts, the model also loads the draft head's weights (mtp.safetensors) when the folder has them: they are among the
    // checkpoint's shards, so the budget below counts them
    const model = fl.Model.load(gpa, io, o.dir, .{ .drafts = o.drafts, .draft_bits = bits }) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Inkling 2 EXL3 engine cannot load this checkpoint ({s}){s}", .{ @errorName(err), if (o.drafts) "; with drafts on it also loads mtp.safetensors, which --no-drafts skips" else "" });
        return null;
    };
    errdefer model.deinit();
    // the head before the budget: its fold joins the checkpoint's shards when it is made, and its scratch is known
    var draft: ?*fl.dspark.Drafter = null;
    if (model.draft) |*head_weights| {
        draft = fl.dspark.Drafter.init(gpa, model.device, model, head_weights, drafting.options()) catch |err| {
            problem.* = try std.fmt.allocPrint(a, "the Inkling 2 draft head cannot start ({s}); --no-drafts serves without it", .{@errorName(err)});
            model.deinit();
            return null;
        };
    } else if (o.drafts) std.log.info("inkling2: no draft head loaded; serving plain and copy-drafted rounds", .{});
    errdefer if (draft) |d| d.deinit();
    if (draft) |d| std.log.info("inkling2: DSpark draft head, {s} layers, {s}, {d:.2} GiB resident", .{ d.w.bits.label(), try drafting.describe(a), gib(d.residentBytes(1)) });
    // refusals that return null free what they hold themselves: errdefer runs only for errors
    var ram: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &ram, &len, null, 0) != 0) ram = 0;
    // --load-limit-gb, else INKLING_LOAD_LIMIT_GB, else 75% of RAM
    const env_text: ?[]const u8 = if (std.c.getenv("INKLING_LOAD_LIMIT_GB")) |v| std.mem.span(v) else null;
    const limit_gb: ?f64 = o.load_limit_gb orelse fl.budget.parseLimit(env_text) catch {
        free(model, draft);
        problem.* = "INKLING_LOAD_LIMIT_GB is not a load limit: give a number of GB above 0 (or --load-limit-gb), or unset it for 75% of RAM";
        return null;
    };
    const c = &model.config.text;
    // the checkpoint's shards, the draft head's file and fold among them
    const weights: u64 = model.checkpoint.residentBytes();
    // what the caches' room is left after: those, the scratch but for the partials that grow with the context (`longest`
    // counts them beside the caches), and the draft head's scratch and its one stream's state
    const held = try fl.budget.held(model, if (draft) |d| d.extraBytes(1) else 0);
    const room = fl.budget.caches(ram, model.device.maxWorkingSet(), limit_gb, held) catch {
        free(model, draft);
        problem.* = "--load-limit-gb is not a load limit: give a number of GB above 0, or leave it out for 75% of RAM";
        return null;
    };
    const fits = fl.budget.longest(c, c.layers, room, c.max_context);
    const window: i64 = o.context orelse @intCast(fits);
    if (window < 1 or window > fits) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: this Mac's Inkling 2 caches fit {d} tokens under its load limit (75% of RAM, or --load-limit-gb, or INKLING_LOAD_LIMIT_GB)", .{ window, fits });
        free(model, draft); // after the message, so a failed allocation leaves them to the errdefers
        return null;
    }
    // the partials of this window come off what the caches may hold
    const budget = room - fl.state.partBytes(c, @as(usize, @intCast(window)) + 1);
    std.log.info("inkling2: {d:.1} GiB of weights, {d:.1} GiB for caches, context up to {d} tokens", .{ gib(weights), gib(budget), window });
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.io = io;
    h.model = model;
    h.resident = null;
    h.draft = draft;
    h.wrap = null;
    h.metal = try fl.backend.Metal.init(gpa, model, .{ .capacity = @intCast(window), .budget = budget });
    errdefer h.metal.deinit();
    try h.metal.warm();
    errdefer if (h.wrap) |*w| w.deinit();
    var facts = h.metal.facts();
    if (draft) |d| {
        // the target writes the drafter's taps from here on, so the windows are timed with them, as a drafted round verifies
        h.wrap = try fl.dspark.wrap(gpa, h.metal.backend(), d);
        try h.metal.measure();
        var text: [256]u8 = undefined;
        std.log.info("inkling2: window forwards by rows, ms: {s}", .{costsText(&text, h.metal.facts().window_costs)});
        facts = fl.dspark.laneFacts(&h.wrap.?, h.metal.facts(), drafting.depth);
    }
    h.config = try lanes.Config.init(gpa, facts, fl.state.window_rows, fl.state.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, if (h.wrap) |*w| w.backend() else h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = 1, .context_window = @intCast(window), .prefill_step = fl.state.prompt_rows, .structures = true, .logprobs = true });
    try fl.residency.hold(h);
    errdefer fl.residency.drop(h);
    // Where the weights exceed the residency set's cap they stay outside it; the busy probe then keeps only the scratch
    // and the tables in use while a request runs, and does nothing for the weights.
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick, .probe = mtl.keepalive.Target.busy };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// What a refusal frees: the draft head before the model it reads.
fn free(model: *fl.Model, draft: ?*fl.dspark.Drafter) void {
    if (draft) |d| d.deinit();
    model.deinit();
}

/// The timed window forwards, "rows: ms" each, for the log.
fn costsText(buf: []u8, costs: []const lanes.config.Cost) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    for (costs, 0..) |cost, i| w.print("{s}{d}: {d:.1}", .{ if (i == 0) "" else ", ", cost.width, cost.ms }) catch break;
    return w.buffered();
}
