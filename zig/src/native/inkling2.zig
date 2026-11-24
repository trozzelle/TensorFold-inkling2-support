//! Inkling 2 (EXL3 and native NVFP4/MXFP8 checkpoints) served by the lane host: one to eight streams, shared verify
//! rounds, interleaved prefill, plain and copy-drafted rounds, and the DSpark draft head's drafts when the model folder
//! has its weights.
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
    cache: ?api.prompt_cache.Store = null, // kept prompt states, in what the streams' caches leave of the budget
    cache_cap: u64 = 0, // --prompt-cache-gib's bytes (default_states when unset)

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        if (h.cache) |*store| store.deinit();
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
        problem.* = "--vision: the Inkling 2 engine reads text only; serve without --vision";
        return null;
    }
    const served = resolvedLanes(o) catch {
        problem.* = "--parallel: the Inkling 2 engine serves 1 to 8 streams; give --parallel 1 through 8";
        return null;
    };
    if (o.prompt_cache_gib) |g| if (!std.math.isFinite(g) or g < 0) {
        problem.* = "--prompt-cache-gib: give a number of GiB at least 0 (0 turns the prompt cache off)";
        return null;
    };
    // how a round drafts and how wide the draft layers' weights are: only a load with drafts reads them, once
    var drafting: fl.dspark.Drafting = .{ .depth = .default };
    var bits: ?fl.dspark_weights.Bits = null;
    if (o.drafts) {
        drafting = fl.dspark.Drafting.parse(if (std.c.getenv("INKLING_DRAFT_DEPTH")) |v| std.mem.span(v) else null, if (std.c.getenv("INKLING_DRAFT_CONFIDENCE")) |v| std.mem.span(v) else null) catch |err| {
            problem.* = switch (err) {
                error.BadDraftDepth => "INKLING_DRAFT_DEPTH is not a draft depth: give a whole number from 1 to 7, or auto; unset it for 7 (4 with INKLING_DRAFT_CONFIDENCE=off)",
                error.BadDraftConfidence => "INKLING_DRAFT_CONFIDENCE is not a draft confidence: give a probability above 0 and below 1, or off to verify every draft a round holds; unset for 0.7",
            };
            return null;
        };
        bits = fl.dspark_weights.Bits.parseOpt(if (std.c.getenv("INKLING_DRAFT_BITS")) |v| std.mem.span(v) else null) catch {
            problem.* = "INKLING_DRAFT_BITS is not a draft weight width: give 8 (quantized at load) or 16 (bf16); unset for model default";
            return null;
        };
    }
    // rows of a prompt chunk at most: read once, and the load's scratch and budget and the host's prefill step follow it
    const prefill_rows = fl.state.parsePrefillRows(if (std.c.getenv("INKLING_PREFILL_ROWS")) |v| std.mem.span(v) else null) catch {
        problem.* = try std.fmt.allocPrint(a, "INKLING_PREFILL_ROWS is not a prompt chunk size: give a multiple of {d} from {d} to {d}, or unset it for {d} ({d} on a GPU without tensor units)", .{ fl.state.prefill_unit, fl.state.prefill_unit, fl.state.max_prefill_rows, fl.state.default_prefill_rows, fl.state.per_op_prefill_rows });
        return null;
    };
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    // with drafts, the model also loads the draft head's weights (mtp.safetensors) when the folder has them: they are among the
    // checkpoint's shards, so the budget below counts them
    const model = fl.Model.load(gpa, io, o.dir, .{ .drafts = o.drafts, .draft_bits = bits, .prefill_rows = prefill_rows }) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Inkling 2 engine cannot load this checkpoint ({s}){s}", .{ @errorName(err), if (o.drafts) "; with drafts on it also loads mtp.safetensors, which --no-drafts skips" else "" });
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
    if (draft) |d| std.log.info("inkling2: DSpark draft head, {s} layers, {s}, {d:.2} GiB resident", .{ d.w.bits.label(), try drafting.describe(a), gib(d.residentBytes(served)) });
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
    // counts them beside the caches), and the draft head's scratch and its served streams' states
    const held = try fl.budget.held(model, if (draft) |d| d.extraBytes(served) else 0);
    const room = fl.budget.caches(ram, model.device.maxWorkingSet(), limit_gb, held) catch {
        free(model, draft);
        problem.* = "--load-limit-gb is not a load limit: give a number of GB above 0, or leave it out for 75% of RAM";
        return null;
    };
    const fits = fl.budget.longest(c, c.virtualLayers(), room, c.max_context, model.prefill_rows);
    const window: i64 = o.context orelse @intCast(@min(fits, fl.budget.default_window_most));
    if (window < 1 or window > fits) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: this Mac's Inkling 2 caches fit {d} tokens under its load limit (75% of RAM, or --load-limit-gb, or INKLING_LOAD_LIMIT_GB)", .{ window, fits });
        free(model, draft); // after the message, so a failed allocation leaves them to the errdefers
        return null;
    }
    // the partials of this window come off what the caches may hold
    const budget = room - fl.state.partBytes(c, @as(usize, @intCast(window)) + 1);
    const fixed_one = fl.state.cacheBytes(c, 1, c.virtualLayers(), model.prefill_rows);
    if (!fitsFixedFloor(c, budget, model.prefill_rows, served)) {
        const most_fit: u64 = budget / fixed_one;
        problem.* = try std.fmt.allocPrint(a, "--parallel {d}: {d} Inkling 2 streams need {d:.2} GiB of fixed rings, more than the {d:.2} GiB left for caches ({d} fit)", .{ served, served, gib(@as(u64, served) * fixed_one), gib(budget), most_fit });
        free(model, draft);
        return null;
    }
    std.log.info("inkling2: {d:.1} GiB of weights, {d:.1} GiB for caches, context up to {d} tokens (the caches fit {d}), {d} stream(s)", .{ gib(weights), gib(budget), window, fits, served });
    std.log.info("inkling2: prompt chunks of {d} rows on the {s} path, {d:.2} GiB of scratch", .{ model.prefill_rows, @tagName(model.promptPath()), gib(try fl.state.scratchBytes(c, 0, model.chunking())) });
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.io = io;
    h.model = model;
    h.resident = null;
    h.draft = draft;
    h.wrap = null;
    h.metal = try fl.backend.Metal.init(gpa, model, .{ .capacity = @intCast(window), .budget = budget, .streams = served });
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
        if (served > 1) {
            var shared_text: [256]u8 = undefined;
            std.log.info("inkling2: shared forwards by rows, ms: {s}", .{costsText(&shared_text, h.metal.facts().shared_costs)});
        }
        facts = fl.dspark.laneFacts(&h.wrap.?, h.metal.facts(), drafting.depth);
    }
    h.config = try lanes.Config.init(gpa, facts, fl.state.window_rows, fl.state.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, if (h.wrap) |*w| w.backend() else h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = served, .context_window = @intCast(window), .prefill_step = model.prefill_rows, .decode_share = o.decode_share, .structures = true, .logprobs = true });
    h.cache = null;
    // kept states take what the streams leave, at most this: by default well under the room, which a load has never filled
    h.cache_cap = if (o.prompt_cache_gib) |g| blk: {
        const cap_bytes = g * (1 << 30);
        break :blk if (cap_bytes >= @as(f64, @floatFromInt(budget))) budget else @intFromFloat(cap_bytes);
    } else @min(budget, fl.budget.default_states);
    if (h.cache_cap > 0) {
        // a prompt row's bits do not depend on its chunk: any row is a place to keep or resume a state, and a pass cuts only at
        // its marks and its own chunk rows
        const rules: api.prompt_cache.Rules = .{ .lookahead = 0, .planned = false, .min_prompt = 0, .chunk = @intCast(model.prefill_rows), .chunk_starts = false };
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = h, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, rules, @min(h.cache_cap, budget));
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
        h.metal.room = .{ .ctx = h, .fit = fitCache };
        std.log.info("inkling2: kept prompt states take up to {d:.1} GiB of the caches' room", .{gib(@min(h.cache_cap, budget))});
    } else std.log.info("inkling2: prompt cache off", .{});
    errdefer if (h.cache) |*store| store.deinit();
    try fl.residency.hold(h);
    errdefer fl.residency.drop(h);
    // Where the weights exceed the residency set's cap they stay outside it; the busy probe then keeps only the scratch
    // and the tables in use while a request runs, and does nothing for the weights.
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick, .probe = mtl.keepalive.Target.busy };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// The prompt cache's copies of a stream: the target's caches at a mark (backend.zig), and the drafter's context there.
const Snaps = struct {
    const M = fl.backend.Metal;
    fn host(ptr: *anyopaque) *Host {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytes(ptr: *anyopaque, at: u32) u64 {
        const h = host(ptr);
        return M.snapBytes(h.metal, at) + if (h.draft) |d| fl.dspark.Drafter.keptBytesFor(d.mtp) else 0;
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const h = host(ptr);
        const saved = try M.snapSave(h.metal, owner, at);
        if (h.draft) |d| d.keepLane(@ptrCast(@alignCast(owner.?)), @ptrCast(@alignCast(saved))) catch |err| {
            M.snapDrop(h.metal, saved);
            return err;
        };
        return saved;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores; // the prompt pass restores, before its first chunk
    }
    fn drop(ptr: *anyopaque, saved: *anyopaque) void {
        const h = host(ptr);
        if (h.draft) |d| d.dropKeep(@ptrCast(@alignCast(saved)));
        M.snapDrop(h.metal, saved);
    }
};

/// Metal.room's fit: the kept states' budget is what the streams leave, at most the cap; the least valuable go first.
fn fitCache(ctx: *anyopaque, bytes: u64, restoring: ?*anyopaque, prompt: []const u32) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const store = if (h.cache) |*owned| owned else return;
    store.budget = @min(h.cache_cap, bytes);
    const Left = struct {
        s: *const api.prompt_cache.Store,
        pub fn fits(l: @This()) bool {
            return l.s.held <= l.s.budget;
        }
    };
    _ = store.giveBack(restoring, prompt, Left{ .s = store });
}

fn resolvedLanes(o: api.Open) !u32 {
    if (!o.lanes_fixed) return 1;
    if (o.lanes == 0 or o.lanes > fl.state.max_streams) return error.InvalidParallel;
    return o.lanes;
}

fn fitsFixedFloor(c: *const fl.config.Text, budget: u64, prefill_rows: usize, served: u32) bool {
    const fixed_one = fl.state.cacheBytes(c, 1, c.virtualLayers(), prefill_rows);
    return @as(u64, served) * fixed_one <= budget;
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

const CacheFixture = struct {
    live: usize = 0,
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 100 + at;
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) !api.prompt_cache.Saved {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        const held = try std.testing.allocator.create(u32);
        held.* = at;
        self.live += 1;
        return held;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: api.prompt_cache.Saved) !void {}
    fn drop(ptr: *anyopaque, saved: api.prompt_cache.Saved) void {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        std.testing.allocator.destroy(@as(*u32, @ptrCast(@alignCast(saved))));
        self.live -= 1;
    }
    fn snapshots(self: *@This()) api.prompt_cache.Snapshots {
        return .{ .ptr = self, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
};

test "inkling2 host: cache callback updates the owned store rather than a temporary value" {
    var f: CacheFixture = .{};
    var h: Host = undefined;
    h.cache_cap = 500;
    h.cache = api.prompt_cache.Store.init(std.testing.allocator, f.snapshots(), .{ .min_prompt = 0 }, 500);
    defer h.cache.?.deinit();
    fitCache(&h, 200, null, &.{});
    try std.testing.expectEqual(@as(u64, 200), h.cache.?.budget);
    fitCache(&h, 800, null, &.{});
    try std.testing.expectEqual(@as(u64, 500), h.cache.?.budget);
}

test "inkling2 host: cache callback evicts from the real store and preserves the restored entry" {
    var f: CacheFixture = .{};
    var h: Host = undefined;
    h.cache_cap = 1000;
    h.cache = api.prompt_cache.Store.init(std.testing.allocator, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer h.cache.?.deinit();
    const store = &h.cache.?;
    try std.testing.expect(store.keep(&.{ 1, 1, 1, 1, 9 }, 4, null, &.{}, &.{}));
    try std.testing.expect(store.keep(&.{ 2, 2, 2, 2, 2, 2, 9 }, 6, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(u64, 210), store.held);
    const kept = store.entries.items[0].saved;
    fitCache(&h, 110, kept, &.{ 1, 1, 1, 1, 9 });
    try std.testing.expectEqual(@as(u64, 110), store.budget);
    try std.testing.expectEqual(@as(u64, 104), store.held);
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    try std.testing.expect(store.entries.items[0].saved == kept);
    try std.testing.expectEqual(@as(usize, 1), f.live);
    try std.testing.expectEqual(@as(u64, 1), store.counts.evicted);
    fitCache(&h, 110, kept, &.{ 1, 1, 1, 1, 9 });
    try std.testing.expectEqual(@as(u64, 1), store.counts.evicted);
    fitCache(&h, 0, null, &.{});
    try std.testing.expectEqual(@as(u64, 0), store.held);
    try std.testing.expectEqual(@as(usize, 0), store.entries.items.len);
    try std.testing.expectEqual(@as(usize, 0), f.live);
    try std.testing.expectEqual(@as(u64, 2), store.counts.evicted);
}

test "inkling2 host: cache callback with no configured cache leaves no state to mutate" {
    var h: Host = undefined;
    h.cache = null;
    fitCache(&h, 0, null, &.{});
    try std.testing.expect(h.cache == null);
}

test "inkling2 host: bad prompt cache cap is refused before loading" {
    var problem: []const u8 = "";
    for ([_]f64{ -1, std.math.nan(f64), std.math.inf(f64) }) |bad| {
        const o: api.Open = .{ .dir = "no-weights", .model_type = "inkling2", .prompt_cache_gib = bad };
        try std.testing.expectEqual(null, try open(std.testing.allocator, std.testing.allocator, std.testing.io, o, &problem));
        try std.testing.expectStringStartsWith(problem, "--prompt-cache-gib:");
    }
}

test "inkling2 host: parallel defaults to 1 when unset and refuses 0 or >8 before loading" {
    try std.testing.expectEqual(@as(u32, 1), try resolvedLanes(.{ .dir = "no-weights", .model_type = "inkling2", .lanes = 4, .lanes_fixed = false }));
    try std.testing.expectEqual(@as(u32, 1), try resolvedLanes(.{ .dir = "no-weights", .model_type = "inkling2", .lanes = 1, .lanes_fixed = true }));
    try std.testing.expectEqual(@as(u32, 4), try resolvedLanes(.{ .dir = "no-weights", .model_type = "inkling2", .lanes = 4, .lanes_fixed = true }));
    try std.testing.expectEqual(@as(u32, 8), try resolvedLanes(.{ .dir = "no-weights", .model_type = "inkling2", .lanes = 8, .lanes_fixed = true }));
    var problem: []const u8 = "";
    for ([_]u32{ 0, 9, 16 }) |bad| {
        const o: api.Open = .{ .dir = "no-weights", .model_type = "inkling2", .lanes = bad, .lanes_fixed = true };
        try std.testing.expectEqual(null, try open(std.testing.allocator, std.testing.allocator, std.testing.io, o, &problem));
        try std.testing.expectStringStartsWith(problem, "--parallel:");
    }
}

test "inkling2 host: fixed-floor check accepts streams that fit and refuses when fixed rings exceed budget" {
    const c: fl.config.Text = .{
        .hidden = 4608,
        .layers = 48,
        .vocab = 200704,
        .unpadded_vocab = 200704,
        .local = .{ .q = 32, .kv = 8, .dim = 128 },
        .global = .{ .q = 32, .kv = 8, .dim = 128 },
        .window = 2048,
        .d_rel = 128,
        .rel_extent = 131072,
        .log_floor = 128,
        .log_alpha = 1,
        .eps = 1e-6,
        .qk_clip = 0,
        .rel_clip = 0,
        .block = 4,
        .experts = 64,
        .topk = 4,
        .expert_inter = 2048,
        .route_scale = 1,
        .gate_cap = 0,
        .up_cap = 0,
        .logit_cap = 0,
        .max_context = 262144,
    };
    const fixed_one = fl.state.cacheBytes(&c, 1, c.virtualLayers(), 4096);
    try std.testing.expect(fitsFixedFloor(&c, 22 * (1 << 30), 4096, 4));
    try std.testing.expect(fitsFixedFloor(&c, 4 * fixed_one, 4096, 4));
    try std.testing.expect(!fitsFixedFloor(&c, 4 * fixed_one - 1, 4096, 4));
}
