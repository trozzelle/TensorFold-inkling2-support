//! tf-inkling-run's `greedy --drafts`: greedy decoding through the lane engine with the DSpark draft head in front of the target,
//! as the server runs it (the same wrapper, facts, window costs, depth and confidence settings and layer weights), so a drafted run's
//! tokens can be set beside the plain `greedy` run's. INKLING_DRAFT_CONFIDENCE is a probability above 0 and below 1, 0.7 when unset:
//! every round holds the deepest chain, 7 drafts when INKLING_DRAFT_DEPTH is unset and at most the number it gives, and verifies the
//! first draft and each next one while its own probability is at least that. With `off` every draft held is verified, and the depth
//! is INKLING_DRAFT_DEPTH's: 1 to 7 holds every round at it, `auto` lets the cost rule choose, and unset is 4. The layers are
//! INKLING_DRAFT_BITS's: 8 (quantized at load, unset too) or 16 (bf16). INKLING_PREFILL_ROWS sets the prompt chunk's rows, as served.
const std = @import("std");
const tf = @import("tensorfold");
const ik = tf.inkling2;
const lanes = tf.lanes;
const settings = @import("settings.zig");

fn now() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

/// `steps` greedy tokens after `ids`: a JSON array of them, then one line of counters. The model loads with its draft head
/// (mtp.safetensors in its folder).
pub fn greedy(init: std.process.Init, model_dir: []const u8, ids: []const u32, steps: usize) !void {
    if (steps == 0 or ids.len == 0) return error.ExpectedPromptAndSteps;
    const gpa = init.gpa;
    const draft = try settings.draft();
    const drafting = draft.drafting;
    const m = try ik.Model.load(gpa, init.io, model_dir, .{ .drafts = true, .draft_bits = draft.bits, .prefill_rows = try settings.prefillRows() });
    defer m.deinit();
    const weights: *ik.dspark_weights.Weights = if (m.draft) |*w| w else return error.NoDraftHead;
    const c = &m.config.text;
    const drafter = try ik.dspark.Drafter.init(gpa, m.device, m, weights, drafting.options());
    defer drafter.deinit();
    // one stream's cache, for the prompt and the tokens after it
    const capacity = ids.len + steps;
    const metal = try ik.backend.Metal.init(gpa, m, .{ .capacity = capacity, .budget = ik.state.cacheBytes(c, capacity + 1, c.virtualLayers(), m.prefill_rows) });
    defer metal.deinit();
    try metal.warm();
    var drafted = try ik.dspark.wrap(gpa, metal.backend(), drafter);
    defer drafted.deinit();
    try metal.measure();
    var cfg = try lanes.Config.init(gpa, ik.dspark.laneFacts(&drafted, metal.facts(), drafting.depth), ik.state.window_rows, ik.state.window_rows - 1);
    defer cfg.deinit(gpa);
    var clock: lanes.backend.WallClock = .{ .io = init.io };
    var engine = lanes.Engine.init(gpa, &cfg, drafted.backend(), clock.clock());
    defer engine.deinit();
    var stream = try lanes.Stream.init(gpa, .{ .id = "greedy", .prompt = ids, .max_new = @intCast(steps) });
    defer stream.deinit(gpa);
    var t0 = now();
    try engine.addStream(&stream);
    const prompt_ms = (now() - t0) * 1e3;
    const first = stream.emitted().len;
    t0 = now();
    while (engine.activeCount() > 0) try engine.step();
    const secs = now() - t0;
    const tokens = stream.emitted();
    var w: std.Io.Writer.Allocating = .init(gpa);
    defer w.deinit();
    try std.json.Stringify.value(tokens, .{}, &w.writer);
    const decoded: f64 = @floatFromInt(tokens.len - @min(first, tokens.len));
    std.debug.print("{s}\n{{\"prompt_tokens\": {d}, \"prompt_ms\": {d:.0}, \"steps\": {d}, \"tokens\": {d}, \"tok_s\": {d:.2}, \"rounds\": {d}, \"drafted\": {d}, \"accepted\": {d}, \"drafts\": true, \"depth\": ", .{ w.written(), ids.len, prompt_ms, steps, tokens.len, decoded / secs, stream.rounds, stream.drafted, stream.accepted });
    switch (drafting.depth) {
        .fixed => |n| std.debug.print("{d}", .{n}),
        .auto => std.debug.print("\"auto\"", .{}),
    }
    if (drafting.confidence) |floor| std.debug.print(", \"confidence\": {d}}}\n", .{floor}) else std.debug.print(", \"confidence\": null}}\n", .{});
    if (tokens.len != steps) return error.DraftedRunEndedShort;
}
