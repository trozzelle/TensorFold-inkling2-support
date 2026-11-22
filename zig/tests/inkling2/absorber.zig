//! The draft head over a prompt, taking its rows in as the drafted lane backend hands them over, chunk by chunk (`prefill --drafts` and
//! `partition --drafts`).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ik = tf.inkling2;
const lanes = tf.lanes;
const settings = @import("settings.zig");

pub const Absorber = struct {
    gpa: std.mem.Allocator,
    drafter: *ik.dspark.Drafter,
    stream: *lanes.Stream, // on the heap: the drafter keeps its stream's state under this address
    row_bytes: u32,

    pub fn init(gpa: std.mem.Allocator, m: *ik.Model, draft: settings.Draft, ids: []const u32) !Absorber {
        const weights: *ik.dspark_weights.Weights = if (m.draft) |*w| w else return error.NoDraftHead;
        const drafter = try ik.dspark.Drafter.init(gpa, m.device, m, weights, draft.drafting.options());
        errdefer drafter.deinit();
        const stream = try gpa.create(lanes.Stream);
        errdefer gpa.destroy(stream);
        stream.* = try lanes.Stream.init(gpa, .{ .id = "prefill", .prompt = ids, .max_new = 1 });
        errdefer stream.deinit(gpa);
        try drafter.drafter().open(stream);
        return .{ .gpa = gpa, .drafter = drafter, .stream = stream, .row_bytes = @intCast(ik.state.feature_taps * @as(usize, m.config.text.hidden) * 2) };
    }

    pub fn deinit(a: Absorber) void {
        a.drafter.drafter().release(a.stream);
        a.stream.deinit(a.gpa);
        a.gpa.destroy(a.stream);
        a.drafter.deinit();
    }

    /// A prompt chunk of `n` rows from `at`, tapped into `taps`, handed over as `Drafted.take` does: every row but the prompt's last
    /// (absorbed with the first token). The absorb's own command buffer is waited for here, so each chunk's GPU seconds are its
    /// own; 0 when the head keeps none of these rows (only the prompt's last 1,023 are ever attended).
    pub fn take(a: Absorber, taps: mtl.Buffer, at: usize, n: usize) !f64 {
        const ids = a.stream.prompt();
        const last = ids.len - 1;
        if (at >= last) return 0;
        const rows: u32 = @intCast(@min(n, last - at));
        const f: lanes.backend.Features = .{ .buffer = @intFromPtr(taps.id), .rows = rows, .row_bytes = a.row_bytes, .space = .device, .dtype = .bf16 };
        try a.drafter.drafter().absorb(&.{.{ .stream = a.stream, .features = f, .start = at, .follow = ids[at + 1 ..][0..rows] }});
        if (a.drafter.pending == null) return 0;
        try a.drafter.sync();
        return a.drafter.absorb_gpu;
    }

    /// The first round after the prompt, as the lane core makes it: the prompt's last row, tapped at row `last` of `taps`, absorbed
    /// with the first token `first`, then a hold of `out.len` drafts after it, whose proposals land in `out`.
    pub fn firstHold(a: Absorber, taps: mtl.Buffer, last: usize, first: u32, out: []u32) !void {
        const ids = a.stream.prompt();
        const row = ids.len - 1;
        const f: lanes.backend.Features = .{ .buffer = @intFromPtr(taps.id), .offset = last * a.row_bytes, .rows = 1, .row_bytes = a.row_bytes, .space = .device, .dtype = .bf16 };
        const d = a.drafter.drafter();
        try d.absorb(&.{.{ .stream = a.stream, .features = f, .start = row, .follow = &.{first} }});
        try d.hold(&.{.{ .stream = a.stream, .pending = first, .position = ids.len + 1, .depth = @intCast(out.len) }});
        try d.held(&.{a.stream}, &.{out});
    }
};
