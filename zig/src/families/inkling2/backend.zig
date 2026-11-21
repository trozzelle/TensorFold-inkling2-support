//! Inkling 2 behind the lane engine: one stream, its caches, one-row rounds and copy-drafted windows drawn on the host.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const greedy = @import("greedy.zig");
const draw = @import("draw.zig");
const Model = @import("model.zig").Model;
const row = @import("../step5/backend.zig").row;
const be = lanes.backend;

pub const Options = struct { capacity: usize, budget: u64 };

/// Widths `measure` times: a pending token and up to seven drafts, the deepest verify a head asks for.
pub const measured_widths = 8;

/// Token handles: drawn tokens and host-fed ones, in a host ring.
const ring = 4096;

/// Seconds a stream's rounds spent, by where (INKLING_ROUND_TIMING): the forward's encode and GPU time by its own
/// clocks, the forward call, the draws, and the time between one round's end and the next one's start.
const Spent = struct {
    rounds: u64 = 0,
    forward: fwd.Timing = .{},
    run: f64 = 0,
    draw: f64 = 0,
    between: f64 = 0,
    ended: f64 = 0,

    fn log(s: Spent) void {
        if (s.rounds == 0) return;
        const per_ms = 1e3 / @as(f64, @floatFromInt(s.rounds));
        std.log.info("inkling2: {d} rounds; a round took gpu {d:.2} ms, encode {d:.2} ms, commit and wake {d:.2} ms, draw {d:.2} ms, between rounds {d:.2} ms", .{ s.rounds, s.forward.gpu * per_ms, s.forward.encode * per_ms, (s.run - s.forward.gpu - s.forward.encode) * per_ms, s.draw * per_ms, s.between * per_ms });
    }
};

pub const Metal = struct {
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    scratch: st.Scratch,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,
    follow: ?mtl.ResidencySet.Follow = null,
    held: u64 = 0,
    tokens: []u32,
    next: u64 = 0,
    /// The host's draws from a logits row: greedy or keyed, and which of a window's rows a round reads.
    drawer: draw.Drawer,
    /// Set by INKLING_ROUND_TIMING: what the stream's rounds spend, logged when it ends; null when off.
    spent: ?Spent = null,
    /// Set by `prepare_features`: every forward taps its rows for the drafter.
    tapped: bool = false,
    /// A drafting prompt pass's: takes each chunk's tapped rows as they are made.
    sink: ?be.FeatureSink = null,
    /// The forward's ms by window rows, timed by `measure` for the depth rule (`timed` of them; none until it runs).
    costs: [measured_widths]lanes.config.Cost = undefined,
    timed: usize = 0,

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Metal {
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        // a stream's cache holds one position past the context window; the warm-up window runs a window's rows at least
        var scratch = try st.Scratch.init(model.device, &model.config.text, @max(options.capacity + 1, st.window_rows));
        errdefer scratch.deinit();
        const tokens = try gpa.alloc(u32, ring);
        errdefer gpa.free(tokens);
        const values = try gpa.alloc(f32, model.config.text.vocab);
        errdefer gpa.free(values);
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = scratch, .tokens = tokens, .drawer = .{ .gpa = gpa, .values = values }, .spent = if (std.c.getenv("INKLING_ROUND_TIMING") != null) Spent{} else null };
        return out;
    }

    pub fn deinit(self: *Metal) void {
        self.setResident(null);
        var it = self.caches.valueIterator();
        while (it.next()) |cache| {
            cache.*.deinit();
            self.gpa.destroy(cache.*);
        }
        self.caches.deinit(self.gpa);
        self.scratch.deinit();
        self.gpa.free(self.tokens);
        self.gpa.free(self.drawer.values);
        self.gpa.destroy(self);
    }

    pub fn setResident(self: *Metal, hook: ?mtl.ResidencySet.Follow) void {
        if (self.follow) |f| mtl.resident.fields.apply(&self.scratch, f, false);
        self.follow = hook;
        if (hook) |f| mtl.resident.fields.apply(&self.scratch, f, true);
        var it = self.caches.valueIterator();
        while (it.next()) |cache| cache.*.setResident(hook);
    }

    /// Rows are independent in every kernel: any window gives one-row bits. One stream; a draft head's facts come from
    /// `Drafted.facts`. The window forwards' costs are those `measure` timed, if it ran.
    pub fn facts(self: *const Metal) lanes.Model {
        return .{ .exact_width = st.window_rows, .gpu_tokens = false, .batch_rows = st.window_rows, .mtp = false, .speculate = false, .speculate_early = false, .drafts = 1, .max_streams = 1, .streams_exact = true, .window_costs = self.costs[0..self.timed] };
    }

    /// A 16-token prompt on a throwaway cache: the first request skips the GPU's first touch of every weight.
    pub fn warm(self: *Metal) !void {
        const c = &self.model.config.text;
        var cache = try st.Cache.init(self.model.device, c, 32, c.layers);
        defer cache.deinit();
        var ids: [st.window_rows]u32 = undefined;
        for (&ids, 0..) |*x, i| x.* = @intCast(1000 + i * 37);
        try fwd.run(self.model, &self.scratch, &cache, &ids, .{ .logits = .all });
    }

    /// A window of each width 1 to `measured_widths` on a throwaway cache at position 0, the second of two runs timed, with
    /// the rows tapped as a drafted round's are once the features are prepared: what the depth rule prices a round's window by.
    pub fn measure(self: *Metal) !void {
        const c = &self.model.config.text;
        var cache = try st.Cache.init(self.model.device, c, 32, c.layers);
        defer cache.deinit();
        var ids: [measured_widths]u32 = undefined;
        for (&ids, 0..) |*x, i| x.* = @intCast(1000 + i * 37);
        const to: ?fwd.B = if (self.tapped) .{ .buf = self.scratch.ftaps } else null;
        for (1..measured_widths + 1) |rows| {
            var ms: f64 = 0;
            for (0..2) |run| {
                cache.len = 0;
                const began = nowSeconds();
                try fwd.run(self.model, &self.scratch, &cache, ids[0..rows], .{ .logits = .all, .feature_taps = to });
                if (run == 1) ms = (nowSeconds() - began) * 1e3;
            }
            self.costs[rows - 1] = .{ .width = @intCast(rows), .ms = ms };
        }
        self.timed = measured_widths;
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .prepare_features = prepareFn, .features = featuresFn, .prefill_features = prefillFeaturesFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .first_row = firstRowFn, .held = heldFn } };
    }

    fn cast(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Metal, stream: *lanes.Stream) !*st.Cache {
        return self.caches.get(stream) orelse error.UnknownInkling2Stream;
    }

    /// The prompt's last row's logits, kept by `feed` for the first draw; no later round writes or reads them.
    fn logits(self: *Metal, cache: *st.Cache) []const u16 {
        return cache.logits.slice(u16, self.model.config.text.vocab);
    }

    fn put(self: *Metal, token: u32) u64 {
        const h = self.next;
        self.next += 1;
        self.tokens[h % ring] = token;
        return h;
    }

    /// A tapped row's bytes: every tap's hidden row, bf16.
    fn tapBytes(self: *const Metal) usize {
        return @as(usize, st.feature_taps) * self.model.config.text.hidden * 2;
    }

    /// A prompt chunk's `n` tapped rows (the first at the cache's end), given to the sink while they are held; the
    /// prompt's last row stays in the cache for the first draft.
    fn holdTaps(self: *Metal, cache: *st.Cache, stream: *lanes.Stream, n: usize, last: bool) !void {
        const width = self.tapBytes();
        if (self.sink) |k| try k.take(k.ptr, stream, .{ .buffer = @intFromPtr(self.scratch.ptaps.id), .rows = @intCast(n), .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 }, cache.len);
        if (!last) return;
        if (cache.last_tap == null) {
            const buf = try self.model.device.buffer(width, mtl.ResourceOptions.shared);
            if (cache.follow) |f| f.join(f.ctx, buf);
            cache.last_tap = buf;
        }
        @memcpy(cache.last_tap.?.slice(u8, width), self.scratch.ptaps.slice(u8, n * width)[(n - 1) * width ..][0..width]);
        cache.tap = .{ .buf = cache.last_tap, .start = cache.len + n - 1, .rows = 1 };
    }

    /// ids at the cache's end in prompt chunks, the last chunk's last logits kept for the first draw. With the features
    /// prepared each chunk is tapped, and `holdTaps` hands its rows on.
    fn feed(self: *Metal, cache: *st.Cache, ids: []const u32, stream: *lanes.Stream) !void {
        var at: usize = 0;
        while (at < ids.len) {
            if (stream.isCancelled()) return error.Cancelled;
            const n = @min(ids.len - at, st.prompt_rows);
            const last = at + n == ids.len;
            const to: ?fwd.B = if (self.tapped) .{ .buf = self.scratch.ptaps } else null;
            try fwd.run(self.model, &self.scratch, cache, ids[at..][0..n], .{ .logits = if (last) .last else .none, .prompt = true, .feature_taps = to });
            if (to != null) try self.holdTaps(cache, stream, n, last);
            if (last) @memcpy(cache.logits.slice(u16, self.model.config.text.vocab), self.scratch.logits.slice(u16, self.model.config.text.vocab));
            cache.len += n;
            lanes.prefill_widths.completed(self.gpa, stream, n);
            at += n;
        }
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidInkling2Prompt;
        const c = &self.model.config.text;
        const capacity = ids.len + stream.max_new + 1;
        const need = st.cacheBytes(c, capacity, c.layers);
        if (self.held + need > self.options.budget) return error.Inkling2CachesFull;
        self.held += need;
        errdefer self.held -= need;
        const cache = try self.gpa.create(st.Cache);
        errdefer self.gpa.destroy(cache);
        cache.* = try st.Cache.init(self.model.device, c, capacity, c.layers);
        errdefer cache.deinit();
        cache.setResident(self.follow);
        try self.caches.put(self.gpa, stream, cache);
        errdefer _ = self.caches.remove(stream);
        stream.cached = 0;
        if (self.spent != null) self.spent = .{};
        try self.feed(cache, ids, stream);
    }

    /// The drafter's layers, told once before any forward. A forward taps the inputs of `fwd.tap_layers` and the finals
    /// only, so a drafter that reads others is refused; from here on every forward taps its rows.
    fn prepareFn(ptr: *anyopaque, taps: []const u32) !void {
        const self = cast(ptr);
        if (!tapsAre(taps, self.model.config.text.layers)) return error.UnsupportedInkling2Taps;
        self.tapped = true;
    }

    /// Rows [start, start + count) of the stream's latest tapped rows, on the device.
    fn featuresFn(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, start: u64, count: u32) !be.Features {
        const self = cast(ptr);
        if (!self.tapped or !tapsAre(taps, self.model.config.text.layers)) return error.UnpreparedInkling2Taps;
        return tappedRows((try self.cacheOf(stream)).tap, self.tapBytes(), start, count);
    }

    /// The prompt pass handing `sink` each chunk's tapped rows as it makes them.
    fn prefillFeaturesFn(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, sink: be.FeatureSink) !void {
        const self = cast(ptr);
        if (!self.tapped or !tapsAre(taps, self.model.config.text.layers)) return error.UnpreparedInkling2Taps;
        self.sink = sink;
        defer self.sink = null;
        return prefillFn(ptr, stream);
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        return cast(ptr).firstSlot(stream, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        return cast(ptr).firstSlot(stream, position, mask);
    }

    fn firstSlot(self: *Metal, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidInkling2Position;
        return self.put(try self.drawer.one(self.logits(cache), stream.sampling, position, mask));
    }

    /// The fed token's forward, the next token drawn from its logits (synchronous: no step runs ahead).
    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, f: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len + 1) return error.InvalidInkling2Position;
        const tok = switch (f) {
            .handle => |h| self.tokens[h % ring],
            .value => |v| v,
        };
        if (tok >= self.model.config.text.vocab) return error.InvalidInkling2Token;
        try fwd.run(self.model, &self.scratch, cache, &.{tok}, .{ .logits = .last });
        cache.len += 1;
        return self.put(try self.drawer.one(self.scratch.logits.slice(u16, self.model.config.text.vocab), stream.sampling, position, &.{}));
    }

    fn readFn(ptr: *anyopaque, handle: u64) !u32 {
        const self = cast(ptr);
        const token = self.tokens[handle % ring];
        if (token >= self.model.config.text.vocab) return error.InvalidInkling2Token;
        return token;
    }

    fn firstRowFn(ptr: *anyopaque, stream: *lanes.Stream) !lanes.LogprobRow {
        const self = cast(ptr);
        const l = self.logits(try self.cacheOf(stream));
        return row(l, greedy.argmax(l), stream.logprobs orelse 0);
    }

    /// One stream's pending token and copy drafts in one forward, the rows drawn that the round reads (draw.zig);
    /// keep() rolls back by length.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        const c = &self.model.config.text;
        if (windows.len != 1 or out.len != 1) return error.UnsupportedInkling2Windows;
        const w = windows[0];
        const o = out[0];
        const rows = w.rows();
        if (rows > st.window_rows or w.parents != null or w.held != 0 or o.sampled.len != rows or o.drafts.len != rows - 1) return error.UnsupportedInkling2Window;
        const cache = try self.cacheOf(w.stream);
        for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidInkling2Position;
        var ids: [st.window_rows]u32 = undefined;
        ids[0] = w.pending;
        @memcpy(ids[1..][0..w.tokens.len], w.tokens);
        const began = nowSeconds();
        if (self.spent) |*s| if (s.ended != 0) {
            s.between += began - s.ended;
        };
        const to: ?fwd.B = if (self.tapped) .{ .buf = self.scratch.ftaps } else null;
        try fwd.run(self.model, &self.scratch, cache, ids[0..rows], .{ .logits = .all, .timing = if (self.spent) |*s| &s.forward else null, .feature_taps = to });
        const ran = nowSeconds();
        // the window's rows are the drafter's next rows, from the pending token's position
        if (to != null) cache.tap = .{ .buf = self.scratch.ftaps, .start = cache.len, .rows = rows };
        cache.len += rows;
        @memcpy(o.drafts, w.tokens);
        const all = self.scratch.logits.slice(u16, rows * c.vocab);
        // a window's logprob rows are made for every row: a forced window commits them along all its rows
        const wanted = o.rows.len == rows;
        try self.drawer.window(all, .{ .drafts = w.tokens, .positions = w.positions, .masks = w.masks, .sampling = w.stream.sampling, .vocab = c.vocab }, o.sampled, !wanted);
        if (wanted) for (0..rows) |r| {
            o.rows[r] = row(all[r * c.vocab ..][0..c.vocab], o.sampled[r], w.stream.logprobs orelse 0);
        };
        if (self.spent) |*s| {
            s.ended = nowSeconds();
            s.rounds += 1;
            s.run += ran - began;
            s.draw += s.ended - ran;
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        for (windows, paths) |w, path| {
            const cache = try self.cacheOf(w.stream);
            for (path, 0..) |r, i| if (r != i) return error.UnsupportedInkling2Tree;
            if (path.len == 0 or path.len > w.rows() or w.rows() > cache.len) return error.UnsupportedInkling2Window;
            cache.len = cache.len - w.rows() + path.len;
        }
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.Inkling2HasNoDraftHead;
    }

    fn heldFn(_: *anyopaque, _: *lanes.Stream, out: []u32) !void {
        if (out.len > 0) return error.Inkling2DraftsNotHeld;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        if (self.caches.fetchRemove(stream)) |entry| {
            const c = &self.model.config.text;
            self.held -= st.cacheBytes(c, entry.value.capacity, c.layers);
            entry.value.deinit();
            self.gpa.destroy(entry.value);
        }
        if (self.spent) |s| {
            s.log();
            self.spent = .{};
        }
    }
};

/// Whether `taps` are the layers a forward taps, in order: the inputs of `fwd.tap_layers`, then the finals, named by the
/// layer count.
fn tapsAre(taps: []const u32, layers: u32) bool {
    const n = fwd.tap_layers.len;
    return taps.len == n + 1 and std.mem.eql(u32, taps[0..n], &fwd.tap_layers) and taps[n] == layers;
}

/// Rows [start, start + count) of the tapped rows `tp` holds, `width` bytes a row, as the drafter reads them.
fn tappedRows(tp: st.Tap, width: usize, start: u64, count: u32) !be.Features {
    const buf = tp.buf orelse return error.UnpreparedInkling2Taps;
    if (count == 0 or start < tp.start or start + count > tp.start + tp.rows) return error.Inkling2TapsNotHeld;
    return .{ .buffer = @intFromPtr(buf.id), .offset = tp.off + (start - tp.start) * width, .rows = count, .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 };
}

fn nowSeconds() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

test "inkling2 backend: the lane backend's functions compile against its interface" {
    std.testing.refAllDecls(Metal);
}

test "inkling2 backend: the taps are the inputs of seven layers and the finals, named by the layer count, in order" {
    try std.testing.expect(tapsAre(&.{ 5, 11, 17, 23, 29, 35, 41, 48 }, 48));
    try std.testing.expect(!tapsAre(&.{ 5, 11, 17, 23, 29, 35, 41 }, 48));
    try std.testing.expect(!tapsAre(&.{ 5, 11, 17, 23, 29, 35, 41, 47 }, 48));
    try std.testing.expect(!tapsAre(&.{ 11, 5, 17, 23, 29, 35, 41, 48 }, 48));
    try std.testing.expect(!tapsAre(&.{ 5, 11, 17, 23, 29, 35, 41, 48, 48 }, 48));
}

test "inkling2 backend: features are the window's rows from the buffer's offset, and nothing outside it" {
    const fake = mtl.Buffer{ .id = @ptrFromInt(0x4000) };
    const width: usize = st.feature_taps * 4608 * 2;
    const tp: st.Tap = .{ .buf = fake, .off = 100, .start = 7, .rows = 3 };
    const whole = try tappedRows(tp, width, 7, 3);
    try std.testing.expectEqual(@as(u64, 0x4000), whole.buffer);
    try std.testing.expectEqual(@as(u64, 100), whole.offset);
    try std.testing.expectEqual(@as(u32, 3), whole.rows);
    try std.testing.expectEqual(@as(u32, width), whole.row_bytes);
    try std.testing.expect(whole.space == .device and whole.dtype == .bf16);
    const tail = try tappedRows(tp, width, 8, 2);
    try std.testing.expectEqual(@as(u64, 100 + width), tail.offset);
    try std.testing.expectEqual(@as(u32, 2), tail.rows);
    try std.testing.expectError(error.Inkling2TapsNotHeld, tappedRows(tp, width, 6, 2));
    try std.testing.expectError(error.Inkling2TapsNotHeld, tappedRows(tp, width, 8, 3));
    try std.testing.expectError(error.Inkling2TapsNotHeld, tappedRows(tp, width, 10, 1));
    try std.testing.expectError(error.Inkling2TapsNotHeld, tappedRows(tp, width, 7, 0));
    try std.testing.expectError(error.UnpreparedInkling2Taps, tappedRows(.{}, width, 0, 1));
}
