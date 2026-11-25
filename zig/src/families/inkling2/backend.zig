//! Inkling 2 lane backend: 1..32 streams, caches, shared verify rounds, and host-drawn copy-drafted windows.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const st = @import("state.zig");
const snapshot = @import("snapshot.zig");
const fwd = @import("forward.zig");
const greedy = @import("greedy.zig");
const draw = @import("draw.zig");
const Model = @import("model.zig").Model;
const row = @import("../step5/backend.zig").row;
const be = lanes.backend;

pub const Options = struct { capacity: usize, budget: u64, streams: u32 = 1 };

/// Widths `measure` times: a pending token and up to seven drafts, the deepest verify a head asks for.
pub const measured_widths = 8;

/// Multi-stream total widths `measure` times when `options.streams > 1` for shared-round cost pricing.
const shared_widths = [_]usize{ 16, 24, 32, 48, 64, 96, 128 };

/// Token handles: drawn tokens and host-fed ones, in a host ring.
const ring = 4096;

/// Per-stream round timings (INKLING_ROUND_TIMING): encode/GPU clocks, forward, draws, and inter-round gaps.
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

/// Prompt cache room callback: tells the host how many bytes of the cache budget streams leave for kept states.
pub const Room = struct {
    ctx: *anyopaque,
    fit: *const fn (ctx: *anyopaque, bytes: u64, restoring: ?*anyopaque, prompt: []const u32) void,
};

pub const Metal = struct {
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    scratch: st.Scratch,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,
    follow: ?mtl.ResidencySet.Follow = null,
    room: ?Room = null,
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
    /// Multi-stream shared forwards' ms by total rows, timed by `measure` when `options.streams > 1`.
    shared: [shared_widths.len]lanes.config.Cost = undefined,
    shared_timed: usize = 0,
    /// Monotonic generation incremented each time `scratch.ftaps` or `scratch.ptaps` is overwritten.
    tap_gen: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Metal {
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        // a stream's cache holds one position past the context window; the warm-up window runs a window's rows at least
        var scratch = try st.Scratch.init(model.device, &model.config.text, @max(options.capacity + 1, st.window_rows), model.chunking());
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

    /// Rows are stream-independent; draft facts come from `Drafted.facts` and costs from `measure`.
    pub fn facts(self: *const Metal) lanes.Model {
        return .{
            .exact_width = st.window_rows,
            .gpu_tokens = false,
            .hidden_rows = self.options.streams > 1,
            .batch_rows = st.batch_rows,
            .mtp = false,
            .speculate = false,
            .speculate_early = false,
            .drafts = 1,
            .max_streams = self.options.streams,
            .streams_exact = true,
            .window_costs = self.costs[0..self.timed],
            .shared_costs = self.shared[0..self.shared_timed],
        };
    }

    /// A 16-token prompt on a throwaway cache: the first request skips the GPU's first touch of every weight.
    pub fn warm(self: *Metal) !void {
        const c = &self.model.config.text;
        var cache = try st.Cache.init(self.model.device, c, 32, c.virtualLayers(), self.scratch.prefill_rows);
        defer cache.deinit();
        var ids: [st.window_rows]u32 = undefined;
        for (&ids, 0..) |*x, i| x.* = @intCast(1000 + i * 37);
        try fwd.run(self.model, &self.scratch, &cache, &ids, .{ .logits = .all });
    }

    /// Times widths 1..`measured_widths` (and `shared_widths` when `options.streams > 1`) on throwaway caches.
    pub fn measure(self: *Metal) !void {
        const c = &self.model.config.text;
        var cache = try st.Cache.init(self.model.device, c, 32, c.virtualLayers(), self.scratch.prefill_rows);
        defer cache.deinit();
        var ids: [st.batch_rows]u32 = undefined;
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
        if (self.options.streams > 1) {
            const max_possible = st.batch_rows / st.window_rows;
            const max_segs = @min(max_possible, @as(usize, self.options.streams));
            const max_rows = @min(@as(usize, st.batch_rows), @as(usize, self.options.streams) * st.window_rows);
            var caches: [max_possible]st.Cache = undefined;
            var made: usize = 0;
            defer for (caches[0..made]) |*cc| cc.deinit();
            for (0..max_segs) |i| {
                caches[i] = try st.Cache.init(self.model.device, c, 32, c.virtualLayers(), self.scratch.prefill_rows);
                made += 1;
            }
            var timed_n: usize = 0;
            for (shared_widths) |rows| {
                if (rows > max_rows) break;
                const n_segs = @max(@as(usize, 2), (rows + st.window_rows - 1) / st.window_rows);
                var segs: [max_possible]fwd.Segment = undefined;
                var rem = rows;
                for (0..n_segs) |si| {
                    const left = n_segs - si;
                    const take = (rem + left - 1) / left;
                    segs[si] = .{ .cache = &caches[si], .rows = take };
                    rem -= take;
                }
                var ms: f64 = 0;
                for (0..2) |run| {
                    for (0..n_segs) |si| caches[si].len = 0;
                    const began = nowSeconds();
                    try fwd.runShared(self.model, &self.scratch, segs[0..n_segs], ids[0..rows], .{ .logits = .all, .feature_taps = to });
                    if (run == 1) ms = (nowSeconds() - began) * 1e3;
                }
                self.shared[timed_n] = .{ .width = @intCast(rows), .ms = ms };
                timed_n += 1;
            }
            self.shared_timed = timed_n;
        }
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{
            .ptr = self,
            .chunk_invariant = true,
            .vtable = &.{
                .prefill = prefillFn,
                .prefill_step = prefillStepFn,
                .cancel_prefill = cancelPrefillFn,
                .reports_fill = true,
                .prepare_features = prepareFn,
                .features = featuresFn,
                .prefill_features = prefillFeaturesFn,
                .first = firstFn,
                .first_masked = firstMaskedFn,
                .queue = queueFn,
                .read = readFn,
                .verify = verifyFn,
                .keep = keepFn,
                .draft = draftFn,
                .release = releaseFn,
                .first_row = firstRowFn,
                .held = heldFn,
            },
        };
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
        return fwd.tapCount(self.model) * self.model.config.text.hidden * 2;
    }

    /// Allocate a stream cache under the budget, evicting prompt snapshots if needed, and restore any prefix snapshot.
    fn openStream(self: *Metal, stream: *lanes.Stream) !*st.Cache {
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidInkling2Prompt;
        const c = &self.model.config.text;
        const capacity = ids.len + stream.max_new + 1;
        const need = st.cacheBytes(c, capacity, c.virtualLayers(), self.scratch.prefill_rows);
        try admitBudget(self.held, need, self.options.budget, self.caches.count());
        self.held += need;
        errdefer self.held -= need;
        if (stream.reuse.saved) |saved| { // a state the stream cannot hold beside its caches: freed now, no restore
            const snap: *snapshot.Snap = @ptrCast(@alignCast(saved));
            if (self.held + snap.bytes > self.options.budget) snap.free();
        }
        // kept states give way first
        if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, stream.reuse.saved, ids);
        const cache = try self.gpa.create(st.Cache);
        errdefer self.gpa.destroy(cache);
        cache.* = try st.Cache.init(self.model.device, c, capacity, c.virtualLayers(), self.scratch.prefill_rows);
        errdefer cache.deinit();
        cache.setResident(self.follow);
        try self.caches.put(self.gpa, stream, cache);
        errdefer _ = self.caches.remove(stream);
        stream.cached = 0;
        if (stream.reuse.saved) |saved| { // a kept state of this prompt's prefix: the pass starts there
            const snap: *const snapshot.Snap = @ptrCast(@alignCast(saved));
            if (snap.at < ids.len) {
                if (snapshot.restore(self.model.queue, c, snap, cache)) stream.cached = snap.at else |_| stream.reuse_failed = true;
            } else stream.reuse_failed = true;
        }
        stream.fill_at.store(@intCast(cache.len), .release);
        if (self.spent != null and self.caches.count() == 1) self.spent = .{};
        return cache;
    }

    /// Runs one prompt chunk (<= `max_rows` or `prefill_rows`), cut at reuse marks; returns true when filled.
    fn prefillStepFn(ptr: *anyopaque, stream: *lanes.Stream, max_rows: u32) !bool {
        const self = cast(ptr);
        const cache = self.caches.get(stream) orelse try self.openStream(stream);
        const ids = stream.prompt();
        if (cache.len >= ids.len) return true;
        if (stream.isCancelled()) return error.Cancelled;
        const cap = if (max_rows > 0) @min(@as(usize, max_rows), self.scratch.prefill_rows) else self.scratch.prefill_rows;
        const at = cache.len;
        var end = @min(ids.len, at + cap);
        // Stop at reuse marks so their state can be saved; chunk boundaries do not affect row bits.
        for (stream.reuse.marks) |x| if (x > at and x < end) {
            end = x;
        };
        const n = end - at;
        const last = end == ids.len;
        const to: ?fwd.B = if (self.tapped) .{ .buf = self.scratch.ptaps } else null;
        try fwd.run(self.model, &self.scratch, cache, ids[at..end], .{ .logits = if (last) .last else .none, .prompt = true, .feature_taps = to });
        if (to != null) {
            self.tap_gen += 1;
            const width = self.tapBytes();
            cache.tap = .{ .buf = self.scratch.ptaps, .off = 0, .start = at, .rows = n, .gen = self.tap_gen };
            if (self.sink) |k| try k.take(k.ptr, stream, .{ .buffer = @intFromPtr(self.scratch.ptaps.id), .rows = @intCast(n), .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 }, at);
            if (last) {
                if (cache.last_tap == null) {
                    const buf = try self.model.device.buffer(width, mtl.ResourceOptions.shared);
                    if (cache.follow) |f| f.join(f.ctx, buf);
                    cache.last_tap = buf;
                }
                @memcpy(cache.last_tap.?.slice(u8, width), self.scratch.ptaps.slice(u8, n * width)[(n - 1) * width ..][0..width]);
                cache.last_tap_pos = ids.len - 1;
            }
        }
        if (last) @memcpy(cache.logits.slice(u16, self.model.config.text.vocab), self.scratch.logits.slice(u16, self.model.config.text.vocab));
        cache.len += n;
        stream.fill_at.store(@intCast(cache.len), .release);
        lanes.prefill_widths.completed(self.gpa, stream, n);
        if (std.mem.indexOfScalar(u32, stream.reuse.marks, @intCast(cache.len)) != null) if (stream.reuse.hook) |k| k.at(k.ptr, stream, @intCast(cache.len));
        return last;
    }

    fn cancelPrefillFn(_: *anyopaque, _: *lanes.Stream) !void {}

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        if (self.caches.contains(stream)) return error.InvalidInkling2Prompt;
        while (!try prefillStepFn(ptr, stream, 0)) {}
    }

    /// Validates the drafter's tap layers (`fwd.tapLayers(model)` + finals) and enables row tapping on forwards.
    fn prepareFn(ptr: *anyopaque, taps: []const u32) !void {
        const self = cast(ptr);
        if (!tapsAre(taps, fwd.tapLayers(self.model), self.model.config.text.virtualLayers())) return error.UnsupportedInkling2Taps;
        self.tapped = true;
    }

    /// Rows [start, start + count) of the stream's latest tapped rows, on the device.
    fn featuresFn(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, start: u64, count: u32) !be.Features {
        const self = cast(ptr);
        if (!self.tapped or !tapsAre(taps, fwd.tapLayers(self.model), self.model.config.text.virtualLayers())) return error.UnpreparedInkling2Taps;
        return featuresOf(try self.cacheOf(stream), self.tapBytes(), start, count, self.tap_gen);
    }

    /// The prompt pass handing `sink` each chunk's tapped rows as it makes them.
    fn prefillFeaturesFn(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, sink: be.FeatureSink) !void {
        const self = cast(ptr);
        if (!self.tapped or !tapsAre(taps, fwd.tapLayers(self.model), self.model.config.text.virtualLayers())) return error.UnpreparedInkling2Taps;
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

    /// Shared forward over 1..N streams' pending tokens and drafts, drawn per stream; keep() rolls back.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        const c = &self.model.config.text;
        if (windows.len == 0 or windows.len != out.len or windows.len > self.options.streams) return error.UnsupportedInkling2Windows;
        var segments: [st.max_streams]fwd.Segment = undefined;
        var ids: [st.batch_rows]u32 = undefined;
        var total: usize = 0;
        for (windows, out, 0..) |w, o, i| {
            const rows = w.rows();
            if (rows > st.window_rows or total + rows > st.batch_rows or w.parents != null or w.held != 0 or o.sampled.len != rows or o.drafts.len != rows - 1) return error.UnsupportedInkling2Window;
            const cache = try self.cacheOf(w.stream);
            for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidInkling2Position;
            segments[i] = .{ .cache = cache, .rows = rows };
            ids[total] = w.pending;
            @memcpy(ids[total + 1 ..][0..w.tokens.len], w.tokens);
            total += rows;
        }
        const began = nowSeconds();
        if (self.spent) |*s| if (s.ended != 0) {
            s.between += began - s.ended;
        };
        const to: ?fwd.B = if (self.tapped) .{ .buf = self.scratch.ftaps } else null;
        try fwd.runShared(self.model, &self.scratch, segments[0..windows.len], ids[0..total], .{ .logits = .all, .timing = if (self.spent) |*s| &s.forward else null, .feature_taps = to });
        const ran = nowSeconds();
        if (to != null) self.tap_gen += 1;
        const all = self.scratch.logits.slice(u16, total * c.vocab);
        const width = self.tapBytes();
        var base: usize = 0;
        for (windows, out, segments[0..windows.len]) |w, o, seg| {
            const rows = seg.rows;
            // the window's rows are the drafter's next rows, from the pending token's position
            if (to != null) seg.cache.tap = .{ .buf = self.scratch.ftaps, .off = base * width, .start = seg.cache.len, .rows = rows, .gen = self.tap_gen };
            seg.cache.len += rows;
            @memcpy(o.drafts, w.tokens);
            const row_logits = all[base * c.vocab ..][0 .. rows * c.vocab];
            // a window's logprob rows are made for every row: a forced window commits them along all its rows
            const wanted = o.rows.len == rows;
            try self.drawer.window(row_logits, .{ .drafts = w.tokens, .positions = w.positions, .masks = w.masks, .sampling = w.stream.sampling, .vocab = c.vocab }, o.sampled, !wanted);
            if (wanted) for (0..rows) |r| {
                o.rows[r] = row(row_logits[r * c.vocab ..][0..c.vocab], o.sampled[r], w.stream.logprobs orelse 0);
            };
            base += rows;
        }
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
            self.held -= st.cacheBytes(c, entry.value.capacity, c.virtualLayers(), self.scratch.prefill_rows);
            entry.value.deinit();
            self.gpa.destroy(entry.value);
            if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, null, &.{});
        }
        if (self.spent) |s| {
            s.log();
            self.spent = .{};
        }
    }

    /// The prompt cache's Snapshots functions: a stream's caches at a mark, copied on the GPU.
    pub fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        const c = &cast(ptr).model.config.text;
        return snapshot.bytes(c, c.virtualLayers(), at);
    }

    pub fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = cast(ptr);
        const cache = try self.cacheOf(@ptrCast(@alignCast(owner orelse return error.NoStream)));
        if (cache.len != at) return error.SnapshotOutOfStep;
        const snap = try self.gpa.create(snapshot.Snap);
        errdefer self.gpa.destroy(snap);
        snap.* = try snapshot.save(self.model.device, self.model.queue, &self.model.config.text, cache);
        return snap;
    }

    pub fn snapRestore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores; // the prompt pass restores, before its first chunk
    }

    pub fn snapDrop(ptr: *anyopaque, saved: *anyopaque) void {
        const snap: *snapshot.Snap = @ptrCast(@alignCast(saved));
        snap.free();
        cast(ptr).gpa.destroy(snap);
    }
};

/// Admit a cache under `budget`: returns `LaneMemory` if other streams are active, else `Inkling2CachesFull`.
fn admitBudget(held: u64, need: u64, budget: u64, active: usize) !void {
    if (held + need <= budget) return;
    return if (active > 0) error.LaneMemory else error.Inkling2CachesFull;
}

/// True when `taps` matches `expected` intermediate layers followed by `layers` (the finals).
fn tapsAre(taps: []const u32, expected: []const u32, layers: u32) bool {
    const n = expected.len;
    return taps.len == n + 1 and std.mem.eql(u32, taps[0..n], expected) and taps[n] == layers;
}

/// Rows [start, start + count) from `cache`, preferring `last_tap` for the final prompt row.
fn featuresOf(cache: *const st.Cache, width: usize, start: u64, count: u32, cur_gen: u64) !be.Features {
    if (count == 1) if (cache.last_tap) |buf| if (cache.last_tap_pos) |pos| if (start == pos) {
        return .{ .buffer = @intFromPtr(buf.id), .offset = 0, .rows = 1, .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 };
    };
    return tappedRows(cache.tap, width, start, count, cur_gen);
}

/// Rows [start, start + count) of the tapped rows `tp` holds, `width` bytes a row, as the drafter reads them.
fn tappedRows(tp: st.Tap, width: usize, start: u64, count: u32, cur_gen: u64) !be.Features {
    const buf = tp.buf orelse return error.UnpreparedInkling2Taps;
    if (tp.gen != cur_gen) return error.StaleInkling2Taps;
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

test "inkling2 backend: the taps are the configured layer inputs and the finals, named by the virtual layer count, in order" {
    try std.testing.expect(tapsAre(&.{ 5, 11, 17, 23, 29, 35, 41, 48 }, &fwd.tap_layers, 48));
    try std.testing.expect(!tapsAre(&.{ 5, 11, 17, 23, 29, 35, 41 }, &fwd.tap_layers, 48));
    try std.testing.expect(!tapsAre(&.{ 5, 11, 17, 23, 29, 35, 41, 47 }, &fwd.tap_layers, 48));
    try std.testing.expect(!tapsAre(&.{ 11, 5, 17, 23, 29, 35, 41, 48 }, &fwd.tap_layers, 48));
    try std.testing.expect(!tapsAre(&.{ 5, 11, 17, 23, 29, 35, 41, 48, 48 }, &fwd.tap_layers, 48));
    // Mini 4-tap schedule
    const mini_taps = [_]u32{ 5, 23, 41 };
    try std.testing.expect(tapsAre(&.{ 5, 23, 41, 48 }, &mini_taps, 48));
    try std.testing.expect(!tapsAre(&.{ 5, 23, 41, 24 }, &mini_taps, 48));
    try std.testing.expect(!tapsAre(&.{ 5, 23, 41, 48 }, &fwd.tap_layers, 48));
}

test "inkling2 backend: features are the window's rows from the buffer's offset, guard stale generations, and preserve last_tap" {
    const width: usize = st.feature_taps * 4608 * 2;
    const tp: st.Tap = .{ .buf = .{ .id = @ptrFromInt(0x4000) }, .off = 100, .start = 7, .rows = 3, .gen = 5 };
    const whole = try tappedRows(tp, width, 7, 3, 5);
    try std.testing.expect(whole.buffer == 0x4000 and whole.offset == 100 and whole.rows == 3 and whole.row_bytes == width and whole.space == .device and whole.dtype == .bf16);
    const tail = try tappedRows(tp, width, 8, 2, 5);
    try std.testing.expect(tail.offset == 100 + width and tail.rows == 2);
    try std.testing.expectError(error.StaleInkling2Taps, tappedRows(tp, width, 7, 3, 6));
    try std.testing.expectError(error.Inkling2TapsNotHeld, tappedRows(tp, width, 6, 2, 5));
    try std.testing.expectError(error.Inkling2TapsNotHeld, tappedRows(tp, width, 8, 3, 5));
    try std.testing.expectError(error.Inkling2TapsNotHeld, tappedRows(tp, width, 10, 1, 5));
    try std.testing.expectError(error.Inkling2TapsNotHeld, tappedRows(tp, width, 7, 0, 5));
    try std.testing.expectError(error.UnpreparedInkling2Taps, tappedRows(.{}, width, 0, 1, 0));
    // last_tap serves the prompt's final row even after another stream advances tap_gen
    var c: st.Cache = undefined;
    c.tap = tp;
    c.last_tap = .{ .id = @ptrFromInt(0x8000) };
    c.last_tap_pos = 9;
    const saved = try featuresOf(&c, width, 9, 1, 99);
    try std.testing.expect(saved.buffer == 0x8000 and saved.offset == 0);
    try std.testing.expectError(error.StaleInkling2Taps, featuresOf(&c, width, 7, 2, 99));
}

test "inkling2 backend: multi-stream facts enable shared rounds in lanes.Config" {
    var m: Metal = undefined;
    m.options = .{ .capacity = 4096, .budget = 1 << 30, .streams = 4 };
    m.timed = 0;
    m.shared_timed = 0;
    const f4 = m.facts();
    try std.testing.expect(f4.hidden_rows and f4.streams_exact and f4.max_streams == 4 and f4.batch_rows == st.batch_rows);
    var cfg4 = try lanes.Config.init(std.testing.allocator, f4, st.window_rows, st.window_rows - 1);
    defer cfg4.deinit(std.testing.allocator);
    try std.testing.expect(cfg4.family_streams and cfg4.batch_streams == 4 and cfg4.batch_rows == st.batch_rows);
    m.options.streams = 1;
    var cfg1 = try lanes.Config.init(std.testing.allocator, m.facts(), st.window_rows, st.window_rows - 1);
    defer cfg1.deinit(std.testing.allocator);
    try std.testing.expect(!cfg1.family_streams and cfg1.batch_streams == 1);
}

test "inkling2 backend: cache budget admission requeues behind active streams and fails alone" {
    try admitBudget(0, 100, 100, 0);
    try admitBudget(50, 50, 100, 1);
    try std.testing.expectError(error.LaneMemory, admitBudget(60, 50, 100, 1));
    try std.testing.expectError(error.LaneMemory, admitBudget(0, 101, 100, 2));
    try std.testing.expectError(error.Inkling2CachesFull, admitBudget(0, 101, 100, 0));
}
