//! Inkling 2 behind the lane engine: one stream, its caches, one-row rounds and copy-drafted windows drawn on the host.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const greedy = @import("greedy.zig");
const Model = @import("model.zig").Model;
const row = @import("../step5/backend.zig").row;
const be = lanes.backend;

pub const Options = struct { capacity: usize, budget: u64 };

/// Token handles: drawn tokens and host-fed ones, in a host ring.
const ring = 4096;

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
    values: []f32,

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
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = scratch, .tokens = tokens, .values = values };
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
        self.gpa.free(self.values);
        self.gpa.destroy(self);
    }

    pub fn setResident(self: *Metal, hook: ?mtl.ResidencySet.Follow) void {
        if (self.follow) |f| mtl.resident.fields.apply(&self.scratch, f, false);
        self.follow = hook;
        if (hook) |f| mtl.resident.fields.apply(&self.scratch, f, true);
        var it = self.caches.valueIterator();
        while (it.next()) |cache| cache.*.setResident(hook);
    }

    /// Rows are independent in every kernel: any window gives one-row bits. One stream; no draft head.
    pub fn facts(self: *const Metal) lanes.Model {
        _ = self;
        return .{ .exact_width = st.window_rows, .gpu_tokens = false, .batch_rows = st.window_rows, .mtp = false, .speculate = false, .speculate_early = false, .drafts = 1, .max_streams = 1, .streams_exact = true };
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

    pub fn backend(self: *Metal) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .first_row = firstRowFn, .held = heldFn } };
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

    /// Greedy argmax or the keyed draw (lanes' gpu_full host twin) at `position`, a grammar mask's tokens only.
    fn draw(self: *Metal, l: []const u16, sampling: ?lanes.Sampling, position: u64, mask: []const u32) !u32 {
        const t = sampled(sampling) orelse {
            if (mask.len == 0) return greedy.argmax(l);
            var best: ?usize = null;
            for (l, 0..) |w, id| {
                if (!lanes.grammar.allows(mask, id)) continue;
                if (best == null or logit(w) > logit(l[best.?])) best = id;
            }
            return @intCast(best orelse 0);
        };
        for (l, self.values, 0..) |w, *v, id| {
            v.* = logit(w);
            if (mask.len > 0 and !lanes.grammar.allows(mask, id)) v.* = -std.math.inf(f32);
        }
        return lanes.gpu_full.draw(self.gpa, self.values, t, @intCast(position), null);
    }

    fn sampled(sampling: ?lanes.Sampling) ?lanes.Sampling {
        const t = sampling orelse return null;
        return if (t.temperature > 0) t else null;
    }

    fn put(self: *Metal, token: u32) u64 {
        const h = self.next;
        self.next += 1;
        self.tokens[h % ring] = token;
        return h;
    }

    /// ids at the cache's end in prompt chunks, the last chunk's last logits kept for the first draw.
    fn feed(self: *Metal, cache: *st.Cache, ids: []const u32, stream: *lanes.Stream) !void {
        var at: usize = 0;
        while (at < ids.len) {
            if (stream.isCancelled()) return error.Cancelled;
            const n = @min(ids.len - at, st.prompt_rows);
            const last = at + n == ids.len;
            try fwd.run(self.model, &self.scratch, cache, ids[at..][0..n], .{ .logits = if (last) .last else .none, .prompt = true });
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
        try self.feed(cache, ids, stream);
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
        return self.put(try self.draw(self.logits(cache), stream.sampling, position, mask));
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
        return self.put(try self.draw(self.scratch.logits.slice(u16, self.model.config.text.vocab), stream.sampling, position, &.{}));
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

    /// One stream's pending token and copy drafts in one forward, every row drawn; keep() rolls back by length.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        const c = &self.model.config.text;
        const words = (c.vocab + 31) / 32;
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
        try fwd.run(self.model, &self.scratch, cache, ids[0..rows], .{ .logits = .all });
        cache.len += rows;
        @memcpy(o.drafts, w.tokens);
        const all = self.scratch.logits.slice(u16, rows * c.vocab);
        for (0..rows) |r| {
            const l = all[r * c.vocab ..][0..c.vocab];
            const mask = if (w.masks.len >= (r + 1) * words) w.masks[r * words ..][0..words] else &.{};
            o.sampled[r] = try self.draw(l, w.stream.sampling, w.positions[r], mask);
            if (o.rows.len == rows) o.rows[r] = row(l, o.sampled[r], w.stream.logprobs orelse 0);
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
    }
};

fn logit(word: u16) f32 {
    return @bitCast(@as(u32, word) << 16);
}
