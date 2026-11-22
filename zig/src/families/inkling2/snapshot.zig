//! A stream's prompt state kept for reuse: global layers' first `at` keys and values, the sliding layers' rows a next row can
//! read, and the convolution rings. Rows are stored in position order, so a state restores into a cache of any capacity.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");
const attention_prefill = @import("attention_prefill.zig");

/// One kept state: its bytes on the GPU (private) and the prompt tokens it holds.
pub const Snap = struct {
    buf: mtl.Buffer,
    at: u32,
    bytes: u64,
    /// The DSpark drafter's context rings for the same stream at `at` (dspark.zig keepLane), when it held rows there.
    draft: ?mtl.Buffer = null,
    /// First context position held in `draft` (the rings hold `[draft_from, at)`).
    draft_from: u64 = 0,
    freed: bool = false, // its storage given back early (a stream needed the room); its entry goes at the pass's report
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn free(self: *Snap) void {
        if (!self.freed) {
            if (self.follow) |f| {
                f.leave(f.ctx, self.buf);
                if (self.draft) |b| f.leave(f.ctx, b);
            }
            self.buf.deinit();
            if (self.draft) |b| b.deinit();
            self.draft = null;
        }
        self.freed = true;
    }

    pub fn setDraft(self: *Snap, buffer: mtl.Buffer, from: u64) void {
        if (self.draft) |old| {
            if (self.follow) |f| f.leave(f.ctx, old);
            self.bytes -= old.length();
            old.deinit();
        }
        self.draft = buffer;
        self.draft_from = from;
        self.bytes += buffer.length();
        if (self.follow) |f| f.join(f.ctx, buffer);
    }
};

/// One contiguous piece of a cache buffer: of `conv` (the convolution rings) or of `kv`.
const Piece = struct { conv: bool = false, off: usize, len: usize };

/// Most pieces a state holds: two a layer's keys and two its values (a ring may wrap), and the convolution rings.
const max_pieces = 4 * cfg.max_layers + 1;

/// Positions [lo, hi) of a ring of `ring_` slots, `width` bytes each, from `base`: one piece, or two where it wraps.
fn ring(base: usize, ring_: usize, width: usize, lo: usize, hi: usize, out: *std.ArrayList(Piece)) void {
    if (hi <= lo) return;
    const first = lo % ring_;
    const n = @min(hi - lo, ring_ - first);
    out.appendAssumeCapacity(.{ .off = base + first * width, .len = n * width });
    if (n < hi - lo) out.appendAssumeCapacity(.{ .off = base, .len = (hi - lo - n) * width });
}

/// A sliding layer's first kept position: the 64-key block of its window's first key, since the chunk attention reads whole
/// blocks.
pub fn slidingFrom(c: *const cfg.Text, at: usize) usize {
    return std.mem.alignBackward(usize, at -| (@as(usize, c.window) - 1), attention_prefill.key_block);
}

/// The pieces a state at `at` holds, in the snapshot's order: what the next rows read, nothing older.
fn pieces(c: *const cfg.Text, cache: *const st.Cache, at: usize, out: *std.ArrayList(Piece)) void {
    for (0..cache.layers) |i| {
        const lo = if (cfg.isGlobal(@intCast(i))) 0 else slidingFrom(c, at);
        ring(cache.keys(i), cache.ring(i), cache.row, lo, at, out);
        ring(cache.values(i), cache.ring(i), cache.row, lo, at, out);
    }
    out.appendAssumeCapacity(.{ .conv = true, .off = 0, .len = cache.conv_layer * cache.layers });
}

/// Bytes a state after `at` prompt tokens takes over the first `layers` layers.
pub fn bytes(c: *const cfg.Text, layers: usize, at: usize) u64 {
    const row: u64 = st.kvBytes(c) / 2;
    var total: u64 = @as(u64, layers) * st.convBytes(c);
    for (0..layers) |i| total += 2 * row * (at - (if (cfg.isGlobal(@intCast(i))) 0 else slidingFrom(c, at)));
    return total;
}

/// Copy `cache`'s state at its length into a new snapshot.
pub fn save(device: mtl.Device, queue: mtl.Queue, c: *const cfg.Text, cache: *const st.Cache) !Snap {
    const n = bytes(c, cache.layers, cache.len);
    const buf = try device.buffer(@intCast(@max(n, 16)), mtl.ResourceOptions.private);
    errdefer buf.deinit();
    try copy(queue, c, cache, buf, true);
    if (cache.follow) |f| f.join(f.ctx, buf);
    return .{ .buf = buf, .at = @intCast(cache.len), .bytes = n, .follow = cache.follow };
}

/// Make `cache` (empty, any capacity past the state) hold `snap`'s state.
pub fn restore(queue: mtl.Queue, c: *const cfg.Text, snap: *const Snap, cache: *st.Cache) !void {
    if (snap.freed or snap.at >= cache.capacity) return error.Inkling2SnapshotGone;
    cache.len = snap.at;
    errdefer cache.len = 0;
    try copy(queue, c, cache, snap.buf, false);
}

/// The pieces between the cache and a snapshot buffer, one blit pass, waited on.
fn copy(queue: mtl.Queue, c: *const cfg.Text, cache: *const st.Cache, buf: mtl.Buffer, out: bool) !void {
    var store: [max_pieces]Piece = undefined;
    var list: std.ArrayList(Piece) = .initBuffer(&store);
    pieces(c, cache, cache.len, &list);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    var at: usize = 0;
    for (list.items) |p| {
        if (p.len == 0) continue;
        const cache_buf = if (p.conv) cache.conv else cache.kv;
        if (out) e.copy(cache_buf, p.off, buf, at, p.len) else e.copy(buf, at, cache_buf, p.off, p.len);
        at += p.len;
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2 prompt state copy: {s}", .{msg});
        return error.GpuFailed;
    }
}

fn fakeCache(c: *const cfg.Text, capacity: usize, layers: usize, rows: usize) struct { cache: st.Cache, kv_bytes: usize } {
    const kv_row = st.kvBytes(c);
    var cache = st.Cache{
        .kv = undefined,
        .conv = undefined,
        .logits = undefined,
        .capacity = capacity,
        .layers = layers,
        .layer_at = @splat(0),
        .slots = @splat(0),
        .row = kv_row / 2,
        .rows = rows,
        .conv_layer = st.convBytes(c),
        .widths = st.convWidths(c),
    };
    var total: usize = 0;
    for (0..layers) |i| {
        const s = st.layerSlots(c, i, capacity, rows);
        cache.layer_at[i] = total;
        cache.slots[i] = @intCast(s);
        total += s * kv_row;
    }
    return .{ .cache = cache, .kv_bytes = total };
}

test "inkling2 snapshot: a state's bytes match its pieces, rings wrapped or not" {
    const c = st.small;
    const Cases = struct { cap: usize, ats: []const usize };
    const sets = [_]Cases{
        .{ .cap = 3000, .ats = &.{ 5, 300, 1023, 1024, 1087, 2999 } },
        .{ .cap = 200_000, .ats = &.{ 5, 300, 1023, 1024, 1087, 2999, 5376, 5377, 9000, 100_003 } },
    };
    for (sets) |set| {
        const f = fakeCache(&c, set.cap, 48, st.default_prefill_rows);
        for (set.ats) |at| {
            var store: [max_pieces]Piece = undefined;
            var list: std.ArrayList(Piece) = .initBuffer(&store);
            pieces(&c, &f.cache, at, &list);
            var sum: u64 = 0;
            for (list.items, 0..) |p, idx| {
                sum += p.len;
                if (idx + 1 == list.items.len) {
                    try std.testing.expect(p.conv);
                    try std.testing.expectEqual(@as(usize, 0), p.off);
                    try std.testing.expectEqual(f.cache.conv_layer * 48, p.len);
                } else {
                    try std.testing.expect(!p.conv);
                    try std.testing.expect(p.off + p.len <= f.kv_bytes);
                }
            }
            try std.testing.expectEqual(bytes(&c, 48, at), sum);
        }
    }
}

test "inkling2 snapshot: a state restores by position into a cache of another layout" {
    const gpa = std.testing.allocator;
    // Small row width (8 bytes = 1 u64 tag per position) so 100k-position images stay fast in host RAM
    var c = st.small;
    c.local = .{ .q = 48, .kv = 1, .dim = 4 };
    c.global = .{ .q = 48, .kv = 1, .dim = 4 };
    c.hidden = 16;
    const layers: usize = 6; // layers 0..4 sliding, layer 5 global
    const Pair = struct { src_cap: usize, dst_cap: usize, at: usize };
    const pairs = [_]Pair{
        .{ .src_cap = 200_000, .dst_cap = 3000, .at = 2500 },
        .{ .src_cap = 3000, .dst_cap = 200_000, .at = 2500 },
        .{ .src_cap = 200_000, .dst_cap = 200_000, .at = 5400 },
        .{ .src_cap = 200_000, .dst_cap = 200_000, .at = 6000 },
        .{ .src_cap = 200_000, .dst_cap = 150_000, .at = 100_003 },
    };
    const tag = struct {
        fn make(layer: usize, half: usize, pos: usize) u64 {
            return (@as(u64, layer) << 40) | (@as(u64, half) << 32) | @as(u64, pos);
        }
    }.make;
    for (pairs) |pr| {
        const src_f = fakeCache(&c, pr.src_cap, layers, st.default_prefill_rows);
        const dst_f = fakeCache(&c, pr.dst_cap, layers, st.default_prefill_rows);
        const src_kv = try gpa.alloc(u8, src_f.kv_bytes);
        defer gpa.free(src_kv);
        const dst_kv = try gpa.alloc(u8, dst_f.kv_bytes);
        defer gpa.free(dst_kv);
        @memset(src_kv, 0xcc);
        @memset(dst_kv, 0xdd);
        const conv_len = src_f.cache.conv_layer * layers;
        const src_conv = try gpa.alloc(u8, conv_len);
        defer gpa.free(src_conv);
        const dst_conv = try gpa.alloc(u8, conv_len);
        defer gpa.free(dst_conv);
        for (src_conv, 0..) |*b, idx| b.* = @truncate(idx * 31 + 7);
        @memset(dst_conv, 0);

        for (0..layers) |l| {
            const lo = if (cfg.isGlobal(@intCast(l))) 0 else slidingFrom(&c, pr.at);
            const slots: usize = src_f.cache.ring(l);
            for (lo..pr.at) |p| {
                const s = p % slots;
                const k_off = src_f.cache.keys(l) + s * src_f.cache.row;
                const v_off = src_f.cache.values(l) + s * src_f.cache.row;
                std.mem.writeInt(u64, src_kv[k_off..][0..8], tag(l, 0, p), .little);
                std.mem.writeInt(u64, src_kv[v_off..][0..8], tag(l, 1, p), .little);
            }
        }
        const snap_buf = try gpa.alloc(u8, @intCast(bytes(&c, layers, pr.at)));
        defer gpa.free(snap_buf);
        var src_store: [max_pieces]Piece = undefined;
        var src_list: std.ArrayList(Piece) = .initBuffer(&src_store);
        pieces(&c, &src_f.cache, pr.at, &src_list);
        var off: usize = 0;
        for (src_list.items) |p| {
            const from = if (p.conv) src_conv else src_kv;
            @memcpy(snap_buf[off..][0..p.len], from[p.off..][0..p.len]);
            off += p.len;
        }
        try std.testing.expectEqual(snap_buf.len, off);

        var dst_store: [max_pieces]Piece = undefined;
        var dst_list: std.ArrayList(Piece) = .initBuffer(&dst_store);
        pieces(&c, &dst_f.cache, pr.at, &dst_list);
        off = 0;
        for (dst_list.items) |p| {
            const to = if (p.conv) dst_conv else dst_kv;
            @memcpy(to[p.off..][0..p.len], snap_buf[off..][0..p.len]);
            off += p.len;
        }
        try std.testing.expectEqual(snap_buf.len, off);
        try std.testing.expectEqualSlices(u8, src_conv, dst_conv);
        for (0..layers) |l| {
            const lo = if (cfg.isGlobal(@intCast(l))) 0 else slidingFrom(&c, pr.at);
            const slots: usize = dst_f.cache.ring(l);
            for (lo..pr.at) |p| {
                const s = p % slots;
                const k_off = dst_f.cache.keys(l) + s * dst_f.cache.row;
                const v_off = dst_f.cache.values(l) + s * dst_f.cache.row;
                try std.testing.expectEqual(tag(l, 0, p), std.mem.readInt(u64, dst_kv[k_off..][0..8], .little));
                try std.testing.expectEqual(tag(l, 1, p), std.mem.readInt(u64, dst_kv[v_off..][0..8], .little));
            }
        }
    }
}

test "inkling2 snapshot: a state holds 32 KiB a token and at most 203 MiB more" {
    const c = st.small;
    try std.testing.expect(bytes(&c, 48, 32768) - 8 * 32768 * 4096 <= 203 << 20);
    try std.testing.expectEqual(@as(usize, 0), slidingFrom(&c, 1023));
    try std.testing.expectEqual(@as(usize, 0), slidingFrom(&c, 1086));
    try std.testing.expectEqual(@as(usize, 64), slidingFrom(&c, 1087));
}
