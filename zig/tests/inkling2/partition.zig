//! tf-inkling-run's `partition`: a prompt fed through the serving prompt path twice, once in chunks of the prefill rows and once in
//! chunks of given lengths, and the two legs' results compared bit for bit.
//!
//! A row's results must not depend on the chunk it was fed in. Each leg starts on a cache whose keys, values, convolution rings and
//! kept logits are NaN; before every chunk the whole scratch is NaN and so is every cache slot no later row can read; after each
//! chunk its rows are read out by position through the slot map and every slot that stayed poisoned must still be NaN. A kernel that
//! reads outside the live rows, or leaves a row unwritten, turns what is compared to NaN and fails the run, as does a write outside
//! the chunk's rows, as does any bit that differs between the legs.
//!
//! Compared: every layer's keys and values of every row, every convolution ring, every row's last-layer output and expert picks;
//! on the whole model also every row's taps (the draft head's features, finals among them), the last row's finals and the logits
//! the prompt keeps for its first draw; with --drafts the proposals of the first hold. A model of the first L layers has no
//! finals, taps or logits.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ik = tf.inkling2;
const settings = @import("settings.zig");
const Absorber = @import("absorber.zig").Absorber;

/// The bytes of the poison: bf16, f16 and f32 words made of them are NaN, and an index made of them is out of range.
const nan_byte: u8 = 0xff;
const nan_word: u16 = 0xffff;
const depth = ik.dspark.max_depth;

const Options = struct { cuts: []const u32, restores: []const u32 = &.{}, layers: ?u32 = null, drafts: bool = false };

fn parseOptions(gpa: std.mem.Allocator, args: []const [:0]const u8) !Options {
    var cuts: std.ArrayList(u32) = .empty;
    errdefer cuts.deinit(gpa);
    var restores: std.ArrayList(u32) = .empty;
    errdefer restores.deinit(gpa);
    var o: Options = .{ .cuts = &.{} };
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--drafts")) {
            o.drafts = true;
        } else if (std.mem.eql(u8, args[i], "--layers")) {
            i += 1;
            if (i == args.len) return error.ExpectedAValueAfterTheOption;
            o.layers = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--restore")) {
            i += 1;
            if (i == args.len) return error.ExpectedAValueAfterTheOption;
            const at = try std.fmt.parseInt(u32, args[i], 10);
            if (at == 0) return error.RestorePositionOutOfRange;
            try restores.append(gpa, at);
        } else if (std.mem.startsWith(u8, args[i], "--")) {
            return error.UnknownPartitionOption;
        } else {
            const cut = try std.fmt.parseInt(u32, args[i], 10);
            if (cut == 0) return error.ACutIsAtLeastOneRow;
            try cuts.append(gpa, cut);
        }
    }
    if (cuts.items.len == 0) return error.ExpectedCuts;
    if (o.layers == 0) return error.ExpectedPositiveLayers;
    if (o.drafts and o.layers != null) return error.DraftsNeedTheWholeModel;
    o.cuts = try cuts.toOwnedSlice(gpa);
    errdefer gpa.free(o.cuts);
    o.restores = try restores.toOwnedSlice(gpa);
    return o;
}

/// The chunk lengths of a leg over `total` rows: `cuts` first, then chunks of `rows` for the rest.
fn chunkLengths(gpa: std.mem.Allocator, total: usize, cuts: []const u32, rows: usize) ![]usize {
    var out: std.ArrayList(usize) = .empty;
    errdefer out.deinit(gpa);
    var at: usize = 0;
    for (cuts) |cut| {
        if (cut > rows) return error.ACutExceedsThePrefillRows;
        if (at + cut > total) return error.CutsExceedThePrompt;
        try out.append(gpa, cut);
        at += cut;
    }
    while (at < total) {
        const n = @min(total - at, rows);
        try out.append(gpa, n);
        at += n;
    }
    return out.toOwnedSlice(gpa);
}

/// What one leg leaves: the cache's rows and rings, every row's last-layer output and picks and, on the whole model, the taps, the
/// last row's finals, the kept logits and the first hold's proposals.
const Leg = struct {
    kv: []u16, // [layer][keys, values][row][kv heads][dim], rows 0 .. T
    conv: []u16, // the rings: the cache's whole conv buffer
    h: []u16, // [row][hidden]
    picks: []u32, // [layer][row][topk]
    taps: ?[]u16 = null, // [row][taps][hidden]
    finals: ?[]u16 = null, // the last row's, [hidden]
    logits: ?[]u16 = null, // [vocab]
    props: ?[depth]u32 = null,

    fn deinit(l: *Leg, gpa: std.mem.Allocator) void {
        gpa.free(l.kv);
        gpa.free(l.conv);
        gpa.free(l.h);
        gpa.free(l.picks);
        if (l.taps) |x| gpa.free(x);
        if (l.finals) |x| gpa.free(x);
        if (l.logits) |x| gpa.free(x);
    }
};

fn finish(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("partition: {s}", .{msg});
        return error.GpuFailed;
    }
}

/// Every scratch buffer, and the `extra` buffers, to NaN bytes.
fn poisonScratch(m: *const ik.Model, s: *ik.state.Scratch, extra: []const mtl.Buffer) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    inline for (ik.state.fields) |f| blit.fill(@field(s, f), 0, @field(s, f).length(), nan_byte);
    for (extra) |buf| blit.fill(buf, 0, buf.length(), nan_byte);
    blit.end();
    try finish(cb);
}

/// Bytes of a cache row of keys (or of values): 8 key heads of 128 dims in bf16.
fn rowBytes(c: *const ik.config.Text) usize {
    return ik.state.kvBytes(c) / 2;
}

/// The first position of layer `l` a row at `at` can read: 0 on a global layer; on a sliding one the 64-key block of its
/// window's first key.
fn liveFrom(c: *const ik.config.Text, l: usize, at: usize) usize {
    return if (ik.config.isGlobal(@intCast(l))) 0 else std.mem.alignBackward(usize, at -| (@as(usize, c.window) - 1), 64);
}

/// Byte offset in the cache's kv buffer of position `p` of layer `l`'s keys (`values`: its values).
fn rowAt(cache: *const ik.state.Cache, l: usize, p: usize, values: bool, row: usize) usize {
    return (if (values) cache.values(l) else cache.keys(l)) + (p % cache.ring(l)) * row;
}

fn fillSlotRange(blit: mtl.BlitEncoder, kv: mtl.Buffer, base: usize, lo: usize, hi: usize, row: usize) void {
    if (hi > lo) blit.fill(kv, base + lo * row, (hi - lo) * row, nan_byte);
}

/// For every layer, NaN every slot except those of positions `[liveFrom(l, pos), pos)`; `start` also poisons the convolution
/// rings and the kept logits.
fn poisonCache(m: *const ik.Model, cache: *ik.state.Cache, pos: usize, start: bool) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const c = &m.config.text;
    const row = rowBytes(c);
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    for (0..cache.layers) |l| {
        const slots: usize = cache.ring(l);
        const lo = liveFrom(c, l, pos);
        for ([_]usize{ cache.keys(l), cache.values(l) }) |base| {
            if (pos <= lo) {
                fillSlotRange(blit, cache.kv, base, 0, slots, row);
            } else if (pos - lo < slots) {
                const first = lo % slots;
                const end = pos % slots;
                if (first < end) {
                    fillSlotRange(blit, cache.kv, base, 0, first, row);
                    fillSlotRange(blit, cache.kv, base, end, slots, row);
                } else {
                    fillSlotRange(blit, cache.kv, base, end, first, row);
                }
            }
        }
    }
    if (start) blit.fill(cache.conv, 0, cache.conv.length(), nan_byte);
    blit.end();
    try finish(cb);
    if (start) @memset(cache.logits.slice(u8, cache.logits.length()), nan_byte);
}

/// Whether slot `s` of a ring of `slots` holds any position in `[lo, hi)`.
fn slotInRange(s: usize, slots: usize, lo: usize, hi: usize) bool {
    if (hi <= lo) return false;
    if (hi - lo >= slots) return true;
    const first = lo % slots;
    const end = hi % slots;
    return if (first < end) (s >= first and s < end) else (s >= first or s < end);
}

/// After a chunk of `n` rows at `pos`: copy the chunk's rows `[pos, pos + n)` of every layer's keys and values into `leg.kv`
/// by position, check they are finite, and confirm every slot outside `[liveFrom(l, pos), pos + n)` mod ring is still NaN.
fn readChunkAndCheck(m: *const ik.Model, cache: *ik.state.Cache, staging: mtl.Buffer, pos: usize, n: usize, T: usize, leg: *Leg) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const c = &m.config.text;
    const row = rowBytes(c);
    const row16 = row / 2;
    for (0..cache.layers) |l| {
        const slots: usize = cache.ring(l);
        const kv_layer = slots * 2 * row;
        const cb = m.queue.commandBuffer();
        const blit = cb.blit();
        blit.copy(cache.kv, cache.keys(l), staging, 0, kv_layer);
        blit.end();
        try finish(cb);
        const words = staging.slice(u16, kv_layer / 2);
        const half16 = slots * row16;
        const lo = liveFrom(c, l, pos);
        for ([_]usize{ 0, half16 }, 0..) |base, kind| {
            const dst = leg.kv[(2 * l + kind) * T * row16 ..];
            for (pos..pos + n) |p| {
                const s = p % slots;
                @memcpy(dst[p * row16 ..][0..row16], words[base + s * row16 ..][0..row16]);
            }
            try allFinite("a layer's keys and values", dst[pos * row16 ..][0 .. n * row16], row16);
            for (0..slots) |s| {
                if (slotInRange(s, slots, lo, pos + n)) continue;
                for (words[base + s * row16 ..][0..row16]) |x| if (x != nan_word) {
                    std.debug.print("partition: the chunk [{d}, {d}) wrote outside its rows: layer {d}, {s}, slot {d}\n", .{ pos, pos + n, l, if (kind == 0) "keys" else "values", s });
                    return error.AChunkWrotePastItsRows;
                };
            }
        }
    }
}

fn finite16(x: u16) bool {
    return x & 0x7f80 != 0x7f80;
}

fn nan16(x: u16) bool {
    return x & 0x7fff > 0x7f80;
}

/// A bf16 word of `words` (rows of `width` words) that is not finite: the poison reached the result `what`.
fn allFinite(what: []const u8, words: []const u16, width: usize) !void {
    for (words, 0..) |x, i| if (!finite16(x)) {
        std.debug.print("partition: {s}: a NaN or infinity at row {d}, word {d}\n", .{ what, i / width, i % width });
        return error.PoisonReachedAResult;
    };
}

/// Every tap of `rows` rows of taps that the model writes is finite: the inputs of the tap layers it holds, and the finals (a model
/// of fewer layers than the first tap layer writes the finals alone).
fn tapsFinite(m: *const ik.Model, taps: []const u16, rows: usize) !void {
    const H: usize = m.config.text.hidden;
    for (0..rows) |r| for (0..ik.state.feature_taps) |k| {
        if (k == ik.state.feature_taps - 1 or ik.forward.tap_layers[k] < m.layers) try allFinite("a row's taps", taps[(r * ik.state.feature_taps + k) * H ..][0..H], H);
    };
}

/// One leg: the prompt in chunks of `lengths`, every result kept. When `restore_at` is positive, `[0, restore_at)` runs first on
/// its own cache, saves a snapshot and restores it into a fresh poisoned cache before `lengths` run over `[restore_at, T)`.
fn runLeg(gpa: std.mem.Allocator, m: *ik.Model, s: *ik.state.Scratch, tp: ik.forward.Taps, ids: []const u32, restore_at: usize, lengths: []const usize, draft: ?settings.Draft) !Leg {
    const c = &m.config.text;
    const H: usize = c.hidden;
    const T = ids.len;
    const L: usize = m.layers;
    const full = m.layers == c.layers;
    const row = rowBytes(c);
    const tap_row: usize = ik.state.feature_taps * H;
    var snap: ?ik.snapshot.Snap = null;
    defer if (snap) |*sp| sp.free();
    if (restore_at > 0) {
        var src = try ik.state.Cache.init(m.device, c, restore_at + 1, m.layers, m.prefill_rows);
        defer src.deinit();
        var pos: usize = 0;
        while (pos < restore_at) {
            const n = @min(restore_at - pos, m.prefill_rows);
            try ik.forward.run(m, s, &src, ids[pos..][0..n], .{ .logits = .none, .prompt = true });
            src.len += n;
            pos += n;
        }
        snap = try ik.snapshot.save(m.device, m.queue, c, &src);
    }
    var cache = try ik.state.Cache.init(m.device, c, T + 1, m.layers, m.prefill_rows);
    defer cache.deinit();
    var leg: Leg = .{
        .kv = try gpa.alloc(u16, L * 2 * T * row / 2),
        .conv = try gpa.alloc(u16, cache.conv.length() / 2),
        .h = try gpa.alloc(u16, T * H),
        .picks = try gpa.alloc(u32, L * T * c.topk),
    };
    errdefer leg.deinit(gpa);
    if (full) {
        leg.taps = try gpa.alloc(u16, T * tap_row);
        leg.finals = try gpa.alloc(u16, H);
        leg.logits = try gpa.alloc(u16, c.vocab);
    }
    var max_layer_bytes: usize = cache.conv.length();
    for (0..cache.layers) |l| max_layer_bytes = @max(max_layer_bytes, @as(usize, cache.ring(l)) * 2 * row);
    const staging = try m.device.buffer(@max(max_layer_bytes, 16), mtl.ResourceOptions.shared);
    defer staging.deinit();
    const absorber: ?Absorber = if (draft) |d| try Absorber.init(gpa, m, d, ids) else null;
    defer if (absorber) |a| a.deinit();
    try poisonCache(m, &cache, 0, true);
    if (snap) |*sp| {
        try ik.snapshot.restore(m.queue, c, sp, &cache);
        sp.free();
    }
    var at: usize = restore_at;
    for (lengths) |n| {
        const last = at + n == T;
        try poisonScratch(m, s, &.{ tp.h, tp.eids });
        try poisonCache(m, &cache, at, false);
        try ik.forward.run(m, s, &cache, ids[at..][0..n], .{
            .logits = if (full and last) .last else .none,
            .prompt = true,
            .taps = tp,
            .feature_taps = if (full) .{ .buf = s.ptaps } else null,
        });
        @memcpy(leg.h[at * H ..][0 .. n * H], s.h.slice(u16, n * H));
        try allFinite("a row's last-layer output", leg.h[at * H ..][0 .. n * H], H);
        const picks = tp.eids.slice(u32, L * n * c.topk);
        for (0..L) |l| @memcpy(leg.picks[(l * T + at) * c.topk ..][0 .. n * c.topk], picks[l * n * c.topk ..][0 .. n * c.topk]);
        if (full) {
            @memcpy(leg.taps.?[at * tap_row ..][0 .. n * tap_row], s.ptaps.slice(u16, n * tap_row));
            try tapsFinite(m, leg.taps.?[at * tap_row ..][0 .. n * tap_row], n);
            if (absorber) |a| _ = try a.take(s.ptaps, at, n);
            if (last) {
                @memcpy(leg.finals.?, s.finals.slice(u16, n * H)[(n - 1) * H ..][0..H]);
                try allFinite("the last row's finals", leg.finals.?, H);
                // the last row's logits stay with the cache for the first draw, as the backend keeps them
                @memcpy(cache.logits.slice(u16, c.vocab), s.logits.slice(u16, c.vocab));
                @memcpy(leg.logits.?, cache.logits.slice(u16, c.vocab));
                for (leg.logits.?, 0..) |x, i| if (nan16(x)) {
                    std.debug.print("partition: the kept logits: a NaN at id {d}\n", .{i});
                    return error.PoisonReachedAResult;
                };
                if (absorber) |a| {
                    var props: [depth]u32 = undefined;
                    try a.firstHold(s.ptaps, n - 1, ik.greedy.argmax(leg.logits.?), &props);
                    leg.props = props;
                }
            }
        }
        try readChunkAndCheck(m, &cache, staging, at, n, T, &leg);
        cache.len += n;
        at += n;
    }
    for (0..L) |l| for (leg.picks[(l * T + restore_at) * c.topk ..][0 .. (T - restore_at) * c.topk]) |p| if (p >= c.experts) {
        std.debug.print("partition: an expert pick outside the table: {d}\n", .{p});
        return error.PoisonReachedAResult;
    };
    try readConv(m, &cache, staging, T, &leg);
    return leg;
}

/// The cache's convolution rings into `leg`.
fn readConv(m: *const ik.Model, cache: *ik.state.Cache, staging: mtl.Buffer, T: usize, leg: *Leg) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    blit.copy(cache.conv, 0, staging, 0, cache.conv.length());
    blit.end();
    try finish(cb);
    @memcpy(leg.conv, staging.slice(u16, cache.conv.length() / 2));
    // every ring slot is written once the prompt is as long as a ring
    if (T >= ik.state.ring) try allFinite("a convolution ring", leg.conv, 1);
}

/// The rows of one kind that differ between the legs.
const Differ = struct {
    what: []const u8,
    rows: usize = 0,
    differ: usize = 0,

    /// Counts a row that differs; the first few say where.
    fn note(d: *Differ, comptime fmt: []const u8, args: anytype) void {
        d.differ += 1;
        if (d.differ <= 3) std.debug.print("  {s} differ:" ++ fmt ++ "\n", .{d.what} ++ args);
    }

    fn report(d: Differ) void {
        std.debug.print("{{\"compared\": \"{s}\", \"rows\": {d}, \"differ\": {d}}}\n", .{ d.what, d.rows, d.differ });
    }
};

/// The byte offset in the conv buffer of layer l's ring `which`.
fn ringOffset(c: *const ik.config.Text, l: usize, which: usize) usize {
    var at = l * ik.state.convBytes(c);
    for (ik.state.convWidths(c)[0..which]) |w| at += ik.state.ring * w * 2;
    return at;
}

/// Every difference between the legs over rows `[from, T)`, printed by what differs; the count, zero when the legs agree to the bit.
fn compare(m: *const ik.Model, a: *const Leg, b: *const Leg, from: usize, T: usize, compare_props: bool) usize {
    const c = &m.config.text;
    const H: usize = c.hidden;
    const L: usize = m.layers;
    const row16 = rowBytes(c) / 2;
    var total: usize = 0;
    var kv: Differ = .{ .what = "keys and values" };
    for (0..L) |l| for (0..2) |kind| for (from..T) |p| {
        kv.rows += 1;
        const r = (2 * l + kind) * T + p;
        if (!std.mem.eql(u16, a.kv[r * row16 ..][0..row16], b.kv[r * row16 ..][0..row16]))
            kv.note(" layer {d} {s} row {d}", .{ l, if (kind == 0) "keys" else "values", p });
    };
    kv.report();
    total += kv.differ;
    var rings: Differ = .{ .what = "convolution rings" };
    for (0..L) |l| for (0..4) |which| {
        const width: usize = ik.state.convWidths(c)[which];
        for (0..ik.state.ring) |p| {
            rings.rows += 1;
            const lo = (ringOffset(c, l, which) + p * width * 2) / 2;
            if (!std.mem.eql(u16, a.conv[lo..][0..width], b.conv[lo..][0..width])) rings.note(" layer {d} ring {d} slot {d}", .{ l, which, p });
        }
    };
    rings.report();
    total += rings.differ;
    var h: Differ = .{ .what = "last-layer outputs" };
    for (from..T) |r| {
        h.rows += 1;
        if (!std.mem.eql(u16, a.h[r * H ..][0..H], b.h[r * H ..][0..H])) h.note(" row {d}", .{r});
    }
    h.report();
    total += h.differ;
    var picks: Differ = .{ .what = "expert picks" };
    for (0..L) |l| for (from..T) |p| {
        picks.rows += 1;
        const r = l * T + p;
        if (!std.mem.eql(u32, a.picks[r * c.topk ..][0..c.topk], b.picks[r * c.topk ..][0..c.topk])) picks.note(" layer {d} row {d}", .{ l, p });
    };
    picks.report();
    total += picks.differ;
    if (a.taps) |ta| {
        const tb = b.taps.?;
        var taps: Differ = .{ .what = "taps" };
        const width = ik.state.feature_taps * H;
        for (from..T) |r| {
            taps.rows += 1;
            if (!std.mem.eql(u16, ta[r * width ..][0..width], tb[r * width ..][0..width])) {
                var k: usize = 0;
                while (std.mem.eql(u16, ta[r * width + k * H ..][0..H], tb[r * width + k * H ..][0..H])) k += 1;
                taps.note(" row {d}, tap {d}", .{ r, k });
            }
        }
        taps.report();
        total += taps.differ;
        var finals: Differ = .{ .what = "the last row's finals", .rows = 1 };
        if (!std.mem.eql(u16, a.finals.?, b.finals.?)) finals.note(" the hidden row", .{});
        finals.report();
        total += finals.differ;
        var logits: Differ = .{ .what = "the kept logits", .rows = 1 };
        if (!std.mem.eql(u16, a.logits.?, b.logits.?)) logits.note(" the vocabulary row", .{});
        logits.report();
        total += logits.differ;
    }
    if (a.props) |pa| {
        if (compare_props) {
            var props: Differ = .{ .what = "the first hold's proposals", .rows = depth };
            for (pa, b.props.?, 0..) |x, y, i| if (x != y) props.note(" proposal {d}: {d} against {d}", .{ i, x, y });
            props.report();
            total += props.differ;
        } else {
            std.debug.print("{{\"compared\": \"the first hold's proposals\", \"proposals\": \"not compared\"}}\n", .{});
        }
    }
    return total;
}

pub fn run(init: std.process.Init, model_dir: []const u8, ids: []const u32, args: []const [:0]const u8) !void {
    const gpa = init.gpa;
    const opt = try parseOptions(gpa, args);
    defer gpa.free(opt.cuts);
    defer gpa.free(opt.restores);
    const T = ids.len;
    if (T == 0) return error.ExpectedAPrompt;
    for (opt.restores) |at| if (at >= T) return error.RestorePositionOutOfRange;
    const draft: ?settings.Draft = if (opt.drafts) try settings.draft() else null;
    const m = try ik.Model.load(gpa, init.io, model_dir, .{
        .layers = opt.layers,
        .drafts = opt.drafts,
        .draft_bits = if (draft) |d| d.bits else .default,
        .prefill_rows = try settings.prefillRows(),
    });
    defer m.deinit();
    const c = &m.config.text;
    const rows: usize = m.prefill_rows;
    const default_chunks = try chunkLengths(gpa, T, &.{}, rows);
    defer gpa.free(default_chunks);
    const cut_chunks = try chunkLengths(gpa, T, opt.cuts, rows);
    defer gpa.free(cut_chunks);
    if (std.mem.eql(usize, default_chunks, cut_chunks)) return error.TheCutsGiveTheDefaultChunking;
    var s = try ik.state.Scratch.init(m.device, c, @max(T + 1, ik.state.window_rows), m.chunking());
    defer s.deinit();
    // what the forward writes besides the cache, for the diagnosis: every layer's picks (and the tap layers' inputs, unread)
    const tp: ik.forward.Taps = .{
        .h = try m.device.buffer(ik.forward.tap_layers.len * rows * c.hidden * 2, mtl.ResourceOptions.shared),
        .eids = try m.device.buffer(@as(usize, m.layers) * rows * 32, mtl.ResourceOptions.shared),
    };
    defer tp.h.deinit();
    defer tp.eids.deinit();
    var a = try runLeg(gpa, m, &s, tp, ids, 0, default_chunks, draft);
    defer a.deinit(gpa);
    var differ: usize = 0;
    {
        var b = try runLeg(gpa, m, &s, tp, ids, 0, cut_chunks, draft);
        defer b.deinit(gpa);
        differ = compare(m, &a, &b, 0, T, true);
    }
    const restore_diffs = try gpa.alloc(usize, opt.restores.len);
    defer gpa.free(restore_diffs);
    var total_differ = differ;
    for (opt.restores, 0..) |restore_at, ri| {
        const tail_chunks = try chunkLengths(gpa, T - restore_at, &.{}, rows);
        defer gpa.free(tail_chunks);
        var r = try runLeg(gpa, m, &s, tp, ids, restore_at, tail_chunks, draft);
        defer r.deinit(gpa);
        restore_diffs[ri] = compare(m, &a, &r, restore_at, T, T - restore_at >= 1023);
        total_differ += restore_diffs[ri];
    }
    std.debug.print("{{\"partition\": \"{s}\", \"prompt_tokens\": {d}, \"layers\": {d}, \"prompt_path\": \"{s}\", \"prefill_rows\": {d}, \"default_chunks\": {d}, \"cut_chunks\": {d}, \"first_cuts\": {any}, \"differ\": {d}, \"restore\": [", .{ if (differ == 0) "identical" else "differs", T, m.layers, @tagName(m.promptPath()), rows, default_chunks.len, cut_chunks.len, cut_chunks[0..@min(cut_chunks.len, 16)], differ });
    for (opt.restores, restore_diffs, 0..) |restore_at, rd, ri| {
        std.debug.print("{s}{{\"at\": {d}, \"result\": \"{s}\"}}", .{ if (ri == 0) "" else ", ", restore_at, if (rd == 0) "identical" else "differs" });
    }
    std.debug.print("]}}\n", .{});
    if (total_differ != 0) return error.PartitionDiffers;
}
