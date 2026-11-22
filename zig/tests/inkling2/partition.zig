//! tf-inkling-run's `partition`: a prompt fed through the serving prompt path twice, once in chunks of the prefill rows and once in
//! chunks of given lengths, and the two legs' results compared bit for bit.
//!
//! A row's results must not depend on the chunk it was fed in. Each leg starts on a cache whose keys, values, convolution rings and
//! kept logits are NaN; before every chunk the whole scratch is NaN and so are the cache's rows from the chunk's first on; after it
//! the 64 rows past the chunk's end must still be NaN. A kernel that reads past the rows it was given, or leaves a row unwritten,
//! turns what is compared to NaN and fails the run, as does a write past the rows, as does any bit that differs between the legs.
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

/// Rows of the cache past a chunk's end that must stay untouched by it: a 64-key block of attention reaches this far.
const reach = 64;
/// The bytes of the poison: bf16, f16 and f32 words made of them are NaN, and an index made of them is out of range.
const nan_byte: u8 = 0xff;
const nan_word: u16 = 0xffff;
const depth = ik.dspark.max_depth;

const Options = struct { cuts: []const u32, layers: ?u32 = null, drafts: bool = false };

fn parseOptions(gpa: std.mem.Allocator, args: []const [:0]const u8) !Options {
    var cuts: std.ArrayList(u32) = .empty;
    errdefer cuts.deinit(gpa);
    var o: Options = .{ .cuts = &.{} };
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--drafts")) {
            o.drafts = true;
        } else if (std.mem.eql(u8, args[i], "--layers")) {
            i += 1;
            if (i == args.len) return error.ExpectedAValueAfterTheOption;
            o.layers = try std.fmt.parseInt(u32, args[i], 10);
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

/// The cache's rows from `from` to its end, of every layer's keys and values, to NaN bytes; `start` also the convolution rings and
/// the kept logits.
fn poisonCache(m: *const ik.Model, cache: *ik.state.Cache, from: usize, start: bool) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const row = rowBytes(&m.config.text);
    const held = ik.state.cachePositions(cache.capacity);
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    if (held > from) for (0..cache.layers) |l| {
        blit.fill(cache.kv, cache.keys(l) + from * row, (held - from) * row, nan_byte);
        blit.fill(cache.kv, cache.values(l) + from * row, (held - from) * row, nan_byte);
    };
    if (start) blit.fill(cache.conv, 0, cache.conv.length(), nan_byte);
    blit.end();
    try finish(cb);
    if (start) @memset(cache.logits.slice(u8, cache.logits.length()), nan_byte);
}

/// The `reach` rows of every layer's keys and values from `from` are still NaN: the chunk before them wrote nothing there.
fn checkReach(m: *const ik.Model, cache: *ik.state.Cache, staging: mtl.Buffer, from: usize) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const row = rowBytes(&m.config.text);
    const held = ik.state.cachePositions(cache.capacity);
    if (from >= held) return;
    const rows = @min(reach, held - from);
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    for (0..cache.layers) |l| {
        blit.copy(cache.kv, cache.keys(l) + from * row, staging, (2 * l) * rows * row, rows * row);
        blit.copy(cache.kv, cache.values(l) + from * row, staging, (2 * l + 1) * rows * row, rows * row);
    }
    blit.end();
    try finish(cb);
    for (staging.slice(u16, 2 * cache.layers * rows * row / 2), 0..) |x, i| if (x != nan_word) {
        const at = i / (row / 2); // the row of 2 * layers * rows
        std.debug.print("partition: the chunk ending at row {d} wrote past its rows: layer {d}, {s}, row {d}\n", .{ from, at / (2 * rows), if (at / rows % 2 == 0) "keys" else "values", from + at % rows });
        return error.AChunkWrotePastItsRows;
    };
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

/// One leg: the prompt in chunks of `lengths`, every result kept.
fn runLeg(gpa: std.mem.Allocator, m: *ik.Model, s: *ik.state.Scratch, tp: ik.forward.Taps, ids: []const u32, lengths: []const usize, draft: ?settings.Draft) !Leg {
    const c = &m.config.text;
    const H: usize = c.hidden;
    const T = ids.len;
    const L: usize = m.layers;
    const full = m.layers == c.layers;
    const row = rowBytes(c);
    const tap_row: usize = ik.state.feature_taps * H;
    var cache = try ik.state.Cache.init(m.device, c, T + 1, m.layers);
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
    const staging = try m.device.buffer(@max(2 * L * reach * row, 16), mtl.ResourceOptions.shared);
    defer staging.deinit();
    const absorber: ?Absorber = if (draft) |d| try Absorber.init(gpa, m, d, ids) else null;
    defer if (absorber) |a| a.deinit();
    try poisonCache(m, &cache, 0, true);
    var at: usize = 0;
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
        try checkReach(m, &cache, staging, at + n);
        cache.len += n;
        at += n;
    }
    for (leg.picks) |p| if (p >= c.experts) {
        std.debug.print("partition: an expert pick outside the table: {d}\n", .{p});
        return error.PoisonReachedAResult;
    };
    try readCache(m, &cache, T, &leg);
    return leg;
}

/// The cache's rows 0 .. T of every layer and its rings into `leg`, the rows past T checked to be NaN still.
fn readCache(m: *const ik.Model, cache: *ik.state.Cache, T: usize, leg: *Leg) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const row = rowBytes(&m.config.text);
    const row16 = row / 2;
    const staging = try m.device.buffer(@max(cache.kv_layer, cache.conv.length()), mtl.ResourceOptions.shared);
    defer staging.deinit();
    const half16 = cache.kv_layer / 2 / 2;
    for (0..cache.layers) |l| {
        const cb = m.queue.commandBuffer();
        const blit = cb.blit();
        blit.copy(cache.kv, cache.keys(l), staging, 0, cache.kv_layer);
        blit.end();
        try finish(cb);
        const words = staging.slice(u16, cache.kv_layer / 2);
        for ([_]usize{ 0, half16 }, 0..) |base, kind| {
            @memcpy(leg.kv[(2 * l + kind) * T * row16 ..][0 .. T * row16], words[base..][0 .. T * row16]);
            for (words[base + T * row16 .. base + half16], 0..) |x, i| if (x != nan_word) {
                std.debug.print("partition: layer {d}'s {s} wrote past the prompt: row {d}\n", .{ l, if (kind == 0) "keys" else "values", T + i / row16 });
                return error.AChunkWrotePastItsRows;
            };
        }
        try allFinite("a layer's keys and values", leg.kv[2 * l * T * row16 ..][0 .. 2 * T * row16], row16);
    }
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

/// Every difference between the legs over `T` rows, printed by what differs; the count, zero when the legs agree to the bit.
fn compare(m: *const ik.Model, a: *const Leg, b: *const Leg, T: usize) usize {
    const c = &m.config.text;
    const H: usize = c.hidden;
    const L: usize = m.layers;
    const row16 = rowBytes(c) / 2;
    var total: usize = 0;
    var kv: Differ = .{ .what = "keys and values" };
    for (0..L * 2 * T) |r| {
        kv.rows += 1;
        if (!std.mem.eql(u16, a.kv[r * row16 ..][0..row16], b.kv[r * row16 ..][0..row16]))
            kv.note(" layer {d} {s} row {d}", .{ r / (2 * T), if (r / T % 2 == 0) "keys" else "values", r % T });
    }
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
    for (0..T) |r| {
        h.rows += 1;
        if (!std.mem.eql(u16, a.h[r * H ..][0..H], b.h[r * H ..][0..H])) h.note(" row {d}", .{r});
    }
    h.report();
    total += h.differ;
    var picks: Differ = .{ .what = "expert picks" };
    for (0..L * T) |r| {
        picks.rows += 1;
        if (!std.mem.eql(u32, a.picks[r * c.topk ..][0..c.topk], b.picks[r * c.topk ..][0..c.topk])) picks.note(" layer {d} row {d}", .{ r / T, r % T });
    }
    picks.report();
    total += picks.differ;
    if (a.taps) |ta| {
        const tb = b.taps.?;
        var taps: Differ = .{ .what = "taps" };
        const width = ik.state.feature_taps * H;
        for (0..T) |r| {
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
        var props: Differ = .{ .what = "the first hold's proposals", .rows = depth };
        for (pa, b.props.?, 0..) |x, y, i| if (x != y) props.note(" proposal {d}: {d} against {d}", .{ i, x, y });
        props.report();
        total += props.differ;
    }
    return total;
}

pub fn run(init: std.process.Init, model_dir: []const u8, ids: []const u32, args: []const [:0]const u8) !void {
    const gpa = init.gpa;
    const opt = try parseOptions(gpa, args);
    defer gpa.free(opt.cuts);
    const T = ids.len;
    if (T == 0) return error.ExpectedAPrompt;
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
    var s = try ik.state.Scratch.init(m.device, c, @max(T + 1, ik.state.window_rows), m.prefill_rows);
    defer s.deinit();
    // what the forward writes besides the cache, for the diagnosis: every layer's picks (and the tap layers' inputs, unread)
    const tp: ik.forward.Taps = .{
        .h = try m.device.buffer(ik.forward.tap_layers.len * rows * c.hidden * 2, mtl.ResourceOptions.shared),
        .eids = try m.device.buffer(@as(usize, m.layers) * rows * 32, mtl.ResourceOptions.shared),
    };
    defer tp.h.deinit();
    defer tp.eids.deinit();
    var a = try runLeg(gpa, m, &s, tp, ids, default_chunks, draft);
    defer a.deinit(gpa);
    var b = try runLeg(gpa, m, &s, tp, ids, cut_chunks, draft);
    defer b.deinit(gpa);
    const differ = compare(m, &a, &b, T);
    std.debug.print("{{\"partition\": \"{s}\", \"prompt_tokens\": {d}, \"layers\": {d}, \"prefill_rows\": {d}, \"default_chunks\": {d}, \"cut_chunks\": {d}, \"first_cuts\": {any}, \"differ\": {d}}}\n", .{ if (differ == 0) "identical" else "differs", T, m.layers, rows, default_chunks.len, cut_chunks.len, cut_chunks[0..@min(cut_chunks.len, 16)], differ });
    if (differ != 0) return error.PartitionDiffers;
}
