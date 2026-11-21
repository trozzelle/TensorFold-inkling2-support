//! The draft head's loader on this Mac's GPU: a tiny mtp.safetensors written from the tiny header fixture, an absent file,
//! and the real one (TF_INKLING2_MTP_DIR=<model folder holding mtp.safetensors>).
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const config = @import("config.zig");
const config_test = @import("config_test.zig");
const dw = @import("dspark_weights.zig");
const mix = @import("dspark_alpha.zig");
const quant = @import("dspark_quant.zig");

const t = std.testing;

/// Element `j` of the `i`th tensor of the tiny file: small multiples of 1/16, and fuse_weight's own values.
fn value(i: usize, j: usize) u16 {
    const k: i64 = @intCast((i * 31 + j * 7) % 97);
    return mix.bf16Of(@as(f32, @floatFromInt(k - 48)) / 16);
}

const Entry = struct { name: []const u8, dims: []const usize };

/// The fixture's tensors in file order: its `{L}` lines over their layer ranges (names_test.unfold's format, which
/// checks the same lines against a digest).
fn fixture(a: std.mem.Allocator, text: []const u8) ![]Entry {
    var out: std.ArrayList(Entry) = .empty;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    _ = lines.next();
    while (lines.next()) |line| {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const pat = f.next().?;
        _ = f.next();
        var dims: std.ArrayList(usize) = .empty;
        var ds = std.mem.tokenizeScalar(u8, f.next().?, 'x');
        while (ds.next()) |d| try dims.append(a, try std.fmt.parseInt(usize, d, 10));
        const layers = f.next() orelse {
            try out.append(a, .{ .name = pat, .dims = dims.items });
            continue;
        };
        var spans = std.mem.tokenizeScalar(u8, layers, ',');
        while (spans.next()) |span| {
            const dash = std.mem.indexOfScalar(u8, span, '-');
            const lo = try std.fmt.parseInt(u32, if (dash) |d| span[0..d] else span, 10);
            const hi = if (dash) |d| try std.fmt.parseInt(u32, span[d + 1 ..], 10) else lo;
            for (lo..hi + 1) |l| {
                var b: [8]u8 = undefined;
                const named = try std.mem.replaceOwned(u8, a, pat, "{L}", try std.fmt.bufPrint(&b, "{d}", .{l}));
                try out.append(a, .{ .name = named, .dims = dims.items });
            }
        }
    }
    return out.items;
}

fn indexOf(entries: []const Entry, name: []const u8) usize {
    for (entries, 0..) |e, i| if (std.mem.eql(u8, e.name, name)) return i;
    unreachable;
}

fn count(e: Entry) usize {
    var n: usize = 1;
    for (e.dims) |d| n *= d;
    return n;
}

/// An element of the file written as other bits than `value(i, j)`.
const Poke = struct { entry: usize, element: usize, bits: u16 };

/// A safetensors file of `entries` in `tmp` as mtp.safetensors, tensor `i` holding `value(i, j)` (but for `poke`).
fn write(a: std.mem.Allocator, tmp: *std.testing.TmpDir, entries: []const Entry, poke: ?Poke) !void {
    var json: std.ArrayList(u8) = .empty;
    try json.append(a, '{');
    var bytes: usize = 0;
    for (entries, 0..) |e, i| {
        try json.print(a, "{s}\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[", .{ if (i > 0) "," else "", e.name });
        for (e.dims, 0..) |d, k| try json.print(a, "{s}{d}", .{ if (k > 0) "," else "", d });
        try json.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ bytes, bytes + 2 * count(e) });
        bytes += 2 * count(e);
    }
    try json.append(a, '}');
    while (json.items.len % 8 != 0) try json.append(a, ' ');
    const file = try a.alloc(u8, 8 + json.items.len + bytes);
    std.mem.writeInt(u64, file[0..8], json.items.len, .little);
    @memcpy(file[8..][0..json.items.len], json.items);
    var at: usize = 8 + json.items.len;
    for (entries, 0..) |e, i| {
        for (0..count(e)) |j| {
            const bits = if (poke) |p| (if (p.entry == i and p.element == j) p.bits else value(i, j)) else value(i, j);
            std.mem.writeInt(u16, file[at + 2 * j ..][0..2], bits, .little);
        }
        at += 2 * count(e);
    }
    try tmp.dir.writeFile(t.io, .{ .sub_path = "mtp.safetensors", .data = file });
}

fn dirOf(a: std.mem.Allocator, tmp: *const std.testing.TmpDir) ![]u8 {
    return std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
}

test "inkling2 draft weights: a tiny mtp.safetensors loads with every shape checked and alpha from fuse_weight" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: config.Why = .{};
    const c = try config.parse(a, config_test.tiny, &why);
    const m = c.mtp.?;
    const entries = try fixture(a, @embedFile("fixtures/tiny_mtp_header.txt"));
    try t.expectEqual(@as(usize, 84), entries.len);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(a, &tmp, entries, null);
    var ckp = ck.Checkpoint.init(t.allocator);
    defer ckp.deinit();
    var w = (try dw.load(t.allocator, t.io, device, &ckp, try dirOf(a, &tmp), m, .sixteen)).?;
    defer w.deinit(t.allocator);

    try t.expectEqual(@as(usize, m.layers), w.layers.len);
    try t.expectEqual(@as(u32, m.layers + 1), w.rows);
    try t.expectEqual(@as(u32, m.taps_n), w.taps);
    const l = w.layers[4];
    try t.expectEqual(@as(u32, 8 * 128), l.q.n());
    try t.expectEqual(@as(u32, 128), l.q.k());
    try t.expectEqual(@as(u32, 4 * 128), l.v_ctx.n());
    try t.expectEqual(@as(u32, 128), l.o.n());
    try t.expectEqual(@as(u32, 8 * 128), l.o.k());
    try t.expectEqual(@as(u32, 128), l.down.k());
    try t.expectEqual(@as(u32, m.vocab), w.markov.w1.n);
    try t.expectEqual(@as(u32, m.rank), w.markov.w1.k);
    try t.expectEqual(@as(u32, 3 * m.rank), w.markov.rnn.n);
    try t.expectEqual(@as(u32, 2 * m.rank + 2 * 128), w.markov.rnn.k);
    try t.expectEqual(@as(u32, 128), w.markov.hidden_out.n);

    // tensors are the file's bytes, not copies of something else
    const mask = indexOf(entries, "mask_embed");
    for (w.mask_embed.host(u16), 0..) |x, j| try t.expectEqual(value(mask, j), x);
    const k_ctx = indexOf(entries, "layers.3.self_attn.k_proj_ctx.weight");
    try t.expectEqual(value(k_ctx, 5), w.layers[3].k_ctx.bf16.t.host(u16)[5]);
    const w1 = indexOf(entries, "markov_head.markov_w1.weight");
    try t.expectEqual(value(w1, 3 * m.rank + 2), w.markov.w1.t.host(u16)[3 * m.rank + 2]);

    // alpha: row r of fuse_weight's softmax, as a host array and as the packed buffer
    const fuse = try a.alloc(u16, w.rows * w.taps);
    const fi = indexOf(entries, "fuse_weight");
    for (fuse, 0..) |*x, j| x.* = value(fi, j);
    var want: [mix.max_rows][mix.max_taps]u16 = undefined;
    mix.softmaxRows(fuse, w.rows, w.taps, &want);
    const packed_alpha = w.alpha_buffer.slice(u16, w.rows * w.taps);
    for (0..w.rows) |r| for (0..w.taps) |j| {
        try t.expectEqual(want[r][j], w.alpha[r][j]);
        try t.expectEqual(want[r][j], packed_alpha[r * w.taps + j]);
    };
    try t.expect(w.fold_w == null and w.fold_b == null);
    try t.expectEqual(dw.Bits.sixteen, w.bits);
    // the file's shard is all the checkpoint holds, and the weights record its bytes for the budget
    try t.expect(w.bytes > 0);
    try t.expectEqual(ckp.residentBytes(), w.bytes);
}

test "inkling2 draft weights: a model folder without mtp.safetensors loads no draft head" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: config.Why = .{};
    const c = try config.parse(a, config_test.tiny, &why);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var ckp = ck.Checkpoint.init(t.allocator);
    defer ckp.deinit();
    try t.expect((try dw.load(t.allocator, t.io, device, &ckp, try dirOf(a, &tmp), c.mtp.?, .eight)) == null);
    try t.expectEqual(@as(usize, 0), ckp.residentBytes());
}

test "inkling2 draft weights: the real mtp.safetensors loads, with fuse_weight's softmax as measured (TF_INKLING2_MTP_DIR)" {
    const dir = std.mem.span(std.c.getenv("TF_INKLING2_MTP_DIR") orelse return error.SkipZigTest);
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, config_test.real, &why);
    var ckp = ck.Checkpoint.init(t.allocator);
    defer ckp.deinit();
    var w = (try dw.load(t.allocator, t.io, device, &ckp, dir, c.mtp.?, .sixteen)) orelse return error.TestUnexpectedResult;
    defer w.deinit(t.allocator);
    try t.expectEqual(@as(usize, 5), w.layers.len);
    try t.expectEqual(@as(usize, 3_423_420_768), ckp.residentBytes());
    try t.expectEqual(@as(u32, 6144), w.layers[0].q.n());
    try t.expectEqual(@as(u32, 18432), w.layers[4].gate.n());
    try t.expectEqual(@as(u32, 201024), w.markov.w1.n);
    try t.expectEqual(@as(u32, 9728), w.markov.rnn.k);
    // rows 0-4 are draft layers, row 5 the Markov head (values to four places)
    const rows = [6][8]f32{
        .{ 0.0442, 0.0432, 0.042, 0.0337, 0.1299, 0.4863, 0.0791, 0.1416 },
        .{ 0.0076, 0.0186, 0.0256, 0.0442, 0.3574, 0.5469, 0.0003, 0.0002 },
        .{ 0.1167, 0.1079, 0.0442, 0.1084, 0.3223, 0.2695, 0.0027, 0.0288 },
        .{ 0.054, 0.0176, 0.0238, 0.0262, 0.1094, 0.3457, 0.3555, 0.0679 },
        .{ 0.1514, 0.0776, 0.0144, 0.0515, 0.1011, 0.127, 0.0004, 0.4766 },
        .{ 0.0669, 0.1289, 0.0815, 0.0557, 0.1553, 0.4043, 0.0605, 0.0466 },
    };
    for (rows, 0..) |row, r| for (row, 0..) |want, j| {
        try t.expectApproxEqAbs(want, mix.f32Of(w.alpha[r][j]), 6e-5);
    };
}

test "inkling2 draft weights: the real mtp.safetensors at 8 bits holds the layers as codes, scales and biases (TF_INKLING2_MTP_DIR)" {
    const dir = std.mem.span(std.c.getenv("TF_INKLING2_MTP_DIR") orelse return error.SkipZigTest);
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, config_test.real, &why);
    var ckp = ck.Checkpoint.init(t.allocator);
    defer ckp.deinit();
    var w = (try dw.load(t.allocator, t.io, device, &ckp, dir, c.mtp.?, .eight)) orelse return error.TestUnexpectedResult;
    defer w.deinit(t.allocator);
    // the five layers' 1,651,507,200 weights at 1.0625 bytes (a bf16 scale and bias a group of 64), and the file's other tensors
    const layer_bytes: usize = 1_651_507_200 * 17 / 16;
    const other_bytes: usize = 3_423_420_768 - 1_651_507_200 * 2;
    try t.expectEqual(w.bytes, ckp.residentBytes());
    try t.expect(w.bytes >= layer_bytes + other_bytes and w.bytes < layer_bytes + other_bytes + (1 << 20));
    std.debug.print("draft file at 8 bits: {d} bytes resident ({d:.3} GiB) against 3,423,420,768 at bf16\n", .{ w.bytes, @as(f64, @floatFromInt(w.bytes)) / (1 << 30) });
    for (w.layers) |l| inline for (.{ l.q, l.k, l.v, l.k_ctx, l.v_ctx, l.o, l.gate, l.up, l.down }) |p| try t.expect(p == .q8);
}

test "inkling2 draft weights: the setting's text is 8 or 16 and unset or blank means 8" {
    for ([_]?[]const u8{ null, "", " \t" }) |text| try t.expectEqual(dw.Bits.eight, try dw.Bits.parse(text));
    try t.expectEqual(dw.Bits.eight, try dw.Bits.parse("8"));
    try t.expectEqual(dw.Bits.sixteen, try dw.Bits.parse(" 16 "));
    for ([_][]const u8{ "0", "4", "bf16", "08", "16.0", "8 bit", "-8", "32" }) |text| try t.expectError(error.BadDraftBits, dw.Bits.parse(text));
    try t.expectEqualStrings("8-bit", dw.Bits.eight.label());
    try t.expectEqualStrings("bf16", dw.Bits.sixteen.label());
}

test "inkling2 draft weights: at 8 bits a tiny file's nine projections a layer are quantized buffers equal to the quantizer's, the bf16 never resident" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: config.Why = .{};
    const c = try config.parse(a, config_test.tiny, &why);
    const m = c.mtp.?;
    const entries = try fixture(a, @embedFile("fixtures/tiny_mtp_header.txt"));
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(a, &tmp, entries, null);
    var ckp = ck.Checkpoint.init(t.allocator);
    defer ckp.deinit();
    var w = (try dw.load(t.allocator, t.io, device, &ckp, try dirOf(a, &tmp), m, .eight)).?;
    defer w.deinit(t.allocator);
    try t.expectEqual(dw.Bits.eight, w.bits);

    // every projection is a quantized buffer of its shape, and the checkpoint reads the file no more
    var quant_bytes: usize = 0;
    for (w.layers) |l| inline for (.{ "q", "k", "v", "k_ctx", "v_ctx", "o", "gate", "up", "down" }) |name| {
        const p = @field(l, name);
        try t.expect(p == .q8);
        quant_bytes += dw.Quant.bytes(p.n(), p.k());
        try t.expectEqual(@as(u32, 0), p.k() % 64);
    };
    for (ckp.files.items) |f| try t.expect(f.fd < 0);

    // the codes, scales and biases are the quantizer's over the file's values
    const proj = [_]struct { name: []const u8, layer: usize, which: []const u8 }{
        .{ .name = "layers.3.self_attn.k_proj_ctx.weight", .layer = 3, .which = "k_ctx" },
        .{ .name = "layers.0.mlp.down_proj.weight", .layer = 0, .which = "down" },
        .{ .name = "layers.4.self_attn.q_proj.weight", .layer = 4, .which = "q" },
    };
    inline for (proj) |pr| {
        const e = entries[indexOf(entries, pr.name)];
        const i = indexOf(entries, pr.name);
        const n = e.dims[0];
        const k = e.dims[1];
        const src = try a.alloc(u16, n * k);
        for (src, 0..) |*x, j| x.* = value(i, j);
        const codes = try a.alloc(u8, n * k);
        const scales = try a.alloc(u16, n * k / 64);
        const biases = try a.alloc(u16, n * k / 64);
        try t.expectEqual(@as(?usize, null), quant.quantizeRows(src, k, codes, scales, biases));
        const q = @field(w.layers[pr.layer], pr.which).q8;
        try t.expectEqual(@as(u32, @intCast(n)), q.n);
        try t.expectEqual(@as(u32, @intCast(k)), q.k);
        const mem = q.buf.contents();
        try t.expectEqualSlices(u8, codes, mem[0 .. n * k]);
        try t.expectEqualSlices(u16, scales, @as([*]const u16, @ptrCast(@alignCast(mem + q.scales)))[0 .. n * k / 64]);
        try t.expectEqualSlices(u16, biases, @as([*]const u16, @ptrCast(@alignCast(mem + q.biases)))[0 .. n * k / 64]);
        // the codes of a row start 16-byte aligned, as the tiles read them
        try t.expectEqual(@as(usize, 0), q.scales % 256);
        try t.expectEqual(@as(usize, 0), q.biases % 256);
    }

    // the file's other tensors are the checkpoint's one shard, apart from the quantized buffers: the bf16 projections are not in it
    var shard: usize = 0;
    for (entries) |e| {
        const is_proj = e.dims.len == 2 and std.mem.startsWith(u8, e.name, "layers.");
        if (is_proj) continue;
        shard = std.mem.alignForward(usize, shard, ck.alignment) + 2 * count(e);
    }
    try t.expectEqual(shard + quant_bytes, w.bytes);
    try t.expectEqual(ckp.residentBytes(), w.bytes);
    var bf16_total: usize = 0;
    for (entries) |e| bf16_total += 2 * count(e);
    try t.expect(w.bytes < bf16_total);

    // the kept tensors are the file's values still
    const mask = indexOf(entries, "mask_embed");
    for (w.mask_embed.host(u16), 0..) |x, j| try t.expectEqual(value(mask, j), x);
    const w1 = indexOf(entries, "markov_head.markov_w1.weight");
    try t.expectEqual(value(w1, 3 * m.rank + 2), w.markov.w1.t.host(u16)[3 * m.rank + 2]);
    const fuse = try a.alloc(u16, w.rows * w.taps);
    for (fuse, 0..) |*x, j| x.* = value(indexOf(entries, "fuse_weight"), j);
    var want: [mix.max_rows][mix.max_taps]u16 = undefined;
    mix.softmaxRows(fuse, w.rows, w.taps, &want);
    for (0..w.rows) |r| for (0..w.taps) |j| try t.expectEqual(want[r][j], w.alpha[r][j]);
}

test "inkling2 draft weights: a projection with a value that is not finite refuses the load at 8 bits and loads at bf16" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    const log_level = t.log_level;
    t.log_level = .err; // the refusal's warning names the group; the test expects it
    defer t.log_level = log_level;
    defer device.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: config.Why = .{};
    const c = try config.parse(a, config_test.tiny, &why);
    const m = c.mtp.?;
    const entries = try fixture(a, @embedFile("fixtures/tiny_mtp_header.txt"));
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(a, &tmp, entries, .{ .entry = indexOf(entries, "layers.2.mlp.up_proj.weight"), .element = 100, .bits = 0x7f80 });
    var bad = ck.Checkpoint.init(t.allocator);
    defer bad.deinit();
    try t.expectError(error.NonFiniteDraftWeight, dw.load(t.allocator, t.io, device, &bad, try dirOf(a, &tmp), m, .eight));
    // at bf16 the file's bytes are the weights, whatever they are
    var ok = ck.Checkpoint.init(t.allocator);
    defer ok.deinit();
    var w = (try dw.load(t.allocator, t.io, device, &ok, try dirOf(a, &tmp), m, .sixteen)).?;
    w.deinit(t.allocator);
}
