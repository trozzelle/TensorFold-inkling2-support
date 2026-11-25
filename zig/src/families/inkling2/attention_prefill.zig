//! Inkling 2's attention for a prompt chunk on the tensor units (kernels/metal/inkling2/pattn.metal): the chunk's rows against the
//! layer's cache, 32 query rows of one head a threadgroup, over 64-key blocks anchored at absolute positions. Each row's relative
//! bias is built first into a table (`ik_pattn_bias`, scalar sums in index order, the same bits the one-row kernels form), which
//! `ik_pattn` reads. A row's bytes depend on that row alone: not on the rows beside it, the chunk's first position, the pieces
//! the table is built in, or where the chunk ends.
//!
//! The kernel reads whole 64-key blocks of the cache, so a cache holds a multiple of `key_block` positions (`state.cacheBytes`
//! pads it), and its keys and values are the layer's `[capacity][8][128]` halves, the values a half after the keys.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");

/// Default shapes: 48 query heads over 8 key heads of 128 dims, a relative state of 16 dims a head.
pub const heads = 48;
pub const kv_heads = 8;
pub const dim = 128;
pub const rel_dims = 16;
/// Backward tokens the relative bias reaches, which is also a local layer's window.
pub const extent = 1024;
/// Positions a cache holds a multiple of.
pub const key_block = 64;

/// ik_pattn's arguments: the call's rows, the first row's position in the sequence, 1 on a global layer, slots in the layer's cache.
pub const Args = extern struct { rows: u32, p0: u32, global: u32, ring: u32 };

comptime {
    std.debug.assert(@sizeOf(Args) == 16);
}

/// A buffer and the byte offset of a view in it.
pub const B = struct { buf: mtl.Buffer, off: usize = 0 };

/// A call's views at its first row: q [rows][q_heads][128], the layer's keys and values [capacity][kv][128],
/// sink [rows][q_heads], gate and out [rows][q_heads][128], all bf16.
pub const Io = struct { q: B, k: B, v: B, sink: B, gate: B, out: B };

/// The kernels' text with nax.h inlined for tensor units (`units`) or simdgroup matrices, parameterized for `q_heads`
/// query heads over `kv` key heads.
pub fn sourceFor(a: std.mem.Allocator, units: bool, q_heads: u32, kv: u32) ![]u8 {
    std.debug.assert(q_heads > 0 and kv > 0 and q_heads % kv == 0 and q_heads % 4 == 0);
    const base = try frags.precision.source(a, units, ks.inkling2_pattn, ks.nax, false);
    if (q_heads == heads and kv == kv_heads) return base;
    defer a.free(base);
    return std.fmt.allocPrint(a, "#define IK_PA_NQ {d}\n#define IK_PA_G {d}\n{s}", .{ q_heads, q_heads / kv, base });
}

/// The kernels' text with nax.h inlined for tensor units (`units`) or simdgroup matrices.
pub fn source(a: std.mem.Allocator, units: bool) ![]u8 {
    return sourceFor(a, units, heads, kv_heads);
}

/// The attention kernel and the table builder, in one library.
pub const Pipes = struct {
    lib: mtl.Library,
    attn: mtl.Pipeline,
    bias: mtl.Pipeline,
    heads: u32 = heads,
    kv_heads: u32 = kv_heads,

    /// The kernels for this GPU: on its tensor units when it has them.
    pub fn init(device: mtl.Device, a: std.mem.Allocator) !Pipes {
        return initFor(device, a, heads, kv_heads);
    }

    /// The kernels for this GPU parameterized for `q_heads` query heads over `kv` key heads.
    pub fn initFor(device: mtl.Device, a: std.mem.Allocator, q_heads: u32, kv: u32) !Pipes {
        return initUnitsFor(device, a, device.tensorUnits(), q_heads, kv);
    }

    /// The kernels on tensor units or on simdgroup matrices (any GPU compiles the latter).
    pub fn initUnits(device: mtl.Device, a: std.mem.Allocator, units: bool) !Pipes {
        return initUnitsFor(device, a, units, heads, kv_heads);
    }

    /// The kernels on tensor units or on simdgroup matrices, parameterized for `q_heads` query heads over `kv` key heads.
    pub fn initUnitsFor(device: mtl.Device, a: std.mem.Allocator, units: bool, q_heads: u32, kv: u32) !Pipes {
        const text = try sourceFor(a, units, q_heads, kv);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        errdefer lib.deinit();
        const attn = try mtl.Pipeline.init(device, lib, "ik_pattn", false);
        errdefer attn.deinit();
        const bias = try mtl.Pipeline.init(device, lib, "ik_pattn_bias", false);
        errdefer bias.deinit();
        // the kernels are written for 32-lane simdgroups and threadgroups of 64
        for ([_]mtl.Pipeline{ attn, bias }) |pipe| if (pipe.simdWidth() != 32 or pipe.maxThreads() < 64) return error.UnsupportedSimd;
        return .{ .lib = lib, .attn = attn, .bias = bias, .heads = q_heads, .kv_heads = kv };
    }

    pub fn deinit(p: *Pipes) void {
        p.attn.deinit();
        p.bias.deinit();
        p.lib.deinit();
    }
};

/// Bytes of the relative-bias table of `rows` rows for `q_heads` query heads: a bf16 for each head's 1,024 slots.
pub fn tableBytesFor(rows: usize, q_heads: u32) usize {
    return rows * @as(usize, q_heads) * extent * 2;
}

/// Bytes of the relative-bias table of `rows` rows: a bf16 for each of a row's 48 heads' 1,024 slots.
pub fn tableBytes(rows: usize) usize {
    return tableBytesFor(rows, heads);
}

/// One call of ik_pattn for `q_heads` query heads: a threadgroup of 64 threads for each 32 rows of each head.
pub fn encodeAttnFor(e: mtl.ComputeEncoder, pipe: mtl.Pipeline, q_heads: u32, a: Args, io: Io, table: B) void {
    std.debug.assert(a.rows > 0 and q_heads > 0);
    e.setPipeline(pipe);
    e.setValue(a, 0);
    const bs = [_]B{ io.q, io.k, io.v, io.sink, table, io.gate, io.out };
    for (bs, 1..) |x, i| e.setBuffer(x.buf, x.off, i);
    e.dispatchGroups(mtl.Size.of((a.rows + 31) / 32, q_heads, 1), mtl.Size.of(64, 1, 1));
}

/// One call of ik_pattn: a threadgroup of 64 threads for each 32 rows of each of the 48 heads. `table` is the rows' relative
/// bias at the call's first row (`encodeBias`'s).
pub fn encodeAttn(e: mtl.ComputeEncoder, pipe: mtl.Pipeline, a: Args, io: Io, table: B) void {
    encodeAttnFor(e, pipe, heads, a, io, table);
}

/// The relative-bias table of `a.rows` rows for `q_heads` query heads from their relative state `reln`
/// [rows][q_heads][16] and the weights' projection `proj` [16][1024] (both bf16), into `table`.
pub fn encodeBiasFor(e: mtl.ComputeEncoder, pipe: mtl.Pipeline, q_heads: u32, a: Args, reln: B, proj: B, table: B) void {
    std.debug.assert(a.rows > 0 and q_heads > 0 and q_heads % 4 == 0);
    std.debug.assert(table.off + tableBytesFor(a.rows, q_heads) <= table.buf.length());
    e.setPipeline(pipe);
    e.setValue(a, 0);
    for ([_]B{ reln, proj, table }, 1..) |x, i| e.setBuffer(x.buf, x.off, i);
    e.dispatchThreads(mtl.Size.of(extent / 4, q_heads / 4, a.rows), mtl.Size.of(64, 1, 1));
}

/// The relative-bias table of `a.rows` rows from their relative state `reln` [rows][48][16] and the weights' projection
/// `proj` [16][1024] (both bf16), into `table`.
pub fn encodeBias(e: mtl.ComputeEncoder, pipe: mtl.Pipeline, a: Args, reln: B, proj: B, table: B) void {
    encodeBiasFor(e, pipe, heads, a, reln, proj, table);
}

fn down(x: B, row_bytes: usize, rows: usize) B {
    return .{ .buf = x.buf, .off = x.off + rows * row_bytes };
}

/// A chunk's attention on one layer: its `a.rows` rows in pieces of `piece` rows (a multiple of 32), each piece's table built
/// into `table` (room for `tableBytesFor(piece, p.heads)` bytes, or for the whole chunk when it is the smaller) and read by
/// ik_pattn. Every view is at the chunk's first row. The encoder must be serial: the attention reads what the builder just
/// wrote. A row's bytes do not depend on `piece` or on `a.p0`.
pub fn encodeChunk(e: mtl.ComputeEncoder, p: *const Pipes, a: Args, io: Io, reln: B, proj: B, table: B, piece: usize) void {
    std.debug.assert(piece > 0 and piece % 32 == 0);
    const q_heads: usize = p.heads;
    const q_row = q_heads * dim;
    var at: usize = 0;
    while (at < a.rows) : (at += piece) {
        const sub: Args = .{ .rows = @intCast(@min(piece, a.rows - at)), .p0 = a.p0 + @as(u32, @intCast(at)), .global = a.global, .ring = a.ring };
        encodeBiasFor(e, p.bias, p.heads, sub, down(reln, q_heads * rel_dims * 2, at), proj, table);
        const part: Io = .{ .q = down(io.q, q_row * 2, at), .k = io.k, .v = io.v, .sink = down(io.sink, q_heads * 2, at), .gate = down(io.gate, q_row * 2, at), .out = down(io.out, q_row * 2, at) };
        encodeAttnFor(e, p.attn, p.heads, sub, part, table);
    }
}

test "attention prefill: the table is a bf16 a head and distance, and ik_pattn's arguments are four words" {
    try std.testing.expectEqual(@as(usize, 48 * 1024 * 2), tableBytes(1));
    try std.testing.expectEqual(@as(usize, 4096 * 48 * 1024 * 2), tableBytes(4096));
    try std.testing.expectEqual(@as(usize, 4096 * 24 * 1024 * 2), tableBytesFor(4096, 24));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Args));
}

test "attention prefill: sourceFor keeps defaults unchanged and defines overrides" {
    const a = std.testing.allocator;
    const def = try source(a, true);
    defer a.free(def);
    const same = try sourceFor(a, true, heads, kv_heads);
    defer a.free(same);
    try std.testing.expectEqualStrings(def, same);

    const g3 = try sourceFor(a, true, 24, 8);
    defer a.free(g3);
    const prefix = "#define IK_PA_NQ 24\n#define IK_PA_G 3\n";
    try std.testing.expect(std.mem.startsWith(u8, g3, prefix));
    try std.testing.expectEqualStrings(def, g3[prefix.len..]);
}
