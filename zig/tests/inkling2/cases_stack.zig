//! Inkling 2 kernel checks: the segmented matvec (ik_mvs_8, ik_mvs_16) against ik_matvec_8/16 run on each segment
//! alone, byte for byte, with x, w and y reached only through GPU addresses declared on the encoder.
const std = @import("std");
const mtl = @import("metal");
const h = @import("harness.zig");

const K = 4608;
const MAXR = 16;
const NS = 6;
/// A local layer's attention inputs: q, k, v, r, gate and sink outputs, all over k = 4608. The sink's rows are padded.
const ns = [NS]u32{ 6144, 1024, 1024, 768, 6144, 48 };
const ldys = [NS]u32{ 6144, 1024, 1024, 768, 6144, 64 };
const sentinel: u16 = 0xaaaa;

const Seg = extern struct { x: u64, w: u64, y: u64, n: u32, k: u32, ldy: u32, first: u32 };
const MvsArgs = extern struct { rows: u32, nseg: u32, total: u32 };
const MvArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };

comptime {
    std.debug.assert(@sizeOf(Seg) == 40);
}

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    try caseStack(g, r);
}

const Run = struct {
    table: mtl.Buffer,
    xs: [NS]mtl.Buffer, // the input of each segment
    w: [NS]mtl.Buffer,
    y: [NS]mtl.Buffer,
};

/// One ik_mvs dispatch over the six segments starting at input row/output row `row`, `rows` rows. Its x, w and y buffers
/// are declared with useResource; the segment table is bound at slot 0.
fn mvs(g: *const h.Gpu, name: [:0]const u8, t: Run, row: usize, rows: u32) !void {
    const segs = t.table.slice(Seg, NS);
    var first: u32 = 0;
    var uses: [3 * NS]h.Use = undefined;
    for (0..NS) |s| {
        segs[s] = .{
            .x = t.xs[s].gpuAddress() + row * K * 2,
            .w = t.w[s].gpuAddress(),
            .y = t.y[s].gpuAddress() + row * ldys[s] * 2,
            .n = ns[s],
            .k = K,
            .ldy = ldys[s],
            .first = first,
        };
        first += ns[s];
        uses[3 * s] = .{ .buf = t.xs[s] };
        uses[3 * s + 1] = .{ .buf = t.w[s] };
        uses[3 * s + 2] = .{ .buf = t.y[s], .write = true };
    }
    const a = MvsArgs{ .rows = rows, .nseg = NS, .total = first };
    try g.dispatch(name, &.{ .{ .buf = .{ .buf = t.table } }, .{ .bytes = std.mem.asBytes(&a) } }, &uses, .{ (first + 7) / 8, 1, 1 }, .{ 256, 1, 1 });
}

/// ik_matvec_8 (rows <= 8) or ik_matvec_16 on one segment alone, with the segment's own ldy.
fn matvec(g: *const h.Gpu, x: mtl.Buffer, w: mtl.Buffer, y: mtl.Buffer, s: usize, rows: u32) !void {
    const a = MvArgs{ .rows = rows, .n = ns[s], .k = K, .ldy = ldys[s] };
    try g.dispatch(if (rows <= 8) "ik_matvec_8" else "ik_matvec_16", &.{ .{ .buf = .{ .buf = x } }, .{ .buf = .{ .buf = w } }, .{ .bytes = std.mem.asBytes(&a) }, .{ .buf = .{ .buf = y } } }, &.{}, .{ (ns[s] + 7) / 8, 1, 1 }, .{ 256, 1, 1 });
}

fn clear(ys: []const mtl.Buffer) void {
    for (ys, 0..) |b, s| @memset(b.slice(u16, MAXR * ldys[s]), sentinel);
}

fn caseStack(g: *const h.Gpu, r: std.Random) !void {
    var t: Run = undefined;
    var xb: [3]mtl.Buffer = undefined; // three distinct inputs of MAXR rows
    var ref: [NS]mtl.Buffer = undefined; // ik_matvec on each segment alone
    var one: [NS]mtl.Buffer = undefined; // the batch's rows run one at a time
    var nan: [NS]mtl.Buffer = undefined; // the batch with a NaN row 1 in the shared input
    t.table = try g.buf(NS * @sizeOf(Seg));
    for (&xb) |*b| {
        b.* = try g.buf(MAXR * K * 2);
        _ = h.fill16(b.*, MAXR * K, r, 1);
    }
    for (0..NS) |s| {
        t.w[s] = try g.buf(@as(usize, ns[s]) * K * 2);
        _ = h.fill16(t.w[s], @as(usize, ns[s]) * K, r, 0.05);
        t.y[s] = try g.buf(MAXR * ldys[s] * 2);
        ref[s] = try g.buf(MAXR * ldys[s] * 2);
        one[s] = try g.buf(MAXR * ldys[s] * 2);
        nan[s] = try g.buf(MAXR * ldys[s] * 2);
    }
    defer {
        t.table.deinit();
        for (xb) |b| b.deinit();
        for (0..NS) |s| for ([_]mtl.Buffer{ t.w[s], t.y[s], ref[s], one[s], nan[s] }) |b| b.deinit();
    }

    var c: h.Check = .{ .name = "mvs" };
    // Config 0: every segment reads one input (a local layer). Config 1: q, k and v read three different inputs, and
    // the other three segments read them again (a global layer).
    const picks = [2][NS]usize{ .{ 0, 0, 0, 0, 0, 0 }, .{ 0, 1, 2, 0, 1, 2 } };
    for (picks, 0..) |pick, cfg| {
        for (0..NS) |s| t.xs[s] = xb[pick[s]];
        for ([_]u32{ 1, 2, 5, 8, 9, 16 }) |rows| {
            const name: [:0]const u8 = if (rows <= 8) "ik_mvs_8" else "ik_mvs_16";
            clear(&ref);
            clear(&t.y);
            clear(&one);
            for (0..NS) |s| try matvec(g, t.xs[s], t.w[s], ref[s], s, rows);
            try mvs(g, name, t, 0, rows);
            for (0..NS) |s| c.same(t.y[s].slice(u8, MAXR * ldys[s] * 2), ref[s].slice(u8, MAXR * ldys[s] * 2));
            // Row independence: each row alone through ik_mvs_8 gives the batch's bits (and writes nothing else).
            var o = t;
            o.y = one;
            for (0..rows) |row| try mvs(g, "ik_mvs_8", o, row, 1);
            for (0..NS) |s| c.same(one[s].slice(u8, MAXR * ldys[s] * 2), t.y[s].slice(u8, MAXR * ldys[s] * 2));
            // Containment: a NaN in row 1 of the shared input stays in row 1.
            if (cfg == 0 and rows == MAXR) {
                const x1 = xb[0].slice(u16, MAXR * K)[K..][0..K];
                var kept: [K]u16 = undefined;
                @memcpy(&kept, x1);
                @memset(x1, 0x7fc0);
                clear(&nan);
                var q = t;
                q.y = nan;
                try mvs(g, name, q, 0, rows);
                @memcpy(x1, &kept);
                for (0..NS) |s| {
                    const want = t.y[s].slice(u8, MAXR * ldys[s] * 2);
                    const got = nan[s].slice(u8, MAXR * ldys[s] * 2);
                    const row = ldys[s] * 2;
                    for (0..MAXR) |row_i| if (row_i != 1) c.same(got[row_i * row ..][0..row], want[row_i * row ..][0..row]);
                }
            }
        }
    }
    try c.done();
}
