//! The Inkling 2 kernel checks' shared pieces: one-kernel GPU runs, bf16 rounding, seeded fills, tallies, and the
//! row-independence and NaN-containment check for row kernels.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const tf = @import("tensorfold");
pub const Args = tf.inkling2.forward.Args;
pub const state = tf.inkling2.state;

pub const Bind = struct { buf: mtl.Buffer, off: usize = 0 };

pub const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    lib: mtl.Library,

    pub fn init() !Gpu {
        const device = try mtl.Device.init();
        const queue = try device.queue();
        const lib = try mtl.Library.fromSource(device, ks.inkling2_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        return .{ .device = device, .queue = queue, .lib = lib };
    }

    pub fn buf(g: *const Gpu, bytes: usize) !mtl.Buffer {
        return g.device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared);
    }

    /// One kernel: args at buffer 0, `binds` from buffer 1; `groups` dispatches threadgroups, else threads.
    pub fn run(g: *const Gpu, name: [:0]const u8, a: Args, binds: []const Bind, grid: [3]usize, group: [3]usize, groups: bool) !void {
        const p = try mtl.Pipeline.init(g.device, g.lib, name, false);
        defer p.deinit();
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(p);
        e.setValue(a, 0);
        for (binds, 1..) |b, i| e.setBuffer(b.buf, b.off, i);
        const g3 = mtl.Size.of(grid[0], grid[1], grid[2]);
        const t3 = mtl.Size.of(group[0], group[1], group[2]);
        if (groups) e.dispatchGroups(g3, t3) else e.dispatchThreads(g3, t3);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("{s}: {s}", .{ name, msg });
            return error.GpuFailed;
        }
    }
};

/// bf16 of x through f32, round to nearest even (Metal's bfloat(float)).
pub fn bf(x: f64) u16 {
    const f: f32 = @floatCast(x);
    if (std.math.isNan(f)) return 0x7fc0;
    const u: u32 = @bitCast(f);
    return @intCast((u +% (0x7fff + ((u >> 16) & 1))) >> 16);
}
pub fn fb(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}
/// A bf16 rounding point in a reference.
pub fn rb(x: f64) f64 {
    return fb(bf(x));
}
fn key(h: u16) i32 {
    return if (h & 0x8000 != 0) -@as(i32, h & 0x7fff) else @as(i32, h);
}
pub fn ulps(a: u16, b: u16) u32 {
    return @abs(key(a) - key(b));
}

pub fn fill16(b: mtl.Buffer, n: usize, r: std.Random, scale: f64) []u16 {
    const s = b.slice(u16, n);
    for (s) |*x| x.* = bf((r.float(f64) * 2 - 1) * scale);
    return s;
}
pub fn fill32(b: mtl.Buffer, n: usize, r: std.Random, scale: f64) []f32 {
    const s = b.slice(f32, n);
    for (s) |*x| x.* = @floatCast((r.float(f64) * 2 - 1) * scale);
    return s;
}

pub const Check = struct {
    name: []const u8,
    n: usize = 0,
    bad: usize = 0,
    worst: f64 = 0,
    first: ?[2]f64 = null, // the first failing output and its reference, printed on failure

    fn miss(c: *Check, got: f64, want: f64) void {
        c.bad += 1;
        if (c.first == null) c.first = .{ got, want };
    }
    /// A bf16 output within one bf16 step of the reference rounded to bf16.
    pub fn b16(c: *Check, got: u16, want: f64) void {
        c.b16n(got, want, 1);
    }
    /// A bf16 output that passes through `n` bf16 rounding points: within n steps of the reference.
    pub fn b16n(c: *Check, got: u16, want: f64, n: u32) void {
        c.n += 1;
        const d = ulps(got, bf(want));
        c.worst = @max(c.worst, @as(f64, @floatFromInt(d)));
        if (d > n) c.miss(fb(got), want);
    }
    /// b16n, also passing when the output is within `abs` of the reference: a sum that cancels toward zero keeps its
    /// absolute error, which is many bf16 steps of a value near zero.
    pub fn b16a(c: *Check, got: u16, want: f64, n: u32, abs: f64) void {
        c.n += 1;
        const d = ulps(got, bf(want));
        if (@abs(fb(got) - want) > abs) c.worst = @max(c.worst, @as(f64, @floatFromInt(d)));
        if (d > n and @abs(fb(got) - want) > abs) c.miss(fb(got), want);
    }
    /// An f32 output within a relative `rel` of the reference.
    pub fn f(c: *Check, got: f32, want: f64, rel: f64) void {
        c.n += 1;
        const e = @abs(@as(f64, got) - want) / @max(@abs(want), 1e-6);
        c.worst = @max(c.worst, e);
        if (e > rel) c.miss(got, want);
    }
    /// An f32 output within `tol` of the reference relative to max(|want|, mag): sums that cancel toward zero keep
    /// their absolute error, so they are judged against the outputs' magnitude.
    pub fn near(c: *Check, got: f32, want: f64, tol: f64, mag: f64) void {
        c.n += 1;
        const e = @abs(@as(f64, got) - want) / @max(@abs(want), mag);
        c.worst = @max(c.worst, e);
        if (e > tol) c.miss(got, want);
    }
    pub fn same(c: *Check, a: []const u8, b: []const u8) void {
        c.n += 1;
        if (!std.mem.eql(u8, a, b)) c.miss(0, 0);
    }
    pub fn done(c: *const Check) !void {
        std.debug.print("{s} {s}: {d} checks, worst {e:.3}, {d} bad\n", .{ if (c.bad == 0) "PASS" else "FAIL", c.name, c.n, c.worst, c.bad });
        if (c.first) |x| std.debug.print("  first: got {e:.6}, want {e:.6}\n", .{ x[0], x[1] });
        if (c.bad != 0) return error.KernelMismatch;
    }
};

/// Row independence and containment for a row kernel `name` whose input `x` (row bytes `xrow`, bf16), extra inputs and
/// output `y` (row bytes `yrow`) are row-strided: each row alone gives the batch's bits; a NaN row 1 stays in row 1.
/// `groups`: one 1024-thread threadgroup a row; else threads over (a.width, rows) in groups of 256.
pub fn independent(g: *const Gpu, name: [:0]const u8, a: Args, x: mtl.Buffer, xrow: usize, extra: []const mtl.Buffer, y: mtl.Buffer, yrow: usize, rows: usize, groups: bool) !void {
    var c: Check = .{ .name = name };
    const all = try std.heap.page_allocator.dupe(u8, y.slice(u8, rows * yrow));
    defer std.heap.page_allocator.free(all);
    for (0..rows) |t| {
        var one = a;
        one.rows = 1;
        var binds: [8]Bind = undefined;
        binds[0] = .{ .buf = x, .off = t * xrow };
        for (extra, 1..) |e, i| binds[i] = .{ .buf = e };
        binds[extra.len + 1] = .{ .buf = y, .off = t * yrow };
        try g.run(name, one, binds[0 .. extra.len + 2], if (groups) .{ 1, 1, 1 } else .{ a.width, 1, 1 }, if (groups) .{ 1024, 1, 1 } else .{ 256, 1, 1 }, groups);
        c.same(y.slice(u8, rows * yrow)[t * yrow ..][0..yrow], all[t * yrow ..][0..yrow]);
    }
    const xs = x.slice(u16, rows * xrow / 2);
    const kept = xs[xrow / 2];
    xs[xrow / 2] = 0x7fc0; // NaN in row 1
    var binds: [8]Bind = undefined;
    binds[0] = .{ .buf = x };
    for (extra, 1..) |e, i| binds[i] = .{ .buf = e };
    binds[extra.len + 1] = .{ .buf = y };
    try g.run(name, a, binds[0 .. extra.len + 2], if (groups) .{ rows, 1, 1 } else .{ a.width, rows, 1 }, if (groups) .{ 1024, 1, 1 } else .{ 256, 1, 1 }, groups);
    for (0..rows) |t| if (t != 1) c.same(y.slice(u8, rows * yrow)[t * yrow ..][0..yrow], all[t * yrow ..][0..yrow]);
    xs[xrow / 2] = kept;
    try c.done();
}
