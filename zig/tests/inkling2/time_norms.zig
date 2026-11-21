//! Inkling 2 kernel timing, run as `tf-inkling-kernels time`: the fused norm kernels' cost a dispatch, for the block counts and
//! query counts of the real layers. Each case encodes `reps` dependent dispatches of one kernel into one command buffer and
//! reports the least GPU time over a few trials, in microseconds a dispatch.
const std = @import("std");
const mtl = @import("metal");
const h = @import("harness.zig");

const H = 4608;
const stride = 512; // the scratch's block rows apart
const reps = 200;

fn scale() f32 {
    return @floatCast(1 / @sqrt(@as(f64, H)));
}

fn micros(g: *const h.Gpu, name: [:0]const u8, a: h.Args, binds: []const h.Bind) !f64 {
    const p = try mtl.Pipeline.init(g.device, g.lib, name, false);
    defer p.deinit();
    var best: f64 = 1e9;
    for (0..6) |_| {
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(p);
        e.setValue(a, 0);
        for (binds, 1..) |b, i| e.setBuffer(b.buf, b.off, i);
        for (0..reps) |_| e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(1024, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        best = @min(best, cb.gpuSeconds());
    }
    return best / reps * 1e6;
}

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    const blocks = try g.buf(9 * stride * H * 4);
    const hb = try g.buf(H * 2);
    const ya = try g.buf(H * 2);
    const query = try g.buf(3 * H * 4);
    const norms = try g.buf(3 * H * 2);
    const nrd = try g.buf(H * 4);
    const xin = try g.buf(3 * H * 2);
    const res = try g.buf(H * 4);
    defer for ([_]mtl.Buffer{ blocks, hb, ya, query, norms, nrd, xin, res }) |b| b.deinit();
    _ = h.fill32(blocks, 9 * stride * H, r, 1.0);
    _ = h.fill16(hb, H, r, 3.0);
    _ = h.fill16(ya, H, r, 3.0);
    _ = h.fill32(query, 3 * H, r, 0.2);
    _ = h.fill16(norms, 3 * H, r, 1.5);
    const rdn_binds = [_]h.Bind{ .{ .buf = hb }, .{ .buf = blocks }, .{ .buf = nrd }, .{ .buf = query }, .{ .buf = norms }, .{ .buf = norms, .off = H * 2 }, .{ .buf = norms, .off = 2 * H * 2 }, .{ .buf = xin } };
    for ([_][2]u32{ .{ 0, 0 }, .{ 1, 8 }, .{ 3, 8 }, .{ 1, 2 }, .{ 3, 2 } }) |qk| {
        const a: h.Args = .{ .rows = 1, .width = H, .queries = qk[0], .blocks = qk[1], .stride = stride, .eps = 1e-5, .scale = scale() };
        std.debug.print("ik_rdn{s} Q {d} K {d}: {d:.2} us\n", .{ if (qk[0] > 1) "3" else "", qk[0], qk[1], try micros(g, if (qk[0] > 1) "ik_rdn3" else "ik_rdn", a, &rdn_binds) });
    }
    const rrdn_binds = [_]h.Bind{ .{ .buf = hb }, .{ .buf = ya }, .{ .buf = res }, .{ .buf = blocks }, .{ .buf = nrd }, .{ .buf = query }, .{ .buf = norms }, .{ .buf = xin } };
    for ([_][2]u32{ .{ 0, 1 }, .{ 7, 1 }, .{ 8, 0 }, .{ 2, 0 } }) |kb| {
        const a: h.Args = .{ .rows = 1, .width = H, .blocks = kb[0], .stride = stride, .flags = if (kb[1] == 1) 4 else 0, .eps = 1e-5, .scale = scale() };
        std.debug.print("ik_rrdn K {d}{s}: {d:.2} us\n", .{ kb[0], if (kb[1] == 1) " boundary" else "", try micros(g, "ik_rrdn", a, &rrdn_binds) });
    }
}
