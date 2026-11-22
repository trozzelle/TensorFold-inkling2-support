//! tf-inkling-bench's `xp` mode: a layer's routed experts for a prompt chunk through the tensor-unit chain
//! (experts_prefill.zig), on the layer's real weights: milliseconds a layer call at each row count, the grouping included.
//! The chunk's rows and picks are recorded ones (a taps file pair: MoE inputs and every layer's picks, concatenated and
//! cycled) or, without them, random rows with picks drawn from a skewed popularity of the experts.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ik = tf.inkling2;
const xp = ik.experts_prefill;
const exl3 = tf.exl3_projection;

const H: usize = 4608;
const I: usize = 2304;
const E: usize = 384;
const TOPK: usize = 8;
const taps_layers: usize = 48; // layers in a taps file's expert-pick rows
const trials = 5;
const caps: xp.Caps = .{ .gate = 4, .up = 25 };

/// One taps file pair read whole, or several concatenated.
const Taps = struct {
    xm: []u16,
    picks: []u32,
    rows: usize,

    fn read(init: std.process.Init, spec: []const u8) !Taps {
        const colon = std.mem.indexOfScalar(u8, spec, ':') orelse return error.BadTapsSpec;
        var xm: std.ArrayList(u16) = .empty;
        var picks: std.ArrayList(u32) = .empty;
        var rows: usize = 0;
        var xs = std.mem.splitScalar(u8, spec[0..colon], ',');
        var es = std.mem.splitScalar(u8, spec[colon + 1 ..], ',');
        while (xs.next()) |xpath| {
            const epath = es.next() orelse return error.BadTapsSpec;
            const xb = try std.Io.Dir.cwd().readFileAlloc(init.io, xpath, init.gpa, .unlimited);
            defer init.gpa.free(xb);
            const eb = try std.Io.Dir.cwd().readFileAlloc(init.io, epath, init.gpa, .unlimited);
            defer init.gpa.free(eb);
            const per_row = taps_layers * TOPK * 4;
            if (eb.len == 0 or eb.len % per_row != 0) return error.BadEids;
            const t = eb.len / per_row;
            if (xb.len != t * 2 * H * 2) return error.BadXm;
            try xm.appendSlice(init.gpa, @alignCast(std.mem.bytesAsSlice(u16, xb)));
            try picks.appendSlice(init.gpa, @alignCast(std.mem.bytesAsSlice(u32, eb)));
            rows += t;
        }
        return .{ .xm = try xm.toOwnedSlice(init.gpa), .picks = try picks.toOwnedSlice(init.gpa), .rows = rows };
    }

    fn deinit(t: Taps, gpa: std.mem.Allocator) void {
        gpa.free(t.xm);
        gpa.free(t.picks);
    }
};

/// The layer's chain with its buffers, for up to `rows_max` rows.
const Bench = struct {
    p: xp.Pipes,
    layer: xp.Layer,
    s: xp.Xp,
    scratch: mtl.Buffer,
    x: mtl.Buffer,
    ids: mtl.Buffer,
    ew: mtl.Buffer,
    out: mtl.Buffer,
    uses: []const mtl.Buffer,

    fn picks(b: *const Bench) xp.Picks {
        return .{ .ids = .{ .buf = b.ids }, .weights = .{ .buf = b.ew } };
    }

    fn encode(b: *const Bench, e: mtl.ComputeEncoder, n: usize) void {
        xp.encode(e, &b.p, b.layer, .{ .buf = b.x }, b.picks(), .{ .buf = b.out }, b.s, n, caps);
    }

    /// One stage alone, on the buffers a full call left behind.
    fn stage(b: *const Bench, e: mtl.ComputeEncoder, n: usize, which: usize) void {
        switch (which) {
            0 => xp.group(e, &b.p, b.layer, b.picks().ids, b.s, n),
            1 => xp.prepare(e, &b.p, b.layer, .{ .buf = b.x }, b.s, n),
            2 => xp.gemm(e, &b.p, .w13, b.layer, b.s, n),
            3 => xp.mid(e, &b.p, b.layer, b.s, n, caps, null),
            4 => xp.gemm(e, &b.p, .w2, b.layer, b.s, n),
            else => xp.finish(e, &b.p, b.layer, b.picks(), .{ .buf = b.out }, b.s, n, null),
        }
    }
};

const stage_names = [_][]const u8{ "group", "prepare", "w13", "mid", "w2", "finish" };

fn median(ms: []f64) f64 {
    std.mem.sort(f64, ms, {}, std.sort.asc(f64));
    return ms[ms.len / 2];
}

/// Milliseconds a call takes: the median over `trials` command buffers of `calls` back-to-back calls of the chain
/// (`which` null) or of one stage, after a discarded warm-up buffer.
fn timeCalls(q: mtl.Queue, b: *const Bench, n: usize, calls: usize, which: ?usize) !f64 {
    var ms: [trials]f64 = undefined;
    for (0..trials + 1) |t| {
        const cb = q.commandBuffer();
        const e = cb.compute(.serial);
        for (b.uses) |u| e.useResource(u, mtl.ResourceUsage.read);
        for (0..calls) |_| if (which) |w| b.stage(e, n, w) else b.encode(e, n);
        e.end();
        const committed = mtl.clock.seconds();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("xp bench: {s}", .{msg});
            return error.GpuFailed;
        }
        if (t > 0) ms[t - 1] = (cb.gpuEnd() - cb.gpuStartOr(committed)) * 1e3 / @as(f64, @floatFromInt(calls));
    }
    return median(&ms);
}

fn rowCounts(text: []const u8, out: *[8]usize) !usize {
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |t| {
        if (n == out.len) return error.TooManyRowCounts;
        out[n] = try std.fmt.parseInt(usize, t, 10);
        if (out[n] == 0 or out[n] > 8192) return error.BadRows;
        n += 1;
    }
    if (n == 0) return error.BadRows;
    return n;
}

pub fn run(init: std.process.Init, args: []const [:0]const u8) !void {
    const gpa = init.gpa;
    if (args.len < 4) {
        std.debug.print("tf-inkling-bench xp <model> <layer> [rows[,rows...]] [taps=XM[,XM...]:EIDS[,EIDS...]] [parts]\n", .{});
        return error.ExpectedModelAndLayer;
    }
    const layer = try std.fmt.parseInt(u32, args[3], 10);
    var counts_buf: [8]usize = .{ 512, 2048, 4096, 0, 0, 0, 0, 0 };
    var nc: usize = 3;
    var taps: ?Taps = null;
    defer if (taps) |t| t.deinit(gpa);
    var parts = false;
    for (args[4..]) |a| {
        if (std.mem.startsWith(u8, a, "taps=")) {
            taps = try Taps.read(init, a[5..]);
        } else if (std.mem.eql(u8, a, "parts")) {
            parts = true;
        } else nc = try rowCounts(a, &counts_buf);
    }
    const counts = counts_buf[0..nc];
    var rows_max: usize = 0;
    for (counts) |c| rows_max = @max(rows_max, c);

    const device = try mtl.Device.init();
    const q = try device.queue();
    var b: Bench = undefined;
    b.p = try xp.Pipes.init(device, gpa);
    defer b.p.deinit();
    // the layer's table, from the pack
    var ck = ik.weights.Checkpoint.init(gpa);
    defer ck.deinit();
    var pack = try ik.expert_pack.Pack.read(gpa, init.io, args[2], layer + 1, E);
    defer pack.deinit();
    var widths: [5]u32 = @splat(0);
    const table = try ik.weights.expertTable(gpa, device, &ck, args[2], pack.layers[layer], layer, H, I, E, &widths);
    defer table.deinit();
    if (!ik.weights.xdOk(table.slice(exl3.ExpertDescriptor, 2 * E), E, H, I)) return error.LayerNotXd;
    b.layer = .{
        .w13 = .{ .descriptors = .{ .buffer = table }, .count = E, .k = H, .n = 2 * I },
        .w2 = .{ .descriptors = .{ .buffer = table, .offset = E * @sizeOf(exl3.ExpertDescriptor) }, .count = E, .k = I, .n = H },
    };
    var uses: std.ArrayList(mtl.Buffer) = .empty;
    defer uses.deinit(gpa);
    try uses.append(gpa, table);
    for (ck.shards.items) |sh| try uses.append(gpa, sh.buffer);
    b.uses = uses.items;
    b.scratch = try device.buffer(xp.xpBytes(rows_max), mtl.ResourceOptions.shared);
    defer b.scratch.deinit();
    b.s = xp.Xp.init(b.scratch, 0, rows_max);
    b.x = try device.buffer(rows_max * H * 2, mtl.ResourceOptions.shared);
    defer b.x.deinit();
    b.ids = try device.buffer(rows_max * TOPK * 4, mtl.ResourceOptions.shared);
    defer b.ids.deinit();
    b.ew = try device.buffer(rows_max * TOPK * 4, mtl.ResourceOptions.shared);
    defer b.ew.deinit();
    b.out = try device.buffer(rows_max * H * 2, mtl.ResourceOptions.shared);
    defer b.out.deinit();
    std.debug.print("{{\"xp\": {{\"layer\": {d}, \"k2\": {d}, \"k3\": {d}, \"taps_rows\": {d}, \"window\": {d}, \"gemm_max_threads\": [{d}, {d}, {d}, {d}]}}}}\n", .{ layer, widths[2], widths[3], if (taps) |t| t.rows else 0, xp.window, b.p.gemm[0][0].maxThreads(), b.p.gemm[0][1].maxThreads(), b.p.gemm[1][0].maxThreads(), b.p.gemm[1][1].maxThreads() });

    var prng = std.Random.DefaultPrng.init(0x5eed);
    const r = prng.random();
    const slot: usize = if (taps != null) blk: {
        for (ik.forward.xm_layers, 0..) |l, k| if (l == layer) break :blk k;
        return error.LayerNotRecorded;
    } else 0;
    for (counts) |n| {
        const S = n * TOPK;
        const xs = b.x.slice(u16, n * H);
        const ids = b.ids.slice(u32, S);
        const ew = b.ew.slice(f32, S);
        if (taps) |t| {
            for (0..n) |row| {
                const src = row % t.rows;
                @memcpy(xs[row * H ..][0..H], t.xm[(src * 2 + slot) * H ..][0..H]);
                for (0..TOPK) |k| {
                    const e = t.picks[(src * taps_layers + layer) * TOPK + k];
                    if (e >= E) return error.BadExpertId;
                    ids[row * TOPK + k] = e;
                }
            }
        } else {
            for (xs) |*v| v.* = @intCast(@as(u32, @bitCast(@as(f32, @floatCast((r.float(f64) * 2 - 1) * 0.1)))) >> 16);
            skewed(r, ids, n);
        }
        for (ew) |*w| w.* = 0.05 + r.float(f32);
        var seen: [E]bool = @splat(false);
        for (ids) |e| seen[e] = true;
        var distinct: usize = 0;
        for (seen) |v| distinct += @intFromBool(v);
        // one call first: the stages' own buffers hold a full call's values for the parts, and a failure shows early
        {
            const cb = q.commandBuffer();
            const e = cb.compute(.serial);
            for (b.uses) |u| e.useResource(u, mtl.ResourceUsage.read);
            b.encode(e, n);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.log.err("xp bench: {s}", .{msg});
                return error.GpuFailed;
            }
        }
        const calls: usize = @max(2, 8192 / n);
        const ms = try timeCalls(q, &b, n, calls, null);
        std.debug.print("{{\"xp\": {{\"layer\": {d}, \"rows\": {d}, \"ms\": {d:.3}, \"distinct\": {d}, \"rows_per_s\": {d:.0}}}}}\n", .{ layer, n, ms, distinct, @as(f64, @floatFromInt(n)) / ms * 1e3 });
        if (parts) {
            std.debug.print("{{\"xp_parts\": {{\"layer\": {d}, \"rows\": {d}", .{ layer, n });
            for (stage_names, 0..) |name, w| {
                const t = try timeCalls(q, &b, n, calls, w);
                std.debug.print(", \"{s}_ms\": {d:.3}", .{ name, t });
            }
            std.debug.print("}}}}\n", .{});
        }
    }
}

/// Picks for `n` rows: each row's 8 distinct experts drawn by popularity 1 / (rank + 1), ranks shuffled over the experts.
fn skewed(r: std.Random, ids: []u32, n: usize) void {
    var perm: [E]u32 = undefined;
    for (&perm, 0..) |*p, i| p.* = @intCast(i);
    r.shuffle(u32, &perm);
    var cdf: [E]f64 = undefined;
    var total: f64 = 0;
    for (&cdf, 0..) |*c, k| {
        total += 1.0 / @as(f64, @floatFromInt(k + 1));
        c.* = total;
    }
    for (0..n) |row| {
        var k: usize = 0;
        while (k < TOPK) {
            const u = r.float(f64) * total;
            var lo: usize = 0;
            var hi: usize = E - 1;
            while (lo < hi) {
                const mid = (lo + hi) / 2;
                if (cdf[mid] < u) lo = mid + 1 else hi = mid;
            }
            const pick = perm[lo];
            if (std.mem.indexOfScalar(u32, ids[row * TOPK ..][0..k], pick) != null) continue;
            ids[row * TOPK + k] = pick;
            k += 1;
        }
    }
}
