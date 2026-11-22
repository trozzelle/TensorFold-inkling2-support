//! Inkling 2 kernel checks: head preparation, relative bias and attention against FP64 references, with the attention
//! row-independence and NaN-containment loops.
const std = @import("std");
const mtl = @import("metal");
const h = @import("harness.zig");

fn tau(p: usize) f64 {
    return 1 + 0.1 * @log(@max((@as(f64, @floatFromInt(p)) + 1) / 8192, 1));
}

fn casePrep(g: *const h.Gpu, r: std.Random) !void {
    const rows = 3;
    const Hq = 48;
    const KV = 8;
    const D = 128;
    const pos0 = 20000; // tau > 1 on a global layer
    const cap = 24000;
    const q = try g.buf(rows * Hq * D * 2);
    const k = try g.buf(rows * KV * D * 2);
    const v = try g.buf(rows * KV * D * 2);
    const rr = try g.buf(rows * Hq * 16 * 2);
    const qs = try g.buf(Hq * D * 2);
    const rs = try g.buf(Hq * 16 * 2);
    const qn = try g.buf(rows * Hq * D * 2);
    const keys = try g.buf(cap * KV * D * 2);
    const vals = try g.buf(cap * KV * D * 2);
    const reln = try g.buf(rows * Hq * 16 * 2);
    defer for ([_]mtl.Buffer{ q, k, v, rr, qs, rs, qn, keys, vals, reln }) |b| b.deinit();
    const qv = h.fill16(q, rows * Hq * D, r, 4);
    const kv = h.fill16(k, rows * KV * D, r, 4);
    const vv = h.fill16(v, rows * KV * D, r, 4);
    const rv = h.fill16(rr, rows * Hq * 16, r, 4);
    const qsv = h.fill16(qs, Hq * D, r, 1.5);
    const rsv = h.fill16(rs, Hq * 16, r, 1.5);
    for ([_]bool{ false, true }) |global| {
        var c: h.Check = .{ .name = if (global) "prep global" else "prep local" };
        try g.run("ik_prep", .{ .rows = rows, .heads = Hq, .kv = KV, .dim = D, .pos = pos0, .flags = if (global) 2 else 0, .ring = cap, .eps = 1e-6, .cap = 22.627416997969522, .thr = 8, .alpha = 0.1, .floor_n = 8192 }, &.{ .{ .buf = q }, .{ .buf = k }, .{ .buf = v }, .{ .buf = rr }, .{ .buf = qs }, .{ .buf = rs }, .{ .buf = qn }, .{ .buf = keys }, .{ .buf = vals }, .{ .buf = reln } }, .{ 2 * Hq + 2 * KV, rows, 1 }, .{ 32, 1, 1 }, true);
        for (0..rows) |t| {
            const tf_ = if (global) tau(pos0 + t) else 1.0;
            for (0..Hq) |hd| { // q: scaled clip at 22.627, tau
                var ss: f64 = 0;
                for (0..D) |d| ss += std.math.pow(f64, h.fb(qv[(t * Hq + hd) * D + d]) * h.fb(qsv[hd * D + d]), 2);
                const f = @min(22.627416997969522 / (@sqrt(ss) + 1e-6), 1);
                for (0..D) |d| {
                    var want = h.rb(h.fb(qv[(t * Hq + hd) * D + d]) * h.fb(qsv[hd * D + d]) * f);
                    if (global) want = want * tf_;
                    c.b16(qn.slice(u16, rows * Hq * D)[(t * Hq + hd) * D + d], want);
                }
            }
            for (0..KV) |hd| { // k: clip at 22.627 unscaled; v: unit RMS; both at position pos0 + t
                var ks2: f64 = 0;
                var vs2: f64 = 0;
                for (0..D) |d| {
                    ks2 += std.math.pow(f64, h.fb(kv[(t * KV + hd) * D + d]), 2);
                    vs2 += std.math.pow(f64, h.fb(vv[(t * KV + hd) * D + d]), 2);
                }
                const f = @min(22.627416997969522 / (@sqrt(ks2) + 1e-6), 1);
                const inv = 1 / @sqrt(vs2 / D + 1e-6);
                for (0..D) |d| {
                    const at = ((pos0 + t) * KV + hd) * D + d;
                    c.b16(keys.slice(u16, cap * KV * D)[at], h.fb(kv[(t * KV + hd) * D + d]) * f);
                    c.b16(vals.slice(u16, cap * KV * D)[at], h.fb(vv[(t * KV + hd) * D + d]) * inv);
                }
            }
            for (0..Hq) |hd| { // rel: scaled clip at 8, tau
                var ss: f64 = 0;
                for (0..16) |i| ss += std.math.pow(f64, h.fb(rv[(t * Hq + hd) * 16 + i]) * h.fb(rsv[hd * 16 + i]), 2);
                const f = @min(8 / (@sqrt(ss) + 1e-6), 1);
                for (0..16) |i| {
                    var want = h.rb(h.fb(rv[(t * Hq + hd) * 16 + i]) * h.fb(rsv[hd * 16 + i]) * f);
                    if (global) want = want * tf_;
                    c.b16(reln.slice(u16, rows * Hq * 16)[(t * Hq + hd) * 16 + i], want);
                }
            }
        }
        try c.done();
    }
}

fn caseAttn(g: *const h.Gpu, r: std.Random) !void {
    const rows = 4;
    const Hq = 48;
    const KV = 8;
    const D = 128;
    const E = 1024;
    const pos0 = 1500; // local rows see 1024 keys, global rows 1500 + t + 1
    const cap = pos0 + rows;
    const qn = try g.buf(rows * Hq * D * 2);
    const keys = try g.buf(cap * KV * D * 2);
    const vals = try g.buf(cap * KV * D * 2);
    const pb = try g.buf(rows * Hq * E * 2);
    const sink = try g.buf(rows * Hq * 2);
    const gate = try g.buf(rows * Hq * D * 2);
    const out = try g.buf(rows * Hq * D * 2);
    const reln = try g.buf(rows * Hq * 16 * 2);
    const proj = try g.buf(16 * E * 2);
    defer for ([_]mtl.Buffer{ qn, keys, vals, pb, sink, gate, out, reln, proj }) |b| b.deinit();
    const qv = h.fill16(qn, rows * Hq * D, r, 0.4);
    const kv = h.fill16(keys, cap * KV * D, r, 0.4);
    const vv = h.fill16(vals, cap * KV * D, r, 1.0);
    const sv = h.fill16(sink, rows * Hq, r, 2.0);
    const gv = h.fill16(gate, rows * Hq * D, r, 3.0);
    const relv = h.fill16(reln, rows * Hq * 16, r, 1.0);
    const pv = h.fill16(proj, 16 * E, r, 1.0);
    var cb: h.Check = .{ .name = "relbias" };
    try g.run("ik_relbias", .{ .rows = rows, .heads = Hq, .extent = E, .dim = 16 }, &.{ .{ .buf = reln }, .{ .buf = proj }, .{ .buf = pb } }, .{ E, Hq, rows }, .{ 256, 1, 1 }, false);
    const pbs = pb.slice(u16, rows * Hq * E);
    for (0..rows * Hq) |th| for (0..E) |d| {
        var s: f64 = 0;
        for (0..16) |i| s += h.fb(relv[th * 16 + i]) * h.fb(pv[i * E + d]);
        cb.b16(pbs[th * E + d], s);
    };
    try cb.done();
    const scale = 1 / @sqrt(@as(f64, D));
    const out_row = Hq * D * 2;
    const saved = try std.heap.page_allocator.alloc(u8, rows * out_row);
    defer std.heap.page_allocator.free(saved);
    for ([_]u32{ 1024, 0 }) |window| {
        const name = if (window > 0) "attention local" else "attention global";
        const full: h.Args = .{ .rows = rows, .heads = Hq, .kv = KV, .dim = D, .pos = pos0, .window = window, .extent = E, .ring = cap, .scale = @floatCast(scale) };
        var c: h.Check = .{ .name = name };
        try g.run("ik_attn", full, &.{ .{ .buf = qn }, .{ .buf = keys }, .{ .buf = vals }, .{ .buf = pb }, .{ .buf = sink }, .{ .buf = gate }, .{ .buf = out } }, .{ Hq, rows, 1 }, .{ 32, 1, 1 }, true);
        for (0..rows) |t| for (0..Hq) |hd| {
            const p = pos0 + t;
            const first = if (window > 0 and p + 1 > window) p + 1 - window else 0;
            const kvh = hd / (Hq / KV);
            var m = h.fb(sv[t * Hq + hd]);
            var logits = try std.heap.page_allocator.alloc(f64, p + 1);
            defer std.heap.page_allocator.free(logits);
            for (first..p + 1) |j| {
                var s: f64 = 0;
                for (0..D) |d| s += h.fb(qv[(t * Hq + hd) * D + d]) * h.fb(kv[(j * KV + kvh) * D + d]);
                s *= scale;
                if (p - j < E) s += h.fb(pbs[(t * Hq + hd) * E + p - j]);
                logits[j] = s;
                m = @max(m, s);
            }
            var z = @exp(h.fb(sv[t * Hq + hd]) - m);
            for (first..p + 1) |j| z += @exp(logits[j] - m);
            for (0..D) |d| {
                var o: f64 = 0;
                for (first..p + 1) |j| o += @exp(logits[j] - m) / z * h.fb(vv[(j * KV + kvh) * D + d]);
                const gt = h.fb(gv[(t * Hq + hd) * D + d]);
                // two rounding points (o, then o times the gate): a flip at the first can move the second by two steps;
                // outputs near zero (weighted sums of +-1 values cancelling) keep the f32 sum's absolute error
                c.b16a(out.slice(u16, rows * Hq * D)[(t * Hq + hd) * D + d], h.rb(o) / (1 + @exp(-gt)), 2, 1e-6);
            }
        };
        try c.done();

        // Row independence: each row alone, with its inputs and output bound at the row's offsets, gives the batch's bits.
        var ci: h.Check = .{ .name = if (window > 0) "attention local independence" else "attention global independence" };
        @memcpy(saved, out.slice(u8, rows * out_row));
        for (0..rows) |t| {
            var one = full;
            one.rows = 1;
            one.pos = pos0 + @as(u32, @intCast(t));
            try g.run("ik_attn", one, &.{ .{ .buf = qn, .off = t * Hq * D * 2 }, .{ .buf = keys }, .{ .buf = vals }, .{ .buf = pb, .off = t * Hq * E * 2 }, .{ .buf = sink, .off = t * Hq * 2 }, .{ .buf = gate, .off = t * Hq * D * 2 }, .{ .buf = out, .off = t * out_row } }, .{ Hq, 1, 1 }, .{ 32, 1, 1 }, true);
            ci.same(out.slice(u8, rows * out_row)[t * out_row ..][0..out_row], saved[t * out_row ..][0..out_row]);
        }
        // NaN containment: a NaN in row 1's query leaves rows 0, 2 and 3 with their bits.
        const nan_at = Hq * D;
        const kept = qv[nan_at];
        qv[nan_at] = 0x7fc0;
        try g.run("ik_attn", full, &.{ .{ .buf = qn }, .{ .buf = keys }, .{ .buf = vals }, .{ .buf = pb }, .{ .buf = sink }, .{ .buf = gate }, .{ .buf = out } }, .{ Hq, rows, 1 }, .{ 32, 1, 1 }, true);
        qv[nan_at] = kept;
        for (0..rows) |t| if (t != 1) ci.same(out.slice(u8, rows * out_row)[t * out_row ..][0..out_row], saved[t * out_row ..][0..out_row]);
        try ci.done();
    }
}

fn caseAttnRing(g: *const h.Gpu, r: std.Random) !void {
    const Hq = 48;
    const KV = 8;
    const D = 128;
    const E = 1024;
    const R_small: usize = 1280;
    const T: usize = 3 * R_small + 100;
    const lin_cap: usize = std.mem.alignForward(usize, T, 64);
    const max_chunk: usize = 64;
    const nan: u16 = 0x7fc0;
    const scale: f32 = @floatCast(1 / @sqrt(@as(f64, D)));
    const q = try g.buf(max_chunk * Hq * D * 2);
    const k = try g.buf(max_chunk * KV * D * 2);
    const v = try g.buf(max_chunk * KV * D * 2);
    const rr = try g.buf(max_chunk * Hq * 16 * 2);
    const qs = try g.buf(Hq * D * 2);
    const rs = try g.buf(Hq * 16 * 2);
    const qn = try g.buf(max_chunk * Hq * D * 2);
    const reln = try g.buf(max_chunk * Hq * 16 * 2);
    const pb = try g.buf(max_chunk * Hq * E * 2);
    const proj = try g.buf(16 * E * 2);
    const sink = try g.buf(max_chunk * Hq * 2);
    const gate = try g.buf(max_chunk * Hq * D * 2);
    const out_lin = try g.buf(max_chunk * Hq * D * 2);
    const out_ring = try g.buf(max_chunk * Hq * D * 2);
    const lin_k = try g.buf(lin_cap * KV * D * 2);
    const lin_v = try g.buf(lin_cap * KV * D * 2);
    const ring_k = try g.buf(lin_cap * KV * D * 2);
    const ring_v = try g.buf(lin_cap * KV * D * 2);
    defer for ([_]mtl.Buffer{ q, k, v, rr, qs, rs, qn, reln, pb, proj, sink, gate, out_lin, out_ring, lin_k, lin_v, ring_k, ring_v }) |b| b.deinit();
    _ = h.fill16(qs, Hq * D, r, 1.5);
    _ = h.fill16(rs, Hq * 16, r, 1.5);
    _ = h.fill16(proj, 16 * E, r, 1.0);
    const chunks = [_]usize{ 64, 17 };
    for ([_]bool{ false, true }) |global| {
        const ring_slots: usize = if (global) lin_cap else R_small;
        const window: u32 = if (global) 0 else E;
        var c: h.Check = .{ .name = if (global) "attention ring global" else "attention ring local" };
        var pos: usize = 0;
        var ci: usize = 0;
        while (pos < T) : (ci += 1) {
            const n = @min(chunks[ci % chunks.len], T - pos);
            _ = h.fill16(q, n * Hq * D, r, 1.0);
            _ = h.fill16(k, n * KV * D, r, 1.0);
            _ = h.fill16(v, n * KV * D, r, 1.0);
            _ = h.fill16(rr, n * Hq * 16, r, 1.0);
            _ = h.fill16(sink, n * Hq, r, 2.0);
            _ = h.fill16(gate, n * Hq * D, r, 3.0);
            var poisoned: [R_small]bool = @splat(false);
            if (!global) {
                const live_lo = std.mem.alignBackward(usize, pos -| (E - 1), 64);
                var keep: [R_small]bool = @splat(false);
                for (live_lo..pos) |p| keep[p % R_small] = true;
                const rk = ring_k.slice(u16, R_small * KV * D);
                const rv = ring_v.slice(u16, R_small * KV * D);
                for (0..R_small) |s| if (!keep[s]) {
                    poisoned[s] = true;
                    @memset(rk[s * KV * D ..][0 .. KV * D], nan);
                    @memset(rv[s * KV * D ..][0 .. KV * D], nan);
                };
            }
            const prep_base: h.Args = .{ .rows = @intCast(n), .heads = Hq, .kv = KV, .dim = D, .pos = @intCast(pos), .flags = if (global) 2 else 0, .eps = 1e-6, .cap = 22.627416997969522, .thr = 8, .alpha = 0.1, .floor_n = 8192 };
            var prep_lin = prep_base;
            prep_lin.ring = @intCast(lin_cap);
            try g.run("ik_prep", prep_lin, &.{ .{ .buf = q }, .{ .buf = k }, .{ .buf = v }, .{ .buf = rr }, .{ .buf = qs }, .{ .buf = rs }, .{ .buf = qn }, .{ .buf = lin_k }, .{ .buf = lin_v }, .{ .buf = reln } }, .{ 2 * Hq + 2 * KV, n, 1 }, .{ 32, 1, 1 }, true);
            var prep_ring = prep_base;
            prep_ring.ring = @intCast(ring_slots);
            try g.run("ik_prep", prep_ring, &.{ .{ .buf = q }, .{ .buf = k }, .{ .buf = v }, .{ .buf = rr }, .{ .buf = qs }, .{ .buf = rs }, .{ .buf = qn }, .{ .buf = ring_k }, .{ .buf = ring_v }, .{ .buf = reln } }, .{ 2 * Hq + 2 * KV, n, 1 }, .{ 32, 1, 1 }, true);
            try g.run("ik_relbias", .{ .rows = @intCast(n), .heads = Hq, .extent = E, .dim = 16 }, &.{ .{ .buf = reln }, .{ .buf = proj }, .{ .buf = pb } }, .{ E, Hq, n }, .{ 256, 1, 1 }, false);
            const attn_base: h.Args = .{ .rows = @intCast(n), .heads = Hq, .kv = KV, .dim = D, .pos = @intCast(pos), .window = window, .extent = E, .scale = scale };
            var attn_lin = attn_base;
            attn_lin.ring = @intCast(lin_cap);
            try g.run("ik_attn", attn_lin, &.{ .{ .buf = qn }, .{ .buf = lin_k }, .{ .buf = lin_v }, .{ .buf = pb }, .{ .buf = sink }, .{ .buf = gate }, .{ .buf = out_lin } }, .{ Hq, n, 1 }, .{ 32, 1, 1 }, true);
            var attn_ring = attn_base;
            attn_ring.ring = @intCast(ring_slots);
            try g.run("ik_attn", attn_ring, &.{ .{ .buf = qn }, .{ .buf = ring_k }, .{ .buf = ring_v }, .{ .buf = pb }, .{ .buf = sink }, .{ .buf = gate }, .{ .buf = out_ring } }, .{ Hq, n, 1 }, .{ 32, 1, 1 }, true);
            const bytes = n * Hq * D * 2;
            c.same(out_ring.slice(u8, bytes), out_lin.slice(u8, bytes));
            for (out_ring.slice(u16, n * Hq * D)) |x| {
                if (std.math.isNan(h.fb(x))) {
                    c.bad += 1;
                    break;
                }
            }
            if (!global) {
                var written: [R_small]bool = @splat(false);
                for (pos..pos + n) |p| written[p % R_small] = true;
                const rk = ring_k.slice(u16, R_small * KV * D);
                const rv = ring_v.slice(u16, R_small * KV * D);
                for (0..R_small) |s| if (poisoned[s] and !written[s]) {
                    c.n += 1;
                    for (rk[s * KV * D ..][0 .. KV * D], rv[s * KV * D ..][0 .. KV * D]) |kv_x, vv_x| {
                        if (kv_x != nan or vv_x != nan) {
                            c.bad += 1;
                            break;
                        }
                    }
                };
            }
            pos += n;
        }
        try c.done();
    }
}

pub fn run(g: *const h.Gpu, r: std.Random) !void {
    try casePrep(g, r);
    try caseAttn(g, r);
    try caseAttnRing(g, r);
}
