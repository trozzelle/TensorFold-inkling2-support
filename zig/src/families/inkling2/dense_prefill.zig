//! Inkling 2's dense prompt projections as one tensor-unit GEMM per matrix: bf16 rows x the release's bf16 [N, K]
//! weight, f32 accumulation in a fixed K order whatever the row count, bf16 rows out (the router's logits in f32).
//! Every row's bits are independent of the rows beside it and of which M/N edge variant computes it.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");

/// A buffer and the byte offset of a matrix in it.
pub const Mat = struct { buf: mtl.Buffer, off: usize = 0 };
/// A bf16 [n, k] weight (torch's Linear layout).
pub const Weight = struct { buf: mtl.Buffer, off: usize = 0, n: usize, k: usize };
/// The element type of the output rows.
pub const Out = enum { bf16, f32 };

/// The kernels' text: gemm_nax with the fragment layout of tensor units (`units`) or of simdgroup matrices inlined.
pub fn source(a: std.mem.Allocator, units: bool) ![]u8 {
    return frags.precision.source(a, units, ks.inkling2_gemm, ks.nax, false);
}

/// The compiled variants: [M edge][N edge] for each output type, indexed by `M % 64 == 0` and `N % 128 == 0`.
pub const Pipes = struct {
    bf16: [2][2]mtl.Pipeline,
    f32: [2][2]mtl.Pipeline,

    /// The variants for this GPU: on its tensor units when it has them.
    pub fn init(device: mtl.Device, a: std.mem.Allocator) !Pipes {
        return initUnits(device, a, device.tensorUnits());
    }

    /// The variants on tensor units or on simdgroup matrices (any GPU runs the latter).
    pub fn initUnits(device: mtl.Device, a: std.mem.Allocator, units: bool) !Pipes {
        const src = try source(a, units);
        defer a.free(src);
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipes = undefined;
        var made: usize = 0;
        errdefer for (0..made) |i| p.at(i).deinit();
        while (made < 8) : (made += 1) {
            var nm: [16]u8 = undefined;
            const name = try std.fmt.bufPrintSentinel(&nm, "ik_gemm{s}_{d}{d}", .{ if (made >= 4) "_f32" else "", (made / 2) % 2, made % 2 }, 0);
            p.at(made).* = try mtl.Pipeline.init(device, lib, name, false);
        }
        return p;
    }

    fn at(p: *Pipes, i: usize) *mtl.Pipeline {
        const set = if (i < 4) &p.bf16 else &p.f32;
        return &set[(i / 2) % 2][i % 2];
    }

    pub fn deinit(p: *Pipes) void {
        for (0..8) |i| p.at(i).deinit();
    }
};

/// out[t] = w x[t] for the `n` rows of `x` ([n][w.k] bf16), into `out` ([n][ldy] bf16 or f32 by `kind`, ldy >= w.n).
/// The variant follows `n % 64` and `w.n % 128`; the rows' bits do not depend on it.
pub fn gemmRows(e: mtl.ComputeEncoder, p: *const Pipes, x: Mat, w: Weight, out: Mat, kind: Out, n: usize, ldy: usize) void {
    std.debug.assert(n > 0 and ldy >= w.n and w.k > 0);
    const tn = (w.n + 127) / 128;
    const tm = (n + 63) / 64;
    const set = if (kind == .bf16) &p.bf16 else &p.f32;
    e.setPipeline(set[@intFromBool(n % 64 == 0)][@intFromBool(w.n % 128 == 0)]);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(w.buf, w.off, 1);
    var par: [16]i32 = @splat(0);
    const vals = [_]usize{ n, w.n, w.k, w.k, w.k, ldy, tn, tm, 2, w.k / 256 };
    for (vals, 0..) |v, i| par[i] = @intCast(v);
    e.setBytes(std.mem.asBytes(&par), 2);
    e.setBuffer(out.buf, out.off, 3);
    e.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
}
