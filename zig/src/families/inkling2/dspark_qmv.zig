//! The draft layers' 8-bit projections on core's row-exact simdgroup-matrix tiles (core/simd_qmm_bits.zig): y = x W^T for a
//! dspark_weights.Quant (MLX's affine form: codes, and a bf16 scale and bias for each group of 64 along k), with f32 sums and one
//! rounding to bf16. The tiles split k by the output width alone, so a row's bits do not depend on how many rows go with it.
const std = @import("std");
const mtl = @import("metal");
const simd = @import("../../core/simd_qmm_bits.zig");
const dw = @import("dspark_weights.zig");
const fwd = @import("forward.zig");

/// The code width of the draft layers' quantized projections.
const bits = 8;

/// The tiles' pipelines: one library, each variant built when a shape first asks for it.
pub const Qmv = struct {
    pipes: simd.Pipelines,

    pub fn init(device: mtl.Device) !Qmv {
        return .{ .pipes = try simd.Pipelines.load(device) };
    }

    pub fn deinit(q: *Qmv) void {
        q.pipes.deinit();
    }

    /// The tiles' variant for `w` and `rows` rows, building it if it is new: an error for a shape or row count they cannot take.
    pub fn check(q: *Qmv, w: dw.Quant, rows: usize) !void {
        _ = try simd.call(&q.pipes, bits, w.n, w.k, rows, w.n, false, 0);
    }

    /// out = in W^T on `e`: `rows` bf16 rows of `in` (k wide, 16-byte aligned) to `rows` rows of `out` (n wide).
    pub fn encode(q: *Qmv, e: mtl.ComputeEncoder, w: dw.Quant, in: fwd.B, out: fwd.B, rows: usize) !void {
        const c = try simd.call(&q.pipes, bits, w.n, w.k, rows, w.n, false, 0);
        simd.encode(e, c, .{ .w = w.buf, .scales = w.buf, .s_off = w.scales, .biases = w.buf, .b_off = w.biases }, in.buf, in.off, out.buf, out.off);
    }
};
