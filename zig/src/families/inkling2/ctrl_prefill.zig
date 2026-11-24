//! Inkling 2 MessageGate prefill forget/gain GEMM on tensor units or simdgroup matrices
//! (kernels/metal/inkling2/ctrl_nax.metal): computes df = z @ forget_w^T and dg = z @ gain_w^T in one dispatch.
//! Every row's arithmetic is independent of neighbouring rows and chunk cuts.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");

/// A buffer and the byte offset of a view in it.
pub const B = struct { buf: mtl.Buffer, off: usize = 0 };

/// ik_ctrl_nax_args in ctrl_nax.metal, field for field.
pub const Args = extern struct {
    rows: u32,
    width: u32,
    rank: u32,
};

comptime {
    std.debug.assert(@sizeOf(Args) == 12);
}

/// Kernel source with nax.h inlined for tensor units (`units`) or simdgroup matrices.
pub fn source(a: std.mem.Allocator, units: bool) ![]u8 {
    return frags.precision.source(a, units, ks.inkling2_ctrl_nax, ks.nax, false);
}

pub const Pipes = struct {
    lib: mtl.Library,
    fg: mtl.Pipeline,

    /// Compile for this GPU: tensor units when available, else simdgroup matrices.
    pub fn init(device: mtl.Device, a: std.mem.Allocator) !Pipes {
        return initUnits(device, a, device.tensorUnits());
    }

    /// Compile for tensor units (`units = true`) or simdgroup matrices (`units = false`).
    pub fn initUnits(device: mtl.Device, a: std.mem.Allocator, units: bool) !Pipes {
        const text = try source(a, units);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, .{
            .language = mtl.CompileOptions.mlx().language,
            .math_functions = .precise,
        });
        errdefer lib.deinit();
        const fg = try mtl.Pipeline.init(device, lib, "ik_msggate_fg_nax", false);
        errdefer fg.deinit();
        if (fg.simdWidth() != 32 or fg.maxThreads() < 128) return error.UnsupportedSimd;
        return .{ .lib = lib, .fg = fg };
    }

    pub fn deinit(p: *Pipes) void {
        p.fg.deinit();
        p.lib.deinit();
    }
};

/// Compute f32 df = z @ forget_w^T and dg = z @ gain_w^T for `rows` rows of bf16 `z` ([rows, rank]).
/// Requires `width % 128 == 0` and `rank % 16 == 0`.
pub fn encodeFg(
    e: mtl.ComputeEncoder,
    p: *const Pipes,
    z: B,
    forget_w: B,
    gain_w: B,
    df: B,
    dg: B,
    rows: usize,
    width: usize,
    rank: usize,
) void {
    std.debug.assert(rows > 0 and width > 0 and width % 128 == 0 and rank > 0 and rank % 16 == 0);
    e.setPipeline(p.fg);
    e.setValue(Args{ .rows = @intCast(rows), .width = @intCast(width), .rank = @intCast(rank) }, 0);
    e.setBuffer(z.buf, z.off, 1);
    e.setBuffer(forget_w.buf, forget_w.off, 2);
    e.setBuffer(gain_w.buf, gain_w.off, 3);
    e.setBuffer(df.buf, df.off, 4);
    e.setBuffer(dg.buf, dg.off, 5);
    e.dispatchGroups(mtl.Size.of(width / 128, (rows + 31) / 32, 1), mtl.Size.of(128, 1, 1));
}

test "ctrl_prefill: source inlines nax.h for both tensor units and simdgroup matrices" {
    const a = std.testing.allocator;
    const tu = try source(a, true);
    defer a.free(tu);
    const sm = try source(a, false);
    defer a.free(sm);
    try std.testing.expect(std.mem.indexOf(u8, tu, "ik_msggate_fg_nax") != null);
    try std.testing.expect(std.mem.indexOf(u8, tu, "mma_16x32") != null);
    try std.testing.expect(std.mem.indexOf(u8, tu, "#define TF_SIMD_FRAGS") == null);
    try std.testing.expect(std.mem.indexOf(u8, sm, "ik_msggate_fg_nax") != null);
    try std.testing.expect(std.mem.indexOf(u8, sm, "#define TF_SIMD_FRAGS") != null);
}
