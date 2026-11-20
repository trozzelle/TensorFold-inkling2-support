//! Inkling 2 rows through every layer in one command buffer at mlx-lm's rounding points; rows stay independent.
const std = @import("std");

/// glue.metal's ik_args, field for field.
pub const Args = extern struct { rows: u32 = 0, width: u32 = 0, heads: u32 = 0, kv: u32 = 0, dim: u32 = 0, pos: u32 = 0, blocks: u32 = 0, queries: u32 = 0, stride: u32 = 0, flags: u32 = 0, window: u32 = 0, extent: u32 = 0, eps: f32 = 0, scale: f32 = 0, cap: f32 = 0, alpha: f32 = 0, floor_n: f32 = 0, thr: f32 = 0 };
pub const f32_in: u32 = 1;
pub const global: u32 = 2;
pub const boundary: u32 = 4;

comptime {
    std.debug.assert(@sizeOf(Args) == 72);
}
