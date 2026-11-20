// Inkling 2's short convolutions at mlx-lm's rounding points: static taps convolve in the weights' bf16, dynamic taps
// (global layers' k and v) in f32; inputs before this chunk come from a 32-position ring. Follows glue.metal.
#include <metal_stdlib>
using namespace metal;
