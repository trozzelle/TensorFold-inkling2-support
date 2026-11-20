// Inkling 2's attention at mlx-lm's rounding points: q, k and relative-state L2 clips, unit-RMS values, the global
// layers' tau, the relative bias bf16(rel . proj), and attention over the cache with the sink logit and output gate.
#include <metal_stdlib>
using namespace metal;
