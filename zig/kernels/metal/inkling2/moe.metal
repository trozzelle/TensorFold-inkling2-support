// Inkling 2's routed experts around the EXL3 projections: fp32 router logits, mlx-lm's expert_routing, grouped 8-row
// tiles by expert for exl3_projection's grouped mode, the capped SwiGLU on interleaved rows and the weighted combine.
#include <metal_stdlib>
using namespace metal;
