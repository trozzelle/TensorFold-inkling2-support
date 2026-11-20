// Inkling 2's glue at mlx-lm's rounding points (mlx_lm/models/inkling2.py): embeddings, norms, depth attention,
// residual adds, copies and the final softcap. Every kernel keeps rows independent. Follows core/bf16_matvec.metal.
#include <metal_stdlib>
using namespace metal;

// forward.zig's Args, field for field.
struct ik_args {
  uint rows, width, heads, kv, dim, pos, blocks, queries;
  uint stride, flags, window, extent;
  float eps, scale, cap, alpha, floor_n, thr;
};

constant uint IK_F32_IN = 1, IK_GLOBAL = 2, IK_BOUNDARY = 4;

inline float ik_in(const device void* x, bool f32in, ulong i) {
  return f32in ? ((const device float*)x)[i] : float(((const device bfloat*)x)[i]);
}

// A threadgroup-wide sum in a fixed order: simd_sum, then the simdgroups in index order; every thread gets it.
inline float ik_sum(float v, threadgroup float* buf, uint sg, uint lane, uint groups) {
  v = simd_sum(v);
  if (lane == 0) buf[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0.0f;
  for (uint s = 0; s < groups; ++s) t += buf[s];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

TF_BF16_MATVECS(ik)
