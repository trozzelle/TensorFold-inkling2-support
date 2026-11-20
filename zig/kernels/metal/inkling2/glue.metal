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

#define IK_ROW uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], \
  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint nt [[threads_per_threadgroup]]

// embed_norm(embed_tokens(ids)): the weighted RMSNorm of the bf16 rows, one rounding to bf16.
[[kernel]] void ik_embed(constant ik_args& a [[buffer(0)]], const device uint* ids [[buffer(1)]],
    const device bfloat* table [[buffer(2)]], const device bfloat* w [[buffer(3)]], device bfloat* y [[buffer(4)]], IK_ROW) {
  threadgroup float buf[32];
  const device bfloat* x = table + ulong(ids[row]) * a.width;
  float ss = 0.0f;
  for (uint d = tid; d < a.width; d += nt) { const float v = float(x[d]); ss += v * v; }
  const float r = rsqrt(ik_sum(ss, buf, sg, lane, nt / 32) / float(a.width) + a.eps);
  for (uint d = tid; d < a.width; d += nt) y[ulong(row) * a.width + d] = bfloat(float(w[d]) * (float(x[d]) * r));
}

// _unweighted_rms_norm (eps 1e-6) of bf16 or f32 rows into f32 rows.
[[kernel]] void ik_rms(constant ik_args& a [[buffer(0)]], const device void* x [[buffer(1)]], device float* y [[buffer(2)]], IK_ROW) {
  threadgroup float buf[32];
  const bool f = a.flags & IK_F32_IN;
  float ss = 0.0f;
  for (uint d = tid; d < a.width; d += nt) { const float v = ik_in(x, f, ulong(row) * a.width + d); ss += v * v; }
  const float r = rsqrt(ik_sum(ss, buf, sg, lane, nt / 32) / float(a.width) + a.eps);
  for (uint d = tid; d < a.width; d += nt) y[ulong(row) * a.width + d] = ik_in(x, f, ulong(row) * a.width + d) * r;
}

// nn.RMSNorm of bf16 or f32 rows (a.stride elements apart) into bf16 rows: w * (x * rsqrt(mean x^2 + eps)).
[[kernel]] void ik_norm(constant ik_args& a [[buffer(0)]], const device void* x [[buffer(1)]],
    const device bfloat* w [[buffer(2)]], device bfloat* y [[buffer(3)]], IK_ROW) {
  threadgroup float buf[32];
  const bool f = a.flags & IK_F32_IN;
  float ss = 0.0f;
  for (uint d = tid; d < a.width; d += nt) { const float v = ik_in(x, f, ulong(row) * a.stride + d); ss += v * v; }
  const float r = rsqrt(ik_sum(ss, buf, sg, lane, nt / 32) / float(a.width) + a.eps);
  for (uint d = tid; d < a.width; d += nt) y[ulong(row) * a.width + d] = bfloat(float(w[d]) * (ik_in(x, f, ulong(row) * a.stride + d) * r));
}

constant uint IK_MAXN = 10, IK_MAXQ = 3;

// BlockAttentionResidual, as mlx-lm's depth_attention kernel: Q learned queries over the a.blocks block states (rows
// a.stride apart) and the partial state; softmax of dot * a.scale over the sources, then their weighted sum.
[[kernel]] void ik_depth(constant ik_args& a [[buffer(0)]], const device float* blocks [[buffer(1)]],
    const device float* partial [[buffer(2)]], const device float* q [[buffer(3)]], device float* out [[buffer(4)]], IK_ROW) {
  threadgroup float buf[IK_MAXN * IK_MAXQ][32];
  const uint N = a.blocks + 1, Q = a.queries, D = a.width;
  float acc[IK_MAXN * IK_MAXQ];
  for (uint i = 0; i < IK_MAXN * IK_MAXQ; ++i) acc[i] = 0.0f;
  for (uint d = tid; d < D; d += nt) {
    for (uint i = 0; i < N; ++i) {
      const float x = i < a.blocks ? blocks[(ulong(i) * a.stride + row) * D + d] : partial[ulong(row) * D + d];
      for (uint j = 0; j < Q; ++j) acc[i * IK_MAXQ + j] += x * q[j * D + d];
    }
  }
  for (uint i = 0; i < N; ++i) for (uint j = 0; j < Q; ++j) {
    const float v = simd_sum(acc[i * IK_MAXQ + j]);
    if (lane == 0) buf[i * IK_MAXQ + j][sg] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float w[IK_MAXN * IK_MAXQ];
  for (uint i = 0; i < N; ++i) for (uint j = 0; j < Q; ++j) {
    float t = 0.0f;
    for (uint s = 0; s < nt / 32; ++s) t += buf[i * IK_MAXQ + j][s];
    w[i * IK_MAXQ + j] = t * a.scale;
  }
  for (uint j = 0; j < Q; ++j) {
    float m = -INFINITY;
    for (uint i = 0; i < N; ++i) m = max(m, w[i * IK_MAXQ + j]);
    float z = 0.0f;
    for (uint i = 0; i < N; ++i) { w[i * IK_MAXQ + j] = exp(w[i * IK_MAXQ + j] - m); z += w[i * IK_MAXQ + j]; }
    for (uint i = 0; i < N; ++i) w[i * IK_MAXQ + j] /= z;
  }
  for (uint d = tid; d < D; d += nt) {
    for (uint j = 0; j < Q; ++j) {
      float o = 0.0f;
      for (uint i = 0; i < N; ++i) {
        const float x = i < a.blocks ? blocks[(ulong(i) * a.stride + row) * D + d] : partial[ulong(row) * D + d];
        o += w[i * IK_MAXQ + j] * x;
      }
      out[(ulong(row) * Q + j) * D + d] = o;
    }
  }
}

// The layer's f32 residual: the attention branch alone at a block boundary, else the layer input plus it.
[[kernel]] void ik_resid(constant ik_args& a [[buffer(0)]], const device bfloat* h [[buffer(1)]],
    const device bfloat* ya [[buffer(2)]], device float* res [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= a.width || gid.y >= a.rows) return;
  const ulong i = ulong(gid.y) * a.width + gid.x;
  res[i] = (a.flags & IK_BOUNDARY ? 0.0f : float(h[i])) + float(ya[i]);
}

// The layer output: bf16(residual + the MLP branch).
[[kernel]] void ik_out(constant ik_args& a [[buffer(0)]], const device float* res [[buffer(1)]],
    const device bfloat* ym [[buffer(2)]], device bfloat* h [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= a.width || gid.y >= a.rows) return;
  const ulong i = ulong(gid.y) * a.width + gid.x;
  h[i] = bfloat(res[i] + float(ym[i]));
}

// bf16 rows copied (a layer dump).
[[kernel]] void ik_copy(constant ik_args& a [[buffer(0)]], const device bfloat* src [[buffer(1)]],
    device bfloat* dst [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= a.width || gid.y >= a.rows) return;
  dst[ulong(gid.y) * a.width + gid.x] = src[ulong(gid.y) * a.width + gid.x];
}

// a.cap * tanh(z / a.cap) over bf16 logits in place; the padded vocabulary (from a.extent) at -inf.
[[kernel]] void ik_softcap(constant ik_args& a [[buffer(0)]], device bfloat* z [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= a.width || gid.y >= a.rows) return;
  const ulong i = ulong(gid.y) * a.width + gid.x;
  z[i] = gid.x >= a.extent ? bfloat(-INFINITY) : bfloat(a.cap * tanh(float(z[i]) / a.cap));
}

TF_BF16_MATVECS(ik)
