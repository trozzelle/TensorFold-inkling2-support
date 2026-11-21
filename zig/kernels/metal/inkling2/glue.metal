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

// embed_norm(embed_tokens(ids)): the weighted RMSNorm of the bf16 rows; as mlx's kernel in bf16, x * rsqrt rounds to bf16
// before the weight multiplies it, then the product rounds again.
[[kernel]] void ik_embed(constant ik_args& a [[buffer(0)]], const device uint* ids [[buffer(1)]],
    const device bfloat* table [[buffer(2)]], const device bfloat* w [[buffer(3)]], device bfloat* y [[buffer(4)]], IK_ROW) {
  threadgroup float buf[32];
  const device bfloat* x = table + ulong(ids[row]) * a.width;
  float ss = 0.0f;
  for (uint d = tid; d < a.width; d += nt) { const float v = float(x[d]); ss += v * v; }
  const float r = rsqrt(ik_sum(ss, buf, sg, lane, nt / 32) / float(a.width) + a.eps);
  for (uint d = tid; d < a.width; d += nt) y[ulong(row) * a.width + d] = bfloat(float(w[d]) * float(bfloat(float(x[d]) * r)));
}

// The row bodies below are shared by the prompt path's kernels and the fused decode kernels (fused.metal): one source of truth, so
// the prompt path and the decode path round alike.

// _unweighted_rms_norm of the D values at x[xo..] into f32 y[yo..]; every thread of the threadgroup takes part.
inline void ik_rms_row(const device void* x, bool f, ulong xo, device float* y, ulong yo, uint D, float eps,
    threadgroup float* buf, uint tid, uint sg, uint lane, uint nt) {
  float ss = 0.0f;
  for (uint d = tid; d < D; d += nt) { const float v = ik_in(x, f, xo + d); ss += v * v; }
  const float r = rsqrt(ik_sum(ss, buf, sg, lane, nt / 32) / float(D) + eps);
  for (uint d = tid; d < D; d += nt) y[yo + d] = ik_in(x, f, xo + d) * r;
}

// The inverse RMS of a row from its sum of squares.
inline float ik_rinv(float ss, uint D, float eps) { return rsqrt(ss / float(D) + eps); }

// One output of nn.RMSNorm: w * (x * r). A bf16 input rounds x * r to bf16 before the weight (mlx's kernel works in the
// input's type); an f32 input rounds once, at the end.
inline bfloat ik_wnorm(float x, float r, bfloat w, bool f32in) {
  const float u = x * r;
  return bfloat(float(w) * (f32in ? u : float(bfloat(u))));
}

// nn.RMSNorm of the D values at x[xo..] into bf16 y[yo..].
inline void ik_norm_row(const device void* x, bool f, ulong xo, const device bfloat* w, device bfloat* y, ulong yo, uint D,
    float eps, threadgroup float* buf, uint tid, uint sg, uint lane, uint nt) {
  float ss = 0.0f;
  for (uint d = tid; d < D; d += nt) { const float v = ik_in(x, f, xo + d); ss += v * v; }
  const float r = ik_rinv(ik_sum(ss, buf, sg, lane, nt / 32), D, eps);
  for (uint d = tid; d < D; d += nt) y[yo + d] = ik_wnorm(ik_in(x, f, xo + d), r, w[d], f);
}

// _unweighted_rms_norm (eps 1e-6) of bf16 or f32 rows into f32 rows.
[[kernel]] void ik_rms(constant ik_args& a [[buffer(0)]], const device void* x [[buffer(1)]], device float* y [[buffer(2)]], IK_ROW) {
  threadgroup float buf[32];
  const ulong o = ulong(row) * a.width;
  ik_rms_row(x, a.flags & IK_F32_IN, o, y, o, a.width, a.eps, buf, tid, sg, lane, nt);
}

// nn.RMSNorm of bf16 or f32 rows (a.stride elements apart) into bf16 rows: w * (x * rsqrt(mean x^2 + eps)). A bf16 row
// rounds x * rsqrt to bf16 before the weight (mlx's kernel works in the input's type); an f32 row rounds once, at the end.
[[kernel]] void ik_norm(constant ik_args& a [[buffer(0)]], const device void* x [[buffer(1)]],
    const device bfloat* w [[buffer(2)]], device bfloat* y [[buffer(3)]], IK_ROW) {
  threadgroup float buf[32];
  ik_norm_row(x, a.flags & IK_F32_IN, ulong(row) * a.stride, w, y, ulong(row) * a.width, a.width, a.eps, buf, tid, sg, lane, nt);
}

constant uint IK_MAXN = 10, IK_MAXQ = 3;

// BlockAttentionResidual, first half: the softmax weights w[i * IK_MAXQ + j] of the Q learned queries over the `nblocks`
// block states (rows `stride` apart) and the partial state, from dot * scale over the sources. buf holds the per-simdgroup
// dot partials; every thread of the threadgroup gets the same weights.
inline void ik_depth_weights(const device float* blocks, const device float* partial, const device float* q, uint nblocks,
    uint stride, uint Q, uint D, float scale, uint row, thread float* w, threadgroup float (*buf)[32], uint tid, uint sg,
    uint lane, uint nt) {
  const uint N = nblocks + 1;
  float acc[IK_MAXN * IK_MAXQ];
  for (uint i = 0; i < IK_MAXN * IK_MAXQ; ++i) acc[i] = 0.0f;
  for (uint d = tid; d < D; d += nt) {
    for (uint i = 0; i < N; ++i) {
      const float x = i < nblocks ? blocks[(ulong(i) * stride + row) * D + d] : partial[ulong(row) * D + d];
      for (uint j = 0; j < Q; ++j) acc[i * IK_MAXQ + j] += x * q[j * D + d];
    }
  }
  for (uint i = 0; i < N; ++i) for (uint j = 0; j < Q; ++j) {
    const float v = simd_sum(acc[i * IK_MAXQ + j]);
    if (lane == 0) buf[i * IK_MAXQ + j][sg] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = 0; i < N; ++i) for (uint j = 0; j < Q; ++j) {
    float t = 0.0f;
    for (uint s = 0; s < nt / 32; ++s) t += buf[i * IK_MAXQ + j][s];
    w[i * IK_MAXQ + j] = t * scale;
  }
  for (uint j = 0; j < Q; ++j) {
    float m = -INFINITY;
    for (uint i = 0; i < N; ++i) m = max(m, w[i * IK_MAXQ + j]);
    float z = 0.0f;
    for (uint i = 0; i < N; ++i) { w[i * IK_MAXQ + j] = exp(w[i * IK_MAXQ + j] - m); z += w[i * IK_MAXQ + j]; }
    for (uint i = 0; i < N; ++i) w[i * IK_MAXQ + j] /= z;
  }
}

// BlockAttentionResidual, second half: output element d of query j, the weighted sum of the sources in index order.
inline float ik_depth_o(const device float* blocks, const device float* partial, thread const float* w, uint nblocks,
    uint stride, uint row, uint D, uint d, uint j) {
  float o = 0.0f;
  for (uint i = 0; i < nblocks + 1; ++i) {
    const float x = i < nblocks ? blocks[(ulong(i) * stride + row) * D + d] : partial[ulong(row) * D + d];
    o += w[i * IK_MAXQ + j] * x;
  }
  return o;
}

// BlockAttentionResidual, as mlx-lm's depth_attention kernel: Q learned queries over the a.blocks block states (rows
// a.stride apart) and the partial state; softmax of dot * a.scale over the sources, then their weighted sum.
[[kernel]] void ik_depth(constant ik_args& a [[buffer(0)]], const device float* blocks [[buffer(1)]],
    const device float* partial [[buffer(2)]], const device float* q [[buffer(3)]], device float* out [[buffer(4)]], IK_ROW) {
  threadgroup float buf[IK_MAXN * IK_MAXQ][32];
  const uint Q = a.queries, D = a.width;
  float w[IK_MAXN * IK_MAXQ];
  ik_depth_weights(blocks, partial, q, a.blocks, a.stride, Q, D, a.scale, row, w, buf, tid, sg, lane, nt);
  for (uint d = tid; d < D; d += nt)
    for (uint j = 0; j < Q; ++j) out[(ulong(row) * Q + j) * D + d] = ik_depth_o(blocks, partial, w, a.blocks, a.stride, row, D, d, j);
}

// One element of the layer's f32 residual: the attention branch alone at a block boundary, else the layer input plus it.
inline float ik_resid_one(bool boundary, bfloat h, bfloat ya) { return (boundary ? 0.0f : float(h)) + float(ya); }

// The layer's f32 residual: the attention branch alone at a block boundary, else the layer input plus it.
[[kernel]] void ik_resid(constant ik_args& a [[buffer(0)]], const device bfloat* h [[buffer(1)]],
    const device bfloat* ya [[buffer(2)]], device float* res [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= a.width || gid.y >= a.rows) return;
  const ulong i = ulong(gid.y) * a.width + gid.x;
  res[i] = ik_resid_one(a.flags & IK_BOUNDARY, h[i], ya[i]);
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
