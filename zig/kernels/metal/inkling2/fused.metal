// Inkling 2's fused decode norm chains: one threadgroup per row runs the RMS, depth-attention, residual and weighted-norm
// chains of the prompt path's kernels (glue.metal) back to back, so a decode step pays one dispatch where the prompt path
// pays three to six. Every value is formed by the same arithmetic, in the same order, as the prompt path's: outputs match
// their chains bit for bit. The threadgroup is 1024 threads; thread t takes the row's columns t, t + 1024, ... and holds them
// in registers, so every device load of a kernel is issued before its first reduction and no value is read back from
// device memory. Each sum adds a thread's terms in column order, takes the simdgroup sum, and adds the 32 simdgroups' sums in
// index order, as ik_sum does. Follows glue.metal, which it must come after.
#include <metal_stdlib>
using namespace metal;

// The unweighted RMS norms always use 1e-6 (mlx-lm's _unweighted_rms_norm); a.eps is the weighted norms'.
constant float IK_RMS_EPS = 1e-6f;

// The widest row the fused kernels take, and the columns a thread holds of it.
constant uint IK_MAXD = 4608;
constant uint IK_PER = (IK_MAXD + 1023) / 1024;

// ik_sum for the fused kernels' 1024 threads: simd_sum, then the 32 simdgroups in index order; every thread gets it. `buf` is
// this sum's own 32 floats, so no second barrier guards its reuse.
inline float ik_sum32(float v, threadgroup float* buf, uint sg, uint lane) {
  v = simd_sum(v);
  if (lane == 0) buf[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint s = 0; s < 32; ++s) t += buf[s];
  return t;
}

// The attention sublayer's input chain: the unweighted RMS of the layer input into nrd (the block slot at a block
// boundary), then, for Q queries, the depth attention over the blocks and nrd and its weighted norm into xin[j].
// h: bf16 rows, f32 when IK_F32_IN. nrd rows are a.width apart; blocks slots a.stride rows apart; xin[j] starts at
// j * a.rows * a.width. norm0..norm2 are the queries' bf16 [H] weights; only the first Q are read, the others may be
// bound to any buffer. Q = a.queries = 0 stops after the RMS (query, norms and xin are not read). a.width <= IK_MAXD.
// ik_rdn takes up to one query (the local layers'), ik_rdn3 up to three (the global layers'): the same body, with fewer
// registers a thread for the one-query instance, which runs the kernel's long chain of loads and sums at least as fast.
template <uint QM>
inline void ik_rdn_run(constant ik_args& a, const device void* h, const device float* blocks, device float* nrd,
    const device float* query, const device bfloat* norm0, const device bfloat* norm1, const device bfloat* norm2,
    device bfloat* xin, threadgroup float (&sums)[1 + QM][32], threadgroup float (&dbuf)[QM][IK_MAXN][33], uint row,
    uint tid, uint sg, uint lane) {
  const uint D = a.width, Q = a.queries, K = a.blocks;
  const ulong o = ulong(row) * D;
  const bool f32in = a.flags & IK_F32_IN;
  float hx[IK_PER] = {}, qv[QM][IK_PER] = {}, bl[IK_MAXN][IK_PER] = {};
  bfloat nw[QM][IK_PER] = {};
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d >= D) continue;
    hx[k] = ik_in(h, f32in, o + d);
    if (Q == 0) continue;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < IK_MAXN - 1; ++i) if (i < K) bl[i][k] = blocks[(ulong(i) * a.stride + row) * D + d];
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < QM; ++j) if (j < Q) {
      qv[j][k] = query[j * D + d];
      nw[j][k] = (j == 0 ? norm0 : (j == 1 ? norm1 : norm2))[d];
    }
  }
  float ss = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) if (tid + 1024 * k < D) ss += hx[k] * hx[k];
  const float r = rsqrt(ik_sum32(ss, sums[0], sg, lane) / float(D) + IK_RMS_EPS);
  float pv[IK_PER] = {};
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d < D) {
      pv[k] = hx[k] * r;
      nrd[o + d] = pv[k];
    }
  }
  if (Q == 0) return;
  // one pass a query: each state's dot with the query, the weights (softmax over the K blocks and the partial state), the
  // depth output (the weighted sum of the states in index order) and its weighted norm. Each pass has its own partial buffer
  _Pragma("clang loop unroll(full)")
  for (uint j = 0; j < QM; ++j) if (j < Q) {
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < IK_MAXN; ++i) if (i <= K) {
      float acc = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < IK_PER; ++k) if (tid + 1024 * k < D) acc += (i < K ? bl[i][k] : pv[k]) * qv[j][k];
      const float v = simd_sum(acc);
      if (lane == 0) dbuf[j][i][sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // lane i of every simdgroup adds the 32 simdgroup sums of state i in index order and hands the result to all its lanes:
    // the sum is the same as every thread adding them all, with 32 loads a thread, not 32 a state
    float tl = 0.0f;
    if (lane <= K) {
      _Pragma("clang loop unroll(full)")
      for (uint s = 0; s < 32; ++s) tl += dbuf[j][lane][s];
      tl *= a.scale;
    }
    float w[IK_MAXN] = {};
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < IK_MAXN; ++i) if (i <= K) w[i] = simd_broadcast(tl, ushort(i));
    float m = -INFINITY;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < IK_MAXN; ++i) if (i <= K) m = max(m, w[i]);
    float z = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < IK_MAXN; ++i) if (i <= K) {
      w[i] = exp(w[i] - m);
      z += w[i];
    }
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < IK_MAXN; ++i) if (i <= K) w[i] /= z;
    float ai[IK_PER] = {}, ssj = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < IK_PER; ++k) if (tid + 1024 * k < D) {
      float v = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < IK_MAXN; ++i) if (i <= K) v += w[i] * (i < K ? bl[i][k] : pv[k]);
      ai[k] = v;
      ssj += v * v;
    }
    const float rj = ik_rinv(ik_sum32(ssj, sums[1 + j], sg, lane), D, a.eps);
    device bfloat* y = xin + (ulong(j) * a.rows + row) * D;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < IK_PER; ++k) if (tid + 1024 * k < D) y[tid + 1024 * k] = ik_wnorm(ai[k], rj, nw[j][k], true);
  }
}

[[kernel, max_total_threads_per_threadgroup(1024)]] void ik_rdn(constant ik_args& a [[buffer(0)]],
    const device void* h [[buffer(1)]], const device float* blocks [[buffer(2)]], device float* nrd [[buffer(3)]],
    const device float* query [[buffer(4)]], const device bfloat* norm0 [[buffer(5)]], const device bfloat* norm1 [[buffer(6)]],
    const device bfloat* norm2 [[buffer(7)]], device bfloat* xin [[buffer(8)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sums[2][32];
  threadgroup float dbuf[1][IK_MAXN][33]; // rows a bank apart, so the lanes' reads of them do not collide
  ik_rdn_run<1>(a, h, blocks, nrd, query, norm0, norm1, norm2, xin, sums, dbuf, row, tid, sg, lane);
}

[[kernel, max_total_threads_per_threadgroup(1024)]] void ik_rdn3(constant ik_args& a [[buffer(0)]],
    const device void* h [[buffer(1)]], const device float* blocks [[buffer(2)]], device float* nrd [[buffer(3)]],
    const device float* query [[buffer(4)]], const device bfloat* norm0 [[buffer(5)]], const device bfloat* norm1 [[buffer(6)]],
    const device bfloat* norm2 [[buffer(7)]], device bfloat* xin [[buffer(8)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sums[4][32];
  threadgroup float dbuf[3][IK_MAXN][33];
  ik_rdn_run<3>(a, h, blocks, nrd, query, norm0, norm1, norm2, xin, sums, dbuf, row, tid, sg, lane);
}

// The MLP sublayer's input chain: the layer's f32 residual into res (the attention branch alone at IK_BOUNDARY), its
// unweighted RMS into ns, the depth attention (one query) over the K + boundary block slots (K = a.blocks; slot K is
// the one the attention sublayer's RMS wrote at a boundary) and ns, and its weighted norm into xm.
// Rows are a.width apart in every buffer but blocks (a.stride apart). a.width <= IK_MAXD.
[[kernel, max_total_threads_per_threadgroup(1024)]] void ik_rrdn(constant ik_args& a [[buffer(0)]],
    const device bfloat* h [[buffer(1)]], const device bfloat* ya [[buffer(2)]], device float* res [[buffer(3)]],
    const device float* blocks [[buffer(4)]], device float* ns [[buffer(5)]], const device float* query [[buffer(6)]],
    const device bfloat* norm [[buffer(7)]], device bfloat* xm [[buffer(8)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sums[2][32];
  threadgroup float dbuf[IK_MAXN][33]; // rows a bank apart, so the lanes' reads below do not collide
  const uint D = a.width;
  const ulong o = ulong(row) * D;
  const bool boundary = a.flags & IK_BOUNDARY;
  const uint K = a.blocks + (boundary ? 1 : 0);
  float qv[IK_PER] = {}, bl[IK_MAXN][IK_PER] = {};
  bfloat hb[IK_PER] = {}, yb[IK_PER] = {}, nw[IK_PER] = {};
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d >= D) continue;
    hb[k] = h[o + d];
    yb[k] = ya[o + d];
    qv[k] = query[d];
    nw[k] = norm[d];
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < IK_MAXN - 1; ++i) if (i < K) bl[i][k] = blocks[(ulong(i) * a.stride + row) * D + d];
  }
  float rs[IK_PER] = {}, ss = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d < D) {
      rs[k] = ik_resid_one(boundary, hb[k], yb[k]);
      res[o + d] = rs[k];
      ss += rs[k] * rs[k];
    }
  }
  const float r = rsqrt(ik_sum32(ss, sums[0], sg, lane) / float(D) + IK_RMS_EPS);
  float pv[IK_PER] = {};
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d < D) {
      pv[k] = rs[k] * r;
      ns[o + d] = pv[k];
    }
  }
  // each state's dot with the query, then the weights: softmax over the K blocks and the partial state
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < IK_MAXN; ++i) if (i <= K) {
    float acc = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < IK_PER; ++k) if (tid + 1024 * k < D) acc += (i < K ? bl[i][k] : pv[k]) * qv[k];
    const float v = simd_sum(acc);
    if (lane == 0) dbuf[i][sg] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // lane i of every simdgroup adds the 32 simdgroup sums of state i in index order and hands the result to all its lanes
  float tl = 0.0f;
  if (lane <= K) {
    _Pragma("clang loop unroll(full)")
    for (uint s = 0; s < 32; ++s) tl += dbuf[lane][s];
    tl *= a.scale;
  }
  float w[IK_MAXN] = {};
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < IK_MAXN; ++i) if (i <= K) w[i] = simd_broadcast(tl, ushort(i));
  float m = -INFINITY;
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < IK_MAXN; ++i) if (i <= K) m = max(m, w[i]);
  float z = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < IK_MAXN; ++i) if (i <= K) {
    w[i] = exp(w[i] - m);
    z += w[i];
  }
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < IK_MAXN; ++i) if (i <= K) w[i] /= z;
  // the depth output, the weighted sum of the states in index order, and its weighted norm
  float ai[IK_PER] = {}, ss2 = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) if (tid + 1024 * k < D) {
    float v = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < IK_MAXN; ++i) if (i <= K) v += w[i] * (i < K ? bl[i][k] : pv[k]);
    ai[k] = v;
    ss2 += v * v;
  }
  const float r2 = ik_rinv(ik_sum32(ss2, sums[1], sg, lane), D, a.eps);
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) if (tid + 1024 * k < D) xm[o + tid + 1024 * k] = ik_wnorm(ai[k], r2, nw[k], true);
}
