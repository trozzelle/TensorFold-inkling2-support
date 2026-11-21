// Inkling 2's fused decode norm chains: one threadgroup per row runs the RMS, depth-attention, residual and weighted-norm
// row bodies (glue.metal) back to back, so a decode step pays one dispatch where the prompt path pays three to six. Every
// value is computed by the same function, in the same order, as the prompt path's kernels: outputs match their chains bit for bit.
// Threadgroup barriers replace kernel boundaries; a device value another thread wrote is read after a mem_device barrier.
// Follows glue.metal, which it must come after.
#include <metal_stdlib>
using namespace metal;

// The unweighted RMS norms always use 1e-6 (mlx-lm's _unweighted_rms_norm); a.eps is the weighted norms'.
constant float IK_RMS_EPS = 1e-6f;

// The widest row the fused kernels hold in threadgroup memory: the depth output of one query.
constant uint IK_MAXD = 4608;

// The attention sublayer's input chain: the unweighted RMS of the layer input into nrd (the block slot at a block
// boundary), then, for Q queries, the depth attention over the blocks and nrd and its weighted norm into xin[j].
// h: bf16 rows, f32 when IK_F32_IN. nrd rows are a.width apart; blocks slots a.stride rows apart; xin[j] starts at
// j * a.rows * a.width. norm0..norm2 are the queries' bf16 [H] weights; only the first Q are read, the others may be
// bound to any buffer. Q = a.queries = 0 stops after the RMS (query, norms and xin are not read). a.width <= IK_MAXD.
// Threadgroup memory: depth partials 3.8 KB, one query's depth output 18 KB.
[[kernel]] void ik_rdn(constant ik_args& a [[buffer(0)]], const device void* h [[buffer(1)]],
    const device float* blocks [[buffer(2)]], device float* nrd [[buffer(3)]], const device float* query [[buffer(4)]],
    const device bfloat* norm0 [[buffer(5)]], const device bfloat* norm1 [[buffer(6)]], const device bfloat* norm2 [[buffer(7)]],
    device bfloat* xin [[buffer(8)]], IK_ROW) {
  threadgroup float buf[32];
  threadgroup float dbuf[IK_MAXN * IK_MAXQ][32];
  threadgroup float ai[IK_MAXD];
  const uint D = a.width, Q = a.queries;
  const ulong o = ulong(row) * D;
  ik_rms_row(h, a.flags & IK_F32_IN, o, nrd, o, D, IK_RMS_EPS, buf, tid, sg, lane, nt);
  if (Q == 0) return;
  threadgroup_barrier(mem_flags::mem_device);
  float w[IK_MAXN * IK_MAXQ];
  ik_depth_weights(blocks, nrd, query, a.blocks, a.stride, Q, D, a.scale, row, w, dbuf, tid, sg, lane, nt);
  for (uint j = 0; j < Q; ++j) {
    float ss = 0.0f;
    for (uint d = tid; d < D; d += nt) {
      const float v = ik_depth_o(blocks, nrd, w, a.blocks, a.stride, row, D, d, j);
      ai[d] = v;
      ss += v * v;
    }
    const float r = ik_rinv(ik_sum(ss, buf, sg, lane, nt / 32), D, a.eps);
    device bfloat* y = xin + (ulong(j) * a.rows + row) * D;
    const device bfloat* norm = j == 0 ? norm0 : (j == 1 ? norm1 : norm2);
    for (uint d = tid; d < D; d += nt) y[d] = ik_wnorm(ai[d], r, norm[d], true);
  }
}

// The MLP sublayer's input chain: the layer's f32 residual into res (the attention branch alone at IK_BOUNDARY), its
// unweighted RMS into ns, the depth attention (one query) over the K + boundary block slots (K = a.blocks; slot K is
// the one the attention sublayer's RMS wrote at a boundary) and ns, and its weighted norm into xm.
// Rows are a.width apart in every buffer but blocks (a.stride apart). a.width <= IK_MAXD.
[[kernel]] void ik_rrdn(constant ik_args& a [[buffer(0)]], const device bfloat* h [[buffer(1)]],
    const device bfloat* ya [[buffer(2)]], device float* res [[buffer(3)]], const device float* blocks [[buffer(4)]],
    device float* ns [[buffer(5)]], const device float* query [[buffer(6)]], const device bfloat* norm [[buffer(7)]],
    device bfloat* xm [[buffer(8)]], IK_ROW) {
  threadgroup float buf[32];
  threadgroup float dbuf[IK_MAXN * IK_MAXQ][32];
  threadgroup float ai[IK_MAXD];
  const uint D = a.width;
  const ulong o = ulong(row) * D;
  const bool boundary = a.flags & IK_BOUNDARY;
  for (uint d = tid; d < D; d += nt) res[o + d] = ik_resid_one(boundary, h[o + d], ya[o + d]);
  threadgroup_barrier(mem_flags::mem_device);
  ik_rms_row(res, true, o, ns, o, D, IK_RMS_EPS, buf, tid, sg, lane, nt);
  threadgroup_barrier(mem_flags::mem_device);
  const uint K = a.blocks + (boundary ? 1 : 0);
  float w[IK_MAXN * IK_MAXQ];
  ik_depth_weights(blocks, ns, query, K, a.stride, 1, D, a.scale, row, w, dbuf, tid, sg, lane, nt);
  float ss = 0.0f;
  for (uint d = tid; d < D; d += nt) {
    const float v = ik_depth_o(blocks, ns, w, K, a.stride, row, D, d, 0);
    ai[d] = v;
    ss += v * v;
  }
  const float r = ik_rinv(ik_sum(ss, buf, sg, lane, nt / 32), D, a.eps);
  for (uint d = tid; d < D; d += nt) xm[o + d] = ik_wnorm(ai[d], r, norm[d], true);
}
