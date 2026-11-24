// Looped-core cross-pass depth attention and rank-R MessageGate controllers at mlx-lm's rounding points.
// Follows glue.metal and fused.metal.
#include <metal_stdlib>
using namespace metal;

constant uint IK_MSG_MAXN = 4, IK_MSG_MAXQ = 4, IK_GATE_MAXR = 128;

// Copy a.rows x a.width f32 values (prelude block save/restore and cross-pass bank capture).
[[kernel]] void ik_copy_f32(constant ik_args& a [[buffer(0)]], const device float* src [[buffer(1)]],
    device float* dst [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= a.width || gid.y >= a.rows) return;
  const ulong i = ulong(gid.y) * a.width + gid.x;
  dst[i] = src[i];
}

// Cross-pass depth attention: Q = 4 queries (q, k, v, mlp) over a.blocks bank blocks (a.stride rows apart) and the
// bank partial state into out [rows, 4, width] f32.
[[kernel, max_total_threads_per_threadgroup(1024)]] void ik_msg_depth(constant ik_args& a [[buffer(0)]],
    const device float* blocks [[buffer(1)]], const device float* partial [[buffer(2)]],
    const device float* q [[buffer(3)]], device float* out [[buffer(4)]], IK_ROW) {
  threadgroup float buf[IK_MSG_MAXN * IK_MSG_MAXQ][32];
  const uint N = a.blocks + 1, Q = a.queries, D = a.width;
  float acc[IK_MSG_MAXN * IK_MSG_MAXQ];
  for (uint i = 0; i < IK_MSG_MAXN * IK_MSG_MAXQ; ++i) acc[i] = 0.0f;
  for (uint d = tid; d < D; d += nt) {
    for (uint i = 0; i < N; ++i) {
      const float x = i < a.blocks ? blocks[(ulong(i) * a.stride + row) * D + d] : partial[ulong(row) * D + d];
      for (uint j = 0; j < Q; ++j) acc[i * IK_MSG_MAXQ + j] += x * q[j * D + d];
    }
  }
  for (uint i = 0; i < N; ++i) for (uint j = 0; j < Q; ++j) {
    const float v = simd_sum(acc[i * IK_MSG_MAXQ + j]);
    if (lane == 0) buf[i * IK_MSG_MAXQ + j][sg] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float w[IK_MSG_MAXN * IK_MSG_MAXQ];
  for (uint i = 0; i < N; ++i) for (uint j = 0; j < Q; ++j) {
    float t = 0.0f;
    for (uint s = 0; s < nt / 32; ++s) t += buf[i * IK_MSG_MAXQ + j][s];
    w[i * IK_MSG_MAXQ + j] = t * a.scale;
  }
  for (uint j = 0; j < Q; ++j) {
    float m = -INFINITY;
    for (uint i = 0; i < N; ++i) m = max(m, w[i * IK_MSG_MAXQ + j]);
    float z = 0.0f;
    for (uint i = 0; i < N; ++i) { w[i * IK_MSG_MAXQ + j] = exp(w[i * IK_MSG_MAXQ + j] - m); z += w[i * IK_MSG_MAXQ + j]; }
    for (uint i = 0; i < N; ++i) w[i * IK_MSG_MAXQ + j] /= z;
  }
  for (uint d = tid; d < D; d += nt) {
    for (uint j = 0; j < Q; ++j) {
      float o = 0.0f;
      for (uint i = 0; i < N; ++i) {
        const float x = i < a.blocks ? blocks[(ulong(i) * a.stride + row) * D + d] : partial[ulong(row) * D + d];
        o += w[i * IK_MSG_MAXQ + j] * x;
      }
      out[(ulong(row) * Q + j) * D + d] = o;
    }
  }
}

// MessageGate input prep: bf16(concat(unweighted_rms_norm(h, 1e-6), unweighted_rms_norm(msg, 1e-6))).
// h rows are a.stride elements apart; msg rows are 4 * a.width elements apart (caller binds msg at slot * a.width).
[[kernel, max_total_threads_per_threadgroup(1024)]] void ik_msggate_prep(constant ik_args& a [[buffer(0)]],
    const device float* h [[buffer(1)]], const device float* msg [[buffer(2)]], device bfloat* out [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sums[2][32];
  const uint D = a.width;
  const ulong ho = ulong(row) * a.stride, mo = ulong(row) * 4 * D, yo = ulong(row) * 2 * D;
  float hv[IK_PER] = {}, mv[IK_PER] = {};
  float ssh = 0.0f, ssm = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d < D) {
      hv[k] = h[ho + d];
      mv[k] = msg[mo + d];
      ssh += hv[k] * hv[k];
      ssm += mv[k] * mv[k];
    }
  }
  const float rh = rsqrt(ik_sum32(ssh, sums[0], sg, lane) / float(D) + IK_RMS_EPS);
  const float rm = rsqrt(ik_sum32(ssm, sums[1], sg, lane) / float(D) + IK_RMS_EPS);
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d < D) {
      out[yo + d] = bfloat(hv[k] * rh);
      out[yo + D + d] = bfloat(mv[k] * rm);
    }
  }
}

// MessageGate forget/gain + blend + trailing weighted RMSNorm into bf16.
// z is bf16 [rows, a.dim] widened to f32; forget_w and gain_w are bf16 [D, a.dim] widened to f32 in registers.
[[kernel, max_total_threads_per_threadgroup(1024)]] void ik_msggate_apply_norm(constant ik_args& a [[buffer(0)]],
    const device float* h [[buffer(1)]], const device float* msg [[buffer(2)]], const device bfloat* z [[buffer(3)]],
    const device bfloat* forget_w [[buffer(4)]], const device bfloat* gain_w [[buffer(5)]],
    const device bfloat* norm_w [[buffer(6)]], device bfloat* out [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float zs[IK_GATE_MAXR];
  threadgroup float sums[32];
  const uint D = a.width, R = a.dim;
  if (tid < R) zs[tid] = float(z[ulong(row) * R + tid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const ulong ho = ulong(row) * a.stride, mo = ulong(row) * 4 * D, yo = ulong(row) * D;
  float bv[IK_PER] = {};
  float ss = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d < D) {
      const device bfloat* fw = forget_w + ulong(d) * R;
      const device bfloat* gw = gain_w + ulong(d) * R;
      float df = 0.0f, dg = 0.0f;
      for (uint r = 0; r < R; ++r) {
        const float zr = zs[r];
        df = fma(zr, float(fw[r]), df);
        dg = fma(zr, float(gw[r]), dg);
      }
      const float v = (1.0f - tanh(df)) * h[ho + d] + tanh(dg) * msg[mo + d];
      bv[k] = v;
      ss += v * v;
    }
  }
  const float rinv = ik_rinv(ik_sum32(ss, sums, sg, lane), D, a.eps);
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d < D) out[yo + d] = ik_wnorm(bv[k], rinv, norm_w[d], true);
  }
}

// 3-slot MessageGate input prep for decode: grid (rows, 3, 1), threadgroup (1024, 1, 1).
// Writes out as [3][rows][2 * D] bf16, bit-identical to 3 calls of ik_msggate_prep.
[[kernel, max_total_threads_per_threadgroup(1024)]] void ik_msggate_prep3(constant ik_args& a [[buffer(0)]],
    const device float* h [[buffer(1)]], const device float* msg [[buffer(2)]], device bfloat* out [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sums[2][32];
  const uint row = tg.x, slot = tg.y, D = a.width;
  const bool glob = (a.flags & IK_GLOBAL) != 0;
  const ulong ho = ulong(row) * a.stride + (glob ? ulong(slot) * D : 0);
  const ulong mo = (ulong(row) * 4 + slot) * D;
  const ulong yo = (ulong(slot) * a.rows + row) * 2 * D;
  float hv[IK_PER] = {}, mv[IK_PER] = {};
  float ssh = 0.0f, ssm = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d < D) {
      hv[k] = h[ho + d];
      mv[k] = msg[mo + d];
      ssh += hv[k] * hv[k];
      ssm += mv[k] * mv[k];
    }
  }
  const float rh = rsqrt(ik_sum32(ssh, sums[0], sg, lane) / float(D) + IK_RMS_EPS);
  const float rm = rsqrt(ik_sum32(ssm, sums[1], sg, lane) / float(D) + IK_RMS_EPS);
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d < D) {
      out[yo + d] = bfloat(hv[k] * rh);
      out[yo + D + d] = bfloat(mv[k] * rm);
    }
  }
}

// 3-slot MessageGate forget/gain + blend + trailing weighted RMSNorm for decode: grid (rows, 3, 1).
// Reads z [3][rows][R] bf16 and writes out [3][rows][D] bf16, bit-identical to 3 calls of ik_msggate_apply_norm.
[[kernel, max_total_threads_per_threadgroup(1024)]] void ik_msggate_apply_norm3(constant ik_args& a [[buffer(0)]],
    const device float* h [[buffer(1)]], const device float* msg [[buffer(2)]], const device bfloat* z [[buffer(3)]],
    const device bfloat* fw0 [[buffer(4)]], const device bfloat* fw1 [[buffer(5)]], const device bfloat* fw2 [[buffer(6)]],
    const device bfloat* gw0 [[buffer(7)]], const device bfloat* gw1 [[buffer(8)]], const device bfloat* gw2 [[buffer(9)]],
    const device bfloat* nw0 [[buffer(10)]], const device bfloat* nw1 [[buffer(11)]], const device bfloat* nw2 [[buffer(12)]],
    device bfloat* out [[buffer(13)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float zs[IK_GATE_MAXR];
  threadgroup float sums[32];
  const uint row = tg.x, slot = tg.y, D = a.width, R = a.dim;
  const bool glob = (a.flags & IK_GLOBAL) != 0;
  const device bfloat* forget_w = slot == 0 ? fw0 : (slot == 1 ? fw1 : fw2);
  const device bfloat* gain_w = slot == 0 ? gw0 : (slot == 1 ? gw1 : gw2);
  const device bfloat* norm_w = !glob || slot == 0 ? nw0 : (slot == 1 ? nw1 : nw2);
  if (tid < R) zs[tid] = float(z[(ulong(slot) * a.rows + row) * R + tid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const ulong ho = ulong(row) * a.stride + (glob ? ulong(slot) * D : 0);
  const ulong mo = (ulong(row) * 4 + slot) * D;
  const ulong yo = (ulong(slot) * a.rows + row) * D;
  float bv[IK_PER] = {};
  float ss = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d < D) {
      const device bfloat* fw = forget_w + ulong(d) * R;
      const device bfloat* gw = gain_w + ulong(d) * R;
      float df = 0.0f, dg = 0.0f;
      for (uint r = 0; r < R; ++r) {
        const float zr = zs[r];
        df = fma(zr, float(fw[r]), df);
        dg = fma(zr, float(gw[r]), dg);
      }
      const float v = (1.0f - tanh(df)) * h[ho + d] + tanh(dg) * msg[mo + d];
      bv[k] = v;
      ss += v * v;
    }
  }
  const float rinv = ik_rinv(ik_sum32(ss, sums, sg, lane), D, a.eps);
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d < D) out[yo + d] = ik_wnorm(bv[k], rinv, norm_w[d], true);
  }
}
