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

// MessageGate blend + trailing weighted RMSNorm from precomputed f32 df and dg into bf16: grid (rows, 1, 1) x 1024.
// h rows are a.stride elements apart; msg rows are 4 * a.width elements apart (caller binds msg at slot * a.width).
[[kernel, max_total_threads_per_threadgroup(1024)]] void ik_msggate_blend_norm(constant ik_args& a [[buffer(0)]],
    const device float* h [[buffer(1)]], const device float* msg [[buffer(2)]], const device float* df [[buffer(3)]],
    const device float* dg [[buffer(4)]], const device bfloat* norm_w [[buffer(5)]], device bfloat* out [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sums[32];
  const uint D = a.width;
  const ulong ho = ulong(row) * a.stride, mo = ulong(row) * 4 * D, yo = ulong(row) * D;
  float bv[IK_PER] = {};
  float ss = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d < D) {
      const float v = (1.0f - tanh(df[yo + d])) * h[ho + d] + tanh(dg[yo + d]) * msg[mo + d];
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

// Coalesced multi-TG decode forget/gain + blend over one slab of up to 16 rows into f32 out_v.
// Each 8 lanes compute one output column d with 128-byte coalesced weight reads and fixed 8-lane xor reduction.
template <uint ROWS>
inline void ik_msggate_fg_slab(const device float* h, const device float* msg, const threadgroup float* zs,
    const device bfloat* forget_w, const device bfloat* gain_w, device float* out_v, uint d, uint D, uint R,
    uint row_base, uint slab_rows, uint total_rows, uint h_stride, uint h_slot, uint msg_slot, uint out_slot,
    uint l8) {
  float df[ROWS], dg[ROWS];
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < ROWS; ++r) {
    df[r] = 0.0f;
    dg[r] = 0.0f;
  }
  if (d < D) {
    const device bfloat* fw = forget_w + ulong(d) * R;
    const device bfloat* gw = gain_w + ulong(d) * R;
    for (uint r0 = l8 * 8u; r0 < R; r0 += 64u) {
      const float4 fw0 = float4(*(const device bfloat4*)(fw + r0));
      const float4 fw1 = float4(*(const device bfloat4*)(fw + r0 + 4u));
      const float4 gw0 = float4(*(const device bfloat4*)(gw + r0));
      const float4 gw1 = float4(*(const device bfloat4*)(gw + r0 + 4u));
      _Pragma("clang loop unroll(full)")
      for (uint r = 0; r < ROWS; ++r) {
        if (r < slab_rows) {
          const threadgroup float* zr = zs + r * R + r0;
          const float4 z0 = *(const threadgroup float4*)zr;
          const float4 z1 = *(const threadgroup float4*)(zr + 4u);
          df[r] += dot(fw0, z0) + dot(fw1, z1);
          dg[r] += dot(gw0, z0) + dot(gw1, z1);
        }
      }
    }
  }
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < ROWS; ++r) {
    float f = df[r];
    f += simd_shuffle_xor(f, 4);
    f += simd_shuffle_xor(f, 2);
    f += simd_shuffle_xor(f, 1);
    float g = dg[r];
    g += simd_shuffle_xor(g, 4);
    g += simd_shuffle_xor(g, 2);
    g += simd_shuffle_xor(g, 1);
    if (d < D && l8 == 0 && r < slab_rows) {
      const uint row = row_base + r;
      const ulong ho = ulong(row) * h_stride + ulong(h_slot) * D;
      const ulong mo = (ulong(row) * 4u + msg_slot) * D;
      const float v = (1.0f - tanh(f)) * h[ho + d] + tanh(g) * msg[mo + d];
      out_v[(ulong(out_slot) * total_rows + row) * D + d] = v;
    }
  }
}

// 1-slot coalesced decode forget/gain + blend into f32 out_v [rows, D]: grid (ceil(D/32), ceil(rows/16), 1) x 256.
// Requires R = a.dim <= IK_GATE_MAXR and R % 8 == 0. Caller binds h at h_slot * D and msg at msg_slot * D.
[[kernel, max_total_threads_per_threadgroup(256)]] void ik_msggate_fg_decode(constant ik_args& a [[buffer(0)]],
    const device float* h [[buffer(1)]], const device float* msg [[buffer(2)]], const device bfloat* z [[buffer(3)]],
    const device bfloat* forget_w [[buffer(4)]], const device bfloat* gain_w [[buffer(5)]],
    device float* out_v [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float zs[16 * IK_GATE_MAXR];
  const uint row_base = tg.y * 16u;
  if (row_base >= a.rows) return;
  const uint slab_rows = min(16u, a.rows - row_base);
  const uint D = a.width, R = a.dim;
  for (uint i = tid; i < slab_rows * R; i += 256u) {
    const uint r = i / R;
    const uint c = i - r * R;
    zs[i] = float(z[ulong(row_base + r) * R + c]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint d = tg.x * 32u + sg * 4u + (lane >> 3);
  const uint l8 = lane & 7u;
  if (slab_rows == 1u) {
    ik_msggate_fg_slab<1>(h, msg, zs, forget_w, gain_w, out_v, d, D, R, row_base, slab_rows, a.rows, a.stride, 0u, 0u,
        0u, l8);
  } else if (slab_rows <= 8u) {
    ik_msggate_fg_slab<8>(h, msg, zs, forget_w, gain_w, out_v, d, D, R, row_base, slab_rows, a.rows, a.stride, 0u, 0u,
        0u, l8);
  } else {
    ik_msggate_fg_slab<16>(h, msg, zs, forget_w, gain_w, out_v, d, D, R, row_base, slab_rows, a.rows, a.stride, 0u, 0u,
        0u, l8);
  }
}

// 3-slot coalesced decode forget/gain + blend into f32 out_v [3][rows][D]:
// grid ((D + 31) / 32, (rows + 15) / 16, 3) x 256. Requires R = a.dim <= IK_GATE_MAXR and R % 8 == 0.
[[kernel, max_total_threads_per_threadgroup(256)]] void ik_msggate_fg_decode3(constant ik_args& a [[buffer(0)]],
    const device float* h [[buffer(1)]], const device float* msg [[buffer(2)]], const device bfloat* z [[buffer(3)]],
    const device bfloat* fw0 [[buffer(4)]], const device bfloat* fw1 [[buffer(5)]], const device bfloat* fw2 [[buffer(6)]],
    const device bfloat* gw0 [[buffer(7)]], const device bfloat* gw1 [[buffer(8)]], const device bfloat* gw2 [[buffer(9)]],
    device float* out_v [[buffer(10)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float zs[16 * IK_GATE_MAXR];
  const uint row_base = tg.y * 16u, slot = tg.z;
  if (row_base >= a.rows) return;
  const uint slab_rows = min(16u, a.rows - row_base);
  const uint D = a.width, R = a.dim;
  const bool glob = (a.flags & IK_GLOBAL) != 0;
  const device bfloat* forget_w = slot == 0 ? fw0 : (slot == 1 ? fw1 : fw2);
  const device bfloat* gain_w = slot == 0 ? gw0 : (slot == 1 ? gw1 : gw2);
  const device bfloat* z_slot = z + ulong(slot) * a.rows * R;
  for (uint i = tid; i < slab_rows * R; i += 256u) {
    const uint r = i / R;
    const uint c = i - r * R;
    zs[i] = float(z_slot[ulong(row_base + r) * R + c]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint d = tg.x * 32u + sg * 4u + (lane >> 3);
  const uint l8 = lane & 7u;
  const uint h_slot = glob ? slot : 0u;
  if (slab_rows == 1u) {
    ik_msggate_fg_slab<1>(h, msg, zs, forget_w, gain_w, out_v, d, D, R, row_base, slab_rows, a.rows, a.stride, h_slot,
        slot, slot, l8);
  } else if (slab_rows <= 8u) {
    ik_msggate_fg_slab<8>(h, msg, zs, forget_w, gain_w, out_v, d, D, R, row_base, slab_rows, a.rows, a.stride, h_slot,
        slot, slot, l8);
  } else {
    ik_msggate_fg_slab<16>(h, msg, zs, forget_w, gain_w, out_v, d, D, R, row_base, slab_rows, a.rows, a.stride, h_slot,
        slot, slot, l8);
  }
}

// 3-slot trailing weighted RMSNorm from f32 x [3][rows][D] into bf16 out [3][rows][D]: grid (rows, 3, 1) x 1024.
[[kernel, max_total_threads_per_threadgroup(1024)]] void ik_norm3_f32(constant ik_args& a [[buffer(0)]],
    const device float* x [[buffer(1)]], const device bfloat* nw0 [[buffer(2)]], const device bfloat* nw1 [[buffer(3)]],
    const device bfloat* nw2 [[buffer(4)]], device bfloat* out [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sums[32];
  const uint row = tg.x, slot = tg.y, D = a.width;
  const bool glob = (a.flags & IK_GLOBAL) != 0;
  const device bfloat* nw = !glob || slot == 0 ? nw0 : (slot == 1 ? nw1 : nw2);
  const ulong o = (ulong(slot) * a.rows + row) * D;
  float xv[IK_PER] = {};
  float ss = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d < D) {
      xv[k] = x[o + d];
      ss += xv[k] * xv[k];
    }
  }
  const float rinv = ik_rinv(ik_sum32(ss, sums, sg, lane), D, a.eps);
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d < D) out[o + d] = ik_wnorm(xv[k], rinv, nw[d], true);
  }
}

// Fused looped-core attention prelude: msg_depth (4 queries) + rms(h -> nrd) + depth(blocks, nrd -> ai) +
// msggate_prep3 -> ctrl_in [3][rows][2 * D] bf16. Bit-identical to the 4-kernel chain.
// Args: rows, width (D <= IK_MAXD), dim (bank_blocks), blocks (K), queries (Q = 1 or 3), stride, scale, flags.
template <uint QM>
inline void ik_loop_attn_prep_run(constant ik_args& a, const device void* h, const device float* bank_blocks,
    const device float* bank_partial, const device float* msg_q, const device float* blocks,
    const device float* attn_q, device float* msg, device float* nrd, device float* ai, device bfloat* ctrl_in,
    threadgroup float (&msums)[3][32], threadgroup float (&mbuf)[IK_MSG_MAXQ][IK_MSG_MAXN][33],
    threadgroup float (&sums)[1 + QM][32], threadgroup float (&dbuf)[QM][IK_MAXN][33], uint row, uint tid, uint sg,
    uint lane) {
  const uint D = a.width, B = a.dim, K = a.blocks, Q = a.queries;
  const ulong o = ulong(row) * D;
  const bool f32in = (a.flags & IK_F32_IN) != 0;

  // Phase 1: cross-pass depth attention into msg [rows, 4, D], plus msg-half of ctrl_in for slots 0..2.
  {
    float bm[IK_MSG_MAXN][IK_PER] = {};
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < IK_PER; ++k) {
      const uint d = tid + 1024 * k;
      if (d >= D) continue;
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < IK_MSG_MAXN; ++i) {
        if (i <= B) bm[i][k] = i < B ? bank_blocks[(ulong(i) * a.stride + row) * D + d] : bank_partial[o + d];
      }
    }
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < IK_MSG_MAXQ; ++j) {
      float mqv[IK_PER] = {};
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < IK_PER; ++k) {
        const uint d = tid + 1024 * k;
        if (d < D) mqv[k] = msg_q[j * D + d];
      }
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < IK_MSG_MAXN; ++i) if (i <= B) {
        float acc = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < IK_PER; ++k) if (tid + 1024 * k < D) acc += bm[i][k] * mqv[k];
        const float v = simd_sum(acc);
        if (lane == 0) mbuf[j][i][sg] = v;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float tl = 0.0f;
      if (lane <= B) {
        _Pragma("clang loop unroll(full)")
        for (uint s = 0; s < 32; ++s) tl += mbuf[j][lane][s];
        tl *= a.scale;
      }
      float w[IK_MSG_MAXN] = {};
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < IK_MSG_MAXN; ++i) if (i <= B) w[i] = simd_broadcast(tl, ushort(i));
      float m = -INFINITY;
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < IK_MSG_MAXN; ++i) if (i <= B) m = max(m, w[i]);
      float z = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < IK_MSG_MAXN; ++i) if (i <= B) {
        w[i] = exp(w[i] - m);
        z += w[i];
      }
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < IK_MSG_MAXN; ++i) if (i <= B) w[i] /= z;
      float mo[IK_PER] = {}, ssm = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < IK_PER; ++k) {
        const uint d = tid + 1024 * k;
        if (d < D) {
          float v = 0.0f;
          _Pragma("clang loop unroll(full)")
          for (uint i = 0; i < IK_MSG_MAXN; ++i) if (i <= B) v += w[i] * bm[i][k];
          mo[k] = v;
          msg[(ulong(row) * IK_MSG_MAXQ + j) * D + d] = v;
          if (j < 3) ssm += v * v;
        }
      }
      if (j < 3) {
        const float rm = rsqrt(ik_sum32(ssm, msums[j], sg, lane) / float(D) + IK_RMS_EPS);
        const ulong yo = (ulong(j) * a.rows + row) * 2 * D + D;
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < IK_PER; ++k) {
          const uint d = tid + 1024 * k;
          if (d < D) ctrl_in[yo + d] = bfloat(mo[k] * rm);
        }
      }
    }
  }

  // Phase 2: unweighted RMS of h into nrd, depth attention into ai [rows, Q, D], plus h-half of ctrl_in.
  float hx[IK_PER] = {}, qv[QM][IK_PER] = {}, bl[IK_MAXN][IK_PER] = {};
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d >= D) continue;
    hx[k] = ik_in(h, f32in, o + d);
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < IK_MAXN - 1; ++i) if (i < K) bl[i][k] = blocks[(ulong(i) * a.stride + row) * D + d];
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < QM; ++j) if (j < Q) qv[j][k] = attn_q[j * D + d];
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
    float av[IK_PER] = {}, ssj = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < IK_PER; ++k) {
      const uint d = tid + 1024 * k;
      if (d < D) {
        float v = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < IK_MAXN; ++i) if (i <= K) v += w[i] * (i < K ? bl[i][k] : pv[k]);
        av[k] = v;
        ai[(ulong(row) * Q + j) * D + d] = v;
        ssj += v * v;
      }
    }
    const float rh = rsqrt(ik_sum32(ssj, sums[1 + j], sg, lane) / float(D) + IK_RMS_EPS);
    if (QM == 1) {
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < IK_PER; ++k) {
        const uint d = tid + 1024 * k;
        if (d < D) {
          const bfloat b = bfloat(av[k] * rh);
          ctrl_in[(ulong(0) * a.rows + row) * 2 * D + d] = b;
          ctrl_in[(ulong(1) * a.rows + row) * 2 * D + d] = b;
          ctrl_in[(ulong(2) * a.rows + row) * 2 * D + d] = b;
        }
      }
    } else {
      const ulong yo = (ulong(j) * a.rows + row) * 2 * D;
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < IK_PER; ++k) {
        const uint d = tid + 1024 * k;
        if (d < D) ctrl_in[yo + d] = bfloat(av[k] * rh);
      }
    }
  }
}

// Local looped-core attention prelude (Q = 1): grid (rows, 1, 1) x 1024.
[[kernel, max_total_threads_per_threadgroup(1024)]] void ik_loop_attn_prep(constant ik_args& a [[buffer(0)]],
    const device void* h [[buffer(1)]], const device float* bank_blocks [[buffer(2)]],
    const device float* bank_partial [[buffer(3)]], const device float* msg_q [[buffer(4)]],
    const device float* blocks [[buffer(5)]], const device float* attn_q [[buffer(6)]], device float* msg [[buffer(7)]],
    device float* nrd [[buffer(8)]], device float* ai [[buffer(9)]], device bfloat* ctrl_in [[buffer(10)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float msums[3][32];
  threadgroup float mbuf[IK_MSG_MAXQ][IK_MSG_MAXN][33];
  threadgroup float sums[2][32];
  threadgroup float dbuf[1][IK_MAXN][33];
  ik_loop_attn_prep_run<1>(a, h, bank_blocks, bank_partial, msg_q, blocks, attn_q, msg, nrd, ai, ctrl_in, msums, mbuf,
      sums, dbuf, row, tid, sg, lane);
}

// Global looped-core attention prelude (Q = 3): grid (rows, 1, 1) x 1024.
[[kernel, max_total_threads_per_threadgroup(1024)]] void ik_loop_attn_prep3(constant ik_args& a [[buffer(0)]],
    const device void* h [[buffer(1)]], const device float* bank_blocks [[buffer(2)]],
    const device float* bank_partial [[buffer(3)]], const device float* msg_q [[buffer(4)]],
    const device float* blocks [[buffer(5)]], const device float* attn_q [[buffer(6)]], device float* msg [[buffer(7)]],
    device float* nrd [[buffer(8)]], device float* ai [[buffer(9)]], device bfloat* ctrl_in [[buffer(10)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float msums[3][32];
  threadgroup float mbuf[IK_MSG_MAXQ][IK_MSG_MAXN][33];
  threadgroup float sums[4][32];
  threadgroup float dbuf[3][IK_MAXN][33];
  ik_loop_attn_prep_run<3>(a, h, bank_blocks, bank_partial, msg_q, blocks, attn_q, msg, nrd, ai, ctrl_in, msums, mbuf,
      sums, dbuf, row, tid, sg, lane);
}

// Fused looped-core MLP prelude: resid + rms(res -> ns) + depth(blocks, ns -> mi) + msggate_prep(mi, msg -> ctrl_in).
// Grid (rows, 1, 1) x 1024. Bit-identical to the 4-kernel chain; caller binds msg at slot 3 (3 * D * 4 bytes).
[[kernel, max_total_threads_per_threadgroup(1024)]] void ik_loop_mlp_prep(constant ik_args& a [[buffer(0)]],
    const device bfloat* h [[buffer(1)]], const device bfloat* ya [[buffer(2)]], device float* res [[buffer(3)]],
    const device float* blocks [[buffer(4)]], device float* ns [[buffer(5)]], const device float* query [[buffer(6)]],
    const device float* msg [[buffer(7)]], device float* mi [[buffer(8)]], device bfloat* ctrl_in [[buffer(9)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sums[3][32];
  threadgroup float dbuf[IK_MAXN][33];
  const uint D = a.width;
  const ulong o = ulong(row) * D, mo = ulong(row) * 4 * D, yo = ulong(row) * 2 * D;
  const bool boundary = (a.flags & IK_BOUNDARY) != 0;
  const uint K = a.blocks + (boundary ? 1 : 0);
  float qv[IK_PER] = {}, mv[IK_PER] = {}, bl[IK_MAXN][IK_PER] = {};
  bfloat hb[IK_PER] = {}, yb[IK_PER] = {};
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d >= D) continue;
    hb[k] = h[o + d];
    yb[k] = ya[o + d];
    qv[k] = query[d];
    mv[k] = msg[mo + d];
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
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < IK_MAXN; ++i) if (i <= K) {
    float acc = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < IK_PER; ++k) if (tid + 1024 * k < D) acc += (i < K ? bl[i][k] : pv[k]) * qv[k];
    const float v = simd_sum(acc);
    if (lane == 0) dbuf[i][sg] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
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
  float miv[IK_PER] = {}, ss_mi = 0.0f, ss_m = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d < D) {
      float v = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < IK_MAXN; ++i) if (i <= K) v += w[i] * (i < K ? bl[i][k] : pv[k]);
      miv[k] = v;
      mi[o + d] = v;
      ss_mi += v * v;
      ss_m += mv[k] * mv[k];
    }
  }
  const float rh = rsqrt(ik_sum32(ss_mi, sums[1], sg, lane) / float(D) + IK_RMS_EPS);
  const float rm = rsqrt(ik_sum32(ss_m, sums[2], sg, lane) / float(D) + IK_RMS_EPS);
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < IK_PER; ++k) {
    const uint d = tid + 1024 * k;
    if (d < D) {
      ctrl_in[yo + d] = bfloat(miv[k] * rh);
      ctrl_in[yo + D + d] = bfloat(mv[k] * rm);
    }
  }
}
