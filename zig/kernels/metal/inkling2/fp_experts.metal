// Weight-only NVFP4 (block-16 E2M1 + E4M3 scale + per-expert global scale) and MXFP8 (block-32 E4M3 + UE8M0 scale)
// routed-expert kernels for Inkling 2. Strictly row-independent and deterministic in K-reduction order.
#include <metal_stdlib>
using namespace metal;

constant float IK_E2M1[16] = {
  0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

inline float ik_e4m3_f32(uint code) {
  const float v = float(as_type<half>(ushort((code & 0x7Fu) << 7))) * 256.0f;
  const float s = (code & 0x80u) ? -v : v;
  return (code & 0x7Fu) == 0x7Fu ? NAN : s;
}

inline float ik_e8m0_f32(uint code) {
  if (code == 255u) return NAN;
  return as_type<float>(code == 0u ? 0x00400000u : code << 23);
}

inline half2 ik_e4m3_h2(uint word) {
  return as_type<half2>(((word & 0x007F007Fu) << 7) | ((word & 0x00800080u) << 8));
}

// Dot product of 16 bf16 inputs at x[0..15] with 16 NVFP4 nibbles at w[0..7] and one E4M3 scale byte.
inline float ik_nvfp4_dot16(const device bfloat* x, const device uchar* w, uchar sc_code) {
  const float sc = ik_e4m3_f32(sc_code);
  const float4 x0 = float4(*(const device bfloat4*)(x + 0));
  const float4 x1 = float4(*(const device bfloat4*)(x + 4));
  const float4 x2 = float4(*(const device bfloat4*)(x + 8));
  const float4 x3 = float4(*(const device bfloat4*)(x + 12));
  const uchar4 b0 = *(const device uchar4*)(w + 0);
  const uchar4 b1 = *(const device uchar4*)(w + 4);
  float s = 0.0f;
  s = fma(x0.x, IK_E2M1[b0.x & 0xFu], s);
  s = fma(x0.y, IK_E2M1[b0.x >> 4],   s);
  s = fma(x0.z, IK_E2M1[b0.y & 0xFu], s);
  s = fma(x0.w, IK_E2M1[b0.y >> 4],   s);
  s = fma(x1.x, IK_E2M1[b0.z & 0xFu], s);
  s = fma(x1.y, IK_E2M1[b0.z >> 4],   s);
  s = fma(x1.z, IK_E2M1[b0.w & 0xFu], s);
  s = fma(x1.w, IK_E2M1[b0.w >> 4],   s);
  s = fma(x2.x, IK_E2M1[b1.x & 0xFu], s);
  s = fma(x2.y, IK_E2M1[b1.x >> 4],   s);
  s = fma(x2.z, IK_E2M1[b1.y & 0xFu], s);
  s = fma(x2.w, IK_E2M1[b1.y >> 4],   s);
  s = fma(x3.x, IK_E2M1[b1.z & 0xFu], s);
  s = fma(x3.y, IK_E2M1[b1.z >> 4],   s);
  s = fma(x3.z, IK_E2M1[b1.w & 0xFu], s);
  s = fma(x3.w, IK_E2M1[b1.w >> 4],   s);
  return s * sc;
}

// Dot product of 32 bf16 inputs at x[0..31] with 32 E4M3 weight bytes at w[0..31] and one UE8M0 scale byte.
inline float ik_mxfp8_dot32(const device bfloat* x, const device uchar* w, uchar sc_code) {
  const uint4 q0 = *(const device uint4*)(w + 0);
  const uint4 q1 = *(const device uint4*)(w + 16);
  const uint4 m0 = (q0 & 0x7F7F7F7Fu) + 0x01010101u;
  const uint4 m1 = (q1 & 0x7F7F7F7Fu) + 0x01010101u;
  if (sc_code > 246u || any(((m0 | m1) & 0x80808080u) != 0u)) {
    const float sc = ik_e8m0_f32(sc_code);
    float s = 0.0f;
    for (uint i = 0; i < 32; ++i) s = fma(float(x[i]), ik_e4m3_f32(w[i]) * sc, s);
    return s;
  }
  const float x256 = ik_e8m0_f32(sc_code) * 256.0f;
  float s = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint j = 0; j < 8; ++j) {
    const uint word = j < 4 ? q0[j] : q1[j - 4];
    const half2 a = ik_e4m3_h2(word), b = ik_e4m3_h2(word >> 8);
    const float4 xv = float4(*(const device bfloat4*)(x + 4 * j));
    s = fma(xv.x, float(a.x), s);
    s = fma(xv.y, float(b.x), s);
    s = fma(xv.z, float(a.y), s);
    s = fma(xv.w, float(b.y), s);
  }
  return s * x256;
}

// NVFP4 w13 + bf16 rounding of gate/up + SoftcappedSwiGLU into act [slots, I].
// 4 simdgroups (128 threads) per threadgroup; each simdgroup computes one (gate, up) output pair i = tg.x * 4 + sg.
[[kernel, max_total_threads_per_threadgroup(128)]] void ik_nvfp4_w13_swiglu(constant ik_args& a [[buffer(0)]],
    const device bfloat* xm [[buffer(1)]], const device uint* eids [[buffer(2)]],
    const device uchar* weight [[buffer(3)]], const device uchar* scale [[buffer(4)]],
    const device float* gscale [[buffer(5)]], device bfloat* act [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint i = tg.x * 4 + sg, slot = tg.y;
  if (i >= a.width || slot >= a.rows) return;
  const uint e = eids[slot], tok = slot / a.queries;
  const uint K = a.dim, N = 2 * a.width;
  const ulong row_g = ulong(e) * N + 2 * i;
  const ulong row_u = row_g + 1;
  const device bfloat* xr = xm + ulong(tok) * K;
  const device uchar* wg = weight + row_g * (K / 2);
  const device uchar* wu = weight + row_u * (K / 2);
  const device uchar* sg_ptr = scale + row_g * (K / 16);
  const device uchar* su_ptr = scale + row_u * (K / 16);
  float ag = 0.0f, au = 0.0f;
  for (uint k0 = 16 * lane; k0 < K; k0 += 512) {
    ag += ik_nvfp4_dot16(xr + k0, wg + k0 / 2, sg_ptr[k0 / 16]);
    au += ik_nvfp4_dot16(xr + k0, wu + k0 / 2, su_ptr[k0 / 16]);
  }
  const float gs = gscale[e];
  const float g_sum = simd_sum(ag) * gs;
  const float u_sum = simd_sum(au) * gs;
  if (lane == 0) {
    const float g = float(bfloat(g_sum));
    const float u = float(bfloat(u_sum));
    act[ulong(slot) * a.width + i] = bfloat((tanh(g / a.cap) * a.cap) * (1.0f / (1.0f + exp(-g))) * (tanh(u / a.thr) * a.thr));
  }
}

// NVFP4 w2 into ey [slots, H] bf16 (followed by ik_combine).
[[kernel, max_total_threads_per_threadgroup(128)]] void ik_nvfp4_w2(constant ik_args& a [[buffer(0)]],
    const device bfloat* act [[buffer(1)]], const device uint* eids [[buffer(2)]],
    const device uchar* weight [[buffer(3)]], const device uchar* scale [[buffer(4)]],
    const device float* gscale [[buffer(5)]], device bfloat* ey [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint d = tg.x * 4 + sg, slot = tg.y;
  if (d >= a.width || slot >= a.rows) return;
  const uint e = eids[slot], K = a.dim, N = a.width;
  const ulong row = ulong(e) * N + d;
  const device bfloat* xr = act + ulong(slot) * K;
  const device uchar* wr = weight + row * (K / 2);
  const device uchar* sr = scale + row * (K / 16);
  float acc = 0.0f;
  for (uint k0 = 16 * lane; k0 < K; k0 += 512)
    acc += ik_nvfp4_dot16(xr + k0, wr + k0 / 2, sr[k0 / 16]);
  const float s = simd_sum(acc) * gscale[e];
  if (lane == 0) ey[ulong(slot) * N + d] = bfloat(s);
}

// MXFP8 w13 + bf16 rounding of gate/up + SoftcappedSwiGLU into act [slots, I].
[[kernel, max_total_threads_per_threadgroup(128)]] void ik_mxfp8_w13_swiglu(constant ik_args& a [[buffer(0)]],
    const device bfloat* xm [[buffer(1)]], const device uint* eids [[buffer(2)]],
    const device uchar* weight [[buffer(3)]], const device uchar* scale [[buffer(4)]],
    device bfloat* act [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint i = tg.x * 4 + sg, slot = tg.y;
  if (i >= a.width || slot >= a.rows) return;
  const uint e = eids[slot], tok = slot / a.queries;
  const uint K = a.dim, N = 2 * a.width;
  const ulong row_g = ulong(e) * N + 2 * i;
  const ulong row_u = row_g + 1;
  const device bfloat* xr = xm + ulong(tok) * K;
  const device uchar* wg = weight + row_g * K;
  const device uchar* wu = weight + row_u * K;
  const device uchar* sg_ptr = scale + row_g * (K / 32);
  const device uchar* su_ptr = scale + row_u * (K / 32);
  float ag = 0.0f, au = 0.0f;
  for (uint k0 = 32 * lane; k0 < K; k0 += 1024) {
    ag += ik_mxfp8_dot32(xr + k0, wg + k0, sg_ptr[k0 / 32]);
    au += ik_mxfp8_dot32(xr + k0, wu + k0, su_ptr[k0 / 32]);
  }
  const float g_sum = simd_sum(ag);
  const float u_sum = simd_sum(au);
  if (lane == 0) {
    const float g = float(bfloat(g_sum));
    const float u = float(bfloat(u_sum));
    act[ulong(slot) * a.width + i] = bfloat((tanh(g / a.cap) * a.cap) * (1.0f / (1.0f + exp(-g))) * (tanh(u / a.thr) * a.thr));
  }
}

// MXFP8 w2 into ey [slots, H] bf16 (followed by ik_combine).
[[kernel, max_total_threads_per_threadgroup(128)]] void ik_mxfp8_w2(constant ik_args& a [[buffer(0)]],
    const device bfloat* act [[buffer(1)]], const device uint* eids [[buffer(2)]],
    const device uchar* weight [[buffer(3)]], const device uchar* scale [[buffer(4)]],
    device bfloat* ey [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint d = tg.x * 4 + sg, slot = tg.y;
  if (d >= a.width || slot >= a.rows) return;
  const uint e = eids[slot], K = a.dim, N = a.width;
  const ulong row = ulong(e) * N + d;
  const device bfloat* xr = act + ulong(slot) * K;
  const device uchar* wr = weight + row * K;
  const device uchar* sr = scale + row * (K / 32);
  float acc = 0.0f;
  for (uint k0 = 32 * lane; k0 < K; k0 += 1024)
    acc += ik_mxfp8_dot32(xr + k0, wr + k0, sr[k0 / 32]);
  const float s = simd_sum(acc);
  if (lane == 0) ey[ulong(slot) * N + d] = bfloat(s);
}

constant uint IK_FPG_TILE = 4;
constant uint IK_FPG_EMAX = 512;
constant uint IK_FPG_SMAX = 128;

// Groups up to 128 decode routing slots across up to 512 experts into tiles of at most 4 slots per expert,
// writing indirect dispatch grids for grouped w13 (dargs[0..2]) and w2 (dargs[4..6]). One 256-thread TG.
[[kernel, max_total_threads_per_threadgroup(256)]] void ik_fp_group(constant ik_args& a [[buffer(0)]],
    const device uint* eids [[buffer(1)]], device uint* u_eid [[buffer(2)]], device uint* u_off [[buffer(3)]],
    device uint* u_cnt [[buffer(4)]], device uint* u_slots [[buffer(5)]], device uint* dargs [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup uint s_eids[IK_FPG_SMAX];
  threadgroup uint counts[IK_FPG_EMAX];
  threadgroup uint offsets[IK_FPG_EMAX];
  const uint E = a.width, S = a.rows;
  if (tid < S) s_eids[tid] = eids[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < E; e += 256) {
    uint c = 0;
    for (uint s = 0; s < S; ++s) c += uint(s_eids[s] == e);
    counts[e] = c;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint slot_at = 0, u = 0;
    for (uint e = 0; e < E; ++e) {
      const uint c = counts[e];
      offsets[e] = slot_at;
      if (c != 0) {
        uint rem = c;
        uint off = slot_at;
        while (rem != 0) {
          const uint take = rem > IK_FPG_TILE ? IK_FPG_TILE : rem;
          u_eid[u] = e;
          u_off[u] = off;
          u_cnt[u] = take;
          ++u;
          off += take;
          rem -= take;
        }
        slot_at += c;
      }
    }
    dargs[0] = (a.dim + 3) / 4;
    dargs[1] = u;
    dargs[2] = 1;
    dargs[3] = 0;
    dargs[4] = (a.stride + 3) / 4;
    dargs[5] = u;
    dargs[6] = 1;
    dargs[7] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < E; e += 256) {
    if (counts[e] != 0) {
      uint at = offsets[e];
      for (uint s = 0; s < S; ++s) {
        if (s_eids[s] == e) u_slots[at++] = s;
      }
    }
  }
}

inline float ik_dot16_f4(const device bfloat* x, float4 w0, float4 w1, float4 w2, float4 w3, float sc) {
  const float4 x0 = float4(*(const device bfloat4*)(x + 0));
  const float4 x1 = float4(*(const device bfloat4*)(x + 4));
  const float4 x2 = float4(*(const device bfloat4*)(x + 8));
  const float4 x3 = float4(*(const device bfloat4*)(x + 12));
  float s = 0.0f;
  s = fma(x0.x, w0.x, s); s = fma(x0.y, w0.y, s); s = fma(x0.z, w0.z, s); s = fma(x0.w, w0.w, s);
  s = fma(x1.x, w1.x, s); s = fma(x1.y, w1.y, s); s = fma(x1.z, w1.z, s); s = fma(x1.w, w1.w, s);
  s = fma(x2.x, w2.x, s); s = fma(x2.y, w2.y, s); s = fma(x2.z, w2.z, s); s = fma(x2.w, w2.w, s);
  s = fma(x3.x, w3.x, s); s = fma(x3.y, w3.y, s); s = fma(x3.z, w3.z, s); s = fma(x3.w, w3.w, s);
  return s * sc;
}

inline void ik_nvfp4_acc16_g(uint cnt, const device bfloat* x0, const device bfloat* x1,
    const device bfloat* x2, const device bfloat* x3, const device uchar* w, uchar sc_code,
    thread float& a0, thread float& a1, thread float& a2, thread float& a3) {
  const float sc = ik_e4m3_f32(sc_code);
  const uchar4 b0 = *(const device uchar4*)(w + 0);
  const uchar4 b1 = *(const device uchar4*)(w + 4);
  const float4 w0 = float4(IK_E2M1[b0.x & 0xFu], IK_E2M1[b0.x >> 4], IK_E2M1[b0.y & 0xFu], IK_E2M1[b0.y >> 4]);
  const float4 w1 = float4(IK_E2M1[b0.z & 0xFu], IK_E2M1[b0.z >> 4], IK_E2M1[b0.w & 0xFu], IK_E2M1[b0.w >> 4]);
  const float4 w2 = float4(IK_E2M1[b1.x & 0xFu], IK_E2M1[b1.x >> 4], IK_E2M1[b1.y & 0xFu], IK_E2M1[b1.y >> 4]);
  const float4 w3 = float4(IK_E2M1[b1.z & 0xFu], IK_E2M1[b1.z >> 4], IK_E2M1[b1.w & 0xFu], IK_E2M1[b1.w >> 4]);
  a0 += ik_dot16_f4(x0, w0, w1, w2, w3, sc);
  if (cnt > 1) a1 += ik_dot16_f4(x1, w0, w1, w2, w3, sc);
  if (cnt > 2) a2 += ik_dot16_f4(x2, w0, w1, w2, w3, sc);
  if (cnt > 3) a3 += ik_dot16_f4(x3, w0, w1, w2, w3, sc);
}

inline float ik_mxfp8_dot32_fast(const device bfloat* x, thread const half2* wa, thread const half2* wb, float x256) {
  float s = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint j = 0; j < 8; ++j) {
    const float4 xv = float4(*(const device bfloat4*)(x + 4 * j));
    s = fma(xv.x, float(wa[j].x), s);
    s = fma(xv.y, float(wb[j].x), s);
    s = fma(xv.z, float(wa[j].y), s);
    s = fma(xv.w, float(wb[j].y), s);
  }
  return s * x256;
}

inline float ik_mxfp8_dot32_slow(const device bfloat* x, const device uchar* w, float sc) {
  float s = 0.0f;
  for (uint i = 0; i < 32; ++i) s = fma(float(x[i]), ik_e4m3_f32(w[i]) * sc, s);
  return s;
}

inline void ik_mxfp8_acc32_g(uint cnt, const device bfloat* x0, const device bfloat* x1,
    const device bfloat* x2, const device bfloat* x3, const device uchar* w, uchar sc_code,
    thread float& a0, thread float& a1, thread float& a2, thread float& a3) {
  const uint4 q0 = *(const device uint4*)(w + 0);
  const uint4 q1 = *(const device uint4*)(w + 16);
  const uint4 m0 = (q0 & 0x7F7F7F7Fu) + 0x01010101u;
  const uint4 m1 = (q1 & 0x7F7F7F7Fu) + 0x01010101u;
  if (sc_code > 246u || any(((m0 | m1) & 0x80808080u) != 0u)) {
    const float sc = ik_e8m0_f32(sc_code);
    a0 += ik_mxfp8_dot32_slow(x0, w, sc);
    if (cnt > 1) a1 += ik_mxfp8_dot32_slow(x1, w, sc);
    if (cnt > 2) a2 += ik_mxfp8_dot32_slow(x2, w, sc);
    if (cnt > 3) a3 += ik_mxfp8_dot32_slow(x3, w, sc);
    return;
  }
  const float x256 = ik_e8m0_f32(sc_code) * 256.0f;
  half2 wa[8], wb[8];
  _Pragma("clang loop unroll(full)")
  for (uint j = 0; j < 8; ++j) {
    const uint word = j < 4 ? q0[j] : q1[j - 4];
    wa[j] = ik_e4m3_h2(word);
    wb[j] = ik_e4m3_h2(word >> 8);
  }
  a0 += ik_mxfp8_dot32_fast(x0, wa, wb, x256);
  if (cnt > 1) a1 += ik_mxfp8_dot32_fast(x1, wa, wb, x256);
  if (cnt > 2) a2 += ik_mxfp8_dot32_fast(x2, wa, wb, x256);
  if (cnt > 3) a3 += ik_mxfp8_dot32_fast(x3, wa, wb, x256);
}

inline bfloat ik_swiglu_bf16(float g_sum, float u_sum, float cap, float thr) {
  const float g = float(bfloat(g_sum));
  const float u = float(bfloat(u_sum));
  return bfloat((tanh(g / cap) * cap) * (1.0f / (1.0f + exp(-g))) * (tanh(u / thr) * thr));
}

// Grouped NVFP4 w13 + bf16 rounding of gate/up + SoftcappedSwiGLU into act [slots, I].
// Dispatched indirectly via ik_fp_group dargs[0..2]; bit-identical to ik_nvfp4_w13_swiglu.
[[kernel, max_total_threads_per_threadgroup(128)]] void ik_nvfp4_w13_swiglu_g(constant ik_args& a [[buffer(0)]],
    const device bfloat* xm [[buffer(1)]], const device uint* u_eid [[buffer(2)]],
    const device uint* u_off [[buffer(3)]], const device uint* u_cnt [[buffer(4)]],
    const device uint* u_slots [[buffer(5)]], const device uchar* weight [[buffer(6)]],
    const device uchar* scale [[buffer(7)]], const device float* gscale [[buffer(8)]],
    device bfloat* act [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint i = tg.x * 4 + sg, u = tg.y;
  if (i >= a.width) return;
  const uint e = u_eid[u], off = u_off[u], cnt = u_cnt[u];
  const uint s0 = u_slots[off];
  const uint s1 = cnt > 1 ? u_slots[off + 1] : s0;
  const uint s2 = cnt > 2 ? u_slots[off + 2] : s0;
  const uint s3 = cnt > 3 ? u_slots[off + 3] : s0;
  const uint K = a.dim, N = 2 * a.width;
  const ulong row_g = ulong(e) * N + 2 * i;
  const ulong row_u = row_g + 1;
  const device bfloat* x0 = xm + ulong(s0 / a.queries) * K;
  const device bfloat* x1 = xm + ulong(s1 / a.queries) * K;
  const device bfloat* x2 = xm + ulong(s2 / a.queries) * K;
  const device bfloat* x3 = xm + ulong(s3 / a.queries) * K;
  const device uchar* wg = weight + row_g * (K / 2);
  const device uchar* wu = weight + row_u * (K / 2);
  const device uchar* sg_ptr = scale + row_g * (K / 16);
  const device uchar* su_ptr = scale + row_u * (K / 16);
  float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
  float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
  for (uint k0 = 16 * lane; k0 < K; k0 += 512) {
    ik_nvfp4_acc16_g(cnt, x0 + k0, x1 + k0, x2 + k0, x3 + k0, wg + k0 / 2, sg_ptr[k0 / 16], ag0, ag1, ag2, ag3);
    ik_nvfp4_acc16_g(cnt, x0 + k0, x1 + k0, x2 + k0, x3 + k0, wu + k0 / 2, su_ptr[k0 / 16], au0, au1, au2, au3);
  }
  const float gs = gscale[e];
  {
    const float g_sum = simd_sum(ag0) * gs, u_sum = simd_sum(au0) * gs;
    if (lane == 0) act[ulong(s0) * a.width + i] = ik_swiglu_bf16(g_sum, u_sum, a.cap, a.thr);
  }
  if (cnt > 1) {
    const float g_sum = simd_sum(ag1) * gs, u_sum = simd_sum(au1) * gs;
    if (lane == 0) act[ulong(s1) * a.width + i] = ik_swiglu_bf16(g_sum, u_sum, a.cap, a.thr);
  }
  if (cnt > 2) {
    const float g_sum = simd_sum(ag2) * gs, u_sum = simd_sum(au2) * gs;
    if (lane == 0) act[ulong(s2) * a.width + i] = ik_swiglu_bf16(g_sum, u_sum, a.cap, a.thr);
  }
  if (cnt > 3) {
    const float g_sum = simd_sum(ag3) * gs, u_sum = simd_sum(au3) * gs;
    if (lane == 0) act[ulong(s3) * a.width + i] = ik_swiglu_bf16(g_sum, u_sum, a.cap, a.thr);
  }
}

// Grouped NVFP4 w2 into ey [slots, H] bf16 (followed by ik_combine).
// Dispatched indirectly via ik_fp_group dargs[4..6]; bit-identical to ik_nvfp4_w2.
[[kernel, max_total_threads_per_threadgroup(128)]] void ik_nvfp4_w2_g(constant ik_args& a [[buffer(0)]],
    const device bfloat* act [[buffer(1)]], const device uint* u_eid [[buffer(2)]],
    const device uint* u_off [[buffer(3)]], const device uint* u_cnt [[buffer(4)]],
    const device uint* u_slots [[buffer(5)]], const device uchar* weight [[buffer(6)]],
    const device uchar* scale [[buffer(7)]], const device float* gscale [[buffer(8)]],
    device bfloat* ey [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint d = tg.x * 4 + sg, u = tg.y;
  if (d >= a.width) return;
  const uint e = u_eid[u], off = u_off[u], cnt = u_cnt[u];
  const uint s0 = u_slots[off];
  const uint s1 = cnt > 1 ? u_slots[off + 1] : s0;
  const uint s2 = cnt > 2 ? u_slots[off + 2] : s0;
  const uint s3 = cnt > 3 ? u_slots[off + 3] : s0;
  const uint K = a.dim, N = a.width;
  const ulong row = ulong(e) * N + d;
  const device bfloat* x0 = act + ulong(s0) * K;
  const device bfloat* x1 = act + ulong(s1) * K;
  const device bfloat* x2 = act + ulong(s2) * K;
  const device bfloat* x3 = act + ulong(s3) * K;
  const device uchar* wr = weight + row * (K / 2);
  const device uchar* sr = scale + row * (K / 16);
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (uint k0 = 16 * lane; k0 < K; k0 += 512)
    ik_nvfp4_acc16_g(cnt, x0 + k0, x1 + k0, x2 + k0, x3 + k0, wr + k0 / 2, sr[k0 / 16], a0, a1, a2, a3);
  const float gs = gscale[e];
  {
    const float v = simd_sum(a0) * gs;
    if (lane == 0) ey[ulong(s0) * N + d] = bfloat(v);
  }
  if (cnt > 1) {
    const float v = simd_sum(a1) * gs;
    if (lane == 0) ey[ulong(s1) * N + d] = bfloat(v);
  }
  if (cnt > 2) {
    const float v = simd_sum(a2) * gs;
    if (lane == 0) ey[ulong(s2) * N + d] = bfloat(v);
  }
  if (cnt > 3) {
    const float v = simd_sum(a3) * gs;
    if (lane == 0) ey[ulong(s3) * N + d] = bfloat(v);
  }
}

// Grouped MXFP8 w13 + bf16 rounding of gate/up + SoftcappedSwiGLU into act [slots, I].
// Dispatched indirectly via ik_fp_group dargs[0..2]; bit-identical to ik_mxfp8_w13_swiglu.
[[kernel, max_total_threads_per_threadgroup(128)]] void ik_mxfp8_w13_swiglu_g(constant ik_args& a [[buffer(0)]],
    const device bfloat* xm [[buffer(1)]], const device uint* u_eid [[buffer(2)]],
    const device uint* u_off [[buffer(3)]], const device uint* u_cnt [[buffer(4)]],
    const device uint* u_slots [[buffer(5)]], const device uchar* weight [[buffer(6)]],
    const device uchar* scale [[buffer(7)]], device bfloat* act [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint i = tg.x * 4 + sg, u = tg.y;
  if (i >= a.width) return;
  const uint e = u_eid[u], off = u_off[u], cnt = u_cnt[u];
  const uint s0 = u_slots[off];
  const uint s1 = cnt > 1 ? u_slots[off + 1] : s0;
  const uint s2 = cnt > 2 ? u_slots[off + 2] : s0;
  const uint s3 = cnt > 3 ? u_slots[off + 3] : s0;
  const uint K = a.dim, N = 2 * a.width;
  const ulong row_g = ulong(e) * N + 2 * i;
  const ulong row_u = row_g + 1;
  const device bfloat* x0 = xm + ulong(s0 / a.queries) * K;
  const device bfloat* x1 = xm + ulong(s1 / a.queries) * K;
  const device bfloat* x2 = xm + ulong(s2 / a.queries) * K;
  const device bfloat* x3 = xm + ulong(s3 / a.queries) * K;
  const device uchar* wg = weight + row_g * K;
  const device uchar* wu = weight + row_u * K;
  const device uchar* sg_ptr = scale + row_g * (K / 32);
  const device uchar* su_ptr = scale + row_u * (K / 32);
  float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
  float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
  for (uint k0 = 32 * lane; k0 < K; k0 += 1024) {
    ik_mxfp8_acc32_g(cnt, x0 + k0, x1 + k0, x2 + k0, x3 + k0, wg + k0, sg_ptr[k0 / 32], ag0, ag1, ag2, ag3);
    ik_mxfp8_acc32_g(cnt, x0 + k0, x1 + k0, x2 + k0, x3 + k0, wu + k0, su_ptr[k0 / 32], au0, au1, au2, au3);
  }
  {
    const float g_sum = simd_sum(ag0), u_sum = simd_sum(au0);
    if (lane == 0) act[ulong(s0) * a.width + i] = ik_swiglu_bf16(g_sum, u_sum, a.cap, a.thr);
  }
  if (cnt > 1) {
    const float g_sum = simd_sum(ag1), u_sum = simd_sum(au1);
    if (lane == 0) act[ulong(s1) * a.width + i] = ik_swiglu_bf16(g_sum, u_sum, a.cap, a.thr);
  }
  if (cnt > 2) {
    const float g_sum = simd_sum(ag2), u_sum = simd_sum(au2);
    if (lane == 0) act[ulong(s2) * a.width + i] = ik_swiglu_bf16(g_sum, u_sum, a.cap, a.thr);
  }
  if (cnt > 3) {
    const float g_sum = simd_sum(ag3), u_sum = simd_sum(au3);
    if (lane == 0) act[ulong(s3) * a.width + i] = ik_swiglu_bf16(g_sum, u_sum, a.cap, a.thr);
  }
}

// Grouped MXFP8 w2 into ey [slots, H] bf16 (followed by ik_combine).
// Dispatched indirectly via ik_fp_group dargs[4..6]; bit-identical to ik_mxfp8_w2.
[[kernel, max_total_threads_per_threadgroup(128)]] void ik_mxfp8_w2_g(constant ik_args& a [[buffer(0)]],
    const device bfloat* act [[buffer(1)]], const device uint* u_eid [[buffer(2)]],
    const device uint* u_off [[buffer(3)]], const device uint* u_cnt [[buffer(4)]],
    const device uint* u_slots [[buffer(5)]], const device uchar* weight [[buffer(6)]],
    const device uchar* scale [[buffer(7)]], device bfloat* ey [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint d = tg.x * 4 + sg, u = tg.y;
  if (d >= a.width) return;
  const uint e = u_eid[u], off = u_off[u], cnt = u_cnt[u];
  const uint s0 = u_slots[off];
  const uint s1 = cnt > 1 ? u_slots[off + 1] : s0;
  const uint s2 = cnt > 2 ? u_slots[off + 2] : s0;
  const uint s3 = cnt > 3 ? u_slots[off + 3] : s0;
  const uint K = a.dim, N = a.width;
  const ulong row = ulong(e) * N + d;
  const device bfloat* x0 = act + ulong(s0) * K;
  const device bfloat* x1 = act + ulong(s1) * K;
  const device bfloat* x2 = act + ulong(s2) * K;
  const device bfloat* x3 = act + ulong(s3) * K;
  const device uchar* wr = weight + row * K;
  const device uchar* sr = scale + row * (K / 32);
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (uint k0 = 32 * lane; k0 < K; k0 += 1024)
    ik_mxfp8_acc32_g(cnt, x0 + k0, x1 + k0, x2 + k0, x3 + k0, wr + k0, sr[k0 / 32], a0, a1, a2, a3);
  {
    const float v = simd_sum(a0);
    if (lane == 0) ey[ulong(s0) * N + d] = bfloat(v);
  }
  if (cnt > 1) {
    const float v = simd_sum(a1);
    if (lane == 0) ey[ulong(s1) * N + d] = bfloat(v);
  }
  if (cnt > 2) {
    const float v = simd_sum(a2);
    if (lane == 0) ey[ulong(s2) * N + d] = bfloat(v);
  }
  if (cnt > 3) {
    const float v = simd_sum(a3);
    if (lane == 0) ey[ulong(s3) * N + d] = bfloat(v);
  }
}
