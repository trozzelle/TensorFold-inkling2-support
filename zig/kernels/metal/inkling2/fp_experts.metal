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
