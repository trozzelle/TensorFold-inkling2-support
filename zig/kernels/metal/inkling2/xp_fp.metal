// Routed-expert prefill chain on M5 tensor units for native NVFP4 and MXFP8 Inkling 2 checkpoints.
// Every row's arithmetic is independent of neighbouring rows, chunk cuts, and window placement.
//
// Dispatches (n rows, S = n * topk slots; ids u32 [S] 16-byte aligned; ew f32 [S]; x bf16 [n, hidden]).
// `a` is ik_xpf_args at buffer 0. Buffers 1.. in order:
//   ik_xpf_count      groups (E) x 256:              ids -> counts u32 [E]
//   ik_xpf_place      groups (E + 1) x 256:          ids, counts -> order, seid, inverse u32 [S], wins, dargs[2]
//   ik_xpf_gather     groups (hidden / 128, S) x 128: x, order, seid -> prep bf16 [S, hidden]
//   ik_xpf_gemm_nvfp4 groups (indirect) x 128:       prep, wins, weight, scale, gscale -> plane f32 [S, n]
//   ik_xpf_gemm_mxfp8 groups (indirect) x 128:       prep, wins, weight, scale -> plane f32 [S, n]
//   ik_xpf_mid        groups (inner / 128, S) x 128: plane, seid -> prep bf16 [S, inner]
//   ik_xpf_finish     groups (hidden / 128, n) x 128: plane, ids, inverse, ew -> moe bf16 [n, hidden]
#include "../nax.h"
using namespace metal;

constant uint IK_XPF_WIN = IK_XPF_WINDOW;
constant uint IK_XPF_BLK = IK_XPF_WIN / 16u;

struct ik_xpf_args {
  uint rows;      // prompt rows n
  uint slots;     // n * topk
  uint experts;   // routed experts count E
  uint topk;      // picks per row
  uint hidden;    // hidden width H
  uint inner;     // expert intermediate width I
  uint n;         // active GEMM output width (2 * inner for w13, hidden for w2)
  uint k;         // active GEMM input width (hidden for w13, inner for w2)
  float cap, thr; // SoftcappedSwiGLU gate and up softcaps
};

struct ik_xpf_win {
  uint start, nlive, eid, pad;
};

constant float IK_XPF_E2M1[16] = {
  0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline float ik_xpf_e4m3_f32(uint code) {
  const float v = float(as_type<half>(ushort((code & 0x7Fu) << 7))) * 256.0f;
  const float s = (code & 0x80u) ? -v : v;
  return (code & 0x7Fu) == 0x7Fu ? NAN : s;
}

static inline float ik_xpf_e8m0_f32(uint code) {
  if (code == 255u) return NAN;
  return as_type<float>(code == 0u ? 0x00400000u : code << 23);
}

static inline half2 ik_xpf_e4m3_h2(uint word) {
  return as_type<half2>(((word & 0x007F007Fu) << 7) | ((word & 0x00800080u) << 8));
}

// Four NVFP4 nibbles (2 packed bytes) times one E4M3 group-16 scale; exact in bfloat (6 sig bits <= 8).
static inline bfloat4 ik_xpf_nvfp4_q4(uchar2 p, uchar sc_code) {
  const float sc = ik_xpf_e4m3_f32(sc_code);
  return bfloat4(float4(IK_XPF_E2M1[p.x & 0xFu], IK_XPF_E2M1[p.x >> 4],
                        IK_XPF_E2M1[p.y & 0xFu], IK_XPF_E2M1[p.y >> 4]) * sc);
}

// Four MXFP8 E4M3 bytes times one UE8M0 group-32 scale; exact in bfloat for normal scales (4 sig bits <= 8).
static inline bfloat4 ik_xpf_mxfp8_q4(uint word, uchar sc_code, float sc256) {
  if (sc_code > 246u || (((word & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u) != 0u) {
    const float sc = ik_xpf_e8m0_f32(sc_code);
    return bfloat4(bfloat(ik_xpf_e4m3_f32(word & 0xFFu) * sc),
                   bfloat(ik_xpf_e4m3_f32((word >> 8) & 0xFFu) * sc),
                   bfloat(ik_xpf_e4m3_f32((word >> 16) & 0xFFu) * sc),
                   bfloat(ik_xpf_e4m3_f32(word >> 24) * sc));
  }
  const half2 a = ik_xpf_e4m3_h2(word);
  const half2 b = ik_xpf_e4m3_h2(word >> 8);
  return bfloat4(float4(float(a.x), float(b.x), float(a.y), float(b.y)) * sc256);
}

// ---- grouping ----

[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik_xpf_count(constant ik_xpf_args& a [[buffer(0)]],
    const device uint4* ids [[buffer(1)]], device uint* counts [[buffer(2)]],
    uint e [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint part[8];
  uint c = 0u;
  for (uint i = tid; i < a.slots / 4u; i += 256u) {
    const uint4 v = ids[i];
    c += uint(v.x == e) + uint(v.y == e) + uint(v.z == e) + uint(v.w == e);
  }
  c = simd_sum(c);
  if (lane == 0u) part[sg] = c;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    uint total = 0u;
    for (uint g = 0u; g < 8u; g++) total += part[g];
    counts[e] = total;
  }
}

[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik_xpf_place(constant ik_xpf_args& a [[buffer(0)]],
    const device uint4* ids [[buffer(1)]], const device uint* counts [[buffer(2)]],
    device uint* order [[buffer(3)]], device uint* seid [[buffer(4)]], device uint* inverse [[buffer(5)]],
    device ik_xpf_win* wins [[buffer(6)]], device uint* dargs [[buffer(7)]],
    uint e [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint part[8][2];
  threadgroup uint base[8];
  const uint lim = min(e, a.experts);
  uint s_slots = 0u, s_wins = 0u;
  for (uint k = tid; k < lim; k += 256u) {
    const uint c = counts[k];
    s_slots += c;
    s_wins += (c + IK_XPF_WIN - 1u) / IK_XPF_WIN;
  }
  s_slots = simd_sum(s_slots);
  s_wins = simd_sum(s_wins);
  if (lane == 0u) {
    part[sg][0] = s_slots;
    part[sg][1] = s_wins;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint t_slots = 0u, t_wins = 0u;
  for (uint g = 0u; g < 8u; g++) {
    t_slots += part[g][0];
    t_wins += part[g][1];
  }
  if (e >= a.experts) {
    if (tid == 0u) {
      dargs[0] = (2u * a.inner) / 128u;
      dargs[1] = t_wins;
      dargs[2] = 1u;
      dargs[3] = 0u;
      dargs[4] = a.hidden / 128u;
      dargs[5] = t_wins;
      dargs[6] = 1u;
      dargs[7] = 0u;
    }
    for (uint pos = t_slots + tid; pos < a.slots; pos += 256u) seid[pos] = 0xffffffffu;
    return;
  }
  const uint n = counts[e];
  if (n == 0u) return;
  const uint off = t_slots;
  const uint run = (((a.slots + 255u) / 256u) + 3u) & ~3u;
  const uint lo = min(tid * run, a.slots);
  const uint hi = min(lo + run, a.slots);
  uint c = 0u;
  for (uint i = lo / 4u; i < hi / 4u; i++) {
    const uint4 v = ids[i];
    c += uint(v.x == e) + uint(v.y == e) + uint(v.z == e) + uint(v.w == e);
  }
  const uint within = simd_prefix_exclusive_sum(c);
  if (lane == 31u) base[sg] = within + c;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint rank = within;
  for (uint g = 0u; g < sg; g++) rank += base[g];
  const device uint* flat = (const device uint*)ids;
  for (uint s = lo; s < hi; s++) {
    if (flat[s] == e) {
      const uint pos = off + rank;
      order[pos] = s;
      seid[pos] = e;
      inverse[s] = pos;
      rank++;
    }
  }
  const uint nw = (n + IK_XPF_WIN - 1u) / IK_XPF_WIN;
  for (uint i = tid; i < nw; i += 256u) {
    ik_xpf_win w;
    w.start = off + i * IK_XPF_WIN;
    w.nlive = min(IK_XPF_WIN, n - i * IK_XPF_WIN);
    w.eid = e;
    w.pad = 0u;
    wins[t_wins + i] = w;
  }
}

// ---- gather ----

[[max_total_threads_per_threadgroup(128)]] [[kernel]] void ik_xpf_gather(constant ik_xpf_args& a [[buffer(0)]],
    const device bfloat* x [[buffer(1)]], const device uint* order [[buffer(2)]],
    const device uint* seid [[buffer(3)]], device bfloat* prep [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint pos = tg.y;
  if (seid[pos] >= a.experts) return;
  const uint col = tg.x * 128u + tid;
  const uint tok = order[pos] / a.topk;
  prep[(size_t)pos * a.hidden + col] = x[(size_t)tok * a.hidden + col];
}

// ---- grouped tensor-unit GEMM ----

static inline void ik_xpf_store(thread const tfp::frag<float>& f, device float* r0, device float* r1,
    bool l0, bool l1) {
  if (l0) *(device float4*)r0 = float4(f[0], f[1], f[2], f[3]);
  if (l1) *(device float4*)r1 = float4(f[4], f[5], f[6], f[7]);
}

template <uint LIVE>
static inline void ik_xpf_nvfp4_tiles(const device bfloat* x, const device uchar* weight, const device uchar* scale,
    uint eid, uint n0, uint n_dim, uint k_dim, uint start, uint wlast, uint lane,
    thread tfp::frag<float> (&acc)[IK_XPF_BLK][2]) {
  const short2 home = tfp::frag_home(ushort(lane));
  const uint hx = uint(home.x), hy = uint(home.y);
  const size_t k_bytes = k_dim >> 1u;
  const size_t k_scales = k_dim >> 4u;
  const size_t e_base = (size_t)eid * n_dim;
  const size_t r00 = e_base + n0 + hy;
  const size_t r01 = r00 + 8u;
  const size_t r10 = r00 + 16u;
  const size_t r11 = r00 + 24u;
  const device uchar* w00 = weight + r00 * k_bytes + (hx >> 1u);
  const device uchar* w01 = weight + r01 * k_bytes + (hx >> 1u);
  const device uchar* w10 = weight + r10 * k_bytes + (hx >> 1u);
  const device uchar* w11 = weight + r11 * k_bytes + (hx >> 1u);
  const device uchar* s00 = scale + r00 * k_scales;
  const device uchar* s01 = scale + r01 * k_scales;
  const device uchar* s10 = scale + r10 * k_scales;
  const device uchar* s11 = scale + r11 * k_scales;
  const device bfloat* xr[2 * LIVE];
  TF_UNROLL
  for (uint b = 0u; b < LIVE; b++) {
    xr[2u * b] = x + (size_t)min(start + 16u * b + hy, wlast) * k_dim + hx;
    xr[2u * b + 1u] = x + (size_t)min(start + 16u * b + hy + 8u, wlast) * k_dim + hx;
  }
  const uint kt = k_dim >> 4u;
#pragma clang loop unroll_count(2)
  for (uint tk = 0u; tk < kt; tk++) {
    const size_t wb = (size_t)tk * 8u;
    const tfp::frag<bfloat> b0 = tfp::frag<bfloat>(
        ik_xpf_nvfp4_q4(*(const device uchar2*)(w00 + wb), s00[tk]),
        ik_xpf_nvfp4_q4(*(const device uchar2*)(w01 + wb), s01[tk]));
    const tfp::frag<bfloat> b1 = tfp::frag<bfloat>(
        ik_xpf_nvfp4_q4(*(const device uchar2*)(w10 + wb), s10[tk]),
        ik_xpf_nvfp4_q4(*(const device uchar2*)(w11 + wb), s11[tk]));
    const size_t k = (size_t)tk * 16u;
    TF_UNROLL
    for (uint b = 0u; b < LIVE; b++) {
      const tfp::frag<bfloat> xf = tfp::frag<bfloat>(
          *(const device bfloat4*)(xr[2u * b] + k),
          *(const device bfloat4*)(xr[2u * b + 1u] + k));
      tfp::mma_16x32<false, true>(acc[b][0], acc[b][1], xf, b0, b1);
    }
  }
}

template <uint LIVE>
static inline void ik_xpf_nvfp4_blocks(uint live, const device bfloat* x, const device uchar* weight,
    const device uchar* scale, uint eid, uint n0, uint n_dim, uint k_dim, uint start, uint wlast, uint lane,
    thread tfp::frag<float> (&acc)[IK_XPF_BLK][2]) {
  if (live == LIVE) {
    ik_xpf_nvfp4_tiles<LIVE>(x, weight, scale, eid, n0, n_dim, k_dim, start, wlast, lane, acc);
    return;
  }
  if constexpr (LIVE > 1u) {
    ik_xpf_nvfp4_blocks<LIVE - 1u>(live, x, weight, scale, eid, n0, n_dim, k_dim, start, wlast, lane, acc);
  }
}

[[max_total_threads_per_threadgroup(128)]] [[kernel]] void ik_xpf_gemm_nvfp4(constant ik_xpf_args& a [[buffer(0)]],
    const device bfloat* x [[buffer(1)]], const device ik_xpf_win* wins [[buffer(2)]],
    const device uchar* weight [[buffer(3)]], const device uchar* scale [[buffer(4)]],
    const device float* gscale [[buffer(5)]], device float* plane [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const ik_xpf_win w = wins[tg.y];
  const uint n0 = tg.x * 128u + sg * 32u;
  tfp::frag<float> acc[IK_XPF_BLK][2];
  TF_UNROLL
  for (uint b = 0u; b < IK_XPF_BLK; b++) {
    acc[b][0] = tfp::frag<float>(0.0f);
    acc[b][1] = tfp::frag<float>(0.0f);
  }
  ik_xpf_nvfp4_blocks<IK_XPF_BLK>((w.nlive + 15u) / 16u, x, weight, scale, w.eid, n0, a.n, a.k, w.start,
                                  w.start + w.nlive - 1u, lane, acc);
  const float gs = gscale[w.eid];
  const short2 home = tfp::frag_home(ushort(lane));
  const uint hx = uint(home.x), hy = uint(home.y);
  device float* y0 = plane + (size_t)(w.start + hy) * a.n + n0 + hx;
  const size_t eight = (size_t)8u * a.n;
  TF_UNROLL
  for (uint b = 0u; b < IK_XPF_BLK; b++) {
    if (w.nlive > 16u * b) {
      device float* yb = y0 + (size_t)b * 2u * eight;
      const bool l0 = hy + 16u * b < w.nlive, l1 = hy + 16u * b + 8u < w.nlive;
      ik_xpf_store(acc[b][0] * gs, yb, yb + eight, l0, l1);
      ik_xpf_store(acc[b][1] * gs, yb + 16u, yb + eight + 16u, l0, l1);
    }
  }
}

template <uint LIVE>
static inline void ik_xpf_mxfp8_tiles(const device bfloat* x, const device uchar* weight, const device uchar* scale,
    uint eid, uint n0, uint n_dim, uint k_dim, uint start, uint wlast, uint lane,
    thread tfp::frag<float> (&acc)[IK_XPF_BLK][2]) {
  const short2 home = tfp::frag_home(ushort(lane));
  const uint hx = uint(home.x), hy = uint(home.y);
  const size_t k_scales = k_dim >> 5u;
  const size_t e_base = (size_t)eid * n_dim;
  const size_t r00 = e_base + n0 + hy;
  const size_t r01 = r00 + 8u;
  const size_t r10 = r00 + 16u;
  const size_t r11 = r00 + 24u;
  const device uchar* w00 = weight + r00 * k_dim + hx;
  const device uchar* w01 = weight + r01 * k_dim + hx;
  const device uchar* w10 = weight + r10 * k_dim + hx;
  const device uchar* w11 = weight + r11 * k_dim + hx;
  const device uchar* s00 = scale + r00 * k_scales;
  const device uchar* s01 = scale + r01 * k_scales;
  const device uchar* s10 = scale + r10 * k_scales;
  const device uchar* s11 = scale + r11 * k_scales;
  const device bfloat* xr[2 * LIVE];
  TF_UNROLL
  for (uint b = 0u; b < LIVE; b++) {
    xr[2u * b] = x + (size_t)min(start + 16u * b + hy, wlast) * k_dim + hx;
    xr[2u * b + 1u] = x + (size_t)min(start + 16u * b + hy + 8u, wlast) * k_dim + hx;
  }
  const uint kb = k_dim >> 5u;
  for (uint bk = 0u; bk < kb; bk++) {
    const uchar sc00 = s00[bk], sc01 = s01[bk], sc10 = s10[bk], sc11 = s11[bk];
    const float f00 = (sc00 <= 246u) ? ik_xpf_e8m0_f32(sc00) * 256.0f : 0.0f;
    const float f01 = (sc01 <= 246u) ? ik_xpf_e8m0_f32(sc01) * 256.0f : 0.0f;
    const float f10 = (sc10 <= 246u) ? ik_xpf_e8m0_f32(sc10) * 256.0f : 0.0f;
    const float f11 = (sc11 <= 246u) ? ik_xpf_e8m0_f32(sc11) * 256.0f : 0.0f;
    TF_UNROLL
    for (uint sub = 0u; sub < 2u; sub++) {
      const size_t k = (size_t)bk * 32u + (size_t)sub * 16u;
      const tfp::frag<bfloat> b0 = tfp::frag<bfloat>(
          ik_xpf_mxfp8_q4(*(const device uint*)(w00 + k), sc00, f00),
          ik_xpf_mxfp8_q4(*(const device uint*)(w01 + k), sc01, f01));
      const tfp::frag<bfloat> b1 = tfp::frag<bfloat>(
          ik_xpf_mxfp8_q4(*(const device uint*)(w10 + k), sc10, f10),
          ik_xpf_mxfp8_q4(*(const device uint*)(w11 + k), sc11, f11));
      TF_UNROLL
      for (uint b = 0u; b < LIVE; b++) {
        const tfp::frag<bfloat> xf = tfp::frag<bfloat>(
            *(const device bfloat4*)(xr[2u * b] + k),
            *(const device bfloat4*)(xr[2u * b + 1u] + k));
        tfp::mma_16x32<false, true>(acc[b][0], acc[b][1], xf, b0, b1);
      }
    }
  }
}

template <uint LIVE>
static inline void ik_xpf_mxfp8_blocks(uint live, const device bfloat* x, const device uchar* weight,
    const device uchar* scale, uint eid, uint n0, uint n_dim, uint k_dim, uint start, uint wlast, uint lane,
    thread tfp::frag<float> (&acc)[IK_XPF_BLK][2]) {
  if (live == LIVE) {
    ik_xpf_mxfp8_tiles<LIVE>(x, weight, scale, eid, n0, n_dim, k_dim, start, wlast, lane, acc);
    return;
  }
  if constexpr (LIVE > 1u) {
    ik_xpf_mxfp8_blocks<LIVE - 1u>(live, x, weight, scale, eid, n0, n_dim, k_dim, start, wlast, lane, acc);
  }
}

[[max_total_threads_per_threadgroup(128)]] [[kernel]] void ik_xpf_gemm_mxfp8(constant ik_xpf_args& a [[buffer(0)]],
    const device bfloat* x [[buffer(1)]], const device ik_xpf_win* wins [[buffer(2)]],
    const device uchar* weight [[buffer(3)]], const device uchar* scale [[buffer(4)]],
    device float* plane [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const ik_xpf_win w = wins[tg.y];
  const uint n0 = tg.x * 128u + sg * 32u;
  tfp::frag<float> acc[IK_XPF_BLK][2];
  TF_UNROLL
  for (uint b = 0u; b < IK_XPF_BLK; b++) {
    acc[b][0] = tfp::frag<float>(0.0f);
    acc[b][1] = tfp::frag<float>(0.0f);
  }
  ik_xpf_mxfp8_blocks<IK_XPF_BLK>((w.nlive + 15u) / 16u, x, weight, scale, w.eid, n0, a.n, a.k, w.start,
                                  w.start + w.nlive - 1u, lane, acc);
  const short2 home = tfp::frag_home(ushort(lane));
  const uint hx = uint(home.x), hy = uint(home.y);
  device float* y0 = plane + (size_t)(w.start + hy) * a.n + n0 + hx;
  const size_t eight = (size_t)8u * a.n;
  TF_UNROLL
  for (uint b = 0u; b < IK_XPF_BLK; b++) {
    if (w.nlive > 16u * b) {
      device float* yb = y0 + (size_t)b * 2u * eight;
      const bool l0 = hy + 16u * b < w.nlive, l1 = hy + 16u * b + 8u < w.nlive;
      ik_xpf_store(acc[b][0], yb, yb + eight, l0, l1);
      ik_xpf_store(acc[b][1], yb + 16u, yb + eight + 16u, l0, l1);
    }
  }
}

// ---- mid ----

[[max_total_threads_per_threadgroup(128)]] [[kernel]] void ik_xpf_mid(constant ik_xpf_args& a [[buffer(0)]],
    const device float* plane [[buffer(1)]], const device uint* seid [[buffer(2)]],
    device bfloat* prep [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint pos = tg.y;
  if (seid[pos] >= a.experts) return;
  const uint i = tg.x * 128u + tid;
  const float2 gu = *(const device float2*)(plane + (size_t)pos * (2u * a.inner) + 2u * i);
  const float g = float(bfloat(gu.x));
  const float u = float(bfloat(gu.y));
  const float act = (tanh(g / a.cap) * a.cap) * (1.0f / (1.0f + exp(-g))) * (tanh(u / a.thr) * a.thr);
  prep[(size_t)pos * a.inner + i] = bfloat(act);
}

// ---- finish ----

[[max_total_threads_per_threadgroup(128)]] [[kernel]] void ik_xpf_finish(constant ik_xpf_args& a [[buffer(0)]],
    const device float* plane [[buffer(1)]], const device uint* ids [[buffer(2)]],
    const device uint* inverse [[buffer(3)]], const device float* ew [[buffer(4)]],
    device bfloat* moe [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint d = tg.x * 128u + tid;
  const uint row = tg.y;
  const uint base_slot = row * a.topk;
  float acc = 0.0f;
  for (uint k = 0u; k < a.topk; k++) {
    const uint slot = base_slot + k;
    const uint id = ids[slot];
    const float ey = (id < a.experts) ? float(bfloat(plane[(size_t)inverse[slot] * a.hidden + d])) : NAN;
    acc = fma(ey, ew[slot], acc);
  }
  moe[(size_t)row * a.hidden + d] = bfloat(acc);
}
