// Inkling 2's routed experts for one to sixteen rows: the EXL3 decode chain, four dispatches from the MoE input to
// the weighted sum. A slot is row * 8 + pick; every slot's arithmetic is independent of the other slots, of the
// number of rows and of the grid, so a window's rows equal the same rows run one at a time, bit for bit.
//
// Adapted from Sushi v1.3.0 (commit 8e4c0af1; MIT License, see LICENSES/Sushi-MIT.txt and THIRD_PARTY_NOTICES.md), as
// carried in mlx-serve's src/sdk/quant/exl3/sushi: kernels/pair_gemv.metal (the input Hadamard and the tile loop),
// kernels/down_fused.metal (the down projection's input and tile loop), kernels/reduce.metal (the output Hadamard's
// geometry), kernels/helpers_tail.metal (the lane bit windows) and the mul1 codeword decode of
// expert_exl3_kernels.zig. What differs here: w13 is one projection of 4608 outputs and its interleaved gate/up
// pairs are formed after the output Hadamard; gu, act and each pick's down output round to bf16 at the reference's
// points; the capped SwiGLU and the weighted sum are the reference's; weights are reached through ExpertDescriptor
// tables (core/exl3_projection.zig), and an expert's rate (K2 or K3, from its descriptor's halves) selects the decode
// per threadgroup, so one dispatch serves a layer of mixed rates.
//
// Dispatches (S = 8 * rows slots; eids u32 [S] in pick order; ew f32 [S]; x bf16 [rows][4608]; descriptor tables:
// w13 at index e, w2 at index e of the table bound at the w2 descriptors' offset). `a` is forward.zig's Args at
// buffer 0; only ik_xd_mid reads it (a.cap, a.thr: the gate and up caps). Buffers 1.. in order:
//   ik_xd_pair   groups (144, S, 2) x 128:   x, eids, w13 table, planes -> planes f32 [S][2][4608]
//   ik_xd_mid    groups (36, S, 1) x 32:     planes, eids, w13 table, act -> act bf16 [S][2304]
//   ik_xd_down   groups (144, S, 1) x 128:   act, eids, w2 table, sums -> sums f32 [S][4608]
//   ik_xd_finish groups (36, rows, 1) x 256: sums, eids, w2 table, ew, moe -> moe bf16 [rows][4608]
// ik_xd_mid_dbg and ik_xd_finish_dbg are the same kernels that also store gu bf16 [S][4608] (buffer 5) and the
// per-pick ey bf16 [S][4608] (buffer 6); ik_xd_decode stores the codebook's value of each state (buffers 1, 2). The
// checks read them, the forward does not.
//
// Order contract. One output element's sums, in this order; any kernel that shares an expert's weights among slots
// must reproduce it for each slot.
//   pair:   x_i * suh_i in f32 (exact: bf16 times f16), the 128-block Hadamard in ik_xd_h128's butterfly order,
//           * 0.08838834764831845f, rounded to f16 into `prepared` (K half `split`: input tiles 144*split ..).
//           Simdgroup sg of four takes the k-tiles sg, sg + 4, sg + 8, .. of its half; per lane eight accumulators
//           each run a chain of fma over those tiles in increasing order (tile tk, then tk + 4, ..); per lane
//           clo = (a0 + a1) + (a2 + a3) and chi = (a4 + a5) + (a6 + a7); then += the lane xor 1, then += the lane
//           xor 2; the four simdgroups' values add from 0.0f in simdgroup order 0, 1, 2, 3.
//           planes[slot][split][col] is that f32 sum.
//   mid:    v = (0 + planes[slot][0][col]) + planes[slot][1][col]; the 128-column-block Hadamard; v * 0.0883.. * svh;
//           gu = bf16(v). Pair j is gu[2j] (gate g) and gu[2j + 1] (up u):
//           act[j] = bf16((tanh(g / cap) * cap) * (1 / (1 + exp(-g))) * (tanh(u / thr) * thr)) from the bf16 g and u.
//   down:   float(act_i) * suh2_i, the same Hadamard and scale, f16 `prepared`; the tile loop and reduction as pair
//           over all 144 input tiles (no split). sums[slot][col] is the f32 sum.
//   finish: the 128-column-block Hadamard of sums[slot], * 0.0883.. * svh2; ey = bf16(v) per pick;
//           moe[row][col] = bf16(fma(ey_7, w_7, .. fma(ey_1, w_1, fma(ey_0, w_0, 0.0f)))), w_k = ew[row * 8 + k].
#include <metal_stdlib>
using namespace metal;

constant uint IK_XD_H = 4608;  // w13's input and w2's output width
constant uint IK_XD_I = 2304;  // the experts' inner width: w2's input, half of w13's output
constant uint IK_XD_TOPK = 8;
constant uint IK_XD_OT = 288;  // output tiles of both projections (4608 / 16)
constant float IK_XD_PSC = 0.08838834764831845f;  // 1 / sqrt(128)

// core/exl3_projection.zig's ExpertDescriptor, field for field.
struct ik_xd_expert {
  const device uchar* trellis;
  const device half* suh;
  const device half* svh;
  const device half* bias;
  ulong tile_k_stride, tile_n_stride;
  uint halves, codebook, suh_signs, svh_signs, has_bias, reserved;
};

// The mul1 codebook: two 16-bit codewords to their f16 weights. The multiply-add is one f16 fma.
static inline half2 ik_xd_pairh(uint2 cw) {
  const uint2 mixed = cw * uint2(0x83DCD12Du);
  const uint2 pair_sums = (mixed & uint2(0x00FF00FFu)) + ((mixed >> uint2(8u)) & uint2(0x00FF00FFu));
  const uint2 byte_sum = uint2(0x6400u) + (pair_sums & uint2(0xFFFFu)) + (pair_sums >> uint2(16u));
  const half2 h = as_type<half2>(ushort2(byte_sum & uint2(0xFFFFu)));
  return fma(h, as_type<half2>(ushort2(0x1EEEu)), as_type<half2>(ushort2(0xC931u)));
}
static inline float2 ik_xd_decode2(uint2 cw) { return float2(ik_xd_pairh(cw)); }

// The codebook's value of each 16-bit state, for the checks: out[2i], out[2i + 1] from states[2i], states[2i + 1].
[[kernel]] void ik_xd_decode(constant ik_args& a [[buffer(0)]], const device ushort* states [[buffer(1)]],
    device half* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  const half2 h = ik_xd_pairh(uint2(uint(states[2u * i]), uint(states[2u * i + 1u])));
  out[2u * i] = h.x;
  out[2u * i + 1u] = h.y;
}

// Eight weights span N / 2 whole bits (N u16 halfwords per tile, K = N / 16 bits a weight), so weight j of an
// eight-weight group ends ik_xd_end(N, j) bits past the group's first bit and its codeword ends ik_xd_sh(N, j) bits
// before the group's last.
static constexpr uint ik_xd_end(uint N, uint j) { return ((j + 1u) * N) >> 4u; }
static constexpr uint ik_xd_sh(uint N, uint j) { return N / 2u - ik_xd_end(N, j); }

// The 64 stream bits ending at bit `end` of a tile of `nwords` words (the tile's bitstream wraps). Valid for N = 32
// and 48, where the codeword farthest from the group's end lies within these bits.
static inline ulong ik_xd_window(const device uint* words, uint end, uint nwords) {
  const uint last = (end - 1u) >> 5u;
  const uint prev = last == 0u ? nwords - 1u : last - 1u;
  const uint s = (0u - end) & 31u;
  return (((ulong)words[prev] << 32u) | (ulong)words[last]) >> s;
}

// The 128-point Hadamard of four elements a lane (indices lane + 32 q) in Sushi's butterfly order: the five lane
// stages by xor shuffles, then the in-lane stages. Returns the outputs at lane, lane + 32, lane + 64, lane + 96,
// unscaled.
static inline float4 ik_xd_h128(float4 v, uint lane) {
  for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    const float p0 = simd_shuffle_xor(v.x, bit);
    const float p1 = simd_shuffle_xor(v.y, bit);
    const float p2 = simd_shuffle_xor(v.z, bit);
    const float p3 = simd_shuffle_xor(v.w, bit);
    const bool lower = (lane & bit) == 0u;
    v.x = lower ? v.x + p0 : p0 - v.x;
    v.y = lower ? v.y + p1 : p1 - v.y;
    v.z = lower ? v.z + p2 : p2 - v.z;
    v.w = lower ? v.w + p3 : p3 - v.w;
  }
  const float s0 = v.x + v.y;
  const float s1 = v.x - v.y;
  const float s2 = v.z + v.w;
  const float s3 = v.z - v.w;
  return float4(s0 + s2, s1 + s3, s0 - s2, s1 - s3);
}

// One simdgroup's tiles of a threadgroup's two output tiles: 144 input tiles from `wp` (tile (sg, ot) of the half),
// the simdgroup's every fourth, two a step (tiles tk and tk + 4) with both tiles' words loaded before either decodes.
// `pp` is the rounded input from the simdgroup's first tile. acc[o][j] are the lane's eight accumulators of output
// tile o: j 0..3 column lane/4 over input rows prow, prow + 1, prow + 8, prow + 9 (prow = (lane & 3) * 2), j 4..7
// column lane/4 + 8.
template <uint N>
static inline void ik_xd_tiles(const device uint* wp, const threadgroup half* pp, uint lane, thread float (&acc)[2][8]) {
  static_assert(N == 32u || N == 48u, "the decode chain reads K2 and K3 tiles");
  constexpr uint W = N / 2u;  // u32 words a tile
  constexpr uint SGS = 4u;
  constexpr uint TILE = 16u;
  const uint prow = (lane & 3u) * 2u;
  const uint end = W * lane + W;
  for (uint step = 0u; step < 18u; step++) {
    ulong merged[2][2];
    for (uint u = 0u; u < 2u; u++) {
      for (uint o = 0u; o < 2u; o++) merged[u][o] = ik_xd_window(wp + u * SGS * IK_XD_OT * W + o * W, end, W);
    }
    for (uint u = 0u; u < 2u; u++) {
      const float in0 = float(pp[u * SGS * TILE + prow]);
      const float in1 = float(pp[u * SGS * TILE + prow + 1u]);
      const float in2 = float(pp[u * SGS * TILE + prow + 8u]);
      const float in3 = float(pp[u * SGS * TILE + prow + 9u]);
      const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
      for (uint o = 0u; o < 2u; o++) {
        for (uint p = 0u; p < 4u; p++) {
          const uint2 cw = uint2(uint(merged[u][o] >> ik_xd_sh(N, p * 2u)), uint(merged[u][o] >> ik_xd_sh(N, p * 2u + 1u))) & uint2(0xffffu);
          const float2 w = ik_xd_decode2(cw);
          acc[o][p * 2u] = fma(ins[p * 2u], w.x, acc[o][p * 2u]);
          acc[o][p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[o][p * 2u + 1u]);
        }
      }
    }
    wp += 2u * SGS * IK_XD_OT * W;
    pp += 2u * SGS * TILE;
  }
}

// The tile loop of the descriptor's rate: tiles tk0 + sg onward of output tiles ot, ot + 1. A threadgroup is one
// slot, so the switch never diverges.
static inline void ik_xd_gemv(const device ik_xd_expert& d, uint tk0, uint ot, const threadgroup half* prepared, uint sg, uint lane, thread float (&acc)[2][8]) {
  const device uint* trellis = (const device uint*)d.trellis;
  switch (d.halves) {
    case 4u: ik_xd_tiles<32u>(trellis + ((size_t)(tk0 + sg) * IK_XD_OT + ot) * 16u, prepared + sg * 16u, lane, acc); break;
    case 6u: ik_xd_tiles<48u>(trellis + ((size_t)(tk0 + sg) * IK_XD_OT + ot) * 24u, prepared + sg * 16u, lane, acc); break;
    default: break;
  }
}

// The lane's accumulators to the threadgroup's 32 output columns (output tiles ot, ot + 1), each the sum of the four
// simdgroups' values in order, stored f32 at out[tile * 16 + column].
static inline void ik_xd_reduce(thread float (&acc)[2][8], threadgroup float* partial, device float* out, uint ot, uint sg, uint lane, uint lid) {
  for (uint o = 0u; o < 2u; o++) {
    float clo = (acc[o][0] + acc[o][1]) + (acc[o][2] + acc[o][3]);
    float chi = (acc[o][4] + acc[o][5]) + (acc[o][6] + acc[o][7]);
    clo += simd_shuffle_xor(clo, 1u);
    chi += simd_shuffle_xor(chi, 1u);
    clo += simd_shuffle_xor(clo, 2u);
    chi += simd_shuffle_xor(chi, 2u);
    if ((lane & 3u) == 0u) {
      partial[(o * 4u + sg) * 16u + (lane >> 2u)] = clo;
      partial[(o * 4u + sg) * 16u + (lane >> 2u) + 8u] = chi;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 32u) {
    const uint o = lid >> 4u;
    float sum = 0.0f;
    for (uint g = 0u; g < 4u; g++) sum += partial[(o * 4u + g) * 16u + (lid & 15u)];
    out[(ot + o) * 16u + (lid & 15u)] = sum;
  }
}

// w13 of the slot's expert over one half of the input: planes[(slot * 2 + split) * 4608 + col].
[[kernel]] void ik_xd_pair(constant ik_args& a [[buffer(0)]], const device bfloat* x [[buffer(1)]],
    const device uint* eids [[buffer(2)]], const device ik_xd_expert* experts [[buffer(3)]],
    device float* planes [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint lid [[thread_index_in_threadgroup]]) {
  threadgroup half prepared[IK_XD_I];
  threadgroup float partial[4 * 16 * 2];
  const uint ot = tg.x * 2u, slot = tg.y, split = tg.z;
  const device ik_xd_expert& d = experts[eids[slot]];
  const uint xrow = slot / IK_XD_TOPK;
  for (uint b = sg; b < IK_XD_I / 128u; b += 4u) {
    const uint pbase = split * IK_XD_I + b * 128u;
    const size_t xr = (size_t)xrow * IK_XD_H + pbase;
    const float x0 = float(x[xr + lane]);
    const float x1 = float(x[xr + lane + 32u]);
    const float x2 = float(x[xr + lane + 64u]);
    const float x3 = float(x[xr + lane + 96u]);
    const float4 s = ik_xd_h128(float4(x0 * float(d.suh[pbase + lane]), x1 * float(d.suh[pbase + lane + 32u]),
                                       x2 * float(d.suh[pbase + lane + 64u]), x3 * float(d.suh[pbase + lane + 96u])), lane);
    prepared[b * 128u + lane] = half(s.x * IK_XD_PSC);
    prepared[b * 128u + lane + 32u] = half(s.y * IK_XD_PSC);
    prepared[b * 128u + lane + 64u] = half(s.z * IK_XD_PSC);
    prepared[b * 128u + lane + 96u] = half(s.w * IK_XD_PSC);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc[2][8] = {};
  ik_xd_gemv(d, split * 144u, ot, prepared, sg, lane, acc);
  ik_xd_reduce(acc, partial, planes + (size_t)(slot * 2u + split) * IK_XD_H, ot, sg, lane, lid);
}

// gu from the two planes and the capped SwiGLU on adjacent pairs; `Dbg` also stores gu.
template <bool Dbg>
static inline void ik_xd_mid_body(constant ik_args& a, const device float* planes, const device uint* eids,
    const device ik_xd_expert* experts, device bfloat* act, device bfloat* gu, uint2 tg, uint lane) {
  const uint blk = tg.x, slot = tg.y;
  const device half* svh = experts[eids[slot]].svh;
  const uint base = blk * 128u;
  float4 v = float4(0.0f);
  for (uint sp = 0u; sp < 2u; sp++) {
    const size_t xb = ((size_t)slot * 2u + sp) * IK_XD_H + base;
    v += float4(planes[xb + lane], planes[xb + lane + 32u], planes[xb + lane + 64u], planes[xb + lane + 96u]);
  }
  const float4 s = ik_xd_h128(v, lane);
  // gu rounds to bf16 here; the pair's partner is the neighbouring lane (the same q, index bit 0).
  const float g0 = float(bfloat(s.x * IK_XD_PSC * float(svh[base + lane])));
  const float g1 = float(bfloat(s.y * IK_XD_PSC * float(svh[base + lane + 32u])));
  const float g2 = float(bfloat(s.z * IK_XD_PSC * float(svh[base + lane + 64u])));
  const float g3 = float(bfloat(s.w * IK_XD_PSC * float(svh[base + lane + 96u])));
  if (Dbg) {
    const size_t gb = (size_t)slot * IK_XD_H + base;
    gu[gb + lane] = bfloat(g0);
    gu[gb + lane + 32u] = bfloat(g1);
    gu[gb + lane + 64u] = bfloat(g2);
    gu[gb + lane + 96u] = bfloat(g3);
  }
  const float u0 = simd_shuffle_xor(g0, 1);
  const float u1 = simd_shuffle_xor(g1, 1);
  const float u2 = simd_shuffle_xor(g2, 1);
  const float u3 = simd_shuffle_xor(g3, 1);
  if ((lane & 1u) == 0u) {
    // an even lane holds the gate, its partner the up
    const size_t ab = (size_t)slot * IK_XD_I + blk * 64u + (lane >> 1u);
    act[ab] = bfloat((tanh(g0 / a.cap) * a.cap) * (1.0f / (1.0f + exp(-g0))) * (tanh(u0 / a.thr) * a.thr));
    act[ab + 16u] = bfloat((tanh(g1 / a.cap) * a.cap) * (1.0f / (1.0f + exp(-g1))) * (tanh(u1 / a.thr) * a.thr));
    act[ab + 32u] = bfloat((tanh(g2 / a.cap) * a.cap) * (1.0f / (1.0f + exp(-g2))) * (tanh(u2 / a.thr) * a.thr));
    act[ab + 48u] = bfloat((tanh(g3 / a.cap) * a.cap) * (1.0f / (1.0f + exp(-g3))) * (tanh(u3 / a.thr) * a.thr));
  }
}

[[kernel]] void ik_xd_mid(constant ik_args& a [[buffer(0)]], const device float* planes [[buffer(1)]],
    const device uint* eids [[buffer(2)]], const device ik_xd_expert* experts [[buffer(3)]],
    device bfloat* act [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  ik_xd_mid_body<false>(a, planes, eids, experts, act, act, tg, lane);
}

[[kernel]] void ik_xd_mid_dbg(constant ik_args& a [[buffer(0)]], const device float* planes [[buffer(1)]],
    const device uint* eids [[buffer(2)]], const device ik_xd_expert* experts [[buffer(3)]],
    device bfloat* act [[buffer(4)]], device bfloat* gu [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  ik_xd_mid_body<true>(a, planes, eids, experts, act, gu, tg, lane);
}

// w2 of the slot's expert over its act row: sums[slot * 4608 + col].
[[kernel]] void ik_xd_down(constant ik_args& a [[buffer(0)]], const device bfloat* act [[buffer(1)]],
    const device uint* eids [[buffer(2)]], const device ik_xd_expert* experts [[buffer(3)]],
    device float* sums [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint lid [[thread_index_in_threadgroup]]) {
  threadgroup half prepared[IK_XD_I];
  threadgroup float partial[4 * 16 * 2];
  const uint ot = tg.x * 2u, slot = tg.y;
  const device ik_xd_expert& d = experts[eids[slot]];
  for (uint b = sg; b < IK_XD_I / 128u; b += 4u) {
    const uint base = b * 128u;
    const size_t ab = (size_t)slot * IK_XD_I + base;
    const float4 s = ik_xd_h128(float4(float(act[ab + lane]) * float(d.suh[base + lane]),
                                       float(act[ab + lane + 32u]) * float(d.suh[base + lane + 32u]),
                                       float(act[ab + lane + 64u]) * float(d.suh[base + lane + 64u]),
                                       float(act[ab + lane + 96u]) * float(d.suh[base + lane + 96u])), lane);
    prepared[base + lane] = half(s.x * IK_XD_PSC);
    prepared[base + lane + 32u] = half(s.y * IK_XD_PSC);
    prepared[base + lane + 64u] = half(s.z * IK_XD_PSC);
    prepared[base + lane + 96u] = half(s.w * IK_XD_PSC);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc[2][8] = {};
  ik_xd_gemv(d, 0u, ot, prepared, sg, lane, acc);
  ik_xd_reduce(acc, partial, sums + (size_t)slot * IK_XD_H, ot, sg, lane, lid);
}

// The output Hadamard of each pick's sums, ey = bf16 per pick, then the weighted sum in pick order; `Dbg` also stores ey.
template <bool Dbg>
static inline void ik_xd_finish_body(const device float* sums, const device uint* eids,
    const device ik_xd_expert* experts, const device float* ew, device bfloat* moe, device bfloat* ey,
    threadgroup float* vals, uint2 tg, uint sg, uint lane) {
  const uint blk = tg.x, row = tg.y;
  const uint slot = row * IK_XD_TOPK + sg;
  const uint base = blk * 128u;
  const device half* svh = experts[eids[slot]].svh;
  const size_t xb = (size_t)slot * IK_XD_H + base;
  const float4 s = ik_xd_h128(float4(sums[xb + lane], sums[xb + lane + 32u], sums[xb + lane + 64u], sums[xb + lane + 96u]), lane);
  const bfloat e0 = bfloat(s.x * IK_XD_PSC * float(svh[base + lane]));
  const bfloat e1 = bfloat(s.y * IK_XD_PSC * float(svh[base + lane + 32u]));
  const bfloat e2 = bfloat(s.z * IK_XD_PSC * float(svh[base + lane + 64u]));
  const bfloat e3 = bfloat(s.w * IK_XD_PSC * float(svh[base + lane + 96u]));
  vals[sg * 128u + lane] = float(e0);
  vals[sg * 128u + lane + 32u] = float(e1);
  vals[sg * 128u + lane + 64u] = float(e2);
  vals[sg * 128u + lane + 96u] = float(e3);
  if (Dbg) {
    ey[xb + lane] = e0;
    ey[xb + lane + 32u] = e1;
    ey[xb + lane + 64u] = e2;
    ey[xb + lane + 96u] = e3;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint k = 0u; k < IK_XD_TOPK; k++) {
      const float w = ew[row * IK_XD_TOPK + k];
      a0 = fma(vals[k * 128u + lane], w, a0);
      a1 = fma(vals[k * 128u + lane + 32u], w, a1);
      a2 = fma(vals[k * 128u + lane + 64u], w, a2);
      a3 = fma(vals[k * 128u + lane + 96u], w, a3);
    }
    const size_t mb = (size_t)row * IK_XD_H + base;
    moe[mb + lane] = bfloat(a0);
    moe[mb + lane + 32u] = bfloat(a1);
    moe[mb + lane + 64u] = bfloat(a2);
    moe[mb + lane + 96u] = bfloat(a3);
  }
}

[[kernel]] void ik_xd_finish(constant ik_args& a [[buffer(0)]], const device float* sums [[buffer(1)]],
    const device uint* eids [[buffer(2)]], const device ik_xd_expert* experts [[buffer(3)]],
    const device float* ew [[buffer(4)]], device bfloat* moe [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float vals[IK_XD_TOPK * 128];
  ik_xd_finish_body<false>(sums, eids, experts, ew, moe, moe, vals, tg, sg, lane);
}

[[kernel]] void ik_xd_finish_dbg(constant ik_args& a [[buffer(0)]], const device float* sums [[buffer(1)]],
    const device uint* eids [[buffer(2)]], const device ik_xd_expert* experts [[buffer(3)]],
    const device float* ew [[buffer(4)]], device bfloat* moe [[buffer(5)]], device bfloat* ey [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float vals[IK_XD_TOPK * 128];
  ik_xd_finish_body<true>(sums, eids, experts, ew, moe, ey, vals, tg, sg, lane);
}
