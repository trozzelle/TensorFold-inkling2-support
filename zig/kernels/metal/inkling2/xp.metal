// Inkling 2's routed experts for a prompt chunk: the EXL3 prefill chain on the tensor units, from the MoE input of
// 1 to thousands of rows to the weighted sum. A slot is row * 8 + pick. Every row's arithmetic is independent of the
// rows beside it, of how a chunk is cut and of which rows share a window: a prompt cut anywhere gives the same bits.
//
// Adapted from Sushi v1.3.0 (commit 8e4c0af1; MIT License, see LICENSES/Sushi-MIT.txt and THIRD_PARTY_NOTICES.md), as
// carried in mlx-serve's src/sdk/quant/exl3/sushi: kernels/token_pair_prepare.metal (the input rotation),
// kernels/group_window.metal (the 32-row windows of one expert), kernels/gemm_nax.metal, gemm_nax_open_loop.metal,
// gemm_nax_decode_lo.metal and gemm_nax_frags.metal (the tensor-unit GEMM over decoded trellis tiles), kernels/mid.metal
// (the middle stage's rotations), kernels/reduce.metal (the finish) and the sorted-table dispatch of
// expert_exl3_kernels.zig (innerGemmSortedTable). What differs here: MLX's argsort is a counting sort by expert, one
// threadgroup per expert, stable by construction; gate and up are one projection of 4608 outputs with one input
// rotation; the sums are stored f32 and the output rotations round at the reference's points (the decode kernels'
// mid and finish, xd.metal); weights are reached through ExpertDescriptor tables (core/exl3_projection.zig), one
// dispatch per rate (K2 or K3) and projection over that rate's windows, the windows counted on the GPU and
// dispatched indirectly.
//
// Dispatches (rows n, S = 8 n slots; ids u32 [S] in pick order, 16-byte aligned (the sort reads it as uint4); ew f32
// [S]; x bf16 [n][4608]; the descriptor tables of w13 and w2). `a` is ik_xp_args at buffer 0. Buffers 1.. in order:
//   ik_xp_count    groups (E) x 256:           ids, w13, w2 -> counts u32 [E], classes u32 [E]
//   ik_xp_place    groups (E + 1) x 256:       ids, counts, classes -> order, seid, inverse u32 [S], windows, indirect arguments
//   ik_xp_prepare  groups (9, S) x 128:        x, order, seid, w13 -> prep f16 [S][4608]
//   ik_xp_gemm13_* groups (indirect) x 128:    prep, windows, w13 -> plane f32 [S][4608]      (_k2 and _k3: the windows of that rate)
//   ik_xp_mid      groups (3, S) x 192:        plane, seid, w13, w2 -> prep f16 [S][2304]    (the down projection's rotated input)
//   ik_xp_gemm2_*  groups (indirect) x 128:    prep, windows, w2 -> plane f32 [S][4608]
//   ik_xp_finish   groups (36, n) x 256:       plane, ids, inverse, w2, ew -> moe bf16 [n][4608]
// The mid and finish also exist as _dbg kernels: the mid also stores act bf16 [S][2304] (buffer 6) and gu bf16
// [S][4608] (buffer 7), the finish the per-slot ey bf16 [S][4608] (buffer 7); the checks read them, the forward does not.
//
// Sorted order: slots by expert id, ascending slot within an expert; `order[pos]` is the slot at sorted position pos,
// `inverse[slot]` its position, `seid[pos]` its expert (0xffffffff past the placed slots, for a pick whose expert id is
// not in the table). A window is up to IK_XP_WINDOW consecutive sorted positions of one expert (ik_xp_win); the
// windows of one (projection, rate) list are in expert order.
//
// Order contract. One output element's sums, in this order; any kernel that shares an expert's weights among slots
// must reproduce it for each slot.
//   prepare: as xd.metal's pair: x_i * suh_i in f32, ik_xd_h128's butterfly, * IK_XD_PSC, rounded to f16 (all of K,
//            one plane).
//   gemm:    plane[pos][col] = the f32 sum over the input tiles tk = 0, 1, .. in increasing order of one 16-deep
//            f16 multiply-accumulate on the tensor unit each, from 0.0f (about 6e-7 relative RMS against the exact
//            sum of the same f16 rows; the decode kernel's 32 short chains give 1.2e-7). A row's value does not depend
//            on its position in the window or on the rows beside it: a padded row reads a clamped neighbour and is
//            never stored.
//   mid:     v = plane[pos][col]; the 128-column-block Hadamard; v * IK_XD_PSC * svh; gu = bf16(v). Pair j is
//            gu[2j] (gate g) and gu[2j + 1] (up u): act[j] = bf16(capped SwiGLU) as xd.metal's mid; then
//            float(act) * suh2, the 128-block Hadamard over 128 acts, * IK_XD_PSC, f16.
//   finish:  the 128-column-block Hadamard of plane[inverse[slot]], * IK_XD_PSC * svh2; ey = bf16 per pick;
//            moe[row][col] = bf16(fma(ey_7, w_7, .. fma(ey_1, w_1, fma(ey_0, w_0, 0.0f)))).
#include "../nax.h"

// IK_XP_WINDOW (experts_prefill.zig's `window`, defined ahead of this text): the rows of a window, a multiple of 16.
constant uint IK_XP_WIN = IK_XP_WINDOW;
constant uint IK_XP_BLK = IK_XP_WIN / 16u;  // its 16-row tensor-unit blocks

// forward-side ik_xp_args, field for field (experts_prefill.zig).
struct ik_xp_args {
  uint rows;      // prompt rows n
  uint slots;     // 8 n
  uint experts;   // entries in each descriptor table
  uint win_cap;   // windows each list has room for
  uint gx;        // threadgroups across a window: outputs / 128
  float cap, thr; // the capped SwiGLU's gate and up caps
};

// A window of one expert: sorted positions [start, start + nlive), nlive 1..IK_XP_WINDOW.
struct ik_xp_win {
  uint start, nlive, eid, pad;
};

// ---- grouping ----

// Slots per expert, and each expert's rate class: bit 0 w13 is K3, bit 1 w2 is K3 (K2 otherwise).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik_xp_count(constant ik_xp_args& a [[buffer(0)]],
    const device uint4* ids [[buffer(1)]], const device ik_xd_expert* w13 [[buffer(2)]],
    const device ik_xd_expert* w2 [[buffer(3)]], device uint* counts [[buffer(4)]], device uint* classes [[buffer(5)]],
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
    classes[e] = (w13[e].halves == 6u ? 1u : 0u) | (w2[e].halves == 6u ? 2u : 0u);
  }
}

// One threadgroup per expert writes its slots, in slot order, at the sum of the counts before it, and its windows at
// the end of each of its two lists (w13's and w2's by rate class). The extra threadgroup writes the indirect arguments.
// Lists 0..3 are w13 K2, w13 K3, w2 K2, w2 K3; list l's windows start at l * a.win_cap.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik_xp_place(constant ik_xp_args& a [[buffer(0)]],
    const device uint4* ids [[buffer(1)]], const device uint* counts [[buffer(2)]],
    const device uint* classes [[buffer(3)]], device uint* order [[buffer(4)]], device uint* seid [[buffer(5)]],
    device uint* inverse [[buffer(6)]], device ik_xp_win* wins [[buffer(7)]], device uint* dargs [[buffer(8)]],
    uint e [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint part[8][5];
  threadgroup uint base[8];
  const uint lim = min(e, a.experts);
  // sums over the experts before this one: slots, then the windows of lists 0..3
  uint s_slots = 0u, s_l0 = 0u, s_l1 = 0u, s_l2 = 0u, s_l3 = 0u;
  for (uint k = tid; k < lim; k += 256u) {
    const uint c = counts[k];
    const uint cl = classes[k];
    const uint wc = (c + IK_XP_WIN - 1u) / IK_XP_WIN;
    s_slots += c;
    s_l0 += (cl & 1u) == 0u ? wc : 0u;
    s_l1 += (cl & 1u) != 0u ? wc : 0u;
    s_l2 += (cl & 2u) == 0u ? wc : 0u;
    s_l3 += (cl & 2u) != 0u ? wc : 0u;
  }
  s_slots = simd_sum(s_slots);
  s_l0 = simd_sum(s_l0);
  s_l1 = simd_sum(s_l1);
  s_l2 = simd_sum(s_l2);
  s_l3 = simd_sum(s_l3);
  if (lane == 0u) {
    part[sg][0] = s_slots;
    part[sg][1] = s_l0;
    part[sg][2] = s_l1;
    part[sg][3] = s_l2;
    part[sg][4] = s_l3;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint t[5] = {0u, 0u, 0u, 0u, 0u};
  for (uint g = 0u; g < 8u; g++) {
    for (uint i = 0u; i < 5u; i++) t[i] += part[g][i];
  }
  if (e >= a.experts) {
    // the dispatch arguments: a window's threadgroups across, the list's windows down
    if (tid < 4u) {
      dargs[tid * 4u] = a.gx;
      dargs[tid * 4u + 1u] = t[1u + tid];
      dargs[tid * 4u + 2u] = 1u;
      dargs[tid * 4u + 3u] = 0u;
    }
    // a slot whose expert id is not in the table is not placed: the positions after the placed ones hold no expert
    for (uint pos = t[0] + tid; pos < a.slots; pos += 256u) seid[pos] = 0xffffffffu;
    return;
  }
  const uint n = counts[e];
  if (n == 0u) return;
  const uint cl = classes[e];
  const uint off = t[0];
  // this expert's slots in slot order: each thread takes a run of ids, counts its matches, the threadgroup scans
  // the counts, and each thread writes its matches from its rank on
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
  const uint nw = (n + IK_XP_WIN - 1u) / IK_XP_WIN;
  const uint wb13 = (cl & 1u) != 0u ? t[2] : t[1];
  const uint wb2 = (cl & 2u) != 0u ? t[4] : t[3];
  device ik_xp_win* list13 = wins + ((cl & 1u) != 0u ? 1u : 0u) * a.win_cap + wb13;
  device ik_xp_win* list2 = wins + (2u + ((cl & 2u) != 0u ? 1u : 0u)) * a.win_cap + wb2;
  for (uint i = tid; i < nw; i += 256u) {
    ik_xp_win w;
    w.start = off + i * IK_XP_WIN;
    w.nlive = min(IK_XP_WIN, n - i * IK_XP_WIN);
    w.eid = e;
    w.pad = 0u;
    list13[i] = w;
    list2[i] = w;
  }
}

// ---- prepare ----

// x * suh of each sorted slot's expert, the 128-block Hadamard and the scale, f16: the w13 GEMM's input rows.
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void ik_xp_prepare(constant ik_xp_args& a [[buffer(0)]],
    const device bfloat* x [[buffer(1)]], const device uint* order [[buffer(2)]], const device uint* seid [[buffer(3)]],
    const device ik_xd_expert* w13 [[buffer(4)]], device half* prep [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint pos = tg.y;
  const uint e = seid[pos];
  if (e >= a.experts) return;  // a hole left by a pick outside the table
  const device ik_xd_expert& d = w13[e];
  const uint base = (tg.x * 4u + sg) * 128u;
  const size_t xr = (size_t)(order[pos] / IK_XD_TOPK) * IK_XD_H + base;
  const float x0 = float(x[xr + lane]);
  const float x1 = float(x[xr + lane + 32u]);
  const float x2 = float(x[xr + lane + 64u]);
  const float x3 = float(x[xr + lane + 96u]);
  const float4 s = ik_xd_h128(float4(x0 * float(d.suh[base + lane]), x1 * float(d.suh[base + lane + 32u]),
                                     x2 * float(d.suh[base + lane + 64u]), x3 * float(d.suh[base + lane + 96u])), lane);
  const size_t pb = (size_t)pos * IK_XD_H + base;
  prep[pb + lane] = half(s.x * IK_XD_PSC);
  prep[pb + lane + 32u] = half(s.y * IK_XD_PSC);
  prep[pb + lane + 64u] = half(s.z * IK_XD_PSC);
  prep[pb + lane + 96u] = half(s.w * IK_XD_PSC);
}

// ---- the grouped tensor-unit GEMM ----

// The weight fragment of a lane for one 16x16 tile (the right operand of nax.h's mma_16x32<false, true>: element e is
// the weight of input row home.x + (e & 3), output column home.y + 8 * (e >> 2)): eight codewords of the tile's
// stream, decoded in registers. A lane's weights are the stream's 16m + 2b + {0, 1, 4, 5} and the same + 8 (b is bit 3
// of the lane), so two 32-bit windows serve all eight. Eight weights span N / 2 whole bits, so weight j of a group
// ends ik_xp_end(N, b, j) bits past the group's first bit.
static constexpr uint ik_xp_end(uint N, uint b, uint j) { return ((2u * b + j + 1u) * N) >> 4u; }
static inline uint ik_xp_sh(uint N, uint b, uint j) {
  const uint s0 = ik_xp_end(N, 0u, 5u) - ik_xp_end(N, 0u, j);
  return s0 + b * (ik_xp_end(N, 1u, 5u) - ik_xp_end(N, 1u, j) - s0);
}

template <uint N>
static inline tfp::frag<half> ik_xp_wfrag(const device uint* words, uint lane) {
  static_assert(N == 32u || N == 48u, "the prefill chain reads K2 and K3 tiles");
  // a quad's codewords lie within 16 bits of its first end, so 32 stream bits hold each group
  static_assert(ik_xp_end(N, 0u, 5u) - ik_xp_end(N, 0u, 0u) <= 16u && ik_xp_end(N, 1u, 5u) - ik_xp_end(N, 1u, 0u) <= 16u,
                "a quad of weights fits 32 stream bits");
  constexpr uint W = N / 2u;
  const uint b = (lane >> 3u) & 1u;
  const uint end = 8u * N * (lane >> 4u) + N * (lane & 7u) + (ik_xp_end(N, 1u, 5u) - ik_xp_end(N, 0u, 5u)) * b + ik_xp_end(N, 0u, 5u);
  const uint s0 = ik_xp_sh(N, b, 0u);
  const uint s1 = ik_xp_sh(N, b, 1u);
  const uint s4 = ik_xp_sh(N, b, 4u);
  const uint lo = uint(ik_xd_window(words, end, W));
  const uint hi = uint(ik_xd_window(words, end + W, W));
  const half2 p0 = ik_xd_pairh(uint2((lo >> s0) & 0xffffu, (lo >> s1) & 0xffffu));
  const half2 p1 = ik_xd_pairh(uint2((lo >> s4) & 0xffffu, lo & 0xffffu));
  const half2 p2 = ik_xd_pairh(uint2((hi >> s0) & 0xffffu, (hi >> s1) & 0xffffu));
  const half2 p3 = ik_xd_pairh(uint2((hi >> s4) & 0xffffu, hi & 0xffffu));
  return tfp::frag<half>(p0.x, p0.y, p2.x, p2.y, p1.x, p1.y, p3.x, p3.y);
}

// One simdgroup's 32 output columns of a window over the input tiles: one multiply-accumulate per tile and 16-row
// block, tiles in increasing order. `LIVE` is the window's 16-row blocks that hold rows: blocks past the window's last
// row are not computed. The lane's rows of block b are rows hy and hy + 8 of it, clamped to the last sorted row, so a
// padded row is a live neighbour and its products are never stored.
template <uint N, uint IDIM, uint LIVE>
static inline void ik_xp_tiles(const device uint* wp, const device half* x, uint start, uint last, uint lane,
    thread tfp::frag<float> (&acc)[IK_XP_BLK][2]) {
  constexpr uint W = N / 2u;
  constexpr uint IT = IDIM / 16u;
  const short2 home = tfp::frag_home(ushort(lane));
  const uint hx = uint(home.x), hy = uint(home.y);
  const device half* xr[2 * LIVE];
  TF_UNROLL
  for (uint b = 0u; b < LIVE; b++) {
    xr[2u * b] = x + (size_t)min(start + 16u * b + hy, last) * IDIM + hx;
    xr[2u * b + 1u] = x + (size_t)min(start + 16u * b + hy + 8u, last) * IDIM + hx;
  }
#pragma clang loop unroll_count(2)
  for (uint tk = 0u; tk < IT; tk++) {
    const tfp::frag<half> b0 = ik_xp_wfrag<N>(wp, lane);
    const tfp::frag<half> b1 = ik_xp_wfrag<N>(wp + W, lane);
    wp += IK_XD_OT * W;
    const uint k = tk * 16u;
    TF_UNROLL
    for (uint b = 0u; b < LIVE; b++) {
      const tfp::frag<half> rows = tfp::frag<half>(*(const device half4*)(xr[2u * b] + k), *(const device half4*)(xr[2u * b + 1u] + k));
      tfp::mma_16x32<false, true>(acc[b][0], acc[b][1], rows, b0, b1);
    }
  }
}

// ik_xp_tiles for the window's `live` blocks, a count the whole simdgroup shares.
template <uint N, uint IDIM, uint LIVE>
static inline void ik_xp_blocks(uint live, const device uint* wp, const device half* x, uint start, uint last, uint lane,
    thread tfp::frag<float> (&acc)[IK_XP_BLK][2]) {
  if (live == LIVE) {
    ik_xp_tiles<N, IDIM, LIVE>(wp, x, start, last, lane, acc);
    return;
  }
  if constexpr (LIVE > 1u) ik_xp_blocks<N, IDIM, LIVE - 1u>(live, wp, x, start, last, lane, acc);
}

// The lane's f32 values of a 16x16 block to its two rows, four columns each; a row past the window is not stored.
static inline void ik_xp_store(thread const tfp::frag<float>& f, device float* r0, device float* r1, bool l0, bool l1) {
  if (l0) *(device float4*)r0 = float4(f[0], f[1], f[2], f[3]);
  if (l1) *(device float4*)r1 = float4(f[4], f[5], f[6], f[7]);
}

// One window (threadgroup row tg.y) of a rate-N list and the simdgroup's 32 of the threadgroup's 128 output columns:
// rows x [IDIM] f16 by the window's expert's tiles.
template <uint N, uint IDIM>
static inline void ik_xp_gemm(constant ik_xp_args& a, const device half* x, const device ik_xp_win* wins,
    const device ik_xd_expert* experts, device float* y, uint3 tg, uint sg, uint lane) {
  constexpr uint W = N / 2u;
  const ik_xp_win w = wins[tg.y];
  const uint ot = tg.x * 8u + sg * 2u;
  const device uint* wp = (const device uint*)experts[w.eid].trellis + (size_t)ot * W;
  tfp::frag<float> acc[IK_XP_BLK][2];
  TF_UNROLL
  for (uint b = 0u; b < IK_XP_BLK; b++) {
    acc[b][0] = tfp::frag<float>(0.0f);
    acc[b][1] = tfp::frag<float>(0.0f);
  }
  ik_xp_blocks<N, IDIM, IK_XP_BLK>((w.nlive + 15u) / 16u, wp, x, w.start, a.slots - 1u, lane, acc);
  const short2 home = tfp::frag_home(ushort(lane));
  const uint hx = uint(home.x), hy = uint(home.y);
  device float* y0 = y + (size_t)(w.start + hy) * IK_XD_H + ot * 16u + hx;
  const size_t eight = (size_t)8u * IK_XD_H;
  TF_UNROLL
  for (uint b = 0u; b < IK_XP_BLK; b++) {
    if (w.nlive > 16u * b) {
      device float* yb = y0 + (size_t)b * 2u * eight;
      const bool l0 = hy + 16u * b < w.nlive, l1 = hy + 16u * b + 8u < w.nlive;
      ik_xp_store(acc[b][0], yb, yb + eight, l0, l1);
      ik_xp_store(acc[b][1], yb + 16u, yb + eight + 16u, l0, l1);
    }
  }
}

#define IK_XP_GEMM_KERNEL(NAME, N, IDIM)                                                                           \
  [[max_total_threads_per_threadgroup(128)]] [[kernel]] void NAME(constant ik_xp_args& a [[buffer(0)]],           \
      const device half* x [[buffer(1)]], const device ik_xp_win* wins [[buffer(2)]],                             \
      const device ik_xd_expert* experts [[buffer(3)]], device float* y [[buffer(4)]],                            \
      uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                       \
      uint lane [[thread_index_in_simdgroup]]) {                                                                   \
    ik_xp_gemm<N, IDIM>(a, x, wins, experts, y, tg, sg, lane);                                                     \
  }

IK_XP_GEMM_KERNEL(ik_xp_gemm13_k2, 32u, 4608u)
IK_XP_GEMM_KERNEL(ik_xp_gemm13_k3, 48u, 4608u)
IK_XP_GEMM_KERNEL(ik_xp_gemm2_k2, 32u, 2304u)
IK_XP_GEMM_KERNEL(ik_xp_gemm2_k3, 48u, 2304u)

// ---- mid ----

// The w13 plane's 128-column block `blk` (36 of them) of a sorted slot to its 64 capped-SwiGLU acts: an even lane
// returns the four it holds, act 16 q + (lane >> 1) of the block at .x .y .z .w for q = 0..3 (zeros at an odd lane).
// `Dbg` also stores gu and act.
template <bool Dbg>
static inline float4 ik_xp_acts(constant ik_xp_args& a, const device float* plane, const device half* svh, uint pos,
    uint blk, device bfloat* act, device bfloat* gu, uint lane) {
  const uint base = blk * 128u;
  const size_t pb = (size_t)pos * IK_XD_H + base;
  const float4 s = ik_xd_h128(float4(plane[pb + lane], plane[pb + lane + 32u], plane[pb + lane + 64u], plane[pb + lane + 96u]), lane);
  // gu rounds to bf16 here; the pair's partner is the neighbouring lane (the same q, index bit 0)
  const float g0 = float(bfloat(s.x * IK_XD_PSC * float(svh[base + lane])));
  const float g1 = float(bfloat(s.y * IK_XD_PSC * float(svh[base + lane + 32u])));
  const float g2 = float(bfloat(s.z * IK_XD_PSC * float(svh[base + lane + 64u])));
  const float g3 = float(bfloat(s.w * IK_XD_PSC * float(svh[base + lane + 96u])));
  if (Dbg) {
    gu[pb + lane] = bfloat(g0);
    gu[pb + lane + 32u] = bfloat(g1);
    gu[pb + lane + 64u] = bfloat(g2);
    gu[pb + lane + 96u] = bfloat(g3);
  }
  const float u0 = simd_shuffle_xor(g0, 1);
  const float u1 = simd_shuffle_xor(g1, 1);
  const float u2 = simd_shuffle_xor(g2, 1);
  const float u3 = simd_shuffle_xor(g3, 1);
  float4 v = float4(0.0f);
  if ((lane & 1u) == 0u) {
    // an even lane holds the gate, its partner the up
    v.x = float(bfloat((tanh(g0 / a.cap) * a.cap) * (1.0f / (1.0f + exp(-g0))) * (tanh(u0 / a.thr) * a.thr)));
    v.y = float(bfloat((tanh(g1 / a.cap) * a.cap) * (1.0f / (1.0f + exp(-g1))) * (tanh(u1 / a.thr) * a.thr)));
    v.z = float(bfloat((tanh(g2 / a.cap) * a.cap) * (1.0f / (1.0f + exp(-g2))) * (tanh(u2 / a.thr) * a.thr)));
    v.w = float(bfloat((tanh(g3 / a.cap) * a.cap) * (1.0f / (1.0f + exp(-g3))) * (tanh(u3 / a.thr) * a.thr)));
    if (Dbg) {
      const size_t ab = (size_t)pos * IK_XD_I + blk * 64u + (lane >> 1u);
      act[ab] = bfloat(v.x);
      act[ab + 16u] = bfloat(v.y);
      act[ab + 32u] = bfloat(v.z);
      act[ab + 48u] = bfloat(v.w);
    }
  }
  return v;
}

// A sorted slot's w13 plane to the down projection's rotated input: one simdgroup takes two 128-column blocks of w13,
// which are 128 acts (act block m: acts [128 m, 128 m + 128), w13 columns [256 m, 256 m + 256)).
template <bool Dbg>
static inline void ik_xp_mid_body(constant ik_xp_args& a, const device float* plane, const device uint* seid,
    const device ik_xd_expert* w13, const device ik_xd_expert* w2, device half* mid, device bfloat* act,
    device bfloat* gu, uint2 tg, uint sg, uint lane) {
  const uint pos = tg.y;
  const uint m = tg.x * 6u + sg;
  const uint e = seid[pos];
  if (e >= a.experts) return;  // a hole left by a pick outside the table
  const device half* svh = w13[e].svh;
  const device half* suhd = w2[e].suh;
  const float4 v0 = ik_xp_acts<Dbg>(a, plane, svh, pos, 2u * m, act, gu, lane);
  const float4 v1 = ik_xp_acts<Dbg>(a, plane, svh, pos, 2u * m + 1u, act, gu, lane);
  // act 64 b + 16 q + (L >> 1) of even lane L (block b, register q) is lane a & 31, register a >> 5 of a vector of 128:
  // lane l takes register r from block b = r >> 1, q = 2 (r & 1) + (l >> 4), lane L = 2 (l & 15). Every lane shuffles
  // both candidates of every register and then keeps its own: a shuffle in a branch that only some lanes take would
  // read lanes that are not executing it, which MSL leaves undefined.
  const ushort from = ushort(2u * (lane & 15u));
  const bool up = lane >= 16u;
  const float a0 = simd_shuffle(v0.x, from), b0 = simd_shuffle(v0.y, from);
  const float a1 = simd_shuffle(v0.z, from), b1 = simd_shuffle(v0.w, from);
  const float a2 = simd_shuffle(v1.x, from), b2 = simd_shuffle(v1.y, from);
  const float a3 = simd_shuffle(v1.z, from), b3 = simd_shuffle(v1.w, from);
  const float t0 = up ? b0 : a0;
  const float t1 = up ? b1 : a1;
  const float t2 = up ? b2 : a2;
  const float t3 = up ? b3 : a3;
  const uint ub = m * 128u;
  const float4 r = ik_xd_h128(float4(t0 * float(suhd[ub + lane]), t1 * float(suhd[ub + lane + 32u]),
                                     t2 * float(suhd[ub + lane + 64u]), t3 * float(suhd[ub + lane + 96u])), lane);
  const size_t mb = (size_t)pos * IK_XD_I + ub;
  mid[mb + lane] = half(r.x * IK_XD_PSC);
  mid[mb + lane + 32u] = half(r.y * IK_XD_PSC);
  mid[mb + lane + 64u] = half(r.z * IK_XD_PSC);
  mid[mb + lane + 96u] = half(r.w * IK_XD_PSC);
}

[[max_total_threads_per_threadgroup(192)]] [[kernel]] void ik_xp_mid(constant ik_xp_args& a [[buffer(0)]],
    const device float* plane [[buffer(1)]], const device uint* seid [[buffer(2)]],
    const device ik_xd_expert* w13 [[buffer(3)]], const device ik_xd_expert* w2 [[buffer(4)]],
    device half* mid [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  ik_xp_mid_body<false>(a, plane, seid, w13, w2, mid, (device bfloat*)mid, (device bfloat*)mid, tg, sg, lane);
}

[[max_total_threads_per_threadgroup(192)]] [[kernel]] void ik_xp_mid_dbg(constant ik_xp_args& a [[buffer(0)]],
    const device float* plane [[buffer(1)]], const device uint* seid [[buffer(2)]],
    const device ik_xd_expert* w13 [[buffer(3)]], const device ik_xd_expert* w2 [[buffer(4)]],
    device half* mid [[buffer(5)]], device bfloat* act [[buffer(6)]], device bfloat* gu [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  ik_xp_mid_body<true>(a, plane, seid, w13, w2, mid, act, gu, tg, sg, lane);
}

// ---- finish ----

// The output rotation of each pick's w2 sums (row `inverse[slot]` of the plane), ey = bf16 per pick, then the weighted
// sum in pick order. A pick whose expert id is not in the table has no sorted row: its ey is NaN, so its row's output
// is, and no other row is touched. `Dbg` also stores ey.
template <bool Dbg>
static inline void ik_xp_finish_body(constant ik_xp_args& a, const device float* sums, const device uint* ids,
    const device uint* inverse, const device ik_xd_expert* w2, const device float* ew, device bfloat* moe,
    device bfloat* ey, threadgroup float* vals, uint2 tg, uint sg, uint lane) {
  const uint blk = tg.x, row = tg.y;
  const uint slot = row * IK_XD_TOPK + sg;
  const uint base = blk * 128u;
  const uint id = ids[slot];
  const bool ok = id < a.experts;
  const device half* svh = w2[ok ? id : 0u].svh;
  const size_t xb = (size_t)(ok ? inverse[slot] : 0u) * IK_XD_H + base;
  const float4 v = ok ? float4(sums[xb + lane], sums[xb + lane + 32u], sums[xb + lane + 64u], sums[xb + lane + 96u])
                      : float4(NAN);
  const float4 s = ik_xd_h128(v, lane);
  const bfloat e0 = bfloat(s.x * IK_XD_PSC * float(svh[base + lane]));
  const bfloat e1 = bfloat(s.y * IK_XD_PSC * float(svh[base + lane + 32u]));
  const bfloat e2 = bfloat(s.z * IK_XD_PSC * float(svh[base + lane + 64u]));
  const bfloat e3 = bfloat(s.w * IK_XD_PSC * float(svh[base + lane + 96u]));
  vals[sg * 128u + lane] = float(e0);
  vals[sg * 128u + lane + 32u] = float(e1);
  vals[sg * 128u + lane + 64u] = float(e2);
  vals[sg * 128u + lane + 96u] = float(e3);
  if (Dbg) {
    const size_t eb = (size_t)slot * IK_XD_H + base;
    ey[eb + lane] = e0;
    ey[eb + lane + 32u] = e1;
    ey[eb + lane + 64u] = e2;
    ey[eb + lane + 96u] = e3;
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

[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik_xp_finish(constant ik_xp_args& a [[buffer(0)]],
    const device float* sums [[buffer(1)]], const device uint* ids [[buffer(2)]], const device uint* inverse [[buffer(3)]],
    const device ik_xd_expert* w2 [[buffer(4)]], const device float* ew [[buffer(5)]], device bfloat* moe [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float vals[IK_XD_TOPK * 128];
  ik_xp_finish_body<false>(a, sums, ids, inverse, w2, ew, moe, moe, vals, tg, sg, lane);
}

[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik_xp_finish_dbg(constant ik_xp_args& a [[buffer(0)]],
    const device float* sums [[buffer(1)]], const device uint* ids [[buffer(2)]], const device uint* inverse [[buffer(3)]],
    const device ik_xd_expert* w2 [[buffer(4)]], const device float* ew [[buffer(5)]], device bfloat* moe [[buffer(6)]],
    device bfloat* ey [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float vals[IK_XD_TOPK * 128];
  ik_xp_finish_body<true>(a, sums, ids, inverse, w2, ew, moe, ey, vals, tg, sg, lane);
}
