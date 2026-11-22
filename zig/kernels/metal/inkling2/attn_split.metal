// Inkling 2's split-K decode attention. ik_attn_part takes each 256-key block of a row's visible keys, blocks anchored at
// absolute positions, to a partial softmax; ik_attn_merge folds a row's blocks in ascending order from the sink logit
// and applies the output gate. A row's result depends on that row alone: not on the window's other rows, its size, or
// which blocks the grid covers (blocks outside a row's range are empty and skipped). Both kernels take 48 q heads in 8
// kv groups of 6 and 128 dims, and every sum has one fixed order.
#include <metal_stdlib>
using namespace metal;

// Keys a block covers, and floats a (row, head, block) partial takes: m, l, then acc[128].
constant uint IK_AS_KEYS = 256, IK_AS_SLOT = 130;

// One step of a transposed sum over the simdgroup: v[0..2N) hold two halves of N slots, lanes with bit N set keep the
// upper half, and each kept slot adds the partner lane's value for it.
template <uint N>
inline void ik_as_halve(thread float* v, uint lane) {
  const bool hi = (lane & N) != 0;
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < N; ++i) {
    const float keep = hi ? v[i + N] : v[i], send = hi ? v[i] : v[i + N];
    v[i] = keep + simd_shuffle_xor(send, ushort(N));
  }
}

// Sums each of v[0..32) over the 32 lanes in one fixed tree (lanes paired by bits 4, 3, 2, 1, 0): lane l gets slot l's
// total. Every slot's total comes from the same tree, so it does not depend on which slot or lane carries it.
inline float ik_as_scatter(thread float* v, uint lane) {
  ik_as_halve<16>(v, lane);
  ik_as_halve<8>(v, lane);
  ik_as_halve<4>(v, lane);
  ik_as_halve<2>(v, lane);
  ik_as_halve<1>(v, lane);
  return v[0];
}

// A simdgroup's relative biases, from the row's relative state and the projection: for each of its two q heads, the bias at
// the distance of a lane's key behind the row, bf16(rel . proj) summed over the 16 dims in index order, the value
// ik_relbias writes. `two` fills out[0], out[1] for heads `head` and `head + 1` of row t at `dist` < extent.
struct IkRelFold {
  const device bfloat* reln; // [rows][heads][16]
  const device bfloat* proj; // [16][extent]
  inline void two(uint t, uint heads, uint head, uint extent, uint dist, thread float* out) const {
    float pv[IK_DREL];
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < IK_DREL; ++i) pv[i] = float(proj[ulong(i) * extent + dist]);
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < 2; ++j) {
      const device bfloat4* r = (const device bfloat4*)(reln + (ulong(t) * heads + head + j) * IK_DREL);
      float s = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (uint c = 0; c < IK_DREL / 4; ++c) {
        const float4 v = float4(r[c]);
        s += v.x * pv[4 * c];
        s += v.y * pv[4 * c + 1];
        s += v.z * pv[4 * c + 2];
        s += v.w * pv[4 * c + 3];
      }
      out[j] = float(bfloat(s));
    }
  }
};

// The partial kernel's body, for any source of the relative bias (`Bias::two`): the threadgroup's three arrays come from the kernel.
template <typename Bias>
inline void ik_attn_part_run(constant ik_args& a, const device bfloat* qn, const device bfloat* keys, const device bfloat* values,
    Bias bias, device float* part, threadgroup float (&tmax)[8][6], threadgroup float (&tsum)[8][6],
    threadgroup float4 (&tacc)[8][6][32], uint3 tg, uint tid, uint sg, uint lane) {
  const uint D = a.dim, g = tg.x, t = tg.y, p = a.pos + t;
  const uint lo = (a.window > 0 && p + 1 > a.window) ? p + 1 - a.window : 0;
  const uint base = (a.queries + tg.z) * IK_AS_KEYS;
  // the block's first slot: a ring is a multiple of the block, or holds every position, so a block never wraps
  const uint wrap = base - base % a.ring;
  const uint from = max(lo, base), to = min(p + 1, base + IK_AS_KEYS);
  device float* slots = part + ((ulong(t) * a.heads + g * 6) * a.blocks + tg.z) * IK_AS_SLOT;
  const ulong head_step = ulong(a.blocks) * IK_AS_SLOT;
  if (from >= to) {
    for (uint i = tid; i < 6 * IK_AS_SLOT; i += 768)
      slots[(i / IK_AS_SLOT) * head_step + i % IK_AS_SLOT] = i % IK_AS_SLOT == 0 ? -INFINITY : 0.0f;
    return;
  }
  const uint kg = sg & 7, h0 = 2 * (sg >> 3);
  const uint k0 = base + 32 * kg, key = k0 + lane;
  const bool reads_bias = key >= from && key < to && p - key < a.extent;
  float4 q[2];
  for (uint j = 0; j < 2; ++j)
    q[j] = float4(*(const device bfloat4*)(qn + (ulong(t) * a.heads + g * 6 + h0 + j) * D + 4 * lane));
  float s[2], bz[2] = {0.0f, 0.0f};
  {
    // the group's 32 key rows, clamped into the visible range (a clamped row is read and given weight zero)
    bfloat4 kr[32];
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 32; ++i)
      kr[i] = *(const device bfloat4*)(keys + (ulong(min(max(k0 + i, from), to - 1) - wrap) * a.kv + g) * D + 4 * lane);
    if (reads_bias) bias.two(t, a.heads, g * 6 + h0, a.extent, p - key, bz);
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < 2; ++j) {
      float pd[32];
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < 32; ++i) {
        const float4 k = float4(kr[i]);
        float x = q[j].x * k.x;
        x = fma(q[j].y, k.y, x);
        x = fma(q[j].z, k.z, x);
        pd[i] = fma(q[j].w, k.w, x);
      }
      float x = ik_as_scatter(pd, lane);
      if (key >= from && key < to) {
        x = x * a.scale;
        if (p - key < a.extent) x += bz[j];
      } else {
        x = -INFINITY;
      }
      s[j] = x;
    }
  }
  _Pragma("clang loop unroll(full)")
  for (uint j = 0; j < 2; ++j) {
    const float x = simd_max(s[j]);
    if (lane == 0) tmax[kg][h0 + j] = x;
  }
  bfloat4 vr[32];
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < 32; ++i)
    vr[i] = *(const device bfloat4*)(values + (ulong(min(max(k0 + i, from), to - 1) - wrap) * a.kv + g) * D + 4 * lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float w[2];
  float4 acc[2];
  _Pragma("clang loop unroll(full)")
  for (uint j = 0; j < 2; ++j) {
    float m = tmax[0][h0 + j];
    for (uint k = 1; k < 8; ++k) m = max(m, tmax[k][h0 + j]);
    w[j] = exp(s[j] - m);
    const float ls = simd_sum(w[j]);
    if (lane == 0) tsum[kg][h0 + j] = ls;
    acc[j] = 0.0f;
  }
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < 32; ++i) {
    const float4 v = float4(vr[i]);
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < 2; ++j) acc[j] = fma(float4(simd_broadcast(w[j], ushort(i))), v, acc[j]);
  }
  _Pragma("clang loop unroll(full)")
  for (uint j = 0; j < 2; ++j) tacc[kg][h0 + j][lane] = acc[j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 192) {
    const uint hh = tid / 32, c = tid % 32;
    float4 o = tacc[0][hh][c];
    for (uint k = 1; k < 8; ++k) o += tacc[k][hh][c];
    device float* dst = slots + hh * head_step + 2 + 4 * c;
    *(device float2*)dst = o.xy;
    *(device float2*)(dst + 2) = o.zw;
  }
  if (tid < 6) {
    float m = tmax[0][tid], l = tsum[0][tid];
    for (uint k = 1; k < 8; ++k) { m = max(m, tmax[k][tid]); l += tsum[k][tid]; }
    slots[tid * head_step] = m;
    slots[tid * head_step + 1] = l;
  }
}

// One (kv head, row, block) a threadgroup of 24 simdgroups (768 threads; dispatch (kv heads, rows, a.blocks)
// threadgroups): simdgroup s takes 32 consecutive keys of the block (s % 8) for two of the kv group's 6 q heads (2 (s / 8)
// and the next), four dims a lane. a.queries is the first block's index and a.blocks the blocks per row: block
// a.queries + tg.z covers keys [256 b, 256 b + 256) clipped to the row's visible range [lo, p]. A simdgroup scores its
// keys one lane a key by a transposed sum over the dims, plus the relative bias at the key's distance when that is below
// a.extent (formed here from reln and proj, as ik_relbias does for the prompt path), the threadgroup takes the block's
// maximum, each lane weights its key by exp(score - maximum), and the weighted values and weight sums of the 8 key
// groups fold in group order into the partial (m, l, acc). An empty block writes m = -inf, l = 0.
[[kernel, max_total_threads_per_threadgroup(768)]] void ik_attn_part(constant ik_args& a [[buffer(0)]],
    const device bfloat* qn [[buffer(1)]], const device bfloat* keys [[buffer(2)]], const device bfloat* values [[buffer(3)]],
    const device bfloat* reln [[buffer(4)]], const device bfloat* proj [[buffer(5)]], device float* part [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tmax[8][6], tsum[8][6];
  threadgroup float4 tacc[8][6][32]; // [key group][head][lane]: the lane's four dims
  ik_attn_part_run(a, qn, keys, values, IkRelFold{reln, proj}, part, tmax, tsum, tacc, tg, tid, sg, lane);
}

// One (head, row) a simdgroup (dispatch (heads, rows) threadgroups of 32 threads), four dims a lane. Starts from the sink
// as a zero-valued key (m = sink, l = 1, acc = 0) and folds the row's blocks in ascending index, skipping empty ones;
// then o = bf16(acc / l) and the output gate.
[[kernel]] void ik_attn_merge(constant ik_args& a [[buffer(0)]], const device float* part [[buffer(1)]],
    const device bfloat* sink [[buffer(2)]], const device bfloat* gate [[buffer(3)]], device bfloat* att [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint h = tg.x, t = tg.y;
  float m = float(sink[ulong(t) * a.heads + h]), l = 1.0f;
  float4 acc = 0.0f;
  const device float* slot = part + (ulong(t) * a.heads + h) * a.blocks * IK_AS_SLOT;
  for (uint b = 0; b < a.blocks; ++b, slot += IK_AS_SLOT) {
    const float mb = slot[0], lb = slot[1];
    if (lb == 0.0f) continue;
    const float4 ab = float4(*(const device float2*)(slot + 2 + 4 * lane), *(const device float2*)(slot + 4 + 4 * lane));
    const float mn = max(m, mb), f = exp(m - mn), fb = exp(mb - mn);
    l = fma(l, f, lb * fb);
    acc = fma(acc, float4(f), ab * fb);
    m = mn;
  }
  const ulong at = (ulong(t) * a.heads + h) * a.dim + 4 * lane;
  const float4 o = float4(bfloat4(acc / l));
  const float4 gt = float4(*(const device bfloat4*)(gate + at));
  *(device bfloat4*)(att + at) = bfloat4(o / (1.0f + exp(-gt)));
}
