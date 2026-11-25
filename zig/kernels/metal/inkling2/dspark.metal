// Inkling 2's DSpark drafter kernels at oMLX's rounding points (omlx/patches/inkling2/dspark.py on mlx 0.32): the tap mix, the
// context and block preparation (norm and rotary embedding), block attention, the elementwise kernels, the Markov head and the
// fold's pick. Its own library, built with precise math functions. The dense projections are not here: they are the target's
// bf16 matvec kernels.
//
// Shapes: hidden width a multiple of 256, q heads = gqa * kv heads (gqa in {3, 6}, kv = 8, head dim 128), block of 8
// rows, Markov rank 256. Layouts, bf16 unless stated: taps [rows][IKD_TAPS][width]; mixed rows [mix][rows][width]
// (draft layers' context rows, then the Markov feature); ring K and V [slot][kv][128] (a.ring + IKD_PAD slots); block q
// [8][heads][128]; block K and V [8][kv][128]; attention partials f32 [row][head][block][130]. Checkpoint tensors are
// 8-byte aligned only: kernels read them in bfloat4 pieces or less. Argument structs are u32 and f32 fields in order.
//
// The drafter's flow over these kernels:
//   absorb: ikd_mix; per layer the k_ctx and v_ctx matvecs, ikd_ctx_prep.
//   hold:   per layer norm, q, k and v matvecs, ikd_block_prep, ikd_attn_part, ikd_attn_merge, o matvec, ikd_add, norm, gate and
//           up matvecs, ikd_swiglu, down matvec, ikd_add; then norm, unembed matvec; per step ikd_rnn_proj, ikd_gru,
//           ikd_fold_pick, ikd_pick.
//   load:   ikd_fold_t turns the folded matvec output [257][vocab] into fold_w [vocab][256] and fold_b [vocab].
#include <metal_stdlib>
using namespace metal;

constant uint IKD_D = 128;     // head dimension
constant uint IKD_RANK = 256;  // Markov rank
constant uint IKD_KEYS = 256;  // keys in one absolute attention block: its partial is one (row, head, block) record
constant uint IKD_SLOT = 130;  // floats in one (row, head, block) attention partial: m, l, acc[128]
constant uint IKD_ROWS = 8;    // the block's rows: one 8-row tile of the matrix unit
constant uint IKD_PAD = 7;     // a ring holds a.ring + IKD_PAD slots: a row of a slot below IKD_PAD is written again past the last

// A threadgroup-wide sum in a fixed order: simd_sum, then the simdgroups in index order; every thread gets it.
inline float ikd_sum(float v, threadgroup float* buf, uint sg, uint lane, uint groups) {
  v = simd_sum(v);
  if (lane == 0) buf[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0.0f;
  for (uint s = 0; s < groups; ++s) t += buf[s];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

// mx.sigmoid on f32: 1 / (1 + exp(|x|)) for x < 0, else 1 minus it (mlx's Sigmoid, precise exp).
inline float ikd_sigmoid(float x) {
  const float y = 1.0f / (1.0f + precise::exp(abs(x)));
  return x < 0.0f ? y : 1.0f - y;
}

// ---- tap mix -----------------------------------------------------------------------------------------------------------

#ifndef IKD_TAPS
#define IKD_TAPS 8
#endif

struct ikd_mix_args { uint rows, width, tap_stride, mix_stride; float eps; };

// out[m][row] = bf16(w_m * (x * rsqrt(mean(x^2) + eps))) with x = sum over IKD_TAPS taps j in index order of
// alpha[m][j] * tap[row][j], in f32 throughout (f32 norm rounds once, at the end). One threadgroup per (row, mix m):
// draft layers' context rows, then the Markov feature. taps: row stride a.tap_stride elements, tap j at j * a.width;
// alpha [mixes][IKD_TAPS]; norms [mixes][width] contiguous norm weights; out: mix m at m * a.mix_stride elements, rows
// a.width apart. Dispatch (rows, mixes) groups of nt threads (multiple of 32), 16 * nt >= width; width multiple of 4.
[[kernel, max_total_threads_per_threadgroup(1024)]] void ikd_mix(constant ikd_mix_args& a [[buffer(0)]],
    const device bfloat* taps [[buffer(1)]], const device bfloat* alpha [[buffer(2)]], const device bfloat* norms [[buffer(3)]],
    device bfloat* out [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint2 nts [[threads_per_threadgroup]]) {
  threadgroup float buf[32];
  const uint row = tg.x, m = tg.y, nt = nts.x;
  const device bfloat* w = norms + ulong(m) * a.width;
  float al[IKD_TAPS];
  _Pragma("clang loop unroll(full)")
  for (uint j = 0; j < IKD_TAPS; ++j) al[j] = float(alpha[m * IKD_TAPS + j]);
  const device bfloat* x = taps + ulong(row) * a.tap_stride;
  float4 v[4];
  float ss = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint c = 0; c < 4; ++c) {
    const uint d = 4 * (tid + c * nt);
    float4 s = 0.0f;
    if (d < a.width) {
      float4 t[IKD_TAPS];
      _Pragma("clang loop unroll(full)")
      for (uint j = 0; j < IKD_TAPS; ++j) t[j] = float4(*(const device bfloat4*)(x + ulong(j) * a.width + d));
      _Pragma("clang loop unroll(full)")
      for (uint j = 0; j < IKD_TAPS; ++j) s = fma(float4(al[j]), t[j], s);
    }
    v[c] = s;
    ss += s.x * s.x + s.y * s.y + s.z * s.z + s.w * s.w;
  }
  const float inv = precise::rsqrt(ikd_sum(ss, buf, sg, lane, nt / 32) / float(a.width) + a.eps);
  device bfloat* y = out + ulong(m) * a.mix_stride + ulong(row) * a.width;
  _Pragma("clang loop unroll(full)")
  for (uint c = 0; c < 4; ++c) {
    const uint d = 4 * (tid + c * nt);
    if (d >= a.width) continue;
    *(device bfloat4*)(y + d) = bfloat4(float4(float(w[d]), float(w[d + 1]), float(w[d + 2]), float(w[d + 3])) * (v[c] * inv));
  }
}

// ---- per-head norm and rotary embedding --------------------------------------------------------------------------------

struct ikd_prep_args { uint rows, heads, kv, pos, ring; float eps, base_log2; };

// One head's 128 dims, one simdgroup: nn.RMSNorm of the bf16 row (w * bf16(x * inv): the normalized value rounds to bf16 before the
// weight multiplies it, and the product rounds again), then mlx's fast RoPE at `pos` over all 128 dims, pairs (i, i + 64): the
// angle pos * 2^(-(i / 64) * base_log2) in f32, fast cos and sin, f32 products, one rounding to bf16. Written to y, and to y2 too
// when `mirror`.
inline void ikd_head(const device bfloat* x, const device bfloat* w, device bfloat* y, device bfloat* y2, bool mirror, uint pos,
    float eps, float base_log2, uint lane) {
  float v[4], ss = 0.0f;
  for (uint j = 0; j < 4; ++j) {
    v[j] = float(x[lane + 32 * j]);
    ss = fma(v[j], v[j], ss);
  }
  const float inv = precise::rsqrt(simd_sum(ss) / float(IKD_D) + eps);
  float n[4];
  for (uint j = 0; j < 4; ++j) n[j] = float(bfloat(float(w[lane + 32 * j]) * float(bfloat(v[j] * inv))));
  for (uint p = 0; p < 2; ++p) {
    const uint i = lane + 32 * p;
    const float freq = exp2(-(float(i) / 64.0f) * base_log2);
    const float theta = float(pos) * freq;
    const float c = fast::cos(theta), s = fast::sin(theta);
    const float x1 = n[p], x2 = n[p + 2];
    y[i] = bfloat(x1 * c - x2 * s);
    y[i + 64] = bfloat(x1 * s + x2 * c);
    if (mirror) {
      y2[i] = y[i];
      y2[i + 64] = y[i + 64];
    }
  }
}

// Context rows into the layer's ring: per (row, kv head) one simdgroup, k_norm_ctx and RoPE at the row's absolute position
// a.pos + row on the k_proj_ctx output, v_proj_ctx output copied unchanged, both into slot (a.pos + row) % a.ring of rk and rv
// [slot][kv][128] (a.ring + IKD_PAD slots; a row of a slot below IKD_PAD is also written to slot a.ring + that slot, so the
// attention reads 8 slots in a row past the last without wrapping). kc, vc [rows][kv * 128]; w [128]. Dispatch (rows) groups of
// 256 threads (8 simdgroups: kv <= 8); rows <= a.ring.
[[kernel, max_total_threads_per_threadgroup(256)]] void ikd_ctx_prep(constant ikd_prep_args& a [[buffer(0)]],
    const device bfloat* kc [[buffer(1)]], const device bfloat* vc [[buffer(2)]], const device bfloat* w [[buffer(3)]],
    device bfloat* rk [[buffer(4)]], device bfloat* rv [[buffer(5)]], uint row [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (sg >= a.kv) return;
  const uint p = a.pos + row, slot = p % a.ring;
  const bool mirror = slot < IKD_PAD;
  const ulong src = (ulong(row) * a.kv + sg) * IKD_D, dst = (ulong(slot) * a.kv + sg) * IKD_D;
  const ulong dst2 = (ulong(slot + a.ring) * a.kv + sg) * IKD_D;
  ikd_head(kc + src, w, rk + dst, rk + dst2, mirror, p, a.eps, a.base_log2, lane);
  for (uint j = 0; j < 4; ++j) {
    const bfloat x = vc[src + lane + 32 * j];
    rv[dst + lane + 32 * j] = x;
    if (mirror) rv[dst2 + lane + 32 * j] = x;
  }
}

// The block's rows at positions a.pos + row: q_norm and RoPE on the q_proj output into qn [rows][heads][128], k_norm and RoPE on the
// k_proj output into kb [rows][kv][128], v_proj output copied to vb: the block's keys never enter the ring. One simdgroup per (row,
// unit), units 0 .. heads - 1 the q heads, then the kv heads. q [rows][heads * 128]; k, v [rows][kv * 128]; qw, kw [128]. Dispatch
// (ceil((heads + kv) / 8), rows) groups of 256 threads.
[[kernel, max_total_threads_per_threadgroup(256)]] void ikd_block_prep(constant ikd_prep_args& a [[buffer(0)]],
    const device bfloat* q [[buffer(1)]], const device bfloat* k [[buffer(2)]], const device bfloat* v [[buffer(3)]],
    const device bfloat* qw [[buffer(4)]], const device bfloat* kw [[buffer(5)]], device bfloat* qn [[buffer(6)]],
    device bfloat* kb [[buffer(7)]], device bfloat* vb [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg.y, unit = tg.x * 8 + sg;
  if (unit >= a.heads + a.kv) return;
  const uint p = a.pos + row;
  if (unit < a.heads) {
    const ulong at = (ulong(row) * a.heads + unit) * IKD_D;
    ikd_head(q + at, qw, qn + at, qn + at, false, p, a.eps, a.base_log2, lane);
    return;
  }
  const ulong at = (ulong(row) * a.kv + (unit - a.heads)) * IKD_D;
  ikd_head(k + at, kw, kb + at, kb + at, false, p, a.eps, a.base_log2, lane);
  for (uint j = 0; j < 4; ++j) vb[at + lane + 32 * j] = v[at + lane + 32 * j];
}

// ---- block attention ---------------------------------------------------------------------------------------------------

struct ikd_attn_args {
  uint heads, kv; // q heads, kv heads
  uint end;       // the block's first position: the ring holds the positions below it
  uint window;    // newest ring positions attended: [max(end - window, 0), end)
  uint ring;      // ring slots, the modulus of a position's slot
  uint blocks;    // blocks per (row, head), at most 32: the ring's absolute IKD_KEYS-key blocks, then the block's own keys
  float scale;
};

// The first ring position attended.
inline uint ikd_lo(constant ikd_attn_args& a) { return a.end > a.window ? a.end - a.window : 0; }

// One (kv head, block) a threadgroup of gqa = a.heads / a.kv simdgroups, one for each q head of the kv group: dispatch
// (kv heads, 1, a.blocks) groups of gqa * 32 threads (192 for gqa = 6, 96 for gqa = 3). Each simdgroup holds its head's
// 8 query rows as one matrix tile and steps through the block's keys 32 at a time on the matrix unit (scores as q . k,
// online softmax, weighted values accumulated into (m, l, acc)). Ring blocks cover absolute positions [IKD_KEYS b,
// IKD_KEYS (b + 1)) meeting [max(a.end - a.window, 0), a.end), read from slot pos % a.ring of rk, rv [slot][kv][128].
// The last block is bk, bv [8][kv][128], seen by every row without a causal mask. Out-of-range keys are masked out.
// Partials are f32 [row][head][block][130] (m, l, acc[128]); qn [8][heads][128].
[[kernel, max_total_threads_per_threadgroup(192)]] void ikd_attn_part(constant ikd_attn_args& a [[buffer(0)]],
    const device bfloat* qn [[buffer(1)]], const device bfloat* rk [[buffer(2)]], const device bfloat* rv [[buffer(3)]],
    const device bfloat* bk [[buffer(4)]], const device bfloat* bv [[buffer(5)]], device float* part [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint g = tg.x, z = tg.z, gqa = a.heads / a.kv, head = g * gqa + sg;
  const bool own = z + 1 == a.blocks;
  uint n = IKD_ROWS, slot0 = 0;
  if (!own) {
    const uint lo = ikd_lo(a), base = (lo / IKD_KEYS + z) * IKD_KEYS;
    const uint from = max(lo, base), to = min(a.end, base + IKD_KEYS);
    n = to > from ? to - from : 0;
    slot0 = from % a.ring;
  }
  // this lane's row of a tile and first column; the 4 lanes of a row are those that differ in bits 0 and 3
  const uint qid = lane / 4, fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
  device float* slots = part + ((ulong(fm) * a.heads + head) * a.blocks + z) * IKD_SLOT;
  if (n == 0) {
    for (uint i = (lane & 1) | ((lane >> 2) & 2); i < IKD_SLOT; i += 4) slots[i] = i == 0 ? -INFINITY : 0.0f;
    return;
  }
  const ulong kvd = ulong(a.kv) * IKD_D;
  const device bfloat* keys = (own ? bk : rk) + ulong(g) * IKD_D;
  const device bfloat* values = (own ? bv : rv) + ulong(g) * IKD_D;
  simdgroup_matrix<bfloat, 8, 8> qa[16];
  _Pragma("clang loop unroll(full)")
  for (uint j = 0; j < 16; ++j)
    simdgroup_load(qa[j], qn + ulong(head) * IKD_D + 8 * j, ulong(a.heads) * IKD_D, ulong2(0, 0), false);
  simdgroup_matrix<float, 8, 8> acc[16];
  _Pragma("clang loop unroll(full)")
  for (uint d = 0; d < 16; ++d) acc[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  float m = -INFINITY, l = 0.0f;
  for (uint c0 = 0; c0 < n; c0 += 32) {
    simdgroup_matrix<float, 8, 8> s[4];
    ulong at[4];
    float mx = -INFINITY;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 4; ++i) {
      s[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      uint slot = own ? c0 + 8 * i : slot0 + c0 + 8 * i;
      if (slot >= a.ring && !own) slot -= a.ring;
      at[i] = ulong(slot) * kvd;
      if (c0 + 8 * i < n) {
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < 16; ++j) {
          simdgroup_matrix<bfloat, 8, 8> kt;
          simdgroup_load(kt, keys + at[i] + 8 * j, kvd, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(s[i], qa[j], kt, s[i]);
        }
        _Pragma("clang loop unroll(full)")
        for (uint e = 0; e < 2; ++e)
          if (c0 + 8 * i + fn + e < n) mx = max(mx, s[i].thread_elements()[e]);
      }
    }
    mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
    mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
    const float mn = max(m, mx), alpha = exp((m - mn) * a.scale);
    m = mn;
    l *= alpha;
    _Pragma("clang loop unroll(full)")
    for (uint d = 0; d < 16; ++d) acc[d].thread_elements() *= alpha;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 4; ++i) {
      if (c0 + 8 * i >= n) continue;
      _Pragma("clang loop unroll(full)")
      for (uint e = 0; e < 2; ++e) {
        const float w = c0 + 8 * i + fn + e < n ? exp((s[i].thread_elements()[e] - mn) * a.scale) : 0.0f;
        s[i].thread_elements()[e] = w;
        l += w;
      }
      const bool row_live = c0 + 8 * i + fm < n;
      _Pragma("clang loop unroll(full)")
      for (uint d = 0; d < 16; ++d) {
        simdgroup_matrix<bfloat, 8, 8> vb;
        simdgroup_matrix<float, 8, 8> vf;
        simdgroup_load(vb, values + at[i] + 8 * d, kvd, ulong2(0, 0), false);
        vf.thread_elements()[0] = row_live ? float(vb.thread_elements()[0]) : 0.0f;
        vf.thread_elements()[1] = row_live ? float(vb.thread_elements()[1]) : 0.0f;
        simdgroup_multiply_accumulate(acc[d], s[i], vf, acc[d]);
      }
    }
  }
  l += simd_shuffle_xor(l, ushort(1));
  l += simd_shuffle_xor(l, ushort(8));
  if ((lane & 9) == 0) {
    slots[0] = m * a.scale;
    slots[1] = l;
  }
  _Pragma("clang loop unroll(full)")
  for (uint d = 0; d < 16; ++d)
    *(device float2*)(slots + 2 + 8 * d + fn) = float2(acc[d].thread_elements()[0], acc[d].thread_elements()[1]);
}

// One (head, row) a simdgroup (dispatch (heads, rows) threadgroups of 32 threads), four dims a lane, over at most 32 blocks.
// The row's blocks weigh by exp(m_b - m) against the largest m among them (an empty block, m = -inf and l = 0, by zero): l is
// the weighted sum of the l_b by a fixed tree, acc the weighted sum of the acc_b in ascending block order, no sink and no gate:
// o = bf16(acc / l).
[[kernel]] void ikd_attn_merge(constant ikd_attn_args& a [[buffer(0)]], const device float* part [[buffer(1)]],
    device bfloat* att [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint h = tg.x, t = tg.y;
  const device float* slot = part + (ulong(t) * a.heads + h) * a.blocks * IKD_SLOT;
  float mb = -INFINITY, lb = 0.0f;
  if (lane < a.blocks) {
    mb = slot[lane * IKD_SLOT];
    lb = slot[lane * IKD_SLOT + 1];
  }
  const float m = simd_max(mb), f = lb != 0.0f ? exp(mb - m) : 0.0f;
  const float l = simd_sum(lb * f);
  float4 acc = 0.0f;
  for (uint b = 0; b < a.blocks; ++b) {
    const float fb = simd_broadcast(f, ushort(b));
    if (fb == 0.0f) continue;
    const device float* ab = slot + b * IKD_SLOT + 2 + 4 * lane;
    acc = fma(float4(fb), float4(*(const device float2*)ab, *(const device float2*)(ab + 2)), acc);
  }
  *(device bfloat4*)(att + (ulong(t) * a.heads + h) * IKD_D + 4 * lane) = bfloat4(acc / l);
}

// ---- elementwise ---------------------------------------------------------------------------------------------------------

struct ikd_ew_args { uint rows, width; };

// nn.silu(gate) * up as mlx computes it in bf16: silu = x * sigmoid(x) is a compiled kernel whose every op rounds to bf16
// (sigmoid: e = exp(|x|), d = 1 + e, y = 1 / d, then y for x < 0 else 1 - y), then the product with `up` rounds again.
// Dispatch threads over (width, rows).
[[kernel]] void ikd_swiglu(constant ikd_ew_args& a [[buffer(0)]], const device bfloat* g [[buffer(1)]],
    const device bfloat* u [[buffer(2)]], device bfloat* y [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= a.width || gid.y >= a.rows) return;
  const ulong i = ulong(gid.y) * a.width + gid.x;
  const float x = float(g[i]);
  const bfloat e = bfloat(precise::exp(abs(x)));
  const bfloat d = bfloat(1.0f + float(e));
  const bfloat inv = bfloat(1.0f / float(d));
  const bfloat sig = x < 0.0f ? inv : bfloat(1.0f - float(inv));
  const bfloat act = bfloat(x * float(sig));
  y[i] = bfloat(float(act) * float(u[i]));
}

// y = bf16(a + b); y may be a or b. Dispatch threads over (width, rows).
[[kernel]] void ikd_add(constant ikd_ew_args& a [[buffer(0)]], const device bfloat* x [[buffer(1)]],
    const device bfloat* z [[buffer(2)]], device bfloat* y [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= a.width || gid.y >= a.rows) return;
  const ulong i = ulong(gid.y) * a.width + gid.x;
  y[i] = bfloat(float(x[i]) + float(z[i]));
}

// ---- Markov head -------------------------------------------------------------------------------------------------------

struct ikd_proj_args { uint n, at, hidden; };

// Eight products of a weight row piece and an input piece, summed as two float4 dots.
inline float ikd_dot8(const device bfloat* w, const device bfloat* x) {
  return dot(float4(*(const device bfloat4*)w), float4(*(const device bfloat4*)x)) +
      dot(float4(*(const device bfloat4*)(w + 4)), float4(*(const device bfloat4*)(x + 4)));
}

// proj[j] = bf16(sum_x W[j][x] * x[x] + bias[j]), one rounding of the f32 sum + bias, over the four-part input
// x = [state (256), w1[seq[a.at]] (256), hid (a.hidden), frnn (a.hidden)]. W [n][2 * 256 + 2 * a.hidden],
// w1 [vocab][256], bias [n]; a.hidden is a multiple of 256. Dispatch (n / 8) groups of 256 threads (8 outputs/group).
[[kernel, max_total_threads_per_threadgroup(256)]] void ikd_rnn_proj(constant ikd_proj_args& a [[buffer(0)]],
    const device bfloat* state [[buffer(1)]], const device bfloat* w1 [[buffer(2)]], const device uint* seq [[buffer(3)]],
    const device bfloat* hid [[buffer(4)]], const device bfloat* frnn [[buffer(5)]], const device bfloat* W [[buffer(6)]],
    const device bfloat* bias [[buffer(7)]], device bfloat* proj [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint j = tg.x * 8 + sg, o = 8 * lane;
  if (j >= a.n) return;
  const device bfloat* w = W + ulong(j) * (2 * IKD_RANK + 2 * ulong(a.hidden));
  float acc = ikd_dot8(w + o, state + o);
  acc += ikd_dot8(w + IKD_RANK + o, w1 + ulong(seq[a.at]) * IKD_RANK + o);
  for (uint i = 0; i < a.hidden; i += 256) acc += ikd_dot8(w + 2 * IKD_RANK + i + o, hid + i + o);
  for (uint i = 0; i < a.hidden; i += 256) acc += ikd_dot8(w + 2 * IKD_RANK + a.hidden + i + o, frnn + i + o);
  const float s = simd_sum(acc);
  if (lane == 0) proj[j] = bfloat(s + float(bias[j]));
}

// One GRU step of the Markov head, one 256-thread group, from proj [768] (the biased rnn_proj output, which used the old state):
// gate = sigmoid(proj[0:256]), kept = gate * state, blend = (1 - gate) * tanh(proj[256:512]), state' = kept + blend, latent =
// tanh(proj[512:768]); the f32 sigmoid and tanh round to bf16, and so does every product, difference and sum. state_out may be state.
[[kernel, max_total_threads_per_threadgroup(256)]] void ikd_gru(const device bfloat* proj [[buffer(0)]],
    const device bfloat* state [[buffer(1)]], device bfloat* state_out [[buffer(2)]], device bfloat* latent [[buffer(3)]],
    uint i [[thread_position_in_threadgroup]]) {
  const bfloat gate = bfloat(ikd_sigmoid(float(proj[i])));
  const bfloat kept = bfloat(float(gate) * float(state[i]));
  const bfloat keep_not = bfloat(1.0f - float(gate));
  const bfloat cand = bfloat(precise::tanh(float(proj[IKD_RANK + i])));
  const bfloat blend = bfloat(float(keep_not) * float(cand));
  state_out[i] = bfloat(float(kept) + float(blend));
  latent[i] = bfloat(precise::tanh(float(proj[2 * IKD_RANK + i])));
}

// ---- fold and pick -------------------------------------------------------------------------------------------------------

struct ikd_fold_args { uint vocab, valid, flags; float cap; };
constant uint IKD_LOGITS = 1; // flags: also write every id's logit
constant uint IKD_MASS = 2;   // flags: also write every group's mass, the sum over its ids of exp(logit - the group's best)

// One step of a transposed sum over the simdgroup: v[0..2N) hold two halves of N slots, lanes with bit N set keep the upper
// half, and each kept slot adds the partner lane's value for it.
template <uint N>
inline void ikd_halve(thread float* v, uint lane) {
  const bool hi = (lane & N) != 0;
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < N; ++i) {
    const float keep = hi ? v[i + N] : v[i], send = hi ? v[i] : v[i + N];
    v[i] = keep + simd_shuffle_xor(send, ushort(N));
  }
}

// Sums each of v[0..32) over the 32 lanes in one fixed tree (lanes paired by bits 4, 3, 2, 1, 0): lane l gets slot l's total.
inline float ikd_scatter(thread float* v, uint lane) {
  ikd_halve<16>(v, lane);
  ikd_halve<8>(v, lane);
  ikd_halve<4>(v, lane);
  ikd_halve<2>(v, lane);
  ikd_halve<1>(v, lane);
  return v[0];
}

// Each id's logit and each group of 256 ids' best. bias = bf16(bf16(latent . fold_w[v]) + fold_b[v]) (the f32 dot rounds to bf16, the
// bias add rounds again); logit = cap * tanh((base[v] + bias) / cap) in f32, -inf for v >= a.valid (the padded vocabulary); a group's
// best is its largest logit, ties to the lower id (a NaN never wins). base [vocab] is the step's row of the unembedding output,
// latent [256], fold_w [vocab][256], fold_b [vocab]; with IKD_LOGITS the logits go to `logits` as bf16. With IKD_MASS a group's mass
// goes to mass[group]: each simdgroup's sum of exp(logit - its best) over its 32 ids (a masked id adds nothing), then the eight
// simdgroups' sums, each scaled by exp(its best - the group's best), added in index order. Dispatch ceil(vocab / 256) groups of 256
// threads: a simdgroup takes 32 ids, reading each one's row across its lanes (8 columns a lane) and summing the dots by a
// transposed sum, so the lane l ends with id l's.
[[kernel, max_total_threads_per_threadgroup(256)]] void ikd_fold_pick(constant ikd_fold_args& a [[buffer(0)]],
    const device bfloat* base [[buffer(1)]], const device bfloat* latent [[buffer(2)]], const device bfloat* fw [[buffer(3)]],
    const device bfloat* fb [[buffer(4)]], device float* best [[buffer(5)]], device uint* best_id [[buffer(6)]],
    device bfloat* logits [[buffer(7)]], device float* mass [[buffer(8)]], uint group [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float vals[8];
  threadgroup uint ids[8];
  threadgroup float sums[8];
  const uint v0 = group * 256 + sg * 32;
  const float4 l0 = float4(*(const device bfloat4*)(latent + 8 * lane)), l1 = float4(*(const device bfloat4*)(latent + 8 * lane + 4));
  float pd[32];
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < 32; ++r) {
    const device bfloat* w = fw + ulong(min(v0 + r, a.vocab - 1)) * IKD_RANK + 8 * lane; // a row past the vocabulary is read, not used
    const float4 x0 = float4(*(const device bfloat4*)w), x1 = float4(*(const device bfloat4*)(w + 4));
    float s = x0.x * l0.x;
    s = fma(x0.y, l0.y, s);
    s = fma(x0.z, l0.z, s);
    s = fma(x0.w, l0.w, s);
    s = fma(x1.x, l1.x, s);
    s = fma(x1.y, l1.y, s);
    s = fma(x1.z, l1.z, s);
    pd[r] = fma(x1.w, l1.w, s);
  }
  const float dot = ikd_scatter(pd, lane);
  const uint v = v0 + lane;
  float score = -INFINITY;
  if (v < a.vocab) {
    const float bias = float(bfloat(float(bfloat(dot)) + float(fb[v])));
    const float z = float(base[v]) + bias;
    score = v < a.valid ? precise::tanh(z / a.cap) * a.cap : -INFINITY;
    if (a.flags & IKD_LOGITS) logits[v] = bfloat(score);
  }
  const float mine = score;
  uint at = v;
  for (ushort d = 16; d > 0; d >>= 1) {
    const float o = simd_shuffle_xor(score, d);
    const uint oi = simd_shuffle_xor(at, d);
    if (o > score || (o == score && oi < at)) { score = o; at = oi; }
  }
  if (a.flags & IKD_MASS) {
    const float s = simd_sum(mine == -INFINITY ? 0.0f : precise::exp(mine - score)); // every lane holds the simdgroup's best now
    if (lane == 0) sums[sg] = s;
  }
  if (lane == 0) { vals[sg] = score; ids[sg] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float b = vals[0];
    uint bi = ids[0];
    for (uint k = 1; k < 8; ++k) if (vals[k] > b || (vals[k] == b && ids[k] < bi)) { b = vals[k]; bi = ids[k]; }
    best[group] = b;
    best_id[group] = bi;
    if (a.flags & IKD_MASS) {
      float m = 0.0f;
      for (uint k = 0; k < 8; ++k) if (vals[k] != -INFINITY) m += sums[k] * precise::exp(vals[k] - b);
      mass[group] = m;
    }
  }
}

constant uint IKD_MAX_BATCH = 8;

struct ikd_fold_batch_args {
  uint vocab, valid, flags, count, base_stride;
  float cap;
  uint slot_map[IKD_MAX_BATCH];
};

// Batched fold-and-pick over 1..IKD_MAX_BATCH active streams sharing one DRAM read of fold_w [vocab][256]. All 256
// threads cooperatively stage 32 rows of fold_w at a time into threadgroup memory across 8 tiles covering the group's
// 256 vocabulary ids; simdgroup sg processes stream sg (when sg < count), reading base at slot_map[sg] * base_stride
// and latent at sg * 256, and writing best, best_id, and (with IKD_MASS) mass at sg * groups + group. Bit-identical
// to count calls of ikd_fold_pick without IKD_LOGITS. Dispatch ceil(vocab / 256) groups of 256 threads.
[[kernel, max_total_threads_per_threadgroup(256)]] void ikd_fold_pick_batch(constant ikd_fold_batch_args& a [[buffer(0)]],
    const device bfloat* base [[buffer(1)]], const device bfloat* latent [[buffer(2)]], const device bfloat* fw [[buffer(3)]],
    const device bfloat* fb [[buffer(4)]], device float* best [[buffer(5)]], device uint* best_id [[buffer(6)]],
    device float* mass [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat fw_s[32][IKD_RANK];
  const uint groups = (a.vocab + 255) / 256;
  const bool active = sg < a.count;
  const device bfloat* base_s = active ? base + ulong(a.slot_map[sg]) * a.base_stride : base;
  float4 l0 = 0.0f, l1 = 0.0f;
  if (active) {
    const device bfloat* lat = latent + ulong(sg) * IKD_RANK + 8 * lane;
    l0 = float4(*(const device bfloat4*)lat);
    l1 = float4(*(const device bfloat4*)(lat + 4));
  }
  const uint c4 = t & 63, r0 = t >> 6;
  float my_val = -INFINITY, my_sum = 0.0f;
  uint my_id = 0;
  for (uint tile = 0; tile < 8; ++tile) {
    const uint v0 = group * 256 + tile * 32;
    _Pragma("clang loop unroll(full)")
    for (uint rr = 0; rr < 32; rr += 4) {
      const uint r = rr + r0;
      const uint vr = min(v0 + r, a.vocab - 1);
      *(threadgroup bfloat4*)(&fw_s[r][c4 * 4]) = *(const device bfloat4*)(fw + ulong(vr) * IKD_RANK + c4 * 4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      float pd[32];
      _Pragma("clang loop unroll(full)")
      for (uint r = 0; r < 32; ++r) {
        const threadgroup bfloat* w = &fw_s[r][8 * lane];
        const float4 x0 = float4(*(const threadgroup bfloat4*)w), x1 = float4(*(const threadgroup bfloat4*)(w + 4));
        float s = x0.x * l0.x;
        s = fma(x0.y, l0.y, s);
        s = fma(x0.z, l0.z, s);
        s = fma(x0.w, l0.w, s);
        s = fma(x1.x, l1.x, s);
        s = fma(x1.y, l1.y, s);
        s = fma(x1.z, l1.z, s);
        pd[r] = fma(x1.w, l1.w, s);
      }
      const float dot = ikd_scatter(pd, lane);
      const uint v = v0 + lane;
      float score = -INFINITY;
      if (v < a.vocab) {
        const float bias = float(bfloat(float(bfloat(dot)) + float(fb[v])));
        const float z = float(base_s[v]) + bias;
        score = v < a.valid ? precise::tanh(z / a.cap) * a.cap : -INFINITY;
      }
      const float mine = score;
      uint at = v;
      for (ushort d = 16; d > 0; d >>= 1) {
        const float o = simd_shuffle_xor(score, d);
        const uint oi = simd_shuffle_xor(at, d);
        if (o > score || (o == score && oi < at)) { score = o; at = oi; }
      }
      float s_mass = 0.0f;
      if (a.flags & IKD_MASS) s_mass = simd_sum(mine == -INFINITY ? 0.0f : precise::exp(mine - score));
      if (lane == tile) {
        my_val = score;
        my_id = at;
        my_sum = s_mass;
      }
    }
    if (tile < 7) threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!active) return;
  float b = simd_broadcast(my_val, ushort(0));
  uint bi = simd_broadcast(my_id, ushort(0));
  _Pragma("clang loop unroll(full)")
  for (uint k = 1; k < 8; ++k) {
    const float vk = simd_broadcast(my_val, ushort(k));
    const uint ik = simd_broadcast(my_id, ushort(k));
    if (vk > b || (vk == b && ik < bi)) { b = vk; bi = ik; }
  }
  const ulong out_i = ulong(sg) * groups + group;
  if (lane == 0) {
    best[out_i] = b;
    best_id[out_i] = bi;
  }
  if (a.flags & IKD_MASS) {
    float m = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < 8; ++k) {
      const float vk = simd_broadcast(my_val, ushort(k));
      const float sk = simd_broadcast(my_sum, ushort(k));
      if (vk != -INFINITY) m += sk * precise::exp(vk - b);
    }
    if (lane == 0) mass[out_i] = m;
  }
}

struct ikd_pick_args { uint groups, step, flags; };

// The step's proposal: the best of every group's best, ties to the lower id, into seq[a.step + 1]. With IKD_MASS also its probability
// under the step's logits, prob[a.step] = 1 / sum over the vocabulary of exp(logit - the best logit), at most 1: each group's mass
// scaled by exp(its best - the largest best), the 1,024 threads' partial sums added by simd_sum and the 32 simdgroups' by one more
// (a fixed order, so every run gives the same bits). One threadgroup of 1024 threads.
[[kernel, max_total_threads_per_threadgroup(1024)]] void ikd_pick(constant ikd_pick_args& a [[buffer(0)]],
    const device float* best [[buffer(1)]], const device uint* best_id [[buffer(2)]], device uint* seq [[buffer(3)]],
    const device float* mass [[buffer(4)]], device float* prob [[buffer(5)]],
    uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float vals[32];
  threadgroup uint ids[32];
  threadgroup float top;
  threadgroup float part[32];
  float b = -INFINITY;
  uint bi = 0xffffffffu;
  for (uint i = t; i < a.groups; i += 1024) if (best[i] > b || (best[i] == b && best_id[i] < bi)) { b = best[i]; bi = best_id[i]; }
  for (ushort d = 16; d > 0; d >>= 1) {
    const float o = simd_shuffle_xor(b, d);
    const uint oi = simd_shuffle_xor(bi, d);
    if (o > b || (o == b && oi < bi)) { b = o; bi = oi; }
  }
  if (lane == 0) { vals[sg] = b; ids[sg] = bi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool sum = (a.flags & IKD_MASS) != 0;
  if (!sum && sg != 0) return;
  if (sg == 0) {
    b = vals[lane];
    bi = ids[lane];
    for (ushort d = 16; d > 0; d >>= 1) {
      const float o = simd_shuffle_xor(b, d);
      const uint oi = simd_shuffle_xor(bi, d);
      if (o > b || (o == b && oi < bi)) { b = o; bi = oi; }
    }
    if (lane == 0) {
      seq[a.step + 1] = bi;
      top = b;
    }
  }
  if (!sum) return;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float e = 0.0f;
  for (uint i = t; i < a.groups; i += 1024) if (best[i] != -INFINITY) e += mass[i] * precise::exp(best[i] - top);
  e = simd_sum(e);
  if (lane == 0) part[sg] = e;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  const float total = simd_sum(part[lane]);
  if (lane == 0) {
    const float p = 1.0f / total;
    prob[a.step] = p > 1.0f ? 1.0f : p;
  }
}

struct ikd_t_args { uint vocab; };

// The fold's matvec output y [257][vocab] (row c < 256 the dot of the unembedding with hidden_out's column c, row 256 with its bias)
// to fold_w [vocab][256] and fold_b [vocab]: a copy through a threadgroup tile of 32 ids. Dispatch ceil(vocab / 32) groups of 256
// threads.
[[kernel, max_total_threads_per_threadgroup(256)]] void ikd_fold_t(constant ikd_t_args& a [[buffer(0)]],
    const device bfloat* y [[buffer(1)]], device bfloat* fw [[buffer(2)]], device bfloat* fb [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
  threadgroup bfloat tile[IKD_RANK + 1][34];
  const uint v0 = group * 32;
  for (uint i = t; i < (IKD_RANK + 1) * 32; i += 256) {
    const uint c = i / 32, v = v0 + i % 32;
    tile[c][i % 32] = v < a.vocab ? y[ulong(c) * a.vocab + v] : bfloat(0.0f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = t; i < 32 * IKD_RANK; i += 256) {
    const uint v = v0 + i / IKD_RANK, c = i % IKD_RANK;
    if (v < a.vocab) fw[ulong(v) * IKD_RANK + c] = tile[c][i / IKD_RANK];
  }
  if (t < 32 && v0 + t < a.vocab) fb[v0 + t] = tile[IKD_RANK][t];
}
