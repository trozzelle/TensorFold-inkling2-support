// Inkling 2's prompt attention on the tensor units: a chunk's rows against the layer's cache, 32 query rows of one head a
// threadgroup, over 64-key blocks anchored at absolute positions (core/prompt_attention_nax.metal's tile structure).
// Scores are f32 mma_16x32 sums of bf16 products; the softmax is online and f32, started at the head's sink logit as a
// zero-valued key (running max = sink, l = 1); masked keys weigh exact zeros; the result is bf16(acc / l), then the output
// gate o / (1 + exp(-gate)). The relative bias is bf16(rel . proj) over 16 dims summed in index order (ik_relbias's and
// ik_attn_part's sum), added to the scaled score for keys below the extent behind the row: ik_pattn_bias builds it for a
// run of rows, ik_pattn reads it.
//
// A row's result depends on that row alone: not on the rows beside it in its tile, the call's first row, its place in the
// tile or the 16-key group the call ends in. A block no key of the row lies in leaves its state exactly as it was, and
// every sum has one order that follows absolute key positions.
#include "../nax.h"
using namespace tfp;

constant constexpr int DH = 128;       // head dim
constant constexpr int NQ = 48;        // query heads
constant constexpr int G = 6;          // query heads a key head
constant constexpr int KVH = 8;        // key heads
constant constexpr int LD = KVH * DH;  // elements in a cache row
constant constexpr int QS = NQ * DH;   // elements in a row of q, gate and the output
constant constexpr int WIN = 1024;     // a local layer's window
constant constexpr int EXT = 1024;     // the relative bias covers distances below this
constant constexpr int REL = 16;       // dims of the relative state
constant constexpr int RT = 32;        // query rows a tile
constant constexpr int NS = DH / 64;   // simdgroups a tile, each its own 64 output columns
constant constexpr int KS = 32;        // keys a simdgroup a block
constant constexpr int BKT = KS * NS;  // keys a block
constant constexpr int PP = BKT + 8;   // the probabilities' row pitch
constant constexpr float kMasked = -3.402823466e+38f;
constant constexpr float kLog2e = 1.44269504089f;
constant constexpr float kScale = 0.08838834764831845f; // 1 / sqrt(DH)

// The probabilities cross threadgroup memory as f16 at half the memory. The tensor op takes them at ten fraction bits,
// which f16 holds for values from 2^-14 up; below that f16 loses precision, and values under about 2^-25 flush to zero,
// which the layer and KL gates tolerate.
using PT = half;
using PT4 = half4;

// rows: the call's rows; p0: the first row's position in the sequence; global: 1 on a global layer, 0 on a local one.
struct IkPattnArgs {
  uint rows, p0, global;
};

// Row max (MAX) or sum over a fragment's 32 columns into r[0], r[1] (rows home.y, home.y + 8).
template <bool MAX>
inline void ik_pa_fold(thread const frag<float>& f, thread float* r) {
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const short b = 4 * h;
    float t = MAX ? max(max(f[b], f[b + 1]), max(f[b + 2], f[b + 3])) : (f[b] + f[b + 1]) + (f[b + 2] + f[b + 3]);
    const float u = simd_shuffle_xor(t, ushort(1));
    t = MAX ? max(t, u) : t + u;
    const float w = simd_shuffle_xor(t, ushort(8));
    t = MAX ? max(t, w) : t + w;
    r[h] = MAX ? max(r[h], t) : r[h] + t;
  }
}

template <bool TR>
inline frag<bfloat> ik_pa_blk(const device bfloat* p, int ld, int r, int c, short2 home) {
#ifndef TF_SIMD_FRAGS
  const device bfloat* q = p + (r + home.y) * ld + c + home.x;
  return frag<bfloat>(*(const device bfloat4*)q, *(const device bfloat4*)(q + 8 * ld));
#else
  frag<bfloat> f;
  if (TR) {
    frag_get_t(f, p, ld, r, c, home);
  } else {
    frag_get(f, p, ld, r, c, home);
  }
  return f;
#endif
}

// The block of a query tile whose rows home.y and home.y + 8 start at r0 and r1.
inline frag<bfloat> ik_pa_rows(const device bfloat* r0, const device bfloat* r1, int c, short2 home) {
#ifndef TF_SIMD_FRAGS
  return frag<bfloat>(*(const device bfloat4*)(r0 + c + home.x), *(const device bfloat4*)(r1 + c + home.x));
#else
  frag<bfloat> f;
  frag_get_rows(f, r0, r1, c, home, true, true);
  return f;
#endif
}

inline frag<PT> ik_pa_prob_frag(const threadgroup PT* pt, int r, int c, short2 home) {
#ifndef TF_SIMD_FRAGS
  const threadgroup PT* q = pt + (r + home.y) * PP + c + home.x;
  return frag<PT>(*(const threadgroup PT4*)q, *(const threadgroup PT4*)(q + 8 * PP));
#else
  frag<PT> f;
  frag_get(f, pt, PP, r, c, home);
  return f;
#endif
}

inline void ik_pa_prob_put(threadgroup PT* pt, int r, int c, short2 home, thread const frag<float>& s) {
#ifndef TF_SIMD_FRAGS
  threadgroup PT* q = pt + (r + home.y) * PP + c + home.x;
  *(threadgroup PT4*)q = PT4(s[0], s[1], s[2], s[3]);
  *(threadgroup PT4*)(q + 8 * PP) = PT4(s[4], s[5], s[6], s[7]);
#else
  TF_UNROLL
  for (short e = 0; e < 8; e++) pt[(r + home.y + (e >> 2) * 8) * PP + c + home.x + TF_COL(e)] = PT(s[e]);
#endif
}

// The relative bias a tile reads from the table ik_pattn_bias builds: bf16 by key slot, [row][head][EXT], the bias of the
// row and the key at slot key & (EXT - 1). Any source of the bias takes this shape: `tile` once, `block` for each block of
// keys before its scores are masked, `at` for each score.
struct IkPaTable {
  const device bfloat* table; // the call's first row
  const device bfloat* tr[4]; // the lane's four rows' slots, rows home.y + 8 j (clamped to the call's last row)
  bfloat4 tv[4][2];           // the block's biases: fragment i, rows home.y and home.y + 8, four keys a lane

  inline void tile(int r0, int q_rows, int qh, int first_pos, uint sg, uint lane, short2 home) {
    TF_UNROLL
    for (short j = 0; j < 4; j++) tr[j] = table + (long(r0 + min(home.y + 8 * j, q_rows - 1)) * NQ + qh) * EXT;
  }

  // kpos: the simdgroup's first key of the block; dmax: the distance of (the tile's first row, the block's first key).
  inline void block(int kpos, int dmax, short2 home) {
#ifndef TF_SIMD_FRAGS
    if (dmax - (BKT - 1) < EXT) { // some score of the block lies within the extent
      TF_UNROLL
      for (short i = 0; i < 4; i++) {
        const int slot = (kpos + 16 * (i & 1) + home.x) & (EXT - 1);
        tv[i][0] = *(const device bfloat4*)(tr[2 * (i >> 1)] + slot);
        tv[i][1] = *(const device bfloat4*)(tr[2 * (i >> 1) + 1] + slot);
      }
    }
#endif
  }

  // fragment i's element e, whose key is `col` and whose distance behind its row is `dist`
  inline float at(short i, short e, int col, int dist) const {
#ifndef TF_SIMD_FRAGS
    return dist < EXT ? float(tv[i][e >> 2][e & 3]) : 0.0f;
#else
    return dist < EXT ? float(tr[2 * (i >> 1) + (e >> 2)][col & (EXT - 1)]) : 0.0f;
#endif
  }
};

// One tile of a head's rows: the rows' scores against the keys of its blocks, the softmax, the weighted values.
template <typename Bias>
inline void ik_pa_run(constant IkPattnArgs& a, const device bfloat* Q, const device bfloat* K, const device bfloat* V,
                      const device bfloat* sinks, const device bfloat* gate, device bfloat* O, thread Bias& bias,
                      threadgroup PT* pt, threadgroup float* rowmax, uint sg, uint lane, uint3 tg) {
  const int R = int(a.rows), p0 = int(a.p0);
  const bool global = a.global != 0;
  const int qh = int(tg.y), kvh = qh / G;
  const int r0 = ((R + RT - 1) / RT - 1 - int(tg.x)) * RT; // the tiles longest first: the last rows see the most keys
  const int q_rows = min(RT, R - r0);
  const short2 home = frag_home(ushort(lane));
  const int first_pos = p0 + r0, last_pos = first_pos + q_rows - 1; // the tile's first and last rows' positions
  const int lo = global ? 0 : max(0, first_pos - WIN + 1);
  const device bfloat* kc = K + kvh * DH;
  const device bfloat* vc = V + kvh * DH + int(sg) * 64;
  // the lane's query rows, clamped to the call's last row: rows past it read a row of the call, never memory beyond it
  const device bfloat* qr[4];
  TF_UNROLL
  for (short j = 0; j < 4; j++) qr[j] = Q + long(r0 + min(16 * (j >> 1) + home.y + 8 * (j & 1), q_rows - 1)) * QS + qh * DH;
  frag<float> acc[8];
  TF_UNROLL
  for (short i = 0; i < 8; i++) acc[i] = frag<float>(0);
  // the running max starts at the row's sink logit and the sum at the sink's weight 1 (carried by the first simdgroup)
  float top[4], total[4];
  TF_UNROLL
  for (short j = 0; j < 4; j++) {
    const int local = home.y + 8 * j;
    top[j] = local < q_rows ? float(sinks[long(r0 + local) * NQ + qh]) * kLog2e : 0.0f;
    total[j] = sg == 0 ? 1.0f : 0.0f;
  }
  bias.tile(r0, q_rows, qh, first_pos, sg, lane, home);
  for (int kb = lo / BKT; kb <= last_pos / BKT; kb++) {
    const int kpos = kb * BKT + int(sg) * KS;
    const device bfloat* kblk = kc + long(kpos) * LD;
    frag<float> s[4];
    TF_UNROLL
    for (short i = 0; i < 4; i++) s[i] = frag<float>(0);
    for (int d = 0; d < DH / 16; d++) {
      const frag<bfloat> k0 = ik_pa_blk<true>(kblk, LD, 0, 16 * d, home);
      const frag<bfloat> k1 = ik_pa_blk<true>(kblk, LD, 16, 16 * d, home);
      TF_UNROLL
      for (short h = 0; h < 2; h++) {
        const frag<bfloat> qf = ik_pa_rows(qr[2 * h], qr[2 * h + 1], 16 * d, home);
        mma_16x32<false, true>(s[2 * h], s[2 * h + 1], qf, k0, k1);
      }
    }
    bias.block(kpos, first_pos - kb * BKT, home);
    TF_UNROLL
    for (short i = 0; i < 4; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int local = 16 * (i >> 1) + home.y + (e >> 2) * 8, row = first_pos + local;
        const int col = kpos + 16 * (i & 1) + home.x + TF_COL(e);
        const int dist = row - col;
        const bool out = col > row || (!global && dist >= WIN);
        s[i][e] = out ? kMasked : fma(s[i][e], kScale * kLog2e, bias.at(i, e, col, dist) * kLog2e);
      }
    }
    float top_new[4] = {top[0], top[1], top[2], top[3]};
    TF_UNROLL
    for (short i = 0; i < 4; i++) ik_pa_fold<true>(s[i], top_new + 2 * (i >> 1));
    if (home.x == 0) {
      TF_UNROLL
      for (short j = 0; j < 4; j++) rowmax[sg * RT + home.y + 8 * j] = top_new[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TF_UNROLL
    for (short j = 0; j < 4; j++) {
      for (int u = 0; u < NS; u++) top_new[j] = max(top_new[j], rowmax[u * RT + home.y + 8 * j]);
    }
    TF_UNROLL
    for (short i = 0; i < 4; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        s[i][e] = s[i][e] == kMasked ? 0.0f : fast::exp2(s[i][e] - top_new[2 * (i >> 1) + (e >> 2)]);
      }
    }
    float keep[4];
    TF_UNROLL
    for (short j = 0; j < 4; j++) {
      keep[j] = top[j] == top_new[j] ? 1.0f : fast::exp2(top[j] - top_new[j]);
      top[j] = top_new[j];
      total[j] = total[j] * keep[j];
    }
    TF_UNROLL
    for (short i = 0; i < 4; i++) ik_pa_fold<false>(s[i], total + 2 * (i >> 1));
    TF_UNROLL
    for (short i = 0; i < 8; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * keep[2 * (i >> 2) + (e >> 2)];
    }
    TF_UNROLL
    for (short i = 0; i < 4; i++) ik_pa_prob_put(pt, 16 * (i >> 1), int(sg) * KS + 16 * (i & 1), home, s[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < BKT / 16; kk++) {
      const int kp = kb * BKT + 16 * kk;
      if (kp > last_pos) break; // keys past the tile's rows: masked, and their cache rows may hold anything
      const int valid = min(16, last_pos + 1 - kp); // a value row never written would poison the tile as 0 x NaN
      const device bfloat* vblk = vc + long(kp) * LD;
      frag<bfloat> v0, v1;
      TF_UNROLL
      for (short d = 0; d < 4; d += 2) {
        if (valid == 16) {
          v0 = ik_pa_blk<false>(vblk, LD, 0, 16 * d, home);
          v1 = ik_pa_blk<false>(vblk, LD, 0, 16 * d + 16, home);
        } else {
          frag_get_in(v0, vblk, LD, 0, 16 * d, home, valid, DH);
          frag_get_in(v1, vblk, LD, 0, 16 * d + 16, home, valid, DH);
        }
        TF_UNROLL
        for (short h = 0; h < 2; h++) {
          const frag<PT> pf = ik_pa_prob_frag(pt, 16 * h, 16 * kk, home);
          mma_16x32<false, false>(acc[4 * h + d], acc[4 * h + d + 1], pf, v0, v1);
        }
      }
    }
  }
  // the rows' sums: each simdgroup's partial sums, scaled alike by the shared maxes
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (home.x == 0) {
    TF_UNROLL
    for (short j = 0; j < 4; j++) rowmax[sg * RT + home.y + 8 * j] = total[j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float l[4];
  TF_UNROLL
  for (short j = 0; j < 4; j++) {
    float t = 0.0f;
    for (int u = 0; u < NS; u++) t += rowmax[u * RT + home.y + 8 * j];
    l[j] = t;
  }
  TF_UNROLL
  for (short i = 0; i < 8; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const int local = 16 * (i >> 2) + home.y + (e >> 2) * 8;
      if (local < q_rows) {
        const long at = long(r0 + local) * QS + qh * DH + int(sg) * 64 + 16 * (i & 3) + home.x + TF_COL(e);
        const float o = float(bfloat(acc[i][e] / l[2 * (i >> 2) + (e >> 2)]));
        O[at] = bfloat(o / (1.0f + precise::exp(-float(gate[at]))));
      }
    }
  }
}

// dispatch ceil(rows / 32) x 48 threadgroups of 64 threads. q [rows][48][128]; K and V the layer's keys and values
// [capacity][8][128] (capacity a multiple of 64); sinks [rows][48]; table the rows' relative bias, ik_pattn_bias's
// [rows][48][1024]; gate and out [rows][48][128]; all bf16, row r at position p0 + r.
[[max_total_threads_per_threadgroup(NS * 32)]]
[[kernel]] void ik_pattn(constant IkPattnArgs& a [[buffer(0)]], const device bfloat* Q [[buffer(1)]],
                         const device bfloat* K [[buffer(2)]], const device bfloat* V [[buffer(3)]],
                         const device bfloat* sinks [[buffer(4)]], const device bfloat* table [[buffer(5)]],
                         const device bfloat* gate [[buffer(6)]], device bfloat* O [[buffer(7)]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup PT4 pt4[RT * PP / 4];
  threadgroup float rowmax[NS * RT];
  IkPaTable bias;
  bias.table = table;
  ik_pa_run(a, Q, K, V, sinks, gate, O, bias, (threadgroup PT*)pt4, rowmax, sg, lane, tg);
}

// The table of the rows' relative bias, bf16(reln . proj) with the 16 products summed in index order from zero as ik_relbias
// does: row t, head h, the distance d = 0 .. EXT - 1 behind the row at slot (p0 + t - d) & (EXT - 1). A thread takes four
// consecutive distances (a multiple of four) of four consecutive heads, so each projection value it converts serves four
// products. Dispatch threads (EXT / 4, 12, rows); reln [rows][48][16] and proj [16][EXT] bf16, table [rows][48][EXT] bf16.
[[kernel]] void ik_pattn_bias(constant IkPattnArgs& a [[buffer(0)]], const device bfloat* reln [[buffer(1)]],
                              const device bfloat* proj [[buffer(2)]], device bfloat* table [[buffer(3)]],
                              uint3 gid [[thread_position_in_grid]]) {
  const int m = int(gid.x), h0 = 4 * int(gid.y), t = int(gid.z);
  if (m >= EXT / 4 || h0 >= NQ || t >= int(a.rows)) return;
  const device bfloat4* r = (const device bfloat4*)(reln + (long(t) * NQ + h0) * REL);
  float4 rl[4][REL / 4]; // head hh's relative state, four dims a float4
  TF_UNROLL
  for (short hh = 0; hh < 4; hh++) {
    TF_UNROLL
    for (short c = 0; c < REL / 4; c++) rl[hh][c] = float4(r[hh * (REL / 4) + c]);
  }
  float4 s[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)}; // head hh's four distances
  TF_UNROLL
  for (short i = 0; i < REL; i++) {
    const float4 pv = float4(*(const device bfloat4*)(proj + i * EXT + 4 * m));
    TF_UNROLL
    for (short hh = 0; hh < 4; hh++) s[hh] = fma(float4(rl[hh][i >> 2][i & 3]), pv, s[hh]);
  }
  const int pos = int(a.p0) + t;
  TF_UNROLL
  for (short hh = 0; hh < 4; hh++) {
    device bfloat* row = table + (long(t) * NQ + h0 + hh) * EXT;
    TF_UNROLL
    for (short k = 0; k < 4; k++) row[(pos - (4 * m + k)) & (EXT - 1)] = bfloat(s[hh][k]);
  }
}
