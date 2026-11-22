// Diagnostic probe kernels appended after inkling2_glue for tf-inkling-bench:
// dependent dispatch latency, read-bandwidth ceiling, expert rotation/decode ablations,
// and a shared-expert tile loop prototype.

// One dependent step in a serial chain: every threadgroup reads buf[0] and writes buf[tg.x],
// creating a true read-after-write dependency across chained dispatches.
[[kernel]] void ik_probe_step(device float* buf [[buffer(0)]],
    uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
  if (lid == 0u) {
    const float v = buf[0];
    buf[tg] = v + 1.0f;
  }
}

// Background 1 MiB read for the second-queue contention probe.
[[kernel]] void ik_probe_bg(const device uint4* src [[buffer(0)]], device uint* dst [[buffer(1)]],
    constant uint& nvec [[buffer(2)]], uint gid [[thread_position_in_grid]],
    uint grid [[threads_per_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint4 acc = uint4(0u);
  for (uint i = gid; i < nvec; i += grid) acc ^= src[i];
  uint v = acc.x ^ acc.y ^ acc.z ^ acc.w;
  v ^= simd_shuffle_xor(v, 16u);
  v ^= simd_shuffle_xor(v, 8u);
  v ^= simd_shuffle_xor(v, 4u);
  v ^= simd_shuffle_xor(v, 2u);
  v ^= simd_shuffle_xor(v, 1u);
  if (lane == 0u && gid < 1024u) dst[gid >> 5u] = v;
}

// Read-only probe with ik_matvec_8's grid and 8-byte weight loads, XORing 16-bit words and writing one word a row.
[[kernel]] void ik_probe_read(const device bfloat* X [[buffer(0)]], const device ushort* W [[buffer(1)]],
    constant tf_bf16_mv_args& a [[buffer(2)]], device ushort* Y [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint j = tg.x * 8u + sg;
  if (j >= a.n) return;
  const device ushort* w = W + (size_t)j * a.k;
  ushort4 acc0 = ushort4(0u), acc1 = ushort4(0u);
  for (uint i = 8u * lane; i < a.k; i += 256u) {
    acc0 ^= *(const device ushort4*)(w + i);
    acc1 ^= *(const device ushort4*)(w + i + 4u);
  }
  const ushort4 m = acc0 ^ acc1;
  uint v = uint(m.x ^ m.y ^ m.z ^ m.w);
  v ^= simd_shuffle_xor(v, 16u);
  v ^= simd_shuffle_xor(v, 8u);
  v ^= simd_shuffle_xor(v, 4u);
  v ^= simd_shuffle_xor(v, 2u);
  v ^= simd_shuffle_xor(v, 1u);
  if (lane == 0u) Y[j] = ushort(v);
}

// Expert ablation: pair with input rotation replaced by a plain f16 copy (timing only).
[[kernel]] void ik_xd_pair_norot(constant ik_args& a [[buffer(0)]], const device bfloat* x [[buffer(1)]],
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
    prepared[b * 128u + lane] = half(x[xr + lane]);
    prepared[b * 128u + lane + 32u] = half(x[xr + lane + 32u]);
    prepared[b * 128u + lane + 64u] = half(x[xr + lane + 64u]);
    prepared[b * 128u + lane + 96u] = half(x[xr + lane + 96u]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc[2][8] = {};
  ik_xd_gemv(d, split * 144u, ot, prepared, sg, lane, acc);
  ik_xd_reduce(acc, partial, planes + (size_t)(slot * 2u + split) * IK_XD_H, ot, sg, lane, lid);
}

// Expert ablation: down with input rotation replaced by a plain f16 copy (timing only).
[[kernel]] void ik_xd_down_norot(constant ik_args& a [[buffer(0)]], const device bfloat* act [[buffer(1)]],
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
    prepared[base + lane] = half(act[ab + lane]);
    prepared[base + lane + 32u] = half(act[ab + lane + 32u]);
    prepared[base + lane + 64u] = half(act[ab + lane + 64u]);
    prepared[base + lane + 96u] = half(act[ab + lane + 96u]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc[2][8] = {};
  ik_xd_gemv(d, 0u, ot, prepared, sg, lane, acc);
  ik_xd_reduce(acc, partial, sums + (size_t)slot * IK_XD_H, ot, sg, lane, lid);
}

// Expert ablation: same trellis loads as ik_xd_tiles, reading the fetched 64-bit words directly as f16 (no bit slicing or mul1).
template <uint N>
static inline void ik_xd_tiles_nodecode(const device uint* wp, const threadgroup half* pp, uint lane, thread float (&acc)[2][8]) {
  constexpr uint W = N / 2u;
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
      for (uint o = 0u; o < 2u; o++) {
        const float4 w = float4(as_type<half4>(merged[u][o]));
        acc[o][0] = fma(in0, w.x, acc[o][0]);
        acc[o][1] = fma(in1, w.y, acc[o][1]);
        acc[o][2] = fma(in2, w.z, acc[o][2]);
        acc[o][3] = fma(in3, w.w, acc[o][3]);
        acc[o][4] = fma(in0, w.x, acc[o][4]);
        acc[o][5] = fma(in1, w.y, acc[o][5]);
        acc[o][6] = fma(in2, w.z, acc[o][6]);
        acc[o][7] = fma(in3, w.w, acc[o][7]);
      }
    }
    wp += 2u * SGS * IK_XD_OT * W;
    pp += 2u * SGS * TILE;
  }
}

static inline void ik_xd_gemv_nodecode(const device ik_xd_expert& d, uint tk0, uint ot, const threadgroup half* prepared, uint sg, uint lane, thread float (&acc)[2][8]) {
  const device uint* trellis = (const device uint*)d.trellis;
  switch (d.halves) {
    case 4u: ik_xd_tiles_nodecode<32u>(trellis + ((size_t)(tk0 + sg) * IK_XD_OT + ot) * 16u, prepared + sg * 16u, lane, acc); break;
    case 6u: ik_xd_tiles_nodecode<48u>(trellis + ((size_t)(tk0 + sg) * IK_XD_OT + ot) * 24u, prepared + sg * 16u, lane, acc); break;
    default: break;
  }
}

[[kernel]] void ik_xd_pair_nodecode(constant ik_args& a [[buffer(0)]], const device bfloat* x [[buffer(1)]],
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
  ik_xd_gemv_nodecode(d, split * 144u, ot, prepared, sg, lane, acc);
  ik_xd_reduce(acc, partial, planes + (size_t)(slot * 2u + split) * IK_XD_H, ot, sg, lane, lid);
}

[[kernel]] void ik_xd_down_nodecode(constant ik_args& a [[buffer(0)]], const device bfloat* act [[buffer(1)]],
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
  ik_xd_gemv_nodecode(d, 0u, ot, prepared, sg, lane, acc);
  ik_xd_reduce(acc, partial, sums + (size_t)slot * IK_XD_H, ot, sg, lane, lid);
}

// Shared-expert pair prototype: one expert and M <= P slots in one threadgroup,
// decoding each tile step's weights once and keeping per-slot accumulators in ik_xd_pair's exact order.
struct ik_probe_entry {
  uint eid;
  uint n;
  uint slot[4];
};

template <uint N, uint P>
static inline void ik_probe_tiles_shared(const device uint* wp, const threadgroup half (&prepared)[P][IK_XD_I],
    uint m, uint sg, uint lane, thread float (&acc)[P][2][8]) {
  constexpr uint W = N / 2u;
  constexpr uint SGS = 4u;
  constexpr uint TILE = 16u;
  const uint prow = (lane & 3u) * 2u;
  const uint end = W * lane + W;
  uint pbase = sg * TILE;
  for (uint step = 0u; step < 18u; step++) {
    ulong merged[2][2];
    for (uint u = 0u; u < 2u; u++) {
      for (uint o = 0u; o < 2u; o++) merged[u][o] = ik_xd_window(wp + u * SGS * IK_XD_OT * W + o * W, end, W);
    }
    for (uint u = 0u; u < 2u; u++) {
      const uint po = pbase + u * SGS * TILE + prow;
      float ins[P][4];
      _Pragma("clang loop unroll(full)")
      for (uint r = 0u; r < P; r++) {
        if (r < m) {
          ins[r][0] = float(prepared[r][po]);
          ins[r][1] = float(prepared[r][po + 1u]);
          ins[r][2] = float(prepared[r][po + 8u]);
          ins[r][3] = float(prepared[r][po + 9u]);
        }
      }
      for (uint o = 0u; o < 2u; o++) {
        for (uint p = 0u; p < 4u; p++) {
          const uint2 cw = uint2(uint(merged[u][o] >> ik_xd_sh(N, p * 2u)), uint(merged[u][o] >> ik_xd_sh(N, p * 2u + 1u))) & uint2(0xffffu);
          const float2 w = ik_xd_decode2(cw);
          const uint i0 = (p * 2u) & 3u;
          const uint i1 = (p * 2u + 1u) & 3u;
          _Pragma("clang loop unroll(full)")
          for (uint r = 0u; r < P; r++) {
            if (r < m) {
              acc[r][o][p * 2u] = fma(ins[r][i0], w.x, acc[r][o][p * 2u]);
              acc[r][o][p * 2u + 1u] = fma(ins[r][i1], w.y, acc[r][o][p * 2u + 1u]);
            }
          }
        }
      }
    }
    wp += 2u * SGS * IK_XD_OT * W;
    pbase += 2u * SGS * TILE;
  }
}

template <uint P>
static inline void ik_probe_pair_body(const device bfloat* x, const device ik_probe_entry* entries,
    const device ik_xd_expert* experts, device float* planes,
    threadgroup half (&prepared)[P][IK_XD_I], threadgroup float* partial,
    uint3 tg, uint sg, uint lane, uint lid) {
  const uint ot = tg.x * 2u, ei = tg.y, split = tg.z;
  const ik_probe_entry e = entries[ei];
  const device ik_xd_expert& d = experts[e.eid];
  for (uint r = 0u; r < e.n; r++) {
    const uint xrow = e.slot[r] / IK_XD_TOPK;
    for (uint b = sg; b < IK_XD_I / 128u; b += 4u) {
      const uint pbase = split * IK_XD_I + b * 128u;
      const size_t xr = (size_t)xrow * IK_XD_H + pbase;
      const float x0 = float(x[xr + lane]);
      const float x1 = float(x[xr + lane + 32u]);
      const float x2 = float(x[xr + lane + 64u]);
      const float x3 = float(x[xr + lane + 96u]);
      const float4 s = ik_xd_h128(float4(x0 * float(d.suh[pbase + lane]), x1 * float(d.suh[pbase + lane + 32u]),
                                         x2 * float(d.suh[pbase + lane + 64u]), x3 * float(d.suh[pbase + lane + 96u])), lane);
      prepared[r][b * 128u + lane] = half(s.x * IK_XD_PSC);
      prepared[r][b * 128u + lane + 32u] = half(s.y * IK_XD_PSC);
      prepared[r][b * 128u + lane + 64u] = half(s.z * IK_XD_PSC);
      prepared[r][b * 128u + lane + 96u] = half(s.w * IK_XD_PSC);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc[P][2][8] = {};
  const device uint* trellis = (const device uint*)d.trellis;
  switch (d.halves) {
    case 4u: ik_probe_tiles_shared<32u, P>(trellis + ((size_t)(split * 144u + sg) * IK_XD_OT + ot) * 16u, prepared, e.n, sg, lane, acc); break;
    case 6u: ik_probe_tiles_shared<48u, P>(trellis + ((size_t)(split * 144u + sg) * IK_XD_OT + ot) * 24u, prepared, e.n, sg, lane, acc); break;
    default: break;
  }
  for (uint r = 0u; r < e.n; r++) {
    ik_xd_reduce(acc[r], partial, planes + (size_t)(e.slot[r] * 2u + split) * IK_XD_H, ot, sg, lane, lid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define IK_PROBE_PAIR(P) \
  [[kernel]] void ik_probe_pair_##P(const device bfloat* x [[buffer(0)]], \
      const device ik_probe_entry* entries [[buffer(1)]], const device ik_xd_expert* experts [[buffer(2)]], \
      device float* planes [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]], \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
      uint lid [[thread_index_in_threadgroup]]) { \
    threadgroup half prepared[P][IK_XD_I]; \
    threadgroup float partial[4 * 16 * 2]; \
    ik_probe_pair_body<P>(x, entries, experts, planes, prepared, partial, tg, sg, lane, lid); \
  }

IK_PROBE_PAIR(2)
IK_PROBE_PAIR(3)
IK_PROBE_PAIR(4)
