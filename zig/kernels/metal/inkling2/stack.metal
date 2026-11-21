// Inkling 2 decode kernels: segmented matvec for a layer's attention inputs.
//
// ik_mvs_8 / ik_mvs_16: one dispatch runs several bf16 matvecs Y = bf16(X W^T) that differ in weight, input and
// output. Output column j of the stacked matrix (segments in ascending `first`, total = sum of n) belongs to the
// simdgroup `tg.x * 8 + sg`; it finds its segment by a linear scan and runs the per-row loop of
// core/bf16_matvec.metal's tf_bf16_matvec_t<ROWS> on that segment's x and w row. Each output element therefore has
// exactly ik_matvec_ROWS's arithmetic: lane l takes i = 8*l + 256*t, acc += dot(w0,x0) + dot(w1,x1), then simd_sum
// and one bf16 rounding. Nothing in a row's arithmetic depends on the other rows, the segments or the grid.
//
// x, w and y are GPU addresses; the host declares those buffers on the encoder (useResource) for each dispatch.

// x: input rows (row stride k), w: weight [n][k] bf16, y: output rows (row stride ldy), first: the segment's first
// column in the stacked matrix.
struct ik_seg { ulong x, w, y; uint n, k, ldy, first; };
struct ik_mvs_args { uint rows, nseg, total; };

// The per-row loop of tf_bf16_matvec_t (core/bf16_matvec.metal), copied verbatim with the row pointers passed in:
// X rows start at `X`, `w` is the weight row, `Y + r * ldy` receives the output of row r.
template <uint ROWS>
inline void ik_mvs_row(const device bfloat* X, const device bfloat* w, uint k, uint rows, uint ldy, device bfloat* Y,
    uint lane) {
  float acc[ROWS];
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;
  for (uint i = 8 * lane; i < k; i += 256) {
    const float4 w0 = float4(*(const device bfloat4*)(w + i)), w1 = float4(*(const device bfloat4*)(w + i + 4));
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < ROWS; ++r) {
      const device bfloat* x = X + (size_t)r * k + i;
      acc[r] += dot(w0, float4(*(const device bfloat4*)x)) + dot(w1, float4(*(const device bfloat4*)(x + 4)));
    }
  }
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < ROWS; ++r) {
    const float s = simd_sum(acc[r]);
    if (lane == 0 && r < rows) Y[(size_t)r * ldy] = bfloat(s);
  }
}

template <uint ROWS>
inline void ik_mvs_t(const device ik_seg* segs, constant ik_mvs_args& a, uint3 tg, uint sg, uint lane) {
  const uint j = tg.x * 8 + sg;
  if (j >= a.total) return;
  uint s = 0;
  while (s + 1 < a.nseg && j >= segs[s + 1].first) ++s;
  const device ik_seg& g = segs[s];
  const uint c = j - g.first;
  ik_mvs_row<ROWS>((const device bfloat*)g.x, (const device bfloat*)g.w + (size_t)c * g.k, g.k, a.rows, g.ldy,
      (device bfloat*)g.y + c, lane);
}

#define IK_MVS(NAME, ROWS)                                                                                         \
  [[kernel]] void NAME(const device ik_seg* segs [[buffer(0)]], constant ik_mvs_args& a [[buffer(1)]],             \
      uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]]) {                                                                   \
    ik_mvs_t<ROWS>(segs, a, tg, sg, lane);                                                                         \
  }
IK_MVS(ik_mvs_8, 8)
IK_MVS(ik_mvs_16, 16)
