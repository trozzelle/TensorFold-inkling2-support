// Tensor-unit forget/gain GEMM for Inkling 2 MessageGate controllers: computes df = z @ forget_w^T and
// dg = z @ gain_w^T in one pass so each z tile is read once. Rows stay independent and out-of-bounds rows are zeroed.
#include "../nax.h"
using namespace metal;

struct ik_ctrl_nax_args {
  uint rows;
  uint width;
  uint rank;
};

// Grid (width / 128, (rows + 31) / 32, 1) x 128 threads (4 simdgroups, each computing 32 cols x up to 32 rows).
// Requires width % 128 == 0 and rank % 16 == 0.
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void ik_msggate_fg_nax(
    constant ik_ctrl_nax_args& a [[buffer(0)]], const device bfloat* z [[buffer(1)]],
    const device bfloat* forget_w [[buffer(2)]], const device bfloat* gain_w [[buffer(3)]],
    device float* df [[buffer(4)]], device float* dg [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r0 = tg.y * 32u;
  if (r0 >= a.rows) return;
  const uint live = a.rows - r0;
  const uint n0 = tg.x * 128u + sg * 32u;
  const short2 home = tfp::frag_home(ushort(lane));
  const uint hy = uint(home.y);
  const bool l00 = hy < live;
  const bool l01 = hy + 8u < live;
  const bool l10 = hy + 16u < live;
  const bool l11 = hy + 24u < live;

  const device bfloat* z00 = l00 ? z + (size_t)(r0 + hy) * a.rank : z;
  const device bfloat* z01 = l01 ? z + (size_t)(r0 + hy + 8u) * a.rank : z;
  const device bfloat* z10 = l10 ? z + (size_t)(r0 + hy + 16u) * a.rank : z;
  const device bfloat* z11 = l11 ? z + (size_t)(r0 + hy + 24u) * a.rank : z;
  const device bfloat* fw = forget_w + (size_t)n0 * a.rank;
  const device bfloat* gw = gain_w + (size_t)n0 * a.rank;

  tfp::frag<float> f00 = tfp::frag<float>(0.0f), f01 = tfp::frag<float>(0.0f);
  tfp::frag<float> g00 = tfp::frag<float>(0.0f), g01 = tfp::frag<float>(0.0f);
  tfp::frag<float> f10 = tfp::frag<float>(0.0f), f11 = tfp::frag<float>(0.0f);
  tfp::frag<float> g10 = tfp::frag<float>(0.0f), g11 = tfp::frag<float>(0.0f);

  for (uint tk = 0u; tk < a.rank; tk += 16u) {
    tfp::frag<bfloat> a0, a1, b0, b1;
    tfp::frag_get_rows(a0, z00, z01, int(tk), home, l00, l01);
    if (live > 16u) tfp::frag_get_rows(a1, z10, z11, int(tk), home, l10, l11);

    tfp::frag_get_t(b0, fw, int(a.rank), 0, int(tk), home);
    tfp::frag_get_t(b1, fw, int(a.rank), 16, int(tk), home);
    tfp::mma_16x32<false, true>(f00, f01, a0, b0, b1);
    if (live > 16u) tfp::mma_16x32<false, true>(f10, f11, a1, b0, b1);

    tfp::frag_get_t(b0, gw, int(a.rank), 0, int(tk), home);
    tfp::frag_get_t(b1, gw, int(a.rank), 16, int(tk), home);
    tfp::mma_16x32<false, true>(g00, g01, a0, b0, b1);
    if (live > 16u) tfp::mma_16x32<false, true>(g10, g11, a1, b0, b1);
  }

  device float* df00 = l00 ? df + (size_t)(r0 + hy) * a.width : df;
  device float* df01 = l01 ? df + (size_t)(r0 + hy + 8u) * a.width : df;
  device float* dg00 = l00 ? dg + (size_t)(r0 + hy) * a.width : dg;
  device float* dg01 = l01 ? dg + (size_t)(r0 + hy + 8u) * a.width : dg;
  tfp::frag_put_rows(f00, df00, df01, int(n0), home, l00, l01);
  tfp::frag_put_rows(f01, df00, df01, int(n0 + 16u), home, l00, l01);
  tfp::frag_put_rows(g00, dg00, dg01, int(n0), home, l00, l01);
  tfp::frag_put_rows(g01, dg00, dg01, int(n0 + 16u), home, l00, l01);
  if (live > 16u) {
    device float* df10 = l10 ? df + (size_t)(r0 + hy + 16u) * a.width : df;
    device float* df11 = l11 ? df + (size_t)(r0 + hy + 24u) * a.width : df;
    device float* dg10 = l10 ? dg + (size_t)(r0 + hy + 16u) * a.width : dg;
    device float* dg11 = l11 ? dg + (size_t)(r0 + hy + 24u) * a.width : dg;
    tfp::frag_put_rows(f10, df10, df11, int(n0), home, l10, l11);
    tfp::frag_put_rows(f11, df10, df11, int(n0 + 16u), home, l10, l11);
    tfp::frag_put_rows(g10, dg10, dg11, int(n0), home, l10, l11);
    tfp::frag_put_rows(g11, dg10, dg11, int(n0 + 16u), home, l10, l11);
  }
}
