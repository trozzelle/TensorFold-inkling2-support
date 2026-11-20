// Inkling 2's short convolutions at mlx-lm's rounding points: static taps convolve in the weights' bf16, dynamic taps
// (global layers' k and v) in f32; inputs before this chunk come from a 32-position ring. Follows glue.metal.
#include <metal_stdlib>
using namespace metal;

// A convolution input at position q: this chunk's row from a.pos on, the ring's before it, zero before position 0.
inline float ik_src(const device bfloat* x, const device bfloat* ring, uint C, uint c, int q, int pos0) {
  if (q < 0) return 0.0f;
  return q >= pos0 ? float(x[ulong(q - pos0) * C + c]) : float(ring[ulong(q % 32) * C + c]);
}

// ShortConvolution: bf16(sum_j w[c][j] x[t-3+j]), then bf16((that + x_t) / a.scale) with a.scale = sqrt(K + 1).
[[kernel]] void ik_conv(constant ik_args& a [[buffer(0)]], const device bfloat* x [[buffer(1)]],
    const device bfloat* ring [[buffer(2)]], const device bfloat* w [[buffer(3)]], device bfloat* y [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, t = gid.y;
  if (c >= a.width || t >= a.rows) return;
  const int p = int(a.pos + t);
  float s = 0.0f;
  for (int j = 0; j < 4; ++j) s += float(w[c * 4 + j]) * ik_src(x, ring, a.width, c, p - 3 + j, int(a.pos));
  y[ulong(t) * a.width + c] = bfloat((float(bfloat(s)) + float(x[ulong(t) * a.width + c])) / a.scale);
}

// DynamicShortConvolution: taps[lag][c] + the row's projection [heads, lags], over x[t - lag] in f32, then
// bf16((that + x_t) / a.scale) with a.scale = sqrt(2K + 1).
[[kernel]] void ik_conv_dyn(constant ik_args& a [[buffer(0)]], const device bfloat* x [[buffer(1)]],
    const device bfloat* ring [[buffer(2)]], const device bfloat* taps [[buffer(3)]], const device bfloat* dyn [[buffer(4)]],
    device bfloat* y [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, t = gid.y;
  if (c >= a.width || t >= a.rows) return;
  const int p = int(a.pos + t);
  const uint h = c / (a.width / a.heads);
  float s = 0.0f;
  for (int lag = 0; lag < 4; ++lag) {
    const float wt = float(taps[lag * a.width + c]) + float(dyn[ulong(t) * a.heads * 4 + h * 4 + lag]);
    s += ik_src(x, ring, a.width, c, p - lag, int(a.pos)) * wt;
  }
  y[ulong(t) * a.width + c] = bfloat((s + float(x[ulong(t) * a.width + c])) / a.scale);
}

// The chunk's last min(rows, 32) inputs into the ring at their positions mod 32 (after the convolution read it).
[[kernel]] void ik_ring(constant ik_args& a [[buffer(0)]], const device bfloat* x [[buffer(1)]],
    device bfloat* ring [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint keep = min(a.rows, 32u);
  if (gid.x >= a.width || gid.y >= keep) return;
  const uint t = a.rows - keep + gid.y;
  ring[ulong((a.pos + t) % 32) * a.width + gid.x] = x[ulong(t) * a.width + gid.x];
}
