// Inkling 2's short convolutions at mlx-lm's rounding points: static taps convolve in the weights' bf16, dynamic taps
// (global layers' k and v) in f32; inputs before this chunk come from a 32-position ring. Follows glue.metal.
#include <metal_stdlib>
using namespace metal;

// A convolution input at position q: this chunk's row from a.pos on, the ring's before it, zero before position 0.
inline float ik_src(const device bfloat* x, const device bfloat* ring, uint C, uint c, int q, int pos0) {
  if (q < 0) return 0.0f;
  return q >= pos0 ? float(x[ulong(q - pos0) * C + c]) : float(ring[ulong(q % 32) * C + c]);
}

// ShortConvolution at channel c, row t: bf16(sum_j w[c][j] x[t-3+j]), then bf16((that + x_t) / a.scale) with
// a.scale = sqrt(K + 1).
inline bfloat ik_conv_at(constant ik_args& a, const device bfloat* x, const device bfloat* ring, const device bfloat* w,
    uint c, uint t) {
  const int p = int(a.pos + t);
  float s = 0.0f;
  for (int j = 0; j < 4; ++j) s += float(w[c * 4 + j]) * ik_src(x, ring, a.width, c, p - 3 + j, int(a.pos));
  return bfloat((float(bfloat(s)) + float(x[ulong(t) * a.width + c])) / a.scale);
}

// DynamicShortConvolution at channel c, row t: taps[lag][c] + the row's projection [heads, lags], over x[t - lag] in
// f32, then bf16((that + x_t) / a.scale) with a.scale = sqrt(2K + 1).
inline bfloat ik_conv_dyn_at(constant ik_args& a, const device bfloat* x, const device bfloat* ring,
    const device bfloat* taps, const device bfloat* dyn, uint c, uint t) {
  const int p = int(a.pos + t);
  const uint h = c / (a.width / a.heads);
  float s = 0.0f;
  for (int lag = 0; lag < 4; ++lag) {
    const float wt = float(taps[lag * a.width + c]) + float(dyn[ulong(t) * a.heads * 4 + h * 4 + lag]);
    s += ik_src(x, ring, a.width, c, p - lag, int(a.pos)) * wt;
  }
  return bfloat((s + float(x[ulong(t) * a.width + c])) / a.scale);
}

[[kernel]] void ik_conv(constant ik_args& a [[buffer(0)]], const device bfloat* x [[buffer(1)]],
    const device bfloat* ring [[buffer(2)]], const device bfloat* w [[buffer(3)]], device bfloat* y [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, t = gid.y;
  if (c >= a.width || t >= a.rows) return;
  y[ulong(t) * a.width + c] = ik_conv_at(a, x, ring, w, c, t);
}

[[kernel]] void ik_conv_dyn(constant ik_args& a [[buffer(0)]], const device bfloat* x [[buffer(1)]],
    const device bfloat* ring [[buffer(2)]], const device bfloat* taps [[buffer(3)]], const device bfloat* dyn [[buffer(4)]],
    device bfloat* y [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, t = gid.y;
  if (c >= a.width || t >= a.rows) return;
  y[ulong(t) * a.width + c] = ik_conv_dyn_at(a, x, ring, taps, dyn, c, t);
}

// The chunk's last min(rows, 32) inputs into the ring at their positions mod 32 (after the convolution read it).
[[kernel]] void ik_ring(constant ik_args& a [[buffer(0)]], const device bfloat* x [[buffer(1)]],
    device bfloat* ring [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint keep = min(a.rows, 32u);
  if (gid.x >= a.width || gid.y >= keep) return;
  const uint t = a.rows - keep + gid.y;
  ring[ulong((a.pos + t) % 32) * a.width + gid.x] = x[ulong(t) * a.width + gid.x];
}

// The decode path's convolution and ring advance in one dispatch (at most 16 rows). A thread writes only its own row's
// input into the ring, at slot (pos + t) % 32: the slots a chunk of fewer than 30 rows writes are not the three,
// (pos - 3 .. pos - 1) % 32, that its convolutions read, so no thread reads a slot another writes.
inline void ik_ring_put(constant ik_args& a, const device bfloat* x, device bfloat* ring, uint c, uint t) {
  if (t + min(a.rows, 32u) >= a.rows) ring[ulong((a.pos + t) % 32) * a.width + c] = x[ulong(t) * a.width + c];
}

// ik_conv then ik_ring.
[[kernel]] void ik_conv_ring(constant ik_args& a [[buffer(0)]], const device bfloat* x [[buffer(1)]],
    device bfloat* ring [[buffer(2)]], const device bfloat* w [[buffer(3)]], device bfloat* y [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, t = gid.y;
  if (c >= a.width || t >= a.rows) return;
  y[ulong(t) * a.width + c] = ik_conv_at(a, x, ring, w, c, t);
  ik_ring_put(a, x, ring, c, t);
}

// ik_conv_dyn then ik_ring.
[[kernel]] void ik_conv_dyn_ring(constant ik_args& a [[buffer(0)]], const device bfloat* x [[buffer(1)]],
    device bfloat* ring [[buffer(2)]], const device bfloat* taps [[buffer(3)]], const device bfloat* dyn [[buffer(4)]],
    device bfloat* y [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, t = gid.y;
  if (c >= a.width || t >= a.rows) return;
  y[ulong(t) * a.width + c] = ik_conv_dyn_at(a, x, ring, taps, dyn, c, t);
  ik_ring_put(a, x, ring, c, t);
}

// The MLP branch: ik_conv, ik_ring, then ik_out (the layer's output h = bf16(res + the convolution's bf16)).
[[kernel]] void ik_conv_out(constant ik_args& a [[buffer(0)]], const device bfloat* x [[buffer(1)]],
    device bfloat* ring [[buffer(2)]], const device bfloat* w [[buffer(3)]], const device float* res [[buffer(4)]],
    device bfloat* h [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, t = gid.y;
  if (c >= a.width || t >= a.rows) return;
  const ulong i = ulong(t) * a.width + c;
  h[i] = bfloat(res[i] + float(ik_conv_at(a, x, ring, w, c, t)));
  ik_ring_put(a, x, ring, c, t);
}
