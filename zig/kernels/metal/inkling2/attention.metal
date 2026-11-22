// Inkling 2's attention at mlx-lm's rounding points: q, k and relative-state L2 clips, unit-RMS values, the global
// layers' tau, the relative bias bf16(rel . proj), and attention over the cache with the sink logit and output gate.
#include <metal_stdlib>
using namespace metal;

constant uint IK_DREL = 16;

inline float ik_tau(uint p, constant ik_args& a) {
  return 1.0f + a.alpha * log(max((float(p) + 1.0f) / a.floor_n, 1.0f));
}

// One simdgroup per head slot: q heads, then k heads, then v heads, then relative-state heads.
[[kernel]] void ik_prep(constant ik_args& a [[buffer(0)]], const device bfloat* q [[buffer(1)]],
    const device bfloat* k [[buffer(2)]], const device bfloat* v [[buffer(3)]], const device bfloat* r [[buffer(4)]],
    const device bfloat* qs [[buffer(5)]], const device bfloat* rs [[buffer(6)]], device bfloat* qn [[buffer(7)]],
    device bfloat* keys [[buffer(8)]], device bfloat* values [[buffer(9)]], device bfloat* reln [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint slot = tg.x, t = tg.y, H = a.heads, D = a.dim, p = a.pos + t;
  const bool global = a.flags & IK_GLOBAL;
  const float tau = global ? ik_tau(p, a) : 1.0f;
  if (slot < H) {
    float s[4], ss = 0.0f;
    for (uint j = 0; j < 4; ++j) {
      const uint d = lane + 32 * j;
      s[j] = float(q[(ulong(t) * H + slot) * D + d]) * float(qs[slot * D + d]);
      ss += s[j] * s[j];
    }
    const float f = min(a.cap / (sqrt(simd_sum(ss)) + 1e-6f), 1.0f);
    for (uint j = 0; j < 4; ++j) {
      float c = float(bfloat(s[j] * f));
      if (global) c = c * tau;
      qn[(ulong(t) * H + slot) * D + lane + 32 * j] = bfloat(c);
    }
  } else if (slot < H + a.kv) {
    const uint h = slot - H;
    const ulong at = ulong(p % a.ring) * a.kv;
    float s[4], ss = 0.0f;
    for (uint j = 0; j < 4; ++j) { s[j] = float(k[(ulong(t) * a.kv + h) * D + lane + 32 * j]); ss += s[j] * s[j]; }
    const float f = min(a.cap / (sqrt(simd_sum(ss)) + 1e-6f), 1.0f);
    for (uint j = 0; j < 4; ++j) keys[(at + h) * D + lane + 32 * j] = bfloat(s[j] * f);
  } else if (slot < H + 2 * a.kv) {
    const uint h = slot - H - a.kv;
    const ulong at = ulong(p % a.ring) * a.kv;
    float s[4], ss = 0.0f;
    for (uint j = 0; j < 4; ++j) { s[j] = float(v[(ulong(t) * a.kv + h) * D + lane + 32 * j]); ss += s[j] * s[j]; }
    const float inv = rsqrt(simd_sum(ss) / float(D) + a.eps);
    for (uint j = 0; j < 4; ++j) values[(at + h) * D + lane + 32 * j] = bfloat(s[j] * inv);
  } else {
    const uint h = slot - H - 2 * a.kv;
    const bool on = lane < IK_DREL;
    const float s = on ? float(r[(ulong(t) * H + h) * IK_DREL + lane]) * float(rs[h * IK_DREL + lane]) : 0.0f;
    const float f = min(a.thr / (sqrt(simd_sum(s * s)) + 1e-6f), 1.0f);
    if (on) {
      float c = float(bfloat(s * f));
      if (global) c = c * tau;
      reln[(ulong(t) * H + h) * IK_DREL + lane] = bfloat(c);
    }
  }
}

// position_bias = bf16(rel . proj) for each row, head and distance below a.extent.
[[kernel]] void ik_relbias(constant ik_args& a [[buffer(0)]], const device bfloat* reln [[buffer(1)]],
    const device bfloat* proj [[buffer(2)]], device bfloat* pb [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
  const uint d = gid.x, h = gid.y, t = gid.z;
  if (d >= a.extent || h >= a.heads || t >= a.rows) return;
  const device bfloat* r = reln + (ulong(t) * a.heads + h) * a.dim;
  float s = 0.0f;
  for (uint i = 0; i < a.dim; ++i) s += float(r[i]) * float(proj[ulong(i) * a.extent + d]);
  pb[(ulong(t) * a.heads + h) * a.extent + d] = bfloat(s);
}

// One row's query head over its keys in ascending order: an online softmax started from the sink logit (a zero-valued
// key), the relative bias at distances below a.extent, a sliding window when a.window > 0; then the output gate.
[[kernel]] void ik_attn(constant ik_args& a [[buffer(0)]], const device bfloat* qn [[buffer(1)]],
    const device bfloat* keys [[buffer(2)]], const device bfloat* values [[buffer(3)]], const device bfloat* pb [[buffer(4)]],
    const device bfloat* sink [[buffer(5)]], const device bfloat* gate [[buffer(6)]], device bfloat* out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint h = tg.x, t = tg.y, D = a.dim, g = h / (a.heads / a.kv), p = a.pos + t;
  const uint first = (a.window > 0 && p + 1 > a.window) ? p + 1 - a.window : 0;
  float q[4], acc[4];
  for (uint j = 0; j < 4; ++j) { q[j] = float(qn[(ulong(t) * a.heads + h) * D + lane + 32 * j]); acc[j] = 0.0f; }
  float m = float(sink[ulong(t) * a.heads + h]), l = 1.0f;
  const device bfloat* bias = pb + (ulong(t) * a.heads + h) * a.extent;
  for (uint key = first; key <= p; ++key) {
    const device bfloat* kr = keys + (ulong(key % a.ring) * a.kv + g) * D;
    float s = 0.0f;
    for (uint j = 0; j < 4; ++j) s += q[j] * float(kr[lane + 32 * j]);
    s = simd_sum(s) * a.scale;
    if (p - key < a.extent) s += float(bias[p - key]);
    const float mn = max(m, s), f = exp(m - mn), e = exp(s - mn);
    l = l * f + e;
    const device bfloat* vr = values + (ulong(key % a.ring) * a.kv + g) * D;
    for (uint j = 0; j < 4; ++j) acc[j] = acc[j] * f + e * float(vr[lane + 32 * j]);
    m = mn;
  }
  for (uint j = 0; j < 4; ++j) {
    const ulong at = ulong(t) * a.heads * D + h * D + lane + 32 * j;
    const float o = float(bfloat(acc[j] / l));
    out[at] = bfloat(o / (1.0f + exp(-float(gate[at]))));
  }
}
