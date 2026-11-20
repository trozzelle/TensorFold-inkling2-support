// Inkling 2's routed experts around the EXL3 projections: fp32 router logits, mlx-lm's expert_routing, grouped 8-row
// tiles by expert for exl3_projection's grouped mode, the capped SwiGLU on interleaved rows and the weighted combine.
#include <metal_stdlib>
using namespace metal;

constant int IK_TOPK = 8;
constant int IK_EMAX = 384;

// logits[t][e] = sum_d x[t][d] w[e][d] in f32 (the reference's fp32 router on bf16-exact weights).
[[kernel]] void ik_router(constant ik_args& a [[buffer(0)]], const device bfloat* x [[buffer(1)]],
    const device bfloat* w [[buffer(2)]], device float* logits [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint e = tg.x * 8 + sg, t = tg.y;
  if (e >= a.width) return;
  const device bfloat* xr = x + ulong(t) * a.dim;
  const device bfloat* wr = w + ulong(e) * a.dim;
  float acc = 0.0f;
  for (uint i = 4 * lane; i < a.dim; i += 128)
    acc += dot(float4(*(const device bfloat4*)(xr + i)), float4(*(const device bfloat4*)(wr + i)));
  const float s = simd_sum(acc);
  if (lane == 0) logits[ulong(t) * a.width + e] = s;
}

// mlx-lm's expert_routing kernel (multiplier 1): sigmoid scores, top-8 of score + bias with ties to the lower id, weights
// score / (sum + a.eps) * a.scale with the sum taken in pick order.
[[kernel]] void ik_route(constant ik_args& a [[buffer(0)]], const device float* logits [[buffer(1)]],
    const device float* bias [[buffer(2)]], device uint* inds [[buffer(3)]], device float* weights [[buffer(4)]],
    uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int R = (IK_EMAX + 31) / 32;
  const uint token = gid / 32;
  const int E = int(a.width);
  const device float* row = logits + ulong(token) * E;
  float score[R], biased[R];
  for (int i = 0; i < R; i++) {
    const int e = int(lane) + 32 * i;
    score[i] = 0.0f;
    biased[i] = -INFINITY;
    if (e < E) {
      const float x = row[e];
      const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
      score[i] = x < 0 ? y : 1.0f - y;
      biased[i] = score[i] + bias[e];
    }
  }
  float w[IK_TOPK];
  for (int k = 0; k < IK_TOPK; k++) {
    float best = -INFINITY;
    int best_e = E;
    for (int i = 0; i < R; i++) {
      const int e = int(lane) + 32 * i;
      if (biased[i] > best || (biased[i] == best && e < best_e)) { best = biased[i]; best_e = e; }
    }
    const float top = simd_max(best);
    const int winner = simd_min(best == top ? best_e : E);
    float s = 0.0f;
    for (int i = 0; i < R; i++) if (int(lane) + 32 * i == winner) { s = score[i]; biased[i] = -INFINITY; }
    w[k] = simd_sum(s);
    if (lane == 0) inds[ulong(token) * IK_TOPK + k] = uint(winner);
  }
  float total = 0.0f;
  for (int k = 0; k < IK_TOPK; k++) total = w[k] + total;
  if (lane == 0) for (int k = 0; k < IK_TOPK; k++) weights[ulong(token) * IK_TOPK + k] = w[k] / (total + a.eps) * a.scale;
}

// The picks grouped by expert into whole 8-row tiles in expert order, slots ascending within an expert: each tile row's
// expert (0xffffffff padding), slot and token. One threadgroup; a.stride rows written.
[[kernel]] void ik_tiles(constant ik_args& a [[buffer(0)]], const device uint* ids [[buffer(1)]],
    device uint* texp [[buffer(2)]], device uint* tslot [[buffer(3)]], device uint* ttok [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup uint start[IK_EMAX];
  const uint E = a.width, S = a.rows;
  for (uint e = tid; e < E; e += 1024) {
    uint c = 0;
    for (uint s = 0; s < S; ++s) c += ids[s] == e;
    start[e] = (c + 7) / 8 * 8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint at = 0;
    for (uint e = 0; e < E; ++e) { const uint n = start[e]; start[e] = at; at += n; }
  }
  for (uint i = tid; i < a.stride; i += 1024) { texp[i] = 0xffffffffu; tslot[i] = 0; ttok[i] = 0; }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  for (uint e = tid; e < E; e += 1024) {
    uint at = start[e];
    for (uint s = 0; s < S; ++s) if (ids[s] == e) { texp[at] = e; tslot[at] = s; ttok[at] = s / a.queries; ++at; }
  }
}

// SoftcappedSwiGLU on interleaved rows (gate at 2i, up at 2i + 1): cap_g tanh(g / cap_g) sigmoid(g) cap_u tanh(u / cap_u).
[[kernel]] void ik_swiglu(constant ik_args& a [[buffer(0)]], const device bfloat* gu [[buffer(1)]],
    device bfloat* act [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x, s = gid.y;
  if (i >= a.width || s >= a.rows) return;
  const float g = float(gu[ulong(s) * 2 * a.width + 2 * i]), u = float(gu[ulong(s) * 2 * a.width + 2 * i + 1]);
  act[ulong(s) * a.width + i] = bfloat((tanh(g / a.cap) * a.cap) * (1.0f / (1.0f + exp(-g))) * (tanh(u / a.thr) * a.thr));
}

// SparseMoeBlock's sum: bf16(sum_k f32(y_k) w_k) in pick order.
[[kernel]] void ik_combine(constant ik_args& a [[buffer(0)]], const device bfloat* ey [[buffer(1)]],
    const device float* w [[buffer(2)]], device bfloat* moe [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
  const uint d = gid.x, t = gid.y;
  if (d >= a.width || t >= a.rows) return;
  float s = 0.0f;
  for (uint k = 0; k < a.queries; ++k) s += float(ey[(ulong(t) * a.queries + k) * a.width + d]) * w[ulong(t) * a.queries + k];
  moe[ulong(t) * a.width + d] = bfloat(s);
}
