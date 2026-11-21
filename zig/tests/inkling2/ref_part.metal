// Test-only, appended to the Inkling 2 glue library by harness.zig: ik_attn_part with its relative bias read from a table of
// bf16 values (the rows ik_relbias writes) instead of formed from the relative state and the projection. Both kernels run
// the same body, so any difference between their partials is the bias alone.

struct IkRelTable {
  const device bfloat* pb; // [rows][heads][extent]
  inline void two(uint t, uint heads, uint head, uint extent, uint dist, thread float* out) const {
    out[0] = float(pb[(ulong(t) * heads + head) * extent + dist]);
    out[1] = float(pb[(ulong(t) * heads + head + 1) * extent + dist]);
  }
};

[[kernel, max_total_threads_per_threadgroup(768)]] void ik_attn_part_table(constant ik_args& a [[buffer(0)]],
    const device bfloat* qn [[buffer(1)]], const device bfloat* keys [[buffer(2)]], const device bfloat* values [[buffer(3)]],
    const device bfloat* pb [[buffer(4)]], device float* part [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tmax[8][6], tsum[8][6];
  threadgroup float4 tacc[8][6][32];
  ik_attn_part_run(a, qn, keys, values, IkRelTable{pb}, part, tmax, tsum, tacc, tg, tid, sg, lane);
}
