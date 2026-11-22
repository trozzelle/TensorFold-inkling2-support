// Inkling 2's dense prompt projections on the tensor units: bf16 x [M, K] by the release's W [N, K] (torch's Linear
// layout), f32 accumulation, one K pass per output element (no split-K), after gemm_nax.metal. ik_gemm_* write bf16
// rows; ik_gemm_f32_* write the f32 rows of the router's logits. The two digits are the M edge and the N edge
// (1: M a multiple of 64, N a multiple of 128).

#define IK_GEMM_KERNEL(NAME, O, AM, AN)                                                                            \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                                \
      const device bfloat16_t* A [[buffer(0)]], const device bfloat16_t* B [[buffer(1)]],                         \
      const device int32_t* P [[buffer(2)]], device O* D [[buffer(3)]],                                           \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                        \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                                 \
    tfp::gemm_nax<bfloat16_t, false, true, AM, AN, false, 64, 128, 256, 2, 4>(A, B, D, P, tg, sg, lane);          \
  }

IK_GEMM_KERNEL(ik_gemm_00, bfloat16_t, false, false)
IK_GEMM_KERNEL(ik_gemm_01, bfloat16_t, false, true)
IK_GEMM_KERNEL(ik_gemm_10, bfloat16_t, true, false)
IK_GEMM_KERNEL(ik_gemm_11, bfloat16_t, true, true)
IK_GEMM_KERNEL(ik_gemm_f32_00, float, false, false)
IK_GEMM_KERNEL(ik_gemm_f32_01, float, false, true)
IK_GEMM_KERNEL(ik_gemm_f32_10, float, true, false)
IK_GEMM_KERNEL(ik_gemm_f32_11, float, true, true)
