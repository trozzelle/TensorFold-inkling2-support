// Test-only, appended to pattn.metal by cases_pattn.zig and bench_pattn.zig: the prompt attention forming its relative bias
// in the kernel (the other way to supply it than reading ik_pattn_bias's table), and the table built on the tensor units.
// ik_pattn_formed runs ik_pattn's body, so any difference between their outputs is the bias alone.

constant constexpr int BP = KS + 8; // the formed bias tile's row pitch

// The relative bias formed from the rows' relative state ([row][head][REL] bf16 at the call's first row) and the projection.

struct IkPaFormed {
  const device bfloat* reln;
  const device float4* projt; // the projection as f32 columns [EXT][REL]
  threadgroup bfloat* bt;     // [NS][RT][BP]: each simdgroup's tile for its 32 keys
  float rl[REL];              // the relative state of the row `lane`
  short2 home;
  uint sg, lane;

  inline void tile(int r0, int q_rows, int qh, int first_pos, uint sg_, uint lane_, short2 home_) {
    sg = sg_;
    lane = lane_;
    home = home_;
    const bool on = int(lane) < q_rows;
    const device bfloat4* r = (const device bfloat4*)(reln + (long(r0 + (on ? int(lane) : 0)) * NQ + qh) * REL);
    TF_UNROLL
    for (short c = 0; c < REL / 4; c++) {
      const float4 v = on ? float4(r[c]) : float4(0.0f);
      rl[4 * c] = v.x;
      rl[4 * c + 1] = v.y;
      rl[4 * c + 2] = v.z;
      rl[4 * c + 3] = v.w;
    }
  }

  // The bias of the row `lane` for the simdgroup's 32 keys c .. c + 32: in each of 32 steps a lane takes the key
  // (lane - step) mod 32, so a step's keys lie on one diagonal of the tile and the lanes of a step share at most two
  // columns. The 16 products sum in index order from zero and the sum rounds to bf16, as ik_pattn_bias's does.
  // dsg: the distance of (the tile's first row, the simdgroup's first key).
  inline void form(int dsg) {
    threadgroup bfloat* row = bt + (sg * RT + lane) * BP;
    _Pragma("clang loop unroll_count(4)")
    for (int step = 0; step < KS; step++) {
      const bool wrap = int(lane) < step;
      const int key = wrap ? int(lane) - step + KS : int(lane) - step;
      const device float4* pc = projt + clamp(dsg + step - (wrap ? KS : 0), 0, EXT - 1) * (REL / 4);
      const float4 p0 = pc[0], p1 = pc[1], p2 = pc[2], p3 = pc[3];
      float s = 0.0f;
      s = fma(rl[0], p0.x, s);
      s = fma(rl[1], p0.y, s);
      s = fma(rl[2], p0.z, s);
      s = fma(rl[3], p0.w, s);
      s = fma(rl[4], p1.x, s);
      s = fma(rl[5], p1.y, s);
      s = fma(rl[6], p1.z, s);
      s = fma(rl[7], p1.w, s);
      s = fma(rl[8], p2.x, s);
      s = fma(rl[9], p2.y, s);
      s = fma(rl[10], p2.z, s);
      s = fma(rl[11], p2.w, s);
      s = fma(rl[12], p3.x, s);
      s = fma(rl[13], p3.y, s);
      s = fma(rl[14], p3.z, s);
      s = fma(rl[15], p3.w, s);
      row[key] = bfloat(s);
    }
  }

  inline void block(int kpos, int dmax, short2) {
    if (dmax - (BKT - 1) < EXT) {
      form(dmax - KS * int(sg));
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  inline float at(short i, short e, int col, int dist) const {
    const int local = 16 * (i >> 1) + home.y + (e >> 2) * 8, ck = 16 * (i & 1) + home.x + TF_COL(e);
    return dist < EXT ? float(bt[(sg * RT + local) * BP + ck]) : 0.0f;
  }
};

// ik_pattn's buffers, with reln [rows][48][16] in place of the table and the f32 projection columns projt [EXT][REL] at
// buffer 8. Dispatch ceil(rows / 32) x 48 threadgroups of 64 threads.
[[max_total_threads_per_threadgroup(NS * 32)]]
[[kernel]] void ik_pattn_formed(constant IkPattnArgs& a [[buffer(0)]], const device bfloat* Q [[buffer(1)]],
                                const device bfloat* K [[buffer(2)]], const device bfloat* V [[buffer(3)]],
                                const device bfloat* sinks [[buffer(4)]], const device bfloat* reln [[buffer(5)]],
                                const device bfloat* gate [[buffer(6)]], device bfloat* O [[buffer(7)]],
                                const device float4* projt [[buffer(8)]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup PT4 pt4[RT * PP / 4];
  threadgroup float rowmax[NS * RT];
  threadgroup bfloat bt[NS * RT * BP];
  IkPaFormed bias;
  bias.reln = reln;
  bias.projt = projt;
  bias.bt = bt;
  ik_pa_run(a, Q, K, V, sinks, gate, O, bias, (threadgroup PT*)pt4, rowmax, sg, lane, tg);
}

// The table on the tensor units: one simdgroup a row and 16 heads, the 1,024 distances in 32 groups of 32 through one
// 16x32x16 product each. Its sums are the unit's, not index order, so it differs from ik_pattn_bias's table in rare bf16
// values: it is measured, not used. Dispatch (rows, 3) threadgroups of 32 threads.
[[kernel]] void ik_pattn_bias_mma(constant IkPattnArgs& a [[buffer(0)]], const device bfloat* reln [[buffer(1)]],
                                  const device bfloat* proj [[buffer(2)]], device bfloat* table [[buffer(3)]],
                                  uint lane [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  const int t = int(tg.x), h0 = 16 * int(tg.y);
  const short2 home = frag_home(ushort(lane));
  const device bfloat* r = reln + (long(t) * NQ + h0) * REL;
  const frag<bfloat> ra = ik_pa_blk<false>(r, REL, 0, 0, home);
  const int pos = int(a.p0) + t;
  device bfloat* rows = table + (long(t) * NQ + h0) * EXT;
  for (int n = 0; n < EXT / 32; n++) {
    const frag<bfloat> b0 = ik_pa_blk<false>(proj, EXT, 0, 32 * n, home);
    const frag<bfloat> b1 = ik_pa_blk<false>(proj, EXT, 0, 32 * n + 16, home);
    frag<float> c0 = frag<float>(0), c1 = frag<float>(0);
    mma_16x32<false, false>(c0, c1, ra, b0, b1);
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const int row = home.y + (e >> 2) * 8, d = 32 * n + home.x + TF_COL(e);
      rows[long(row) * EXT + ((pos - d) & (EXT - 1))] = bfloat(c0[e]);
      rows[long(row) * EXT + ((pos - d - 16) & (EXT - 1))] = bfloat(c1[e]);
    }
  }
}
