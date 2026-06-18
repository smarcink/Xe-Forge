#include <cm/cm.h>

// Register-blocked GEMM  (no XMX/DPAS, no SLM, single-buffered).
// D = A @ B,  A:[M,K] half,  B:[K,N] half,  D:[M,N] half  (fp16 fast-MAD accum).
//
// Each thread owns a BLOCK_M x BLOCK_N output tile, held in registers across the
// whole K-loop, and updates it by a rank-BLOCK_K outer product each step. The
// loaded A/B elements are reused BLOCK_N / BLOCK_M times respectively, which is
// what keeps the half MAD units fed.
//
// MEASURED FINDING (ARL Xe-LPG, 4096^3, .temp/gemm_sweep.py): on this in-order,
// DPAS-less iGPU the kernel is OCCUPANCY/LATENCY-bound, not reuse-bound. Three
// non-obvious results drove this final form:
//   * SINGLE-buffering beats source-level double-buffering / software pipelining.
//     A hand-rolled "prefetch next K-slab into _nxt regs, copy _nxt->_cur" loop
//     was NO faster (the HW already overlaps the independent LSC loads with the
//     MADs) and SLOWER once it cost enough GRF to drop a tile size -- the copy is
//     pure overhead. So there is no prefetch here; the EU hides the latency.
//   * Small tile + GRF=128 (the DEFAULT file) wins over big tile + GRF=256:
//     more resident hardware threads hide latency better than more per-thread
//     reuse. 8x64 in the default register file beat every 16x/32x large-GRF tile.
//   * Shallow BLOCK_K (8) edges deeper K -- again favouring occupancy headroom.
// Net: ~1.4-1.6 TFLOPS (~40% of the 3.855 TF fp16 ALU peak), ~2.4x the naive
// seed. No large-GRF build directive is needed -- the tile fits the default file.
//
// LSC block loads need >= 32-bit elements, so 2 halves are packed per uint32
// (cm_load<uint32_t, N/2>(...).format<half>()); rows are contiguous -> 1D loads.

#define BLOCK_M 8
#define BLOCK_N 64
#define BLOCK_K 8

#define GROUP_M 1
#define GROUP_N 1

extern "C" _GENX_MAIN_ void
cm_gemm(SurfaceIndex surfA [[type("buffer_t")]],
        SurfaceIndex surfB [[type("buffer_t")]],
        SurfaceIndex surfD [[type("buffer_t")]],
        int M, int N, int K) {
  // One BLOCK_M x BLOCK_N output tile per thread (GROUP=1 -> global_id == group_id).
  const int tm = cm_global_id(0) * BLOCK_M;   // first output row
  const int tn = cm_global_id(1) * BLOCK_N;   // first output col

  // fp16 accumulator, resident in registers across the whole K-loop.
  matrix<half, BLOCK_M, BLOCK_N> acc = 0.0f;

  matrix<half, BLOCK_M, BLOCK_K> a;
  matrix<half, BLOCK_K, BLOCK_N> b;

  for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
    // ---- load this K-slab: A[BLOCK_M x BLOCK_K] and B[BLOCK_K x BLOCK_N] ----
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++)
      a.row(i) =
          cm_load<uint32_t, BLOCK_K / 2>(surfA, ((tm + i) * K + k0) * sizeof(half)).format<half>();
    #pragma unroll
    for (int kk = 0; kk < BLOCK_K; kk++)
      b.row(kk) =
          cm_load<uint32_t, BLOCK_N / 2>(surfB, ((k0 + kk) * N + tn) * sizeof(half)).format<half>();

    // ---- rank-BLOCK_K outer-product update; BLOCK_M independent acc rows give
    //      the ILP the in-order EU uses to hide the load + MAD latency ----
    #pragma unroll
    for (int kk = 0; kk < BLOCK_K; kk++) {
      vector<half, BLOCK_N> brow = b.row(kk);
      #pragma unroll
      for (int i = 0; i < BLOCK_M; i++)
        acc.row(i) += a(i, kk) * brow;   // half scalar * half row, broadcast
    }
  }

  // ---- store the fp16 tile (2 halves packed per uint32) ----
  #pragma unroll
  for (int i = 0; i < BLOCK_M; i++)
    cm_store<uint32_t, BLOCK_N / 2>(surfD, ((tm + i) * N + tn) * sizeof(half),
                                    acc.row(i).format<uint32_t>());
}
