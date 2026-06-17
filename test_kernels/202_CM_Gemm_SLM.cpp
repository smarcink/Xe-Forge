#include <cm/cm.h>

// Cooperative SLM GEMM reference: D = A @ B.
//   A: [M, K] fp16,  B: [K, N] fp16,  D: [M, N] fp32.
//
// This is the cooperative counterpart of the naive seed 200_CM_Gemm.cpp. It
// shows the one optimization the naive kernel cannot express on its own:
// reusing a shared input tile across a *thread group* through Shared Local
// Memory (SLM).
//
// MODEL: each thread still owns one BLOCK_M x BLOCK_N output tile, indexed by
// cm_global_id (so the output mapping is identical to the seed). GROUP_M
// threads are grouped along M (local = (GROUP_M, 1)); every thread in a group
// shares the SAME B sub-tile B[k0:k0+BLOCK_K, tn:tn+BLOCK_N] each K-step. So the
// group stages that B tile into SLM ONCE (each thread loads BLOCK_K/GROUP_M
// rows), barriers, and then all GROUP_M threads read it back from SLM — cutting
// B's global-memory traffic by GROUP_M x. Each thread's A rows are private and
// stay in registers.
//
// GROUP_M is a cooperative work-group-size knob: it drives the launch grid
// (global.x = ceil(M / (BLOCK_M*GROUP_M)) * GROUP_M, local.x = GROUP_M) and MUST
// divide BLOCK_K so the staged B rows split evenly across the group.

#define BLOCK_M 8
#define BLOCK_N 16
#define BLOCK_K 16
#define GROUP_M 4   // cooperating threads per group (must divide BLOCK_K)
#define GROUP_N 1   // (kept for grid symmetry with the seed; 1 = share B only)

#define BN2 (BLOCK_N / 2)               // halfs packed two-per-uint32 along N
#define KROWS_PER_THREAD (BLOCK_K / GROUP_M)
#define SLM_BYTES (BLOCK_K * BN2 * 4)   // B tile staged as packed uint32

extern "C" _GENX_MAIN_ void
cm_gemm(SurfaceIndex surfA [[type("buffer_t")]],
        SurfaceIndex surfB [[type("buffer_t")]],
        SurfaceIndex surfD [[type("buffer_t")]],
        int M, int N, int K) {
  const uint lid = cm_local_id(0);             // 0 .. GROUP_M-1 within the group
  const int tm = cm_global_id(0) * BLOCK_M;    // this thread's private row base
  const int tn = cm_group_id(1) * BLOCK_N;     // column base shared by the group

  cm_slm_init(SLM_BYTES);
  uint slm = cm_slm_alloc(SLM_BYTES);

  // Accumulate the output tile in float for accuracy.
  matrix<float, BLOCK_M, BLOCK_N> acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
    // --- cooperatively stage the shared B tile into SLM (packed uint32) ---
    #pragma unroll
    for (int r = 0; r < KROWS_PER_THREAD; r++) {
      int krow = lid * KROWS_PER_THREAD + r;   // B row this thread is responsible for
      vector<uint, BN2> brow =
          cm_load<uint32_t, BN2>(surfB, ((k0 + krow) * N + tn) * sizeof(half));
      cm_store_slm<uint, BN2>((krow * BN2) * sizeof(uint), brow);
    }
    cm_slm_fence(CM_GLOBAL_COHERENT_FENCE);    // order the SLM writes
    cm_barrier();                              // make them visible to the group

    // --- load this thread's private A rows from HBM ---
    matrix<half, BLOCK_M, BLOCK_K> a;
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++)
      a.row(i) = cm_load<uint32_t, BLOCK_K / 2>(surfA, ((tm + i) * K + k0) * sizeof(half)).format<half>();

    // --- read the shared B tile back from SLM and multiply ---
    matrix<half, BLOCK_K, BLOCK_N> b;
    #pragma unroll
    for (int kk = 0; kk < BLOCK_K; kk++)
      b.row(kk) = cm_load_slm<uint, BN2>((kk * BN2) * sizeof(uint)).format<half>();

    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++)
      #pragma unroll
      for (int kk = 0; kk < BLOCK_K; kk++)
        acc.row(i) += a(i, kk) * b.row(kk);

    cm_barrier();   // all reads done before the next K-step overwrites the SLM tile
  }

  #pragma unroll
  for (int i = 0; i < BLOCK_M; i++)
    cm_store<float, BLOCK_N>(surfD, ((tm + i) * N + tn) * sizeof(float), acc.row(i));
}
