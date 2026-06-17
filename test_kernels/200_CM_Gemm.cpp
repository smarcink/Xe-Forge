#include <cm/cm.h>

#define BLOCK_M 8
#define BLOCK_N 16
#define BLOCK_K 16

#define GROUP_M 1
#define GROUP_N 1

extern "C" _GENX_MAIN_ void
cm_gemm(SurfaceIndex surfA [[type("buffer_t")]],
        SurfaceIndex surfB [[type("buffer_t")]],
        SurfaceIndex surfD [[type("buffer_t")]],
        int M, int N, int K) {
  // cm_global_id == cm_group_id * GROUP + cm_local_id, so each thread owns one
  // distinct BLOCK_M x BLOCK_N output tile whatever the group size is (identical
  // to cm_group_id when GROUP_M = GROUP_N = 1).
  const int tm = cm_global_id(0) * BLOCK_M;
  const int tn = cm_global_id(1) * BLOCK_N;

  // Accumulate the output tile in float for accuracy.
  matrix<float, BLOCK_M, BLOCK_N> acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
    matrix<half, BLOCK_M, BLOCK_K> a;
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++)
      a.row(i) = cm_load<uint32_t, BLOCK_K/2>(surfA, ((tm + i) * K + k0) * sizeof(half)).format<half>();

    matrix<half, BLOCK_K, BLOCK_N> b;
    #pragma unroll
    for (int kk = 0; kk < BLOCK_K; kk++)
      b.row(kk) = cm_load<uint32_t, BLOCK_N/2>(surfB, ((k0 + kk) * N + tn) * sizeof(half)).format<half>();

    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++)
      #pragma unroll
      for (int kk = 0; kk < BLOCK_K; kk++)
        acc.row(i) += a(i, kk) * b.row(kk);
  }

  #pragma unroll
  for (int i = 0; i < BLOCK_M; i++)
    cm_store<float, BLOCK_N>(surfD, ((tm + i) * N + tn) * sizeof(float), acc.row(i));
}
