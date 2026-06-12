#include <cm/cm.h>

#define TILE_M 8
#define TILE_N 16
#define TILE_K 16

extern "C" _GENX_MAIN_ void
cm_gemm(SurfaceIndex surfA [[type("buffer_t")]],
        SurfaceIndex surfB [[type("buffer_t")]],
        SurfaceIndex surfD [[type("buffer_t")]],
        int M, int N, int K) {
  const int tm = cm_group_id(0) * TILE_M;
  const int tn = cm_group_id(1) * TILE_N;

  // Accumulate the output tile in float for accuracy.
  matrix<float, TILE_M, TILE_N> acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += TILE_K) {
    matrix<bfloat16, TILE_M, TILE_K> a;
    #pragma unroll
    for (int i = 0; i < TILE_M; i++)
      a.row(i) = cm_load<bfloat16, TILE_K>(surfA, ((tm + i) * K + k0) * sizeof(bfloat16));

    matrix<bfloat16, TILE_K, TILE_N> b;
    #pragma unroll
    for (int kk = 0; kk < TILE_K; kk++)
      b.row(kk) = cm_load<bfloat16, TILE_N>(surfB, ((k0 + kk) * N + tn) * sizeof(bfloat16));

    #pragma unroll
    for (int i = 0; i < TILE_M; i++)
      #pragma unroll
      for (int kk = 0; kk < TILE_K; kk++)
        acc.row(i) += a(i, kk) * b.row(kk);
  }

  #pragma unroll
  for (int i = 0; i < TILE_M; i++)
    cm_store<float, TILE_N>(surfD, ((tm + i) * N + tn) * sizeof(float), acc.row(i));
}
