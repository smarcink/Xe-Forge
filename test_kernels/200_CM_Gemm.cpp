#include <cm/cm.h>

// Stateless ABI: kernel args are raw 64-bit pointers (svmptr_t), bound to plain
// OpenCL buffers by the harness. Because the access is STATELESS, the kernel may
// use the full stateless LSC API -- in particular the untyped 2D block load with
// VNNI transform, which packs the DPAS Src1 operand in HARDWARE from natural
// [K,N] B (no manual VNNI shuffle, no offline pre-pack). Requires -DCM_PTRSIZE=64.
#define BLOCK_M 8
#define BLOCK_N 16
#define BLOCK_K 16

#define GROUP_M 1
#define GROUP_N 1

extern "C" _GENX_MAIN_ void
cm_gemm(svmptr_t A, svmptr_t B, svmptr_t D,
        int M, int N, int K) {
  uint32_t *pA = (uint32_t *)A;
  uint32_t *pB = (uint32_t *)B;
  uint32_t *pD = (uint32_t *)D;
  const int tm = cm_global_id(0) * BLOCK_M;
  const int tn = cm_global_id(1) * BLOCK_N;

  matrix<half, BLOCK_M, BLOCK_N> acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
    matrix<half, BLOCK_M, BLOCK_K> a;
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++)
      a.row(i) = cm_ptr_load<uint32_t, BLOCK_K/2>(pA, ((tm + i) * K + k0) * (unsigned)sizeof(half)).format<half>();

    matrix<half, BLOCK_K, BLOCK_N> b;
    #pragma unroll
    for (int kk = 0; kk < BLOCK_K; kk++)
      b.row(kk) = cm_ptr_load<uint32_t, BLOCK_N/2>(pB, ((k0 + kk) * N + tn) * (unsigned)sizeof(half)).format<half>();

    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++)
      #pragma unroll
      for (int kk = 0; kk < BLOCK_K; kk++)
        acc.row(i) += a(i, kk) * b.row(kk);
  }

  #pragma unroll
  for (int i = 0; i < BLOCK_M; i++)
    cm_ptr_store<uint32_t, BLOCK_N/2>(pD, ((tm + i) * N + tn) * (unsigned)sizeof(half), acc.row(i).format<uint32_t>());
}
