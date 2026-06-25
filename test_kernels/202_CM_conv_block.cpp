#include <cm/cm.h>


#define KH   3                            // kernel height
#define KW   3                            // kernel width
#define PAD  1                            // = KH/2 = KW/2 (kernel default)
#define POOL 2                            // AvgPool2d kernel/stride
#define CHI  128                          // input channels  (CIN)  — structural
#define CHO  128                          // output channels (COUT) — structural

#define LWS_X 1
#define LWS_Y 16

extern "C" _GENX_MAIN_ void
cm_conv_block(svmptr_t X,     // [N, H, W, CIN]        activation (NHWC)
              svmptr_t Wt,    // [KH, KW, CIN, COUT]   conv weight (channels-last)
              svmptr_t Bs,    // [COUT]                conv bias
              svmptr_t Y,     // [N, OH, OW, COUT]     output (NHWC, OH=H/2, OW=W/2)
              int N, int CIN, int COUT, int H, int W,
              int kH, int kW, int OH, int OW) {
  (void)CIN; (void)COUT; (void)kH; (void)kW;   // structural sizes are #defines
  uint32_t *pX = (uint32_t *)X;
  uint32_t *pW = (uint32_t *)Wt;
  uint32_t *pB = (uint32_t *)Bs;
  uint32_t *pY = (uint32_t *)Y;

  const int gx = cm_global_id(0);         // 0 .. N*OH-1 (+ padded tail)
  const int pw = cm_global_id(1);         // 0 .. OW-1   (+ padded tail)
  if (gx >= N * OH || pw >= OW) return;   // drop the LWS padding work-items
  const int n  = gx / OH;
  const int ph = gx % OH;

  vector<float, CHO> bias = cm_ptr_load<uint32_t, CHO / 2>(pB, 0).format<half>();

  // Average over the 2x2 conv window {oh, oh+1} x {ow, ow+1} feeding this pixel.
  vector<float, CHO> pooled = 0.0f;
  #pragma unroll
  for (int dh = 0; dh < POOL; dh++) {
    #pragma unroll
    for (int dw = 0; dw < POOL; dw++) {
      const int oh = POOL * ph + dh;
      const int ow = POOL * pw + dw;

      vector<float, CHO> acc = bias;
      #pragma unroll
      for (int kh = 0; kh < KH; kh++) {
        const int ih = oh + kh - PAD;
        if (ih < 0 || ih >= H) continue;            // zero-padded rows
        #pragma unroll
        for (int kw = 0; kw < KW; kw++) {
          const int iw = ow + kw - PAD;
          if (iw < 0 || iw >= W) continue;          // zero-padded cols

          // Contiguous CHI-vector of input channels at (n, ih, iw).
          const unsigned xoff =
              (unsigned)(((n * H + ih) * W + iw) * CHI) * (unsigned)sizeof(half);
          vector<float, CHI> xv = cm_ptr_load<uint32_t, CHI / 2>(pX, xoff).format<half>();

          // Weight rows for this tap: W[kh,kw, ic, :] is CHO contiguous.
          const unsigned wrow0 = (unsigned)((kh * KW + kw) * CHI) * CHO;
          for (int ic = 0; ic < CHI; ic++) {
            vector<float, CHO> wrow =
                cm_ptr_load<uint32_t, CHO / 2>(
                    pW, (wrow0 + (unsigned)ic * CHO) * (unsigned)sizeof(half))
                    .format<half>();
            acc += xv(ic) * wrow;
          }
        }
      }

      acc.merge(0.0f, acc < 0.0f);                   // ReLU
      pooled += acc;
    }
  }

  pooled = pooled * (1.0f / (POOL * POOL));           // AvgPool2d divide-by-4
  vector<half, CHO> outh = pooled;
  const unsigned yoff =
      (unsigned)(((n * OH + ph) * OW + pw) * CHO) * (unsigned)sizeof(half);
  cm_ptr_store<uint32_t, CHO / 2>(pY, yoff, outh.format<uint32_t>());
}
