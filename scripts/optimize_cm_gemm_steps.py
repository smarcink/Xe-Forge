#!/usr/bin/env python3
"""Step-by-step optimizer + storyteller for the CM GEMM kernel on Intel Xe (BMG).

Takes the XeForge-optimized GEMM as *stage 0* and applies a sequence of
incremental, individually-measured optimizations on top of it. For every stage
this script:

  * generates the CM ``.cpp`` (or, for stage 0, reads the seed verbatim),
  * optionally **rearranges the A / B tensors offline** so the kernel can load
    DPAS operands directly (no in-kernel VNNI shuffle),
  * compiles it ONLINE via the Intel runtime's ``-cmc`` VC frontend,
  * validates the result against a numpy golden (cosine similarity),
  * times it (min-of-iters) and converts to TFLOPS,
  * writes a *stamped* kernel to ``outputs/kernels/steps/<ts>/`` whose header
    records exactly what changed and what it measured.

At the end it prints a "story" table with the per-step and cumulative speedup
so you can see **which step gave the biggest boost**, and emits a roofline-ready
CSV you can feed straight into ``scripts/roofline.py``.

The offline B (and A) rearrangement is the key lever the user asked for: instead
of VNNI-packing B inside the hot loop every k-step, B is pre-packed on the host
into the exact DPAS Src1 byte layout, so the kernel just issues one contiguous
``cm_load`` per atom. The packed bytes are *identical* to what the in-kernel
shuffle produced, so the DPAS call is unchanged and the result is bit-for-bit
the same as the baseline -- only faster.

Run (use the venv python; bare ``python`` may be a deps-less system interpreter):

    # validate the offline packers on the CPU only -- no GPU needed
    python scripts/optimize_cm_gemm_steps.py --self-test

    # run the full staged story on the BMG device
    python scripts/optimize_cm_gemm_steps.py --m 4096 --n 4096 --k 4096 --iters 100

    # narrow / widen the roof used for the printed % of peak
    python scripts/optimize_cm_gemm_steps.py --peak-tflops 150 --peak-bandwidth 400
"""

from __future__ import annotations

import argparse
import datetime as _dt
import pathlib
import sys

import numpy as np

# Reuse the proven device picker + #define extractor from the scratch harness so
# this script tracks the same ABI / grid convention without duplicating it.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scratch_pyopencl import (  # noqa: E402  (after sys.path tweak)
    _type_label,
    extract_defines,
    pick_device,
)

DEFAULT_BASELINE = "outputs/kernels/cm_gemm_20260618_103122_optimized.cpp"

# DPAS atom geometry for fp16 on Xe: RepeatCount rows x ExecSize cols out,
# SystolicDepth along K, 2 fp16 per dword (VNNI2) -> K consumed per atom = 16.
RPT = 8       # DPAS RepeatCount (A rows per atom)
DPAS_N = 16   # DPAS output columns per atom (ExecSize)
DPAS_K = 16   # K consumed per atom = SystolicDepth(8) * 2


# ===========================================================================
# Offline tensor rearrangement (the "avoid the in-kernel VNNI" lever)
# ===========================================================================
# Both packers produce byte-for-byte the same operand the baseline built inside
# the loop, just laid out so each DPAS atom is ONE contiguous cm_load.
def pack_B_vnni(B: np.ndarray) -> np.ndarray:
    """``B[K, N]`` fp16 -> VNNI-blocked uint32, atoms ordered ``[jb][kb][kp][col]``.

    Per (N-block ``jb``, K-block ``kb``) the 128-dword operand is
    ``dword(kp, col) = pack2(B[kb*16 + 2*kp + 0, jb*16 + col],
                             B[kb*16 + 2*kp + 1, jb*16 + col])``
    with the even K-row in the low half and the odd K-row in the high half --
    exactly the bytes the baseline produced via ``bv ... .format<uint32_t>()``.
    """
    K, N = B.shape
    assert K % DPAS_K == 0 and N % DPAS_N == 0, "B dims must tile by 16"
    KBLK, NBLK = K // DPAS_K, N // DPAS_N
    # [kb, kp, two, jb, col] -> [jb, kb, kp, col, two]; the trailing (col,two)
    # pair of fp16 fuses into one little-endian dword (low=even, high=odd K-row).
    t = B.reshape(KBLK, DPAS_K // 2, 2, NBLK, DPAS_N).transpose(3, 0, 1, 4, 2)
    t = np.ascontiguousarray(t)
    return t.view(np.uint32).reshape(-1)


def pack_A_blocked(A: np.ndarray) -> np.ndarray:
    """``A[M, K]`` fp16 -> row-atom-blocked uint32, atoms ordered ``[ra][kb][r][c]``.

    Per (row-atom ``ra``, K-block ``kb``) the 64-dword operand is the RPT(8) x
    DPAS_K(16) half tile ``A[ra*8 + r, kb*16 + c]`` in row-major order -- exactly
    the bytes the baseline produced via ``asub.format<uint32_t>()``.
    """
    M, K = A.shape
    assert M % RPT == 0 and K % DPAS_K == 0, "A dims must tile by (8, 16)"
    MABLK, KBLK = M // RPT, K // DPAS_K
    t = A.reshape(MABLK, RPT, KBLK, DPAS_K).transpose(0, 2, 1, 3)
    t = np.ascontiguousarray(t)
    return t.view(np.uint32).reshape(-1)


def _depack_B_vnni(packed: np.ndarray, K: int, N: int) -> np.ndarray:
    """Inverse of :func:`pack_B_vnni` (used only by ``--self-test``)."""
    KBLK, NBLK = K // DPAS_K, N // DPAS_N
    d = packed.view(np.float16).reshape(NBLK, KBLK, DPAS_K // 2, DPAS_N, 2)
    # invert transpose(3,0,1,4,2): [jb,kb,kp,col,two] -> [kb,kp,two,jb,col]
    return d.transpose(1, 2, 4, 0, 3).reshape(K, N)


def _depack_A_blocked(packed: np.ndarray, M: int, K: int) -> np.ndarray:
    """Inverse of :func:`pack_A_blocked` (used only by ``--self-test``)."""
    MABLK, KBLK = M // RPT, K // DPAS_K
    d = packed.view(np.float16).reshape(MABLK, KBLK, RPT, DPAS_K)
    return d.transpose(0, 2, 1, 3).reshape(M, K)


def self_test() -> int:
    """Round-trip the offline packers on the CPU -- no GPU / driver needed."""
    rng = np.random.default_rng(0)
    M, K, N = 24, 32, 48  # multiples of (8, 16, 16); odd enough to catch strides
    A = rng.standard_normal((M, K)).astype(np.float16)
    B = rng.standard_normal((K, N)).astype(np.float16)

    okA = np.array_equal(_depack_A_blocked(pack_A_blocked(A), M, K), A)
    okB = np.array_equal(_depack_B_vnni(pack_B_vnni(B), K, N), B)
    print(f"pack_A_blocked round-trip : {'OK' if okA else 'FAIL'}")
    print(f"pack_B_vnni   round-trip  : {'OK' if okB else 'FAIL'}")

    # Bonus: confirm sizes match the DPAS operand counts the kernel will load.
    assert pack_B_vnni(B).size == (N // DPAS_N) * (K // DPAS_K) * (DPAS_K // 2) * DPAS_N
    assert pack_A_blocked(A).size == (M // RPT) * (K // DPAS_K) * RPT * (DPAS_K // 2)
    print("operand sizes             : OK")
    return 0 if (okA and okB) else 1


# ===========================================================================
# Kernel generation
# ===========================================================================
# Token-substituted templates (no brace-escaping headaches). The load snippets
# use the BLOCK/DPAS macros, so they are independent of the concrete tile size.
_B_LOAD_OFFLINE = """    // B operands: pre-VNNI-packed on the host. The DPAS Src1 is 128 dwords,
    // but one LSC cm_load tops out at 64 dwords, so fill it with two loads.
    #pragma unroll
    for (int nb = 0; nb < NB; nb++) {
      const int cb = tn / DPAS_N + nb;
      const int bo = (cb * KBLK + kb) * ((DPAS_K/2)*DPAS_N);  // dword offset
      bpk[nb].select<64, 1>(0)  = cm_load<uint32_t, 64>(surfB, (bo +  0) * sizeof(uint32_t));
      bpk[nb].select<64, 1>(64) = cm_load<uint32_t, 64>(surfB, (bo + 64) * sizeof(uint32_t));
    }"""

_A_LOAD_OFFLINE = """    // A operands: pre-blocked on the host -> one contiguous load per atom.
    #pragma unroll
    for (int mb = 0; mb < MB; mb++) {
      const int ra = tm / RPT + mb;
      apk[mb] = cm_load<uint32_t, (RPT*DPAS_K/2)>(
          surfA, ((ra * KBLK + kb) * (RPT*DPAS_K/2)) * sizeof(uint32_t));
    }"""

_A_LOAD_NATURAL = """    // A operands: natural [M,K] row-major load + in-register repack (baseline).
    matrix<half, BLOCK_M, DPAS_K> a_nat;
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++)
      a_nat.row(i) = cm_load<uint32_t, DPAS_K/2>(
          surfA, ((tm + i) * K + k0) * sizeof(half)).format<half>();
    #pragma unroll
    for (int mb = 0; mb < MB; mb++) {
      matrix<half, RPT, DPAS_K> asub;
      #pragma unroll
      for (int r = 0; r < RPT; r++)
        asub.row(r) = a_nat.row(mb * RPT + r);
      apk[mb] = asub.format<uint32_t>();
    }"""

_K_DEFINES = """#include <cm/cm.h>
#define BLOCK_M __BM__
#define BLOCK_N __BN__
#define BLOCK_K 16
#define GROUP_M __GM__
#define GROUP_N __GN__
#define RPT 8
#define SD  8
#define DPAS_N 16
#define DPAS_K 16
#define MB (BLOCK_M / RPT)
#define NB (BLOCK_N / DPAS_N)
"""

_K_STORE = """  // Store: convert float acc -> half, write packed (D is natural [M,N]).
  #pragma unroll
  for (int mb = 0; mb < MB; mb++)
    #pragma unroll
    for (int nb = 0; nb < NB; nb++) {
      matrix_ref<float, RPT, DPAS_N> accm = acc[mb][nb].format<float, RPT, DPAS_N>();
      #pragma unroll
      for (int r = 0; r < RPT; r++) {
        vector<half, DPAS_N> outr = accm.row(r);
        int row = tm + mb * RPT + r;
        int col = tn + nb * DPAS_N;
        cm_store<uint32_t, DPAS_N/2>(surfD, (row * N + col) * sizeof(half),
                                     outr.format<uint32_t>());
      }
    }
}
"""

_K_BODY = """
extern "C" _GENX_MAIN_ void
cm_gemm(SurfaceIndex surfA [[type("buffer_t")]],
        SurfaceIndex surfB [[type("buffer_t")]],
        SurfaceIndex surfD [[type("buffer_t")]],
        int M, int N, int K) {
  const int tm = cm_global_id(0) * BLOCK_M;
  const int tn = cm_global_id(1) * BLOCK_N;
  const int KBLK = K / DPAS_K;

  vector<float, RPT * DPAS_N> acc[MB][NB];
  #pragma unroll
  for (int mb = 0; mb < MB; mb++)
    #pragma unroll
    for (int nb = 0; nb < NB; nb++)
      acc[mb][nb] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += DPAS_K) {
    const int kb = k0 / DPAS_K;

    vector<uint32_t, (DPAS_K/2)*DPAS_N> bpk[NB];
__BLOAD__

    vector<uint32_t, RPT*DPAS_K/2> apk[MB];
__ALOAD__

    #pragma unroll
    for (int nb = 0; nb < NB; nb++)
      #pragma unroll
      for (int mb = 0; mb < MB; mb++)
        acc[mb][nb] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SD, RPT>(
            acc[mb][nb], bpk[nb], apk[mb]);
  }

"""

_K_BODY_PREFETCH_HEAD = """
extern "C" _GENX_MAIN_ void
cm_gemm(SurfaceIndex surfA [[type("buffer_t")]],
        SurfaceIndex surfB [[type("buffer_t")]],
        SurfaceIndex surfD [[type("buffer_t")]],
        int M, int N, int K) {
  const int tm = cm_global_id(0) * BLOCK_M;
  const int tn = cm_global_id(1) * BLOCK_N;
  const int KBLK = K / DPAS_K;

  vector<float, RPT * DPAS_N> acc[MB][NB];
  #pragma unroll
  for (int mb = 0; mb < MB; mb++)
    #pragma unroll
    for (int nb = 0; nb < NB; nb++)
      acc[mb][nb] = 0.0f;

"""


def _emit_B_load(dst: str, kexpr: str, ind: str) -> str:
    return (
        f"{ind}#pragma unroll\n"
        f"{ind}for (int nb = 0; nb < NB; nb++) {{\n"
        f"{ind}  const int cb = tn / DPAS_N + nb;\n"
        f"{ind}  const int bo = (cb * KBLK + ({kexpr})) * ((DPAS_K/2)*DPAS_N);\n"
        f"{ind}  {dst}[nb].select<64, 1>(0)  = cm_load<uint32_t, 64>(surfB, (bo +  0) * sizeof(uint32_t));\n"
        f"{ind}  {dst}[nb].select<64, 1>(64) = cm_load<uint32_t, 64>(surfB, (bo + 64) * sizeof(uint32_t));\n"
        f"{ind}}}\n"
    )


def _emit_A_load(dst: str, kexpr: str, ind: str) -> str:
    return (
        f"{ind}#pragma unroll\n"
        f"{ind}for (int mb = 0; mb < MB; mb++) {{\n"
        f"{ind}  const int ra = tm / RPT + mb;\n"
        f"{ind}  {dst}[mb] = cm_load<uint32_t, (RPT*DPAS_K/2)>(\n"
        f"{ind}      surfA, ((ra * KBLK + ({kexpr})) * (RPT*DPAS_K/2)) * sizeof(uint32_t));\n"
        f"{ind}}}\n"
    )


def _emit_dpas(bsrc: str, asrc: str, ind: str) -> str:
    return (
        f"{ind}#pragma unroll\n"
        f"{ind}for (int nb = 0; nb < NB; nb++)\n"
        f"{ind}  #pragma unroll\n"
        f"{ind}  for (int mb = 0; mb < MB; mb++)\n"
        f"{ind}    acc[mb][nb] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SD, RPT>(\n"
        f"{ind}        acc[mb][nb], {bsrc}[nb], {asrc}[mb]);\n"
    )


def _build_pipeline_body(pa: bool, pb: bool) -> str:
    """Generate a K-loop software-pipelined over whichever operands are prefetched.

    K is unrolled by 2 with EXPLICITLY-NAMED operand buffers (set 0 = even
    k-blocks, set 1 = odd) -- a runtime-indexed register array would spill to
    stack. ``pa``/``pb`` independently choose to prefetch A / B one k-block
    ahead (hiding its load latency behind DPAS) or load it synchronously. The
    single-sided variants (pa^pb) exist so the 32x64 best-tile can pipeline the
    latency-dominant B load without the full double-buffer footprint that
    overflows the 256-GRF file.
    """
    s = [_K_BODY_PREFETCH_HEAD]
    # operand buffer declarations
    if pb:
        s.append("  vector<uint32_t, (DPAS_K/2)*DPAS_N> b0[NB], b1[NB];\n")
    else:
        s.append("  vector<uint32_t, (DPAS_K/2)*DPAS_N> bcur[NB];\n")
    if pa:
        s.append("  vector<uint32_t, RPT*DPAS_K/2>      a0[MB], a1[MB];\n")
    else:
        s.append("  vector<uint32_t, RPT*DPAS_K/2>      acur[MB];\n")
    s.append("\n  // preload k-block 0 for the prefetched operand(s) into set 0\n")
    if pb:
        s.append(_emit_B_load("b0", "0", "  "))
    if pa:
        s.append(_emit_A_load("a0", "0", "  "))

    s.append("\n  for (int kb = 0; kb < KBLK; kb += 2) {\n")
    # ---- even half: process block kb ----
    if pb:
        s.append("    if (kb + 1 < KBLK) {  // prefetch B(kb+1) into set 1\n")
        s.append(_emit_B_load("b1", "kb+1", "      "))
        s.append("    }\n")
    if pa:
        s.append("    if (kb + 1 < KBLK) {  // prefetch A(kb+1) into set 1\n")
        s.append(_emit_A_load("a1", "kb+1", "      "))
        s.append("    }\n")
    if not pb:
        s.append("    // synchronous B load for block kb\n")
        s.append(_emit_B_load("bcur", "kb", "    "))
    if not pa:
        s.append("    // synchronous A load for block kb\n")
        s.append(_emit_A_load("acur", "kb", "    "))
    s.append("    // DPAS on block kb\n")
    s.append(_emit_dpas("b0" if pb else "bcur", "a0" if pa else "acur", "    "))

    # ---- odd half: process block kb+1 ----
    s.append("\n    if (kb + 1 < KBLK) {\n")
    if pb:
        s.append("      if (kb + 2 < KBLK) {  // prefetch B(kb+2) into set 0\n")
        s.append(_emit_B_load("b0", "kb+2", "        "))
        s.append("      }\n")
    if pa:
        s.append("      if (kb + 2 < KBLK) {  // prefetch A(kb+2) into set 0\n")
        s.append(_emit_A_load("a0", "kb+2", "        "))
        s.append("      }\n")
    if not pb:
        s.append("      // synchronous B load for block kb+1\n")
        s.append(_emit_B_load("bcur", "kb+1", "      "))
    if not pa:
        s.append("      // synchronous A load for block kb+1\n")
        s.append(_emit_A_load("acur", "kb+1", "      "))
    s.append("      // DPAS on block kb+1\n")
    s.append(_emit_dpas("b1" if pb else "bcur", "a1" if pa else "acur", "      "))
    s.append("    }\n")

    s.append("  }\n\n")
    return "".join(s)


def gen_kernel(bm: int, bn: int, *, offline_A: bool, offline_B: bool,
               prefetch: bool = False, prefetch_a: bool | None = None,
               prefetch_b: bool | None = None, gm: int = 1, gn: int = 1) -> str:
    """Emit a CM GEMM variant for the given tile + offline-pack / pipeline flags.

    ``prefetch=True`` is shorthand for prefetching BOTH operands; ``prefetch_a``
    / ``prefetch_b`` override that to pipeline a single operand (the rest load
    synchronously). Any pipelined variant requires offline-packed A and B.
    """
    pa = prefetch if prefetch_a is None else prefetch_a
    pb = prefetch if prefetch_b is None else prefetch_b
    if pa or pb:
        assert offline_A and offline_B, "pipelined variant assumes offline A and B"
        body = _build_pipeline_body(pa, pb)
    else:
        bload = _B_LOAD_OFFLINE if offline_B else None
        aload = _A_LOAD_OFFLINE if offline_A else _A_LOAD_NATURAL
        assert bload is not None, "natural-B path is only the verbatim baseline"
        body = _K_BODY.replace("__BLOAD__", bload).replace("__ALOAD__", aload)
    src = _K_DEFINES + body + _K_STORE
    return (src.replace("__BM__", str(bm)).replace("__BN__", str(bn))
            .replace("__GM__", str(gm)).replace("__GN__", str(gn)))


# ===========================================================================
# Stage definitions -- the "story"
# ===========================================================================
class Stage:
    def __init__(self, key, title, story, *, source=None, gen=None,
                 offline_A=False, offline_B=False, regfile=128, ref=None):
        self.key = key
        self.title = title
        self.story = story
        self._source = source      # explicit source (stage 0 reads the seed)
        self._gen = gen            # or a zero-arg callable returning source
        self.offline_A = offline_A
        self.offline_B = offline_B
        self.regfile = regfile
        # key of the stage this one builds on, for an honest step-speedup. None
        # => the previous stage (a linear story). Exploratory BRANCH stages set
        # this so their step compares to their real parent, not whatever probe
        # happened to run just before them.
        self.ref = ref

    def source(self) -> str:
        return self._source if self._source is not None else self._gen()


def build_stages(baseline_src: str) -> list[Stage]:
    return [
        Stage(
            "00_baseline", "Baseline (XeForge optimized)",
            "The XeForge-optimized seed, verbatim: 16x32 tile, in-kernel VNNI "
            "shuffle of B every k-step, natural A/B layout, 128-GRF.",
            source=baseline_src, offline_A=False, offline_B=False, regfile=128,
        ),
        Stage(
            "01_offlineB", "Offline VNNI-pack B",
            "B is pre-VNNI-packed on the HOST into the DPAS Src1 byte layout, so "
            "the kernel drops the per-k-step shuffle loop and loads each B atom "
            "as one contiguous 128-dword message. A still loaded the baseline way. "
            "Same 16x32 tile / 128-GRF -> isolates the VNNI-removal win.",
            gen=lambda: gen_kernel(16, 32, offline_A=False, offline_B=True),
            offline_A=False, offline_B=True, regfile=128,
        ),
        Stage(
            "02_offlineAB", "Offline-pack A too",
            "A is also pre-blocked on the host so each A atom is a single "
            "contiguous 64-dword load (no row-by-row gather + reformat). "
            "Both operands now stream straight into DPAS. 16x32 / 128-GRF.",
            gen=lambda: gen_kernel(16, 32, offline_A=True, offline_B=True),
            offline_A=True, offline_B=True, regfile=128,
        ),
        Stage(
            "03_tile16x64", "Wider N tile (16x64) + 256-GRF",
            "Grow the per-thread tile to 16x64 (NB 2->4) so each loaded A atom "
            "feeds 4 column atoms -> 2x the A-operand reuse. Switch to 256-GRF "
            "so the extra accumulators/operands fit. Offline A+B keep loads cheap "
            "(the win that big tiles couldn't get with the in-kernel shuffle).",
            gen=lambda: gen_kernel(16, 64, offline_A=True, offline_B=True),
            offline_A=True, offline_B=True, regfile=256,
        ),
        Stage(
            "04_kprefetch", "Double-buffered K pipeline (16x64)",
            "Same 16x64 tile as the previous stage, now SOFTWARE-PIPELINED: unroll "
            "K by 2 with two explicitly-named operand buffers (set 0 = even "
            "k-blocks, set 1 = odd) and prefetch one k-block ahead so the LSC "
            "loads overlap DPAS compute. Named buffers matter -- a runtime-indexed "
            "register array (buf[cur]) spills operands to stack and tanks "
            "throughput. Compare directly against stage 3 to read the pipeline win.",
            gen=lambda: gen_kernel(16, 64, offline_A=True, offline_B=True,
                                   prefetch=True),
            offline_A=True, offline_B=True, regfile=256,
        ),
        Stage(
            "05_tile32x64", "Square-ish tile (32x64) + 256-GRF",
            "Grow to 32x64 (MB 4 x NB 4 = 16 atoms/thread). Each A atom feeds 4 "
            "N-atoms and each B atom feeds 4 M-atoms (reuse 16/8 = 2.0), "
            "maximising operand reuse per memory load -- the single biggest reuse "
            "lever so far. 256-GRF. (32x64 is left un-pipelined: double-buffering "
            "it overflows the GRF file.)",
            gen=lambda: gen_kernel(32, 64, offline_A=True, offline_B=True),
            offline_A=True, offline_B=True, regfile=256,
        ),
        Stage(
            "06_tile32x128", "Wider tile 32x128 (push reuse) + 256-GRF",
            "Push the reuse ratio further: 32x128 = MB 4 x NB 8 = 32 DPAS "
            "atoms/thread, reuse MB*NB/(MB+NB) = 2.67 vs 2.0 at 32x64 (+33%). "
            "But 32 fp32 accumulators = 32*512 B = 16 KB, which nearly fills the "
            "256-GRF file by itself -- this stage probes whether the extra reuse "
            "beats the register pressure or spills. N-heavy load mix (8 B-atoms, "
            "4 A-atoms per k-step).",
            gen=lambda: gen_kernel(32, 128, offline_A=True, offline_B=True),
            offline_A=True, offline_B=True, regfile=256, ref="05_tile32x64",
        ),
        Stage(
            "07_tile64x64", "Tall tile 64x64 (push reuse) + 256-GRF",
            "Same 2.67 reuse and same 16 KB accumulator footprint as 32x128, but "
            "SQUARE (MB 8 x NB 4 = 32 atoms): balances the load-message mix (4 "
            "B-atoms + 8 A-atoms per k-step) instead of being B-heavy. Compare "
            "against 32x128 to see whether load-mix shape matters once both sit "
            "at the same reuse / register pressure. This brackets the GRF wall: "
            "if both regress vs 32x64, 16 atoms is the output-tile ceiling.",
            gen=lambda: gen_kernel(64, 64, offline_A=True, offline_B=True),
            offline_A=True, offline_B=True, regfile=256, ref="05_tile32x64",
        ),
        Stage(
            "08_32x64_pfB", "32x64 + B-only K pipeline + 256-GRF",
            "Combine the two proven wins: the 32x64 best tile AND K-pipelining "
            "(which gave +18% at 16x64). Full double-buffering 32x64 overflows "
            "the GRF, so prefetch ONLY B -- the latency-dominant operand (2 load "
            "messages per atom, 2 KB/k-step vs A's 1 KB). One extra B buffer "
            "(~2 KB) is the cheapest way to hide the bigger load behind DPAS.",
            gen=lambda: gen_kernel(32, 64, offline_A=True, offline_B=True,
                                   prefetch_b=True),
            offline_A=True, offline_B=True, regfile=256, ref="05_tile32x64",
        ),
        Stage(
            "09_32x64_pfA", "32x64 + A-only K pipeline + 256-GRF",
            "The other single-sided pipeline: prefetch only A (smaller, +1 KB "
            "buffer) while B loads synchronously. Lower GRF pressure than B-pf so "
            "it definitely fits, but hides only the smaller A latency. Brackets "
            "08: comparing A-pf vs B-pf vs un-pipelined 32x64 isolates WHICH "
            "operand's latency actually stalls the DPAS array.",
            gen=lambda: gen_kernel(32, 64, offline_A=True, offline_B=True,
                                   prefetch_a=True),
            offline_A=True, offline_B=True, regfile=256, ref="05_tile32x64",
        ),
        Stage(
            "10_16x64_occ", "16x64 pipeline @ 128-GRF (occupancy)",
            "Occupancy play: the winning levers all force 256-GRF, which halves "
            "the threads resident per Xe-core. Re-run the 16x64 full pipeline at "
            "128-GRF -- 8 atoms = 4 KB acc may fit the smaller file, doubling "
            "occupancy. Tests whether thread-level latency hiding (more threads) "
            "beats register-tile ILP (fewer, fatter threads) on this device.",
            gen=lambda: gen_kernel(16, 64, offline_A=True, offline_B=True,
                                   prefetch=True),
            offline_A=True, offline_B=True, regfile=128, ref="04_kprefetch",
        ),
    ]


# ===========================================================================
# Benchmark one stage
# ===========================================================================
class Result:
    def __init__(self, ok, status, min_ms=None, median_ms=None, tflops=None,
                 cos=None, rel=None, note=""):
        self.ok = ok
        self.status = status
        self.min_ms = min_ms
        self.median_ms = median_ms
        self.tflops = tflops
        self.cos = cos
        self.rel = rel
        self.note = note


def _grid_from_defines(defines, M, N, K):
    BLOCK_M = defines["BLOCK_M"]
    BLOCK_N = defines["BLOCK_N"]
    BLOCK_K = defines.get("BLOCK_K", 1)
    GROUP_M = defines.get("GROUP_M", 1)
    GROUP_N = defines.get("GROUP_N", 1)
    TILE_M, TILE_N = BLOCK_M * GROUP_M, BLOCK_N * GROUP_N
    if M % TILE_M or N % TILE_N or K % BLOCK_K:
        raise ValueError(
            f"problem {M}x{N}x{K} not divisible by tile "
            f"({TILE_M}x{TILE_N}, K%{BLOCK_K})")
    gsize = ((M // TILE_M) * GROUP_M, (N // TILE_N) * GROUP_N)
    lsize = (GROUP_M, GROUP_N)
    return gsize, lsize, (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, GROUP_N)


def bench_stage(cl, ctx, stage: Stage, src: str, *, M, N, K,
                A, B, ref, A_packed, B_packed, iters, warmup,
                jit_target) -> Result:
    import pyopencl  # noqa: F401  (cl is the module, passed in)

    opts = f"-cmc -Qxcm_register_file_size={stage.regfile}"
    if jit_target:
        opts += f" -Qxcm_jit_target={jit_target}"

    try:
        prg = cl.Program(ctx, src).build(options=opts)
    except cl.RuntimeError as e:
        # Surface the first real compiler 'error:' line, not the trailing
        # options string that pyopencl appends to the exception text.
        errs = [ln.strip() for ln in str(e).splitlines() if "error:" in ln]
        note = errs[0] if errs else str(e).strip().splitlines()[-1]
        return Result(False, "BUILD-FAIL", note=note)

    if "cm_gemm" not in [k.function_name for k in prg.all_kernels()]:
        return Result(False, "NO-ENTRY", note="no cm_gemm entry")

    defines = extract_defines(src)
    try:
        gsize, lsize, _ = _grid_from_defines(defines, M, N, K)
    except ValueError as e:
        return Result(False, "GRID-FAIL", note=str(e))

    mf = cl.mem_flags
    a_host = A_packed if stage.offline_A else A
    b_host = B_packed if stage.offline_B else B
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_host)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_host)
    d_g = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * np.dtype(np.float16).itemsize)
    kargs = (a_g, b_g, d_g, np.int32(M), np.int32(N), np.int32(K))

    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    kernel = cl.Kernel(prg, "cm_gemm")

    try:
        for _ in range(max(0, warmup)):
            kernel(queue, gsize, lsize, *kargs)
        queue.finish()
        events = [kernel(queue, gsize, lsize, *kargs) for _ in range(max(1, iters))]
        queue.finish()
    except cl.RuntimeError as e:
        return Result(False, "RUN-FAIL", note=str(e).strip().splitlines()[-1])

    per_ms = sorted((e.profile.end - e.profile.start) * 1e-6 for e in events)
    min_ms = per_ms[0]
    median_ms = per_ms[len(per_ms) // 2]
    tflops = (2.0 * M * N * K) / (min_ms * 1e-3) / 1e12

    d = np.empty((M, N), np.float16)
    cl.enqueue_copy(queue, d, d_g)
    queue.finish()

    dv = d.astype(np.float32).ravel()
    rv = ref.ravel()
    cos = float(dv @ rv) / (float(np.linalg.norm(dv) * np.linalg.norm(rv)) + 1e-12)
    rel = float(np.max(np.abs(dv - rv))) / (float(np.max(np.abs(rv))) + 1e-12)
    ok = cos > 0.999
    return Result(ok, "OK" if ok else "WRONG", min_ms, median_ms, tflops,
                  cos, rel, "" if ok else f"cos={cos:.4f}")


# ===========================================================================
# Stamped kernel output
# ===========================================================================
def stamp_header(stage: Stage, res: Result, idx: int, step_x, cum_x,
                 M, N, K) -> str:
    bar = "// " + "=" * 72
    lines = [
        bar,
        f"//  Xe-Forge step-by-step GEMM optimization  --  STAGE {idx}: {stage.title}",
        bar,
        f"//  problem      : {M}x{N}x{K}  (fp16 A/B, fp16 D)",
        f"//  offline pack : A={'yes' if stage.offline_A else 'no'}  "
        f"B={'yes' if stage.offline_B else 'no'}    GRF={stage.regfile}",
        "//",
    ]
    for chunk in _wrap(stage.story, 70):
        lines.append(f"//  {chunk}")
    lines.append("//")
    if res.ok:
        lines += [
            f"//  measured     : {res.tflops:.2f} TFLOPS   min={res.min_ms:.3f} ms"
            f"   cos={res.cos:.5f}",
            f"//  step speedup : {step_x}      cumulative vs stage 0: {cum_x}",
        ]
    else:
        lines.append(f"//  status       : {res.status}  ({res.note})")
    lines.append(bar)
    return "\n".join(lines) + "\n\n"


def _wrap(text: str, width: int) -> list[str]:
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


# ===========================================================================
# Main
# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default=DEFAULT_BASELINE,
                    help=f"stage-0 seed kernel (default {DEFAULT_BASELINE})")
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--jit-target", default="BMG",
                    help="-Qxcm_jit_target value (default BMG)")
    ap.add_argument("--peak-tflops", type=float, default=150.0,
                    help="compute roof for the printed %% of peak (default 150)")
    ap.add_argument("--peak-bandwidth", type=float, default=400.0,
                    help="DRAM roof GB/s, for the roofline CSV (default 400)")
    ap.add_argument("--self-test", action="store_true",
                    help="validate the offline packers on the CPU and exit")
    ap.add_argument("--outdir", default="outputs/kernels/steps")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    M, N, K = args.m, args.n, args.k
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    baseline_path = pathlib.Path(args.baseline)
    if not baseline_path.is_file():
        print(f"[fail] baseline not found: {baseline_path}")
        return 1
    baseline_src = baseline_path.read_text()

    # ---- host tensors + golden (computed ONCE; reused for every stage) ----
    print(f"[setup] building A/B and fp32 golden for {M}x{N}x{K} ...")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float16)
    B = rng.standard_normal((K, N)).astype(np.float16)
    ref = (A.astype(np.float32) @ B.astype(np.float32))  # fp32 golden, [M,N]
    A_packed = pack_A_blocked(A)
    B_packed = pack_B_vnni(B)

    # ---- device ----
    import pyopencl as cl
    # The CM VC frontend emits a benign 'stateful buffer not supported' warning
    # on every build; silence pyopencl's CompilerWarning so the story stays readable.
    try:
        import warnings
        warnings.simplefilter("ignore", cl.CompilerWarning)
    except Exception:
        pass
    platform, device = pick_device()
    print(f"[device] {platform.name} | {device.name} ({_type_label(device)})")
    print(f"[roof]   {args.peak_tflops:g} TFLOPS / {args.peak_bandwidth:g} GB/s")
    ctx = cl.Context(devices=[device])

    stages = build_stages(baseline_src)
    outdir = pathlib.Path(args.outdir) / ts
    outdir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[Stage, Result]] = []
    by_key: dict[str, Result] = {}   # stage.key -> Result, for ref lookups
    prev_ok_tflops = None   # last successful stage -> linear step reference
    base_tflops = None      # stage-0 -> cumulative speedup reference

    flop = 2 * M * N * K
    nbytes = (M * K + K * N + M * N) * np.dtype(np.float16).itemsize

    print("\n[run] benchmarking stages (cosine-validated vs numpy golden)\n")
    for idx, stage in enumerate(stages):
        src = stage.source()
        res = bench_stage(cl, ctx, stage, src, M=M, N=N, K=K, A=A, B=B, ref=ref,
                          A_packed=A_packed, B_packed=B_packed,
                          iters=args.iters, warmup=args.warmup,
                          jit_target=args.jit_target)

        step_x = cum_x = "-"
        if res.ok:
            if base_tflops is None:
                base_tflops = res.tflops
            cum_x = f"{res.tflops / base_tflops:.2f}x"
            # step is vs the stage's declared parent (branch stages) or, for the
            # linear story, the previous OK stage.
            ref_res = by_key.get(stage.ref) if stage.ref else None
            ref_tf = ref_res.tflops if (ref_res and ref_res.ok) else prev_ok_tflops
            if ref_tf is not None:
                step_x = f"{res.tflops / ref_tf:.2f}x"
            prev_ok_tflops = res.tflops
        by_key[stage.key] = res

        # stamped kernel
        stamped = stamp_header(stage, res, idx, step_x, cum_x, M, N, K) + src
        (outdir / f"{stage.key}.cpp").write_text(stamped)

        status = res.status
        tag = f" (vs {stage.ref})" if stage.ref else ""
        if res.ok:
            pct = 100.0 * res.tflops / args.peak_tflops
            print(f"  [{idx}] {stage.title:<36} {res.tflops:6.2f} TF  "
                  f"{res.min_ms:7.3f} ms  step {step_x:>6}  cum {cum_x:>6}  "
                  f"{pct:4.1f}% peak  cos={res.cos:.4f}{tag}")
        else:
            print(f"  [{idx}] {stage.title:<36} {status:<10} {res.note[:48]}{tag}")

        results.append((stage, res))

    _print_story(results, base_tflops, args.peak_tflops)
    _write_csvs(results, ts, M, N, K, flop, nbytes, args)
    print(f"\n[out] stamped kernels  -> {outdir}")
    return 0


def _print_story(results, base_tflops, peak_tflops):
    by_key = {st.key: rs for st, rs in results}
    print("\n" + "=" * 70)
    print("STORY  (which step moved the needle)")
    print("=" * 70)
    best = None          # (title, tflops, idx) of the fastest stage overall
    best_step = None     # (title, gain, idx) biggest MAIN-LINE jump (ref=None)
    prev = None          # previous OK stage tflops (linear fallback)
    for idx, (stage, res) in enumerate(results):
        if not res.ok:
            note = f"FAILED ({res.status})"
            mark = " *" if stage.ref else "  "
            print(f" {mark}{idx}. {stage.title:<36} {note}")
            continue
        # step vs declared parent (branch) or previous OK stage (linear)
        ref_res = by_key.get(stage.ref) if stage.ref else None
        ref_tf = ref_res.tflops if (ref_res and ref_res.ok) else prev
        if ref_tf is not None:
            gain = res.tflops / ref_tf - 1.0
            tag = f"{gain * 100:+5.1f}%"
            # only the linear main line (ref=None) competes for "biggest step";
            # branch experiments are explorations off the main line.
            if not stage.ref and (best_step is None or gain > best_step[1]):
                best_step = (stage.title, gain, idx)
        else:
            tag = "  base"
        mark = " *" if stage.ref else "  "
        suffix = f"   (vs {stage.ref})" if stage.ref else ""
        print(f" {mark}{idx}. {stage.title:<36} {res.tflops:6.2f} TF   "
              f"{tag}{suffix}")
        if best is None or res.tflops > best[1]:
            best = (stage.title, res.tflops, idx)
        prev = res.tflops

    print("-" * 70)
    print("  ( * = exploratory branch off an earlier stage, not the main line)")
    if base_tflops and best:
        print(f"  BEST: stage {best[2]} '{best[0]}'  {best[1]:.2f} TFLOPS "
              f"({best[1] / base_tflops:.2f}x over stage 0, "
              f"{100 * best[1] / peak_tflops:.1f}% of {peak_tflops:g} TF roof)")
    if best_step:
        print(f"  biggest main-line step: '{best_step[0]}' "
              f"(+{best_step[1] * 100:.1f}%, stage {best_step[2]})")
    print("=" * 70)


def _write_csvs(results, ts, M, N, K, flop, nbytes, args):
    logs = pathlib.Path("outputs/logs")
    logs.mkdir(parents=True, exist_ok=True)

    story_csv = logs / f"optimize_story_{ts}.csv"
    with open(story_csv, "w", newline="") as f:
        f.write("idx,key,title,status,min_ms,median_ms,tflops,cos,rel,"
                "regfile,offline_A,offline_B\n")
        for idx, (stage, res) in enumerate(results):
            f.write(
                f"{idx},{stage.key},\"{stage.title}\",{res.status},"
                f"{_n(res.min_ms)},{_n(res.median_ms)},{_n(res.tflops)},"
                f"{_n(res.cos)},{_n(res.rel)},{stage.regfile},"
                f"{int(stage.offline_A)},{int(stage.offline_B)}\n")

    # roofline-ready CSV: one row per OK stage, linkable into scripts/roofline.py
    roof_csv = logs / f"optimize_story_{ts}_roofline.csv"
    with open(roof_csv, "w", newline="") as f:
        f.write("series,pair,label,tflops,flop,time_us,bytes\n")
        for idx, (stage, res) in enumerate(results):
            if not res.ok:
                continue
            series = "original" if idx == 0 else "optimized"
            label = f"{idx}:{stage.title}"
            f.write(f"{series},gemm,\"{label}\",{res.tflops:.4f},{flop},"
                    f"{res.min_ms * 1000:.3f},{nbytes}\n")

    print(f"\n[out] story CSV        -> {story_csv}")
    print(f"[out] roofline CSV     -> {roof_csv}")
    print(f"      plot it: python scripts/roofline.py {roof_csv} "
          f"--peak-tflops {args.peak_tflops:g} "
          f"--peak-bandwidth {args.peak_bandwidth:g} --connect --annotate "
          f"-o plots/gemm_steps.png")


def _n(x):
    return "" if x is None else f"{x:.6f}"


if __name__ == "__main__":
    sys.exit(main())
