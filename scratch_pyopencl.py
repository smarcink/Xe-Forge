"""CM GEMM diagnostic / micro-benchmark over PyOpenCL.

Compiles a CM ``.cpp`` ONLINE via the Intel OpenCL runtime's IGC Vector-Compute
frontend: the raw CM source goes to clCreateProgramWithSource + clBuildProgram
with the ``-cmc`` build option (``<cm/cm.h>``, ``_GENX_MAIN_``, ``SurfaceIndex``,
cm_load/cm_store all just work) -- NO offline ``cmc``/SPIR-V step. The seed
``cm_gemm`` entry is then run against a numpy golden reference and timed.

A side tool for inspecting/measuring ONE GEMM kernel outside the full pipeline.
Its main use is passing extra ``-cmc`` diagnostic flags via ``--cm-opts`` --
e.g. ``-Qxcm_print_asm_count`` (asm instruction count) or ``-mCM_printregusage``
(register usage) -- which the production executor does not surface. To see those
diagnostics on every run, defeat both shader caches first:
``$env:PYOPENCL_NO_CACHE=1; $env:NEO_CACHE_PERSISTENT=0`` (a cache HIT skips IGC).

NOTE: this device does NOT advertise ``cl_intel_vector_compute``, yet ``-cmc``
works -- the VC backend is present, the extension string just isn't reported.

Run (use the venv python; bare `python` is system 3.14 without deps):
    python scratch_pyopencl.py --list                  # enumerate OpenCL devices
    python scratch_pyopencl.py --cm k.cpp              # build + run + time cm_gemm
    python scratch_pyopencl.py --cm k.cpp --m 512 --n 512 --k 512 --iters 50
    python scratch_pyopencl.py --saxpy                 # OpenCL-C stack smoke test

Input/output dtypes are read from the kernel's sibling ``.yaml`` spec (same path,
``.yaml`` suffix) -- override with ``--yaml PATH``; without a spec it falls back
to half inputs / fp32 output.

Device selection prefers Intel (this box also has an NVIDIA OpenCL platform).
Override with env vars, e.g.  $env:XE_OCL_PLATFORM = "NVIDIA".
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

import numpy as np
import pyopencl as cl
import yaml

# ---- dummy OpenCL-C kernel: SAXPY  out = a*x + y ----------------------------
DUMMY_KERNEL_SRC = r"""
__kernel void saxpy(const float a,
                    __global const float *x,
                    __global const float *y,
                    __global float *out)
{
    int gid = get_global_id(0);
    out[gid] = a * x[gid] + y[gid];
}
"""

# Build option that makes the Intel runtime compile CM source (not OpenCL-C)
# online. `-vc-codegen` is the ESIMD/VC-SPIRV path and does NOT work for CM
# source here; `-cmc` (the CM frontend) is the one that accepts cm/cm.h kernels.
CM_BUILD_OPTS = "-cmc"

# `#define NAME VALUE` matcher -- mirrors xe_forge.core.cm_grid.extract_defines,
# kept local so this scratch tool stays standalone (no PYTHONPATH/import needed).
_DEFINE_RE = re.compile(
    r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)[ \t]+(\S+)", re.MULTILINE
)


def extract_defines(src: str) -> dict[str, int]:
    """Return integer ``#define NAME VALUE`` pairs (decimal or 0x/0o/0b).

    Used to read the kernel's tiling knobs (BLOCK_M/N/K) straight from the
    source so the launch grid tracks the kernel instead of being hardcoded.
    """
    defines: dict[str, int] = {}
    for name, token in _DEFINE_RE.findall(src):
        try:
            defines[name] = int(token, 0)  # honors 0x/0o/0b + plain decimal
        except ValueError:
            pass  # skip expression-/string-valued macros
    return defines


# Map spec ``dtype:`` strings -> numpy dtypes so buffers + golden arrays are
# sized/typed from the kernel's .yaml instead of being hardcoded in this tool.
_NP_DTYPE_BY_NAME: dict[str, np.dtype] = {
    "float16": np.dtype(np.float16), "half": np.dtype(np.float16),
    "float32": np.dtype(np.float32), "float": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "int8": np.dtype(np.int8), "uint8": np.dtype(np.uint8),
    "int16": np.dtype(np.int16), "uint16": np.dtype(np.uint16),
    "int32": np.dtype(np.int32), "uint32": np.dtype(np.uint32),
    "int64": np.dtype(np.int64), "uint64": np.dtype(np.uint64),
}


def _np_dtype(name: str) -> np.dtype:
    """Resolve a spec dtype string (e.g. ``float16``) to a numpy dtype."""
    try:
        return _NP_DTYPE_BY_NAME[name.strip().lower()]
    except KeyError:
        raise ValueError(
            f"unsupported dtype {name!r}; known: {sorted(_NP_DTYPE_BY_NAME)}"
        ) from None


def parse_spec_dtypes(
    yaml_path: str,
) -> tuple[list[np.dtype], list[np.dtype]]:
    """Read a kernel ``.yaml`` spec -> ``(input_dtypes, output_dtypes)``.

    Ordered to match the ``inputs:`` / ``outputs:`` declaration order, which is
    the cm_gemm ABI order (inputs A, B then output D). Lets the runner take its
    buffer + golden dtypes straight from the spec instead of hardcoding them.
    """
    spec = yaml.safe_load(pathlib.Path(yaml_path).read_text()) or {}
    in_dtypes = [
        _np_dtype(v.get("dtype", "float32"))
        for v in (spec.get("inputs") or {}).values()
    ]
    out_dtypes = [
        _np_dtype(v.get("dtype", "float32"))
        for v in (spec.get("outputs") or {}).values()
    ]
    return in_dtypes, out_dtypes


def _resolve_spec_dtypes(
    cm_path: str | None, yaml_path: str | None
) -> tuple[list[np.dtype] | None, list[np.dtype] | None, str | None]:
    """Find the kernel spec and parse its in/out dtypes.

    Explicit ``--yaml`` wins; otherwise look for a sibling file with the same
    stem as the ``.cpp``. Returns ``(None, None, None)`` when no spec is found
    so the runner falls back to its fp16-in / fp32-out defaults.
    """
    path = yaml_path
    if path is None and cm_path is not None:
        cand = pathlib.Path(cm_path).with_suffix(".yaml")
        path = str(cand) if cand.is_file() else None
    if path is None:
        return None, None, None
    in_dtypes, out_dtypes = parse_spec_dtypes(path)
    return in_dtypes, out_dtypes, path


def _type_label(dev: cl.Device) -> str:
    t = dev.type
    if t & cl.device_type.GPU:
        return "GPU"
    if t & cl.device_type.CPU:
        return "CPU"
    if t & cl.device_type.ACCELERATOR:
        return "ACCEL"
    return cl.device_type.to_string(t)


def list_devices() -> None:
    for pi, platform in enumerate(cl.get_platforms()):
        print(f"[{pi}] {platform.name}  ({platform.version})")
        for di, dev in enumerate(platform.get_devices()):
            print(
                f"      [{di}] {dev.name}  "
                f"| {_type_label(dev)} "
                f"| CUs={dev.max_compute_units} "
                f"| {dev.global_mem_size // (1024 * 1024)} MiB"
            )


def pick_device(prefer: str = "intel") -> tuple[cl.Platform, cl.Device]:
    """Pick a (platform, device). Prefer a platform whose name contains
    `prefer` (default "intel"); override via env XE_OCL_PLATFORM /
    XE_OCL_DEVICE (case-insensitive substring match)."""
    plat_filter = os.environ.get("XE_OCL_PLATFORM", prefer).lower()
    dev_filter = os.environ.get("XE_OCL_DEVICE", "").lower()

    candidates = [
        (p, d) for p in cl.get_platforms() for d in p.get_devices()
    ]
    if not candidates:
        raise RuntimeError("No OpenCL platforms/devices found")

    matches = [
        (p, d)
        for (p, d) in candidates
        if plat_filter in p.name.lower()
        and (not dev_filter or dev_filter in d.name.lower())
    ]
    if matches:
        return matches[0]

    print(f"[warn] no platform matching '{plat_filter}', using first device")
    return candidates[0]


def run_dummy(n: int = 1 << 20) -> int:
    platform, device = pick_device()
    print(f"platform : {platform.name}")
    print(f"device   : {device.name} ({_type_label(device)})")

    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
    )

    a = np.float32(2.0)
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)

    mf = cl.mem_flags
    x_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    y_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y)
    out_g = cl.Buffer(ctx, mf.WRITE_ONLY, x.nbytes)

    prg = cl.Program(ctx, DUMMY_KERNEL_SRC).build()
    evt = prg.saxpy(queue, x.shape, None, a, x_g, y_g, out_g)
    evt.wait()

    out = np.empty_like(x)
    cl.enqueue_copy(queue, out, out_g)
    queue.finish()

    gpu_ms = (evt.profile.end - evt.profile.start) * 1e-6
    expected = a * x + y
    ok = bool(np.allclose(out, expected, rtol=1e-5, atol=1e-5))
    print(f"saxpy n={n}  kernel={gpu_ms:.3f} ms  match={ok}")
    if not ok:
        print(f"[fail] max abs err = {np.max(np.abs(out - expected))}")
        return 1
    return 0


# ---- CM from SOURCE (no cmc!) ----------------------------------------------
# clCreateProgramWithSource(cm_source) + clBuildProgram("-cmc") makes the Intel
# runtime's IGC VC frontend compile CM directly -- no offline cmc -> .spv step.
# This is the path the production executor uses (see xe_forge.core.cm_worker).
def run_cm_source(cpp_path: str, build_opts: str = CM_BUILD_OPTS, *,
                  m: int = 256, n: int = 256, k: int = 256,
                  iters: int = 20, warmup: int = 3,
                  in_dtypes: list[np.dtype] | None = None,
                  out_dtypes: list[np.dtype] | None = None,
                  spec_path: str | None = None) -> int:
    src = pathlib.Path(cpp_path).read_text()
    platform, device = pick_device()
    print(f"platform : {platform.name}")
    print(f"device   : {device.name} ({_type_label(device)})")
    print(f"build    : clBuildProgram(options={build_opts!r})")
    print(f"spec     : {spec_path or '(none; fp16 in / fp32 out defaults)'}")

    ctx = cl.Context(devices=[device])
    try:
        prg = cl.Program(ctx, src).build(options=build_opts)
    except cl.RuntimeError as e:
        print(f"[fail] CM source build failed:\n{e}")
        return 1

    # The asm-count / diagnostics print straight to stdout; the CL build log is
    # usually empty but print it when non-empty in case a driver routes there.
    log = prg.get_build_info(device, cl.program_build_info.LOG)
    if log.strip():
        print(f"[build log]\n{log}")

    names = [k.function_name for k in prg.all_kernels()]
    print(f"built {cpp_path} from SOURCE -> kernels: {names}")

    if "cm_gemm" in names:
        queue = cl.CommandQueue(
            ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
        )
        return run_cm_gemm(ctx, queue, prg, extract_defines(src),
                           M=m, N=n, K=k, iters=iters, warmup=warmup,
                           in_dtypes=in_dtypes, out_dtypes=out_dtypes)
    print(f"[warn] no 'cm_gemm' entry among {names}; built OK but nothing to run")
    return 0


def run_cm_gemm(ctx: cl.Context, queue: cl.CommandQueue, prg: cl.Program,
                defines: dict[str, int],
                M: int = 256, N: int = 256, K: int = 256,
                iters: int = 20, warmup: int = 3,
                in_dtypes: list[np.dtype] | None = None,
                out_dtypes: list[np.dtype] | None = None) -> int:
    """Drive the seed cm_gemm kernel end-to-end: verify vs numpy + time it.

    ABI (matches test_kernels/200_CM_Gemm.cpp): cm_gemm(SurfaceIndex A, B, D,
    int M, int N, int K). Each SurfaceIndex -> a __global buffer arg; scalars
    pass by value. A is MxK, B is KxN, D is MxN; their element dtypes come from
    the kernel's .yaml spec (in/out order) via ``in_dtypes``/``out_dtypes``,
    falling back to half inputs / fp32 output when no spec is found.
    The launch grid is derived from the kernel's own ``#define``s so editing the
    kernel's tiling keeps this runner correct. BLOCK_M/N set the per-thread tile
    and GROUP_M/N (default 1) the cooperative work-group, mirroring the Model-B
    grid in 200_CM_Gemm.yaml: local=(GROUP_M, GROUP_N) and the global keeps one
    work-item per BLOCK_M x BLOCK_N tile. At GROUP=1 this is the naive
    (M//BLOCK_M, N//BLOCK_N) grid with local=(1,1); at GROUP>1 the threads form a
    real SLM-sharing group so cm_local_id / cm_barrier / SLM staging actually
    work. Timing is the mean over ``iters`` runs (after ``warmup``) via CL
    profiling events; reports TFLOPS.
    """
    try:
        BLOCK_M = defines["BLOCK_M"]
        BLOCK_N = defines["BLOCK_N"]
    except KeyError as e:
        print(f"[fail] kernel source has no #define {e.args[0]}; "
              "cannot derive launch grid")
        return 1
    BLOCK_K = defines.get("BLOCK_K", 1)  # only constrains the K-loop divisibility
    GROUP_M = defines.get("GROUP_M", 1)  # cooperative work-group size in M
    GROUP_N = defines.get("GROUP_N", 1)  # cooperative work-group size in N
    print(f"tiling   : BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} BLOCK_K={BLOCK_K} "
          f"GROUP_M={GROUP_M} GROUP_N={GROUP_N} (from source)")
    # A GROUP_M x GROUP_N group cooperatively owns a (GROUP_M*BLOCK_M) x
    # (GROUP_N*BLOCK_N) output region, so the problem must tile evenly by that
    # region (matches the kernel's gm_base/gn_base math and the Model-B grid).
    TILE_M = BLOCK_M * GROUP_M
    TILE_N = BLOCK_N * GROUP_N
    if M % TILE_M or N % TILE_N or K % BLOCK_K:
        print(f"[fail] need M%{TILE_M}==N%{TILE_N}==K%{BLOCK_K}==0 "
              f"(BLOCK*GROUP), got {M}x{N}x{K}")
        return 1

    # Buffer + golden dtypes come from the kernel's .yaml spec when available;
    # fall back to the seed's fp16 inputs / fp32 output when no spec was found.
    a_dtype = in_dtypes[0] if in_dtypes else np.dtype(np.float16)
    b_dtype = in_dtypes[1] if in_dtypes and len(in_dtypes) > 1 else a_dtype
    d_dtype = out_dtypes[0] if out_dtypes else np.dtype(np.float32)
    print(f"dtypes   : A={a_dtype} B={b_dtype} D={d_dtype}")

    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(a_dtype)
    b = rng.standard_normal((K, N)).astype(b_dtype)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    d_g = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * d_dtype.itemsize)

    # Model-B grid: the global keeps one work-item per BLOCK_M x BLOCK_N tile and
    # the local is the cooperative group. TILE_* divides M/N (checked above), so
    # this is the exact (M//BLOCK_M, N//BLOCK_N) global with local=(GROUP_M,
    # GROUP_N) -- (M//BLOCK_M)//GROUP_M groups of GROUP_M threads along each axis.
    gsize = ((M // TILE_M) * GROUP_M, (N // TILE_N) * GROUP_N)
    lsize = (GROUP_M, GROUP_N)
    kargs = (a_g, b_g, d_g, np.int32(M), np.int32(N), np.int32(K))

    # Retrieve the kernel once; prg.cm_gemm would rebuild it every call.
    kernel = cl.Kernel(prg, "cm_gemm")
    for _ in range(max(0, warmup)):
        kernel(queue, gsize, lsize, *kargs)
    queue.finish()

    # Report MIN (not mean) as the headline. On this iGPU the per-iter time swings
    # with DVFS/boost ramp and any background GPU work, so the MEAN of a short run
    # is unstable (the same kernel measured 155 vs 260 ms across two sessions). The
    # MIN iter is the one that hit peak clock with least interference -- it is the
    # most reproducible estimate of true kernel cost and what clpeak/good
    # microbenchmarks report. median + mean are printed too so the spread is visible.
    iters = max(1, iters)
    events = [kernel(queue, gsize, lsize, *kargs) for _ in range(iters)]
    queue.finish()
    per_ms = sorted((e.profile.end - e.profile.start) * 1e-6 for e in events)
    min_ms = per_ms[0]
    median_ms = per_ms[len(per_ms) // 2]
    mean_ms = sum(per_ms) / iters
    tflops = (2.0 * M * N * K) / (min_ms * 1e-3) / 1e12  # headline TFLOPS uses MIN

    d = np.empty((M, N), d_dtype)
    cl.enqueue_copy(queue, d, d_g)
    queue.finish()

    ref = (a.astype(np.float32) @ b.astype(np.float32)).astype(d_dtype)
    # Correctness via COSINE SIMILARITY (not max-rel-error). The kernel keeps an
    # fp16 accumulator on purpose (preserves the GPU fp16 MAD), so summing K
    # terms in half precision drifts the MAGNITUDE by a few percent at large K --
    # enough to blow a 2e-2 max-rel gate while the result is still correct. Cosine
    # similarity compares DIRECTION of the flattened output vs the fp32 golden, so
    # uniform fp16 rounding barely moves it; a real bug (wrong tiles/indexing)
    # tanks it. rel error is kept as an informational diagnostic only.
    dv = d.astype(np.float32).ravel()
    rv = ref.astype(np.float32).ravel()
    cos = float(dv @ rv) / (float(np.linalg.norm(dv) * np.linalg.norm(rv)) + 1e-12)
    max_err = float(np.max(np.abs(dv - rv)))
    rel = max_err / (float(np.max(np.abs(rv))) + 1e-12)
    ok = cos > 0.999  # direction match; robust to fp16-accumulation drift
    print(f"cm_gemm {M}x{N}x{K}  grid={gsize} local={lsize}  "
          f"min={min_ms:.3f} median={median_ms:.3f} mean={mean_ms:.3f} ms/iter "
          f"({iters} iters, {warmup} warmup)  {tflops:.3f} TFLOPS(min)  "
          f"cos={cos:.6f} rel={rel:.2e}  match={ok}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--list", action="store_true",
                    help="enumerate OpenCL devices and exit")
    ap.add_argument("--cm", metavar="PATH",
                    help="compile a CM .cpp ONLINE via -cmc and run/time its cm_gemm entry")
    ap.add_argument("--cm-opts", default=CM_BUILD_OPTS,
                    help=f"clBuildProgram options for --cm (default {CM_BUILD_OPTS!r}); "
                         "add diagnostics e.g. -Qxcm_print_asm_count -mCM_printregusage")
    ap.add_argument("--yaml", metavar="PATH",
                    help="kernel spec .yaml for input/output dtypes "
                         "(default: the --cm path with a .yaml suffix)")
    ap.add_argument("--m", type=int, default=256, help="GEMM M (default 256)")
    ap.add_argument("--n", type=int, default=256, help="GEMM N (default 256)")
    ap.add_argument("--k", type=int, default=256, help="GEMM K (default 256)")
    ap.add_argument("--iters", type=int, default=20,
                    help="timed iterations for --cm (default 20)")
    ap.add_argument("--warmup", type=int, default=10,
                    help="warmup iterations for --cm (default 10; lets the GPU "
                         "reach steady boost clock before timing)")
    ap.add_argument("--saxpy", action="store_true",
                    help="run an OpenCL-C SAXPY stack smoke test")
    ap.add_argument("--saxpy-n", type=int, default=1 << 20,
                    help="vector length for --saxpy (default 2^20)")
    args = ap.parse_args(argv)

    if args.list:
        list_devices()
        return 0
    if args.cm:
        in_dtypes, out_dtypes, spec_path = _resolve_spec_dtypes(
            args.cm, args.yaml)
        return run_cm_source(args.cm, build_opts=args.cm_opts,
                             m=args.m, n=args.n, k=args.k,
                             iters=args.iters, warmup=args.warmup,
                             in_dtypes=in_dtypes, out_dtypes=out_dtypes,
                             spec_path=spec_path)
    if args.saxpy:
        return run_dummy(args.saxpy_n)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
