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

Device selection prefers Intel (this box also has an NVIDIA OpenCL platform).
Override with env vars, e.g.  $env:XE_OCL_PLATFORM = "NVIDIA".
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import pyopencl as cl

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
                  iters: int = 20, warmup: int = 3) -> int:
    src = pathlib.Path(cpp_path).read_text()
    platform, device = pick_device()
    print(f"platform : {platform.name}")
    print(f"device   : {device.name} ({_type_label(device)})")
    print(f"build    : clBuildProgram(options={build_opts!r})")

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
        return run_cm_gemm(ctx, queue, prg, M=m, N=n, K=k, iters=iters, warmup=warmup)
    print(f"[warn] no 'cm_gemm' entry among {names}; built OK but nothing to run")
    return 0


def run_cm_gemm(ctx: cl.Context, queue: cl.CommandQueue, prg: cl.Program,
                M: int = 256, N: int = 256, K: int = 256,
                iters: int = 20, warmup: int = 3) -> int:
    """Drive the seed cm_gemm kernel end-to-end: verify vs numpy + time it.

    ABI (matches test_kernels/200_CM_Gemm.cpp): cm_gemm(SurfaceIndex A, B, D,
    int M, int N, int K). Each SurfaceIndex -> a __global buffer arg; scalars
    pass by value. A is MxK half, B is KxN half, D is MxN float (fp32 accum).
    Grid: tile (BLOCK_M=8) x (BLOCK_N=16); launch one work-item per tile
    (local=(1,1)) so cm_group_id(d) == get_global_id(d). Timing is the mean over
    ``iters`` runs (after ``warmup``) via CL profiling events; reports TFLOPS.
    """
    BLOCK_M, BLOCK_N, BLOCK_K = 8, 16, 16
    if M % BLOCK_M or N % BLOCK_N or K % BLOCK_K:
        print(f"[fail] need M%{BLOCK_M}==N%{BLOCK_N}==K%{BLOCK_K}==0, got {M}x{N}x{K}")
        return 1

    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    d_g = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)

    gsize = (M // BLOCK_M, N // BLOCK_N)  # cm_group_id(0)->M tiles, (1)->N tiles
    lsize = (1, 1)
    kargs = (a_g, b_g, d_g, np.int32(M), np.int32(N), np.int32(K))

    # Retrieve the kernel once; prg.cm_gemm would rebuild it every call.
    kernel = cl.Kernel(prg, "cm_gemm")
    for _ in range(max(0, warmup)):
        kernel(queue, gsize, lsize, *kargs)
    queue.finish()

    iters = max(1, iters)
    events = [kernel(queue, gsize, lsize, *kargs) for _ in range(iters)]
    queue.finish()
    mean_ms = sum(e.profile.end - e.profile.start for e in events) / iters * 1e-6
    tflops = (2.0 * M * N * K) / (mean_ms * 1e-3) / 1e12

    d = np.empty((M, N), np.float32)
    cl.enqueue_copy(queue, d, d_g)
    queue.finish()

    ref = a.astype(np.float32) @ b.astype(np.float32)
    max_err = float(np.max(np.abs(d - ref)))
    rel = max_err / (float(np.max(np.abs(ref))) + 1e-12)
    ok = rel < 2e-2  # fp16 inputs -> loose tolerance
    print(f"cm_gemm {M}x{N}x{K}  grid={gsize} local={lsize}  "
          f"mean={mean_ms:.3f} ms/iter ({iters} iters)  {tflops:.3f} TFLOPS  "
          f"max_err={max_err:.4f} rel={rel:.2e}  match={ok}")
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
    ap.add_argument("--m", type=int, default=256, help="GEMM M (default 256)")
    ap.add_argument("--n", type=int, default=256, help="GEMM N (default 256)")
    ap.add_argument("--k", type=int, default=256, help="GEMM K (default 256)")
    ap.add_argument("--iters", type=int, default=20,
                    help="timed iterations for --cm (default 20)")
    ap.add_argument("--warmup", type=int, default=3,
                    help="warmup iterations for --cm (default 3)")
    ap.add_argument("--saxpy", action="store_true",
                    help="run an OpenCL-C SAXPY stack smoke test")
    ap.add_argument("--saxpy-n", type=int, default=1 << 20,
                    help="vector length for --saxpy (default 2^20)")
    args = ap.parse_args(argv)

    if args.list:
        list_devices()
        return 0
    if args.cm:
        return run_cm_source(args.cm, build_opts=args.cm_opts,
                             m=args.m, n=args.n, k=args.k,
                             iters=args.iters, warmup=args.warmup)
    if args.saxpy:
        return run_dummy(args.saxpy_n)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
