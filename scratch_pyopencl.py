"""scratch_pyopencl.py -- throwaway PyOpenCL playground (DELETE LATER).

Purpose: prove out PyOpenCL on this machine's Intel GPU before wiring it up to
run CM kernels. Two CM paths are demonstrated:

  1. ONLINE source build (no cmc step!): hand the raw CM `.cpp` to the Intel
     OpenCL runtime via clCreateProgramWithSource + clBuildProgram with the
     `-cmc` build option. The driver's IGC Vector-Compute frontend compiles CM
     directly -- `<cm/cm.h>`, `_GENX_MAIN_`, `SurfaceIndex`, cm_load/cm_store
     all just work. See run_cm_source().  <-- this is the interesting one.
  2. OFFLINE SPIR-V: cmc emits SPIR-V (`cmc k.cpp -o k.spv -emit-spirv
     -mcpu=<plat>`) and PyOpenCL loads it via the 2-arg Program(ctx, bytes)
     form (auto clCreateProgramWithIL). See run_spirv_stub().

NOTE: this device does NOT advertise `cl_intel_vector_compute`, yet BOTH paths
work -- the VC backend is present, the extension string just isn't reported.

This is intentionally scratch -- remove it once the real CM executor exists.

Run (use the venv python; bare `python` is system 3.14 without deps):
    .venv\\Scripts\\python.exe scratch_pyopencl.py            # run dummy SAXPY
    .venv\\Scripts\\python.exe scratch_pyopencl.py --list     # enumerate devices
    .venv\\Scripts\\python.exe scratch_pyopencl.py --cm k.cpp # build CM source online
    .venv\\Scripts\\python.exe scratch_pyopencl.py --spirv k.spv   # peek a .spv

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
# runtime's IGC VC frontend compile CM directly. This is what cm_compiler.run()
# could use to skip the offline cmc -> .spv step entirely.
def run_cm_source(cpp_path: str, build_opts: str = CM_BUILD_OPTS,
                  run: bool = True) -> int:
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

    if run and "cm_gemm" in names:
        queue = cl.CommandQueue(
            ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
        )
        return run_cm_gemm(ctx, queue, prg)
    return 0


def run_cm_gemm(ctx: cl.Context, queue: cl.CommandQueue, prg: cl.Program,
                M: int = 64, N: int = 64, K: int = 64) -> int:
    """Drive the seed cm_gemm kernel end-to-end and verify against numpy.

    ABI (matches test_kernels/200_CM_Gemm.cpp): cm_gemm(SurfaceIndex A, B, D,
    int M, int N, int K). Each SurfaceIndex -> a __global buffer arg; scalars
    pass by value. A is MxK half, B is KxN half, D is MxN float (fp32 accum).
    Grid: tile (BLOCK_M=8) x (BLOCK_N=16); launch one work-item per tile
    (local=(1,1)) so cm_group_id(d) == get_global_id(d).
    """
    BLOCK_M, BLOCK_N = 8, 16
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % 16 == 0

    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    d_g = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)

    gsize = (M // BLOCK_M, N // BLOCK_N)  # cm_group_id(0)->M tiles, (1)->N tiles
    lsize = (1, 1)
    evt = prg.cm_gemm(queue, gsize, lsize, a_g, b_g, d_g,
                      np.int32(M), np.int32(N), np.int32(K))
    evt.wait()

    d = np.empty((M, N), np.float32)
    cl.enqueue_copy(queue, d, d_g)
    queue.finish()

    ref = a.astype(np.float32) @ b.astype(np.float32)
    gpu_ms = (evt.profile.end - evt.profile.start) * 1e-6
    max_err = float(np.max(np.abs(d - ref)))
    rel = max_err / (float(np.max(np.abs(ref))) + 1e-12)
    ok = rel < 2e-2  # fp16 inputs -> loose tolerance
    print(f"cm_gemm {M}x{N}x{K}  grid={gsize} local={lsize}  "
          f"kernel={gpu_ms:.3f} ms  max_err={max_err:.4f} rel={rel:.2e}  match={ok}")
    return 0 if ok else 1


# ---- OFFLINE: load a cmc-emitted .spv (Program auto-detects SPIR-V magic) ---
def run_spirv_stub(spv_path: str) -> int:
    spv = pathlib.Path(spv_path).read_bytes()
    _, device = pick_device()
    ctx = cl.Context(devices=[device])
    prg = cl.Program(ctx, spv).build()
    names = [k.function_name for k in prg.all_kernels()]
    print(f"loaded {spv_path} ({len(spv)} bytes) -> kernels: {names}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true",
                    help="enumerate OpenCL devices and exit")
    ap.add_argument("--n", type=int, default=1 << 20,
                    help="vector length for the dummy SAXPY (default 2^20)")
    ap.add_argument("--cm", metavar="PATH",
                    help="compile a CM .cpp ONLINE via -cmc (no cmc step); "
                         "runs cm_gemm end-to-end if present")
    ap.add_argument("--cm-opts", default=CM_BUILD_OPTS,
                    help=f"clBuildProgram options for --cm (default {CM_BUILD_OPTS!r})")
    ap.add_argument("--spirv", metavar="PATH",
                    help="load a .spv (e.g. a cmc kernel) and list its kernels")
    args = ap.parse_args(argv)

    if args.list:
        list_devices()
        return 0
    if args.cm:
        return run_cm_source(args.cm, build_opts=args.cm_opts)
    if args.spirv:
        return run_spirv_stub(args.spirv)
    return run_dummy(args.n)


if __name__ == "__main__":
    sys.exit(main())
