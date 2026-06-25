"""ConvBlock CM diagnostic / micro-benchmark over PyOpenCL.

The Conv counterpart of ``scratch_swin.py`` / ``scratch_gemm.py``: it compiles
the ConvBlock CM ``.cpp`` ONLINE via the Intel OpenCL runtime's IGC
Vector-Compute frontend (``-cmc``, no offline ``cmc``/SPIR-V step), runs it
against a Torch golden reference of the same math, verifies correctness and
times it (min-of-iters), reporting TFLOPS.

The op is the fp16 ConvBlock from ``test_kernels/conv_pool/conv_block_fp16.py``:

    y = AvgPool2d(2)( ReLU( Conv2d(x, 3x3, pad=1, stride=1, bias) ) )       (NCHW)

Buffer order, shapes and dtypes are read straight from the kernel's ``.yaml``
spec (``inputs:`` -> ``outputs:`` -> per-variant ``dims:``), so the ABI tracks
the spec exactly:

    inputs (params order) -> output Y -> scalars (dims order)

matching ``cm_conv_block(X, weight, bias, Y, N,CIN,COUT,H,W,KH,KW,OH,OW)``. The
launch grid is one work-item per pooled output row, padded up to a multiple of
the kernel's tunable ``LWS_X``/``LWS_Y`` work-group dims (read from ``#define``);
the kernel guards the padded tail.

Run (use the venv python; bare ``python`` may be a deps-less interpreter):
    python scratch_conv.py                                   # ci variant, default kernel
    python scratch_conv.py --variant bench-gpu --iters 50    # full 1x128x360x640 workload
    python scratch_conv.py --height 256 --width 256          # override H/W
    python scratch_conv.py --cm path/to/optimized.cpp        # measure an optimized kernel
    python scratch_conv.py --list                            # enumerate OpenCL devices

Pass extra ``-cmc`` diagnostic flags via ``--cm-opts`` (e.g.
``-mCM_printregusage`` / ``-Qxcm_print_asm_count``). To force them to print on
every run, defeat the shader caches first:
``$env:PYOPENCL_NO_CACHE=1; $env:NEO_CACHE_PERSISTENT=0`` (a cache HIT skips IGC).

Device selection prefers Intel; override with env XE_OCL_PLATFORM / XE_OCL_DEVICE.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import re
import sys

# Surface the IGC/Vector-Compute build log (register usage, asm count, warnings)
# instead of swallowing it behind pyopencl's CompilerWarning. Set BEFORE importing
# pyopencl so it takes effect at build time. Override with PYOPENCL_COMPILER_OUTPUT=0.
os.environ.setdefault("PYOPENCL_COMPILER_OUTPUT", "1")

import numpy as np
import pyopencl as cl
import yaml

DEFAULT_CM = "test_kernels/202_CM_conv_block.cpp"
# The Conv kernel uses the STATELESS svmptr_t ABI, so -DCM_PTRSIZE=64 is required.
DEFAULT_CM_OPTS = "-cmc -DCM_PTRSIZE=64"

_DEFINE_RE = re.compile(
    r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)[ \t]+(\S+)", re.MULTILINE
)

_NP_DTYPE_BY_NAME: dict[str, np.dtype] = {
    "float16": np.dtype(np.float16), "half": np.dtype(np.float16),
    "float32": np.dtype(np.float32), "float": np.dtype(np.float32),
}


def extract_defines(src: str) -> dict[str, int]:
    """Return integer ``#define NAME VALUE`` pairs (decimal or 0x/0o/0b)."""
    out: dict[str, int] = {}
    for name, token in _DEFINE_RE.findall(src):
        try:
            out[name] = int(token, 0)
        except ValueError:
            pass
    return out


def _np_dtype(name: str) -> np.dtype:
    try:
        return _NP_DTYPE_BY_NAME[name.strip().lower()]
    except KeyError:
        raise ValueError(f"unsupported dtype {name!r}") from None


def _type_label(dev: cl.Device) -> str:
    t = dev.type
    if t & cl.device_type.GPU:
        return "GPU"
    if t & cl.device_type.CPU:
        return "CPU"
    return cl.device_type.to_string(t)


def list_devices() -> None:
    for pi, platform in enumerate(cl.get_platforms()):
        print(f"[{pi}] {platform.name}  ({platform.version})")
        for di, dev in enumerate(platform.get_devices()):
            print(f"      [{di}] {dev.name}  | {_type_label(dev)} "
                  f"| CUs={dev.max_compute_units} "
                  f"| {dev.global_mem_size // (1024 * 1024)} MiB")


def pick_device(prefer: str = "intel") -> tuple[cl.Platform, cl.Device]:
    """Pick a (platform, device), preferring Intel; override via env
    XE_OCL_PLATFORM / XE_OCL_DEVICE (case-insensitive substring match)."""
    plat_filter = os.environ.get("XE_OCL_PLATFORM", prefer).lower()
    dev_filter = os.environ.get("XE_OCL_DEVICE", "").lower()
    candidates = [(p, d) for p in cl.get_platforms() for d in p.get_devices()]
    if not candidates:
        raise RuntimeError("No OpenCL platforms/devices found")
    matches = [
        (p, d) for (p, d) in candidates
        if plat_filter in p.name.lower() and (not dev_filter or dev_filter in d.name.lower())
    ]
    if matches:
        return matches[0]
    print(f"[warn] no platform matching '{plat_filter}', using first device")
    return candidates[0]


def load_spec(yaml_path: str, variant: str, overrides: dict[str, int]):
    """Parse the kernel spec -> (input_specs, output_spec, dims, flop_formula).

    ``input_specs`` is an ordered list of (name, shape_symbols, np_dtype) in the
    variant's ``params`` (ABI) order; ``dims`` is the resolved dim map with CLI
    overrides applied (insertion order = scalar-arg order). H/W overrides also
    refresh the derived OH/OW = H/2, W/2 so the spatial dims stay consistent.
    """
    spec = yaml.safe_load(pathlib.Path(yaml_path).read_text()) or {}
    inputs = spec.get("inputs") or {}
    outputs = spec.get("outputs") or {}

    vlist = spec.get(variant)
    if not vlist:
        raise SystemExit(
            f"variant {variant!r} not in spec; have: "
            f"{[k for k in spec if isinstance(spec.get(k), list)]}"
        )
    v = vlist[0]
    dims = dict(v.get("dims") or {})
    for k, val in overrides.items():
        if val is not None and k in dims:
            dims[k] = val
    # Keep the pooled output dims consistent with any H/W override.
    if "OH" in dims and "H" in dims:
        dims["OH"] = dims["H"] // 2
    if "OW" in dims and "W" in dims:
        dims["OW"] = dims["W"] // 2

    params = v.get("params") or list(inputs.keys())
    input_specs = []
    for name in params:
        spec_in = inputs[name]
        input_specs.append((name, spec_in["shape"], _np_dtype(spec_in.get("dtype", "float16"))))

    out_name, out_def = next(iter(outputs.items()))
    out_spec = (out_name, out_def["shape"], _np_dtype(out_def.get("dtype", "float16")))
    return input_specs, out_spec, dims, v.get("flop")


def eval_flop(formula: str | None, dims: dict[str, int]) -> float | None:
    """Evaluate the spec's flop formula over the resolved dims (no builtins)."""
    if not formula:
        return None
    try:
        return float(eval(formula, {"__builtins__": {}}, dict(dims)))  # noqa: S307
    except Exception:
        return None


def conv_reference(bufs, dims: dict[str, int]):
    """Torch golden reference matching the kernel (fp32 accumulate).

    Conv2d(3x3, pad=1, stride=1, bias) -> ReLU -> AvgPool2d(2). The kernel is
    NHWC with channels-last weight [KH,KW,CIN,COUT]; torch wants NCHW + OIHW, so
    permute in, permute out.
    """
    import torch
    import torch.nn.functional as F

    def t(name):  # fp16 numpy -> fp32 torch
        return torch.from_numpy(bufs[name].astype(np.float32))

    pad = dims["KH"] // 2
    x = t("X").permute(0, 3, 1, 2).contiguous()          # NHWC -> NCHW
    w = t("weight").permute(3, 2, 0, 1).contiguous()     # [KH,KW,CIN,COUT] -> OIHW
    y = F.conv2d(x, w, t("bias"), stride=1, padding=pad)
    y = F.relu(y)
    y = F.avg_pool2d(y, kernel_size=2)
    return y.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float16)  # NCHW -> NHWC


def run_conv(cm_path: str, yaml_path: str, variant: str, overrides: dict,
             build_opts: str, iters: int, warmup: int, seed: int) -> int:
    src = pathlib.Path(cm_path).read_text()
    defines = extract_defines(src)
    lws_x = max(1, defines.get("LWS_X", 1))
    lws_y = max(1, defines.get("LWS_Y", 1))
    # Pooled output pixels (along OW) each work-item computes (register blocking).
    pw_tile = max(1, defines.get("PW_TILE", 1))

    input_specs, out_spec, dims, flop_formula = load_spec(yaml_path, variant, overrides)

    def resolve(shape_syms):
        return tuple(int(dims[s]) for s in shape_syms)

    platform, device = pick_device()
    print(f"platform : {platform.name}")
    print(f"device   : {device.name} ({_type_label(device)})")
    print(f"build    : clBuildProgram(options={build_opts!r})")
    print(f"spec     : {yaml_path}  variant={variant}")
    print(f"dims     : {dims}  (LWS_X={lws_x} LWS_Y={lws_y} PW_TILE={pw_tile})")

    ctx = cl.Context(devices=[device])
    try:
        prg = cl.Program(ctx, src).build(options=build_opts)
    except cl.RuntimeError as e:
        print(f"[fail] CM source build failed:\n{e}")
        return 1
    names = [k.function_name for k in prg.all_kernels()]
    entry = names[0] if names else None
    if entry is None:
        print("[fail] program exposes no kernels")
        return 1
    print(f"built {cm_path} -> entry: {entry}")

    # Random inputs (fixed seed) for every buffer the spec declares, by name.
    rng = np.random.default_rng(seed)
    bufs: dict[str, np.ndarray] = {}
    for name, shape_syms, dt in input_specs:
        shape = resolve(shape_syms)
        bufs[name] = (rng.standard_normal(shape) * 0.5).astype(dt)

    # Golden reference (Torch) BEFORE touching the GPU.
    ref = conv_reference(bufs, dims)

    mf = cl.mem_flags
    in_bufs = [
        cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                  hostbuf=np.ascontiguousarray(bufs[name]))
        for name, _, _ in input_specs
    ]
    out_shape = resolve(out_spec[1])
    out_dt = out_spec[2]
    out_nbytes = int(np.prod(out_shape)) * out_dt.itemsize
    y_buf = cl.Buffer(ctx, mf.WRITE_ONLY, out_nbytes)

    # Scalars: dims values in spec (insertion) order -> int32, matching the ABI.
    scalars = [np.int32(v) for v in dims.values()]

    # Grid: one work-item per pooled output row (N*OH) x block of PW_TILE pooled
    # columns (ceil(OW/PW_TILE)), padded up to a multiple of the tunable work-group
    # dims; local = (LWS_X, LWS_Y). The kernel guards the padded/blocked tail.
    N, OH, OW = dims["N"], dims["OH"], dims["OW"]
    gx = math.ceil(N * OH / lws_x) * lws_x
    gy = math.ceil(math.ceil(OW / pw_tile) / lws_y) * lws_y
    gsize = (gx, gy)
    lsize = (lws_x, lws_y)
    print(f"grid     : global={gsize} local={lsize}")

    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    kernel = cl.Kernel(prg, entry)
    args = (*in_bufs, y_buf, *scalars)

    for _ in range(max(0, warmup)):
        kernel(queue, gsize, lsize, *args)
    queue.finish()

    iters = max(1, iters)
    events = [kernel(queue, gsize, lsize, *args) for _ in range(iters)]
    queue.finish()
    per_ms = sorted((e.profile.end - e.profile.start) * 1e-6 for e in events)
    min_ms = per_ms[0]
    median_ms = per_ms[len(per_ms) // 2]
    mean_ms = sum(per_ms) / iters

    y = np.empty(int(np.prod(out_shape)), out_dt)
    cl.enqueue_copy(queue, y, y_buf)
    queue.finish()
    y = y.reshape(out_shape)

    # Correctness via relative L2 (robust to fp16 reordering drift) + cosine.
    a = ref.astype(np.float64).ravel()
    bb = y.astype(np.float64).ravel()
    rel_l2 = float(np.linalg.norm(bb - a) / (np.linalg.norm(a) + 1e-12))
    cos = float(a @ bb) / (float(np.linalg.norm(a) * np.linalg.norm(bb)) + 1e-12)
    finite = bool(np.all(np.isfinite(y)))
    ok = finite and rel_l2 < 5e-2

    flop = eval_flop(flop_formula, dims)
    tflops = (flop / (min_ms * 1e-3) / 1e12) if flop else None
    tf = f"{tflops:.3f} TFLOPS(min)" if tflops is not None else "TFLOPS=n/a"
    print(f"conv {variant} {tuple(out_shape)}  "
          f"min={min_ms:.3f} median={median_ms:.3f} mean={mean_ms:.3f} ms/iter "
          f"({iters} iters, {warmup} warmup)  {tf}  "
          f"rel-L2={rel_l2:.2e} cos={cos:.6f}  finite={finite}  match={ok}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--list", action="store_true", help="enumerate OpenCL devices and exit")
    ap.add_argument("--cm", metavar="PATH", default=DEFAULT_CM,
                    help=f"CM .cpp to compile + run (default {DEFAULT_CM})")
    ap.add_argument("--yaml", metavar="PATH",
                    help="kernel spec .yaml (default: the --cm path with a .yaml suffix)")
    ap.add_argument("--variant", default="ci", help="spec variant (default: ci)")
    ap.add_argument("--cm-opts", default=DEFAULT_CM_OPTS,
                    help=f"clBuildProgram options (default {DEFAULT_CM_OPTS!r})")
    ap.add_argument("--height", type=int, default=None, help="override H")
    ap.add_argument("--width", type=int, default=None, help="override W")
    ap.add_argument("--iters", type=int, default=20, help="timed iterations (default 20)")
    ap.add_argument("--warmup", type=int, default=10, help="warmup iterations (default 10)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for inputs (default 0)")
    args = ap.parse_args(argv)

    if args.list:
        list_devices()
        return 0

    yaml_path = args.yaml or str(pathlib.Path(args.cm).with_suffix(".yaml"))
    if not pathlib.Path(yaml_path).is_file():
        print(f"[fail] spec not found: {yaml_path} (pass --yaml)")
        return 1

    overrides = {"H": args.height, "W": args.width}
    return run_conv(args.cm, yaml_path, args.variant, overrides,
                    args.cm_opts, args.iters, args.warmup, args.seed)


if __name__ == "__main__":
    sys.exit(main())
