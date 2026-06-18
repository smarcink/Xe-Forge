"""
Isolated PyOpenCL worker that compiles and runs a single CM kernel.

This module is meant to be launched as a short-lived subprocess::

    python -m xe_forge.core.cm_worker <manifest.json>

It compiles the CM ``.cpp`` source ONLINE via PyOpenCL's ``-cmc`` build option
(the Intel IGC Vector-Compute frontend — no offline ``cmc``/SPIR-V step), binds
the input/output buffers and scalar arguments described by the manifest, launches
the kernel over the requested grid, times it, and dumps the output tensors.

Running in a separate process is deliberate: an LLM-generated CM kernel can hang
or trip a GPU TDR (timeout detection & recovery), which poisons the OpenCL/driver
context. Isolating each run means the parent (:class:`xe_forge.core.cm_compiler.CMCompiler`)
can kill a wedged worker and keep going instead of crashing the whole optimizer.

The worker is intentionally torch-free (numpy + pyopencl only) so it starts fast.
It communicates results back over stdout as a single sentinel-prefixed JSON line
(``RESULT_PREFIX``); any driver chatter on stdout is ignored by the parent, which
scans for that prefix. Exit code is 0 on success, 1 on any failure, with the
failure ``stage`` and ``error`` captured in the JSON line.

Manifest schema (all paths absolute)::

    {
      "source_path": "/abs/kernel.cpp",
      "build_options": "-cmc",
      "entry": null | "kernel_name",
      "input_dir": "/abs/inputs",
      "inputs": ["input_0.bin", "input_1.bin"],          # ABI order
      "outputs": [{"file": "output_0.bin", "bytes": 65536}],
      "scalars": [{"value": 256, "type": "int32"}, ...],  # ABI order (after outputs)
      "grid": {"global": [x, y, z], "local": [x, y, z]},
      "output_dir": "/abs/out",
      "warmup": 3,
      "iterations": 20
    }

ABI: kernel args are bound as inputs (spec order) -> outputs (spec order) ->
scalars (spec order), matching the seed ``cm_gemm(A, B, D, M, N, K)``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# Sentinel that prefixes the single machine-readable result line on stdout. The
# parent (CMCompiler.run) scans stdout for this so it is robust to any other
# text the driver prints (e.g. -cmc asm-count diagnostics).
RESULT_PREFIX = "__CM_WORKER_RESULT__ "


def _emit(result: dict) -> None:
    """Print the single sentinel-prefixed JSON result line and flush stdout."""
    sys.stdout.flush()
    print(RESULT_PREFIX + json.dumps(result), flush=True)


def _fail(stage: str, error: object) -> int:
    """Emit a failure result for ``stage`` and return process exit code 1."""
    _emit({"success": False, "stage": stage, "error": str(error)})
    return 1


def _pick_device(cl, prefer: str = "intel"):
    """Pick a (platform, device), preferring a platform whose name contains
    ``prefer``. Overridable via env ``XE_OCL_PLATFORM`` / ``XE_OCL_DEVICE``
    (case-insensitive substring match). This box may also expose an NVIDIA
    OpenCL platform, so the Intel default matters."""
    plat_filter = os.environ.get("XE_OCL_PLATFORM", prefer).lower()
    dev_filter = os.environ.get("XE_OCL_DEVICE", "").lower()

    candidates = [(p, d) for p in cl.get_platforms() for d in p.get_devices()]
    if not candidates:
        raise RuntimeError("No OpenCL platforms/devices found")

    matches = [
        (p, d)
        for (p, d) in candidates
        if plat_filter in p.name.lower() and (not dev_filter or dev_filter in d.name.lower())
    ]
    return matches[0] if matches else candidates[0]


def _scalar(np, spec: dict):
    """Build a by-value numpy scalar for a manifest scalar spec."""
    t = spec["type"]
    if t == "int32":
        return np.int32(spec["value"])
    if t == "float32":
        return np.float32(spec["value"])
    raise ValueError(f"Unsupported scalar type: {t!r}")


def run(manifest: dict) -> int:
    """Compile + launch + time the kernel described by ``manifest``.

    Returns a process exit code (0 success, 1 failure) and emits exactly one
    sentinel result line describing the outcome.
    """
    try:
        import numpy as np
        import pyopencl as cl
    except Exception as e:  # pragma: no cover - environment guard
        return _fail("import", f"pyopencl/numpy import failed: {e}")

    # --- device + context ---
    try:
        _platform, device = _pick_device(cl)
        ctx = cl.Context(devices=[device])
        queue = cl.CommandQueue(
            ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
        )
    except Exception as e:
        return _fail("device", e)

    # --- online -cmc compile (no offline cmc / .spv) ---
    src_path = manifest["source_path"]
    build_opts = manifest.get("build_options", "-cmc")
    try:
        src = pathlib.Path(src_path).read_text()
    except OSError as e:
        return _fail("io", f"cannot read source {src_path!r}: {e}")
    try:
        prg = cl.Program(ctx, src).build(options=build_opts)
    except Exception as e:
        return _fail("compile", e)

    try:
        names = [k.function_name for k in prg.all_kernels()]
    except Exception as e:
        return _fail("compile", f"could not enumerate kernels: {e}")
    entry = manifest.get("entry") or (names[0] if names else None)
    if entry is None:
        return _fail("compile", "program exposes no kernels")
    if entry not in names:
        return _fail("compile", f"entry {entry!r} not found; kernels={names}")

    # --- bind args: inputs -> outputs -> scalars ---
    try:
        mf = cl.mem_flags
        input_dir = manifest.get("input_dir") or ""
        in_bufs = []
        for fname in manifest.get("inputs", []):
            data = pathlib.Path(input_dir, fname).read_bytes()
            host = np.frombuffer(data, dtype=np.uint8)
            in_bufs.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host))
        out_specs = manifest.get("outputs", [])
        out_bufs = [cl.Buffer(ctx, mf.WRITE_ONLY, int(s["bytes"])) for s in out_specs]
        scalars = [_scalar(np, s) for s in manifest.get("scalars", [])]
    except Exception as e:
        return _fail("run", f"argument binding failed: {e}")

    gsize = tuple(manifest["grid"]["global"])
    lsize = tuple(manifest["grid"]["local"])
    args = [*in_bufs, *out_bufs, *scalars]

    # --- launch + time ---
    try:
        kernel = cl.Kernel(prg, entry)
        warmup = max(0, int(manifest.get("warmup", 3)))
        iterations = max(1, int(manifest.get("iterations", 20)))
        for _ in range(warmup):
            kernel(queue, gsize, lsize, *args)
        queue.finish()
        events = [kernel(queue, gsize, lsize, *args) for _ in range(iterations)]
        queue.finish()
        # The MIN iteration is the
        # one that hit peak clock with the least interference — the most
        # reproducible estimate of true kernel cost (this is what clpeak reports).
        # median + mean are emitted too so the parent can see the spread.
        per_ms = sorted((e.profile.end - e.profile.start) * 1e-6 for e in events)
        min_ms = per_ms[0]
        median_ms = per_ms[len(per_ms) // 2]
        mean_ms = sum(per_ms) / iterations
    except Exception as e:
        return _fail("run", f"kernel launch failed: {e}")

    # --- dump outputs ---
    try:
        out_dir = manifest.get("output_dir")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            for spec, buf in zip(out_specs, out_bufs):
                host = np.empty(int(spec["bytes"]), dtype=np.uint8)
                cl.enqueue_copy(queue, host, buf)
                queue.finish()
                host.tofile(os.path.join(out_dir, spec["file"]))
    except Exception as e:
        return _fail("io", f"output dump failed: {e}")

    _emit(
        {
            "success": True,
            "time_ms": min_ms,  # headline = MIN (robust to DVFS/interference)
            "time_median_ms": median_ms,
            "time_mean_ms": mean_ms,
            "entry": entry,
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Isolated PyOpenCL CM kernel worker.")
    ap.add_argument("manifest", help="path to the launch manifest JSON")
    args = ap.parse_args(argv)
    try:
        manifest = json.loads(pathlib.Path(args.manifest).read_text())
    except Exception as e:
        return _fail("io", f"cannot read manifest {args.manifest!r}: {e}")
    return run(manifest)


if __name__ == "__main__":
    sys.exit(main())
