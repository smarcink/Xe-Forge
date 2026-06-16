"""
CM ("C for Metal") kernel launcher.

Runs a CM ``.cpp`` kernel by spawning an isolated worker process
(:mod:`xe_forge.core.cm_worker`) that compiles the kernel ONLINE via PyOpenCL's
``-cmc`` build option (the Intel IGC Vector-Compute frontend — no offline
``cmc``/SPIR-V step) and launches it on the GPU.

Running each kernel in a separate, short-lived process is what makes the
optimization loop robust to a hung kernel or a GPU TDR (timeout detection &
recovery): :meth:`CMCompiler.run` enforces a hard wall-clock timeout and kills
the worker (and its whole process tree) on expiry, returning
``CMRunResult(success=False, ...)`` instead of taking the application down.

``CMCompiler`` + ``CMRunResult`` keep the shape ``CMExecutor`` already expects so
the optimizer consumes CM results exactly like SYCL ones.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from xe_forge.core.cm_grid import GridConfig
from xe_forge.core.cm_worker import RESULT_PREFIX

logger = logging.getLogger(__name__)

# Default hard timeout (seconds) for a single kernel launch before the worker is
# considered hung/TDR'd and force-killed. A healthy kernel runs in milliseconds.
_DEFAULT_HANG_TIMEOUT = 30


@dataclass
class CMRunResult:
    """Result of running a compiled CM kernel.

    Mirrors ``ai_bench.sycl.compiler.SYCLRunResult`` so CMExecutor can consume
    it the same way the SYCL executor consumes its run result.
    """

    success: bool
    passed: bool | None = None
    time_ms: float | None = None
    tflops: float | None = None
    error: str = ""
    grid: GridConfig | None = None  # Grid configuration used for launch


class CMCompiler:
    """Launches CM kernels in an isolated PyOpenCL worker subprocess.

    Each :meth:`run` spawns :mod:`xe_forge.core.cm_worker` to compile (online
    ``-cmc``) and execute one kernel, bounded by a hard ``hang_timeout`` so a
    hanging kernel / GPU TDR cannot wedge the parent process.

    Args:
        hang_timeout: Seconds to wait for a kernel run before killing the worker
            and reporting a graceful failure. A healthy kernel runs in ms.
        build_options: PyOpenCL ``clBuildProgram`` options for the online CM
            compile (``-cmc`` selects the IGC Vector-Compute frontend).
    """

    def __init__(
        self,
        hang_timeout: int = _DEFAULT_HANG_TIMEOUT,
        build_options: str = "-cmc",
    ):
        self.hang_timeout = hang_timeout
        self.build_options = build_options

    def run(
        self,
        source_path: str | Path,
        *,
        dims: dict[str, int | float] | None = None,
        grid: GridConfig,
        output_sizes: list[int],
        input_dir: str | None = None,
        output_dir: str | None = None,
        iterations: int = 20,
        warmup: int = 3,
        entry: str | None = None,
    ) -> CMRunResult:
        """Compile + launch a CM kernel in an isolated worker and time it.

        The kernel ABI is inputs (``input_0.bin``, ``input_1.bin``, ... from
        ``input_dir``, in order) -> outputs (one ``output_<i>.bin`` per entry in
        ``output_sizes``, written to ``output_dir``) -> scalars (``dims`` values,
        in order). ``grid`` is the launch geometry from
        :func:`xe_forge.core.cm_grid.compute_grid`.

        A hung kernel / GPU TDR is bounded by ``self.hang_timeout``: the worker
        (and its process tree) is force-killed and ``CMRunResult(success=False)``
        is returned, so the optimizer degrades gracefully instead of crashing.
        """
        source_path = Path(source_path)
        if not source_path.is_file():
            return CMRunResult(
                success=False, error=f"CM source not found: {source_path}", grid=grid
            )

        work_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="cm_run_"))
        work_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "source_path": str(source_path),
            "build_options": self.build_options,
            "entry": entry,
            "input_dir": str(input_dir) if input_dir else "",
            "inputs": _ordered_inputs(input_dir),
            "outputs": [
                {"file": f"output_{i}.bin", "bytes": int(n)}
                for i, n in enumerate(output_sizes)
            ],
            "scalars": [_scalar_spec(v) for v in (dims or {}).values()],
            "grid": {"global": list(grid.global_size), "local": list(grid.local_size)},
            "output_dir": str(output_dir) if output_dir else "",
            "warmup": warmup,
            "iterations": iterations,
        }
        manifest_path = work_dir / "cm_launch.json"
        manifest_path.write_text(json.dumps(manifest))

        cmd = [sys.executable, "-m", "xe_forge.core.cm_worker", str(manifest_path)]
        popen_kwargs: dict = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
        }
        # Put the worker in its own process group/session so a wedged driver
        # thread can't orphan grandchildren — we kill the whole tree on timeout.
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        logger.info("Launching CM worker: %s", " ".join(cmd))
        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)
        except OSError as e:
            return CMRunResult(success=False, error=f"failed to spawn CM worker: {e}", grid=grid)

        try:
            stdout, stderr = proc.communicate(timeout=self.hang_timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            logger.warning(
                "CM worker timed out after %ss — killed (hang/TDR).", self.hang_timeout
            )
            return CMRunResult(
                success=False,
                error=(
                    f"hang/TDR: CM kernel did not finish within {self.hang_timeout}s "
                    "and was force-killed"
                ),
                grid=grid,
            )

        result = _parse_result(stdout)
        if result is None:
            tail = (stderr or stdout or "").strip()[-2000:]
            return CMRunResult(
                success=False,
                error=f"CM worker produced no result (exit {proc.returncode}):\n{tail}",
                grid=grid,
            )
        if not result.get("success"):
            stage = result.get("stage", "?")
            return CMRunResult(
                success=False,
                error=f"[{stage}] {result.get('error', 'unknown error')}",
                grid=grid,
            )
        return CMRunResult(
            success=True, passed=None, time_ms=result.get("time_ms"), grid=grid
        )


def _ordered_inputs(input_dir: str | None) -> list[str]:
    """Return ``input_0.bin``, ``input_1.bin``, ... that exist in ``input_dir``,
    in contiguous numeric (ABI) order."""
    if not input_dir:
        return []
    base = Path(input_dir)
    names: list[str] = []
    i = 0
    while (base / f"input_{i}.bin").is_file():
        names.append(f"input_{i}.bin")
        i += 1
    return names


def _scalar_spec(value: int | float) -> dict:
    """Describe a scalar kernel arg for the manifest (int -> int32, else float32)."""
    if isinstance(value, bool):  # bool is an int subclass — treat as int32
        return {"value": int(value), "type": "int32"}
    if isinstance(value, int):
        return {"value": value, "type": "int32"}
    return {"value": float(value), "type": "float32"}


def _parse_result(stdout: str | None) -> dict | None:
    """Extract the worker's sentinel-prefixed JSON result from ``stdout``.

    Scans from the end so any earlier driver chatter (e.g. ``-cmc`` asm-count
    output) is ignored. Returns ``None`` if no valid result line is present.
    """
    if not stdout:
        return None
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX):])
            except json.JSONDecodeError:
                return None
    return None


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Force-kill the worker and all its descendants.

    A GPU TDR can leave driver threads wedged; killing only the immediate child
    may orphan grandchildren. On Windows use ``taskkill /T``; on POSIX kill the
    process group created via ``start_new_session``.
    """
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
