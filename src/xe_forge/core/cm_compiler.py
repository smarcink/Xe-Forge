"""
CM ("C for Metal") compiler wrapper.

Drives the Intel CM toolchain:
  * :meth:`CMCompiler.compile` invokes the ``cmc`` offline compiler to lower a CM
    ``.cpp`` kernel to SPIR-V (``-emit-spirv``), and
  * :meth:`CMCompiler.run` (still a STUB) is meant to drive an OpenCL / Level-Zero
    host harness that uploads inputs, launches the kernel, times it over N
    iterations, and dumps the output tensor.

The public interface intentionally mirrors
``ai_bench.sycl.compiler.SYCLCompiler`` so that :class:`CMExecutor` can drive it
exactly the way ``SyclExecutor`` drives the SYCL compiler.

Environment variables:
  * ``CMC_BIN``  — path to the ``cmc`` compiler binary (default: ``cmc``).
  * ``CM_ROOT``  — root of the CM SDK (headers under ``$CM_ROOT/include``).
  * ``CM_MCPU``  — default ``-mcpu`` platform when no device target is resolved.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from xe_forge.core.cm_grid import GridConfig

logger = logging.getLogger(__name__)

CM_ROOT = os.environ.get("CM_ROOT", "")
CMC_BIN = os.environ.get("CMC_BIN", "cmc")

# cmc lowers CM to vISA for a concrete GPU platform passed via ``-mcpu=<PLATFORM>``
# (BMG / DG2 / MTL / TGLLP / PVC). The rest of the codebase identifies devices by
# architecture family ("xe2", "xehpg", "xehpc"); map those — and common aliases —
# onto the platform names cmc expects. Unknown values are passed through upcased.
_MCPU_BY_TARGET: dict[str, str] = {
    "xe2": "BMG",
    "bmg": "BMG",
    "battlemage": "BMG",
    "lnl": "BMG",  # Lunar Lake is also Xe2; BMG codegen applies
    "xehpg": "DG2",
    "dg2": "DG2",
    "arc": "DG2",
    "xehpc": "PVC",
    "pvc": "PVC",
    "mtl": "MTL",
    "arl": "MTL",
    "tgllp": "TGLLP",
    "tgl": "TGLLP",
}

# Used when no device target can be resolved (e.g. compiling on a host without an
# XPU). The repo's seed kernels use LSC/DPAS, so default to an Xe target.
_DEFAULT_MCPU = os.environ.get("CM_MCPU", "BMG")

# Reason surfaced to callers for the (still-stubbed) run path.
_STUB_REASON = (
    "CM run path is not implemented yet (OpenCL/L0 host harness is stubbed). "
    "Implement CMCompiler.run() to launch the compiled SPIR-V and time it."
)


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
    """Compile and run CM kernels via the ``cmc`` toolchain — STUB.

    Args:
        include_dirs: Header search paths handed to ``cmc`` (``-I``).
        target_device: AOT device target (e.g. ``"bmg"`, ``"pvc"``). May be
            ``None`` to let the compiler pick a default.
        cmc_bin: Path to the ``cmc`` binary.
        cm_root: Root of the CM SDK (its ``include`` dir is added automatically).
    """

    def __init__(
        self,
        include_dirs: list[str] | None = None,
        target_device: str | None = None,
        cmc_bin: str = CMC_BIN,
        cm_root: str = CM_ROOT,
    ):
        self.include_dirs: list[str] = list(include_dirs or [])
        if cm_root:
            sdk_include = str(Path(cm_root) / "include")
            if sdk_include not in self.include_dirs:
                self.include_dirs.append(sdk_include)
        self.target_device = target_device
        self.cmc_bin = cmc_bin
        self.cm_root = cm_root
        self.last_compile_error: str | None = None

    @property
    def available(self) -> bool:
        """True when the ``cmc`` compiler can be found on PATH / at CMC_BIN."""
        return shutil.which(self.cmc_bin) is not None or Path(self.cmc_bin).is_file()

    @property
    def mcpu(self) -> str:
        """The ``-mcpu`` platform name cmc should target."""
        target = (self.target_device or "").strip().lower()
        if not target:
            return _DEFAULT_MCPU
        return _MCPU_BY_TARGET.get(target, target.upper())

    def _subprocess_env(self) -> dict[str, str]:
        """Environment for invoking cmc, ensuring its sibling DLLs/.so resolve.

        cmc loads ``clangFEWrapper`` from its own directory, so that directory is
        prepended to PATH (Windows) / LD_LIBRARY_PATH (Linux).
        """
        env = os.environ.copy()
        cmc_path = shutil.which(self.cmc_bin) or self.cmc_bin
        cmc_dir = str(Path(cmc_path).resolve().parent)
        if os.name == "nt":
            env["PATH"] = cmc_dir + os.pathsep + env.get("PATH", "")
        else:
            env["LD_LIBRARY_PATH"] = cmc_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        return env

    def compile(self, src_path: str | Path) -> Path | None:
        """Compile a CM ``.cpp`` source to SPIR-V via ``cmc``.

        Invokes ``cmc <src> -o <src>.spv -emit-spirv -mcpu=<PLATFORM>`` (plus any
        configured ``-I`` include dirs). Emitting SPIR-V keeps compilation offline
        and device-free — the runtime JITs it at load — so this does not require
        ``ocloc``. Returns the path to the ``.spv`` on success, or ``None`` on
        failure (with :attr:`last_compile_error` populated).
        """
        src_path = Path(src_path)
        if not src_path.is_file():
            self.last_compile_error = f"CM source not found: {src_path}"
            return None

        if not self.available:
            self.last_compile_error = f"cmc compiler not found (CMC_BIN={self.cmc_bin!r})."
            logger.warning(self.last_compile_error)
            return None

        out_path = src_path.with_suffix(".spv")
        cmd = [
            self.cmc_bin,
            str(src_path),
            "-o", str(out_path),
            "-emit-spirv",
            f"-mcpu={self.mcpu}",
            *[f"-I{d}" for d in self.include_dirs],
        ]
        logger.info("Compiling CM kernel: %s", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self._subprocess_env(),
                check=False,
            )
        except OSError as e:
            self.last_compile_error = f"Failed to invoke cmc ({self.cmc_bin!r}): {e}"
            logger.warning(self.last_compile_error)
            return None

        if proc.returncode != 0 or not out_path.is_file():
            self.last_compile_error = (
                proc.stderr.strip() or proc.stdout.strip() or "cmc failed (no diagnostics)"
            )
            logger.warning("cmc compilation failed:\n%s", self.last_compile_error)
            return None

        self.last_compile_error = None
        logger.info("Compiled CM kernel -> %s", out_path)
        return out_path

    def run(
        self,
        binary: str | Path,
        dims: dict[str, int | float] | None = None,
        iterations: int = 20,
        verify: int = 0,
        input_dir: str | None = None,
        output_dir: str | None = None,
        grid: GridConfig | None = None,
    ) -> CMRunResult:
        """Run a compiled CM kernel and parse its timing output — STUB.

        ``dims`` is a generic name->int map (e.g. ``{"M": .., "N": .., "K": ..}``
        for a GEMM, but any kernel's shape parameters) passed to the host harness
        as CLI args. ``grid`` is the concrete launch geometry resolved by
        :func:`xe_forge.core.cm_grid.compute_grid` from the kernel's ``#define``
        block sizes; the harness binds it as the ND-range / work-group size.
        ``input_dir`` (when given) holds the shared input tensors the harness
        binds as STATEFUL buffers (``input_0.bin``, ``input_1.bin``, ...);
        ``output_dir`` is where it dumps the result (``output_0.bin``) for
        external correctness comparison. When ``input_dir`` is set, ``verify`` is
        typically 0 because correctness is checked in Python against those dumps.

        TODO(cm): execute the host harness with the given dims, iteration count,
        and ``grid.global_size`` / ``grid.local_size`` (written into the launch
        manifest), loading inputs from ``input_dir`` and dumping
        ``output_dir/output_0.bin``, parse "<tflops> TFlop/s (<ms>) ms", and
        return timing.
        """
        return CMRunResult(success=False, error=_STUB_REASON, grid=grid)
