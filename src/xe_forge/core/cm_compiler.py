"""
CM ("C for Metal") compiler wrapper — STUB.

This is a placeholder for the Intel CM toolchain:
  * the ``cmc`` offline kernel compiler, and
  * an OpenCL / Level-Zero host harness that uploads inputs, launches the
    kernel, times it over N iterations, and dumps the output tensor.

The public interface intentionally mirrors
``ai_bench.sycl.compiler.SYCLCompiler`` so that :class:`CMExecutor` can drive it
exactly the way ``SyclExecutor`` drives the SYCL compiler. Once the real
toolchain is available, fill in :meth:`CMCompiler.compile` and
:meth:`CMCompiler.run`; nothing else in the pipeline should need to change.

Environment variables:
  * ``CMC_BIN``  — path to the ``cmc`` compiler binary (default: ``cmc``).
  * ``CM_ROOT``  — root of the CM SDK (headers under ``$CM_ROOT/include``).
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CM_ROOT = os.environ.get("CM_ROOT", "")
CMC_BIN = os.environ.get("CMC_BIN", "cmc")

# Reason surfaced to callers until the real toolchain is wired in.
_STUB_REASON = (
    "CM toolchain is not implemented yet (cmc compiler + OpenCL/L0 host harness "
    "are stubbed). Set CMC_BIN/CM_ROOT and implement CMCompiler.compile()/run()."
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

    def compile(self, src_path: str | Path) -> Path | None:
        """Compile a CM ``.cpp`` source to a runnable binary — STUB.

        Returns the path to the compiled binary on success, or ``None`` on
        failure (with :attr:`last_compile_error` populated).

        TODO(cm): implement the real invocation, roughly::

            out = Path(src_path).with_suffix("")
            cmd = [
                self.cmc_bin,
                f"-march={self.target_device or 'bmg'}",
                *[f"-I{d}" for d in self.include_dirs],
                "-o", str(out_isa),
                str(src_path),
            ]
            # then build the OpenCL/L0 host program that loads the generated
            # ISA/SPIR-V, binds the inputs as STATEFUL buffers (SurfaceIndex;
            # input_0.bin, input_1.bin, ...) plus the output buffer, runs
            # `iterations`, prints a "<tflops> TFlop/s (<ms>) ms" line, and
            # dumps output_0.bin.

        For now this is a placeholder so the rest of the pipeline can be wired
        and tested end-to-end without the toolchain present.
        """
        src_path = Path(src_path)
        if not src_path.is_file():
            self.last_compile_error = f"CM source not found: {src_path}"
            return None

        if not self.available:
            self.last_compile_error = (
                f"cmc compiler not found (CMC_BIN={self.cmc_bin!r}). {_STUB_REASON}"
            )
            logger.warning(self.last_compile_error)
            return None

        # cmc is present but the build recipe (flags + host harness link) is not
        # implemented yet. Fail loudly rather than producing a bogus binary.
        self.last_compile_error = _STUB_REASON
        logger.warning("CMCompiler.compile() is a stub: %s", _STUB_REASON)
        return None

    def run(
        self,
        binary: str | Path,
        dims: dict[str, int | float] | None = None,
        iterations: int = 20,
        verify: int = 0,
        input_dir: str | None = None,
        output_dir: str | None = None,
    ) -> CMRunResult:
        """Run a compiled CM kernel and parse its timing output — STUB.

        ``dims`` is a generic name->int map (e.g. ``{"M": .., "N": .., "K": ..}``
        for a GEMM, but any kernel's shape parameters) passed to the host harness
        as CLI args. ``input_dir`` (when given) holds the shared input tensors the
        harness binds as STATEFUL buffers (``input_0.bin``, ``input_1.bin``, ...);
        ``output_dir`` is where it dumps the result (``output_0.bin``) for
        external correctness comparison. When ``input_dir`` is set, ``verify`` is
        typically 0 because correctness is checked in Python against those dumps.

        TODO(cm): execute the host harness with the given dims and iteration
        count, loading inputs from ``input_dir`` and dumping
        ``output_dir/output_0.bin``, parse "<tflops> TFlop/s (<ms>) ms", and
        return timing.
        """
        return CMRunResult(success=False, error=_STUB_REASON)
