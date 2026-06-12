"""
CM ("C for Metal") Kernel Executor — compiles and benchmarks CM C++ kernels.

Mirrors :class:`xe_forge.core.sycl_executor.SyclExecutor`: it drives the CM
toolchain (:class:`xe_forge.core.cm_compiler.CMCompiler`) to compile an original
and an optimized kernel, run both on identical inputs, and report speedup +
correctness via :class:`CMComparisonResult` so the optimization loop can consume
the same feedback shape it gets from SYCL.

The underlying compiler is currently a stub (see ``cm_compiler.py``); until the
real ``cmc`` + host harness is wired in, compilation fails gracefully with an
informative message and the optimizer falls back to static checks.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ai_bench.harness.runner.benchmark_compare import set_all_seeds

from xe_forge.core.cm_compiler import CM_ROOT, CMCompiler, CMRunResult
from xe_forge.core.sycl_executor import KernelType, _save_tensor
from xe_forge.models import ExecutionResult

logger = logging.getLogger(__name__)

# cmc AOT device target. Override explicitly with the CM_TARGET env var or
# CMExecutor(device_target=...). Auto-detection prefers the PCI device id /
# architecture from torch.xpu (stable hardware identifiers) over the driver
# "name" string, which only reflects the installed driver build.
_CM_TARGET_ENV = "CM_TARGET"


def _target_from_pci_id(pci_id: int) -> str:
    """Map an Intel GPU PCI device id to a cmc architecture family."""
    if 0xE200 <= pci_id <= 0xE2FF:  # Battlemage (Xe2), e.g. B580=0xE20B
        return "xe2"
    if pci_id in (0x6420, 0x64A0, 0x64B0):  # Lunar Lake (Xe2 iGPU)
        return "xe2"
    if (0x4F80 <= pci_id <= 0x4F88) or (0x5690 <= pci_id <= 0x56BF):  # DG2 / Arc A-series (Xe-HPG)
        return "xehpg"
    if 0x0BD0 <= pci_id <= 0x0BDF:  # Ponte Vecchio (Xe-HPC)
        return "xehpc"
    return ""


# Legacy fallback only: substring match on the (often unreliable) device name.
_CM_DEVICE_NAME_TO_TARGET: dict[str, str] = {
    "battlemage": "xe2",
    "bmg": "xe2",
    "b580": "xe2",
    "b570": "xe2",
    "lunar lake": "xe2",
    "arc": "xehpg",
    "a770": "xehpg",
    "a750": "xehpg",
    "a580": "xehpg",
    "a380": "xehpg",
    "ponte vecchio": "xehpc",
    "data center gpu max": "xehpc",
    "max 1550": "xehpc",
    "max 1100": "xehpc",
}


def _detect_device_target() -> str:
    """Auto-detect the cmc AOT device target.

    Priority: CM_TARGET env override -> PCI device-id range -> name fallback.
    Returns "" when undetermined (cmc then picks its own default). The device
    ``architecture`` int is logged to help extend the mapping for new parts.
    """
    env = os.environ.get(_CM_TARGET_ENV, "").strip()
    if env:
        logger.info("CM target '%s' (from %s)", env, _CM_TARGET_ENV)
        return env
    try:
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            return ""
        props = torch.xpu.get_device_properties(torch.xpu.current_device())
        pci_id = int(getattr(props, "device_id", 0) or 0)
        arch = getattr(props, "architecture", None)
        target = _target_from_pci_id(pci_id)
        if target:
            logger.info("CM target '%s' (PCI 0x%04X, architecture=%s)", target, pci_id, arch)
            return target
        name = (getattr(props, "name", "") or "").lower()
        for key, t in _CM_DEVICE_NAME_TO_TARGET.items():
            if key in name:
                logger.info("CM target '%s' (name fallback: '%s')", t, name)
                return t
        logger.warning(
            "Could not map XPU device to a CM target (PCI 0x%04X, architecture=%s, name=%r). "
            "Set %s=<xe2|xehpg|xehpc|...> to override.",
            pci_id,
            arch,
            getattr(props, "name", ""),
            _CM_TARGET_ENV,
        )
        return ""
    except Exception as e:
        logger.debug("CM device target detection failed: %s", e)
        return ""


# --- Input/output dtype handling -------------------------------------------

# Name dumped by the host harness for the (single) output tensor.
_OUTPUT_FILE = "output_0.bin"

_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float32": torch.float32, "float": torch.float32, "fp32": torch.float32, "f32": torch.float32,
    "float16": torch.float16, "half": torch.float16, "fp16": torch.float16, "f16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "int8": torch.int8, "uint8": torch.uint8, "int32": torch.int32, "int": torch.int32,
}

_OUTPUT_NP: dict[str, np.dtype] = {
    "float32": np.float32, "float": np.float32, "fp32": np.float32, "f32": np.float32,
    "float16": np.float16, "half": np.float16, "fp16": np.float16, "f16": np.float16,
    "int8": np.int8, "uint8": np.uint8, "int32": np.int32, "int": np.int32,
}


def _to_torch_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """Resolve a torch.dtype or dtype name string to a torch.dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower().replace("torch.", "")
    if key in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[key]
    raise ValueError(f"Unsupported CM dtype: {dtype!r}")


def _resolve_np_dtype(dtype: torch.dtype | str | np.dtype) -> np.dtype:
    """Resolve an output dtype (str/torch/np) to a numpy dtype for loading.

    Raises ValueError on dtypes numpy cannot represent natively (e.g. bfloat16):
    output_<i>.bin is raw bytes with no dtype tag, so a wrong numpy dtype would
    silently reinterpret the bytes and corrupt the correctness comparison. Fail
    loudly instead. (bf16 outputs would need a torch round-trip, not np.fromfile.)
    """
    if isinstance(dtype, np.dtype) or (isinstance(dtype, type) and issubclass(dtype, np.generic)):
        return dtype
    key = str(dtype).lower().replace("torch.", "")
    if key in _OUTPUT_NP:
        return _OUTPUT_NP[key]
    raise ValueError(
        f"Unsupported CM output dtype {dtype!r}: numpy cannot load it via np.fromfile "
        f"(supported: {sorted(set(_OUTPUT_NP))}). bf16 outputs need a torch round-trip. "
        "Pass output_dtype explicitly to compare_kernels()."
    )


def _random_tensor(shape: tuple[int, ...], dtype: torch.dtype | str) -> torch.Tensor:
    """Random tensor for a shape/dtype — randint for integer types, randn otherwise."""
    dt = _to_torch_dtype(dtype)
    shape = tuple(shape)
    if dt in (torch.int8, torch.int32):
        return torch.randint(-8, 8, shape, dtype=dt)
    if dt is torch.uint8:
        return torch.randint(0, 16, shape, dtype=dt)
    return torch.randn(shape, dtype=dt)


def _include_dirs(cm_root: str, kernel_type: KernelType = KernelType.GEMM) -> list[str]:
    """Header search paths for CM kernels.

    TODO(cm): extend with kernel-type-specific helper headers once the SDK
    layout is finalized.
    """
    dirs: list[str] = []
    if cm_root:
        dirs.append(f"{cm_root}/include")
    return dirs


@dataclass
class CMComparisonResult:
    """Result of comparing original vs optimized CM kernel performance.

    Field-compatible with :class:`SyclComparisonResult` so the optimizer's
    verify path can treat CM and SYCL identically.
    """

    original_time_ms: float
    optimized_time_ms: float
    speedup: float
    original_tflops: float | None = None
    optimized_tflops: float | None = None
    original_correct: bool = True
    optimized_correct: bool = True
    is_slower: bool = False
    feedback_message: str = ""

    @property
    def original_time_us(self) -> float:
        return self.original_time_ms * 1000

    @property
    def optimized_time_us(self) -> float:
        return self.optimized_time_ms * 1000


class CMExecutor:
    """Compiles and runs CM C++ kernels and measures performance.

    Wraps :class:`CMCompiler` for the underlying compile/run pipeline and adds
    source-string input, generic dims, shared-input correctness comparison, and
    optimization-loop feedback — the CM counterpart of ``SyclExecutor``.
    """

    def __init__(
        self,
        cm_root: str = CM_ROOT,
        device_target: str | None = None,
        compile_timeout: int = 300,
        run_timeout: int = 120,
        iterations: int = 20,
        verify: bool = True,
        kernel_type: KernelType | str = KernelType.GEMM,
    ):
        if isinstance(kernel_type, str):
            kernel_type = KernelType(kernel_type)
        self.kernel_type = kernel_type
        if device_target is None:
            device_target = _detect_device_target()
        self._compiler = CMCompiler(
            include_dirs=_include_dirs(cm_root, kernel_type),
            target_device=device_target or None,
            cm_root=cm_root,
        )
        self.iterations = iterations
        self.verify = verify
        self._build_dir: str | None = None
        self._cached_input_dir: str | None = None
        self._cached_input_key: tuple | None = None

    @property
    def build_dir(self) -> str:
        if self._build_dir is None:
            self._build_dir = tempfile.mkdtemp(prefix="cm_build_")
        return self._build_dir

    def compile(
        self,
        source_code: str | None = None,
        source_path: str | None = None,
        output_name: str = "kernel_cm",
    ) -> tuple[bool, str, str]:
        """Compile CM C++ source to a binary. Returns (success, binary, error)."""
        if source_code is not None:
            src_path = Path(self.build_dir) / f"{output_name}.cpp"
            src_path.write_text(source_code)
        elif source_path is not None:
            src_path = Path(source_path)
        else:
            return False, "", "No source code or path provided"

        src_parent = str(src_path.parent)
        if src_parent not in self._compiler.include_dirs:
            self._compiler.include_dirs.append(src_parent)

        logger.info("Compiling CM kernel: %s", src_path)
        binary = self._compiler.compile(src_path)
        if binary is None:
            err = self._compiler.last_compile_error or "Compilation failed (no details)"
            return False, "", err
        logger.info("Compilation succeeded: %s", binary)
        return True, str(binary), ""

    @staticmethod
    def _gemm_specs_from_dims(
        dims: dict[str, int | float] | None,
        dtype: torch.dtype | str = torch.bfloat16,
    ) -> tuple[list[tuple[int, ...]], list[torch.dtype]]:
        """GEMM fallback: build (input_shapes, input_dtypes) for A[M,K], B[K,N].

        Used only when a caller does not pass explicit ``input_shapes`` (e.g. the
        GEMM seed). Arbitrary kernels supply their own shapes instead.
        """
        d = dims or {}
        m = int(d.get("M", d.get("N", 1024)))
        n = int(d.get("N", m))
        k = int(d.get("K", m))
        dt = _to_torch_dtype(dtype)
        return [(m, k), (k, n)], [dt, dt]

    def generate_inputs(
        self,
        output_dir: str,
        input_shapes: list[tuple[int, ...]],
        input_dtypes: list[torch.dtype | str] | None = None,
        seed: int | None = None,
    ) -> None:
        """Generate random input tensors -> input_0.bin, input_1.bin, ... (any shapes/dtypes)."""
        if seed is not None:
            set_all_seeds(seed)
        os.makedirs(output_dir, exist_ok=True)
        dtypes = list(input_dtypes) if input_dtypes else [torch.bfloat16] * len(input_shapes)
        for i, (shape, dt) in enumerate(zip(input_shapes, dtypes, strict=False)):
            tensor = _random_tensor(tuple(shape), dt)
            _save_tensor(tensor, os.path.join(output_dir, f"input_{i}.bin"))
            logger.info(
                "Generated input_%d: shape=%s dtype=%s -> %s",
                i, tuple(shape), _to_torch_dtype(dt), output_dir,
            )

    def get_or_create_inputs(
        self,
        input_shapes: list[tuple[int, ...]],
        input_dtypes: list[torch.dtype | str] | None = None,
        seed: int = 42,
    ) -> str:
        """Return a directory with deterministic input tensors, caching across calls."""
        dtypes = [_to_torch_dtype(d) for d in (input_dtypes or [torch.bfloat16] * len(input_shapes))]
        shapes = [tuple(s) for s in input_shapes]
        key = (tuple(shapes), tuple(str(d) for d in dtypes), seed)
        if self._cached_input_dir is not None and self._cached_input_key == key:
            return self._cached_input_dir
        if self._cached_input_dir is not None:
            try:
                shutil.rmtree(self._cached_input_dir)
            except Exception:
                pass
        input_dir = tempfile.mkdtemp(prefix="cm_inputs_")
        self.generate_inputs(input_dir, shapes, dtypes, seed=seed)
        self._cached_input_dir = input_dir
        self._cached_input_key = key
        return input_dir

    @staticmethod
    def load_output(path: str, dtype: np.dtype = np.float32) -> np.ndarray:
        """Load a binary tensor file dumped by the CM kernel."""
        return np.fromfile(path, dtype=dtype)

    @staticmethod
    def compare_outputs(
        output_a: np.ndarray,
        output_b: np.ndarray,
        rtol: float = 1e-2,
        atol: float = 1e-3,
    ) -> tuple[bool, str]:
        """Compare two output tensors element-wise. Returns (passed, message)."""
        if output_a.shape != output_b.shape:
            return False, f"Shape mismatch: {output_a.shape} vs {output_b.shape}"
        if np.allclose(output_a, output_b, rtol=rtol, atol=atol):
            return True, "Outputs match"
        diff = np.abs(output_a - output_b)
        num_mismatch = int(np.sum(~np.isclose(output_a, output_b, rtol=rtol, atol=atol)))
        total = output_a.size
        return False, (
            f"Outputs differ: max_diff={float(np.max(diff)):.6f}, "
            f"mean_diff={float(np.mean(diff)):.6f}, "
            f"mismatched={num_mismatch}/{total} ({100 * num_mismatch / total:.1f}%)"
        )

    def execute(
        self,
        kernel_code: str | None = None,
        kernel_path: str | None = None,
        dims: dict[str, int | float] | None = None,
        m: int = 1024,
        n: int = 1024,
        k: int = 1024,
        output_name: str = "kernel_cm",
        input_dir: str | None = None,
        output_dir: str | None = None,
    ) -> ExecutionResult:
        """Compile and run a CM kernel, returning structured results.

        ``dims`` is a generic name->int map handed to the host harness (any
        kernel, not just GEMM); m/n/k are a GEMM convenience folded into dims.
        """
        success, binary_path, err = self.compile(
            source_code=kernel_code,
            source_path=kernel_path,
            output_name=output_name,
        )
        if not success:
            return ExecutionResult(
                success=False,
                error_message=f"Compilation failed:\n{err[-2000:]}",
            )

        effective_dims = dims or {"M": m, "N": n, "K": k}
        logger.info("Running CM kernel: %s (dims=%s)", binary_path, effective_dims)
        # Skip the harness's internal verify when using file-based I/O — we
        # compare the dumped outputs in Python via compare_outputs() instead.
        use_verify = 0 if input_dir else (1 if self.verify else 0)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        result: CMRunResult = self._compiler.run(
            Path(binary_path),
            dims=effective_dims,
            iterations=self.iterations,
            verify=use_verify,
            input_dir=input_dir,
            output_dir=output_dir,
        )
        return self._to_execution_result(result)

    @staticmethod
    def _to_execution_result(r: CMRunResult) -> ExecutionResult:
        if not r.success:
            return ExecutionResult(success=False, error_message=f"Execution failed: {r.error}")
        if r.passed is False:
            return ExecutionResult(
                success=False,
                output_correct=False,
                execution_time_ms=r.time_ms,
                tflops=r.tflops,
                error_message="Correctness verification failed",
            )
        return ExecutionResult(
            success=True,
            execution_time_ms=r.time_ms,
            tflops=r.tflops,
            output_correct=r.passed,
        )

    def compare_kernels(
        self,
        original_code: str | None = None,
        optimized_code: str | None = None,
        original_path: str | None = None,
        optimized_path: str | None = None,
        m: int = 1024,
        n: int = 1024,
        k: int = 1024,
        dims: dict[str, int | float] | None = None,
        input_shapes: list[tuple[int, ...]] | None = None,
        input_dtypes: list[torch.dtype | str] | None = None,
        output_dtype: torch.dtype | str = "float32",
        rtol: float = 1e-2,
        atol: float = 1e-3,
        input_dir: str | None = None,
        seed: int = 42,
    ) -> CMComparisonResult:
        """Compare performance and correctness of original vs optimized CM kernel.

        Inputs are described generically by ``input_shapes`` / ``input_dtypes``
        (any kernel, like the Triton path). If omitted, a GEMM is assumed and the
        shapes are derived from ``dims`` (or m/n/k). Both kernels run on identical
        inputs; outputs (``output_0.bin``) are compared in numpy.
        """
        effective_dims = dims or {"M": m, "N": n, "K": k}
        if input_shapes is not None:
            spec_shapes = [tuple(s) for s in input_shapes]
            spec_dtypes = [
                _to_torch_dtype(d)
                for d in (input_dtypes or [torch.bfloat16] * len(spec_shapes))
            ]
        else:
            spec_shapes, spec_dtypes = self._gemm_specs_from_dims(effective_dims)

        caller_owns_inputs = input_dir is not None
        io_dir = tempfile.mkdtemp(prefix="cm_compare_")
        if not caller_owns_inputs:
            input_dir = self.get_or_create_inputs(spec_shapes, spec_dtypes, seed=seed)
        orig_output_dir = os.path.join(io_dir, "orig_out")
        opt_output_dir = os.path.join(io_dir, "opt_out")

        orig_result = self.execute(
            kernel_code=original_code,
            kernel_path=original_path,
            dims=effective_dims,
            output_name="original_cm",
            input_dir=input_dir,
            output_dir=orig_output_dir,
        )
        opt_result = self.execute(
            kernel_code=optimized_code,
            kernel_path=optimized_path,
            dims=effective_dims,
            output_name="optimized_cm",
            input_dir=input_dir,
            output_dir=opt_output_dir,
        )

        if not orig_result.success:
            return CMComparisonResult(
                original_time_ms=float("inf"),
                optimized_time_ms=float("inf"),
                speedup=0.0,
                original_correct=False,
                feedback_message=f"FAILURE: Original kernel failed: {orig_result.error_message}",
            )
        if not opt_result.success:
            return CMComparisonResult(
                original_time_ms=orig_result.execution_time_ms or float("inf"),
                optimized_time_ms=float("inf"),
                speedup=0.0,
                optimized_correct=False,
                feedback_message=(
                    f"FAILURE: Optimized kernel failed: {opt_result.error_message}. "
                    "Fix compilation or runtime errors."
                ),
            )

        orig_ms = orig_result.execution_time_ms or float("inf")
        opt_ms = opt_result.execution_time_ms or float("inf")
        speedup = orig_ms / opt_ms if opt_ms > 0 else 0.0
        is_slower = speedup < 1.0
        orig_tflops = orig_result.tflops
        opt_tflops = opt_result.tflops

        # Correctness: compare dumped outputs (output_0.bin) when available.
        opt_correct = True
        correctness_msg = ""
        np_dt = _resolve_np_dtype(output_dtype)
        orig_out = os.path.join(orig_output_dir, _OUTPUT_FILE)
        opt_out = os.path.join(opt_output_dir, _OUTPUT_FILE)
        if os.path.exists(orig_out) and os.path.exists(opt_out):
            passed, detail = self.compare_outputs(
                self.load_output(orig_out, np_dt), self.load_output(opt_out, np_dt), rtol=rtol, atol=atol
            )
            opt_correct = passed
            correctness_msg = f" CORRECTNESS FAILED: {detail}." if not passed else " Correctness: PASSED."
            logger.info("Output comparison (rtol=%s, atol=%s): %s", rtol, atol, detail)
        else:
            correctness_msg = " (no output files for comparison)"
            logger.warning("Output dump files not found — skipping correctness check")

        try:
            shutil.rmtree(io_dir)
        except Exception:
            pass

        if not opt_correct:
            msg = (
                f"CORRECTNESS FAILURE: Optimized kernel produces wrong results. "
                f"{correctness_msg.strip()} Original: {orig_ms:.4f}ms, Optimized: {opt_ms:.4f}ms. "
                "Fix numerical correctness before optimizing for speed."
            )
        elif is_slower:
            slowdown = 1.0 / speedup if speedup > 0 else float("inf")
            msg = (
                f"PERFORMANCE REGRESSION: Optimized kernel is {slowdown:.2f}x SLOWER. "
                f"Original: {orig_ms:.4f}ms, Optimized: {opt_ms:.4f}ms. "
                f"{correctness_msg.strip()} Try a different approach."
            )
        elif speedup >= 2.0:
            msg = (
                f"SUCCESS: Excellent! {speedup:.2f}x speedup. "
                f"Original: {orig_ms:.4f}ms, Optimized: {opt_ms:.4f}ms.{correctness_msg}"
            )
        elif speedup >= 1.2:
            msg = (
                f"SUCCESS: Good {speedup:.2f}x speedup. "
                f"Original: {orig_ms:.4f}ms, Optimized: {opt_ms:.4f}ms. "
                f"{correctness_msg.strip()} Consider further optimizations."
            )
        else:
            msg = (
                f"MARGINAL: Only {speedup:.2f}x speedup. "
                f"Original: {orig_ms:.4f}ms, Optimized: {opt_ms:.4f}ms. "
                f"{correctness_msg.strip()} Try more aggressive optimizations."
            )

        return CMComparisonResult(
            original_time_ms=orig_ms,
            optimized_time_ms=opt_ms,
            speedup=speedup,
            original_tflops=orig_tflops,
            optimized_tflops=opt_tflops,
            original_correct=True,
            optimized_correct=opt_correct,
            is_slower=is_slower,
            feedback_message=msg,
        )

    def __del__(self):
        for d in (self._build_dir, self._cached_input_dir):
            if d is not None:
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass
