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

from xe_forge.core.cm_compiler import CMCompiler, CMRunResult
from xe_forge.core.cm_grid import compute_grid, parse_build_directives
from xe_forge.core.sycl_executor import _save_tensor
from xe_forge.core.xpu_query import xmx_enabled
from xe_forge.models import ExecutionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CMDeviceCaps:
    """Hardware features that gate which CM codegen patterns are usable.

    DPAS is assumed available on every Intel Xe target: the driver's
    ``has_subgroup_matrix_multiply_accumulate`` flag is unreliable (under-reported
    on some Battlemage runtime builds) and is deliberately not consulted. The
    DPAS path is suppressed only when explicitly opted out via the
    ``XE_FORGE_NO_XMX`` env var — e.g. to target Xe-LPG / Meteor Lake / Arrow
    Lake, which have no XMX systolic array.
    """

    has_dpas: bool = True  # XMX systolic matmul (cm_dpas); opt out via XE_FORGE_NO_XMX


def _detect_device_capabilities() -> CMDeviceCaps:
    """Resolve XMX/DPAS availability for CM codegen.

    DPAS is assumed available rather than probed from the buggy driver flag; opt
    into the no-DPAS path (vector FMA / dp4a) with ``XE_FORGE_NO_XMX=1``.
    """
    has_dpas = xmx_enabled()
    logger.info("CM device capabilities: XMX/DPAS=%s", "yes" if has_dpas else "NO")
    return CMDeviceCaps(has_dpas=has_dpas)


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
        hang_timeout: int = 60,
        iterations: int = 20,
    ):
        self.device_caps = _detect_device_capabilities()
        self._compiler = CMCompiler(hang_timeout=hang_timeout)
        self.iterations = iterations
        self.grid_spec: dict | None = None
        self._build_dir: str | None = None
        self._cached_input_dir: str | None = None
        self._cached_input_key: tuple | None = None

    @property
    def build_dir(self) -> str:
        if self._build_dir is None:
            self._build_dir = tempfile.mkdtemp(prefix="cm_build_")
        return self._build_dir

    def _resolve_output_sizes(
        self,
        output_shapes: list[tuple[int, ...]] | None,
        output_dtypes: list[torch.dtype | str] | None,
    ) -> list[int]:
        """Byte size of each output buffer the worker allocates and dumps.

        Output shapes/dtypes come from the spec's ``outputs:`` section.
        """
        if not output_shapes:
            raise ValueError(
                "CM output sizes require explicit output_shapes from the spec "
                "'outputs:' section; none were provided."
            )
        shapes = [tuple(s) for s in output_shapes]
        dtypes = [
            _to_torch_dtype(d)
            for d in (output_dtypes or [torch.float32] * len(shapes))
        ]
        return [
            int(np.prod(shape)) * torch.empty((), dtype=dt).element_size()
            for shape, dt in zip(shapes, dtypes, strict=False)
        ]

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
        *,
        dtype: np.dtype | None = None,
    ) -> tuple[bool, str]:
        """Compare two output tensors for numerical equivalence.

        Operation-agnostic by design: it flattens the tensors and compares them
        with a RELATIVE L2 (Euclidean) norm — ``‖b - a‖ / ‖a‖`` — rather than an
        element-wise ``allclose``. This is the generic correctness gate for an
        optimizer that may legitimately change the *order* or *precision* of a
        reduction:

        * Scale-invariant — the normalized error means the same thing whether the
          outputs are ~1 or ~1e6, so no per-kernel ``atol`` tuning is needed.
        * Robust to low-precision (fp16/bf16) reordering drift, which spreads
          small errors across many elements but stays tiny *in aggregate*.
        * Robust to near-zero catastrophic cancellation: an output element whose
          true value is ~0 has a huge element-wise relative error but contributes
          almost nothing to the norm (element-wise ``allclose`` fails on it; the
          norm shrugs it off).
        * Still catches real bugs — a wrong scale, a systematic offset, or a large
          localized error all move the norm well past the threshold. (Unlike
          cosine similarity, the L2 norm is NOT magnitude-blind, so a kernel that
          is 2x the reference everywhere correctly fails.)

        ``a`` is the reference (the original kernel's output). The pass threshold
        is ``rtol``; when ``dtype`` is given and the caller left ``rtol`` at the
        strict default, a precision-appropriate default is used instead (fp16/bf16
        tolerate more reordering drift than fp32). NaN/Inf in the optimized output
        is always a failure. ``atol`` is accepted for signature compatibility and
        only used as a floor on the reference norm.
        """
        if output_a.shape != output_b.shape:
            return False, f"Shape mismatch: {output_a.shape} vs {output_b.shape}"

        a = output_a.astype(np.float64).ravel()
        b = output_b.astype(np.float64).ravel()

        # NaN / Inf in the candidate is unambiguously wrong, whatever the metric.
        if not np.all(np.isfinite(b)):
            n_bad = int(np.sum(~np.isfinite(b)))
            return False, f"Optimized output has {n_bad} non-finite value(s) (NaN/Inf)"

        # Precision-appropriate default threshold (operation-agnostic — keyed only
        # on the output dtype, which the spec declares). Used when the caller kept
        # the strict 1e-2 default; an explicit override always wins.
        threshold = rtol
        if dtype is not None and abs(rtol - 1e-2) < 1e-12:
            name = np.dtype(dtype).name
            if name in ("float16", "bfloat16"):
                threshold = 5e-2  # half/bf16: long reductions drift more
            elif name == "float32":
                threshold = 1e-2

        ref_norm = float(np.linalg.norm(a))
        err_norm = float(np.linalg.norm(b - a))
        # Guard a near-zero reference (all-zero output): fall back to an absolute
        # norm floor so we don't divide by ~0.
        denom = ref_norm if ref_norm > atol else max(atol, 1.0)
        rel_l2 = err_norm / denom

        if rel_l2 <= threshold:
            return True, f"Outputs match (rel-L2={rel_l2:.2e} <= {threshold:.2e})"

        diff = np.abs(b - a)
        max_diff = float(np.max(diff)) if diff.size else 0.0
        mean_diff = float(np.mean(diff)) if diff.size else 0.0
        return False, (
            f"Outputs differ: rel-L2={rel_l2:.4e} > tol={threshold:.2e} "
            f"(max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})"
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
        input_shapes: list[tuple[int, ...]] | None = None,
        input_dtypes: list[torch.dtype | str] | None = None,
        output_shapes: list[tuple[int, ...]] | None = None,
        output_dtypes: list[torch.dtype | str] | None = None,
        flop: float | None = None,
        seed: int = 42,
    ) -> ExecutionResult:
        """Compile (online ``-cmc`` in an isolated worker) and run a CM kernel.

        ``dims`` is a generic name->int map whose values become the kernel's
        trailing scalar args; m/n/k are a scalar-dim convenience folded into
        dims. When ``input_dir`` is omitted, inputs are materialized from
        ``input_shapes``/``input_dtypes``; output buffer sizes come from
        ``output_shapes``/``output_dtypes``. ``flop`` (when given) yields TFLOPS.
        """
        if kernel_code is not None:
            src_path = Path(self.build_dir) / f"{output_name}.cpp"
            src_path.write_text(kernel_code)
        elif kernel_path is not None:
            src_path = Path(kernel_path)
        else:
            return ExecutionResult(success=False, error_message="No source code or path provided")

        effective_dims = dims or {"M": m, "N": n, "K": k}
        kernel_source = kernel_code if kernel_code is not None else src_path.read_text()
        try:
            grid = compute_grid(kernel_source, self.grid_spec, effective_dims)
        except ValueError as e:
            return ExecutionResult(success=False, error_message=f"Grid computation failed: {e}")

        # Materialize inputs from the spec's input_shapes unless the caller
        # supplied a prepared input_dir.
        if input_dir is None:
            if not input_shapes:
                raise ValueError(
                    "CM execute() needs either input_dir or explicit input_shapes "
                    "from the spec 'inputs:' section to bind kernel input buffers; "
                    "neither was provided."
                )
            input_dir = self.get_or_create_inputs(input_shapes, input_dtypes, seed=seed)

        output_sizes = self._resolve_output_sizes(output_shapes, output_dtypes)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Honor an in-source ``// xe-forge-build:`` directive (e.g. the GRF
        # register-file size chosen by the autotuner) so a saved kernel compiles
        # the same way the autotuner measured it.
        extra_build_options = parse_build_directives(kernel_source)

        logger.info("Running CM kernel: %s (dims=%s)", src_path, effective_dims)
        result: CMRunResult = self._compiler.run(
            src_path,
            dims=effective_dims,
            grid=grid,
            output_sizes=output_sizes,
            input_dir=input_dir,
            output_dir=output_dir,
            iterations=self.iterations,
            extra_build_options=extra_build_options,
        )
        return self._to_execution_result(result, flop=flop)

    @staticmethod
    def _to_execution_result(r: CMRunResult, flop: float | None = None) -> ExecutionResult:
        if not r.success:
            return ExecutionResult(success=False, error_message=f"Execution failed: {r.error}")
        tflops = r.tflops
        if tflops is None and flop and r.time_ms and r.time_ms > 0:
            tflops = flop / (r.time_ms * 1e-3) / 1e12
        if r.passed is False:
            return ExecutionResult(
                success=False,
                output_correct=False,
                execution_time_ms=r.time_ms,
                tflops=tflops,
                error_message="Correctness verification failed",
            )
        return ExecutionResult(
            success=True,
            execution_time_ms=r.time_ms,
            tflops=tflops,
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
        output_shapes: list[tuple[int, ...]] | None = None,
        output_dtypes: list[torch.dtype | str] | None = None,
        flop: float | None = None,
        rtol: float = 1e-2,
        atol: float = 1e-3,
        input_dir: str | None = None,
        seed: int = 42,
    ) -> CMComparisonResult:
        """Compare performance and correctness of original vs optimized CM kernel.

        Inputs and outputs are described by the spec's
        ``input_shapes``/``input_dtypes`` and ``output_shapes``/``output_dtypes``.
        Both kernels run on identical inputs; the first output
        (``output_0.bin``) is compared in numpy. ``flop`` (when given) yields
        the original/optimized TFLOPS.
        """
        effective_dims = dims or {"M": m, "N": n, "K": k}
        if not input_shapes:
            raise ValueError(
                "CM compare_kernels() requires explicit input_shapes from the "
                "spec 'inputs:' section; none were provided."
            )
        spec_shapes = [tuple(s) for s in input_shapes]
        spec_dtypes = [
            _to_torch_dtype(d)
            for d in (input_dtypes or [torch.bfloat16] * len(spec_shapes))
        ]

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
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            flop=flop,
        )
        opt_result = self.execute(
            kernel_code=optimized_code,
            kernel_path=optimized_path,
            dims=effective_dims,
            output_name="optimized_cm",
            input_dir=input_dir,
            output_dir=opt_output_dir,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            flop=flop,
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
        # The first output's dtype drives the numpy reinterpret of output_0.bin.
        np_dt = _resolve_np_dtype(output_dtypes[0] if output_dtypes else "float32")
        orig_out = os.path.join(orig_output_dir, _OUTPUT_FILE)
        opt_out = os.path.join(opt_output_dir, _OUTPUT_FILE)
        if os.path.exists(orig_out) and os.path.exists(opt_out):
            passed, detail = self.compare_outputs(
                self.load_output(orig_out, np_dt),
                self.load_output(opt_out, np_dt),
                rtol=rtol,
                atol=atol,
                dtype=np_dt,
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
