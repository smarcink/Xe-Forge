"""
CM autotuning sweep.

The CM ``autotuning`` optimization stage proposes — via the LLM — a shortlist of
candidate tuning configurations (the kernel's grid-driving ``#define`` block
sizes plus an optional GRF register-file size) and this module *measures* them.
For each candidate it rewrites the ``#define``s, stamps the GRF build-directive,
compiles + runs the kernel against the known-good baseline for correctness,
times it, and keeps the fastest correct candidate.

CM has no runtime autotuner: block sizes are compile-time ``#define`` constants
and the GRF size is a ``clBuildProgram`` flag (``-Qxcm_register_file_size``).
So the LLM narrows the search space using hardware knowledge and the pipeline
performs the actual sweep. Measuring on the benchmark problem size (where signal
dominates timing noise) is what makes the winner meaningful.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from xe_forge.core.cm_grid import (
    extract_defines,
    rewrite_defines,
    stamp_build_directive,
)

logger = logging.getLogger(__name__)

# A swept candidate must beat the incumbent by this factor to be adopted — the
# same noise gate the optimizer uses per stage, so timing jitter is not mistaken
# for a real improvement.
_MIN_IMPROVEMENT = 1.02

# Hard cap on candidates actually measured regardless of how many the LLM
# proposes — each candidate is a full compile + run subprocess.
_DEFAULT_MAX_CANDIDATES = 24

# Keys the LLM may use for the register-file size (mapped to the GRF flag).
_GRF_KEYS = ("grf", "GRF", "grf_size", "register_file_size")


@dataclass
class CMAutotuneCandidate:
    """One measured configuration in the sweep."""

    defines: dict[str, int]
    grf: int | None
    time_ms: float | None = None
    tflops: float | None = None
    speedup: float | None = None
    correct: bool = False
    error: str = ""


@dataclass
class CMAutotuneResult:
    """Outcome of a sweep: the winning kernel plus a ranked candidate table."""

    best_code: str | None = None
    best_defines: dict[str, int] = field(default_factory=dict)
    best_grf: int | None = None
    best_time_ms: float | None = None
    best_tflops: float | None = None
    best_speedup: float | None = None
    improved: bool = False
    candidates_proposed: int = 0
    candidates_tried: int = 0
    candidates_ok: int = 0
    ranked: list[CMAutotuneCandidate] = field(default_factory=list)
    message: str = ""


def normalize_configs(
    configs: list[dict[str, Any]],
    tunable_defines: set[str],
    *,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
) -> list[tuple[dict[str, int], int | None]]:
    """Validate and dedupe LLM-proposed configs into ``(define_overrides, grf)`` pairs.

    - Keeps only ``#define`` names that actually exist in the kernel
      (``tunable_defines``); unknown names are dropped.
    - Coerces values to ``int``; a key whose value is not int-coercible is dropped.
    - Extracts the special GRF key (``grf``/``register_file_size``/...).
    - Drops empty and duplicate candidates and caps the total.
    """
    seen: set[tuple] = set()
    out: list[tuple[dict[str, int], int | None]] = []
    for cfg in configs:
        if not isinstance(cfg, dict):
            continue
        grf: int | None = None
        defines: dict[str, int] = {}
        for key, val in cfg.items():
            if key in _GRF_KEYS:
                try:
                    grf = int(val)
                except (TypeError, ValueError):
                    grf = None
                continue
            if key not in tunable_defines:
                logger.debug("autotune: dropping unknown define %r", key)
                continue
            try:
                defines[key] = int(val)
            except (TypeError, ValueError):
                logger.debug("autotune: dropping non-int define %r=%r", key, val)
        if not defines and grf is None:
            continue
        signature = (tuple(sorted(defines.items())), grf)
        if signature in seen:
            continue
        seen.add(signature)
        out.append((defines, grf))
        if len(out) >= max_candidates:
            break
    return out


def _materialize(base_code: str, defines: dict[str, int], grf: int | None) -> str:
    """Build a candidate kernel: rewritten ``#define``s + stamped GRF directive."""
    code = rewrite_defines(base_code, defines)
    if grf is not None:
        code = stamp_build_directive(code, [f"-Qxcm_register_file_size={int(grf)}"])
    return code


def sweep_cm(
    *,
    configs: list[dict[str, Any]],
    base_code: str,
    baseline_code: str,
    executor: Any,
    dims: dict[str, int] | None,
    input_shapes: list[tuple[int, ...]] | None,
    input_dtypes: list[Any] | None = None,
    output_shapes: list[tuple[int, ...]] | None = None,
    output_dtypes: list[Any] | None = None,
    flop: float | None = None,
    rtol: float = 1e-2,
    atol: float = 1e-3,
    incumbent_ms: float | None = None,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
) -> CMAutotuneResult:
    """Measure LLM-proposed CM tuning configs; return the fastest correct one.

    Args:
        configs: Raw config objects from the LLM, each mapping ``#define`` knob
            names to ints with an optional ``grf`` key.
        base_code: Kernel whose ``#define``s are swept (typically the post-stage
            optimized kernel).
        baseline_code: Known-good reference the candidates are checked against
            for correctness (typically the original seed kernel).
        executor: A ``CMExecutor``-like object exposing ``compare_kernels(...)``.
        incumbent_ms: Current best time (ms) the winner must beat by the noise
            gate to be adopted; when unknown, any correct best is accepted.

    Each candidate is compared against ``baseline_code`` so its reported
    ``speedup`` is relative to the original baseline — consistent with how the
    pipeline interprets per-stage speedups. A candidate that fails to compile or
    fails correctness is skipped (fail-soft); one bad combo never aborts the sweep.
    """
    result = CMAutotuneResult(candidates_proposed=len(configs))
    tunable = set(extract_defines(base_code))
    candidates = normalize_configs(configs, tunable, max_candidates=max_candidates)
    if not candidates:
        result.message = "No valid autotune candidates after validation."
        logger.info("CM autotune: %s", result.message)
        return result

    best: CMAutotuneCandidate | None = None
    for defines, grf in candidates:
        candidate_code = _materialize(base_code, defines, grf)
        cand = CMAutotuneCandidate(defines=defines, grf=grf)
        result.candidates_tried += 1
        try:
            cmp = executor.compare_kernels(
                original_code=baseline_code,
                optimized_code=candidate_code,
                dims=dims,
                input_shapes=input_shapes,
                input_dtypes=input_dtypes,
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
                flop=flop,
                rtol=rtol,
                atol=atol,
            )
        except Exception as e:  # fail-soft: a single bad combo must not abort
            cand.error = str(e)
            result.ranked.append(cand)
            logger.warning("autotune: candidate %s grf=%s raised: %s", defines, grf, e)
            continue

        cand.correct = bool(getattr(cmp, "optimized_correct", False))
        cand.time_ms = getattr(cmp, "optimized_time_ms", None)
        cand.tflops = getattr(cmp, "optimized_tflops", None)
        cand.speedup = getattr(cmp, "speedup", None)
        if not cand.correct:
            cand.error = getattr(cmp, "feedback_message", "") or "incorrect result"
            result.ranked.append(cand)
            logger.info("autotune: candidate %s grf=%s INCORRECT", defines, grf)
            continue

        result.candidates_ok += 1
        result.ranked.append(cand)
        logger.info(
            "autotune: candidate %s grf=%s -> %.4fms (%.3f TFlop/s)",
            defines,
            grf,
            cand.time_ms if cand.time_ms is not None else float("nan"),
            cand.tflops or 0.0,
        )
        if cand.time_ms is not None and (
            best is None or best.time_ms is None or cand.time_ms < best.time_ms
        ):
            best = cand

    result.ranked.sort(
        key=lambda c: c.time_ms if (c.correct and c.time_ms is not None) else float("inf")
    )

    if best is None:
        result.message = (
            f"Swept {result.candidates_tried} candidate(s); "
            "none compiled and passed correctness."
        )
        logger.info("CM autotune: %s", result.message)
        return result

    result.best_defines = best.defines
    result.best_grf = best.grf
    result.best_time_ms = best.time_ms
    result.best_tflops = best.tflops
    result.best_speedup = best.speedup
    result.best_code = _materialize(base_code, best.defines, best.grf)

    if incumbent_ms and best.time_ms and incumbent_ms > 0:
        result.improved = best.time_ms < incumbent_ms / _MIN_IMPROVEMENT
    else:
        result.improved = True

    result.message = (
        f"Best {best.defines} grf={best.grf}: "
        f"{best.time_ms:.4f}ms ({best.tflops or 0:.3f} TFlop/s); "
        f"{result.candidates_ok}/{result.candidates_tried} correct"
        + ("" if result.improved else " (no improvement over incumbent)")
    )
    logger.info("CM autotune: %s", result.message)
    return result
