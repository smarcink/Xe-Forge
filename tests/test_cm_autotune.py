"""
Tests for the CM autotuning sweep and its source-rewriting helpers.

These tests are torch-free and GPU-free: the sweep talks to a fake executor that
returns canned timings, so we exercise candidate validation, fail-soft handling,
best-pick selection, the noise gate, and the #define / build-directive rewriting
without compiling or running anything.
"""

from xe_forge.core.cm_autotune import normalize_configs, sweep_cm
from xe_forge.core.cm_grid import (
    BUILD_DIRECTIVE_PREFIX,
    extract_defines,
    parse_build_directives,
    rewrite_defines,
    stamp_build_directive,
)

BASE = """#include <cm/cm.h>

#define BLOCK_M 8
#define BLOCK_N 16
#define BLOCK_K 16
#define SCALE (BLOCK_M * 2)

extern "C" _GENX_MAIN_ void cm_gemm(SurfaceIndex a, int M) {
  // body uses BLOCK_M / BLOCK_N / BLOCK_K
}
"""


# --------------------------------------------------------------------------- #
# Source-rewriting helpers
# --------------------------------------------------------------------------- #
def test_rewrite_defines_replaces_only_named_int_macros():
    out = rewrite_defines(BASE, {"BLOCK_M": 32, "BLOCK_N": 64})
    d = extract_defines(out)
    assert d["BLOCK_M"] == 32
    assert d["BLOCK_N"] == 64
    assert d["BLOCK_K"] == 16  # untouched
    # The expression-valued macro must never be rewritten, even if named.
    assert "#define SCALE (BLOCK_M * 2)" in out


def test_rewrite_defines_ignores_unknown_and_expression_macros():
    # SCALE is an expression macro; an attempt to override it is ignored.
    out = rewrite_defines(BASE, {"NOPE": 5, "SCALE": 9})
    assert "#define SCALE (BLOCK_M * 2)" in out
    assert "NOPE" not in out
    assert out == BASE  # nothing changed


def test_rewrite_defines_empty_is_noop():
    assert rewrite_defines(BASE, {}) == BASE


def test_parse_build_directives_allowlist():
    src = (
        f"{BUILD_DIRECTIVE_PREFIX} -Qxcm_register_file_size=256 -rf-evil-flag\n"
        + BASE
    )
    toks = parse_build_directives(src)
    assert toks == ["-Qxcm_register_file_size=256"]  # evil flag dropped


def test_stamp_build_directive_roundtrip_and_idempotent():
    stamped = stamp_build_directive(BASE, ["-Qxcm_register_file_size=128"])
    assert parse_build_directives(stamped) == ["-Qxcm_register_file_size=128"]
    # Re-stamping replaces rather than appends.
    restamped = stamp_build_directive(stamped, ["-Qxcm_register_file_size=256"])
    assert parse_build_directives(restamped) == ["-Qxcm_register_file_size=256"]
    assert restamped.count(BUILD_DIRECTIVE_PREFIX) == 1
    # Empty tokens strip the directive entirely.
    assert parse_build_directives(stamp_build_directive(stamped, [])) == []


# --------------------------------------------------------------------------- #
# Config normalization
# --------------------------------------------------------------------------- #
def test_normalize_configs_prunes_coerces_dedupes_and_caps():
    tunable = {"BLOCK_M", "BLOCK_N", "BLOCK_K"}
    configs = [
        {"BLOCK_N": 32, "grf": 256},          # ok
        {"BLOCK_N": "64"},                     # str coerced to int
        {"BLOCK_N": 32, "grf": 256},          # duplicate of first
        {"UNKNOWN": 4, "BLOCK_M": 16},        # unknown key pruned, keeps BLOCK_M
        {"BLOCK_K": "abc"},                    # non-int dropped -> empty -> skipped
        {},                                     # empty -> skipped
    ]
    out = normalize_configs(configs, tunable, max_candidates=10)
    assert ({"BLOCK_N": 32}, 256) in out
    assert ({"BLOCK_N": 64}, None) in out
    assert ({"BLOCK_M": 16}, None) in out
    assert len(out) == 3  # dupes/empties removed

    capped = normalize_configs(
        [{"BLOCK_N": n} for n in range(100)], tunable, max_candidates=5
    )
    assert len(capped) == 5


# --------------------------------------------------------------------------- #
# Sweep behavior (fake executor)
# --------------------------------------------------------------------------- #
class _FakeCmp:
    def __init__(self, correct=True, ms=1.0, tflops=1.0, speedup=1.0, msg=""):
        self.optimized_correct = correct
        self.optimized_time_ms = ms
        self.optimized_tflops = tflops
        self.speedup = speedup
        self.feedback_message = msg


class _FakeExecutor:
    """compare_kernels() returns canned timings keyed by the candidate BLOCK_N/GRF."""

    def __init__(self, ms_by_blockn, *, incorrect=(), raises=(), grf_factor=None):
        self.ms_by_blockn = ms_by_blockn
        self.incorrect = set(incorrect)
        self.raises = set(raises)
        self.grf_factor = grf_factor or {}
        self.calls: list[str] = []

    def compare_kernels(self, *, original_code, optimized_code, **kwargs):
        self.calls.append(optimized_code)
        d = extract_defines(optimized_code)
        bn = d.get("BLOCK_N")
        if bn in self.raises:
            raise RuntimeError("compile boom")
        grf = None
        for tok in parse_build_directives(optimized_code):
            if tok.startswith("-Qxcm_register_file_size="):
                grf = int(tok.split("=")[1])
        ms = self.ms_by_blockn.get(bn, 9.9) * self.grf_factor.get(grf, 1.0)
        return _FakeCmp(
            correct=bn not in self.incorrect,
            ms=ms,
            tflops=(1.0 / ms if ms else 0.0),
            speedup=(2.0 / ms if ms else 0.0),
        )


def test_sweep_picks_fastest_correct():
    ex = _FakeExecutor({16: 1.0, 32: 0.5, 64: 0.7})
    res = sweep_cm(
        configs=[{"BLOCK_N": 16}, {"BLOCK_N": 32}, {"BLOCK_N": 64}],
        base_code=BASE,
        baseline_code=BASE,
        executor=ex,
        dims={"M": 256, "N": 256, "K": 256},
        input_shapes=[(256, 256), (256, 256)],
    )
    assert res.candidates_ok == 3
    assert res.best_defines == {"BLOCK_N": 32}
    assert res.best_time_ms == 0.5
    assert extract_defines(res.best_code)["BLOCK_N"] == 32
    assert res.improved is True


def test_sweep_selects_grf_winner_and_stamps_directive():
    # grf=256 makes the 32-tile 20% faster; it should win and be stamped.
    ex = _FakeExecutor({32: 0.5}, grf_factor={256: 0.8, 128: 1.0})
    res = sweep_cm(
        configs=[{"BLOCK_N": 32}, {"BLOCK_N": 32, "grf": 256}],
        base_code=BASE,
        baseline_code=BASE,
        executor=ex,
        dims=None,
        input_shapes=None,
    )
    assert res.best_grf == 256
    assert abs(res.best_time_ms - 0.4) < 1e-9
    assert parse_build_directives(res.best_code) == ["-Qxcm_register_file_size=256"]


def test_sweep_is_failsoft_on_exception_and_incorrect():
    ex = _FakeExecutor({32: 0.5, 64: 0.3, 128: 0.2}, incorrect=(128,), raises=(64,))
    res = sweep_cm(
        configs=[{"BLOCK_N": 32}, {"BLOCK_N": 64}, {"BLOCK_N": 128}],
        base_code=BASE,
        baseline_code=BASE,
        executor=ex,
        dims=None,
        input_shapes=None,
    )
    # 64 raised, 128 incorrect -> only 32 is a valid winner.
    assert res.candidates_tried == 3
    assert res.candidates_ok == 1
    assert res.best_defines == {"BLOCK_N": 32}
    assert len(res.ranked) == 3


def test_sweep_improved_gate_rejects_noise():
    ex = _FakeExecutor({32: 0.5})
    # Incumbent already at 0.5ms; a 0.5ms "win" is within the noise gate.
    res = sweep_cm(
        configs=[{"BLOCK_N": 32}],
        base_code=BASE,
        baseline_code=BASE,
        executor=ex,
        dims=None,
        input_shapes=None,
        incumbent_ms=0.5,
    )
    assert res.best_defines == {"BLOCK_N": 32}
    assert res.improved is False


def test_sweep_no_valid_candidates():
    ex = _FakeExecutor({})
    res = sweep_cm(
        configs=[{"UNKNOWN": 1}],
        base_code=BASE,
        baseline_code=BASE,
        executor=ex,
        dims=None,
        input_shapes=None,
    )
    assert res.best_code is None
    assert res.candidates_ok == 0
    assert ex.calls == []  # nothing measured
    assert "No valid" in res.message
