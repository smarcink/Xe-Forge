"""
Tests for the CM kernel launcher.

Two layers:

* Unit tests with a mocked ``subprocess.Popen`` exercise ``CMCompiler.run`` in
  isolation — manifest construction, the hang/TDR timeout-and-kill path, and
  worker-reported failures — with no GPU or pyopencl required.
* GPU-gated integration tests actually compile + run a CM kernel via the worker.
  They skip unless an Intel OpenCL platform is present.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from xe_forge.core.cm_compiler import (
    CMCompiler,
    _ordered_inputs,
    _parse_result,
    _scalar_spec,
)
from xe_forge.core.cm_grid import GridConfig
from xe_forge.core.cm_worker import RESULT_PREFIX

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_GEMM = REPO_ROOT / "test_kernels" / "200_CM_Gemm.cpp"
OPT_GEMM = REPO_ROOT / "test_kernels" / "203_CM_Gemm_Pipelined.cpp"


# --------------------------------------------------------------------------- #
# Mocked-subprocess unit tests (no GPU)                                        #
# --------------------------------------------------------------------------- #


class _FakePopen:
    """Stand-in for ``subprocess.Popen`` used to drive ``CMCompiler.run``."""

    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0, timeout: bool = False):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._timeout = timeout
        self.pid = 4321
        self.kill_called = False

    def communicate(self, timeout=None):
        if self._timeout:
            raise subprocess.TimeoutExpired(cmd="cm_worker", timeout=timeout)
        return self._stdout, self._stderr

    def kill(self):
        self.kill_called = True


def _make_compiler() -> CMCompiler:
    return CMCompiler(hang_timeout=5)


def _src(tmp_path: Path) -> Path:
    p = tmp_path / "kernel.cpp"
    p.write_text("// dummy CM source; worker is mocked\n")
    return p


def _grid() -> GridConfig:
    return GridConfig((32, 16, 1), (1, 1, 1))


def test_run_builds_expected_manifest(tmp_path):
    """A successful run writes a manifest with the inputs->outputs->scalars ABI."""
    src = _src(tmp_path)
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "input_0.bin").write_bytes(b"\x00" * 8)
    (input_dir / "input_1.bin").write_bytes(b"\x00" * 8)
    out_dir = tmp_path / "out"

    fake = _FakePopen(
        stdout=RESULT_PREFIX + json.dumps({"success": True, "time_ms": 0.123, "entry": "cm_gemm"})
    )
    with mock.patch("xe_forge.core.cm_compiler.subprocess.Popen", return_value=fake) as mpopen:
        result = _make_compiler().run(
            src,
            dims={"M": 256, "N": 128, "K": 64},
            grid=_grid(),
            output_sizes=[256 * 128 * 4],
            input_dir=str(input_dir),
            output_dir=str(out_dir),
            iterations=20,
            warmup=3,
        )

    assert result.success is True
    assert result.time_ms == pytest.approx(0.123)

    # Worker was launched as `python -m xe_forge.core.cm_worker <manifest>`.
    cmd = mpopen.call_args.args[0]
    assert cmd[1:3] == ["-m", "xe_forge.core.cm_worker"]

    manifest = json.loads((out_dir / "cm_launch.json").read_text())
    assert manifest["source_path"] == str(src)
    assert manifest["build_options"] == "-cmc"
    assert manifest["inputs"] == ["input_0.bin", "input_1.bin"]
    assert manifest["outputs"] == [{"file": "output_0.bin", "bytes": 256 * 128 * 4}]
    assert manifest["scalars"] == [
        {"value": 256, "type": "int32"},
        {"value": 128, "type": "int32"},
        {"value": 64, "type": "int32"},
    ]
    assert manifest["grid"] == {"global": [32, 16, 1], "local": [1, 1, 1]}
    assert manifest["iterations"] == 20
    assert manifest["warmup"] == 3


def test_run_timeout_kills_worker(tmp_path):
    """A hung kernel trips the timeout, force-kills the tree, and fails gracefully."""
    src = _src(tmp_path)
    fake = _FakePopen(timeout=True)
    with (
        mock.patch("xe_forge.core.cm_compiler.subprocess.Popen", return_value=fake),
        mock.patch("xe_forge.core.cm_compiler._kill_process_tree") as mkill,
    ):
        result = _make_compiler().run(
            src,
            dims={"M": 256, "N": 256, "K": 256},
            grid=_grid(),
            output_sizes=[4],
            output_dir=str(tmp_path / "out"),
        )

    assert result.success is False
    assert "hang/TDR" in result.error
    mkill.assert_called_once()


def test_run_reports_worker_failure(tmp_path):
    """A worker compile failure is surfaced with its stage and message."""
    src = _src(tmp_path)
    fake = _FakePopen(
        stdout=RESULT_PREFIX
        + json.dumps({"success": False, "stage": "compile", "error": "build failed: bad token"}),
        returncode=1,
    )
    with mock.patch("xe_forge.core.cm_compiler.subprocess.Popen", return_value=fake):
        result = _make_compiler().run(
            src,
            dims={"M": 256, "N": 256, "K": 256},
            grid=_grid(),
            output_sizes=[4],
            output_dir=str(tmp_path / "out"),
        )

    assert result.success is False
    assert "compile" in result.error
    assert "bad token" in result.error


def test_run_no_result_line(tmp_path):
    """If the worker emits no sentinel line, run() fails (not hangs)."""
    src = _src(tmp_path)
    fake = _FakePopen(stdout="some driver chatter\nno sentinel here\n", stderr="boom", returncode=1)
    with mock.patch("xe_forge.core.cm_compiler.subprocess.Popen", return_value=fake):
        result = _make_compiler().run(
            src,
            dims={"M": 256, "N": 256, "K": 256},
            grid=_grid(),
            output_sizes=[4],
            output_dir=str(tmp_path / "out"),
        )

    assert result.success is False
    assert "no result" in result.error.lower()


def test_run_missing_source(tmp_path):
    """A nonexistent source path fails before any subprocess is spawned."""
    with mock.patch("xe_forge.core.cm_compiler.subprocess.Popen") as mpopen:
        result = _make_compiler().run(
            tmp_path / "does_not_exist.cpp",
            dims={"M": 1, "N": 1, "K": 1},
            grid=_grid(),
            output_sizes=[4],
        )
    assert result.success is False
    assert "not found" in result.error
    mpopen.assert_not_called()


# --------------------------------------------------------------------------- #
# Pure helper tests                                                            #
# --------------------------------------------------------------------------- #


def test_ordered_inputs_stops_at_first_gap(tmp_path):
    for i in (0, 1, 2):
        (tmp_path / f"input_{i}.bin").write_bytes(b"x")
    (tmp_path / "input_4.bin").write_bytes(b"x")  # gap at 3 -> 4 must be ignored
    assert _ordered_inputs(str(tmp_path)) == ["input_0.bin", "input_1.bin", "input_2.bin"]
    assert _ordered_inputs(None) == []


def test_scalar_spec_typing():
    assert _scalar_spec(256) == {"value": 256, "type": "int32"}
    assert _scalar_spec(True) == {"value": 1, "type": "int32"}  # bool -> int32
    assert _scalar_spec(1.5) == {"value": 1.5, "type": "float32"}


def test_parse_result_scans_from_end():
    stdout = "asm count: 123\n" + RESULT_PREFIX + json.dumps({"success": True, "time_ms": 1.0})
    assert _parse_result(stdout) == {"success": True, "time_ms": 1.0}
    assert _parse_result("no sentinel\n") is None
    assert _parse_result("") is None


# --------------------------------------------------------------------------- #
# GPU-gated integration tests                                                  #
# --------------------------------------------------------------------------- #


def _intel_ocl_available() -> bool:
    try:
        import pyopencl as cl
    except Exception:
        return False
    try:
        return any(
            "intel" in p.name.lower() and p.get_devices() for p in cl.get_platforms()
        )
    except Exception:
        return False


requires_intel_ocl = pytest.mark.skipif(
    not _intel_ocl_available(), reason="no Intel OpenCL platform / pyopencl not installed"
)


@requires_intel_ocl
def test_seed_gemm_runs_and_self_compares():
    """End-to-end: compile + run the seed GEMM via the worker; orig vs orig matches."""
    from xe_forge.core.cm_executor import CMExecutor

    executor = CMExecutor(hang_timeout=30)
    result = executor.compare_kernels(
        original_path=str(SEED_GEMM),
        optimized_path=str(SEED_GEMM),
        dims={"M": 256, "N": 256, "K": 256},
        input_shapes=[(256, 256), (256, 256)],
        input_dtypes=["float16", "float16"],
        output_shapes=[(256, 256)],
        output_dtypes=["float16"],
        flop=2 * 256 * 256 * 256,
        rtol=1e-2,
        atol=1e-2,
    )

    assert result.original_correct, result.feedback_message
    assert result.optimized_correct, result.feedback_message
    assert result.original_time_ms > 0.0
    assert result.original_tflops and result.original_tflops > 0.0
    # Identical kernels: speedup hovers around 1.0 (timing noise); sanity-band it.
    assert 0.25 < result.speedup < 4.0, result.feedback_message


@requires_intel_ocl
def test_regblocked_gemm_matches_seed():
    """The hand-written register-blocked GEMM (203) matches the naive seed (200).

    Proves an optimized hand-written kernel runs end-to-end through the worker and
    produces the same result as the one-thread-per-tile seed. 203 uses a larger
    per-thread tile (BLOCK_M x BLOCK_N) but the same one-tile-per-thread model
    (GROUP = 1), so its grid comes from its own spec. Both kernels are fp16 in /
    fp16 accumulate / fp16 out, so the comparison uses an fp16 output.
    """
    from xe_forge.core.cm_executor import CMExecutor
    from xe_forge.core.spec_loader import load_spec

    spec = load_spec(REPO_ROOT / "test_kernels" / "203_CM_Gemm_Pipelined.yaml")
    executor = CMExecutor(hang_timeout=30)
    executor.grid_spec = spec.grid
    result = executor.compare_kernels(
        original_path=str(SEED_GEMM),
        optimized_path=str(OPT_GEMM),
        dims={"M": 256, "N": 256, "K": 256},
        input_shapes=[(256, 256), (256, 256)],
        input_dtypes=["float16", "float16"],
        output_shapes=[(256, 256)],
        output_dtypes=["float16"],
        flop=2 * 256 * 256 * 256,
        # Both kernels accumulate in fp16 but with different tiling / summation
        # order, so the results differ by fp16 rounding drift. Most elements are
        # within ~1% (rtol), but outputs whose true value is near zero suffer
        # catastrophic cancellation where the absolute fp16 error (~0.2 at K=256)
        # dwarfs the tiny magnitude — hence the loose atol. Both kernels are
        # correct; this tolerance accepts fp16-accumulation drift, not a bug.
        rtol=5e-2,
        atol=0.5,
    )

    # 203 (optimized) is compared against the seed (original): equal => correct.
    assert result.optimized_correct, result.feedback_message
    assert result.optimized_time_ms and result.optimized_time_ms > 0.0


def test_compare_kernels_without_input_shapes_fails_loud():
    """Missing input_shapes is a hard error, not a fabricated GEMM fallback."""
    from xe_forge.core.cm_executor import CMExecutor

    executor = CMExecutor(hang_timeout=30)
    with pytest.raises(ValueError, match="input_shapes"):
        executor.compare_kernels(
            original_path=str(SEED_GEMM),
            optimized_path=str(SEED_GEMM),
            dims={"M": 256, "N": 256, "K": 256},
            output_shapes=[(256, 256)],
        )


# A data-dependent infinite loop the CM frontend won't optimize away. Running it
# can trip a real GPU TDR that briefly resets the display, so it is opt-in.
_HANG_KERNEL = """\
#include <cm/cm.h>
extern "C" _GENX_MAIN_ void
cm_hang(SurfaceIndex surfD [[type("buffer_t")]], int M, int N, int K) {
  volatile int x = 1;
  int acc = M;
  while (x) {
    acc += N;
    cm_store<int, 1>(surfD, 0, acc);
  }
}
"""


@requires_intel_ocl
@pytest.mark.skipif(
    os.environ.get("XE_CM_TDR_TEST") != "1",
    reason="real GPU hang can trigger a display TDR; set XE_CM_TDR_TEST=1 to run",
)
def test_hanging_kernel_is_killed(tmp_path):
    """A wedged kernel is force-killed at the timeout; the parent survives."""
    src = tmp_path / "hang.cpp"
    src.write_text(_HANG_KERNEL)

    result = CMCompiler(hang_timeout=5).run(
        src,
        dims={"M": 256, "N": 256, "K": 256},
        grid=GridConfig((1, 1, 1), (1, 1, 1)),
        output_sizes=[4],
        output_dir=str(tmp_path / "out"),
    )

    assert result.success is False
    assert "hang/TDR" in result.error
