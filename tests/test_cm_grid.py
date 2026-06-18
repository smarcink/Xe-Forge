"""
Test grid parsing and evaluation for CM kernels.
"""

from pathlib import Path

from xe_forge.core.cm_grid import (
    _check_symbols,
    _safe_eval,
    compute_grid,
    describe_grid_contract,
    extract_defines,
)
from xe_forge.core.spec_loader import load_spec


def test_grid_parser_basic():
    """Test basic grid expression parsing with ceil, min, max."""
    # Test simple division with ceil
    result = _safe_eval("ceil(256 / 8)", {})
    assert result == 32, f"Expected 32, got {result}"

    # Test with context variables
    result = _safe_eval("ceil(M / BLOCK_M)", {"M": 256, "BLOCK_M": 8})
    assert result == 32, f"Expected 32, got {result}"

    # Test min/max
    result = _safe_eval("min(256, 512)", {})
    assert result == 256, f"Expected 256, got {result}"

    result = _safe_eval("max(8, 16)", {})
    assert result == 16, f"Expected 16, got {result}"

    print("[OK] Basic grid parser tests passed")


def test_define_extraction():
    """Test extracting #define values from kernel source."""
    kernel_source = """
#include <cm/cm.h>

#define BLOCK_M 8
#define BLOCK_N 16
#define BLOCK_K 32

extern "C" _GENX_MAIN_ void
cm_gemm(SurfaceIndex A, int M, int N) {
  // kernel body
}
"""
    defines = extract_defines(kernel_source)

    assert defines == {"BLOCK_M": 8, "BLOCK_N": 16, "BLOCK_K": 32}, f"Got {defines}"
    print("[OK] Define extraction test passed")


def test_validate_references():
    """Test symbol validation."""
    dims = {"M": 256, "N": 256, "K": 256}
    defines = {"BLOCK_M": 8, "BLOCK_N": 16}

    # Valid expression
    try:
        _check_symbols("ceil(M / BLOCK_M)", dims, defines, "grid")
        print("[OK] Valid reference check passed")
    except ValueError as e:
        raise AssertionError(f"Valid expression rejected: {e}") from e

    # Invalid expression (missing BLOCK_K)
    try:
        _check_symbols("ceil(K / BLOCK_K)", dims, defines, "grid")
        raise AssertionError("Should have rejected missing BLOCK_K")
    except ValueError as e:
        assert "BLOCK_K" in str(e), f"Expected BLOCK_K in error, got: {e}"
        print("[OK] Invalid reference check passed")


def test_define_extraction_hex():
    """Hex/oct/bin #define values are parsed, function-like macros are skipped."""
    defines = extract_defines(
        "#define BLOCK_M 0x10\n"
        "#define BLOCK_N 8  // trailing comment\n"
        "#define SQUARE(x) ((x)*(x))\n"
    )
    assert defines == {"BLOCK_M": 16, "BLOCK_N": 8}, f"Got {defines}"
    print("[OK] Hex/comment/macro define extraction test passed")


def test_unsafe_expression_rejected():
    """The evaluator must reject power operators and arbitrary calls (no eval())."""
    for bad in ["9 ** 9 ** 9", "__import__('os')", "M.__class__", "open('x')"]:
        try:
            _safe_eval(bad, {"M": 256})
            raise AssertionError(f"Should have rejected unsafe expression: {bad!r}")
        except ValueError:
            pass
    print("[OK] Unsafe expression rejection test passed")


def test_missing_symbol_diagnostic():
    """A missing knob is reported as a #define gap, not lumped with problem dims."""
    kernel_source = "#define BLOCK_M 8\n"  # BLOCK_N intentionally absent
    grid_spec = {"x": "ceil(M / BLOCK_M)", "y": "ceil(N / BLOCK_N)", "z": 1}
    try:
        compute_grid(kernel_source, grid_spec, {"M": 256, "N": 256})
        raise AssertionError("Should have rejected missing BLOCK_N")
    except ValueError as e:
        msg = str(e)
        assert "BLOCK_N" in msg, f"Expected BLOCK_N in error, got: {msg}"
        # The message must distinguish kernel #defines (BLOCK_M present) from dims.
        assert "BLOCK_M" in msg and "#define" in msg, f"Diagnostic not actionable: {msg}"
        print("[OK] Missing-symbol diagnostic test passed")


def test_grid_contract_names_actual_knobs():
    """The prompt contract lists the real knob names from the formula, not hardcoded ones."""
    # Custom knob names (not BLOCK_*) to prove extraction is name-agnostic.
    kernel_source = "#define TILE_ROWS 16\n#define TILE_COLS 32\n"
    grid_spec = {"x": "ceil(M / TILE_ROWS)", "y": "ceil(N / TILE_COLS)", "z": 1}
    contract = describe_grid_contract(grid_spec, kernel_source)

    assert "TILE_ROWS = 16" in contract, contract
    assert "TILE_COLS = 32" in contract, contract
    # Problem dims are reported as fixed, not as tunable knobs.
    assert "M, N" in contract, contract
    # Hardening language must be present.
    assert "do NOT rename" in contract, contract

    # No grid_spec -> default BLOCK_M/BLOCK_N contract from the kernel's defines.
    default_contract = describe_grid_contract(None, "#define BLOCK_M 8\n#define BLOCK_N 16\n")
    assert "BLOCK_M = 8" in default_contract, default_contract
    assert "BLOCK_N = 16" in default_contract, default_contract
    print("[OK] Grid contract knob-naming test passed")


def test_cooperative_grid_multiply():
    """A local (work-group) block groups threads without changing tile count.

    Each thread owns one output tile via cm_global_id; GROUP_M only controls how
    those threads are partitioned into groups (so adjacent tiles that share an
    input sub-tile can cooperate through SLM). #groups = global / local. At
    GROUP=1 the grid is byte-identical to the non-cooperative default; raising
    GROUP_M keeps the SAME global work size (no redundant threads) and just forms
    larger groups.
    """
    kernel_source = (
        "#define BLOCK_M 8\n#define BLOCK_N 16\n#define GROUP_M 1\n#define GROUP_N 1\n"
    )
    grid_spec = {
        "x": "ceil(M / (BLOCK_M * GROUP_M)) * GROUP_M",
        "y": "ceil(N / (BLOCK_N * GROUP_N)) * GROUP_N",
        "z": 1,
        "local": {"x": "GROUP_M", "y": "GROUP_N", "z": 1},
    }
    dims = {"M": 256, "N": 256, "K": 256}

    # GROUP_*=1 -> identity with the non-cooperative grid.
    grid = compute_grid(kernel_source, grid_spec, dims)
    assert grid.global_size == (32, 16, 1), grid.global_size
    assert grid.local_size == (1, 1, 1), grid.local_size

    # Raise GROUP_M to 4 -> groups of 4 threads, SAME 32 row-tiles, no redundancy.
    coop_source = kernel_source.replace("#define GROUP_M 1", "#define GROUP_M 4")
    coop = compute_grid(coop_source, grid_spec, dims)
    assert coop.global_size == (32, 16, 1), coop.global_size  # unchanged total work
    assert coop.local_size == (4, 1, 1), coop.local_size
    # #groups along x = global / local = 8 cooperating groups.
    assert coop.global_size[0] // coop.local_size[0] == 8
    print("[OK] Cooperative grid work-group test passed")


def test_grid_divisibility_guard():
    """global must divide evenly by local; a bad pairing raises ValueError."""
    kernel_source = "#define BLOCK_M 8\n#define GROUP_M 3\n"
    # global.x = ceil(256/8) = 32, local.x = 3 -> 32 % 3 != 0 -> reject.
    grid_spec = {"x": "ceil(M / BLOCK_M)", "y": 1, "z": 1, "local": {"x": "GROUP_M"}}
    try:
        compute_grid(kernel_source, grid_spec, {"M": 256, "N": 256})
        raise AssertionError("Should have rejected indivisible global/local")
    except ValueError as e:
        assert "multiple of local" in str(e), e
        print("[OK] Grid divisibility guard test passed")


def test_grid_contract_cooperative_annotation():
    """Work-group-size knobs are flagged as cooperative levers in the contract."""
    kernel_source = (
        "#define BLOCK_M 8\n#define BLOCK_N 16\n#define GROUP_M 1\n#define GROUP_N 1\n"
    )
    grid_spec = {
        "x": "ceil(M / BLOCK_M) * GROUP_M",
        "y": "ceil(N / BLOCK_N) * GROUP_N",
        "z": 1,
        "local": {"x": "GROUP_M", "y": "GROUP_N", "z": 1},
    }
    contract = describe_grid_contract(grid_spec, kernel_source)
    # Both tile and group knobs are listed.
    assert "BLOCK_M = 8" in contract, contract
    assert "GROUP_M = 1" in contract, contract
    # The group knobs are annotated and the cooperative SLM guidance is present.
    assert "work-group size" in contract, contract
    assert "COOPERATIVE THREAD GROUPS" in contract, contract
    assert "cm_local_id" in contract and "cm_store_slm" in contract, contract
    # A non-cooperative kernel (no local block) gets no cooperative note.
    plain = describe_grid_contract(
        {"x": "ceil(M / BLOCK_M)", "y": "ceil(N / BLOCK_N)", "z": 1},
        "#define BLOCK_M 8\n#define BLOCK_N 16\n",
    )
    assert "COOPERATIVE THREAD GROUPS" not in plain, plain
    print("[OK] Grid contract cooperative annotation test passed")


def test_grid_config_from_seed_kernel():
    """Test grid evaluation against the actual seed kernel."""
    kernel_source = """
#include <cm/cm.h>

#define BLOCK_M 8
#define BLOCK_N 16
#define BLOCK_K 16

extern "C" _GENX_MAIN_ void
cm_gemm(SurfaceIndex surfA [[type("buffer_t")]],
        SurfaceIndex surfB [[type("buffer_t")]],
        SurfaceIndex surfD [[type("buffer_t")]],
        int M, int N, int K) {
  const int tm = cm_group_id(0) * BLOCK_M;
  const int tn = cm_group_id(1) * BLOCK_N;
  // kernel body
}
"""

    grid_spec = {
        "x": "ceil(M / BLOCK_M)",
        "y": "ceil(N / BLOCK_N)",
        "z": 1,
    }

    dims = {"M": 256, "N": 256, "K": 256}

    grid = compute_grid(kernel_source, grid_spec, dims)

    # For 256x256 with BLOCK_M=8, BLOCK_N=16:
    # x = ceil(256 / 8) = 32
    # y = ceil(256 / 16) = 16
    assert grid.global_size == (32, 16, 1), f"Expected (32, 16, 1), got {grid.global_size}"
    assert grid.local_size == (1, 1, 1), f"Expected (1, 1, 1), got {grid.local_size}"

    print("[OK] Seed kernel grid evaluation test passed")


def test_load_spec_with_grid_and_outputs():
    """Test loading the seed kernel spec with grid and outputs."""
    # Try multiple possible paths
    possible_paths = [
        Path(__file__).parent.parent / "test_kernels" / "200_CM_Gemm.yaml",
        Path("test_kernels") / "200_CM_Gemm.yaml",
        Path("c:\\Users\\gta\\Desktop\\smarcink\\Xe-Forge\\test_kernels\\200_CM_Gemm.yaml"),
    ]

    spec_path = None
    for p in possible_paths:
        if p.exists():
            spec_path = p
            break

    if not spec_path:
        print(f"[SKIP] spec load test (file not found in any of: {possible_paths})")
        return

    spec = load_spec(spec_path)

    # Check outputs
    assert "outputs" in dir(spec), "spec should have outputs attribute"
    assert "D" in spec.outputs, "spec.outputs should have D"
    assert spec.outputs["D"].dtype == "float16", "D should be float16"
    assert spec.outputs["D"].shape_vars == ["M", "N"], "D shape should be [M, N]"

    # Check grid
    assert spec.grid is not None, "spec.grid should not be None"
    assert "x" in spec.grid, "grid should have x"
    assert spec.grid["x"] == "ceil(M / (BLOCK_M * GROUP_M)) * GROUP_M", f"Got {spec.grid['x']}"
    assert spec.grid["y"] == "ceil(N / (BLOCK_N * GROUP_N)) * GROUP_N", f"Got {spec.grid['y']}"
    # The cooperative work-group (local) block is parsed too.
    assert spec.grid.get("local", {}).get("x") == "GROUP_M", f"Got {spec.grid.get('local')}"

    print("[OK] Spec load with grid/outputs test passed")


if __name__ == "__main__":
    test_grid_parser_basic()
    test_define_extraction()
    test_validate_references()
    test_define_extraction_hex()
    test_unsafe_expression_rejected()
    test_missing_symbol_diagnostic()
    test_grid_contract_names_actual_knobs()
    test_cooperative_grid_multiply()
    test_grid_divisibility_guard()
    test_grid_contract_cooperative_annotation()
    test_grid_config_from_seed_kernel()
    test_load_spec_with_grid_and_outputs()
    print("\nAll grid tests passed!")
