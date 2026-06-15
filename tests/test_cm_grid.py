"""
Test grid parsing and evaluation for CM kernels.
"""

from pathlib import Path

from xe_forge.core.cm_grid import GridParser, parse_grid_from_kernel_and_spec
from xe_forge.core.spec_loader import load_spec


def test_grid_parser_basic():
    """Test basic grid expression parsing with ceil, min, max."""
    parser = GridParser()

    # Test simple division with ceil
    result = parser._safe_eval("ceil(256 / 8)", {})
    assert result == 32, f"Expected 32, got {result}"

    # Test with context variables
    result = parser._safe_eval("ceil(M / BLOCK_M)", {"M": 256, "BLOCK_M": 8})
    assert result == 32, f"Expected 32, got {result}"

    # Test min/max
    result = parser._safe_eval("min(256, 512)", {})
    assert result == 256, f"Expected 256, got {result}"

    result = parser._safe_eval("max(8, 16)", {})
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
    parser = GridParser()
    defines = parser.extract_defines(kernel_source)

    assert defines == {"BLOCK_M": 8, "BLOCK_N": 16, "BLOCK_K": 32}, f"Got {defines}"
    print("[OK] Define extraction test passed")


def test_validate_references():
    """Test symbol validation."""
    parser = GridParser()

    dims = {"M": 256, "N": 256, "K": 256}
    defines = {"BLOCK_M": 8, "BLOCK_N": 16}

    # Valid expression
    try:
        parser.validate_references("ceil(M / BLOCK_M)", dims, defines)
        print("[OK] Valid reference check passed")
    except ValueError as e:
        raise AssertionError(f"Valid expression rejected: {e}") from e

    # Invalid expression (missing BLOCK_K)
    try:
        parser.validate_references("ceil(K / BLOCK_K)", dims, defines)
        raise AssertionError("Should have rejected missing BLOCK_K")
    except ValueError as e:
        assert "BLOCK_K" in str(e), f"Expected BLOCK_K in error, got: {e}"
        print("[OK] Invalid reference check passed")


def test_define_extraction_hex():
    """Hex/oct/bin #define values are parsed, function-like macros are skipped."""
    parser = GridParser()
    defines = parser.extract_defines(
        "#define BLOCK_M 0x10\n"
        "#define BLOCK_N 8  // trailing comment\n"
        "#define SQUARE(x) ((x)*(x))\n"
    )
    assert defines == {"BLOCK_M": 16, "BLOCK_N": 8}, f"Got {defines}"
    print("[OK] Hex/comment/macro define extraction test passed")


def test_unsafe_expression_rejected():
    """The evaluator must reject power operators and arbitrary calls (no eval())."""
    parser = GridParser()
    for bad in ["9 ** 9 ** 9", "__import__('os')", "M.__class__", "open('x')"]:
        try:
            parser._safe_eval(bad, {"M": 256})
            raise AssertionError(f"Should have rejected unsafe expression: {bad!r}")
        except ValueError:
            pass
    print("[OK] Unsafe expression rejection test passed")


def test_missing_symbol_diagnostic():
    """A missing knob is reported as a #define gap, not lumped with problem dims."""
    kernel_source = "#define BLOCK_M 8\n"  # BLOCK_N intentionally absent
    grid_spec = {"x": "ceil(M / BLOCK_M)", "y": "ceil(N / BLOCK_N)", "z": 1}
    try:
        parse_grid_from_kernel_and_spec(kernel_source, grid_spec, {"M": 256, "N": 256})
        raise AssertionError("Should have rejected missing BLOCK_N")
    except ValueError as e:
        msg = str(e)
        assert "BLOCK_N" in msg, f"Expected BLOCK_N in error, got: {msg}"
        # The message must distinguish kernel #defines (BLOCK_M present) from dims.
        assert "BLOCK_M" in msg and "#define" in msg, f"Diagnostic not actionable: {msg}"
        print("[OK] Missing-symbol diagnostic test passed")



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

    grid = parse_grid_from_kernel_and_spec(kernel_source, grid_spec, dims)

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
    assert spec.outputs["D"].dtype == "float32", "D should be float32"
    assert spec.outputs["D"].shape_vars == ["M", "N"], "D shape should be [M, N]"

    # Check grid
    assert spec.grid is not None, "spec.grid should not be None"
    assert "x" in spec.grid, "grid should have x"
    assert spec.grid["x"] == "ceil(M / BLOCK_M)", f"Got {spec.grid['x']}"
    assert spec.grid["y"] == "ceil(N / BLOCK_N)", f"Got {spec.grid['y']}"

    print("[OK] Spec load with grid/outputs test passed")


if __name__ == "__main__":
    test_grid_parser_basic()
    test_define_extraction()
    test_validate_references()
    test_define_extraction_hex()
    test_unsafe_expression_rejected()
    test_missing_symbol_diagnostic()
    test_grid_config_from_seed_kernel()
    test_load_spec_with_grid_and_outputs()
    print("\nAll grid tests passed!")
