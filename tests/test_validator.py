"""Tests for static kernel validation."""

from xe_forge.core.validator import KernelValidator

VALID_1D_SWIZZLED_GRID = """\
import triton
import triton.language as tl

GROUP_SIZE_M = 4


@triton.jit
def kernel():
    pass


class Model:
    pass


def launch():
    grid = lambda META: (triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]),)
    kernel[grid]()
"""


INVALID_2D_SWIZZLED_GRID = """\
import triton
import triton.language as tl

GROUP_SIZE_M = 4


@triton.jit
def kernel():
    pass


class Model:
    pass


def launch():
    grid = lambda META: (triton.cdiv(M, META["BM"]), triton.cdiv(N, META["BN"]))
    kernel[grid]()
"""


INVALID_2D_TUPLE_SWIZZLED_GRID = """\
import triton
import triton.language as tl

GROUP_SIZE_M = 4


@triton.jit
def kernel():
    pass


class Model:
    pass


def launch():
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 256))
    kernel[grid]()
"""


class TestGridSwizzleValidation:
    def test_1d_grid_with_swizzle_is_allowed(self):
        issues = KernelValidator().validate(VALID_1D_SWIZZLED_GRID, dsl="triton")
        assert all(issue.check_name != "grid_swizzle_conflict" for issue in issues)

    def test_2d_grid_with_swizzle_is_rejected(self):
        issues = KernelValidator().validate(INVALID_2D_SWIZZLED_GRID, dsl="triton")
        assert any(issue.check_name == "grid_swizzle_conflict" for issue in issues)

    def test_2d_tuple_grid_with_swizzle_is_rejected(self):
        issues = KernelValidator().validate(INVALID_2D_TUPLE_SWIZZLED_GRID, dsl="triton")
        assert any(issue.check_name == "grid_swizzle_conflict" for issue in issues)


VALID_CM_KERNEL = """\
#include <cm/cm.h>

extern "C" _GENX_MAIN_ void cm_gemm(SurfaceIndex a, SurfaceIndex b, SurfaceIndex d) {
    matrix<float, 8, 16> acc = 0.0f;
    write(d, 0, 0, acc);
}
"""

CM_MISSING_GENX_MAIN = """\
#include <cm/cm.h>

void helper() {}
"""

CM_MISSING_INCLUDE = """\
extern "C" _GENX_MAIN_ void cm_gemm(SurfaceIndex a) {}
"""


class TestCMValidation:
    def test_valid_cm_kernel_has_no_structural_issues(self):
        issues = KernelValidator().validate(VALID_CM_KERNEL, dsl="cm")
        names = {i.check_name for i in issues}
        assert "missing_include" not in names
        assert "missing_genx_main" not in names
        assert "missing_cm_header" not in names

    def test_missing_genx_main_is_flagged(self):
        issues = KernelValidator().validate(CM_MISSING_GENX_MAIN, dsl="cm")
        assert any(i.check_name == "missing_genx_main" for i in issues)

    def test_missing_include_is_flagged(self):
        issues = KernelValidator().validate(CM_MISSING_INCLUDE, dsl="cm")
        assert any(i.check_name == "missing_include" for i in issues)

    def test_cooperative_slm_kernel_has_no_slm_warnings(self):
        code = (
            "#include <cm/cm.h>\n"
            'extern "C" _GENX_MAIN_ void k(SurfaceIndex d) {\n'
            "  uint lid = cm_local_id(0);\n"
            "  cm_slm_init(64); uint slm = cm_slm_alloc(64);\n"
            "  vector<uint,1> v; v[0] = lid;\n"
            "  cm_store_slm<uint,1>(lid*4, v);\n"
            "  cm_slm_fence(CM_GLOBAL_COHERENT_FENCE); cm_barrier();\n"
            "}\n"
        )
        names = {i.check_name for i in KernelValidator().validate(code, dsl="cm")}
        assert "slm_without_thread_group" not in names
        assert "barrier_without_slm_fence" not in names

    def test_slm_without_cm_local_id_is_flagged(self):
        code = (
            "#include <cm/cm.h>\n"
            'extern "C" _GENX_MAIN_ void k(SurfaceIndex d) {\n'
            "  cm_slm_init(64); uint slm = cm_slm_alloc(64);\n"
            "  cm_slm_fence(CM_GLOBAL_COHERENT_FENCE); cm_barrier();\n"
            "}\n"
        )
        names = {i.check_name for i in KernelValidator().validate(code, dsl="cm")}
        assert "slm_without_thread_group" in names

    def test_barrier_without_fence_is_flagged(self):
        code = (
            "#include <cm/cm.h>\n"
            'extern "C" _GENX_MAIN_ void k(SurfaceIndex d) {\n'
            "  uint lid = cm_local_id(0);\n"
            "  cm_barrier();\n"
            "}\n"
        )
        names = {i.check_name for i in KernelValidator().validate(code, dsl="cm")}
        assert "barrier_without_slm_fence" in names

