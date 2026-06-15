"""
CM Grid Configuration and Evaluation.

Parses and validates grid specifications from YAML and kernel #defines:
- Grid formulas: expressions over problem dims (from spec) and tuning knobs (from kernel)
- Supports: ceil(), min(), max(), arithmetic
- Validates that referenced symbols are either problem dims or #define'd in kernel
- Evaluates to concrete global/local work sizes for kerneles launch
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GridConfig:
    """Parsed and validated grid configuration."""

    global_size: tuple[int, int, int]  # ND-range [x, y, z]
    local_size: tuple[int, int, int] = (1, 1, 1)  # Work-group size [x, y, z]
    formulas: dict[str, str] | None = None  # Original formulas for debugging

    def __post_init__(self):
        """Validate that global and local sizes are positive."""
        for dim, name in zip(self.global_size, ("global_x", "global_y", "global_z")):
            if dim <= 0:
                raise ValueError(f"{name} must be > 0, got {dim}")
        for dim, name in zip(self.local_size, ("local_x", "local_y", "local_z")):
            if dim <= 0:
                raise ValueError(f"{name} must be > 0, got {dim}")


class GridParser:
    """Parses grid specifications from YAML and evaluates them against dims + kernel defines."""

    # Regex to extract #define NAME VALUE
    _DEFINE_PATTERN = re.compile(r"^\s*#define\s+(\w+)\s+(\d+)", re.MULTILINE)

    # Symbols that are built-in functions
    _BUILTINS = {"ceil", "min", "max"}

    def __init__(self):
        self._defines: dict[str, int] = {}
        self._dims: dict[str, int] = {}

    def extract_defines(self, kernel_source: str) -> dict[str, int]:
        """Extract all integer #define NAME VALUE from kernel source.

        Returns a dict of {name: value} for symbols referenced in grid expressions.
        Only extracts simple integer defines; complex macros are ignored.
        """
        defines = {}
        for match in self._DEFINE_PATTERN.finditer(kernel_source):
            name, value_str = match.groups()
            try:
                defines[name] = int(value_str)
            except ValueError:
                logger.debug(f"Skipping non-integer #define {name} = {value_str}")
        logger.debug(f"Extracted #defines from kernel: {defines}")
        return defines

    def validate_references(
        self,
        grid_expr: str,
        dims: dict[str, int],
        defines: dict[str, int],
        expr_name: str = "grid",
    ) -> None:
        """Validate that all symbols in grid_expr are either dims or #define'd.

        Raises ValueError if a symbol is missing.
        """
        # Extract all identifiers (word characters)
        identifiers = set(re.findall(r"\b([a-zA-Z_]\w*)\b", grid_expr))

        # Remove builtins
        unknowns = identifiers - self._BUILTINS

        # Check each unknown symbol
        missing = []
        for sym in unknowns:
            if sym not in dims and sym not in defines:
                missing.append(sym)

        if missing:
            raise ValueError(
                f"{expr_name} references undefined symbols: {sorted(missing)}. "
                f"Expected to find them in problem dims {set(dims.keys())} "
                f"or kernel #defines {set(defines.keys())}"
            )

    def evaluate(
        self,
        grid_spec: dict[str, Any] | None,
        dims: dict[str, int],
        kernel_source: str,
    ) -> GridConfig:
        """Parse and evaluate grid specification.

        Args:
            grid_spec: Dict with keys "x", "y", "z" (global) and optionally "local"
                      Each value is a string expression to evaluate.
            dims: Problem dimension values (e.g., {"M": 1024, "N": 1024, "K": 1024})
            kernel_source: The full kernel C++ source (to extract #defines)

        Returns:
            GridConfig with evaluated global/local sizes.

        Raises:
            ValueError: If grid_spec is malformed, references missing symbols,
                       or evaluation fails.
        """
        if grid_spec is None:
            # Default grid: covers problem size with BLOCK_M/N (must be in kernel)
            logger.warning("No grid specified; using default (requires BLOCK_M, BLOCK_N in kernel)")
            grid_spec = {
                "x": "ceil(M / BLOCK_M)",
                "y": "ceil(N / BLOCK_N)",
                "z": 1,
            }

        # Extract kernel defines
        defines = self.extract_defines(kernel_source)

        # Build evaluation context
        context = {**dims, **defines}

        # Parse global (x, y, z)
        global_size = self._evaluate_grid_dims(grid_spec, context, "global")

        # Parse local (optional, default [1,1,1])
        local_spec = grid_spec.get("local", {})
        local_size = self._evaluate_grid_dims(local_spec, context, "local", default=1)

        # Validate coverage: global >= local (each dimension)
        for g, l, axis in zip(global_size, local_size, "xyz"):
            if l > g:
                logger.warning(
                    f"local.{axis} ({l}) > global.{axis} ({g}); "
                    "OpenCL will auto-reduce local to fit global"
                )

        result = GridConfig(global_size=global_size, local_size=local_size, formulas=dict(grid_spec))
        logger.info(f"Evaluated grid: global={result.global_size}, local={result.local_size}")
        return result

    def _evaluate_grid_dims(
        self,
        spec: dict[str, Any] | None,
        context: dict[str, int],
        kind: str,
        default: int = 1,
    ) -> tuple[int, int, int]:
        """Evaluate grid dimensions [x, y, z] from spec dict.

        Args:
            spec: Dict with optional keys "x", "y", "z" (string expressions).
            context: Symbol table {name: value} for evaluation.
            kind: "global" or "local" (for error messages).
            default: Default value for omitted dimensions.

        Returns:
            Tuple (x, y, z) of evaluated integers.

        Raises:
            ValueError: If expressions can't be evaluated or reference missing symbols.
        """
        if not spec:
            return (default, default, default)

        dims_dict = {}
        for axis in "xyz":
            if axis in spec:
                expr = spec[axis]
                if isinstance(expr, int):
                    dims_dict[axis] = expr
                else:
                    expr_str = str(expr).strip()
                    # Validate references
                    self.validate_references(expr_str, context, {}, f"{kind}.{axis}")
                    # Evaluate
                    try:
                        val = self._safe_eval(expr_str, context)
                        dims_dict[axis] = int(val)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to evaluate {kind}.{axis} = {expr_str!r}: {e}"
                        ) from e
            else:
                dims_dict[axis] = default

        return (dims_dict["x"], dims_dict["y"], dims_dict["z"])

    def _safe_eval(self, expr: str, context: dict[str, Any]) -> Any:
        """Safely evaluate a mathematical expression with ceil, min, max support.

        Args:
            expr: String expression like "ceil(M/BLOCK_M)" or "min(N, 256)".
            context: Symbol table {name: value}.

        Returns:
            Evaluated result (typically an int or float).

        Raises:
            ValueError: If expression is malformed or references missing symbols.
        """
        import math

        # Build safe builtins: only math functions we allow
        safe_builtins = {
            "ceil": math.ceil,
            "min": min,
            "max": max,
            "__builtins__": {},
        }

        # Combine with context (problem dims + kernel defines)
        eval_context = {**context, **safe_builtins}

        try:
            return eval(expr, {"__builtins__": {}}, eval_context)
        except NameError as e:
            raise ValueError(f"Undefined symbol in expression: {e}") from e
        except ZeroDivisionError as e:
            raise ValueError(f"Division by zero in expression: {e}") from e
        except Exception as e:
            raise ValueError(f"Evaluation error: {e}") from e


def parse_grid_from_kernel_and_spec(
    kernel_source: str,
    grid_spec: dict[str, Any] | None,
    dims: dict[str, int],
) -> GridConfig:
    """Top-level function to parse, validate, and evaluate grid from spec + kernel.

    Args:
        kernel_source: Full C++ kernel source (to extract #defines).
        grid_spec: Grid spec from YAML (e.g., {"x": "ceil(M/BLOCK_M)", "y": "ceil(N/BLOCK_N)", "z": 1}).
        dims: Problem dimensions for this variant (e.g., {"M": 1024, "N": 1024, "K": 1024}).

    Returns:
        GridConfig with concrete global/local work sizes.

    Raises:
        ValueError: If grid is malformed or can't be evaluated.
    """
    parser = GridParser()
    return parser.evaluate(grid_spec, dims, kernel_source)
