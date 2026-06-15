"""
CM Grid Configuration and Evaluation.

Parses and validates grid specifications from YAML and kernel #defines:
- Grid formulas: expressions over problem dims (from spec) and tuning knobs (from kernel)
- Supports: ceil(), floor(), min(), max(), and + - * / // % arithmetic
- Validates that referenced symbols are either problem dims or #define'd in kernel
- Evaluates to concrete global/local work sizes for kernel launch

Expressions are evaluated with a restricted AST walker (no ``eval``): only the
whitelisted functions/operators above and numeric literals are permitted, so a
malformed or hostile grid formula cannot execute arbitrary code or hang the
process.
"""

from __future__ import annotations

import ast
import logging
import math
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Whitelisted helpers usable inside grid expressions.
_ALLOWED_FUNCS = {"ceil": math.ceil, "floor": math.floor, "min": min, "max": max}
# Whitelisted operators. Pow (``**``) is intentionally excluded: grid math never
# needs it, and ``a ** b`` with large operands is a cheap denial-of-service vector.
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


@dataclass
class GridConfig:
    """Parsed and validated grid configuration."""

    global_size: tuple[int, int, int]  # ND-range [x, y, z]
    local_size: tuple[int, int, int] = (1, 1, 1)  # Work-group size [x, y, z]
    formulas: dict[str, Any] | None = None  # Original formulas for debugging

    def __post_init__(self):
        """Validate that every global and local size is positive."""
        names = ("global_x", "global_y", "global_z", "local_x", "local_y", "local_z")
        for value, name in zip((*self.global_size, *self.local_size), names, strict=False):
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")


class GridParser:
    """Parses grid specs from YAML and evaluates them against dims + kernel defines."""

    # #define NAME VALUE — value token captured up to whitespace/comment, parsed below.
    _DEFINE_PATTERN = re.compile(r"^[ \t]*#define[ \t]+(\w+)[ \t]+([^\s/]+)", re.MULTILINE)

    # Identifiers that are built-in helpers, not symbols to resolve.
    _BUILTINS = frozenset(_ALLOWED_FUNCS)

    @staticmethod
    def _parse_int_token(token: str) -> int | None:
        """Parse an integer #define value (decimal or 0x/0o/0b prefixed), else None."""
        try:
            return int(token, 0)  # honors 0x.., 0o.., 0b.. prefixes
        except ValueError:
            pass
        try:
            return int(token)  # plain decimal, incl. accidental leading zeros
        except ValueError:
            return None

    def extract_defines(self, kernel_source: str) -> dict[str, int]:
        """Extract integer ``#define NAME VALUE`` pairs from kernel source.

        Only simple integer defines (decimal or hex/oct/bin) are returned;
        expression-valued or function-like macros are ignored.
        """
        defines: dict[str, int] = {}
        for name, token in self._DEFINE_PATTERN.findall(kernel_source):
            value = self._parse_int_token(token)
            if value is None:
                logger.debug("Skipping non-integer #define %s = %s", name, token)
            else:
                defines[name] = value
        logger.debug("Extracted #defines from kernel: %s", defines)
        return defines

    def validate_references(
        self,
        grid_expr: str,
        dims: dict[str, int],
        defines: dict[str, int],
        expr_name: str = "grid",
    ) -> None:
        """Ensure every symbol in ``grid_expr`` is a problem dim or a kernel #define.

        Raises ValueError naming the missing symbol(s) and which bucket they
        belong to, so the kernel author / LLM gets an actionable message.
        """
        identifiers = set(re.findall(r"[A-Za-z_]\w*", grid_expr))
        missing = sorted(
            s for s in identifiers - self._BUILTINS if s not in dims and s not in defines
        )
        if missing:
            raise ValueError(
                f"{expr_name} references undefined symbol(s) {missing}. "
                f"Known problem dims: {sorted(dims)}; kernel #defines: {sorted(defines)}. "
                f"Add a tuning knob as `#define <NAME> <int>` in the kernel, "
                f"or add a problem dimension to the spec's dims."
            )

    def evaluate(
        self,
        grid_spec: dict[str, Any] | None,
        dims: dict[str, int],
        kernel_source: str,
    ) -> GridConfig:
        """Parse and evaluate a grid specification.

        Args:
            grid_spec: Dict with keys "x", "y", "z" (global) and an optional "local"
                       nested dict of the same shape. Values are ints or string
                       expressions over problem dims and kernel #defines.
            dims: Problem dimension values (e.g., {"M": 1024, "N": 1024, "K": 1024}).
            kernel_source: Full kernel C++ source (to extract #defines from).

        Returns:
            GridConfig with evaluated global/local sizes.

        Raises:
            ValueError: If grid_spec is malformed, references missing symbols,
                        or an expression cannot be evaluated.
        """
        if grid_spec is None:
            # Default grid: cover the problem with BLOCK_M/N (must exist in kernel).
            logger.warning("No grid specified; using default (requires BLOCK_M, BLOCK_N in kernel)")
            grid_spec = {"x": "ceil(M / BLOCK_M)", "y": "ceil(N / BLOCK_N)", "z": 1}

        defines = self.extract_defines(kernel_source)

        global_size = self._evaluate_grid_dims(grid_spec, dims, defines, "global")
        local_size = self._evaluate_grid_dims(grid_spec.get("local"), dims, defines, "local")

        for axis, g, loc in zip("xyz", global_size, local_size, strict=True):
            if loc > g:
                logger.warning(
                    "local.%s (%d) > global.%s (%d); the runtime will clamp it to fit",
                    axis,
                    loc,
                    axis,
                    g,
                )

        result = GridConfig(global_size=global_size, local_size=local_size, formulas=dict(grid_spec))
        logger.info("Evaluated grid: global=%s, local=%s", result.global_size, result.local_size)
        return result

    def _evaluate_grid_dims(
        self,
        spec: dict[str, Any] | None,
        dims: dict[str, int],
        defines: dict[str, int],
        kind: str,
        default: int = 1,
    ) -> tuple[int, int, int]:
        """Evaluate the [x, y, z] of a global/local grid spec dict.

        ``dims`` and ``defines`` are kept separate (not merged) so reference
        validation can report exactly where a missing symbol should be defined.
        """
        if not spec:
            return (default, default, default)

        symbols = {**dims, **defines}
        out: dict[str, int] = {}
        for axis in "xyz":
            if axis not in spec:
                out[axis] = default
                continue
            expr = spec[axis]
            if isinstance(expr, bool):
                raise ValueError(f"{kind}.{axis} must be an int/expression, got bool {expr!r}")
            if isinstance(expr, int):
                out[axis] = expr
                continue
            expr_str = str(expr).strip()
            self.validate_references(expr_str, dims, defines, f"{kind}.{axis}")
            try:
                out[axis] = int(self._safe_eval(expr_str, symbols))
            except ValueError as e:
                raise ValueError(f"Failed to evaluate {kind}.{axis} = {expr_str!r}: {e}") from e

        return (out["x"], out["y"], out["z"])

    def _safe_eval(self, expr: str, context: dict[str, Any]) -> int | float:
        """Evaluate ``expr`` against ``context`` using a restricted AST walker.

        Supports the whitelisted functions (ceil/floor/min/max), numeric literals,
        and + - * / // % with unary +/-. Anything else (attribute access, calls to
        other names, ``**``, comprehensions, ...) raises ValueError.
        """
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid grid expression {expr!r}: {e.msg}") from e
        return self._eval_node(tree.body, context, expr)

    def _eval_node(self, node: ast.AST, context: dict[str, Any], expr: str) -> int | float:
        """Recursively evaluate a single whitelisted AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int | float):
                raise ValueError(f"Non-numeric literal {node.value!r} in {expr!r}")
            return node.value
        if isinstance(node, ast.Name):
            if node.id in context:
                return context[node.id]
            raise ValueError(f"Undefined symbol {node.id!r} in expression {expr!r}")
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise ValueError(f"Operator {type(node.op).__name__} not allowed in {expr!r}")
            left = self._eval_node(node.left, context, expr)
            right = self._eval_node(node.right, context, expr)
            return self._apply_binop(node.op, left, right)
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARYOPS):
                raise ValueError(f"Unary {type(node.op).__name__} not allowed in {expr!r}")
            operand = self._eval_node(node.operand, context, expr)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.Call):
            return self._eval_call(node, context, expr)
        raise ValueError(f"Unsupported expression element {type(node).__name__} in {expr!r}")

    def _eval_call(self, node: ast.Call, context: dict[str, Any], expr: str) -> int | float:
        """Evaluate a call to one of the whitelisted helper functions."""
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            name = getattr(node.func, "id", type(node.func).__name__)
            raise ValueError(f"Call to {name!r} not allowed in {expr!r}")
        if node.keywords:
            raise ValueError(f"Keyword arguments not allowed in {expr!r}")
        args = [self._eval_node(a, context, expr) for a in node.args]
        try:
            return _ALLOWED_FUNCS[node.func.id](*args)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Bad call to {node.func.id}() in {expr!r}: {e}") from e

    @staticmethod
    def _apply_binop(op: ast.operator, left: int | float, right: int | float) -> int | float:
        """Apply a whitelisted binary operator, mapping div-by-zero to ValueError."""
        try:
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return left / right
            if isinstance(op, ast.FloorDiv):
                return left // right
            return left % right  # ast.Mod — only remaining allowed op
        except ZeroDivisionError as e:
            raise ValueError(f"Division by zero: {e}") from e


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
