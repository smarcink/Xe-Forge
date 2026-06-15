"""
CM grid evaluation.

Turns a YAML grid spec plus the kernel's ``#define`` knobs into concrete OpenCL
work sizes. Grid expressions support ``ceil``/``floor``/``min``/``max`` and
``+ - * / // %`` over problem dims (from the spec) and tuning knobs (from the
kernel). Expressions are evaluated with a restricted AST walker — never
``eval`` — so a malformed or hostile formula cannot run arbitrary code, and
``**`` is rejected to avoid a cheap denial-of-service.

Public surface: :class:`GridConfig`, :func:`compute_grid`, :func:`extract_defines`.
"""

from __future__ import annotations

import ast
import logging
import math
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Helpers and operators permitted inside grid expressions. Pow (``**``) is
# deliberately excluded: grid math never needs it and ``a ** b`` with large
# operands is a cheap denial-of-service vector.
_FUNCS = {"ceil": math.ceil, "floor": math.floor, "min": min, "max": max}
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
_UNARYOPS = (ast.UAdd, ast.USub)
_DEFINE_RE = re.compile(r"^[ \t]*#define[ \t]+(\w+)[ \t]+([^\s/]+)", re.MULTILINE)

# Grid used when a CM spec declares none: cover the problem with the kernel's
# BLOCK_M/BLOCK_N tiles. Shared by compute_grid() and describe_grid_contract().
_DEFAULT_GRID: dict[str, Any] = {"x": "ceil(M / BLOCK_M)", "y": "ceil(N / BLOCK_N)", "z": 1}


@dataclass
class GridConfig:
    """Concrete launch grid: global/local work sizes and the source formulas."""

    global_size: tuple[int, int, int]
    local_size: tuple[int, int, int] = (1, 1, 1)
    formulas: dict[str, Any] | None = None  # kept for logging/debugging

    def __post_init__(self) -> None:
        names = ("global_x", "global_y", "global_z", "local_x", "local_y", "local_z")
        for value, name in zip((*self.global_size, *self.local_size), names, strict=True):
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")


def extract_defines(kernel_source: str) -> dict[str, int]:
    """Return integer ``#define NAME VALUE`` pairs (decimal or ``0x``/``0o``/``0b``).

    Function-like or expression-valued macros are ignored.
    """
    defines: dict[str, int] = {}
    for name, token in _DEFINE_RE.findall(kernel_source):
        try:
            defines[name] = int(token, 0)  # honors 0x/0o/0b prefixes and plain decimal
        except ValueError:
            logger.debug("Skipping non-integer #define %s = %s", name, token)
    return defines


def compute_grid(
    kernel_source: str,
    grid_spec: dict[str, Any] | None,
    dims: dict[str, int | float],
) -> GridConfig:
    """Evaluate a grid spec into concrete global/local work sizes.

    Args:
        kernel_source: Full CM kernel source; its ``#define``s are the tuning knobs.
        grid_spec: ``{"x", "y", "z"}`` (plus an optional nested ``"local"``) of ints
            or string expressions over problem dims and kernel ``#define``s. ``None``
            defaults to covering the problem with ``BLOCK_M``/``BLOCK_N``.
        dims: Problem dimensions, e.g. ``{"M": 1024, "N": 1024, "K": 1024}`` (floats
            from the CLI are coerced to ints).

    Raises:
        ValueError: If the spec is malformed, references an undefined symbol, or an
            expression cannot be evaluated.
    """
    if grid_spec is None:
        logger.warning("No grid specified; defaulting to ceil(M/BLOCK_M) x ceil(N/BLOCK_N)")
        grid_spec = dict(_DEFAULT_GRID)

    dims = {k: int(v) for k, v in dims.items()}
    defines = extract_defines(kernel_source)

    global_size = _eval_axes(grid_spec, dims, defines, "global")
    local_size = _eval_axes(grid_spec.get("local"), dims, defines, "local")

    for axis, g, loc in zip("xyz", global_size, local_size, strict=True):
        if loc > g:
            logger.warning(
                "local.%s (%d) > global.%s (%d); runtime will clamp", axis, loc, axis, g
            )

    grid = GridConfig(global_size, local_size, formulas=dict(grid_spec))
    logger.info("Evaluated grid: global=%s local=%s", grid.global_size, grid.local_size)
    return grid


def describe_grid_contract(grid_spec: dict[str, Any] | None, kernel_source: str) -> str:
    """Render the launch-grid contract for the optimizer prompt.

    Names the exact integer ``#define`` knobs that drive the grid — found by
    intersecting the grid formulas with the kernel's own ``#define``s — and shows
    the formulas plus each knob's current value. This tells the optimizer which
    ``#define``s it may tune and must not rename, whatever they happen to be
    called in this kernel/spec.
    """
    effective = grid_spec or _DEFAULT_GRID
    defines = extract_defines(kernel_source)
    local = effective.get("local")
    local = local if isinstance(local, dict) else None

    exprs = [effective[ax] for ax in ("x", "y", "z") if ax in effective]
    formulas = [f"  global.{ax} = {effective[ax]}" for ax in ("x", "y", "z") if ax in effective]
    if local:
        exprs += [local[ax] for ax in ("x", "y", "z") if ax in local]
        formulas += [f"  local.{ax} = {local[ax]}" for ax in ("x", "y", "z") if ax in local]

    symbols: set[str] = set()
    for expr in exprs:
        symbols |= set(re.findall(r"[A-Za-z_]\w*", str(expr)))
    symbols -= _FUNCS.keys()
    knobs = sorted(s for s in symbols if s in defines)
    dims = sorted(s for s in symbols if s not in defines)

    if knobs:
        knob_lines = "\n".join(f"    {k} = {defines[k]}" for k in knobs)
    else:
        knob_lines = "    (no grid-driving #define found in the current kernel)"

    return (
        "The harness computes the launch grid from these formulas (you never set the "
        "grid yourself):\n"
        + "\n".join(formulas)
        + "\nFixed problem dimensions (set at runtime, not tunable): "
        + (", ".join(dims) if dims else "(none)")
        + "\nGrid-driving tuning knobs — integer #defines in this kernel you MAY tune:\n"
        + knob_lines
        + "\nChanging a knob's value is safe: the dispatch grid follows it automatically. "
        "But keep each knob a plain integer `#define <NAME> <int>` with the SAME name shown "
        "above — do NOT rename it, remove it, inline its value, or turn it into an expression "
        "or function-like macro, or the grid can no longer be computed and the kernel is "
        "rejected."
    )


def _eval_axes(
    spec: dict[str, Any] | None,
    dims: dict[str, int],
    defines: dict[str, int],
    label: str,
    default: int = 1,
) -> tuple[int, int, int]:
    """Evaluate the [x, y, z] of a global/local spec dict (missing axes -> default)."""
    if not spec:
        return (default, default, default)

    symbols = {**dims, **defines}
    out: list[int] = []
    for axis in "xyz":
        value = spec.get(axis, default)
        if isinstance(value, bool):  # bool is an int subclass; reject YAML true/false
            raise ValueError(f"{label}.{axis} must be an int or expression, got bool {value!r}")
        if isinstance(value, int):
            out.append(value)
            continue
        expr = str(value).strip()
        _check_symbols(expr, dims, defines, f"{label}.{axis}")
        try:
            out.append(int(_safe_eval(expr, symbols)))
        except ValueError as e:
            raise ValueError(f"Failed to evaluate {label}.{axis} = {expr!r}: {e}") from e
    return (out[0], out[1], out[2])


def _check_symbols(expr: str, dims: dict[str, int], defines: dict[str, int], label: str) -> None:
    """Raise an actionable error if ``expr`` names something that is neither a
    problem dim nor a kernel ``#define`` (dims and defines stay separate so the
    message can point the author at the right place)."""
    used = set(re.findall(r"[A-Za-z_]\w*", expr)) - _FUNCS.keys()
    missing = sorted(s for s in used if s not in dims and s not in defines)
    if missing:
        raise ValueError(
            f"{label} references undefined symbol(s) {missing}. "
            f"Problem dims: {sorted(dims)}; kernel #defines: {sorted(defines)}. "
            f"Add `#define <NAME> <int>` to the kernel, or a dim to the spec."
        )


def _safe_eval(expr: str, symbols: dict[str, Any]) -> int | float:
    """Evaluate ``expr`` against ``symbols`` with a restricted AST walker.

    Permits the whitelisted functions, numeric literals, and ``+ - * / // %`` with
    unary ``+``/``-``. Anything else (attribute access, other calls, ``**``,
    comprehensions, ...) raises ValueError.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression {expr!r}: {e.msg}") from e
    return _eval(tree.body, symbols, expr)


def _eval(node: ast.AST, symbols: dict[str, Any], expr: str) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ValueError(f"Non-numeric literal {node.value!r} in {expr!r}")
        return node.value
    if isinstance(node, ast.Name):
        if node.id in symbols:
            return symbols[node.id]
        raise ValueError(f"Undefined symbol {node.id!r} in {expr!r}")
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _BINOPS):
            raise ValueError(f"Operator {type(node.op).__name__} not allowed in {expr!r}")
        left = _eval(node.left, symbols, expr)
        right = _eval(node.right, symbols, expr)
        return _binop(node.op, left, right, expr)
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _UNARYOPS):
            raise ValueError(f"Unary {type(node.op).__name__} not allowed in {expr!r}")
        operand = _eval(node.operand, symbols, expr)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.Call):
        return _call(node, symbols, expr)
    raise ValueError(f"Unsupported expression element {type(node).__name__} in {expr!r}")


def _call(node: ast.Call, symbols: dict[str, Any], expr: str) -> int | float:
    if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
        name = getattr(node.func, "id", type(node.func).__name__)
        raise ValueError(f"Call to {name!r} not allowed in {expr!r}")
    if node.keywords:
        raise ValueError(f"Keyword arguments not allowed in {expr!r}")
    args = [_eval(a, symbols, expr) for a in node.args]
    try:
        return _FUNCS[node.func.id](*args)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Bad call to {node.func.id}() in {expr!r}: {e}") from e


def _binop(op: ast.operator, left: int | float, right: int | float, expr: str) -> int | float:
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
        raise ValueError(f"Division by zero in {expr!r}") from e
