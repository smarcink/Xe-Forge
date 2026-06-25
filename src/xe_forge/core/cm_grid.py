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
        for axis, g, loc in zip("xyz", self.global_size, self.local_size, strict=True):
            if g % loc != 0:
                raise ValueError(
                    f"global.{axis} ({g}) must be a whole multiple of local.{axis} "
                    f"({loc}); the global work size must divide evenly into "
                    f"work-groups of the local size"
                )


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


# A ``#define`` line with separately captured leading whitespace / name / value,
# used by :func:`rewrite_defines` to patch a value in place without disturbing
# the rest of the line.
_DEFINE_LINE_RE = re.compile(
    r"(?P<lead>^[ \t]*)#define[ \t]+(?P<name>\w+)[ \t]+(?P<val>[^\s/]+)", re.MULTILINE
)


def rewrite_defines(kernel_source: str, overrides: dict[str, int]) -> str:
    """Return *kernel_source* with the value of each named ``#define`` replaced.

    Only integer object-like ``#define``s already present in the source — the
    same kind :func:`extract_defines` reports — are rewritten. Names not present,
    or macros whose current value is not an integer literal, are left untouched
    so a function-like or expression macro can never be clobbered. This is the
    mechanism the autotuner uses to try candidate block sizes.
    """
    if not overrides:
        return kernel_source

    def _sub(m: "re.Match[str]") -> str:
        name = m.group("name")
        if name not in overrides:
            return m.group(0)
        try:
            int(m.group("val"), 0)  # only replace genuine integer-literal macros
        except ValueError:
            return m.group(0)
        return f"{m.group('lead')}#define {name} {int(overrides[name])}"

    return _DEFINE_LINE_RE.sub(_sub, kernel_source)


# Build-directive comment the executor honors to pass extra ``clBuildProgram``
# options that cannot be expressed in the kernel source as a ``#define`` —
# chiefly the GRF register-file size (``-Qxcm_register_file_size``), which is a
# compiler flag, not a language construct. The autotuner stamps the winning
# value here so the saved ``.cpp`` is self-describing and any later compile
# (final re-measure, a human re-run) uses the same build options.
BUILD_DIRECTIVE_PREFIX = "// xe-forge-build:"

# Strict allowlist of flags honored from a build directive. The directive feeds
# ``clBuildProgram`` and the surrounding source may be LLM-generated, so only
# known-safe flags are accepted — anything else is dropped. Extend deliberately.
# The register-file-size value is an integer (e.g. 128, 256) or ``auto`` (let the
# compiler pick the large file when the kernel needs it).
_ALLOWED_BUILD_FLAG_RE = re.compile(r"^-Qxcm_register_file_size=(\d+|auto)$")


def parse_build_directives(kernel_source: str) -> list[str]:
    """Extract allow-listed extra build flags from ``// xe-forge-build:`` lines.

    Tokens that do not match the allowlist are dropped with a warning so a
    (possibly LLM-generated) source comment cannot inject arbitrary compiler
    flags into ``clBuildProgram``.
    """
    tokens: list[str] = []
    for line in kernel_source.splitlines():
        stripped = line.strip()
        if not stripped.startswith(BUILD_DIRECTIVE_PREFIX):
            continue
        for tok in stripped[len(BUILD_DIRECTIVE_PREFIX):].split():
            if _ALLOWED_BUILD_FLAG_RE.match(tok):
                tokens.append(tok)
            else:
                logger.warning("Ignoring non-allowlisted CM build directive token: %r", tok)
    return tokens


def stamp_build_directive(kernel_source: str, tokens: list[str]) -> str:
    """Return *kernel_source* carrying a single ``// xe-forge-build:`` line.

    Any existing build-directive lines are removed first, so stamping is
    idempotent. With no *tokens*, the source is returned with directives
    stripped. The directive is placed at the top (it is a comment, valid before
    ``#include``) so it survives independently of the kernel body.
    """
    trailing_nl = kernel_source.endswith("\n")
    body = [
        ln
        for ln in kernel_source.splitlines()
        if not ln.strip().startswith(BUILD_DIRECTIVE_PREFIX)
    ]
    if tokens:
        body.insert(0, f"{BUILD_DIRECTIVE_PREFIX} {' '.join(tokens)}")
    out = "\n".join(body)
    return out + "\n" if trailing_nl else out


# Grid-directive comment that lets a kernel OWN its launch grid instead of
# inheriting the spec's. It is the mechanism by which an optimizer changes the
# threads-vs-work-per-thread split — e.g. give each thread several output tiles
# so a DPAS thread has enough rows to run at RepeatCount=8 and reuse each loaded
# weight across more rows — WITHOUT editing the spec. Honored by compute_grid().
GRID_DIRECTIVE_PREFIX = "// xe-forge-grid:"

# Axis keys honored in a grid directive: the three global axes plus an optional
# work-group (local) size per axis. Anything else is dropped with a warning so a
# (possibly LLM-generated) comment cannot smuggle in unexpected keys.
_GRID_DIRECTIVE_KEYS = {"x", "y", "z", "local.x", "local.y", "local.z"}


def parse_grid_directive(kernel_source: str) -> dict[str, Any] | None:
    """Extract a kernel-declared launch grid from ``// xe-forge-grid:`` line(s).

    Returns a grid spec dict shaped exactly like the YAML ``grid:`` block —
    ``{"x", "y", "z"}`` (plus an optional nested ``"local"``) of integer-or-
    expression strings over problem dims and kernel ``#define``s — or ``None``
    when no directive is present.

    Format (one or more comment lines; ``;``-separated ``key = expr`` pairs, so an
    expression may itself contain spaces)::

        // xe-forge-grid: x = ceil(N * OH / PH); y = ceil(OW / PW); z = 1
        // xe-forge-grid: local.x = LWS_X; local.y = LWS_Y

    Expressions are NOT evaluated here; they flow through the same safe AST
    evaluator as the YAML grid (see :func:`compute_grid` / :func:`_eval_axes`),
    so only the allow-listed functions/operators over dims + ``#define``s are ever
    computed — a directive can never execute arbitrary code.
    """
    pairs: list[str] = []
    for line in kernel_source.splitlines():
        stripped = line.strip()
        if not stripped.startswith(GRID_DIRECTIVE_PREFIX):
            continue
        pairs.extend(stripped[len(GRID_DIRECTIVE_PREFIX):].split(";"))

    spec: dict[str, Any] = {}
    local: dict[str, Any] = {}
    found = False
    for pair in pairs:
        if not pair.strip():
            continue
        if "=" not in pair:
            logger.warning("Ignoring malformed // xe-forge-grid: entry (no '='): %r", pair.strip())
            continue
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip()
        if key not in _GRID_DIRECTIVE_KEYS:
            logger.warning(
                "Ignoring unknown // xe-forge-grid: key %r (allowed: %s)",
                key, sorted(_GRID_DIRECTIVE_KEYS),
            )
            continue
        if not value:
            logger.warning("Ignoring empty // xe-forge-grid: value for %r", key)
            continue
        found = True
        if key.startswith("local."):
            local[key.split(".", 1)[1]] = value
        else:
            spec[key] = value

    if not found:
        return None
    if local:
        spec["local"] = local
    return spec


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
    # A kernel-declared // xe-forge-grid: directive OWNS the grid: it overrides
    # the spec so an optimizer can change the threads-vs-work-per-thread split
    # (e.g. own several output tiles per thread to fill the DPAS RepeatCount)
    # without editing the spec.
    directive_spec = parse_grid_directive(kernel_source)
    if directive_spec is not None:
        logger.info("Using kernel-declared // xe-forge-grid: directive (overrides spec grid)")
        grid_spec = directive_spec
    elif grid_spec is None:
        logger.warning("No grid specified; defaulting to ceil(M/BLOCK_M) x ceil(N/BLOCK_N)")
        grid_spec = dict(_DEFAULT_GRID)

    dims = {k: int(v) for k, v in dims.items()}
    defines = extract_defines(kernel_source)

    global_size = _eval_axes(grid_spec, dims, defines, "global")
    local_size = _eval_axes(grid_spec.get("local"), dims, defines, "local")

    # GridConfig.__post_init__ enforces positivity and global%local divisibility.
    grid = GridConfig(global_size, local_size, formulas=dict(grid_spec))
    logger.info("Evaluated grid: global=%s local=%s", grid.global_size, grid.local_size)
    return grid


def describe_grid_contract(grid_spec: dict[str, Any] | None, kernel_source: str) -> str:
    """Render the launch-grid contract for the optimizer prompt.

    Names the exact integer ``#define`` knobs that drive the grid — found by
    intersecting the grid formulas with the kernel's own ``#define``s — and shows
    the formulas plus each knob's current value. This tells the optimizer which
    ``#define``s it may tune and must not rename, whatever they happen to be
    called in this kernel/spec. If the kernel declares its own grid via a
    ``// xe-forge-grid:`` directive, that grid is shown (it overrides the spec).
    """
    directive = parse_grid_directive(kernel_source)
    effective = directive or grid_spec or _DEFAULT_GRID
    defines = extract_defines(kernel_source)
    local = effective.get("local")
    local = local if isinstance(local, dict) else None

    exprs = [effective[ax] for ax in ("x", "y", "z") if ax in effective]
    formulas = [f"  global.{ax} = {effective[ax]}" for ax in ("x", "y", "z") if ax in effective]
    local_exprs: list[Any] = []
    if local:
        local_exprs = [local[ax] for ax in ("x", "y", "z") if ax in local]
        exprs += local_exprs
        formulas += [f"  local.{ax} = {local[ax]}" for ax in ("x", "y", "z") if ax in local]

    symbols: set[str] = set()
    for expr in exprs:
        symbols |= set(re.findall(r"[A-Za-z_]\w*", str(expr)))
    symbols -= _FUNCS.keys()
    knobs = sorted(s for s in symbols if s in defines)
    dims = sorted(s for s in symbols if s not in defines)

    # Knobs that appear in a local (work-group size) formula are the cooperative
    # group-size levers: raising one past 1 forms a real thread group whose
    # members can share Shared Local Memory.
    local_symbols: set[str] = set()
    for expr in local_exprs:
        local_symbols |= set(re.findall(r"[A-Za-z_]\w*", str(expr)))
    local_knobs = {s for s in (local_symbols - _FUNCS.keys()) if s in defines}

    if knobs:
        knob_lines = "\n".join(
            f"    {k} = {defines[k]}"
            + ("   (work-group size: raise > 1 to form a cooperative thread group)"
               if k in local_knobs else "")
            for k in knobs
        )
    else:
        knob_lines = "    (no grid-driving #define found in the current kernel)"

    coop_note = ""
    if local_knobs:
        coop_note = (
            "\n\nCOOPERATIVE THREAD GROUPS: the work-group-size knob(s) above ("
            + ", ".join(sorted(local_knobs))
            + ") set how many threads share one thread group (#groups = global / local). "
            "At size 1 each thread runs alone, so Shared Local Memory and cm_barrier do "
            "nothing. Raise a work-group-size knob to make the threads in a group "
            "cooperate: partition the shared input tile across them by cm_local_id(...), "
            "stage it once in SLM (cm_store_slm / cm_load_slm) with cm_slm_fence + "
            "cm_barrier, then let every thread read it back — eliminating redundant global "
            "loads. Index each thread's own output tile with cm_global_id(...) (= "
            "cm_group_id * GROUP + cm_local_id) so it stays correct at any group size; a "
            "group then owns several adjacent tiles that share an input sub-tile."
        )

    return (
        "The harness computes the launch grid from these formulas"
        + (" (declared by this kernel's // xe-forge-grid: directive)"
           if directive else " (from the spec)")
        + ":\n"
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
        + _grid_directive_help()
        + coop_note
    )


def _grid_directive_help() -> str:
    """Steering paragraph: how (and why) a kernel may declare its OWN launch grid."""
    return (
        "\n\nDECLARE YOUR OWN GRID (// xe-forge-grid:) — the lever to change how much "
        "work each thread owns: the formulas above are only the DEFAULT. You may OVERRIDE "
        "the launch grid from inside the kernel with a comment line\n"
        "    // xe-forge-grid: x = <expr>; y = <expr>; z = <expr>; local.x = <expr>; local.y = <expr>\n"
        "whose expressions are over the problem dims and this kernel's integer #defines "
        "(same functions as above: ceil/floor/min/max). The harness then launches THIS grid "
        "instead of the spec's. This is the ONLY way to change the threads-vs-work-per-thread "
        "split — and it is the key to filling the DPAS systolic array: if the spec maps one "
        "thread per output element, a DPAS thread only has 1 row (RepeatCount=1) and cannot "
        "reuse weights. Add a per-thread tile-count #define (e.g. PH, PW), DIVIDE the matching "
        "grid axis by it in the directive so the harness launches proportionally FEWER threads, "
        "and have each thread compute that whole tile of output elements — now the thread has "
        "enough rows to run cm_dpas at RepeatCount=8 and amortize each loaded weight across the "
        "tile. Index every thread's tile from cm_global_id(...) consistently with the grid you "
        "declare, guard the tail, and keep the directive's expressions referencing only the "
        "dims and integer #defines shown above."
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
