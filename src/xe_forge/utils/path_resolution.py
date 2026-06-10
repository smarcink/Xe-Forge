"""Helpers for resolving symlink-backed repo files across platforms."""

from pathlib import Path


def resolve_linked_path(path: str | Path, *, max_hops: int = 8) -> Path:
    """Resolve real symlinks and git symlink placeholders to a concrete file path."""
    current = Path(path)
    seen: set[Path] = set()

    for _ in range(max_hops):
        marker = current.resolve(strict=False)
        if marker in seen:
            break
        seen.add(marker)

        if current.is_symlink():
            current = current.resolve()
            continue

        if not current.is_file():
            return current

        try:
            raw = current.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return current

        if not raw or "\n" in raw or "\r" in raw:
            return current

        target = Path(raw)
        if not target.is_absolute():
            target = current.parent / target

        if not target.exists() or not target.is_file():
            return current

        current = target.resolve()

    return current


def read_linked_text(path: str | Path, *, encoding: str = "utf-8") -> str:
    """Read text from a path after resolving symlinks or git symlink placeholders."""
    return resolve_linked_path(path).read_text(encoding=encoding)