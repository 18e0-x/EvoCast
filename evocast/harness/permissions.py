"""Permission helpers for the EvoCast v3 harness."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from evocast.variant.protected_paths import check_file_paths, validate_variant_path


class PermissionError(ValueError):
    """Raised when a tool request violates harness permissions."""


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DENIED_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
}

DENIED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".dll",
    ".exe",
}

READ_ALLOWED_PREFIXES = (
    ".evocast/",
    "evocast/",
    "ts_benchmark/",
    "scripts/",
    "config/",
)


def repo_root() -> Path:
    return PROJECT_ROOT


def normalize_repo_path(path: str | os.PathLike[str]) -> str:
    raw = str(path or "").replace("\\", "/").strip()
    if not raw:
        raise PermissionError("empty path")
    candidate = Path(raw)
    resolved = candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    try:
        rel = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise PermissionError(f"path outside repository: {path}") from exc
    return rel


def resolve_read_path(path: str | os.PathLike[str]) -> Path:
    rel = normalize_repo_path(path)
    if Path(rel).name in DENIED_FILE_NAMES:
        raise PermissionError(f"reading secret-like file is denied: {rel}")
    if Path(rel).suffix.lower() in DENIED_SUFFIXES:
        raise PermissionError(f"reading binary/generated file is denied: {rel}")
    if not rel.startswith(READ_ALLOWED_PREFIXES):
        raise PermissionError(
            "path is not in the v2 read whitelist: "
            f"{rel}. Allowed prefixes: {', '.join(READ_ALLOWED_PREFIXES)}"
        )
    resolved = (PROJECT_ROOT / rel).resolve()
    if not resolved.is_file():
        raise PermissionError(f"path does not exist or is not a file: {rel}")
    return resolved


def resolve_read_dir(path: str | os.PathLike[str]) -> Path:
    rel = normalize_repo_path(path)
    if not rel.endswith("/"):
        rel_for_prefix = rel + "/"
    else:
        rel_for_prefix = rel
    if Path(rel).name in DENIED_FILE_NAMES:
        raise PermissionError(f"reading secret-like directory is denied: {rel}")
    if not rel_for_prefix.startswith(READ_ALLOWED_PREFIXES):
        raise PermissionError(
            "path is not in the v2 read whitelist: "
            f"{rel}. Allowed prefixes: {', '.join(READ_ALLOWED_PREFIXES)}"
        )
    resolved = (PROJECT_ROOT / rel).resolve()
    if not resolved.is_dir():
        raise PermissionError(f"path does not exist or is not a directory: {rel}")
    return resolved


def assert_write_paths_allowed(paths: Iterable[str]) -> list[str]:
    normalized = [normalize_repo_path(path) for path in paths]
    check = check_file_paths(normalized)
    if not check.get("ok"):
        details = "; ".join(f"{path}: {reason}" for path, reason in check.get("violations", []))
        raise PermissionError(f"write path denied: {details}")
    return list(check.get("allowed", []))


def assert_variant_path(path: str) -> str:
    rel = normalize_repo_path(path)
    check = validate_variant_path(rel)
    if not check.get("ok"):
        raise PermissionError(str(check.get("error") or f"invalid variant path: {rel}"))
    return rel

