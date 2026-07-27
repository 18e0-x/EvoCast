"""Formal benchmark source copy and fingerprint utilities.

The formal source tree is the checked-in ``ts_benchmark`` implementation used
by baseline models. It deliberately excludes generated research candidates and
interpreter/cache artifacts.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, List, Tuple


FORMAL_SOURCE_SCHEMA_VERSION = "formal_source_v1"

FORMAL_SOURCE_EXCLUDE_PATTERNS = (
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.pyd",
    "**/*.dll",
    "**/*.exe",
    "ts_benchmark/baselines/research_variants/**",
)


@dataclass(frozen=True)
class CopiedSourceFile:
    path: str
    sha1: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha1": self.sha1}


def normalize_source_relpath(value: str | Path) -> str:
    return str(value).replace("\\", "/").strip().lstrip("./")


def is_formal_source_excluded(relative_path: str | Path) -> bool:
    normalized = normalize_source_relpath(relative_path)
    return any(fnmatch(normalized, pattern) for pattern in FORMAL_SOURCE_EXCLUDE_PATTERNS)


def _canonical_source_relpath(path: Path, repo_root: Path, source_root: Path) -> str:
    source = source_root.resolve()
    try:
        rel = path.resolve().relative_to(source).as_posix()
        if source.name == "ts_benchmark":
            return f"ts_benchmark/{rel}" if rel else "ts_benchmark"
    except ValueError:
        pass
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        rel = path.resolve().relative_to(source).as_posix()
        return f"{source.name}/{rel}" if rel else source.name


def iter_formal_source_files(repo_root: Path, source_root: Path | None = None) -> Iterable[Tuple[Path, str]]:
    root = Path(repo_root).resolve()
    source = Path(source_root).resolve() if source_root is not None else root / "ts_benchmark"
    if not source.is_dir():
        return
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = _canonical_source_relpath(path, root, source)
        if is_formal_source_excluded(rel):
            continue
        yield path, rel


def sha1_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def formal_source_fingerprint(repo_root: Path, source_root: Path | None = None) -> str:
    root = Path(repo_root).resolve()
    digest = hashlib.sha256()
    for path, rel in iter_formal_source_files(root, source_root):
        digest.update(rel.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def copy_formal_source_tree(
    *,
    repo_root: Path,
    source_root: Path | None = None,
    destination_root: Path,
    overwrite: bool = False,
) -> List[dict[str, str]]:
    root = Path(repo_root).resolve()
    source = Path(source_root).resolve() if source_root is not None else root / "ts_benchmark"
    destination = Path(destination_root)
    copied: list[dict[str, str]] = []
    if not source.is_dir():
        return copied
    for src, rel in iter_formal_source_files(root, source):
        dst = destination / rel
        if dst.exists() and not overwrite:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(CopiedSourceFile(path=rel, sha1=sha1_file(src)).to_dict())
    return copied
