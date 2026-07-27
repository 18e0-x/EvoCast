from __future__ import annotations

import difflib
import hashlib
import json
import shutil
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any


SNAPSHOT_SCHEMA_VERSION = "source_snapshot_v1"

SNAPSHOT_EXCLUDE_GLOBS = (
    ".git/**",
    ".git",
    ".evocast/**",
    "**/__pycache__/**",
    "__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.pyd",
    "**/*.dll",
    "**/*.exe",
)


def normalize_relpath(value: str | Path) -> str:
    return str(value or "").replace("\\", "/").strip().lstrip("./")


def is_excluded(path: str | Path) -> bool:
    normalized = normalize_relpath(path)
    return any(fnmatch(normalized, pattern) for pattern in SNAPSHOT_EXCLUDE_GLOBS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path) -> list[tuple[Path, str]]:
    base = Path(root).resolve()
    result: list[tuple[Path, str]] = []
    if not base.is_dir():
        return result
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.resolve().relative_to(base).as_posix()
        if is_excluded(rel):
            continue
        result.append((path, rel))
    return result


def source_manifest(root: str | Path) -> dict[str, Any]:
    base = Path(root).resolve()
    files = [
        {"path": rel, "sha256": sha256_file(path), "size": path.stat().st_size}
        for path, rel in _iter_files(base)
    ]
    manifest_hash = hashlib.sha256(
        json.dumps(files, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "root": str(base),
        "manifest_hash": manifest_hash,
        "snapshot_id": manifest_hash[:24],
        "files": files,
    }


def manifest_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        normalize_relpath(item.get("path")): dict(item)
        for item in list((manifest or {}).get("files") or [])
        if normalize_relpath(item.get("path"))
    }


def copy_source_tree(source_root: str | Path, destination_root: str | Path, *, overwrite: bool = True) -> list[dict[str, str]]:
    source = Path(source_root).resolve()
    destination = Path(destination_root).resolve()
    if not source.is_dir():
        raise RuntimeError(f"source snapshot root does not exist: {source}")
    if destination.exists() and overwrite:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    for src, rel in _iter_files(source):
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({"path": rel, "sha256": sha256_file(dst)})
    return copied


def changed_files_between(base_manifest: dict[str, Any], workspace: str | Path) -> list[str]:
    before = manifest_map(base_manifest)
    after_manifest = source_manifest(workspace)
    after = manifest_map(after_manifest)
    paths = sorted(set(before) | set(after))
    return [
        path
        for path in paths
        if before.get(path, {}).get("sha256") != after.get(path, {}).get("sha256")
    ]


def _read_text_or_marker(path: Path) -> tuple[list[str], bool]:
    if not path.is_file():
        return [], False
    data = path.read_bytes()
    if b"\x00" in data:
        return [f"<binary file sha256={hashlib.sha256(data).hexdigest()}>\n"], True
    try:
        return data.decode("utf-8").splitlines(keepends=True), False
    except UnicodeDecodeError:
        return [f"<non-utf8 file sha256={hashlib.sha256(data).hexdigest()}>\n"], True


def unified_diff_between(base_root: str | Path, candidate_root: str | Path, changed_files: list[str]) -> str:
    base = Path(base_root).resolve()
    candidate = Path(candidate_root).resolve()
    chunks: list[str] = []
    for raw in changed_files:
        rel = normalize_relpath(raw)
        if not rel:
            continue
        before_path = base / rel
        after_path = candidate / rel
        before, before_binary = _read_text_or_marker(before_path)
        after, after_binary = _read_text_or_marker(after_path)
        chunks.append(f"diff --snapshot a/{rel} b/{rel}\n")
        if before_binary or after_binary:
            chunks.append(f"--- a/{rel}\n")
            chunks.append(f"+++ b/{rel}\n")
            chunks.extend([f"-{line}" for line in before])
            chunks.extend([f"+{line}" for line in after])
            continue
        chunks.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="\n",
            )
        )
        if chunks and not chunks[-1].endswith("\n"):
            chunks[-1] += "\n"
    return "".join(chunks)


@dataclass(frozen=True)
class CandidateSnapshot:
    snapshot_id: str
    source_checkout: str
    manifest_path: str
    patch_path: str
    changed_files: list[str]
    manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "snapshot_id": self.snapshot_id,
            "source_checkout": self.source_checkout,
            "manifest_path": self.manifest_path,
            "patch_path": self.patch_path,
            "changed_files": list(self.changed_files),
            "manifest": dict(self.manifest),
        }


def materialize_candidate_snapshot(
    *,
    base_checkout: str | Path,
    agent_workspace: str | Path,
    destination_dir: str | Path,
    changed_files: list[str],
) -> CandidateSnapshot:
    base = Path(base_checkout).resolve()
    agent = Path(agent_workspace).resolve()
    destination = Path(destination_dir).resolve()
    source_checkout = destination / "source"
    copy_source_tree(base, source_checkout, overwrite=True)
    normalized_changed = [normalize_relpath(path) for path in changed_files if normalize_relpath(path)]
    for rel in normalized_changed:
        src = agent / rel
        dst = source_checkout / rel
        if src.exists():
            if not src.is_file():
                raise RuntimeError(f"changed path is not a regular file: {rel}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        elif dst.exists():
            dst.unlink()
    manifest = source_manifest(source_checkout)
    snapshot_id = str(manifest["snapshot_id"])
    manifest_path = destination / "manifest.json"
    patch_path = destination / "patch.diff"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    patch_path.write_text(
        unified_diff_between(base, source_checkout, normalized_changed),
        encoding="utf-8",
    )
    return CandidateSnapshot(
        snapshot_id=snapshot_id,
        source_checkout=str(source_checkout),
        manifest_path=str(manifest_path),
        patch_path=str(patch_path),
        changed_files=normalized_changed,
        manifest=manifest,
    )
