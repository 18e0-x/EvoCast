"""Baseline model binding and immutable source snapshot utilities."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import ast
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

from evocast.domain.atomic_io import atomic_write_json
from evocast.domain.formal_source import (
    FORMAL_SOURCE_EXCLUDE_PATTERNS,
    FORMAL_SOURCE_SCHEMA_VERSION,
    copy_formal_source_tree,
    formal_source_fingerprint,
)
from evocast.domain.knowledge_paths import agent_base_dir, task_knowledge_dir


SCHEMA_VERSION = "baseline_identity_v1"
SOURCE_BINDING_SCHEMA_VERSION = "source_binding_v1"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_hash(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return _sha256_bytes(canonical.encode("utf-8"))


def _binding_dir(base_dir: str, task_id: str) -> Path:
    path = task_knowledge_dir(base_dir, task_id) / "model_bindings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _repo_root() -> Path:
    return agent_base_dir().parent.resolve()


def _repo_rel(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(_repo_root()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _repo_rel_to(path: str | Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _is_tsl_model_file(path: Path, root: Path) -> bool:
    rel = _repo_rel_to(path, root).replace("\\", "/")
    return rel.startswith("ts_benchmark/baselines/time_series_library/models/")


def _import_from_candidates(
    *,
    entry_file: Path,
    node: ast.ImportFrom,
    package_root: Path,
    package_import_prefix: str,
) -> list[Path]:
    module = node.module or ""
    if node.level > 0:
        base = entry_file.parent
        for _ in range(max(0, int(node.level) - 1)):
            base = base.parent
        if module:
            return [base / Path(module.replace(".", "/") + ".py")]
        return []
    if module.startswith(package_import_prefix + "."):
        rel_module = module[len(package_import_prefix) + 1 :]
        return [package_root / Path(rel_module.replace(".", "/") + ".py")]
    return []


def _resolve_local_python_imports(
    *,
    entry_file: Path,
    root: Path,
    package_root: Path,
    package_import_prefix: str,
) -> list[str]:
    tree = ast.parse(entry_file.read_text(encoding="utf-8"))
    resolved: list[str] = []
    entry_rel = _repo_rel_to(entry_file, root)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for candidate in _import_from_candidates(
            entry_file=entry_file,
            node=node,
            package_root=package_root,
            package_import_prefix=package_import_prefix,
        ):
            if not candidate.is_file():
                continue
            rel = _repo_rel_to(candidate, root)
            if rel != entry_rel:
                resolved.append(rel)
    return list(dict.fromkeys(resolved))


def _tsl_model_source_binding(entry_file: Path, *, root: Path) -> Dict[str, Any]:
    entry_file = entry_file.resolve()
    source_root = "ts_benchmark/baselines/time_series_library"
    package_root = root / source_root
    entry_rel = _repo_rel_to(entry_file, root)
    referenced = _resolve_local_python_imports(
        entry_file=entry_file,
        root=root,
        package_root=package_root,
        package_import_prefix="ts_benchmark.baselines.time_series_library",
    )
    source_files = list(dict.fromkeys([entry_rel, *referenced]))
    return {
        "schema_version": SOURCE_BINDING_SCHEMA_VERSION,
        "entry_file": entry_rel,
        "source_root": source_root,
        "source_files": source_files,
        "core_files": [path for path in source_files if path != entry_rel and not path.endswith("/__init__.py")],
        "support_files": [path for path in source_files if path.endswith("/__init__.py")],
        "binding_policy": "tsl_entry_plus_direct_local_imports_editable",
    }


def _package_source_binding(module_file: Path, entry_file: Path, *, root: Path) -> Dict[str, Any]:
    package_dir = module_file.resolve().parent
    source_files = sorted(
        {
            _repo_rel_to(path, root)
            for path in package_dir.rglob("*.py")
            if "__pycache__" not in path.parts and path.is_file()
        }
    )
    entry_rel = _repo_rel_to(entry_file, root)
    if entry_rel not in source_files:
        source_files.insert(0, entry_rel)
    core_files = [
        path
        for path in source_files
        if path != entry_rel and not path.endswith("/__init__.py")
    ]
    support_files = [
        path
        for path in source_files
        if path.endswith("/__init__.py")
    ]
    return {
        "schema_version": SOURCE_BINDING_SCHEMA_VERSION,
        "entry_file": entry_rel,
        "source_root": _repo_rel_to(package_dir, root),
        "source_files": source_files,
        "core_files": core_files,
        "support_files": support_files,
        "binding_policy": "package_all_python_files_editable",
    }


def source_binding_from_entry_file(
    entry_file: str | Path,
    *,
    wrapper_module: str = "",
    repo_dir: str | Path | None = None,
) -> Dict[str, Any]:
    root = Path(repo_dir or _repo_root()).resolve()
    entry_path = Path(entry_file)
    if not entry_path.is_absolute():
        entry_path = root / entry_path
    entry_path = entry_path.resolve()
    if _is_tsl_model_file(entry_path, root):
        return _tsl_model_source_binding(entry_path, root=root)
    module_file = entry_path
    if wrapper_module:
        module = importlib.import_module(wrapper_module)
        module_file = Path(getattr(module, "__file__", "") or entry_path).resolve()
    return _package_source_binding(module_file, entry_path, root=root)


def _module_package_source_files(wrapper_module: str, entry_file: Path) -> Dict[str, Any]:
    return source_binding_from_entry_file(entry_file, wrapper_module=wrapper_module)


@dataclass(frozen=True)
class VerifiedModelBinding:
    model_key: str
    public_import_path: str
    wrapper_module: str
    wrapper_class: str
    constructor_signature: str
    source_file: str
    source_fingerprint: str
    adapter: str | None = None
    runtime_channel_contract: Dict[str, Any] = field(default_factory=dict)
    source_binding: Dict[str, Any] = field(default_factory=dict)
    verified_at: str = field(default_factory=_now)
    schema_version: str = SCHEMA_VERSION
    binding_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if not payload["binding_hash"]:
            payload["binding_hash"] = _stable_hash({key: value for key, value in payload.items() if key != "binding_hash"})
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VerifiedModelBinding":
        data = dict(payload)
        binding = cls(
            model_key=str(data["model_key"]),
            public_import_path=str(data["public_import_path"]),
            wrapper_module=str(data["wrapper_module"]),
            wrapper_class=str(data["wrapper_class"]),
            constructor_signature=str(data.get("constructor_signature") or ""),
            source_file=str(data.get("source_file") or ""),
            source_fingerprint=str(data.get("source_fingerprint") or ""),
            adapter=data.get("adapter"),
            runtime_channel_contract=dict(data.get("runtime_channel_contract") or {}),
            source_binding=dict(data.get("source_binding") or {}),
            verified_at=str(data.get("verified_at") or _now()),
            schema_version=str(data.get("schema_version") or SCHEMA_VERSION),
            binding_hash=str(data.get("binding_hash") or ""),
        )
        expected = binding.to_dict()["binding_hash"]
        if binding.binding_hash and binding.binding_hash != expected:
            legacy_data = {key: value for key, value in binding.to_dict().items() if key not in {"binding_hash", "source_binding"}}
            legacy_expected = _stable_hash(legacy_data)
            if "source_binding" in data or binding.binding_hash != legacy_expected:
                raise ValueError("verified model binding hash mismatch")
        return binding


def resolve_and_verify_model_binding(
    *,
    model_key: str,
    public_import_path: str,
    adapter: str | None = None,
    runtime_channel_contract: Mapping[str, Any] | None = None,
) -> VerifiedModelBinding:
    path = str(public_import_path or "").strip()
    if not path or "." not in path:
        raise ValueError("a fully-qualified public_import_path is required for model binding")
    public_module, exported_name = path.rsplit(".", 1)
    module = importlib.import_module(public_module)
    exported = getattr(module, exported_name)
    if not inspect.isclass(exported):
        raise TypeError(f"public import {path} did not resolve to a class")
    wrapper_module = str(exported.__module__)
    wrapper_class = str(exported.__name__)
    defining_module = importlib.import_module(wrapper_module)
    if getattr(defining_module, wrapper_class, None) is not exported:
        raise RuntimeError(f"model binding identity verification failed for {path}")
    source_file = inspect.getsourcefile(exported) or ""
    if not source_file:
        raise RuntimeError(f"model binding has no inspectable source file: {path}")
    source_path = Path(source_file)
    if not source_path.is_file():
        raise RuntimeError(f"model binding source file does not exist: {source_file}")
    try:
        signature = str(inspect.signature(exported))
    except (TypeError, ValueError):
        signature = ""
    binding = VerifiedModelBinding(
        model_key=str(model_key),
        public_import_path=path,
        wrapper_module=wrapper_module,
        wrapper_class=wrapper_class,
        constructor_signature=signature,
        source_file=str(source_path),
        source_fingerprint=_sha256_bytes(source_path.read_bytes()),
        adapter=str(adapter) if adapter else None,
        runtime_channel_contract=dict(runtime_channel_contract or {}),
        source_binding=_module_package_source_files(wrapper_module, source_path),
    )
    return VerifiedModelBinding.from_dict(binding.to_dict())


def persist_model_binding(base_dir: str, task_id: str, binding: VerifiedModelBinding, *, candidate_id: str) -> Dict[str, str]:
    payload = binding.to_dict()
    safe_id = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(candidate_id or binding.model_key))
    path = _binding_dir(base_dir, task_id) / f"{safe_id}.json"
    atomic_write_json(path, payload, ensure_ascii=False)
    return {"path": str(path), "binding_hash": payload["binding_hash"]}


def load_model_binding(path: str | Path) -> VerifiedModelBinding:
    return VerifiedModelBinding.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def verify_persisted_model_binding(path: str | Path) -> VerifiedModelBinding:
    binding = load_model_binding(path)
    source_path = Path(binding.source_file)
    if not source_path.is_file():
        raise RuntimeError("verified model binding source file no longer exists")
    if _sha256_bytes(source_path.read_bytes()) != binding.source_fingerprint:
        raise RuntimeError("verified model binding source fingerprint is stale")
    return binding


@dataclass(frozen=True)
class ImmutableBaselineSnapshot:
    snapshot_id: str
    source_root: str
    source_fingerprint: str
    binding_hash: str
    source_schema_version: str = FORMAL_SOURCE_SCHEMA_VERSION
    excluded_patterns: list[str] = field(default_factory=lambda: list(FORMAL_SOURCE_EXCLUDE_PATTERNS))
    created_at: str = field(default_factory=_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tree_fingerprint(root: Path) -> str:
    repo = agent_base_dir().parent
    return formal_source_fingerprint(repo, root)


def create_immutable_baseline_snapshot(base_dir: str, task_id: str, binding: VerifiedModelBinding) -> ImmutableBaselineSnapshot:
    repo = agent_base_dir().parent
    source = repo / "ts_benchmark"
    if not source.is_dir():
        raise RuntimeError(f"baseline source tree not found: {source}")
    fingerprint = formal_source_fingerprint(repo, source)
    snapshot_id = _stable_hash({
        "binding_hash": binding.to_dict()["binding_hash"],
        "source_fingerprint": fingerprint,
        "source_schema_version": FORMAL_SOURCE_SCHEMA_VERSION,
    })[:20]
    root = task_knowledge_dir(base_dir, task_id) / "baseline_snapshots" / snapshot_id
    source_root = root / "source" / "ts_benchmark"
    manifest = root / "snapshot.json"
    if not source_root.exists():
        source_root.parent.mkdir(parents=True, exist_ok=True)
        copy_formal_source_tree(repo_root=repo, source_root=source, destination_root=source_root.parent)
    snapshot = ImmutableBaselineSnapshot(
        snapshot_id=snapshot_id,
        source_root=str(source_root),
        source_fingerprint=fingerprint,
        binding_hash=binding.to_dict()["binding_hash"],
    )
    atomic_write_json(manifest, snapshot.to_dict(), ensure_ascii=False)
    return snapshot


def verify_immutable_baseline_snapshot(
    snapshot: ImmutableBaselineSnapshot | Mapping[str, Any],
    binding: VerifiedModelBinding,
) -> ImmutableBaselineSnapshot:
    data = snapshot.to_dict() if isinstance(snapshot, ImmutableBaselineSnapshot) else dict(snapshot)
    checked = ImmutableBaselineSnapshot(
        snapshot_id=str(data["snapshot_id"]),
        source_root=str(data["source_root"]),
        source_fingerprint=str(data["source_fingerprint"]),
        binding_hash=str(data["binding_hash"]),
        source_schema_version=str(data.get("source_schema_version") or FORMAL_SOURCE_SCHEMA_VERSION),
        excluded_patterns=list(data.get("excluded_patterns") or FORMAL_SOURCE_EXCLUDE_PATTERNS),
        created_at=str(data.get("created_at") or _now()),
        schema_version=str(data.get("schema_version") or SCHEMA_VERSION),
    )
    if checked.binding_hash != binding.to_dict()["binding_hash"]:
        raise ValueError("baseline snapshot and model binding do not match")
    root = Path(checked.source_root)
    repo = agent_base_dir().parent
    if not root.is_dir() or formal_source_fingerprint(repo, root) != checked.source_fingerprint:
        raise ValueError("immutable baseline snapshot source fingerprint is stale")
    try:
        relative_source = Path(binding.source_file).resolve().relative_to((repo / "ts_benchmark").resolve())
    except ValueError as exc:
        raise ValueError("model binding source is outside the baseline source tree") from exc
    copied_source = root / relative_source
    if not copied_source.is_file() or _sha256_bytes(copied_source.read_bytes()) != binding.source_fingerprint:
        raise ValueError("immutable baseline snapshot does not contain the verified model source")
    return checked
