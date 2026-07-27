from __future__ import annotations

from pathlib import Path
from typing import Optional


def package_root() -> Path:
    """Return the Python package root: <repo>/evocast."""
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """Return the repository root."""
    return package_root().parent


def _looks_like_repo_root(path: Path) -> bool:
    return (path / "evocast").is_dir() and (path / "ts_benchmark").is_dir()


def _looks_like_package_root(path: Path) -> bool:
    return (path / "core").is_dir() and (path / "harness").is_dir()


def runtime_root(base_dir: Optional[str] = None) -> Path:
    """Return the writable runtime root.

    Repository/package roots are never used directly for runtime artifacts.
    Their task state, runs, and shared knowledge are stored under <repo>/.evocast.
    Temporary or explicitly external roots keep their historical layout unless
    the caller already passes a .evocast directory.
    """
    if base_dir is None:
        return repo_root() / ".evocast"
    base = Path(base_dir).resolve()
    if base.name == ".evocast":
        return base
    if _looks_like_repo_root(base):
        return base / ".evocast"
    if _looks_like_package_root(base):
        return base.parent / ".evocast"
    return base


def agent_base_dir(base_dir: Optional[str] = None) -> Path:
    """Return the source package root, not the runtime artifact root."""
    if base_dir is not None:
        base = Path(base_dir).resolve()
        if _looks_like_package_root(base):
            return base
        if _looks_like_repo_root(base):
            return base / "evocast"
    return package_root()


def task_knowledge_root(base_dir: Optional[str] = None) -> Path:
    return runtime_root(base_dir) / "task_knowledge"


def task_knowledge_dir(base_dir: Optional[str], task_id: str) -> Path:
    return task_knowledge_root(base_dir) / str(task_id)


def runs_root(base_dir: Optional[str] = None) -> Path:
    return runtime_root(base_dir) / "runs"


def task_runs_dir(base_dir: Optional[str], task_id: str) -> Path:
    return runs_root(base_dir) / str(task_id)


def dataset_knowledge_root(base_dir: Optional[str] = None) -> Path:
    return runtime_root(base_dir) / "dataset_knowledge"


def dataset_registry_path(base_dir: Optional[str] = None) -> Path:
    return dataset_knowledge_root(base_dir) / "registry.json"


def dataset_record_dir(base_dir: Optional[str], dataset_id: str) -> Path:
    return dataset_knowledge_root(base_dir) / "datasets" / str(dataset_id)


def dataset_view_dir(base_dir: Optional[str], dataset_id: str, view_id: str) -> Path:
    return dataset_record_dir(base_dir, dataset_id) / "views" / str(view_id)


def dataset_task_binding_path(base_dir: Optional[str], task_id: str) -> Path:
    return dataset_knowledge_root(base_dir) / "task_bindings" / f"{task_id}.json"


def baseline_knowledge_root(base_dir: Optional[str] = None) -> Path:
    return runtime_root(base_dir) / "baseline_knowledge"


def baseline_evaluation_dir(base_dir: Optional[str], signature_hash: str) -> Path:
    return baseline_knowledge_root(base_dir) / "evaluations" / str(signature_hash)


def baseline_task_binding_path(base_dir: Optional[str], task_id: str) -> Path:
    return baseline_knowledge_root(base_dir) / "tasks" / f"{task_id}.json"


def baseline_knowledge_registry_path(base_dir: Optional[str] = None) -> Path:
    return baseline_knowledge_root(base_dir) / "registry.json"


def shared_knowledge_dir(base_dir: Optional[str] = None) -> Path:
    """Global, cross-task cache directory for model cards/structure facts."""
    return runtime_root(base_dir) / "knowledge"
