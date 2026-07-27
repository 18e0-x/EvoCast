from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable


def _is_workspace_path(value: object) -> bool:
    text = str(value or "").replace("\\", "/")
    return (
        "/round_sources/" in text
        or (
            "/task_knowledge/" in text
            and (
                "/rounds/Research" in text
                or "/rounds/Ablation" in text
                or "/workspace/" in text
            )
        )
    )


def _safe_resolved(value: object) -> str:
    try:
        return str(Path(str(value)).resolve())
    except Exception:
        return str(value or "")


def purge_workspace_modules(*, purge_ts_benchmark: bool = False) -> list[str]:
    """Remove modules that may have been imported from a variant workspace.

    Workspace variants shadow ``ts_benchmark`` with edited copies.  Removing the
    workspace path from ``sys.path`` is not enough: Python keeps already imported
    modules in ``sys.modules``.  Baseline/reference runs must therefore purge
    workspace modules before importing model code again.
    """

    removed: list[str] = []
    for name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", "")
        remove = False
        if name.startswith("evocast_workspace"):
            remove = True
        elif _is_workspace_path(module_file):
            remove = True
        elif purge_ts_benchmark and (name == "ts_benchmark" or name.startswith("ts_benchmark.")):
            remove = True
        if remove:
            sys.modules.pop(name, None)
            removed.append(name)
    sys.path[:] = [item for item in sys.path if not _is_workspace_path(_safe_resolved(item))]
    importlib.invalidate_caches()
    return removed


def collect_variant_paths(model_config: dict | None) -> list[str]:
    paths: list[str] = []
    for entry in list((model_config or {}).get("models") or []):
        if isinstance(entry, dict) and str(entry.get("variant_path") or "").strip():
            paths.append(str(entry.get("variant_path")).strip())
    return paths


@contextmanager
def model_execution_import_context(
    variant_paths: Iterable[str] | None = None,
    *,
    source_checkout: str | Path | None = None,
):
    """Import boundary for one pipeline execution.

    Baseline executions enter with no variant paths and must start from a clean
    repository import state.  Variant executions may load workspace modules, but
    those modules are purged again when the run completes so later baseline or
    reference runs cannot reuse edited workspace code.
    """

    has_variant = any(str(item or "").strip() for item in list(variant_paths or []))
    checkout_path = Path(source_checkout).resolve() if source_checkout else None
    inserted_checkout = False
    if checkout_path:
        if not checkout_path.is_dir():
            raise RuntimeError(f"source_checkout is not a directory: {checkout_path}")
        purge_workspace_modules(purge_ts_benchmark=True)
        checkout_text = str(checkout_path)
        if checkout_text not in sys.path:
            sys.path.insert(0, checkout_text)
            inserted_checkout = True
    elif not has_variant:
        purge_workspace_modules(purge_ts_benchmark=True)
    try:
        yield
    finally:
        if inserted_checkout and checkout_path:
            checkout_text = str(checkout_path)
            sys.path[:] = [item for item in sys.path if str(Path(item).resolve()) != checkout_text]
        purge_workspace_modules(purge_ts_benchmark=True)
