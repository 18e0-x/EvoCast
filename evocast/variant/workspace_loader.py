"""Single entry point for loading Model classes from variant paths.

ALL variant loading — preflight, smoke, runtime probe, experiment — must go
through ``load_model_class()``.  Baseline models (no variant) are also handled
by the same function, making it the one call site for "give me a Model class."

Principle: a ``variant_path`` is a file path to a ``round_entry.py`` in a
task workspace.  The loader handles two runtime cases:

1. Workspace path  →  file-path import of round_entry.py
2. None / empty     →  standard importlib import of baseline model

All intra-repo import caching is purged on each load so repair cycles always
see fresh code.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Optional, Type

from evocast.domain.execution_ids import (
    extract_execution_label_from_variant_path,
    format_research_id,
    is_ablation_workspace_path,
    is_research_source_path,
    is_research_workspace_path,
    normalize_path_text,
)

# ── Stable module namespace ──────────────────────────────────────────────
_WORKSPACE_MODULE_PREFIX = "evocast_workspace"


# ── Path utilities ───────────────────────────────────────────────────────

def _abs_path(variant_path: str) -> Path:
    p = Path(str(variant_path))
    return p.resolve() if p.is_absolute() else p.resolve()


def workspace_module_name(variant_path: str) -> str:
    """Generate a stable, unique module name for a workspace entry file.

    Format: ``evocast_workspace_<task_id>_<execution_label>``
    """
    path = _abs_path(variant_path)
    parts = path.parts
    task_id = ""
    execution_label = ""
    for i, part in enumerate(parts):
        if part == "round_sources" and i + 2 < len(parts):
            task_id = parts[i + 1]
            execution_label = parts[i + 2]
    execution_label = extract_execution_label_from_variant_path(str(path))
    if task_id and execution_label:
        safe_task = re.sub(r"[^0-9a-zA-Z_]+", "_", task_id)
        safe_round = re.sub(r"[^0-9a-zA-Z_]+", "_", execution_label)
        return f"{_WORKSPACE_MODULE_PREFIX}_{safe_task}_{safe_round}"
    import hashlib
    path_hash = hashlib.sha1(str(path).encode()).hexdigest()[:12]
    return f"{_WORKSPACE_MODULE_PREFIX}_{path_hash}"


def _ensure_workspace_on_path(workspace_root: Path) -> None:
    ws_str = str(workspace_root.resolve())
    if ws_str not in sys.path:
        sys.path.insert(0, ws_str)


def _remove_workspace_from_path(workspace_root: Path) -> None:
    ws_str = str(workspace_root.resolve())
    sys.path[:] = [item for item in sys.path if str(Path(item).resolve()) != ws_str]


def _purge_workspace_shadowed_modules(workspace_root: Path) -> None:
    """Remove cached repo modules that have workspace copies.

    Python will not re-resolve ``ts_benchmark`` imports just because the
    workspace is placed earlier on sys.path.  Any already-imported package keeps
    its old __path__, so exact-edits silently execute the global baseline.
    Purge modules that are present in this workspace before loading round_entry.
    """
    root = workspace_root.resolve()
    prefixes: set[str] = set()
    for package_dir in ("ts_benchmark",):
        if (root / package_dir).exists():
            prefixes.add(package_dir)
    if not prefixes:
        return
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            sys.modules.pop(name, None)


def is_workspace_variant_path(variant_path: str) -> bool:
    """Return True if *variant_path* points to a task workspace entry file."""
    normalized = normalize_path_text(variant_path)
    return is_research_source_path(normalized)


def variant_path_to_display_name(variant_path: str) -> str:
    """Convert a workspace variant path to a human-readable display name."""
    normalized = normalize_path_text(variant_path)
    parts = normalized.split("/")
    task_id = ""
    execution_label = ""
    for i, part in enumerate(parts):
        if part == "round_sources" and i + 2 < len(parts):
            task_id = parts[i + 1]
            execution_label = parts[i + 2]
    execution_label = extract_execution_label_from_variant_path(normalized)
    if task_id and execution_label:
        return f"{task_id}/{execution_label}"
    return workspace_module_name(normalized)


def normalize_variant_path(variant_path: str, *, base_dir: Optional[str] = None) -> str:
    """Normalize a variant_path to an absolute file path.

    Handles:
    - Already-absolute paths
    - Relative paths (resolved against base_dir or CWD)
    - Empty string → empty string (baseline, no variant)
    """
    if not variant_path or not str(variant_path).strip():
        return ""
    raw = str(variant_path).replace("\\", "/").strip()
    if not raw:
        return ""

    p = Path(raw)
    if p.is_absolute():
        return str(p.resolve()).replace("\\", "/")

    if base_dir:
        p = (Path(base_dir) / raw).resolve()
        return str(p).replace("\\", "/")

    return str(p.resolve()).replace("\\", "/")


def resolve_entry_path(variant_path: str) -> str:
    """Resolve a variant path to the actual round_entry.py file.

    If variant_path already points to a .py file, returns it normalized.
    If variant_path is a directory, looks for round_entry.py inside.
    """
    normalized = normalize_variant_path(variant_path)
    if not normalized:
        return ""
    p = Path(normalized)
    if p.is_file() and p.suffix == ".py":
        return normalized
    if p.is_dir():
        entry = p / "round_entry.py"
        if entry.is_file():
            return str(entry.resolve()).replace("\\", "/")
    return normalized


# ── Core loading functions ────────────────────────────────────────────────

def load_module_from_variant_path(variant_path: str, *, keep_workspace_on_path: bool = False) -> Any:
    """Import a Python module from a workspace entry file.

    ALWAYS reloads from disk — purges __pycache__, invalidates import
    caches, and removes the module from sys.modules before loading.
    This ensures repair/modification cycles see the latest code.
    """
    path = _abs_path(variant_path)
    if not path.exists() or not path.is_file():
        raise ImportError(f"Variant entry file not found: {variant_path}")

    mod_name = workspace_module_name(variant_path)

    importlib.invalidate_caches()
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    pycache = path.parent / "__pycache__"
    if pycache.exists():
        stem = path.stem
        for cached in pycache.glob(f"{stem}*.pyc"):
            try:
                cached.unlink()
            except OSError:
                pass

    workspace_root = path.parent
    _purge_workspace_shadowed_modules(workspace_root)
    _ensure_workspace_on_path(workspace_root)

    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for: {variant_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    finally:
        # The workspace package modules remain in sys.modules for the loaded
        # variant class, but the workspace root must not keep shadowing future
        # baseline imports in the same long-lived agent process.
        if not keep_workspace_on_path:
            _remove_workspace_from_path(workspace_root)

    return module


def load_model_class_from_variant_path(variant_path: str) -> type:
    """Import the ``Model`` class from a workspace entry file."""
    module = load_module_from_variant_path(variant_path)
    if hasattr(module, "Model"):
        model_cls = getattr(module, "Model")
        if isinstance(model_cls, type):
            return model_cls
        raise AttributeError(
            f"Model attribute in {variant_path} is {type(model_cls).__name__}, not a class"
        )
    raise AttributeError(f"No 'Model' attribute found in {variant_path}")


# ── Unified entry point ───────────────────────────────────────────────────

def load_model_class(
    variant_path: Optional[str] = None,
    *,
    model_name: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> type:
    """**Single entry point** for loading a Model class.

    Handles both cases:
    1. Workspace variant path  →  file-path import of round_entry.py
    2. Baseline (no variant)   →  standard importlib.import_module

    Args:
        variant_path: File path to variant entry (round_entry.py or dir).
        model_name: Fully-qualified import path for baseline models
                    (e.g. ``ts_benchmark.baselines.time_series_library.PatchTST``).
        base_dir: Root directory for resolving relative paths.

    Returns:
        The Model class (a ``type``).

    Raises:
        ImportError: When the module cannot be loaded.
        AttributeError: When the Model class is not found.
    """
    # ── Case 1 & 2: variant path provided ────────────────────────────
    if variant_path and str(variant_path).strip():
        entry = resolve_entry_path(variant_path)
        if entry and Path(entry).is_file():
            return load_model_class_from_variant_path(entry)
        # Path doesn't point to a file — fall through to baseline

    # ── Case 3: baseline model (by name) ─────────────────────────────
    name = str(model_name or "").strip()
    if not name:
        raise ImportError(
            "No variant_path or model_name provided — cannot load Model class."
        )

    # Strip global. prefix if present
    if name.startswith("global."):
        name = name[len("global."):]

    if "." not in name:
        raise ImportError(
            f"model_name '{name}' is not a fully-qualified import path."
        )

    module_name, attr = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, attr, None)
    if cls is None:
        raise AttributeError(
            f"'{attr}' not found in module '{module_name}'"
        )
    if not isinstance(cls, type):
        raise AttributeError(
            f"'{attr}' in '{module_name}' is {type(cls).__name__}, not a class"
        )
    return cls


def variant_module_name(variant_path: str) -> str:
    """Return a display-only identifier for a variant path."""
    return variant_path_to_display_name(variant_path)
