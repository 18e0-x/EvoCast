"""Generic discovery of TFB objective/loss edit boundaries.

This module intentionally avoids model-name rules.  It reads Python source and
finds adapter-level ``_process`` methods that return the TFB ``out_loss`` dict
consumed by ``DeepForecastingModelBase``.  Those boundaries are the safe default
surface for loss/objective research; inner-model forward graphs remain the
default surface for architecture/data-flow research.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


OBJECTIVE_BOUNDARY_SCHEMA_VERSION = "objective_boundary_v1"


def _clean_path(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    marker = "ts_benchmark/"
    if marker in text:
        text = text[text.index(marker) :]
    return text


def _is_project_model_source(path: str) -> bool:
    clean = _clean_path(path)
    if not clean.startswith("ts_benchmark/"):
        return False
    if clean.endswith("__init__.py"):
        return False
    if "/research_variants/" in clean:
        return False
    # Shared framework/base files describe the training protocol; they are not
    # per-model objective edit targets.
    excluded_suffixes = {
        "ts_benchmark/baselines/deep_forecasting_model_base.py",
        "ts_benchmark/models/model_base.py",
    }
    if clean in excluded_suffixes:
        return False
    return clean.endswith(".py")


def _literal_string(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _dict_has_key(node: ast.AST, key: str) -> bool:
    if not isinstance(node, ast.Dict):
        return False
    return any(_literal_string(k) == key for k in node.keys if k is not None)


def _return_dict_keys(method: ast.FunctionDef) -> Set[str]:
    keys: Set[str] = set()
    for node in ast.walk(method):
        if not isinstance(node, ast.Return):
            continue
        value = node.value
        if isinstance(value, ast.Dict):
            for key in value.keys:
                text = _literal_string(key) if key is not None else ""
                if text:
                    keys.add(text)
    return keys


def _assigned_out_loss_names(method: ast.FunctionDef) -> Set[str]:
    names: Set[str] = set()
    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and _dict_has_key(node.value, "output"):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _returned_names(method: ast.FunctionDef) -> Set[str]:
    names: Set[str] = set()
    for node in ast.walk(method):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
            names.add(node.value.id)
    return names


def _subscript_key(node: ast.AST) -> str:
    if isinstance(node, ast.Subscript):
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return sl.value
    return ""


def _assigns_additional_loss(method: ast.FunctionDef, out_names: Set[str]) -> bool:
    for node in ast.walk(method):
        targets: Iterable[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        for target in targets:
            if not isinstance(target, ast.Subscript):
                continue
            if _subscript_key(target) != "additional_loss":
                continue
            owner = target.value
            if isinstance(owner, ast.Name) and (not out_names or owner.id in out_names):
                return True
    return False


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _inner_calls(method: ast.FunctionDef) -> List[str]:
    calls: List[str] = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if not name:
            continue
        if name.startswith("self.") and name not in calls:
            calls.append(name)
    return calls[:20]


def _class_methods(tree: ast.AST) -> Iterable[tuple[str, ast.FunctionDef]]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                yield node.name, child


def discover_objective_boundaries(
    *,
    source_files: Iterable[Dict[str, Any]],
    project_root: str | Path,
    inner_models: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return per-model objective boundaries discovered from source.

    The discovery is structural: it scans project source files for adapter
    ``_process`` methods that return a dict with an ``output`` key.  It never
    switches on model names.
    """

    root = Path(project_root)
    inner_sources: Set[str] = set()
    for model in list(inner_models or []):
        for item in list((model or {}).get("source_files") or []):
            path = _clean_path((item or {}).get("path"))
            if path:
                inner_sources.add(path)

    boundaries: List[Dict[str, Any]] = []
    seen: Set[tuple[str, str, str]] = set()
    for item in list(source_files or []):
        source = _clean_path((item or {}).get("path"))
        if not _is_project_model_source(source):
            continue
        path = root / source
        if not path.exists() or not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for class_name, method in _class_methods(tree):
            if method.name != "_process":
                continue
            out_names = _assigned_out_loss_names(method)
            return_keys = _return_dict_keys(method)
            returns_out_loss_name = bool(out_names & _returned_names(method))
            returns_output_dict = "output" in return_keys or returns_out_loss_name
            if not returns_output_dict:
                continue
            key = (source, class_name, method.name)
            if key in seen:
                continue
            seen.add(key)
            has_additional_loss = "additional_loss" in return_keys or _assigns_additional_loss(method, out_names)
            calls = _inner_calls(method)
            boundaries.append(
                {
                    "schema_version": OBJECTIVE_BOUNDARY_SCHEMA_VERSION,
                    "kind": "tfb_process_boundary",
                    "adapter_file": source,
                    "adapter_class": class_name,
                    "method": "_process",
                    "safe_objective_edit_file": source,
                    "returns_out_loss_dict": True,
                    "supports_additional_loss": True,
                    "already_emits_additional_loss": bool(has_additional_loss),
                    "inner_calls": calls,
                    "inner_model_files": sorted(inner_sources),
                    "source_role": "adapter_process_boundary",
                    "boundary_rule": (
                        "Emit additional_loss from adapter _process.  Final out_loss['output'] "
                        "must be a Tensor; do not let an inner forward dict become nested under output."
                    ),
                }
            )
    return boundaries


def normalize_tfb_process_output(out: Any) -> Dict[str, Any]:
    """Normalize a TFB model/_process output and flag nested dict mistakes."""

    if isinstance(out, dict):
        prediction = out.get("output")
        additional_loss = out.get("additional_loss")
        nested = isinstance(prediction, dict)
        return {
            "raw_kind": "dict",
            "prediction": prediction,
            "additional_loss": additional_loss,
            "has_additional_loss": additional_loss is not None,
            "nested_output_dict": nested,
            "error_type": "objective_boundary_mismatch" if nested else "",
            "error_message": (
                "out_loss['output'] is a dict.  This usually means an inner forward "
                "returned {'output': tensor, 'additional_loss': ...} and adapter _process "
                "wrapped that dict as the public prediction output."
                if nested
                else ""
            ),
        }
    return {
        "raw_kind": type(out).__name__,
        "prediction": out,
        "additional_loss": None,
        "has_additional_loss": False,
        "nested_output_dict": False,
        "error_type": "",
        "error_message": "",
    }

