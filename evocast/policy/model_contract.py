"""Pre-run validation for generated model entries.

The benchmark already validates model loading at runtime.  These checks move
cheap, deterministic failures to the harness boundary so the agent gets one
clear rejection instead of spending a full experiment turn.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, List, Optional

from evocast.research.model_registry import build_registry


class ModelContractError(ValueError):
    """Raised when a model entry violates a deterministic harness contract."""


def available_adapters() -> List[str]:
    try:
        from ts_benchmark.baselines import ADAPTER

        return sorted(str(key) for key in ADAPTER)
    except Exception:
        return []


def _import_attribute(path: str) -> Any:
    normalized = str(path or "").strip()
    if normalized.startswith("global."):
        normalized = normalized[len("global.") :]
    module_name, attr = normalized.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _candidate_model_paths(model_name: str) -> List[str]:
    name = str(model_name or "").strip()
    if not name:
        return []
    candidates = []
    if name.startswith("global."):
        candidates.append(name)
    if not name.startswith("ts_benchmark.baselines.") and "." not in name:
        candidates.append(f"ts_benchmark.baselines.{name}")
    candidates.append(name)
    seen = set()
    ordered = []
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _model_info_for_contract(model_name: str, *, variant_path: Optional[str] = None) -> Any:
    """Import the model class/registry entry.

    When variant_path is provided, uses workspace_loader which handles
    workspace variants, legacy paths, and baseline models through a
    single entry point.  Otherwise falls back to importlib-based
    candidate search (backward compat).
    """
    if variant_path:
        from evocast.variant.workspace_loader import load_model_class as _ws_load_model_class
        # When caller explicitly provides variant_path, the workspace loader
        # is authoritative — fallthrough to importlib would hide real errors
        # (e.g. ModuleNotFoundError for a workspace module that importlib
        # cannot resolve by model_name alone).
        return _ws_load_model_class(variant_path=variant_path, model_name=model_name or "")
    last_error: Exception | None = None
    for candidate in _candidate_model_paths(model_name):
        try:
            return _import_attribute(candidate)
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ImportError("empty model_name")


def _required_hparams(model_info: Any) -> Dict[str, Any]:
    if isinstance(model_info, dict):
        return dict(model_info.get("required_hyper_params") or {})
    if hasattr(model_info, "required_hyper_params") and callable(model_info.required_hyper_params):
        return dict(model_info.required_hyper_params() or {})
    return {}


def _resolve_registry_spec(entry: Dict[str, Any]) -> Dict[str, Any]:
    candidates = [
        str(entry.get("model_key") or "").strip(),
        str(entry.get("base_model_key") or "").strip(),
        str(entry.get("source_model_key") or "").strip(),
        str(entry.get("model_name") or "").strip(),
    ]
    normalized = {item for item in candidates if item}
    normalized.update(item[len("global.") :] for item in list(normalized) if item.startswith("global."))
    for spec in build_registry(verify=False):
        aliases = {
            str(spec.get("model_key") or ""),
            str(spec.get("import_path") or ""),
            f"global.{str(spec.get('import_path') or '')}",
        }
        if normalized & aliases:
            return spec
    return {}


def _schema_type_issue(key: str, value: Any, rule: Dict[str, Any]) -> Optional[str]:
    expected = str(rule.get("type") or "").strip().lower()
    if not expected:
        return None
    if expected == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            return f"model_hyper_params.{key} must be int, got {type(value).__name__}"
    elif expected in {"float", "numeric", "number"}:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"model_hyper_params.{key} must be numeric, got {type(value).__name__}"
    elif expected == "bool":
        if not isinstance(value, bool):
            return f"model_hyper_params.{key} must be bool, got {type(value).__name__}"
    elif expected == "str":
        if not isinstance(value, str):
            return f"model_hyper_params.{key} must be str, got {type(value).__name__}"
    elif expected == "int_list":
        if not isinstance(value, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            return f"model_hyper_params.{key} must be list[int], got {type(value).__name__}"
        min_length = int(rule.get("min_length") or 0)
        if min_length and len(value) < min_length:
            return f"model_hyper_params.{key} must contain at least {min_length} item(s)"
    elif expected == "int_or_int_list":
        if isinstance(value, int) and not isinstance(value, bool):
            return None
        if isinstance(value, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            min_length = int(rule.get("min_length") or 0)
            if min_length and len(value) < min_length:
                return f"model_hyper_params.{key} must contain at least {min_length} item(s)"
            return None
        return f"model_hyper_params.{key} must be int or list[int], got {type(value).__name__}"
    return None


def validate_model_config(
    model_entry: Dict[str, Any],
    *,
    recommend_model_hyper_params: Optional[Dict[str, Any]] = None,
    baseline_validated_hyper_params: Optional[Dict[str, Any]] = None,
    require_import: bool = True,
    variant_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a structured validation result and raise on hard errors.

    variant_path: when provided, the import check routes through
    workspace_loader.load_model_class() instead of importlib-only
    candidate search.  This gives a single loading path for workspace
    variants, legacy research_variants, and baseline models.
    """
    entry = dict(model_entry or {})
    errors: List[str] = []
    warnings: List[str] = []
    adapter = entry.get("adapter")
    adapters = available_adapters()
    if adapter is not None and str(adapter) not in adapters:
        errors.append(
            "Unknown adapter "
            f"{adapter!r}. Available adapters: {', '.join(adapters) if adapters else '(adapter registry unavailable)'}"
        )

    registry_spec = _resolve_registry_spec(entry)
    hparam_schema = dict(registry_spec.get("hparam_schema") or {})
    inherited_hparams = dict(baseline_validated_hyper_params or {})
    hparams = dict(entry.get("model_hyper_params") or {})
    for key, value in hparams.items():
        schema_rule = hparam_schema.get(key) or hparam_schema.get(str(key).lower())
        has_schema_rule = isinstance(schema_rule, dict)
        issue = _schema_type_issue(key, value, schema_rule) if has_schema_rule else None
        if issue:
            errors.append(issue)

    model_name = str(entry.get("model_name") or "").strip()
    required: Dict[str, Any] = {}
    if require_import and model_name:
        try:
            model_info = _model_info_for_contract(model_name, variant_path=variant_path)
            required = _required_hparams(model_info)
        except Exception as exc:
            errors.append(f"model_name cannot be imported during preflight: {type(exc).__name__}: {exc}")

    if required:
        recommended = dict(recommend_model_hyper_params or {})
        missing = [
            param_name
            for param_name, std_name in required.items()
            if param_name not in hparams and std_name not in recommended
        ]
        if missing:
            errors.append(f"required_hyper_params missing after preflight: {missing}")

    if model_name and not inspect.isclass(entry.get("model_factory", object)):
        warnings.append("model preflight checked import path and config only; runtime shape is validated by experiments")

    result = {
        "status": "ok" if not errors else "error",
        "errors": errors,
        "warnings": warnings,
        "adapter": adapter,
        "available_adapters": adapters,
        "required_hyper_params": required,
        "model_key": registry_spec.get("model_key") or entry.get("model_key"),
        "hparam_schema": hparam_schema,
    }
    if errors:
        raise ModelContractError("; ".join(errors))
    return result


def training_hparam_overrides(explicit_hparams: Dict[str, Any], applied_hparams: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    training_keys = {
        "num_epochs",
        "epochs",
        "epoch",
        "patience",
        "batch_size",
        "lr",
        "learning_rate",
        "optimizer",
        "scheduler",
        "weight_decay",
    }
    for key, requested in dict(explicit_hparams or {}).items():
        if str(key).strip().lower() not in training_keys:
            continue
        applied = applied_hparams.get(key)
        if applied != requested:
            warnings.append(f"{key}={requested!r} was overridden by the central training policy to {applied!r}")
    return warnings
