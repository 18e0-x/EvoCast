from __future__ import annotations

from typing import Any, Dict, List


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def task_requires_univariate(config_data: Dict[str, Any]) -> bool:
    data_config = dict(config_data.get("data_config") or {})
    feature_dict = dict(data_config.get("feature_dict") or {})
    if "if_univariate" in feature_dict:
        return _truthy(feature_dict.get("if_univariate"))
    semantics = dict(data_config.get("task_semantics") or {})
    topology = str(semantics.get("input_variable_topology") or "").strip().lower()
    mode = str(semantics.get("task_mode") or "").strip().lower()
    return topology == "univariate" or mode in {"ss", "univariate"}


def model_supports_task(spec: Dict[str, Any], config_data: Dict[str, Any]) -> bool:
    if task_requires_univariate(config_data):
        return bool(spec.get("supports_univariate", True))
    return bool(spec.get("supports_multivariate", True))


def build_baseline_candidate_brief(*, spec: Dict[str, Any], config_data: Dict[str, Any]) -> Dict[str, Any]:
    tags = [str(item) for item in list(spec.get("tags") or []) if str(item).strip()]
    selection_tags = [str(item) for item in list(spec.get("selection_tags") or tags) if str(item).strip()]
    supports_univariate = bool(spec.get("supports_univariate", True))
    supports_multivariate = bool(spec.get("supports_multivariate", True))
    verified_import = bool(spec.get("verified_import", False))
    task_supported = model_supports_task(spec, config_data)
    local_code = bool(spec.get("local_code", False))
    family = str(spec.get("family") or "unknown")
    model_key = str(spec.get("model_key") or "")
    facts: List[str] = [
        f"family={family}",
        f"verified_import={verified_import}",
        f"task_supported={task_supported}",
        f"local_code={local_code}",
    ]
    if selection_tags:
        facts.append("selection_tags=" + ",".join(selection_tags[:8]))
    if spec.get("facts_confidence"):
        facts.append(f"facts_confidence={spec.get('facts_confidence')}")
    if spec.get("facts_backbone"):
        facts.append(f"facts_backbone={spec.get('facts_backbone')}")
    if spec.get("facts_tokenization"):
        facts.append(f"facts_tokenization={spec.get('facts_tokenization')}")

    return {
        "model_key": model_key,
        "family": family,
        "tags": tags,
        "selection_tags": selection_tags,
        "verified_import": verified_import,
        "supports_univariate": supports_univariate,
        "supports_multivariate": supports_multivariate,
        "task_supported": task_supported,
        "researchable_source": local_code,
        "local_code": local_code,
        "import_path": str(spec.get("import_path") or ""),
        "adapter": spec.get("adapter"),
        "selection_facts": facts,
    }


def build_baseline_candidate_briefs(
    *,
    registry: List[Dict[str, Any]],
    config_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [build_baseline_candidate_brief(spec=dict(spec), config_data=config_data) for spec in list(registry or [])]
