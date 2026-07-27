from __future__ import annotations

from typing import Any, Dict, List, Tuple

from evocast.research.baseline_candidate_brief import build_baseline_candidate_brief


def build_baseline_candidate_pool(
    *,
    registry: List[Dict[str, Any]],
    config_data: Dict[str, Any],
    max_pool_size: int,
    protected_model_keys: List[str] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build the complete eligible pool.

    ``max_pool_size`` and ``protected_model_keys`` remain in the signature for
    compatibility with callers and persisted policies. Candidate budgeting is
    now owned by the family-aware curator; truncating here would make the
    result depend on legacy metadata ordering instead of family/alphabetical
    order.
    """
    del max_pool_size, protected_model_keys
    pool: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for spec in list(registry or []):
        item = dict(spec or {})
        brief = build_baseline_candidate_brief(spec=item, config_data=config_data)
        model_key = str(item.get("model_key") or "")
        if not model_key:
            rejected.append({"model_key": "", "reason": "missing_model_key", "brief": brief})
            continue
        if not brief["verified_import"]:
            rejected.append({"model_key": model_key, "reason": "unverified_import", "brief": brief})
            continue
        if not brief["task_supported"]:
            rejected.append({"model_key": model_key, "reason": "task_mode_not_supported", "brief": brief})
            continue
        if not brief["researchable_source"]:
            rejected.append({"model_key": model_key, "reason": "not_researchable_external_model", "brief": brief})
            continue
        pool.append({"spec": item, "brief": brief})

    return pool, rejected
