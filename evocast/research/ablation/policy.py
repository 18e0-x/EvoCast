from __future__ import annotations

from typing import Any, Dict


def _normalize_metric_direction(metric_direction: str) -> str:
    normalized = str(metric_direction or "").strip().lower()
    if normalized in {"lower", "lower_is_better", "min", "minimize"}:
        return "lower"
    if normalized in {"higher", "higher_is_better", "max", "maximize"}:
        return "higher"
    raise ValueError(f"Unsupported metric_direction: {metric_direction}")


def relative_improvement(
    *,
    reference_value: Any,
    candidate_value: Any,
    metric_direction: str,
) -> Dict[str, Any]:
    try:
        reference = float(reference_value)
        candidate = float(candidate_value)
    except Exception:
        return {
            "status": "unknown",
            "reason": "non_numeric_metric",
            "metric_direction": str(metric_direction or ""),
            "reference_value": reference_value,
            "candidate_value": candidate_value,
            "relative_improvement": None,
            "absolute_delta": None,
        }

    if reference == 0.0:
        return {
            "status": "unknown",
            "reason": "zero_reference_value",
            "metric_direction": _normalize_metric_direction(metric_direction),
            "reference_value": reference,
            "candidate_value": candidate,
            "relative_improvement": None,
            "absolute_delta": candidate - reference,
        }

    direction = _normalize_metric_direction(metric_direction)
    absolute_delta = candidate - reference
    if direction == "lower":
        improvement = (reference - candidate) / abs(reference)
    else:
        improvement = (candidate - reference) / abs(reference)
    return {
        "status": "ok",
        "reason": "",
        "metric_direction": direction,
        "reference_value": reference,
        "candidate_value": candidate,
        "relative_improvement": improvement,
        "absolute_delta": absolute_delta,
    }


def classify_ablation_effect(
    *,
    objective_metric: str,
    metric_direction: str,
    reference_value: Any,
    candidate_value: Any,
    threshold: float,
) -> Dict[str, Any]:
    outcome = relative_improvement(
        reference_value=reference_value,
        candidate_value=candidate_value,
        metric_direction=metric_direction,
    )
    payload = {
        "threshold": float(threshold),
        "objective_metric": str(objective_metric or ""),
        "metric_direction": outcome.get("metric_direction") or str(metric_direction or ""),
        "reference_value": outcome.get("reference_value"),
        "candidate_value": outcome.get("candidate_value"),
        "relative_improvement": outcome.get("relative_improvement"),
        "absolute_delta": outcome.get("absolute_delta"),
        "module_effect": "unknown",
        "recommended_action": "retry_or_skip",
        "policy": "insufficient_metric_evidence",
        "status": outcome.get("status"),
        "reason": outcome.get("reason"),
    }
    improvement = payload["relative_improvement"]
    if improvement is None:
        return payload
    if improvement <= -float(threshold):
        payload.update(
            {
                "module_effect": "essential",
                "recommended_action": "protect_current_mechanism",
                "policy": "avoid_destructive_edits_without_replacement",
            }
        )
    elif improvement >= float(threshold):
        payload.update(
            {
                "module_effect": "harmful_or_redundant_candidate",
                "recommended_action": "run_seed_eval",
                "policy": "seed_eval_required_before_removal",
            }
        )
    else:
        payload.update(
            {
                "module_effect": "weak_or_uncertain",
                "recommended_action": "record_as_context",
                "policy": "available_design_space",
            }
        )
    return payload


def classify_seed_eval_ablation_effect(
    *,
    objective_metric: str,
    metric_direction: str,
    reference_value: Any,
    candidate_value: Any,
    threshold: float,
    seed_eval_ready: bool = True,
    failure_reason: str = "",
) -> Dict[str, Any]:
    payload = classify_ablation_effect(
        objective_metric=objective_metric,
        metric_direction=metric_direction,
        reference_value=reference_value,
        candidate_value=candidate_value,
        threshold=threshold,
    )
    if not seed_eval_ready or payload.get("relative_improvement") is None:
        payload.update(
            {
                "module_effect": "seed_eval_failed_or_inconclusive",
                "recommended_action": "keep_current_baseline",
                "policy": "keep_current_baseline",
                "status": "unknown",
                "reason": failure_reason or payload.get("reason") or "seed_eval_inconclusive",
            }
        )
        return payload

    improvement = float(payload["relative_improvement"])
    if improvement >= float(threshold):
        payload.update(
            {
                "module_effect": "confirmed_harmful_or_redundant",
                "recommended_action": "promote_ablation_variant_to_current_best",
                "policy": "promote_ablation_variant_to_current_best",
            }
        )
    elif improvement <= -float(threshold):
        payload.update(
            {
                "module_effect": "confirmed_essential",
                "recommended_action": "keep_current_baseline",
                "policy": "avoid_destructive_edits_without_replacement",
            }
        )
    else:
        payload.update(
            {
                "module_effect": "seed_eval_weak_or_uncertain",
                "recommended_action": "keep_current_baseline",
                "policy": "available_design_space",
            }
        )
    return payload
