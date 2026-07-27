"""Deterministic gate for evocast.

Compares a run node against the current best reference.
Makes structured decisions: accept, reject, needs_seed_eval, needs_review, needs_repair.

The direction of improvement (lower_is_better or higher_is_better) is configurable
per metric, not hard-coded.
"""

from enum import Enum
from typing import Dict, Optional


class GateDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    MARGINAL_NO_SEED_EVAL = "marginal_no_seed_eval"
    NEEDS_SEED_EVAL = "needs_seed_eval"
    NEEDS_REVIEW = "needs_review"
    NEEDS_REPAIR = "needs_repair"
    STOP = "stop"


# Default lower-is-better metrics for TFB fixed forecasting.
DEFAULT_METRIC_DIRECTIONS = {
    "mae": "lower",
    "mse": "lower",
    "rmse": "lower",
    "mape": "lower",
    "smape": "lower",
    "mase": "lower",
    "wape": "lower",
    "msmape": "lower",
}


def get_metric_direction(
    metric_name: str,
    overrides: Optional[Dict[str, str]] = None,
) -> str:
    """Get the direction for a metric. Returns 'lower' or 'higher'."""
    if overrides and metric_name in overrides:
        return overrides[metric_name]
    return DEFAULT_METRIC_DIRECTIONS.get(metric_name, "lower")


def is_better(
    current: float,
    reference: float,
    direction: str = "lower",
    min_relative_improvement: float = 0.0,
) -> bool:
    """Check if current value is better than reference.

    Args:
        current: The new value.
        reference: The baseline/best value.
        direction: "lower" or "higher".
        min_relative_improvement: Minimum relative improvement required.
    """
    if direction == "lower":
        if min_relative_improvement > 0 and reference != 0:
            threshold = reference * (1 - min_relative_improvement)
            return current < threshold
        return current < reference
    else:
        if min_relative_improvement > 0 and reference != 0:
            threshold = reference * (1 + min_relative_improvement)
            return current > threshold
        return current > reference


def relative_improvement(current: float, reference: float, direction: str = "lower") -> Optional[float]:
    if reference == 0:
        return None
    if direction == "lower":
        return (reference - current) / abs(reference)
    return (current - reference) / abs(reference)


def gate_decision(
    metrics: Dict[str, float],
    reference_metrics: Dict[str, float],
    objective_metric: str,
    *,
    min_relative_improvement: float,
    metric_direction_overrides: Optional[Dict[str, str]] = None,
    min_seed_eval_relative_improvement: float,
) -> Dict:
    """Make a gate decision by comparing metrics.

    Args:
        metrics: The candidate model's metrics.
        reference_metrics: The current_best metrics.
        objective_metric: The primary metric to optimize.
        min_relative_improvement: Minimum relative improvement over current_best.
        metric_direction_overrides: Override default metric directions.

    Returns:
        Dict with decision, reason, and comparison details.
    """
    direction = get_metric_direction(objective_metric, metric_direction_overrides)
    reference = dict(reference_metrics or {})

    result = {
        "decision": GateDecision.REJECT.value,
        "reason": "",
        "objective_metric": objective_metric,
        "objective_direction": direction,
        "current_value": metrics.get(objective_metric),
        "reference_kind": "current_best",
        "reference_value": reference.get(objective_metric),
        "current_best_value": reference.get(objective_metric),
        "reference_relative_improvement": None,
        "current_best_relative_improvement": None,
        "beats_reference": False,
        "beats_reference_threshold": False,
        "min_relative_improvement": min_relative_improvement,
        "min_seed_eval_relative_improvement": min_seed_eval_relative_improvement,
    }

    current_obj = metrics.get(objective_metric)
    reference_obj = reference.get(objective_metric)

    # If objective metric is missing, needs repair (metric_missing)
    if current_obj is None:
        result["decision"] = GateDecision.NEEDS_REPAIR.value
        result["reason"] = f"Objective metric '{objective_metric}' missing from candidate metrics"
        return result

    if reference_obj is None:
        result["decision"] = GateDecision.NEEDS_REVIEW.value
        result["reason"] = "No current_best value to compare against"
        return result

    result["reference_relative_improvement"] = relative_improvement(float(current_obj), float(reference_obj), direction)
    result["current_best_relative_improvement"] = result["reference_relative_improvement"]
    result["beats_reference"] = is_better(current_obj, reference_obj, direction, 0.0)
    result["beats_reference_threshold"] = is_better(current_obj, reference_obj, direction, min_relative_improvement)

    if result["beats_reference_threshold"]:
        rel_pct = (
            f"{result['reference_relative_improvement'] * 100:.2f}%"
            if result["reference_relative_improvement"] is not None
            else "n/a"
        )
        result["decision"] = GateDecision.ACCEPT.value
        result["reason"] = (
            f"{objective_metric}: {current_obj} vs current_best {reference_obj} "
            f"(relative improvement {rel_pct})"
        )
    elif (
        result["beats_reference"]
        and not result["beats_reference_threshold"]
        and result["reference_relative_improvement"] is not None
        and 0 < result["reference_relative_improvement"] < float(min_seed_eval_relative_improvement or 0.0)
    ):
        result["decision"] = GateDecision.MARGINAL_NO_SEED_EVAL.value
        result["reason"] = (
            f"{objective_metric}: numeric improvement over current_best is only "
            f"{result['reference_relative_improvement'] * 100:.4f}%, below the "
            f"minimum meaningful seed-eval threshold "
            f"{float(min_seed_eval_relative_improvement or 0.0) * 100:.2f}%"
        )
    elif result["beats_reference"] and not result["beats_reference_threshold"]:
        result["decision"] = GateDecision.NEEDS_SEED_EVAL.value
        result["reason"] = (
            f"{objective_metric}: numeric improvement over current_best is below "
            f"min_relative_improvement={min_relative_improvement}"
        )
    elif result["reference_relative_improvement"] is not None and result["reference_relative_improvement"] <= 0:
        result["decision"] = GateDecision.REJECT.value
        result["reason"] = (
            f"{objective_metric}: non-positive improvement "
            f"{result['reference_relative_improvement'] * 100:.4f}%; no seed evaluation is required"
        )
    else:
        regression_pct = (
            abs((float(current_obj) - float(reference_obj)) / float(reference_obj) * 100)
            if float(reference_obj) != 0
            else float("inf")
        )
        result["decision"] = GateDecision.REJECT.value
        result["reason"] = (
            f"{objective_metric}: {current_obj} vs current_best {reference_obj} "
            f"(regression of {regression_pct:.2f}%)"
        )

    return result


def seed_eval_significance_decision(
    variant_mean: Optional[float],
    reference_mean: Optional[float],
    reference_std: Optional[float],
    reference_seed_count: Optional[int],
    objective_metric: str,
    *,
    direction: str = "lower",
    min_effect_size: float,
    variant_seed_count: Optional[int] = None,
    min_relative_improvement: float,
    min_absolute_improvement: float,
    reference_std_multiplier: float,
    relative_denominator_floor: float = 1e-6,
    required_seed_count: int,
) -> Dict:
    """Statistical admission rule for multi-seed variant evaluation."""
    result = {
        "objective_metric": objective_metric,
        "variant_mean": variant_mean,
        "reference_kind": "current_best",
        "reference_mean": reference_mean,
        "current_best_mean": reference_mean,
        "reference_std": reference_std,
        "current_best_std": reference_std,
        "reference_seed_count": reference_seed_count or 0,
        "current_best_seed_count": reference_seed_count or 0,
        "variant_seed_count": variant_seed_count or 0,
        "required_seed_count": required_seed_count,
        "effect_size": None,
        "relative_improvement": None,
        "relative_denominator": None,
        "relative_accept_threshold": float(min_relative_improvement or 0.0),
        "absolute_accept_threshold": float(min_absolute_improvement or 0.0),
        "threshold": None,
        "significant": False,
        "accept_rule": None,
        "decision": GateDecision.NEEDS_REVIEW.value,
        "reason": "",
    }
    if variant_mean is None or reference_mean is None:
        result["decision"] = GateDecision.NEEDS_REPAIR.value
        result["reason"] = "missing variant or current_best reference mean for seed-eval significance"
        return result

    if direction == "lower":
        improvement = reference_mean - variant_mean
    else:
        improvement = variant_mean - reference_mean

    reference_std = float(reference_std or 0.0)
    result["effect_size"] = improvement
    result["reference_std_multiplier"] = float(reference_std_multiplier)
    result["threshold"] = max(float(min_effect_size or 0.0), float(reference_std_multiplier) * reference_std)
    denominator = max(abs(float(reference_mean)), float(relative_denominator_floor or 1e-6))
    relative = improvement / denominator
    result["relative_denominator"] = denominator
    result["relative_improvement"] = relative

    reference_count = int(reference_seed_count or 0)
    variant_count = int(variant_seed_count or 0)
    required_count = max(1, int(required_seed_count or 3))
    if reference_count < required_count or variant_count < required_count:
        result["decision"] = GateDecision.REJECT.value
        result["reason"] = (
            f"seed-eval rejection: high-confidence acceptance requires current_best_seed_count>={required_count} and "
            f"variant_seed_count>={required_count}; got current_best={reference_count}, "
            f"variant={variant_count}"
        )
        return result

    if improvement > float(min_absolute_improvement or 0.0) and relative > float(min_relative_improvement or 0.0):
        result["decision"] = GateDecision.ACCEPT.value
        result["significant"] = True
        result["accept_rule"] = "three_seed_relative_improvement"
        result["reason"] = (
            f"seed eval relative improvement {relative * 100:.2f}% exceeds "
            f"{float(min_relative_improvement or 0.0) * 100:.2f}% with absolute improvement "
            f"{improvement} > {float(min_absolute_improvement or 0.0)}"
        )
    elif improvement > 0:
        result["decision"] = GateDecision.REJECT.value
        if improvement <= float(min_absolute_improvement or 0.0):
            result["reason"] = (
                f"seed-eval rejection: numeric improvement {improvement} is below absolute floor "
                f"{float(min_absolute_improvement or 0.0)}"
            )
        else:
            result["reason"] = (
                f"seed-eval rejection: numeric improvement {improvement} has relative improvement {relative * 100:.2f}%, "
                f"below unified accept threshold {float(min_relative_improvement or 0.0) * 100:.2f}%"
            )
    else:
        result["decision"] = GateDecision.REJECT.value
        result["reason"] = f"no positive seed-eval improvement; effect_size={improvement}"
    return result
