"""Metric policy for evocast.

EvoCast now aligns its public metric names with ts_benchmark directly.
That means normalized metrics are first-class metric names, not hidden
behind canonical aliases.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from evocast.domain.frequency_semantics import resolve_mase_seasonality_generic

PUBLIC_METRICS = (
    "mse",
    "mae",
    "rmse",
    "mape",
    "smape",
    "mase",
    "wape",
    "msmape",
    "mse_norm",
    "mae_norm",
    "rmse_norm",
    "mape_norm",
    "smape_norm",
    "mase_norm",
    "wape_norm",
    "msmape_norm",
)

NORMALIZED_METRICS = tuple(
    metric_name for metric_name in PUBLIC_METRICS if metric_name.endswith("_norm")
)

DEFAULT_OBJECTIVE_METRIC = "mse_norm"


def get_default_objective_metric(task_mode: Optional[str]) -> str:
    mode = str(task_mode or "").strip().upper()
    if mode == "MM":
        return "mse_norm"
    return DEFAULT_OBJECTIVE_METRIC

_PUBLIC_METRIC_SET = set(PUBLIC_METRICS)

_MASE_METRICS = {"mase", "mase_norm"}


def is_supported_objective_metric(metric_name: str) -> bool:
    return str(metric_name or "").strip() in _PUBLIC_METRIC_SET


def get_objective_metric_choices(task_mode: Optional[str]) -> tuple[str, ...]:
    mode = str(task_mode or "").strip().upper()
    if mode == "MM":
        return NORMALIZED_METRICS
    return PUBLIC_METRICS


def validate_public_metric_name(metric_name: str) -> str:
    metric = str(metric_name or "").strip()
    if metric not in _PUBLIC_METRIC_SET:
        raise ValueError(
            "Unsupported objective metric. "
            f"EvoCast supports metrics aligned with ts_benchmark: {list(PUBLIC_METRICS)}. "
            f"Received: {metric_name!r}"
        )
    return metric


def validate_objective_metric_for_task_mode(
    metric_name: str,
    task_mode: Optional[str],
) -> str:
    metric = validate_public_metric_name(metric_name)
    allowed_metrics = get_objective_metric_choices(task_mode)
    if metric not in allowed_metrics:
        mode = str(task_mode or "").strip().upper() or "UNKNOWN"
        raise ValueError(
            f"Unsupported objective metric {metric!r} for task_mode={mode}. "
            f"Allowed metrics: {list(allowed_metrics)}"
        )
    return metric


def resolve_mase_seasonality(frequency: Optional[str]) -> int:
    """Resolve the seasonality parameter required by TFB's MASE metrics."""
    return resolve_mase_seasonality_generic(frequency)


def build_public_metric_configs(*, frequency: Optional[str] = None) -> list[Any]:
    """Build evaluation metric configs, expanding MASE-like metrics with seasonality."""
    mase_seasonality = resolve_mase_seasonality(frequency)
    metric_configs: list[Any] = []
    for metric_name in PUBLIC_METRICS:
        if metric_name in _MASE_METRICS:
            metric_configs.append({
                "name": metric_name,
                "seasonality": mase_seasonality,
            })
        else:
            metric_configs.append(metric_name)
    return metric_configs


def canonicalize_metric_values(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    metric_values = dict(metrics or {})
    return {
        name: metric_values[name]
        for name in PUBLIC_METRICS
        if name in metric_values
    }


def metric_semantics(metric_name: str) -> Dict[str, Any]:
    metric = validate_public_metric_name(metric_name)
    return {
        "metric_name": metric,
        "scale_kind": "normalized_scale" if metric.endswith("_norm") else "raw_scale",
        "display_name": metric,
        "description": (
            "EvoCast forecasting metric aligned directly with ts_benchmark naming and scale."
        ),
    }


def annotate_metric_payload(
    *,
    objective_metric: Optional[str],
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metric_values = canonicalize_metric_values(metrics)
    objective = validate_public_metric_name(str(objective_metric or DEFAULT_OBJECTIVE_METRIC))
    return {
        "objective_metric": metric_semantics(objective),
        "metrics": {
            name: metric_semantics(name)
            for name in metric_values.keys()
        },
    }
