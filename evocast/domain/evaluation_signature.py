"""Canonical evaluation signature helpers.

These helpers capture the parts of a task/evaluation setup that make metric
values comparable. We use the same signature everywhere instead of letting
different runtime artifacts silently compare metrics from incompatible tasks.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from evocast.domain.metric_semantics import DEFAULT_OBJECTIVE_METRIC, validate_public_metric_name


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    return value


def _normalize_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return os.path.normpath(text).replace("\\", "/").lower()


def _normalize_target_columns(value: Any) -> list[str]:
    cols = []
    for item in list(value or []):
        text = str(item or "").strip()
        if text:
            cols.append(text)
    return cols


def build_evaluation_signature(
    *,
    task_config: Optional[Dict[str, Any]] = None,
    compiled_config: Optional[Dict[str, Any]] = None,
    objective_metric: str = "",
) -> Dict[str, Any]:
    task_config = dict(task_config or {})
    compiled_config = dict(compiled_config or {})

    data_config = dict(compiled_config.get("data_config", {}) or {})
    eval_config = dict(compiled_config.get("evaluation_config", {}) or {})
    strategy_args = dict(eval_config.get("strategy_args", {}) or {})
    semantics = dict(data_config.get("task_semantics", {}) or {})

    resolved_objective = validate_public_metric_name(
        str(
            objective_metric
            or semantics.get("objective_metric")
            or task_config.get("objective_metric")
            or compiled_config.get("objective_metric")
            or DEFAULT_OBJECTIVE_METRIC
        ).strip()
    )

    dataset_path = (
        task_config.get("dataset_path")
        or data_config.get("dataset_path")
        or semantics.get("dataset_path")
        or ""
    )
    if dataset_path and not os.path.isabs(str(dataset_path)):
        dataset_path = _normalize_path(dataset_path)
    else:
        dataset_path = _normalize_path(dataset_path)

    feature_dict = dict(data_config.get("feature_dict", {}) or {})
    model_hp = dict((compiled_config.get("model_config", {}) or {}).get("recommend_model_hyper_params", {}) or {})
    canonical_frequency = (
        semantics.get("frequency")
        or feature_dict.get("canonical_freq")
        or feature_dict.get("freq")
        or task_config.get("feature_dict", {}).get("canonical_freq")
        or task_config.get("feature_dict", {}).get("freq")
        or ""
    )

    return {
        "strategy_name": str(
            strategy_args.get("strategy_name")
            or semantics.get("strategy_name")
            or ""
        ).strip(),
        "objective_metric": resolved_objective,
        "dataset_path": dataset_path,
        "time_col": str(
            task_config.get("time_col")
            or data_config.get("time_col")
            or semantics.get("time_col")
            or ""
        ).strip(),
        "task_mode": str(semantics.get("task_mode") or "").strip(),
        "target_columns": _normalize_target_columns(
            semantics.get("target_columns")
            or data_config.get("target_columns")
            or task_config.get("task_semantics", {}).get("target_columns")
            or []
        ),
        "target_channel": list(strategy_args.get("target_channel") or []),
        "horizon": int(
            strategy_args.get("horizon")
            or task_config.get("horizon")
            or 0
        ),
        "stride": _optional_int(strategy_args.get("stride")),
        "num_rollings": _optional_int(strategy_args.get("num_rollings")),
        "tv_ratio": strategy_args.get("tv_ratio"),
        "train_ratio_in_tv": _normalize_json_value(strategy_args.get("train_ratio_in_tv")),
        "deterministic": str(strategy_args.get("deterministic") or "").strip(),
        "seq_len": int(
            task_config.get("seq_len")
            or model_hp.get("input_chunk_length")
            or semantics.get("input_chunk_length")
            or 0
        ),
        "frequency": str(
            canonical_frequency
            or ""
        ).strip(),
    }


def signatures_match(saved_signature: Dict[str, Any], current_signature: Dict[str, Any]) -> bool:
    return bool(saved_signature) and saved_signature == current_signature
