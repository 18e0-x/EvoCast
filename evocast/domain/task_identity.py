"""Canonical task semantics and generated task identity helpers."""

from __future__ import annotations

import os
import re
import hashlib
from datetime import datetime
from typing import Any, Dict

from evocast.domain.knowledge_paths import task_knowledge_dir
from evocast.domain.metric_semantics import DEFAULT_OBJECTIVE_METRIC


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "")).strip("_").lower()
    return cleaned or "task"


def build_task_semantics_from_intent(intent: Dict[str, Any]) -> Dict[str, Any]:
    task_mode = str(intent.get("task_mode", "MM") or "MM").upper()
    target_columns = [str(col) for col in list(intent.get("target_columns", []) or []) if str(col).strip()]
    horizons = [int(h) for h in list(intent.get("horizons", []) or [])]
    input_chunk_length = int(intent.get("input_chunk_length") or (horizons[0] if horizons else 0))

    if task_mode == "SS":
        input_topology = "univariate"
        target_selection = "single_target"
    elif task_mode == "MS":
        input_topology = "multivariate"
        target_selection = "single_target"
    elif task_mode == "MM":
        input_topology = "multivariate"
        target_selection = "all_targets"
    else:
        raise ValueError(f"Unsupported task_mode for semantic compilation: {task_mode}")

    return {
        "task_mode": task_mode,
        "input_variable_topology": input_topology,
        "prediction_target_selection": target_selection,
        "target_columns": target_columns,
        "time_col": str(intent.get("time_col", "") or ""),
        "dataset_path": str(intent.get("dataset_path", "") or ""),
        "frequency": str(intent.get("frequency", "") or ""),
        "strategy_name": str(intent.get("strategy_name", "rolling_forecast") or "rolling_forecast"),
        "objective_metric": str(intent.get("objective_metric", DEFAULT_OBJECTIVE_METRIC) or DEFAULT_OBJECTIVE_METRIC),
        "horizons": horizons,
        "input_chunk_length": input_chunk_length,
    }


def build_task_semantics_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    data_config = config.get("data_config", {}) or {}
    semantics = dict(data_config.get("task_semantics", {}) or {})
    if not semantics:
        raise ValueError("Compiled config is missing data_config.task_semantics. Legacy semantic inference is disabled.")
    semantics["target_columns"] = [str(col) for col in list(semantics.get("target_columns", []) or [])]
    horizons = list(semantics.get("horizons", []) or [])
    semantics["horizons"] = [int(h) for h in horizons]
    return semantics


def generate_task_id(semantics: Dict[str, Any], created_at: datetime | None = None) -> str:
    created_at = created_at or datetime.now()
    return created_at.strftime("%Y%m%d_%H%M%S")


def validate_task_semantics(semantics: Dict[str, Any]) -> None:
    task_mode = str(semantics.get("task_mode", "") or "").upper()
    topology = str(semantics.get("input_variable_topology", "") or "")
    target_selection = str(semantics.get("prediction_target_selection", "") or "")
    target_columns = list(semantics.get("target_columns", []) or [])
    horizons = list(semantics.get("horizons", []) or [])
    strategy_name = str(semantics.get("strategy_name", "rolling_forecast") or "rolling_forecast")

    errors: list[str] = []
    if task_mode not in {"MM", "MS", "SS"}:
        errors.append(f"Unsupported task_mode: {task_mode}")
    if topology not in {"univariate", "multivariate"}:
        errors.append(f"Unsupported input_variable_topology: {topology}")
    if target_selection not in {"single_target", "all_targets"}:
        errors.append(f"Unsupported prediction_target_selection: {target_selection}")
    if task_mode == "MM":
        if topology != "multivariate" or target_selection != "all_targets":
            errors.append("MM requires multivariate inputs and all-target prediction.")
        if target_columns:
            errors.append("MM must not declare explicit target_columns.")
    if task_mode == "MS":
        if topology != "multivariate" or target_selection != "single_target":
            errors.append("MS requires multivariate inputs and single-target prediction.")
        if len(target_columns) != 1:
            errors.append("MS requires exactly one target column.")
    if task_mode == "SS":
        if topology != "univariate" or target_selection != "single_target":
            errors.append("SS requires univariate inputs and single-target prediction.")
        if len(target_columns) > 1:
            errors.append("SS supports at most one target column.")
    if not horizons or any(int(h) <= 0 for h in horizons):
        errors.append("At least one positive horizon is required.")
    if int(semantics.get("input_chunk_length") or 0) <= 0:
        errors.append("A positive input_chunk_length is required.")
    if strategy_name != "rolling_forecast":
        errors.append("Only rolling_forecast is supported.")

    if errors:
        raise ValueError("Invalid task semantics:\n- " + "\n- ".join(errors))


def validate_compiled_config_semantics(config: Dict[str, Any]) -> Dict[str, Any]:
    semantics = build_task_semantics_from_config(config)
    validate_task_semantics(semantics)

    data_config = config.get("data_config", {}) or {}
    feature_dict = data_config.get("feature_dict", {}) or {}
    strategy_args = (config.get("evaluation_config", {}) or {}).get("strategy_args", {}) or {}
    horizon = strategy_args.get("horizon")
    output_chunk_length = (
        (config.get("model_config", {}) or {})
        .get("recommend_model_hyper_params", {})
        .get("output_chunk_length")
    )
    task_mode = semantics["task_mode"]
    target_selection = semantics["prediction_target_selection"]
    topology = semantics["input_variable_topology"]
    target_channel = strategy_args.get("target_channel")

    errors: list[str] = []
    if topology == "univariate" and feature_dict.get("if_univariate") is not True:
        errors.append("Univariate semantics require feature_dict.if_univariate=true.")
    if topology == "multivariate" and feature_dict.get("if_univariate") is True:
        errors.append("Multivariate semantics require feature_dict.if_univariate=false.")
    if target_selection == "all_targets" and target_channel is not None:
        errors.append("All-target semantics require target_channel=null.")
    if target_selection == "single_target" and task_mode == "MS" and target_channel is None:
        errors.append("MS semantics require an explicit target_channel.")
    expected_horizon = int((semantics.get("horizons") or [0])[0])
    if horizon is not None and int(horizon) != expected_horizon:
        errors.append(f"evaluation horizon {horizon} does not match semantic horizon {expected_horizon}.")
    if output_chunk_length is not None and int(output_chunk_length) != expected_horizon:
        errors.append(
            f"output_chunk_length {output_chunk_length} does not match semantic horizon {expected_horizon}."
        )
    if errors:
        raise ValueError("Compiled config semantic validation failed:\n- " + "\n- ".join(errors))
    return semantics


def summarize_task_semantics(semantics: Dict[str, Any]) -> str:
    target_columns = list(semantics.get("target_columns", []) or [])
    target_repr = ",".join(target_columns) if target_columns else "all"
    horizon = (semantics.get("horizons") or ["?"])[0]
    return (
        f"mode={semantics.get('task_mode')} "
        f"topology={semantics.get('input_variable_topology')} "
        f"target={target_repr} horizon={horizon}"
    )


def resolve_compiled_config_path(task_id: str, base_dir: str) -> str:
    config_path = str(task_knowledge_dir(base_dir, task_id) / "compiled_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"compiled_config.json not found for task '{task_id}'. "
            "All execution paths now require a compiled task created by the wizard/configurator flow."
        )
    return config_path


def compact_result_save_path(task_id: str, *parts: Any, max_len: int = 120) -> str:
    normalized_parts = [slugify(task_id)] + [slugify(part) for part in parts if str(part or "").strip()]
    raw = "EvoCast_" + "_".join(part for part in normalized_parts if part)
    if len(raw) <= max_len:
        return raw
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    task_prefix = slugify(task_id)[:48]
    tail_prefix = "_".join(slugify(part)[:18] for part in parts[:2] if str(part or "").strip())
    compact = "EvoCast_" + "_".join(part for part in [task_prefix, tail_prefix, digest] if part)
    return compact[:max_len]
