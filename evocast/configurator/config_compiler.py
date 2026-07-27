"""Deterministic config compiler.

Converts a validated intent into TFB-compatible data_config,
model_config, evaluation_config, and report_config.

Also generates multi-horizon manifests when multiple horizons
are specified.
"""

import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

from .config_validator import ValidationReport
from evocast.domain.frequency_semantics import resolve_tfb_runtime_freq_token
from evocast.domain.metric_semantics import (
    DEFAULT_OBJECTIVE_METRIC,
    PUBLIC_METRICS,
    build_public_metric_configs,
    validate_public_metric_name,
)
from evocast.domain.task_identity import build_task_semantics_from_intent, validate_task_semantics
from evocast.policy.agent_control_policy import build_mode_policy


# ── Default templates ─────────────────────────────────────────────────────

_BASE_TEMPLATE: Dict = {
    "data_config": {
        "feature_dict": {
            "if_univariate": False,
            "if_trend": None,
            "has_timestamp": None,
            "if_season": None,
        },
        "data_set_name": "large_forecast",
        "dataset_path": "",
    },
    "model_config": {
        "models": [],
        "recommend_model_hyper_params": {
            "input_chunk_length": 96,
            "output_chunk_length": 96,
            "add_relative_index": True,
            "norm": True,
        },
    },
    "evaluation_config": {
        "metrics": "all",
        "strategy_args": {
            "strategy_name": "rolling_forecast",
            "horizon": 96,
            "tv_ratio": 0.8,
            "train_ratio_in_tv": {
                "ETTm1.csv": 0.75,
                "ETTm2.csv": 0.75,
                "PEMS04.csv": 0.75,
                "PEMS08.csv": 0.75,
                "PEMS03.csv": 0.75,
                "PEMS07.csv": 0.75,
                "AQShunyi.csv": 0.75,
                "AQWan.csv": 0.75,
                "ETTh1.csv": 0.75,
                "ETTh2.csv": 0.75,
                "Solar.csv": 0.75,
                "__default__": 0.875,
            },
            "stride": 1,
            "num_rollings": 48000,
            "seed": 2021,
            "deterministic": "efficient",
            "save_true_pred": False,
            "target_channel": None,
        },
    },
    "report_config": {
        "aggregate_type": "mean",
        "report_metrics": ["mse", "mae", "rmse", "mape", "smape", "wape", "msmape"],
        "fill_type": "mean_value",
        "null_value_threshold": "0.3",
    },
}

# ── Main entry points ─────────────────────────────────────────────────────


def compile_config(
    intent: Dict,
    report: Optional[ValidationReport] = None,
    seed: int = 2021,
    build_mode: bool = False,
) -> Dict:
    """Compile a validated intent into a single TFB config dict.

    Args:
        intent: Validated config intent dict.
        report: Optional ValidationReport for inferred hints.
        seed: Random seed.
        build_mode: If True, minimize evaluation (1 rolling, no val split).

    Returns:
        TFB config dict with data_config, model_config, evaluation_config,
        report_config.
    """
    task_mode = intent.get("task_mode", "MS")
    horizons = intent.get("horizons", [96])
    if not isinstance(horizons, list):
        horizons = [horizons]
    primary_horizon = horizons[0] if horizons else 96
    strategy_name = _normalize_strategy_name(intent.get("strategy_name", "rolling_forecast"))
    validate_public_metric_name(str(intent.get("objective_metric", DEFAULT_OBJECTIVE_METRIC)))

    config = deepcopy(_BASE_TEMPLATE)
    semantics = build_task_semantics_from_intent(intent)
    validate_task_semantics(semantics)

    # ── data_config ───────────────────────────────────────────────────
    _compile_data_config(config, intent, report, semantics)

    # ── model_config ──────────────────────────────────────────────────
    _compile_model_config(config, intent, report, semantics, primary_horizon)

    # ── evaluation_config ─────────────────────────────────────────────
    _compile_evaluation_config(config, intent, report, semantics, primary_horizon, strategy_name, seed, build_mode=build_mode)

    # ── report_config ─────────────────────────────────────────────────
    _compile_report_config(config, intent, report)

    return config


def _normalize_strategy_name(strategy_name: Any) -> str:
    """Normalize UI labels into the only supported strategy registry key."""
    value = str(strategy_name or "rolling_forecast").strip()
    first_token = value.split(" ", 1)[0].strip()
    if first_token and first_token != "rolling_forecast":
        raise ValueError(
            f"Unsupported strategy_name '{strategy_name}'. "
            "EvoCast now only supports rolling_forecast."
        )
    return "rolling_forecast"


def compile_multi_horizon_manifest(
    intent: Dict,
    report: Optional[ValidationReport] = None,
    seed: int = 2021,
    output_dir: Optional[str] = None,
) -> Dict:
    """Compile a multi-horizon manifest.

    When multiple horizons are specified (e.g., [96, 192, 336, 720]),
    generate one independent config per horizon + a manifest file.

    Returns:
        Dict with:
          - manifest_path
          - configs: {horizon: config_dict}
          - primary_horizon
    """
    horizons = intent.get("horizons", [96])
    if not isinstance(horizons, list):
        horizons = [horizons]

    configs = {}
    for h in horizons:
        single_intent = dict(intent)
        single_intent["horizons"] = [h]
        configs[h] = compile_config(single_intent, report, seed, build_mode=bool(intent.get("build_mode")))

    manifest = {
        "task_mode": intent.get("task_mode", "MS"),
        "dataset_path": intent.get("dataset_path", ""),
        "time_col": intent.get("time_col", "date"),
        "target_columns": intent.get("target_columns", []),
        "horizons": horizons,
        "primary_horizon": horizons[0] if horizons else 96,
        "objective_metric": intent.get("objective_metric", DEFAULT_OBJECTIVE_METRIC),
        "strategy_name": intent.get("strategy_name", "rolling_forecast"),
        "seed": seed,
        "_note": "Each horizon config is independent. Run them sequentially or via orchestrator.",
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, "multi_horizon_manifest.json")
        # Write manifest
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        # Write per-horizon configs
        for h, cfg in configs.items():
            cfg_path = os.path.join(output_dir, f"config_horizon_{h}.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        manifest["config_files"] = {
            h: os.path.join(output_dir, f"config_horizon_{h}.json")
            for h in horizons
        }

    return {
        "manifest": manifest,
        "configs": configs,
        "primary_horizon": horizons[0] if horizons else 96,
    }


# ── Sub-compilers ─────────────────────────────────────────────────────────


def _compile_data_config(
    config: Dict,
    intent: Dict,
    report: Optional[ValidationReport],
    semantics: Dict[str, Any],
) -> None:
    dc = config["data_config"]
    fd = dc["feature_dict"]

    # If a specific dataset file was validated, use data_name_list
    # to bypass metadata filtering and run only that dataset
    dataset_path = intent.get("dataset_path", "")
    if dataset_path:
        dc["dataset_path"] = dataset_path
    if dataset_path and report and report.valid:
        basename = os.path.basename(dataset_path)
        dc["data_name_list"] = [basename]
    dc["time_col"] = semantics.get("time_col", "")
    if semantics.get("target_columns"):
        dc["target_columns"] = list(semantics["target_columns"])
    dc["task_semantics"] = deepcopy(semantics)

    fd["if_univariate"] = semantics["input_variable_topology"] == "univariate"

    # Preserve two frequency views:
    # - canonical_frequency: human/metric semantics (e.g. 15min)
    # - freq: TFB runtime token for model internals (e.g. t)
    freq = intent.get("frequency")
    if report and report.detected_frequency:
        freq = report.detected_frequency
    if freq:
        dc["task_semantics"]["frequency"] = freq
        fd["canonical_freq"] = freq
        fd["freq"] = resolve_tfb_runtime_freq_token(freq)


def _compile_model_config(
    config: Dict,
    intent: Dict,
    report: Optional[ValidationReport],
    semantics: Dict[str, Any],
    horizon: int,
) -> None:
    mc = config["model_config"]
    hp = mc["recommend_model_hyper_params"]

    hp["output_chunk_length"] = horizon
    hp["input_chunk_length"] = _resolve_input_chunk_length(intent, report, horizon)
    hp["norm"] = True
    hp["add_relative_index"] = True


def _compile_evaluation_config(
    config: Dict,
    intent: Dict,
    report: Optional[ValidationReport],
    semantics: Dict[str, Any],
    horizon: int,
    strategy_name: str,
    seed: int,
    build_mode: bool = False,
) -> None:
    ec = config["evaluation_config"]
    sa = ec["strategy_args"]

    sa["strategy_name"] = strategy_name
    sa["horizon"] = horizon
    sa["seed"] = seed

    # Task mode → target_channel
    task_mode = semantics.get("task_mode", "MS")
    target_columns = list(semantics.get("target_columns", []) or [])

    if task_mode == "MS":
        # Single target: set target_channel to column index.
        # For long-format CSV, use the index in TFB's pd.unique() order
        # (preserved in long_format_vars).  Pass as a list — TFB iterates it.
        tc = target_columns[0] if target_columns else ""
        if report and tc:
            if tc in report.numeric_target_indices:
                sa["target_channel"] = [report.numeric_target_indices[tc]]
            elif tc in report.target_indices:
                sa["target_channel"] = [report.target_indices[tc]]
            elif report.is_long_format and tc in report.long_format_vars:
                idx = report.long_format_vars.index(tc)
                sa["target_channel"] = [idx]
    elif task_mode == "MM":
        sa["target_channel"] = None  # predict all
    elif task_mode == "SS":
        sa["target_channel"] = None  # univariate

    # Metrics
    objective = validate_public_metric_name(str(intent.get("objective_metric", DEFAULT_OBJECTIVE_METRIC)))
    ec["metrics"] = "all"

    # TV ratio from intent or default
    tv_ratio = intent.get("tv_ratio")
    if tv_ratio is not None:
        sa["tv_ratio"] = float(tv_ratio)

    train_ratio = intent.get("train_ratio_in_tv")
    if train_ratio is not None:
        if isinstance(train_ratio, (int, float)):
            sa["train_ratio_in_tv"] = float(train_ratio)
        else:
            sa["train_ratio_in_tv"] = train_ratio

    # ── build_mode: one-window evaluation ─────────────────────────────
    if build_mode:
        # Keep the task's normal train/validation split so the model executes
        # all three phases.  The smoke training policy limits both train and
        # validation loaders to one batch; here we limit test to one rolling
        # window.  Setting train_ratio_in_tv=1 used to skip validation entirely.
        sa["num_rollings"] = int(build_mode_policy().get("num_rollings") or 1)
        sa["stride"] = horizon


def _compile_report_config(
    config: Dict,
    intent: Dict,
    report: Optional[ValidationReport] = None,
) -> None:
    rc = config["report_config"]
    validate_public_metric_name(str(intent.get("objective_metric", DEFAULT_OBJECTIVE_METRIC)))
    frequency = (
        (report.detected_frequency if report and report.detected_frequency else None)
        or ((config.get("data_config", {}) or {}).get("task_semantics", {}) or {}).get("frequency")
        or ((config.get("data_config", {}) or {}).get("feature_dict", {}) or {}).get("canonical_freq")
        or intent.get("frequency")
    )
    rc["report_metrics"] = [
        "mse_norm",
        "mae_norm",
        "rmse_norm",
        "mape_norm",
        "smape_norm",
        "wape_norm",
        "msmape_norm",
    ]


def _resolve_input_chunk_length(
    intent: Dict,
    report: Optional[ValidationReport],
    horizon: int,
) -> int:
    """Determine input_chunk_length from intent, report hints, or defaults."""
    # Explicit in intent
    icl = intent.get("input_chunk_length")
    if icl is not None:
        return int(icl)

    # From report hints
    if report and report.suggested_input_chunk_length:
        return report.suggested_input_chunk_length

    # Default: equal to horizon (lookback = forecast length)
    return horizon


# ── Utility ───────────────────────────────────────────────────────────────
