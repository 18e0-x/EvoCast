from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from evocast.domain.knowledge_paths import runtime_root, task_knowledge_dir
from evocast.state.runtime.store import sync_best_baseline
from evocast.harness.api_client import ProviderClient
from evocast.harness.session import AgentSession


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def base_dir() -> str:
    return str(runtime_root(str(PROJECT_ROOT)))


def unique_task_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def create_dtaf_task(prefix: str, *, max_rounds: int = 6) -> AgentSession:
    task_id = unique_task_id(prefix)
    knowledge_dir = task_knowledge_dir(base_dir(), task_id)
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    semantics = {
        "task_mode": "SS",
        "input_variable_topology": "univariate",
        "prediction_target_selection": "single_target",
        "target_columns": ["Usage_kWh"],
        "time_col": "date",
        "dataset_path": "dataset/forecasting/steel.csv",
        "frequency": "15min",
        "strategy_name": "rolling_forecast",
        "objective_metric": "mse_norm",
        "horizons": [96],
        "input_chunk_length": 96,
    }
    compiled = {
        "data_config": {
            "feature_dict": {
                "if_univariate": True,
                "if_trend": None,
                "has_timestamp": None,
                "if_season": None,
                "canonical_freq": "15min",
                "freq": "t",
            },
            "data_set_name": "large_forecast",
            "dataset_path": "dataset/forecasting/steel.csv",
            "data_name_list": ["steel.csv"],
            "time_col": "date",
            "target_columns": ["Usage_kWh"],
            "task_semantics": semantics,
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
                "train_ratio_in_tv": {"steel.csv": 0.875, "__default__": 0.875},
                "stride": 1,
                "num_rollings": 16,
                "seed": 2021,
                "deterministic": "efficient",
                "save_true_pred": False,
                "target_channel": None,
            },
        },
        "report_config": {
            "aggregate_type": "mean",
            "report_metrics": [
                "mse_norm",
                "mae_norm",
                "rmse_norm",
                "mape_norm",
                "smape_norm",
                "wape_norm",
                "msmape_norm",
            ],
            "fill_type": "mean_value",
            "null_value_threshold": "0.3",
        },
    }
    task_config = {
        "task_id": task_id,
        "config_path": str(knowledge_dir / "compiled_config.json"),
        "objective_metric": "mse_norm",
        "metric_direction": "lower_is_better",
        "budget": "unified",
        "max_rounds": max_rounds,
        "force_full_rounds": True,
        "baseline_strategy": "manual",
        "baseline_models": ["DTAF"],
        "data_set_name": "large_forecast",
        "dataset_path": str(PROJECT_ROOT / "dataset" / "forecasting" / "steel.csv"),
        "horizon": 96,
        "seq_len": 96,
        "task_semantics": semantics,
        "feature_dict": compiled["data_config"]["feature_dict"],
    }
    write_json(knowledge_dir / "compiled_config.json", compiled)
    write_json(knowledge_dir / "task_config.json", task_config)
    sync_best_baseline(
        base_dir(),
        task_id,
        {
            "candidate_id": "baseline_dtaf",
            "candidate_kind": "baseline",
            "display_name": "DTAF",
            "model_name": "DTAF",
            "import_path": "ts_benchmark.baselines.dtaf.DTAF",
            "adapter": None,
            "source": "regression_test_fixture",
            "metrics": {"mse_norm": 1.0},
            "objective_metric": "mse_norm",
            "node_id": "baseline_dtaf",
            "model_config": {
                "model_name": "ts_benchmark.baselines.dtaf.DTAF",
                "adapter": None,
                "model_hyper_params": {},
            },
        },
    )
    session = AgentSession(
        task_id=task_id,
        base_dir=base_dir(),
        client=ProviderClient(task_id=task_id),
        dry_run=False,
        max_tool_result_chars=120_000,
    )
    session.ensure_dirs()
    return session
