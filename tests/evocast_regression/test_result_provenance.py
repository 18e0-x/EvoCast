from __future__ import annotations

import base64
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from evocast.domain.metric_parser import parse_metrics_from_paths
from evocast.domain.result_provenance import (
    build_result_provenance,
    model_entry_hash,
    stamp_result_artifacts,
    validate_result_artifact_provenance,
    variant_source_sha256,
)
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.rolling_forecast import RollingForecast
from ts_benchmark.recording import read_record_file, write_record_file


def _encoded(value: object) -> str:
    return base64.b64encode(pickle.dumps(value)).decode("utf-8")


def _record(path: Path) -> str:
    df = pd.DataFrame(
        [
            {
                "model_name": "ETSformer",
                "strategy_args": "{}",
                "model_params": "{}",
                "mse": 1.25,
                "mse_norm": 1.5,
                "mae": 0.5,
                "file_name": "fixture.csv",
                "fit_time": 0.1,
                "inference_time": 0.2,
                "actual_data": _encoded(np.asarray([[1.0, 2.0], [3.0, 4.0]])),
                "inference_data": _encoded(np.asarray([[1.5, 2.5], [3.5, 4.5]])),
                "forecast_execution_path": "batch_forecast",
                "log_info": "",
            }
        ]
    )
    return write_record_file(df, str(path / "ETSformer.fixture.csv"), compress_method="gz")


def test_variant_identity_hash_includes_source_sha(tmp_path: Path) -> None:
    workspace = tmp_path / "Research001" / "variant"
    workspace.mkdir(parents=True)
    entry = workspace / "round_entry.py"
    helper = workspace / "helper.py"
    entry.write_text("from helper import X\nclass Model:\n    pass\n", encoding="utf-8")
    helper.write_text("X = 1\n", encoding="utf-8")

    model_entry = {
        "model_name": "ts_benchmark.baselines.time_series_library.ETSformer",
        "variant_path": str(entry),
        "model_hyper_params": {"seq_len": 16},
    }
    first_source_hash = variant_source_sha256(str(entry))
    first_entry_hash = model_entry_hash(model_entry, variant_path=str(entry))

    helper.write_text("X = 2\n", encoding="utf-8")

    assert variant_source_sha256(str(entry)) != first_source_hash
    assert model_entry_hash(model_entry, variant_path=str(entry)) != first_entry_hash


def test_result_artifact_contains_variant_provenance(tmp_path: Path) -> None:
    workspace = tmp_path / "Research001" / "variant"
    workspace.mkdir(parents=True)
    entry = workspace / "round_entry.py"
    entry.write_text("class Model:\n    pass\n", encoding="utf-8")
    log_path = _record(tmp_path)
    model_entry = {
        "model_name": "ts_benchmark.baselines.time_series_library.ETSformer",
        "variant_path": str(entry),
        "model_hyper_params": {},
    }
    expected = build_result_provenance(
        task_id="task_x",
        run_id="run_x",
        candidate_id=str(entry),
        candidate_kind="variant",
        model_entry=model_entry,
        evaluation_budget="build_mode",
        build_mode=True,
        variant_path=str(entry),
    )

    stamped = stamp_result_artifacts([log_path], expected)
    validation = validate_result_artifact_provenance(
        [log_path],
        expected,
        require_prediction_hash=True,
        require_batch_forecast=True,
    )
    df = read_record_file(log_path)

    assert stamped and stamped[0]["prediction_hashes"]
    assert validation["status"] == "ok"
    assert df["evocast_variant_path"].iloc[0] == expected["variant_path"]
    assert df["evocast_variant_source_sha256"].iloc[0] == expected["variant_source_sha256"]
    assert df["evocast_model_entry_hash"].iloc[0] == expected["model_entry_hash"]
    assert df["evocast_prediction_hash"].iloc[0]
    assert df["evocast_forecast_execution_path"].iloc[0] == "batch_forecast"


def test_evaluator_rejects_missing_variant_provenance(tmp_path: Path) -> None:
    workspace = tmp_path / "Research001" / "variant"
    workspace.mkdir(parents=True)
    entry = workspace / "round_entry.py"
    entry.write_text("class Model:\n    pass\n", encoding="utf-8")
    log_path = _record(tmp_path)
    expected = build_result_provenance(
        task_id="task_x",
        run_id="run_x",
        candidate_id=str(entry),
        candidate_kind="variant",
        model_entry={"model_name": "ETSformer", "variant_path": str(entry), "model_hyper_params": {}},
        evaluation_budget="build_mode",
        build_mode=True,
        variant_path=str(entry),
    )

    validation = validate_result_artifact_provenance([log_path], expected)

    assert validation["status"] == "failed"
    assert "missing provenance columns" in validation["failures"][0]


def test_baseline_artifact_allows_empty_variant_identity(tmp_path: Path) -> None:
    log_path = _record(tmp_path)
    model_entry = {"model_name": "ETSformer", "model_hyper_params": {}}
    expected = build_result_provenance(
        task_id="task_x",
        run_id="baseline_x",
        candidate_id="baseline",
        candidate_kind="baseline",
        model_entry=model_entry,
        evaluation_budget="build_mode",
        build_mode=True,
        variant_path=None,
    )

    stamp_result_artifacts([log_path], expected)
    validation = validate_result_artifact_provenance([log_path], expected, require_batch_forecast=True)

    assert validation["status"] == "ok"


def test_metric_parser_ignores_evocast_provenance_numeric_columns(tmp_path: Path) -> None:
    workspace = tmp_path / "Research001" / "variant"
    workspace.mkdir(parents=True)
    entry = workspace / "round_entry.py"
    entry.write_text("class Model:\n    pass\n", encoding="utf-8")
    log_path = _record(tmp_path)
    expected = build_result_provenance(
        task_id="task_x",
        run_id="run_x",
        candidate_id=str(entry),
        candidate_kind="variant",
        model_entry={"model_name": "ETSformer", "variant_path": str(entry), "model_hyper_params": {}},
        evaluation_budget="build_mode",
        build_mode=True,
        variant_path=str(entry),
    )
    stamp_result_artifacts([log_path], expected)

    parsed = parse_metrics_from_paths([log_path], objective_metric="mse_norm")

    assert parsed["metric_values"]["mse_norm"] == 1.5
    assert "evocast_prediction_mean" not in parsed["metric_values"]


def test_formal_forecast_probe_uses_batch_forecast_path() -> None:
    assert FieldNames.FORECAST_EXECUTION_PATH in RollingForecast.field_names.fget(  # type: ignore[arg-type]
        type("Dummy", (), {"evaluator": type("Eval", (), {"metric_names": []})()})()
    )
