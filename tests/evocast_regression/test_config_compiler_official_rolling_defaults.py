from __future__ import annotations

from evocast.configurator.config_compiler import compile_config
from evocast.configurator.config_validator import ValidationReport


def test_compiler_preserves_official_rolling_forecast_defaults() -> None:
    intent = {
        "dataset_path": "dataset/forecasting/fixture.csv",
        "time_col": "date",
        "target_columns": ["target"],
        "task_mode": "MS",
        "horizons": [120],
        "frequency": "15min",
        "objective_metric": "mse_norm",
        "strategy_name": "rolling_forecast",
    }
    report = ValidationReport(
        intent=intent,
        valid=True,
        row_count=60000,
        column_names=["date", "target", "aux"],
        numeric_target_indices={"target": 1},
        target_indices={"target": 1},
    )

    config = compile_config(intent, report)
    strategy_args = config["evaluation_config"]["strategy_args"]

    assert strategy_args["stride"] == 1
    assert strategy_args["num_rollings"] == 48000


def test_build_mode_keeps_validation_and_limits_test_to_one_window() -> None:
    intent = {
        "dataset_path": "dataset/forecasting/fixture.csv",
        "time_col": "date",
        "target_columns": ["target"],
        "task_mode": "MS",
        "horizons": [120],
        "frequency": "15min",
        "objective_metric": "mse_norm",
        "strategy_name": "rolling_forecast",
    }
    report = ValidationReport(
        intent=intent,
        valid=True,
        row_count=60000,
        column_names=["date", "target", "aux"],
        numeric_target_indices={"target": 1},
        target_indices={"target": 1},
    )

    strategy_args = compile_config(intent, report, build_mode=True)["evaluation_config"]["strategy_args"]

    assert strategy_args["num_rollings"] == 1
    assert strategy_args["stride"] == 120
    assert strategy_args["train_ratio_in_tv"] != 1
