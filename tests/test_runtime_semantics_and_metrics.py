import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from evocast.runners.tfb_pipeline_runner import (
    _complete_runtime_shape_hparams,
    build_run_configs,
)
from ts_benchmark.evaluation.metrics.regression_metrics import (
    mae_norm,
    mase,
    mase_norm,
    mape,
    mape_norm,
)
from ts_benchmark.evaluation.strategy.forecasting import ForecastingStrategy


FIXTURE_DATASET = str(Path(__file__).resolve().parent / "fixtures" / "air_quality_ready.csv")


def test_mm_runtime_shape_uses_all_numeric_dataset_columns():
    config = {
        "data_config": {
            "dataset_path": FIXTURE_DATASET,
            "time_col": "date",
            "task_semantics": {"task_mode": "MM", "target_columns": []},
        },
        "model_config": {"recommend_model_hyper_params": {}},
        "evaluation_config": {"strategy_args": {}},
    }

    hparams = _complete_runtime_shape_hparams(config, {})

    assert hparams["enc_in"] == 12
    assert hparams["dec_in"] == 12
    assert hparams["c_out"] == 12


def test_build_run_configs_records_all_mm_channels():
    config = {
        "data_config": {
            "dataset_path": FIXTURE_DATASET,
            "time_col": "date",
            "task_semantics": {"task_mode": "MM", "target_columns": []},
        },
        "model_config": {"models": [], "recommend_model_hyper_params": {}},
        "evaluation_config": {"strategy_args": {}},
    }

    _, model_config, _ = build_run_configs(
        config,
        [{"model_name": "ts_benchmark.baselines.time_series_library.DLinear"}],
    )
    recorded = model_config["models"][0]["model_hyper_params"]

    assert {recorded["enc_in"], recorded["dec_in"], recorded["c_out"]} == {12}


def test_mape_ignores_zero_actual_values_and_mape_norm_keeps_original_scale():
    actual = np.array([[0.0], [2.0], [4.0]])
    predicted = np.array([[1.0], [1.0], [2.0]])
    scaler = StandardScaler().fit(np.array([[0.0], [2.0], [4.0], [6.0]]))

    assert mape(actual, predicted) == 50.0
    assert mape_norm(actual, predicted, scaler) == 50.0
    assert np.isnan(mape(np.zeros((3, 1)), np.ones((3, 1))))


def test_mase_uses_requested_seasonality_instead_of_sentinel_value():
    history = np.arange(1.0, 9.0)[:, None]
    actual = np.array([[9.0], [10.0]])
    predicted = np.array([[8.0], [9.0]])
    scaler = StandardScaler().fit(history)

    assert mase(actual, predicted, history, seasonality=1) == 1.0
    assert np.isclose(
        mase_norm(actual, predicted, scaler, history, seasonality=1), 1.0
    )
    assert np.isnan(mase(actual, predicted, history, seasonality=99))


def test_seasonality_matches_hourly_and_quarter_hourly_data():
    hourly = pd.DataFrame(index=pd.date_range("2024-01-01", periods=48, freq="h"))
    quarter_hourly = pd.DataFrame(
        index=pd.date_range("2024-01-01", periods=384, freq="15min")
    )

    assert ForecastingStrategy._get_seasonality(None, hourly, None) == 24
    assert ForecastingStrategy._get_seasonality(None, quarter_hourly, None) == 96
