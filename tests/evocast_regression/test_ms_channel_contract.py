from __future__ import annotations

from pathlib import Path

from evocast.variant.contract import _resolve_channel_contract, validate_variant_runtime_contract


def _ms_config(dataset_path: Path) -> dict:
    return {
        "data_config": {
            "dataset_path": str(dataset_path),
            "time_col": "date",
            "target_columns": ["Usage_kWh"],
            "task_semantics": {
                "task_mode": "MS",
                "input_variable_topology": "multivariate",
                "prediction_target_selection": "single_target",
                "target_columns": ["Usage_kWh"],
                "time_col": "date",
                "dataset_path": str(dataset_path),
                "horizons": [120],
                "input_chunk_length": 720,
            },
        },
        "evaluation_config": {
            "strategy_args": {
                "horizon": 120,
                "target_channel": [0],
            }
        },
        "model_config": {
            "recommend_model_hyper_params": {
                "input_chunk_length": 720,
                "output_chunk_length": 120,
                "norm": True,
            }
        },
    }


def _write_steel_like_csv(path: Path) -> None:
    columns = ["date", "Usage_kWh", *[f"exog_{idx}" for idx in range(1, 10)]]
    path.write_text(",".join(columns) + "\n2026-01-01 00:00:00," + ",".join(["0"] * 10) + "\n", encoding="utf-8")


def test_ms_single_target_channel_contract_separates_input_and_eval_channels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "steel_industry_ready.csv"
    _write_steel_like_csv(dataset_path)
    hparams = {"enc_in": 10, "dec_in": 10, "c_out": 10, "seq_len": 720, "horizon": 120, "pred_len": 120}

    contract = _resolve_channel_contract(_ms_config(dataset_path), hparams)

    assert contract["task_mode"] == "MS"
    assert contract["input_channels"] == 10
    assert contract["raw_output_channels"] == 10
    assert contract["target_channels"] == 1
    assert contract["needs_target_slice"] is True


def test_ms_single_target_runtime_probe_accepts_full_raw_channels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "steel_industry_ready.csv"
    _write_steel_like_csv(dataset_path)
    variant_dir = tmp_path / "round_sources" / "task" / "Research001"
    variant_dir.mkdir(parents=True)
    variant_path = variant_dir / "round_entry.py"
    variant_path.write_text(
        """
from types import SimpleNamespace
import torch


class Inner(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input):
        batch = input.shape[0]
        horizon = int(self.config.horizon)
        channels = int(self.config.c_out)
        return input.new_zeros(batch, horizon, channels) + self.weight


class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = SimpleNamespace(**kwargs)
        self.model = Inner(self.config)

    def forecast_fit(self, *args, **kwargs):
        return self

    def forecast(self, *args, **kwargs):
        raise NotImplementedError

    def batch_forecast(self, *args, **kwargs):
        raise NotImplementedError

    def _process(self, input, target, input_mark, target_mark):
        return {"output": self.model(input)}
""".strip(),
        encoding="utf-8",
    )
    entry = {
        "model_name": "example.Model",
        "variant_path": str(variant_path),
        "model_hyper_params": {
            "enc_in": 10,
            "dec_in": 10,
            "c_out": 10,
            "seq_len": 720,
            "label_len": 360,
            "horizon": 120,
            "pred_len": 120,
        },
    }

    result = validate_variant_runtime_contract(
        tfb_config=_ms_config(dataset_path),
        variant_entry=entry,
        variant_path=str(variant_path),
        seed=2027,
    )

    assert result["status"] == "ok", result
    assert result["input_shape"] == [2, 720, 10]
    assert result["output_shape"] == [2, 120, 10]
    assert result["expected_output_shape"] == [2, 120, 10]
    assert result["expected_eval_output_shape"] == [2, 120, 1]
    assert result["accepted_output_shape"] == [2, 120, 1]
    assert result["output_slice_contract"]["status"] == "target_channel_sliceable"
