from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evocast.policy.error_taxonomy import ErrorLabel
from evocast.policy.model_hparam_compat import (
    apply_model_hparam_compatibility,
    validate_model_hparam_compatibility,
)
from evocast.runners.baseline_runner import _validate_model_constraints, align_model_hparams_to_task
from evocast.runners.tfb_pipeline_runner import build_run_configs


def _compiled_config(seq_len: int = 96, horizon: int = 96) -> dict:
    return {
        "data_config": {
            "seq_len": seq_len,
            "horizon": horizon,
        },
        "model_config": {
            "recommend_model_hyper_params": {
                "input_chunk_length": seq_len,
                "output_chunk_length": horizon,
            }
        },
        "evaluation_config": {"strategy_args": {"horizon": horizon}},
    }


def _spec(model_key: str) -> dict:
    return {"model_key": model_key, "source": "time_series_library"}


def test_etsformer_layers_are_aligned_before_baseline_run() -> None:
    hparams, notes = align_model_hparams_to_task(
        _spec("ETSformer"),
        _compiled_config(),
        {"e_layers": 2, "d_layers": 1, "seq_len": 96, "pred_len": 96},
    )

    assert hparams["e_layers"] == 2
    assert hparams["d_layers"] == 2
    assert any("ETSformer" in note for note in notes)
    assert validate_model_hparam_compatibility("ETSformer", hparams, _compiled_config()) is None


def test_temporal_fusion_transformer_heads_are_aligned() -> None:
    hparams, notes = apply_model_hparam_compatibility(
        "TemporalFusionTransformer",
        {"d_model": 510, "n_heads": 8},
        _compiled_config(),
    )

    assert 510 % hparams["n_heads"] == 0
    assert any("TemporalFusionTransformer" in note for note in notes)


def test_lightts_unfixable_constructor_chunk_constraint_fails_fast() -> None:
    error = _validate_model_constraints(
        _spec("LightTS"),
        _compiled_config(seq_len=95, horizon=24),
        {"model_hyper_params": {"seq_len": 95, "pred_len": 24}},
    )

    assert error is not None
    assert error["error_type"] == ErrorLabel.SHAPE_CONSTRAINT_VIOLATION.value
    assert "LightTS requires seq_len divisible" in error["error_message"]


def test_generic_patch_window_params_are_capped_by_task_seq_len() -> None:
    hparams, notes = apply_model_hparam_compatibility(
        "AnyPatchModel",
        {"seq_len": 96, "patch_len": 720, "patch_size": 128, "window_size": 192},
        _compiled_config(seq_len=96, horizon=96),
    )

    assert hparams["patch_len"] == 96
    assert hparams["patch_size"] == 96
    assert hparams["window_size"] == 96
    assert validate_model_hparam_compatibility("AnyPatchModel", hparams, _compiled_config()) is None
    assert len(notes) == 3


def test_timefilter_default_patch_len_is_runtime_compatible_with_build_window() -> None:
    hparams, notes = apply_model_hparam_compatibility(
        "TimeFilter",
        {"seq_len": 96, "horizon": 96, "patch_len": 720},
        _compiled_config(seq_len=96, horizon=96),
    )

    assert hparams["patch_len"] == 96
    assert any("patch_len" in note for note in notes)


def test_pipeline_runner_applies_generic_patch_window_compatibility() -> None:
    _, model_config, _ = build_run_configs(
        _compiled_config(seq_len=96, horizon=96),
        [
            {
                "model_key": "TIMEFILTER",
                "model_name": "TimeFilter",
                "model_hyper_params": {"seq_len": 96, "horizon": 96, "patch_len": 720},
            }
        ],
        save_path="compat_test",
        seed=2021,
    )

    entry = model_config["models"][0]
    assert entry["model_hyper_params"]["patch_len"] == 96
    assert entry["model_hyper_params"]["c_out"] == 1
    assert entry["model_hyper_params"]["enc_in"] == 1
    assert entry["model_hyper_params"]["pred_len"] == 96
    assert any("patch_len" in note for note in entry["model_hparam_compatibility_notes"])


def test_pipeline_runner_uses_source_model_key_for_workspace_variant_compatibility() -> None:
    _, model_config, _ = build_run_configs(
        _compiled_config(seq_len=96, horizon=96),
        [
            {
                "model_key": "TIMEFILTER",
                "model_name": "evocast_workspace.round_entry",
                "model_hyper_params": {"seq_len": 96, "horizon": 96, "patch_len": 720},
            }
        ],
        save_path="compat_variant_test",
        seed=2021,
    )

    entry = model_config["models"][0]
    assert entry["model_hyper_params"]["patch_len"] == 96
    assert any("patch_len" in note for note in entry["model_hparam_compatibility_notes"])


def test_build_run_configs_does_not_mutate_strategy_args_with_overrides() -> None:
    config = _compiled_config(seq_len=720, horizon=120)
    config["evaluation_config"]["strategy_args"].update(
        {
            "strategy_name": "rolling_forecast",
            "stride": 1,
            "num_rollings": 48000,
        }
    )

    _, _, smoke_eval = build_run_configs(
        config,
        [{"model_name": "Fixture", "model_hyper_params": {}}],
        save_path="smoke",
        seed=2027,
        override_eval_args={"save_true_pred": True, "stride": 120, "num_rollings": 1},
    )
    _, _, formal_eval = build_run_configs(
        config,
        [{"model_name": "Fixture", "model_hyper_params": {}}],
        save_path="formal",
        seed=2027,
        override_eval_args={"save_true_pred": True},
    )

    assert smoke_eval["strategy_args"]["stride"] == 120
    assert smoke_eval["strategy_args"]["num_rollings"] == 1
    assert formal_eval["strategy_args"]["stride"] == 1
    assert formal_eval["strategy_args"]["num_rollings"] == 48000
    assert config["evaluation_config"]["strategy_args"]["stride"] == 1
    assert config["evaluation_config"]["strategy_args"]["num_rollings"] == 48000
