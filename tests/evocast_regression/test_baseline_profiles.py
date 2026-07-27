from __future__ import annotations

import json
from pathlib import Path

from evocast.domain.effective_model_config import resolve_effective_model_config


def _prepare_task(tmp_path: Path, *, task_id: str = "aq_profile_task") -> tuple[Path, dict]:
    repo = tmp_path / "repo"
    runtime = repo / ".evocast"
    knowledge = runtime / "task_knowledge" / task_id
    (repo / "evocast" / "core").mkdir(parents=True)
    (repo / "evocast" / "harness").mkdir(parents=True)
    (repo / "ts_benchmark").mkdir(parents=True)
    (repo / "evocast" / "configs" / "policies").mkdir(parents=True)
    baseline_profiles_src = Path(__file__).resolve().parents[2] / "evocast" / "configs" / "policies" / "baseline_profiles.yaml"
    (repo / "evocast" / "configs" / "policies" / "baseline_profiles.yaml").write_text(
        baseline_profiles_src.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    knowledge.mkdir(parents=True)
    compiled = {
        "data_config": {
            "dataset_path": "dataset/forecasting/air_quality_ready.csv",
            "seq_len": 168,
            "horizon": 24,
        },
        "model_config": {
            "recommend_model_hyper_params": {
                "input_chunk_length": 168,
                "output_chunk_length": 24,
            }
        },
        "evaluation_config": {
            "strategy_args": {
                "horizon": 24,
            }
        },
    }
    task_config = {
        "task_id": task_id,
        "dataset_path": "dataset/forecasting/air_quality_ready.csv",
        "seq_len": 168,
        "horizon": 24,
        "task_semantics": {
            "task_mode": "MM",
        },
    }
    (knowledge / "compiled_config.json").write_text(json.dumps(compiled), encoding="utf-8")
    (knowledge / "task_config.json").write_text(json.dumps(task_config), encoding="utf-8")
    return repo, compiled


def test_amplifier_air_quality_profile_overrides_unified_policy(tmp_path: Path) -> None:
    repo, compiled = _prepare_task(tmp_path)
    resolved = resolve_effective_model_config(
        config_data=compiled,
        base_dir=str(repo),
        task_id="aq_profile_task",
        model_entry={
            "model_key": "Amplifier",
            "model_name": "ts_benchmark.baselines.amplifier.Amplifier",
            "model_hyper_params": {},
        },
        requested_budget="unified",
        smoke=False,
    )
    hparams = dict(resolved.effective_model_hyper_params)
    assert hparams["lr"] == 0.06
    assert hparams["label_len"] == 84
    assert hparams["hidden_size"] == 128
    assert resolved.entry["baseline_profile_id"] == "air_quality_mm_amplifier_best"


def test_timekan_air_quality_profile_applies_to_research_variant_runs(tmp_path: Path) -> None:
    repo, compiled = _prepare_task(tmp_path)
    resolved = resolve_effective_model_config(
        config_data=compiled,
        base_dir=str(repo),
        task_id="aq_profile_task",
        model_entry={
            "model_key": "TimeKAN",
            "source_model_key": "TimeKAN",
            "model_name": "ts_benchmark.baselines.timekan.TimeKAN",
            "variant_path": "dummy_variant.py",
            "model_hyper_params": {},
        },
        baseline_model_config={
            "model_name": "ts_benchmark.baselines.timekan.TimeKAN",
            "model_hyper_params": {},
        },
        requested_budget="unified",
        smoke=False,
    )
    hparams = dict(resolved.effective_model_hyper_params)
    assert hparams["lr"] == 0.003
    assert hparams["label_len"] == 84
    assert hparams["d_model"] == 16
    assert resolved.entry["baseline_profile_id"] == "air_quality_mm_timekan_best"
