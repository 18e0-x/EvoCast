from __future__ import annotations

import json
from pathlib import Path

from evocast.research import baseline_reference
from evocast.research.baseline_knowledge import (
    build_candidate_signature,
    build_reference_signature,
    load_candidate_result,
    load_reference_result,
    signature_hash,
    write_candidate_result,
)
from evocast.research.dataset_profile import write_skipped_dataset_profile
from evocast.state.domain_store import save_round_record, save_task_config


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_series_csv(path: Path, rows: int = 96) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["date,target,aux"]
    for idx in range(rows):
        lines.append(f"2024-01-{(idx % 28) + 1:02d} {idx % 24:02d}:00:00,{idx * 0.1:.4f},{idx * 0.2:.4f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_baseline_reference_must_precede_a_canonical_research_round(tmp_path: Path) -> None:
    task_id = "canonical_reference_order"
    save_task_config(str(tmp_path), task_id, {"task_id": task_id, "build_mode": True})
    save_round_record(
        str(tmp_path), task_id,
        {"round_id": 1, "research_id": "Research001", "status": "round_started"},
    )

    try:
        baseline_reference.write_initial_baseline_reference(
            task_id=task_id,
            base_dir=str(tmp_path),
            config_path=str(tmp_path / "compiled_config.json"),
            baseline_record={},
        )
    except RuntimeError as exc:
        assert "BASELINE_REFERENCE_MUST_PRECEDE_RESEARCH" in str(exc)
    else:
        raise AssertionError("canonical Research round must prevent late baseline reference creation")


def _prepare_task(
    base_dir: Path,
    task_id: str,
    dataset_path: Path,
    *,
    stride: int = 1,
    num_rollings: int = 4,
) -> Path:
    knowledge_dir = base_dir / "task_knowledge" / task_id
    semantics = {
        "task_mode": "MM",
        "input_variable_topology": "multivariate",
        "prediction_target_selection": "all_targets",
        "target_columns": ["target", "aux"],
        "time_col": "date",
        "dataset_path": str(dataset_path),
        "frequency": "hourly",
        "strategy_name": "rolling_forecast",
        "objective_metric": "mse_norm",
        "input_chunk_length": 48,
    }
    compiled = {
        "data_config": {
            "data_set_name": "fixture_forecast",
            "dataset_path": str(dataset_path),
            "time_col": "date",
            "target_columns": ["target", "aux"],
            "task_semantics": semantics,
            "feature_dict": {"canonical_freq": "hourly", "freq": "h"},
            "scale": True,
        },
        "model_config": {"models": [], "recommend_model_hyper_params": {"input_chunk_length": 48, "output_chunk_length": 24}},
        "evaluation_config": {
            "strategy_args": {
                "strategy_name": "rolling_forecast",
                "horizon": 24,
                "stride": stride,
                "num_rollings": num_rollings,
                "seed": 2021,
            }
        },
    }
    task_config = {
        "task_id": task_id,
        "config_path": str(knowledge_dir / "compiled_config.json"),
        "objective_metric": "mse_norm",
        "metric_direction": "lower_is_better",
        "data_set_name": "fixture_forecast",
        "dataset_path": str(dataset_path),
        "horizon": 24,
        "seq_len": 48,
        "task_semantics": semantics,
        "feature_dict": compiled["data_config"]["feature_dict"],
        "build_mode": False,
    }
    _write_json(knowledge_dir / "compiled_config.json", compiled)
    _write_json(knowledge_dir / "task_config.json", task_config)
    write_skipped_dataset_profile(task_id=task_id, base_dir=str(base_dir))
    return knowledge_dir / "compiled_config.json"


def test_baseline_candidate_result_reuses_across_equivalent_tasks(tmp_path: Path) -> None:
    base_dir = tmp_path / "evocast"
    dataset_path = base_dir / "dataset" / "forecasting" / "fixture.csv"
    _write_series_csv(dataset_path)
    config_a = _prepare_task(base_dir, "baseline_cache_a", dataset_path)
    _prepare_task(base_dir, "baseline_cache_b", dataset_path)
    config_data = json.loads(config_a.read_text(encoding="utf-8"))
    model_entry = {
        "model_name": "ts_benchmark.baselines.fixture.Fixture",
        "model_hyper_params": {"batch_size": 32, "num_epochs": 10, "patience": 3, "input_chunk_length": 48, "output_chunk_length": 24},
    }
    signature_a = build_candidate_signature(
        task_id="baseline_cache_a",
        base_dir=str(base_dir),
        config_data=config_data,
        model_key="Fixture",
        model_entry=model_entry,
        objective_metric="mse_norm",
        budget="unified",
        seed=2021,
    )
    signature_b = build_candidate_signature(
        task_id="baseline_cache_b",
        base_dir=str(base_dir),
        config_data=config_data,
        model_key="Fixture",
        model_entry=model_entry,
        objective_metric="mse_norm",
        budget="unified",
        seed=2021,
    )
    assert signature_hash(signature_a) == signature_hash(signature_b)

    write_candidate_result(
        base_dir=str(base_dir),
        task_id="baseline_cache_a",
        signature=signature_a,
        result={
            "model_key": "Fixture",
            "node_id": "baseline_001_Fixture",
            "status": "success",
            "objective_metric": "mse_norm",
            "objective_value": 0.42,
            "metrics": {"mse_norm": 0.42, "mae_norm": 0.5},
            "model_config": model_entry,
            "seed": 2021,
        },
    )
    reused = load_candidate_result(str(base_dir), signature_b)

    assert reused["status"] == "success"
    assert reused["metrics"]["mse_norm"] == 0.42
    assert reused["baseline_knowledge"]["reused"] is True
    assert reused["baseline_knowledge"]["origin_task_id"] == "baseline_cache_a"


def test_baseline_candidate_signature_includes_rolling_evaluation_policy(tmp_path: Path) -> None:
    base_dir = tmp_path / "evocast"
    dataset_path = base_dir / "dataset" / "forecasting" / "fixture.csv"
    _write_series_csv(dataset_path)
    config_a = _prepare_task(base_dir, "baseline_cache_full", dataset_path, stride=1, num_rollings=48000)
    config_b = _prepare_task(base_dir, "baseline_cache_one", dataset_path, stride=24, num_rollings=1)
    model_entry = {
        "model_name": "ts_benchmark.baselines.fixture.Fixture",
        "model_hyper_params": {"batch_size": 32, "num_epochs": 10, "patience": 3, "input_chunk_length": 48, "output_chunk_length": 24},
    }

    signature_a = build_candidate_signature(
        task_id="baseline_cache_full",
        base_dir=str(base_dir),
        config_data=json.loads(config_a.read_text(encoding="utf-8")),
        model_key="Fixture",
        model_entry=model_entry,
        objective_metric="mse_norm",
        budget="unified",
        seed=2021,
    )
    signature_b = build_candidate_signature(
        task_id="baseline_cache_one",
        base_dir=str(base_dir),
        config_data=json.loads(config_b.read_text(encoding="utf-8")),
        model_key="Fixture",
        model_entry=model_entry,
        objective_metric="mse_norm",
        budget="unified",
        seed=2021,
    )

    assert signature_a["evaluation_signature"]["stride"] == 1
    assert signature_a["evaluation_signature"]["num_rollings"] == 48000
    assert signature_b["evaluation_signature"]["stride"] == 24
    assert signature_b["evaluation_signature"]["num_rollings"] == 1
    assert signature_hash(signature_a) != signature_hash(signature_b)

    write_candidate_result(
        base_dir=str(base_dir),
        task_id="baseline_cache_full",
        signature=signature_a,
        result={
            "model_key": "Fixture",
            "node_id": "baseline_001_Fixture",
            "status": "success",
            "objective_metric": "mse_norm",
            "objective_value": 0.42,
            "metrics": {"mse_norm": 0.42},
            "model_config": model_entry,
            "seed": 2021,
        },
    )
    assert load_candidate_result(str(base_dir), signature_b) == {}


def test_baseline_reference_reuses_across_equivalent_tasks_without_rerunning_seed_eval(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / "evocast"
    dataset_path = base_dir / "dataset" / "forecasting" / "fixture.csv"
    _write_series_csv(dataset_path)
    config_a = _prepare_task(base_dir, "reference_cache_a", dataset_path)
    config_b = _prepare_task(base_dir, "reference_cache_b", dataset_path)
    baseline_record = {
        "candidate_id": "baseline_001_Fixture",
        "node_id": "baseline_001_Fixture",
        "model_name": "Fixture",
        "display_name": "Fixture",
        "metrics": {"mse_norm": 0.40, "mae_norm": 0.50},
        "seed": 2021,
        "model_config": {
            "model_name": "ts_benchmark.baselines.fixture.Fixture",
            "model_hyper_params": {"batch_size": 32, "num_epochs": 10, "patience": 3, "input_chunk_length": 48, "output_chunk_length": 24},
        },
    }
    calls = {"count": 0}

    def _fake_seed_eval(**kwargs):
        calls["count"] += 1
        precomputed = list(kwargs.get("precomputed_seed_values") or [])
        per_seed = [
            {"seed": item["seed"], "metrics": dict(item["metrics"] or {}), "success": True}
            for item in precomputed
        ]
        for seed in list(kwargs.get("seed_list") or []):
            per_seed.append({"seed": seed, "metrics": {"mse_norm": 0.40 + (seed - 2021) * 0.001, "mae_norm": 0.50}, "success": True})
        mean_mse = sum(item["metrics"]["mse_norm"] for item in per_seed) / len(per_seed)
        return {
            "status": "completed",
            "seed_list": [item["seed"] for item in per_seed],
            "per_seed": per_seed,
            "valid_metric_seeds": len(per_seed),
            "mean_metrics": {"mse_norm": mean_mse, "mae_norm": 0.50},
            "result_path": str(base_dir / "runs" / "seed_eval.json"),
        }

    monkeypatch.setattr(baseline_reference, "run_seed_evaluation", _fake_seed_eval)
    first = baseline_reference.write_initial_baseline_reference(
        task_id="reference_cache_a",
        base_dir=str(base_dir),
        config_path=str(config_a),
        baseline_record=baseline_record,
        objective_metric="mse_norm",
        seed_list=[2021, 2022, 2023],
    )
    second = baseline_reference.write_initial_baseline_reference(
        task_id="reference_cache_b",
        base_dir=str(base_dir),
        config_path=str(config_b),
        baseline_record=baseline_record,
        objective_metric="mse_norm",
        seed_list=[2021, 2022, 2023],
    )

    assert calls["count"] == 1
    assert second["source"] == "baseline_knowledge_reuse"
    assert second["origin_task_id"] == "reference_cache_a"
    assert second["metric_stats"]["mse_norm"] == first["metric_stats"]["mse_norm"]
    signature = build_reference_signature(
        task_id="reference_cache_b",
        base_dir=str(base_dir),
        config_data=json.loads(config_b.read_text(encoding="utf-8")),
        baseline_record=baseline_record,
        objective_metric="mse_norm",
        seed_list=[2021, 2022, 2023],
    )
    assert load_reference_result(str(base_dir), signature, objective_metric="mse_norm")
