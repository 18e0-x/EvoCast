"""Focused regression tests for EvoCast seed-gate policy fixes."""

from __future__ import annotations

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evocast.policy.gate import gate_decision, seed_eval_significance_decision
from evocast.runners.seed_runner import _seed_success_counts, run_seed_evaluation
from evocast.state.runtime.store import load_runtime_state, sync_best_baseline, sync_current_best


SEED_DECISION_POLICY = {
    "min_effect_size": 0.0,
    "min_relative_improvement": 0.005,
    "min_absolute_improvement": 0.0001,
    "reference_std_multiplier": 2.0,
    "required_seed_count": 3,
}

SINGLE_SEED_GATE_POLICY = {
    "min_relative_improvement": 0.01,
    "min_seed_eval_relative_improvement": 0.005,
}


def test_round10_three_seed_relative_accept() -> None:
    decision = seed_eval_significance_decision(
        variant_mean=0.3159918578377337,
        reference_mean=0.33458745108124427,
        reference_std=0.014764420257831751,
        reference_seed_count=3,
        variant_seed_count=3,
        objective_metric="mse_norm",
        direction="lower",
        **SEED_DECISION_POLICY,
    )
    assert decision["decision"] == "accept", decision
    assert decision["accept_rule"] == "three_seed_relative_improvement", decision
    assert decision["relative_improvement"] > SEED_DECISION_POLICY["min_relative_improvement"], decision


def test_seed_accept_requires_three_valid_seeds() -> None:
    decision = seed_eval_significance_decision(
        variant_mean=0.90,
        reference_mean=1.00,
        reference_std=0.01,
        reference_seed_count=1,
        variant_seed_count=3,
        objective_metric="mse_norm",
        direction="lower",
        **SEED_DECISION_POLICY,
    )
    assert decision["decision"] != "accept", decision


def test_seed_success_counts_require_valid_objective_metric() -> None:
    counts = _seed_success_counts(
        [
            {"success": True, "objective_value": 0.10},
            {"success": True, "objective_value": None},
            {"success": False, "objective_value": None},
        ]
    )
    assert counts["pipeline_successful_seeds"] == 2, counts
    assert counts["successful_seeds"] == 1, counts
    assert counts["valid_metric_seeds"] == 1, counts


def test_tiny_current_best_requires_absolute_floor() -> None:
    decision = seed_eval_significance_decision(
        variant_mean=0.00099,
        reference_mean=0.001,
        reference_std=0.0,
        reference_seed_count=3,
        variant_seed_count=3,
        objective_metric="loss",
        direction="lower",
        **SEED_DECISION_POLICY,
    )
    assert decision["decision"] == "reject", decision


def test_single_seed_marginal_improvement_has_no_seed_obligation() -> None:
    decision = gate_decision(
        metrics={"mse_norm": 0.999},
        reference_metrics={"mse_norm": 1.0},
        objective_metric="mse_norm",
        **SINGLE_SEED_GATE_POLICY,
    )
    assert decision["decision"] == "marginal_no_seed_eval", decision


def test_non_positive_improvement_rejects_without_seed_obligation() -> None:
    decision = gate_decision(
        metrics={"mse_norm": 1.001},
        reference_metrics={"mse_norm": 1.0},
        objective_metric="mse_norm",
        **SINGLE_SEED_GATE_POLICY,
    )
    assert decision["decision"] == "reject", decision
    assert "no seed evaluation is required" in decision["reason"], decision


def test_current_best_promotion_reuses_min_relative_improvement(tmp_path: Path) -> None:
    base_dir = tmp_path / "repo"
    task_id = "unified_promotion_threshold"
    sync_best_baseline(
        str(base_dir),
        task_id,
        {
            "candidate_id": "baseline",
            "display_name": "Baseline",
            "objective_metric": "mse_norm",
            "metrics": {"mse_norm": 1.0},
        },
    )

    sync_current_best(
        str(base_dir),
        task_id,
        {
            "candidate_id": "tiny_gain",
            "display_name": "Tiny gain",
            "objective_metric": "mse_norm",
            "metrics": {"mse_norm": 0.995},
        },
    )
    state = load_runtime_state(str(base_dir), task_id, auto_migrate=False)
    assert state.current_best.candidate_id == "baseline"

    sync_current_best(
        str(base_dir),
        task_id,
        {
            "candidate_id": "meaningful_gain",
            "display_name": "Meaningful gain",
            "objective_metric": "mse_norm",
            "metrics": {"mse_norm": 0.989},
        },
    )
    state = load_runtime_state(str(base_dir), task_id, auto_migrate=False)
    assert state.current_best.candidate_id == "meaningful_gain"


def test_seed_eval_below_unified_promotion_threshold_does_not_accept(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / "repo"
    task_id = "seed_eval_below_unified_threshold"
    task_dir = base_dir / "task_knowledge" / task_id
    task_dir.mkdir(parents=True)
    config_path = task_dir / "compiled_config.json"
    config_path.write_text(json.dumps({"model_config": {"models": []}}), encoding="utf-8")
    sync_best_baseline(
        str(base_dir),
        task_id,
        {
            "candidate_id": "baseline",
            "display_name": "Baseline",
            "objective_metric": "mse_norm",
            "metrics": {"mse_norm": 1.0},
        },
    )
    sync_current_best(
        str(base_dir),
        task_id,
        {
            "candidate_id": "incumbent",
            "display_name": "Incumbent",
            "objective_metric": "mse_norm",
            "metrics": {"mse_norm": 0.965},
            "seed_eval": {
                "metric_stats": {
                    "mse_norm": {"mean": 0.965, "std": 0.0, "seed_count": 3}
                }
            },
        },
    )

    monkeypatch.setattr("evocast.runners.seed_runner.build_run_configs", lambda *args, **kwargs: ({}, {}, {}))
    monkeypatch.setattr("evocast.runners.seed_runner.run_pipeline", lambda *args, **kwargs: {"success": True, "log_paths": ["fixture.csv"]})
    monkeypatch.setattr("evocast.runners.seed_runner.stamp_result_artifacts", lambda *args, **kwargs: [])
    monkeypatch.setattr("evocast.runners.seed_runner.validate_result_artifact_provenance", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(
        "evocast.runners.seed_runner.parse_metrics_from_paths",
        lambda *args, **kwargs: {"metric_values": {"mse_norm": 0.962}},
    )

    result = run_seed_evaluation(
        task_id=task_id,
        node_id="candidate_run",
        candidate_id="Research004",
        model_config={"model_name": "fixture.Model", "model_hyper_params": {}},
        config_path=str(config_path),
        objective_metric="mse_norm",
        seed_list=[2021, 2022, 2023],
        base_dir=str(base_dir),
        reference_mean=0.965,
        reference_std=0.0,
        reference_seed_count=3,
        promote_on_accept=True,
    )

    state = load_runtime_state(str(base_dir), task_id, auto_migrate=False)
    assert result["significance_decision"]["decision"] == "reject"
    assert result["promoted_to_current_best"] is False
    assert "promoted_record" not in result
    assert "promotion_decision" not in result
    assert state.current_best.candidate_id == "incumbent"


def main() -> None:
    test_round10_three_seed_relative_accept()
    test_seed_accept_requires_three_valid_seeds()
    test_seed_success_counts_require_valid_objective_metric()
    test_tiny_baseline_requires_absolute_floor()
    test_single_seed_marginal_improvement_has_no_seed_obligation()
    test_non_positive_improvement_rejects_without_seed_obligation()
    print("PASS: seed gate policy regression tests")


if __name__ == "__main__":
    main()
