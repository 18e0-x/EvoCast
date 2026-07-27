from __future__ import annotations

from evocast.domain import round_semantics
from evocast.harness import rounds as compatibility_rounds
from evocast.state.round_projection import (
    gate_review_obligations,
    project_round_progress,
    seed_eval_obligations,
)


def test_one_idea_round_keeps_multiple_attempts_in_separate_statistics() -> None:
    record = {
        "round_id": 1,
        "research_id": "Research001",
        "round_scope": "research",
        "counts_toward_research_budget": True,
        "status": "experiment_failed",
        "phase": "closed",
        "failure_kind": "validation_failure",
        "runs": [
            {
                "run_id": "run_001",
                "status": "failed",
                "stage": "runtime_contract_probe",
                "error_type": "invalid_mechanism_contract",
            },
            {
                "run_id": "run_002",
                "status": "failed",
                "stage": "runtime_contract_probe",
                "error_type": "invalid_variant_contract",
            },
            {
                "run_id": "run_003",
                "status": "failed",
                "stage": "runtime_contract_probe",
                "error_type": "invalid_mechanism_contract",
            },
        ],
    }

    progress = project_round_progress(
        [record],
        current_round_id=None,
        current_research_id="",
    )

    assert progress["research_rounds"] == 1
    assert progress["terminal_rounds"] == 1
    assert progress["valid_trial_rounds"] == 0
    assert progress["metric_production_rate"] == 0.0
    assert progress["variant_attempts"] == 3
    assert progress["failed_variant_attempts"] == 3
    assert progress["attempt_failure_rate"] == 1.0
    assert progress["attempt_metric_production_rate"] == 0.0


def test_successful_scientific_rejection_is_valid_round_and_metric_attempt() -> None:
    record = {
        "round_id": 1,
        "research_id": "Research001",
        "round_scope": "research",
        "counts_toward_research_budget": True,
        "status": "scientific_rejected",
        "phase": "closed",
        "variant_path": "variant.py",
        "gate_decision": "reject",
        "gate_event": {
            "run_id": "run_001",
            "candidate_value": 0.6,
            "baseline_value": 0.5,
        },
        "runs": [
            {
                "run_id": "run_001",
                "status": "success",
                "stage": "experiment",
                "metrics": {"mse_norm": 0.6},
            }
        ],
    }

    progress = project_round_progress(
        [record],
        current_round_id=None,
        current_research_id="",
    )

    assert progress["valid_trial_rounds"] == 1
    assert progress["successful_rounds"] == 1
    assert progress["scientific_rejection_rounds"] == 1
    assert progress["metric_production_rate"] == 1.0
    assert progress["attempt_metric_production_rate"] == 1.0
    assert progress["attempt_failure_rate"] == 0.0


def test_obligation_projections_are_read_only_and_agree() -> None:
    record = {
        "round_id": 2,
        "research_id": "Research002",
        "round_scope": "research",
        "status": "gate_checked",
        "fit_point": "Informer",
        "variant_path": "variant.py",
        "gate_decision": "needs_seed_eval",
        "gate_event": {
            "run_id": "run_002",
            "candidate_name": "candidate",
            "objective_metric": "mse_norm",
            "candidate_value": 0.4,
            "baseline_value": 0.5,
        },
    }

    seed = seed_eval_obligations([record])
    gate = gate_review_obligations([record])

    assert seed[0]["run_id"] == gate[0]["run_id"] == "run_002"
    assert seed[0]["required_action"] == "run_seed_eval"
    assert gate[0]["required_action"] == "run_seed_eval_before_same_fit_point"
    assert "seed_eval" not in record


def test_compatibility_facade_delegates_to_domain_semantics() -> None:
    failed = {
        "status": "experiment_failed",
        "round_scope": "research",
        "runs": [
            {
                "run_id": "run_001",
                "status": "failed",
                "stage": "runtime_contract_probe",
                "error_type": "invalid_variant_contract",
            }
        ],
    }

    assert compatibility_rounds.classify_failure_kind(
        "experiment_failed",
        "invalid_variant_contract",
        False,
    ) == round_semantics.classify_failure_kind(
        "experiment_failed",
        "invalid_variant_contract",
        False,
    )
    assert compatibility_rounds._resolve_failure_kind(
        failed
    ) == round_semantics.resolve_failure_kind(failed)
    assert compatibility_rounds._round_counts_toward_research_budget(
        failed
    ) == round_semantics.counts_toward_research_budget(failed)
