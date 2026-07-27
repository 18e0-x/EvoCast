from __future__ import annotations

from evocast.tools.tfb_ablation import finalize_baseline_diagnosis
from tests.evocast_regression.support import create_dtaf_task


def test_baseline_diagnosis_counts_usable_evidence_only_from_valid_ablation() -> None:
    session = create_dtaf_task("baseline_diagnosis_semantics", max_rounds=1)
    payload = finalize_baseline_diagnosis(
        session,
        baseline_model="DTAF",
        objective_metric="mse_norm",
        baseline_metrics={"mse_norm": 1.0},
        plan={
            "evaluation_stage": "smoke",
            "targets": [{"target_id": "q001"}, {"target_id": "q002"}, {"target_id": "q003"}],
            "planner_source": "active_path_summary",
        },
        review={
            "status": "approved",
            "evaluation_stage": "smoke",
            "reviewed_targets": [{"target_id": "q001"}, {"target_id": "q002"}, {"target_id": "q003"}],
            "errors": [],
            "corrections": [],
        },
        results=[
            {
                "target_id": "q001",
                "target_name": "q001",
                "mechanism_id": "m_x",
                "mechanism_name": "mechanism x",
                "exact_edit_intent": "remove mechanism x",
                "status": "success",
                "evaluation_stage": "smoke",
                "metrics": {"mse_norm": 1.3},
                "metric_delta": {"mse_norm": {"delta": 0.3}},
                "usable_evidence_status": "usable_evidence",
                "artifact_paths": [],
                "created_at": "2026-07-04T00:00:00",
            },
            {
                "target_id": "q002",
                "target_name": "q002",
                "mechanism_id": "m_y",
                "mechanism_name": "mechanism y",
                "exact_edit_intent": "remove mechanism y",
                "status": "failed_ablation_round",
                "usable_evidence_status": "failed_evidence",
                "artifact_paths": [],
                "created_at": "2026-07-04T00:00:01",
            },
        ],
    )

    assert payload["target_discovery"]["status"] == "success"
    assert payload["target_discovery"]["target_count"] == 3
    assert payload["ablation_execution"]["status"] == "partial"
    assert payload["ablation_execution"]["successful_count"] == 1
    assert payload["ablation_execution"]["failed_count"] == 1
    assert payload["usable_evidence"]["status"] == "available"
    assert payload["usable_evidence"]["count"] == 1


def test_baseline_diagnosis_marks_zero_usable_execution_as_failed_evidence() -> None:
    session = create_dtaf_task("baseline_diagnosis_failed_evidence", max_rounds=1)
    payload = finalize_baseline_diagnosis(
        session,
        baseline_model="DTAF",
        objective_metric="mse_norm",
        baseline_metrics={"mse_norm": 1.0},
        plan={
            "evaluation_stage": "smoke",
            "targets": [{"target_id": "q001"}, {"target_id": "q002"}],
            "planner_source": "active_path_summary",
        },
        review={
            "status": "approved",
            "evaluation_stage": "smoke",
            "reviewed_targets": [{"target_id": "q001"}, {"target_id": "q002"}],
            "errors": [],
            "corrections": [],
        },
        results=[
            {
                "target_id": "q001",
                "target_name": "q001",
                "mechanism_id": "m_x",
                "status": "failed_ablation_round",
                "usable_evidence_status": "failed_evidence",
            },
            {
                "target_id": "q002",
                "target_name": "q002",
                "mechanism_id": "m_y",
                "status": "failed_ablation_round",
                "usable_evidence_status": "failed_evidence",
            },
        ],
    )

    assert payload["status"] == "failed_evidence"
    assert payload["ablation_execution"]["status"] == "failed"
    assert payload["usable_evidence"]["status"] == "missing"


def test_baseline_diagnosis_marks_rejected_review_as_invalid_plan() -> None:
    session = create_dtaf_task("baseline_diagnosis_invalid_plan", max_rounds=1)
    payload = finalize_baseline_diagnosis(
        session,
        baseline_model="DTAF",
        objective_metric="mse_norm",
        baseline_metrics={"mse_norm": 1.0},
        plan={
            "evaluation_stage": "smoke",
            "targets": [{"target_id": "q001"}],
            "planner_source": "active_path_summary",
        },
        review={
            "status": "rejected",
            "evaluation_stage": "smoke",
            "reviewed_targets": [],
            "errors": ["insufficient_valid_targets:0"],
            "corrections": [],
        },
        results=[],
    )

    assert payload["status"] == "invalid_plan"
    assert payload["target_discovery"]["status"] == "rejected"
    assert payload["usable_evidence"]["status"] == "missing"
