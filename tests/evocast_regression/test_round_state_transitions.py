from __future__ import annotations

import json
from pathlib import Path

import pytest

from evocast.harness.rounds import (
    close_current_round,
    current_round,
    list_rounds,
    record_experiment_run,
    record_gate_event,
    record_variant_write_failure,
    round_progress,
    seed_eval_obligations,
    start_round,
)
from evocast.policy.agent_control_policy import round_control_policy
from evocast.probe.failure_signature import failure_signature


def _write_task_config(base_dir: Path, task_id: str, *, max_rounds: int = 6) -> None:
    path = base_dir / "task_knowledge" / task_id / "task_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "objective_metric": "mse_norm",
                "metric_direction": "lower_is_better",
                "force_full_rounds": True,
                "max_rounds": max_rounds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _start(base_dir: Path, task_id: str):
    return start_round(
        base_dir=str(base_dir),
        task_id=task_id,
        fit_point="model.predictor",
        hypothesis="Replace predictor with a shape-preserving variant.",
        evidence_source="test fixture",
    )


def _signature(variant_path: str, *, stage: str = "variant_write") -> dict:
    return failure_signature(
        error_type="WRAPPER_SUBCLASS_REQUIRED",
        message="Variant must subclass the expected baseline wrapper",
        traceback_text="RefuseGate blocked fixture",
        variant_path=variant_path,
        stage=stage,
    )


def test_write_failures_stay_auditable_and_terminal_failure_counts_as_round(tmp_path: Path) -> None:
    base_dir = tmp_path / "runtime"
    task_id = "round_write_failure"
    _write_task_config(base_dir, task_id)
    variant_path = f"round_sources/{task_id}/Research001/round_entry.py"

    record = _start(base_dir, task_id)
    assert record["status"] == "round_started"
    assert record["research_id"] == "Research001"

    updated = record_variant_write_failure(
        base_dir=str(base_dir),
        task_id=task_id,
        variant_path=variant_path,
        error_type="WRAPPER_SUBCLASS_REQUIRED",
        error_message="Variant must subclass DTAF",
        failure_signature=_signature(variant_path),
    )
    assert updated is not None
    assert updated["status"] == "repair_needed"
    assert updated["variant_path"] == variant_path
    assert updated["runs"][-1]["stage"] == "variant_write"

    limit = int(round_control_policy(str(base_dir)).get("repeated_failure_signature_limit") or 2)
    for attempt in range(2, limit + 1):
        updated = record_variant_write_failure(
            base_dir=str(base_dir),
            task_id=task_id,
            variant_path=variant_path,
            error_type="WRAPPER_SUBCLASS_REQUIRED",
            error_message=f"Still invalid attempt {attempt}",
            failure_signature=_signature(variant_path),
        )

    assert updated is not None
    assert updated["status"] == "write_failed"
    assert updated["failure_analysis"]["terminal_reason"] == "repeated_failure_signature"
    assert current_round(str(base_dir), task_id) is None

    progress = round_progress(str(base_dir), task_id)
    assert progress["research_rounds"] == 1
    assert progress["terminal_rounds"] == 1
    assert progress["build_failure_rounds"] == 1
    assert progress["failed_metric_production_rounds"] == 1


def test_terminal_round_closes_pointer_and_next_research_id_advances(tmp_path: Path) -> None:
    base_dir = tmp_path / "runtime"
    task_id = "round_terminal_next"
    _write_task_config(base_dir, task_id)

    _start(base_dir, task_id)
    closed = close_current_round(
        base_dir=str(base_dir),
        task_id=task_id,
        status="proposal_failed",
        reason="fixture terminal proposal failure",
    )
    assert closed["status"] == "proposal_failed"
    assert current_round(str(base_dir), task_id) is None

    next_record = _start(base_dir, task_id)
    assert next_record["round_id"] == 2
    assert next_record["research_id"] == "Research002"


def test_explicit_terminal_round_close_is_idempotent(tmp_path: Path) -> None:
    base_dir = tmp_path / "runtime"
    task_id = "round_terminal_idempotent_close"
    _write_task_config(base_dir, task_id)

    opened = _start(base_dir, task_id)
    closed = close_current_round(
        base_dir=str(base_dir),
        task_id=task_id,
        status="experiment_failed",
        reason="first terminal close",
        round_id=opened["round_id"],
        research_id=opened["research_id"],
    )
    assert current_round(str(base_dir), task_id) is None

    repeated = close_current_round(
        base_dir=str(base_dir),
        task_id=task_id,
        status="infra_failed",
        reason="second close must not overwrite",
        round_id=opened["round_id"],
        research_id=opened["research_id"],
    )

    assert repeated["status"] == "experiment_failed"
    assert repeated["close_reason"] == "first terminal close"
    assert repeated["closed_at"] == closed["closed_at"]
    assert round_progress(str(base_dir), task_id)["terminal_rounds"] == 1


def test_positive_candidate_requires_seed_eval_before_close_or_next_round(tmp_path: Path) -> None:
    base_dir = tmp_path / "runtime"
    task_id = "round_seed_eval_obligation"
    _write_task_config(base_dir, task_id)
    variant_path = f"round_sources/{task_id}/Research001/round_entry.py"

    _start(base_dir, task_id)
    gated = record_gate_event(
        base_dir=str(base_dir),
        task_id=task_id,
        gate_event={
            "run_id": "seed_eval_required",
            "variant_path": variant_path,
            "candidate_name": variant_path,
            "decision": "needs_seed_eval",
            "objective_metric": "mse_norm",
            "candidate_value": 0.99,
            "reference_kind": "current_best",
            "reference_value": 1.0,
            "current_best_value": 1.0,
            "relative_improvement": 0.01,
            "candidate_metrics": {"mse_norm": 0.99},
            "gate": {
                "decision": "needs_seed_eval",
                "reference_relative_improvement": 0.01,
                "reason": "positive candidate below acceptance threshold",
            },
        },
    )

    assert gated is not None
    assert gated["status"] == "gate_checked"
    assert seed_eval_obligations(str(base_dir), task_id)

    with pytest.raises(RuntimeError, match="NEEDS_SEED_EVAL_OBLIGATION"):
        close_current_round(
            base_dir=str(base_dir),
            task_id=task_id,
            status="rejected",
            reason="cannot close without seed eval",
        )

    with pytest.raises(RuntimeError, match="NEEDS_SEED_EVAL_OBLIGATION"):
        start_round(
            base_dir=str(base_dir),
            task_id=task_id,
            fit_point="model.patch_embedding",
            hypothesis="Another concrete variant.",
            evidence_source="test fixture",
        )


def test_experiment_failure_uses_canonical_round_sources_path(tmp_path: Path) -> None:
    base_dir = tmp_path / "runtime"
    task_id = "round_path_canonical"
    _write_task_config(base_dir, task_id)
    relative_variant = f"round_sources/{task_id}/Research001/round_entry.py"
    absolute_variant = tmp_path / "repo" / relative_variant

    _start(base_dir, task_id)
    failed = record_experiment_run(
        base_dir=str(base_dir),
        task_id=task_id,
        run_id="failed_experiment",
        variant_path=str(absolute_variant),
        status="failed",
        error_type="invalid_variant_contract",
        metrics={},
        failure_signature=failure_signature(
            error_type="invalid_variant_contract",
            message="synthetic failed experiment",
            traceback_text="Traceback synthetic",
            variant_path=str(absolute_variant),
            stage="experiment",
        ),
        failure_evidence={"innermost_project_frame": {"file": str(absolute_variant), "line": 1}},
    )

    assert failed is not None
    assert failed["status"] == "repair_needed"
    assert failed["variant_path"] == relative_variant
    assert failed["runs"][-1]["variant_path"] == relative_variant

    follow_up = record_experiment_run(
        base_dir=str(base_dir),
        task_id=task_id,
        run_id="failed_experiment_again",
        variant_path=relative_variant,
        status="failed",
        error_type="invalid_variant_contract",
        metrics={},
    )
    assert follow_up is not None
    assert follow_up["runs"][-1]["variant_path"] == relative_variant
    assert len(list_rounds(str(base_dir), task_id)) == 1
