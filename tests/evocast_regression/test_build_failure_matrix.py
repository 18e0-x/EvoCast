from __future__ import annotations

from pathlib import Path

import pytest

from evocast.build.contract import BuildContract
from evocast.build.ledger import ResearchRoundLedger
from evocast.build.round_terminal import RoundTerminalService
from evocast.build.source_snapshot import source_manifest
from evocast.harness.rounds import (
    FAILURE_KIND_ACTIVATION,
    FAILURE_KIND_IMPLEMENTATION,
    FAILURE_KIND_RUNTIME,
    FAILURE_KIND_VALIDATION,
    record_experiment_run,
    record_gate_event,
    record_seed_eval_result,
    round_progress,
)
from evocast.state.domain_store import load_round_record


def _contract(source: Path) -> BuildContract:
    manifest = source_manifest(source)
    return BuildContract(
        research_id="Research001",
        base_snapshot_id=manifest["snapshot_id"],
        semantic_goal="Exercise one auditable build failure.",
        hypothesis="The failure matrix preserves one terminal Research round.",
        target_model="FixtureModel",
        base_source_ref={
            "kind": "source_snapshot",
            "base_snapshot_id": manifest["snapshot_id"],
            "source_checkout": str(source),
        },
        allowed_edit_files=["model.py"],
    )


@pytest.mark.parametrize(
    ("case", "stage", "error_type", "close_status", "expected_kind"),
    [
        (
            "build",
            "variant_write",
            "syntax_error",
            "implementation_rejected",
            FAILURE_KIND_IMPLEMENTATION,
        ),
        (
            "import",
            "variant_write",
            "import_error",
            "implementation_rejected",
            FAILURE_KIND_IMPLEMENTATION,
        ),
        (
            "shape",
            "smoke",
            "shape_mismatch",
            "runtime_probe_failed",
            FAILURE_KIND_VALIDATION,
        ),
        (
            "probe",
            "runtime_probe",
            "activation_failure",
            "runtime_probe_failed",
            FAILURE_KIND_ACTIVATION,
        ),
        (
            "runtime",
            "experiment",
            "runtime_error",
            "experiment_failed",
            FAILURE_KIND_RUNTIME,
        ),
        (
            "training",
            "experiment",
            "metric_failed",
            "experiment_failed",
            FAILURE_KIND_RUNTIME,
        ),
    ],
)
def test_concrete_build_failure_closes_one_auditable_research_round(
    tmp_path: Path,
    case: str,
    stage: str,
    error_type: str,
    close_status: str,
    expected_kind: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("VALUE = 'baseline'\n", encoding="utf-8")
    runtime = tmp_path / "runtime"
    task_id = f"failure_matrix_{case}"
    ledger = ResearchRoundLedger(base_dir=str(runtime), task_id=task_id)
    opened = ledger.open_round(_contract(source))
    evidence_path = ledger.record_artifact(
        opened.round_id,
        f"{case}_failure.json",
        {
            "stage": stage,
            "error_type": error_type,
            "traceback": f"{case} fixture traceback",
        },
    )
    record_experiment_run(
        base_dir=str(runtime),
        task_id=task_id,
        run_id=f"{case}_failure",
        variant_path=None,
        status="failed",
        error_type=error_type,
        error_message=f"{case} fixture failure",
        failure_evidence={
            "artifact_path": str(evidence_path),
            "traceback": f"{case} fixture traceback",
        },
        stage=stage,
    )

    terminal = RoundTerminalService(
        base_dir=str(runtime),
        task_id=task_id,
        ledger=ledger,
    )
    terminal.close_failed_if_open(
        opened,
        close_status,
        f"{case}_failure",
        {"artifact_path": str(evidence_path)},
    )

    record = load_round_record(
        str(runtime),
        task_id,
        opened.research_id,
    )
    progress = round_progress(str(runtime), task_id)
    assert record["status"] == close_status
    assert record["phase"] == "closed"
    assert record["failure_kind"] == expected_kind
    assert record["runs"][-1]["failure_evidence"]["artifact_path"] == str(evidence_path)
    assert evidence_path.is_file()
    assert progress["terminal_rounds"] == 1
    assert progress["research_open_rounds"] == 0


def test_build_orchestrator_retains_terminal_ownership_across_metric_repairs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("VALUE = 'baseline'\n", encoding="utf-8")
    runtime = tmp_path / "runtime"
    task_id = "deferred_metric_terminal"
    ledger = ResearchRoundLedger(base_dir=str(runtime), task_id=task_id)
    opened = ledger.open_round(_contract(source))

    for attempt in range(1, 6):
        record = record_experiment_run(
            base_dir=str(runtime),
            task_id=task_id,
            run_id=f"probe_{attempt}",
            variant_path=None,
            status="failed",
            error_type="invalid_mechanism_contract",
            error_message="probe shape mismatch",
            failure_evidence={"traceback": f"attempt {attempt} traceback"},
            stage="runtime_contract_probe",
            allow_terminal_transition=False,
        )
        assert record is not None
        assert record["status"] == "repair_needed"

    terminal = RoundTerminalService(
        base_dir=str(runtime),
        task_id=task_id,
        ledger=ledger,
    )
    terminal.close_failed_if_open(
        opened,
        "experiment_failed",
        "probe shape mismatch",
        {"terminal_reason": "variant_forge_attempts_exhausted"},
    )

    record = load_round_record(str(runtime), task_id, opened.research_id)
    assert record["status"] == "experiment_failed"
    assert record["phase"] == "closed"
    assert record["failure_kind"] == FAILURE_KIND_VALIDATION
    assert len(record["runs"]) == 5
    assert record["runs"][-1]["stage"] == "runtime_contract_probe"
    assert record["close_reason"] == "probe shape mismatch"


def test_valid_metric_rejection_supersedes_prior_probe_failures(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("VALUE = 'baseline'\n", encoding="utf-8")
    runtime = tmp_path / "runtime"
    task_id = "scientific_rejection_after_repairs"
    ledger = ResearchRoundLedger(base_dir=str(runtime), task_id=task_id)
    opened = ledger.open_round(_contract(source))
    variant_path = str(source / "model.py")

    for attempt in range(1, 4):
        record_experiment_run(
            base_dir=str(runtime),
            task_id=task_id,
            run_id=f"probe_{attempt}",
            variant_path=variant_path,
            status="failed",
            error_type="invalid_mechanism_contract",
            stage="runtime_contract_probe",
            allow_terminal_transition=False,
        )
    record_experiment_run(
        base_dir=str(runtime),
        task_id=task_id,
        run_id="metric_success",
        variant_path=variant_path,
        status="success",
        metrics={"mse_norm": 0.9},
        stage="experiment",
    )
    record_gate_event(
        base_dir=str(runtime),
        task_id=task_id,
        gate_event={
            "run_id": "metric_success",
            "decision": "needs_seed_eval",
            "objective_metric": "mse_norm",
            "candidate_metrics": {"mse_norm": 0.9},
        },
    )
    record_seed_eval_result(
        base_dir=str(runtime),
        task_id=task_id,
        variant_path=variant_path,
        result={
            "node_id": "seed_eval_metric_success",
            "mean": 1.1,
            "std": 0.1,
            "successful_seeds": 3,
            "significance_decision": {"decision": "reject"},
            "promoted_to_current_best": False,
        },
    )

    record = load_round_record(str(runtime), task_id, opened.research_id)
    progress = round_progress(str(runtime), task_id)
    assert record["status"] == "rejected"
    assert record["phase"] == "closed"
    assert record["gate_decision"] == "reject"
    assert record["failure_kind"] == "scientific_failure"
    assert progress["valid_trial_rounds"] == 1
    assert progress["scientific_rejection_rounds"] == 1
    assert progress["runtime_failure_rounds"] == 0
    assert progress["metric_production_rate"] == 1.0
