from __future__ import annotations

from pathlib import Path

from evocast.build.contract import BuildContract
from evocast.build.ledger import ResearchRoundLedger
from evocast.build.round_terminal import RoundTerminalService
from evocast.build.source_snapshot import source_manifest
from evocast.evaluation.decision_kernel import EvaluationDecisionKernel
from evocast.harness import rounds
from evocast.harness.session import AgentSession
from evocast.state.domain_store import load_round_record
from evocast.state.runtime.store import sync_current_best


def _opened_round(
    tmp_path: Path,
    task_id: str,
) -> tuple[ResearchRoundLedger, object, str]:
    source = tmp_path / "source"
    source.mkdir()
    variant = source / "model.py"
    variant.write_text("VALUE = 'candidate'\n", encoding="utf-8")
    manifest = source_manifest(source)
    contract = BuildContract(
        research_id="Research001",
        base_snapshot_id=manifest["snapshot_id"],
        semantic_goal="Exercise the evaluation decision boundary.",
        hypothesis="Retries are exactly once.",
        target_model="FixtureModel",
        base_source_ref={
            "kind": "source_snapshot",
            "base_snapshot_id": manifest["snapshot_id"],
            "source_checkout": str(source),
        },
        allowed_edit_files=["model.py"],
    )
    ledger = ResearchRoundLedger(
        base_dir=str(tmp_path),
        task_id=task_id,
    )
    return ledger, ledger.open_round(contract), str(variant)


def test_attempt_replay_is_exactly_once_before_and_after_terminal(
    tmp_path: Path,
) -> None:
    task_id = "attempt_replay"
    ledger, opened, variant_path = _opened_round(tmp_path, task_id)
    kernel = EvaluationDecisionKernel(
        base_dir=str(tmp_path),
        task_id=task_id,
    )
    payload = {
        "run_id": "run_probe_001",
        "variant_path": variant_path,
        "status": "failed",
        "error_type": "invalid_mechanism_contract",
        "error_message": "shape mismatch",
        "stage": "runtime_contract_probe",
        "orchestrator_owns_terminal": True,
    }

    first = kernel.record_attempt(**payload)
    replay_while_open = kernel.record_attempt(**payload)
    assert first is not None
    assert replay_while_open is not None
    assert len(replay_while_open["runs"]) == 1

    RoundTerminalService(
        base_dir=str(tmp_path),
        task_id=task_id,
        ledger=ledger,
    ).close_failed_if_open(
        opened,
        "runtime_probe_failed",
        "shape mismatch",
    )
    replay_after_close = kernel.record_attempt(**payload)
    assert replay_after_close is not None
    assert replay_after_close["status"] == "runtime_probe_failed"
    assert len(replay_after_close["runs"]) == 1
    assert rounds.round_progress(str(tmp_path), task_id)["terminal_rounds"] == 1


def test_seed_decision_replay_does_not_close_round_twice(
    tmp_path: Path,
) -> None:
    task_id = "seed_decision_replay"
    _, opened, variant_path = _opened_round(tmp_path, task_id)
    kernel = EvaluationDecisionKernel(
        base_dir=str(tmp_path),
        task_id=task_id,
    )
    kernel.record_attempt(
        run_id="run_metric_001",
        variant_path=variant_path,
        status="success",
        metrics={"mse_norm": 0.9},
        stage="experiment",
        orchestrator_owns_terminal=True,
    )
    rounds.record_gate_event(
        base_dir=str(tmp_path),
        task_id=task_id,
        gate_event={
            "run_id": "run_metric_001",
            "decision": "needs_seed_eval",
            "objective_metric": "mse_norm",
            "candidate_metrics": {"mse_norm": 0.9},
        },
    )
    result = {
        "node_id": "seed_eval_run_metric_001",
        "mean": 1.1,
        "std": 0.1,
        "successful_seeds": 3,
        "significance_decision": {"decision": "reject"},
        "promoted_to_current_best": False,
    }

    first = kernel.record_seed_result(
        variant_path=variant_path,
        result=result,
    )
    replay = kernel.record_seed_result(
        variant_path=variant_path,
        result=result,
    )
    assert first is not None
    assert replay is not None
    assert first["closed_at"] == replay["closed_at"]

    record = load_round_record(
        str(tmp_path),
        task_id,
        opened.research_id,
    )
    progress = rounds.round_progress(str(tmp_path), task_id)
    assert record["status"] == "rejected"
    assert record["phase"] == "closed"
    assert record["failure_kind"] == "scientific_failure"
    assert record["gate_decision"] == "reject"
    assert progress["terminal_rounds"] == 1
    assert progress["valid_trial_rounds"] == 1
    assert progress["scientific_rejection_rounds"] == 1


def test_gate_replay_does_not_duplicate_canonical_or_jsonl_event(
    tmp_path: Path,
) -> None:
    task_id = "gate_replay"
    _, opened, variant_path = _opened_round(tmp_path, task_id)
    session = AgentSession(
        task_id=task_id,
        base_dir=str(tmp_path),
        client=None,  # type: ignore[arg-type]
    )
    session.ensure_dirs()
    sync_current_best(
        str(tmp_path),
        task_id,
        {
            "candidate_id": "baseline",
            "display_name": "Fixture baseline",
            "objective_metric": "mse_norm",
            "metrics": {"mse_norm": 1.0},
        },
    )
    kernel = EvaluationDecisionKernel.from_session(session)
    kernel.record_attempt(
        run_id="run_metric_gate",
        variant_path=variant_path,
        status="success",
        metrics={"mse_norm": 0.9},
        stage="experiment",
        orchestrator_owns_terminal=True,
    )
    gate_args = {
        "session": session,
        "run_id": "run_metric_gate",
        "variant_path": variant_path,
        "candidate_id": "candidate",
        "candidate_kind": "variant",
        "candidate_name": "Fixture candidate",
        "metrics": {"mse_norm": 0.9},
        "objective_metric": "mse_norm",
        "evaluation_stage": "build_mode",
        "smoke": False,
    }

    first = kernel.record_success_gate(**gate_args)
    replay = kernel.record_success_gate(**gate_args)
    assert first["timestamp"] == replay["timestamp"]

    record = load_round_record(
        str(tmp_path),
        task_id,
        opened.research_id,
    )
    events = (
        session.knowledge_dir / "gate_events.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert record["gate_event"]["run_id"] == "run_metric_gate"
    assert len(events) == 1


def test_seed_failure_replay_is_exactly_once(
    tmp_path: Path,
) -> None:
    task_id = "seed_failure_replay"
    _, opened, variant_path = _opened_round(tmp_path, task_id)
    kernel = EvaluationDecisionKernel(
        base_dir=str(tmp_path),
        task_id=task_id,
    )
    kernel.record_attempt(
        run_id="run_metric_before_seed_failure",
        variant_path=variant_path,
        status="success",
        metrics={"mse_norm": 0.9},
        stage="experiment",
        orchestrator_owns_terminal=True,
    )
    rounds.record_gate_event(
        base_dir=str(tmp_path),
        task_id=task_id,
        gate_event={
            "run_id": "run_metric_before_seed_failure",
            "decision": "needs_seed_eval",
            "objective_metric": "mse_norm",
            "candidate_metrics": {"mse_norm": 0.9},
        },
    )
    failure = {
        "node_id": "seed_eval_failed_001",
        "error_type": "RuntimeError",
        "error_message": "seed worker crashed",
    }

    first = kernel.record_seed_failure(
        variant_path=variant_path,
        result=failure,
    )
    replay = kernel.record_seed_failure(
        variant_path=variant_path,
        result=failure,
    )
    assert first is not None
    assert replay is not None
    assert first["closed_at"] == replay["closed_at"]

    record = load_round_record(
        str(tmp_path),
        task_id,
        opened.research_id,
    )
    assert record["status"] == "experiment_failed"
    assert record["phase"] == "closed"
    assert record["seed_eval"]["status"] == "failed"
    assert rounds.round_progress(str(tmp_path), task_id)["terminal_rounds"] == 1
