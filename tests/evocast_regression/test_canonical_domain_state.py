from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evocast.build.contract import BuildContract
from evocast.harness.rounds import close_current_round, record_phase_transition, start_round
from evocast.state.domain_store import (
    domain_state_path,
    load_build_contract,
    load_domain_state,
    load_round_record,
    load_task_config,
    save_build_contract,
    save_task_config,
    mutate_domain_state,
)
from evocast.state.runtime.store import mark_task_interrupted, sync_task_stage, update_runtime_state
from evocast.domain.state_model import RoundRecord
from evocast.policy.experiment_policy import task_build_mode, training_budget_for_task


def _create_task(tmp_path: Path, task_id: str = "canonical_state") -> tuple[str, str]:
    base_dir = str(tmp_path / "runtime")
    save_task_config(
        base_dir,
        task_id,
        {
            "task_id": task_id,
            "objective_metric": "mse_norm",
            "max_rounds": 2,
            "build_mode": True,
            "research_intent": "Improve Informer while preserving the evaluation protocol.",
        },
    )
    return base_dir, task_id


def _contract() -> BuildContract:
    return BuildContract(
        research_id="Research001",
        base_snapshot_id="baseline-snapshot",
        semantic_goal="Improve the active Informer.",
        hypothesis="A focused residual path improves mse_norm.",
        target_model="Informer",
        allowed_edit_files=["models/informer.py"],
        research_intent="Improve Informer while preserving the evaluation protocol.",
    )


def test_creation_writes_one_canonical_task_definition(tmp_path: Path) -> None:
    base_dir, task_id = _create_task(tmp_path)

    assert domain_state_path(base_dir, task_id).is_file()
    assert not (Path(base_dir) / "task_knowledge" / task_id / "task_config.json").exists()
    assert load_task_config(base_dir, task_id)["research_intent"] == "Improve Informer while preserving the evaluation protocol."


def test_execution_records_contract_and_round_in_canonical_state(tmp_path: Path) -> None:
    base_dir, task_id = _create_task(tmp_path)
    contract = _contract()
    save_build_contract(base_dir, task_id, contract.research_id, contract.to_dict())
    record = start_round(
        base_dir=base_dir,
        task_id=task_id,
        fit_point="Informer",
        hypothesis=contract.hypothesis,
        evidence_source={"kind": "test"},
        research_intent=contract.research_intent,
    )
    record_phase_transition(base_dir=base_dir, task_id=task_id, phase="build")

    assert load_build_contract(base_dir, task_id, "Research001") == contract.to_dict()
    assert load_round_record(base_dir, task_id, "Research001")["phase"] == "build"
    assert record["research_id"] == "Research001"


def test_interrupt_then_resume_retains_intent_and_open_round(tmp_path: Path) -> None:
    base_dir, task_id = _create_task(tmp_path)
    sync_task_stage(base_dir, task_id, stage="research_loop", status="running")
    start_round(base_dir=base_dir, task_id=task_id, fit_point="Informer", hypothesis="h", evidence_source={"kind": "test"})
    mark_task_interrupted(base_dir, task_id)

    restored = load_domain_state(base_dir, task_id)
    assert restored["task"]["research_intent"] == "Improve Informer while preserving the evaluation protocol."
    assert restored["runtime"]["task_status"] == "interrupted"
    assert restored["rounds"]["Research001"]["status"] == "round_started"


def test_failure_is_a_persisted_research_round(tmp_path: Path) -> None:
    base_dir, task_id = _create_task(tmp_path)
    start_round(base_dir=base_dir, task_id=task_id, fit_point="Informer", hypothesis="h", evidence_source={"kind": "test"})
    close_current_round(base_dir=base_dir, task_id=task_id, status="implementation_rejected", reason="compile failed")

    record = load_round_record(base_dir, task_id, "Research001")
    assert record["status"] == "implementation_rejected"
    assert record["decision"]["reason"] == "compile failed"


def test_completion_has_authoritative_evaluation_and_decision(tmp_path: Path) -> None:
    base_dir, task_id = _create_task(tmp_path)
    start_round(base_dir=base_dir, task_id=task_id, fit_point="Informer", hypothesis="h", evidence_source={"kind": "test"})
    close_current_round(
        base_dir=base_dir,
        task_id=task_id,
        status="completed",
        reason="verified improvement",
        extra={"metric_result": {"status": "ACCEPTED", "metrics": {"mse_norm": 0.41}}},
    )

    record = load_round_record(base_dir, task_id, "Research001")
    assert record["evaluation"]["metrics"] == {"mse_norm": 0.41}
    assert record["decision"]["reason"] == "verified improvement"


def test_legacy_state_is_migrated_without_becoming_a_write_target(tmp_path: Path) -> None:
    base_dir, task_id = str(tmp_path / "runtime"), "legacy_state"
    root = Path(base_dir) / "task_knowledge" / task_id
    root.mkdir(parents=True)
    (root / "task_config.json").write_text(json.dumps({"task_id": task_id, "research_intent": "legacy intent"}), encoding="utf-8")
    legacy_round = root / "rounds" / "Research001" / "round.json"
    legacy_round.parent.mkdir(parents=True)
    legacy_round.write_text(json.dumps({"round_id": 1, "research_id": "Research001", "status": "completed"}), encoding="utf-8")

    migrated = load_domain_state(base_dir, task_id)

    assert migrated["task"]["research_intent"] == "legacy intent"
    assert migrated["rounds"]["Research001"]["status"] == "completed"
    assert domain_state_path(base_dir, task_id).is_file()


def test_canonical_round_trip_preserves_explicit_evaluation_and_decision() -> None:
    payload = {
        "round_id": 1,
        "research_id": "Research001",
        "evaluation": {"status": "completed", "metrics": {"mse_norm": 0.25}, "evidence_paths": ["metric.json"]},
        "decision": {"status": "accept", "reason": "verified", "next_required": "publish"},
    }
    evaluation = RoundRecord.from_dict(payload).to_dict()["evaluation"]
    assert evaluation["status"] == "completed"
    assert evaluation["metrics"] == {"mse_norm": 0.25}
    assert evaluation["evidence_paths"] == ["metric.json"]
    assert RoundRecord.from_dict(payload).to_dict()["decision"] == payload["decision"]


def test_canonical_state_never_reimports_legacy_records_after_migration(tmp_path: Path) -> None:
    base_dir, task_id = str(tmp_path / "runtime"), "one_time_migration"
    root = Path(base_dir) / "task_knowledge" / task_id / "rounds" / "Research001"
    root.mkdir(parents=True)
    (root / "round.json").write_text(json.dumps({"round_id": 1, "research_id": "Research001", "status": "completed"}), encoding="utf-8")
    load_domain_state(base_dir, task_id)
    mutate_domain_state(base_dir, task_id, lambda state: state["rounds"].pop("Research001"))
    assert "Research001" not in load_domain_state(base_dir, task_id)["rounds"]


def test_fresh_canonical_build_mode_uses_smoke_budget(tmp_path: Path) -> None:
    base_dir, task_id = _create_task(tmp_path)
    assert task_build_mode(base_dir, task_id) is True
    assert training_budget_for_task(base_dir=base_dir, task_id=task_id) == "smoke_test"


def test_runtime_mutations_do_not_overwrite_each_other(tmp_path: Path) -> None:
    base_dir, task_id = _create_task(tmp_path)

    def _write(index: int) -> None:
        update_runtime_state(
            base_dir,
            task_id,
            lambda state: state.research.__setitem__(f"writer_{index}", index),
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(_write, range(12)))

    research = load_domain_state(base_dir, task_id)["runtime"]["research"]
    assert {research[f"writer_{index}"] for index in range(12)} == set(range(12))
