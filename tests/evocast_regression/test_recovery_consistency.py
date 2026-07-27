from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evocast.services.resume_service import (
    ResumeDependencies,
    ResumeRequest,
    ResumeService,
)
from evocast.services.task_finalization import (
    FinalizationDependencies,
    FinalizationRequest,
    TaskFinalizationService,
)
from evocast.state.domain_store import save_task_config
from evocast.state.runtime.store import (
    load_runtime_state,
    record_runtime_event,
    runtime_events_path,
    sync_task_stage,
)


def _events(base_dir: str, task_id: str) -> list[dict]:
    path = Path(runtime_events_path(base_dir, task_id))
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_runtime_terminal_event_is_exactly_once_across_concurrent_replay(
    tmp_path: Path,
) -> None:
    base_dir = str(tmp_path)
    task_id = "concurrent_terminal"
    payload = {
        "status": "completed",
        "reason": "max_rounds_exhausted",
        "idempotency_key": "task_terminal:max_rounds_exhausted",
    }

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _index: record_runtime_event(
                    base_dir,
                    task_id,
                    "task_terminal",
                    payload,
                ),
                range(16),
            )
        )

    assert results.count(True) == 1
    assert results.count(False) == 15
    terminal = [
        event for event in _events(base_dir, task_id)
        if event.get("event_type") == "task_terminal"
    ]
    assert len(terminal) == 1


def test_task_finalization_replay_keeps_one_terminal_event(tmp_path: Path) -> None:
    base_dir = str(tmp_path)
    task_id = "finalization_replay"
    save_task_config(
        base_dir,
        task_id,
        {"task_id": task_id, "objective_metric": "mse_norm", "max_rounds": 1},
    )
    service = TaskFinalizationService(
        base_dir=base_dir,
        dependencies=FinalizationDependencies(
            sync_task_stage=sync_task_stage,
            record_runtime_event=record_runtime_event,
            build_dashboard=lambda *_args, **_kwargs: None,
            generate_report=lambda **_kwargs: {"status": "skipped"},
            open_report=lambda _path: {},
        ),
    )
    request = FinalizationRequest(task_id=task_id, review_report=False)
    progress = {
        "terminal_rounds": 1,
        "completed_rounds": 1,
        "valid_trial_rounds": 0,
        "research_open_rounds": 0,
    }

    first = service.finalize_research(request, progress=progress, max_rounds=1)
    second = service.finalize_research(request, progress=progress, max_rounds=1)

    assert first["status"] == second["status"] == "completed_with_failed_rounds"
    assert load_runtime_state(base_dir, task_id).task_status == "completed_with_failed_rounds"
    terminal = [
        event for event in _events(base_dir, task_id)
        if event.get("event_type") == "task_terminal"
    ]
    assert len(terminal) == 1
    assert terminal[0]["payload"]["reason"] == "max_rounds_exhausted"


def test_finalization_does_not_treat_consumed_failed_round_as_success() -> None:
    stages: list[dict] = []
    service = TaskFinalizationService(
        base_dir="runtime",
        dependencies=FinalizationDependencies(
            sync_task_stage=lambda _base, _task, **kwargs: stages.append(kwargs),
            record_runtime_event=lambda *_args: None,
            build_dashboard=lambda *_args, **_kwargs: None,
            generate_report=lambda **_kwargs: {"status": "skipped"},
            open_report=lambda _path: {},
        ),
    )

    result = service.finalize_research(
        FinalizationRequest(task_id="failed_round", review_report=False),
        progress={
            "terminal_rounds": 1,
            "completed_rounds": 1,
            "valid_trial_rounds": 0,
            "research_open_rounds": 0,
        },
        max_rounds=1,
    )

    assert result["status"] == "completed_with_failed_rounds"
    assert stages[0]["extra"]["research"]["completed_rounds"] == 0
    assert stages[0]["extra"]["research"]["failed_rounds"] == 1


def test_keyed_terminal_replay_recognizes_legacy_unkeyed_event(tmp_path: Path) -> None:
    base_dir = str(tmp_path)
    task_id = "legacy_terminal"
    assert record_runtime_event(
        base_dir,
        task_id,
        "task_terminal",
        {"status": "completed", "reason": "max_rounds_exhausted"},
    )

    appended = record_runtime_event(
        base_dir,
        task_id,
        "task_terminal",
        {
            "status": "completed",
            "reason": "max_rounds_exhausted",
            "idempotency_key": "task_terminal:max_rounds_exhausted",
        },
    )

    assert appended is False
    terminal = [
        event for event in _events(base_dir, task_id)
        if event.get("event_type") == "task_terminal"
    ]
    assert len(terminal) == 1


def test_terminal_status_reconciliation_appends_correction_not_duplicate(
    tmp_path: Path,
) -> None:
    base_dir = str(tmp_path)
    task_id = "terminal_correction"
    record_runtime_event(
        base_dir,
        task_id,
        "task_terminal",
        {"status": "completed", "reason": "max_rounds_exhausted"},
    )
    corrected = {
        "status": "completed_with_failed_rounds",
        "reason": "max_rounds_exhausted",
        "idempotency_key": "task_terminal:max_rounds_exhausted",
    }

    assert record_runtime_event(
        base_dir, task_id, "task_terminal", corrected
    ) is False
    assert record_runtime_event(
        base_dir, task_id, "task_terminal", corrected
    ) is False

    events = _events(base_dir, task_id)
    assert len([e for e in events if e["event_type"] == "task_terminal"]) == 1
    corrections = [
        event for event in events
        if event["event_type"] == "task_terminal_correction"
    ]
    assert len(corrections) == 1
    assert corrections[0]["payload"]["previous_status"] == "completed"
    assert corrections[0]["payload"]["status"] == "completed_with_failed_rounds"


def test_zero_round_finalization_records_one_terminal_event(tmp_path: Path) -> None:
    base_dir = str(tmp_path)
    task_id = "zero_round_terminal"
    save_task_config(
        base_dir,
        task_id,
        {"task_id": task_id, "objective_metric": "mse_norm", "max_rounds": 0},
    )
    service = TaskFinalizationService(
        base_dir=base_dir,
        dependencies=FinalizationDependencies(
            sync_task_stage=sync_task_stage,
            record_runtime_event=record_runtime_event,
            build_dashboard=lambda *_args, **_kwargs: None,
            generate_report=lambda **_kwargs: {"status": "skipped"},
            open_report=lambda _path: {},
        ),
    )
    request = FinalizationRequest(task_id=task_id, review_report=False)

    service.finalize_zero_rounds(request)
    service.finalize_zero_rounds(request)

    terminal = [
        event for event in _events(base_dir, task_id)
        if event.get("event_type") == "task_terminal"
    ]
    assert len(terminal) == 1
    assert terminal[0]["payload"]["reason"] == "max_rounds_zero_baseline_diagnosis_complete"


def _resume_dependencies(**overrides):
    calls = overrides.pop("calls")
    values = dict(
        base_dir="runtime",
        load_task_config=lambda _base, _task: {
            "task_id": "task",
            "objective_metric": "mse_norm",
            "max_rounds": 1,
        },
        state_exists=lambda _base, _task: True,
        list_rounds=lambda _base, _task: [
            {
                "research_id": "Research001",
                "round_scope": "research",
                "status": "completed",
            }
        ],
        current_round=lambda _base, _task: None,
        counts_as_research=lambda record: record.get("round_scope") == "research",
        phase_build="build",
        gate_resolution_decisions={"needs_review", "needs_seed_eval"},
        load_contract=lambda *_args: {},
        record_seed_eval_failure=lambda **kwargs: calls.append(("seed_failure", kwargs)),
        close_round=lambda **kwargs: calls.append(("close_round", kwargs)),
        run_agent=lambda *_args: calls.append(("run_agent", _args)),
        round_progress=lambda _base, _task: {
            "terminal_rounds": 1,
            "completed_rounds": 1,
            "research_open_rounds": 0,
        },
        run_research_loop=lambda _request: (_ for _ in ()).throw(
            AssertionError("terminal task must not restart the research loop")
        ),
        load_diagnosis=lambda *_args: {},
        build_dashboard=lambda *_args: calls.append(("dashboard", _args)),
        finalize_research=lambda *args: calls.append(("finalize", args))
        or {"status": "completed"},
        record_agent_failure=lambda *args: calls.append(("agent_failure", args)),
    )
    values.update(overrides)
    return ResumeDependencies(**values)


def test_resume_of_terminal_rounds_reconciles_canonical_task_terminal() -> None:
    calls: list[tuple] = []
    service = ResumeService(_resume_dependencies(calls=calls))

    result = service.resume(
        ResumeRequest(
            task_id="task",
            api_config="providers/minimax.yaml",
            language="zh",
            idea_planner="none",
            agent_ablation="none",
            review_report=False,
        )
    )

    assert result["status"] == "already_completed"
    assert result["completion"]["status"] == "completed"
    assert [name for name, _payload in calls] == ["finalize"]


def test_interrupted_seed_eval_uses_single_failure_boundary() -> None:
    calls: list[tuple] = []
    open_round = {
        "round_id": 1,
        "research_id": "Research001",
        "round_scope": "research",
        "status": "metric_completed",
        "phase": "experiment",
        "gate_decision": "needs_seed_eval",
        "gate_event": {"run_id": "run_001"},
        "variant_path": "variant.py",
    }
    service = ResumeService(
        _resume_dependencies(
            calls=calls,
            list_rounds=lambda *_args: [open_round],
            current_round=lambda *_args: open_round,
            record_seed_eval_failure=lambda **kwargs: calls.append(
                ("seed_failure", kwargs)
            )
            or {"status": "experiment_failed"},
        )
    )

    recovered = service.recover_open_round(
        "task",
        {"api_config": "providers/minimax.yaml"},
        "providers/minimax.yaml",
    )

    assert recovered is False
    assert [name for name, _payload in calls] == ["seed_failure"]
    assert calls[0][1]["result"]["node_id"] == "run_001"
    assert calls[0][1]["result"]["error_type"] == "InterruptedSeedEval"


def test_resume_failure_is_finalized_before_exception_is_re_raised() -> None:
    calls: list[tuple] = []
    open_round = {
        "round_id": 1,
        "research_id": "Research001",
        "round_scope": "research",
        "status": "round_started",
        "phase": "build",
    }
    service = ResumeService(
        _resume_dependencies(
            calls=calls,
            list_rounds=lambda *_args: [open_round],
            current_round=lambda *_args: open_round,
            load_contract=lambda *_args: {"research_id": "Research001"},
            run_agent=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("resume build failed")
            ),
            round_progress=lambda *_args: {
                "terminal_rounds": 0,
                "completed_rounds": 0,
                "research_open_rounds": 1,
            },
        )
    )

    try:
        service.resume(
            ResumeRequest(
                task_id="task",
                api_config="providers/minimax.yaml",
                language="zh",
                idea_planner="none",
                agent_ablation="none",
                review_report=False,
            )
        )
    except RuntimeError as exc:
        assert str(exc) == "resume build failed"
    else:
        raise AssertionError("resume failure must be re-raised")

    assert [name for name, _payload in calls] == ["agent_failure"]
    assert isinstance(calls[0][1][1], RuntimeError)
