from __future__ import annotations

from evocast.services.research_loop import (
    ResearchContractRequest,
    ResearchLoopDependencies,
    ResearchLoopRequest,
    ResearchLoopService,
)
from evocast.services.task_finalization import (
    FinalizationDependencies,
    FinalizationRequest,
    TaskFinalizationService,
)


def test_research_loop_uses_terminal_round_budget_for_failed_attempts() -> None:
    progress = {"terminal_rounds": 0, "completed_rounds": 0, "research_open_rounds": 0}
    contracts: list[ResearchContractRequest] = []

    def create_contract(request: ResearchContractRequest) -> str:
        contracts.append(request)
        return f"Research{len(contracts):03d}"

    def run_agent(_task_id: str, _api_config: str, _research_id: str) -> None:
        progress["terminal_rounds"] += 1

    service = ResearchLoopService(
        base_dir="runtime",
        dependencies=ResearchLoopDependencies(
            round_progress=lambda _base, _task: dict(progress),
            create_contract=create_contract,
            run_agent=run_agent,
        ),
    )

    result = service.run(
        ResearchLoopRequest(
            task_id="task",
            objective_metric="mse_norm",
            api_config="providers/minimax.yaml",
            max_rounds=2,
        )
    )

    assert result["terminal_rounds"] == 2
    assert [item.task_id for item in contracts] == ["task", "task"]


def test_research_loop_stops_without_creating_contract_for_open_round() -> None:
    service = ResearchLoopService(
        base_dir="runtime",
        dependencies=ResearchLoopDependencies(
            round_progress=lambda _base, _task: {
                "terminal_rounds": 0,
                "completed_rounds": 0,
                "research_open_rounds": 1,
            },
            create_contract=lambda _request: (_ for _ in ()).throw(
                AssertionError("open round must be recovered before creating a contract")
            ),
            run_agent=lambda *_args: None,
        ),
    )

    result = service.run(
        ResearchLoopRequest(
            task_id="task",
            objective_metric="mse_norm",
            api_config="providers/minimax.yaml",
            max_rounds=1,
        )
    )

    assert result["research_open_rounds"] == 1


def test_finalization_classifies_failed_rounds_and_records_terminal_event() -> None:
    stages: list[dict] = []
    events: list[tuple[str, dict]] = []
    dashboards: list[str] = []
    service = TaskFinalizationService(
        base_dir="runtime",
        dependencies=FinalizationDependencies(
            sync_task_stage=lambda _base, _task, **kwargs: stages.append(kwargs),
            record_runtime_event=lambda _base, _task, event, payload: events.append((event, payload)),
            build_dashboard=lambda _base, task, **_kwargs: dashboards.append(task),
            generate_report=lambda **_kwargs: {"status": "ok", "html_path": "report.html"},
            open_report=lambda path: {"status": "opened", "path": path},
        ),
    )

    result = service.finalize_research(
        FinalizationRequest(task_id="task", open_report=True),
        progress={
            "terminal_rounds": 2,
            "completed_rounds": 1,
            "research_open_rounds": 0,
        },
        max_rounds=2,
    )

    assert result["status"] == "completed_with_failed_rounds"
    assert stages[0]["status"] == "completed_with_failed_rounds"
    assert stages[0]["extra"]["research"]["failed_rounds"] == 1
    assert events[0][0] == "task_terminal"
    assert dashboards == ["task"]
    assert result["report"]["open_result"]["status"] == "opened"


def test_finalization_does_not_complete_before_round_budget_is_terminal() -> None:
    stages: list[dict] = []
    service = TaskFinalizationService(
        base_dir="runtime",
        dependencies=FinalizationDependencies(
            sync_task_stage=lambda _base, _task, **kwargs: stages.append(kwargs),
            record_runtime_event=lambda *_args: None,
            build_dashboard=lambda *_args, **_kwargs: None,
            generate_report=lambda **_kwargs: {"status": "ok"},
            open_report=lambda _path: {},
        ),
    )

    result = service.finalize_research(
        FinalizationRequest(task_id="task"),
        progress={
            "terminal_rounds": 1,
            "completed_rounds": 1,
            "research_open_rounds": 0,
        },
        max_rounds=2,
    )

    assert result["status"] == "not_terminal"
    assert stages == []
