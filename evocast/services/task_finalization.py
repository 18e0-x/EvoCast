"""Canonical task completion, failure recording, dashboard, and report flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class FinalizationRequest:
    task_id: str
    review_report: bool = True
    open_report: bool = False
    strict_report: bool = False


@dataclass
class FinalizationDependencies:
    sync_task_stage: Callable[..., Any]
    record_runtime_event: Callable[..., Any]
    build_dashboard: Callable[..., Any]
    generate_report: Callable[..., Dict[str, Any]]
    open_report: Callable[[str], Dict[str, Any]]


class TaskFinalizationService:
    def __init__(self, *, base_dir: str, dependencies: FinalizationDependencies) -> None:
        self.base_dir = base_dir
        self.d = dependencies

    def _report(self, request: FinalizationRequest) -> Dict[str, Any]:
        if not request.review_report:
            return {"status": "skipped"}
        report = self.d.generate_report(task_id=request.task_id, base_dir=self.base_dir)
        if report.get("status") != "ok":
            if request.strict_report:
                raise RuntimeError(f"review_report_failed: {report}")
            return report
        if request.open_report and report.get("html_path"):
            report = {**report, "open_result": self.d.open_report(str(report["html_path"]))}
        return report

    def finalize_zero_rounds(self, request: FinalizationRequest) -> Dict[str, Any]:
        terminal_payload = {
            "status": "completed",
            "reason": "max_rounds_zero_baseline_diagnosis_complete",
            "idempotency_key": "task_terminal:max_rounds_zero",
        }
        self.d.sync_task_stage(
            self.base_dir,
            request.task_id,
            stage="completed",
            status="completed",
            extra={
                "research": {
                    "finalize_reason": "max_rounds_zero_baseline_diagnosis_complete",
                    "next_round": 1,
                    "valid_trial_rounds": 0,
                    "terminal_rounds": 0,
                }
            },
        )
        self.d.record_runtime_event(
            self.base_dir,
            request.task_id,
            "task_terminal",
            terminal_payload,
        )
        self.d.build_dashboard(self.base_dir, request.task_id, persist=True)
        return {"status": "completed", "report": self._report(request)}

    def finalize_research(
        self,
        request: FinalizationRequest,
        *,
        progress: Dict[str, Any],
        max_rounds: int,
    ) -> Dict[str, Any]:
        if (
            int(progress.get("research_open_rounds") or 0) > 0
            or int(progress.get("terminal_rounds") or 0) < int(max_rounds)
        ):
            self.d.build_dashboard(self.base_dir, request.task_id, persist=True)
            return {"status": "not_terminal", "round_progress": progress}
        # ``completed_rounds`` is the historical budget-consumption counter:
        # engineering-failed terminal rounds are included. Scientific
        # execution success is represented by ``valid_trial_rounds``.
        successful = int(
            progress.get("valid_trial_rounds")
            if "valid_trial_rounds" in progress
            else progress.get("completed_rounds")
            or 0
        )
        failed = max(0, int(progress.get("terminal_rounds") or 0) - successful)
        status = "completed" if failed == 0 else "completed_with_failed_rounds"
        self.d.sync_task_stage(
            self.base_dir,
            request.task_id,
            stage="completed",
            status=status,
            extra={
                "research": {
                    "finalize_reason": "max_rounds_exhausted",
                    "completed_rounds": successful,
                    "valid_trial_rounds": successful,
                    "failed_rounds": failed,
                    "terminal_rounds": int(progress.get("terminal_rounds") or 0),
                }
            },
        )
        self.d.record_runtime_event(
            self.base_dir,
            request.task_id,
            "task_terminal",
            {
                "status": status,
                "reason": "max_rounds_exhausted",
                "round_progress": progress,
                "idempotency_key": "task_terminal:max_rounds_exhausted",
            },
        )
        self.d.build_dashboard(self.base_dir, request.task_id, persist=True)
        return {"status": status, "round_progress": progress, "report": self._report(request)}

    def record_agent_failure(
        self,
        request: FinalizationRequest,
        *,
        error: Exception,
        progress: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "status": "failed",
            "reason": "build_orchestrator_exception",
            "error_type": type(error).__name__,
            "message": str(error),
            "stage": "agent_launch",
            "round_progress": progress,
            "idempotency_key": "task_terminal:build_orchestrator_exception",
        }
        self.d.sync_task_stage(
            self.base_dir,
            request.task_id,
            stage="agent_launch",
            status="failed",
            extra={"research": payload},
        )
        self.d.record_runtime_event(self.base_dir, request.task_id, "task_terminal", payload)
        return {**payload, "report": self._report(request)}
