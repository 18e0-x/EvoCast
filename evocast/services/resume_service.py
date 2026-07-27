"""Non-interactive canonical-task resume workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from evocast.services.research_loop import ResearchLoopRequest


@dataclass(frozen=True)
class ResumeRequest:
    task_id: str
    api_config: str
    language: str
    idea_planner: str
    agent_ablation: str
    review_report: bool


@dataclass
class ResumeDependencies:
    base_dir: str
    load_task_config: Callable[[str, str], Dict[str, Any]]
    state_exists: Callable[[str, str], bool]
    list_rounds: Callable[[str, str], list[Dict[str, Any]]]
    current_round: Callable[[str, str], Dict[str, Any] | None]
    counts_as_research: Callable[[Dict[str, Any]], bool]
    phase_build: str
    gate_resolution_decisions: set[str]
    load_contract: Callable[[str, str, str], Dict[str, Any]]
    record_seed_eval_failure: Callable[..., Any]
    close_round: Callable[..., Any]
    run_agent: Callable[[str, str, str], Any]
    round_progress: Callable[[str, str], Dict[str, Any]]
    run_research_loop: Callable[[ResearchLoopRequest], Dict[str, Any]]
    load_diagnosis: Callable[[str, str], Dict[str, Any]]
    build_dashboard: Callable[[str, str], Any]
    finalize_research: Callable[[str, Dict[str, Any], int, bool], Any]
    record_agent_failure: Callable[[str, Exception, Dict[str, Any], bool], Any]


class ResumeService:
    """Resume only durable research state; UI and CLI stay at the edge."""

    def __init__(self, dependencies: ResumeDependencies) -> None:
        self.d = dependencies

    def load_task_config(self, task_id: str) -> Dict[str, Any]:
        payload = self.d.load_task_config(self.d.base_dir, task_id)
        if not self.d.state_exists(self.d.base_dir, task_id):
            raise FileNotFoundError(f"Cannot resume {task_id}: canonical task state is missing.")
        return payload

    def _close_interrupted_round(self, task_id: str, record: Dict[str, Any]) -> None:
        research_id = str(record.get("research_id") or "").strip()
        phase = str(record.get("phase") or "unknown").strip() or "unknown"
        if (
            record.get("gate_decision") in self.d.gate_resolution_decisions
            and dict(record.get("seed_eval") or {}).get("status") not in {"completed", "failed"}
        ):
            saved = self.d.record_seed_eval_failure(
                base_dir=self.d.base_dir,
                task_id=task_id,
                variant_path=record.get("variant_path") or None,
                result={
                    "node_id": str((dict(record.get("gate_event") or {})).get("run_id") or ""),
                    "error_type": "InterruptedSeedEval",
                    "error_message": f"resume_process_interrupted_at_{phase}",
                    "summary": "resume_process_interrupted_before_seed_eval_completed",
                },
            )
            if saved:
                return
        self.d.close_round(
            base_dir=self.d.base_dir, task_id=task_id, research_id=research_id,
            status="infra_failed", reason=f"resume_process_interrupted_at_{phase}",
            extra={"terminal_reason": "resume_process_interrupted", "resume_preflight": True, "interrupted_phase": phase},
        )

    def recover_open_round(self, task_id: str, task_config: Dict[str, Any], fallback_api_config: str) -> bool:
        record = self.d.current_round(self.d.base_dir, task_id)
        if not record:
            return False
        research_id = str(record.get("research_id") or "").strip()
        is_formal = self.d.counts_as_research(record)
        has_formal = any(self.d.counts_as_research(item) for item in self.d.list_rounds(self.d.base_dir, task_id))
        if not is_formal and has_formal:
            self._close_interrupted_round(task_id, record)
            return False
        if is_formal and str(record.get("phase") or "").strip() == self.d.phase_build and self.d.load_contract(self.d.base_dir, task_id, research_id):
            self.d.run_agent(task_id, str(task_config.get("api_config") or fallback_api_config), research_id)
            return True
        self._close_interrupted_round(task_id, record)
        return False

    def resume(self, request: ResumeRequest) -> Dict[str, Any]:
        task_config = self.load_task_config(request.task_id)
        all_records = self.d.list_rounds(self.d.base_dir, request.task_id)
        if not any(self.d.counts_as_research(record) for record in all_records):
            raise RuntimeError("--resume supports tasks after formal Research has started. This task has no committed Research round yet.")
        api_config = str(task_config.get("api_config") or request.api_config)
        objective_metric = str(task_config.get("objective_metric") or "mse_norm")
        max_rounds = int(task_config.get("max_rounds") or 0)
        if max_rounds <= 0:
            raise RuntimeError(f"Cannot resume {request.task_id}: persisted max_rounds must be positive.")
        initial = self.d.round_progress(self.d.base_dir, request.task_id)
        if not self.d.current_round(self.d.base_dir, request.task_id) and int(initial.get("research_open_rounds") or 0) <= 0 and int(initial.get("terminal_rounds") or 0) >= max_rounds:
            completion = self.d.finalize_research(
                request.task_id,
                initial,
                max_rounds,
                request.review_report,
            )
            return {
                "status": "already_completed",
                "task_id": request.task_id,
                "round_progress": initial,
                "completion": completion,
            }
        try:
            self.recover_open_round(request.task_id, task_config, api_config)
            progress = self.d.run_research_loop(
                ResearchLoopRequest(
                    task_id=request.task_id,
                    objective_metric=objective_metric,
                    api_config=api_config,
                    max_rounds=max_rounds,
                    language=str(task_config.get("language") or request.language),
                    idea_planner=request.idea_planner,
                    agent_ablation=str(task_config.get("agent_ablation") or request.agent_ablation),
                    diagnosis=self.d.load_diagnosis(self.d.base_dir, request.task_id),
                )
            )
        except Exception as exc:
            self.d.record_agent_failure(
                request.task_id,
                exc,
                self.d.round_progress(self.d.base_dir, request.task_id),
                request.review_report,
            )
            raise
        self.d.finalize_research(request.task_id, progress, max_rounds, request.review_report)
        return {"status": "resumed", "task_id": request.task_id, "round_progress": progress}
