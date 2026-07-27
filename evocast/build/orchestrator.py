"""Facade and outer repair loop for one BuildContract Research round."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evocast.build.attempt_evaluator import BuildAttemptEvaluator, MetricRunner
from evocast.build.backends.base import AgentRunResult, CodingAgentBackend
from evocast.build.candidate_workspace import CandidateWorkspaceService
from evocast.build.contract import BuildContract
from evocast.build.diff_audit import DiffAuditor
from evocast.build.ledger import OpenRound, ResearchRoundLedger
from evocast.build.metadata_writer import ResearchMetadataWriter
from evocast.build.result import BuildDecision, BuildOutcome, VerificationResult
from evocast.build.round_terminal import RoundTerminalService, SeedEvalRunner
from evocast.build.source_snapshot import CandidateSnapshot
from evocast.build.verifier import ExternalVerifier
from evocast.domain.atomic_io import atomic_write_json
from evocast.harness import rounds as round_records
from evocast.state.domain_store import load_build_contract


class ResearchBuildOrchestrator:
    """Keep session/retry/recovery orchestration at a stable public boundary."""

    def __init__(
        self,
        *,
        base_dir: str,
        task_id: str,
        repo_dir: str | Path,
        backend: CodingAgentBackend,
        auditor: DiffAuditor | None = None,
        verifier: ExternalVerifier | None = None,
        metric_runner: MetricRunner | None = None,
        seed_eval_runner: SeedEvalRunner | None = None,
        metadata_writer: ResearchMetadataWriter | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.task_id = task_id
        self.repo_dir = Path(repo_dir).resolve()
        self.backend = backend
        self.ledger = ResearchRoundLedger(base_dir=base_dir, task_id=task_id)
        self.auditor = auditor or DiffAuditor()
        self.verifier = verifier or ExternalVerifier()
        self.metric_runner = metric_runner
        self.seed_eval_runner = seed_eval_runner
        self.metadata_writer = metadata_writer
        self.workspace = CandidateWorkspaceService()
        self.terminal = RoundTerminalService(
            base_dir=base_dir,
            task_id=task_id,
            ledger=self.ledger,
            seed_eval_runner=seed_eval_runner,
        )
        self.attempt_evaluator = BuildAttemptEvaluator(
            base_dir=base_dir,
            task_id=task_id,
            ledger=self.ledger,
            workspace=self.workspace,
            terminal=self.terminal,
            auditor=self.auditor,
            verifier=self.verifier,
            metric_runner=metric_runner,
            metadata_writer=metadata_writer,
        )

    def run(self, contract: BuildContract) -> BuildOutcome:
        stale = self._matching_stale_build_round(contract)
        if stale is not None:
            try:
                return self._recover_stale_build_round(stale, contract)
            except Exception as exc:
                self._close_failed_if_open(
                    stale,
                    "infra_failed",
                    "stale_build_round_recovery_failed",
                    {"error": str(exc), "type": type(exc).__name__},
                    already_closed_event="stale_build_round_recovery_unclosed",
                )
                raise
        opened = self.ledger.open_round(contract)
        contract = BuildContract.from_dict(
            load_build_contract(self.base_dir, self.task_id, opened.research_id)
        )
        direction = dict((contract.metric_protocol or {}).get("research_direction") or {})
        if direction:
            atomic_write_json(
                opened.round_dir / "research_direction.json",
                direction,
                ensure_ascii=False,
            )
            self.ledger.append_event(
                opened.round_id,
                "research_direction_selected",
                {
                    "planner": direction.get("planner"),
                    "terminal_display_title": direction.get("terminal_display_title"),
                    "round_role": direction.get("round_role"),
                    "hypothesis": direction.get("hypothesis"),
                    "artifact_path": direction.get("artifact_path"),
                },
            )
        try:
            return self._run_open_round(opened, contract)
        except Exception as exc:
            self._close_failed_if_open(
                opened,
                "infra_failed",
                "build_orchestrator_exception",
                {"error": str(exc), "type": type(exc).__name__},
                already_closed_event="orchestrator_exception_unclosed",
            )
            raise

    def _run_open_round(
        self,
        opened: OpenRound,
        contract: BuildContract,
    ) -> BuildOutcome:
        agent_workspace = opened.round_dir / "agent_workspace"
        self._create_workspace_from_snapshot(agent_workspace, contract)
        atomic_write_json(
            opened.round_dir / "workspaces.json",
            {"agent_workspace": str(agent_workspace)},
            ensure_ascii=False,
        )
        round_records.record_phase_transition(
            base_dir=self.base_dir,
            task_id=self.task_id,
            phase=round_records.PHASE_BUILD,
        )

        session_id = self.backend.start(agent_workspace, contract)
        message = self._initial_message(contract)
        last_patch: Path | None = None
        last_changed: list[str] = []
        last_feedback: VerificationResult | None = None
        backend_max_attempts = getattr(self.backend, "max_attempts", None)
        max_attempts = (
            max(1, int(contract.repair_budget or 0) + 1)
            if backend_max_attempts is None
            else max(1, int(backend_max_attempts or 1))
        )

        for attempt in range(1, max_attempts + 1):
            attempt_dir = opened.round_dir / f"build_attempt_{attempt:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            turn_result = self.backend.run_turn(
                session_id,
                message,
                contract.timeout_seconds,
            )
            self.ledger.record_agent_turn(
                opened.round_id,
                attempt,
                turn_result.to_dict(),
            )

            if turn_result.status not in {"completed", "success"}:
                last_feedback = VerificationResult(
                    status=BuildDecision.REPAIR_REQUIRED,
                    reason_code="agent_turn_failed",
                    summary=(
                        turn_result.summary
                        or turn_result.error
                        or "Coding backend failed before producing an accepted diff."
                    ),
                    repair_instructions=[
                        "Repair the code generation response inside the VariantForge attempt sequence.",
                        "Return a valid complete-file JSON payload that satisfies the BuildContract.",
                    ],
                    raw_payload={
                        "stage": "build_coding_agent",
                        "last_turn": turn_result.to_dict(),
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                    },
                )
                if attempt < max_attempts:
                    self.ledger.record_repair_feedback(
                        opened.round_id,
                        attempt,
                        last_feedback.to_dict(),
                    )
                    message = self._repair_message(last_feedback)
                    continue
                break

            patch_text = self._capture_patch(agent_workspace, contract)
            changed_files = self._changed_files(agent_workspace, contract)
            last_changed = changed_files
            patch_path = attempt_dir / "agent.patch"
            patch_path.write_text(patch_text, encoding="utf-8")
            last_patch = patch_path
            outcome, repair_feedback = self._evaluate_completed_build(
                opened=opened,
                contract=contract,
                attempt=attempt,
                repair_attempts=attempt - 1,
                attempt_dir=attempt_dir,
                patch_path=patch_path,
                patch_text=patch_text,
                changed_files=changed_files,
                agent_workspace=agent_workspace,
                turn_result=turn_result,
            )
            if outcome is not None:
                return outcome
            last_feedback = repair_feedback
            if repair_feedback is not None and attempt < max_attempts:
                self.ledger.record_repair_feedback(
                    opened.round_id,
                    attempt,
                    repair_feedback.to_dict(),
                )
                message = self._repair_message(repair_feedback)
                continue
            break

        failed_close_status = "implementation_rejected"
        if (
            str(
                ((last_feedback.raw_payload if last_feedback else {}) or {}).get(
                    "stage"
                )
                or ""
            )
            == "canonical_metric"
        ):
            failed_close_status = "experiment_failed"
        self._close_failed_if_open(
            opened,
            failed_close_status,
            (
                last_feedback.summary
                if last_feedback
                else "build_evaluation_failed"
            ),
            {
                "feedback": last_feedback.to_dict() if last_feedback else {},
                "patch_path": str(last_patch) if last_patch else None,
                "terminal_reason": "variant_forge_attempts_exhausted",
                "variant_forge_attempts_exhausted": True,
                "variant_forge_max_attempts": max_attempts,
                "outer_repair_attempts_disabled": True,
            },
        )
        return self._outcome(
            opened,
            BuildDecision.TERMINAL_REJECTED,
            last_patch,
            last_changed,
            max(0, max_attempts - 1),
            (
                last_feedback.summary
                if last_feedback
                else "build_evaluation_failed"
            ),
        )

    def _matching_stale_build_round(
        self,
        contract: BuildContract,
    ) -> OpenRound | None:
        record = round_records.current_round(self.base_dir, self.task_id)
        if not record:
            return None
        if str(record.get("research_id") or "") != str(contract.research_id):
            return None
        if str(record.get("status") or "") != "round_started":
            return None
        if str(record.get("phase") or "") != round_records.PHASE_BUILD:
            return None
        round_id = int(record.get("round_id") or 0)
        if round_id <= 0:
            return None
        round_dir = self.ledger.round_dir(round_id)
        if not load_build_contract(
            self.base_dir,
            self.task_id,
            str(record.get("research_id") or ""),
        ):
            return None
        return OpenRound(
            round_id=round_id,
            research_id=str(
                record.get("research_id") or contract.research_id
            ),
            round_dir=round_dir,
            record=record,
        )

    def _recover_stale_build_round(
        self,
        opened: OpenRound,
        contract: BuildContract,
    ) -> BuildOutcome:
        stored_contract = load_build_contract(
            self.base_dir,
            self.task_id,
            opened.research_id,
        )
        if stored_contract:
            contract = BuildContract.from_dict(stored_contract)
        workspaces_path = opened.round_dir / "workspaces.json"
        workspaces = (
            json.loads(workspaces_path.read_text(encoding="utf-8"))
            if workspaces_path.is_file()
            else {}
        )
        agent_workspace = Path(
            str(
                workspaces.get("agent_workspace")
                or workspaces.get("agent_worktree")
                or ""
            )
        ).resolve()
        if not agent_workspace.is_dir():
            self._close_failed_if_open(
                opened,
                "infra_failed",
                "stale_build_round_workspace_missing",
                {"agent_workspace": str(agent_workspace)},
            )
            return self._outcome(
                opened,
                BuildDecision.TERMINAL_REJECTED,
                None,
                [],
                0,
                "stale_build_round_workspace_missing",
            )
        patch_text = self._capture_patch(agent_workspace, contract)
        changed_files = self._changed_files(agent_workspace, contract)
        if not patch_text.strip() or not changed_files:
            self._close_failed_if_open(
                opened,
                "infra_failed",
                "stale_build_round_has_no_workspace_diff",
                {"agent_workspace": str(agent_workspace)},
            )
            return self._outcome(
                opened,
                BuildDecision.TERMINAL_REJECTED,
                None,
                changed_files,
                0,
                "stale_build_round_has_no_workspace_diff",
            )
        attempt = self._next_recovery_attempt_index(opened.round_dir)
        attempt_dir = opened.round_dir / f"build_attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        patch_path = attempt_dir / "agent.patch"
        patch_path.write_text(patch_text, encoding="utf-8")
        turn_result = AgentRunResult(
            session_id="stale-build-recovery",
            turn_id=f"recovery-{attempt:03d}",
            status="completed",
            summary="Recovered stale build round from existing workspace diff.",
            execution_summary=(
                "Recovered stale build round from existing workspace diff."
            ),
            changed_files=changed_files,
        )
        self.ledger.record_agent_turn(
            opened.round_id,
            attempt,
            turn_result.to_dict(),
        )
        outcome, feedback = self._evaluate_completed_build(
            opened=opened,
            contract=contract,
            attempt=attempt,
            repair_attempts=0,
            attempt_dir=attempt_dir,
            patch_path=patch_path,
            patch_text=patch_text,
            changed_files=changed_files,
            agent_workspace=agent_workspace,
            turn_result=turn_result,
        )
        if outcome is not None:
            return outcome
        self._close_failed_if_open(
            opened,
            "implementation_rejected",
            (
                feedback.summary
                if feedback
                else "stale_build_round_recovery_requires_repair"
            ),
            {
                "feedback": feedback.to_dict() if feedback else {},
                "patch_path": str(patch_path),
            },
        )
        return self._outcome(
            opened,
            BuildDecision.TERMINAL_REJECTED,
            patch_path,
            changed_files,
            0,
            (
                feedback.summary
                if feedback
                else "stale_build_round_recovery_requires_repair"
            ),
        )

    @staticmethod
    def _next_recovery_attempt_index(round_dir: Path) -> int:
        indexes: list[int] = []
        for path in round_dir.glob("build_attempt_*"):
            suffix = path.name.rsplit("_", 1)[-1]
            try:
                indexes.append(int(suffix))
            except ValueError:
                continue
        return max(indexes or [0]) + 1

    @staticmethod
    def _initial_message(_contract: BuildContract) -> str:
        return (
            "Implement the current BuildContract using the backend-provided execution_contract and source_files. "
            "Preserve task semantics and produce a concrete source edit."
        )

    @staticmethod
    def _repair_message(feedback: VerificationResult) -> str:
        payload = dict(feedback.raw_payload or {})
        failure_evidence = payload.get("failure_evidence") or {}
        primary_frame = {}
        if isinstance(failure_evidence, dict):
            primary_frame = (
                failure_evidence.get("innermost_editable_frame")
                or failure_evidence.get("innermost_project_frame")
                or failure_evidence.get("innermost_frame")
                or {}
            )
        expanded = {
            "status": (
                feedback.status.value
                if hasattr(feedback.status, "value")
                else str(feedback.status)
            ),
            "reason_code": feedback.reason_code,
            "summary": feedback.summary,
            "repair_instructions": list(feedback.repair_instructions or []),
            "primary_error": {
                "stage": payload.get("stage"),
                "error_type": payload.get("error_type"),
                "error_message": payload.get("error_message"),
                "traceback": (
                    payload.get("traceback")
                    or payload.get("error_traceback")
                    or ""
                ),
                "innermost_repair_frame": primary_frame,
                "failure_evidence": failure_evidence,
            },
            "candidate_context": {
                "source_checkout": payload.get("source_checkout"),
                "source_entry_file": payload.get("source_entry_file"),
                "changed_files": payload.get("changed_files") or [],
                "patch_path": payload.get("patch_path"),
                "artifact_paths": list(feedback.artifact_paths or []),
                "parsed_status": payload.get("parsed_status"),
                "artifact_provenance": payload.get("artifact_provenance") or {},
            },
            "full_feedback": feedback.to_dict(),
        }
        return (
            "The previous implementation attempt was not accepted. Continue in this same thread and repair it.\n"
            "Treat primary_error.traceback and primary_error.failure_evidence as authoritative. "
            "If metric artifacts contain no metrics, repair the model/runtime error recorded there; do not repair provenance noise.\n"
            "<evocast_feedback>\n"
            + json.dumps(expanded, ensure_ascii=False, indent=2)
            + "\n</evocast_feedback>\n"
        )

    # Compatibility facades retained for callers and tests that inspect the
    # former monolithic orchestrator surface.
    def _close_failed_if_open(
        self,
        opened: OpenRound,
        status: str,
        reason: str,
        extra: dict[str, Any] | None = None,
        *,
        already_closed_event: str = "round_close_skipped_already_terminal",
    ) -> dict[str, Any] | None:
        return self.terminal.close_failed_if_open(
            opened,
            status,
            reason,
            extra,
            already_closed_event=already_closed_event,
        )

    def _round_is_terminal(self, opened: OpenRound) -> bool:
        return self.terminal.is_terminal(opened)

    def _evaluate_completed_build(self, **kwargs: Any):
        return self.attempt_evaluator.evaluate(**kwargs)

    def _finalize_metric_completed(self, **kwargs: Any):
        return self.terminal.finalize_metric_completed(**kwargs)

    def _outcome(
        self,
        opened: OpenRound,
        decision: BuildDecision,
        patch_path: Path | None,
        changed_files: list[str],
        repair_attempts: int,
        summary: str,
        candidate_snapshot: CandidateSnapshot | None = None,
    ) -> BuildOutcome:
        return self.terminal.outcome(
            opened,
            decision,
            patch_path,
            changed_files,
            repair_attempts,
            summary,
            candidate_snapshot,
        )

    def _base_checkout(self, contract: BuildContract) -> Path:
        return self.workspace.base_checkout(contract)

    def _create_workspace_from_snapshot(
        self,
        path: Path,
        contract: BuildContract,
    ) -> None:
        self.workspace.create_from_snapshot(path, contract)

    def _repair_feedback_with_context(
        self,
        feedback: VerificationResult,
        **kwargs: Any,
    ) -> VerificationResult:
        return self.attempt_evaluator.repair_feedback_with_context(
            feedback,
            **kwargs,
        )

    def _materialize_candidate_snapshot_from_agent_state(
        self,
        **kwargs: Any,
    ) -> CandidateSnapshot:
        return self.workspace.materialize_candidate(**kwargs)

    @staticmethod
    def _normalize_candidate_path(path: str) -> str:
        return CandidateWorkspaceService.normalize_path(path)

    @staticmethod
    def _matches(path: str, patterns: list[str]) -> bool:
        return CandidateWorkspaceService.matches(path, patterns)

    def _validate_candidate_changed_files(
        self,
        changed_files: list[str],
        contract: BuildContract,
        *,
        base_checkout: Path | None = None,
    ) -> None:
        self.workspace.validate_changed_files(
            changed_files,
            contract,
            base_checkout=base_checkout,
        )

    def _capture_patch(
        self,
        workspace: Path,
        contract: BuildContract,
    ) -> str:
        return self.workspace.capture_patch(workspace, contract)

    def _changed_files(
        self,
        workspace: Path,
        contract: BuildContract,
    ) -> list[str]:
        return self.workspace.changed_files(workspace, contract)

    @staticmethod
    def _is_generated_untracked(path: str) -> bool:
        normalized = str(path or "").strip().replace("\\", "/")
        return (
            normalized.startswith(".evocast")
            or "/__pycache__/" in f"/{normalized}"
            or normalized.endswith(".pyc")
            or normalized.endswith(".pyo")
        )
