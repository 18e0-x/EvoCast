"""Audit, verify, and evaluate one completed coding-agent build attempt."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evocast.build.backends.base import AgentRunResult
from evocast.build.candidate_workspace import CandidateWorkspaceService
from evocast.build.contract import BuildContract
from evocast.build.diff_audit import DiffAuditor
from evocast.build.ledger import OpenRound, ResearchRoundLedger
from evocast.build.metadata_writer import ResearchMetadata, ResearchMetadataWriter
from evocast.build.result import BuildDecision, BuildOutcome, VerificationResult
from evocast.build.round_terminal import RoundTerminalService
from evocast.build.verifier import ExternalVerifier
from evocast.harness import rounds as round_records

MetricRunner = Callable[[Path, BuildContract, Path], VerificationResult]


def _has_valid_objective_metric(metrics: dict[str, Any], objective_metric: str) -> bool:
    value = dict(metrics or {}).get(str(objective_metric or ""))
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


class BuildAttemptEvaluator:
    """Evaluate an existing patch without owning the outer repair loop."""

    def __init__(
        self,
        *,
        base_dir: str,
        task_id: str,
        ledger: ResearchRoundLedger,
        workspace: CandidateWorkspaceService,
        terminal: RoundTerminalService,
        auditor: DiffAuditor,
        verifier: ExternalVerifier,
        metric_runner: MetricRunner | None,
        metadata_writer: ResearchMetadataWriter | None,
    ) -> None:
        self.base_dir = base_dir
        self.task_id = task_id
        self.ledger = ledger
        self.workspace = workspace
        self.terminal = terminal
        self.auditor = auditor
        self.verifier = verifier
        self.metric_runner = metric_runner
        self.metadata_writer = metadata_writer

    def evaluate(
        self,
        *,
        opened: OpenRound,
        contract: BuildContract,
        attempt: int,
        repair_attempts: int,
        attempt_dir: Path,
        patch_path: Path,
        patch_text: str,
        changed_files: list[str],
        agent_workspace: Path,
        turn_result: AgentRunResult,
    ) -> tuple[BuildOutcome | None, VerificationResult | None]:
        audit = self.auditor.audit(
            contract=contract,
            changed_files=changed_files,
            patch_text=patch_text,
            output_dir=attempt_dir,
        )
        self.ledger.record_artifact(
            opened.round_id,
            f"build_attempt_{attempt:02d}_audit.json",
            audit.to_dict(),
        )
        if audit.status != BuildDecision.ACCEPTED:
            feedback = audit.to_verification_result([str(attempt_dir / "diff_audit.json")])
            if audit.status == BuildDecision.REPAIR_REQUIRED:
                return None, feedback
            self.terminal.close_failed_if_open(
                opened,
                "implementation_rejected",
                audit.summary,
                {"audit": audit.to_dict(), "patch_path": str(patch_path)},
            )
            return (
                self.terminal.outcome(
                    opened,
                    BuildDecision.TERMINAL_REJECTED,
                    patch_path,
                    changed_files,
                    repair_attempts,
                    audit.summary,
                ),
                None,
            )

        metadata = self._write_round_metadata(
            opened=opened,
            contract=contract,
            attempt=attempt,
            turn_result=turn_result,
            patch_text=patch_text,
            changed_files=changed_files,
            attempt_dir=attempt_dir,
        )
        if metadata is None:
            return (
                self.terminal.outcome(
                    opened,
                    BuildDecision.TERMINAL_REJECTED,
                    patch_path,
                    changed_files,
                    repair_attempts,
                    "metadata_generation_failed",
                ),
                None,
            )

        candidate_dir = opened.round_dir / f"candidate_{attempt:02d}"
        try:
            candidate_snapshot = self.workspace.materialize_candidate(
                agent_workspace=agent_workspace,
                candidate_dir=candidate_dir,
                changed_files=changed_files,
                contract=contract,
            )
        except RuntimeError as exc:
            feedback = VerificationResult(
                status=BuildDecision.REPAIR_REQUIRED,
                reason_code="candidate_state_rejected",
                summary=(
                    "Candidate final-state materialization failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                repair_instructions=[
                    "Continue in the same agent worktree and repair the final changed files.",
                    "Only edit BuildContract allowed files or allowed new-file roots.",
                    "Do not edit generated, data, result, runtime, protected, or forbidden paths.",
                ],
                artifact_paths=[
                    str(patch_path),
                    str(attempt_dir / "diff_audit.json"),
                ],
                raw_payload={
                    "stage": "candidate_materialization",
                    "changed_files": list(changed_files),
                    "patch_path": str(patch_path),
                },
            )
            self.ledger.record_artifact(
                opened.round_id,
                f"build_attempt_{attempt:02d}_candidate_state_rejected.json",
                feedback.to_dict(),
            )
            return None, feedback

        verifier_workspace = opened.round_dir / f"verifier_{attempt:02d}"
        self.workspace.copy_candidate(candidate_snapshot, verifier_workspace)
        round_records.record_phase_transition(
            base_dir=self.base_dir,
            task_id=self.task_id,
            phase=round_records.PHASE_VERIFY,
        )
        verification = self.verifier.verify(
            contract=contract,
            checkout_dir=verifier_workspace,
            output_dir=attempt_dir / "verifier",
        )
        self.ledger.record_artifact(
            opened.round_id,
            f"build_attempt_{attempt:02d}_verifier.json",
            verification.to_dict(),
        )
        if verification.status != BuildDecision.ACCEPTED:
            if verification.status == BuildDecision.REPAIR_REQUIRED:
                return None, verification
            self.terminal.close_failed_if_open(
                opened,
                "runtime_probe_failed",
                verification.summary,
                {
                    "verification": verification.to_dict(),
                    "patch_path": str(patch_path),
                },
            )
            return (
                self.terminal.outcome(
                    opened,
                    BuildDecision.TERMINAL_REJECTED,
                    patch_path,
                    changed_files,
                    repair_attempts,
                    verification.summary,
                    candidate_snapshot,
                ),
                None,
            )

        metric_checkout = opened.round_dir / f"metric_checkout_{attempt:02d}"
        self.workspace.copy_candidate(candidate_snapshot, metric_checkout)
        if self.metric_runner is None:
            self.ledger.append_event(
                opened.round_id,
                "candidate_accepted_for_metric",
                {
                    "candidate_snapshot_id": candidate_snapshot.snapshot_id,
                    "metric_checkout": str(metric_checkout),
                },
            )
            self.terminal.close_failed_if_open(
                opened,
                "experiment_failed",
                "metric_runner_missing_after_verified_candidate",
                {
                    "candidate_snapshot_id": candidate_snapshot.snapshot_id,
                    "patch_path": str(patch_path),
                },
            )
            return (
                self.terminal.outcome(
                    opened,
                    BuildDecision.TERMINAL_REJECTED,
                    patch_path,
                    changed_files,
                    repair_attempts,
                    "metric_runner_missing_after_verified_candidate",
                    candidate_snapshot,
                ),
                None,
            )

        round_records.record_phase_transition(
            base_dir=self.base_dir,
            task_id=self.task_id,
            phase=round_records.PHASE_EXPERIMENT,
        )
        metric_result = self.metric_runner(
            metric_checkout,
            contract,
            opened.round_dir / "metric",
        )
        self.ledger.record_artifact(
            opened.round_id,
            "metric_result.json",
            metric_result.to_dict(),
        )
        if metric_result.status == BuildDecision.METRIC_COMPLETED:
            objective_metric = str(
                (contract.metric_protocol or {}).get("objective_metric") or "mse_norm"
            )
            if not _has_valid_objective_metric(metric_result.metrics, objective_metric):
                return None, self.repair_feedback_with_context(
                    VerificationResult(
                        status=BuildDecision.REPAIR_REQUIRED,
                        reason_code="metric_completed_without_valid_objective_metric",
                        summary=(
                            "MetricRunner returned METRIC_COMPLETED without a valid "
                            f"numeric {objective_metric}."
                        ),
                        repair_instructions=[
                            "Repair the candidate or metric execution so the completed metric result includes the objective metric.",
                            f"Required objective metric: {objective_metric}.",
                        ],
                        artifact_paths=list(metric_result.artifact_paths or []),
                        raw_payload={
                            **dict(metric_result.raw_payload or {}),
                            "metric_result": metric_result.to_dict(),
                            "objective_metric": objective_metric,
                        },
                    ),
                    changed_files=changed_files,
                    patch_path=patch_path,
                    candidate_snapshot_id=candidate_snapshot.snapshot_id,
                    stage="canonical_metric",
                )
            decision, summary = self.terminal.finalize_metric_completed(
                opened=opened,
                contract=contract,
                candidate_snapshot=candidate_snapshot,
                metric_checkout=metric_checkout,
                patch_path=patch_path,
                changed_files=changed_files,
                metric_result=metric_result,
            )
        elif metric_result.status == BuildDecision.ACCEPTED:
            raise RuntimeError(
                "MetricRunner must return METRIC_COMPLETED for produced metrics; "
                "ACCEPTED is reserved for final gate/seed-eval acceptance."
            )
        else:
            if metric_result.status == BuildDecision.REPAIR_REQUIRED:
                return None, self.repair_feedback_with_context(
                    metric_result,
                    changed_files=changed_files,
                    patch_path=patch_path,
                    candidate_snapshot_id=candidate_snapshot.snapshot_id,
                    stage="canonical_metric",
                )
            self.terminal.close_failed_if_open(
                opened,
                "experiment_failed",
                metric_result.summary,
                {
                    "candidate_snapshot_id": candidate_snapshot.snapshot_id,
                    "metric_result": metric_result.to_dict(),
                },
            )
            decision, summary = metric_result.status, metric_result.summary
        return (
            self.terminal.outcome(
                opened,
                decision,
                patch_path,
                changed_files,
                repair_attempts,
                summary,
                candidate_snapshot,
            ),
            None,
        )

    def repair_feedback_with_context(
        self,
        feedback: VerificationResult,
        *,
        changed_files: list[str],
        patch_path: Path,
        candidate_snapshot_id: str | None,
        stage: str,
    ) -> VerificationResult:
        return VerificationResult(
            status=BuildDecision.REPAIR_REQUIRED,
            reason_code=feedback.reason_code,
            summary=feedback.summary,
            repair_instructions=[
                *list(feedback.repair_instructions or []),
                f"Failure stage: {stage}.",
                "Changed files: "
                + (", ".join(changed_files) if changed_files else "<none>")
                + ".",
                f"Patch artifact: {patch_path}.",
                "Allowed edit files remain limited to BuildContract allowed_edit_files and allowed_new_file_roots.",
            ],
            artifact_paths=list(feedback.artifact_paths or []),
            metrics=dict(feedback.metrics or {}),
            raw_payload={
                **dict(feedback.raw_payload or {}),
                "stage": stage,
                "changed_files": list(changed_files),
                "patch_path": str(patch_path),
                "candidate_snapshot_id": candidate_snapshot_id,
            },
        )

    def _write_round_metadata(
        self,
        *,
        opened: OpenRound,
        contract: BuildContract,
        attempt: int,
        turn_result: AgentRunResult,
        patch_text: str,
        changed_files: list[str],
        attempt_dir: Path,
    ) -> ResearchMetadata | None:
        if self.metadata_writer is None:
            self.terminal.close_failed_if_open(
                opened,
                "metadata_generation_failed",
                "metadata_writer_missing",
                {
                    "attempt": attempt,
                    "patch_path": str(attempt_dir / "agent.patch"),
                    "changed_files": list(changed_files),
                },
            )
            return None
        try:
            metadata = self.metadata_writer.write(
                round_num=opened.round_id,
                contract=contract,
                patch_text=patch_text,
                changed_files=changed_files,
                coding_summary=str(
                    turn_result.execution_summary or turn_result.summary or ""
                ),
                output_dir=attempt_dir / "metadata",
            )
        except Exception as exc:
            self.terminal.close_failed_if_open(
                opened,
                "metadata_generation_failed",
                f"metadata_generation_failed: {type(exc).__name__}: {exc}",
                {
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "patch_path": str(attempt_dir / "agent.patch"),
                    "changed_files": list(changed_files),
                },
            )
            return None
        payload = metadata.to_dict()
        execution_summary = str(
            turn_result.execution_summary or turn_result.summary or ""
        ).strip()
        round_records.record_round_display_metadata(
            base_dir=self.base_dir,
            task_id=self.task_id,
            display_idea=payload["display_idea"],
            idea_summary=payload["idea_summary"],
            mechanism_summary=payload["idea_summary"],
            execution_summary=execution_summary,
            mechanism_name=payload["mechanism_name"],
            changed_mechanism=payload["changed_mechanism"],
            expected_effect=payload["expected_effect"],
        )
        self.ledger.append_event(
            opened.round_id,
            "round_display_metadata",
            {
                "attempt": attempt,
                "source": "llm_metadata_writer",
                "display_idea": payload["display_idea"],
                "idea_summary": payload["idea_summary"],
                "mechanism_summary": payload["idea_summary"],
                "execution_summary": execution_summary,
                "mechanism_name": payload["mechanism_name"],
                "changed_mechanism": payload["changed_mechanism"],
                "expected_effect": payload["expected_effect"],
                "artifact_path": str(
                    attempt_dir / "metadata" / "round_metadata.json"
                ),
            },
        )
        return metadata
