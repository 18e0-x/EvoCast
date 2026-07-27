"""Canonical round closure and metric-completion decisions for build attempts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from evocast.build.contract import BuildContract
from evocast.build.ledger import OpenRound, ResearchRoundLedger
from evocast.build.result import BuildDecision, BuildOutcome, VerificationResult
from evocast.build.source_snapshot import CandidateSnapshot
from evocast.domain.atomic_io import atomic_write_json
from evocast.evaluation.decision_kernel import EvaluationDecisionKernel
from evocast.harness import rounds as round_records
from evocast.state.domain_store import load_build_contract, load_round_record

SeedEvalRunner = Callable[[dict[str, Any]], dict[str, Any]]


class RoundTerminalService:
    """Own terminal round writes; evaluation components only request decisions."""

    def __init__(
        self,
        *,
        base_dir: str,
        task_id: str,
        ledger: ResearchRoundLedger,
        seed_eval_runner: SeedEvalRunner | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.task_id = task_id
        self.ledger = ledger
        self.seed_eval_runner = seed_eval_runner

    def close_failed_if_open(
        self,
        opened: OpenRound,
        status: str,
        reason: str,
        extra: dict[str, Any] | None = None,
        *,
        already_closed_event: str = "round_close_skipped_already_terminal",
    ) -> dict[str, Any] | None:
        current = round_records.current_round(self.base_dir, self.task_id)
        if not current:
            self.ledger.append_event(
                opened.round_id,
                already_closed_event,
                {
                    "status": status,
                    "reason": reason,
                    "extra": extra or {},
                    "state": "no_current_round",
                },
            )
            return None
        if int(current.get("round_id") or 0) != int(opened.round_id):
            self.ledger.append_event(
                opened.round_id,
                already_closed_event,
                {
                    "status": status,
                    "reason": reason,
                    "extra": extra or {},
                    "state": "different_current_round",
                    "current_round_id": current.get("round_id"),
                    "current_research_id": current.get("research_id"),
                },
            )
            return current
        if str(current.get("status") or "") in round_records.TERMINAL_ROUND_STATUSES:
            self.ledger.append_event(
                opened.round_id,
                already_closed_event,
                {
                    "status": status,
                    "reason": reason,
                    "extra": extra or {},
                    "state": "current_round_already_terminal",
                    "current_status": current.get("status"),
                },
            )
            return current
        return self.ledger.close_failed(opened.round_id, status, reason, extra or {})

    def is_terminal(self, opened: OpenRound) -> bool:
        current = round_records.current_round(self.base_dir, self.task_id)
        if not current:
            return True
        if int(current.get("round_id") or 0) != int(opened.round_id):
            return True
        return str(current.get("status") or "") in round_records.TERMINAL_ROUND_STATUSES

    def round_record(self, opened: OpenRound) -> dict[str, Any]:
        return load_round_record(self.base_dir, self.task_id, opened.research_id)

    def finalize_metric_completed(
        self,
        *,
        opened: OpenRound,
        contract: BuildContract,
        candidate_snapshot: CandidateSnapshot,
        metric_checkout: Path,
        patch_path: Path,
        changed_files: list[str],
        metric_result: VerificationResult,
    ) -> tuple[BuildDecision, str]:
        record = self.round_record(opened)
        source_binding = dict((contract.base_source_ref or {}).get("source_binding") or {})
        model_binding_ref = str((contract.base_source_ref or {}).get("model_binding_ref") or "")
        source_ref = {
            "kind": "source_snapshot",
            "candidate_snapshot_id": candidate_snapshot.snapshot_id,
            "base_snapshot_id": contract.base_snapshot_id,
            "patch_path": str(patch_path),
            "changed_files": list(changed_files),
            "parent_candidate_id": contract.base_candidate_id,
            "source_checkout": str(metric_checkout),
        }
        if source_binding:
            source_ref["source_binding"] = source_binding
        if model_binding_ref:
            source_ref["model_binding_ref"] = model_binding_ref
        objective_metric = str(
            (record.get("gate_event") or {}).get("objective_metric")
            or (contract.metric_protocol or {}).get("objective_metric")
            or "mse_norm"
        )
        round_records.record_metric_completed_result(
            base_dir=self.base_dir,
            task_id=self.task_id,
            round_id=opened.round_id,
            research_id=opened.research_id,
            metrics=dict(metric_result.metrics or {}),
            objective_metric=objective_metric,
            metric_result=metric_result.to_dict(),
            candidate_snapshot_id=candidate_snapshot.snapshot_id,
            patch_path=patch_path,
            changed_files=changed_files,
            source_ref=source_ref,
        )
        record = self.round_record(opened)
        gate_decision = str(record.get("gate_decision") or "").strip()
        round_status = str(record.get("status") or "").strip()
        if gate_decision == "needs_seed_eval":
            if self.seed_eval_runner is None:
                raise RuntimeError(
                    "BuildContract Research requires seed_eval_runner for gate_decision=needs_seed_eval"
                )
            gate_event = dict(record.get("gate_event") or {})
            objective_metric = str(
                gate_event.get("objective_metric")
                or (contract.metric_protocol or {}).get("objective_metric")
                or "mse_norm"
            )
            variant_path = str(
                record.get("variant_path")
                or (contract.metric_protocol or {}).get("variant_path")
                or ""
            )
            model_entry = dict((contract.metric_protocol or {}).get("model_config") or {})
            parent_id = contract.base_candidate_id or (
                (contract.metric_protocol or {}).get("baseline") or {}
            ).get("candidate_id")
            source_ref["parent_candidate_id"] = parent_id
            seed_args = {
                "variant_path": variant_path,
                "model_name": model_entry.get("model_name") or contract.target_model,
                "adapter": model_entry.get("adapter"),
                "model_hyper_params": dict(
                    model_entry.get("effective_model_hyper_params")
                    or model_entry.get("model_hyper_params")
                    or {}
                ),
                "source_checkout": str(metric_checkout),
                "objective_metric": objective_metric,
                "candidate_id": record.get("research_id") or opened.research_id,
                "run_id": gate_event.get("run_id") or f"{opened.research_id}_seed_eval",
                "promote_on_accept": True,
                "promotion_metadata": {
                    "source": "research_seed_eval_accept",
                    "candidate_kind": "source_patch",
                    "parent_candidate_id": parent_id,
                    "candidate_snapshot_id": candidate_snapshot.snapshot_id,
                    "base_snapshot_id": contract.base_snapshot_id,
                    "patch_path": str(patch_path),
                    "changed_files": list(changed_files),
                    "source_ref": source_ref,
                    "source_checkout": str(metric_checkout),
                    "research_id": opened.research_id,
                    "source_binding": source_binding,
                    "model_binding_ref": model_binding_ref,
                },
            }
            try:
                seed_result = self.seed_eval_runner(seed_args)
            except Exception as exc:
                failure_record = EvaluationDecisionKernel(
                    base_dir=self.base_dir,
                    task_id=self.task_id,
                ).record_seed_failure(
                    variant_path=variant_path or None,
                    result={
                        "node_id": seed_args.get("run_id"),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "summary": "seed_eval_runner_failed",
                    },
                )
                self.ledger.append_event(
                    opened.round_id,
                    "seed_eval_failed",
                    {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "recorded": bool(failure_record),
                    },
                )
                return BuildDecision.TERMINAL_REJECTED, (
                    "Seed evaluation failed for the BuildContract research candidate: "
                    f"{type(exc).__name__}: {exc}"
                )
            seed_decision = str(
                (dict(seed_result.get("significance_decision") or {})).get("decision") or ""
            ).strip()
            self.ledger.append_event(
                opened.round_id,
                "seed_eval_completed",
                {
                    "decision": seed_decision,
                    "result_path": seed_result.get("result_path"),
                    "promoted_to_current_best": seed_result.get("promoted_to_current_best"),
                },
            )
            if seed_decision == "accept" and seed_result.get("promoted_to_current_best"):
                return (
                    BuildDecision.ACCEPTED,
                    "Seed evaluation accepted the BuildContract research candidate.",
                )
            if seed_decision == "accept":
                return BuildDecision.TERMINAL_REJECTED, (
                    "Seed evaluation accepted but did not promote current_best under the unified threshold."
                )
            return BuildDecision.TERMINAL_REJECTED, (
                "Seed evaluation rejected the BuildContract research candidate: "
                + (seed_decision or "missing_decision")
            )
        if gate_decision == "needs_review":
            return (
                BuildDecision.METRIC_COMPLETED,
                "Metric completed and gate_decision=needs_review; round remains open for review.",
            )
        if gate_decision in {"reject", "rejected"} or round_status in {
            "rejected",
            "scientific_rejected",
            "marginal_no_seed_eval",
        }:
            return (
                BuildDecision.TERMINAL_REJECTED,
                "Metric completed and gate rejected the BuildContract research candidate.",
            )
        if gate_decision == "accept" or (
            round_status == "completed" and record.get("metrics_source") != "metric_completed"
        ):
            return (
                BuildDecision.ACCEPTED,
                "Metric completed and gate accepted the BuildContract research candidate.",
            )
        self.ledger.close_completed(
            opened.round_id,
            {
                "candidate_snapshot_id": candidate_snapshot.snapshot_id,
                "base_snapshot_id": contract.base_snapshot_id,
                "source_ref": {
                    "kind": "source_snapshot",
                    "candidate_snapshot_id": candidate_snapshot.snapshot_id,
                    "base_snapshot_id": contract.base_snapshot_id,
                    "patch_path": str(patch_path),
                    "changed_files": list(changed_files),
                    "parent_candidate_id": contract.base_candidate_id,
                    "source_checkout": str(metric_checkout),
                    **({"source_binding": source_binding} if source_binding else {}),
                    **({"model_binding_ref": model_binding_ref} if model_binding_ref else {}),
                },
                "metric_result": metric_result.to_dict(),
            },
        )
        return BuildDecision.ACCEPTED, metric_result.summary

    def outcome(
        self,
        opened: OpenRound,
        decision: BuildDecision,
        patch_path: Path | None,
        changed_files: list[str],
        repair_attempts: int,
        summary: str,
        candidate_snapshot: CandidateSnapshot | None = None,
    ) -> BuildOutcome:
        try:
            base_snapshot_id = BuildContract.from_dict(
                load_build_contract(self.base_dir, self.task_id, opened.research_id)
            ).base_snapshot_id
        except Exception:
            base_snapshot_id = None
        outcome = BuildOutcome(
            status=decision,
            research_id=opened.research_id,
            round_id=opened.round_id,
            round_dir=opened.round_dir,
            candidate_snapshot_id=(
                candidate_snapshot.snapshot_id if candidate_snapshot else None
            ),
            candidate_checkout=(
                Path(candidate_snapshot.source_checkout) if candidate_snapshot else None
            ),
            base_snapshot_id=base_snapshot_id,
            patch_path=patch_path,
            changed_files=changed_files,
            repair_attempts=repair_attempts,
            summary=summary,
        )
        atomic_write_json(
            opened.round_dir / "build_outcome.json",
            outcome.to_dict(),
            ensure_ascii=False,
            default=str,
        )
        round_records.record_build_outcome(
            base_dir=self.base_dir,
            task_id=self.task_id,
            research_id=opened.research_id,
            round_id=opened.round_id,
            candidate_snapshot_id=outcome.candidate_snapshot_id,
            patch_path=outcome.patch_path,
            changed_files=outcome.changed_files,
            status=outcome.status,
        )
        return outcome
