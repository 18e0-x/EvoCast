"""Single canonical write boundary for evaluation attempts and decisions."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from evocast.domain.knowledge_paths import runs_root
from evocast.harness import rounds as round_records
from evocast.harness.session import AgentSession
from evocast.policy.agent_control_policy import gate_policy, promotion_policy
from evocast.policy.experiment_policy import task_build_mode
from evocast.policy.gate import gate_decision, get_metric_direction
from evocast.state.runtime.store import load_runtime_state
from evocast.state.runtime.trial_journal import append_node, create_node


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    return value


class EvaluationDecisionKernel:
    """Own exactly-once round attempts, gates, and seed-evaluation terminals."""

    def __init__(self, *, base_dir: str, task_id: str) -> None:
        self.base_dir = base_dir
        self.task_id = task_id

    @classmethod
    def from_session(cls, session: AgentSession) -> "EvaluationDecisionKernel":
        return cls(base_dir=session.base_dir, task_id=session.task_id)

    def _round_with_run(self, run_id: str) -> dict[str, Any] | None:
        for record in reversed(
            round_records.list_rounds(self.base_dir, self.task_id)
        ):
            if any(
                str(run.get("run_id") or "") == run_id
                for run in list(record.get("runs") or [])
            ):
                return record
        return None

    def _round_with_gate(self, run_id: str) -> dict[str, Any] | None:
        for record in reversed(
            round_records.list_rounds(self.base_dir, self.task_id)
        ):
            gate = dict(record.get("gate_event") or {})
            if str(gate.get("run_id") or "") == run_id:
                return record
        return None

    def _round_with_seed_node(self, node_id: str) -> dict[str, Any] | None:
        for record in reversed(
            round_records.list_rounds(self.base_dir, self.task_id)
        ):
            seed_eval = dict(record.get("seed_eval") or {})
            if str(seed_eval.get("node_id") or "") == node_id:
                return record
        return None

    def record_attempt(
        self,
        *,
        run_id: str,
        variant_path: str | None,
        status: str,
        error_type: Any = None,
        metrics: dict[str, Any] | None = None,
        failure_signature: dict[str, Any] | None = None,
        failure_evidence: dict[str, Any] | None = None,
        error_message: str = "",
        stage: str = "experiment",
        orchestrator_owns_terminal: bool = False,
    ) -> dict[str, Any] | None:
        existing = self._round_with_run(run_id)
        if existing is not None:
            return existing
        return round_records.record_experiment_run(
            base_dir=self.base_dir,
            task_id=self.task_id,
            run_id=run_id,
            variant_path=variant_path,
            status=status,
            error_type=error_type,
            metrics=metrics,
            failure_signature=failure_signature,
            failure_evidence=failure_evidence,
            error_message=error_message,
            stage=stage,
            allow_terminal_transition=not orchestrator_owns_terminal,
        )

    def record_success_gate(
        self,
        *,
        session: AgentSession,
        run_id: str,
        variant_path: str,
        candidate_id: str | None,
        candidate_kind: str,
        candidate_name: str | None,
        metrics: dict[str, Any],
        objective_metric: str,
        evaluation_stage: str,
        smoke: bool,
        behavior_delta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing = self._round_with_gate(run_id)
        if existing is not None:
            return dict(existing.get("gate_event") or {})

        state = load_runtime_state(
            self.base_dir,
            self.task_id,
            auto_migrate=False,
        )
        current_best = state.current_best
        current_best_metrics = (
            dict(current_best.metrics or {})
            if current_best and current_best.candidate_id
            else {}
        )
        if not current_best_metrics:
            return {"status": "skipped", "reason": "missing_current_best_metrics"}

        policy = gate_policy(self.base_dir)
        build_mode = bool(task_build_mode(self.base_dir, self.task_id))
        evaluation_budget = "build_mode" if build_mode else evaluation_stage
        direction = get_metric_direction(objective_metric)
        gate = gate_decision(
            metrics=metrics,
            reference_metrics=current_best_metrics,
            objective_metric=objective_metric,
            min_relative_improvement=policy["min_relative_improvement"],
            min_seed_eval_relative_improvement=policy[
                "min_seed_eval_relative_improvement"
            ],
        )
        single_seed_decision = str(gate.get("decision") or "")
        gate = dict(gate)
        if smoke:
            gate["evaluation_stage"] = "smoke"
            gate["engineering_status"] = "verified"
            gate["scientific_status"] = (
                "formal_build_mode_result"
                if build_mode
                else "single_seed_smoke_candidate"
            )
        elif build_mode:
            gate["evaluation_stage"] = "build_mode"
            gate["engineering_status"] = "verified"
            gate["scientific_status"] = "formal_build_mode_result"
        gate["evaluation_budget"] = evaluation_budget
        gate["build_mode"] = build_mode

        promotion = promotion_policy(self.base_dir)
        if (
            gate.get("decision") == "accept"
            and promotion["require_seed_eval_for_current_best"]
        ):
            gate["decision"] = "needs_seed_eval"
            gate["reason"] = (
                f"{evaluation_stage} single-seed metrics are comparable but "
                "cannot update current_best; run seed_eval"
            )
        if (
            behavior_delta
            and behavior_delta.get("suspected_noop")
            and not behavior_delta.get("objective_noop_allowed")
            and gate.get("decision") in {"accept", "needs_seed_eval"}
        ):
            gate["decision"] = "needs_review"
            gate["reason"] = (
                "behavior_delta_probe flagged suspected_noop; require "
                "review/seed evidence before treating the candidate as a "
                "meaningful architecture change"
            )

        candidate_value = metrics.get(objective_metric)
        current_best_value = current_best_metrics.get(objective_metric)
        relative_improvement = None
        if (
            isinstance(candidate_value, (int, float))
            and isinstance(current_best_value, (int, float))
            and current_best_value != 0
        ):
            if direction == "lower":
                relative_improvement = (
                    float(current_best_value) - float(candidate_value)
                ) / abs(float(current_best_value))
            else:
                relative_improvement = (
                    float(candidate_value) - float(current_best_value)
                ) / abs(float(current_best_value))

        event = {
            "timestamp": datetime.now().isoformat(),
            "task_id": self.task_id,
            "run_id": run_id,
            "variant_path": variant_path or None,
            "candidate_name": candidate_name or candidate_id or variant_path,
            "decision": gate.get("decision"),
            "objective_metric": objective_metric,
            "objective_direction": direction,
            "candidate_value": candidate_value,
            "reference_kind": "current_best",
            "reference_value": current_best_value,
            "current_best_value": current_best_value,
            "relative_improvement": relative_improvement,
            "threshold_met": single_seed_decision == "accept",
            "single_seed_decision": single_seed_decision,
            "candidate_metrics": metrics,
            "reference_metrics": current_best_metrics,
            "current_best_metrics": current_best_metrics,
            "gate": gate,
            "behavior_delta": behavior_delta,
            "suspected_noop": (
                bool((behavior_delta or {}).get("suspected_noop"))
                if behavior_delta
                else None
            ),
            "objective_noop_allowed": (
                bool((behavior_delta or {}).get("objective_noop_allowed"))
                if behavior_delta
                else None
            ),
            "auto_gate": True,
            "evaluation_stage": evaluation_stage,
            "evaluation_budget": evaluation_budget,
            "build_mode": build_mode,
            "promoted_to_current_best": False,
        }
        gate_events_path = self._append_gate_event(session, event)
        gate_node = create_node(
            self.task_id,
            f"gate_auto_{run_id}",
            action_type="gate",
            model_name=candidate_name or candidate_id or variant_path,
            variant_path=variant_path or None,
            objective_metric=objective_metric,
            metrics=metrics,
            status="success",
            gate_decision=gate,
            artifact_paths=[gate_events_path],
            llm_summary="automatic gate after successful evaluation",
        )
        append_node(
            self.task_id,
            gate_node,
            str(runs_root(self.base_dir)),
        )
        round_records.record_gate_event(
            base_dir=self.base_dir,
            task_id=self.task_id,
            gate_event=event,
        )
        return event

    def _append_gate_event(
        self,
        session: AgentSession,
        event: dict[str, Any],
    ) -> str:
        path = session.knowledge_dir / "gate_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    _json_safe(event),
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
        return str(path)

    def record_seed_result(
        self,
        *,
        variant_path: str | None,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        node_id = str(result.get("node_id") or "").strip()
        if node_id:
            existing = self._round_with_seed_node(node_id)
            if existing is not None:
                return existing
        return round_records.record_seed_eval_result(
            base_dir=self.base_dir,
            task_id=self.task_id,
            variant_path=variant_path,
            result=result,
        )

    def record_seed_failure(
        self,
        *,
        variant_path: str | None,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        node_id = str(result.get("node_id") or "").strip()
        if node_id:
            existing = self._round_with_seed_node(node_id)
            if existing is not None:
                return existing
        return round_records.record_seed_eval_failure(
            base_dir=self.base_dir,
            task_id=self.task_id,
            variant_path=variant_path,
            result=result,
        )
