"""Canonical, serialisable domain objects for an EvoCast task.

The objects in this module deliberately contain *facts*, not presentation or
cache data.  ``state/domain_store.py`` is their sole persistence boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


DOMAIN_STATE_VERSION = 2


@dataclass(frozen=True)
class ResearchIntent:
    """The user's durable research objective, written only at task creation."""

    text: str = ""

    @classmethod
    def from_value(cls, value: Any) -> "ResearchIntent":
        if isinstance(value, dict):
            value = value.get("text") or value.get("research_intent") or ""
        return cls(text=str(value or "").strip())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskConfig:
    """Canonical task definition.

    ``settings`` retains task-specific protocol fields without creating a
    second schema authority.  New fields belong there until they earn a stable
    first-class name.
    """

    task_id: str
    research_intent: ResearchIntent = field(default_factory=ResearchIntent)
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None, *, task_id: str = "") -> "TaskConfig":
        data = dict(payload or {})
        resolved_task_id = str(data.pop("task_id", task_id) or task_id).strip()
        intent = ResearchIntent.from_value(data.pop("research_intent", ""))
        data.pop("research_intent_record", None)
        return cls(task_id=resolved_task_id, research_intent=intent, settings=data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "research_intent": self.research_intent.text,
            "research_intent_record": self.research_intent.to_dict(),
            **dict(self.settings),
        }


@dataclass(frozen=True)
class EvaluationResult:
    """A round's evaluator-owned outcome; never inferred by reports."""

    status: str = "not_run"
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence_paths: list[str] = field(default_factory=list)
    failure_kind: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "EvaluationResult":
        data = dict(payload or {})
        return cls(
            status=str(data.get("status") or "not_run"),
            metrics=dict(data.get("metrics") or {}),
            evidence_paths=[str(path) for path in list(data.get("evidence_paths") or []) if path],
            failure_kind=str(data.get("failure_kind") or ""),
        )

    @classmethod
    def from_round(cls, record: dict[str, Any]) -> "EvaluationResult":
        close_extra = dict(record.get("close_extra") or {})
        metric_result = dict(close_extra.get("metric_result") or {})
        runs = [dict(item or {}) for item in list(record.get("runs") or [])]
        experiment_runs = [item for item in runs if str(item.get("stage") or "") == "experiment"]
        latest = experiment_runs[-1] if experiment_runs else {}
        metrics = dict(metric_result.get("metrics") or latest.get("metrics") or {})
        status = str(metric_result.get("status") or latest.get("status") or "not_run")
        evidence = [str(path) for path in (record.get("artifact_paths") or []) if path]
        return cls(
            status=status,
            metrics=metrics,
            evidence_paths=evidence,
            failure_kind=str(record.get("failure_kind") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    """The gate/round-closer decision for a round."""

    status: str = "pending"
    reason: str = ""
    next_required: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "Decision":
        data = dict(payload or {})
        return cls(
            status=str(data.get("status") or "pending"),
            reason=str(data.get("reason") or ""),
            next_required=str(data.get("next_required") or ""),
        )

    @classmethod
    def from_round(cls, record: dict[str, Any]) -> "Decision":
        close_extra = dict(record.get("close_extra") or {})
        gate = dict(record.get("gate_event") or {})
        status = str(record.get("gate_decision") or record.get("status") or "pending")
        return cls(
            status=status,
            reason=str(record.get("close_reason") or gate.get("reason") or ""),
            next_required=str(close_extra.get("next_required") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoundRecord:
    """Canonical round fact with legacy-compatible details retained verbatim."""

    round_id: int
    research_id: str
    status: str = "round_started"
    phase: str = "design"
    evaluation: EvaluationResult = field(default_factory=EvaluationResult)
    decision: Decision = field(default_factory=Decision)
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RoundRecord":
        data = dict(payload or {})
        # Canonical payloads must round-trip without consulting historical
        # compatibility fields.  Legacy migration is the sole place where the
        # derived fallbacks below are used.
        has_canonical_evaluation = isinstance(data.get("evaluation"), dict)
        has_canonical_decision = isinstance(data.get("decision"), dict)
        evaluation = (
            EvaluationResult.from_dict(data.pop("evaluation"))
            if has_canonical_evaluation
            else EvaluationResult.from_round(data)
        )
        decision = (
            Decision.from_dict(data.pop("decision"))
            if has_canonical_decision
            else Decision.from_round(data)
        )
        return cls(
            round_id=int(data.pop("round_id", 0) or 0),
            research_id=str(data.pop("research_id", "") or ""),
            status=str(data.pop("status", "round_started") or "round_started"),
            phase=str(data.pop("phase", "design") or "design"),
            evaluation=evaluation,
            decision=decision,
            details=data,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **dict(self.details),
            "round_id": self.round_id,
            "research_id": self.research_id,
            "status": self.status,
            "phase": self.phase,
            "evaluation": self.evaluation.to_dict(),
            "decision": self.decision.to_dict(),
        }
