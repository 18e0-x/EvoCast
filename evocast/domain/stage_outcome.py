from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


ALLOWED_STAGE_STATUSES = {
    "completed",
    "degraded",
    "blocked",
    "failed",
    "skipped",
    "inherited",
}


@dataclass
class StageAttempt:
    stage_name: str
    status: str
    reason_code: str = ""
    degradation_level: int = 0
    fallback_used: Optional[str] = None
    llm_requested_abort: bool = False
    artifact_keys: List[str] = field(default_factory=list)
    input_artifact_ids: List[str] = field(default_factory=list)
    output_artifact_ids: List[str] = field(default_factory=list)
    candidate_id: str = ""
    wall_time_ms: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_STAGE_STATUSES:
            raise ValueError(f"Unsupported StageAttempt.status: {self.status}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StageAttempt":
        return cls(
            stage_name=str(payload.get("stage_name") or ""),
            status=str(payload.get("status") or "failed"),
            reason_code=str(payload.get("reason_code") or ""),
            degradation_level=int(payload.get("degradation_level", 0) or 0),
            fallback_used=payload.get("fallback_used"),
            llm_requested_abort=bool(payload.get("llm_requested_abort", False)),
            artifact_keys=list(payload.get("artifact_keys", []) or []),
            input_artifact_ids=list(payload.get("input_artifact_ids", []) or []),
            output_artifact_ids=list(payload.get("output_artifact_ids", []) or []),
            candidate_id=str(payload.get("candidate_id") or ""),
            wall_time_ms=float(payload.get("wall_time_ms", 0.0) or 0.0),
            payload=dict(payload.get("payload", {}) or {}),
        )


StageOutcome = StageAttempt


def completed_stage(stage_name: str, **payload: Any) -> StageAttempt:
    return StageAttempt(stage_name=stage_name, status="completed", payload=dict(payload or {}))


def failed_stage(stage_name: str, reason_code: str, **payload: Any) -> StageAttempt:
    return StageAttempt(
        stage_name=stage_name,
        status="failed",
        reason_code=reason_code,
        payload=dict(payload or {}),
    )
