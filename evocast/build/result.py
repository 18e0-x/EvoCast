from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class BuildDecision(str, Enum):
    ACCEPTED = "ACCEPTED"
    METRIC_COMPLETED = "METRIC_COMPLETED"
    REPAIR_REQUIRED = "REPAIR_REQUIRED"
    TERMINAL_REJECTED = "TERMINAL_REJECTED"


@dataclass(frozen=True)
class VerificationResult:
    status: BuildDecision
    reason_code: str
    summary: str
    repair_instructions: list[str] = field(default_factory=list)
    artifact_paths: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason_code": self.reason_code,
            "summary": self.summary,
            "repair_instructions": list(self.repair_instructions),
            "artifact_paths": list(self.artifact_paths),
            "metrics": dict(self.metrics),
            "raw_payload": dict(self.raw_payload),
        }


@dataclass(frozen=True)
class BuildOutcome:
    status: BuildDecision
    research_id: str
    round_id: int
    round_dir: Path
    candidate_snapshot_id: str | None = None
    candidate_checkout: Path | None = None
    base_snapshot_id: str | None = None
    patch_path: Path | None = None
    changed_files: list[str] = field(default_factory=list)
    repair_attempts: int = 0
    summary: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "research_id": self.research_id,
            "round_id": self.round_id,
            "round_dir": str(self.round_dir),
            "candidate_snapshot_id": self.candidate_snapshot_id,
            "candidate_checkout": str(self.candidate_checkout) if self.candidate_checkout else None,
            "base_snapshot_id": self.base_snapshot_id,
            "patch_path": str(self.patch_path) if self.patch_path else None,
            "changed_files": list(self.changed_files),
            "repair_attempts": self.repair_attempts,
            "summary": self.summary,
            "artifacts": dict(self.artifacts),
        }
