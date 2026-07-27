from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from evocast.build.contract import BuildContract


@dataclass(frozen=True)
class AgentRunResult:
    session_id: str
    turn_id: str
    status: str
    summary: str
    execution_summary: str = ""
    display_idea: str = ""
    idea_summary: str = ""
    mechanism_name: str = ""
    changed_mechanism: str = ""
    expected_effect: str = ""
    changed_files: list[str] = field(default_factory=list)
    patch_path: Path | None = None
    transcript_path: Path | None = None
    internal_checks: list[dict[str, Any]] = field(default_factory=list)
    edit_deadline_step: int | None = None
    first_successful_edit_step: int | None = None
    edit_deadline_triggered: bool = False
    edit_deadline_rejections: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "status": self.status,
            "summary": self.summary,
            "execution_summary": self.execution_summary or self.summary,
            "display_idea": self.display_idea,
            "idea_summary": self.idea_summary,
            "mechanism_name": self.mechanism_name,
            "changed_mechanism": self.changed_mechanism,
            "expected_effect": self.expected_effect,
            "changed_files": list(self.changed_files),
            "patch_path": str(self.patch_path) if self.patch_path else None,
            "transcript_path": str(self.transcript_path) if self.transcript_path else None,
            "internal_checks": list(self.internal_checks),
            "edit_deadline_step": self.edit_deadline_step,
            "first_successful_edit_step": self.first_successful_edit_step,
            "edit_deadline_triggered": self.edit_deadline_triggered,
            "edit_deadline_rejections": self.edit_deadline_rejections,
            "error": self.error,
        }


class CodingAgentBackend(Protocol):
    def start(self, workspace: Path, contract: BuildContract) -> str:
        ...

    def run_turn(self, session_id: str, message: str, timeout_seconds: int) -> AgentRunResult:
        ...

    def interrupt(self, session_id: str) -> None:
        ...
