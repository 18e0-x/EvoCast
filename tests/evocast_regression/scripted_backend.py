from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evocast.build.backends.base import AgentRunResult
from evocast.build.contract import BuildContract
from evocast.build.source_snapshot import changed_files_between, source_manifest
from evocast.domain.atomic_io import atomic_write_json


ScriptedTurn = Callable[[Path, BuildContract, str, int], dict[str, Any]]


class ScriptedBackend:
    """Deterministic backend used by regression tests."""

    def __init__(self, turns: list[ScriptedTurn]) -> None:
        self._turns = list(turns)
        self._sessions: dict[str, dict[str, Any]] = {}
        self._session_counter = itertools.count(1)

    def start(self, workspace: Path, contract: BuildContract) -> str:
        session_id = f"scripted-{next(self._session_counter)}"
        self._sessions[session_id] = {
            "workspace": workspace,
            "contract": contract,
            "turn": 0,
            "base_manifest": source_manifest(workspace),
        }
        return session_id

    def run_turn(self, session_id: str, message: str, timeout_seconds: int) -> AgentRunResult:
        state = self._sessions[session_id]
        turn_index = int(state["turn"])
        state["turn"] = turn_index + 1
        workspace = Path(state["workspace"])
        contract = state["contract"]
        if turn_index >= len(self._turns):
            payload = {"status": "failed", "summary": "scripted backend has no remaining turn"}
        else:
            payload = dict(self._turns[turn_index](workspace, contract, message, timeout_seconds) or {})
        changed = changed_files_between(dict(state.get("base_manifest") or {}), workspace)
        transcript = workspace.parent / f"{session_id}_turn_{turn_index + 1:02d}_transcript.json"
        atomic_write_json(
            transcript,
            {
                "session_id": session_id,
                "turn_id": f"turn-{turn_index + 1}",
                "message": message,
                "payload": payload,
                "changed_files": changed,
            },
            ensure_ascii=False,
        )
        return AgentRunResult(
            session_id=session_id,
            turn_id=f"turn-{turn_index + 1}",
            status=str(payload.get("status") or "completed"),
            summary=str(payload.get("summary") or ""),
            execution_summary=str(payload.get("execution_summary") or payload.get("summary") or ""),
            display_idea=str(payload.get("display_idea") or ""),
            idea_summary=str(payload.get("idea_summary") or ""),
            mechanism_name=str(payload.get("mechanism_name") or ""),
            changed_mechanism=str(payload.get("changed_mechanism") or ""),
            expected_effect=str(payload.get("expected_effect") or ""),
            changed_files=changed,
            transcript_path=transcript,
            internal_checks=list(payload.get("internal_checks") or []),
            error=payload.get("error"),
        )

    def interrupt(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
