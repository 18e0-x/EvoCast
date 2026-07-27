"""Session state for the EvoCast multi-agent workflow harness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from evocast.harness.api_client import ProviderClient
from evocast.domain.atomic_io import atomic_write_json
from evocast.state.runtime.dashboard import compact_dashboard_payload, build_dashboard
from evocast.domain.knowledge_paths import runtime_root, task_knowledge_dir, task_runs_dir, runs_root
from evocast.state.runtime.store import load_runtime_state
from evocast.state.runtime.trial_journal import journal_summary
from evocast.harness.task_system import TaskBoard


def _now() -> str:
    return datetime.now().isoformat()


def _atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload, ensure_ascii=False)


@dataclass
class AgentSession:
    task_id: str
    base_dir: str
    client: ProviderClient
    max_tool_result_chars: int = 16000
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.base_dir = str(runtime_root(self.base_dir))

    @property
    def knowledge_dir(self) -> Path:
        return task_knowledge_dir(self.base_dir, self.task_id)

    @property
    def runs_dir(self) -> Path:
        return task_runs_dir(self.base_dir, self.task_id)

    @property
    def transcript_dir(self) -> Path:
        return self.knowledge_dir / "transcripts"

    @property
    def meta_path(self) -> Path:
        return self.knowledge_dir / "session.json"

    @property
    def task_board(self) -> TaskBoard:
        return TaskBoard(self.base_dir, self.task_id)

    def ensure_dirs(self) -> None:
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

    def _load_meta(self) -> Dict[str, Any]:
        if not self.meta_path.exists():
            return {
                "task_id": self.task_id,
                "tool_turn_count": 0,
                "created_at": _now(),
                "updated_at": _now(),
            }
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {"task_id": self.task_id, "tool_turn_count": 0, "created_at": _now(), "updated_at": _now()}

    def _save_meta(self, meta: Dict[str, Any]) -> None:
        meta = dict(meta or {})
        meta["task_id"] = self.task_id
        meta["updated_at"] = _now()
        _atomic_write_json(self.meta_path, meta)

    def tool_turn_count(self) -> int:
        return int(self._load_meta().get("tool_turn_count", 0) or 0)

    def bump_tool_turn(self) -> int:
        meta = self._load_meta()
        meta["tool_turn_count"] = int(meta.get("tool_turn_count", 0) or 0) + 1
        self._save_meta(meta)
        return int(meta["tool_turn_count"])

    def rehydrate_summary(self) -> str:
        dashboard = build_dashboard(self.base_dir, self.task_id, persist=True)
        compact_dashboard = compact_dashboard_payload(dashboard, base_dir=self.base_dir, task_id=self.task_id)
        state = load_runtime_state(self.base_dir, self.task_id, auto_migrate=False)
        tasks = self.task_board.list_tasks()
        try:
            journal = journal_summary(self.task_id, str(runs_root(self.base_dir)))
        except Exception:
            journal = {}
        payload = {
            "task_id": self.task_id,
            "objective_metric": state.objective_metric,
            "dashboard": compact_dashboard,
            "task_board": tasks,
            "journal_summary": journal,
        }
        return "<state-rehydration>\n" + json.dumps(payload, indent=2, ensure_ascii=False) + "\n</state-rehydration>"
