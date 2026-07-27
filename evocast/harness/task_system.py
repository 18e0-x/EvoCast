"""Persistent model-visible task board for EvoCast v3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from evocast.domain.atomic_io import atomic_write_json
from evocast.domain.knowledge_paths import task_knowledge_dir

class TaskBoardError(ValueError):
    """Raised for invalid task board operations."""


def _now() -> str:
    return datetime.now().isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_json(path, payload, ensure_ascii=False)


@dataclass
class TaskBoard:
    base_dir: str
    task_id: str

    @property
    def root(self) -> Path:
        return task_knowledge_dir(self.base_dir, self.task_id) / "tasks"

    def _task_path(self, task_id: int | str) -> Path:
        return self.root / f"task_{int(task_id):04d}.json"

    def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.root.exists():
            return []
        tasks: List[Dict[str, Any]] = []
        for path in sorted(self.root.glob("task_*.json")):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if status and item.get("status") != status:
                continue
            tasks.append(item)
        return tasks

    def _next_id(self) -> int:
        existing = [int(item.get("id", 0) or 0) for item in self.list_tasks()]
        return (max(existing) + 1) if existing else 1

    def create_task(
        self,
        subject: str,
        description: str = "",
        blocked_by: Optional[List[int]] = None,
        owner: str = "lead",
    ) -> Dict[str, Any]:
        subject = str(subject or "").strip()
        if not subject:
            raise TaskBoardError("task subject is required")
        task_id = self._next_id()
        ts = _now()
        task = {
            "id": task_id,
            "subject": subject,
            "description": str(description or ""),
            "status": "pending",
            "owner": str(owner or "lead"),
            "blockedBy": [int(x) for x in list(blocked_by or [])],
            "artifacts": [],
            "created_at": ts,
            "updated_at": ts,
        }
        _atomic_write_json(self._task_path(task_id), task)
        return task

    def update_task(
        self,
        task_id: int,
        *,
        status: Optional[str] = None,
        owner: Optional[str] = None,
        add_blocked_by: Optional[List[int]] = None,
        remove_blocked_by: Optional[List[int]] = None,
        artifacts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        path = self._task_path(task_id)
        if not path.exists():
            raise TaskBoardError(f"task does not exist: {task_id}")
        task = json.loads(path.read_text(encoding="utf-8"))

        blocked = [int(x) for x in list(task.get("blockedBy") or [])]
        for item in list(add_blocked_by or []):
            value = int(item)
            if value not in blocked:
                blocked.append(value)
        for item in list(remove_blocked_by or []):
            value = int(item)
            blocked = [x for x in blocked if x != value]
        task["blockedBy"] = blocked

        if owner is not None:
            task["owner"] = str(owner or "lead")
        if artifacts:
            existing = list(task.get("artifacts") or [])
            for artifact in artifacts:
                if artifact not in existing:
                    existing.append(str(artifact))
            task["artifacts"] = existing
        if status is not None:
            status = str(status or "").strip()
            if status not in {"pending", "in_progress", "completed", "blocked", "failed"}:
                raise TaskBoardError(f"unsupported task status: {status}")
            if status == "in_progress":
                if task.get("blockedBy"):
                    raise TaskBoardError("blocked task cannot move to in_progress")
                for other in self.list_tasks():
                    if int(other.get("id", 0) or 0) != int(task_id) and other.get("status") == "in_progress":
                        raise TaskBoardError("only one task can be in_progress")
            task["status"] = status

        task["updated_at"] = _now()
        _atomic_write_json(path, task)

        if task.get("status") == "completed":
            self._remove_completed_blocker(int(task_id))
        return task

    def _remove_completed_blocker(self, completed_id: int) -> None:
        for item in self.list_tasks():
            blocked = [int(x) for x in list(item.get("blockedBy") or [])]
            if completed_id not in blocked:
                continue
            item["blockedBy"] = [x for x in blocked if x != completed_id]
            item["updated_at"] = _now()
            _atomic_write_json(self._task_path(item["id"]), item)
