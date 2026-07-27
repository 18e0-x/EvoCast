"""Append-only candidate/run registry used for compact agent context."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from evocast.domain.knowledge_paths import task_knowledge_dir


def _now() -> str:
    return datetime.now().isoformat()


def candidate_registry_path(base_dir: str, task_id: str) -> Path:
    path = task_knowledge_dir(base_dir, task_id) / "candidates.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def record_candidate(base_dir: str, task_id: str, record: Dict[str, Any]) -> str:
    path = candidate_registry_path(base_dir, task_id)
    payload = dict(record or {})
    payload.setdefault("task_id", task_id)
    payload.setdefault("created_at", _now())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(payload), ensure_ascii=False, default=str) + "\n")
    return str(path)


def read_candidates(base_dir: str, task_id: str, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = candidate_registry_path(base_dir, task_id)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    if limit and limit > 0:
        return rows[-limit:]
    return rows


def compact_candidate_history(base_dir: str, task_id: str, *, limit: int = 8) -> List[Dict[str, Any]]:
    rows = read_candidates(base_dir, task_id, limit=limit)
    compact: List[Dict[str, Any]] = []
    for row in rows:
        metric = row.get("objective_metric")
        metrics = dict(row.get("metrics") or {})
        compact.append(
            {
                "run_id": row.get("run_id"),
                "candidate_id": row.get("candidate_id") or row.get("variant_path") or row.get("model_name"),
                "variant_path": row.get("variant_path"),
                "model_name": row.get("display_model_name") or row.get("model_name"),
                "import_model_name": row.get("import_model_name") or row.get("model_name"),
                "status": row.get("status"),
                "evaluation_stage": row.get("evaluation_stage"),
                "seed": row.get("seed"),
                "objective_metric": metric,
                "objective_value": metrics.get(metric) if metric else None,
                "gate_decision": row.get("gate_decision"),
                "config_hash": row.get("config_hash"),
                "created_at": row.get("created_at"),
            }
        )
    return compact
