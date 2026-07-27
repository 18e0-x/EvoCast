"""Build a unified stage timing and cost summary for a task run."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from evocast.domain.knowledge_paths import task_knowledge_dir
from evocast.state.cost_ledger import ledger_path, load_cost_events
from evocast.state.token_usage import build_token_usage_summary, write_token_usage_summary


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _token_cost_by_stage(token_summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    by_stage: Dict[str, Dict[str, Any]] = {}
    for entry in list(token_summary.get("entries") or []):
        if not isinstance(entry, dict):
            continue
        stage = str(entry.get("stage_label") or entry.get("stage_family") or "").strip()
        if not stage:
            continue
        bucket = by_stage.setdefault(
            stage,
            {
                "api_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_hit_tokens": 0,
                "cache_miss_input_tokens": 0,
                "reasoning_tokens": 0,
                "cost": {
                    "cache_hit_cost": 0.0,
                    "cache_miss_input_cost": 0.0,
                    "output_cost": 0.0,
                    "total_cost": 0.0,
                },
            },
        )
        bucket["api_calls"] += 1
        bucket["prompt_tokens"] += int(entry.get("prompt_tokens") or 0)
        bucket["completion_tokens"] += int(entry.get("completion_tokens") or 0)
        bucket["total_tokens"] += int(entry.get("total_tokens") or 0)
        bucket["cache_hit_tokens"] += int(entry.get("prompt_cache_hit_tokens") or 0)
        bucket["cache_miss_input_tokens"] += int(entry.get("prompt_cache_miss_tokens") or 0)
        bucket["reasoning_tokens"] += int(entry.get("reasoning_tokens") or 0)
        cost = entry.get("cost") if isinstance(entry.get("cost"), dict) else {}
        for key in list(bucket["cost"]):
            bucket["cost"][key] += _coerce_float(cost.get(key))
    return by_stage


def build_stage_timing_summary(task_id: str, base_dir: str) -> Dict[str, Any]:
    write_token_usage_summary(task_id, base_dir)
    token_summary = build_token_usage_summary(task_id, base_dir)
    rows: List[Dict[str, Any]] = []
    for event in load_cost_events(base_dir, task_id):
        kind = str(event.get("kind") or "stage")
        usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        rows.append(
            {
                "event_id": event.get("event_id"),
                "parent_id": event.get("parent_id"),
                "name": event.get("stage"),
                "category": kind,
                "round_id": event.get("round_id"),
                "attempt_id": event.get("attempt_id"),
                "status": event.get("status"),
                "started_at": event.get("started_at"),
                "end_time": event.get("finished_at") or event.get("recorded_at"),
                "elapsed_seconds": event.get("elapsed_seconds"),
                "fit_time_seconds": event.get("fit_time_seconds"),
                "inference_time_seconds": event.get("inference_time_seconds"),
                "api_calls": 1 if kind == "llm_api" else 0,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "cache_hit_tokens": usage.get("prompt_cache_hit_tokens") or (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "cache_miss_input_tokens": usage.get("prompt_cache_miss_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "cost": event.get("cost") if isinstance(event.get("cost"), dict) else {},
                "details": details,
                "error_type": event.get("error_type"),
                "error_message": event.get("error_message"),
            }
        )
    rows.sort(key=lambda row: str(row.get("started_at") or row.get("end_time") or ""))
    total_execution_seconds = sum(_coerce_float(row.get("elapsed_seconds")) for row in rows if row.get("category") == "execution")
    return {
        "schema_version": "stage_timing_summary_v1",
        "task_id": task_id,
        "generated_at": datetime.now().isoformat(),
        "notes": [
            "All token, API cost, and elapsed-time records are read from cost_ledger.jsonl.",
            "API cost is calculated from the task provider pricing configuration and returned provider usage.",
        ],
        "totals": {
            "execution_elapsed_seconds": round(total_execution_seconds, 3),
            "api_total_cost": (token_summary.get("cost") or {}).get("total_cost"),
            "api_prompt_tokens": (token_summary.get("totals") or {}).get("prompt_tokens"),
            "api_completion_tokens": (token_summary.get("totals") or {}).get("completion_tokens"),
            "api_cache_hit_tokens": (token_summary.get("totals") or {}).get("prompt_cache_hit_tokens"),
            "api_cache_miss_input_tokens": (token_summary.get("totals") or {}).get("prompt_cache_miss_tokens"),
        },
        "ledger_path": str(ledger_path(base_dir, task_id)),
        "rows": rows,
    }


def write_stage_timing_summary(task_id: str, base_dir: str) -> str:
    summary = build_stage_timing_summary(task_id, base_dir)
    output_path = task_knowledge_dir(base_dir, task_id) / "stage_timing_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return str(output_path)
