"""Read-only task discovery used by the wizard resume prompt."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

from evocast.domain.execution_ids import format_research_id
from evocast.domain.knowledge_paths import task_knowledge_root


def discover_resumable_tasks(
    base_dir: Path,
    *,
    summary_loader: Callable[[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    knowledge_root = task_knowledge_root(str(base_dir))
    if not knowledge_root.exists():
        return []
    tasks: List[Dict[str, Any]] = []
    for task_dir in knowledge_root.iterdir():
        if not task_dir.is_dir():
            continue
        summary = summary_loader(str(base_dir), task_dir.name)
        has_any_state = (
            summary.get("has_task_config")
            or summary.get("has_best_baseline")
            or summary.get("baseline_search_status")
            or bool(summary.get("completed_rounds"))
        )
        if not has_any_state:
            continue
        summary["mtime"] = max((path.stat().st_mtime for path in task_dir.glob("*") if path.is_file()), default=0)
        tasks.append(summary)
    return sorted(tasks, key=lambda item: float(item.get("mtime") or 0), reverse=True)


def format_resumable_task(summary: Dict[str, Any], *, english: bool) -> str:
    stage = summary.get("stage") or summary.get("task_state", {}).get("stage") or "未知阶段"
    status = summary.get("stage_status") or summary.get("task_state", {}).get("stage_status") or "unknown"
    detail = summary.get("stage_detail") or ""
    stage_text = f"{stage} {detail}" if detail else stage
    baseline = "baseline OK" if summary.get("has_best_baseline") else ("baseline incomplete" if english else "baseline 未完成")
    rounds, pending = summary.get("completed_rounds") or [], summary.get("pending_rounds") or []
    best = ""
    if summary.get("best_model"):
        best = f", best={summary['best_model']}"
        if summary.get("best_value") is not None:
            best += f"({summary['best_value']:.6g})"
    dry = ", dry-run" if summary.get("is_dry_run") else ""
    pending_text = (f", pending round: {format_research_id(int(pending[0]))}" if english else f", 未完成轮: {format_research_id(int(pending[0]))}") if pending else ""
    completed, next_label = ("completed research rounds", "next") if english else ("已完成创新轮", "下一轮")
    updated = ""
    try:
        mtime = float(summary.get("mtime") or 0)
        if mtime > 0:
            updated = f", updated={datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}" if english else f", 最近更新={datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception:
        pass
    return (
        f"{summary['task_id']} - {stage_text}/{status}, {baseline}{best}{dry}{updated}, "
        f"{completed}: {len(rounds)}{pending_text}, {next_label}: {summary.get('next_research_id') or format_research_id(int(summary.get('next_round', 1) or 1))}"
    )
