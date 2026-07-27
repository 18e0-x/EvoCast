"""Resume/checkpoint helpers for evocast."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from evocast.domain.atomic_io import atomic_write_json as _atomic_write_json
from evocast.domain.execution_ids import format_research_id
from evocast.domain.knowledge_paths import task_knowledge_dir, task_runs_dir
from evocast.state.domain_store import load_task_config


RESUME_VERSION = 1


def _now() -> str:
    return datetime.now().isoformat()


def atomic_write_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Write JSON atomically enough for crash recovery on local filesystems."""
    _atomic_write_json(path, data, ensure_ascii=False, default=str, trailing_newline=True)


def load_json(path: str | Path, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def task_state_path(base_dir: str, task_id: str) -> str:
    return str(task_knowledge_dir(base_dir, task_id) / "task_state.json")


def baseline_search_state_path(base_dir: str, task_id: str) -> str:
    return str(task_knowledge_dir(base_dir, task_id) / "baseline_search_state.json")


def round_state_path(base_dir: str, task_id: str, round_number: int) -> str:
    return os.path.join(
        str(task_runs_dir(base_dir, task_id)),
        "variants",
        f"{format_research_id(round_number)}_state.json",
    )


def write_task_state(
    base_dir: str,
    task_id: str,
    stage: str,
    status: str = "running",
    completed_stages: Optional[Iterable[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    state = load_json(task_state_path(base_dir, task_id), {}) or {}
    existing_completed = list(state.get("completed_stages", []))
    for item in completed_stages or []:
        if item not in existing_completed:
            existing_completed.append(item)
    state.update(
        {
            "task_id": task_id,
            "resume_version": RESUME_VERSION,
            "stage": stage,
            "stage_status": status,
            "completed_stages": existing_completed,
            "last_updated": _now(),
        }
    )
    if extra:
        state.update(extra)
    atomic_write_json(task_state_path(base_dir, task_id), state)


def mark_task_stage_complete(
    base_dir: str,
    task_id: str,
    stage: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    write_task_state(
        base_dir=base_dir,
        task_id=task_id,
        stage=stage,
        status="completed",
        completed_stages=[stage],
        extra=extra,
    )


def task_has_artifact(base_dir: str, task_id: str, name: str) -> bool:
    return (task_knowledge_dir(base_dir, task_id) / name).exists()


def best_baseline_complete(base_dir: str, task_id: str) -> bool:
    from evocast.state.runtime.store import load_runtime_state

    runtime_state = load_runtime_state(base_dir, task_id, auto_migrate=False)
    return bool(runtime_state.baseline.candidate_id or runtime_state.baseline.display_name)


def baseline_search_has_success(base_dir: str, task_id: str) -> bool:
    state = load_json(baseline_search_state_path(base_dir, task_id), {}) or {}
    for result in state.get("run_results", []) or []:
        if result.get("status") == "success" and result.get("objective_value") is not None:
            return True
    for result in state.get("cheap_results", []) or []:
        if result.get("status") == "success" and result.get("objective_value") is not None:
            return True
    return False


def read_journal_nodes(base_dir: str, task_id: str) -> List[Dict[str, Any]]:
    path = str(task_runs_dir(base_dir, task_id) / "trial_journal.jsonl")
    if not os.path.exists(path):
        return []
    nodes: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                nodes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return nodes


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


_ROUND_RE = re.compile(r"(?:round_|Research)(\d{3,})")


def completed_research_rounds(base_dir: str, task_id: str) -> List[int]:
    """Return completed non-baseline round numbers from the journal."""
    rounds: set[int] = set()
    for node in read_journal_nodes(base_dir, task_id):
        if node.get("action_type") == "baseline":
            continue
        node_id = str(node.get("node_id", ""))
        match = _ROUND_RE.search(node_id)
        if match:
            rounds.add(int(match.group(1)))
    return sorted(rounds)


def next_research_round(base_dir: str, task_id: str) -> int:
    pending = pending_round_state_numbers(base_dir, task_id)
    if pending:
        return pending[0]
    rounds = completed_research_rounds(base_dir, task_id)
    return (max(rounds) + 1) if rounds else 1


def pending_round_state_numbers(base_dir: str, task_id: str) -> List[int]:
    variants_dir = str(task_runs_dir(base_dir, task_id) / "variants")
    if not os.path.isdir(variants_dir):
        return []
    completed = set(completed_research_rounds(base_dir, task_id))
    pending: List[int] = []
    for name in os.listdir(variants_dir):
        match = re.fullmatch(r"(?:round_|Research)(\d{3,})_state\.json", name)
        if not match:
            continue
        number = int(match.group(1))
        if number in completed:
            continue
        state = load_json(os.path.join(variants_dir, name), {}) or {}
        if state.get("status") != "completed":
            pending.append(number)
    return sorted(pending)


def load_round_state(base_dir: str, task_id: str, round_number: int) -> Dict[str, Any]:
    primary = round_state_path(base_dir, task_id, round_number)
    state = load_json(primary, {}) or {}
    if state:
        return state
    legacy = str(task_runs_dir(base_dir, task_id) / "variants" / f"round_{round_number:03d}_state.json")
    return load_json(legacy, {}) or {}


def save_round_state(
    base_dir: str,
    task_id: str,
    round_number: int,
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    state = load_round_state(base_dir, task_id, round_number)
    state.update(updates)
    state.setdefault("task_id", task_id)
    state.setdefault("round", format_research_id(round_number))
    state["resume_version"] = RESUME_VERSION
    state["last_updated"] = _now()
    atomic_write_json(round_state_path(base_dir, task_id, round_number), state)
    return state


def resume_summary(base_dir: str, task_id: str) -> Dict[str, Any]:
    from evocast.state.runtime.store import load_runtime_state

    knowledge_dir = str(task_knowledge_dir(base_dir, task_id))
    runs_dir = str(task_runs_dir(base_dir, task_id))
    runtime_state = load_runtime_state(base_dir, task_id, auto_migrate=False)
    search_state = load_json(baseline_search_state_path(base_dir, task_id), {}) or {}
    task_state = load_json(task_state_path(base_dir, task_id), {}) or {}
    search_log = load_json(os.path.join(knowledge_dir, "search_round_log.json"), {}) or {}
    budget_funnel = load_json(os.path.join(knowledge_dir, "budget_funnel.json"), {}) or {}
    baseline_runs = read_jsonl(os.path.join(knowledge_dir, "baseline_runs.jsonl"))

    rounds = completed_research_rounds(base_dir, task_id)
    pending_rounds = pending_round_state_numbers(base_dir, task_id)
    next_round = pending_rounds[0] if pending_rounds else ((max(rounds) + 1) if rounds else 1)
    has_best_baseline = best_baseline_complete(base_dir, task_id)
    has_successful_baseline_search = baseline_search_has_success(base_dir, task_id)

    control_search = dict(runtime_state.baseline_search_progress or {})
    control_research = dict(runtime_state.research or {})
    control_best_baseline = runtime_state.baseline.to_dict() if runtime_state.baseline.candidate_id else {}

    search_total_run = len(search_state.get("run_results", []) or [])
    if not search_total_run:
        search_total_run = int(control_search.get("cheap_completed") or 0)
    if not search_total_run:
        search_total_run = int(search_log.get("total_run") or 0)
    if not search_total_run and baseline_runs:
        search_total_run = len(
            {r.get("model_key") for r in baseline_runs if r.get("tier") == "cheap"}
        )

    search_max_total = (
        control_search.get("cheap_target")
        or search_state.get("max_cheap_total")
        or search_log.get("max_cheap_total")
    )
    if not search_max_total and budget_funnel.get("cheap"):
        search_max_total = budget_funnel["cheap"].get("total")

    best_model = (
        control_search.get("best_model")
        or control_best_baseline.get("model_name")
        or search_state.get("best_model_key")
        or search_log.get("best_model")
    )
    best_value = control_search.get(
        "best_value",
        search_state.get("best_value", search_log.get("best_value")),
    )

    is_dry_run = any(
        r.get("error_type") == "dry_run" or r.get("metrics", {}).get("_dry_run")
        for r in baseline_runs[:5]
    )
    if control_search.get("is_dry_run") is True:
        is_dry_run = True

    stage_label_map = {
        "init": "初始化",
        "profile": "任务画像",
        "registry": "模型注册",
        "baseline_search": "模型寻优",
        "diagnosis": "诊断分析",
        "research_loop": "创新阶段",
        "export": "导出",
        "completed": "已完成",
    }
    round_stage_label_map = {
        "context": "上下文准备",
        "synthesis": "创新生成",
        "judge": "方案评审",
        "planner": "实现规划",
        "implementer": "代码实现",
        "experiment": "实验执行",
        "metrics": "指标解析",
        "gate": "结果判定",
        "review": "复核",
        "journal": "日志归档",
        "completed": "轮次完成",
    }

    inferred_stage_key = runtime_state.current_stage or task_state.get("stage")
    inferred_status = runtime_state.current_stage_status or task_state.get("stage_status")
    inferred_stage = stage_label_map.get(inferred_stage_key, inferred_stage_key)
    stage_detail = ""

    if not inferred_stage:
        if pending_rounds:
            inferred_stage = f"创新阶段 {format_research_id(pending_rounds[0])}"
            pending_state = load_round_state(base_dir, task_id, pending_rounds[0])
            inferred_status = pending_state.get("stage") or "running"
        elif rounds:
            inferred_stage = "创新阶段"
            inferred_status = "ready"
            stage_detail = f"下一轮 {format_research_id(next_round)}"
        elif has_best_baseline:
            inferred_stage = "模型寻优"
            inferred_status = "completed"
        elif (control_search.get("status") == "completed" or search_state.get("status") == "completed") and not has_successful_baseline_search:
            inferred_stage = "模型寻优"
            inferred_status = "failed"
        elif search_total_run:
            inferred_stage = "模型寻优"
            inferred_status = "running"
        elif task_has_artifact(base_dir, task_id, "model_registry_snapshot.json"):
            inferred_stage = "模型注册"
            inferred_status = "completed"
        elif task_has_artifact(base_dir, task_id, "task_profile.json"):
            inferred_stage = "任务画像"
            inferred_status = "completed"
        elif load_task_config(base_dir, task_id).get("task_id"):
            inferred_stage = "初始化"
            inferred_status = "completed"
        else:
            inferred_stage = "未初始化"
            inferred_status = "unknown"

    if inferred_stage == "模型寻优" and search_total_run:
        stage_detail = f"{search_total_run}/{search_max_total}" if search_max_total else str(search_total_run)
        if inferred_status == "running":
            stage_detail = f"{stage_detail}/running"

    control_completed_rounds = control_research.get("completed_rounds")
    if isinstance(control_completed_rounds, list) and control_completed_rounds:
        rounds = sorted({int(r) for r in control_completed_rounds})

    control_pending_round = control_research.get("pending_round")
    if isinstance(control_pending_round, int) and control_pending_round not in pending_rounds:
        pending_rounds = sorted(pending_rounds + [control_pending_round])
        next_round = pending_rounds[0]

    current_round = control_research.get("current_round")
    current_research_id = control_research.get("current_research_id")
    current_round_stage = control_research.get("current_round_stage")
    if inferred_stage == "创新阶段" and isinstance(current_round, int):
        round_stage_label = round_stage_label_map.get(current_round_stage, current_round_stage or "unknown")
        stage_detail = f"{current_research_id or format_research_id(current_round)}/{round_stage_label}/{inferred_status or 'unknown'}"

    return {
        "task_id": task_id,
        "knowledge_dir": knowledge_dir,
        "runs_dir": runs_dir,
        "control_state": {},
        "task_state": task_state,
        "stage": inferred_stage,
        "stage_status": inferred_status,
        "stage_detail": stage_detail,
        "has_task_config": bool(load_task_config(base_dir, task_id).get("task_id")),
        "has_task_profile": task_has_artifact(base_dir, task_id, "task_profile.json"),
        "has_registry": task_has_artifact(base_dir, task_id, "model_registry_snapshot.json"),
        "has_best_baseline": has_best_baseline,
        "baseline_search_status": (
            ("completed" if has_best_baseline and search_total_run else None)
            if not (control_search.get("status") or search_state.get("status"))
            else (
                "failed"
                if (control_search.get("status") == "completed" or search_state.get("status") == "completed")
                and not has_best_baseline
                and not has_successful_baseline_search
                else (control_search.get("status") or search_state.get("status"))
            )
        ),
        "baseline_search_phase": (
            control_search.get("phase")
            or search_state.get("phase")
            or ("done" if has_best_baseline and search_total_run else None)
        ),
        "baseline_search_total_run": search_total_run,
        "baseline_search_max_total": search_max_total,
        "best_model": best_model,
        "best_value": best_value,
        "is_dry_run": bool(is_dry_run),
        "completed_rounds": rounds,
        "pending_rounds": pending_rounds,
        "next_round": next_round,
        "next_research_id": format_research_id(next_round),
    }
