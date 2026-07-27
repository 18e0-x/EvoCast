"""Single authoritative persistence boundary for EvoCast domain state.

The canonical file is ``task_knowledge/<task>/domain_state.json``.  Legacy
files are read once and migrated when no canonical file exists; normal writes
never recreate them.  Event logs and experiment artifacts remain evidence,
not competing sources of business state.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

from evocast.domain.atomic_io import atomic_write_json
from evocast.domain.execution_ids import format_research_id
from evocast.domain.knowledge_paths import task_knowledge_dir
from evocast.domain.state_model import DOMAIN_STATE_VERSION, RoundRecord, TaskConfig


def _now() -> str:
    return datetime.now().isoformat()


def domain_state_path(base_dir: str, task_id: str) -> Path:
    return task_knowledge_dir(base_dir, task_id) / "domain_state.json"


def domain_state_lock_path(base_dir: str, task_id: str) -> Path:
    return domain_state_path(base_dir, task_id).with_suffix(".lock")


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def _empty(task_id: str) -> dict[str, Any]:
    return {
        "schema_version": DOMAIN_STATE_VERSION,
        "task": TaskConfig(task_id=task_id).to_dict(),
        "runtime": {},
        "contracts": {},
        "rounds": {},
        "created_at": _now(),
        "updated_at": _now(),
    }


def _migrate_legacy(base_dir: str, task_id: str) -> dict[str, Any]:
    root = task_knowledge_dir(base_dir, task_id)
    state = _empty(task_id)
    state["task"] = TaskConfig.from_dict(_read_json(root / "task_config.json", {}), task_id=task_id).to_dict()
    state["runtime"] = dict(_read_json(root / "runtime_state.json", {}) or {})
    # One-time migration of pre-canonical baseline facts.  After this point
    # reports use ``runtime`` exclusively for baseline/current-best state.
    if not state["runtime"].get("baseline"):
        legacy_baseline = _read_json(root / "best_baseline.json", {}) or {}
        legacy_diagnosis = _read_json(root / "baseline_diagnosis.json", {}) or {}
        baseline = dict(legacy_baseline or legacy_diagnosis.get("active_baseline_after_diagnosis") or {})
        if not baseline and legacy_diagnosis:
            baseline = {
                "candidate_id": legacy_diagnosis.get("baseline_candidate_id") or legacy_diagnosis.get("baseline_id"),
                "display_name": legacy_diagnosis.get("baseline_model"),
                "metrics": legacy_diagnosis.get("baseline_metrics") or {},
                "model_config": legacy_diagnosis.get("baseline_model_config") or {},
            }
        if baseline:
            state["runtime"]["baseline"] = baseline
            state["runtime"].setdefault("current_best", dict(baseline))
    if not state["runtime"].get("baseline_search_progress"):
        state["runtime"]["baseline_search_progress"] = dict(_read_json(root / "baseline_search_state.json", {}) or {})
    if not state["runtime"].get("baseline_diagnosis"):
        state["runtime"]["baseline_diagnosis"] = dict(_read_json(root / "baseline_diagnosis.json", {}) or {})
    rounds: dict[str, dict[str, Any]] = {}
    legacy_paths = [
        *root.glob("rounds/Research*/round.json"),
        *root.glob("rounds/Ablation*/round.json"),
        # Historical baseline ablations persisted their facts as
        # ablation_record.json rather than round.json.
        *root.glob("rounds/Ablation*/ablation_record.json"),
    ]
    for path in sorted(legacy_paths):
        raw = dict(_read_json(path, {}) or {})
        if not raw:
            continue
        raw.setdefault("research_id", path.parent.name)
        raw.setdefault("round_id", int("".join(ch for ch in path.parent.name if ch.isdigit()) or 0))
        if str(raw.get("research_id") or "").startswith("Ablation"):
            raw.setdefault("round_scope", "baseline_diagnosis")
            raw.setdefault("counts_toward_research_budget", False)
        record = RoundRecord.from_dict(raw).to_dict()
        rounds[str(record["research_id"])] = record
        contract = _read_json(path.parent / "build_contract.json", {}) or {}
        if contract:
            state["contracts"][str(record["research_id"])] = dict(contract)
    state["rounds"] = rounds
    return state


def _has_legacy_state(base_dir: str, task_id: str) -> bool:
    root = task_knowledge_dir(base_dir, task_id)
    return any(
        path.exists()
        for path in (
            root / "task_config.json",
            root / "runtime_state.json",
            root / "best_baseline.json",
            root / "baseline_search_state.json",
            root / "baseline_diagnosis.json",
        )
    ) or any(root.glob("rounds/*/round.json")) or any(root.glob("rounds/*/ablation_record.json"))


def load_domain_state(base_dir: str, task_id: str, *, auto_migrate: bool = True) -> dict[str, Any]:
    path = domain_state_path(base_dir, task_id)
    state = _read_json(path, {}) or {}
    if state:
        state.setdefault("schema_version", DOMAIN_STATE_VERSION)
        state.setdefault("task", TaskConfig(task_id=task_id).to_dict())
        state.setdefault("runtime", {})
        state.setdefault("contracts", {})
        state.setdefault("rounds", {})
        return state
    state = _migrate_legacy(base_dir, task_id)
    if auto_migrate and _has_legacy_state(base_dir, task_id):
        save_domain_state(base_dir, task_id, state)
    return state


def save_domain_state(base_dir: str, task_id: str, state: dict[str, Any]) -> dict[str, Any]:
    payload = dict(state or {})
    payload["schema_version"] = DOMAIN_STATE_VERSION
    payload["task"] = TaskConfig.from_dict(dict(payload.get("task") or {}), task_id=task_id).to_dict()
    payload.setdefault("runtime", {})
    payload.setdefault("contracts", {})
    payload.setdefault("rounds", {})
    payload.setdefault("created_at", _now())
    payload["updated_at"] = _now()
    path = domain_state_path(base_dir, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(domain_state_lock_path(base_dir, task_id))):
        atomic_write_json(path, payload, ensure_ascii=False, default=str, trailing_newline=True)
    return payload


def mutate_domain_state(base_dir: str, task_id: str, mutator) -> dict[str, Any]:
    """Read-modify-write canonical state under one inter-process file lock."""
    path = domain_state_path(base_dir, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(domain_state_lock_path(base_dir, task_id))):
        state = _read_json(path, {}) or _migrate_legacy(base_dir, task_id)
        if not state:
            state = _empty(task_id)
        state.setdefault("task", TaskConfig(task_id=task_id).to_dict())
        state.setdefault("runtime", {})
        state.setdefault("contracts", {})
        state.setdefault("rounds", {})
        mutator(state)
        payload = dict(state)
        payload["schema_version"] = DOMAIN_STATE_VERSION
        payload["task"] = TaskConfig.from_dict(dict(payload.get("task") or {}), task_id=task_id).to_dict()
        payload["updated_at"] = _now()
        payload.setdefault("created_at", _now())
        atomic_write_json(path, payload, ensure_ascii=False, default=str, trailing_newline=True)
        return payload


def load_task_config(base_dir: str, task_id: str) -> dict[str, Any]:
    return TaskConfig.from_dict(load_domain_state(base_dir, task_id).get("task") or {}, task_id=task_id).to_dict()


def save_task_config(base_dir: str, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    state = mutate_domain_state(
        base_dir, task_id,
        lambda current: current.__setitem__("task", TaskConfig.from_dict(payload, task_id=task_id).to_dict()),
    )
    return dict(state["task"])


def load_runtime_payload(base_dir: str, task_id: str) -> dict[str, Any]:
    return dict(load_domain_state(base_dir, task_id).get("runtime") or {})


def save_runtime_payload(base_dir: str, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    state = mutate_domain_state(base_dir, task_id, lambda current: current.__setitem__("runtime", dict(payload or {})))
    return dict(state["runtime"])


def list_round_records(base_dir: str, task_id: str) -> list[dict[str, Any]]:
    rows = [RoundRecord.from_dict(item).to_dict() for item in dict(load_domain_state(base_dir, task_id).get("rounds") or {}).values()]
    return sorted(rows, key=lambda item: (str(item.get("research_id") or ""), int(item.get("round_id") or 0)))


def load_round_record(base_dir: str, task_id: str, research_id: str) -> dict[str, Any]:
    return dict(load_domain_state(base_dir, task_id).get("rounds", {}).get(str(research_id), {}) or {})


def save_round_record(base_dir: str, task_id: str, record: dict[str, Any]) -> dict[str, Any]:
    canonical = RoundRecord.from_dict(record).to_dict()
    label = str(canonical.get("research_id") or format_research_id(int(canonical.get("round_id") or 0)))
    canonical["research_id"] = label
    mutate_domain_state(base_dir, task_id, lambda current: current["rounds"].__setitem__(label, canonical))
    return canonical


def save_build_contract(base_dir: str, task_id: str, research_id: str, contract: dict[str, Any]) -> None:
    mutate_domain_state(
        base_dir, task_id,
        lambda current: current["contracts"].__setitem__(str(research_id), dict(contract or {})),
    )


def load_build_contract(base_dir: str, task_id: str, research_id: str) -> dict[str, Any]:
    return dict(load_domain_state(base_dir, task_id).get("contracts", {}).get(str(research_id), {}) or {})
