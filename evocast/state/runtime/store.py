from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from filelock import FileLock

from evocast.domain.execution_ids import format_research_id
from evocast.policy.gate import get_metric_direction, is_better
from evocast.domain.knowledge_paths import task_knowledge_dir
from evocast.domain.metric_semantics import DEFAULT_OBJECTIVE_METRIC
from evocast.state.runtime.state import Candidate, ResearchRoundState, RuntimeState
from evocast.state.domain_store import (
    load_domain_state,
    load_runtime_payload,
    load_task_config,
    mutate_domain_state,
    save_runtime_payload,
)
from evocast.domain.stage_outcome import StageOutcome
from evocast.policy.agent_control_policy import gate_policy


def _now() -> str:
    return datetime.now().isoformat()


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    last_exc: Optional[Exception] = None
    for attempt in range(8):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.05 * (attempt + 1))
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    if last_exc:
        raise last_exc


def _load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def runtime_state_path(base_dir: str, task_id: str) -> str:
    return str(task_knowledge_dir(base_dir, task_id) / "runtime_state.json")


def runtime_events_path(base_dir: str, task_id: str) -> str:
    return str(task_knowledge_dir(base_dir, task_id) / "runtime_events.jsonl")


def executor_snapshot_path(base_dir: str, task_id: str) -> str:
    return str(task_knowledge_dir(base_dir, task_id) / "executor_config_snapshot.json")


def _task_mode(base_dir: str, task_id: str) -> str:
    task_config = load_task_config(base_dir, task_id)
    configured = str(task_config.get("mode") or "").strip().lower()
    if configured in {"strict", "exploration"}:
        return configured
    return "exploration"


def load_runtime_state(base_dir: str, task_id: str, *, auto_migrate: bool = True) -> RuntimeState:
    domain = load_domain_state(base_dir, task_id, auto_migrate=auto_migrate)
    payload = dict(domain.get("runtime") or {})
    if payload:
        return RuntimeState.from_dict(payload)
    task_config = load_task_config(base_dir, task_id)
    objective_metric = str(task_config.get("objective_metric") or DEFAULT_OBJECTIVE_METRIC)
    return RuntimeState.empty(task_id, objective_metric=objective_metric, mode=_task_mode(base_dir, task_id))


def save_runtime_state(base_dir: str, task_id: str, state: RuntimeState | Dict[str, Any]) -> RuntimeState:
    runtime_state = state if isinstance(state, RuntimeState) else RuntimeState.from_dict(state)
    runtime_state.touch()
    save_runtime_payload(base_dir, task_id, runtime_state.to_dict())
    return runtime_state


def record_runtime_event(
    base_dir: str,
    task_id: str,
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
) -> bool:
    """Append a runtime event, suppressing replays with the same durable key."""
    path = runtime_events_path(base_dir, task_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    event_payload = dict(payload or {})
    idempotency_key = str(event_payload.get("idempotency_key") or "").strip()
    record = {
        "task_id": task_id,
        "ts": _now(),
        "event_type": event_type,
        "payload": event_payload,
    }
    with FileLock(path + ".lock"):
        if idempotency_key and os.path.isfile(path):
            prior_events: list[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as existing:
                for line in existing:
                    try:
                        prior_events.append(json.loads(line))
                    except (TypeError, ValueError):
                        continue
            for prior in prior_events:
                prior_key = str(
                    (prior.get("payload") or {}).get("idempotency_key") or ""
                ).strip()
                legacy_terminal_match = (
                    not prior_key
                    and str(event_type) == "task_terminal"
                    and str(prior.get("event_type") or "") == "task_terminal"
                    and str((prior.get("payload") or {}).get("reason") or "").strip()
                    == str(event_payload.get("reason") or "").strip()
                )
                if (
                    str(prior.get("event_type") or "") == str(event_type)
                    and (prior_key == idempotency_key or legacy_terminal_match)
                ):
                    prior_status = str(
                        (prior.get("payload") or {}).get("status") or ""
                    ).strip()
                    next_status = str(event_payload.get("status") or "").strip()
                    if (
                        str(event_type) == "task_terminal"
                        and prior_status
                        and next_status
                        and prior_status != next_status
                    ):
                        correction_key = (
                            f"task_terminal_correction:{idempotency_key}:{next_status}"
                        )
                        correction_exists = any(
                            str(item.get("event_type") or "")
                            == "task_terminal_correction"
                            and str(
                                (item.get("payload") or {}).get(
                                    "idempotency_key"
                                )
                                or ""
                            ).strip()
                            == correction_key
                            for item in prior_events
                        )
                        if not correction_exists:
                            correction = {
                                "task_id": task_id,
                                "ts": _now(),
                                "event_type": "task_terminal_correction",
                                "payload": {
                                    **event_payload,
                                    "idempotency_key": correction_key,
                                    "previous_status": prior_status,
                                    "corrects": idempotency_key,
                                },
                            }
                            with open(path, "a", encoding="utf-8") as f:
                                f.write(
                                    json.dumps(
                                        correction,
                                        ensure_ascii=False,
                                        default=str,
                                    )
                                    + "\n"
                                )
                    return False
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return True


def publish_runtime_event(base_dir: str, task_id: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """Append one factual runtime observation."""
    record_runtime_event(base_dir, task_id, event_type, payload)


def record_stage_outcome(base_dir: str, task_id: str, outcome: StageOutcome) -> None:
    payload = outcome.to_dict()
    record_runtime_event(base_dir, task_id, "stage_outcome", payload)
    if outcome.degradation_level > 0 or outcome.status in {"degraded", "blocked", "failed", "inherited"}:
        append_degradation(
            base_dir,
            task_id,
            {
                "stage_name": outcome.stage_name,
                "status": outcome.status,
                "reason_code": outcome.reason_code,
                "degradation_level": outcome.degradation_level,
                "fallback_used": outcome.fallback_used,
            },
        )


def append_degradation(base_dir: str, task_id: str, degradation: Dict[str, Any]) -> RuntimeState:
    def _apply(state: RuntimeState) -> None:
        state.degradation_chain.append(dict(degradation))
    return update_runtime_state(base_dir, task_id, _apply)


def update_runtime_state(
    base_dir: str,
    task_id: str,
    updater: Callable[[RuntimeState], None],
) -> RuntimeState:
    # Apply the user mutation while domain_state's file lock is held.  Loading
    # before taking that lock lets two workers overwrite one another's runtime
    # changes even when each individual JSON replacement is atomic.
    updated: RuntimeState | None = None

    def _mutate(domain: Dict[str, Any]) -> None:
        nonlocal updated
        payload = dict(domain.get("runtime") or {})
        if payload:
            state = RuntimeState.from_dict(payload)
        else:
            task_config = dict(domain.get("task") or {})
            objective_metric = str(task_config.get("objective_metric") or DEFAULT_OBJECTIVE_METRIC)
            configured_mode = str(task_config.get("mode") or "").strip().lower()
            mode = configured_mode if configured_mode in {"strict", "exploration"} else "exploration"
            state = RuntimeState.empty(task_id, objective_metric=objective_metric, mode=mode)
        updater(state)
        state.touch()
        domain["runtime"] = state.to_dict()
        updated = state

    mutate_domain_state(base_dir, task_id, _mutate)
    assert updated is not None
    return updated


def sync_executor_snapshot(
    base_dir: str,
    task_id: str,
    snapshot: Dict[str, Any],
) -> RuntimeState:
    def _apply(state: RuntimeState) -> None:
        state.executor_config_snapshot = dict(snapshot or {})

    return update_runtime_state(base_dir, task_id, _apply)


def sync_best_baseline(base_dir: str, task_id: str, baseline_record: Dict[str, Any]) -> RuntimeState:
    def _apply(state: RuntimeState) -> None:
        state.baseline = Candidate.from_dict(baseline_record)
        policy = gate_policy(base_dir)
        metric = (
            state.baseline.objective_metric
            or state.objective_metric
            or DEFAULT_OBJECTIVE_METRIC
        )
        if (
            not state.current_best.candidate_id
            or _candidate_beats_existing(
                state.baseline,
                state.current_best,
                metric,
                min_relative_improvement=policy["min_relative_improvement"],
            )
        ):
            if state.current_best.candidate_id and state.current_best.candidate_id != state.baseline.candidate_id:
                state.consistency_warnings.append(
                    "Replaced current_best with better baseline during baseline sync "
                    f"on {metric}: {state.baseline.candidate_id}"
                )
            state.current_best = Candidate.from_dict(baseline_record or {})
        active = state.current_best if state.current_best and state.current_best.candidate_id else state.baseline
        if active and active.candidate_id:
            state.research["active_development_baseline_id"] = active.candidate_id
        if baseline_record.get("objective_metric"):
            state.objective_metric = str(baseline_record.get("objective_metric") or state.objective_metric)

    state = update_runtime_state(base_dir, task_id, _apply)
    payload = {
        "candidate_id": state.baseline.candidate_id,
        "display_name": state.baseline.display_name,
        "objective_metric": state.baseline.objective_metric or state.objective_metric,
        "metrics": dict(state.baseline.metrics or {}),
    }
    record_runtime_event(base_dir, task_id, "baseline_selected", payload)
    return state


def sync_current_best(base_dir: str, task_id: str, record: Dict[str, Any]) -> RuntimeState:
    def _apply(state: RuntimeState) -> None:
        candidate = Candidate.from_dict(record or {})
        policy = gate_policy(base_dir)
        metric = candidate.objective_metric or state.objective_metric or DEFAULT_OBJECTIVE_METRIC
        incumbent = state.current_best if state.current_best and state.current_best.candidate_id else state.baseline
        if _candidate_beats_existing(
            candidate,
            incumbent,
            metric,
            min_relative_improvement=policy["min_relative_improvement"],
        ):
            state.current_best = candidate
            if state.current_best and state.current_best.candidate_id:
                state.research["active_development_baseline_id"] = state.current_best.candidate_id
        else:
            state.consistency_warnings.append(
                "Ignored current_best update because candidate did not beat incumbent "
                f"on {metric}: {candidate.candidate_id}"
            )

    state = update_runtime_state(base_dir, task_id, _apply)
    payload = {
        "candidate_id": state.current_best.candidate_id,
        "display_name": state.current_best.display_name,
        "objective_metric": state.current_best.objective_metric or state.objective_metric,
        "metrics": dict(state.current_best.metrics or {}),
    }
    record_runtime_event(base_dir, task_id, "current_best", payload)
    return state


def _candidate_beats_existing(
    new_candidate: Candidate,
    existing: Optional[Candidate],
    objective_metric: str,
    *,
    min_relative_improvement: float,
) -> bool:
    if existing is None or not existing.candidate_id:
        return True
    metric = objective_metric or new_candidate.objective_metric or existing.objective_metric or DEFAULT_OBJECTIVE_METRIC
    new_value = (new_candidate.metrics or {}).get(metric)
    old_value = (existing.metrics or {}).get(metric)
    if not isinstance(new_value, (int, float)):
        return False
    if not isinstance(old_value, (int, float)):
        return True
    return is_better(float(new_value), float(old_value), get_metric_direction(metric), float(min_relative_improvement or 0.0))


def sync_provisional_best(base_dir: str, task_id: str, record: Dict[str, Any]) -> RuntimeState:
    """Sync a single-seed improved candidate as provisional_best (NOT verified).

    Unlike current_best, provisional_best does NOT represent scientific success.
    It is only used to record a single-seed improvement that has not yet passed
    multi-seed verification. It can never be exported as the verified best.
    """
    def _apply(state: RuntimeState) -> None:
        candidate = Candidate.from_dict(record or {})
        policy = gate_policy(base_dir)
        metric = candidate.objective_metric or state.objective_metric or DEFAULT_OBJECTIVE_METRIC
        if _candidate_beats_existing(
            candidate,
            state.provisional_best,
            metric,
            min_relative_improvement=policy["min_relative_improvement"],
        ):
            state.provisional_best = candidate
        else:
            state.consistency_warnings.append(
                "Ignored provisional_best update because candidate did not beat existing provisional "
                f"on {metric}: {candidate.candidate_id}"
            )

    return update_runtime_state(base_dir, task_id, _apply)


def promote_provisional_to_current_best(
    base_dir: str,
    task_id: str,
    verified_record: Optional[Dict[str, Any]] = None,
    *,
    clear_provisional: bool = True,
) -> RuntimeState:
    """Promote provisional_best to current_best after multi-seed verification passes.

    Only call this after seed verification confirms the improvement.
    """
    def _apply(state: RuntimeState) -> None:
        policy = gate_policy(base_dir)
        if verified_record:
            candidate = Candidate.from_dict(verified_record or {})
            metric = candidate.objective_metric or state.objective_metric or DEFAULT_OBJECTIVE_METRIC
            incumbent = state.current_best if state.current_best and state.current_best.candidate_id else state.baseline
            if _candidate_beats_existing(
                candidate,
                incumbent,
                metric,
                min_relative_improvement=policy["min_relative_improvement"],
            ):
                state.current_best = candidate
                if state.current_best and state.current_best.candidate_id:
                    state.research["active_development_baseline_id"] = state.current_best.candidate_id
            else:
                state.consistency_warnings.append(
                    "Ignored verified current_best promotion because candidate did not beat incumbent "
                    f"on {metric}: {candidate.candidate_id}"
                )
            if clear_provisional:
                state.provisional_best = None
            return
        if state.provisional_best is None or not state.provisional_best.candidate_id:
            return
        metric = state.provisional_best.objective_metric or state.objective_metric or DEFAULT_OBJECTIVE_METRIC
        incumbent = state.current_best if state.current_best and state.current_best.candidate_id else state.baseline
        if _candidate_beats_existing(
            state.provisional_best,
            incumbent,
            metric,
            min_relative_improvement=policy["min_relative_improvement"],
        ):
            state.current_best = state.provisional_best
            if state.current_best and state.current_best.candidate_id:
                state.research["active_development_baseline_id"] = state.current_best.candidate_id
        else:
            state.consistency_warnings.append(
                "Ignored provisional current_best promotion because candidate did not beat incumbent "
                f"on {metric}: {state.provisional_best.candidate_id}"
            )
        if clear_provisional:
            state.provisional_best = None

    return update_runtime_state(base_dir, task_id, _apply)


def sync_baseline_search_progress(base_dir: str, task_id: str, progress: Dict[str, Any]) -> RuntimeState:
    def _apply(state: RuntimeState) -> None:
        state.baseline_search_progress = dict(progress or {})

    state = update_runtime_state(base_dir, task_id, _apply)
    payload = {
        "status": progress.get("status"),
        "phase": progress.get("phase"),
        "completed": progress.get("completed"),
        "total": progress.get("total"),
        "strategy": progress.get("strategy"),
    }
    record_runtime_event(base_dir, task_id, "baseline_search_progress", payload)
    return state


def sync_baseline_diagnosis(base_dir: str, task_id: str, diagnosis: Dict[str, Any]) -> RuntimeState:
    """Persist the diagnosis summary as RuntimeState business fact."""
    def _apply(state: RuntimeState) -> None:
        state.baseline_diagnosis = dict(diagnosis or {})

    return update_runtime_state(base_dir, task_id, _apply)


def sync_task_stage(
    base_dir: str,
    task_id: str,
    *,
    stage: str,
    status: str,
    extra: Optional[Dict[str, Any]] = None,
) -> RuntimeState:
    def _apply(state: RuntimeState) -> None:
        state.current_stage = stage
        state.current_stage_status = status
        status_l = str(status or "").strip().lower()
        if stage == "completed" and status_l in {"completed", "completed_with_failed_rounds"}:
            state.task_status = status_l
            state.research["resume_recommended"] = False
            state.research.pop("error_type", None)
            state.research.pop("error_message", None)
            state.research.pop("interruption_path", None)
        elif status_l in {"failed", "blocked"}:
            state.task_status = status_l
        elif (
            status_l.endswith("_failed")
            or status_l.endswith("_interrupted")
            or status_l in {
                "api_error",
                "api_error_interrupted",
                "interrupted",
                "exception_interrupted",
                "experiment_failed",
                "variant_write_failed",
                "round_start_failed",
                "idea_rejected",
                "implementation_rejected",
                "idea_review_exhausted",
                "protocol_blocked",
            }
        ):
            state.task_status = "failed"
        else:
            state.task_status = "running"
        if extra:
            if "baseline_search" in extra and isinstance(extra["baseline_search"], dict):
                state.baseline_search_progress.update(dict(extra["baseline_search"] or {}))
            if "research" in extra and isinstance(extra["research"], dict):
                state.research.update(dict(extra["research"] or {}))
            if "best_baseline" in extra and isinstance(extra["best_baseline"], dict):
                state.baseline = Candidate.from_dict(extra["best_baseline"])

    state = update_runtime_state(base_dir, task_id, _apply)
    payload = {"stage": stage, "status": status, "extra": dict(extra or {})}
    record_runtime_event(base_dir, task_id, "task_stage", payload)
    return state


def sync_round_stage(
    base_dir: str,
    task_id: str,
    *,
    round_number: int,
    stage: str,
    status: str,
    action: str = "",
    result: Optional[Dict[str, Any]] = None,
) -> RuntimeState:
    def _apply(state: RuntimeState) -> None:
        state.research["current_round"] = round_number
        state.research["current_round_stage"] = stage
        if stage == "completed":
            completed = set(state.research.get("completed_rounds", []) or [])
            if status == "completed":
                completed.add(round_number)
            state.research["completed_rounds"] = sorted(completed)
            state.research["pending_round"] = None
        else:
            state.research["pending_round"] = round_number
        state.current_round = ResearchRoundState(
            round_number=round_number,
            round_id=format_research_id(round_number),
            current_stage=stage,
            status=status,
            action=action,
            best_variant_id=(result or {}).get("node_id"),
            best_variant_metric=(result or {}).get("objective_value"),
            payload=dict(result or {}),
        )

    state = update_runtime_state(base_dir, task_id, _apply)
    payload = {
        "round_number": round_number,
        "round_id": format_research_id(round_number),
        "stage": stage,
        "status": status,
        "action": action,
        "result": dict(result or {}),
    }
    record_runtime_event(base_dir, task_id, "round_stage", payload)
    return state


def sync_export_result(base_dir: str, task_id: str, export_result: Dict[str, Any]) -> RuntimeState:
    def _apply(state: RuntimeState) -> None:
        state.export = dict(export_result or {})

    return update_runtime_state(base_dir, task_id, _apply)


# ═══════════════════════════════════════════════════════════════════════════════
# P2-1: Heartbeat and crash detection
# ═══════════════════════════════════════════════════════════════════════════════

_HEARTBEAT_TIMEOUT_SECONDS = 300  # 5 minutes — same as prompt cache TTL

def record_heartbeat(base_dir: str, task_id: str, *, pid: int | None = None) -> RuntimeState:
    """P2-1: Write current timestamp + PID so dead processes can be detected."""
    def _apply(state: RuntimeState) -> None:
        state.research["heartbeat_at"] = _now()
        if pid is not None:
            state.research["heartbeat_pid"] = pid
    return update_runtime_state(base_dir, task_id, _apply)


def detect_stale_heartbeat(base_dir: str, task_id: str, *, timeout_seconds: int = _HEARTBEAT_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """P2-1: Check if the wizard process is dead.

    Returns {"stale": True, ...} if heartbeat is older than timeout_seconds
    or the PID is no longer alive.
    """
    state = load_runtime_state(base_dir, task_id, auto_migrate=False)
    heartbeat_at = (state.research or {}).get("heartbeat_at")
    heartbeat_pid = (state.research or {}).get("heartbeat_pid")
    current_stage = state.current_stage
    current_status = state.current_stage_status

    result: Dict[str, Any] = {
        "stale": False,
        "heartbeat_at": heartbeat_at,
        "heartbeat_pid": heartbeat_pid,
        "current_stage": current_stage,
        "current_status": current_status,
    }

    if not heartbeat_at:
        result["stale"] = current_status in {"running", "active"}
        result["reason"] = "no heartbeat recorded" if result["stale"] else "task not running"
        return result

    try:
        last_hb = datetime.fromisoformat(str(heartbeat_at).replace("Z", "+00:00"))
        age = (datetime.now() - last_hb.replace(tzinfo=None)).total_seconds()
    except Exception:
        result["stale"] = True
        result["reason"] = "unparseable heartbeat_at"
        return result

    if age > timeout_seconds:
        result["stale"] = True
        result["reason"] = f"heartbeat age {age:.0f}s exceeds timeout {timeout_seconds}s"

    # Check PID if available
    if heartbeat_pid and not result["stale"]:
        try:
            import ctypes
            import ctypes.wintypes
            SYNCHRONIZE = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, int(heartbeat_pid))
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
            else:
                result["stale"] = True
                result["reason"] = f"PID {heartbeat_pid} not found"
        except Exception:
            pass  # Non-Windows or permission error — skip PID check

    return result


def mark_task_interrupted(base_dir: str, task_id: str) -> RuntimeState:
    """P2-1: Mark a stale task as interrupted so dashboard shows correct state."""
    def _apply(state: RuntimeState) -> None:
        if state.current_stage_status in {"running", "active"}:
            state.current_stage_status = "interrupted"
            state.task_status = "interrupted"
            state.research["interrupted_at"] = _now()
    return update_runtime_state(base_dir, task_id, _apply)


def check_consecutive_no_progress(base_dir: str, task_id: str, *, max_consecutive: int = 3) -> Dict[str, Any]:
    """P2-2: Detect consecutive no-improvement rounds for auto-exit.

    Returns {"should_exit": True, ...} when max_consecutive is reached.
    """
    state = load_runtime_state(base_dir, task_id, auto_migrate=False)
    count = int((state.research or {}).get("consecutive_no_improvement") or 0)
    completed = list((state.research or {}).get("completed_rounds") or [])
    return {
        "should_exit": count >= max_consecutive,
        "consecutive_no_improvement": count,
        "max_consecutive": max_consecutive,
        "completed_rounds": completed,
        "completed_count": len(completed),
    }
