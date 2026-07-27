"""Pure mutations for attempt, gate, and seed-evaluation round transitions."""

from __future__ import annotations

from typing import Any, Callable, Dict

from evocast.domain.round_semantics import (
    FAILURE_KIND_SCIENTIFIC,
    FAILURE_KIND_VALIDATION,
    PHASE_CLOSED,
    TERMINAL_ROUND_STATUSES,
)


def apply_experiment_attempt(
    record: Dict[str, Any],
    run: Dict[str, Any],
    *,
    failure_transition: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    record["runs"] = [*list(record.get("runs") or []), dict(run)]
    if str(run.get("status") or "") == "success":
        record["status"] = "experiment_succeeded"
        record["failure_analysis"] = {}
        return record
    return failure_transition(record, run)


def apply_gate_event(
    record: Dict[str, Any],
    gate_event: Dict[str, Any],
) -> Dict[str, Any]:
    event = dict(gate_event or {})
    decision = event.get("decision")
    record["gate_decision"] = decision
    record["gate_event"] = event
    metrics = event.get("candidate_metrics")
    if isinstance(metrics, dict) and metrics:
        record["candidate_metrics"] = dict(metrics)
        record["metrics"] = dict(metrics)
    statuses = {
        "accept": "completed",
        "reject": "rejected",
        "marginal_no_seed_eval": "marginal_no_seed_eval",
        "needs_review": "gate_checked",
        "needs_seed_eval": "gate_checked",
    }
    if decision in statuses:
        record["status"] = statuses[decision]
    elif record.get("status") not in TERMINAL_ROUND_STATUSES:
        record["status"] = "gate_checked"
    return record


def apply_seed_result(
    record: Dict[str, Any],
    result: Dict[str, Any],
    *,
    recorded_at: str,
) -> Dict[str, Any]:
    decision = dict(result.get("significance_decision") or {})
    decision_value = str(decision.get("decision") or "").strip()
    promoted = bool(result.get("promoted_to_current_best"))
    record["seed_eval"] = {
        "status": "completed" if result.get("mean") is not None else "failed",
        "node_id": result.get("node_id"),
        "result_path": result.get("result_path"),
        "mean": result.get("mean"),
        "std": result.get("std"),
        "successful_seeds": result.get("successful_seeds"),
        "significance_decision": decision,
        "recorded_at": recorded_at,
        "promoted_to_current_best": promoted,
    }
    if result.get("promotion_decision"):
        record["seed_eval"]["promotion_decision"] = dict(
            result.get("promotion_decision") or {}
        )
    if decision_value == "accept" and promoted:
        record["status"] = "completed"
        record["gate_decision"] = "accept"
        record.pop("failure_kind", None)
        record["close_reason"] = "seed_eval accepted and promoted candidate"
    elif decision_value == "accept":
        record["status"] = "rejected"
        record["gate_decision"] = "reject"
        record["failure_kind"] = FAILURE_KIND_SCIENTIFIC
        record["close_reason"] = (
            "seed_eval accepted but did not promote current_best under the unified threshold"
        )
    else:
        record["status"] = "rejected"
        record["gate_decision"] = "reject"
        record["failure_kind"] = FAILURE_KIND_SCIENTIFIC
        record["close_reason"] = (
            f"seed_eval did not accept candidate: {decision_value or 'unknown'}"
        )
    record["phase"] = PHASE_CLOSED
    record["closed_at"] = recorded_at
    return record


def apply_seed_failure(
    record: Dict[str, Any],
    result: Dict[str, Any],
    *,
    recorded_at: str,
) -> Dict[str, Any]:
    error_type = str(result.get("error_type") or "SeedEvalError")
    error_message = str(
        result.get("error_message") or result.get("summary") or "seed_eval_failed"
    )
    record["seed_eval"] = {
        "status": "failed",
        "node_id": result.get("node_id"),
        "result_path": result.get("result_path"),
        "error_type": error_type,
        "error_message": error_message[:2000],
        "recorded_at": recorded_at,
        "promoted_to_current_best": False,
    }
    record["status"] = "experiment_failed"
    record["phase"] = PHASE_CLOSED
    record["close_reason"] = (
        f"seed_eval failed: {error_type}: {error_message[:500]}"
    )
    record["failure_kind"] = FAILURE_KIND_VALIDATION
    record["closed_at"] = recorded_at
    return record
