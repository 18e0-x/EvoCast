"""Read-only projections over canonical Research round records."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from evocast.domain.execution_ids import format_research_id
from evocast.domain.round_semantics import (
    FAILURE_KIND_ACTIVATION,
    FAILURE_KIND_IMPLEMENTATION,
    FAILURE_KIND_VALIDATION,
    GATE_RESOLUTION_DECISIONS,
    TERMINAL_ROUND_STATUSES,
    counts_as_completed_round,
    counts_as_valid_trial,
    counts_toward_research_budget,
    resolve_failure_kind,
)


def seed_eval_obligations(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    obligations: List[Dict[str, Any]] = []
    for record in records:
        if not counts_toward_research_budget(record):
            continue
        if record.get("status") in TERMINAL_ROUND_STATUSES:
            continue
        if record.get("gate_decision") not in GATE_RESOLUTION_DECISIONS:
            continue
        if dict(record.get("seed_eval") or {}).get("status") in {"completed", "failed"}:
            continue
        event = dict(record.get("gate_event") or {})
        obligations.append(
            {
                "round_id": record.get("round_id"),
                "variant_path": record.get("variant_path"),
                "run_id": event.get("run_id"),
                "candidate_name": event.get("candidate_name"),
                "objective_metric": event.get("objective_metric"),
                "candidate_value": event.get("candidate_value"),
                "baseline_value": event.get("baseline_value"),
                "decision": record.get("gate_decision"),
                "reason": ((event.get("gate") or {}).get("reason") or event.get("reason")),
                "required_action": "run_seed_eval",
            }
        )
    return obligations


def gate_review_obligations(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    obligations: List[Dict[str, Any]] = []
    for record in records:
        if not counts_toward_research_budget(record):
            continue
        if record.get("status") in TERMINAL_ROUND_STATUSES:
            continue
        if record.get("gate_decision") not in GATE_RESOLUTION_DECISIONS:
            continue
        if dict(record.get("seed_eval") or {}).get("status") in {"completed", "failed"}:
            continue
        event = dict(record.get("gate_event") or {})
        obligations.append(
            {
                "round_id": record.get("round_id"),
                "fit_point": record.get("fit_point"),
                "variant_path": record.get("variant_path"),
                "run_id": event.get("run_id"),
                "candidate_name": event.get("candidate_name"),
                "decision": record.get("gate_decision"),
                "objective_metric": event.get("objective_metric"),
                "candidate_value": event.get("candidate_value"),
                "baseline_value": event.get("baseline_value"),
                "reason": ((event.get("gate") or {}).get("reason") or event.get("reason")),
                "required_action": "run_seed_eval_before_same_fit_point",
            }
        )
    return obligations


def project_round_progress(
    records: List[Dict[str, Any]],
    *,
    current_round_id: Optional[int],
    current_research_id: str,
) -> Dict[str, Any]:
    budget = [r for r in records if counts_toward_research_budget(r)]
    diagnostic = [r for r in records if not counts_toward_research_budget(r)]
    all_closed = [r for r in records if r.get("status") in TERMINAL_ROUND_STATUSES]
    closed = [r for r in budget if r.get("status") in TERMINAL_ROUND_STATUSES]
    diagnostic_closed = [
        r for r in diagnostic if r.get("status") in TERMINAL_ROUND_STATUSES
    ]
    completed = [r for r in closed if counts_as_completed_round(r)]
    valid = [r for r in closed if counts_as_valid_trial(r)]
    scientific_rejections = [
        r
        for r in valid
        if str(r.get("status") or "") in {"rejected", "scientific_rejected"}
        or str(r.get("gate_decision") or "") in {"reject", "rejected"}
    ]
    rejection_ids = {int(r.get("round_id") or 0) for r in scientific_rejections}
    effective = [
        r for r in valid if int(r.get("round_id") or 0) not in rejection_ids
    ]
    build_failures = [
        r
        for r in closed
        if str(r.get("status") or "") != "completed"
        and str(r.get("failure_kind") or resolve_failure_kind(r))
        in {FAILURE_KIND_IMPLEMENTATION, FAILURE_KIND_ACTIVATION}
    ]
    runtime_failures = [
        r
        for r in closed
        if str(r.get("status") or "") != "completed"
        and str(r.get("failure_kind") or resolve_failure_kind(r))
        == FAILURE_KIND_VALIDATION
    ]
    attempts = [
        dict(run or {})
        for record in budget
        for run in list(record.get("runs") or [])
        if str((run or {}).get("run_id") or "").strip()
    ]
    successful_attempts = [
        run for run in attempts if str(run.get("status") or "") == "success"
    ]
    metric_attempts = [
        run for run in successful_attempts if bool(run.get("metrics") or {})
    ]
    failed_attempts = [
        run for run in attempts if str(run.get("status") or "") == "failed"
    ]
    latest_round = max([int(r.get("round_id") or 0) for r in records] or [0])
    latest_research = max([int(r.get("round_id") or 0) for r in budget] or [0])
    return {
        "total_rounds": len(budget),
        "all_round_records": len(records),
        "research_rounds": len(budget),
        "diagnostic_rounds": len(diagnostic),
        "terminal_rounds": len(closed),
        "all_terminal_rounds": len(all_closed),
        "diagnostic_terminal_rounds": 0,
        "completed_rounds": len(completed),
        "completed_round_ids": [int(r.get("round_id") or 0) for r in completed],
        "valid_trial_rounds": len(valid),
        "successful_rounds": len(valid),
        "effective_metric_rounds": len(effective),
        "scientific_rejection_rounds": len(scientific_rejections),
        "build_failure_rounds": len(build_failures),
        "runtime_failure_rounds": len(runtime_failures),
        "valid_trial_round_ids": [int(r.get("round_id") or 0) for r in valid],
        "closed_rounds": len(all_closed),
        "research_closed_rounds": len(closed),
        "failed_metric_production_rounds": len(closed) - len(valid),
        "metric_production_rate": (len(valid) / len(closed)) if closed else 0.0,
        "variant_attempts": len(attempts),
        "successful_variant_attempts": len(successful_attempts),
        "failed_variant_attempts": len(failed_attempts),
        "metric_variant_attempts": len(metric_attempts),
        "attempt_metric_production_rate": (
            len(metric_attempts) / len(attempts) if attempts else 0.0
        ),
        "attempt_failure_rate": (
            len(failed_attempts) / len(attempts) if attempts else 0.0
        ),
        "open_rounds": len(records) - len(all_closed),
        "research_open_rounds": len(budget) - len(closed),
        "diagnostic_open_rounds": len(diagnostic) - len(diagnostic_closed),
        "latest_round_id": latest_round,
        "latest_research_id": (
            format_research_id(latest_research) if latest_research > 0 else None
        ),
        "records": records,
        "current_round_id": current_round_id,
        "current_research_id": current_research_id
        or (format_research_id(current_round_id) if current_round_id else None),
        "valid_trial_research_ids": [
            str(
                r.get("research_id")
                or format_research_id(int(r.get("round_id") or 0))
            )
            for r in valid
        ],
    }
