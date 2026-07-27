"""Deterministic task cockpit for the v3-style harness.

The dashboard is intentionally factual.  It summarizes persisted state for the
model, but it does not decide the next research action.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from evocast.domain.atomic_io import atomic_write_json
from evocast.state.runtime.candidate_registry import compact_candidate_history
from evocast.domain.execution_ids import format_research_id
from evocast.domain.knowledge_paths import task_knowledge_dir, runs_root
from evocast.state.runtime.store import load_runtime_state
from evocast.state.runtime.trial_journal import journal_summary, latest_nodes_by_id, read_journal
from evocast.harness.rounds import TERMINAL_ROUND_STATUSES, gate_review_obligations, list_rounds
from evocast.harness.rounds import seed_eval_obligations
from evocast.state.domain_store import load_task_config


def _now() -> str:
    return datetime.now().isoformat()


def dashboard_path(base_dir: str, task_id: str) -> Path:
    return task_knowledge_dir(base_dir, task_id) / "dashboard.json"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_jsonl_tail(path: Path, limit: int = 8) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows[-limit:]


def _task_config(base_dir: str, task_id: str) -> Dict[str, Any]:
    return load_task_config(base_dir, task_id)


def _compiled_config(base_dir: str, task_id: str) -> Dict[str, Any]:
    return dict(_read_json(task_knowledge_dir(base_dir, task_id) / "compiled_config.json", {}) or {})


def _recent_failures(
    nodes: List[Dict[str, Any]],
    limit: int = 5,
    *,
    exclude_node_ids: set[str] | None = None,
) -> List[Dict[str, Any]]:
    exclude = set(exclude_node_ids or set())
    failures = [
        node
        for node in nodes
        if node.get("status") == "failed" and str(node.get("node_id") or "") not in exclude
    ]
    compact: List[Dict[str, Any]] = []
    for node in failures[-limit:]:
        compact.append(
            {
                "node_id": node.get("node_id"),
                "model_name": node.get("model_name"),
                "variant_path": node.get("variant_path"),
                "error_type": node.get("error_type"),
                "error_message": str(node.get("error_message") or "")[:500],
                "created_at": node.get("created_at"),
            }
        )
    return compact


def _pending_obligations(base_dir: str, task_id: str, dashboard: Dict[str, Any]) -> List[str]:
    obligations: List[str] = []
    provisional = dict(dashboard.get("provisional_best") or {})
    if provisional.get("candidate_id"):
        obligations.append("provisional_best_requires_seed_verify")
        obligations.append("provisional_best_requires_mechanism_ablation_or_journaled_waiver")

    gate_events = _read_jsonl_tail(task_knowledge_dir(base_dir, task_id) / "gate_events.jsonl", limit=6)
    accepted = [event for event in gate_events if event.get("decision") == "accept"]
    if accepted and not provisional.get("candidate_id") and not dict(dashboard.get("current_best") or {}).get("candidate_id"):
        obligations.append("accepted_gate_event_not_synced_to_candidate_state")
    for item in seed_eval_obligations(base_dir, task_id):
        round_id = item.get("round_id")
        variant = item.get("variant_path") or item.get("candidate_name") or "candidate"
        decision = item.get("decision") or "seed_eval_required"
        obligations.append(f"{format_research_id(int(round_id or 0))}_{decision}_requires_seed_eval:{variant}")
    for item in gate_review_obligations(base_dir, task_id):
        if item.get("decision") in {"needs_review", "needs_seed_eval"}:
            continue
        round_id = item.get("round_id")
        fit_point = item.get("fit_point") or "unknown_fit_point"
        variant = item.get("variant_path") or item.get("candidate_name") or "candidate"
        obligations.append(f"{format_research_id(int(round_id or 0))}_needs_review_same_fit_point:{fit_point}:{variant}")
    return sorted(set(obligations))


def _compact_candidate(candidate: Dict[str, Any], objective_metric: str) -> Dict[str, Any]:
    if not candidate:
        return {}
    metrics = dict(candidate.get("metrics") or {})
    return {
        "candidate_id": candidate.get("candidate_id"),
        "node_id": candidate.get("node_id"),
        "display_name": candidate.get("display_name") or candidate.get("model_name"),
        "source": candidate.get("source"),
        "objective_metric": candidate.get("objective_metric") or objective_metric,
        "objective_value": metrics.get(candidate.get("objective_metric") or objective_metric),
        "scientific_status": candidate.get("scientific_status"),
        "engineering_status": candidate.get("engineering_status"),
        "requires_seed_verify": candidate.get("requires_seed_verify"),
        "requires_mechanism_ablation": candidate.get("requires_mechanism_ablation"),
    }


def _compact_gate_event(event: Dict[str, Any]) -> Dict[str, Any]:
    gate = dict(event.get("gate") or {})
    return {
        "run_id": event.get("run_id"),
        "variant_path": event.get("variant_path"),
        "decision": event.get("decision"),
        "objective_metric": event.get("objective_metric"),
        "candidate_value": event.get("candidate_value"),
        "baseline_value": event.get("baseline_value"),
        "incumbent_value": event.get("incumbent_value"),
        "relative_improvement": event.get("relative_improvement"),
        "evaluation_stage": event.get("evaluation_stage"),
        "reason": str(gate.get("reason") or event.get("reason") or "")[:240],
    }


def _compact_ablation(item: Dict[str, Any], objective_metric: str) -> Dict[str, Any]:
    delta = dict(item.get("metric_delta") or {})
    objective_delta = delta.get(objective_metric) if objective_metric else None
    return {
        "ablation_id": item.get("ablation_id"),
        "target_id": item.get("target_id"),
        "target_name": item.get("target_name"),
        "exact_edit_intent": item.get("exact_edit_intent"),
        "status": item.get("status"),
        "run_id": item.get("run_id"),
        "objective_delta": objective_delta,
        "usable_evidence_status": item.get("usable_evidence_status"),
        "created_at": item.get("created_at"),
    }


def _diagnosis_failures(diagnosis: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    failures = []
    for item in list(diagnosis.get("failed_ablations") or [])[-limit:]:
        if not isinstance(item, dict):
            continue
        failures.append(
            {
                "node_id": "baseline_diagnosis",
                "model_name": diagnosis.get("baseline_model"),
                "variant_path": None,
                "error_type": item.get("failure_type") or "baseline_diagnosis_partial",
                "error_message": str(item.get("reason") or item.get("failure_reason") or "")[:500],
                "created_at": diagnosis.get("updated_at"),
            }
        )
    return failures


def _failed_fit_points(base_dir: str, task_id: str, objective_metric: str) -> List[Dict[str, Any]]:
    from evocast.harness.rounds import (
        INFRA_FAILURE_STATUSES,
        FAILURE_KIND_SCIENTIFIC,
        classify_failure_kind,
    )

    groups: Dict[str, Dict[str, Any]] = {}
    for record in list_rounds(base_dir, task_id):
        if record.get("status") not in TERMINAL_ROUND_STATUSES:
            continue
        if record.get("status") == "completed":
            continue
        # P0-4: Only scientific_failure enters failed_fit_points.
        # implementation/activation/validation failures are engineering issues,
        # not research evidence that "this idea doesn't work".
        failure_kind = record.get("failure_kind") or classify_failure_kind(
            record.get("status") or "",
            str((record.get("failure_analysis") or {}).get("error_type") or ""),
        )
        if failure_kind != FAILURE_KIND_SCIENTIFIC:
            continue
        # Also exclude explicit infra statuses (belt and suspenders).
        if record.get("status") in INFRA_FAILURE_STATUSES:
            continue
        fit_point = str(record.get("fit_point") or "unknown")
        group = groups.setdefault(
            fit_point,
            {
                "fit_point": fit_point,
                "attempts": 0,
                "best_value": None,
                "best_relative_improvement": None,
                "variants_tried": [],
                "round_ids": [],
            },
        )
        group["attempts"] += 1
        group["round_ids"].append(record.get("round_id"))
        mechanism = str((record.get("proposal") or {}).get("modified_mechanism") or record.get("operation_family") or "")
        if mechanism and mechanism not in group["variants_tried"]:
            group["variants_tried"].append(mechanism)
        gate = dict((record.get("gate_event") or {}).get("gate") or {})
        value = gate.get("current_value")
        rel = gate.get("reference_relative_improvement")
        if isinstance(value, (int, float)):
            best = group.get("best_value")
            if best is None or value < best:
                group["best_value"] = value
                group["best_relative_improvement"] = rel
    summaries: List[Dict[str, Any]] = []
    for group in groups.values():
        variants = list(group.get("variants_tried") or [])
        attempts = int(group.get("attempts") or 0)
        conclusion = (
            f"{attempts} attempted mechanism(s) at {group['fit_point']} have not improved {objective_metric}: "
            + (", ".join(variants[:5]) if variants else "mechanisms unavailable")
            + ". This only rules against these tried mechanisms; a genuinely different, failure-grounded revision remains allowed."
        )
        summaries.append({**group, "conclusion": conclusion})
    return sorted(summaries, key=lambda item: int(item.get("attempts") or 0), reverse=True)


def compact_dashboard_payload(dashboard: Dict[str, Any], *, base_dir: str = "", task_id: str = "") -> Dict[str, Any]:
    objective_metric = str(dashboard.get("objective_metric") or "mse")
    successful_candidate_node_ids = {
        str(candidate.get("node_id") or "")
        for candidate in (
            dict(dashboard.get("baseline") or {}),
            dict(dashboard.get("current_best") or {}),
            dict(dashboard.get("provisional_best") or {}),
        )
        if candidate.get("node_id")
    }
    recent_failures = []
    for item in list(dashboard.get("recent_failures") or [])[-5:]:
        if str(item.get("node_id") or "") in successful_candidate_node_ids:
            continue
        recent_failures.append(
            {
                "node_id": item.get("node_id"),
                "variant_path": item.get("variant_path"),
                "error_type": item.get("error_type"),
                "error_message": str(item.get("error_message") or "")[:180],
            }
        )
    payload = {
        "generated_at": dashboard.get("generated_at"),
        "task_id": dashboard.get("task_id"),
        "objective_metric": objective_metric,
        "dataset": dashboard.get("dataset"),
        "runtime_stage": dashboard.get("runtime_stage"),
        "baseline": _compact_candidate(dict(dashboard.get("baseline") or {}), objective_metric),
        "current_best": _compact_candidate(dict(dashboard.get("current_best") or {}), objective_metric),
        "provisional_best": _compact_candidate(dict(dashboard.get("provisional_best") or {}), objective_metric),
        "pending_obligations": list(dashboard.get("pending_obligations") or []),
        "journal_summary": dashboard.get("journal_summary"),
        "recent_failures": recent_failures,
        "recent_gate_events": [
            _compact_gate_event(event)
            for event in list(dashboard.get("recent_gate_events") or [])[-5:]
            if event.get("decision") in {"accept", "needs_seed_eval", "needs_review", "reject"}
        ],
        "recent_ablation_results": [
            _compact_ablation(item, objective_metric)
            for item in list(dashboard.get("recent_ablation_results") or [])[-5:]
            if item.get("status") == "success"
        ],
        "baseline_diagnosis": dashboard.get("baseline_diagnosis"),
        "failed_fit_points": list(dashboard.get("failed_fit_points") or [])[:6],
    }
    if base_dir and task_id:
        payload["recent_candidates"] = compact_candidate_history(base_dir, task_id, limit=8)
    return payload


def build_dashboard(base_dir: str, task_id: str, *, persist: bool = True) -> Dict[str, Any]:
    state = load_runtime_state(base_dir, task_id, auto_migrate=False)
    task_config = _task_config(base_dir, task_id)
    compiled = _compiled_config(base_dir, task_id)
    data_config = dict(compiled.get("data_config") or {})
    semantics = dict(data_config.get("task_semantics") or task_config.get("task_semantics") or {})
    strategy_args = dict((compiled.get("evaluation_config") or {}).get("strategy_args") or {})
    journal_nodes = latest_nodes_by_id(read_journal(task_id, str(runs_root(base_dir))))
    try:
        summary = journal_summary(task_id, str(runs_root(base_dir)))
    except Exception:
        summary = {}

    successful_candidate_node_ids = {
        str(candidate.get("node_id") or "")
        for candidate in (
            state.baseline.to_dict() if state.baseline.candidate_id else {},
            state.current_best.to_dict() if state.current_best.candidate_id else {},
            state.provisional_best.to_dict() if state.provisional_best and state.provisional_best.candidate_id else {},
        )
        if candidate.get("node_id")
    }
    successful_journal_node_ids = {
        str(node.get("node_id") or "")
        for node in journal_nodes
        if node.get("status") == "success" and node.get("node_id")
    }

    diagnosis = dict(state.baseline_diagnosis or {})
    recent_failures = _recent_failures(
        journal_nodes,
        exclude_node_ids=successful_candidate_node_ids | successful_journal_node_ids,
    )
    recent_failures.extend(_diagnosis_failures(diagnosis))
    dashboard = {
        "generated_at": _now(),
        "task_id": task_id,
        "objective_metric": state.objective_metric,
        "metric_direction": task_config.get("metric_direction") or "lower_is_better",
        "dataset": {
            "path": data_config.get("dataset_path") or task_config.get("dataset_path") or semantics.get("dataset_path"),
            "data_set_name": data_config.get("data_set_name") or task_config.get("data_set_name"),
            "task_mode": semantics.get("task_mode"),
            "horizon": strategy_args.get("horizon") or task_config.get("horizon"),
            "seq_len": task_config.get("seq_len") or (compiled.get("model_config") or {}).get("recommend_model_hyper_params", {}).get("input_chunk_length"),
            "frequency": (data_config.get("feature_dict") or {}).get("canonical_freq") or semantics.get("frequency"),
        },
        "baseline": state.baseline.to_dict() if state.baseline.candidate_id else {},
        "current_best": state.current_best.to_dict() if state.current_best.candidate_id else {},
        "provisional_best": state.provisional_best.to_dict() if state.provisional_best and state.provisional_best.candidate_id else {},
        "runtime_stage": {
            "current_stage": state.current_stage,
            "current_stage_status": state.current_stage_status,
            "task_status": state.task_status,
            "mode": state.mode,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        },
        "journal_summary": summary,
        "recent_failures": recent_failures[-5:],
        "recent_gate_events": _read_jsonl_tail(task_knowledge_dir(base_dir, task_id) / "gate_events.jsonl", limit=5),
        "recent_ablation_results": _read_jsonl_tail(task_knowledge_dir(base_dir, task_id) / "rounds" / "ablation_results.jsonl", limit=5),
        "model_structure": {},
        "failed_fit_points": _failed_fit_points(base_dir, task_id, state.objective_metric or "mse"),
        "pending_obligations": [],
    }
    if isinstance(diagnosis, dict):
        dashboard["baseline_diagnosis"] = {
            "baseline_model": diagnosis.get("baseline_model"),
            "objective_metric": diagnosis.get("objective_metric"),
            "target_discovery": diagnosis.get("target_discovery"),
            "ablation_execution": diagnosis.get("ablation_execution"),
            "usable_evidence": diagnosis.get("usable_evidence"),
            "evidence_completeness": diagnosis.get("evidence_completeness"),
            "planned_ablation_count": diagnosis.get("planned_ablation_count", len(list((diagnosis.get("ablation_plan") or {}).get("targets") or []))),
            "executed_ablation_count": diagnosis.get("executed_ablation_count", len(list(diagnosis.get("ablations") or []))),
            "usable_ablation_count": diagnosis.get("usable_ablation_count", sum(1 for item in list(diagnosis.get("ablations") or []) if item.get("usable_evidence_status") == "usable_evidence")),
            "failed_ablation_count": diagnosis.get("failed_ablation_count", len(list(diagnosis.get("failed_ablations") or []))),
        }
    dashboard["pending_obligations"] = _pending_obligations(base_dir, task_id, dashboard)

    if persist:
        path = dashboard_path(base_dir, task_id)
        atomic_write_json(path, dashboard, ensure_ascii=False, default=str)
    return dashboard


def dashboard_block(base_dir: str, task_id: str) -> str:
    payload = build_dashboard(base_dir, task_id, persist=True)
    compact = compact_dashboard_payload(payload, base_dir=base_dir, task_id=task_id)
    return "<dashboard>\n" + json.dumps(compact, indent=2, ensure_ascii=False, default=str) + "\n</dashboard>"
