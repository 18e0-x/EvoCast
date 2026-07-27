"""Structured exact-edit ablation helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from evocast.domain.execution_ids import ROUNDS_DIR, format_ablation_id, parse_ablation_id
from evocast.research.ablation.policy import classify_seed_eval_ablation_effect
from evocast.policy.experiment_policy import (
    baseline_diagnosis_policy,
    normalize_budget,
    task_build_mode,
)
from evocast.policy.gate import get_metric_direction
from evocast.state.runtime.store import load_runtime_state, sync_baseline_diagnosis
from evocast.harness.ablation_round import run_ablation_round
from evocast.harness.session import AgentSession


class AblationToolError(ValueError):
    """Raised when ablation evidence cannot be safely recorded."""


def next_ablation_index(session: AgentSession) -> int:
    """Return the next numeric ablation id from the canonical rounds directory."""
    root = session.knowledge_dir / ROUNDS_DIR
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in root.iterdir():
        if path.is_dir():
            parsed = parse_ablation_id(path.name)
            if parsed is not None:
                existing.append(parsed)
    return max(existing or [0]) + 1


def _ablation_dir(session: AgentSession) -> Path:
    path = session.knowledge_dir / ROUNDS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ablation_index_path(session: AgentSession) -> Path:
    return _ablation_dir(session) / "ablation_results.jsonl"


def _diagnosis_path(session: AgentSession) -> Path:
    return session.knowledge_dir / "baseline_diagnosis" / "diagnosis_summary.json"


def _target_dir(session: AgentSession, ablation_id: str) -> Path:
    path = _ablation_dir(session) / str(ablation_id or "unknown_ablation")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return str(path)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _record_to_compact(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ablation_id": record.get("ablation_id"),
        "ablation_index": record.get("ablation_index"),
        "target_id": record.get("target_id"),
        "target_name": record.get("target_name"),
        "mechanism_id": record.get("mechanism_id"),
        "mechanism_name": record.get("mechanism_name"),
        "causal_variable": record.get("causal_variable"),
        "exact_edit_intent": record.get("exact_edit_intent"),
        "evidence_files": record.get("evidence_files"),
        "evidence_anchors": record.get("evidence_anchors"),
        "status": record.get("status"),
        "execution_status": record.get("execution_status"),
        "gate_decision": record.get("gate_decision"),
        "scientific_decision": record.get("scientific_decision"),
        "failure_type": record.get("failure_type"),
        "failure_reason": record.get("failure_reason"),
        "error": record.get("error"),
        "traceback": record.get("traceback"),
        "usable_evidence_status": record.get("usable_evidence_status"),
        "evaluation_stage": record.get("evaluation_stage"),
        "execution_surface": record.get("execution_surface"),
        "metrics": record.get("metrics"),
        "metrics_source": record.get("metrics_source"),
        "final_attempt_metrics": record.get("final_attempt_metrics"),
        "best_diagnostic_attempt": record.get("best_diagnostic_attempt"),
        "best_diagnostic_metrics": record.get("best_diagnostic_metrics"),
        "metric_delta": record.get("metric_delta"),
        "interpretation": record.get("interpretation"),
        "seed_eval": record.get("seed_eval"),
        "exact_patch_audit": record.get("exact_patch_audit"),
        "exact_ablation_target": record.get("exact_ablation_target"),
        "variant_path": record.get("variant_path"),
        "artifact_paths": record.get("artifact_paths"),
        "created_at": record.get("created_at"),
    }


def _active_state(session: AgentSession):
    return load_runtime_state(session.base_dir, session.task_id, auto_migrate=False)


def _baseline_model_record(session: AgentSession) -> Dict[str, Any]:
    state = _active_state(session)
    return state.baseline.to_dict() if state.baseline and state.baseline.candidate_id else {}


def active_baseline_record(session: AgentSession) -> Dict[str, Any]:
    state = _active_state(session)
    if state.current_best and state.current_best.candidate_id:
        return state.current_best.to_dict()
    return state.baseline.to_dict() if state.baseline and state.baseline.candidate_id else {}


def active_baseline_metrics(
    session: AgentSession,
    *,
    fallback_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    record = active_baseline_record(session)
    return dict(record.get("metrics") or record.get("best_metrics") or fallback_metrics or {})


def write_ablation_record(session: AgentSession, record: Dict[str, Any]) -> str:
    ablation_id = str(record.get("ablation_id") or "").strip()
    if not ablation_id:
        return ""
    path = _target_dir(session, ablation_id) / "ablation_record.json"
    _write_json(path, record)
    return str(path)


def summarize_architecture_evidence(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "protected_mechanisms": [],
        "removed_mechanisms_in_current_best": [],
        "open_design_space_mechanisms": [],
        "pending_seed_eval_removal_candidates": [],
        "inconclusive_mechanisms": [],
    }
    for item in results:
        if not isinstance(item, dict):
            continue
        interpretation = dict(item.get("interpretation") or {})
        seed_eval = dict(item.get("seed_eval") or {})
        mechanism_id = str(item.get("mechanism_id") or "").strip()
        if not mechanism_id:
            continue
        payload = {
            "target_id": item.get("target_id"),
            "mechanism_id": mechanism_id,
            "mechanism_name": item.get("mechanism_name"),
            "module_effect": seed_eval.get("module_effect") or interpretation.get("module_effect"),
            "recommended_action": seed_eval.get("recommended_action") or interpretation.get("recommended_action"),
            "development_policy": seed_eval.get("development_policy") or interpretation.get("development_policy"),
        }
        effect = str(payload.get("module_effect") or "")
        if effect in {"essential", "confirmed_essential"}:
            buckets["protected_mechanisms"].append(payload)
        elif effect == "confirmed_harmful_or_redundant":
            buckets["removed_mechanisms_in_current_best"].append(payload)
        elif effect == "harmful_or_redundant_candidate":
            buckets["pending_seed_eval_removal_candidates"].append(payload)
        elif effect in {"weak_or_uncertain", "seed_eval_weak_or_uncertain"}:
            buckets["open_design_space_mechanisms"].append(payload)
        else:
            buckets["inconclusive_mechanisms"].append(payload)
    return buckets


def classify_seed_eval_for_ablation(
    *,
    objective_metric: str,
    seed_eval_result: Dict[str, Any],
    base_dir: str | None = None,
) -> Dict[str, Any]:
    policy = baseline_diagnosis_policy(base_dir)
    threshold = float(policy["ablation_effect_threshold"])
    decision = str(dict(seed_eval_result.get("significance_decision") or {}).get("decision") or "").strip().lower()
    ready = (
        isinstance(seed_eval_result.get("mean"), (int, float))
        and isinstance(seed_eval_result.get("reference_mean"), (int, float))
        and int(seed_eval_result.get("valid_metric_seeds") or seed_eval_result.get("successful_seeds") or 0) >= 1
    )
    payload = classify_seed_eval_ablation_effect(
        objective_metric=objective_metric,
        metric_direction=get_metric_direction(objective_metric),
        reference_value=seed_eval_result.get("reference_mean"),
        candidate_value=seed_eval_result.get("mean"),
        threshold=threshold,
        seed_eval_ready=ready,
        failure_reason=str(
            seed_eval_result.get("error")
            or seed_eval_result.get("failure_reason")
            or ("" if ready else f"seed_eval_not_ready:{decision or 'missing_decision'}")
        ),
    )
    if ready and payload.get("module_effect") == "confirmed_harmful_or_redundant" and decision != "accept":
        payload.update(
            {
                "module_effect": "seed_eval_weak_or_uncertain",
                "recommended_action": "keep_current_baseline",
                "policy": "available_design_space",
                "reason": f"positive_seed_eval_not_accepted:{decision or 'missing_decision'}",
            }
        )
    return payload


def persist_ablation_plan(session: AgentSession, plan: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, str]:
    ablation_dir = _ablation_dir(session)
    plan_name = "mechanism_ablation_plan.json" if str(plan.get("schema_version") or "").startswith("mechanism_") else "ablation_plan.json"
    review_name = "mechanism_ablation_plan_review.json" if str(review.get("schema_version") or "").startswith("mechanism_") else "ablation_plan_review.json"
    plan_path = ablation_dir / plan_name
    review_path = ablation_dir / review_name
    _write_json(plan_path, plan)
    _write_json(review_path, review)
    artifact_dir = session.knowledge_dir / "baseline_diagnosis"
    artifact_plan_path = artifact_dir / plan_name
    artifact_review_path = artifact_dir / review_name
    _write_json(artifact_plan_path, plan)
    _write_json(artifact_review_path, review)
    return {
        "plan_path": str(plan_path),
        "review_path": str(review_path),
        "baseline_plan_path": str(artifact_plan_path),
        "baseline_review_path": str(artifact_review_path),
    }


def finalize_baseline_diagnosis(
    session: AgentSession,
    *,
    baseline_model: str,
    objective_metric: str,
    baseline_metrics: Dict[str, Any],
    reference_metrics: Dict[str, Any] | None = None,
    plan: Dict[str, Any],
    review: Dict[str, Any],
    results: List[Dict[str, Any]],
    mechanism_understanding: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    compact_results = [_record_to_compact(dict(item)) for item in results if isinstance(item, dict)]
    planned_targets = list(plan.get("targets") or [])
    reviewed_targets = list(review.get("reviewed_targets") or [])
    executed = [item for item in compact_results if item.get("status")]
    successful = [item for item in compact_results if item.get("status") == "success"]
    failed = [item for item in compact_results if item.get("status") != "success"]
    usable = [item for item in compact_results if item.get("usable_evidence_status") == "usable_evidence"]
    policy_results = [dict(item.get("interpretation") or {}) for item in compact_results if isinstance(item, dict)]
    essential_count = sum(1 for item in policy_results if item.get("module_effect") == "essential")
    harmful_candidate_count = sum(1 for item in policy_results if item.get("module_effect") == "harmful_or_redundant_candidate")
    weak_count = sum(1 for item in policy_results if item.get("module_effect") == "weak_or_uncertain")
    seed_eval_triggered_count = sum(1 for item in compact_results if dict(item.get("seed_eval") or {}).get("triggered"))
    seed_eval_promoted_count = sum(1 for item in compact_results if dict(item.get("seed_eval") or {}).get("promoted_to_current_best"))
    evidence_policy = summarize_architecture_evidence([dict(item) for item in results if isinstance(item, dict)])
    active_baseline = active_baseline_record(session)
    comparison_reference_metrics = dict(reference_metrics or (active_baseline.get("metrics") or active_baseline.get("best_metrics") or baseline_metrics or {}))
    diagnosis_policy = baseline_diagnosis_policy(session.base_dir)
    ablation_effect_threshold = float(diagnosis_policy["ablation_effect_threshold"])

    if successful and not failed:
        execution_status = "success"
    elif successful and failed:
        execution_status = "partial"
    elif failed:
        execution_status = "failed"
    else:
        execution_status = "not_started"

    review_status = str(review.get("status") or "").strip()
    if review_status == "skipped":
        target_discovery_status = "skipped"
    elif review_status == "rejected":
        target_discovery_status = "rejected"
    elif reviewed_targets:
        target_discovery_status = "success"
    elif planned_targets:
        target_discovery_status = "insufficient_executable_targets"
    else:
        target_discovery_status = "failed"

    if review_status == "skipped":
        diagnosis_status = "skipped"
    elif review_status == "rejected":
        diagnosis_status = "invalid_plan"
    elif usable and execution_status == "success" and len(usable) == len(reviewed_targets) and reviewed_targets:
        diagnosis_status = "completed"
    elif usable:
        diagnosis_status = "partial"
    elif executed:
        diagnosis_status = "failed_evidence"
    elif reviewed_targets:
        diagnosis_status = "not_started"
    else:
        diagnosis_status = "invalid_plan" if planned_targets else "not_started"

    payload = {
        "schema_version": "baseline_diagnosis_v3",
        "status": diagnosis_status,
        "task_id": session.task_id,
        "baseline_diagnosis_path": str(_diagnosis_path(session)),
        "baseline_model": baseline_model,
        "objective_metric": objective_metric,
        "baseline_metrics": baseline_metrics,
        "reference_kind": "current_best",
        "reference_metrics": comparison_reference_metrics,
        "current_best_reference_metrics": comparison_reference_metrics,
        "reference_kind": "current_best",
        "reference_metrics": baseline_metrics,
        "evaluation_stage": plan.get("evaluation_stage") or review.get("evaluation_stage"),
        "planned_ablation_count": len(planned_targets),
        "reviewed_targets": reviewed_targets,
        "executed_ablation_count": len(executed),
        "usable_ablation_count": len(usable),
        "failed_ablation_count": len(failed),
        "target_discovery": {
            "status": target_discovery_status,
            "target_count": len(planned_targets),
            "reviewed_target_count": len(reviewed_targets),
        },
        "ablation_execution": {
            "status": execution_status,
            "successful_count": len(successful),
            "failed_count": len(failed),
        },
        "usable_evidence": {
            "status": "available" if usable else "missing",
            "count": len(usable),
        },
        "plan_review": {
            "status": review.get("status"),
            "review_errors": list(review.get("errors") or []),
            "review_corrections": list(review.get("corrections") or []),
        },
        "ablation_policy_summary": {
            "threshold": ablation_effect_threshold,
            "essential_count": essential_count,
            "harmful_or_redundant_candidate_count": harmful_candidate_count,
            "weak_or_uncertain_count": weak_count,
            "seed_eval_triggered_count": seed_eval_triggered_count,
            "seed_eval_promoted_count": seed_eval_promoted_count,
        },
        "mechanism_development_policy": [
            {
                "target_id": item.get("target_id"),
                "mechanism_id": item.get("mechanism_id"),
                "mechanism_name": item.get("mechanism_name"),
                "module_effect": dict(item.get("seed_eval") or {}).get("module_effect")
                or dict(item.get("interpretation") or {}).get("module_effect"),
                "recommended_action": dict(item.get("seed_eval") or {}).get("recommended_action")
                or dict(item.get("interpretation") or {}).get("recommended_action"),
                "development_policy": dict(item.get("seed_eval") or {}).get("development_policy")
                or dict(item.get("interpretation") or {}).get("development_policy"),
            }
            for item in compact_results
        ],
        "architecture_evidence_policy": evidence_policy,
        "active_baseline_after_diagnosis": active_baseline,
        "mechanism_understanding": dict(mechanism_understanding or plan.get("mechanism_understanding") or {}),
        "ablation_plan": plan,
        "ablation_plan_review": review,
        "ablations": compact_results,
        "created_at": datetime.now().isoformat(),
    }
    sync_baseline_diagnosis(session.base_dir, session.task_id, payload)
    _write_json(_diagnosis_path(session), payload)
    _write_json(_ablation_dir(session) / "diagnosis_summary.json", payload)
    return payload


def _persist_record(session: AgentSession, record: Dict[str, Any]) -> None:
    compact = _record_to_compact(record)
    _append_jsonl(_ablation_index_path(session), compact)


def _forbidden_field_errors(args: Dict[str, Any], target: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for field in (
        "ablation_kind",
        "fallback_ablation_kind",
        "component_path",
        "canonical_component_path",
        "display_component_path",
        "minimal_counterfactual",
        "mechanism_family",
        "exact_edit_strategy",
        "target_files",
        "target_classes",
        "edit_anchors",
    ):
        if field in args or field in target:
            errors.append(f"forbidden_field:{field}")
    return errors


def _stage_from_args(budget: str, smoke: bool) -> str:
    if smoke:
        return "smoke"
    if str(budget or "").lower() == "seed_eval":
        return "seed_eval"
    return "experiment"


def run_ablation(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    target = dict(args.get("target") or {})
    target_id = str(args.get("target_id") or target.get("target_id") or "").strip() or "unknown_target"
    ablation_id = str(args.get("ablation_id") or "").strip() or format_ablation_id(next_ablation_index(session))
    ablation_index = int(parse_ablation_id(ablation_id) or 0)
    target_kind = str(args.get("target_kind") or target.get("target_kind") or "").strip()
    model_key = str(args.get("model_key") or args.get("baseline_model") or "").strip()
    reference_metrics = dict(args.get("reference_metrics") or args.get("current_best_metrics") or {})
    objective_metric = str(args.get("objective_metric") or "mse_norm")
    budget = normalize_budget(args.get("budget") or "unified", session.base_dir)
    smoke = bool(args.get("smoke", False) or task_build_mode(session.base_dir, session.task_id))
    candidate_stage = str(args.get("candidate_stage") or "").strip().lower()
    if candidate_stage == "standard":
        candidate_stage = "experiment"
    evaluation_stage = _stage_from_args(budget, smoke)
    forbidden_field_errors = _forbidden_field_errors(args, target)
    if forbidden_field_errors:
        record = {
            "ablation_id": ablation_id,
            "ablation_index": ablation_index,
            "target_id": target_id,
            "target_name": str(args.get("target_name") or target_id),
            "mechanism_id": str(args.get("mechanism_id") or target.get("mechanism_id") or ""),
            "mechanism_name": str(args.get("mechanism_name") or target.get("mechanism_name") or ""),
            "exact_edit_intent": str(args.get("exact_edit_intent") or target.get("exact_edit_intent") or ""),
            "status": "failed_ablation_round",
            "failure_type": "invalid_ablation_task",
            "failure_reason": "; ".join(forbidden_field_errors),
            "usable_evidence_status": "failed_evidence",
            "artifact_paths": [],
            "created_at": datetime.now().isoformat(),
        }
        _persist_record(session, record)
        return {
            "status": "failed",
            "record": record,
            "ablation_results_path": str(_ablation_index_path(session)),
            "baseline_diagnosis_path": str(_diagnosis_path(session)),
        }
    if candidate_stage and candidate_stage != evaluation_stage:
        record = {
            "ablation_id": ablation_id,
            "ablation_index": ablation_index,
            "target_id": target_id,
            "target_name": str(args.get("target_name") or target_id),
            "mechanism_id": str(args.get("mechanism_id") or target.get("mechanism_id") or ""),
            "mechanism_name": str(args.get("mechanism_name") or target.get("mechanism_name") or ""),
            "exact_edit_intent": str(args.get("exact_edit_intent") or target.get("exact_edit_intent") or ""),
            "status": "failed_ablation_round",
            "failure_type": "stage_mismatch",
            "failure_reason": (
                f"candidate_stage={candidate_stage}, requested_stage={evaluation_stage}. "
                "Ablation execution was blocked before run start."
            ),
            "usable_evidence_status": "failed_evidence",
            "artifact_paths": [],
            "created_at": datetime.now().isoformat(),
        }
        _persist_record(session, record)
        return {
            "status": "failed",
            "record": record,
            "ablation_results_path": str(_ablation_index_path(session)),
            "baseline_diagnosis_path": str(_diagnosis_path(session)),
        }
    if not model_key:
        raise AblationToolError("run_ablation requires model_key or baseline_model")
    if target_kind != "mechanism_ablation":
        raise AblationToolError("run_ablation requires target_kind=mechanism_ablation")

    ablation_task = {
        **target,
        "ablation_id": ablation_id,
        "target_id": target_id,
        "target_kind": target_kind,
        "mechanism_id": str(args.get("mechanism_id") or target.get("mechanism_id") or ""),
        "mechanism_name": str(args.get("mechanism_name") or target.get("mechanism_name") or ""),
        "causal_variable": str(args.get("causal_variable") or target.get("causal_variable") or ""),
        "diagnosis_question": str(target.get("diagnosis_question") or args.get("ablation_description") or ""),
        "exact_edit_intent": str(args.get("exact_edit_intent") or target.get("exact_edit_intent") or ""),
        "edit_spec": dict(args.get("edit_spec") or target.get("edit_spec") or {}),
        "evidence_files": list(target.get("evidence_files") or args.get("evidence_files") or []),
        "evidence_anchors": list(target.get("evidence_anchors") or args.get("evidence_anchors") or []),
        "runtime_paths": list(target.get("runtime_paths") or args.get("runtime_paths") or []),
        "preserve_contract": dict(args.get("preserve_contract") or target.get("preserve_contract") or {}),
        "expected_behavior_delta": str(
            args.get("expected_behavior_delta")
            or target.get("expected_behavior_delta")
            or "same-input forecast should change if this mechanism is active"
        ),
        "risk": str(target.get("risk") or "medium"),
        "evaluation_stage": evaluation_stage,
    }
    result = run_ablation_round(
        session,
        ablation_task,
        model_key=model_key,
        objective_metric=objective_metric,
        reference_metrics=reference_metrics,
        budget=budget,
        seed=int(args.get("seed") or 2021),
        max_repair_attempts=int(
            args.get("max_repair_attempts")
            if args.get("max_repair_attempts") is not None
            else baseline_diagnosis_policy(session.base_dir).get("ablation_repair_attempts", 3)
        ),
    )
    record = dict(result.get("record") or {})
    if record:
        record_path = write_ablation_record(session, record)
        if record_path:
            record["artifact_paths"] = list(record.get("artifact_paths") or []) + [record_path]
            write_ablation_record(session, record)
        _persist_record(session, record)
    return {
        "status": result.get("status") or ("ok" if record.get("status") == "success" else "failed"),
        "record": record,
        "ablation_results_path": str(_ablation_index_path(session)),
        "baseline_diagnosis_path": str(_diagnosis_path(session)),
    }


def read_ablation_results(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    rows = _read_jsonl(_ablation_index_path(session))
    limit = int(args.get("limit") or 20)
    if limit > 0:
        rows = rows[-limit:]
    diagnosis = dict(load_runtime_state(session.base_dir, session.task_id).baseline_diagnosis or {})
    return {
        "status": "ok",
        "count": len(rows),
        "ablations": rows,
        "baseline_diagnosis": diagnosis,
        "ablation_results_path": str(_ablation_index_path(session)),
        "baseline_diagnosis_path": str(_diagnosis_path(session)),
    }
