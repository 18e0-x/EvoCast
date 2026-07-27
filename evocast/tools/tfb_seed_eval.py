"""Seed-evaluation and promotion tools for EvoCast v3."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from evocast.state.runtime.candidate_registry import record_candidate
from evocast.research.baseline_reference import require_baseline_reference
from evocast.policy.cost_profile import ERROR_STATUS, SUCCESS_STATUS
from evocast.state.cost_ledger import record_execution_cost
from evocast.policy.model_contract import training_hparam_overrides, validate_model_config
from evocast.policy.experiment_policy import fixed_seed_list
from evocast.domain.effective_model_config import resolve_effective_model_config
from evocast.state.runtime.store import load_runtime_state, promote_provisional_to_current_best, sync_current_best
from evocast.domain.task_identity import resolve_compiled_config_path
from evocast.harness.session import AgentSession
from evocast.evaluation.decision_kernel import EvaluationDecisionKernel
from evocast.runners.seed_runner import run_seed_evaluation
from evocast.runners.tfb_pipeline_runner import load_config_json
from evocast.tools.tfb_experiment import (
    _display_model_name,
    _normalize_model_name,
    _objective_metric,
    _registry_model_entry,
    _runtime_model_name_for_variant,
    _stable_hash,
)
from evocast.state.cost_ledger import tracked_stage


class SeedEvalToolError(ValueError):
    """Raised when seed evaluation or promotion cannot be safely performed."""


def _build_seed_eval_model_entry(session: AgentSession, args: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    config_path = resolve_compiled_config_path(session.task_id, session.base_dir)
    tfb_config = load_config_json(config_path)
    variant_path = str(args.get("variant_path") or "").strip()
    model_key = str(args.get("model_key") or "").strip()
    model_name = str(args.get("model_name") or model_key).strip()
    registry_entry = _registry_model_entry(model_name)
    state = load_runtime_state(session.base_dir, session.task_id, auto_migrate=False)
    baseline = state.baseline.to_dict() if state.baseline.candidate_id else {}
    if variant_path:
        model_name = _runtime_model_name_for_variant(
            registry_entry=registry_entry,
            baseline=baseline,
            fallback_model_name=model_name,
        )
    else:
        model_name = _normalize_model_name(registry_entry.get("model_name") or model_name)
    if not model_name:
        raise SeedEvalToolError("run_seed_eval requires variant_path or model_name")
    display_model_name = _display_model_name(model_name, variant_path)

    base_hparams = (
        dict((baseline.get("model_config") or {}).get("model_hyper_params") or {})
        if variant_path
        else {}
    )
    source_model_key = (
        model_key
        or str(registry_entry.get("model_key") or "")
        or str(baseline.get("display_name") or "")
        or str(baseline.get("model_name") or "")
    ).strip()
    explicit_hparams = dict(args.get("model_hyper_params") or {})
    raw_entry = {
        "model_key": source_model_key,
        "model_name": model_name,
        "adapter": args.get("adapter") or baseline.get("adapter") or registry_entry.get("adapter") or None,
        "model_hyper_params": explicit_hparams,
    }
    recommend_hparams = dict((tfb_config.get("model_config") or {}).get("recommend_model_hyper_params") or {})
    resolved = resolve_effective_model_config(
        config_data=tfb_config,
        base_dir=session.base_dir,
        task_id=session.task_id,
        model_entry=raw_entry,
        baseline_model_config=(baseline.get("model_config") or {}) if variant_path else {},
        explicit_model_hyper_params=explicit_hparams,
        requested_budget="unified",
        smoke=False,
    )
    entry = resolved.entry
    if variant_path:
        entry["variant_path"] = variant_path
    hparams = dict(resolved.effective_model_hyper_params)
    training_budget = resolved.policy_budget
    preflight = validate_model_config(
        entry,
        recommend_model_hyper_params=recommend_hparams,
        baseline_validated_hyper_params=base_hparams,
        require_import=True,
    )
    provenance = {
        "config_hash": _stable_hash(entry),
        "display_model_name": display_model_name,
        "explicit_model_hyper_params": explicit_hparams,
        "applied_model_hyper_params": hparams,
        "effective_model_hyper_params": hparams,
        "policy_budget": training_budget,
        "warnings": training_hparam_overrides(explicit_hparams, hparams),
        "preflight": preflight,
    }
    return entry, provenance


def _current_best_seed_reference(session: AgentSession, objective_metric: str, configured_seeds: list[int]) -> Dict[str, Any]:
    state = load_runtime_state(session.base_dir, session.task_id, auto_migrate=False)
    current_best = state.current_best.to_dict() if state.current_best and state.current_best.candidate_id else {}
    if not current_best:
        raise SeedEvalToolError("CURRENT_BEST_REFERENCE_REQUIRED: seed_eval requires runtime current_best.")
    seed_eval = dict(current_best.get("seed_eval") or {})
    metric_stats = dict(seed_eval.get("metric_stats") or {})
    source_path = str(seed_eval.get("result_path") or "")
    if not metric_stats and str(current_best.get("candidate_kind") or "").lower() == "baseline":
        initial_reference = require_baseline_reference(
            session.base_dir,
            session.task_id,
            objective_metric=objective_metric,
        )
        metric_stats = dict(initial_reference.get("metric_stats") or {})
        source_path = str(initial_reference.get("result_path") or initial_reference.get("path") or "")
    objective_stats = dict((metric_stats.get(objective_metric) or {}))
    required_seeds = len(configured_seeds) if configured_seeds else 1
    if int(objective_stats.get("seed_count") or 0) < required_seeds:
        raise SeedEvalToolError(
            "CURRENT_BEST_REFERENCE_INSUFFICIENT_SEEDS: "
            f"seed evaluation requires current_best metric_stats[{objective_metric}].seed_count "
            f">={required_seeds}; got {int(objective_stats.get('seed_count') or 0)}"
        )
    return {
        "kind": "current_best",
        "candidate_id": current_best.get("candidate_id"),
        "variant_path": current_best.get("variant_path"),
        "source_checkout": current_best.get("source_checkout"),
        "model_config_hash": current_best.get("model_config_hash"),
        "source_clean": current_best.get("source_clean"),
        "metrics": dict(current_best.get("metrics") or {}),
        "metric_stats": metric_stats,
        "result_path": source_path,
        "seed_count": objective_stats.get("seed_count"),
        "reference_mean": objective_stats.get("mean"),
        "reference_std": objective_stats.get("std"),
    }


@tracked_stage(
    "seed_eval",
    lambda session, args: (
        session.base_dir,
        session.task_id,
        str((args or {}).get("round_id") or ""),
        str((args or {}).get("node_id") or (args or {}).get("run_id") or ""),
    ),
)
def run_seed_eval(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    if session.dry_run:
        return {"status": "dry_run", "message": "run_seed_eval was not executed because the session is in dry-run mode."}

    objective_metric = _objective_metric(session, args.get("objective_metric"))
    configured_seeds = fixed_seed_list(session.base_dir)
    node_id = str(args.get("node_id") or args.get("run_id") or f"seed_eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    model_entry, provenance = _build_seed_eval_model_entry(session, args)
    state = load_runtime_state(session.base_dir, session.task_id, auto_migrate=False)
    current_best = state.current_best.to_dict() if state.current_best and state.current_best.candidate_id else {}
    variant_path = str(args.get("variant_path") or "").strip()
    source_checkout = str(args.get("source_checkout") or "").strip()
    is_candidate_seed_eval = bool(variant_path or source_checkout)
    current_best_reference = {}
    reference_mean = None
    reference_std = None
    reference_seed_count = None
    if is_candidate_seed_eval:
        current_best_reference = _current_best_seed_reference(session, objective_metric, configured_seeds)
        reference_mean = current_best_reference.get("reference_mean")
        reference_std = current_best_reference.get("reference_std")
        reference_seed_count = current_best_reference.get("seed_count")
    result = run_seed_evaluation(
        task_id=session.task_id,
        node_id=node_id,
        model_config=model_entry,
        config_path=resolve_compiled_config_path(session.task_id, session.base_dir),
        objective_metric=objective_metric,
        num_seeds=len(configured_seeds),
        base_seed=configured_seeds[0] if configured_seeds else 2021,
        seed_list=configured_seeds,
        base_dir=session.base_dir,
        reference_metrics=dict(current_best_reference.get("metrics") or current_best.get("metrics") or current_best.get("best_metrics") or {}),
        reference_mean=reference_mean,
        reference_std=reference_std,
        reference_seed_count=reference_seed_count,
        candidate_id=str(args.get("candidate_id") or variant_path or node_id),
        variant_path=variant_path or None,
        source_checkout=source_checkout or None,
        promotion_metadata=dict(args.get("promotion_metadata") or {}),
        promote_on_accept=bool(args.get("promote_on_accept", True)),
        min_effect_size=None,
    )
    if is_candidate_seed_eval:
        result["comparison_reference"] = {
            "kind": "current_best",
            "result_path": str(current_best_reference.get("result_path") or ""),
            "candidate_id": current_best_reference.get("candidate_id"),
            "variant_path": current_best_reference.get("variant_path"),
            "source_checkout": current_best_reference.get("source_checkout"),
            "model_config_hash": current_best_reference.get("model_config_hash"),
            "source_clean": current_best_reference.get("source_clean"),
            "seed_count": reference_seed_count,
            "reference_mean": reference_mean,
            "reference_std": reference_std,
            "current_best_mean": reference_mean,
            "current_best_std": reference_std,
        }
        if result.get("result_path"):
            Path(str(result["result_path"])).write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
    decision = dict(result.get("significance_decision") or {})
    metrics = dict(result.get("mean_metrics") or {})
    record_candidate(
        session.base_dir,
        session.task_id,
        {
            "run_id": node_id,
            "candidate_id": str(args.get("candidate_id") or variant_path or node_id),
            "model_name": provenance.get("display_model_name") or model_entry.get("model_name"),
            "import_model_name": model_entry.get("model_name"),
            "display_model_name": provenance.get("display_model_name"),
            "variant_path": variant_path or None,
            "adapter": model_entry.get("adapter"),
            "status": "success" if result.get("mean") is not None else "failed",
            "metrics": metrics,
            "objective_metric": objective_metric,
            "evaluation_stage": "seed_eval",
            "seed": configured_seeds[0] if configured_seeds else 2021,
            "gate_decision": decision.get("decision"),
            "config_hash": provenance["config_hash"],
            "seed_eval_result_path": result.get("result_path"),
            "promoted_to_current_best": result.get("promoted_to_current_best"),
        },
    )
    record_execution_cost(
        session.base_dir,
        session.task_id,
        {
            "model_key": provenance.get("display_model_name") or model_entry.get("model_name") or node_id,
            "candidate_id": str(args.get("candidate_id") or variant_path or node_id),
            "run_id": node_id,
            "variant_path": variant_path or None,
            "tier": "seed_eval",
            "budget": "seed_eval",
            "evaluation_stage": "seed_eval",
            "seed": configured_seeds[0] if configured_seeds else 2021,
            "status": SUCCESS_STATUS if result.get("mean") is not None else ERROR_STATUS,
            "elapsed_seconds": result.get("elapsed_seconds"),
            "elapsed_seconds_total": result.get("elapsed_seconds_total") or result.get("elapsed_seconds"),
            "objective_metric": objective_metric,
        },
    )
    round_record = EvaluationDecisionKernel.from_session(
        session
    ).record_seed_result(
        variant_path=variant_path or None,
        result={
            **result,
            "node_id": node_id,
        },
    )
    promoted_record = dict(result.get("promoted_record") or {})
    if result.get("promoted_to_current_best"):
        promotion_metadata = dict(args.get("promotion_metadata") or {})
        state_after = load_runtime_state(session.base_dir, session.task_id, auto_migrate=False)
        current_best = state_after.current_best.to_dict() if state_after.current_best and state_after.current_best.candidate_id else {}
        if promotion_metadata and current_best:
            extras = dict(current_best.get("extras") or {})
            if current_best.get("removed_or_bypassed_mechanism") and not extras.get("removed_or_bypassed_mechanism"):
                extras["removed_or_bypassed_mechanism"] = current_best.get("removed_or_bypassed_mechanism")
            for key in ("removed_or_bypassed_mechanism", "exact_edit_intent"):
                if promotion_metadata.get(key):
                    extras[key] = promotion_metadata.get(key)
            source_ref_metadata = dict(promotion_metadata.get("source_ref") or {})
            current_source_ref = dict(current_best.get("source_ref") or {})
            source_binding = (
                promotion_metadata.get("source_binding")
                or source_ref_metadata.get("source_binding")
                or current_best.get("source_binding")
                or current_source_ref.get("source_binding")
            )
            model_binding_ref = (
                promotion_metadata.get("model_binding_ref")
                or source_ref_metadata.get("model_binding_ref")
                or current_best.get("model_binding_ref")
                or current_source_ref.get("model_binding_ref")
            )
            enriched_source_ref = {
                **current_source_ref,
                **source_ref_metadata,
                "kind": "source_snapshot",
                "candidate_snapshot_id": promotion_metadata.get("candidate_snapshot_id") or current_best.get("candidate_snapshot_id"),
                "base_snapshot_id": promotion_metadata.get("base_snapshot_id") or current_best.get("base_snapshot_id"),
                "patch_path": promotion_metadata.get("patch_path") or current_best.get("patch_path"),
                "changed_files": list(promotion_metadata.get("changed_files") or current_best.get("changed_files") or []),
                "parent_candidate_id": promotion_metadata.get("parent_candidate_id") or current_best.get("parent_candidate_id"),
                "source_checkout": str(args.get("source_checkout") or promotion_metadata.get("source_checkout") or ""),
            }
            if source_binding:
                enriched_source_ref["source_binding"] = dict(source_binding)
            if model_binding_ref:
                enriched_source_ref["model_binding_ref"] = str(model_binding_ref)
            enriched = {
                **current_best,
                "source": str(promotion_metadata.get("source") or "ablation_seed_eval_accept"),
                "candidate_kind": str(promotion_metadata.get("candidate_kind") or "ablation_variant"),
                "parent_candidate_id": promotion_metadata.get("parent_candidate_id") or current_best.get("parent_candidate_id"),
                "candidate_snapshot_id": promotion_metadata.get("candidate_snapshot_id") or current_best.get("candidate_snapshot_id"),
                "base_snapshot_id": promotion_metadata.get("base_snapshot_id") or current_best.get("base_snapshot_id"),
                "patch_path": promotion_metadata.get("patch_path") or current_best.get("patch_path"),
                "changed_files": list(promotion_metadata.get("changed_files") or current_best.get("changed_files") or []),
                "source_ref": enriched_source_ref,
                **({"source_binding": dict(source_binding)} if source_binding else {}),
                **({"model_binding_ref": str(model_binding_ref)} if model_binding_ref else {}),
                "extras": extras,
            }
            sync_current_best(session.base_dir, session.task_id, enriched)
            promoted_record = {
                "candidate_id": enriched.get("candidate_id"),
                "objective_metric": current_best.get("objective_metric"),
                "objective_value": (current_best.get("metrics") or {}).get(current_best.get("objective_metric")),
                "source": enriched.get("source"),
                "candidate_kind": enriched.get("candidate_kind"),
                "parent_candidate_id": enriched.get("parent_candidate_id"),
                "extras": extras,
            }
    return {
        "status": "ok",
        "node_id": node_id,
        "success": result.get("mean") is not None,
        "objective_metric": objective_metric,
        "mean": result.get("mean"),
        "std": result.get("std"),
        "successful_seeds": result.get("successful_seeds"),
        "valid_metric_seeds": result.get("valid_metric_seeds"),
        "reference_mean": result.get("reference_mean"),
        "reference_std": result.get("reference_std"),
        "reference_seed_count": result.get("reference_seed_count"),
        "current_best_mean": result.get("current_best_mean"),
        "current_best_std": result.get("current_best_std"),
        "current_best_seed_count": result.get("current_best_seed_count"),
        "comparison_reference": result.get("comparison_reference") or {},
        "significance_decision": decision,
        "promoted_to_current_best": result.get("promoted_to_current_best"),
        "promoted_record": promoted_record,
        "result_path": result.get("result_path"),
        "round_record": round_record,
        "config_hash": provenance["config_hash"],
        "config_provenance_warnings": provenance["warnings"],
    }


def promote_current_best(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    result_path = str(args.get("seed_eval_result_path") or "").strip()
    verified_record = dict(args.get("verified_record") or {})
    seed_result: Dict[str, Any] = {}
    if result_path:
        path = Path(result_path)
        if not path.is_absolute():
            path = Path(session.base_dir) / result_path
        if not path.exists():
            raise SeedEvalToolError(f"seed_eval_result_path not found: {path}")
        seed_result = json.loads(path.read_text(encoding="utf-8"))
    decision = dict(args.get("significance_decision") or seed_result.get("significance_decision") or {})
    if decision.get("decision") != "accept":
        raise SeedEvalToolError("promotion requires seed_eval significance_decision.decision == 'accept'")

    state = load_runtime_state(session.base_dir, session.task_id, auto_migrate=False)
    provisional = state.provisional_best.to_dict() if state.provisional_best else {}
    objective_metric = str(args.get("objective_metric") or seed_result.get("objective_metric") or provisional.get("objective_metric") or state.objective_metric)
    metrics = dict(verified_record.get("metrics") or seed_result.get("mean_metrics") or provisional.get("metrics") or {})
    if objective_metric not in metrics and seed_result.get("mean") is not None:
        metrics[objective_metric] = seed_result.get("mean")
    if objective_metric not in metrics:
        raise SeedEvalToolError(f"promotion requires verified metric '{objective_metric}'")

    record = {
        **provisional,
        **verified_record,
        "candidate_id": verified_record.get("candidate_id") or provisional.get("candidate_id") or seed_result.get("node_id"),
        "display_name": verified_record.get("display_name") or provisional.get("display_name") or provisional.get("variant_path") or seed_result.get("node_id"),
        "metrics": metrics,
        "objective_metric": objective_metric,
        "source": "manual_seed_eval_promotion",
        "scientific_status": "verified_seed_eval",
        "engineering_status": "verified",
        "seed_eval": {
            "result_path": result_path or seed_result.get("result_path"),
            "decision": decision,
            "mean": seed_result.get("mean"),
            "std": seed_result.get("std"),
        },
    }
    state_after = promote_provisional_to_current_best(session.base_dir, session.task_id, record, clear_provisional=bool(args.get("clear_provisional", True)))
    promoted = bool(
        state_after.current_best
        and state_after.current_best.candidate_id == record.get("candidate_id")
    )
    return {
        "status": "ok",
        "promoted_to_current_best": promoted,
        "candidate_id": record.get("candidate_id"),
        "objective_metric": objective_metric,
        "objective_value": metrics.get(objective_metric),
    }
