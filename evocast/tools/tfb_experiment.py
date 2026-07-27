"""TFB experiment tools for EvoCast v3."""

from __future__ import annotations

import json
import hashlib
import math
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from evocast.state.runtime.candidate_registry import record_candidate
from evocast.policy.cost_profile import ERROR_STATUS, SUCCESS_STATUS
from evocast.state.cost_ledger import record_execution_cost
from evocast.state.cost_ledger import tracked_stage
from evocast.policy.error_taxonomy import classify_from_result
from evocast.probe.execution_evidence import extract_traceback_evidence
from evocast.probe.failure_signature import failure_signature
from evocast.domain.metric_parser import parse_metrics_from_paths
from evocast.domain.metric_semantics import DEFAULT_OBJECTIVE_METRIC
from evocast.domain.knowledge_paths import runs_root
from evocast.policy.model_contract import training_hparam_overrides, validate_model_config
from evocast.research.model_registry import build_registry
from evocast.policy.experiment_policy import normalize_budget, task_build_mode
from evocast.domain.result_provenance import (
    build_result_provenance,
    model_entry_hash,
    stamp_result_artifacts,
    validate_result_artifact_provenance,
)
from evocast.domain.effective_model_config import resolve_effective_model_config
from evocast.research.baseline_reference import load_baseline_reference
from evocast.state.runtime.store import (
    load_runtime_state,
    record_runtime_event,
)
from evocast.domain.task_identity import compact_result_save_path, resolve_compiled_config_path
from evocast.state.runtime.trial_journal import append_node, create_node
from evocast.variant.contract import probe_variant_behavior_delta, validate_variant_runtime_contract
from evocast.variant.workspace_loader import (
    load_module_from_variant_path as _workspace_load_module,
    is_workspace_variant_path,
    variant_module_name as _variant_module_name,
)
from evocast.variant.import_isolation import model_execution_import_context
from evocast.harness.permissions import assert_variant_path
from evocast.harness.session import AgentSession
from evocast.runners.tfb_pipeline_runner import (
    build_run_configs,
    load_config_json,
    run_pipeline,
)
from evocast.evaluation.decision_kernel import EvaluationDecisionKernel
from evocast.evaluation.executor import (
    execute_variant,
    require_formal_model_config_binding as _require_formal_model_config_binding,
    require_variant_model_entry_binding as _require_variant_model_entry_binding,
    resolved_path_text as _resolved_path_text,
)


class ExperimentToolError(ValueError):
    """Raised for invalid experiment requests."""


def _runs_index_dir(session: AgentSession) -> Path:
    path = session.knowledge_dir / "run_records"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_record_path(session: AgentSession, run_id: str) -> Path:
    return _runs_index_dir(session) / f"{run_id}.json"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _stable_hash(payload: Dict[str, Any]) -> str:
    text = json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _runtime_contract_summary(runtime_contract: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not runtime_contract:
        return None
    source_entrypoint = dict((runtime_contract or {}).get("source_entrypoint") or {})
    fact_pack = dict((runtime_contract or {}).get("runtime_fact_pack") or {})
    return {
        "status": (runtime_contract or {}).get("status"),
        "input_shape": (runtime_contract or {}).get("input_shape"),
        "output_shape": (runtime_contract or {}).get("output_shape"),
        "expected_output_shape": (runtime_contract or {}).get("expected_output_shape"),
        "expected_eval_output_shape": (runtime_contract or {}).get("expected_eval_output_shape"),
        "accepted_output_shape": (runtime_contract or {}).get("accepted_output_shape"),
        "output_slice_contract": (runtime_contract or {}).get("output_slice_contract"),
        "channel_contract": (runtime_contract or {}).get("channel_contract"),
        "error_type": (runtime_contract or {}).get("error_type"),
        "error_message": (runtime_contract or {}).get("error_message"),
        "failure_evidence": (runtime_contract or {}).get("failure_evidence"),
        "failure_chain": (runtime_contract or {}).get("failure_chain"),
        "mechanism_probe": (runtime_contract or {}).get("mechanism_probe"),
        "failure_signature": (runtime_contract or {}).get("failure_signature"),
        "runtime_fact_pack": {
            "status": fact_pack.get("status"),
            "model_kind": fact_pack.get("model_kind"),
            "failed_module": fact_pack.get("failed_module"),
            "module_trace_count": fact_pack.get("module_trace_count"),
            "confidence": fact_pack.get("confidence"),
            "ambiguities": fact_pack.get("ambiguities"),
        } if fact_pack else None,
        "source_entrypoint": {
            "entry_class": source_entrypoint.get("entry_class"),
            "factory_model_class": source_entrypoint.get("factory_model_class"),
            "adapter_model_class_matches_entry": source_entrypoint.get("adapter_model_class_matches_entry"),
            "outer_model_class": source_entrypoint.get("outer_model_class"),
            "inner_model_class": source_entrypoint.get("inner_model_class"),
            "actual_runtime_root_class": source_entrypoint.get("actual_runtime_root_class"),
            "factory_model_source": source_entrypoint.get("factory_model_source"),
            "inner_model_source": source_entrypoint.get("inner_model_source"),
            "workspace_root": source_entrypoint.get("workspace_root"),
            "workspace_source_bound": source_entrypoint.get("workspace_source_bound"),
            "entry_init_owner": source_entrypoint.get("entry_init_owner"),
            "entry_init_model_owner": source_entrypoint.get("entry_init_model_owner"),
            "uses_wrapper_entrypoint_text": source_entrypoint.get("uses_wrapper_entrypoint_text"),
            "transformer_inner_entrypoint_text": source_entrypoint.get("transformer_inner_entrypoint_text"),
            "variant_module": source_entrypoint.get("variant_module"),
            "runtime_variant_modules": source_entrypoint.get("runtime_variant_modules"),
            "ineffective_local_definitions": source_entrypoint.get("ineffective_local_definitions"),
            "binding_warnings": source_entrypoint.get("binding_warnings"),
        } if source_entrypoint else None,
    }


def _objective_noop_allowed(runtime_contract: Dict[str, Any] | None) -> bool:
    """Pure objective edits may keep same-input predictions unchanged before training.

    This exemption is deliberately evidence-based: it is allowed only when the
    runtime mechanism probe proves that the TFB train/backward path saw a valid
    scalar additional_loss. Architecture/data-flow/combination edits still need
    observable same-input behavior delta.
    """
    if not isinstance(runtime_contract, dict):
        return False
    mechanism_probe = runtime_contract.get("mechanism_probe") or {}
    if not isinstance(mechanism_probe, dict) or mechanism_probe.get("status") != "ok":
        return False
    for case in mechanism_probe.get("cases") or []:
        if not isinstance(case, dict):
            continue
        if case.get("name") != "full_model.train_backward":
            continue
        return bool(case.get("has_additional_loss") and case.get("status") == "ok")
    return False


def _behavior_delta_summary(behavior_delta: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not behavior_delta:
        return None
    return {
        "status": behavior_delta.get("status"),
        "suspected_noop": bool(behavior_delta.get("suspected_noop")),
        "objective_noop_allowed": bool(behavior_delta.get("objective_noop_allowed")),
        "reason": behavior_delta.get("reason"),
        "comparison": behavior_delta.get("comparison"),
        "error_type": behavior_delta.get("error_type"),
        "error_message": behavior_delta.get("error_message"),
        "failure_evidence": behavior_delta.get("failure_evidence"),
    }


def _baseline_entry_for_behavior_delta(
    *,
    baseline: Dict[str, Any],
    source_model_key: str,
    hparams: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_model_config = dict(baseline.get("model_config") or {})
    registry_entry = _registry_model_entry(source_model_key)
    baseline_model_name = str(
        registry_entry.get("model_name")
        or baseline.get("display_name")
        or baseline.get("model_name")
        or source_model_key
        or baseline_model_config.get("model_name")
        or baseline.get("import_path")
        or ""
    ).strip()
    if not baseline_model_name:
        return {}
    return {
        "model_key": str(baseline.get("display_name") or baseline.get("model_name") or source_model_key),
        "model_name": _normalize_model_name(baseline_model_name),
        "adapter": (
            baseline_model_config.get("adapter")
            if baseline_model_config.get("adapter") is not None
            else baseline.get("adapter")
        ),
        "model_hyper_params": hparams,
    }


def _baseline_from_reference(
    *,
    session: AgentSession,
    registry_entry: Dict[str, Any],
    fallback_model_name: str,
) -> Dict[str, Any]:
    reference = load_baseline_reference(session.base_dir, session.task_id)
    if not reference or reference.get("candidate_kind") != "baseline":
        return {}
    model_key = str(reference.get("model") or fallback_model_name or "").strip()
    if not model_key:
        return {}
    ref_registry_entry = _registry_model_entry(model_key) or registry_entry
    model_name = _normalize_model_name(str(ref_registry_entry.get("model_name") or model_key))
    if not model_name:
        return {}
    return {
        "candidate_id": reference.get("candidate_id") or f"baseline_reference_{model_key}",
        "candidate_kind": "baseline",
        "display_name": model_key,
        "model_name": model_key,
        "import_path": model_name,
        "adapter": ref_registry_entry.get("adapter"),
        "metrics": dict(reference.get("single_seed_metrics") or {}),
        "metric_stats": dict(reference.get("metric_stats") or {}),
        "model_config": {
            "model_name": model_name,
            "adapter": ref_registry_entry.get("adapter"),
            "model_hyper_params": {},
        },
        "baseline_reference_path": str(reference.get("path") or ""),
        "source": "baseline_reference",
    }


def _behavior_delta_repair_hint(variant_path: str) -> Dict[str, Any]:
    return {
        "status": "semantic_noop",
        "primary_file": variant_path,
        "required_evidence": (
            "Repair must make the edited mechanism active in the runtime module tree "
            "and produce a non-identical same-input behavior_delta before any training run."
        ),
        "baseline_training_rerun_allowed": False,
    }


def _behavior_delta_failure_evidence(behavior_delta: Dict[str, Any], variant_path: str) -> Dict[str, Any]:
    message = str(
        behavior_delta.get("reason")
        or "baseline and variant probe outputs are identical or numerically indistinguishable"
    )
    return {
        "status": "ok",
        "final_error": message,
        "stage": "behavior_delta_preflight",
        "comparison": behavior_delta.get("comparison"),
        "probe_policy": "same_input_forward_only_no_baseline_training",
        "repair_scope_hint": _behavior_delta_repair_hint(variant_path),
    }


def _noop_failure_signature(
    *,
    variant_path: str,
    model_name: str,
    behavior_delta: Dict[str, Any],
) -> Dict[str, Any]:
    message = str(
        behavior_delta.get("reason")
        or "baseline and variant probe outputs are identical or numerically indistinguishable"
    )
    comparison = behavior_delta.get("comparison") or {}
    traceback_text = json.dumps(
        {
            "stage": behavior_delta.get("stage") or "behavior_delta_probe",
            "reason": message,
            "comparison": comparison,
        },
        ensure_ascii=False,
        default=str,
    )
    return failure_signature(
        error_type="noop_variant_contract",
        message=message,
        traceback_text=traceback_text,
        variant_path=variant_path or f"config:{model_name}",
        stage="behavior_delta_preflight",
    )


def _prediction_hashes_from_validation(validation: Dict[str, Any] | None) -> List[str]:
    hashes: set[str] = set()
    if not isinstance(validation, dict):
        return []
    for record in list(validation.get("records") or []):
        if not isinstance(record, dict):
            continue
        for item in list(record.get("prediction_hashes") or []):
            text = str(item or "").strip()
            if text:
                hashes.add(text)
    return sorted(hashes)


def _prediction_hashes_from_run_result(run_result: Dict[str, Any] | None) -> List[str]:
    provenance = dict(((run_result or {}).get("artifact_provenance") or {}))
    return _prediction_hashes_from_validation(dict(provenance.get("validation") or {}))


def _behavior_delta_proves_active(behavior_delta: Dict[str, Any] | None) -> bool:
    if not isinstance(behavior_delta, dict):
        return False
    if str(behavior_delta.get("status") or "").strip().lower() not in {"ok", "passed", "success"}:
        return False
    return not bool(behavior_delta.get("suspected_noop"))


def _find_duplicate_prediction_hash_evidence(
    *,
    session: AgentSession,
    current_prediction_hashes: List[str],
    current_source_sha256: str,
    current_run_id: str,
) -> Dict[str, Any] | None:
    current_hash_set = {str(item) for item in current_prediction_hashes if str(item)}
    current_source = str(current_source_sha256 or "").strip()
    if not current_hash_set or not current_source:
        return None
    records_dir = _runs_index_dir(session)
    for path in sorted(records_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("run_id") or "") == str(current_run_id):
            continue
        prior_provenance = dict(payload.get("config_provenance") or {})
        prior_source = str(prior_provenance.get("variant_source_sha256") or "").strip()
        if not prior_source or prior_source == current_source:
            continue
        prior_hashes = set(_prediction_hashes_from_run_result(dict(payload.get("run_result") or {})))
        overlap = sorted(current_hash_set.intersection(prior_hashes))
        if not overlap:
            continue
        return {
            "status": "failed",
            "stage": "post_metric_prediction_hash_gate",
            "reason": (
                "candidate prediction hash is identical to a prior run with different source hash; "
                "the edited mechanism is likely not active in the effective forecast path"
            ),
            "prediction_hashes": sorted(current_hash_set),
            "duplicate_prediction_hashes": overlap,
            "current_source_sha256": current_source,
            "prior_run_id": payload.get("run_id"),
            "prior_source_sha256": prior_source,
            "prior_record_path": str(path),
        }
    return None


def _find_current_best_metric_noop_evidence(
    *,
    current_best: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any] | None:
    reference_metrics = dict(current_best.get("metrics") or current_best.get("best_metrics") or {})
    if not reference_metrics or not metrics:
        return None
    preferred = ["mse_norm", "mae_norm"]
    common = [key for key in preferred if key in reference_metrics and key in metrics]
    if len(common) < 2:
        common = sorted(set(reference_metrics).intersection(metrics))
    numeric_common = [
        key
        for key in common
        if isinstance(reference_metrics.get(key), (int, float)) and isinstance(metrics.get(key), (int, float))
    ]
    if len(numeric_common) < 2:
        return None
    if not all(abs(float(metrics[key]) - float(reference_metrics[key])) <= 1e-12 for key in numeric_common):
        return None
    return {
        "status": "failed",
        "stage": "post_metric_current_best_metric_gate",
        "reason": "candidate formal metrics are exactly identical to current_best reference metrics",
        "matching_metrics": {key: metrics[key] for key in numeric_common},
        "reference_kind": "current_best",
        "current_best_candidate_id": current_best.get("candidate_id"),
        "current_best_source": current_best.get("source") or "runtime_state",
    }


def _smoke_log_dir(session: AgentSession) -> Path:
    path = session.knowledge_dir / "smoke_logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_smoke_log(
    session: AgentSession,
    *,
    smoke_id: str,
    variant_path: str,
    payload: Dict[str, Any],
) -> str:
    path = _smoke_log_dir(session) / f"{smoke_id}.json"
    _write_json(path, {"smoke_id": smoke_id, "variant_path": variant_path, **payload})
    return str(path)


def _full_traceback_excerpt(text: str, *, max_chars: int = 20000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _evaluation_stage(budget: str, smoke: bool, *, build_mode: bool = False) -> str:
    if build_mode and not smoke:
        return "build_mode"
    if smoke:
        return "smoke"
    normalized = str(budget or "unified").strip().lower()
    if normalized == "seed_eval":
        return "seed_eval"
    return "experiment"


def _eligible_for_promotion(stage: str) -> bool:
    return str(stage or "").lower() in {"build_mode", "smoke", "experiment", "standard", "seed_eval"}


def _module_from_variant_path(variant_path: str) -> str:
    """Return the display identifier for a variant file."""
    return _variant_module_name(variant_path)


def _runtime_model_name_for_variant(
    *,
    registry_entry: Dict[str, Any],
    baseline: Dict[str, Any],
    fallback_model_name: str,
) -> str:
    """Return the actual baseline import path/class used to execute a variant.

    Workspace variants are loaded through ``variant_path``.  The pipeline still
    needs the baseline wrapper import path in model_config.model_name.
    """
    candidates = (
        (baseline.get("model_config") or {}).get("model_name"),
        baseline.get("import_path"),
        registry_entry.get("model_name"),
        fallback_model_name,
    )
    for value in candidates:
        normalized = _normalize_model_name(str(value or "").strip())
        if normalized and not normalized.startswith("global."):
            return normalized
    return _normalize_model_name(str(fallback_model_name or "").strip())


def _normalize_model_name(model_name: str) -> str:
    """Normalize a model name for baseline/config execution."""
    name = str(model_name or "").strip()
    if name.startswith("global.") and name.endswith(".Model"):
        return name
    return name


def _display_model_name(model_name: str, variant_path: str = "") -> str:
    if variant_path:
        return variant_path
    name = str(model_name or "").strip()
    if name.startswith("global."):
        return name[len("global.") :]
    return name


def _registry_model_entry(model_key_or_name: str) -> Dict[str, Any]:
    key = str(model_key_or_name or "").strip()
    if not key:
        return {}
    for spec in build_registry(verify=False):
        if key in {str(spec.get("model_key") or ""), str(spec.get("import_path") or "")}:
            return {
                "model_key": str(spec.get("model_key") or key),
                "model_name": str(spec.get("import_path") or key),
                "adapter": spec.get("adapter"),
                "default_hyper_params": dict(spec.get("default_hyper_params") or {}),
                "hparam_schema": dict(spec.get("hparam_schema") or {}),
            }
    return {}


def _objective_metric(session: AgentSession, requested: str | None = None) -> str:
    if requested:
        return str(requested)
    state = load_runtime_state(session.base_dir, session.task_id, auto_migrate=False)
    return state.objective_metric or DEFAULT_OBJECTIVE_METRIC


def _append_gate_event(session: AgentSession, event: Dict[str, Any]) -> str:
    path = session.knowledge_dir / "gate_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(event), ensure_ascii=False, default=str) + "\n")
    return str(path)


def _auto_gate_successful_run(
    *,
    session: AgentSession,
    run_id: str,
    variant_path: str,
    candidate_id: str | None = None,
    candidate_kind: str = "variant",
    candidate_name: str | None = None,
    metrics: Dict[str, Any],
    baseline: Dict[str, Any],
    objective_metric: str,
    evaluation_stage: str,
    smoke: bool,
    behavior_delta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    # Compatibility facade: canonical decision ownership lives in the
    # evaluation kernel. Tests and older callers may still patch this symbol.
    del baseline
    return EvaluationDecisionKernel.from_session(session).record_success_gate(
        session=session,
        run_id=run_id,
        variant_path=variant_path,
        candidate_id=candidate_id,
        candidate_kind=candidate_kind,
        candidate_name=candidate_name,
        metrics=metrics,
        objective_metric=objective_metric,
        evaluation_stage=evaluation_stage,
        smoke=smoke,
        behavior_delta=behavior_delta,
    )


def _execute_variant(
    *,
    base_dir: str,
    task_id: str,
    run_id: str,
    candidate_id: str,
    candidate_kind: str,
    tfb_config: Dict[str, Any],
    variant_entry: Dict[str, Any],
    objective_metric: str,
    save_path: str,
    seed: int,
    evaluation_budget: str,
    build_mode: bool,
    source_checkout: str | None = None,
    source_entry_file: str | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str | None]:
    return execute_variant(
        base_dir=base_dir,
        task_id=task_id,
        run_id=run_id,
        candidate_id=candidate_id,
        candidate_kind=candidate_kind,
        tfb_config=tfb_config,
        variant_entry=variant_entry,
        objective_metric=objective_metric,
        save_path=save_path,
        seed=seed,
        evaluation_budget=evaluation_budget,
        build_mode=build_mode,
        source_checkout=source_checkout,
        source_entry_file=source_entry_file,
        build_run_configs_fn=build_run_configs,
        run_pipeline_fn=run_pipeline,
        parse_metrics_fn=parse_metrics_from_paths,
        build_provenance_fn=build_result_provenance,
        stamp_artifacts_fn=stamp_result_artifacts,
        validate_provenance_fn=validate_result_artifact_provenance,
        classify_result_fn=classify_from_result,
    )


def _experiment_request_context(
    session: AgentSession,
    args: Dict[str, Any],
    *,
    smoke_policy: bool = True,
    empty_model_message: str,
) -> Dict[str, Any]:
    config_path = resolve_compiled_config_path(session.task_id, session.base_dir)
    tfb_config = load_config_json(config_path)
    variant_path = str(args.get("variant_path") or "").strip()
    source_checkout = str(args.get("source_checkout") or "").strip()
    source_entry_file = str(args.get("source_entry_file") or args.get("source_target_file") or "").strip()
    model_key = str(args.get("model_key") or "").strip()
    model_name = str(args.get("model_name") or model_key).strip()
    registry_entry = _registry_model_entry(model_name)
    runtime_state = load_runtime_state(session.base_dir, session.task_id, auto_migrate=False)
    baseline = runtime_state.baseline.to_dict() if runtime_state.baseline.candidate_id else {}
    current_best = runtime_state.current_best.to_dict() if runtime_state.current_best and runtime_state.current_best.candidate_id else {}
    if not baseline:
        baseline = _baseline_from_reference(
            session=session,
            registry_entry=registry_entry,
            fallback_model_name=model_name or model_key,
        )
    if variant_path:
        model_name = _runtime_model_name_for_variant(
            registry_entry=registry_entry,
            baseline=baseline,
            fallback_model_name=model_name,
        )
    else:
        model_name = _normalize_model_name(registry_entry.get("model_name") or model_name)
    if not model_name:
        raise ExperimentToolError(empty_model_message)

    budget = normalize_budget(args.get("budget") or "unified", session.base_dir)
    seed = int(args.get("seed") or 2021)
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
    resolved = resolve_effective_model_config(
        config_data=tfb_config,
        base_dir=session.base_dir,
        task_id=session.task_id,
        model_entry=raw_entry,
        baseline_model_config=(baseline.get("model_config") or {}) if variant_path else {},
        explicit_model_hyper_params=explicit_hparams,
        requested_budget=budget,
        smoke=smoke_policy,
    )
    variant_entry = resolved.entry
    hparams = dict(resolved.effective_model_hyper_params)
    merged_hparams = dict(resolved.merged_model_hyper_params_before_policy)
    training_budget = resolved.policy_budget
    if variant_path:
        variant_entry["variant_path"] = variant_path
        _require_variant_model_entry_binding(
            variant_entry,
            variant_path,
            stage="request_context.model_entry",
        )
    return {
        "tfb_config": tfb_config,
        "variant_entry": variant_entry,
        "variant_path": variant_path,
        "source_checkout": source_checkout,
        "source_entry_file": source_entry_file,
        "model_name": model_name,
        "model_key": model_key,
        "registry_entry": registry_entry,
        "seed": seed,
        "baseline": baseline,
        "current_best": current_best,
        "source_model_key": source_model_key,
        "hparams": hparams,
        "budget": budget,
        "base_hparams": base_hparams,
        "explicit_hparams": explicit_hparams,
        "merged_hparams": merged_hparams,
        "training_budget": training_budget,
    }


def _variant_entry_for_request(
    session: AgentSession,
    args: Dict[str, Any],
    *,
    smoke_policy: bool = True,
) -> tuple[Dict[str, Any], Dict[str, Any], str, str, int, Dict[str, Any], str, Dict[str, Any]]:
    context = _experiment_request_context(
        session,
        args,
        smoke_policy=smoke_policy,
        empty_model_message="variant smoke test requires variant_path or model_name",
    )
    return (
        context["tfb_config"],
        context["variant_entry"],
        context["variant_path"],
        context["model_name"],
        context["seed"],
        context["baseline"],
        context["source_model_key"],
        context["hparams"],
    )


def _build_activation_check(
    *,
    runtime_contract: Dict[str, Any] | None,
    variant_path: str,
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    """P0-3: Verify the runtime-loaded module IS the variant, and the target
    module exists in the actual forward path.

    This runs BEFORE behavior_check so noop detection has a definitive verdict
    on whether the variant code actually hooked into the model.
    """
    result: Dict[str, Any] = {
        "status": "skipped",
        "passed": False,
        "variant_module_loaded": False,
        "target_module_in_tree": False,
        "new_parameters_present": False,
        "bindings": {},
    }
    if not variant_path or not isinstance(runtime_contract, dict):
        return result

    source_entrypoint = runtime_contract.get("source_entrypoint")
    if not isinstance(source_entrypoint, dict):
        if str(runtime_contract.get("status") or "").lower() == "ok":
            result["status"] = "skipped"
            result["passed"] = True
            result["skip_reason"] = "runtime_contract has no source_entrypoint diagnostics"
        else:
            result["status"] = "failed"
            result["error"] = "no source_entrypoint in runtime_contract"
        return result

    result["status"] = "running"

    # 1. Check that variant modules exist in the runtime tree.
    #    For transformer_adapter models, the root class is always
    #    TransformerAdapter — the variant is the INNER model, never the root.
    #    We check both paths: (a) variant as root (non-adapter models), and
    #    (b) variant modules present in the runtime tree (adapter models).
    actual_root = str(source_entrypoint.get("actual_runtime_root_class") or "")
    # P2 (workspace): derive module name from variant path.
    normalized_path = variant_path.replace("\\", "/")
    variant_module_name = _variant_module_name(normalized_path)
    result["expected_module"] = variant_module_name
    result["actual_root_class"] = actual_root

    # 2. Check target variant modules exist in the runtime tree.
    runtime_modules = list(source_entrypoint.get("runtime_variant_modules") or [])
    workspace_source_bound = bool(source_entrypoint.get("workspace_source_bound"))
    result["runtime_variant_modules"] = [
        {"name": m.get("name"), "class_path": m.get("class_path")}
        for m in runtime_modules[:20]
    ]
    result["workspace_source_bound"] = workspace_source_bound
    result["factory_model_source"] = source_entrypoint.get("factory_model_source")
    result["inner_model_source"] = source_entrypoint.get("inner_model_source")

    # A variant with at least one non-root module means new code was loaded.
    has_non_root_variant_modules = any(
        m.get("name") != "<root>" for m in runtime_modules
    ) if runtime_modules else False
    result["target_module_in_tree"] = has_non_root_variant_modules

    # P1: Adapter-aware variant_module_loaded check.
    # For transformer_adapter models, the root is TransformerAdapter and the
    # variant is the inner model. variant_module_loaded should pass when the
    # variant's modules ARE in the runtime tree, even if the root class isn't
    # the variant module itself.
    is_transformer_adapter = (
        actual_root and "TransformerAdapter" in actual_root
    )
    if is_transformer_adapter:
        inner_model_class = str(source_entrypoint.get("inner_model_class") or "")
        result["variant_module_loaded"] = (
            workspace_source_bound
            or (
                has_non_root_variant_modules
                and variant_module_name in inner_model_class
            )
        )
    else:
        result["variant_module_loaded"] = (
            variant_module_name in actual_root
            or workspace_source_bound
        )

    runtime_contract_status = str(runtime_contract.get("status") or "").lower()
    runtime_contract_failed = runtime_contract_status not in {"", "ok"}
    result["runtime_contract_failed"] = runtime_contract_failed
    result["runtime_contract_error_type"] = str(runtime_contract.get("error_type") or "")

    # 3. Check for binding warnings — these indicate the variant code may not
    #    actually be what's executing.
    binding_warnings = list(source_entrypoint.get("binding_warnings") or [])
    ineffective = list(source_entrypoint.get("ineffective_local_definitions") or [])
    result["binding_warnings"] = binding_warnings
    result["ineffective_definitions"] = ineffective

    # 4. Determine pass/fail.
    # Warnings about unbound local definitions are advisory for assembled
    # exact-edit variants: the module can legitimately contain legacy
    # baseline classes that are no longer the active runtime binding.
    # Activation should fail only when the variant is not actually loaded or
    # when no non-root variant module reached the runtime tree.
    result["passed"] = (
        result["variant_module_loaded"]
        and (
            workspace_source_bound
            or
            runtime_contract_failed
            or result["target_module_in_tree"]
        )
    )
    result["status"] = "passed" if result["passed"] else "failed"
    if not result["passed"]:
        reasons = []
        if not result["variant_module_loaded"]:
            reasons.append(f"expected {variant_module_name}, got {actual_root[:120]}")
        if not runtime_contract_failed and not result["target_module_in_tree"]:
            if ineffective:
                reasons.append(f"ineffective local definitions: {ineffective[:3]}")
            if binding_warnings:
                reasons.append(f"binding warnings: {binding_warnings[:3]}")
        result["failure_reason"] = "; ".join(reasons)

    return result


def run_variant_smoke_test(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Run deterministic pre-experiment checks for a candidate variant.

    P0-3: Split into activation_check (module identity, forward path binding)
    followed by behavior_check (output delta).  activation_check MUST pass
    before behavior_check is meaningful; a noop detected after a failed
    activation_check is classified as activation_failure, not validation_failure.
    """
    if session.dry_run:
        return {
            "status": "dry_run",
            "success": True,
            "message": "run_variant_smoke_test was not executed because the session is in dry-run mode.",
        }

    smoke_id = datetime.now().strftime("smoke_%Y%m%d_%H%M%S_%f")
    (
        tfb_config,
        variant_entry,
        variant_path,
        model_name,
        seed,
        baseline,
        source_model_key,
        hparams,
    ) = _variant_entry_for_request(
        session,
        args,
        smoke_policy=True,
    )
    source_checkout = str(args.get("source_checkout") or "").strip()
    source_entry_file = str(args.get("source_entry_file") or args.get("source_target_file") or "").strip()
    contract_variant_path = source_entry_file or variant_path or f"config:{model_name}"
    if variant_path:
        assert_variant_path(variant_path)
    recommend_hparams = dict((tfb_config.get("model_config") or {}).get("recommend_model_hyper_params") or {})
    base_hparams = dict((baseline.get("model_config") or {}).get("model_hyper_params") or {}) if variant_path else {}

    # ── Stage 0: Config preflight ─────────────────────────────────────────
    preflight_failed = False
    preflight_error = ""
    preflight_traceback = ""
    try:
        if source_checkout:
            with model_execution_import_context(source_checkout=source_checkout):
                preflight = validate_model_config(
                    variant_entry,
                    recommend_model_hyper_params=recommend_hparams,
                    baseline_validated_hyper_params=base_hparams,
                    require_import=True,
                    variant_path=variant_path or None,
                )
        else:
            preflight = validate_model_config(
                variant_entry,
                recommend_model_hyper_params=recommend_hparams,
                baseline_validated_hyper_params=base_hparams,
                require_import=True,
                variant_path=variant_path or None,
            )
    except Exception as exc:
        preflight_failed = True
        preflight_error = f"{type(exc).__name__}: {exc}"
        preflight_traceback = traceback.format_exc()
        preflight = {
            "status": "failed",
            "errors": [preflight_error],
            "warnings": [],
            "adapter": variant_entry.get("adapter"),
        }

    # ── Stage 1: Runtime contract probe (loads variant) ───────────────────
    runtime_contract: Dict[str, Any] | None = None
    if not preflight_failed:
        runtime_contract = validate_variant_runtime_contract(
            tfb_config=tfb_config,
            variant_entry=variant_entry,
            variant_path=contract_variant_path,
            source_checkout=source_checkout or None,
            seed=seed,
            base_dir=session.base_dir,
        )

    # ── Stage 2: Activation check (P0-3) ──────────────────────────────────
    activation_check: Dict[str, Any] = {}
    if variant_path and isinstance(runtime_contract, dict):
        activation_check = _build_activation_check(
            runtime_contract=runtime_contract,
            variant_path=variant_path,
            baseline=baseline,
        )

    # ── Stage 3: Behavior delta probe (only if activation passed) ──────────
    behavior_delta: Dict[str, Any] | None = None
    activation_passed_for_behavior = bool(source_checkout) or activation_check.get("passed", False)
    if (
        (variant_path or source_checkout)
        and isinstance(runtime_contract, dict)
        and runtime_contract.get("status") == "ok"
        and baseline
        and activation_passed_for_behavior
    ):
        baseline_entry = _baseline_entry_for_behavior_delta(
            baseline=baseline,
            source_model_key=source_model_key,
            hparams=hparams,
        )
        if baseline_entry:
            behavior_delta = probe_variant_behavior_delta(
                tfb_config=tfb_config,
                baseline_entry=baseline_entry,
                variant_entry=variant_entry,
                variant_path=variant_path,
                source_checkout=source_checkout or None,
                source_entry_file=source_entry_file or "",
                seed=seed,
                fit_point=str(args.get("fit_point") or ""),
                base_dir=session.base_dir,
            )
            if behavior_delta and _objective_noop_allowed(runtime_contract):
                behavior_delta["objective_noop_allowed"] = True
                if behavior_delta.get("suspected_noop"):
                    behavior_delta["reason"] = (
                        str(behavior_delta.get("reason") or "")
                        + " | allowed because mechanism_probe proved a scalar additional_loss in the train/backward path"
                    ).strip(" |")

    # ── Aggregate success ─────────────────────────────────────────────────
    success = bool(
        not preflight_failed
        and isinstance(runtime_contract, dict)
        and str(runtime_contract.get("status") or "").lower() == "ok"
    )
    if (
        behavior_delta
        and behavior_delta.get("suspected_noop")
        and not behavior_delta.get("objective_noop_allowed")
    ):
        success = False

    # If activation failed, don't blame behavior_delta — it was never reached.
    if not activation_check.get("passed", False) and activation_check.get("status") == "failed":
        success = False

    error_type = ""
    error_message = ""
    full_traceback = ""
    if preflight_failed:
        error_type = "config_error"
        error_message = preflight_error
        full_traceback = preflight_traceback
    elif (
        not activation_check.get("passed", False)
        and activation_check.get("status") == "failed"
        and not activation_check.get("runtime_contract_failed", False)
    ):
        error_type = "activation_failure"
        error_message = str(activation_check.get("failure_reason") or "variant not activated in forward path")
        full_traceback = json.dumps(activation_check, ensure_ascii=False, default=str)
    elif (
        behavior_delta
        and behavior_delta.get("suspected_noop")
        and not behavior_delta.get("objective_noop_allowed")
    ):
        error_type = "noop_variant_contract"
        error_message = str(
            behavior_delta.get("reason")
            or "baseline and variant probe outputs are identical or numerically indistinguishable"
        )
        full_traceback = json.dumps(
            {
                "stage": behavior_delta.get("stage") or "behavior_delta_probe",
                "reason": error_message,
                "comparison": behavior_delta.get("comparison"),
            },
            ensure_ascii=False,
            default=str,
        )
    elif isinstance(runtime_contract, dict) and runtime_contract.get("status") != "ok":
        error_type = str(runtime_contract.get("error_type") or "invalid_variant_contract")
        error_message = str(runtime_contract.get("error_message") or "runtime contract failed")
        full_traceback = str(runtime_contract.get("error_traceback") or "")

    signature: Dict[str, Any] = {}
    if not success:
        signature = dict((runtime_contract or {}).get("failure_signature") or {})
        if not signature and error_type == "noop_variant_contract" and behavior_delta:
            signature = _noop_failure_signature(
                variant_path=variant_path or source_entry_file,
                model_name=model_name,
                behavior_delta=behavior_delta,
            )
        if not signature:
            signature = failure_signature(
                error_type=error_type or "smoke_precheck_failed",
                message=error_message,
                traceback_text=full_traceback,
                variant_path=variant_path or source_entry_file or f"config:{model_name}",
                stage="variant_smoke_test",
            )

    log_payload = {
        "status": "ok" if success else "failed",
        "success": success,
        "task_id": session.task_id,
        "model_entry": variant_entry,
        "preflight": preflight,
        "runtime_contract": runtime_contract,
        "activation_check": activation_check,      # P0-3: NEW
        "behavior_delta": behavior_delta,
        "error_type": error_type if not success else None,
        "error_message": error_message if not success else "",
        "full_traceback": full_traceback,
        "failure_evidence": (
            _behavior_delta_failure_evidence(behavior_delta, variant_path or source_entry_file)
            if error_type == "noop_variant_contract" and behavior_delta
            else extract_traceback_evidence(full_traceback) if full_traceback else (runtime_contract or {}).get("failure_evidence")
        ),
        "failure_signature": signature if not success else None,
        "runtime_fact_pack": (runtime_contract or {}).get("runtime_fact_pack") if isinstance(runtime_contract, dict) else None,
        "created_at": datetime.now().isoformat(),
    }
    log_path = _write_smoke_log(
        session,
        smoke_id=smoke_id,
        variant_path=variant_path or source_entry_file or f"config:{model_name}",
        payload=log_payload,
    )
    record_runtime_event(
        session.base_dir,
        session.task_id,
        "variant_smoke_test",
        {
            "smoke_id": smoke_id,
            "variant_path": variant_path or source_entry_file or None,
            "success": success,
            "error_type": error_type if not success else None,
            "error_message": error_message if not success else "",
            "log_path": log_path,
        },
    )
    return {
        "status": "ok" if success else "failed",
        "success": success,
        "smoke_id": smoke_id,
        "variant_path": variant_path or source_entry_file or None,
        "model_name": model_name,
        "preflight_status": preflight.get("status"),
        "runtime_contract": _runtime_contract_summary(runtime_contract),
        "activation_check": {                        # P0-3: NEW structured field
            "passed": activation_check.get("passed"),
            "variant_module_loaded": activation_check.get("variant_module_loaded"),
            "target_module_in_tree": activation_check.get("target_module_in_tree"),
            "binding_warnings": activation_check.get("binding_warnings"),
            "failure_reason": activation_check.get("failure_reason"),
        } if activation_check else None,
        "behavior_delta": _behavior_delta_summary(behavior_delta),
        "error_type": error_type if not success else None,
        "error_message": error_message[:1000] if not success else "",
        "full_log_path": log_path,
        "full_traceback_excerpt": _full_traceback_excerpt(full_traceback),
        "failure_signature": signature if not success else None,
        "failure_evidence": (
            _behavior_delta_failure_evidence(behavior_delta, variant_path)
            if error_type == "noop_variant_contract" and behavior_delta
            else
            extract_traceback_evidence(full_traceback)
            if full_traceback
            else (runtime_contract or {}).get("failure_evidence")
        ) if not success else None,
        "runtime_fact_pack": (runtime_contract or {}).get("runtime_fact_pack") if isinstance(runtime_contract, dict) else None,
        "next_action": (
            "repair_variant_using_full_log_and_source_context"
            if not success
            else "run_experiment"
        ),
    }


@tracked_stage(
    "experiment",
    lambda session, args: (
        session.base_dir,
        session.task_id,
        str((args or {}).get("round_id") or ""),
        str((args or {}).get("run_id") or ""),
    ),
)
def run_experiment(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    if session.dry_run:
        return {
            "status": "dry_run",
            "message": "run_experiment was not executed because the session is in dry-run mode.",
        }

    objective_metric = _objective_metric(session, args.get("objective_metric"))

    budget = normalize_budget(args.get("budget") or "unified", session.base_dir)
    build_mode = task_build_mode(session.base_dir, session.task_id)
    smoke = bool(args.get("smoke", False))
    evaluation_stage = _evaluation_stage(budget, smoke, build_mode=build_mode)
    context = _experiment_request_context(
        session,
        args,
        smoke_policy=smoke,
        empty_model_message="run_experiment requires variant_path or model_name",
    )
    tfb_config = context["tfb_config"]
    variant_entry = context["variant_entry"]
    variant_path = context["variant_path"]
    source_checkout = context["source_checkout"]
    source_entry_file = context["source_entry_file"]
    model_name = context["model_name"]
    seed = context["seed"]
    baseline = context["baseline"]
    current_best = context["current_best"]
    source_model_key = context["source_model_key"]
    hparams = context["hparams"]
    base_hparams = context["base_hparams"]
    explicit_hparams = context["explicit_hparams"]
    merged_hparams = context["merged_hparams"]
    training_budget = context["training_budget"]
    candidate_kind = "variant" if variant_path else "config"
    is_source_candidate = bool(source_checkout)
    is_executable_candidate = bool(variant_path or source_checkout)
    candidate_display_name = _display_model_name(model_name, variant_path)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    initial_config_hash = _stable_hash(variant_entry)
    candidate_id = variant_path or f"{model_name}@{initial_config_hash[:8]}"
    save_path = compact_result_save_path(session.task_id, run_id, evaluation_stage)
    recommend_hparams = dict((tfb_config.get("model_config") or {}).get("recommend_model_hyper_params") or {})
    preflight_failed = False
    preflight_error = ""
    preflight_traceback = ""
    try:
        if is_source_candidate:
            with model_execution_import_context(source_checkout=source_checkout):
                preflight = validate_model_config(
                    variant_entry,
                    recommend_model_hyper_params=recommend_hparams,
                    baseline_validated_hyper_params=base_hparams,
                    require_import=True,
                    variant_path=variant_path or None,
                )
        else:
            preflight = validate_model_config(
                variant_entry,
                recommend_model_hyper_params=recommend_hparams,
                baseline_validated_hyper_params=base_hparams,
                require_import=True,
                variant_path=variant_path or None,
            )
    except Exception as exc:
        preflight_failed = True
        preflight_error = f"{type(exc).__name__}: {exc}"
        preflight_traceback = traceback.format_exc()
        preflight = {
            "status": "failed",
            "errors": [preflight_error],
            "warnings": [],
            "adapter": variant_entry.get("adapter"),
        }
    provenance_warnings = training_hparam_overrides(explicit_hparams, hparams)
    provenance = {
        "declared_model_name": model_name,
        "declared_adapter": args.get("adapter"),
        "explicit_model_hyper_params": explicit_hparams,
        "merged_model_hyper_params_before_policy": merged_hparams,
        "applied_model_hyper_params": hparams,
        "policy_budget": training_budget,
        "config_hash": initial_config_hash,
        "model_entry_hash": None,
        "warnings": provenance_warnings,
        "preflight": preflight,
    }

    runtime_contract: Dict[str, Any] | None = None
    behavior_delta: Dict[str, Any] | None = None
    if preflight_failed:
        run_result = {
            "success": False,
            "log_paths": [],
            "error": RuntimeError(preflight_error),
            "error_traceback": preflight_traceback,
            "elapsed_seconds": 0,
        }
        parsed = {"metric_values": {}, "status": "preflight_failed", "warnings": []}
        metrics = {}
        error_type = "config_error"
        if is_executable_candidate:
            provenance["failure_signature"] = failure_signature(
                error_type=error_type,
                message=preflight_error,
                traceback_text=preflight_traceback,
                variant_path=variant_path or f"config:{model_name}",
                stage="config_preflight",
            )
    elif is_executable_candidate:
        runtime_contract = validate_variant_runtime_contract(
            tfb_config=tfb_config,
            variant_entry=variant_entry,
            variant_path=source_entry_file or variant_path or f"config:{model_name}",
            source_checkout=source_checkout or None,
            seed=seed,
            base_dir=session.base_dir,
        )
        provenance["runtime_contract"] = runtime_contract
        if is_executable_candidate and runtime_contract.get("status") == "ok" and baseline:
            baseline_entry = _baseline_entry_for_behavior_delta(
                baseline=baseline,
                source_model_key=source_model_key,
                hparams=hparams,
            )
            if baseline_entry:
                behavior_delta = probe_variant_behavior_delta(
                    tfb_config=tfb_config,
                    baseline_entry=baseline_entry,
                    variant_entry=variant_entry,
                    variant_path=variant_path,
                    source_checkout=source_checkout or None,
                    source_entry_file=source_entry_file or "",
                    seed=seed,
                    fit_point=str(args.get("fit_point") or ""),
                    base_dir=session.base_dir,
                )
                if behavior_delta and _objective_noop_allowed(runtime_contract):
                    behavior_delta["objective_noop_allowed"] = True
                    if behavior_delta.get("suspected_noop"):
                        behavior_delta["reason"] = (
                            str(behavior_delta.get("reason") or "")
                            + " | allowed because mechanism_probe proved a scalar additional_loss in the train/backward path"
                        ).strip(" |")
                provenance["behavior_delta"] = behavior_delta

    smoke_precheck: Dict[str, Any] | None = None
    if preflight_failed:
        pass
    elif runtime_contract and runtime_contract.get("status") != "ok":
        run_result = {
            "success": False,
            "log_paths": [],
            "error": RuntimeError(str(runtime_contract.get("error_message") or "runtime contract failed")),
            "error_traceback": runtime_contract.get("error_traceback") or "",
            "elapsed_seconds": 0,
        }
        parsed = {"metric_values": {}, "status": "contract_failed", "warnings": []}
        metrics: Dict[str, Any] = {}
        error_type = str(runtime_contract.get("error_type") or "invalid_variant_contract")
    elif (
        behavior_delta
        and behavior_delta.get("suspected_noop")
        and not behavior_delta.get("objective_noop_allowed")
    ):
        message = str(
            behavior_delta.get("reason")
            or "baseline and variant probe outputs are identical or numerically indistinguishable"
        )
        run_result = {
            "success": False,
            "log_paths": [],
            "error": RuntimeError(message),
            "error_traceback": json.dumps(
                {
                    "stage": behavior_delta.get("stage") or "behavior_delta_probe",
                    "reason": message,
                    "comparison": behavior_delta.get("comparison"),
                },
                ensure_ascii=False,
                default=str,
            ),
            "elapsed_seconds": 0,
        }
        parsed = {"metric_values": {}, "status": "noop_variant_contract", "warnings": []}
        metrics = {}
        error_type = "noop_variant_contract"
        provenance["failure_signature"] = _noop_failure_signature(
            variant_path=variant_path or source_entry_file,
            model_name=model_name,
            behavior_delta=behavior_delta,
        )
        provenance["failure_evidence"] = {
            **_behavior_delta_failure_evidence(behavior_delta, variant_path or source_entry_file),
            "final_error": message,
        }
    else:
        if is_executable_candidate and not smoke:
            smoke_resolved = resolve_effective_model_config(
                config_data=tfb_config,
                base_dir=session.base_dir,
                task_id=session.task_id,
                model_entry=variant_entry,
                baseline_model_config=baseline.get("model_config") or {},
                explicit_model_hyper_params=explicit_hparams,
                requested_budget="smoke_test",
                smoke=True,
            )
            smoke_entry = smoke_resolved.entry
            smoke_save_path = compact_result_save_path(session.task_id, f"{run_id}_smoke", "smoke")
            smoke_run_result, smoke_parsed, smoke_metrics, smoke_error_type = _execute_variant(
                base_dir=session.base_dir,
                task_id=session.task_id,
                run_id=f"{run_id}_smoke",
                candidate_id=candidate_id,
                candidate_kind=candidate_kind,
                tfb_config=tfb_config,
                variant_entry=smoke_entry,
                objective_metric=objective_metric,
                save_path=smoke_save_path,
                seed=seed,
                evaluation_budget="smoke_precheck",
                build_mode=build_mode,
                source_checkout=source_checkout or None,
                source_entry_file=source_entry_file or None,
            )
            smoke_precheck = {
                "status": "success" if smoke_run_result.get("success") and smoke_metrics else "failed",
                "model_entry": smoke_entry,
                "save_path": smoke_save_path,
                "run_result": smoke_run_result,
                "parsed": smoke_parsed,
                "metrics": smoke_metrics,
                "error_type": smoke_error_type,
            }
            if smoke_precheck["status"] != "success":
                run_result = {
                    "success": False,
                    "log_paths": list(smoke_run_result.get("log_paths") or []),
                    "error": RuntimeError("variant smoke precheck failed; formal experiment was not executed"),
                    "error_traceback": str(smoke_run_result.get("error_traceback") or ""),
                    "elapsed_seconds": smoke_run_result.get("elapsed_seconds", 0),
                }
                parsed = smoke_parsed
                metrics = {}
                error_type = str(smoke_error_type or "smoke_precheck_failed")
                provenance["failure_signature"] = failure_signature(
                    error_type=error_type,
                    message=str(smoke_run_result.get("error") or "variant smoke precheck failed"),
                    traceback_text=str(smoke_run_result.get("error_traceback") or ""),
                    variant_path=variant_path or f"config:{model_name}",
                    stage="smoke_precheck",
                )
            else:
                run_result, parsed, metrics, error_type = _execute_variant(
                    base_dir=session.base_dir,
                    task_id=session.task_id,
                    run_id=run_id,
                    candidate_id=candidate_id,
                    candidate_kind=candidate_kind,
                    tfb_config=tfb_config,
                    variant_entry=variant_entry,
                    objective_metric=objective_metric,
                    save_path=save_path,
                    seed=seed,
                    evaluation_budget="build_mode" if build_mode else evaluation_stage,
                    build_mode=build_mode,
                    source_checkout=source_checkout or None,
                    source_entry_file=source_entry_file or None,
                )
        else:
            run_result, parsed, metrics, error_type = _execute_variant(
                base_dir=session.base_dir,
                task_id=session.task_id,
                run_id=run_id,
                candidate_id=candidate_id,
                candidate_kind=candidate_kind,
                tfb_config=tfb_config,
                variant_entry=variant_entry,
                objective_metric=objective_metric,
                save_path=save_path,
                seed=seed,
                evaluation_budget="build_mode" if build_mode else evaluation_stage,
                build_mode=build_mode,
                source_checkout=source_checkout or None,
                source_entry_file=source_entry_file or None,
            )
        if is_executable_candidate and not bool(run_result.get("success")):
            sig = failure_signature(
                error_type=str(error_type or "runtime_error"),
                message=str(run_result.get("error") or ""),
                traceback_text=str(run_result.get("error_traceback") or ""),
                variant_path=variant_path or f"config:{model_name}",
                stage=evaluation_stage,
            )
            provenance["failure_signature"] = sig
            provenance["failure_evidence"] = extract_traceback_evidence(
                str(run_result.get("error_traceback") or "")
            )

    artifact_expected = dict(((run_result or {}).get("artifact_provenance") or {}).get("expected") or {})
    if artifact_expected.get("model_entry_hash"):
        provenance["model_entry_hash"] = artifact_expected.get("model_entry_hash")
        provenance["config_hash"] = artifact_expected.get("model_entry_hash")
    if artifact_expected.get("variant_source_sha256"):
        provenance["variant_source_sha256"] = artifact_expected.get("variant_source_sha256")
    if artifact_expected.get("workspace_root"):
        provenance["workspace_root"] = artifact_expected.get("workspace_root")
    post_metric_noop_gate = None
    if (
        is_executable_candidate
        and bool(run_result.get("success"))
        and metrics
        and not _behavior_delta_proves_active(behavior_delta)
        and not _objective_noop_allowed(runtime_contract)
    ):
        post_metric_noop_gate = _find_duplicate_prediction_hash_evidence(
            session=session,
            current_prediction_hashes=_prediction_hashes_from_run_result(run_result),
            current_source_sha256=str(provenance.get("variant_source_sha256") or ""),
            current_run_id=run_id,
        )
        if not post_metric_noop_gate:
            post_metric_noop_gate = _find_current_best_metric_noop_evidence(
                current_best=current_best,
                metrics=metrics,
            )
        if post_metric_noop_gate:
            original_metrics = dict(metrics)
            message = str(post_metric_noop_gate.get("reason") or "duplicate prediction hash indicates semantic no-op")
            synthetic_behavior_delta = {
                "status": "ok",
                "stage": "post_metric_prediction_hash_gate",
                "suspected_noop": True,
                "reason": message,
                "comparison": {
                    "prediction_hashes": post_metric_noop_gate.get("prediction_hashes"),
                    "duplicate_prediction_hashes": post_metric_noop_gate.get("duplicate_prediction_hashes"),
                    "prior_run_id": post_metric_noop_gate.get("prior_run_id"),
                },
            }
            run_result["success"] = False
            run_result["error"] = RuntimeError(message)
            run_result["error_traceback"] = json.dumps(post_metric_noop_gate, ensure_ascii=False, default=str)
            parsed["metric_values"] = {}
            parsed["status"] = "noop_variant_contract"
            metrics = {}
            error_type = "noop_variant_contract"
            provenance["post_metric_noop_gate"] = post_metric_noop_gate
            provenance["post_metric_noop_original_metrics"] = original_metrics
            provenance["failure_signature"] = _noop_failure_signature(
                variant_path=variant_path or source_entry_file,
                model_name=model_name,
                behavior_delta=synthetic_behavior_delta,
            )
            provenance["failure_evidence"] = {
                **_behavior_delta_failure_evidence(synthetic_behavior_delta, variant_path or source_entry_file),
                "post_metric_noop_gate": post_metric_noop_gate,
                "final_error": message,
            }
    formal_model_config_variant_path = None
    run_model_entries = list(((run_result or {}).get("model_config") or {}).get("models") or [])
    if run_model_entries and isinstance(run_model_entries[0], dict):
        formal_model_config_variant_path = run_model_entries[0].get("variant_path")
    artifact_validation = dict(((run_result or {}).get("artifact_provenance") or {}).get("validation") or {})

    record = {
        "run_id": run_id,
        "task_id": session.task_id,
        "model_entry": variant_entry,
        "variant_path": variant_path or None,
        "budget": budget,
        "build_mode": build_mode,
        "smoke": smoke,
        "evaluation_stage": evaluation_stage,
        "eligible_for_performance_gate": _eligible_for_promotion(evaluation_stage),
        "build_mode": build_mode,
        "seed": seed,
        "save_path": save_path,
        "objective_metric": objective_metric,
        "config_provenance": provenance,
        "runtime_contract": runtime_contract,
        "behavior_delta": behavior_delta,
        "smoke_precheck": smoke_precheck,
        "run_result": run_result,
        "parsed": parsed,
        "metrics": metrics,
        "created_at": datetime.now().isoformat(),
    }

    node = create_node(
        session.task_id,
        run_id,
        action_type="draft",
        model_name=model_name,
        variant_path=variant_path or None,
        model_config=variant_entry,
        objective_metric=objective_metric,
        metrics=metrics,
        status="success" if run_result.get("success") and metrics else "failed",
        error_type=error_type,
        error_message=str(run_result.get("error") or "")[:1000],
        artifact_paths=list(run_result.get("log_paths") or []),
    )
    journal_path = append_node(session.task_id, node, str(runs_root(session.base_dir)))
    success = bool(run_result.get("success") and metrics)
    failure_sig: Dict[str, Any] = {}
    if is_executable_candidate and not success:
        if isinstance(runtime_contract, dict):
            failure_sig = dict(runtime_contract.get("failure_signature") or {})
        if not failure_sig:
            failure_sig = dict(provenance.get("failure_signature") or {})
    decision_kernel = EvaluationDecisionKernel.from_session(session)
    round_record = decision_kernel.record_attempt(
        run_id=run_id,
        variant_path=variant_path or None,
        status="success" if success else "failed",
        error_type=error_type,
        metrics=metrics,
        failure_signature=failure_sig,
        failure_evidence=(
            ((runtime_contract or {}).get("failure_evidence") if isinstance(runtime_contract, dict) else None)
            or provenance.get("failure_evidence")
        ),
        error_message=str(run_result.get("error") or ""),
        stage=str(failure_sig.get("stage") or "experiment"),
        orchestrator_owns_terminal=bool(
            args.get("orchestrator_owns_round_terminal")
        ),
    )
    gate_decision = None
    auto_gate = None
    gate = None
    evaluation_budget = "build_mode" if build_mode else evaluation_stage
    if success:
        auto_gate = _auto_gate_successful_run(
            session=session,
            run_id=run_id,
            variant_path=variant_path,
            candidate_id=candidate_id,
            candidate_kind=candidate_kind,
            candidate_name=candidate_display_name,
            metrics=metrics,
            baseline=baseline,
            objective_metric=objective_metric,
            evaluation_stage=evaluation_stage,
            smoke=smoke,
            behavior_delta=behavior_delta,
        )
        gate_decision = (auto_gate or {}).get("decision")
        gate = (auto_gate or {}).get("gate")
    candidates_path = record_candidate(
        session.base_dir,
        session.task_id,
        {
            "run_id": run_id,
            "candidate_id": candidate_id,
            "candidate_kind": candidate_kind,
            "model_name": candidate_display_name,
            "import_model_name": model_name,
            "display_model_name": candidate_display_name,
            "variant_path": variant_path or None,
            "adapter": variant_entry.get("adapter"),
            "status": "success" if success else "failed",
            "error_type": error_type if not success else None,
            "metrics": metrics,
            "objective_metric": objective_metric,
            "evaluation_stage": evaluation_stage,
            "seed": seed,
            "gate_decision": gate_decision,
            "config_hash": provenance["config_hash"],
            "provenance_warnings": provenance_warnings,
            "runtime_contract_status": (runtime_contract or {}).get("status") if runtime_contract else None,
            "behavior_delta_status": (behavior_delta or {}).get("status") if behavior_delta else None,
            "suspected_noop": bool((behavior_delta or {}).get("suspected_noop")) if behavior_delta else None,
            "failure_signature": failure_sig if failure_sig else None,
        },
    )
    cost_path = record_execution_cost(
        session.base_dir,
        session.task_id,
        {
            "model_key": candidate_display_name or candidate_id,
            "candidate_id": candidate_id,
            "run_id": run_id,
            "variant_path": variant_path or None,
            "tier": evaluation_stage,
            "budget": budget,
            "evaluation_stage": evaluation_stage,
            "seed": seed,
            "status": SUCCESS_STATUS if success else ERROR_STATUS,
            "elapsed_seconds": run_result.get("elapsed_seconds"),
            "elapsed_seconds_total": run_result.get("elapsed_seconds"),
            "error_type": error_type if not success else None,
            "objective_metric": objective_metric,
        },
    )
    record.update({
        "round_id": (round_record or {}).get("round_id"),
        "round_record": round_record,
        "auto_gate": auto_gate,
        "gate_event": auto_gate,
        "gate": gate,
        "gate_decision": gate_decision,
        "candidate_id": candidate_id,
        "candidate_kind": candidate_kind,
        "candidate_record_path": candidates_path,
        "cost_ledger_path": cost_path,
        "evaluation_budget": evaluation_budget,
        "journal_path": journal_path,
    })
    _write_json(_run_record_path(session, run_id), record)
    return {
        "status": "ok",
        "run_id": run_id,
        "success": bool(run_result.get("success")),
        "metrics": metrics,
        "auto_gate": auto_gate,
        "gate_event": auto_gate,
        "gate": gate,
        "evaluation_stage": evaluation_stage,
        "evaluation_budget": evaluation_budget,
        "formal_model_config_variant_path": formal_model_config_variant_path,
        "artifact_provenance": (run_result or {}).get("artifact_provenance"),
        "formal_artifact_provenance_status": artifact_validation.get("status"),
        "eligible_for_performance_gate": _eligible_for_promotion(evaluation_stage),
        "build_mode": build_mode,
        "parsed_status": parsed.get("status"),
        "log_paths": list(run_result.get("log_paths") or []),
        "error_type": error_type if not success else None,
        "error_message": str(run_result.get("error") or "")[:500] if not success else "",
        "error_traceback": str(run_result.get("error_traceback") or "") if not success else "",
        "failure_evidence": (
            ((runtime_contract or {}).get("failure_evidence") if isinstance(runtime_contract, dict) else None)
            or provenance.get("failure_evidence")
        ) if not success else None,
        "config_hash": provenance["config_hash"],
        "config_provenance_warnings": provenance_warnings,
        "runtime_contract": _runtime_contract_summary(runtime_contract),
        "behavior_delta": {
            "status": (behavior_delta or {}).get("status"),
            "suspected_noop": bool((behavior_delta or {}).get("suspected_noop")) if behavior_delta else None,
            "objective_noop_allowed": bool((behavior_delta or {}).get("objective_noop_allowed")) if behavior_delta else None,
            "reason": (behavior_delta or {}).get("reason"),
            "comparison": (behavior_delta or {}).get("comparison"),
            "error_message": (behavior_delta or {}).get("error_message"),
        } if behavior_delta else None,
        "smoke_precheck": {
            "status": (smoke_precheck or {}).get("status"),
            "metrics": (smoke_precheck or {}).get("metrics"),
            "error_type": (smoke_precheck or {}).get("error_type"),
            "log_paths": list(((smoke_precheck or {}).get("run_result") or {}).get("log_paths") or []),
        } if smoke_precheck else None,
        "failure_signature": failure_sig if not success and failure_sig else None,
        "run_record_path": str(_run_record_path(session, run_id)),
        "candidates_path": candidates_path,
        "cost_ledger_path": cost_path,
        "journal_path": journal_path,
        "round_id": (round_record or {}).get("round_id"),
    }


def read_metrics(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    objective_metric = _objective_metric(session, args.get("objective_metric"))
    log_paths: List[str] = list(args.get("log_paths") or [])
    run_id = str(args.get("run_id") or "").strip()
    if run_id:
        path = _run_record_path(session, run_id)
        if not path.exists():
            raise ExperimentToolError(f"unknown run_id: {run_id}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if not log_paths:
            log_paths = list((record.get("run_result") or {}).get("log_paths") or [])
    if not log_paths:
        raise ExperimentToolError("read_metrics requires run_id or log_paths")
    parsed = parse_metrics_from_paths(log_paths, objective_metric=objective_metric)
    return {"status": "ok", "run_id": run_id or None, "parsed": parsed, "metrics": parsed.get("metric_values", {})}
