"""Pure Research-round semantics shared by writers and read models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

FAILURE_KIND_IMPLEMENTATION = "implementation_failure"
FAILURE_KIND_ACTIVATION = "activation_failure"
FAILURE_KIND_VALIDATION = "validation_failure"
FAILURE_KIND_RUNTIME = "runtime_failure"
FAILURE_KIND_SCIENTIFIC = "scientific_failure"

FAILURE_KIND_IDEA_INVALID = "idea_invalid"
FAILURE_KIND_IDEA_INSUFFICIENT = "idea_insufficient"
FAILURE_KIND_CONTRACT_INVALID = "contract_invalid"
FAILURE_KIND_CONTRACT_INACTIVE = "contract_inactive"
FAILURE_KIND_CONTRACT_AMBIGUOUS = "contract_ambiguous"
FAILURE_KIND_IMPL_INVALID = "implementation_invalid"
FAILURE_KIND_IMPL_UNFAITHFUL = "implementation_unfaithful"
FAILURE_KIND_IMPL_INACTIVE = "implementation_inactive"
FAILURE_KIND_RUNTIME_IMPORT = "runtime_import_error"
FAILURE_KIND_RUNTIME_SHAPE = "runtime_shape_error"
FAILURE_KIND_RUNTIME_CUDA = "runtime_cuda_error"
FAILURE_KIND_INFRA_TIMEOUT = "infra_timeout"
FAILURE_KIND_INFRA_OOM = "infra_oom"
FAILURE_KIND_INFRA_DISK = "infra_disk_full"
FAILURE_KIND_SCIENCE_WORSE = "science_worse"
FAILURE_KIND_SCIENCE_NEUTRAL = "science_neutral"
FAILURE_KIND_SCIENCE_BETTER = "science_better"

MAX_CONSECUTIVE_PIPELINE_FAILURES = 3
_PIPELINE_FAILURE_KINDS = {
    FAILURE_KIND_IMPL_INVALID,
    FAILURE_KIND_IMPL_UNFAITHFUL,
    FAILURE_KIND_IMPL_INACTIVE,
    FAILURE_KIND_RUNTIME_IMPORT,
    FAILURE_KIND_RUNTIME_SHAPE,
    FAILURE_KIND_RUNTIME_CUDA,
    FAILURE_KIND_IMPLEMENTATION,
    FAILURE_KIND_ACTIVATION,
    FAILURE_KIND_VALIDATION,
    FAILURE_KIND_RUNTIME,
}

PHASE_DESIGN = "design"
PHASE_BUILD = "build"
PHASE_VERIFY = "verify"
PHASE_EXPERIMENT = "experiment"
PHASE_CLOSED = "closed"
ROUND_PHASES = [PHASE_DESIGN, PHASE_BUILD, PHASE_VERIFY, PHASE_EXPERIMENT]

TERMINAL_ROUND_STATUSES = {
    "completed",
    "abandoned",
    "rejected",
    "failed",
    "idea_rejected",
    "idea_review_exhausted",
    "implementation_rejected",
    "contract_invalid",
    "contract_inactive",
    "write_failed",
    "edit_apply_failed",
    "compile_failed",
    "runtime_probe_failed",
    "smoke_failed",
    "experiment_failed",
    "scientific_rejected",
    "proposal_failed",
    "protocol_blocked",
    "metadata_generation_failed",
    "marginal_no_seed_eval",
    "infra_failed",
}
INFRA_FAILURE_STATUSES = {
    "infra_failed",
    "protocol_blocked",
    "write_failed",
    "edit_apply_failed",
    "compile_failed",
    "runtime_probe_failed",
    "smoke_failed",
    "implementation_rejected",
    "proposal_failed",
    "metadata_generation_failed",
    "contract_invalid",
    "contract_inactive",
}
ROUND_OPEN_STATUSES = {
    "round_started",
    "variant_write_blocked",
    "variant_written",
    "repair_needed",
    "experiment_succeeded",
    "gate_checked",
}
CRITICAL_VARIANT_WRITE_ERRORS = {
    "CODE_HYPERPARAMETER_CHANGE_FORBIDDEN",
    "DESIGN_ROUTE_BLOCKED",
    "SKELETON_CONTRACT_REQUIRED",
    "WRAPPER_SUBCLASS_REQUIRED",
    "INNER_MODULE_SUBCLASS_FORBIDDEN",
    "MECHANISM_REQUIRES_FORBIDDEN_CONFIG_ARGS",
    "PLANNED_WRAPPER_SUBCLASS_MISMATCH",
    "METADATA_MISMATCH",
}
GATE_RESOLUTION_DECISIONS = {"needs_review", "needs_seed_eval"}
ROUND_SCOPE_RESEARCH = "research"
ROUND_SCOPE_BASELINE_DIAGNOSIS = "baseline_diagnosis"


def _rule_implementation(status: str, error_type: str, _has_experiment: bool) -> bool:
    return status in {
        "edit_apply_failed",
        "compile_failed",
        "implementation_rejected",
        "implementation_invalid",
        "contract_invalid",
        "contract_inactive",
        "protocol_blocked",
        "write_failed",
    } or error_type in {
        "static_contract_error",
        "syntax_error",
        "import_error",
        "compile_error",
        *CRITICAL_VARIANT_WRITE_ERRORS,
    }


def _rule_activation(_status: str, error_type: str, _has_experiment: bool) -> bool:
    return error_type == "activation_failure"


def _rule_validation(_status: str, error_type: str, has_experiment: bool) -> bool:
    return error_type in {
        "invalid_variant_contract",
        "invalid_mechanism_contract",
        "config_error",
        "shape_mismatch",
        "smoke_precheck_failed",
        "runtime_contract",
        "shape_contract",
        "runtime_binding",
        "patch_context",
        "noop_variant_contract",
        "semantic_noop",
    } or has_experiment is False


def _rule_runtime(_status: str, error_type: str, _has_experiment: bool) -> bool:
    return error_type in {
        "runtime_error",
        "shape_error",
        "shape_mismatch",
        "import_error",
        "evaluator_error",
        "metric_runner_exception",
        "metric_failed",
        "provenance_validation_failed",
    }


def classify_failure_kind(
    status: str,
    error_type: str = "",
    has_experiment_run: bool = False,
) -> str:
    status_norm = str(status or "").strip().lower()
    error_norm = str(error_type or "").strip().lower()
    rules = (
        (FAILURE_KIND_IMPLEMENTATION, _rule_implementation),
        (FAILURE_KIND_ACTIVATION, _rule_activation),
        (FAILURE_KIND_VALIDATION, _rule_validation),
        (FAILURE_KIND_RUNTIME, _rule_runtime),
    )
    for kind, rule in rules:
        if rule(status_norm, error_norm, has_experiment_run):
            return kind
    return FAILURE_KIND_SCIENTIFIC


def next_phase(current: str) -> str:
    try:
        index = ROUND_PHASES.index(current)
        if index + 1 < len(ROUND_PHASES):
            return ROUND_PHASES[index + 1]
    except ValueError:
        pass
    return PHASE_CLOSED


def infer_round_scope(record: Dict[str, Any]) -> str:
    explicit = str(record.get("round_scope") or "").strip()
    if explicit:
        return explicit
    path = str(record.get("variant_path") or "").replace("\\", "/").strip()
    if path and ("/Ablation" in f"/{path}/" or "/ablation" in f"/{path}/"):
        return ROUND_SCOPE_BASELINE_DIAGNOSIS
    return ROUND_SCOPE_RESEARCH


def is_ablation_variant_path(value: Any) -> bool:
    path = str(value or "").replace("\\", "/").strip()
    return bool(path and ("/Ablation" in f"/{path}/" or "/ablation" in f"/{path}/"))


def counts_toward_research_budget(record: Dict[str, Any]) -> bool:
    if record.get("counts_toward_research_budget") is False:
        return False
    return infer_round_scope(record) == ROUND_SCOPE_RESEARCH


def apply_round_budget_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    record["round_scope"] = infer_round_scope(record)
    record["counts_toward_research_budget"] = counts_toward_research_budget(record)
    return record


def counts_as_completed_round(record: Dict[str, Any]) -> bool:
    return (
        counts_toward_research_budget(record)
        and record.get("status") in TERMINAL_ROUND_STATUSES
    )


def successful_experiment_runs(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = [
        dict(run or {})
        for run in list(record.get("runs") or [])
        if str((run or {}).get("stage") or "") == "experiment"
        and str((run or {}).get("status") or "") == "success"
        and bool((run or {}).get("metrics") or {})
    ]
    snapshot_id = str(record.get("candidate_snapshot_id") or "").strip()
    metric_result = dict((record.get("close_extra") or {}).get("metric_result") or {})
    if (
        str(record.get("status") or "") == "completed"
        and str(metric_result.get("status") or "") == "ACCEPTED"
        and snapshot_id
    ):
        runs.append(
            {
                "run_id": f"{record.get('research_id') or record.get('round_id')}:candidate_metric",
                "stage": "experiment",
                "status": "success",
                "metrics": dict(metric_result.get("metrics") or {"accepted_candidate": True}),
                "candidate_snapshot_id": snapshot_id,
                "metric_result": metric_result,
            }
        )
    return runs


def _numeric(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except Exception:
        return None


def gate_has_result_delta(record: Dict[str, Any]) -> bool:
    event = dict(record.get("gate_event") or {})
    behavior_delta = dict(event.get("behavior_delta") or {})
    if behavior_delta and behavior_delta.get("suspected_noop"):
        return False
    comparison = dict(behavior_delta.get("comparison") or {})
    if any(
        (value := _numeric(comparison.get(key))) is not None and value > 1e-12
        for key in ("max_abs_diff", "mean_abs_diff", "l2_diff")
    ):
        return True
    candidate = _numeric(event.get("candidate_value"))
    baseline = _numeric(event.get("baseline_value"))
    if candidate is not None and baseline is not None:
        return abs(candidate - baseline) > 1e-12
    return bool(event)


def counts_as_valid_trial(record: Dict[str, Any]) -> bool:
    if not counts_as_completed_round(record):
        return False
    if not (
        str(record.get("variant_path") or "").strip()
        or str(record.get("candidate_snapshot_id") or "").strip()
    ):
        return False
    if not successful_experiment_runs(record):
        return False
    return bool(
        record.get("gate_event")
        or str(record.get("candidate_snapshot_id") or "").strip()
    )


def resolve_failure_kind(record: Dict[str, Any]) -> str:
    if str(record.get("status") or "") == "completed":
        return ""
    error_type = ""
    for run in reversed(list(record.get("runs") or [])):
        if run.get("status") == "failed":
            error_type = str(run.get("error_type") or "")
            break
    has_experiment = any(
        str(run.get("stage") or "") == "experiment"
        for run in list(record.get("runs") or [])
    )
    return classify_failure_kind(
        str(record.get("status") or ""),
        error_type,
        has_experiment,
    )


# Compatibility aliases used by the existing facade while callers migrate.
_next_phase = next_phase
_infer_round_scope = infer_round_scope
_round_counts_toward_research_budget = counts_toward_research_budget
_apply_round_budget_metadata = apply_round_budget_metadata
_record_counts_as_completed_round = counts_as_completed_round
_successful_experiment_runs = successful_experiment_runs
_gate_has_result_delta = gate_has_result_delta
_record_counts_as_valid_trial = counts_as_valid_trial
_resolve_failure_kind = resolve_failure_kind
