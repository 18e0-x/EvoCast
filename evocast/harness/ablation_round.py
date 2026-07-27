"""BuildContract-backed source-edit ablation round."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from evocast.build.backends.variant_forge_backend import VariantForgeBackend
from evocast.build.contract_compiler import build_ablation_contract
from evocast.build.metadata_writer import ResearchMetadataWriter
from evocast.build.metric_runner import TFBExperimentMetricRunner
from evocast.build.orchestrator import ResearchBuildOrchestrator
from evocast.build.result import BuildDecision
from evocast.domain.knowledge_paths import repo_root
from evocast.harness.session import AgentSession
from evocast.policy.agent_control_policy import research_repair_budget
from evocast.policy.experiment_policy import baseline_seed, normalize_budget
from evocast.research.ablation.exact_contract import audit_exact_patch_hit
from evocast.state.runtime.store import load_runtime_state
from evocast.tools.tfb_seed_eval import run_seed_eval


def _selected_baseline_source_authority(session: AgentSession, fallback_metrics: dict[str, Any]) -> dict[str, Any]:
    state = load_runtime_state(session.base_dir, session.task_id, auto_migrate=False)
    baseline = state.baseline.to_dict() if state.baseline and state.baseline.candidate_id else {}
    if not baseline:
        raise RuntimeError("run_ablation_round requires selected baseline in runtime_state")
    current_best = state.current_best.to_dict() if state.current_best and state.current_best.candidate_id else {}
    reference_metrics = dict(fallback_metrics or current_best.get("metrics") or {})
    if reference_metrics and not baseline.get("metrics"):
        baseline["metrics"] = reference_metrics
    if reference_metrics:
        baseline["reference_metrics"] = reference_metrics
    if current_best.get("candidate_id"):
        baseline["reference_candidate_id"] = str(current_best.get("candidate_id") or "")
    return baseline


def _outcome_record(
    *,
    session: AgentSession,
    target: Dict[str, Any],
    outcome: Any,
    contract_path: Path,
    variant_path: str,
    objective_metric: str,
) -> Dict[str, Any]:
    metric_result = {}
    metric_path = Path(str(outcome.round_dir)) / "metric" / "metric_result.json"
    if metric_path.is_file():
        try:
            import json

            metric_result = json.loads(metric_path.read_text(encoding="utf-8"))
        except Exception:
            metric_result = {}
    contract_payload = {}
    try:
        import json

        contract_payload = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    except Exception:
        contract_payload = {}
    metric_protocol = dict(contract_payload.get("metric_protocol") or {})
    result = dict(metric_result.get("result") or {})
    metrics = dict(result.get("metrics") or {})
    exact_target = dict((metric_result.get("contract") if isinstance(metric_result.get("contract"), dict) else {}) or {})
    exact_target = dict(exact_target.get("exact_ablation_target") or {})
    if not exact_target:
        exact_target = dict(metric_protocol.get("exact_ablation_target") or {})
    exact_patch_audit = audit_exact_patch_hit(patch_path=outcome.patch_path, exact_target=exact_target)
    exact_patch_audit_path = Path(str(outcome.round_dir)) / "exact_patch_audit.json"
    try:
        import json

        exact_patch_audit_path.write_text(json.dumps(exact_patch_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    metric_completed = bool(metrics)
    patch_hit = bool(exact_patch_audit.get("passed"))
    success = metric_completed and patch_hit
    gate_decision = outcome.status.value if isinstance(outcome.status, BuildDecision) else str(outcome.status or "")
    if success and gate_decision == BuildDecision.ACCEPTED.value:
        scientific_decision = "accepted"
    elif success and gate_decision == BuildDecision.METRIC_COMPLETED.value:
        scientific_decision = "metric_completed_pending_review"
    elif success:
        scientific_decision = "scientific_rejected"
    else:
        scientific_decision = "invalid_evidence"
    execution_status = "metric_completed" if metric_completed else "metric_missing_or_build_failed"
    source_identity = (
        result.get("variant_path")
        or metric_result.get("source_checkout")
        or (f"candidate_snapshot:{outcome.candidate_snapshot_id}" if outcome.candidate_snapshot_id else "")
        or (str((repo_root() / variant_path).resolve()) if variant_path else "")
    )
    return {
        "ablation_id": str(target.get("ablation_id") or target.get("target_id") or outcome.research_id),
        "target_id": target.get("target_id"),
        "target_name": target.get("mechanism_name") or target.get("mechanism_id"),
        "mechanism_id": target.get("mechanism_id"),
        "mechanism_name": target.get("mechanism_name"),
        "causal_variable": target.get("causal_variable"),
        "exact_edit_intent": target.get("exact_edit_intent"),
        "evidence_files": list(target.get("evidence_files") or []),
        "evidence_anchors": list(target.get("evidence_anchors") or []),
        "status": "success" if success else "failed_ablation_round",
        "execution_status": execution_status,
        "gate_decision": gate_decision,
        "scientific_decision": scientific_decision,
        "failure_type": None if success else ("anchor_area_patch_miss" if metric_completed else "build_contract_execution_failed"),
        "failure_reason": "" if success else (str(exact_patch_audit.get("reason") or "") or outcome.summary),
        "usable_evidence_status": "usable_evidence" if success else "failed_evidence",
        "evaluation_stage": str(target.get("evaluation_stage") or metric_protocol.get("evaluation_stage") or "experiment"),
        "execution_surface": "build_contract",
        "metrics": metrics,
        "metrics_source": str(metric_path) if metric_path.is_file() else None,
        "final_attempt_metrics": metrics or None,
        "variant_path": source_identity,
        "artifact_paths": [
            str(contract_path),
            str(outcome.round_dir / "build_outcome.json"),
            str(exact_patch_audit_path),
            *([str(outcome.patch_path)] if outcome.patch_path else []),
            *list(result.get("log_paths") or []),
        ],
        "exact_patch_audit": exact_patch_audit,
        "exact_ablation_target": exact_target,
        "build_outcome": outcome.to_dict(),
        "objective_metric": objective_metric,
        "created_at": datetime.now().isoformat(),
        "task_id": session.task_id,
    }


def run_ablation_round(
    session: AgentSession,
    target: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    objective_metric = str(kwargs.get("objective_metric") or "mse_norm")
    budget = normalize_budget(kwargs.get("budget") or "unified", session.base_dir)
    default_seed = baseline_seed(session.base_dir)
    baseline = _selected_baseline_source_authority(session, dict(kwargs.get("reference_metrics") or {}))
    contract = build_ablation_contract(
        base_dir=session.base_dir,
        task_id=session.task_id,
        target=target,
        baseline=baseline,
        objective_metric=objective_metric,
        repo_dir=repo_root(),
        repair_budget=(
            int(kwargs["max_repair_attempts"])
            if kwargs.get("max_repair_attempts") is not None
            else research_repair_budget(session.base_dir)
        ),
    )
    contract_path = session.knowledge_dir / "build_contracts" / f"{contract.research_id}_{target.get('target_id') or 'ablation'}.json"
    contract.write_json(contract_path)
    backend = VariantForgeBackend(client=session.client)
    orchestrator = ResearchBuildOrchestrator(
        base_dir=session.base_dir,
        task_id=session.task_id,
        repo_dir=repo_root(),
        backend=backend,
        metric_runner=TFBExperimentMetricRunner(
            session=session,
            budget=budget,
            seed=int(kwargs.get("seed") if kwargs.get("seed") is not None else default_seed),
        ),
        seed_eval_runner=lambda seed_args: run_seed_eval(session, seed_args),
        metadata_writer=ResearchMetadataWriter(client=session.client),
    )
    outcome = orchestrator.run(contract)
    variant_path = str((contract.metric_protocol or {}).get("variant_path") or "")
    record = _outcome_record(
        session=session,
        target=target,
        outcome=outcome,
        contract_path=contract_path,
        variant_path=variant_path,
        objective_metric=objective_metric,
    )
    return {
        "status": "ok" if record.get("status") == "success" else "failed_ablation_round",
        "record": record,
        "reason": outcome.summary,
    }
