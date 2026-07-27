"""Dataset diagnosis, baseline establishment, and mechanism-ablation flow."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from evocast.domain.knowledge_paths import task_knowledge_dir
from evocast.harness.ablation_round import run_ablation_round
from evocast.harness.api_client import create_task_client
from evocast.harness.mechanism_ablation_diagnosis import generate_mechanism_ablation_plan
from evocast.harness.session import AgentSession
from evocast.policy.experiment_policy import baseline_diagnosis_policy, normalize_budget, task_build_mode
from evocast.research.dataset_profile import (
    dataset_profile_path,
    ensure_dataset_profile,
    write_skipped_dataset_profile,
)
from evocast.runners.baseline_search_runner import run_baseline_search
from evocast.state.domain_store import load_task_config
from evocast.state.runtime.store import load_runtime_state, sync_baseline_diagnosis, sync_task_stage
from evocast.tools.model_structure import analyze_model_structure
from evocast.tools.tfb_ablation import (
    active_baseline_record,
    classify_seed_eval_for_ablation,
    finalize_baseline_diagnosis,
    persist_ablation_plan,
    read_ablation_results,
    write_ablation_record,
)
from evocast.tools.tfb_seed_eval import run_seed_eval


@dataclass(frozen=True)
class BaselineFlowRequest:
    task_id: str
    objective_metric: str
    budget: str
    metric_direction: str
    baseline_strategy: str
    baseline_models: List[str]
    seed: int
    dataset_diagnosis_mode: str = "required"
    max_ablation_targets: int = 0
    language: str = "zh"


class BaselineFlowService:
    def __init__(
        self,
        *,
        base_dir: str,
        progress: Callable[[str], None] | None = None,
        analyze_model_structure_fn: Callable[..., Any] = analyze_model_structure,
        generate_mechanism_ablation_plan_fn: Callable[..., Any] = generate_mechanism_ablation_plan,
        persist_ablation_plan_fn: Callable[..., Any] = persist_ablation_plan,
        run_ablation_round_fn: Callable[..., Any] = run_ablation_round,
        run_seed_eval_fn: Callable[..., Any] = run_seed_eval,
        read_ablation_results_fn: Callable[..., Any] = read_ablation_results,
    ) -> None:
        self.base_dir = base_dir
        self.progress = progress or (lambda _message: None)
        self.analyze_model_structure = analyze_model_structure_fn
        self.generate_mechanism_ablation_plan = generate_mechanism_ablation_plan_fn
        self.persist_ablation_plan = persist_ablation_plan_fn
        self.run_ablation_round = run_ablation_round_fn
        self.run_seed_eval = run_seed_eval_fn
        self.read_ablation_results = read_ablation_results_fn

    @staticmethod
    def _english(language: str) -> bool:
        return str(language or "").strip().lower() in {"en", "english"}

    @staticmethod
    def _structure_has_content(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        analysis = dict(value.get("analysis") or value)
        return bool(
            list(analysis.get("source_files") or [])
            and (
                list(analysis.get("inner_models") or [])
                or list(analysis.get("components") or [])
                or list(analysis.get("safe_fit_points") or [])
            )
        )

    def run(self, request: BaselineFlowRequest) -> Dict[str, Any]:
        english = self._english(request.language)
        mode = str(request.dataset_diagnosis_mode or "required").strip().lower()
        profile: Dict[str, Any] = {}
        if mode == "skip":
            self.progress(
                "[Wizard] Skipping dataset analysis and going straight to baseline/research."
                if english
                else "[向导] 已跳过数据集分析，直接进入 baseline/research 流程。"
            )
            profile = write_skipped_dataset_profile(
                task_id=request.task_id,
                base_dir=self.base_dir,
                language=request.language,
            )
        elif mode == "reuse" and not dataset_profile_path(self.base_dir, request.task_id).exists():
            raise RuntimeError(
                "dataset_diagnosis_mode=reuse requires an existing dataset profile."
                if english
                else "dataset_diagnosis_mode=reuse 需要已有 dataset profile。"
            )
        else:
            self.progress(
                "[Wizard] Running dataset diagnosis before baseline."
                if english
                else "[向导] 启动 baseline 前先运行数据集诊断。"
            )
            try:
                prefix = "[Wizard][Dataset Diagnosis]" if english else "[向导][Dataset Diagnosis]"
                profile = ensure_dataset_profile(
                    task_id=request.task_id,
                    base_dir=self.base_dir,
                    progress=lambda message: self.progress(f"{prefix} {message}"),
                    language=request.language,
                )
                claims = [
                    str(item.get("claim") or "")
                    for item in list(profile.get("derived_claims") or [])
                    if str(item.get("claim") or "").strip()
                ][:3]
                self.progress(
                    (
                        "[Wizard] Dataset diagnosis complete: "
                        f"quality={profile.get('profile_quality')}, claims={claims or ['<none>']}, "
                        f"profile={dataset_profile_path(self.base_dir, request.task_id)}"
                    )
                    if english
                    else (
                        "[向导] 数据集诊断完成："
                        f" quality={profile.get('profile_quality')}, claims={claims or ['<none>']}, "
                        f"profile={dataset_profile_path(self.base_dir, request.task_id)}"
                    )
                )
            except Exception as exc:
                self.progress(
                    f"[Wizard] Dataset diagnosis failed, baseline will continue: {type(exc).__name__}: {exc}"
                    if english
                    else f"[向导] 数据集诊断失败，但不会阻断 baseline： {type(exc).__name__}: {exc}"
                )

        self.progress(
            f"[Wizard] Running baseline before agent launch: strategy={request.baseline_strategy}, budget={request.budget}"
            if english
            else f"[向导] 启动 agent 前先运行 baseline：strategy={request.baseline_strategy}, budget={request.budget}"
        )
        baseline_summary = run_baseline_search(
            task_id=request.task_id,
            budget=request.budget,
            objective_metric=request.objective_metric,
            metric_direction=request.metric_direction,
            baseline_strategy=request.baseline_strategy,
            manual_models=list(request.baseline_models),
            seed=request.seed,
            base_dir=self.base_dir,
            dry_run=False,
        )
        if baseline_summary.get("status") != "completed" or not baseline_summary.get("best_baseline"):
            sync_task_stage(
                self.base_dir,
                request.task_id,
                stage="baseline_search",
                status="failed",
                extra={"baseline_search": baseline_summary, "research": {"finalize_reason": "baseline_failed"}},
            )
            return {
                "status": "baseline_failed",
                "baseline_summary": baseline_summary,
                "dataset_profile": profile,
                "diagnosis_summary": {},
            }

        best = dict(baseline_summary.get("best_baseline") or {})
        value = dict(best.get("metrics") or {}).get(request.objective_metric)
        self.progress(
            f"[Wizard] Baseline complete: {best.get('display_name') or best.get('best_model_name')} {request.objective_metric}={value}"
            if english
            else f"[向导] baseline 完成：{best.get('display_name') or best.get('best_model_name')} {request.objective_metric}={value}"
        )
        sync_task_stage(
            self.base_dir,
            request.task_id,
            stage="baseline_diagnosis",
            status="running",
            extra={"best_baseline": best},
        )
        diagnosis = self.run_diagnosis_nonblocking(
            task_id=request.task_id,
            baseline=best,
            objective_metric=request.objective_metric,
            budget=request.budget,
            seed=request.seed,
            max_targets=request.max_ablation_targets,
            language=request.language,
        )
        count = int(dict(diagnosis.get("diagnosis") or {}).get("count") or 0)
        sync_task_stage(
            self.base_dir,
            request.task_id,
            stage="baseline_diagnosis",
            status=str(diagnosis.get("status") or "completed"),
            extra={
                "best_baseline": best,
                "research": {
                    "baseline_diagnosis_status": diagnosis.get("status"),
                    "baseline_diagnosis_count": count,
                },
            },
        )
        self.progress(
            f"[Wizard] Baseline diagnosis evidence records: {count}, status={diagnosis.get('status')}"
            if english
            else f"[向导] baseline 诊断证据记录：{count} 条，status={diagnosis.get('status')}"
        )
        return {
            "status": "completed",
            "baseline_summary": baseline_summary,
            "best_baseline": best,
            "dataset_profile": profile,
            "diagnosis_summary": diagnosis,
        }

    def run_diagnosis(
        self,
        *,
        task_id: str,
        baseline: Dict[str, Any],
        objective_metric: str,
        budget: str,
        seed: int,
        max_targets: int | None = None,
        language: str = "zh",
    ) -> Dict[str, Any]:
        policy = baseline_diagnosis_policy(self.base_dir)
        max_targets = max(0, int(max_targets if max_targets is not None else policy.get("max_ablation_targets", 3)))
        max_repairs = max(0, int(policy.get("ablation_repair_attempts", 3)))
        load_task_config(self.base_dir, task_id)
        session = AgentSession(
            task_id=task_id,
            base_dir=self.base_dir,
            client=create_task_client(base_dir=self.base_dir, task_id=task_id),
            dry_run=False,
        )
        session.ensure_dirs()
        model_key = str(
            baseline.get("model_name")
            or baseline.get("display_name")
            or baseline.get("best_model_name")
            or ""
        )
        if not model_key:
            return {"status": "skipped", "reason": "missing_baseline_model_key"}
        normalized_budget = normalize_budget(budget, self.base_dir)
        evaluation_stage = (
            "smoke"
            if task_build_mode(self.base_dir, task_id)
            else ("seed_eval" if normalized_budget == "seed_eval" else "experiment")
        )
        reference = active_baseline_record(session) or dict(baseline)
        reference_metrics = dict(
            reference.get("metrics")
            or reference.get("best_metrics")
            or baseline.get("metrics")
            or baseline.get("best_metrics")
            or {}
        )
        display_metrics = dict(baseline.get("metrics") or baseline.get("best_metrics") or {})

        if max_targets == 0:
            plan = {
                "schema_version": "mechanism_ablation_plan_v1",
                "planner_source": "wizard",
                "base_model": model_key,
                "model_key": model_key,
                "objective_metric": objective_metric,
                "evaluation_stage": evaluation_stage,
                "targets": [],
                "max_ablation_targets": 0,
                "skip_reason": "max_ablation_targets_zero",
                "created_at": datetime.now().isoformat(),
            }
            review = {
                "schema_version": "mechanism_ablation_plan_review_v1",
                "status": "skipped",
                "reviewed_targets": [],
                "rejected_targets": [],
                "errors": [],
                "corrections": [],
                "evaluation_stage": evaluation_stage,
                "skip_reason": "max_ablation_targets_zero",
                "created_at": datetime.now().isoformat(),
            }
            self.persist_ablation_plan(session, plan, review)
            diagnosis_payload = finalize_baseline_diagnosis(
                session,
                baseline_model=model_key,
                objective_metric=objective_metric,
                baseline_metrics=display_metrics,
                reference_metrics=reference_metrics,
                plan=plan,
                review=review,
                results=[],
                mechanism_understanding={
                    "schema_version": "active_path_summary_ablation_v1",
                    "status": "skipped",
                    "reason": "max_ablation_targets_zero",
                },
            )
            return {
                "status": "skipped",
                "reason": "max_ablation_targets_zero",
                "model_key": model_key,
                "targets": [],
                "plan": plan,
                "review": review,
                "results": [],
                "diagnosis": {
                    "status": "ok",
                    "count": 0,
                    "usable_ablation_count": 0,
                    "ablations": [],
                    "baseline_diagnosis": diagnosis_payload,
                    "baseline_diagnosis_path": str(
                        task_knowledge_dir(self.base_dir, task_id)
                        / "baseline_diagnosis"
                        / "diagnosis_summary.json"
                    ),
                },
            }

        analysis = self.analyze_model_structure(session, {"model_key": model_key, "run_shape_probe": False})
        bundle = self.generate_mechanism_ablation_plan(
            session,
            model_key=model_key,
            objective_metric=objective_metric,
            evaluation_stage=evaluation_stage,
            analysis=analysis,
            max_targets=max_targets,
        )
        plan, review = dict(bundle.get("plan") or {}), dict(bundle.get("review") or {})
        understanding = {
            "schema_version": "active_path_summary_ablation_v1",
            "evidence_graph": bundle.get("evidence_graph"),
            "target_plan": bundle.get("target_plan"),
            "plan": bundle.get("plan"),
            "review": bundle.get("review"),
            "artifact_paths": bundle.get("artifact_paths"),
        }
        self.persist_ablation_plan(session, plan, review)
        targets = list(review.get("reviewed_targets") or [])
        if review.get("status") == "rejected" or not targets:
            diagnosis_payload = finalize_baseline_diagnosis(
                session,
                baseline_model=model_key,
                objective_metric=objective_metric,
                baseline_metrics=display_metrics,
                reference_metrics=reference_metrics,
                plan=plan,
                review=review,
                results=[],
                mechanism_understanding=understanding,
            )
            raise RuntimeError(
                "baseline_diagnosis_ablation_targets_required_but_none_executable: "
                f"max_targets={max_targets}, review_status={review.get('status')}, "
                f"review_errors={json.dumps(review.get('errors') or [], ensure_ascii=False)}, "
                f"baseline_diagnosis_path={diagnosis_payload.get('baseline_diagnosis_path')}"
            )

        results: List[Dict[str, Any]] = []
        for target in targets:
            target_name = str(target.get("target_id") or target.get("mechanism_id") or "mechanism")
            mechanism = str(target.get("mechanism_name") or target.get("mechanism_id") or target_name)
            self.progress(
                f"[Wizard] Baseline mechanism ablation: {target_name} ({mechanism})"
                if self._english(language)
                else f"[向导] baseline 机制消融：{target_name} ({mechanism})"
            )
            try:
                result = self.run_ablation_round(
                    session,
                    target,
                    model_key=model_key,
                    objective_metric=objective_metric,
                    reference_metrics=reference_metrics,
                    budget=normalized_budget,
                    seed=seed,
                    max_repair_attempts=max_repairs,
                )
            except Exception as exc:
                trace = traceback.format_exc()
                result = {
                    "status": "error",
                    "target_id": target.get("target_id"),
                    "target_name": target_name,
                    "mechanism_id": target.get("mechanism_id"),
                    "mechanism_name": target.get("mechanism_name"),
                    "granularity": target.get("granularity"),
                    "exact_edit_intent": target.get("exact_edit_intent"),
                    "failure_type": "exception",
                    "failure_reason": str(exc),
                    "usable_evidence_status": "failed_evidence",
                    "error": str(exc),
                    "traceback": trace,
                }
                failure_dir = session.knowledge_dir / "rounds" / str(target.get("target_id") or target_name)
                try:
                    failure_dir.mkdir(parents=True, exist_ok=True)
                    trace_path = failure_dir / "exception_traceback.txt"
                    record_path = failure_dir / "exception_record.json"
                    trace_path.write_text(trace, encoding="utf-8")
                    record_path.write_text(
                        json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8",
                    )
                    result["artifact_paths"] = [str(trace_path), str(record_path)]
                except Exception:
                    pass

            record = dict(result.get("record") or result)
            interpretation = dict(record.get("interpretation") or {})
            if interpretation.get("recommended_action") == "run_seed_eval":
                target_dir = (
                    session.knowledge_dir
                    / "rounds"
                    / str(record.get("ablation_id") or target.get("target_id") or target_name)
                )
                seed_result = self.run_seed_eval(
                    session,
                    {
                        "variant_path": record.get("variant_path"),
                        "objective_metric": objective_metric,
                        "candidate_id": record.get("target_id") or record.get("variant_path"),
                        "promote_on_accept": True,
                        "promotion_metadata": {
                            "source": "ablation_seed_eval_accept",
                            "candidate_kind": "ablation_variant",
                            "parent_candidate_id": reference.get("candidate_id"),
                            "removed_or_bypassed_mechanism": target.get("mechanism_id"),
                            "exact_edit_intent": record.get("exact_edit_intent"),
                        },
                    },
                )
                classification = classify_seed_eval_for_ablation(
                    objective_metric=objective_metric,
                    seed_eval_result=seed_result,
                    base_dir=self.base_dir,
                )
                seed_payload = {
                    **seed_result,
                    "triggered": True,
                    "module_effect": classification.get("module_effect"),
                    "recommended_action": classification.get("recommended_action"),
                    "development_policy": classification.get("policy"),
                    "relative_improvement": classification.get("relative_improvement"),
                    "threshold": classification.get("threshold"),
                    "build_mode_seed_eval": bool(task_build_mode(self.base_dir, task_id)),
                }
                target_dir.mkdir(parents=True, exist_ok=True)
                (target_dir / "seed_eval_result.json").write_text(
                    json.dumps(seed_payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                record["seed_eval"] = seed_payload
                write_ablation_record(session, record)
                result["record"] = record
            results.append(result)

        self.persist_ablation_plan(session, plan, review)
        diagnosis_payload = finalize_baseline_diagnosis(
            session,
            baseline_model=model_key,
            objective_metric=objective_metric,
            baseline_metrics=display_metrics,
            reference_metrics=reference_metrics,
            plan=plan,
            review=review,
            results=[dict(item.get("record") or item) for item in results],
            mechanism_understanding=understanding,
        )
        diagnosis = self.read_ablation_results(session, {"limit": max_targets})
        success_count = int((diagnosis_payload.get("ablation_execution") or {}).get("successful_count") or 0)
        usable_count = int(diagnosis_payload.get("usable_ablation_count") or 0)
        diagnosis["count"] = usable_count
        diagnosis["usable_ablation_count"] = usable_count
        diagnosis["baseline_diagnosis"] = diagnosis_payload
        status = (
            "failed_non_blocking"
            if review.get("status") == "rejected"
            else "completed"
            if usable_count > 0 and success_count == len(targets)
            else "partial"
            if usable_count > 0
            else "failed_non_blocking"
        )
        return {
            "status": status,
            "model_key": model_key,
            "targets": targets,
            "plan": plan,
            "review": review,
            "results": results,
            "diagnosis": diagnosis,
        }

    def run_diagnosis_nonblocking(
        self,
        *,
        diagnosis_runner: Callable[..., Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            return (diagnosis_runner or self.run_diagnosis)(**kwargs)
        except Exception as exc:
            baseline = dict(kwargs.get("baseline") or {})
            reason = f"{type(exc).__name__}: {exc}"
            path = self.write_nonblocking_diagnosis(
                task_id=str(kwargs["task_id"]),
                baseline=baseline,
                objective_metric=str(kwargs.get("objective_metric") or ""),
                reason=reason,
                targets=[],
                results=[],
            )
            return {
                "status": "failed_non_blocking",
                "reason": reason,
                "exception_type": type(exc).__name__,
                "model_key": (
                    baseline.get("model_name")
                    or baseline.get("display_name")
                    or baseline.get("best_model_name")
                    or ""
                ),
                "targets": [],
                "results": [],
                "diagnosis": {
                    "status": "failed_non_blocking",
                    "count": 0,
                    "usable_ablation_count": 0,
                    "baseline_diagnosis_path": str(path),
                },
            }

    def write_nonblocking_diagnosis(
        self,
        *,
        task_id: str,
        baseline: Dict[str, Any],
        objective_metric: str,
        reason: str,
        targets: Optional[List[Dict[str, Any]]] = None,
        results: Optional[List[Dict[str, Any]]] = None,
    ) -> Path:
        path = task_knowledge_dir(self.base_dir, task_id) / "baseline_diagnosis" / "diagnosis_summary.json"
        existing = dict(load_runtime_state(self.base_dir, task_id).baseline_diagnosis or {})
        if list(existing.get("ablations") or []):
            return path
        failed = list(existing.get("failed_ablations") or [])
        for index, result in enumerate(list(results or []), start=1):
            if isinstance(result, dict) and result.get("status") not in {"ok", "success", "completed"}:
                failed.append(
                    {
                        "target": result.get("target_name") or result.get("mechanism_id") or f"diagnosis_target_{index}",
                        "failure_type": result.get("failure_type") or "baseline_diagnosis_non_blocking_failure",
                        "reason": result.get("error") or result.get("reason") or reason,
                        "action": "continue_agent_with_partial_evidence",
                    }
                )
        if not failed:
            failed.append(
                {
                    "target": "baseline_diagnosis",
                    "failure_type": "baseline_diagnosis_non_blocking_failure",
                    "reason": reason,
                    "action": "continue_agent_with_partial_evidence",
                }
            )
        model_key = (
            baseline.get("model_name")
            or baseline.get("display_name")
            or baseline.get("best_model_name")
            or existing.get("baseline_model")
            or ""
        )
        structure = existing.get("model_structure") or {}
        if not self._structure_has_content(structure) and model_key:
            try:
                session = AgentSession(task_id=task_id, base_dir=self.base_dir, client=None, dry_run=False)  # type: ignore[arg-type]
                session.ensure_dirs()
                refreshed = self.analyze_model_structure(session, {"model_key": model_key, "force_refresh": True})
                if self._structure_has_content(refreshed):
                    structure = refreshed
            except Exception:
                structure = structure or {}
        ablations = list(existing.get("ablations") or [])
        usable = int(existing.get("usable_ablation_count") or 0)
        payload = {
            **existing,
            "schema_version": existing.get("schema_version") or "baseline_diagnosis_v2",
            "task_id": task_id,
            "baseline_model": model_key,
            "objective_metric": objective_metric or existing.get("objective_metric") or "",
            "baseline_metrics": existing.get("baseline_metrics") or baseline.get("metrics") or baseline.get("best_metrics") or {},
            "ablations": ablations,
            "failed_ablations": failed,
            "diagnosis_targets": targets or existing.get("diagnosis_targets") or [],
            "model_structure": structure,
            "planned_ablation_count": existing.get("planned_ablation_count") or len(targets or existing.get("diagnosis_targets") or []),
            "executed_ablation_count": existing.get("executed_ablation_count") or len(ablations),
            "usable_ablation_count": usable,
            "failed_ablation_count": len(failed),
            "target_discovery": existing.get("target_discovery") or {
                "status": "success" if (targets or existing.get("diagnosis_targets")) else "failed",
                "target_count": len(targets or existing.get("diagnosis_targets") or []),
            },
            "ablation_execution": existing.get("ablation_execution") or {
                "status": "partial",
                "successful_count": sum(1 for item in ablations if item.get("status") == "success"),
                "failed_count": len(failed),
            },
            "usable_evidence": existing.get("usable_evidence") or {
                "status": "available" if usable else "unavailable",
                "count": usable,
            },
            "evidence_completeness": {
                "status": "partial",
                "successful_ablations": sum(1 for item in ablations if item.get("status") == "success"),
                "failed_ablations": len(failed),
                "usable_ablations": usable,
                "non_blocking": True,
            },
            "recent_failures": failed[-5:],
            "updated_at": datetime.now().isoformat(),
        }
        sync_baseline_diagnosis(self.base_dir, task_id, payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path
