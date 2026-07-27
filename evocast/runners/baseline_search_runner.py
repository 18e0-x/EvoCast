"""Baseline search runner for evocast."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from evocast.research.baseline_facts import augment_registry_with_facts
from evocast.research.baseline_reference import write_initial_baseline_reference
from evocast.research.baseline_selector import select_baseline_candidates
from evocast.research.dataset_profile import dataset_profile_path, has_dataset_profile_or_intentional_skip, load_dataset_profile
from evocast.policy.experiment_policy import baseline_seed, fixed_seed_list
from evocast.state.runtime.current_best import build_baseline_current_best
from evocast.domain.knowledge_paths import runtime_root, task_knowledge_dir
from evocast.domain.model_key_utils import resolve_model_key
from evocast.domain.baseline_identity import (
    create_immutable_baseline_snapshot,
    persist_model_binding,
    resolve_and_verify_model_binding,
)
from evocast.research.model_registry import build_registry
from evocast.state.runtime.resume_state import atomic_write_json, baseline_search_state_path, load_json
from evocast.state.runtime.trial_journal import latest_nodes_by_id, read_journal
from evocast.policy.agent_control_policy import build_mode_policy, execution_timeout_policy
from evocast.policy.experiment_policy import baseline_search_policy, normalize_budget, task_build_mode
from evocast.state.runtime.store import sync_best_baseline, sync_baseline_search_progress
from evocast.runners.baseline_runner import run_baseline_candidate
from evocast.runners.command_generator import generate_model_entry
from evocast.runners.tfb_pipeline_runner import load_config_json


DEFAULT_PROFILE = {
    "candidate_count": 9,
    "registry_pool_size": 36,
    "initial_seeds": [],
    "preferred_families": ["linear", "transformer", "mlp", "cnn", "gnn", "others"],
}


def _base_dir() -> str:
    return str(runtime_root())


def _load_profile(base_dir: str, budget: str) -> Dict[str, Any]:
    del budget
    profile = dict(DEFAULT_PROFILE)
    profile.update(baseline_search_policy(base_dir))
    return profile


def _load_registry_spec(model_key: str, registry: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    specs = list(registry or [])
    key, _ = resolve_model_key(model_key, specs)
    if not key:
        key = str(model_key or "").strip()
    for spec in specs:
        if key in {str(spec.get("model_key") or ""), str(spec.get("import_path") or "")}:
            return dict(spec)
    return None


def _sort_key(result: Dict[str, Any], direction: str) -> tuple:
    value = result.get("objective_value")
    if not isinstance(value, (int, float)):
        return (1, 0.0)
    return (0, float(value) if direction == "lower_is_better" else -float(value))


def _leaderboard_rows(results: List[Dict[str, Any]], metric: str, direction: str) -> List[Dict[str, Any]]:
    rows = []
    for result in sorted(results, key=lambda item: _sort_key(item, direction)):
        rows.append(
            {
                "rank": "",
                "model_key": result.get("model_key", ""),
                "status": result.get("status", ""),
                "objective_metric": metric,
                "objective_value": result.get("objective_value"),
                "error_type": result.get("error_type", ""),
                "family": result.get("family", ""),
                "tier": result.get("tier") or result.get("budget", ""),
                "node_id": result.get("node_id", ""),
                "elapsed_seconds": result.get("elapsed_seconds", 0),
            }
        )
    rank = 1
    for row in rows:
        if row["status"] == "success" and isinstance(row["objective_value"], (int, float)):
            row["rank"] = rank
            rank += 1
    return rows


def _write_leaderboard(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "model_key",
        "status",
        "objective_metric",
        "objective_value",
        "error_type",
        "family",
        "tier",
        "node_id",
        "elapsed_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _candidate_order(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, spec in enumerate(candidates, start=1):
        model_key = str(spec.get("model_key") or f"model_{index}")
        rows.append(
            {
                "index": index,
                "node_id": f"baseline_{index:03d}_{model_key}",
                "model_key": model_key,
                "spec": dict(spec or {}),
            }
        )
    return rows


def _candidates_from_order(candidate_order: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for item in list(candidate_order or []):
        spec = dict(item.get("spec") or {})
        if not spec:
            spec = {"model_key": str(item.get("model_key") or "")}
        spec.setdefault("model_key", str(item.get("model_key") or ""))
        candidates.append(spec)
    return candidates


def _journal_node_to_result(node: Dict[str, Any], objective_metric: str, budget: str, tier: str) -> Dict[str, Any]:
    metrics = dict(node.get("metrics") or {})
    objective_value = metrics.get(objective_metric)
    model_config = dict(node.get("model_config") or {})
    return {
        "status": node.get("status"),
        "objective_value": objective_value,
        "metrics": metrics,
        "model_key": str(node.get("model_name") or ""),
        "family": node.get("family") or "",
        "node_id": node.get("node_id"),
        "elapsed_seconds": node.get("elapsed_seconds", 0),
        "model_config": model_config,
        "run_result": {"log_paths": list(node.get("artifact_paths") or [])},
        "artifact_paths": list(node.get("artifact_paths") or []),
        "error_type": node.get("error_type"),
        "error_message": node.get("error_message"),
        "tier": tier,
        "budget": budget,
        "adapter": model_config.get("adapter"),
        "resumed_from_journal": True,
        "seed": node.get("seed"),
    }


def _completed_baseline_results_by_node(
    *,
    task_id: str,
    base_dir: str,
    objective_metric: str,
    budget: str,
    tier: str,
) -> Dict[str, Dict[str, Any]]:
    completed: Dict[str, Dict[str, Any]] = {}
    for node in latest_nodes_by_id(read_journal(task_id, base_dir)):
        if node.get("action_type") != "baseline":
            continue
        if node.get("status") not in {"success", "failed"}:
            continue
        if not node.get("completed_at"):
            continue
        node_id = str(node.get("node_id") or "")
        if not node_id:
            continue
        completed[node_id] = _journal_node_to_result(node, objective_metric, budget, tier)
    return completed


def _removed_candidate_result(
    *,
    spec: Dict[str, Any],
    node_id: str,
    budget: str,
    tier: str,
) -> Dict[str, Any]:
    model_key = str(spec.get("model_key") or "")
    return {
        "status": "failed",
        "objective_value": None,
        "metrics": {},
        "model_key": model_key,
        "family": str(spec.get("family") or ""),
        "node_id": node_id,
        "elapsed_seconds": 0,
        "model_config": generate_model_entry(spec) if spec.get("import_path") else {},
        "run_result": {"log_paths": []},
        "error_type": "skipped_removed_candidate",
        "error_message": f"Baseline candidate '{model_key}' is no longer registered as runnable.",
        "tier": tier,
        "budget": budget,
        "resumed_from_state": True,
    }


def _write_running_baseline_state(
    *,
    path: str,
    dataset_diagnosis: Dict[str, Any],
    strategy: str,
    budget: str,
    training_budget: str,
    build_mode: bool,
    objective_metric: str,
    metric_direction: str,
    candidate_order: List[Dict[str, Any]],
    selection_report: Dict[str, Any],
    results: List[Dict[str, Any]],
    completed: int,
    resume: bool,
) -> None:
    atomic_write_json(
        path,
        {
            "status": "running",
            "dataset_diagnosis": dataset_diagnosis,
            "strategy": strategy,
            "budget": budget,
            "training_budget": training_budget,
            "build_mode": build_mode,
            "objective_metric": objective_metric,
            "metric_direction": metric_direction,
            "total": len(candidate_order),
            "completed": completed,
            "candidate_order": candidate_order,
            "run_results": results,
            "selection_report": selection_report if strategy == "auto" else {},
            "resume_enabled": resume,
            "updated_at": datetime.now().isoformat(),
        },
    )


def run_baseline_search(
    *,
    task_id: str,
    budget: str = "unified",
    objective_metric: str = "mse_norm",
    metric_direction: str = "lower_is_better",
    baseline_strategy: str = "auto",
    manual_models: Optional[List[str]] = None,
    seed: int | None = None,
    base_dir: Optional[str] = None,
    dry_run: bool = False,
    pipeline_timeout: float = 3600,
    resume: bool = True,
    retry_failed_baselines: bool = False,
) -> Dict[str, Any]:
    """Run baseline search and persist the best baseline into runtime state."""
    search_started_at = datetime.now()
    base_dir = str(runtime_root(base_dir or _base_dir()))
    requested_budget = str(budget or "unified")
    budget = normalize_budget(requested_budget, base_dir)
    seed = int(seed if seed is not None else baseline_seed(base_dir))
    knowledge_dir = task_knowledge_dir(base_dir, task_id)
    config_path = knowledge_dir / "compiled_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"compiled_config.json not found: {config_path}")
    if not has_dataset_profile_or_intentional_skip(base_dir, task_id):
        raise RuntimeError(
            "DATASET_DIAGNOSIS_REQUIRED: run_dataset_diagnosis must complete before run_baseline_search."
        )
    dataset_profile = load_dataset_profile(base_dir, task_id)
    dataset_diagnosis: Dict[str, Any] = {
        "status": str(dataset_profile.get("status") or ""),
        "dataset_profile_path": str(dataset_profile_path(base_dir, task_id)),
        "characteristics_engine": str(dataset_profile.get("characteristics_engine") or ""),
        "diagnostic_count": len(list(dataset_profile.get("diagnostics") or [])),
        "claim_count": len(list(dataset_profile.get("derived_claims") or [])),
    }
    build_mode = task_build_mode(base_dir, task_id)
    build_policy = build_mode_policy(base_dir) if build_mode else {}
    training_budget = "smoke_test" if build_mode and bool(build_policy.get("force_smoke_experiment", True)) else budget
    if build_mode:
        timeout_policy = execution_timeout_policy(base_dir)
        pipeline_timeout = min(float(pipeline_timeout), float(timeout_policy["build_mode_baseline_pipeline"]))

    strategy = str(baseline_strategy or "auto").strip().lower()
    if strategy not in {"auto", "manual"}:
        raise ValueError(
            f"Unsupported baseline strategy: {baseline_strategy}. "
            "Research tasks require a baseline; use auto or manual."
        )

    tfb_config = load_config_json(str(config_path))
    registry = augment_registry_with_facts(build_registry(verify=True))
    profile = _load_profile(base_dir, budget)
    requested_models = [m.strip() for m in (manual_models or []) if str(m).strip()]
    selection_report: Dict[str, Any] = {}
    search_state_path = baseline_search_state_path(base_dir, task_id)
    previous_state = load_json(search_state_path, {}) or {}
    resume_candidate_order = (
        resume
        and previous_state.get("status") == "running"
        and str(previous_state.get("strategy") or "") == strategy
        and str(previous_state.get("budget") or "") == budget
        and isinstance(previous_state.get("candidate_order"), list)
        and previous_state.get("candidate_order")
    )

    if resume_candidate_order:
        candidate_order = list(previous_state.get("candidate_order") or [])
        candidates = _candidates_from_order(candidate_order)
        selection_report = dict(previous_state.get("selection_report") or {})
    elif strategy == "manual":
        if not requested_models:
            raise ValueError("manual baseline strategy requires at least one model")
        candidates = []
        missing = []
        resolved_models = []
        for model_key in requested_models:
            resolved_key, resolution_kind = resolve_model_key(model_key, registry)
            lookup_key = resolved_key or model_key
            spec = _load_registry_spec(lookup_key, registry)
            if spec is None:
                missing.append(model_key)
            else:
                candidates.append(spec)
                if resolved_key and resolved_key != model_key:
                    resolved_models.append(
                        {"requested": model_key, "resolved": resolved_key, "resolution": resolution_kind}
                    )
        if missing:
            known = sorted(str(spec.get("model_key") or "") for spec in registry if spec.get("model_key"))
            raise ValueError(
                f"Manual baseline model(s) not found in registry: {', '.join(missing)}. "
                f"Known examples: {', '.join(known[:30])}"
            )
        candidate_order = _candidate_order(candidates)
    else:
        max_models = int(profile.get("candidate_count") or 5)
        candidates, selection_report = select_baseline_candidates(
            registry=registry,
            config_data=tfb_config,
            candidate_count=max(1, max_models),
            registry_pool_size=max(1, int(profile.get("registry_pool_size") or max_models)),
            initial_seeds=list(profile.get("initial_seeds") or []),
            preferred_families=list(profile.get("preferred_families") or []),
        )
        candidate_order = _candidate_order(candidates)

    progress = {
        "status": "running",
        "phase": "baseline_search",
        "strategy": strategy,
        "budget": budget,
        "total": len(candidate_order),
        "completed": 0,
        "started_at": datetime.now().isoformat(),
        "resume": bool(resume),
    }
    if strategy == "manual":
        progress["requested_models"] = requested_models
        progress["resolved_models"] = resolved_models if "resolved_models" in locals() else []
    else:
        progress["selection_strategy"] = selection_report.get("strategy")
        progress["selection_report"] = selection_report
    sync_baseline_search_progress(base_dir, task_id, progress)

    results: List[Dict[str, Any]] = []
    completed_by_node = _completed_baseline_results_by_node(
        task_id=task_id,
        base_dir=base_dir,
        objective_metric=objective_metric,
        budget=budget,
        tier="manual" if strategy == "manual" else "tournament",
    ) if resume else {}
    registry_keys = {str(spec.get("model_key") or "") for spec in registry if spec.get("model_key")}
    _write_running_baseline_state(
        path=search_state_path,
        dataset_diagnosis=dataset_diagnosis,
        strategy=strategy,
        budget=budget,
        training_budget=training_budget,
        build_mode=build_mode,
        objective_metric=objective_metric,
        metric_direction=metric_direction,
        candidate_order=candidate_order,
        selection_report=selection_report,
        results=results,
        completed=0,
        resume=resume,
    )

    for item in candidate_order:
        index = int(item.get("index") or (len(results) + 1))
        spec = dict(item.get("spec") or {})
        model_key = str(item.get("model_key") or spec.get("model_key") or f"model_{index}")
        spec.setdefault("model_key", model_key)
        node_id = str(item.get("node_id") or f"baseline_{index:03d}_{model_key}")
        previous_result = completed_by_node.get(node_id)
        if previous_result and (previous_result.get("status") == "success" or not retry_failed_baselines):
            result = dict(previous_result)
        elif resume and model_key not in registry_keys:
            result = _removed_candidate_result(
                spec=spec,
                node_id=node_id,
                budget=budget,
                tier="manual" if strategy == "manual" else "tournament",
            )
        else:
            result = run_baseline_candidate(
                spec=spec,
                config_data=tfb_config,
                task_id=task_id,
                node_id=node_id,
                objective_metric=objective_metric,
                budget=training_budget,
                tier="manual" if strategy == "manual" else "tournament",
                seed=seed,
                base_dir=base_dir,
                dry_run=dry_run,
                pipeline_timeout=pipeline_timeout,
            )
        result["tier"] = "manual" if strategy == "manual" else "tournament"
        result["budget"] = budget
        result["model_config"] = dict(result.get("model_config") or generate_model_entry(spec))
        if spec.get("adapter") is not None:
            result["adapter"] = spec.get("adapter")
        results.append(result)
        progress["completed"] = index
        sync_baseline_search_progress(base_dir, task_id, progress)
        _write_running_baseline_state(
            path=search_state_path,
            dataset_diagnosis=dataset_diagnosis,
            strategy=strategy,
            budget=budget,
            training_budget=training_budget,
            build_mode=build_mode,
            objective_metric=objective_metric,
            metric_direction=metric_direction,
            candidate_order=candidate_order,
            selection_report=selection_report,
            results=results,
            completed=len(results),
            resume=resume,
        )

    rows = _leaderboard_rows(results, objective_metric, metric_direction)
    leaderboard_path = knowledge_dir / "baseline_leaderboard.csv"
    _write_leaderboard(leaderboard_path, rows)
    selection_report_path = knowledge_dir / "baseline_selection_report.json"
    if strategy == "auto":
        atomic_write_json(str(selection_report_path), selection_report)

    successful = [
        result for result in results
        if result.get("status") == "success" and isinstance(result.get("objective_value"), (int, float))
    ]
    best = sorted(successful, key=lambda item: _sort_key(item, metric_direction))[0] if successful else {}

    best_record: Dict[str, Any] = {}
    baseline_reference: Dict[str, Any] = {}
    if best:
        best_spec = _load_registry_spec(str(best.get("model_key") or ""), registry) or {}
        best_record = build_baseline_current_best(
            {
                "source": "baseline_search",
                "candidate_kind": "baseline",
                "candidate_id": best.get("node_id") or best.get("model_key"),
                "model_name": best.get("model_key"),
                "display_name": best.get("model_key"),
                "node_id": best.get("node_id"),
                "tier": best.get("tier"),
                "metrics": dict(best.get("metrics") or {}),
                "objective_metric": objective_metric,
                "model_config": dict(best.get("model_config") or {}),
                "adapter": best.get("adapter"),
                "family": best.get("family"),
                "tags": list(best.get("tags") or []),
                "artifact_paths": list((best.get("run_result") or {}).get("log_paths") or []),
                "seed": best.get("seed", seed),
            },
            objective_metric=objective_metric,
            all_results=results,
        )
        # A baseline metric alone is insufficient for research.  Resolve the
        # public registry entry to the class that actually defines the model
        # and persist that immutable fact before this baseline is made active.
        public_import_path = str(
            best_spec.get("import_path")
            or (best.get("model_config") or {}).get("model_name")
            or ""
        ).strip()
        try:
            binding = resolve_and_verify_model_binding(
                model_key=str(best.get("model_key") or best_record.get("display_name") or ""),
                public_import_path=public_import_path,
                adapter=best.get("adapter"),
            )
            binding_ref = persist_model_binding(
                base_dir,
                task_id,
                binding,
                candidate_id=str(best_record.get("candidate_id") or best.get("node_id") or best.get("model_key") or "baseline"),
            )
            best_record["import_path"] = public_import_path
            best_record["model_binding_ref"] = binding_ref["path"]
            best_record["model_binding_hash"] = binding_ref["binding_hash"]
            snapshot = create_immutable_baseline_snapshot(base_dir, task_id, binding)
            best_record["baseline_snapshot"] = snapshot.to_dict()
            best_record["research_binding_status"] = "verified"
        except Exception as exc:
            # Keep the leaderboard evidence, but never let an unresolved
            # Python entry silently enter the research workflow.
            best_record["research_binding_status"] = "failed"
            best_record["research_binding_error"] = f"{type(exc).__name__}: {exc}"
        sync_best_baseline(base_dir, task_id, best_record)
        configured_seeds = fixed_seed_list(base_dir)
        baseline_reference = write_initial_baseline_reference(
            task_id=task_id,
            base_dir=base_dir,
            config_path=str(config_path),
            baseline_record=best_record,
            objective_metric=objective_metric,
            num_seeds=len(configured_seeds),
            base_seed=configured_seeds[0] if configured_seeds else seed,
            seed_list=configured_seeds,
        )
        best_record["baseline_reference_path"] = baseline_reference.get("path")
        best_record["seed_eval"] = {
            "status": "completed",
            "seed_list": list(baseline_reference.get("seed_list") or []),
            "metric_stats": dict(baseline_reference.get("metric_stats") or {}),
            "result_path": baseline_reference.get("result_path"),
            "reference_kind": "current_best",
        }
        sync_best_baseline(base_dir, task_id, best_record)

    summary = {
        "status": "completed" if best_record else "failed",
        "dataset_diagnosis": dataset_diagnosis,
        "strategy": strategy,
        "budget": budget,
        "training_budget": training_budget,
        "build_mode": build_mode,
        "objective_metric": objective_metric,
        "metric_direction": metric_direction,
        "total": len(results),
        "successes": len(successful),
        "failures": len(results) - len(successful),
        "best_baseline": best_record,
        "baseline_reference": baseline_reference,
        "leaderboard_path": str(leaderboard_path),
        "selection_report_path": str(selection_report_path) if strategy == "auto" else "",
        "selection_report": selection_report if strategy == "auto" else {},
        "candidate_order": candidate_order,
        "resume_enabled": resume,
        "retry_failed_baselines": retry_failed_baselines,
        "run_results": results,
        "started_at": search_started_at.isoformat(),
        "completed_at": datetime.now().isoformat(),
        "elapsed_seconds": (datetime.now() - search_started_at).total_seconds(),
    }
    sync_baseline_search_progress(base_dir, task_id, summary)
    return summary
