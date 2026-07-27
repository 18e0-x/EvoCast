"""Seed runner for evocast.

Runs a model with multiple random seeds to test stability.
Used when the gate marks a result as `needs_seed_eval`.
"""

import json
import os
from datetime import datetime
from typing import Any

import numpy as np

from evocast.policy.gate import get_metric_direction, seed_eval_significance_decision
from evocast.policy.agent_control_policy import execution_timeout_policy, gate_policy
from evocast.domain.knowledge_paths import runs_root, task_runs_dir
from evocast.state.runtime.trial_journal import create_node, append_node
from evocast.domain.metric_semantics import DEFAULT_OBJECTIVE_METRIC
from evocast.domain.metric_parser import parse_metrics_from_paths
from evocast.policy.experiment_policy import task_build_mode
from evocast.domain.result_provenance import (
    build_result_provenance,
    model_entry_hash,
    stamp_result_artifacts,
    validate_result_artifact_provenance,
)
from evocast.state.runtime.store import load_runtime_state, promote_provisional_to_current_best, record_runtime_event
from evocast.domain.task_identity import compact_result_save_path
from evocast.runners.tfb_pipeline_runner import (
    load_config_json,
    build_run_configs,
    run_pipeline,
)


def _model_config_hash(model_config: dict[str, Any]) -> str:
    return model_entry_hash(model_config or {}, variant_path=str((model_config or {}).get("variant_path") or "") or None)


def _numeric_mean_metrics(seed_values: list[dict]) -> dict[str, float]:
    values_by_metric: dict[str, list[float]] = {}
    for item in seed_values:
        for key, value in dict(item.get("metrics") or {}).items():
            if isinstance(value, (int, float)):
                values_by_metric.setdefault(key, []).append(float(value))
    return {
        key: float(np.array(values).mean())
        for key, values in values_by_metric.items()
        if values
    }


def _seed_success_counts(seed_values: list[dict]) -> dict[str, int]:
    valid_metric_count = sum(1 for item in seed_values if item.get("objective_value") is not None)
    return {
        "pipeline_successful_seeds": sum(1 for item in seed_values if item.get("success")),
        "successful_seeds": valid_metric_count,
        "valid_metric_seeds": valid_metric_count,
    }


def _reference_stats(
    *,
    base_dir: str,
    task_id: str,
    objective_metric: str,
    reference_metrics: dict[str, Any] | None = None,
    reference_mean: float | None = None,
    reference_std: float | None = None,
    reference_seed_count: int | None = None,
) -> tuple[float | None, float | None, int]:
    state = load_runtime_state(base_dir, task_id, auto_migrate=False)
    current_best = state.current_best.to_dict() if state.current_best and state.current_best.candidate_id else {}
    metrics = dict(reference_metrics or current_best.get("metrics") or current_best.get("best_metrics") or {})
    mean = reference_mean
    if mean is None:
        mean = metrics.get(objective_metric)
    std = reference_std
    count = reference_seed_count
    if count is None:
        count = 1 if mean is not None else 0
    return (
        float(mean) if isinstance(mean, (int, float)) else None,
        float(std) if isinstance(std, (int, float)) else None,
        int(count or 0),
    )


def run_seed_evaluation(
    task_id: str,
    node_id: str,
    model_config: dict,
    config_path: str,
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC,
    num_seeds: int = 3,
    base_seed: int = 2021,
    seed_list: list[int] | None = None,
    seed_universe: list[int] | None = None,
    precomputed_seed_values: list[dict] | None = None,
    base_dir: str | None = None,
    reference_metrics: dict[str, Any] | None = None,
    reference_mean: float | None = None,
    reference_std: float | None = None,
    reference_seed_count: int | None = None,
    candidate_id: str | None = None,
    variant_path: str | None = None,
    source_checkout: str | None = None,
    promotion_metadata: dict[str, Any] | None = None,
    promote_on_accept: bool = True,
    min_effect_size: float | None = None,
    min_relative_improvement: float | None = None,
    min_absolute_improvement: float | None = None,
    reference_std_multiplier: float | None = None,
) -> dict[str, Any]:
    """Run a model with multiple seeds for stability testing.

    Args:
        task_id: Task identifier.
        node_id: Journal node ID of the parent variant.
        model_config: Model config entry.
        config_path: Path to TFB config JSON.
        objective_metric: Objective metric name.
        num_seeds: Number of seeds to evaluate when seed_list is not provided.
        base_seed: Starting seed (incremented per run) when seed_list is not provided.
        seed_list: Explicit seeds to evaluate. Takes precedence over num_seeds/base_seed.
        base_dir: Override base directory.

    Returns:
        Dict with seed evaluation results (mean, std, per_seed values).
    """
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "..")

    runs_dir = str(task_runs_dir(base_dir, task_id))
    os.makedirs(runs_dir, exist_ok=True)

    tfb_config = load_config_json(config_path)
    seed_values: list[dict] = []
    all_metrics: list[float] = []
    evaluation_seed_list = [int(seed) for seed in seed_list] if seed_list is not None else [
        int(base_seed) + i for i in range(max(1, int(num_seeds or 1)))
    ]
    actual_seed_list = [int(seed) for seed in (seed_universe or evaluation_seed_list)]
    if len(set(actual_seed_list)) != len(actual_seed_list):
        raise ValueError(f"seed_universe contains duplicate seeds: {actual_seed_list}")
    precomputed_by_seed = {
        int(item["seed"]): dict(item)
        for item in list(precomputed_seed_values or [])
        if isinstance(item, dict) and item.get("seed") is not None
    }
    missing_precomputed = set(precomputed_by_seed) - set(actual_seed_list)
    if missing_precomputed:
        raise ValueError(
            f"precomputed seed values are outside seed_universe: {sorted(missing_precomputed)}"
        )
    for seed in actual_seed_list:
        if seed in precomputed_by_seed:
            seed_values.append(precomputed_by_seed[seed])
            val = seed_values[-1].get("objective_value")
            if val is not None:
                all_metrics.append(float(val))
    num_seeds = len(actual_seed_list)

    for evaluation_index, seed in enumerate(evaluation_seed_list, start=1):
        if seed in precomputed_by_seed:
            continue
        i = actual_seed_list.index(seed)
        seed_node_id = f"{node_id}_seed{i}"
        save_path = compact_result_save_path(task_id, seed_node_id)

        print(
            f"[seed_eval] Seed {seed} "
            f"(execution {evaluation_index}/{len(evaluation_seed_list)}, "
            f"sample {i + 1}/{num_seeds})..."
        )

        data_config, mc, evaluation_config = build_run_configs(
            tfb_config,
            [model_config],
            save_path=save_path,
            seed=seed,
            override_eval_args={"save_true_pred": True},
        )

        run_result = run_pipeline(
            data_config, mc, evaluation_config,
            timeout=int(execution_timeout_policy(base_dir)["seed_eval_pipeline"]),
            source_checkout=source_checkout,
        )

        metrics = {}
        artifact_provenance = {}
        if run_result["success"] and run_result["log_paths"]:
            seed_candidate_id = candidate_id or variant_path or node_id
            expected_provenance = build_result_provenance(
                task_id=task_id,
                run_id=seed_node_id,
                candidate_id=seed_candidate_id,
                candidate_kind="variant" if variant_path else "config",
                model_entry=model_config,
                evaluation_budget="seed_eval",
                build_mode=bool(task_build_mode(base_dir, task_id)),
                variant_path=variant_path,
            )
            stamped = stamp_result_artifacts(run_result["log_paths"], expected_provenance)
            validation = validate_result_artifact_provenance(
                run_result["log_paths"],
                expected_provenance,
                require_prediction_hash=bool(variant_path),
                require_batch_forecast=(
                    str(
                        ((tfb_config.get("evaluation_config") or {}).get("strategy_args") or {}).get("strategy_name")
                        or ""
                    )
                    == "rolling_forecast"
                ),
            )
            artifact_provenance = {
                "expected": expected_provenance,
                "stamped_records": stamped,
                "validation": validation,
            }
            run_result["artifact_provenance"] = artifact_provenance
            if validation.get("status") == "ok":
                parsed = parse_metrics_from_paths(
                    run_result["log_paths"],
                    objective_metric=objective_metric,
                )
                metrics = parsed["metric_values"]
            else:
                run_result["success"] = False
                run_result["error"] = "result artifact provenance validation failed"
                run_result["error_traceback"] = json.dumps(validation, ensure_ascii=False, default=str)

        val = metrics.get(objective_metric)
        if val is not None:
            all_metrics.append(val)

        seed_values.append({
            "seed": seed,
            "objective_value": val,
            "metrics": metrics,
            "success": run_result["success"],
            "error_type": (
                "evaluator_error"
                if artifact_provenance and artifact_provenance.get("validation", {}).get("status") != "ok"
                else None
            ),
            "artifact_provenance": artifact_provenance or None,
        })

        # Journal node
        node = create_node(
            task_id=task_id,
            node_id=seed_node_id,
            action_type="seed_eval",
            model_name=model_config.get("model_name", ""),
            model_config=model_config,
            objective_metric=objective_metric,
            status="success" if val is not None else "failed",
        )
        node["metrics"] = metrics
        node["parent_id"] = node_id
        node["seed"] = seed
        node["completed_at"] = datetime.now().isoformat()
        node["artifact_paths"] = run_result.get("log_paths", [])
        append_node(task_id, node, str(runs_root(base_dir)))

    seed_values_by_seed = {int(item["seed"]): item for item in seed_values}
    seed_values = [seed_values_by_seed[seed] for seed in actual_seed_list if seed in seed_values_by_seed]

    # Compute summary statistics
    seed_counts = _seed_success_counts(seed_values)
    result: dict[str, Any] = {
        "node_id": node_id,
        "num_seeds": num_seeds,
        "seed_list": actual_seed_list,
        "evaluated_seed_list": evaluation_seed_list,
        "precomputed_seed_list": sorted(precomputed_by_seed),
        "model_config_hash": _model_config_hash(model_config),
        "policy_budget": "seed_eval",
        **seed_counts,
        "per_seed": seed_values,
    }

    if all_metrics:
        arr = np.array(all_metrics)
        result["mean"] = float(arr.mean())
        result["std"] = float(arr.std())
        result["min"] = float(arr.min())
        result["max"] = float(arr.max())
        result["coefficient_of_variation"] = float(arr.std() / arr.mean()) if arr.mean() != 0 else 0.0

        print(f"[seed_eval] {objective_metric}: "
              f"mean={result['mean']:.6f} std={result['std']:.6f} "
              f"cv={result['coefficient_of_variation']:.4f}")
    else:
        result["mean"] = None
        result["std"] = None
        print("[seed_eval] No valid metrics across seeds!")

    mean_metrics = _numeric_mean_metrics(seed_values)
    result["mean_metrics"] = mean_metrics
    reference_mean_value, reference_std_value, reference_count = _reference_stats(
        base_dir=base_dir,
        task_id=task_id,
        objective_metric=objective_metric,
        reference_metrics=reference_metrics,
        reference_mean=reference_mean,
        reference_std=reference_std,
        reference_seed_count=reference_seed_count,
    )
    policy = gate_policy(base_dir)
    significance = seed_eval_significance_decision(
        variant_mean=result.get("mean"),
        reference_mean=reference_mean_value,
        reference_std=reference_std_value,
        reference_seed_count=reference_count,
        objective_metric=objective_metric,
        direction=get_metric_direction(objective_metric),
        min_effect_size=(
            float(min_effect_size)
            if min_effect_size is not None
            else 0.0
        ),
        variant_seed_count=result.get("valid_metric_seeds"),
        min_relative_improvement=(
            float(min_relative_improvement)
            if min_relative_improvement is not None
            else policy["min_relative_improvement"]
        ),
        min_absolute_improvement=(
            float(min_absolute_improvement)
            if min_absolute_improvement is not None
            else policy["seed_eval_min_absolute_improvement"]
        ),
        reference_std_multiplier=(
            float(reference_std_multiplier)
            if reference_std_multiplier is not None
            else 0.0
        ),
        required_seed_count=num_seeds,
    )
    result["reference_kind"] = "current_best"
    result["reference_mean"] = reference_mean_value
    result["reference_std"] = reference_std_value
    result["reference_seed_count"] = reference_count
    result["current_best_mean"] = reference_mean_value
    result["current_best_std"] = reference_std_value
    result["current_best_seed_count"] = reference_count
    result["significance_decision"] = significance
    result["promoted_to_current_best"] = False

    # Save result
    artifact_name = compact_result_save_path(task_id, node_id, "seed_eval") + ".json"
    result_path = os.path.join(runs_dir, artifact_name)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    result["result_path"] = result_path

    summary_node = create_node(
        task_id=task_id,
        node_id=f"{node_id}_seed_eval_summary",
        action_type="seed_eval",
        model_name=model_config.get("model_name", ""),
        model_config=model_config,
        objective_metric=objective_metric,
        metrics=mean_metrics,
        status="success" if result.get("mean") is not None else "failed",
        gate_decision=significance,
        artifact_paths=[result_path],
        llm_summary="multi-seed evaluation summary with significance decision",
    )
    summary_node["parent_id"] = node_id
    summary_node["seed_eval_result"] = {
        "mean": result.get("mean"),
        "std": result.get("std"),
        "successful_seeds": result.get("successful_seeds"),
        "valid_metric_seeds": result.get("valid_metric_seeds"),
        "reference_kind": "current_best",
        "reference_mean": reference_mean_value,
        "reference_std": reference_std_value,
        "reference_seed_count": reference_count,
        "decision": significance.get("decision"),
    }
    append_node(task_id, summary_node, str(runs_root(base_dir)))

    if promote_on_accept and significance.get("decision") == "accept":
        promotion = dict(promotion_metadata or {})
        source_ref = dict(promotion.get("source_ref") or {})
        candidate_snapshot_id = str(promotion.get("candidate_snapshot_id") or source_ref.get("candidate_snapshot_id") or "").strip()
        if candidate_snapshot_id:
            source_ref.update(
                {
                    "kind": "source_snapshot",
                    "candidate_snapshot_id": candidate_snapshot_id,
                    "base_snapshot_id": str(promotion.get("base_snapshot_id") or source_ref.get("base_snapshot_id") or ""),
                    "patch_path": str(promotion.get("patch_path") or source_ref.get("patch_path") or ""),
                    "changed_files": list(promotion.get("changed_files") or source_ref.get("changed_files") or []),
                    "parent_candidate_id": str(promotion.get("parent_candidate_id") or source_ref.get("parent_candidate_id") or ""),
                    "source_checkout": str(source_checkout or promotion.get("source_checkout") or ""),
                }
            )
        verified_record = {
            "candidate_id": candidate_id or variant_path or node_id,
            "candidate_kind": str(promotion.get("candidate_kind") or ("variant" if variant_path else "source_patch")),
            "display_name": variant_path or model_config.get("model_name") or candidate_id or node_id,
            "import_path": model_config.get("model_name", ""),
            "adapter": model_config.get("adapter"),
            "metrics": mean_metrics or {objective_metric: result.get("mean")},
            "node_id": node_id,
            "source": str(promotion.get("source") or "seed_eval_accept"),
            "objective_metric": objective_metric,
            "model_config": model_config,
            "parent_candidate_id": promotion.get("parent_candidate_id"),
            "candidate_snapshot_id": candidate_snapshot_id or None,
            "base_snapshot_id": promotion.get("base_snapshot_id") or None,
            "patch_path": promotion.get("patch_path") or None,
            "changed_files": list(promotion.get("changed_files") or []),
            "source_ref": source_ref if source_ref else {},
            "scientific_status": "verified_seed_eval",
            "engineering_status": "verified",
            "best_artifact_paths": [result_path],
            "variant_path": variant_path,
            "seed_eval": {
                "num_seeds": num_seeds,
                "seed_list": actual_seed_list,
                "successful_seeds": result.get("successful_seeds"),
                "mean": result.get("mean"),
                "std": result.get("std"),
                "metric_stats": {
                    objective_metric: {
                        "mean": result.get("mean"),
                        "std": result.get("std"),
                        "seed_count": result.get("valid_metric_seeds"),
                    }
                },
                "decision": significance,
            },
        }
        state_after_promotion = promote_provisional_to_current_best(base_dir, task_id, verified_record, clear_provisional=True)
        promoted = bool(
            state_after_promotion.current_best
            and state_after_promotion.current_best.candidate_id == verified_record["candidate_id"]
        )
        result["promoted_to_current_best"] = promoted
        promotion_policy = gate_policy(base_dir)
        result["promotion_decision"] = {
            "decision": "promote" if promoted else "reject",
            "candidate_id": verified_record["candidate_id"],
            "current_best_candidate_id": (
                state_after_promotion.current_best.candidate_id
                if state_after_promotion.current_best
                else ""
            ),
            "objective_metric": objective_metric,
            "candidate_value": verified_record["metrics"].get(objective_metric),
            "current_best_value": (
                (state_after_promotion.current_best.metrics or {}).get(objective_metric)
                if state_after_promotion.current_best
                else None
            ),
            "min_relative_improvement": promotion_policy["min_relative_improvement"],
            "reason": (
                "candidate met the unified promotion threshold"
                if promoted
                else "seed eval did not promote current_best under the unified threshold"
            ),
        }
        if promoted:
            result["promoted_record"] = {
                "candidate_id": verified_record["candidate_id"],
                "objective_metric": objective_metric,
                "objective_value": verified_record["metrics"].get(objective_metric),
                "source": verified_record["source"],
            }

    record_runtime_event(
        base_dir,
        task_id,
        "seed_eval_completed",
        {
            "node_id": node_id,
            "decision": significance.get("decision"),
            "promoted_to_current_best": result["promoted_to_current_best"],
            "promotion_decision": result.get("promotion_decision") or {},
            "result_path": result_path,
        },
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    return result
