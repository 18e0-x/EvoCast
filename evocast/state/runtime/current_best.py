"""Runtime-state backed helpers for the current best candidate."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from evocast.domain.metric_semantics import annotate_metric_payload, canonicalize_metric_values
from evocast.policy.gate import get_metric_direction
from evocast.state.runtime.store import load_runtime_state, runtime_state_path, sync_current_best


def normalize_best_record(record: Dict[str, Any], source: str = "unknown") -> Dict[str, Any]:
    if not record:
        return {}
    metrics = canonicalize_metric_values(record.get("best_metrics") or record.get("metrics") or {})
    model_name = (
        record.get("display_name")
        or record.get("model_name")
        or record.get("best_model_name")
        or record.get("best_model")
        or record.get("candidate_id")
        or record.get("node_id")
    )
    best_node_id = record.get("node_id") or record.get("best_node_id")
    normalized = {
        "source": record.get("source", source),
        "best_node_id": best_node_id,
        "best_model_name": model_name,
        "best_tier": record.get("tier") or record.get("best_tier"),
        "best_seed": record.get("best_seed") or record.get("seed"),
        "best_metrics": metrics,
        "best_artifact_paths": record.get("best_artifact_paths") or record.get("artifact_paths") or [],
        "model_config": record.get("model_config", {}),
        "adapter": record.get("adapter"),
        "family": record.get("family"),
        "tags": record.get("tags", []),
        "candidate_id": record.get("candidate_id") or best_node_id or model_name,
        "candidate_kind": record.get("candidate_kind") or ("baseline" if "baseline" in source else "variant"),
        "display_name": model_name,
        "metrics": metrics,
        "node_id": best_node_id,
        "tier": record.get("tier") or record.get("best_tier"),
    }
    objective_metric = record.get("objective_metric")
    if objective_metric:
        normalized["objective_metric"] = objective_metric
        normalized["metric_semantics"] = annotate_metric_payload(
            objective_metric=str(objective_metric),
            metrics=metrics,
        )
    return normalized


def load_current_best(base_dir: str, task_id: str) -> Dict[str, Any]:
    runtime_state = load_runtime_state(base_dir, task_id)
    if runtime_state.current_best.candidate_id and runtime_state.baseline.candidate_id:
        metric = (
            runtime_state.current_best.objective_metric
            or runtime_state.baseline.objective_metric
            or runtime_state.objective_metric
        )
        current_value = (runtime_state.current_best.metrics or {}).get(metric)
        baseline_value = (runtime_state.baseline.metrics or {}).get(metric)
        if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
            direction = get_metric_direction(metric)
            baseline_better = is_better_value(float(baseline_value), float(current_value), direction)
            if baseline_better:
                return normalize_best_record(
                    runtime_state.baseline.to_dict(),
                    source=(runtime_state.baseline.source or "runtime_baseline"),
                )
    if runtime_state.current_best.candidate_id:
        return normalize_best_record(
            runtime_state.current_best.to_dict(),
            source=(runtime_state.current_best.source or "runtime_state"),
        )
    if runtime_state.baseline.candidate_id:
        return normalize_best_record(
            runtime_state.baseline.to_dict(),
            source=(runtime_state.baseline.source or "runtime_baseline"),
        )
    return {}


def save_current_best(base_dir: str, task_id: str, record: Dict[str, Any]) -> str:
    payload = normalize_best_record(record, source=record.get("source", "runtime_state"))
    sync_current_best(base_dir, task_id, payload)
    return runtime_state_path(base_dir, task_id)


def build_baseline_current_best(
    best_baseline: Dict[str, Any],
    objective_metric: str,
    all_results: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    record = normalize_best_record(best_baseline, source="baseline_search")
    if objective_metric and record.get("best_metrics") and objective_metric not in record["best_metrics"]:
        value = (best_baseline.get("metrics") or {}).get(objective_metric)
        if value is not None:
            record["best_metrics"][objective_metric] = value
            record["metrics"][objective_metric] = value
    return record


def is_better_value(candidate: float, reference: float, direction: str = "lower") -> bool:
    return candidate < reference if direction == "lower" else candidate > reference
