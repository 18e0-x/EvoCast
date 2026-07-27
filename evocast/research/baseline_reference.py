from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from evocast.domain.knowledge_paths import task_knowledge_dir
from evocast.domain.metric_semantics import DEFAULT_OBJECTIVE_METRIC
from evocast.policy.experiment_policy import baseline_seed, task_build_mode
from evocast.research.baseline_knowledge import (
    bind_reference_to_task,
    build_reference_signature,
    load_reference_result,
    write_reference_result,
)
from evocast.runners.seed_runner import run_seed_evaluation
from evocast.state.domain_store import list_round_records

SCHEMA_VERSION = "baseline_reference_v1"


def baseline_reference_path(base_dir: str, task_id: str) -> Path:
    return task_knowledge_dir(base_dir, task_id) / "baseline_reference.json"


def _stable_hash(payload: Any) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_baseline_reference(base_dir: str, task_id: str) -> Dict[str, Any]:
    payload = _read_json(baseline_reference_path(base_dir, task_id), {})
    return payload if isinstance(payload, dict) else {}


def has_valid_baseline_reference(base_dir: str, task_id: str, *, objective_metric: str = DEFAULT_OBJECTIVE_METRIC) -> bool:
    payload = load_baseline_reference(base_dir, task_id)
    stats = ((payload.get("metric_stats") or {}).get(objective_metric) or {})
    return bool(
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("candidate_kind") == "baseline"
        and not payload.get("variant_path")
        and payload.get("source_clean") is True
        and int(stats.get("seed_count") or 0) >= 1
        and isinstance(stats.get("mean"), (int, float))
    )


def require_baseline_reference(base_dir: str, task_id: str, *, objective_metric: str = DEFAULT_OBJECTIVE_METRIC) -> Dict[str, Any]:
    payload = load_baseline_reference(base_dir, task_id)
    if not has_valid_baseline_reference(base_dir, task_id, objective_metric=objective_metric):
        raise RuntimeError(
            "BASELINE_REFERENCE_REQUIRED: frozen clean baseline_reference.json must exist before variant seed evaluation."
        )
    return payload


def _metric_stats_from_seed_result(seed_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    mean_metrics = dict(seed_result.get("mean_metrics") or {})
    per_seed = [item for item in list(seed_result.get("per_seed") or []) if isinstance(item, dict)]
    for key, mean in mean_metrics.items():
        if not isinstance(mean, (int, float)):
            continue
        values = [
            float((item.get("metrics") or {}).get(key))
            for item in per_seed
            if isinstance((item.get("metrics") or {}).get(key), (int, float))
        ]
        std = None
        if len(values) >= 2:
            avg = sum(values) / len(values)
            std = (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5
        stats[key] = {"mean": float(mean), "std": std, "seed_count": len(values) or int(seed_result.get("valid_metric_seeds") or 0)}
    return stats


def write_initial_baseline_reference(
    *,
    task_id: str,
    base_dir: str,
    config_path: str,
    baseline_record: Dict[str, Any],
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC,
    num_seeds: int = 3,
    base_seed: int = 2021,
    seed_list: list[int] | None = None,
    force: bool = False,
) -> Dict[str, Any]:
    path = baseline_reference_path(base_dir, task_id)
    if path.exists() and not force:
        existing = load_baseline_reference(base_dir, task_id)
        if has_valid_baseline_reference(base_dir, task_id, objective_metric=objective_metric):
            return existing
    has_research_round = any(
        bool(record.get("counts_toward_research_budget"))
        if record.get("counts_toward_research_budget") is not None
        else str(record.get("round_scope") or "research").strip().lower() == "research"
        for record in list_round_records(base_dir, task_id)
    )
    if not force and has_research_round:
        raise RuntimeError("BASELINE_REFERENCE_MUST_PRECEDE_RESEARCH: generate baseline_reference.json before any Research round.")

    model_config = dict(baseline_record.get("model_config") or {})
    if not model_config.get("model_name"):
        raise RuntimeError("BASELINE_REFERENCE_REQUIRES_MODEL_CONFIG: baseline_record.model_config.model_name is required.")
    if model_config.get("variant_path"):
        raise RuntimeError("BASELINE_REFERENCE_CONTAMINATED_CONFIG: baseline model_config must not contain variant_path.")

    actual_seed_list = [int(seed) for seed in seed_list] if seed_list is not None else [
        int(base_seed or 2021) + i for i in range(max(1, int(num_seeds or 1)))
    ]
    initial_seed = int(baseline_record.get("seed") or baseline_seed(base_dir))
    if initial_seed not in actual_seed_list:
        raise ValueError(
            f"initial baseline seed {initial_seed} is not in seed_eval seed list {actual_seed_list}"
        )

    formal_baseline_knowledge_enabled = not bool(task_build_mode(base_dir, task_id))
    reference_signature: Dict[str, Any] = {}
    if formal_baseline_knowledge_enabled:
        config_data = _read_json(Path(config_path), {})
        reference_signature = build_reference_signature(
            task_id=task_id,
            base_dir=base_dir,
            config_data=config_data if isinstance(config_data, dict) else {},
            baseline_record=baseline_record,
            objective_metric=objective_metric,
            seed_list=actual_seed_list,
        )
        cached_reference = load_reference_result(base_dir, reference_signature, objective_metric=objective_metric)
        if cached_reference:
            return bind_reference_to_task(
                base_dir=base_dir,
                task_id=task_id,
                cached_payload=cached_reference,
                objective_metric=objective_metric,
            )

    remaining_seed_list = [seed for seed in actual_seed_list if seed != initial_seed]
    initial_metrics = dict(baseline_record.get("metrics") or {})
    initial_value = initial_metrics.get(objective_metric)
    precomputed_seed_values = [{
        "seed": initial_seed,
        "objective_value": initial_value,
        "metrics": initial_metrics,
        "success": initial_value is not None,
        "source": "baseline_initial_run",
        "artifact_provenance": None,
    }]
    seed_result = run_seed_evaluation(
        task_id=task_id,
        node_id=f"{baseline_record.get('node_id') or baseline_record.get('candidate_id') or 'baseline'}_baseline_reference",
        model_config=model_config,
        config_path=config_path,
        objective_metric=objective_metric,
        num_seeds=len(actual_seed_list),
        base_seed=initial_seed,
        seed_list=remaining_seed_list,
        seed_universe=actual_seed_list,
        precomputed_seed_values=precomputed_seed_values,
        base_dir=base_dir,
        reference_metrics=dict(baseline_record.get("metrics") or {}),
        reference_mean=(baseline_record.get("metrics") or {}).get(objective_metric),
        reference_std=0.0,
        reference_seed_count=1,
        candidate_id=baseline_record.get("candidate_id") or baseline_record.get("node_id") or "baseline",
        variant_path=None,
        promote_on_accept=False,
        min_effect_size=None,
    )
    metric_stats = _metric_stats_from_seed_result(seed_result)
    objective_stats = metric_stats.get(objective_metric) or {}
    if int(objective_stats.get("seed_count") or 0) < len(actual_seed_list):
        raise RuntimeError(
            "BASELINE_REFERENCE_INCOMPLETE: baseline reference requires valid metrics for all configured seeds."
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "candidate_id": baseline_record.get("candidate_id"),
        "node_id": baseline_record.get("node_id"),
        "model": baseline_record.get("display_name") or baseline_record.get("model_name"),
        "candidate_kind": "baseline",
        "variant_path": None,
        "objective_metric": objective_metric,
        "seed_list": list(seed_result.get("seed_list") or []),
        "single_seed_metrics": dict(baseline_record.get("metrics") or {}),
        "metric_stats": metric_stats,
        "source_clean": True,
        "generated_before_first_variant": True,
        "model_config_hash": _stable_hash(model_config),
        "path": str(path),
        "result_path": seed_result.get("result_path"),
        "created_at": datetime.now().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if formal_baseline_knowledge_enabled:
        write_reference_result(
            base_dir=base_dir,
            task_id=task_id,
            signature=reference_signature,
            baseline_reference=payload,
        )
    return payload
