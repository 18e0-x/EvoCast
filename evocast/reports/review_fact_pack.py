from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from evocast.research.baseline_reference import load_baseline_reference
from evocast.domain.knowledge_paths import task_knowledge_dir, task_runs_dir
from evocast.research.dataset_profile import dataset_profile_path, load_dataset_profile
from evocast.state.stage_timing import build_stage_timing_summary, write_stage_timing_summary
from evocast.state.token_usage import build_token_usage_summary, write_token_usage_summary
from evocast.state.cost_ledger import ledger_path
from evocast.state.domain_store import load_domain_state, load_task_config

METRIC_KEYS = ["mse_norm", "mae_norm", "rmse_norm", "mse", "mae", "rmse"]


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_jsonl(path: Path, *, limit: int = 200) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(item)
    except Exception:
        return rows
    return rows


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _compact_metrics(metrics: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    return {key: metrics.get(key) for key in METRIC_KEYS if key in metrics}


def _baseline_search_fact(search_state: Dict[str, Any], *, objective_metric: str, metric_direction: str) -> Dict[str, Any]:
    """Return the compact, auditable baseline-tournament rows for the report."""
    state = dict(search_state or {})
    order = {
        str(item.get("node_id") or ""): int(item.get("index") or index)
        for index, item in enumerate(list(state.get("candidate_order") or []), start=1)
        if isinstance(item, dict)
    }
    rows: List[Dict[str, Any]] = []
    for fallback_index, result in enumerate(list(state.get("run_results") or []), start=1):
        if not isinstance(result, dict):
            continue
        metrics = dict(result.get("metrics") or {})
        value = metrics.get(objective_metric)
        rows.append(
            {
                "order": order.get(str(result.get("node_id") or ""), fallback_index),
                "model": result.get("model_key"),
                "type": result.get("family") or result.get("tier"),
                "metric": value if isinstance(value, (int, float)) else None,
                "elapsed_seconds": result.get("elapsed_seconds"),
                "status": result.get("status"),
                "rank": None,
            }
        )
    reverse = str(metric_direction or "lower_is_better") == "higher_is_better"
    ranked = sorted(
        [row for row in rows if isinstance(row.get("metric"), (int, float))],
        key=lambda row: float(row["metric"]),
        reverse=reverse,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    rows.sort(key=lambda row: int(row.get("order") or 0))
    return {
        "strategy": state.get("strategy"),
        "objective_metric": objective_metric,
        "rows": rows,
    }


def _compact_metric_delta(delta: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(delta, dict):
        return {}
    return {key: delta.get(key) for key in METRIC_KEYS if key in delta}


def _metric_stat_mean(seed_eval: Dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(seed_eval, dict):
        return None
    stats = seed_eval.get("metric_stats")
    if not isinstance(stats, dict):
        return None
    item = stats.get(key)
    if not isinstance(item, dict):
        return None
    mean = item.get("mean")
    return float(mean) if isinstance(mean, (int, float)) else None


def _numeric_metric(metrics: Dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _metric_comparison(
    *,
    metrics: Dict[str, Any] | None,
    seed_eval: Dict[str, Any] | None,
    reference_metrics: Dict[str, Any] | None,
    reference_seed_eval: Dict[str, Any] | None,
    objective_metric: str = "",
    metric_direction: str = "lower_is_better",
) -> Dict[str, Any]:
    """Build one metric comparison model shared by ablations and research rounds."""
    rows: Dict[str, Any] = {}
    candidate_has_seed = any(_metric_stat_mean(seed_eval, key) is not None for key in METRIC_KEYS)
    candidate_source = "seed_eval" if candidate_has_seed else "single_seed"
    reference_source = "seed_eval" if candidate_has_seed else "single_seed"
    lower_is_better = str(metric_direction or "lower_is_better") != "higher_is_better"
    for key in METRIC_KEYS:
        value = _metric_stat_mean(seed_eval, key) if candidate_has_seed else _numeric_metric(metrics, key)
        reference = (
            _metric_stat_mean(reference_seed_eval, key)
            if candidate_has_seed
            else _numeric_metric(reference_metrics, key)
        )
        if value is None and reference is None:
            continue
        delta = None
        relative_delta = None
        improvement = None
        outcome = ""
        if value is not None and reference not in (None, 0):
            delta = value - float(reference)
            relative_delta = delta / abs(float(reference))
            improvement = -relative_delta if lower_is_better else relative_delta
            if abs(improvement) < 1e-12:
                outcome = "unchanged"
            elif improvement > 0:
                outcome = "improved"
            else:
                outcome = "degraded"
        rows[key] = {
            "value": value,
            "reference_kind": "current_best",
            "reference": reference,
            "delta": delta,
            "relative_delta": relative_delta,
            "relative_improvement": improvement,
            "outcome": outcome,
            "candidate_source": candidate_source,
            "reference_source": reference_source,
            "direction": "lower_is_better" if lower_is_better else "higher_is_better",
        }
    objective = rows.get(objective_metric) if objective_metric else {}
    return {
        "schema_version": "metric_comparison_v1",
        "reference_kind": "current_best",
        "candidate_source": candidate_source,
        "reference_source": reference_source,
        "objective_metric": objective_metric,
        "metric_direction": "lower_is_better" if lower_is_better else "higher_is_better",
        "metrics": rows,
        "objective": objective or {},
    }


def _as_number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _comparison_row(
    *,
    key: str,
    value: float | None,
    reference: float | None,
    lower_is_better: bool,
    candidate_source: str,
    reference_source: str,
    reference_local: bool,
) -> Dict[str, Any]:
    delta = None
    relative_delta = None
    improvement = None
    outcome = ""
    if value is not None and reference not in (None, 0):
        delta = float(value) - float(reference)
        relative_delta = delta / abs(float(reference))
        improvement = -relative_delta if lower_is_better else relative_delta
        if abs(improvement) < 1e-12:
            outcome = "unchanged"
        elif improvement > 0:
            outcome = "improved"
        else:
            outcome = "degraded"
    return {
        "value": value,
        "reference_kind": "current_best",
        "reference": reference,
        "delta": delta,
        "relative_delta": relative_delta,
        "relative_improvement": improvement,
        "outcome": outcome,
        "candidate_source": candidate_source,
        "reference_source": reference_source,
        "reference_local": reference_local,
        "direction": "lower_is_better" if lower_is_better else "higher_is_better",
    }


def _recorded_seed_objective_comparison(
    record: Dict[str, Any],
    *,
    objective_metric: str,
    lower_is_better: bool,
) -> Dict[str, Any] | None:
    seed_eval = record.get("seed_eval") if isinstance(record.get("seed_eval"), dict) else {}
    decision = seed_eval.get("significance_decision") if isinstance(seed_eval.get("significance_decision"), dict) else {}
    if str(decision.get("reference_kind") or "current_best") != "current_best":
        return None
    value = _as_number(decision.get("variant_mean"))
    if value is None:
        value = _metric_stat_mean(seed_eval, objective_metric)
    reference = _as_number(decision.get("reference_mean"))
    if value is None and reference is None:
        return None
    row = _comparison_row(
        key=objective_metric,
        value=value,
        reference=reference,
        lower_is_better=lower_is_better,
        candidate_source="seed_eval",
        reference_source="seed_eval_significance",
        reference_local=True,
    )
    recorded_relative = _as_number(decision.get("relative_improvement"))
    if recorded_relative is not None:
        row["relative_improvement"] = recorded_relative
        row["relative_delta"] = -recorded_relative if lower_is_better else recorded_relative
        if abs(recorded_relative) < 1e-12:
            row["outcome"] = "unchanged"
        elif recorded_relative > 0:
            row["outcome"] = "improved"
        else:
            row["outcome"] = "degraded"
    row["decision"] = decision.get("decision")
    row["accept_rule"] = decision.get("accept_rule")
    return row


def _recorded_gate_objective_comparison(
    record: Dict[str, Any],
    *,
    objective_metric: str,
    lower_is_better: bool,
) -> Dict[str, Any] | None:
    gate_event = record.get("gate_event") if isinstance(record.get("gate_event"), dict) else {}
    gate = record.get("gate") if isinstance(record.get("gate"), dict) else {}
    nested_gate = gate_event.get("gate") if isinstance(gate_event.get("gate"), dict) else {}
    gate = {**nested_gate, **gate}
    value = _as_number(gate_event.get("candidate_value"))
    if value is None:
        value = _as_number(gate.get("current_value"))
    if value is None:
        value = _numeric_metric(record.get("metrics"), objective_metric)
    reference = _as_number(gate_event.get("reference_value"))
    if reference is None:
        reference = _as_number(gate.get("reference_value"))
    if reference is None:
        reference = _as_number(gate.get("current_best_value"))
    if value is None and reference is None:
        return None
    row = _comparison_row(
        key=objective_metric,
        value=value,
        reference=reference,
        lower_is_better=lower_is_better,
        candidate_source="single_seed",
        reference_source="gate_event",
        reference_local=True,
    )
    recorded_relative = _as_number(gate_event.get("relative_improvement"))
    if recorded_relative is None:
        recorded_relative = _as_number(gate.get("reference_relative_improvement"))
    if recorded_relative is not None:
        row["relative_improvement"] = recorded_relative
        row["relative_delta"] = -recorded_relative if lower_is_better else recorded_relative
        if abs(recorded_relative) < 1e-12:
            row["outcome"] = "unchanged"
        elif recorded_relative > 0:
            row["outcome"] = "improved"
        else:
            row["outcome"] = "degraded"
    row["decision"] = gate_event.get("decision") or gate.get("decision")
    row["threshold_met"] = gate_event.get("threshold_met") if gate_event.get("threshold_met") is not None else gate.get("beats_reference_threshold")
    return row


def _recorded_metric_comparison(
    record: Dict[str, Any],
    *,
    fallback_reference_metrics: Dict[str, Any] | None,
    fallback_reference_seed_eval: Dict[str, Any] | None,
    objective_metric: str = "",
    metric_direction: str = "lower_is_better",
) -> Dict[str, Any]:
    """Build report comparison from recorded gate/seed-eval evidence first.

    Reports must restate the live decision path. They must not recompute every
    round against one global report-time current_best, because current_best can
    change between rounds.
    """
    lower_is_better = str(metric_direction or "lower_is_better") != "higher_is_better"
    fallback = _metric_comparison(
        metrics=record.get("metrics"),
        seed_eval=record.get("seed_eval"),
        reference_metrics=fallback_reference_metrics,
        reference_seed_eval=fallback_reference_seed_eval,
        objective_metric=objective_metric,
        metric_direction=metric_direction,
    )
    rows = dict(fallback.get("metrics") or {})
    objective_row = None
    if objective_metric:
        objective_row = _recorded_seed_objective_comparison(
            record,
            objective_metric=objective_metric,
            lower_is_better=lower_is_better,
        )
        if objective_row is None:
            objective_row = _recorded_gate_objective_comparison(
                record,
                objective_metric=objective_metric,
                lower_is_better=lower_is_better,
            )
    if objective_metric and objective_row is not None:
        rows[objective_metric] = objective_row
    objective = rows.get(objective_metric) if objective_metric else {}
    reference_source = (
        str((objective or {}).get("reference_source") or "")
        or str(fallback.get("reference_source") or "")
    )
    candidate_source = (
        str((objective or {}).get("candidate_source") or "")
        or str(fallback.get("candidate_source") or "")
    )
    return {
        "schema_version": "metric_comparison_v2",
        "reference_kind": "current_best",
        "candidate_source": candidate_source,
        "reference_source": reference_source,
        "objective_metric": objective_metric,
        "metric_direction": "lower_is_better" if lower_is_better else "higher_is_better",
        "metrics": rows,
        "objective": objective or {},
    }


def _ablation_metrics(record: Dict[str, Any], final_validity: Dict[str, Any]) -> Dict[str, Any]:
    usable_status = str(record.get("usable_evidence_status") or final_validity.get("usable_evidence_status") or "")
    validity_status = str(final_validity.get("validity_status") or "")
    status = str(record.get("status") or final_validity.get("status") or "")
    if usable_status and usable_status != "usable_evidence":
        return {}
    if validity_status and validity_status not in {"valid_ablation", "usable_evidence"}:
        return {}
    if status.startswith("failed"):
        return {}
    return _compact_metrics(record.get("metrics"))


def attach_token_usage_summary(fact_pack: Dict[str, Any], *, task_id: str, base_dir: str) -> Dict[str, Any]:
    updated = dict(fact_pack)
    write_token_usage_summary(task_id, base_dir)
    updated["token_usage"] = build_token_usage_summary(task_id, base_dir)
    return updated


def attach_stage_timing_summary(fact_pack: Dict[str, Any], *, task_id: str, base_dir: str) -> Dict[str, Any]:
    updated = dict(fact_pack)
    write_stage_timing_summary(task_id, base_dir)
    updated["stage_timing"] = build_stage_timing_summary(task_id, base_dir)
    return updated


def _compact_research_state(state: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(state, dict) or not state:
        return {}
    beliefs = []
    for item in list((state.get("beliefs") or {}).values()):
        if not isinstance(item, dict):
            continue
        beliefs.append(
            {
                "belief_id": item.get("belief_id"),
                "claim": item.get("claim"),
                "belief_type": item.get("belief_type"),
                "confidence": item.get("confidence"),
                "support_event_ids": list(item.get("support_event_ids") or [])[:6],
                "refute_event_ids": list(item.get("refute_event_ids") or [])[:6],
                "uncertainty": list(item.get("uncertainty") or [])[:6],
                "status": item.get("status"),
            }
        )
    beliefs.sort(key=lambda row: float(row.get("confidence") or 0), reverse=True)
    questions = []
    for item in list((state.get("questions") or {}).values()):
        if not isinstance(item, dict):
            continue
        questions.append(
            {
                "question_id": item.get("question_id"),
                "question": item.get("question"),
                "competing_hypotheses": list(item.get("competing_hypotheses") or [])[:4],
                "falsifiable_prediction": item.get("falsifiable_prediction"),
                "information_value": item.get("information_value"),
                "status": item.get("status"),
            }
        )
    questions.sort(key=lambda row: float(row.get("information_value") or 0), reverse=True)
    agenda = []
    for item in list(state.get("agenda") or []):
        if not isinstance(item, dict):
            continue
        agenda.append(
            {
                "agenda_id": item.get("agenda_id"),
                "question_id": item.get("question_id"),
                "action_type": item.get("action_type"),
                "priority": item.get("priority"),
                "rationale": item.get("rationale"),
                "status": item.get("status"),
                "target_path": ((item.get("intervention_spec") or {}).get("target_path") if isinstance(item.get("intervention_spec"), dict) else None)
                or (((item.get("instrument_spec") or {}).get("inputs") or {}).get("target_path") if isinstance(item.get("instrument_spec"), dict) else None),
                "completion_criteria": list(item.get("completion_criteria") or [])[:4],
            }
        )
    agenda.sort(key=lambda row: (row.get("status") != "pending", -float(row.get("priority") or 0)))
    interpretations = []
    for item in list((state.get("interpretations") or {}).values()):
        if not isinstance(item, dict):
            continue
        interpretations.append(
            {
                "interpretation_id": item.get("interpretation_id"),
                "question_id": item.get("question_id"),
                "result_status": item.get("result_status"),
                "conclusion": item.get("conclusion"),
                "evidence_event_ids": list(item.get("evidence_event_ids") or [])[:6],
            }
        )
    return {
        "schema_version": state.get("schema_version"),
        "current_question_id": state.get("current_question_id"),
        "event_count": len(list(state.get("event_index") or [])),
        "artifact_count": len(dict(state.get("artifact_index") or {})),
        "top_beliefs": beliefs[:8],
        "open_questions": questions[:8],
        "agenda": agenda[:8],
        "interpretations": interpretations[:8],
    }


def _compact_dataset_diagnosis(profile: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(profile, dict) or not profile:
        return {}
    basic = dict(profile.get("basic_facts") or {})
    raw = dict(profile.get("raw_characteristics") or {})
    narrative = dict(profile.get("llm_narrative") or {})
    return {
        "status": profile.get("status"),
        "characteristics_engine": profile.get("characteristics_engine"),
        "basic_facts": {
            "dataset_name": basic.get("dataset_name"),
            "dataset_path": basic.get("dataset_path"),
            "task_mode": basic.get("task_mode"),
            "frequency": basic.get("frequency"),
            "seq_len": basic.get("seq_len"),
            "horizon": basic.get("horizon"),
            "num_variables": basic.get("num_variables"),
            "train_shape": basic.get("train_shape"),
            "missing_ratio": basic.get("missing_ratio"),
            "is_univariate": basic.get("is_univariate"),
        },
        "raw_characteristics": {
            key: raw.get(key)
            for key in [
                "Correlation",
                "Transition",
                "Shifting",
                "Seasonality",
                "Trend",
                "Stationarity",
                "Short_term_jsd",
                "Long_term_jsd",
            ]
            if key in raw
        },
        "derived_claims": [
            {
                "claim": item.get("claim"),
                "confidence": item.get("confidence"),
                "evidence": item.get("evidence"),
                "research_implication": item.get("research_implication"),
            }
            for item in list(profile.get("derived_claims") or [])
            if isinstance(item, dict)
        ],
        "research_implications": [
            {
                "topic": item.get("topic"),
                "priority": item.get("priority"),
                "confidence": item.get("confidence"),
                "reason": item.get("reason"),
            }
            for item in list(profile.get("research_implications") or [])
            if isinstance(item, dict)
        ],
        "llm_narrative": {
            "status": narrative.get("status"),
            "dataset_summary": narrative.get("dataset_summary"),
            "research_interpretation": narrative.get("research_interpretation"),
            "suggested_opportunity_themes": [
                item
                for item in list(narrative.get("suggested_opportunity_themes") or [])
                if isinstance(item, dict)
            ],
            "limitations": [
                str(item)
                for item in list(narrative.get("limitations") or [])
                if str(item).strip()
            ],
        },
        "diagnostics": [
            {
                "severity": item.get("severity"),
                "code": item.get("code"),
                "message": item.get("message"),
            }
            for item in list(profile.get("diagnostics") or [])
            if isinstance(item, dict)
        ],
    }


def _resolve_existing_path(path_text: Any, *, root: Path) -> Path | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    raw = Path(text)
    candidates = [raw] if raw.is_absolute() else [root / raw, root.parent / raw, Path.cwd() / raw]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _seed_eval_metric_stats(
    seed_eval: Dict[str, Any] | None,
    *,
    objective_metric: str = "",
    root: Path,
) -> Dict[str, Any]:
    if not isinstance(seed_eval, dict) or not seed_eval:
        return {}
    detail = dict(seed_eval)
    detail_path = _resolve_existing_path(seed_eval.get("result_path"), root=root)
    if detail_path:
        loaded = _read_json(detail_path, {}) or {}
        if isinstance(loaded, dict):
            detail = {**loaded, **seed_eval}

    valid_seed_count = int(detail.get("valid_metric_seeds") or detail.get("successful_seeds") or 0)
    if valid_seed_count < 2:
        return {}

    stats: Dict[str, Dict[str, Any]] = {}
    mean_metrics = dict(detail.get("mean_metrics") or {})
    for key in METRIC_KEYS:
        if isinstance(mean_metrics.get(key), (int, float)):
            values = [
                float((item.get("metrics") or {}).get(key))
                for item in list(detail.get("per_seed") or [])
                if isinstance(item, dict) and isinstance((item.get("metrics") or {}).get(key), (int, float))
            ]
            stats[key] = {
                "mean": float(mean_metrics[key]),
                "std": _std(values) if len(values) >= 2 else None,
                "seed_count": len(values) or valid_seed_count,
            }

    if (
        objective_metric
        and objective_metric in METRIC_KEYS
        and objective_metric not in stats
        and isinstance(detail.get("mean"), (int, float))
    ):
        stats[objective_metric] = {
            "mean": float(detail["mean"]),
            "std": float(detail["std"]) if isinstance(detail.get("std"), (int, float)) else None,
            "seed_count": valid_seed_count,
        }
    return {
        "status": detail.get("status") or ("completed" if detail.get("mean") is not None else ""),
        "node_id": detail.get("node_id"),
        "result_path": str(detail_path or detail.get("result_path") or ""),
        "valid_metric_seeds": valid_seed_count,
        "metric_stats": stats,
        "promoted_to_current_best": detail.get("promoted_to_current_best"),
        "promotion_decision": detail.get("promotion_decision"),
        "significance_decision": detail.get("significance_decision"),
    }


def _artifact_index(paths: Iterable[Path], *, root: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in sorted({p for p in paths if p.exists()}):
        if path.is_dir():
            continue
        items.append(
            {
                "path": _safe_rel(path, root),
                "name": path.name,
                "size_bytes": path.stat().st_size,
            }
        )
    return items


def _exact_edits(ablation_dir: Path) -> List[Dict[str, Any]]:
    payload = _read_json(ablation_dir / "exact_edits.json", {})
    edits = payload.get("edits") if isinstance(payload, dict) else []
    if not isinstance(edits, list):
        return []
    result: List[Dict[str, Any]] = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        result.append(
            {
                "target_file": str(edit.get("target_file") or edit.get("path") or ""),
                "intent": str(edit.get("intent") or ""),
                "old_string": str(edit.get("old_string") or ""),
                "new_string": str(edit.get("new_string") or ""),
                "replace_all": bool(edit.get("replace_all")),
            }
        )
    return result


def _latest_json_by_prefix(directory: Path, prefix: str) -> Dict[str, Any]:
    candidates = sorted(directory.glob(f"{prefix}*.json"))
    if not candidates:
        return {}
    return _read_json(candidates[-1], {}) or {}


def _read_round_record(directory: Path) -> Dict[str, Any]:
    record = _read_json(directory / "round.json", {}) or {}
    if isinstance(record, dict):
        return record
    return {}


def _read_build_contract(directory: Path) -> Dict[str, Any]:
    contract = _read_json(directory / "build_contract.json", {}) or {}
    if isinstance(contract, dict):
        return contract
    return {}


def _read_metric_result(directory: Path) -> Dict[str, Any]:
    payload = _read_json(directory / "metric_result.json", {}) or _latest_json_by_prefix(directory, "metric_result")
    if isinstance(payload, dict):
        return payload
    return {}


def _metric_result_metrics(metric_result: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("metrics", "candidate_metrics", "result_metrics"):
        value = metric_result.get(key)
        if isinstance(value, dict):
            return value
    result = metric_result.get("result")
    if isinstance(result, dict):
        for key in ("metrics", "candidate_metrics"):
            value = result.get(key)
            if isinstance(value, dict):
                return value
    return {}


def _contract_direction(contract: Dict[str, Any]) -> Dict[str, Any]:
    protocol = contract.get("metric_protocol") if isinstance(contract.get("metric_protocol"), dict) else {}
    direction = protocol.get("research_direction")
    return direction if isinstance(direction, dict) else {}


def _contract_changed_files(record: Dict[str, Any], build_outcome: Dict[str, Any], contract: Dict[str, Any]) -> List[str]:
    candidates: List[Any] = [
        record.get("changed_files"),
        build_outcome.get("changed_files"),
        (record.get("build_outcome") or {}).get("changed_files") if isinstance(record.get("build_outcome"), dict) else None,
        contract.get("allowed_edit_files"),
    ]
    for value in candidates:
        if isinstance(value, list) and any(str(item).strip() for item in value):
            return sorted({str(item) for item in value if str(item).strip()})
    return []


def _compact_round_artifact_paths(artifacts: List[Dict[str, Any]]) -> Dict[str, str]:
    by_name: Dict[str, str] = {}
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        path = str(item.get("path") or "")
        if name and path:
            by_name[name] = path
    return by_name


def _ablation_fact(
    ablation_dir: Path,
    *,
    root: Path,
    objective_metric: str = "",
    canonical_record: Dict[str, Any] | None = None,
    canonical_contract: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    record = dict(canonical_record or {})
    if not record:  # compatibility input, migrated before normal report reads
        round_record = _read_round_record(ablation_dir)
        legacy_record = _read_json(ablation_dir / "ablation_record.json", {}) or {}
        record = {**legacy_record, **round_record}
    final_validity = _read_json(ablation_dir / "final_validity.json", {}) or {}
    contract = dict(canonical_contract or {}) or _read_build_contract(ablation_dir)
    build_outcome = _read_json(ablation_dir / "build_outcome.json", {}) or {}
    metric_result = _read_metric_result(ablation_dir)
    protocol = contract.get("metric_protocol") if isinstance(contract.get("metric_protocol"), dict) else {}
    materialize = _latest_json_by_prefix(ablation_dir, "materialize_variant")
    smoke = _latest_json_by_prefix(ablation_dir, "smoke_result")
    experiment = _latest_json_by_prefix(ablation_dir, "experiment_result")
    runtime_probe = _latest_json_by_prefix(ablation_dir, "runtime_probe")
    seed_eval = _read_json(ablation_dir / "seed_eval_result.json", {}) or record.get("seed_eval") or {}
    exact_edits = _exact_edits(ablation_dir)
    changed_files = sorted(
        {
            str(edit.get("target_file") or "")
            for edit in exact_edits
            if str(edit.get("target_file") or "")
        }
    )
    artifacts = _artifact_index(ablation_dir.glob("*.json"), root=root)
    seed_eval_summary = _seed_eval_metric_stats(seed_eval, objective_metric=objective_metric, root=root)
    metrics = (
        _compact_metrics((record.get("evaluation") or {}).get("metrics"))
        or _ablation_metrics(record, final_validity)
        or _compact_metrics(record.get("metrics"))
        or _compact_metrics(_metric_result_metrics(metric_result))
    )
    artifact_paths = _compact_round_artifact_paths(artifacts)
    display_idea = record.get("display_idea") or record.get("idea_summary") or protocol.get("research_direction")
    mechanism_id = (
        record.get("mechanism_id")
        or protocol.get("mechanism_id")
        or record.get("target_id")
        or contract.get("name")
        or ablation_dir.name
    )
    mechanism_name = record.get("mechanism_name") or record.get("display_idea") or mechanism_id
    return {
        "id": ablation_dir.name,
        "target_id": record.get("target_id"),
        "mechanism_id": mechanism_id,
        "mechanism_name": mechanism_name,
        "display_idea": display_idea,
        "summary": record.get("mechanism_summary") or record.get("idea_summary") or record.get("hypothesis"),
        "hypothesis": record.get("hypothesis") or contract.get("hypothesis"),
        "intent": record.get("exact_edit_intent") or record.get("research_intent") or contract.get("semantic_goal"),
        "status": record.get("status") or final_validity.get("status"),
        "usable_evidence_status": record.get("usable_evidence_status") or final_validity.get("usable_evidence_status"),
        "evaluation_stage": record.get("evaluation_stage") or protocol.get("evaluation_stage"),
        "variant_path": record.get("variant_path") or build_outcome.get("candidate_checkout") or materialize.get("variant_path"),
        "changed_files": changed_files or _contract_changed_files(record, build_outcome, contract),
        "exact_edits": exact_edits,
        "metrics": metrics,
        "metric_delta": _compact_metric_delta(record.get("metric_delta")),
        "interpretation": record.get("interpretation") or _read_json(ablation_dir / "interpretation.json", {}) or {},
        "seed_eval": seed_eval_summary or seed_eval,
        "gate_event": record.get("gate_event") if isinstance(record.get("gate_event"), dict) else {},
        "gate": record.get("gate") if isinstance(record.get("gate"), dict) else {},
        "build_contract": {
            "semantic_goal": contract.get("semantic_goal"),
            "hypothesis": contract.get("hypothesis"),
            "allowed_edit_files": list(contract.get("allowed_edit_files") or []),
            "research_direction": _contract_direction(contract),
        },
        "build_outcome": {
            "status": build_outcome.get("status"),
            "candidate_snapshot_id": build_outcome.get("candidate_snapshot_id"),
            "candidate_checkout": build_outcome.get("candidate_checkout"),
            "patch_path": build_outcome.get("patch_path"),
            "summary": build_outcome.get("summary"),
        },
        "artifact_paths": artifact_paths,
        "materialize": {
            "status": materialize.get("status"),
            "execution_scope": materialize.get("execution_scope"),
            "record_research_round": materialize.get("record_research_round"),
            "source_view": materialize.get("source_view"),
        },
        "runtime_probe": {
            "status": runtime_probe.get("status") or runtime_probe.get("runtime_contract", {}).get("status"),
            "passed": runtime_probe.get("passed"),
        },
        "smoke": {
            "status": smoke.get("status"),
            "success": smoke.get("success"),
            "next_action": smoke.get("next_action"),
        },
        "experiment": {
            "status": experiment.get("status"),
            "success": experiment.get("success"),
            "run_id": experiment.get("run_id"),
        },
        "artifacts": artifacts,
    }


def _research_rounds(
    knowledge_dir: Path,
    *,
    root: Path,
    objective_metric: str = "",
    canonical_records: List[Dict[str, Any]] | None = None,
    canonical_contracts: Dict[str, Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    rounds: List[Dict[str, Any]] = []
    round_dir = knowledge_dir / "rounds"
    if not round_dir.is_dir() and not canonical_records:
        return rounds
    candidates = _read_jsonl(knowledge_dir / "candidates.jsonl", limit=500)
    candidates_by_variant = {
        str(item.get("variant_path") or ""): item
        for item in candidates
        if isinstance(item, dict) and str(item.get("variant_path") or "")
    }
    candidates_by_run_id = {
        str(item.get("run_id") or ""): item
        for item in candidates
        if isinstance(item, dict) and str(item.get("run_id") or "")
    }
    records = list(canonical_records or [])
    if not records:
        records = [
            {**(_read_json(path, {}) or {}), "research_id": (_read_json(path, {}) or {}).get("research_id") or path.parent.name}
            for path in sorted(round_dir.glob("Research*/round.json"))
        ]
    for record in records:
        if not str(record.get("research_id") or "").startswith("Research"):
            continue
        round_id = str(record.get("research_id"))
        round_artifact_dir = round_dir / round_id
        round_files = sorted(round_artifact_dir.glob("*.json"))
        contract = dict((canonical_contracts or {}).get(round_id) or {}) or _read_build_contract(round_artifact_dir)
        build_outcome = _read_json(round_artifact_dir / "build_outcome.json", {}) or {}
        metric_result = _read_metric_result(round_artifact_dir)
        direction = (
            _read_json(round_artifact_dir / "research_direction.json", {}) or _contract_direction(contract)
        )
        if not isinstance(direction, dict):
            direction = {}
        experiment = _latest_json_by_prefix(round_artifact_dir, "run_experiment_result")
        module_spec = (
            _latest_json_by_prefix(round_artifact_dir, "architecture_module_spec")
            or (record.get("proposal") or {}).get("architecture_module_spec")
            or {}
        )
        module_manifest = _latest_json_by_prefix(round_artifact_dir, "module_manifest")
        module_validity = _latest_json_by_prefix(round_artifact_dir, "module_validity_probe")
        module_decision = _latest_json_by_prefix(round_artifact_dir, "module_experiment_decision")
        evolution_decision = _latest_json_by_prefix(round_artifact_dir, "module_evolution_decision")
        candidate = (
            candidates_by_variant.get(str(record.get("variant_path") or ""))
            or candidates_by_run_id.get(str(experiment.get("run_id") or record.get("run_id") or ""))
            or {}
        )
        metrics = (
            ((record.get("evaluation") or {}).get("metrics"))
            or record.get("metrics")
            or record.get("candidate_metrics")
            or _metric_result_metrics(metric_result)
            or experiment.get("metrics")
            or experiment.get("candidate_metrics")
            or candidate.get("metrics")
            or candidate.get("candidate_metrics")
        )
        artifacts = _artifact_index(round_files, root=root)
        artifact_paths = _compact_round_artifact_paths(artifacts)
        seed_eval = dict(record.get("seed_eval") or {})
        proposal = record.get("proposal") if isinstance(record.get("proposal"), dict) else {}
        opportunity = dict(proposal)
        opportunity_id = str(opportunity.get("opportunity_id") or "")
        target_mechanism = direction.get("target_mechanism") or record.get("mechanism_id") or record.get("display_idea")
        target_code_region = direction.get("target_code_region") or record.get("fit_point")
        display_title = direction.get("terminal_display_title") or record.get("display_idea") or direction.get("idea_title") or round_id
        idea_text = (
            record.get("idea")
            or record.get("idea_name")
            or record.get("idea_summary")
            or direction.get("hypothesis")
            or record.get("hypothesis")
        )
        rounds.append(
            {
                "id": round_id,
                "round_id": record.get("round_id"),
                "status": record.get("status"),
                "display_idea": display_title,
                "idea": idea_text,
                "idea_summary": record.get("idea_summary") or record.get("mechanism_summary"),
                "mechanism_summary": record.get("mechanism_summary"),
                "hypothesis": record.get("hypothesis") or direction.get("hypothesis") or contract.get("hypothesis"),
                "round_role": direction.get("round_role"),
                "planner": direction.get("planner"),
                "research_direction": direction,
                "opportunity": {
                    "opportunity_id": opportunity.get("opportunity_id") or opportunity_id,
                    "research_question": opportunity.get("research_question") or direction.get("what_this_round_will_teach"),
                    "mechanism_id": opportunity.get("mechanism_id") or target_mechanism,
                    "primary_fit_point": opportunity.get("primary_fit_point") or opportunity.get("fit_point") or target_code_region,
                    "linked_evidence": list(opportunity.get("linked_evidence") or proposal.get("linked_evidence") or []),
                    "source_influence": dict(opportunity.get("source_influence") or {}),
                    "selection_reasons": list(opportunity.get("selection_reasons") or ([direction.get("why_this_role_now")] if direction.get("why_this_role_now") else [])),
                    "risks": list(opportunity.get("risks") or ([direction.get("failure_to_avoid")] if direction.get("failure_to_avoid") else [])),
                },
                "context_reasoning": dict(proposal.get("context_reasoning") or {}),
                "autonomy_reasoning": proposal.get("autonomy_reasoning"),
                "why_not_merely_repeat_previous_work": proposal.get("why_not_merely_repeat_previous_work"),
                "module_family_memory_reference": dict(proposal.get("module_family_memory_reference") or {}),
                "architecture_module_spec": module_spec,
                "module_manifest": module_manifest,
                "module_validity": module_validity,
                "module_experiment_decision": module_decision,
                "module_evolution_decision": evolution_decision,
                "target_code_region": target_code_region,
                "changed_files": _contract_changed_files(record, build_outcome, contract),
                "variant_path": record.get("variant_path") or build_outcome.get("candidate_checkout"),
                "metrics": _compact_metrics(metrics),
                "seed_eval": _seed_eval_metric_stats(seed_eval, objective_metric=objective_metric, root=root) or seed_eval,
                "gate_event": record.get("gate_event") if isinstance(record.get("gate_event"), dict) else {},
                "gate": record.get("gate") if isinstance(record.get("gate"), dict) else {},
                "decision": record.get("decision") or record.get("gate_decision"),
                "failure_type": record.get("failure_type"),
                "failure_reason": record.get("failure_reason") or record.get("error"),
                "build_contract": {
                    "semantic_goal": contract.get("semantic_goal"),
                    "hypothesis": contract.get("hypothesis"),
                    "allowed_edit_files": list(contract.get("allowed_edit_files") or []),
                    "research_direction": direction,
                },
                "build_outcome": {
                    "status": build_outcome.get("status"),
                    "candidate_snapshot_id": build_outcome.get("candidate_snapshot_id"),
                    "candidate_checkout": build_outcome.get("candidate_checkout"),
                    "patch_path": build_outcome.get("patch_path"),
                    "summary": build_outcome.get("summary"),
                    "repair_attempts": build_outcome.get("repair_attempts"),
                },
                "experiment": {
                    "status": experiment.get("status"),
                    "success": experiment.get("success"),
                    "run_id": experiment.get("run_id") or candidate.get("run_id"),
                },
                "artifact_paths": artifact_paths,
                "artifacts": artifacts,
            }
        )
    return rounds


def _final_state(
    knowledge_dir: Path,
    baseline: Dict[str, Any],
    research_rounds: List[Dict[str, Any]],
    diagnosis_current_best: Dict[str, Any] | None = None,
    runtime_state: Dict[str, Any] | None = None,
    *,
    root: Path,
    objective_metric: str = "",
) -> Dict[str, Any]:
    runtime = runtime_state if isinstance(runtime_state, dict) else {}
    runtime_current = runtime.get("current_best") if isinstance(runtime.get("current_best"), dict) else {}
    runtime_baseline = runtime.get("baseline") if isinstance(runtime.get("baseline"), dict) else {}
    candidates = _read_jsonl(knowledge_dir / "candidates.jsonl", limit=500)
    best = {}
    if runtime_current and (
        runtime_current.get("candidate_id")
        or runtime_current.get("display_name")
        or runtime_current.get("model_name")
        or runtime_current.get("metrics")
    ):
        best = runtime_current
    elif runtime_baseline and (
        runtime_baseline.get("candidate_id")
        or runtime_baseline.get("display_name")
        or runtime_baseline.get("model_name")
        or runtime_baseline.get("metrics")
    ):
        best = runtime_baseline
    else:
        for candidate in candidates:
            if candidate.get("is_current_best") or candidate.get("promoted_to_current_best"):
                best = candidate
    if not best:
        best = diagnosis_current_best if isinstance(diagnosis_current_best, dict) and diagnosis_current_best else baseline
    final_name = best.get("display_name") or best.get("model_name") or best.get("candidate_id")
    baseline_id = (
        runtime_baseline.get("candidate_id")
        or baseline.get("candidate_id")
        or runtime_baseline.get("model_name")
        or baseline.get("model_name")
        or runtime_baseline.get("display_name")
        or baseline.get("display_name")
    )
    final_id = best.get("candidate_id") or best.get("model_name") or best.get("display_name")
    return {
        "final_best": {
            "candidate_id": final_id,
            "display_name": final_name,
            "metrics": _compact_metrics(best.get("metrics") or best.get("best_metrics")),
            "seed_eval": _seed_eval_metric_stats(
                dict(best.get("seed_eval") or {}),
                objective_metric=objective_metric,
                root=root,
            ) or dict(best.get("seed_eval") or {}),
            "variant_path": best.get("variant_path"),
        },
        "changed_from_baseline": bool(final_id and baseline_id and final_id != baseline_id),
        "formal_research_round_count": len(research_rounds),
    }


def build_review_fact_pack(*, task_id: str, base_dir: str) -> Dict[str, Any]:
    root = task_knowledge_dir(base_dir, task_id).parents[1]
    knowledge_dir = task_knowledge_dir(base_dir, task_id)
    runs_dir = task_runs_dir(base_dir, task_id)
    domain_state = load_domain_state(base_dir, task_id)
    task_config = load_task_config(base_dir, task_id)
    compiled = _read_json(knowledge_dir / "compiled_config.json", {}) or {}
    dataset_profile = load_dataset_profile(base_dir, task_id) or {}
    runtime_state = dict(domain_state.get("runtime") or {})
    canonical_diagnosis = dict(runtime_state.get("baseline_diagnosis") or {})
    module_family_memory = _read_json(knowledge_dir / "module_family_memory.json", {}) or {}
    objective_metric = str(task_config.get("objective_metric") or runtime_state.get("objective_metric") or "")
    baseline_reference = load_baseline_reference(str(root), task_id)
    initial_baseline = dict(runtime_state.get("baseline") or {})
    diagnosis_current_best = dict(runtime_state.get("current_best") or {})
    baseline_metrics = _compact_metrics(initial_baseline.get("metrics"))
    baseline_seed_eval = {
        "status": "completed" if baseline_reference.get("metric_stats") else "",
        "node_id": baseline_reference.get("node_id"),
        "result_path": baseline_reference.get("result_path"),
        "valid_metric_seeds": ((baseline_reference.get("metric_stats") or {}).get(objective_metric) or {}).get("seed_count"),
        "metric_stats": baseline_reference.get("metric_stats") or {},
        "reference_path": baseline_reference.get("path"),
        "source_clean": baseline_reference.get("source_clean"),
        "generated_before_first_variant": baseline_reference.get("generated_before_first_variant"),
    } if baseline_reference else {}
    reference_metrics = _compact_metrics(
        diagnosis_current_best.get("metrics")
        or baseline_metrics
    )
    reference_seed_eval = dict(diagnosis_current_best.get("seed_eval") or {})
    if not reference_seed_eval and diagnosis_current_best.get("candidate_kind") == "baseline":
        reference_seed_eval = baseline_seed_eval
    canonical_rounds = list(dict(domain_state.get("rounds") or {}).values())
    canonical_contracts = dict(domain_state.get("contracts") or {})
    ablation_root = knowledge_dir / "rounds"
    ablations = [
        _ablation_fact(
            ablation_root / str(record.get("research_id")),
            root=root,
            objective_metric=objective_metric,
            canonical_record=record,
            canonical_contract=dict(canonical_contracts.get(str(record.get("research_id"))) or {}),
        )
        for record in canonical_rounds
        if str(record.get("research_id") or "").startswith("Ablation")
    ]
    research = _research_rounds(
        knowledge_dir,
        root=root,
        objective_metric=objective_metric,
        canonical_records=canonical_rounds,
        canonical_contracts=canonical_contracts,
    )
    metric_direction = str(task_config.get("metric_direction") or "lower_is_better")
    for item in ablations:
        if isinstance(item, dict):
            item["metric_comparison"] = _recorded_metric_comparison(
                item,
                fallback_reference_metrics=reference_metrics,
                fallback_reference_seed_eval=reference_seed_eval,
                objective_metric=objective_metric,
                metric_direction=metric_direction,
            )
    for item in research:
        if isinstance(item, dict):
            item["metric_comparison"] = _recorded_metric_comparison(
                item,
                fallback_reference_metrics=reference_metrics,
                fallback_reference_seed_eval=reference_seed_eval,
                objective_metric=objective_metric,
                metric_direction=metric_direction,
            )
    mechanism_understanding = dict(canonical_diagnosis.get("mechanism_understanding") or {})
    selected_mechanisms = list(
        (mechanism_understanding.get("target_plan") or {}).get("ablation_questions")
        or ((canonical_diagnosis.get("ablation_plan") or {}).get("targets") or [])
        or []
    )
    artifact_paths: List[Path] = [
        task_knowledge_dir(base_dir, task_id) / "domain_state.json",
        knowledge_dir / "compiled_config.json",
        dataset_profile_path(base_dir, task_id),
        knowledge_dir / "baseline_diagnosis.json",
        knowledge_dir / "module_family_memory.json",
        knowledge_dir / "stage_timing_summary.json",
    ]
    research_dir = knowledge_dir / "rounds"
    if research_dir.is_dir():
        artifact_paths.extend(research_dir.glob("Research*/round.json"))
        artifact_paths.extend(research_dir.glob("Research*/build_contract.json"))
        artifact_paths.extend(research_dir.glob("Research*/research_direction.json"))
        artifact_paths.extend(research_dir.glob("Ablation*/round.json"))
        artifact_paths.extend(research_dir.glob("Ablation*/build_contract.json"))
        artifact_paths.extend(research_dir.glob("Ablation*/ablation_record.json"))
    artifact_paths.append(ledger_path(base_dir, task_id))
    api_log_dir = runs_dir / "logs" / "api"
    if api_log_dir.is_dir():
        artifact_paths.extend(api_log_dir.glob("*.parsed.json"))

    final_state = _final_state(
        knowledge_dir,
        initial_baseline,
        research,
        diagnosis_current_best,
        runtime_state,
        root=root,
        objective_metric=objective_metric,
    )
    final_state["architecture_search_status"] = (
        "not_started" if int(task_config.get("max_rounds") or 0) == 0 else "completed_or_attempted"
    )
    fact_pack = {
        "schema_version": "review_fact_pack_v1",
        "task": {
            "task_id": task_id,
            "language": task_config.get("language") or "zh",
            "dataset": task_config.get("dataset_path") or ((compiled.get("data_config") or {}).get("dataset_path")),
            "data_name_list": ((compiled.get("data_config") or {}).get("data_name_list") or []),
            "target": task_config.get("target") or task_config.get("target_columns") or ((compiled.get("data_config") or {}).get("target_columns")),
            "horizon": task_config.get("horizon") or ((compiled.get("evaluation_config") or {}).get("strategy_args") or {}).get("horizon"),
            "seq_len": task_config.get("seq_len") or ((compiled.get("model_config") or {}).get("recommend_model_hyper_params") or {}).get("seq_len"),
            "objective_metric": task_config.get("objective_metric") or runtime_state.get("objective_metric"),
            "budget": task_config.get("budget"),
            "build_mode": bool(task_config.get("build_mode")),
        },
        "baseline": {
            "candidate_id": initial_baseline.get("candidate_id"),
            "model": initial_baseline.get("display_name") or initial_baseline.get("model_name"),
            "metrics": baseline_metrics,
            "seed_eval": baseline_seed_eval,
            "reference": baseline_reference,
            "config": initial_baseline.get("model_config") or {},
        },
        "baseline_search": _baseline_search_fact(
            dict(runtime_state.get("baseline_search_progress") or {}),
            objective_metric=objective_metric,
            metric_direction=metric_direction,
        ),
        "dataset_diagnosis": _compact_dataset_diagnosis(dataset_profile),
        "diagnosis_current_best": {
            "candidate_id": diagnosis_current_best.get("candidate_id"),
            "model": diagnosis_current_best.get("display_name") or diagnosis_current_best.get("model_name"),
            "metrics": _compact_metrics(diagnosis_current_best.get("metrics")),
            "variant_path": (diagnosis_current_best.get("model_config") or {}).get("variant_path") or diagnosis_current_best.get("variant_path"),
            "source": diagnosis_current_best.get("source"),
            "parent_candidate_id": diagnosis_current_best.get("parent_candidate_id"),
        },
        "comparison_reference": {
            "kind": "current_best",
            "metrics": reference_metrics,
            "seed_eval": reference_seed_eval,
            "candidate_id": diagnosis_current_best.get("candidate_id"),
            "model": diagnosis_current_best.get("display_name") or diagnosis_current_best.get("model_name"),
        },
        "baseline_diagnosis": {
            "status": str(canonical_diagnosis.get("status") or ""),
            "planned": canonical_diagnosis.get("planned_ablation_count", len(ablations)),
            "executed": canonical_diagnosis.get("executed_ablation_count", len(ablations)),
            "usable": canonical_diagnosis.get("usable_ablation_count", sum(1 for item in ablations if str(item.get("usable_evidence_status") or "") == "usable_evidence")),
            "failed": canonical_diagnosis.get("failed_ablation_count", sum(1 for item in ablations if str(item.get("status") or "").startswith("failed"))),
            "target_discovery": canonical_diagnosis.get("target_discovery") or {},
            "ablation_execution": canonical_diagnosis.get("ablation_execution") or {},
            "selected_mechanisms": selected_mechanisms,
            "ablations": ablations,
        },
        "research_rounds": research,
        "module_family_memory": module_family_memory,
        "research_state": _compact_research_state(dict(runtime_state.get("research") or {})),
        "token_usage": build_token_usage_summary(task_id, base_dir),
        "stage_timing": build_stage_timing_summary(task_id, base_dir),
        "final_state": final_state,
        "artifacts": _artifact_index(artifact_paths, root=root),
    }
    return fact_pack


def write_review_fact_pack(*, task_id: str, base_dir: str) -> tuple[Dict[str, Any], Path]:
    write_token_usage_summary(task_id, base_dir)
    write_stage_timing_summary(task_id, base_dir)
    fact_pack = build_review_fact_pack(task_id=task_id, base_dir=base_dir)
    path = task_knowledge_dir(base_dir, task_id) / "review_report_fact_pack.json"
    path.write_text(json.dumps(fact_pack, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return fact_pack, path
