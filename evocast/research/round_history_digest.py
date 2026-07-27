from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from evocast.domain.knowledge_paths import task_knowledge_dir
from evocast.harness.api_client import create_task_client
from evocast.state.domain_store import list_round_records


DIGEST_SCHEMA_VERSION = "round_history_digest_v3"

DIGEST_SCHEMA_HINT = json.dumps(
    {
        "required": ["items"],
        "items": [
            {
                "round_id": "Research001",
                "one_line": "One factual sentence stating changed location, mechanism/information flow, operation type, and final metric/status.",
            }
        ],
    },
    ensure_ascii=False,
)

COMPRESSOR_SYSTEM_PROMPT = (
    "Compress EvoCast Research rounds into factual one-line history items. "
    "Return exactly one JSON object. No markdown. No commentary. "
    "Do not evaluate, recommend, or explain why a round failed. "
    "Do not write a chronological narrative. "
    "Never infer current_best from a lower single-seed metric or completed execution status. "
    "State promoted_to_current_best explicitly when present; if it is false, say the round did not promote current_best. "
    "Do not preserve long catchy module names unless needed for precision. "
    "Each sentence must state where the code changed, what information flow or mechanism changed, "
    "whether it added/replaced/modified a path, the final metric/status, and the promotion outcome."
)


def _read_json(path: Path, default: Any) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _round_sort_key(value: Path | Dict[str, Any]) -> tuple[int, str]:
    name = value.name if isinstance(value, Path) else str(value.get("research_id") or value.get("round_id") or "")
    digits = "".join(ch for ch in name if ch.isdigit())
    return (int(digits or 0), name)


def _counts_toward_research_budget(record: Dict[str, Any]) -> bool:
    explicit = record.get("counts_toward_research_budget")
    if explicit is not None:
        return bool(explicit)
    return str(record.get("round_scope") or "research").strip().lower() == "research"


def _metric_value(metrics: Dict[str, Any], key: str) -> Any:
    value = metrics.get(key)
    if isinstance(value, float):
        return round(value, 6)
    return value


def _gate_event(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("gate_event")
    return value if isinstance(value, dict) else {}


def _seed_eval(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("seed_eval")
    return value if isinstance(value, dict) else {}


def _promoted_to_current_best(record: Dict[str, Any]) -> bool:
    seed_eval = _seed_eval(record)
    promotion = record.get("promotion_decision") if isinstance(record.get("promotion_decision"), dict) else {}
    seed_promotion = seed_eval.get("promotion_decision") if isinstance(seed_eval.get("promotion_decision"), dict) else {}
    decisions = [
        promotion.get("decision"),
        seed_promotion.get("decision"),
        record.get("promotion_decision") if not isinstance(record.get("promotion_decision"), dict) else None,
    ]
    return bool(
        record.get("promoted_to_current_best") is True
        or record.get("is_current_best") is True
        or seed_eval.get("promoted_to_current_best") is True
        or any(str(value or "").strip().lower() in {"promote", "promoted"} for value in decisions)
    )


def _effective_status(record: Dict[str, Any], gate_decision: Any, promoted: bool) -> str:
    decision = str(gate_decision or "").strip().lower()
    raw_status = str(record.get("status") or "").strip()
    if decision == "marginal_no_seed_eval":
        return "marginal_no_seed_eval"
    if decision in {"reject", "rejected"}:
        return "rejected"
    if decision == "accept" and not promoted:
        return "needs_seed_eval"
    return raw_status


def _direction_for_round(root: Path, round_id: str) -> Dict[str, Any]:
    direct = root / "rounds" / round_id / "research_direction.json"
    payload = _read_json(direct, {})
    if isinstance(payload, dict) and payload:
        return payload
    candidates = sorted((root / "research_directions").glob(f"{round_id}_*.json"))
    if not candidates:
        return {}
    payload = _read_json(candidates[-1], {})
    return payload if isinstance(payload, dict) else {}


def _metadata_for_round(root: Path, round_id: str) -> Dict[str, Any]:
    round_dir = root / "rounds" / round_id
    candidates = sorted(round_dir.glob("attempt_*/metadata/round_metadata.json"))
    if not candidates:
        candidates = sorted(round_dir.glob("*/metadata/round_metadata.json"))
    if not candidates:
        return {}
    payload = _read_json(candidates[-1], {})
    return payload if isinstance(payload, dict) else {}


def research_round_facts(base_dir: str, task_id: str, *, limit: int = 20) -> List[Dict[str, Any]]:
    root = task_knowledge_dir(base_dir, task_id)
    rows: List[Dict[str, Any]] = []
    records = [record for record in list_round_records(base_dir, task_id) if _counts_toward_research_budget(record)]
    for record in sorted(records, key=_round_sort_key):
        round_id = str(record.get("research_id") or record.get("round_id") or "")
        if not round_id:
            continue
        direction = _direction_for_round(root, round_id)
        metadata = _metadata_for_round(root, round_id)
        evaluation = record.get("evaluation") if isinstance(record.get("evaluation"), dict) else {}
        metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else dict(evaluation.get("metrics") or {})
        gate_event = _gate_event(record)
        gate = gate_event.get("gate") if isinstance(gate_event.get("gate"), dict) else {}
        seed_eval = _seed_eval(record)
        gate_decision = record.get("gate_decision") or gate_event.get("decision") or gate.get("decision")
        promoted = _promoted_to_current_best(record)
        rows.append(
            {
                "round_id": round_id,
                "status": _effective_status(record, gate_decision, promoted),
                "raw_status": record.get("status"),
                "failure_kind": record.get("failure_kind"),
                "error_type": record.get("error_type"),
                "gate_decision": gate_decision,
                "promoted_to_current_best": promoted,
                "seed_eval_promoted_to_current_best": seed_eval.get("promoted_to_current_best"),
                "relative_improvement": gate_event.get("relative_improvement")
                if gate_event.get("relative_improvement") is not None
                else gate.get("reference_relative_improvement"),
                "threshold_met": gate_event.get("threshold_met")
                if gate_event.get("threshold_met") is not None
                else gate.get("beats_reference_threshold"),
                "reference_kind": gate_event.get("reference_kind") or gate.get("reference_kind"),
                "reference_value": gate_event.get("reference_value") or gate.get("reference_value"),
                "current_best_value": gate_event.get("current_best_value") or gate.get("current_best_value"),
                "mse_norm": _metric_value(metrics, "mse_norm"),
                "mae_norm": _metric_value(metrics, "mae_norm"),
                "display_idea": record.get("display_idea") or metadata.get("display_idea") or direction.get("terminal_display_title"),
                "idea_summary": record.get("idea_summary") or metadata.get("idea_summary"),
                "target_mechanism": direction.get("target_mechanism"),
                "target_code_region": direction.get("target_code_region"),
                "changed_mechanism": record.get("changed_mechanism") or metadata.get("changed_mechanism"),
                "round_role": direction.get("round_role"),
            }
        )
    return rows[-max(1, int(limit or 1)) :]


def round_status_table(base_dir: str, task_id: str, *, limit: int = 20) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in research_round_facts(base_dir, task_id, limit=limit):
        rows.append(
            {
                key: value
                for key, value in {
                    "round_id": item.get("round_id"),
                    "status": item.get("status"),
                    "failure_kind": item.get("failure_kind"),
                    "error_type": item.get("error_type"),
                    "gate_decision": item.get("gate_decision"),
                    "promoted_to_current_best": item.get("promoted_to_current_best"),
                    "relative_improvement": item.get("relative_improvement"),
                    "threshold_met": item.get("threshold_met"),
                    "mse_norm": item.get("mse_norm"),
                    "mae_norm": item.get("mae_norm"),
                }.items()
                if value not in (None, "", [], {})
            }
        )
    return rows


def _digest_path(base_dir: str, task_id: str) -> Path:
    return task_knowledge_dir(base_dir, task_id) / "round_history_digest.json"


def _digest_is_current(payload: Dict[str, Any], facts: List[Dict[str, Any]]) -> bool:
    if payload.get("schema_version") != DIGEST_SCHEMA_VERSION:
        return False
    item_ids = [str(item.get("round_id") or "") for item in list(payload.get("items") or []) if isinstance(item, dict)]
    fact_ids = [str(item.get("round_id") or "") for item in facts]
    return item_ids == fact_ids


def _mechanical_one_line(fact: Dict[str, Any]) -> str:
    round_id = str(fact.get("round_id") or "Research").strip()
    location = str(fact.get("target_code_region") or "").strip()
    mechanism = str(
        fact.get("changed_mechanism")
        or fact.get("target_mechanism")
        or fact.get("idea_summary")
        or fact.get("display_idea")
        or "未记录具体机制"
    ).strip()
    metric_parts: List[str] = []
    if fact.get("mse_norm") is not None:
        metric_parts.append(f"mse_norm={fact.get('mse_norm')}")
    if fact.get("mae_norm") is not None:
        metric_parts.append(f"mae_norm={fact.get('mae_norm')}")
    gate_decision = str(fact.get("gate_decision") or "").strip()
    promoted = fact.get("promoted_to_current_best") is True
    if gate_decision:
        metric_parts.append(f"gate_decision={gate_decision}")
    if fact.get("relative_improvement") is not None:
        try:
            metric_parts.append(f"relative_improvement={float(fact.get('relative_improvement')):.4%}")
        except Exception:
            metric_parts.append(f"relative_improvement={fact.get('relative_improvement')}")
    if fact.get("threshold_met") is not None:
        metric_parts.append(f"threshold_met={bool(fact.get('threshold_met'))}")
    metric_parts.append(f"promoted_to_current_best={str(promoted).lower()}")
    status_parts = [
        str(fact.get("status") or "").strip(),
        str(fact.get("failure_kind") or "").strip(),
        str(fact.get("error_type") or "").strip(),
    ]
    status = "/".join(part for part in status_parts if part) or "status=unknown"
    promotion_note = "已晋升为 current_best" if promoted else "未晋升，不是 current_best"
    prefix = f"{round_id}: "
    where = f"在 {location} " if location else ""
    metrics = f"，{', '.join(metric_parts)}" if metric_parts else ""
    return f"{prefix}{where}执行 {mechanism}{metrics}，状态 {status}，{promotion_note}。"


def _mechanical_digest_items(facts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {"round_id": str(fact.get("round_id") or ""), "one_line": _mechanical_one_line(fact)}
        for fact in facts
        if str(fact.get("round_id") or "").strip()
    ]


def _normalize_digest_items(raw_items: Any, facts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    fact_ids = [str(item.get("round_id") or "") for item in facts]
    fact_by_id = {str(item.get("round_id") or ""): item for item in facts}
    by_id: Dict[str, str] = {}
    positional: List[str] = []
    if isinstance(raw_items, dict):
        iterable = [{"round_id": key, "one_line": value} for key, value in raw_items.items()]
    else:
        iterable = list(raw_items or [])
    for item in iterable:
        if isinstance(item, str):
            text = item.strip()
            if text:
                positional.append(text)
            continue
        if not isinstance(item, dict):
            continue
        round_id = str(item.get("round_id") or "").strip()
        one_line = str(
            item.get("one_line")
            or item.get("compressed_fact")
            or item.get("fact")
            or item.get("sentence")
            or item.get("summary")
            or item.get("text")
            or item.get("description")
            or item.get("content")
            or ""
        ).strip()
        if round_id and one_line:
            by_id[round_id] = one_line
        elif one_line:
            positional.append(one_line)

    items: List[Dict[str, str]] = []
    positional_index = 0
    for round_id in fact_ids:
        one_line = by_id.get(round_id, "")
        if not one_line and positional_index < len(positional):
            candidate = positional[positional_index]
            positional_index += 1
            if candidate.startswith(round_id):
                one_line = candidate
            else:
                one_line = f"{round_id}: {candidate}"
        if not one_line:
            one_line = _mechanical_one_line(fact_by_id.get(round_id, {"round_id": round_id}))
        items.append({"round_id": round_id, "one_line": one_line})
    return items


def load_round_history_digest(base_dir: str, task_id: str) -> Dict[str, Any]:
    payload = _read_json(_digest_path(base_dir, task_id), {})
    return payload if isinstance(payload, dict) else {}


def ensure_round_history_digest(
    *,
    base_dir: str,
    task_id: str,
    api_config: str,
    history_limit: int = 20,
) -> Dict[str, Any]:
    facts = research_round_facts(base_dir, task_id, limit=history_limit)
    path = _digest_path(base_dir, task_id)
    existing = load_round_history_digest(base_dir, task_id)
    if _digest_is_current(existing, facts):
        return existing
    if not facts:
        payload = {
            "schema_version": DIGEST_SCHEMA_VERSION,
            "task_id": task_id,
            "items": [],
            "source_round_count": 0,
            "updated_at": datetime.now().isoformat(),
        }
        _write_json(path, payload)
        return payload

    messages = [
        {"role": "system", "content": COMPRESSOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Compress each Research round into exactly one factual sentence.",
                    "sentence_rules": [
                        "No evaluation or recommendation.",
                        "No explanation of why it failed.",
                        "No chronological narrative.",
                        "State changed location, mechanism or information flow, operation type, final metric/status, and promotion outcome.",
                        "If promoted_to_current_best is false, state that the round did not promote current_best even if its single-seed metric is lower.",
                        "Prefer concise Chinese when the source task uses Chinese; otherwise concise English is acceptable.",
                    ],
                    "round_facts": facts,
                },
                ensure_ascii=False,
            ),
        },
    ]
    source = "llm"
    fallback_reason = ""
    try:
        client = create_task_client(
            base_dir=base_dir,
            task_id=f"{task_id}_round_history_digest",
            explicit_config=api_config,
        )
        if not client.api_available:
            raise RuntimeError("round_history_digest_requires_real_api_key")
        result = client.call_json(
            "planner",
            0,
            messages,
            schema_hint=DIGEST_SCHEMA_HINT,
            execution_label="round_history_digest",
            require_all_top_level_keys=True,
            stream_override=False,
        )
        raw_items = []
        if isinstance(result, dict):
            raw_items = result.get("items") or result.get("history") or result
        items = _normalize_digest_items(raw_items, facts)
    except Exception as exc:
        source = "mechanical_fallback"
        fallback_reason = f"{type(exc).__name__}: {exc}"
        items = _mechanical_digest_items(facts)

    payload = {
        "schema_version": DIGEST_SCHEMA_VERSION,
        "task_id": task_id,
        "items": items,
        "source_round_count": len(facts),
        "source": source,
        "updated_at": datetime.now().isoformat(),
    }
    if fallback_reason:
        payload["fallback_reason"] = fallback_reason
    _write_json(path, payload)
    return payload
