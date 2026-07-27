from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evocast.domain.i18n import locale_matches_text
from evocast.harness.api_client import ProviderClient
from evocast.domain.knowledge_paths import task_knowledge_dir


REVIEW_NARRATIVE_SCHEMA_VERSION = "review_narrative_v1"


def _shorten(value: Any, *, max_string: int = 1800, max_list: int = 30, depth: int = 0) -> Any:
    if depth > 8:
        return "<truncated:max_depth>"
    if isinstance(value, str):
        if len(value) <= max_string:
            return value
        return value[:max_string] + f"\n<truncated {len(value) - max_string} chars>"
    if isinstance(value, list):
        items = [_shorten(item, max_string=max_string, max_list=max_list, depth=depth + 1) for item in value[:max_list]]
        if len(value) > max_list:
            items.append({"truncated_items": len(value) - max_list})
        return items
    if isinstance(value, dict):
        return {
            str(key): _shorten(item, max_string=max_string, max_list=max_list, depth=depth + 1)
            for key, item in value.items()
        }
    return value


def compact_fact_pack_for_narrative(fact_pack: Dict[str, Any]) -> Dict[str, Any]:
    compact = _shorten(fact_pack, max_string=1600, max_list=40)
    if not isinstance(compact, dict):
        return {}
    return compact


def _default_narrative() -> Dict[str, Any]:
    return {
        "schema_version": REVIEW_NARRATIVE_SCHEMA_VERSION,
        "title": "",
        "executive_summary": [],
        "timeline": [],
        "baseline_summary": "",
        "baseline_diagnosis_summary": "",
        "ablation_narratives": [],
        "research_round_narratives": [],
        "final_architecture_summary": "",
        "review_checklist": [],
        "risk_notes": [],
    }


def _normalize_text_block_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            normalized.append({"text": item, "evidence_refs": []})
        elif isinstance(item, dict):
            normalized.append(
                {
                    "text": str(item.get("text") or item.get("summary") or ""),
                    "evidence_refs": list(item.get("evidence_refs") or []),
                }
            )
    return [item for item in normalized if item.get("text")]


def normalize_narrative(payload: Dict[str, Any], *, language: str = "zh") -> Dict[str, Any]:
    narrative = _default_narrative()
    if isinstance(payload, dict):
        narrative.update(payload)
    narrative["schema_version"] = REVIEW_NARRATIVE_SCHEMA_VERSION
    english = str(language or "zh").strip().lower() in {"en", "english"}
    default_title = "EvoCast Review Report" if english else "EvoCast 实验复盘"
    narrative["title"] = str(narrative.get("title") or default_title)
    narrative["executive_summary"] = _normalize_text_block_list(narrative.get("executive_summary"))
    narrative["timeline"] = [item for item in list(narrative.get("timeline") or []) if isinstance(item, dict)]
    narrative["ablation_narratives"] = [
        item for item in list(narrative.get("ablation_narratives") or []) if isinstance(item, dict)
    ]
    narrative["research_round_narratives"] = [
        item for item in list(narrative.get("research_round_narratives") or []) if isinstance(item, dict)
    ]
    narrative["review_checklist"] = [
        str(item) if not isinstance(item, dict) else str(item.get("text") or item.get("item") or "")
        for item in list(narrative.get("review_checklist") or [])
    ]
    narrative["review_checklist"] = [item for item in narrative["review_checklist"] if item]
    narrative["risk_notes"] = [
        str(item) if not isinstance(item, dict) else str(item.get("text") or item.get("risk") or "")
        for item in list(narrative.get("risk_notes") or [])
    ]
    narrative["risk_notes"] = [item for item in narrative["risk_notes"] if item]
    for key in ("baseline_summary", "baseline_diagnosis_summary", "final_architecture_summary"):
        narrative[key] = str(narrative.get(key) or "")
    return narrative


ABLATION_REQUIRED_FIELDS: List[str] = []
RESEARCH_REQUIRED_FIELDS: List[str] = []


def _required_ids(fact_pack: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    diagnosis = fact_pack.get("baseline_diagnosis") if isinstance(fact_pack.get("baseline_diagnosis"), dict) else {}
    ablation_ids = [
        str(item.get("id") or "").strip()
        for item in list((diagnosis or {}).get("ablations") or [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    ]
    research_ids = [
        str(item.get("id") or "").strip()
        for item in list(fact_pack.get("research_rounds") or [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    ]
    return ablation_ids, research_ids


def _narrative_by_id(narrative: Dict[str, Any], key: str) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("id")): dict(item)
        for item in list(narrative.get(key) or [])
        if isinstance(item, dict) and item.get("id")
    }


def _field_missing_or_wrong_language(value: Any, *, language: str) -> bool:
    if isinstance(value, list):
        values = [str(item or "").strip() for item in value if str(item or "").strip()]
        return not values or not all(_narrative_locale_matches(item, language=language) for item in values)
    text = str(value or "").strip()
    if not text:
        return True
    return not _narrative_locale_matches(text, language=language)


def _narrative_locale_matches(value: Any, *, language: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    # Code snippets, paths, hashes, and symbols are allowed to remain stable
    # inside Chinese prose; they should not make a human-readable sentence look
    # like the wrong language.
    scrubbed = re.sub(r"`[^`]*`", " ", text)
    scrubbed = re.sub(r"\b[\w./\\-]+\.(?:py|json|jsonl|txt|md|csv|html|patch)\b", " ", scrubbed)
    scrubbed = re.sub(r"\b[0-9a-f]{12,40}\b", " ", scrubbed, flags=re.IGNORECASE)
    scrubbed = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", " ", scrubbed)
    scrubbed = re.sub(r"\b[A-Za-z_]+(?:_[A-Za-z0-9_]+){1,}\b", " ", scrubbed)
    scrubbed = re.sub(r"\s+", " ", scrubbed).strip()
    return locale_matches_text(scrubbed or text, language)


def _missing_fields(item: Dict[str, Any] | None, required_fields: List[str], *, language: str) -> List[str]:
    if not isinstance(item, dict):
        return list(required_fields)
    return [field for field in required_fields if _field_missing_or_wrong_language(item.get(field), language=language)]


def _failed_ablation_requires_no_mechanism_conclusion(record: Dict[str, Any], narrative_item: Dict[str, Any] | None) -> bool:
    status = str(record.get("status") or record.get("raw_status") or "").strip()
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    if not status.startswith("failed") and metrics:
        return False
    if not status.startswith("failed"):
        return False
    text = str((narrative_item or {}).get("result_interpretation") or "")
    required_markers = [
        ("实现", "失败", "机制结论"),
        ("implementation", "failed", "mechanism conclusion"),
        ("implementation", "failure", "mechanism conclusion"),
    ]
    lowered = text.lower()
    return not any(all(marker.lower() in lowered for marker in markers) for markers in required_markers)


def _narrative_gaps(narrative: Dict[str, Any], fact_pack: Dict[str, Any], *, language: str) -> Dict[str, Dict[str, List[str]]]:
    ablation_ids, research_ids = _required_ids(fact_pack)
    ablation_by_id = _narrative_by_id(narrative, "ablation_narratives")
    research_by_id = _narrative_by_id(narrative, "research_round_narratives")
    diagnosis = fact_pack.get("baseline_diagnosis") if isinstance(fact_pack.get("baseline_diagnosis"), dict) else {}
    ablation_records = {
        str(item.get("id") or ""): item
        for item in list((diagnosis or {}).get("ablations") or [])
        if isinstance(item, dict)
    }
    ablations = {}
    for aid in ablation_ids:
        item = ablation_by_id.get(aid)
        missing = _missing_fields(item, ABLATION_REQUIRED_FIELDS, language=language)
        if _failed_ablation_requires_no_mechanism_conclusion(ablation_records.get(aid) or {}, item):
            missing.append("result_interpretation")
        if missing:
            ablations[aid] = sorted(set(missing), key=missing.index)
    research = {
        rid: missing
        for rid in research_ids
        if (missing := _missing_fields(research_by_id.get(rid), RESEARCH_REQUIRED_FIELDS, language=language))
    }
    return {"ablations": ablations, "research_rounds": research}


def _has_gaps(gaps: Dict[str, Dict[str, List[str]]]) -> bool:
    return any(bool(items) for items in gaps.values())


def _first_gap(gaps: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[str]]]:
    for group in ("ablations", "research_rounds"):
        items = gaps.get(group) if isinstance(gaps.get(group), dict) else {}
        for key, fields in items.items():
            return {group: {key: list(fields)}, "research_rounds" if group == "ablations" else "ablations": {}}
    return {"ablations": {}, "research_rounds": {}}


def _subset_facts_for_ids(fact_pack: Dict[str, Any], gaps: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
    task = fact_pack.get("task") if isinstance(fact_pack.get("task"), dict) else {}
    baseline = fact_pack.get("baseline") if isinstance(fact_pack.get("baseline"), dict) else {}
    diagnosis = fact_pack.get("baseline_diagnosis") if isinstance(fact_pack.get("baseline_diagnosis"), dict) else {}
    ablation_ids = set(gaps.get("ablations") or {})
    research_ids = set(gaps.get("research_rounds") or {})
    return {
        "task": task,
        "baseline": baseline,
        "baseline_diagnosis": {
            key: diagnosis.get(key)
            for key in ["status", "planned", "executed", "usable", "failed"]
        } | {
            "ablations": [
                item
                for item in list(diagnosis.get("ablations") or [])
                if isinstance(item, dict) and str(item.get("id") or "") in ablation_ids
            ]
        },
        "research_rounds": [
            item
            for item in list(fact_pack.get("research_rounds") or [])
            if isinstance(item, dict) and str(item.get("id") or "") in research_ids
        ],
    }


def _coerce_supplement_payload(payload: Dict[str, Any], gaps: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("ablation_narratives"), list) or isinstance(payload.get("research_round_narratives"), list):
        return payload
    item_id = str(payload.get("id") or "").strip()
    if not item_id:
        return payload
    if item_id in set(gaps.get("ablations") or {}):
        return {"ablation_narratives": [payload], "research_round_narratives": []}
    if item_id in set(gaps.get("research_rounds") or {}):
        return {"ablation_narratives": [], "research_round_narratives": [payload]}
    return payload


def _merge_narrative_items(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]], required_fields: List[str]) -> List[Dict[str, Any]]:
    by_id = {
        str(item.get("id")): dict(item)
        for item in existing
        if isinstance(item, dict) and item.get("id")
    }
    for item in incoming:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        key = str(item.get("id"))
        merged = dict(by_id.get(key) or {"id": key})
        for field in required_fields:
            if str(item.get(field) or "").strip():
                merged[field] = item.get(field)
        for field, value in item.items():
            if field not in merged and str(value or "").strip():
                merged[field] = value
        by_id[key] = merged
    return list(by_id.values())


def _merge_supplement(narrative: Dict[str, Any], supplement: Dict[str, Any], *, language: str) -> Dict[str, Any]:
    normalized = normalize_narrative(supplement, language=language)
    merged = dict(narrative)
    merged["ablation_narratives"] = _merge_narrative_items(
        list(merged.get("ablation_narratives") or []),
        list(normalized.get("ablation_narratives") or []),
        ABLATION_REQUIRED_FIELDS,
    )
    merged["research_round_narratives"] = _merge_narrative_items(
        list(merged.get("research_round_narratives") or []),
        list(normalized.get("research_round_narratives") or []),
        RESEARCH_REQUIRED_FIELDS,
    )
    return normalize_narrative(merged, language=language)


def _metric_sentence(metrics: Dict[str, Any] | None, objective: str = "") -> str:
    if not isinstance(metrics, dict) or not metrics:
        return "未记录有效指标"
    key = objective if objective and objective in metrics else next(iter(metrics))
    return f"{key}={metrics.get(key)}"


def _changed_files_sentence(files: Any) -> str:
    values = [str(item) for item in list(files or []) if str(item or "").strip()]
    return "、".join(values) if values else "未记录修改文件"


def _localize_round_status(value: Any, *, language: str) -> str:
    text = str(value or "").strip()
    if language in {"en", "english"}:
        return text or "unknown"
    mapping = {
        "rejected": "已拒绝",
        "completed": "已完成",
        "failed": "失败",
        "marginal_no_seed_eval": "边际变化，未晋升",
        "needs_seed_eval": "需要多种子评估",
        "TERMINAL_REJECTED": "终态拒绝",
    }
    return mapping.get(text, text or "未记录状态")


def _fallback_ablation_narrative(record: Dict[str, Any], *, language: str, objective: str) -> Dict[str, Any]:
    rid = str(record.get("id") or "")
    raw_name = str(record.get("display_idea") or record.get("mechanism_name") or record.get("mechanism_id") or rid)
    name = raw_name if language in {"en", "english"} else f"{rid} 的机制消融"
    changed = _changed_files_sentence(record.get("changed_files"))
    metric = _metric_sentence(record.get("metrics") if isinstance(record.get("metrics"), dict) else {}, objective)
    status = _localize_round_status(record.get("status"), language=language)
    build = record.get("build_outcome") if isinstance(record.get("build_outcome"), dict) else {}
    snapshot_id = str(build.get("candidate_snapshot_id") or record.get("variant_path") or "").strip()
    if language in {"en", "english"}:
        return {
            "id": rid,
            "mechanism_display": name,
            "what_changed": f"This ablation tested {name}.",
            "why_it_matters": "It checks whether the targeted baseline mechanism is necessary for the observed metric.",
            "code_change": f"Changed files: {changed}.",
            "result_interpretation": f"The round status is {status}; the recorded metric is {metric}.",
            "review_risk": "This fallback summary was generated from structured artifacts because the LLM narrative was incomplete.",
            "audit_trace": f"Audit artifacts are stored under {rid}; candidate source snapshot: {snapshot_id[:12] if snapshot_id else 'not recorded'}.",
        }
    return {
        "id": rid,
        "mechanism_display": name,
        "what_changed": f"该消融围绕记录中的目标机制展开，用来观察移除或削弱该机制后的指标变化；原始短名为 {raw_name}。",
        "why_it_matters": "它用于判断该基线机制是否真正在当前任务中承担关键作用。",
        "code_change": f"修改文件：{changed}。",
        "result_interpretation": f"该轮状态为{status}；记录到的指标为 {metric}。",
        "review_risk": "这是根据结构化产物生成的保底中文说明；若需要更细粒度解释，应查看对应轮次产物。",
        "audit_trace": f"审计线索位于 {rid} 轮次产物；候选源码快照：{snapshot_id[:12] if snapshot_id else '未记录'}。",
    }


def _fallback_research_narrative(record: Dict[str, Any], *, language: str, objective: str) -> Dict[str, Any]:
    rid = str(record.get("id") or "")
    raw_title = str(record.get("display_idea") or record.get("idea") or rid)
    title = raw_title if language in {"en", "english"} else f"{rid} 的候选机制研究"
    role = str(record.get("round_role") or "")
    planner = str(record.get("planner") or "")
    changed = _changed_files_sentence(record.get("changed_files"))
    metric = _metric_sentence(record.get("metrics") if isinstance(record.get("metrics"), dict) else {}, objective)
    status = _localize_round_status(record.get("status"), language=language)
    opportunity = record.get("opportunity") if isinstance(record.get("opportunity"), dict) else {}
    build = record.get("build_outcome") if isinstance(record.get("build_outcome"), dict) else {}
    snapshot_id = str(build.get("candidate_snapshot_id") or record.get("variant_path") or "").strip()
    mechanism = str(opportunity.get("mechanism_id") or title)
    if language in {"en", "english"}:
        return {
            "id": rid,
            "idea": title,
            "research_question": f"Whether {mechanism} can improve the target metric.",
            "idea_source": f"Planner: {planner or 'not recorded'}; round role: {role or 'not recorded'}.",
            "implementation": f"The coding agent produced a candidate variant for {title}.",
            "code_change": f"Changed files: {changed}.",
            "result_interpretation": f"The round status is {status}; the recorded metric is {metric}.",
            "relation_to_previous": "This fallback summary preserves the recorded round order; inspect artifacts for detailed cross-round reasoning.",
            "review_risk": "This fallback summary was generated from structured artifacts because the LLM narrative was incomplete.",
            "audit_trace": f"Audit artifacts are stored under {rid}; candidate source snapshot: {snapshot_id[:12] if snapshot_id else 'not recorded'}.",
        }
    return {
        "id": rid,
        "idea": title,
        "research_question": f"该轮检验“{mechanism}”是否能改善目标指标。",
        "idea_source": f"Idea 来源记录为规划器 {planner or '未记录'}，轮次角色为 {role or '未记录'}。",
        "implementation": f"coding agent 围绕该轮候选方向生成了一个候选变体；原始短名为 {raw_title}。",
        "code_change": f"修改文件：{changed}。",
        "result_interpretation": f"该轮状态为{status}；记录到的指标为 {metric}。",
        "relation_to_previous": "该保底说明保留了轮次顺序；更细的跨轮推理请查看该轮 research_direction 与 round 产物。",
        "review_risk": "这是根据结构化产物生成的保底中文说明；若需要更细粒度解释，应查看对应轮次产物。",
        "audit_trace": f"审计线索位于 {rid} 轮次产物；候选源码快照：{snapshot_id[:12] if snapshot_id else '未记录'}。",
    }


def _fill_narrative_fallbacks(narrative: Dict[str, Any], fact_pack: Dict[str, Any], *, language: str) -> Dict[str, Any]:
    objective = str((fact_pack.get("task") or {}).get("objective_metric") or "")
    merged = dict(narrative)
    by_ablation = _narrative_by_id(merged, "ablation_narratives")
    diagnosis = fact_pack.get("baseline_diagnosis") if isinstance(fact_pack.get("baseline_diagnosis"), dict) else {}
    for record in list(diagnosis.get("ablations") or []):
        if not isinstance(record, dict):
            continue
        rid = str(record.get("id") or "")
        if not rid:
            continue
        fallback = _fallback_ablation_narrative(record, language=language, objective=objective)
        current = dict(by_ablation.get(rid) or {"id": rid})
        for field in ABLATION_REQUIRED_FIELDS:
            if _field_missing_or_wrong_language(current.get(field), language=language):
                current[field] = fallback.get(field)
        by_ablation[rid] = current
    by_research = _narrative_by_id(merged, "research_round_narratives")
    for record in list(fact_pack.get("research_rounds") or []):
        if not isinstance(record, dict):
            continue
        rid = str(record.get("id") or "")
        if not rid:
            continue
        fallback = _fallback_research_narrative(record, language=language, objective=objective)
        current = dict(by_research.get(rid) or {"id": rid})
        for field in RESEARCH_REQUIRED_FIELDS:
            if _field_missing_or_wrong_language(current.get(field), language=language):
                current[field] = fallback.get(field)
        by_research[rid] = current
    merged["ablation_narratives"] = list(by_ablation.values())
    merged["research_round_narratives"] = list(by_research.values())
    return normalize_narrative(merged, language=language)


def _request_missing_narrative(
    *,
    client: Any,
    fact_pack: Dict[str, Any],
    narrative: Dict[str, Any],
    gaps: Dict[str, Dict[str, List[str]]],
    language: str,
    task_id: str,
    attempt: int,
) -> Dict[str, Any]:
    english = language in {"en", "english"}
    language_instruction = (
        "Write all human-facing narrative text in English."
        if english
        else "Write all human-facing narrative text in Chinese."
    )
    schema_hint = {
        "ablation_narratives": [
            {
                "id": "string",
                "mechanism_display": "string",
                "what_changed": "string",
                "why_it_matters": "string",
                "code_change": "string",
                "result_interpretation": "string",
                "review_risk": "string",
                "audit_trace": "string",
            }
        ],
        "research_round_narratives": [
            {
                "id": "string",
                "idea": "string",
                "research_question": "string",
                "idea_source": "string",
                "implementation": "string",
                "code_change": "string",
                "result_interpretation": "string",
                "relation_to_previous": "string",
                "review_risk": "string",
                "audit_trace": "string",
            }
        ],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are repairing missing user-facing narrative fields for an offline EvoCast HTML report. "
                "Use only the provided facts and existing narrative. Return exactly one JSON object with only "
                "ablation_narratives and research_round_narratives arrays. Fill every requested missing or wrong-language field. "
                f"{language_instruction} Do not output HTML or Markdown. "
                "Do not mention internal JSON parse errors, field names, stack traces, raw tool failures, schema names, "
                "fact_pack, evidence_refs, or source paths. If an ablation failed before usable implementation evidence, "
                "state plainly for the audience: 实现阶段失败，因此没有机制结论. In English, say: "
                "the implementation stage failed, so there is no mechanism conclusion."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Fill missing report narratives only.",
                    "language": "English" if english else "Chinese",
                    "missing_fields": gaps,
                    "schema_hint": schema_hint,
                    "relevant_facts": _subset_facts_for_ids(fact_pack, gaps),
                    "existing_narrative": {
                        "ablation_narratives": narrative.get("ablation_narratives") or [],
                        "research_round_narratives": narrative.get("research_round_narratives") or [],
                    },
                },
                ensure_ascii=False,
                default=str,
            ),
        },
    ]
    payload = client.call_json(
        stage="review_report_supplement",
        round_num=0,
        messages=messages,
        schema_hint=json.dumps(schema_hint, ensure_ascii=False),
        stage_label="review_report_supplement",
        execution_label=f"{task_id}_supplement_{attempt}",
    )
    return _coerce_supplement_payload(payload, gaps)


_ROUND_FACT_FIELDS = (
    "id", "display_idea", "idea_summary", "mechanism_name", "changed_mechanism",
    "expected_effect", "metrics", "status", "decision", "seed_eval", "failure_reason",
    "metric_comparison", "gate_event", "gate",
)


def _round_fact_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: record.get(key) for key in _ROUND_FACT_FIELDS}


def _isolated_report_client(client: Any, *, task_id: str) -> Any:
    """Create an API client per concurrent report batch to isolate request state and logs."""
    if isinstance(client, ProviderClient):
        return ProviderClient(config_path=client.config_path, task_id=task_id, base_dir=client.base_dir)
    return client


def _summary_labels(*, language: str) -> Tuple[str, str, str]:
    if language in {"en", "english"}:
        return "Change:", "Result:", "Conclusion:"
    return "改动：", "结果：", "结论："


def _valid_round_summary(item: Any, *, requested_ids: set[str], language: str) -> Dict[str, str] | None:
    if not isinstance(item, dict):
        return None
    round_id = str(item.get("id") or "").strip()
    summary = str(item.get("summary") or "").strip()
    labels = _summary_labels(language=language)
    if round_id not in requested_ids or not summary or not all(label in summary for label in labels):
        return None
    return {"id": round_id, "implementation": summary}


def _request_round_batch(
    *,
    client: Any,
    task_id: str,
    batch_index: int,
    records: List[Dict[str, Any]],
    language: str,
) -> List[Dict[str, str]]:
    requested_ids = {str(record["id"]) for record in records}
    labels = _summary_labels(language=language)
    english = language in {"en", "english"}
    language_name = "English" if english else "Chinese"
    length_limit = "300 English characters" if english else "180 Chinese characters"
    messages = [
        {
            "role": "system",
            "content": (
                "You write factual per-round summaries for a EvoCast HTML experiment report. "
                "Use only the supplied records; do not invent facts. First distinguish mechanism facts from "
                "metric, gate, and seed-evaluation facts. For per-round metric comparisons, use "
                "metric_comparison.objective exactly: its reference_kind is current_best, and its "
                "relative_improvement is the recorded gate or seed-evaluation comparison. Do not recompute "
                "relative improvement and do not describe this per-round comparison as relative to the baseline "
                "unless the supplied record explicitly says reference_kind is baseline. Promotion is also relative "
                "to current_best. Do not output Markdown. "
                f"Keep each summary concise: no more than {length_limit}. "
                "Return exactly JSON: {\"rounds\":[{\"id\":\"Research007\",\"summary\":\"...\"}]}. "
                f"Write all summaries in {language_name}. Every summary must contain exactly three labelled sentences: "
                f"{labels[0]}... {labels[1]}... {labels[2]}..."
            ),
        },
        {"role": "user", "content": json.dumps({"rounds": records}, ensure_ascii=False, default=str)},
    ]
    for attempt in range(1, 3):
        attempt_messages = list(messages)
        if attempt == 2:
            attempt_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your prior response was incomplete or invalid. Return every requested id exactly once, "
                        "with a non-empty three-sentence summary. Do not omit any item."
                    ),
                }
            )
        batch_client = _isolated_report_client(
            client,
            task_id=f"{task_id}_review_batch_{batch_index:02d}_attempt_{attempt}",
        )
        payload = batch_client.call_json(
            stage="review_report_round",
            round_num=0,
            messages=attempt_messages,
            schema_hint=json.dumps({"rounds": [{"id": "string", "summary": "string"}]}, ensure_ascii=False),
            stage_label="review_report_round",
            execution_label=f"{task_id}_batch_{batch_index:02d}_attempt_{attempt}",
            stream_override=False,
            max_tokens_override=2500,
        )
        output = payload.get("rounds") if isinstance(payload, dict) else None
        if not isinstance(output, list):
            continue
        summaries: Dict[str, Dict[str, str]] = {}
        for item in output:
            normalized = _valid_round_summary(item, requested_ids=requested_ids, language=language)
            if normalized is None or normalized["id"] in summaries:
                summaries = {}
                break
            summaries[normalized["id"]] = normalized
        if set(summaries) == requested_ids and len(summaries) == len(records):
            return [summaries[str(record["id"])] for record in records]
    return []


def _batch_round_summaries(*, client: Any, task_id: str, records: List[Dict[str, Any]], language: str) -> List[Dict[str, str]]:
    batches = [records[index:index + 5] for index in range(0, len(records), 5)]
    if not batches:
        return []
    completed: Dict[int, List[Dict[str, str]]] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
        futures = {
            executor.submit(_request_round_batch, client=client, task_id=task_id, batch_index=index, records=batch, language=language): index
            for index, batch in enumerate(batches, start=1)
        }
        for future in as_completed(futures):
            batch_index = futures[future]
            try:
                completed[batch_index] = future.result()
            except Exception:
                completed[batch_index] = []
    return [item for index in range(1, len(batches) + 1) for item in completed.get(index, [])]


def _request_global_narrative(*, client: Any, compact_pack: Dict[str, Any], narrative: Dict[str, Any], language: str) -> Dict[str, Any]:
    english = language in {"en", "english"}
    schema_hint = {
        "title": "string", "executive_summary": [{"text": "string"}], "baseline_summary": "string",
        "baseline_diagnosis_summary": "string", "final_architecture_summary": "string",
        "review_checklist": ["string"], "risk_notes": ["string"],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Generate the concise global narrative for a EvoCast HTML report from the supplied final facts and "
                "already-written per-round summaries. Do not rewrite per-round summaries and do not invent facts. "
                "Return exactly one JSON object matching the schema. "
                + ("Write user-facing text in English." if english else "Write user-facing text in Chinese.")
                + " Return at most five executive items, three checklist items, and three risk items."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "language": "English" if english else "Chinese",
                    "task": compact_pack.get("task"), "baseline": compact_pack.get("baseline"),
                    "final_state": compact_pack.get("final_state"),
                    "ablation_summaries": narrative.get("ablation_narratives"),
                    "research_summaries": narrative.get("research_round_narratives"),
                    "schema_hint": schema_hint,
                }, ensure_ascii=False, default=str,
            ),
        },
    ]
    payload = client.call_json(
        stage="review_report", round_num=0, messages=messages,
        schema_hint=json.dumps(schema_hint, ensure_ascii=False), stage_label="review_report",
        execution_label=f"{compact_pack.get('task', {}).get('task_id', 'unknown')}_global",
        stream_override=False,
    )
    return payload if isinstance(payload, dict) else {}


def build_review_narrative(*, client: Any, fact_pack: Dict[str, Any]) -> Dict[str, Any]:
    compact_pack = compact_fact_pack_for_narrative(fact_pack)
    task = compact_pack.get("task") if isinstance(compact_pack, dict) else {}
    task_id = str((task or {}).get("task_id") or "unknown")
    language = str((task or {}).get("language") or "zh").strip().lower()
    research_records = [_round_fact_record(item) for item in list(compact_pack.get("research_rounds") or []) if isinstance(item, dict) and item.get("id")]
    diagnosis = compact_pack.get("baseline_diagnosis") if isinstance(compact_pack.get("baseline_diagnosis"), dict) else {}
    ablation_records = [_round_fact_record(item) for item in list(diagnosis.get("ablations") or []) if isinstance(item, dict) and item.get("id")]

    narrative = _default_narrative()
    narrative["ablation_narratives"] = _batch_round_summaries(
        client=client, task_id=f"{task_id}_ablation", records=ablation_records, language=language,
    )
    narrative["research_round_narratives"] = _batch_round_summaries(
        client=client, task_id=task_id, records=research_records, language=language,
    )
    global_payload = _request_global_narrative(client=client, compact_pack=compact_pack, narrative=narrative, language=language)
    global_payload["ablation_narratives"] = narrative["ablation_narratives"]
    global_payload["research_round_narratives"] = narrative["research_round_narratives"]
    return normalize_narrative(global_payload, language=language)


def write_review_narrative(*, task_id: str, base_dir: str, client: Any, fact_pack: Dict[str, Any]) -> tuple[Dict[str, Any], Path]:
    narrative = build_review_narrative(client=client, fact_pack=fact_pack)
    path = task_knowledge_dir(base_dir, task_id) / "review_report_narrative.json"
    path.write_text(json.dumps(narrative, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return narrative, path
