"""Build a locale-specific, user-facing report view from immutable facts.

The fact pack remains an audit artifact and therefore retains stable English
codes and original agent prose.  HTML is rendered only from this view model.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from evocast.domain.i18n import (
    is_english,
    locale_matches_text,
    localize_code,
    localize_evidence,
    localize_source,
    localize_status,
    normalize_language,
)


def _text(value: Any, fallback: str, *, language: str, field: str, warnings: List[Dict[str, str]]) -> str:
    text = str(value or "").strip()
    if text and _locale_matches_human_text(text, language):
        return text
    if text:
        warnings.append({"field": field, "reason": "locale_mismatch", "original": text})
    return fallback


def _locale_matches_human_text(value: Any, language: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    scrubbed = re.sub(r"`[^`]*`", " ", text)
    scrubbed = re.sub(r"\b[\w./\\-]+\.(?:py|json|jsonl|txt|md|csv|html|patch)\b", " ", scrubbed)
    scrubbed = re.sub(r"\b[0-9a-f]{12,40}\b", " ", scrubbed, flags=re.IGNORECASE)
    scrubbed = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", " ", scrubbed)
    scrubbed = re.sub(r"\b[A-Za-z_]+(?:_[A-Za-z0-9_]+){1,}\b", " ", scrubbed)
    scrubbed = re.sub(r"\s+", " ", scrubbed).strip()
    return locale_matches_text(scrubbed or text, language)


def _dataset_fallback(diagnosis: Dict[str, Any], *, language: str) -> Tuple[str, str]:
    facts = dict(diagnosis.get("basic_facts") or {})
    claims = [str(item.get("claim") or "") for item in list(diagnosis.get("derived_claims") or []) if isinstance(item, dict)]
    topics = [str(item.get("topic") or "") for item in list(diagnosis.get("research_implications") or []) if isinstance(item, dict)]
    if is_english(language):
        summary = f"The dataset has {int(facts.get('num_variables') or 0)} variables."
        interpretation = "Recommended research themes: " + ", ".join(topics[:3]) + "." if topics else "No dominant dataset-driven research theme was detected."
        return summary, interpretation
    traits = [localize_code(item, "claim", language) for item in claims[:3]]
    summary = f"该数据集包含 {int(facts.get('num_variables') or 0)} 个变量"
    if traits:
        summary += "，呈现" + "、".join(traits) + "等特征。"
    else:
        summary += "，当前诊断未识别出占主导地位的结构特征。"
    themes = [localize_code(item, "topic", language) for item in topics[:3]]
    interpretation = "建议优先关注：" + "、".join(themes) + "。" if themes else "该诊断仅作为研究先验，不单独构成模型性能结论。"
    return summary, interpretation


def _mechanism_display(item: Dict[str, Any], *, language: str) -> str:
    name = str(item.get("display_idea") or item.get("mechanism_name") or "").strip()
    code = str(item.get("mechanism_id") or "").strip()
    if name and (not code or name != code):
        return name
    code = code or name
    if not code:
        return str(item.get("display_idea") or item.get("id") or "").strip() or "-"
    localized = localize_code(code, "mechanism", language)
    if localized != code or is_english(language):
        return localized
    return f"机制（{code}）"


def _narrative_by_id(narrative: Dict[str, Any], key: str) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("id")): dict(item)
        for item in list(narrative.get(key) or [])
        if isinstance(item, dict) and item.get("id")
    }


def build_report_view_model(*, fact_pack: Dict[str, Any], narrative: Dict[str, Any]) -> Dict[str, Any]:
    view_facts = deepcopy(dict(fact_pack or {}))
    view_narrative = deepcopy(dict(narrative or {}))
    language = normalize_language((view_facts.get("task") or {}).get("language"))
    warnings: List[Dict[str, str]] = []

    task = dict(view_facts.get("task") or {})
    task["language"] = language
    task["build_mode_display"] = "构建模式" if not is_english(language) else "Build mode"
    view_facts["task"] = task

    default_title = "EvoCast Review Report" if is_english(language) else "EvoCast 实验复盘"
    view_narrative["title"] = _text(view_narrative.get("title"), default_title, language=language, field="title", warnings=warnings)
    architecture_search_not_started = str((view_facts.get("final_state") or {}).get("architecture_search_status") or "") == "not_started"
    for key, fallback in {
        "baseline_summary": "已记录基线实验结果。" if not is_english(language) else "Baseline experiment results were recorded.",
        "baseline_diagnosis_summary": "已汇总基线诊断与消融证据。" if not is_english(language) else "Baseline diagnosis and ablation evidence were summarized.",
        "final_architecture_summary": "最终架构由已验证的候选及其评估结果确定。" if not is_english(language) else "The final architecture is determined by verified candidates and their evaluation results.",
    }.items():
        view_narrative[key] = _text(view_narrative.get(key), fallback, language=language, field=key, warnings=warnings)
    summaries = []
    for index, item in enumerate(list(view_narrative.get("executive_summary") or [])):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        row["text"] = _text(row.get("text"), "已根据结构化实验事实生成本次执行摘要。" if not is_english(language) else "This summary was generated from structured experiment facts.", language=language, field=f"executive_summary[{index}]", warnings=warnings)
        summaries.append(row)
    view_narrative["executive_summary"] = summaries
    if architecture_search_not_started:
        view_narrative["executive_summary"] = [{
            "text": "Architecture search was not started because max_rounds=0." if is_english(language) else "架构搜索未启动：max_rounds=0，仅完成 baseline 与诊断消融。",
            "evidence_refs": [],
        }]
        view_narrative["final_architecture_summary"] = (
            "No architecture search result exists because max_rounds=0."
            if is_english(language)
            else "未产生架构搜索结果：max_rounds=0。"
        )
    timeline = []
    for index, item in enumerate(list(view_narrative.get("timeline") or [])):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        row["title"] = _text(row.get("title"), "执行阶段" if not is_english(language) else "Execution stage", language=language, field=f"timeline[{index}].title", warnings=warnings)
        row["summary"] = _text(row.get("summary"), "该阶段已根据结构化产物记录。" if not is_english(language) else "This stage is recorded from structured artifacts.", language=language, field=f"timeline[{index}].summary", warnings=warnings)
        row["outcome"] = _text(row.get("outcome"), "已记录" if not is_english(language) else "Recorded", language=language, field=f"timeline[{index}].outcome", warnings=warnings)
        timeline.append(row)
    view_narrative["timeline"] = timeline

    diagnosis = dict(view_facts.get("dataset_diagnosis") or {})
    if diagnosis:
        summary_fallback, interpretation_fallback = _dataset_fallback(diagnosis, language=language)
        diagnosis["status"] = localize_status(diagnosis.get("status"), language)
        diagnosis["characteristics_engine"] = localize_code(diagnosis.get("characteristics_engine"), "status", language)
        basic = dict(diagnosis.get("basic_facts") or {})
        basic["frequency"] = localize_code(basic.get("frequency"), "frequency", language)
        diagnosis["basic_facts"] = basic
        dataset_narrative = dict(diagnosis.get("llm_narrative") or {})
        dataset_narrative["dataset_summary"] = _text(dataset_narrative.get("dataset_summary"), summary_fallback, language=language, field="dataset_summary", warnings=warnings)
        dataset_narrative["research_interpretation"] = _text(dataset_narrative.get("research_interpretation"), interpretation_fallback, language=language, field="dataset_research_interpretation", warnings=warnings)
        diagnosis["llm_narrative"] = dataset_narrative
        claims = []
        for item in list(diagnosis.get("derived_claims") or []):
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row["claim"] = localize_code(row.get("claim"), "claim", language)
            row["evidence"] = localize_evidence(row.get("evidence"), language)
            source_code = str(item.get("claim") or "")
            row["research_implication"] = localize_code(
                "trend_aware_modeling" if source_code in {"strong_trend", "trend_flag_present"} else str(item.get("implication_code") or ""),
                "topic",
                language,
            ) or _text(row.get("research_implication"), "该判断可作为后续研究方向的参考。" if not is_english(language) else "This finding can guide later research.", language=language, field="claim_implication", warnings=warnings)
            claims.append(row)
        diagnosis["derived_claims"] = claims
        implications = []
        for item in list(diagnosis.get("research_implications") or []):
            if not isinstance(item, dict):
                continue
            row = dict(item)
            topic_code = str(item.get("topic") or "")
            row["topic"] = localize_code(topic_code, "topic", language)
            row["priority"] = localize_code(row.get("priority"), "priority", language)
            fallback = f"该方向与当前数据集诊断相符。" if not is_english(language) else "This direction is aligned with the dataset diagnosis."
            row["reason"] = _text(row.get("reason"), fallback, language=language, field=f"implication:{topic_code}", warnings=warnings)
            implications.append(row)
        diagnosis["research_implications"] = implications
        diagnostics = []
        for index, item in enumerate(list(diagnosis.get("diagnostics") or [])):
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row["severity"] = localize_status(row.get("severity"), language)
            row["message"] = _text(row.get("message"), "已记录诊断信息。" if not is_english(language) else "Diagnostic information was recorded.", language=language, field=f"diagnostic[{index}]", warnings=warnings)
            diagnostics.append(row)
        diagnosis["diagnostics"] = diagnostics
        view_facts["dataset_diagnosis"] = diagnosis

    baseline_diagnosis = dict(view_facts.get("baseline_diagnosis") or {})
    baseline_diagnosis["status"] = localize_status(baseline_diagnosis.get("status"), language)
    ablation_narratives = _narrative_by_id(view_narrative, "ablation_narratives")
    ablations = []
    for item in list(baseline_diagnosis.get("ablations") or []):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        aid = str(row.get("id") or "")
        row["mechanism_name"] = _mechanism_display(row, language=language)
        row["raw_status"] = row.get("status")
        row["raw_usable_evidence_status"] = row.get("usable_evidence_status")
        row["status"] = localize_status(row.get("status"), language)
        row["usable_evidence_status"] = localize_status(row.get("usable_evidence_status"), language)
        ablations.append(row)
        text = dict(ablation_narratives.get(aid) or {})
        if text:
            text["mechanism_display"] = _text(text.get("mechanism_display"), row["mechanism_name"], language=language, field=f"{aid}.mechanism_display", warnings=warnings)
            for field in ["what_changed", "why_it_matters", "code_change", "result_interpretation", "review_risk", "audit_trace"]:
                text[field] = _text(text.get(field), "", language=language, field=f"{aid}.{field}", warnings=warnings)
            ablation_narratives[aid] = text
    baseline_diagnosis["ablations"] = ablations
    view_facts["baseline_diagnosis"] = baseline_diagnosis

    research_narratives = _narrative_by_id(view_narrative, "research_round_narratives")
    research = []
    for item in list(view_facts.get("research_rounds") or []):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        rid = str(row.get("id") or "")
        text = dict(research_narratives.get(rid) or {})
        opportunity = dict(row.get("opportunity") or {})
        mechanism = _mechanism_display(opportunity, language=language)
        row["idea"] = _text(text.get("idea"), "", language=language, field=f"{rid}.idea", warnings=warnings)
        row["status"] = localize_status(row.get("status"), language)
        row["decision"] = localize_status(row.get("decision"), language)
        opportunity["mechanism_id"] = mechanism
        opportunity["research_question"] = _text(text.get("research_question"), "", language=language, field=f"{rid}.research_question", warnings=warnings)
        reasons = text.get("selection_reasons")
        if isinstance(reasons, list) and reasons and all(locale_matches_text(value, language) for value in reasons):
            opportunity["selection_reasons"] = [str(value) for value in reasons]
        else:
            opportunity["selection_reasons"] = []
            if reasons:
                warnings.append({"field": f"{rid}.selection_reasons", "reason": "locale_mismatch", "original": str(reasons)})
        row["opportunity"] = opportunity
        context = dict(text.get("context_reasoning") or {})
        row["context_reasoning"] = {
            localize_source(key, language): _text(
                value,
                "",
                language=language,
                field=f"{rid}.context.{key}",
                warnings=warnings,
            )
            for key, value in context.items()
        }
        row["autonomy_reasoning"] = _text(text.get("autonomy_reasoning"), "", language=language, field=f"{rid}.autonomy_reasoning", warnings=warnings)
        row["why_not_merely_repeat_previous_work"] = _text(text.get("non_repetition"), "", language=language, field=f"{rid}.non_repetition", warnings=warnings)
        text["idea_source"] = _text(text.get("idea_source"), "", language=language, field=f"{rid}.idea_source", warnings=warnings)
        text["implementation"] = _text(text.get("implementation"), "", language=language, field=f"{rid}.implementation", warnings=warnings)
        text["code_change"] = _text(text.get("code_change"), "", language=language, field=f"{rid}.code_change", warnings=warnings)
        text["result_interpretation"] = _text(text.get("result_interpretation"), "", language=language, field=f"{rid}.result_interpretation", warnings=warnings)
        text["relation_to_previous"] = _text(text.get("relation_to_previous"), "", language=language, field=f"{rid}.relation_to_previous", warnings=warnings)
        text["review_risk"] = _text(text.get("review_risk"), "", language=language, field=f"{rid}.review_risk", warnings=warnings)
        text["audit_trace"] = _text(text.get("audit_trace"), "", language=language, field=f"{rid}.audit_trace", warnings=warnings)
        research.append(row)
        research_narratives[rid] = text
    view_facts["research_rounds"] = research
    view_narrative["ablation_narratives"] = list(ablation_narratives.values())
    view_narrative["research_round_narratives"] = list(research_narratives.values())
    for key, fallback in {
        "review_checklist": "已完成报告事实与产物索引检查。" if not is_english(language) else "Report facts and artifact indexes were checked.",
        "risk_notes": "请结合完整实验协议解读当前结果。" if not is_english(language) else "Interpret current results together with the full experiment protocol.",
    }.items():
        values = []
        for index, value in enumerate(list(view_narrative.get(key) or [])):
            values.append(_text(value, fallback, language=language, field=f"{key}[{index}]", warnings=warnings))
        view_narrative[key] = values

    return {
        "schema_version": "review_view_model_v1",
        "fact_pack": view_facts,
        "narrative": view_narrative,
        "locale_validation": {"language": language, "status": "ok" if not warnings else "fallback_applied", "warnings": warnings},
    }
