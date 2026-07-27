from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from evocast.domain.i18n import localize_characteristic, localize_code, localize_source, localize_status
from evocast.domain.knowledge_paths import runtime_root
from evocast.reports.view_model import build_report_view_model


METRIC_KEYS = ["mse_norm", "mae_norm", "rmse_norm", "mse", "mae", "rmse"]

LABELS = {
    "zh": {
        "task": "任务",
        "stage": "阶段",
        "dataset_diagnosis": "数据集诊断",
        "ablations": "消融实验",
        "research": "正式研究",
        "research_brain": "研究大脑",
        "final": "最终结果",
        "planned_usable_failed": "计划 {planned} / 可用 {usable} / 失败 {failed}",
        "formal_rounds": "{count} 个正式轮次",
        "metrics": "指标",
        "value": "数值",
        "delta": "变化量",
        "relative_delta": "相对 current_best",
        "no_narrative": "没有记录叙述内容。",
        "no_summary": "叙述阶段没有返回执行摘要。",
        "dataset_characteristics": "数据集特征",
        "no_dataset": "该任务没有记录数据集诊断。",
        "status": "状态",
        "characteristics_engine": "特征引擎",
        "dataset_name": "数据集名称",
        "task_mode": "任务模式",
        "frequency": "频率",
        "train_shape": "训练集形状",
        "variables": "变量数",
        "missing_ratio": "缺失率",
        "research_interpretation": "研究解释",
        "derived_claims": "派生判断",
        "claim": "判断",
        "confidence": "置信度",
        "evidence": "证据",
        "implication": "研究启发",
        "research_implications": "研究启发",
        "topic": "主题",
        "priority": "优先级",
        "reason": "原因",
        "diagnostics": "诊断信息",
        "severity": "级别",
        "code": "代码",
        "message": "信息",
        "baseline": "基线",
        "baseline_search": "Baseline 寻优",
        "order": "顺序",
        "type": "类型",
        "baseline_diagnosis": "基线诊断",
        "baseline_metrics": "基线指标",
        "baseline_config": "基线配置",
        "model": "模型",
        "candidate": "候选项",
        "planned_executed_usable_failed": "计划 / 执行 / 可用 / 失败",
        "changed_files": "修改文件",
        "idea_planner": "Idea 规划器",
        "round_role": "轮次角色",
        "target_code_region": "目标代码区",
        "what_changed": "改了什么",
        "why_it_matters": "为什么重要",
        "result_interpretation": "结果解释",
        "review_risk": "审查风险",
        "artifacts": "产物",
        "source_influence": "证据来源影响",
        "source": "来源",
        "influence": "影响",
        "research_opportunity": "研究机会",
        "beliefs": "信念",
        "questions": "研究问题",
        "agenda": "下一议程",
        "interpretations": "解释更新",
        "current_question": "当前问题",
        "event_count": "证据事件数",
        "artifact_count": "工件数",
        "belief": "信念",
        "confidence": "置信度",
        "priority": "优先级",
        "hypotheses": "竞争解释",
        "prediction": "可证伪预测",
        "target_path": "目标路径",
        "completion": "完成条件",
        "conclusion": "结论",
        "opportunity": "机会",
        "research_question": "研究问题",
        "mechanism": "机制",
        "primary_fit_point": "主要编辑点",
        "linked_evidence": "关联证据",
        "selection_reasons": "选择理由",
        "designer_context_reasoning": "设计者上下文推理",
        "autonomy_reasoning": "自主创新理由",
        "non_repetition": "非重复性",
        "risks": "风险",
        "no_research": "该任务没有记录正式 Research 轮次。",
        "idea": "想法",
        "decision": "决策",
        "variant": "变体路径",
        "implementation": "实现",
        "round_artifacts": "轮次产物",
        "display_name": "显示名称",
        "changed_from_baseline": "是否相对基线改变",
        "variant_path": "变体路径",
        "final_metrics": "最终指标",
        "no_artifacts": "没有索引到面向 review 的产物。",
        "name": "名称",
        "path": "路径",
        "size": "大小",
        "executive_summary": "执行摘要",
        "timeline": "时间线",
        "baseline_diagnosis_and_ablations": "基线诊断与消融",
        "formal_research_rounds": "正式研究轮次",
        "final_architecture": "最终架构",
        "review_checklist": "审查检查清单",
        "risk_notes": "风险记录",
        "artifact_appendix": "产物附录",
        "task_id": "任务 ID",
        "dataset": "数据集",
        "objective_metric": "优化指标",
        "build_mode": "构建模式",
        "token_usage": "Token 消耗",
        "stage_timing": "阶段耗时与成本",
        "category": "类别",
        "elapsed_seconds": "耗时秒数",
        "api_cost": "API 费用",
        "cache_hit": "缓存命中",
        "cache_miss": "缓存未命中",
        "output_tokens": "输出 tokens",
        "llm_calls": "LLM 调用数",
        "total_tokens": "总 tokens",
        "prompt_tokens": "输入 tokens",
        "completion_tokens": "输出 tokens",
        "reasoning_tokens": "推理 tokens",
        "prompt_cache_hit_tokens": "缓存命中 tokens",
        "prompt_cache_miss_tokens": "缓存未命中 tokens",
        "token_by_stage": "按阶段统计",
        "module_research": "模块研究",
        "module_name": "模块名称",
        "module_family": "模块族",
        "exploration_scale": "探索尺度",
        "internal_components": "内部组件",
        "controls": "控制项",
        "module_validity": "模块有效性",
        "failure_kind": "失败类型",
        "repair_target": "修复目标",
        "runtime_detected_components": "运行时检出组件",
        "validity_checks": "有效性检查",
        "module_family_decision": "模块族决策",
        "metric_signal": "指标信号",
        "control_status": "控制状态",
        "promotion_decision": "Gate 决策",
        "evaluation_budget": "评估预算",
        "next_required": "下一步要求",
        "module_memory_reference": "模块族记忆引用",
        "component_traces": "组件运行轨迹",
        "component_kind": "组件类型",
        "trainable_params": "可训练参数",
        "affects_output": "影响最终输出",
        "called": "已调用",
        "output_norm": "输出范数",
        "has_grad": "有梯度",
        "monitor_title": "EvoCast 运行审计报告",
        "run_overview": "运行总览",
        "metric_dashboard": "指标看板",
        "round_monitor": "研究轮次监控",
        "event_timeline": "执行时间线",
        "diagnosis_snapshot": "诊断快照",
        "collapsed_artifacts": "产物索引",
        "baseline_model": "基线模型",
        "final_best": "最终最优",
        "best_metric": "最优指标",
        "baseline_value": "基线值",
        "baseline_single": "基线单次",
        "baseline_reference": "基线多种子参考",
        "smoke_value": "快速验证指标",
        "seed_value": "多种子指标",
        "vs_baseline": "相对 current_best",
        "vs_incumbent": "相对当前最优",
        "gate_decision": "门控决策",
        "round_status": "轮次状态",
        "fit_point": "改动位置",
        "integration_site": "集成位置",
        "hypothesis_short": "这轮在做什么",
        "implementation_summary": "实现摘要",
        "result_summary": "结果摘要",
        "seed_count": "种子数",
        "round": "轮次",
        "module": "模块",
        "module_type": "模块类型",
        "metric_source": "指标来源",
        "artifact_link": "产物",
        "no_seed_eval": "未进行多种子评估",
        "baseline_stage": "基线评估",
        "dataset_stage": "数据集分析",
        "baseline_diagnosis_stage": "基线诊断",
        "report_stage": "报告生成",
        "completed_stage": "已完成",
        "skipped_stage": "已跳过",
        "round_result": "轮次结果",
        "artifact_details": "查看产物",
        "raw_artifact_note": "以下为机器审计用产物索引，默认折叠；正文只保留监控所需信息。",
        "language_validation": "语言检查",
        "language_ok": "中文模式：正文标签和默认说明已中文化。",
        "implementation_failed": "实现失败",
        "no_mechanism_conclusion": "无机制结论",
        "cost_estimate": "成本估算",
        "total_cost": "总费用",
        "cache_hit_cost": "缓存命中费用",
        "cache_miss_input_cost": "缓存未命中输入费用",
        "output_cost": "输出费用",
        "pricing_source": "价格来源",
        "provider": "服务商",
        "valid_metric_rounds": "有效指标轮次",
        "clean_evidence_rounds": "干净证据轮次",
        "estimated_wall_time": "估算总耗时",
        "round_monitor_table": "轮次监控总表",
        "round_audit_table": "轮次总览",
        "round_audit_cards": "轮次记录",
        "round_type": "类型",
        "ablation_round": "消融",
        "research_round": "Research",
        "evidence_quality": "证据质量",
        "clean_positive_evidence": "干净正证据",
        "clean_negative_evidence": "干净负证据",
        "dirty_metric": "脏指标",
        "no_metric": "无有效指标",
        "round_elapsed": "轮次耗时",
        "round_tokens": "轮次 tokens",
        "round_cost": "轮次费用",
        "round_count": "总轮次",
        "success_failed": "成功 / 失败",
        "changed_position": "改动位置",
        "components": "组件",
        "idea_from_agent": "Agent 提出的 idea",
        "audit_appendix": "审计附录",
        "end_time": "结束时间",
        "result": "结果",
        "details_folded": "展开查看细节",
    },
    "en": {},
}
LABELS["en"] = {
    "task": "Task",
    "stage": "Stage",
    "dataset_diagnosis": "Dataset Diagnosis",
    "ablations": "Ablations",
    "research": "Research",
    "research_brain": "Research Brain",
    "final": "Final",
    "planned_usable_failed": "planned {planned} / usable {usable} / failed {failed}",
    "formal_rounds": "{count} formal rounds",
    "metrics": "Metrics",
    "value": "Value",
    "delta": "Delta",
    "relative_delta": "Vs current_best",
    "no_narrative": "No narrative items recorded.",
    "no_summary": "No executive summary was returned by the narrative stage.",
    "dataset_characteristics": "Dataset Characteristics",
    "no_dataset": "No dataset diagnosis was recorded for this task.",
    "status": "Status",
    "characteristics_engine": "Characteristics engine",
    "dataset_name": "Dataset name",
    "task_mode": "Task mode",
    "frequency": "Frequency",
    "train_shape": "Train shape",
    "variables": "Variables",
    "missing_ratio": "Missing ratio",
    "research_interpretation": "Research interpretation",
    "derived_claims": "Derived Claims",
    "claim": "Claim",
    "confidence": "Confidence",
    "evidence": "Evidence",
    "implication": "Implication",
    "research_implications": "Research Implications",
    "topic": "Topic",
    "priority": "Priority",
    "reason": "Reason",
    "diagnostics": "Diagnostics",
    "severity": "Severity",
    "code": "Code",
    "message": "Message",
        "baseline": "Baseline",
        "baseline_search": "Baseline Search",
        "order": "Order",
        "type": "Type",
    "baseline_diagnosis": "Baseline Diagnosis",
    "baseline_metrics": "Baseline Metrics",
    "baseline_config": "Baseline config",
    "model": "Model",
    "candidate": "Candidate",
    "planned_executed_usable_failed": "Planned / Executed / Usable / Failed",
    "changed_files": "Changed files",
    "idea_planner": "Idea planner",
    "round_role": "Round role",
    "target_code_region": "Target code region",
    "what_changed": "What changed",
    "why_it_matters": "Why it matters",
    "result_interpretation": "Result interpretation",
    "review_risk": "Review risk",
    "artifacts": "Artifacts",
    "source_influence": "Source Influence",
    "source": "Source",
    "influence": "Influence",
    "research_opportunity": "Research Opportunity",
    "beliefs": "Beliefs",
    "questions": "Research Questions",
    "agenda": "Next Agenda",
    "interpretations": "Interpretation Updates",
    "current_question": "Current Question",
    "event_count": "Evidence Events",
    "artifact_count": "Artifacts",
    "belief": "Belief",
    "confidence": "Confidence",
    "priority": "Priority",
    "hypotheses": "Competing hypotheses",
    "prediction": "Falsifiable prediction",
    "target_path": "Target path",
    "completion": "Completion criteria",
    "conclusion": "Conclusion",
    "opportunity": "Opportunity",
    "research_question": "Research question",
    "mechanism": "Mechanism",
    "primary_fit_point": "Primary fit point",
    "linked_evidence": "Linked evidence",
    "selection_reasons": "Selection reasons",
    "designer_context_reasoning": "Designer Context Reasoning",
    "autonomy_reasoning": "Autonomy reasoning",
    "non_repetition": "Non-repetition",
    "risks": "Risks",
    "no_research": "No formal Research rounds were recorded for this task.",
    "idea": "Idea",
    "decision": "Decision",
    "variant": "Variant",
    "implementation": "Implementation",
    "round_artifacts": "Round artifacts",
    "display_name": "Display name",
    "changed_from_baseline": "Changed from baseline",
    "variant_path": "Variant path",
    "final_metrics": "Final Metrics",
    "no_artifacts": "No review-facing artifacts were indexed.",
    "name": "Name",
    "path": "Path",
    "size": "Size",
    "executive_summary": "Executive Summary",
    "timeline": "Timeline",
    "baseline_diagnosis_and_ablations": "Baseline Diagnosis And Ablations",
    "formal_research_rounds": "Formal Research Rounds",
    "final_architecture": "Final Architecture",
    "review_checklist": "Review Checklist",
    "risk_notes": "Risk Notes",
    "artifact_appendix": "Artifact Appendix",
    "task_id": "Task",
    "dataset": "Dataset",
    "objective_metric": "Objective",
    "build_mode": "Build mode",
    "token_usage": "Token Usage",
    "stage_timing": "Stage Timing And Cost",
    "category": "Category",
    "elapsed_seconds": "Elapsed seconds",
    "api_cost": "API cost",
    "cache_hit": "Cache hit",
    "cache_miss": "Cache miss",
    "output_tokens": "Output tokens",
    "llm_calls": "LLM calls",
    "total_tokens": "Total tokens",
    "prompt_tokens": "Prompt tokens",
    "completion_tokens": "Completion tokens",
    "reasoning_tokens": "Reasoning tokens",
    "prompt_cache_hit_tokens": "Prompt cache hit tokens",
    "prompt_cache_miss_tokens": "Prompt cache miss tokens",
    "token_by_stage": "By stage",
    "module_research": "Module Research",
    "module_name": "Module name",
    "module_family": "Module family",
    "exploration_scale": "Exploration scale",
    "internal_components": "Internal components",
    "controls": "Controls",
    "module_validity": "Module validity",
    "failure_kind": "Failure kind",
    "repair_target": "Repair target",
    "runtime_detected_components": "Runtime detected components",
    "validity_checks": "Validity checks",
    "module_family_decision": "Module family decision",
    "metric_signal": "Metric signal",
    "control_status": "Control status",
    "promotion_decision": "Gate decision",
    "evaluation_budget": "Evaluation budget",
    "next_required": "Next required",
    "module_memory_reference": "Module memory reference",
    "component_traces": "Component traces",
    "component_kind": "Component kind",
    "trainable_params": "Trainable params",
    "affects_output": "Affects final output",
    "called": "Called",
    "output_norm": "Output norm",
    "has_grad": "Has grad",
    "monitor_title": "EvoCast Run Audit Report",
    "run_overview": "Run Overview",
    "metric_dashboard": "Metric Dashboard",
    "round_monitor": "Research Round Monitor",
    "event_timeline": "Execution Timeline",
    "diagnosis_snapshot": "Diagnosis Snapshot",
    "collapsed_artifacts": "Artifact Index",
    "baseline_model": "Baseline model",
    "final_best": "Final best",
    "best_metric": "Best metric",
    "baseline_value": "Baseline",
    "baseline_single": "Baseline single",
    "baseline_reference": "Baseline reference",
    "smoke_value": "Smoke",
    "seed_value": "Seed eval",
    "vs_baseline": "vs current_best",
    "vs_incumbent": "vs incumbent",
    "gate_decision": "Gate decision",
    "round_status": "Round status",
    "fit_point": "Fit point",
    "integration_site": "Integration site",
    "hypothesis_short": "What this round did",
    "implementation_summary": "Implementation summary",
    "result_summary": "Result summary",
    "seed_count": "Seeds",
    "round": "Round",
    "module": "Module",
    "module_type": "Module type",
    "metric_source": "Metric source",
    "artifact_link": "Artifacts",
    "no_seed_eval": "No seed eval",
    "baseline_stage": "Baseline evaluation",
    "dataset_stage": "Dataset analysis",
    "baseline_diagnosis_stage": "Baseline diagnosis",
    "report_stage": "Report generation",
    "completed_stage": "Completed",
    "skipped_stage": "Skipped",
    "round_result": "Round result",
    "artifact_details": "View artifacts",
    "raw_artifact_note": "The following artifact index is for machine audit and is collapsed by default.",
    "language_validation": "Language validation",
    "language_ok": "Human-facing labels and fallback text match the selected language.",
    "implementation_failed": "Implementation failed",
    "no_mechanism_conclusion": "No mechanism conclusion",
    "cost_estimate": "Cost estimate",
    "total_cost": "Total cost",
    "cache_hit_cost": "Cache-hit input cost",
    "cache_miss_input_cost": "Cache-miss input cost",
    "output_cost": "Output cost",
    "pricing_source": "Pricing source",
    "provider": "Provider",
    "valid_metric_rounds": "Valid metric rounds",
    "clean_evidence_rounds": "Clean evidence rounds",
    "estimated_wall_time": "Estimated wall time",
    "round_monitor_table": "Round monitor table",
    "round_audit_table": "Round overview",
    "round_audit_cards": "Round records",
    "round_type": "Type",
    "ablation_round": "Ablation",
    "research_round": "Research",
    "evidence_quality": "Evidence quality",
    "clean_positive_evidence": "Clean positive evidence",
    "clean_negative_evidence": "Clean negative evidence",
    "dirty_metric": "Dirty metric",
    "no_metric": "No valid metric",
    "round_elapsed": "Round elapsed",
    "round_tokens": "Round tokens",
    "round_cost": "Round cost",
    "round_count": "Rounds",
    "success_failed": "Success / failed",
    "changed_position": "Changed position",
    "components": "Components",
    "idea_from_agent": "Agent idea",
    "audit_appendix": "Audit appendix",
    "end_time": "End time",
    "result": "Result",
    "details_folded": "Expand details",
}


def _language(fact_pack: Dict[str, Any]) -> str:
    task = fact_pack.get("task") if isinstance(fact_pack.get("task"), dict) else {}
    value = str(task.get("language") or "zh").strip().lower()
    return "en" if value in {"en", "english"} else "zh"


def _t(lang: str, key: str) -> str:
    return LABELS.get(lang, LABELS["zh"]).get(key, LABELS["en"].get(key, key))


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _human_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\bfact_pack(?:\.[A-Za-z0-9_\[\]]+)+", "", text)
    text = re.sub(r"\bevidence_refs\b", "", text)
    text = re.sub(r"\bbaseline_diagnosis(?:\.[A-Za-z0-9_\[\]]+)+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _he(value: Any) -> str:
    return _e(_human_text(value))


def _cell(value: Any, fallback: str = "-") -> str:
    text = _human_text(value)
    return _e(text if text else fallback)


def _json(value: Any) -> str:
    return _e(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_count(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "-"


def _format_money(value: Any, currency: str = "USD") -> str:
    try:
        return f"{currency} {float(value):.6f}"
    except Exception:
        return "-"


def _format_bool(value: Any, *, lang: str) -> Any:
    if isinstance(value, bool):
        if lang == "en":
            return "yes" if value else "no"
        return "是" if value else "否"
    return value


def _compact_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = text.replace("\\", "/")
    marker = "/dataset/"
    if marker in normalized:
        return "dataset/" + normalized.split(marker, 1)[1]
    marker = "/.evocast/"
    if marker in normalized:
        return ".evocast/" + normalized.split(marker, 1)[1]
    return text


def _dataset_display(value: Any) -> str:
    text = _compact_path(value)
    if not text:
        return ""
    return Path(text.replace("\\", "/")).name or text


def _compact_integration_site(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("target_model_path", "runtime_path", "declared_path", "path", "module_path"):
            if value.get(key):
                return str(value.get(key))
        source = _compact_path(value.get("target_source_file"))
        insertion = str(value.get("insertion_mode") or "")
        if source and insertion:
            return f"{source} · {insertion}"
        return source or ""
    return str(value or "")


def _format_seed_metric(stats: Dict[str, Any] | None) -> str:
    if not isinstance(stats, dict):
        return "-"
    mean = stats.get("mean")
    std = stats.get("std")
    if not isinstance(mean, (int, float)):
        return "-"
    if isinstance(std, (int, float)):
        return f"{_format_metric(float(mean))} ± {_format_metric(float(std))}"
    return _format_metric(float(mean))


def _objective_metric(fact_pack: Dict[str, Any]) -> str:
    task = fact_pack.get("task") if isinstance(fact_pack.get("task"), dict) else {}
    return str(task.get("objective_metric") or "mse_norm")


def _seed_metric_stats(seed_eval: Dict[str, Any] | None, key: str) -> Dict[str, Any]:
    if not isinstance(seed_eval, dict):
        return {}
    stats = seed_eval.get("metric_stats")
    if not isinstance(stats, dict):
        return {}
    value = stats.get(key)
    return value if isinstance(value, dict) else {}


def _seed_metric_mean(seed_eval: Dict[str, Any] | None, key: str) -> float | None:
    stats = _seed_metric_stats(seed_eval, key)
    mean = stats.get("mean")
    return float(mean) if isinstance(mean, (int, float)) else None


def _display_metric(metrics: Dict[str, Any] | None, seed_eval: Dict[str, Any] | None, key: str) -> str:
    stats = _seed_metric_stats(seed_eval, key)
    if stats:
        return _format_seed_metric(stats)
    return _format_metric(_metric_value(metrics, key))


def _metric_number(metrics: Dict[str, Any] | None, seed_eval: Dict[str, Any] | None, key: str) -> float | None:
    seed_mean = _seed_metric_mean(seed_eval, key)
    if seed_mean is not None:
        return seed_mean
    value = _metric_value(metrics, key)
    return float(value) if isinstance(value, (int, float)) else None


def _relative_to_baseline(value: float | None, baseline: float | None, *, lang: str = "zh") -> str:
    if value is None or baseline is None or baseline == 0:
        return "-"
    return _format_relative_improvement((baseline - value) / abs(baseline), lang=lang)


def _format_relative_improvement(value: Any, *, lang: str) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    percent = abs(float(value)) * 100
    if percent < 1e-10:
        return "unchanged" if lang == "en" else "持平"
    text = f"{percent:.3g}%"
    if float(value) > 0:
        return f"improved {text}" if lang == "en" else f"改善 {text}"
    return f"degraded {text}" if lang == "en" else f"退化 {text}"


def _comparison_metric(comparison: Dict[str, Any] | None, key: str) -> Dict[str, Any]:
    if not isinstance(comparison, dict):
        return {}
    metrics = comparison.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    item = metrics.get(key)
    return item if isinstance(item, dict) else {}


def _comparison_objective(comparison: Dict[str, Any] | None, objective: str) -> Dict[str, Any]:
    if not isinstance(comparison, dict):
        return {}
    objective_item = comparison.get("objective")
    if isinstance(objective_item, dict) and objective_item:
        return objective_item
    return _comparison_metric(comparison, objective)


def _relative_from_comparison(comparison: Dict[str, Any] | None, objective: str, *, lang: str) -> str:
    item = _comparison_objective(comparison, objective)
    return _format_relative_improvement(item.get("relative_improvement"), lang=lang)


def _relative_reference(
    metrics: Dict[str, Any] | None,
    seed_eval: Dict[str, Any] | None,
    reference: Dict[str, Any],
    objective: str,
) -> tuple[float | None, float | None]:
    seed_value = _seed_metric_mean(seed_eval, objective)
    if seed_value is not None:
        return seed_value, _seed_metric_mean(reference.get("seed_eval"), objective)
    value = _metric_number(metrics, None, objective)
    return value, _metric_number(reference.get("metrics"), None, objective)


def _comparison_reference(fact_pack: Dict[str, Any]) -> Dict[str, Any]:
    reference = fact_pack.get("comparison_reference") if isinstance(fact_pack.get("comparison_reference"), dict) else {}
    if reference:
        return reference
    baseline = fact_pack.get("baseline") if isinstance(fact_pack.get("baseline"), dict) else {}
    return {"kind": "current_best", "metrics": baseline.get("metrics") or {}, "seed_eval": baseline.get("seed_eval") or {}}


def _dashboard_status(record: Dict[str, Any], *, lang: str) -> str:
    status = str(record.get("raw_status") or record.get("status") or "")
    if status == "failed_ablation_round" and not record.get("metrics"):
        return _t(lang, "implementation_failed")
    return _status(status, lang=lang)


def _promoted_to_current_best(record: Dict[str, Any]) -> bool:
    seed_eval = record.get("seed_eval") if isinstance(record.get("seed_eval"), dict) else {}
    if seed_eval.get("promoted_to_current_best") is True:
        return True
    if record.get("promoted_to_current_best") is True or record.get("is_current_best") is True:
        return True
    promotion = record.get("promotion_decision") if isinstance(record.get("promotion_decision"), dict) else {}
    seed_promotion = seed_eval.get("promotion_decision") if isinstance(seed_eval.get("promotion_decision"), dict) else {}
    decisions = [
        promotion.get("decision"),
        seed_promotion.get("decision"),
        record.get("promotion_decision") if not isinstance(record.get("promotion_decision"), dict) else None,
    ]
    return any(str(value or "").strip().lower() in {"promote", "promoted"} for value in decisions)


def _promotion_outcome_status(record: Dict[str, Any], *, lang: str) -> str:
    raw = str(record.get("raw_status") or record.get("status") or "").strip().lower()
    if raw == "failed_ablation_round" and not record.get("metrics"):
        return _t(lang, "implementation_failed")
    if raw in {"running", "pending", "started", "round_started", "proposal", "build", "experiment"}:
        return _status("running", lang=lang)
    if raw:
        return _status(raw, lang=lang)
    return _status("completed" if record.get("metrics") else "infra_failed", lang=lang)


def _ablation_gate(record: Dict[str, Any], *, lang: str) -> str:
    if str(record.get("raw_status") or record.get("status") or "") == "failed_ablation_round" and not record.get("metrics"):
        return _t(lang, "no_mechanism_conclusion")
    interpretation = record.get("interpretation") if isinstance(record.get("interpretation"), dict) else {}
    return interpretation.get("module_effect") or interpretation.get("recommended_action") or record.get("usable_evidence_status") or "-"


def _status(value: Any, *, lang: str) -> str:
    return localize_status(value, lang)


def _code(value: Any, category: str, *, lang: str) -> str:
    return localize_code(value, category, lang)


def _narrative_lookup(narrative: Dict[str, Any], key: str) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("id")): dict(item)
        for item in list(narrative.get(key) or [])
        if isinstance(item, dict) and item.get("id")
    }


def _round_module(record: Dict[str, Any]) -> Dict[str, Any]:
    spec = record.get("architecture_module_spec") if isinstance(record.get("architecture_module_spec"), dict) else {}
    manifest = record.get("module_manifest") if isinstance(record.get("module_manifest"), dict) else {}
    opportunity = record.get("opportunity") if isinstance(record.get("opportunity"), dict) else {}
    decision = record.get("module_experiment_decision") if isinstance(record.get("module_experiment_decision"), dict) else {}
    direction = record.get("research_direction") if isinstance(record.get("research_direction"), dict) else {}
    build_contract = record.get("build_contract") if isinstance(record.get("build_contract"), dict) else {}
    fit_point = opportunity.get("primary_fit_point") or spec.get("integration_site") or manifest.get("integration_site")
    integration_site = spec.get("integration_site") or manifest.get("integration_site")
    target_code_region = record.get("target_code_region") or direction.get("target_code_region")
    target_mechanism = opportunity.get("mechanism_id") or direction.get("target_mechanism")
    changed_files = ", ".join(str(item) for item in list(record.get("changed_files") or []) if str(item).strip())
    return {
        "name": spec.get("module_name")
        or manifest.get("module_name")
        or decision.get("module_name")
        or record.get("display_idea")
        or direction.get("terminal_display_title")
        or direction.get("idea_title")
        or target_mechanism
        or record.get("id"),
        "summary": str(record.get("idea_summary") or "").strip(),
        "family": spec.get("module_family") or manifest.get("module_family") or decision.get("module_family") or record.get("round_role"),
        "scale": spec.get("exploration_scale") or record.get("round_role"),
        "fit_point": _compact_integration_site(fit_point) or str(target_code_region or "") or changed_files,
        "integration_site": _compact_integration_site(integration_site) or changed_files or str(target_code_region or ""),
        "metric_signal": decision.get("metric_signal"),
        "control_status": decision.get("control_status"),
        "promotion_decision": decision.get("promotion_decision") or record.get("decision") or (record.get("build_outcome") or {}).get("status"),
        "evaluation_budget": decision.get("evaluation_budget"),
        "next_required": decision.get("next_required") or build_contract.get("semantic_goal"),
        "validity": (record.get("module_validity") or {}).get("status") if isinstance(record.get("module_validity"), dict) else (record.get("build_outcome") or {}).get("status", ""),
    }


def _stage_rows(fact_pack: Dict[str, Any]) -> List[Dict[str, Any]]:
    summary = fact_pack.get("stage_timing") if isinstance(fact_pack.get("stage_timing"), dict) else {}
    rows = []
    for item in list(summary.get("rows") or []):
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "")
        name = str(item.get("name") or "")
        if category == "runtime_event":
            continue
        rows.append(item)
    return rows


def _parse_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _format_duration(seconds: Any, *, lang: str = "zh") -> str:
    if not isinstance(seconds, (int, float)):
        return "-"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}min" if lang == "en" else f"{minutes:.1f} 分钟"
    hours = minutes / 60.0
    return f"{hours:.2f}h" if lang == "en" else f"{hours:.2f} 小时"


def _estimated_wall_seconds(fact_pack: Dict[str, Any]) -> float | None:
    stamps = [_parse_iso_datetime(item.get("end_time")) for item in _stage_rows(fact_pack)]
    stamps = [item for item in stamps if item is not None]
    if len(stamps) >= 2:
        return max(0.0, (max(stamps) - min(stamps)).total_seconds())
    elapsed_values = []
    for item in _stage_rows(fact_pack):
        value = item.get("elapsed_seconds")
        if value is None:
            value = item.get("elapsed_since_previous_seconds")
        if isinstance(value, (int, float)):
            elapsed_values.append(float(value))
    return sum(elapsed_values) if elapsed_values else None


def _round_stage_rows(fact_pack: Dict[str, Any], round_id: str) -> List[Dict[str, Any]]:
    prefix = f"{round_id}:"
    return [
        item
        for item in _stage_rows(fact_pack)
        if str(item.get("round_id") or "") == str(round_id)
        or str(item.get("name") or "").startswith(prefix)
    ]


def _round_elapsed_seconds(fact_pack: Dict[str, Any], round_id: str) -> float | None:
    rows = _round_stage_rows(fact_pack, round_id)
    starts = [_parse_iso_datetime(item.get("started_at")) for item in rows]
    ends = [_parse_iso_datetime(item.get("end_time")) for item in rows]
    starts = [value for value in starts if value is not None]
    ends = [value for value in ends if value is not None]
    if starts and ends:
        return max(0.0, (max(ends) - min(starts)).total_seconds())

    values = []
    for item in rows:
        if str(item.get("category") or "") != "stage":
            continue
        value = item.get("elapsed_seconds")
        if value is None:
            value = item.get("elapsed_since_previous_seconds")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return max(values) if values else None


def _round_token_totals(fact_pack: Dict[str, Any], round_id: str) -> Dict[str, int]:
    usage = fact_pack.get("token_usage") if isinstance(fact_pack.get("token_usage"), dict) else {}
    total = {"calls": 0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    round_key = str(round_id or "").strip()
    entry_prefix = f"{round_key}_".lower()
    for item in list(usage.get("entries") or []):
        if not isinstance(item, dict):
            continue
        if not str(item.get("file") or "").lower().startswith(entry_prefix):
            continue
        for key in total:
            value = item.get(key)
            if isinstance(value, (int, float)):
                total[key] += int(value)
    if total["total_tokens"]:
        return total

    match = re.match(r"^(?:research|ablation)(\d+)$", round_key, flags=re.IGNORECASE)
    if match:
        numeric_round = str(int(match.group(1)))
        for item in list(usage.get("by_round") or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("name") or "") != numeric_round:
                continue
            for key in total:
                value = item.get(key)
                if isinstance(value, (int, float)):
                    total[key] += int(value)
            return total

    marker = round_key.lower()
    for item in list(usage.get("by_stage") or []):
        if not isinstance(item, dict):
            continue
        if marker not in str(item.get("name") or "").lower():
            continue
        for key in total:
            value = item.get(key)
            if isinstance(value, (int, float)):
                total[key] += int(value)
    return total


def _round_cost_estimate(fact_pack: Dict[str, Any], round_id: str) -> tuple[float | None, str]:
    usage = fact_pack.get("token_usage") if isinstance(fact_pack.get("token_usage"), dict) else {}
    cost = usage.get("cost") if isinstance(usage.get("cost"), dict) else {}
    totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else {}
    currency = str(cost.get("currency") or "USD")
    round_key = str(round_id or "").strip()
    entry_prefix = f"{round_key}_".lower()
    entry_cost = 0.0
    matched_cost_entries = 0
    for item in list(usage.get("entries") or []):
        if not isinstance(item, dict):
            continue
        if not str(item.get("file") or "").lower().startswith(entry_prefix):
            continue
        item_cost = item.get("cost") if isinstance(item.get("cost"), dict) else {}
        value = item_cost.get("total_cost")
        if isinstance(value, (int, float)):
            entry_cost += float(value)
            matched_cost_entries += 1
    if matched_cost_entries:
        return entry_cost, currency

    total_cost = cost.get("total_cost")
    total_tokens = totals.get("total_tokens")
    round_tokens = _round_token_totals(fact_pack, round_id).get("total_tokens")
    if not isinstance(total_cost, (int, float)) or not isinstance(total_tokens, (int, float)) or not total_tokens or not round_tokens:
        return None, currency
    return float(total_cost) * (float(round_tokens) / float(total_tokens)), currency


def _round_tokens_display(fact_pack: Dict[str, Any], round_id: str, *, lang: str) -> str:
    tokens = _round_token_totals(fact_pack, round_id)
    total_tokens = tokens.get("total_tokens")
    if total_tokens:
        return _format_count(total_tokens)
    usage = fact_pack.get("token_usage") if isinstance(fact_pack.get("token_usage"), dict) else {}
    totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else {}
    if totals.get("total_tokens"):
        return "未归因" if lang == "zh" else "Unattributed"
    return "未记录" if lang == "zh" else "Not recorded"


def _round_cost_display(fact_pack: Dict[str, Any], round_id: str, *, lang: str) -> str:
    round_cost, currency = _round_cost_estimate(fact_pack, round_id)
    if round_cost is not None:
        return _format_money(round_cost, currency)
    usage = fact_pack.get("token_usage") if isinstance(fact_pack.get("token_usage"), dict) else {}
    cost = usage.get("cost") if isinstance(usage.get("cost"), dict) else {}
    if cost.get("total_cost") is None:
        return "未记录价格" if lang == "zh" else "Price not recorded"
    return "未归因" if lang == "zh" else "Unattributed"


def _round_objective_comparison(record: Dict[str, Any], objective: str) -> Dict[str, Any]:
    return _comparison_objective(record.get("metric_comparison"), objective)


def _round_has_valid_metric(record: Dict[str, Any], objective: str) -> bool:
    if _seed_metric_mean(record.get("seed_eval") if isinstance(record.get("seed_eval"), dict) else None, objective) is not None:
        return True
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    return isinstance(metrics.get(objective), (int, float))


def _round_evidence_quality(record: Dict[str, Any], objective: str, *, lang: str) -> str:
    if not _round_has_valid_metric(record, objective):
        return _t(lang, "no_metric")
    validity = record.get("module_validity") if isinstance(record.get("module_validity"), dict) else {}
    seed_eval = record.get("seed_eval") if isinstance(record.get("seed_eval"), dict) else {}
    clean = validity.get("status") == "module_valid" and seed_eval.get("status") == "completed"
    if not clean:
        return _t(lang, "dirty_metric")
    outcome = str(_round_objective_comparison(record, objective).get("outcome") or "").lower()
    if outcome == "improved" or str(record.get("decision") or "").lower() == "accept":
        return _t(lang, "clean_positive_evidence")
    return _t(lang, "clean_negative_evidence")


def _round_gate_decision(record: Dict[str, Any], objective: str, *, lang: str) -> str:
    seed_eval = record.get("seed_eval") if isinstance(record.get("seed_eval"), dict) else {}
    comparison = _round_objective_comparison(record, objective)
    if seed_eval.get("status") == "completed":
        decision = _status(record.get("decision") or comparison.get("outcome") or "completed", lang=lang)
        relative = _format_relative_improvement(comparison.get("relative_improvement"), lang=lang)
        return f"{decision} · {relative}" if relative != "-" else decision
    module = _round_module(record)
    return _status(module.get("promotion_decision") or record.get("decision"), lang=lang)


def _research_round_records(fact_pack: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [item for item in list(fact_pack.get("research_rounds") or []) if isinstance(item, dict)]


def _valid_metric_round_count(fact_pack: Dict[str, Any]) -> int:
    objective = _objective_metric(fact_pack)
    return sum(1 for item in _research_round_records(fact_pack) if _round_has_valid_metric(item, objective))


def _round_success_failed_counts(fact_pack: Dict[str, Any]) -> tuple[int, int]:
    records = _research_round_records(fact_pack)
    objective = _objective_metric(fact_pack)
    success = sum(1 for item in records if _round_has_valid_metric(item, objective))
    return success, len(records) - success


def _clean_evidence_round_count(fact_pack: Dict[str, Any]) -> int:
    objective = _objective_metric(fact_pack)
    lang = _language(fact_pack)
    return sum(
        1
        for item in _research_round_records(fact_pack)
        if _round_evidence_quality(item, objective, lang=lang)
        in {_t(lang, "clean_positive_evidence"), _t(lang, "clean_negative_evidence")}
    )


def _best_candidate_summary(fact_pack: Dict[str, Any]) -> tuple[str, str]:
    objective = _objective_metric(fact_pack)
    final_state = fact_pack.get("final_state") if isinstance(fact_pack.get("final_state"), dict) else {}
    final_best = final_state.get("final_best") if isinstance(final_state.get("final_best"), dict) else {}
    if final_best:
        raw_name = str(final_best.get("display_name") or final_best.get("candidate_id") or final_best.get("model") or "-")
        name = raw_name
        compact = _compact_path(raw_name)
        if ".evocast/" in compact or compact.endswith(".py") or "/" in compact or "\\" in raw_name:
            variant = str(final_best.get("variant_path") or raw_name)
            for record in _research_round_records(fact_pack):
                rid = str(record.get("id") or "")
                module = _round_module(record)
                record_variant = str(record.get("variant_path") or "")
                if (record_variant and record_variant == variant) or (rid and rid in variant):
                    name = f"{rid} · {module.get('name') or '-'}"
                    break
        return str(name), _display_metric(final_best.get("metrics"), final_best.get("seed_eval"), objective)
    candidates: List[tuple[float, str, str]] = []
    baseline = fact_pack.get("baseline") if isinstance(fact_pack.get("baseline"), dict) else {}
    baseline_value = _metric_number(baseline.get("metrics"), baseline.get("seed_eval"), objective)
    if baseline_value is not None:
        candidates.append((baseline_value, str(baseline.get("model") or _t(_language(fact_pack), "baseline")), _display_metric(baseline.get("metrics"), baseline.get("seed_eval"), objective)))
    for record in _research_round_records(fact_pack):
        value = _metric_number(record.get("metrics"), record.get("seed_eval"), objective)
        if value is None:
            continue
        module = _round_module(record)
        label = f"{record.get('id') or '-'} · {module.get('name') or '-'}"
        candidates.append((value, label, _display_metric(record.get("metrics"), record.get("seed_eval"), objective)))
    if not candidates:
        return "-", "-"
    _, label, metric = min(candidates, key=lambda item: item[0])
    return label, metric


def _module_components(record: Dict[str, Any]) -> str:
    spec = record.get("architecture_module_spec") if isinstance(record.get("architecture_module_spec"), dict) else {}
    names = [
        str(item.get("name") or "").strip()
        for item in list(spec.get("internal_components") or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    if names:
        return ", ".join(names)
    direction = record.get("research_direction") if isinstance(record.get("research_direction"), dict) else {}
    opportunity = record.get("opportunity") if isinstance(record.get("opportunity"), dict) else {}
    mechanism = direction.get("target_mechanism") or opportunity.get("mechanism_id")
    return str(mechanism or "").strip()


def _display_title(*, lang: str, task: Dict[str, Any], narrative: Dict[str, Any]) -> str:
    if lang == "en":
        return str(narrative.get("title") or f"{_t(lang, 'monitor_title')} {task.get('task_id', '')}").strip()
    return f"{_t(lang, 'monitor_title')}（{task.get('task_id', '')}）"


def _monitor_cards(fact_pack: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    baseline = fact_pack.get("baseline") or {}
    objective = _objective_metric(fact_pack)
    best_name, best_metric = _best_candidate_summary(fact_pack)
    success, failed = _round_success_failed_counts(fact_pack)
    cards = [
        (_t(lang, "round_count"), len(_research_round_records(fact_pack))),
        (_t(lang, "success_failed"), f"{success} / {failed}"),
        (_t(lang, "baseline_reference"), _display_metric(None, baseline.get("seed_eval"), objective) if baseline.get("seed_eval") else _display_metric(baseline.get("metrics"), None, objective)),
        (_t(lang, "final_best"), f"{best_name} · {best_metric}（{objective}）" if best_metric != "-" else best_name),
        (_t(lang, "estimated_wall_time"), _format_duration(_estimated_wall_seconds(fact_pack), lang=lang)),
    ]
    return '<div class="monitor-cards">' + "".join(
        f'<div class="monitor-card"><div class="card-label">{_e(label)}</div><div class="card-value">{_he(value)}</div></div>'
        for label, value in cards
    ) + "</div>"


def _metric_dashboard(fact_pack: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    objective = _objective_metric(fact_pack)
    baseline = fact_pack.get("baseline") or {}
    reference = _comparison_reference(fact_pack)
    rows = [
        {
            "round": _t(lang, "baseline"),
            "module": baseline.get("model"),
            "status": _status("completed", lang=lang),
            "smoke": _display_metric(baseline.get("metrics"), None, objective),
            "seed": _display_metric(None, baseline.get("seed_eval"), objective) if baseline.get("seed_eval") else "-",
            "vs": "0%",
            "gate": "-",
        }
    ]
    diagnosis = fact_pack.get("baseline_diagnosis") if isinstance(fact_pack.get("baseline_diagnosis"), dict) else {}
    for record in diagnosis.get("ablations") or []:
        if not isinstance(record, dict):
            continue
        value, reference_value = _relative_reference(record.get("metrics"), record.get("seed_eval"), reference, objective)
        relative = _relative_from_comparison(record.get("metric_comparison"), objective, lang=lang)
        if relative == "-":
            relative = _relative_to_baseline(value, reference_value, lang=lang)
        rows.append(
            {
                "round": record.get("id"),
                "module": record.get("mechanism_name") or record.get("mechanism_id"),
                "status": _promotion_outcome_status(record, lang=lang),
                "smoke": _display_metric(record.get("metrics"), None, objective),
                "seed": _display_metric(None, record.get("seed_eval"), objective) if record.get("seed_eval") else _t(lang, "no_seed_eval"),
                "vs": relative,
                "gate": _ablation_gate(record, lang=lang),
            }
        )
    for record in fact_pack.get("research_rounds") or []:
        if not isinstance(record, dict):
            continue
        module = _round_module(record)
        value, reference_value = _relative_reference(record.get("metrics"), record.get("seed_eval"), reference, objective)
        relative = _relative_from_comparison(record.get("metric_comparison"), objective, lang=lang)
        if relative == "-":
            relative = _relative_to_baseline(value, reference_value, lang=lang)
        rows.append(
            {
                "round": record.get("id"),
                "module": module.get("name"),
                "status": _promotion_outcome_status(record, lang=lang),
                "smoke": _display_metric(record.get("metrics"), None, objective),
                "seed": _display_metric(None, record.get("seed_eval"), objective) if record.get("seed_eval") else _t(lang, "no_seed_eval"),
                "vs": relative,
                "gate": _round_gate_decision(record, objective, lang=lang),
            }
        )
    body = "".join(
        "<tr>"
        f"<td>{_he(row['round'])}</td>"
        f"<td>{_he(row['module'])}</td>"
        f"<td>{_he(row['status'])}</td>"
        f'<td class="metric-value">{_e(row["smoke"])}</td>'
        f'<td class="metric-value">{_e(row["seed"])}</td>'
        f'<td class="metric-value">{_e(row["vs"])}</td>'
        f"<td>{_he(_status(row['gate'], lang=lang))}</td>"
        "</tr>"
        for row in rows
    )
    return (
        '<table class="monitor-table"><thead><tr>'
        f'<th>{_e(_t(lang, "round"))}</th>'
        f'<th>{_e(_t(lang, "module"))}</th>'
        f'<th>{_e(_t(lang, "status"))}</th>'
        f'<th>{_e(_t(lang, "smoke_value"))}（{_e(objective)}）</th>'
        f'<th>{_e(_t(lang, "seed_value"))}（{_e(objective)}）</th>'
        f'<th>{_e(_t(lang, "vs_baseline"))}</th>'
        f'<th>{_e(_t(lang, "gate_decision"))}</th>'
        f'</tr></thead><tbody>{body}</tbody></table>'
    )


def _round_monitor_table(fact_pack: Dict[str, Any], narrative: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    objective = _objective_metric(fact_pack)
    reference = _comparison_reference(fact_pack)
    rows = []
    diagnosis = fact_pack.get("baseline_diagnosis") if isinstance(fact_pack.get("baseline_diagnosis"), dict) else {}
    ablation_by_id = _ablation_lookup(narrative)
    research_by_id = _research_lookup(narrative)
    for record in list(diagnosis.get("ablations") or []):
        if not isinstance(record, dict):
            continue
        rid = str(record.get("id") or "")
        value, reference_value = _relative_reference(record.get("metrics"), record.get("seed_eval"), reference, objective)
        relative = _relative_from_comparison(record.get("metric_comparison"), objective, lang=lang)
        if relative == "-":
            relative = _relative_to_baseline(value, reference_value, lang=lang)
        text = ablation_by_id.get(rid, {})
        rows.append(
            "<tr>"
            f"<td>{_he(_t(lang, 'ablation_round'))}</td>"
            f"<td>{_he(rid)}</td>"
            f"<td>{_he(_promotion_outcome_status(record, lang=lang))}</td>"
            f"<td>{_he(text.get('mechanism_display') or record.get('mechanism_name') or record.get('mechanism_id'))}</td>"
            f'<td class="metric-value">{_e(_display_metric(record.get("metrics"), record.get("seed_eval") if isinstance(record.get("seed_eval"), dict) else None, objective))}</td>'
            f'<td class="metric-value">{_e(relative)}</td>'
            f'<td class="metric-value">{_e(_format_duration(_round_elapsed_seconds(fact_pack, rid), lang=lang))}</td>'
            f'<td class="metric-value">{_e(_round_tokens_display(fact_pack, rid, lang=lang))}</td>'
            f'<td class="metric-value">{_e(_round_cost_display(fact_pack, rid, lang=lang))}</td>'
            "</tr>"
        )
    for record in _research_round_records(fact_pack):
        rid = str(record.get("id") or "")
        module = _round_module(record)
        value, reference_value = _relative_reference(record.get("metrics"), record.get("seed_eval"), reference, objective)
        relative = _relative_from_comparison(record.get("metric_comparison"), objective, lang=lang)
        if relative == "-":
            relative = _relative_to_baseline(value, reference_value, lang=lang)
        text = research_by_id.get(rid, {})
        rows.append(
            "<tr>"
            f"<td>{_he(_t(lang, 'research_round'))}</td>"
            f"<td>{_he(rid)}</td>"
            f"<td>{_he(_promotion_outcome_status(record, lang=lang))}</td>"
            f"<td>{_he(text.get('idea') or module.get('name'))}</td>"
            f'<td class="metric-value">{_e(_display_metric(record.get("metrics"), record.get("seed_eval") if isinstance(record.get("seed_eval"), dict) else None, objective))}</td>'
            f'<td class="metric-value">{_e(relative)}</td>'
            f'<td class="metric-value">{_e(_format_duration(_round_elapsed_seconds(fact_pack, rid), lang=lang))}</td>'
            f'<td class="metric-value">{_e(_round_tokens_display(fact_pack, rid, lang=lang))}</td>'
            f'<td class="metric-value">{_e(_round_cost_display(fact_pack, rid, lang=lang))}</td>'
            "</tr>"
        )
    if not rows:
        return f'<p class="muted">{_e(_t(lang, "no_research"))}</p>'
    return (
        '<table class="monitor-table round-audit-table"><thead><tr>'
        f'<th>{_e(_t(lang, "round_type"))}</th>'
        f'<th>{_e(_t(lang, "round"))}</th>'
        f'<th>{_e(_t(lang, "status"))}</th>'
        f'<th>{_e(_t(lang, "module"))}</th>'
        f'<th>{_e(objective)}</th>'
        f'<th>{_e(_t(lang, "vs_baseline"))}</th>'
        f'<th>{_e(_t(lang, "round_elapsed"))}</th>'
        f'<th>{_e(_t(lang, "round_tokens"))}</th>'
        f'<th>{_e(_t(lang, "round_cost"))}</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def _round_monitor(fact_pack: Dict[str, Any], narrative: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    objective = _objective_metric(fact_pack)
    baseline = fact_pack.get("baseline") or {}
    narrative_by_id = _narrative_lookup(narrative, "research_round_narratives")
    parts = []
    for record in fact_pack.get("research_rounds") or []:
        if not isinstance(record, dict):
            continue
        rid = str(record.get("id") or "")
        text = narrative_by_id.get(rid, {})
        module = _round_module(record)
        value, reference_value = _relative_reference(record.get("metrics"), record.get("seed_eval"), reference, objective)
        relative = _relative_from_comparison(record.get("metric_comparison"), objective, lang=lang)
        if relative == "-":
            relative = _relative_to_baseline(value, reference_value, lang=lang)
        seed_eval = record.get("seed_eval") if isinstance(record.get("seed_eval"), dict) else {}
        seed_count = seed_eval.get("valid_metric_seeds") or (_seed_metric_stats(seed_eval, objective).get("seed_count") if seed_eval else "")
        parts.append('<article class="round-card">')
        parts.append(
            '<div class="round-card-head">'
            f'<div><div class="round-id">{_e(rid)}</div><h3>{_he(module.get("name"))}</h3></div>'
            f'<div class="status-pill">{_he(_promotion_outcome_status(record, lang=lang))}</div>'
            '</div>'
        )
        parts.append(_kv_table([
            (_t(lang, "module_type"), _code(module.get("family"), "module_family", lang=lang)),
            (_t(lang, "exploration_scale"), _code(module.get("scale"), "exploration_scale", lang=lang)),
            (_t(lang, "fit_point"), module.get("fit_point")),
            (_t(lang, "integration_site"), module.get("integration_site")),
            (_t(lang, "module_validity"), _status(module.get("validity"), lang=lang)),
            (_t(lang, "metric_signal"), _status(module.get("metric_signal"), lang=lang)),
            (_t(lang, "control_status"), _status(module.get("control_status"), lang=lang)),
            (_t(lang, "gate_decision"), _status(module.get("promotion_decision"), lang=lang)),
            (_t(lang, "evaluation_budget"), _status(module.get("evaluation_budget"), lang=lang)),
            (_t(lang, "seed_count"), seed_count or "-"),
            (_t(lang, "vs_baseline"), relative),
        ]))
        parts.append(_metrics_table(
            record.get("metrics"),
            caption=f"{rid} {_t(lang, 'smoke_value')}",
            comparison=record.get("metric_comparison"),
            seed_eval=None,
            lang=lang,
        ))
        if seed_eval:
            parts.append(_metrics_table(None, caption=f"{rid} {_t(lang, 'seed_value')}", seed_eval=seed_eval, lang=lang))
        for label, key in [
            (_t(lang, "hypothesis_short"), "idea"),
            (_t(lang, "implementation_summary"), "implementation"),
            (_t(lang, "result_summary"), "result_interpretation"),
        ]:
            if text.get(key):
                parts.append(f'<p class="round-note"><strong>{_e(label)}：</strong>{_he(text.get(key))}</p>')
        parts.append('</article>')
    if not parts:
        return f'<p class="muted">{_e(_t(lang, "no_research"))}</p>'
    return "".join(parts)


def _round_audit_cards(fact_pack: Dict[str, Any], narrative: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    objective = _objective_metric(fact_pack)
    reference = _comparison_reference(fact_pack)
    narrative_by_id = _narrative_lookup(narrative, "research_round_narratives")
    parts = []
    for record in _research_round_records(fact_pack):
        rid = str(record.get("id") or "")
        text = narrative_by_id.get(rid, {})
        module = _round_module(record)
        value, reference_value = _relative_reference(record.get("metrics"), record.get("seed_eval"), reference, objective)
        relative = _relative_from_comparison(record.get("metric_comparison"), objective, lang=lang)
        if relative == "-":
            relative = _relative_to_baseline(value, reference_value, lang=lang)
        validity = record.get("module_validity") if isinstance(record.get("module_validity"), dict) else {}
        failure = record.get("failure_reason") or record.get("error") or validity.get("failure_kind")
        headline = f"{rid} · {module.get('name')} · {_promotion_outcome_status(record, lang=lang)} · {_display_metric(record.get('metrics'), record.get('seed_eval'), objective)} · {relative}"
        parts.append(f'<details class="audit-round-card"><summary>{_he(headline)}</summary>')
        parts.append(
            '<div class="audit-round-head">'
            f'<div><div class="round-id">{_e(rid)}</div><h3>{_he(text.get("idea") or module.get("name"))}</h3></div>'
            f'<div class="status-pill">{_he(_promotion_outcome_status(record, lang=lang))}</div>'
            '</div>'
        )
        if text.get("implementation"):
            parts.append(
                f'<div class="idea-box"><div class="fact-label">{_e(_t(lang, "implementation_summary"))}</div>'
                f'<div class="idea-text">{_he(text.get("implementation"))}</div></div>'
            )
        elif text.get("research_question"):
            parts.append(
                f'<div class="idea-box"><div class="fact-label">{_e(_t(lang, "research_question"))}</div>'
                f'<div class="idea-text">{_he(text.get("research_question"))}</div></div>'
            )
        elif module.get("summary") and (lang == "en" or any("\u4e00" <= char <= "\u9fff" for char in str(module.get("summary")))):
            parts.append(
                f'<div class="idea-box"><div class="fact-label">{_e(_t(lang, "implementation_summary"))}</div>'
                f'<div class="idea-text">{_he(module.get("summary"))}</div></div>'
            )
        parts.append(
            '<div class="fact-grid">'
            f'<div class="fact"><div class="fact-label">{_e(objective)}</div><div class="fact-value metric-value">{_e(_display_metric(record.get("metrics"), record.get("seed_eval") if isinstance(record.get("seed_eval"), dict) else None, objective))}</div></div>'
            f'<div class="fact"><div class="fact-label">{_e(_t(lang, "vs_baseline"))}</div><div class="fact-value metric-value">{_e(relative)}</div></div>'
            f'<div class="fact"><div class="fact-label">{_e(_t(lang, "round_elapsed"))}</div><div class="fact-value metric-value">{_e(_format_duration(_round_elapsed_seconds(fact_pack, rid), lang=lang))}</div></div>'
            f'<div class="fact"><div class="fact-label">{_e(_t(lang, "round_tokens"))} / {_e(_t(lang, "round_cost"))}</div><div class="fact-value metric-value">{_e(_round_tokens_display(fact_pack, rid, lang=lang))} / {_e(_round_cost_display(fact_pack, rid, lang=lang))}</div></div>'
            '</div>'
        )
        if failure:
            parts.append(f'<p class="round-note"><strong>{_e(_t(lang, "failure_kind"))}：</strong>{_he(failure)}</p>')
        parts.append(f'<details><summary>{_e(_t(lang, "details_folded"))}</summary>')
        parts.append(_metrics_table(
            record.get("metrics"),
            caption=f"{rid} {_t(lang, 'metrics')}",
            comparison=record.get("metric_comparison"),
            seed_eval=record.get("seed_eval"),
            lang=lang,
        ))
        detail_rows = [
            (_t(lang, "selection_reasons"), text.get("idea_source")),
            (_t(lang, "implementation"), text.get("implementation")),
            (_t(lang, "result_interpretation"), text.get("result_interpretation")),
            (_t(lang, "non_repetition"), text.get("relation_to_previous")),
            (_t(lang, "review_risk"), text.get("review_risk")),
            (_t(lang, "artifacts"), text.get("audit_trace")),
            (_t(lang, "module_type"), _code(module.get("family"), "module_family", lang=lang)),
            (_t(lang, "module_validity"), _status(module.get("validity"), lang=lang)),
            (_t(lang, "exploration_scale"), _code(module.get("scale"), "exploration_scale", lang=lang)),
            (_t(lang, "variant"), _compact_path(record.get("variant_path"))),
        ]
        parts.append(_kv_table([(label, value) for label, value in detail_rows if str(value or "").strip() not in {"", "-"}]))
        parts.append(f'<details><summary>{_e(_t(lang, "round_artifacts"))}</summary><pre>{_json(record.get("artifacts") or [])}</pre></details>')
        parts.append('</details>')
        parts.append('</details>')
    if not parts:
        return f'<p class="muted">{_e(_t(lang, "no_research"))}</p>'
    return "".join(parts)


def _event_timeline_monitor(fact_pack: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    rows_data = _stage_rows(fact_pack)
    if rows_data:
        rows = []
        for item in rows_data:
            name = str(item.get("name") or "")
            if "fact_pack" in name:
                continue
            elapsed = item.get("elapsed_seconds")
            if elapsed is None:
                elapsed = item.get("elapsed_since_previous_seconds")
            rows.append(
                "<tr>"
                f"<td>{_he(item.get('category'))}</td>"
                f"<td>{_he(name)}</td>"
                f"<td>{_he(str(item.get('end_time') or '').replace('T', ' ')[:19])}</td>"
                f'<td class="metric-value">{_e(_format_duration(elapsed, lang=lang))}</td>'
                f"<td>{_he(_status(item.get('status'), lang=lang))}</td>"
                "</tr>"
            )
        return (
            '<table class="monitor-table timeline-table"><thead><tr>'
            f'<th>{_e(_t(lang, "category"))}</th>'
            f'<th>{_e(_t(lang, "stage"))}</th>'
            f'<th>{_e(_t(lang, "end_time"))}</th>'
            f'<th>{_e(_t(lang, "elapsed_seconds"))}</th>'
            f'<th>{_e(_t(lang, "result"))}</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
        )
    dataset_diagnosis = fact_pack.get("dataset_diagnosis") or {}
    baseline_diagnosis = fact_pack.get("baseline_diagnosis") or {}
    baseline = fact_pack.get("baseline") or {}
    items = [
        ("01", _t(lang, "dataset_stage"), dataset_diagnosis.get("status") or "-"),
        ("02", _t(lang, "baseline_stage"), f"{baseline.get('model') or '-'} · {_t(lang, 'completed_stage')}"),
        ("03", _t(lang, "baseline_diagnosis_stage"), baseline_diagnosis.get("status") or "-"),
    ]
    index = 4
    for record in fact_pack.get("research_rounds") or []:
        if not isinstance(record, dict):
            continue
        module = _round_module(record)
        items.append((f"{index:02d}", str(record.get("id") or ""), f"{module.get('name') or '-'} · {_promotion_outcome_status(record, lang=lang)}"))
        index += 1
    items.append((f"{index:02d}", _t(lang, "report_stage"), _t(lang, "completed_stage")))
    return "".join(
        '<div class="timeline-row">'
        f'<div class="timeline-step">{_e(step)}</div>'
        f'<div class="timeline-title">{_he(title)}</div>'
        f'<div class="timeline-outcome">{_he(_status(outcome, lang=lang))}</div>'
        '</div>'
        for step, title, outcome in items
    )


def _diagnosis_snapshot(fact_pack: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    dataset_diagnosis = fact_pack.get("dataset_diagnosis") or {}
    baseline_diagnosis = fact_pack.get("baseline_diagnosis") or {}
    basic = dataset_diagnosis.get("basic_facts") if isinstance(dataset_diagnosis.get("basic_facts"), dict) else {}
    body = _kv_table([
        (_t(lang, "dataset_diagnosis"), dataset_diagnosis.get("status")),
        (_t(lang, "characteristics_engine"), dataset_diagnosis.get("characteristics_engine")),
        (_t(lang, "dataset_name"), basic.get("dataset_name")),
        (_t(lang, "frequency"), basic.get("frequency")),
        (_t(lang, "variables"), basic.get("num_variables")),
        (_t(lang, "baseline_diagnosis"), baseline_diagnosis.get("status")),
        (_t(lang, "planned_executed_usable_failed"), f"{baseline_diagnosis.get('planned', 0)} / {baseline_diagnosis.get('executed', 0)} / {baseline_diagnosis.get('usable', 0)} / {baseline_diagnosis.get('failed', 0)}"),
        (_t(lang, "language_validation"), _t(lang, "language_ok")),
    ])
    ablations = [item for item in list(baseline_diagnosis.get("ablations") or []) if isinstance(item, dict)]
    if ablations:
        objective = _objective_metric(fact_pack)
        rows = "".join(
            "<tr>"
            f"<td>{_he(item.get('id'))}</td>"
            f"<td>{_he(item.get('mechanism_name') or item.get('mechanism_id'))}</td>"
            f"<td>{_he(_status(item.get('status'), lang=lang))}</td>"
            f'<td class="metric-value">{_e(_display_metric(item.get("metrics"), item.get("seed_eval"), objective))}</td>'
            f"<td>{_he(_status(item.get('usable_evidence_status'), lang=lang))}</td>"
            "</tr>"
            for item in ablations
        )
        body += (
            f'<div class="table-title">{_e(_t(lang, "baseline_diagnosis_and_ablations"))}</div>'
            f'<table class="monitor-table"><thead><tr>'
            f'<th>{_e(_t(lang, "round"))}</th><th>{_e(_t(lang, "mechanism"))}</th><th>{_e(_t(lang, "status"))}</th>'
            f'<th>{_e(objective)}</th><th>{_e(_t(lang, "evidence"))}</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>'
        )
    return body


def _token_usage(fact_pack: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    usage = fact_pack.get("token_usage") if isinstance(fact_pack.get("token_usage"), dict) else {}
    totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else {}
    if not totals:
        return f'<p class="muted">{_e(_t(lang, "no_narrative"))}</p>'
    parts = [
        _kv_table([
            (_t(lang, "llm_calls"), _format_count(totals.get("calls"))),
            (_t(lang, "total_tokens"), _format_count(totals.get("total_tokens"))),
            (_t(lang, "prompt_tokens"), _format_count(totals.get("prompt_tokens"))),
            (_t(lang, "completion_tokens"), _format_count(totals.get("completion_tokens"))),
            (_t(lang, "reasoning_tokens"), _format_count(totals.get("reasoning_tokens"))),
            (_t(lang, "prompt_cache_hit_tokens"), _format_count(totals.get("prompt_cache_hit_tokens"))),
            (_t(lang, "prompt_cache_miss_tokens"), _format_count(totals.get("prompt_cache_miss_tokens"))),
        ])
    ]
    cost = usage.get("cost") if isinstance(usage.get("cost"), dict) else {}
    if cost:
        currency = str(cost.get("currency") or "USD")
        parts.append(
            f'<div class="table-title">{_e(_t(lang, "cost_estimate"))}</div>'
            + _kv_table([
                (_t(lang, "provider"), cost.get("provider")),
                (_t(lang, "model"), cost.get("model")),
                (_t(lang, "total_cost"), _format_money(cost.get("total_cost"), currency)),
                (
                    _t(lang, "cache_hit_cost"),
                    f'{_format_money(cost.get("cache_hit_cost"), currency)} / {_format_count(cost.get("cache_hit_tokens"))} tokens',
                ),
                (
                    _t(lang, "cache_miss_input_cost"),
                    f'{_format_money(cost.get("cache_miss_input_cost"), currency)} / {_format_count(cost.get("cache_miss_input_tokens"))} tokens',
                ),
                (
                    _t(lang, "output_cost"),
                    f'{_format_money(cost.get("output_cost"), currency)} / {_format_count(cost.get("output_tokens"))} tokens',
                ),
                (_t(lang, "pricing_source"), cost.get("source")),
            ])
        )
    by_stage = [item for item in list(usage.get("by_stage") or []) if isinstance(item, dict)]
    if by_stage:
        rows = "".join(
            "<tr>"
            f"<td>{_he(item.get('name'))}</td>"
            f'<td class="metric-value">{_e(_format_count(item.get("calls")))}</td>'
            f'<td class="metric-value">{_e(_format_count(item.get("total_tokens")))}</td>'
            f'<td class="metric-value">{_e(_format_count(item.get("prompt_tokens")))}</td>'
            f'<td class="metric-value">{_e(_format_count(item.get("completion_tokens")))}</td>'
            f'<td class="metric-value">{_e(_format_count(item.get("reasoning_tokens")))}</td>'
            "</tr>"
            for item in by_stage
        )
        parts.append(
            f'<div class="table-title">{_e(_t(lang, "token_by_stage"))}</div>'
            f'<table class="monitor-table"><thead><tr>'
            f'<th>{_e(_t(lang, "stage"))}</th>'
            f'<th>{_e(_t(lang, "llm_calls"))}</th>'
            f'<th>{_e(_t(lang, "total_tokens"))}</th>'
            f'<th>{_e(_t(lang, "prompt_tokens"))}</th>'
            f'<th>{_e(_t(lang, "completion_tokens"))}</th>'
            f'<th>{_e(_t(lang, "reasoning_tokens"))}</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>'
        )
    return "".join(parts)


def _stage_timing(fact_pack: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    summary = fact_pack.get("stage_timing") if isinstance(fact_pack.get("stage_timing"), dict) else {}
    rows_data = _stage_rows(fact_pack)
    if not rows_data:
        return f'<p class="muted">{_e(_t(lang, "no_narrative"))}</p>'
    totals = summary.get("totals") if isinstance(summary.get("totals"), dict) else {}
    parts = []
    if totals:
        currency = "USD"
        token_usage = fact_pack.get("token_usage") if isinstance(fact_pack.get("token_usage"), dict) else {}
        cost = token_usage.get("cost") if isinstance(token_usage.get("cost"), dict) else {}
        currency = str(cost.get("currency") or currency)
        parts.append(
            _kv_table([
                (_t(lang, "estimated_wall_time"), _format_duration(_estimated_wall_seconds(fact_pack), lang=lang)),
                (_t(lang, "total_cost"), _format_money(totals.get("api_total_cost"), currency)),
                (_t(lang, "prompt_tokens"), _format_count(totals.get("api_prompt_tokens"))),
                (_t(lang, "completion_tokens"), _format_count(totals.get("api_completion_tokens"))),
                (_t(lang, "prompt_cache_hit_tokens"), _format_count(totals.get("api_cache_hit_tokens"))),
                (_t(lang, "prompt_cache_miss_tokens"), _format_count(totals.get("api_cache_miss_input_tokens"))),
            ])
        )
    display_rows = rows_data[-80:]
    body = []
    for item in display_rows:
        cost = item.get("cost") if isinstance(item.get("cost"), dict) else {}
        currency = "USD"
        token_usage = fact_pack.get("token_usage") if isinstance(fact_pack.get("token_usage"), dict) else {}
        usage_cost = token_usage.get("cost") if isinstance(token_usage.get("cost"), dict) else {}
        currency = str(usage_cost.get("currency") or currency)
        elapsed = item.get("elapsed_seconds")
        if elapsed is None:
            elapsed = item.get("elapsed_since_previous_seconds")
        artifact = _compact_path(item.get("artifact"))
        if "fact_pack" in artifact:
            artifact = ""
        body.append(
            "<tr>"
            f"<td>{_cell(item.get('category'))}</td>"
            f"<td>{_cell(item.get('name'))}</td>"
            f"<td>{_cell(_status(item.get('status'), lang=lang))}</td>"
            f'<td class="metric-value">{_e(_format_metric(elapsed))}</td>'
            f'<td class="metric-value">{_e(_format_money(cost.get("total_cost"), currency) if cost else "-")}</td>'
            f'<td class="metric-value">{_e(_format_count(item.get("cache_hit_tokens")))}</td>'
            f'<td class="metric-value">{_e(_format_count(item.get("cache_miss_input_tokens")))}</td>'
            f'<td class="metric-value">{_e(_format_count(item.get("completion_tokens") or item.get("output_tokens")))}</td>'
            f'<td class="metric-value">{_e(_format_count(item.get("total_tokens")))}</td>'
            f"<td>{_cell(artifact)}</td>"
            "</tr>"
        )
    parts.append(
        f'<table class="monitor-table"><thead><tr>'
        f'<th>{_e(_t(lang, "category"))}</th>'
        f'<th>{_e(_t(lang, "stage"))}</th>'
        f'<th>{_e(_t(lang, "status"))}</th>'
        f'<th>{_e(_t(lang, "elapsed_seconds"))}</th>'
        f'<th>{_e(_t(lang, "api_cost"))}</th>'
        f'<th>{_e(_t(lang, "cache_hit"))}</th>'
        f'<th>{_e(_t(lang, "cache_miss"))}</th>'
        f'<th>{_e(_t(lang, "output_tokens"))}</th>'
        f'<th>{_e(_t(lang, "total_tokens"))}</th>'
        f'<th>{_e(_t(lang, "artifact_link"))}</th>'
        f'</tr></thead><tbody>{"".join(body)}</tbody></table>'
    )
    notes = [str(item) for item in list(summary.get("notes") or []) if str(item).strip()]
    if notes:
        parts.append('<p class="muted">' + _he(" / ".join(notes)) + '</p>')
    return "".join(parts)


def _metric_value(metrics: Dict[str, Any] | None, key: str) -> Any:
    if not isinstance(metrics, dict) or not metrics:
        return None
    return metrics.get(key)


def _delta_value(delta: Dict[str, Any] | None, key: str, field: str) -> Any:
    if not isinstance(delta, dict):
        return None
    value = delta.get(key)
    if isinstance(value, dict):
        return value.get(field)
    if field == "delta":
        return value
    return None


def _metrics_table(
    metrics: Dict[str, Any] | None,
    *,
    caption: str = "Metrics",
    delta: Dict[str, Any] | None = None,
    comparison: Dict[str, Any] | None = None,
    seed_eval: Dict[str, Any] | None = None,
    lang: str = "zh",
) -> str:
    has_comparison = isinstance(comparison, dict) and bool((comparison.get("metrics") or {}))
    has_delta = isinstance(delta, dict) and bool(delta)
    seed_stats = dict((seed_eval or {}).get("metric_stats") or {}) if isinstance(seed_eval, dict) else {}
    head = "<thead><tr><th></th>" + "".join(f"<th>{_e(key)}</th>" for key in METRIC_KEYS) + "</tr></thead>"
    rows = [
        f"<tr><th>{_e(_t(lang, 'value'))}</th>"
        + "".join(
            f'<td class="metric-value">{_e(_format_seed_metric(seed_stats.get(key)) if key in seed_stats else _format_metric(_metric_value(metrics, key)))}</td>'
            for key in METRIC_KEYS
        )
        + "</tr>"
    ]
    if has_comparison:
        rows.append(
            f"<tr><th>{_e(_t(lang, 'delta'))}</th>"
            + "".join(
                f'<td class="metric-value">{_e(_format_metric(_comparison_metric(comparison, key).get("delta")))}</td>'
                for key in METRIC_KEYS
            )
            + "</tr>"
        )
        rows.append(
            f"<tr><th>{_e(_t(lang, 'relative_delta'))}</th>"
            + "".join(
                f'<td class="metric-value">{_e(_format_relative_improvement(_comparison_metric(comparison, key).get("relative_improvement"), lang=lang))}</td>'
                for key in METRIC_KEYS
            )
            + "</tr>"
        )
    elif has_delta:
        rows.append(
            f"<tr><th>{_e(_t(lang, 'delta'))}</th>"
            + "".join(
                f'<td class="metric-value">{_e(_format_metric(_delta_value(delta, key, "delta")))}</td>'
                for key in METRIC_KEYS
            )
            + "</tr>"
        )
        rel_cells = []
        for key in METRIC_KEYS:
            rel = _delta_value(delta, key, "relative_delta")
            rel_text = _format_relative_improvement(-float(rel), lang=lang) if isinstance(rel, (int, float)) else str(rel or "-")
            rel_cells.append(f'<td class="metric-value">{_e(rel_text)}</td>')
        rows.append(f"<tr><th>{_e(_t(lang, 'relative_delta'))}</th>" + "".join(rel_cells) + "</tr>")
    return (
        f'<div class="metric-block"><div class="table-title">{_e(caption)}</div>'
        f'<table class="metrics-table">{head}<tbody>{"".join(rows)}</tbody></table></div>'
    )


def _kv_table(rows: Iterable[tuple[str, Any]]) -> str:
    body = "".join(
        f"<tr><th>{_e(label)}</th><td>{_he(value) if str(value or '').strip() else '-'}</td></tr>"
        for label, value in rows
    )
    return f"<table><tbody>{body}</tbody></table>"


def _section(title: str, body: str) -> str:
    return f'<section class="section"><h2>{_e(title)}</h2>{body}</section>'


def _list(items: Iterable[Any], *, lang: str = "zh") -> str:
    rows = [f"<li>{_he(item)}</li>" for item in items if str(item or "").strip()]
    if not rows:
        return f'<p class="muted">{_e(_t(lang, "no_narrative"))}</p>'
    return "<ul>" + "".join(rows) + "</ul>"


def _summary_blocks(items: List[Dict[str, Any]], *, lang: str = "zh") -> str:
    if not items:
        return f'<p class="muted">{_e(_t(lang, "no_summary"))}</p>'
    parts = []
    for item in items:
        parts.append(f'<div class="narrative"><p>{_he(item.get("text"))}</p></div>')
    return "".join(parts)


def _cards(fact_pack: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    diagnosis = fact_pack.get("baseline_diagnosis") or {}
    dataset_diagnosis = fact_pack.get("dataset_diagnosis") or {}
    research = fact_pack.get("research_rounds") or []
    final_state = fact_pack.get("final_state") or {}
    cards = [
        (_t(lang, "task"), (fact_pack.get("task") or {}).get("task_id", "")),
        (_t(lang, "dataset_diagnosis"), dataset_diagnosis.get("status") or ""),
        (_t(lang, "ablations"), _t(lang, "planned_usable_failed").format(planned=diagnosis.get("planned", 0), usable=diagnosis.get("usable", 0), failed=diagnosis.get("failed", 0))),
        (_t(lang, "research"), _t(lang, "formal_rounds").format(count=len(research))),
        (_t(lang, "final"), ((final_state.get("final_best") or {}).get("display_name") or (final_state.get("final_best") or {}).get("candidate_id") or "")),
    ]
    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="card-label">{_e(label)}</div><div class="card-value">{_e(value)}</div></div>'
        for label, value in cards
    ) + "</div>"


def _dataset_characteristics_table(raw: Dict[str, Any] | None, *, lang: str = "zh") -> str:
    raw = raw if isinstance(raw, dict) else {}
    keys = [
        "Correlation",
        "Transition",
        "Shifting",
        "Seasonality",
        "Trend",
        "Stationarity",
        "Short_term_jsd",
        "Long_term_jsd",
    ]
    head = "<thead><tr>" + "".join(f"<th>{_e(localize_characteristic(key, lang))}</th>" for key in keys) + "</tr></thead>"
    body = "<tbody><tr>" + "".join(
        f'<td class="metric-value">{_e(_format_metric(raw.get(key)))}</td>'
        for key in keys
    ) + "</tr></tbody>"
    return (
        f'<div class="metric-block"><div class="table-title">{_e(_t(lang, "dataset_characteristics"))}</div>'
        f'<table class="metrics-table">{head}{body}</table></div>'
    )


def _dataset_diagnosis(fact_pack: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    diagnosis = fact_pack.get("dataset_diagnosis") or {}
    if not diagnosis:
        return f'<p class="muted">{_e(_t(lang, "no_dataset"))}</p>'
    basic = diagnosis.get("basic_facts") or {}
    narrative = diagnosis.get("llm_narrative") or {}
    claims = [item for item in list(diagnosis.get("derived_claims") or []) if isinstance(item, dict)]
    implications = [item for item in list(diagnosis.get("research_implications") or []) if isinstance(item, dict)]
    diagnostics = [item for item in list(diagnosis.get("diagnostics") or []) if isinstance(item, dict)]
    parts = [
        _kv_table([
            (_t(lang, "status"), diagnosis.get("status")),
            (_t(lang, "characteristics_engine"), diagnosis.get("characteristics_engine")),
            (_t(lang, "dataset_name"), basic.get("dataset_name")),
            (_t(lang, "task_mode"), basic.get("task_mode")),
            (_t(lang, "frequency"), basic.get("frequency")),
            (_t(lang, "train_shape"), basic.get("train_shape")),
            (_t(lang, "variables"), basic.get("num_variables")),
            (_t(lang, "missing_ratio"), basic.get("missing_ratio")),
        ]),
        _dataset_characteristics_table(diagnosis.get("raw_characteristics"), lang=lang),
    ]
    if narrative.get("dataset_summary"):
        parts.append(f'<p class="narrative">{_he(narrative.get("dataset_summary"))}</p>')
    if narrative.get("research_interpretation"):
        parts.append(f'<p><strong>{_e(_t(lang, "research_interpretation"))}:</strong> {_he(narrative.get("research_interpretation"))}</p>')
    if claims:
        rows = "".join(
            "<tr>"
            f"<td>{_he(item.get('claim'))}</td>"
            f"<td class=\"metric-value\">{_e(_format_metric(item.get('confidence')))}</td>"
            f"<td>{_he(item.get('evidence'))}</td>"
            f"<td>{_he(item.get('research_implication'))}</td>"
            "</tr>"
            for item in claims
        )
        parts.append(
            f'<div class="table-title">{_e(_t(lang, "derived_claims"))}</div>'
            f'<table><thead><tr><th>{_e(_t(lang, "claim"))}</th><th>{_e(_t(lang, "confidence"))}</th><th>{_e(_t(lang, "evidence"))}</th><th>{_e(_t(lang, "implication"))}</th></tr></thead>'
            f"<tbody>{rows}</tbody></table>"
        )
    if implications:
        rows = "".join(
            "<tr>"
            f"<td>{_he(item.get('topic'))}</td>"
            f"<td>{_he(item.get('priority'))}</td>"
            f"<td class=\"metric-value\">{_e(_format_metric(item.get('confidence')))}</td>"
            f"<td>{_he(item.get('reason'))}</td>"
            "</tr>"
            for item in implications
        )
        parts.append(
            f'<div class="table-title">{_e(_t(lang, "research_implications"))}</div>'
            f'<table><thead><tr><th>{_e(_t(lang, "topic"))}</th><th>{_e(_t(lang, "priority"))}</th><th>{_e(_t(lang, "confidence"))}</th><th>{_e(_t(lang, "reason"))}</th></tr></thead>'
            f"<tbody>{rows}</tbody></table>"
        )
    if diagnostics:
        rows = "".join(
            "<tr>"
            f"<td>{_he(item.get('severity'))}</td>"
            f"<td>{_he(item.get('code'))}</td>"
            f"<td>{_he(item.get('message'))}</td>"
            "</tr>"
            for item in diagnostics
        )
        parts.append(
            f'<div class="table-title">{_e(_t(lang, "diagnostics"))}</div>'
            f'<table><thead><tr><th>{_e(_t(lang, "severity"))}</th><th>{_e(_t(lang, "code"))}</th><th>{_e(_t(lang, "message"))}</th></tr></thead>'
            f"<tbody>{rows}</tbody></table>"
        )
    return "".join(parts)


def _timeline(narrative: Dict[str, Any], fact_pack: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    items = narrative.get("timeline") or []
    if not items:
        diagnosis = fact_pack.get("baseline_diagnosis") or {}
        items = [
            {"id": "baseline", "title": _t(lang, "baseline"), "summary": "Baseline metrics recorded." if lang == "en" else "已记录基线指标。", "outcome": ""},
            {
                "id": "baseline_diagnosis",
                "title": _t(lang, "baseline_diagnosis"),
                "summary": f"{diagnosis.get('executed', 0)} ablations executed" if lang == "en" else f"已执行 {diagnosis.get('executed', 0)} 个消融实验",
                "outcome": str(diagnosis.get("status") or ""),
            },
        ]
        items.extend(
            {
                "id": item.get("id"),
                "title": item.get("id"),
                "summary": str(item.get("idea") or ""),
                "outcome": str(item.get("status") or ""),
            }
            for item in fact_pack.get("research_rounds") or []
        )
    rows = []
    for item in items:
        rows.append(
            '<div class="timeline-item">'
            f'<div class="timeline-id">{_he(item.get("id"))}</div>'
            f'<div><h3>{_he(item.get("title"))}</h3><p>{_he(item.get("summary"))}</p>'
            f'<p class="outcome">{_he(item.get("outcome"))}</p></div>'
            '</div>'
        )
    return "".join(rows)


def _baseline(fact_pack: Dict[str, Any], narrative: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    baseline = fact_pack.get("baseline") or {}
    body = [
        f'<p class="narrative">{_he(narrative.get("baseline_summary"))}</p>' if narrative.get("baseline_summary") else "",
        _kv_table([
            (_t(lang, "model"), baseline.get("model")),
            (_t(lang, "candidate"), baseline.get("candidate_id")),
        ]),
        _metrics_table(baseline.get("metrics"), caption=_t(lang, "baseline_metrics"), seed_eval=baseline.get("seed_eval"), lang=lang),
        f'<details><summary>{_e(_t(lang, "baseline_config"))}</summary><pre>{_json(baseline.get("config") or {})}</pre></details>',
    ]
    return "".join(body)


def _baseline_search(fact_pack: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    search = fact_pack.get("baseline_search") if isinstance(fact_pack.get("baseline_search"), dict) else {}
    rows = [row for row in list(search.get("rows") or []) if isinstance(row, dict)]
    if not rows:
        return '<p class="muted">-</p>'
    objective = str(search.get("objective_metric") or _objective_metric(fact_pack))
    body_rows = []
    for row in rows:
        rank = row.get("rank")
        emphasis = " baseline-search-best" if rank == 1 else " baseline-search-second" if rank == 2 else ""
        body_rows.append(
            f'<tr class="{emphasis.strip()}">'
            f'<td>{_e(row.get("order"))}</td>'
            f'<td>{_e(row.get("model"))}</td>'
            f'<td>{_e(row.get("type"))}</td>'
            f'<td class="metric-value">{_e(_format_metric(row.get("metric")))}</td>'
            f'<td>{_e(_format_duration(row.get("elapsed_seconds"), lang=lang))}</td>'
            '</tr>'
        )
    return (
        '<table class="monitor-table baseline-search-table"><thead><tr>'
        f'<th>{_e(_t(lang, "order"))}</th><th>{_e(_t(lang, "model"))}</th>'
        f'<th>{_e(_t(lang, "type"))}</th><th>{_e(objective)}</th><th>{_e(_t(lang, "round_elapsed"))}</th>'
        f'</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'
    )


def _ablation_lookup(narrative: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for item in narrative.get("ablation_narratives") or []:
        if isinstance(item, dict) and item.get("id"):
            result[str(item["id"])] = item
    return result


def _ablations(fact_pack: Dict[str, Any], narrative: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    diagnosis = fact_pack.get("baseline_diagnosis") or {}
    narrative_by_id = _ablation_lookup(narrative)
    parts = []
    if narrative.get("baseline_diagnosis_summary"):
        parts.append(f'<p class="narrative">{_he(narrative.get("baseline_diagnosis_summary"))}</p>')
    overview_rows = [
        (_t(lang, "status"), diagnosis.get("status")),
        (_t(lang, "planned_executed_usable_failed"), f"{diagnosis.get('planned')} / {diagnosis.get('executed')} / {diagnosis.get('usable')} / {diagnosis.get('failed')}"),
    ]
    parts.append(_kv_table([(label, value) for label, value in overview_rows if str(value or "").strip() not in {"", "-"}]))
    for ablation in diagnosis.get("ablations") or []:
        aid = str(ablation.get("id") or "")
        text = narrative_by_id.get(aid, {})
        headline = f"{aid} · {ablation.get('mechanism_name') or ablation.get('mechanism_id')} · {_promotion_outcome_status(ablation, lang=lang)} · {_display_metric(ablation.get('metrics'), ablation.get('seed_eval'), _objective_metric(fact_pack))}"
        parts.append(f'<details class="round"><summary>{_he(headline)}</summary>')
        ablation_rows = [
            (_t(lang, "status"), ablation.get("status")),
            (_t(lang, "evidence"), ablation.get("usable_evidence_status")),
        ]
        parts.append(_kv_table([(label, value) for label, value in ablation_rows if str(value or "").strip() not in {"", "-"}]))
        parts.append(_metrics_table(
            ablation.get("metrics"),
            caption=f"{aid} {_t(lang, 'metrics')}",
            delta=ablation.get("metric_delta"),
            comparison=ablation.get("metric_comparison"),
            seed_eval=ablation.get("seed_eval"),
            lang=lang,
        ))
        if text.get("implementation"):
            parts.append(
                f'<div class="idea-box"><div class="fact-label">{_e(_t(lang, "implementation_summary"))}</div>'
                f'<div class="idea-text">{_he(text.get("implementation"))}</div></div>'
            )
        for label, key in [
            (_t(lang, "what_changed"), "what_changed"),
            (_t(lang, "why_it_matters"), "why_it_matters"),
            (_t(lang, "result_interpretation"), "result_interpretation"),
            (_t(lang, "review_risk"), "review_risk"),
            (_t(lang, "artifacts"), "audit_trace"),
        ]:
            if text.get(key):
                parts.append(f'<p><strong>{_e(label)}:</strong> {_he(text.get(key))}</p>')
        parts.append('</details>')
    return "".join(parts)


def _research_lookup(narrative: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for item in narrative.get("research_round_narratives") or []:
        if isinstance(item, dict) and item.get("id"):
            result[str(item["id"])] = item
    return result


def _source_influence_table(source_influence: Dict[str, Any], *, lang: str = "zh") -> str:
    if not isinstance(source_influence, dict) or not source_influence:
        return ""
    order = ["dataset", "user_intent", "ablation", "model_structure", "trial_history", "current_best"]
    rows = "".join(
        "<tr>"
        f"<td>{_he(localize_source(key, lang))}</td>"
        f"<td class=\"metric-value\">{_he(source_influence.get(key))}</td>"
        "</tr>"
        for key in order
        if key in source_influence
    )
    return (
        f'<div class="table-title">{_e(_t(lang, "source_influence"))}</div>'
        f'<table class="compact-table"><thead><tr><th>{_e(_t(lang, "source"))}</th><th>{_e(_t(lang, "influence"))}</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>"
    )


def _opportunity_panel(record: Dict[str, Any], *, lang: str = "zh") -> str:
    opportunity = record.get("opportunity") if isinstance(record.get("opportunity"), dict) else {}
    if not opportunity:
        return ""
    context_reasoning = record.get("context_reasoning") if isinstance(record.get("context_reasoning"), dict) else {}
    parts = ['<div class="opportunity-panel">']
    parts.append(f"<h4>{_e(_t(lang, 'research_opportunity'))}</h4>")
    parts.append(_kv_table([
        (_t(lang, "opportunity"), opportunity.get("opportunity_id")),
        (_t(lang, "research_question"), opportunity.get("research_question")),
        (_t(lang, "mechanism"), opportunity.get("mechanism_id")),
        (_t(lang, "primary_fit_point"), opportunity.get("primary_fit_point")),
        (_t(lang, "linked_evidence"), ", ".join(str(item) for item in list(opportunity.get("linked_evidence") or []))),
        (_t(lang, "selection_reasons"), "; ".join(str(item) for item in list(opportunity.get("selection_reasons") or []))),
    ]))
    parts.append(_source_influence_table(opportunity.get("source_influence") or {}, lang=lang))
    if context_reasoning:
        parts.append(f'<div class="table-title">{_e(_t(lang, "designer_context_reasoning"))}</div>')
        parts.append(_kv_table([(key.replace("_", " ").title(), value) for key, value in context_reasoning.items()]))
    if record.get("autonomy_reasoning"):
        parts.append(f'<p><strong>{_e(_t(lang, "autonomy_reasoning"))}:</strong> {_he(record.get("autonomy_reasoning"))}</p>')
    if record.get("why_not_merely_repeat_previous_work"):
        parts.append(f'<p><strong>{_e(_t(lang, "non_repetition"))}:</strong> {_he(record.get("why_not_merely_repeat_previous_work"))}</p>')
    risks = [str(item) for item in list(opportunity.get("risks") or []) if str(item).strip()]
    if risks:
        parts.append(f'<p><strong>{_e(_t(lang, "risks"))}:</strong> {_he("; ".join(risks))}</p>')
    parts.append("</div>")
    return "".join(parts)


def _module_research_panel(record: Dict[str, Any], *, lang: str = "zh") -> str:
    spec = record.get("architecture_module_spec") if isinstance(record.get("architecture_module_spec"), dict) else {}
    manifest = record.get("module_manifest") if isinstance(record.get("module_manifest"), dict) else {}
    validity = record.get("module_validity") if isinstance(record.get("module_validity"), dict) else {}
    decision = record.get("module_experiment_decision") if isinstance(record.get("module_experiment_decision"), dict) else {}
    evolution = record.get("module_evolution_decision") if isinstance(record.get("module_evolution_decision"), dict) else {}
    memory_reference = record.get("module_family_memory_reference") if isinstance(record.get("module_family_memory_reference"), dict) else {}
    if not any([spec, manifest, validity, decision, evolution]):
        return ""
    components = [
        str(item.get("name") or "")
        for item in list(spec.get("internal_components") or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    controls = list(spec.get("controls") or [])
    parts = ['<div class="opportunity-panel module-panel">']
    parts.append(f"<h4>{_e(_t(lang, 'module_research'))}</h4>")
    parts.append(_kv_table([
        (_t(lang, "module_name"), spec.get("module_name") or manifest.get("module_name")),
        (_t(lang, "module_family"), spec.get("module_family") or manifest.get("module_family")),
        (_t(lang, "exploration_scale"), spec.get("exploration_scale")),
        (_t(lang, "internal_components"), ", ".join(components)),
        (_t(lang, "controls"), ", ".join(str(item) for item in controls)),
        (_t(lang, "module_validity"), validity.get("status")),
        (_t(lang, "failure_kind"), validity.get("failure_kind")),
        (_t(lang, "repair_target"), validity.get("repair_target")),
        (_t(lang, "metric_signal"), decision.get("metric_signal")),
        (_t(lang, "control_status"), decision.get("control_status")),
        (_t(lang, "promotion_decision"), decision.get("promotion_decision")),
        (_t(lang, "module_family_decision"), decision.get("module_family_decision") or evolution.get("decision")),
        (_t(lang, "evaluation_budget"), decision.get("evaluation_budget")),
        (_t(lang, "next_required"), "; ".join(str(item) for item in list(decision.get("next_required") or evolution.get("next_required") or []))),
        (_t(lang, "module_memory_reference"), "; ".join(
            str(item)
            for item in [
                *(memory_reference.get("referenced_family_ids") or []),
                *(memory_reference.get("referenced_previous_decisions") or []),
                *(memory_reference.get("referenced_tested_modules") or []),
                *(memory_reference.get("next_required_addressed") or []),
            ]
            if str(item).strip()
        ) or memory_reference.get("rationale")),
    ]))
    traces = validity.get("component_traces") if isinstance(validity.get("component_traces"), dict) else {}
    runtime_probe = traces.get("_runtime_probe") if isinstance(traces.get("_runtime_probe"), dict) else {}
    detected_components = runtime_probe.get("runtime_detected_components") if isinstance(runtime_probe.get("runtime_detected_components"), dict) else {}
    if detected_components:
        parts.append(f'<div class="table-title">{_e(_t(lang, "runtime_detected_components"))}</div>')
        parts.append(_kv_table([(name, path) for name, path in detected_components.items()]))
    checks = validity.get("checks") if isinstance(validity.get("checks"), dict) else {}
    if checks:
        parts.append(f'<details><summary>{_e(_t(lang, "validity_checks"))}</summary><pre>{_json(checks)}</pre></details>')
    alignment = runtime_probe.get("manifest_code_alignment") if isinstance(runtime_probe.get("manifest_code_alignment"), dict) else {}
    if alignment and not alignment.get("passed", True):
        parts.append(f'<details open><summary>{_e(_t(lang, "diagnostics"))}</summary><pre>{_json(alignment)}</pre></details>')
    component_rows = []
    for name, trace in traces.items():
        if str(name).startswith("_") or not isinstance(trace, dict):
            continue
        component_rows.append(
            "<tr>"
            f"<td>{_he(name)}</td>"
            f"<td>{_he(trace.get('runtime_path') or trace.get('declared_path'))}</td>"
            f"<td>{_he(trace.get('component_kind'))}</td>"
            f"<td>{_he(trace.get('called'))}</td>"
            f"<td class=\"metric-value\">{_e(_format_count(trace.get('trainable_param_count')))}</td>"
            f"<td class=\"metric-value\">{_e(_format_metric(trace.get('output_norm')))}</td>"
            f"<td>{_he(trace.get('has_grad'))}</td>"
            f"<td>{_he(trace.get('affects_final_output'))}</td>"
            "</tr>"
        )
    if component_rows:
        parts.append(
            f'<div class="table-title">{_e(_t(lang, "component_traces"))}</div>'
            f'<table class="compact-table"><thead><tr>'
            f'<th>{_e(_t(lang, "name"))}</th><th>{_e(_t(lang, "target_path"))}</th><th>{_e(_t(lang, "component_kind"))}</th><th>{_e(_t(lang, "called"))}</th><th>{_e(_t(lang, "trainable_params"))}</th><th>{_e(_t(lang, "output_norm"))}</th><th>{_e(_t(lang, "has_grad"))}</th><th>{_e(_t(lang, "affects_output"))}</th>'
            f'</tr></thead><tbody>{"".join(component_rows)}</tbody></table>'
        )
    parts.append("</div>")
    return "".join(parts)


def _research_rounds(fact_pack: Dict[str, Any], narrative: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    rounds = fact_pack.get("research_rounds") or []
    if not rounds:
        return f'<p class="muted">{_e(_t(lang, "no_research"))}</p>'
    narrative_by_id = _research_lookup(narrative)
    parts = []
    for record in rounds:
        rid = str(record.get("id") or "")
        text = narrative_by_id.get(rid, {})
        module = _round_module(record)
        objective = _objective_metric(fact_pack)
        parts.append('<article class="round">')
        parts.append(f'<h3>{_e(rid)} · {_he(module.get("name"))}</h3>')
        parts.append(_kv_table([
            (_t(lang, "status"), record.get("status")),
            (_t(lang, "hypothesis_short"), record.get("idea")),
            (_t(lang, "seed_value"), _display_metric(None, record.get("seed_eval") if isinstance(record.get("seed_eval"), dict) else None, objective)),
            (_t(lang, "vs_baseline"), _relative_from_comparison(record.get("metric_comparison"), objective, lang=lang)),
            (_t(lang, "evidence_quality"), _round_evidence_quality(record, objective, lang=lang)),
            (_t(lang, "decision"), _round_gate_decision(record, objective, lang=lang)),
        ]))
        parts.append(_metrics_table(
            record.get("metrics"),
            caption=f"{rid} {_t(lang, 'metrics')}",
            comparison=record.get("metric_comparison"),
            seed_eval=record.get("seed_eval"),
            lang=lang,
        ))
        for label, key in [(_t(lang, "implementation"), "implementation"), (_t(lang, "result_interpretation"), "result_interpretation")]:
            if text.get(key):
                parts.append(f'<p><strong>{_e(label)}:</strong> {_he(text.get(key))}</p>')
        parts.append(f'<details><summary>{_e(_t(lang, "details_folded"))}</summary>')
        parts.append(_kv_table([
            (_t(lang, "module_type"), _code(module.get("family"), "module_family", lang=lang)),
            (_t(lang, "exploration_scale"), _code(module.get("scale"), "exploration_scale", lang=lang)),
            (_t(lang, "fit_point"), module.get("fit_point")),
            (_t(lang, "integration_site"), module.get("integration_site")),
            (_t(lang, "variant"), _compact_path(record.get("variant_path"))),
        ]))
        parts.append(_opportunity_panel(record, lang=lang))
        parts.append(_module_research_panel(record, lang=lang))
        parts.append(f'<details><summary>{_e(_t(lang, "round_artifacts"))}</summary><pre>{_json(record.get("artifacts") or [])}</pre></details>')
        parts.append('</details>')
        parts.append('</article>')
    return "".join(parts)


def _research_brain(fact_pack: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    state = fact_pack.get("research_state") if isinstance(fact_pack.get("research_state"), dict) else {}
    if not state:
        return f'<p class="muted">{_e(_t(lang, "no_narrative"))}</p>'
    parts = [
        _kv_table([
            (_t(lang, "current_question"), state.get("current_question_id")),
            (_t(lang, "event_count"), state.get("event_count")),
            (_t(lang, "artifact_count"), state.get("artifact_count")),
        ])
    ]
    beliefs = [item for item in list(state.get("top_beliefs") or []) if isinstance(item, dict)]
    if beliefs:
        rows = "".join(
            "<tr>"
            f"<td>{_he(item.get('claim'))}</td>"
            f"<td>{_he(item.get('belief_type'))}</td>"
            f"<td class=\"metric-value\">{_e(_format_metric(item.get('confidence')))}</td>"
            f"<td>{_he('; '.join(str(x) for x in list(item.get('uncertainty') or [])))}</td>"
            "</tr>"
            for item in beliefs
        )
        parts.append(
            f'<div class="table-title">{_e(_t(lang, "beliefs"))}</div>'
            f'<table><thead><tr><th>{_e(_t(lang, "belief"))}</th><th>{_e(_t(lang, "mechanism"))}</th><th>{_e(_t(lang, "confidence"))}</th><th>{_e(_t(lang, "risks"))}</th></tr></thead><tbody>{rows}</tbody></table>'
        )
    questions = [item for item in list(state.get("open_questions") or []) if isinstance(item, dict)]
    if questions:
        rows = "".join(
            "<tr>"
            f"<td>{_he(item.get('question'))}</td>"
            f"<td>{_he('; '.join(str(x) for x in list(item.get('competing_hypotheses') or [])))}</td>"
            f"<td>{_he(item.get('falsifiable_prediction'))}</td>"
            f"<td class=\"metric-value\">{_e(_format_metric(item.get('information_value')))}</td>"
            "</tr>"
            for item in questions
        )
        parts.append(
            f'<div class="table-title">{_e(_t(lang, "questions"))}</div>'
            f'<table><thead><tr><th>{_e(_t(lang, "research_question"))}</th><th>{_e(_t(lang, "hypotheses"))}</th><th>{_e(_t(lang, "prediction"))}</th><th>{_e(_t(lang, "priority"))}</th></tr></thead><tbody>{rows}</tbody></table>'
        )
    agenda = [item for item in list(state.get("agenda") or []) if isinstance(item, dict)]
    if agenda:
        rows = "".join(
            "<tr>"
            f"<td>{_he(item.get('action_type'))}</td>"
            f"<td>{_he(item.get('target_path'))}</td>"
            f"<td class=\"metric-value\">{_e(_format_metric(item.get('priority')))}</td>"
            f"<td>{_he(item.get('rationale'))}</td>"
            f"<td>{_he('; '.join(str(x) for x in list(item.get('completion_criteria') or [])))}</td>"
            "</tr>"
            for item in agenda
        )
        parts.append(
            f'<div class="table-title">{_e(_t(lang, "agenda"))}</div>'
            f'<table><thead><tr><th>{_e(_t(lang, "decision"))}</th><th>{_e(_t(lang, "target_path"))}</th><th>{_e(_t(lang, "priority"))}</th><th>{_e(_t(lang, "reason"))}</th><th>{_e(_t(lang, "completion"))}</th></tr></thead><tbody>{rows}</tbody></table>'
        )
    interpretations = [item for item in list(state.get("interpretations") or []) if isinstance(item, dict)]
    if interpretations:
        rows = "".join(
            "<tr>"
            f"<td>{_he(item.get('question_id'))}</td>"
            f"<td>{_he(item.get('result_status'))}</td>"
            f"<td>{_he(item.get('conclusion'))}</td>"
            "</tr>"
            for item in interpretations
        )
        parts.append(
            f'<div class="table-title">{_e(_t(lang, "interpretations"))}</div>'
            f'<table><thead><tr><th>{_e(_t(lang, "research_question"))}</th><th>{_e(_t(lang, "status"))}</th><th>{_e(_t(lang, "conclusion"))}</th></tr></thead><tbody>{rows}</tbody></table>'
        )
    return "".join(parts)


def _final_architecture(fact_pack: Dict[str, Any], narrative: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    final_state = fact_pack.get("final_state") or {}
    best = final_state.get("final_best") or {}
    body = []
    if narrative.get("final_architecture_summary"):
        body.append(f'<p class="narrative">{_he(narrative.get("final_architecture_summary"))}</p>')
    body.append(_kv_table([
        (_t(lang, "candidate"), best.get("candidate_id")),
        (_t(lang, "display_name"), best.get("display_name")),
        (_t(lang, "changed_from_baseline"), final_state.get("changed_from_baseline")),
        (_t(lang, "variant_path"), best.get("variant_path")),
    ]))
    body.append(_metrics_table(best.get("metrics"), caption=_t(lang, "final_metrics"), seed_eval=best.get("seed_eval"), lang=lang))
    return "".join(body)


def _appendix(fact_pack: Dict[str, Any], narrative: Dict[str, Any]) -> str:
    lang = _language(fact_pack)
    artifacts = []
    artifact_sources = list(fact_pack.get("artifacts") or [])
    diagnosis = fact_pack.get("baseline_diagnosis") if isinstance(fact_pack.get("baseline_diagnosis"), dict) else {}
    for ablation in diagnosis.get("ablations") or []:
        if isinstance(ablation, dict):
            artifact_sources.extend(list(ablation.get("artifacts") or []))
    for record in fact_pack.get("research_rounds") or []:
        if isinstance(record, dict):
            artifact_sources.extend(list(record.get("artifacts") or []))
    seen_paths = set()
    for item in artifact_sources:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        if not path or path in seen_paths or path.endswith(".parsed.json") or "/logs/api/" in path or "\\logs\\api\\" in path:
            continue
        seen_paths.add(path)
        artifacts.append({"name": item.get("name"), "path": path, "size_bytes": item.get("size_bytes")})
    parts = [f'<details><summary>{_e(_t(lang, "token_usage"))}</summary>{_token_usage(fact_pack)}</details>']
    if not artifacts:
        parts.append(f'<p class="muted">{_e(_t(lang, "no_artifacts"))}</p>')
        parts.append(f'<details><summary>{_e(_t(lang, "stage_timing"))}</summary>{_stage_timing(fact_pack)}</details>')
        return "".join(parts)
    rows = "".join(
        "<tr>"
        f"<td>{_cell(item.get('name') or Path(str(item.get('path') or '')).name)}</td>"
        f"<td>{_cell(item.get('path'))}</td>"
        f"<td class=\"metric-value\">{_e(item.get('size_bytes'))}</td>"
        "</tr>"
        for item in artifacts
    )
    parts.append(
        f'<table><thead><tr><th>{_e(_t(lang, "name"))}</th><th>{_e(_t(lang, "path"))}</th><th>{_e(_t(lang, "size"))}</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>"
    )
    parts.append(f'<details><summary>{_e(_t(lang, "stage_timing"))}</summary>{_stage_timing(fact_pack)}</details>')
    return "".join(parts)


def render_review_html(*, fact_pack: Dict[str, Any], narrative: Dict[str, Any]) -> str:
    view_model = build_report_view_model(fact_pack=fact_pack, narrative=narrative)
    fact_pack = dict(view_model["fact_pack"])
    narrative = dict(view_model["narrative"])
    lang = _language(fact_pack)
    task = fact_pack.get("task") or {}
    title = _display_title(lang=lang, task=task, narrative=narrative)
    html_lang = "en" if lang == "en" else "zh-CN"
    css = """
    :root {
      color-scheme: light;
      --ink: #18202a;
      --muted: #667085;
      --line: #d9dee7;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --soft: #f9fafb;
      --accent: #175cd3;
      --accent-soft: #eff4ff;
      --good: #067647;
      --warn: #b54708;
      --bad: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.55;
    }
    header { padding: 28px 36px 20px; background: #ffffff; border-bottom: 1px solid var(--line); }
    main { max-width: 1120px; margin: 0 auto; padding: 24px 24px 56px; }
    h1 { margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }
    h2 { margin: 0 0 14px; font-size: 21px; letter-spacing: 0; }
    h3 { margin: 0 0 10px; font-size: 16px; letter-spacing: 0; }
    h4 { margin: 0 0 10px; font-size: 14px; letter-spacing: 0; color: #25443f; }
    .meta { color: var(--muted); display: flex; gap: 16px; flex-wrap: wrap; font-size: 14px; }
    .section { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 22px; margin: 18px 0; box-shadow: 0 8px 24px rgba(16, 24, 40, 0.035); }
    .monitor-cards { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin: 16px 0; }
    .monitor-card { border: 1px solid var(--line); border-radius: 16px; padding: 16px; background: linear-gradient(180deg, #ffffff, #fbfcff); min-height: 92px; }
    .card-label { color: var(--muted); font-size: 12px; text-transform: uppercase; }
    .card-value { margin-top: 8px; font-size: 17px; font-weight: 700; overflow-wrap: anywhere; }
    .narrative, .round-note { background: var(--soft); border-left: 3px solid var(--accent); padding: 10px 12px; border-radius: 8px; }
    .refs, .muted { color: var(--muted); font-size: 13px; }
    .timeline-row { display: grid; grid-template-columns: 58px minmax(160px, 0.55fr) minmax(0, 1fr); gap: 12px; align-items: center; padding: 11px 0; border-top: 1px solid var(--line); }
    .timeline-step { width: 36px; height: 36px; display: grid; place-items: center; border-radius: 999px; background: var(--accent-soft); color: var(--accent); font-weight: 800; }
    .timeline-title { font-weight: 750; }
    .timeline-outcome { color: var(--muted); overflow-wrap: anywhere; }
    .round-card { border: 1px solid var(--line); border-radius: 14px; padding: 16px; margin-top: 14px; background: #fff; }
    .audit-round-card { border: 1px solid var(--line); border-radius: 18px; padding: 18px; margin-top: 16px; background: #fff; }
    .audit-round-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; margin-bottom: 14px; }
    .round-card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin-bottom: 10px; }
    .round-id { color: var(--accent); font-weight: 800; font-size: 13px; margin-bottom: 2px; }
    .status-pill { border-radius: 999px; padding: 5px 10px; background: var(--accent-soft); color: var(--accent); font-weight: 800; font-size: 13px; white-space: nowrap; }
    .idea-box { border: 1px solid #d8e3f8; background: #f7faff; border-radius: 14px; padding: 14px; margin: 12px 0 16px; }
    .idea-text { margin-top: 5px; font-size: 15px; }
    .fact-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .fact { border: 1px solid var(--line); border-radius: 13px; padding: 12px; background: #fcfcfd; min-height: 78px; }
    .fact-label { color: var(--muted); font-size: 12px; font-weight: 700; letter-spacing: .01em; }
    .fact-value { margin-top: 6px; font-weight: 650; overflow-wrap: anywhere; }
    .opportunity-panel { border: 1px solid #cfe5df; background: #f8fcfb; border-radius: 8px; padding: 12px; margin: 12px 0; }
    table { width: 100%; border-collapse: collapse; margin: 10px 0 14px; font-size: 14px; }
    th, td { text-align: left; vertical-align: top; border-top: 1px solid var(--line); padding: 8px 10px; overflow-wrap: anywhere; }
    th { width: 220px; color: var(--muted); font-weight: 600; }
    .table-title { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; margin: 14px 0 4px; }
    .monitor-table th { width: auto; background: var(--soft); color: #344054; border-top: 0; }
    .monitor-table { border: 1px solid var(--line); border-radius: 12px; border-collapse: separate; border-spacing: 0; overflow: hidden; }
    .monitor-table tbody tr:nth-child(even) td { background: #fbfcfe; }
    .baseline-search-best td { color: #c62828; font-weight: 800; }
    .baseline-search-second td { color: #1565c0; text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 2px; }
    .round-audit-table th, .round-audit-table td { padding: 12px 12px; }
    .metrics-table { border: 1px solid var(--line); border-radius: 8px; border-collapse: separate; border-spacing: 0; overflow: hidden; }
    .metrics-table thead th { background: var(--accent-soft); color: #1d4ed8; border-top: 0; width: auto; }
    .metrics-table tbody tr:nth-child(even) td { background: #fbfcfe; }
    .compact-table th { width: auto; }
    .metric-value { font-variant-numeric: tabular-nums; }
    details { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; margin: 10px 0; background: #fbfcfe; }
    summary { cursor: pointer; font-weight: 650; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; line-height: 1.45; }
    @media (max-width: 760px) {
      header { padding: 22px 18px 14px; }
      main { padding: 14px; }
      .monitor-cards { grid-template-columns: 1fr; }
      .fact-grid { grid-template-columns: 1fr; }
      .timeline-row { grid-template-columns: 1fr; gap: 4px; }
      .round-card-head { display: block; }
      .audit-round-head { display: block; }
      th { width: 130px; }
    }
    """
    body = [
        "<!doctype html>",
        f'<html lang="{html_lang}">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_e(title)}</title>",
        f"<style>{css}</style>",
        "</head>",
        "<body>",
        "<header>",
        f"<h1>{_e(title)}</h1>",
        '<div class="meta">'
        f'<span>{_e(_t(lang, "task_id"))}: {_e(task.get("task_id"))}</span>'
        f'<span>{_e(_t(lang, "dataset"))}: {_e(_dataset_display(task.get("dataset")))}</span>'
        f'<span>{_e(_t(lang, "objective_metric"))}: {_e(task.get("objective_metric"))}</span>'
        f'<span>{_e(_t(lang, "build_mode"))}: {_e(_format_bool(task.get("build_mode"), lang=lang))}</span>'
        "</div>",
        "</header>",
        "<main>",
        _section(_t(lang, "run_overview"), _monitor_cards(fact_pack)),
        _section(
            _t(lang, "executive_summary"),
            _summary_blocks(list(narrative.get("executive_summary") or []), lang=lang)
            + _final_architecture(fact_pack, narrative),
        ),
        _section(_t(lang, "round_audit_table"), _round_monitor_table(fact_pack, narrative)),
        _section(_t(lang, "dataset_diagnosis"), _dataset_diagnosis(fact_pack)),
        _section(_t(lang, "baseline_search"), _baseline_search(fact_pack)),
        _section(_t(lang, "ablations"), _ablations(fact_pack, narrative)),
        _section(_t(lang, "round_audit_cards"), _round_audit_cards(fact_pack, narrative)),
        _section(
            _t(lang, "audit_appendix"),
            f'<p class="muted">{_e(_t(lang, "raw_artifact_note"))}</p>'
            f'<details><summary>{_e(_t(lang, "metric_dashboard"))}</summary>{_metric_dashboard(fact_pack)}</details>'
            f'<details><summary>{_e(_t(lang, "artifact_details"))}</summary>{_appendix(fact_pack, narrative)}</details>',
        ),
        "</main>",
        "</body></html>",
    ]
    return "\n".join(body)


def write_review_html(*, task_id: str, base_dir: str, fact_pack: Dict[str, Any], narrative: Dict[str, Any]) -> Path:
    runtime = runtime_root(base_dir)
    output_root = runtime.parent if runtime.name == ".evocast" else runtime
    path = output_root / "agent_reports" / f"{task_id}_review_report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_review_html(fact_pack=fact_pack, narrative=narrative), encoding="utf-8")
    return path
