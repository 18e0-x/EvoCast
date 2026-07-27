"""Small, deterministic localization helpers for user-facing EvoCast views.

Runtime records deliberately keep stable English codes.  This module is the
only place that turns those codes into human-facing labels.
"""

from __future__ import annotations

import re
from typing import Any


def normalize_language(value: Any) -> str:
    return "en" if str(value or "zh").strip().lower() in {"en", "english"} else "zh"


def is_english(value: Any) -> bool:
    return normalize_language(value) == "en"


_ZH = {
    "status": {
        "ok": "正常",
        "completed": "已完成",
        "success": "成功",
        "failed": "失败",
        "rejected": "已拒绝",
        "accepted": "已接受",
        "needs_seed_eval": "需要多种子评估",
        "needs_review": "需要复核",
        "marginal_no_seed_eval": "边际改善，未晋升",
        "idea_review_exhausted": "想法审查次数耗尽",
        "proposal_failed": "提案失败",
        "protocol_blocked": "协议阻断",
        "failed_ablation_round": "消融轮失败",
        "usable_evidence": "可用证据",
        "failed_evidence": "证据失败",
        "deterministic": "确定性生成",
        "skipped": "已跳过",
        "partial": "部分完成",
        "failed_non_blocking": "失败但不阻断",
        "reject": "拒绝",
        "accept": "接受",
        "proceed": "继续",
        "approve": "通过",
        "module_valid": "模块有效",
        "passed": "通过",
        "present": "已存在",
        "positive_single_seed": "单种子正信号",
        "negative_single_seed": "单种子负信号",
        "neutral_single_seed": "单种子中性信号",
        "build_mode": "构建模式",
    },
    "priority": {"high": "高", "medium": "中", "low": "低"},
    "frequency": {
        "hourly": "小时级",
        "daily": "日级",
        "weekly": "周级",
        "monthly": "月级",
        "quarterly": "季度级",
        "yearly": "年度级",
        "minutely": "分钟级",
        "15min": "15 分钟级",
    },
    "characteristic": {
        "Correlation": "变量相关性",
        "Transition": "状态转移性",
        "Shifting": "分布漂移性",
        "Seasonality": "季节性",
        "Trend": "趋势性",
        "Stationarity": "平稳性",
        "Short_term_jsd": "短期分布差异",
        "Long_term_jsd": "长期分布差异",
    },
    "claim": {
        "strong_seasonality": "强季节性",
        "seasonality_flag_present": "存在季节性标记",
        "strong_trend": "强趋势",
        "trend_flag_present": "存在趋势标记",
        "nonstationary_behavior": "非平稳行为",
        "distribution_shift": "分布漂移",
        "strong_cross_variable_correlation": "强跨变量相关性",
        "structured_state_transition": "结构化状态转移",
        "local_non_gaussianity": "局部非高斯性",
        "global_distribution_complexity": "全局分布复杂性",
    },
    "topic": {
        "frequency_or_periodic_modeling": "频率或周期建模",
        "trend_aware_modeling": "趋势感知建模",
        "nonstationary_adaptation": "非平稳自适应",
        "cross_variable_modeling": "跨变量建模",
        "state_sensitive_temporal_modeling": "状态敏感时序建模",
        "local_robustness": "局部鲁棒性",
        "distribution_robustness": "分布鲁棒性",
    },
    "mechanism": {
        "series_decomposition": "序列分解",
        "seasonal_trend_fusion": "季节—趋势融合",
        "moving_avg_decomposition": "移动平均分解",
        "multi_scale_patch_embedding": "多尺度补丁嵌入",
        "data_embedding_fusion": "数据嵌入融合",
        "masked_full_attention": "带掩码的全注意力",
        "decoder_cross_attention": "解码器交叉注意力",
    },
    "module_family": {
        "multi_scale_decomposition": "多尺度分解",
        "dynamic_routing_frequency": "动态频率路由",
        "state_adaptive_embedding": "状态自适应嵌入",
        "frequency_conditioning": "频率条件化",
        "patch_embedding": "补丁嵌入",
    },
    "exploration_scale": {
        "module_level": "模块级",
        "architecture_level": "架构级",
        "component_level": "组件级",
        "parameter_level": "参数级",
    },
    "source": {
        "dataset": "数据集",
        "user_intent": "用户意图",
        "ablation": "消融证据",
        "model_structure": "模型结构",
        "trial_history": "试验历史",
        "current_best": "当前最优候选",
    },
}


def localize_code(value: Any, category: str, language: Any = "zh") -> str:
    text = str(value or "")
    if is_english(language):
        return text
    return _ZH.get(category, {}).get(text, text)


def localize_status(value: Any, language: Any = "zh") -> str:
    return localize_code(value, "status", language)


def localize_characteristic(value: Any, language: Any = "zh") -> str:
    return localize_code(value, "characteristic", language)


def localize_source(value: Any, language: Any = "zh") -> str:
    return localize_code(value, "source", language)


def localize_evidence(value: Any, language: Any = "zh") -> str:
    text = str(value or "")
    if is_english(language):
        return text
    match = re.fullmatch(r"([A-Za-z_]+)=(-?(?:\d+(?:\.\d*)?|\.\d+))", text)
    if not match:
        return text
    return f"{localize_characteristic(match.group(1), language)}={match.group(2)}"


def chinese_text(value: Any) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(value or "")))


def locale_matches_text(value: Any, language: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if is_english(language):
        return not chinese_text(text)
    if not chinese_text(text):
        return False
    english_words = re.findall(r"[A-Za-z]{3,}", text)
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    # Keep model names and metric identifiers usable in Chinese prose, but
    # reject a mostly-English sentence with only a short Chinese wrapper.
    return not (len(english_words) >= 6 and len(english_words) * 2 > chinese_chars)
