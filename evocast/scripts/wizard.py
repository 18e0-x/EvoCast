"""EvoCast interactive launch wizard.

Usage:
  python -m evocast
  python -m evocast wizard
  python -m evocast.scripts.wizard --dataset ETTh1.csv --yes --dry-run

The wizard converts step-by-step answers into a validated TFB config and can
launch the v3 agent harness after confirmation.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import difflib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from evocast.configurator.config_compiler import compile_config
from evocast.configurator.config_validator import (
    _try_parse_time,
    get_wide_numeric_feature_columns,
    validate_intent,
)
from evocast.state.runtime.dashboard import build_dashboard
from evocast.policy.experiment_policy import (
    baseline_seed,
    baseline_diagnosis_policy,
    baseline_search_policy,
    normalize_budget,
)
from evocast.domain.knowledge_paths import runtime_root, task_knowledge_dir
from evocast.domain.metric_semantics import get_default_objective_metric, get_objective_metric_choices
from evocast.domain.model_key_utils import compact_model_key, resolve_model_key
from evocast.domain.task_identity import build_task_semantics_from_intent, generate_task_id
from evocast.research.model_registry import build_registry
from evocast.harness.api_client import ProviderClient, resolve_provider_config_path
from evocast.harness.ablation_round import run_ablation_round
from evocast.harness import rounds as round_records
from evocast.harness.rounds import round_progress
from evocast.harness.mechanism_ablation_diagnosis import generate_mechanism_ablation_plan
from evocast.services.task_lifecycle import initialize_task, write_compiled_config as persist_compiled_config
from evocast.services.task_discovery import discover_resumable_tasks as discover_tasks, format_resumable_task
from evocast.services.resume_service import ResumeDependencies, ResumeRequest, ResumeService
from evocast.evaluation.decision_kernel import EvaluationDecisionKernel
from evocast.services.baseline_flow import BaselineFlowRequest, BaselineFlowService
from evocast.services.research_loop import (
    ResearchContractRequest,
    ResearchContractService,
    ResearchLoopDependencies,
    ResearchLoopRequest,
    ResearchLoopService,
)
from evocast.services.task_finalization import (
    FinalizationDependencies,
    FinalizationRequest,
    TaskFinalizationService,
)
from evocast.scripts.run_agent_v3 import main as run_agent_v3_main
from evocast.state.runtime.resume_state import resume_summary
from evocast.state.runtime.store import sync_task_stage, record_runtime_event
from evocast.state.runtime.store import load_runtime_state
from evocast.state.domain_store import domain_state_path, load_build_contract, load_task_config
from evocast.state.runtime.terminal_ui import TerminalUI
from evocast.reports.review_report import generate_review_report
from evocast.reports.browser_open import open_html_report
from evocast.tools.model_structure import analyze_model_structure
from evocast.tools.tfb_ablation import (
    persist_ablation_plan,
    read_ablation_results,
)
from evocast.tools.tfb_seed_eval import run_seed_eval


def _configure_windows_safe_io() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("CONDA_NO_PLUGINS", "true")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


DEFAULT_HORIZONS = [96]
DEFAULT_SEQ_LEN = 96
DEFAULT_OBJECTIVE = ""
DEFAULT_BUDGET = "unified"
DEFAULT_TASK_MODE = "MM"
DEFAULT_STRATEGY = "rolling_forecast"
DEFAULT_API_CONFIG = "providers/deepseek.yaml"
LANGUAGE_CHOICES = ["zh", "en"]
LANGUAGE_DESCRIPTIONS = {
    "zh": "中文向导与中文报告",
    "en": "English wizard and English report",
}


def normalize_language(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"en", "english"}:
        return "en"
    if text in {"zh", "cn", "chinese", "中文"}:
        return "zh"
    raise ValueError("--language only supports zh or en.")


def is_english(intent_or_lang: Any) -> bool:
    if isinstance(intent_or_lang, dict):
        return normalize_language(str(intent_or_lang.get("language") or "zh")) == "en"
    return normalize_language(str(intent_or_lang or "zh")) == "en"
DEFAULT_BUILD_MODE = True
DEFAULT_MAX_ROUNDS = 3
AGENT_ABLATION_CHOICES = ["none", "A3", "A4", "A5"]
API_CONFIG_CHOICES = [
    "providers/deepseek.yaml",
    "providers/deepseek_test.yaml",
    "providers/openai.yaml",
    "providers/minimax.yaml",
]
TIME_COLUMN_DESCRIPTIONS = {
    "date": "时间戳列，表示每条观测发生的时间。当前 TFB 数据通常推荐用它作为时间列。",
    "time": "时间戳列，表示每条观测发生的时间。",
    "timestamp": "时间戳列，表示每条观测发生的时间。",
    "datetime": "时间戳列，表示每条观测发生的时间。",
    "ds": "时间戳列，常见于 Prophet 风格数据。",
    "data": "数值列。在长表格式数据中，它保存某个变量在某个时间点的观测值，不推荐作为时间列。",
    "cols": "变量名列。在长表格式数据中，它保存变量名称，例如 HUFL、HULL、OT，不推荐作为时间列。",
}
TIME_COLUMN_DESCRIPTIONS_EN = {
    "date": "Timestamp column for each observation. Current TFB forecasting datasets usually use it as the time column.",
    "time": "Timestamp column for each observation.",
    "timestamp": "Timestamp column for each observation.",
    "datetime": "Timestamp column for each observation.",
    "ds": "Timestamp column commonly used by Prophet-style datasets.",
    "data": "Numeric observation column in long-format data. It is not recommended as the time column.",
    "cols": "Variable-name column in long-format data, such as HUFL, HULL, or OT. It is not recommended as the time column.",
}
OBJECTIVE_DESCRIPTIONS = {
    "mse": "原尺度均方误差，越低越好。数值受变量量纲影响，不应直接和归一化 benchmark MSE 混比。",
    "mse_norm": "归一化均方误差，越低越好。TFB 官方 rolling forecast 的多变量报告默认使用归一化指标。",
    "mae": "平均绝对误差，越低越好，对异常值比 mse 更不敏感。",
    "mae_norm": "归一化平均绝对误差，越低越好。",
    "rmse": "均方根误差，越低越好，量纲接近原始数据。",
    "rmse_norm": "归一化均方根误差，越低越好。",
    "mape": "平均绝对百分比误差，越低越好；数据接近 0 时可能不稳定。",
    "mape_norm": "归一化平均绝对百分比误差，越低越好。",
    "smape": "对称百分比误差，越低越好。",
    "smape_norm": "归一化对称百分比误差，越低越好。",
    "mase": "平均绝对缩放误差，越低越好；依赖 seasonality。",
    "mase_norm": "归一化平均绝对缩放误差，越低越好；依赖 seasonality。",
    "wape": "加权绝对百分比误差，越低越好。",
    "wape_norm": "归一化加权绝对百分比误差，越低越好。",
    "msmape": "修正 smape，越低越好。",
    "msmape_norm": "归一化修正 smape，越低越好。",
}
OBJECTIVE_DESCRIPTIONS_EN = {
    "mse": "Original-scale mean squared error. Lower is better. Values depend on variable scale.",
    "mse_norm": "Normalized mean squared error. Lower is better. This is the default normalized metric for TFB multivariate rolling forecast reports.",
    "mae": "Mean absolute error. Lower is better and less sensitive to outliers than mse.",
    "mae_norm": "Normalized mean absolute error. Lower is better.",
    "rmse": "Root mean squared error. Lower is better and its scale is close to the original data.",
    "rmse_norm": "Normalized root mean squared error. Lower is better.",
    "mape": "Mean absolute percentage error. Lower is better and can be unstable near zero.",
    "mape_norm": "Normalized mean absolute percentage error. Lower is better.",
    "smape": "Symmetric percentage error. Lower is better.",
    "smape_norm": "Normalized symmetric percentage error. Lower is better.",
    "mase": "Mean absolute scaled error. Lower is better and depends on seasonality.",
    "mase_norm": "Normalized mean absolute scaled error. Lower is better and depends on seasonality.",
    "wape": "Weighted absolute percentage error. Lower is better.",
    "wape_norm": "Normalized weighted absolute percentage error. Lower is better.",
    "msmape": "Modified smape. Lower is better.",
    "msmape_norm": "Normalized modified smape. Lower is better.",
}
METRIC_DIRECTION_DESCRIPTIONS = {
    "lower_is_better": "指标越低越好。mse、mae、rmse 等误差指标推荐使用它。",
    "higher_is_better": "指标越高越好。只有准确率、得分这类指标才使用它。",
}
METRIC_DIRECTION_DESCRIPTIONS_EN = {
    "lower_is_better": "Lower metric values are better. Recommended for error metrics such as mse, mae, and rmse.",
    "higher_is_better": "Higher metric values are better. Use only for score-like metrics.",
}
API_CONFIG_DESCRIPTIONS = {
    "providers/deepseek.yaml": "生产配置：关键代码与规划阶段优先使用更强模型，适合正式实验。",
    "providers/deepseek_test.yaml": "测试配置：保留完整架构分工，但整体更省钱，适合联调与冒烟。",
    "providers/openai.yaml": "OpenAI/GPT 配置：按 Chat Completions 兼容协议接入，适合使用 GPT 模型。",
    "providers/minimax.yaml": "MiniMax 配置：按兼容 Chat Completions 协议接入，并开启 reasoning_split。",
}
API_CONFIG_DESCRIPTIONS_EN = {
    "providers/deepseek.yaml": "Production API config. Uses stronger models for key code and planning stages.",
    "providers/deepseek_test.yaml": "Test API config. Keeps the same agent roles for integration and smoke runs.",
    "providers/openai.yaml": "OpenAI/GPT config through the Chat Completions-compatible protocol.",
    "providers/minimax.yaml": "MiniMax config through the Chat Completions-compatible protocol with reasoning_split enabled.",
}
BUILD_MODE_DESCRIPTION = "构建期极速训练链路：baseline 自动寻优仍使用正式候选池、排序和选择逻辑；每个候选及后续实验强制使用 smoke 训练/测试预算（典型 1epoch/1batch/batch_size=1）。"
BUILD_MODE_DESCRIPTION_EN = "Build-mode keeps the formal baseline candidate pool, ranking, and selection path, while each baseline and later experiment uses smoke-scale training/testing (typically 1 epoch, 1 batch, batch_size=1)."
BASELINE_STRATEGY_CHOICES = ["auto", "manual"]
BASELINE_STRATEGY_DESCRIPTIONS = {
    "auto": "启动 agent 前自动运行 baseline tournament，并把最佳 baseline 写入 runtime_state。",
    "manual": "启动 agent 前只运行你指定的 baseline 模型。",
}
BASELINE_STRATEGY_DESCRIPTIONS_EN = {
    "auto": "Run a baseline tournament before launching the agent and write the best baseline into runtime_state.",
    "manual": "Run only the baseline model or models you specify before launching the agent.",
}
DATASET_DIAGNOSIS_MODE_CHOICES = ["required", "reuse", "skip"]
DATASET_DIAGNOSIS_INTERACTIVE_CHOICES = ["required", "skip"]
DATASET_DIAGNOSIS_MODE_DESCRIPTIONS = {
    "required": "正常分析：有旧结果就复用，没有就先分析数据集；适合正式运行。",
    "reuse": "高级模式：只使用已有数据集分析结果；没有就报错，不自动重跑。",
    "skip": "跳过分析，直接测试 research 流程；系统会记录这次没有做数据集分析。",
}
DATASET_DIAGNOSIS_MODE_DESCRIPTIONS_EN = {
    "required": "Normal: reuse an existing analysis when available; otherwise analyze the dataset before research.",
    "reuse": "Advanced: only use an existing dataset analysis; fail if it is missing.",
    "skip": "Skip analysis and go straight to testing the research flow; the run records that dataset analysis was not performed.",
}
MODE_OPTIONS = ["MM", "MS", "SS"]
MODE_DESCRIPTIONS = {
    "MM": "多变量输入，预测所有变量。符合 TFB 官方 rolling forecast 对 ETTh1 等多变量数据的默认约定。",
    "MS": "多变量输入，只预测一个目标变量，例如显式指定 OT。",
    "SS": "单变量输入，只预测单个目标变量。",
}
MODE_DESCRIPTIONS_EN = {
    "MM": "Multivariate input, predict all variables. This matches the default TFB rolling forecast setup for datasets such as ETTh1.",
    "MS": "Multivariate input, predict one target variable such as OT.",
    "SS": "Univariate input, predict one target variable.",
}


class WizardBack(Exception):
    """Raised when the interactive wizard should return to the previous step."""


def _back_hint(lang: str) -> str:
    return "Press Esc to go back." if is_english(lang) else "按 Esc 返回上一步。"


def _read_line(prompt: str, *, lang: str = "zh", allow_back: bool = True) -> str:
    """Read one console line; on Windows, Esc immediately returns to the previous step."""
    if not allow_back:
        return input(prompt)
    if os.name != "nt" or not sys.stdin.isatty():
        value = input(prompt)
        if value.strip().lower() in {":back", "/back", "back"}:
            raise WizardBack()
        return value
    import msvcrt

    print(prompt, end="", flush=True)
    chars: List[str] = []
    while True:
        ch = msvcrt.getwch()
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x1b":
            print()
            raise WizardBack()
        if ch in ("\r", "\n"):
            print()
            return "".join(chars)
        if ch in ("\b", "\x7f"):
            if chars:
                chars.pop()
                print("\b \b", end="", flush=True)
            continue
        if ch in ("\x00", "\xe0"):
            msvcrt.getwch()
            continue
        chars.append(ch)
        print(ch, end="", flush=True)


def _interactive_back_message(lang: str) -> str:
    return "[Wizard] Back to previous step." if is_english(lang) else "[向导] 已返回上一步。"


def descriptions_for(name: str, lang: str) -> Dict[str, str]:
    english = is_english(lang)
    if name == "time_column":
        return TIME_COLUMN_DESCRIPTIONS_EN if english else TIME_COLUMN_DESCRIPTIONS
    if name == "objective":
        return OBJECTIVE_DESCRIPTIONS_EN if english else OBJECTIVE_DESCRIPTIONS
    if name == "metric_direction":
        return METRIC_DIRECTION_DESCRIPTIONS_EN if english else METRIC_DIRECTION_DESCRIPTIONS
    if name == "api_config":
        return API_CONFIG_DESCRIPTIONS_EN if english else API_CONFIG_DESCRIPTIONS
    if name == "baseline_strategy":
        return BASELINE_STRATEGY_DESCRIPTIONS_EN if english else BASELINE_STRATEGY_DESCRIPTIONS
    if name == "dataset_diagnosis_mode":
        return DATASET_DIAGNOSIS_MODE_DESCRIPTIONS_EN if english else DATASET_DIAGNOSIS_MODE_DESCRIPTIONS
    if name == "mode":
        return MODE_DESCRIPTIONS_EN if english else MODE_DESCRIPTIONS
    return {}


class ChineseArgumentParser(argparse.ArgumentParser):
    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "用法:")

    def format_help(self) -> str:
        text = super().format_help()
        return (
            text.replace("usage:", "用法:")
            .replace("options:", "选项:")
            .replace("show this help message and exit", "显示帮助信息并退出")
        )

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: 参数错误：{message}\n")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def agent_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def runtime_dir() -> Path:
    return runtime_root(project_root())


def discover_datasets(root: Optional[Path] = None) -> List[Path]:
    root = root or project_root()
    dataset_root = root / "dataset"
    if not dataset_root.exists():
        return []

    datasets: List[Path] = []
    for path in dataset_root.rglob("*.csv"):
        if path.name.upper() == "FORECAST_META.CSV":
            continue
        datasets.append(path)

    return sorted(datasets, key=lambda p: (0 if "forecasting" in p.parts else 1, str(p).lower()))


def resolve_dataset(value: str, root: Optional[Path] = None) -> Path:
    root = root or project_root()
    raw = Path(value)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([
            root / raw,
            root / "dataset" / "forecasting" / value,
            root / "dataset" / value,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Return the most helpful unresolved path for validation messages.
    return candidates[0].resolve() if candidates else raw.resolve()


def relative_to_root(path: Path, root: Optional[Path] = None) -> str:
    root = root or project_root()
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def read_sample_rows(path: Path, max_rows: int = 50) -> List[List[str]]:
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            rows.append(row)
    return rows


def is_long_format_header(columns: Iterable[str]) -> bool:
    lowered = [c.lower().strip() for c in columns]
    return "data" in lowered and "cols" in lowered and ("date" in lowered or "time" in lowered)


def read_long_format_variables(path: Path, columns: List[str], max_rows: int = 200000) -> List[str]:
    lowered = [c.lower().strip() for c in columns]
    if "cols" not in lowered:
        return []

    cols_idx = lowered.index("cols")
    seen: Dict[str, bool] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            if len(row) <= cols_idx:
                continue
            value = row[cols_idx].strip()
            if value:
                seen.setdefault(value, True)
    return list(seen.keys())


def build_column_descriptions(columns: List[str], *, lang: str = "zh") -> Dict[str, str]:
    descriptions: Dict[str, str] = {}
    long_format = is_long_format_header(columns)
    time_descriptions = descriptions_for("time_column", lang)
    english = is_english(lang)
    for col in columns:
        key = col.lower().strip()
        if key in time_descriptions:
            descriptions[col] = time_descriptions[key]
        elif long_format:
            descriptions[col] = "CSV field. Choose according to data semantics; the time column is usually a different field." if english else "CSV 表中的字段。请根据数据语义选择；时间列通常不是这个字段。"
        else:
            descriptions[col] = "Data variable column that can be used as the target in MS/SS mode." if english else "数据变量列，可作为 MS/SS 模式的目标列。"
    return descriptions


def build_target_descriptions(targets: List[str], long_format: bool, *, lang: str = "zh") -> Dict[str, str]:
    descriptions: Dict[str, str] = {}
    english = is_english(lang)
    for target in targets:
        if long_format:
            descriptions[target] = "Variable name from the CSV cols column. Selecting it means predicting only this variable." if english else "变量名，来自 CSV 的 cols 列。选择后表示只预测这个变量。"
        else:
            descriptions[target] = "Variable column in the CSV. Selecting it means predicting this target column." if english else "CSV 中的变量列。选择后表示只预测这个目标列。"
    return descriptions


def get_objective_choices_for_task_mode(task_mode: str) -> List[str]:
    return list(get_objective_metric_choices(task_mode))


def choose_described_option(
    prompt: str,
    options: List[str],
    descriptions: Dict[str, str],
    default_index: int = 0,
    *,
    lang: str = "zh",
) -> str:
    english = is_english(lang)
    if not options:
        raise ValueError(f"{prompt} has no available options." if english else f"{prompt} 没有可选项。")
    if len(options) == 1:
        print(f"{prompt}: {options[0]} (only option, selected automatically)" if english else f"{prompt}：{options[0]}（唯一可选，已自动选择）")
        return options[0]
    print("Meanings:" if english else "含义：")
    for idx, option in enumerate(options, start=1):
        desc = descriptions.get(option, "")
        mark = " (recommended)" if english and idx - 1 == default_index else "（推荐）" if idx - 1 == default_index else ""
        suffix = f": {desc}" if english and desc else f"：{desc}" if desc else ""
        print(f"  - {idx}. {option}{mark}{suffix}")
    while True:
        raw = _read_line("Enter number: " if english else "请输入序号: ", lang=lang).strip()
        if not raw:
            return options[default_index]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("Enter a valid number." if english else "请输入有效序号。")


def format_validation_report_zh(report) -> str:
    def _warning_zh(message: str) -> str:
        text = str(message or "")
        if "uses ts_benchmark normalized error semantics" in text:
            metric = "归一化指标"
            marker = "objective_metric '"
            if marker in text:
                try:
                    metric = text.split(marker, 1)[1].split("'", 1)[0]
                except Exception:
                    metric = "归一化指标"
            return (
                f"objective_metric '{metric}' 使用 ts_benchmark 的归一化误差语义："
                "误差会按训练集标准差缩放。对 MS/SS 任务，这可能比原始 mse/mae 小很多。"
                "如果需要直接解释原始量纲误差，建议使用 'mse' 或 'mae'。"
            )
        if text.startswith("Horizon ") and "forecast span" in text:
            rendered = (
                text.replace("Horizon", "预测步长")
                .replace(" on frequency ", " 在频率 ")
                .replace(" means ", " 下表示 ")
                .replace(" of forecast span, not that many days. Confirm this matches your intended evaluation window.", " 的预测跨度，不是同等数量的天数。请确认这符合你的评估窗口预期。")
                .replace(" days", " 天")
                .replace(" day", " 天")
                .replace("'hourly'", "小时级")
                .replace("'daily'", "天级")
                .replace("'weekly'", "周级")
                .replace("'monthly'", "月级")
            )
            return (
                rendered
            )
        return text

    freq_zh = {
        "hourly": "小时级",
        "daily": "天级",
        "weekly": "周级",
        "monthly": "月级",
        "quarterly": "季度级",
        "yearly": "年级",
        "other": "其他",
    }
    lines = [
        "=" * 60,
        "配置校验报告",
        "=" * 60,
        f"校验结果：{'通过' if report.valid else '未通过'}",
        f"数据集：{report.intent.get('dataset_path', '?')}",
        f"CSV 列名：{report.column_names}",
        f"数据行数：{report.row_count}",
        f"检测到的频率：{freq_zh.get(report.detected_frequency, report.detected_frequency or '未知')}",
        (
            f"频率置信度：{report.detected_frequency_confidence:.2f}"
            if report.detected_frequency_confidence is not None
            else "频率置信度：未知"
        ),
        f"频率识别方法：{report.detected_frequency_method or '未知'}",
        f"时间范围：{report.time_min or '?'} -> {report.time_max or '?'}",
        f"推荐输入窗口长度：{report.suggested_input_chunk_length}",
        "",
    ]
    if report.errors:
        lines.append("错误：")
        for err in report.errors:
            lines.append(f"  - {err}")
        lines.append("")
    if report.warnings:
        lines.append("提醒：")
        for warning in report.warnings:
            lines.append(f"  - {_warning_zh(warning)}")
        lines.append("")
    if report.valid and not report.warnings:
        lines.append("全部检查通过。")
    return "\n".join(lines)


def format_validation_report_en(report) -> str:
    lines = [
        "=" * 60,
        "Configuration Validation Report",
        "=" * 60,
        f"Validation: {'passed' if report.valid else 'failed'}",
        f"Dataset: {report.intent.get('dataset_path', '?')}",
        f"CSV columns: {report.column_names}",
        f"Rows: {report.row_count}",
        f"Detected frequency: {report.detected_frequency or 'unknown'}",
        (
            f"Frequency confidence: {report.detected_frequency_confidence:.2f}"
            if report.detected_frequency_confidence is not None
            else "Frequency confidence: unknown"
        ),
        f"Frequency method: {report.detected_frequency_method or 'unknown'}",
        f"Time range: {report.time_min or '?'} -> {report.time_max or '?'}",
        f"Suggested input window: {report.suggested_input_chunk_length}",
        "",
    ]
    if report.errors:
        lines.append("Errors:")
        for err in report.errors:
            lines.append(f"  - {err}")
        lines.append("")
    if report.warnings:
        lines.append("Warnings:")
        for warning in report.warnings:
            lines.append(f"  - {warning}")
        lines.append("")
    if report.valid and not report.warnings:
        lines.append("All checks passed.")
    return "\n".join(lines)


def format_validation_report(report, language: str) -> str:
    return format_validation_report_en(report) if is_english(language) else format_validation_report_zh(report)


def infer_time_col(columns: Iterable[str]) -> str:
    cols = list(columns)
    lowered = {c.lower().strip(): c for c in cols}
    for name in ("date", "time", "timestamp", "datetime", "ds"):
        if name in lowered:
            return lowered[name]
    return cols[0] if cols else "date"


def infer_target_col(columns: Iterable[str], time_col: str) -> str:
    candidates = [c for c in columns if c != time_col]
    if not candidates:
        return ""
    for preferred in ("OT", "target", "value", "data", "y"):
        for col in candidates:
            if col.lower() == preferred.lower():
                return col
    return candidates[-1]


def parse_horizons(value: str | List[int] | None) -> List[int]:
    if value is None or value == "":
        return list(DEFAULT_HORIZONS)
    if isinstance(value, list):
        return [int(v) for v in value]
    parts = [p.strip() for p in re.split(r"[, ]+", value) if p.strip()]
    if not all(p.isdigit() for p in parts):
        raise ValueError("预测步长必须是正整数，多个值用逗号分隔，例如：96 或 96,192。")
    horizons = [int(p) for p in parts]
    if any(h <= 0 for h in horizons):
        raise ValueError("预测步长必须是正整数，例如：96 或 96,192。")
    return horizons or list(DEFAULT_HORIZONS)


def prompt_text(prompt: str, default: str = "", *, lang: str = "zh") -> str:
    suffix = f" [{default}]" if default else ""
    value = _read_line(f"{prompt}{suffix}: ", lang=lang).strip()
    return value or default


def format_choices(options: Iterable[str], max_items: int = 20, *, lang: str = "zh") -> str:
    items = list(options)
    visible = items[:max_items]
    suffix = "" if len(items) <= max_items else (f", ... ({len(items)} total)" if is_english(lang) else f", ...（共 {len(items)} 个）")
    return ", ".join(visible) + suffix


def _choice_help(
    options: List[str],
    descriptions: Optional[Dict[str, str]] = None,
    recommended: str = "",
    *,
    lang: str = "zh",
) -> str:
    english = is_english(lang)
    descriptions = descriptions or {}
    lines = [f"Available choices: {format_choices(options, lang=lang)}" if english else f"可选范围：{format_choices(options, lang=lang)}"]
    described = [opt for opt in options if opt in descriptions]
    if described:
        lines.append("Meanings:" if english else "含义：")
        for opt in described:
            mark = " (recommended)" if english and recommended and opt == recommended else "（推荐）" if recommended and opt == recommended else ""
            sep = ": " if english else "："
            lines.append(f"  - {opt}{mark}{sep}{descriptions[opt]}")
    elif recommended:
        lines.append(f"Recommended: {recommended}" if english else f"推荐：{recommended}")
    if recommended and described and recommended not in described:
        lines.append(f"Recommended: {recommended}" if english else f"推荐：{recommended}")
    return "\n".join(lines)


def require_choice(
    value: str,
    options: List[str],
    label: str,
    descriptions: Optional[Dict[str, str]] = None,
    recommended: str = "",
    *,
    lang: str = "zh",
) -> str:
    if value in options:
        return value
    english = is_english(lang)
    raise ValueError(
        (f"{label} is invalid: {value}\n" if english else f"{label} 输入无效：{value}\n")
        + f"{_choice_help(options, descriptions=descriptions, recommended=recommended, lang=lang)}"
    )


def prompt_choice_text(
    prompt: str,
    options: List[str],
    default: str = "",
    descriptions: Optional[Dict[str, str]] = None,
    *,
    lang: str = "zh",
) -> str:
    english = is_english(lang)
    if not options:
        raise ValueError(f"{prompt} has no available choices. Check the dataset first." if english else f"{prompt} 没有可选项，请先检查数据集。")
    if len(options) == 1:
        print(f"{prompt}: {options[0]} (only option, selected automatically)" if english else f"{prompt}：{options[0]}（唯一可选，已自动选择）")
        return options[0]
    default = default if default in options else options[0]
    print(_choice_help(options, descriptions=descriptions, recommended=default, lang=lang))
    while True:
        value = prompt_text(prompt, default, lang=lang)
        if value in options:
            return value
        print(f"Invalid input: {value}" if english else f"输入无效：{value}")
        print(_choice_help(options, descriptions=descriptions, recommended=default, lang=lang))


def time_column_error(
    value: str,
    columns: List[str],
    sample_rows: List[List[str]],
    recommended: str,
    *,
    lang: str = "zh",
) -> str:
    english = is_english(lang)
    if value not in columns:
        return f"Time column does not exist: {value}" if english else f"时间列不存在：{value}"

    col_idx = columns.index(value)
    values = [row[col_idx].strip() for row in sample_rows if len(row) > col_idx and row[col_idx].strip()]
    parsed = [v for v in values if _try_parse_time(v)]
    if parsed:
        return ""

    sample = values[:3]
    lines = (
        [
            f"Invalid time column: {value}",
            f"Reason: sample values from this column cannot be parsed as timestamps: {sample or ['<empty>']}",
            f"Recommended: use {recommended} as the time column.",
        ]
        if english
        else [
            f"时间列选择无效：{value}",
            f"原因：这一列的样例值无法解析为时间戳：{sample or ['<空值>']}",
            f"推荐：请选择 {recommended} 作为时间列。",
        ]
    )
    if value.lower() == "data":
        lines.append("Note: data is a numeric observation column, for example 5.827, not a timestamp." if english else "说明：data 是数值观测值列，例如 5.827，不是日期时间。")
    elif value.lower() == "cols":
        lines.append("Note: cols is a variable-name column, for example HUFL or OT, not a timestamp." if english else "说明：cols 是变量名列，例如 HUFL、OT，不是日期时间。")
    return "\n".join(lines)


def require_time_column(
    value: str,
    columns: List[str],
    sample_rows: List[List[str]],
    label: str,
    descriptions: Optional[Dict[str, str]] = None,
    recommended: str = "",
    *,
    lang: str = "zh",
) -> str:
    value = require_choice(
        value, columns, label,
        descriptions=descriptions, recommended=recommended, lang=lang,
    )
    error = time_column_error(value, columns, sample_rows, recommended or infer_time_col(columns), lang=lang)
    if error:
        raise ValueError((f"{label} is invalid: {value}\n{error}" if is_english(lang) else f"{label} 输入无效：{value}\n{error}"))
    return value


def prompt_time_column(
    columns: List[str],
    sample_rows: List[List[str]],
    default: str,
    descriptions: Optional[Dict[str, str]] = None,
    *,
    lang: str = "zh",
) -> str:
    recommended = default if default in columns else infer_time_col(columns)
    while True:
        value = prompt_choice_text(
            "Select time column" if is_english(lang) else "请选择时间列",
            columns,
            recommended,
            descriptions=descriptions,
            lang=lang,
        )
        error = time_column_error(value, columns, sample_rows, recommended, lang=lang)
        if not error:
            return value
        print(error)


def parse_horizons_prompt(prompt: str, default: str, *, lang: str = "zh") -> List[int]:
    english = is_english(lang)
    while True:
        raw = prompt_text(prompt, default, lang=lang)
        try:
            return parse_horizons(raw)
        except ValueError as exc:
            print(f"Invalid input: {raw}" if english else f"输入无效：{raw}")
            print(str(exc))


def require_positive_int(value: int, label: str, *, lang: str = "zh") -> int:
    if value > 0:
        return value
    raise ValueError(f"{label} is invalid: {value}. Use a positive integer such as 1, 5, or 10." if is_english(lang) else f"{label} 输入无效：{value}。请选择正整数，例如：1、5、10。")


def require_non_negative_int(value: int, label: str, *, lang: str = "zh") -> int:
    if value >= 0:
        return value
    raise ValueError(f"{label} is invalid: {value}. Use 0 or a positive integer." if is_english(lang) else f"{label} 输入无效：{value}。请选择 0 或正整数。")


def _research_intent_from_args(args: argparse.Namespace) -> str:
    inline = str(getattr(args, "research_intent", "") or "").strip()
    path_text = str(getattr(args, "research_intent_file", "") or "").strip()
    if not path_text:
        return inline
    path = Path(path_text)
    if not path.is_absolute():
        path = project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"--research-intent-file not found: {path}")
    file_intent = path.read_text(encoding="utf-8").strip()
    if inline and file_intent and inline != file_intent:
        raise ValueError("--research-intent and --research-intent-file provide different values.")
    return inline or file_intent


def _review_report_enabled(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "suppress_review_report", False))


def _normalize_agent_ablation_arg(value: Any) -> str:
    text = str(value or "none").strip()
    if not text or text.lower() == "none":
        return "none"
    upper = text.upper()
    if upper in {"A3", "A4", "A5"}:
        return upper
    raise ValueError(f"Unsupported --agent-ablation: {value}. Use one of: {', '.join(AGENT_ABLATION_CHOICES)}")


def _build_contract_args(args: argparse.Namespace, task_cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    path_text = str(getattr(args, "build_contract", "") or (task_cfg or {}).get("build_contract_path") or "").strip()
    if not path_text:
        raise RuntimeError(
            "BuildContract is required to launch the Research path."
        )
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    if not path.is_file():
        raise FileNotFoundError(f"--build-contract not found: {path}")
    return ["--build-contract", str(path)]


def _ensure_research_build_contract(
    *,
    args: argparse.Namespace,
    task_id: str,
    intent: Dict[str, Any],
    objective_metric: str,
    api_config_name: str,
    diagnosis_summary: Optional[Dict[str, Any]] = None,
) -> Path:
    research_id = _research_contract_service().create(
        ResearchContractRequest(
            task_id=task_id,
            objective_metric=objective_metric,
            api_config=api_config_name,
            language="en" if is_english(intent) else "zh",
            idea_planner=str(
                getattr(args, "idea_planner", "research_program_hypothesis_competition")
                or "research_program_hypothesis_competition"
            ),
            diagnosis=dict((diagnosis_summary or {}).get("diagnosis") or {}),
            explicit_contract=str(getattr(args, "build_contract", "") or "").strip(),
        )
    )
    return Path(research_id)


def _research_contract_service() -> ResearchContractService:
    return ResearchContractService(
        base_dir=str(runtime_dir()),
        project_root=project_root(),
        progress=lambda message: print(message, flush=True),
    )


def _run_research_agent(task_id: str, api_config: str, research_id: str) -> Any:
    return run_agent_v3_main(
        ["--task-id", task_id, "--api-config", api_config, "--research-id", research_id]
    )


def _research_loop_service() -> ResearchLoopService:
    def create_contract(request: ResearchContractRequest) -> str:
        path = _ensure_research_build_contract(
            args=argparse.Namespace(
                build_contract=request.explicit_contract,
                idea_planner=request.idea_planner,
            ),
            task_id=request.task_id,
            intent={"language": request.language},
            objective_metric=request.objective_metric,
            api_config_name=request.api_config,
            diagnosis_summary={"diagnosis": dict(request.diagnosis or {})},
        )
        return str(path)

    return ResearchLoopService(
        base_dir=str(runtime_dir()),
        dependencies=ResearchLoopDependencies(
            round_progress=round_progress,
            create_contract=create_contract,
            run_agent=_run_research_agent,
        ),
    )


def _finalization_service() -> TaskFinalizationService:
    return TaskFinalizationService(
        base_dir=str(runtime_dir()),
        dependencies=FinalizationDependencies(
            sync_task_stage=sync_task_stage,
            record_runtime_event=record_runtime_event,
            build_dashboard=build_dashboard,
            generate_report=generate_review_report,
            open_report=open_html_report,
        ),
    )


def _baseline_flow_service() -> BaselineFlowService:
    return BaselineFlowService(
        base_dir=str(runtime_dir()),
        progress=lambda message: print(message, flush=True),
        analyze_model_structure_fn=analyze_model_structure,
        generate_mechanism_ablation_plan_fn=generate_mechanism_ablation_plan,
        persist_ablation_plan_fn=persist_ablation_plan,
        run_ablation_round_fn=run_ablation_round,
        run_seed_eval_fn=run_seed_eval,
        read_ablation_results_fn=read_ablation_results,
    )


def prompt_positive_int(prompt: str, default: int, *, lang: str = "zh") -> int:
    english = is_english(lang)
    while True:
        raw = prompt_text(prompt, str(default), lang=lang)
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print(f"Invalid input: {raw}" if english else f"输入无效：{raw}")
        print("Use a positive integer such as 1, 5, or 10." if english else "请选择正整数，例如：1、5、10。")


def prompt_non_negative_int(prompt: str, default: int, *, lang: str = "zh") -> int:
    english = is_english(lang)
    while True:
        raw = prompt_text(prompt, str(default), lang=lang)
        if raw.isdigit():
            return int(raw)
        print(f"Invalid input: {raw}" if english else f"输入无效：{raw}")
        print("Use 0 or a positive integer." if english else "请选择 0 或正整数。")


def prompt_yes_no(prompt: str, default: bool = False, *, lang: str = "zh") -> bool:
    english = is_english(lang)
    marker = "yes/no, default yes" if english and default else "yes/no, default no" if english else "是/否，默认是" if default else "是/否，默认否"
    while True:
        value = _read_line(f"{prompt} [{marker}]: ", lang=lang).strip().lower()
        if not value:
            return default
        if value in ("y", "yes", "是", "对", "true", "1"):
            return True
        if value in ("n", "no", "否", "不", "false", "0"):
            return False
        print("Enter yes or no." if english else "请输入 是 或 否。")


def discover_resumable_tasks(base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    return discover_tasks(base_dir or runtime_dir(), summary_loader=resume_summary)


def _format_resumable_task(summary: Dict[str, Any], *, lang: str = "zh") -> str:
    return format_resumable_task(summary, english=is_english(lang))


def maybe_resume_interactive(args: argparse.Namespace, *, lang: str = "zh") -> Optional[Dict[str, Any]]:
    """Ask whether to resume an existing task when launched interactively."""
    if args.yes or args.dry_run or args.configure_only:
        return None
    if args.dataset or args.start_run:
        return None

    tasks = discover_resumable_tasks(agent_dir())
    if not tasks:
        return None

    english = is_english(lang)
    print("Existing EvoCast tasks detected:" if english else "检测到已有 EvoCast 任务状态：")
    labels = [_format_resumable_task(t, lang=lang) for t in tasks[:10]]
    create_new_label = "Do not resume; create/configure a new task" if english else "不续跑，创建/配置新任务"
    labels.append(create_new_label)
    selected = choose_option("Resume from the previous interruption?" if english else "是否从上一次中断处继续？", labels, default_index=len(labels) - 1, lang=lang)
    if selected == create_new_label:
        return None

    idx = labels.index(selected)
    task_id = tasks[idx]["task_id"]
    task_cfg = load_task_config(str(runtime_dir()), task_id)

    print(f"[Wizard] Resuming task {task_id}." if english else f"[向导] 将从任务 {task_id} 的断点继续。")
    args.resume = True
    args.task_id = task_id
    return resume_existing_task(args)


def _load_existing_task_config(task_id: str) -> Dict[str, Any]:
    return _resume_service().load_task_config(task_id)


def _resume_service() -> ResumeService:
    base_dir = str(runtime_dir())
    loop_service = _research_loop_service()
    finalizer = _finalization_service()

    return ResumeService(ResumeDependencies(
        base_dir=base_dir, load_task_config=load_task_config, state_exists=lambda root, task: domain_state_path(root, task).is_file(),
        list_rounds=round_records.list_rounds, current_round=round_records.current_round,
        counts_as_research=round_records._round_counts_toward_research_budget, phase_build=round_records.PHASE_BUILD,
        gate_resolution_decisions=round_records.GATE_RESOLUTION_DECISIONS, load_contract=load_build_contract,
        record_seed_eval_failure=lambda **kwargs: EvaluationDecisionKernel(
            base_dir=str(kwargs["base_dir"]),
            task_id=str(kwargs["task_id"]),
        ).record_seed_failure(
            variant_path=kwargs.get("variant_path"),
            result=dict(kwargs.get("result") or {}),
        ),
        close_round=round_records.close_current_round,
        run_agent=_run_research_agent, round_progress=round_progress, run_research_loop=loop_service.run,
        load_diagnosis=lambda root, task: dict(load_runtime_state(root, task).baseline_diagnosis or {}),
        build_dashboard=lambda root, task: build_dashboard(root, task, persist=True),
        finalize_research=lambda task, progress, max_rounds, review: finalizer.finalize_research(
            FinalizationRequest(task_id=task, review_report=review, strict_report=True),
            progress=progress,
            max_rounds=max_rounds,
        ),
        record_agent_failure=lambda task, error, progress, review: finalizer.record_agent_failure(
            FinalizationRequest(task_id=task, review_report=review, strict_report=False),
            error=error,
            progress=progress,
        ),
    ))


def _close_interrupted_round_for_resume(task_id: str, record: Dict[str, Any]) -> None:
    _resume_service()._close_interrupted_round(task_id, record)


def _resume_open_round(task_id: str, task_config: Dict[str, Any], args: argparse.Namespace) -> bool:
    """Recover an open build round when its durable workspace permits it.

    Returns True only when an existing BuildContract was handed to the build
    orchestrator. Other interrupted rounds are closed with their evidence
    intact, because their live process state cannot be reconstructed safely.
    """
    return _resume_service().recover_open_round(task_id, task_config, str(args.api_config or DEFAULT_API_CONFIG))


def resume_existing_task(args: argparse.Namespace) -> Dict[str, Any]:
    """Continue a persisted formal research task without replaying its prelude."""
    task_id = str(getattr(args, "task_id", "") or "").strip()
    if not task_id:
        raise ValueError("--resume requires --task-id.")
    if getattr(args, "configure_only", False) or getattr(args, "dry_run", False):
        raise ValueError("--resume cannot be combined with --configure-only or --dry-run.")
    if any(
        (
            str(getattr(args, "dataset", "") or "").strip(),
            str(getattr(args, "time_col", "") or "").strip(),
            str(getattr(args, "target", "") or "").strip(),
            getattr(args, "seq_len", None) is not None,
        )
    ):
        raise ValueError(
            "--resume reads task semantics from the persisted task; "
            "do not pass --dataset, --time-col, --target, or --seq-len."
        )

    terminal_ui = TerminalUI(str(runtime_dir()), task_id).start()
    atexit.register(terminal_ui.stop)
    try:
        return _resume_service().resume(ResumeRequest(
            task_id=task_id,
            api_config=str(getattr(args, "api_config", "") or DEFAULT_API_CONFIG),
            language=str(getattr(args, "language", "") or "zh"),
            idea_planner=str(getattr(args, "idea_planner", "research_program_hypothesis_competition") or "research_program_hypothesis_competition"),
            agent_ablation=str(getattr(args, "agent_ablation", "none") or "none"),
            review_report=_review_report_enabled(args),
        ))
    finally:
        terminal_ui.stop()


def choose_option(prompt: str, options: List[str], default_index: int = 0, *, lang: str = "zh") -> str:
    english = is_english(lang)
    if not options:
        raise ValueError("No available options." if english else "没有可选项。")
    if len(options) == 1:
        print(f"{prompt} {options[0]} (only option, selected automatically)" if english else f"{prompt} {options[0]}（唯一可选，已自动选择）")
        return options[0]
    print(prompt)
    for idx, option in enumerate(options, start=1):
        default_marker = " (recommended)" if english and idx - 1 == default_index else "（推荐）" if idx - 1 == default_index else ""
        print(f"  {idx}. {option}{default_marker}")
    while True:
        raw = _read_line("Enter number: " if english else "请输入序号: ", lang=lang).strip()
        if not raw:
            return options[default_index]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("Enter a valid number." if english else "请输入有效序号。")


def registry_model_keys() -> List[str]:
    try:
        registry = build_registry(verify=False)
    except Exception:
        return []
    keys = sorted({str(spec.get("model_key") or "") for spec in registry if spec.get("model_key")})
    return keys


def registry_models_by_family() -> Dict[str, List[str]]:
    try:
        registry = build_registry(verify=False)
    except Exception:
        return {}
    grouped: Dict[str, List[str]] = {}
    for spec in registry:
        key = str(spec.get("model_key") or "").strip()
        if not key:
            continue
        family = str(spec.get("family") or "unknown").strip() or "unknown"
        grouped.setdefault(family, []).append(key)
    return {family: sorted(set(keys)) for family, keys in grouped.items()}


def format_registry_models_by_family(*, lang: str = "zh") -> str:
    grouped = registry_models_by_family()
    if not grouped:
        return ""
    try:
        preferred = [
            str(item)
            for item in list(baseline_search_policy(str(runtime_dir())).get("preferred_families") or [])
            if str(item).strip()
        ]
    except Exception:
        preferred = []
    ordered = [family for family in preferred if family in grouped]
    ordered.extend(family for family in sorted(grouped) if family not in set(ordered))
    total = sum(len(grouped[family]) for family in grouped)
    if is_english(lang):
        lines = [f"Available baseline models by family ({total} total):"]
        for family in ordered:
            lines.append(f"  {family} ({len(grouped[family])}): {', '.join(grouped[family])}")
    else:
        lines = [f"可选 baseline 模型（按 family 分组，共 {total} 个）："]
        for family in ordered:
            lines.append(f"  {family}（{len(grouped[family])}）：{', '.join(grouped[family])}")
    return "\n".join(lines)


def parse_model_list(value: str | List[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[, ]+", str(value))
    return [str(item).strip() for item in raw_items if str(item).strip()]


def normalize_manual_model_list(models: List[str]) -> List[str]:
    if not models:
        return []
    try:
        registry = build_registry(verify=False)
    except Exception:
        registry = []
    normalized: List[str] = []
    for model in models:
        resolved, _ = resolve_model_key(model, registry)
        value = resolved or str(model or "").strip()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _manual_model_suggestions(model: str, registry: List[Dict[str, Any]], *, limit: int = 3) -> List[str]:
    requested = compact_model_key(model)
    if not requested:
        return []
    by_compact: Dict[str, List[str]] = {}
    for spec in registry:
        key = str(spec.get("model_key") or "").strip()
        import_path = str(spec.get("import_path") or "").strip()
        aliases = [key, key.lower()]
        if import_path:
            aliases.extend([import_path, import_path.rsplit(".", 1)[-1]])
        for alias in aliases:
            compact = compact_model_key(alias)
            if compact:
                by_compact.setdefault(compact, []).append(key)
    suggestions: List[str] = []
    for compact in difflib.get_close_matches(requested, sorted(by_compact), n=limit * 2, cutoff=0.72):
        for key in by_compact.get(compact) or []:
            if key and key not in suggestions:
                suggestions.append(key)
            if len(suggestions) >= limit:
                return suggestions
    return suggestions


def validate_manual_model_list(models: List[str]) -> tuple[List[str], List[Dict[str, Any]]]:
    try:
        registry = build_registry(verify=False)
    except Exception:
        registry = []
    normalized: List[str] = []
    missing: List[Dict[str, Any]] = []
    for model in models:
        requested = str(model or "").strip()
        if not requested:
            continue
        resolved, resolution = resolve_model_key(requested, registry)
        if resolved:
            if resolved not in normalized:
                normalized.append(resolved)
            continue
        missing.append(
            {
                "requested": requested,
                "suggestions": _manual_model_suggestions(requested, registry),
                "resolution": resolution,
            }
        )
    return normalized, missing


def format_missing_manual_models(missing: List[Dict[str, Any]], *, lang: str = "zh") -> str:
    english = is_english(lang)
    parts: List[str] = []
    for item in missing:
        requested = str(item.get("requested") or "")
        suggestions = [str(value) for value in item.get("suggestions") or [] if str(value)]
        if suggestions:
            if english:
                parts.append(f"{requested} (did you mean: {', '.join(suggestions)}?)")
            else:
                parts.append(f"{requested}（你是不是想输入：{', '.join(suggestions)}？）")
        else:
            parts.append(requested)
    return "; ".join(parts)


def check_api_config_available(api_config_name: str, *, task_id: str = "wizard_api_preflight") -> tuple[bool, str]:
    try:
        client = ProviderClient(config_path=resolve_provider_config_path(api_config_name), task_id=task_id)
        client.call_json(
            stage="api_preflight",
            round_num=0,
            execution_label="wizard_api_preflight",
            messages=[
                {"role": "system", "content": "Return exactly one JSON object."},
                {"role": "user", "content": "{\"status\":\"ok\"}"},
            ],
            stream=False,
        )
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def choose_api_config(args: argparse.Namespace, *, lang: str) -> str:
    english = is_english(lang)
    if args.yes:
        api_config_name = require_choice(
            args.api_config,
            API_CONFIG_CHOICES,
            "--api-config",
            descriptions=descriptions_for("api_config", lang),
            recommended=DEFAULT_API_CONFIG,
            lang=lang,
        )
        ok, error = check_api_config_available(api_config_name)
        if not ok:
            raise ValueError(
                f"API config preflight failed for {api_config_name}: {error}"
                if english
                else f"API 配置预检失败：{api_config_name}，错误：{error}"
            )
        return api_config_name

    while True:
        api_config_name = choose_described_option(
            "Select API config:" if english else "请选择 API 配置：",
            API_CONFIG_CHOICES,
            descriptions_for("api_config", lang),
            default_index=API_CONFIG_CHOICES.index(args.api_config) if args.api_config in API_CONFIG_CHOICES else 0,
            lang=lang,
        )
        print(
            f"[Wizard] Checking API config: {api_config_name}"
            if english
            else f"[向导] 正在预检 API 配置：{api_config_name}"
        )
        ok, error = check_api_config_available(api_config_name)
        if ok:
            return api_config_name
        print(
            f"[Wizard] API config failed: {error}. Please choose another config."
            if english
            else f"[向导] API 配置不可用：{error}。请重新选择。"
        )


def prompt_manual_baselines(default: str = "", *, lang: str = "zh") -> List[str]:
    english = is_english(lang)
    grouped = format_registry_models_by_family(lang=lang)
    if grouped:
        print(grouped)
    while True:
        raw = prompt_text("Enter baseline model names, separated by commas" if english else "请输入 baseline 模型名，多个用逗号分隔", default)
        models = parse_model_list(raw)
        if models:
            normalized, missing = validate_manual_model_list(models)
            if not missing:
                return normalized
            print(
                "Unknown baseline model(s): " + format_missing_manual_models(missing, lang=lang) + ". Please enter again."
                if english
                else "未找到 baseline 模型：" + format_missing_manual_models(missing, lang=lang) + "。请重新输入。"
            )
            continue
        print("manual baseline requires at least one model name." if english else "manual baseline 至少需要一个模型名。")


def choose_dataset(datasets: List[Path], args: argparse.Namespace, *, lang: str = "zh") -> Path:
    english = is_english(lang)
    root = project_root()
    if args.dataset:
        dataset_path = resolve_dataset(args.dataset, root)
        if not dataset_path.exists():
            known = [relative_to_root(p, root) for p in datasets]
            raise FileNotFoundError(
                (
                    f"--dataset is invalid: {args.dataset}. File does not exist.\n"
                    f"Available datasets: {format_choices(known, lang=lang)}\n"
                    "Recommended first run: dataset/forecasting/ETTh1.csv."
                )
                if english
                else (
                    f"--dataset 输入无效：{args.dataset}。文件不存在。\n"
                    f"当前可选数据集：{format_choices(known, lang=lang)}\n"
                    "推荐：先选择 dataset/forecasting/ETTh1.csv 做流程验证。"
                )
            )
        return dataset_path

    if args.yes:
        if not datasets:
            raise FileNotFoundError("No CSV datasets found under dataset/." if english else "未在 dataset/ 目录下找到 CSV 数据集。")
        return datasets[0]

    labels = [relative_to_root(p, root) for p in datasets]
    if labels:
        manual_label = "Enter another path manually" if english else "手动输入其他路径"
        labels.append(manual_label)
        selected = choose_option("Select dataset:" if english else "请选择数据集：", labels, default_index=0, lang=lang)
        if selected != manual_label:
            return resolve_dataset(selected, root)

    while True:
        raw = prompt_text("Enter dataset CSV path" if english else "请输入数据集 CSV 路径", "dataset/forecasting/ETTh1.csv")
        dataset_path = resolve_dataset(raw, root)
        if dataset_path.exists():
            return dataset_path
        known = [relative_to_root(p, root) for p in datasets]
        print(f"Dataset does not exist: {raw}" if english else f"数据集不存在：{raw}")
        if known:
            print(f"Available datasets: {format_choices(known, lang=lang)}" if english else f"当前可选数据集：{format_choices(known, lang=lang)}")
            print("Recommended: dataset/forecasting/ETTh1.csv" if english else "推荐：dataset/forecasting/ETTh1.csv")


def build_intent_interactive(args: argparse.Namespace, *, lang: str) -> Dict[str, Any]:
    english = is_english(lang)
    root = project_root()
    answers: Dict[str, Any] = {}
    context: Dict[str, Any] = {}
    step = 0
    max_step = 11

    def back() -> None:
        nonlocal step
        if step <= 0:
            print("[Wizard] Already at the first configurable step." if english else "[向导] 已经在第一个可配置步骤。")
            return
        if step == 4 and str(answers.get("task_mode") or "") == "MM":
            step = 2
        elif step == 8 and str(answers.get("baseline_strategy") or "") != "manual":
            step = 6
        else:
            step -= 1
        print(_interactive_back_message(lang))

    while step <= max_step:
        try:
            if step == 0:
                datasets = discover_datasets(root)
                dataset_path = choose_dataset(datasets, args, lang=lang)
                columns = read_header(dataset_path)
                sample_rows = read_sample_rows(dataset_path)
                long_format = is_long_format_header(columns)
                column_descriptions = build_column_descriptions(columns, lang=lang)
                target_choices = read_long_format_variables(dataset_path, columns) if long_format else []
                if not target_choices:
                    numeric_target_choices, _ = get_wide_numeric_feature_columns(
                        str(dataset_path), infer_time_col(columns)
                    )
                    target_choices = numeric_target_choices or [c for c in columns if c != infer_time_col(columns)]
                target_descriptions = build_target_descriptions(target_choices, long_format, lang=lang)
                context.update(
                    {
                        "dataset_path": dataset_path,
                        "columns": columns,
                        "sample_rows": sample_rows,
                        "column_descriptions": column_descriptions,
                        "target_choices": target_choices,
                        "target_descriptions": target_descriptions,
                    }
                )
                answers["dataset_path"] = relative_to_root(dataset_path, root)
                step += 1
                continue

            if step == 1:
                columns = list(context["columns"])
                sample_rows = list(context["sample_rows"])
                time_default = args.time_col or infer_time_col(columns)
                time_col = prompt_time_column(
                    columns,
                    sample_rows,
                    time_default,
                    descriptions=dict(context.get("column_descriptions") or {}),
                    lang=lang,
                )
                answers["time_col"] = time_col
                step += 1
                continue

            if step == 2:
                mode_default = args.task_mode or DEFAULT_TASK_MODE
                if mode_default not in MODE_OPTIONS:
                    mode_default = DEFAULT_TASK_MODE
                task_mode = choose_option(
                    "Select forecasting mode:" if english else "请选择预测模式：",
                    [
                        "MM - multivariate input, predict all variables (TFB default)" if english else "MM - 多变量输入，预测所有变量（TFB 官方默认）",
                        "MS - multivariate input, predict one target variable" if english else "MS - 多变量输入，只预测一个目标变量",
                        "SS - univariate input, predict one target variable" if english else "SS - 单变量输入，只预测单个目标变量",
                    ],
                    default_index=["MM", "MS", "SS"].index(mode_default),
                    lang=lang,
                ).split(" ", 1)[0]
                answers["task_mode"] = task_mode
                if task_mode == "MM":
                    answers["target_columns"] = []
                    step = 4
                else:
                    step += 1
                continue

            if step == 3:
                target_choices = list(context["target_choices"])
                target_descriptions = dict(context.get("target_descriptions") or {})
                inferred_target = infer_target_col(target_choices, str(answers.get("time_col") or ""))
                target_default = args.target if args.target in target_choices else inferred_target
                default_index = target_choices.index(target_default) if target_default in target_choices else 0
                print("Select target variable:" if english else "请选择目标变量：")
                target_raw = choose_described_option(
                    "Select target variable" if english else "请选择目标变量",
                    target_choices,
                    descriptions=target_descriptions,
                    default_index=default_index,
                    lang=lang,
                )
                answers["target_columns"] = [target_raw] if target_raw else []
                step += 1
                continue

            if step == 4:
                try:
                    horizons_default = ",".join(str(h) for h in parse_horizons(args.horizons))
                except ValueError as exc:
                    raise ValueError((f"--horizons is invalid: {args.horizons}. {exc}" if english else f"--horizons 输入无效：{args.horizons}。{exc}")) from exc
                answers["horizons"] = parse_horizons_prompt("Enter forecast horizons" if english else "请输入预测步长", horizons_default, lang=lang)
                step += 1
                continue

            if step == 5:
                seq_len_default = require_positive_int(args.seq_len, "--seq-len", lang=lang) if args.seq_len is not None else DEFAULT_SEQ_LEN
                answers["input_chunk_length"] = prompt_positive_int("Enter input window length seq_len" if english else "请输入输入窗口长度 seq_len", seq_len_default, lang=lang)
                step += 1
                continue

            if step == 6:
                baseline_strategy = choose_described_option(
                    "Select baseline strategy" if english else "请选择 baseline 策略",
                    BASELINE_STRATEGY_CHOICES,
                    descriptions_for("baseline_strategy", lang),
                    default_index=BASELINE_STRATEGY_CHOICES.index(args.baseline_strategy)
                    if args.baseline_strategy in BASELINE_STRATEGY_CHOICES
                    else 1,
                    lang=lang,
                )
                answers["baseline_strategy"] = baseline_strategy
                answers["baseline_models"] = parse_model_list(args.baseline_models)
                step = 7 if baseline_strategy == "manual" else 8
                continue

            if step == 7:
                baseline_models = prompt_manual_baselines(args.baseline_models, lang=lang)
                answers["baseline_models"] = baseline_models
                step += 1
                continue

            if step == 8:
                answers["build_mode"] = prompt_yes_no("Enable build mode" if english else "是否开启 build mode（极速闭环验证模式）", default=bool(args.build_mode), lang=lang)
                step += 1
                continue

            if step == 9:
                answers["dataset_diagnosis_mode"] = choose_described_option(
                    "Select dataset diagnosis mode" if english else "请选择数据集诊断模式",
                    DATASET_DIAGNOSIS_INTERACTIVE_CHOICES,
                    descriptions_for("dataset_diagnosis_mode", lang),
                    default_index=DATASET_DIAGNOSIS_INTERACTIVE_CHOICES.index(args.dataset_diagnosis_mode)
                    if args.dataset_diagnosis_mode in DATASET_DIAGNOSIS_INTERACTIVE_CHOICES
                    else 0,
                    lang=lang,
                )
                step += 1
                continue

            if step == 10:
                answers["baseline_diagnosis_max_ablation_targets"] = prompt_non_negative_int(
                    "Enter maximum baseline diagnosis ablations (0 skips ablation)" if english else "请输入 baseline 诊断最大消融数量（0 表示跳过消融）",
                    args.max_ablation_targets,
                    lang=lang,
                )
                step += 1
                continue

            if step == 11:
                task_mode = str(answers["task_mode"])
                objective_choices = get_objective_choices_for_task_mode(task_mode)
                task_default_objective = get_default_objective_metric(task_mode)
                objective_default = args.objective if args.objective in objective_choices else task_default_objective
                answers["objective_metric"] = prompt_choice_text(
                    "Select optimization metric" if english else "请选择优化指标",
                    objective_choices,
                    objective_default,
                    descriptions=descriptions_for("objective", lang),
                    lang=lang,
                )
                step += 1
                continue
        except WizardBack:
            back()

    return {
        "language": lang,
        "dataset_path": answers["dataset_path"],
        "time_col": answers["time_col"],
        "target_columns": list(answers.get("target_columns") or []),
        "input_columns": "all_except_time",
        "task_mode": answers["task_mode"],
        "horizons": list(answers["horizons"]),
        "input_chunk_length": int(answers["input_chunk_length"]),
        "strategy_name": DEFAULT_STRATEGY,
        "baseline_strategy": answers["baseline_strategy"],
        "baseline_models": list(answers.get("baseline_models") or []),
        "build_mode": bool(answers["build_mode"]),
        "dataset_diagnosis_mode": answers["dataset_diagnosis_mode"],
        "baseline_diagnosis_max_ablation_targets": int(answers["baseline_diagnosis_max_ablation_targets"]),
        "agent_ablation": _normalize_agent_ablation_arg(getattr(args, "agent_ablation", "none")),
        "objective_metric": answers["objective_metric"],
    }


def build_intent_from_answers(args: argparse.Namespace) -> Dict[str, Any]:
    lang = normalize_language(str(getattr(args, "language", "zh") or "zh"))
    english = is_english(lang)
    if not args.yes:
        return build_intent_interactive(args, lang=lang)
    root = project_root()
    datasets = discover_datasets(root)
    dataset_path = choose_dataset(datasets, args, lang=lang)
    columns = read_header(dataset_path)
    sample_rows = read_sample_rows(dataset_path)
    long_format = is_long_format_header(columns)
    column_descriptions = build_column_descriptions(columns, lang=lang)
    target_choices = read_long_format_variables(dataset_path, columns) if long_format else []
    if not target_choices:
        numeric_target_choices, _ = get_wide_numeric_feature_columns(
            str(dataset_path), infer_time_col(columns)
        )
        target_choices = numeric_target_choices or [c for c in columns if c != infer_time_col(columns)]
    target_descriptions = build_target_descriptions(target_choices, long_format, lang=lang)

    time_default = args.time_col or infer_time_col(columns)
    if args.time_col:
        time_col = require_time_column(
            args.time_col, columns, sample_rows, "--time-col",
            descriptions=column_descriptions, recommended=infer_time_col(columns), lang=lang,
        )
    elif args.yes:
        time_col = require_time_column(
            time_default, columns, sample_rows, "--time-col",
            descriptions=column_descriptions, recommended=infer_time_col(columns), lang=lang,
        )
    else:
        time_col = prompt_time_column(
            columns,
            sample_rows,
            time_default,
            descriptions=column_descriptions,
            lang=lang,
        )

    mode_default = args.task_mode or DEFAULT_TASK_MODE
    if mode_default not in MODE_OPTIONS:
        require_choice(
            mode_default, MODE_OPTIONS, "--task-mode",
            descriptions=descriptions_for("mode", lang), recommended=DEFAULT_TASK_MODE, lang=lang,
        )
    if args.yes:
        task_mode = require_choice(
            mode_default, MODE_OPTIONS, "--task-mode",
            descriptions=descriptions_for("mode", lang), recommended=DEFAULT_TASK_MODE, lang=lang,
        )
    else:
        task_mode = choose_option(
            "Select forecasting mode:" if english else "请选择预测模式：",
            [
                "MM - multivariate input, predict all variables (TFB default)" if english else "MM - 多变量输入，预测所有变量（TFB 官方默认）",
                "MS - multivariate input, predict one target variable" if english else "MS - 多变量输入，只预测一个目标变量",
                "SS - univariate input, predict one target variable" if english else "SS - 单变量输入，只预测一个目标变量",
            ],
            default_index=["MM", "MS", "SS"].index(mode_default) if mode_default in ("MM", "MS", "SS") else 0,
            lang=lang,
        ).split(" ", 1)[0]

    objective_choices = get_objective_choices_for_task_mode(task_mode)
    task_default_objective = get_default_objective_metric(task_mode)
    objective_default = args.objective if args.objective in objective_choices else task_default_objective

    inferred_target = infer_target_col(target_choices, time_col)
    target_default = args.target if args.target in target_choices else inferred_target
    target_columns: List[str] = []
    if task_mode in ("MS", "SS"):
        if args.target:
            target_raw = require_choice(
                args.target, target_choices, "--target",
                descriptions=target_descriptions, recommended=inferred_target, lang=lang,
            )
        elif args.yes:
            target_raw = require_choice(
                target_default, target_choices, "--target",
                descriptions=target_descriptions, recommended=target_default, lang=lang,
            )
        else:
            default_index = (
                target_choices.index(target_default)
                if target_default in target_choices
                else 0
            )
            print("Select target variable:" if english else "请选择目标变量：")
            target_raw = choose_described_option(
                "Select target variable" if english else "请选择目标变量",
                target_choices,
                descriptions=target_descriptions,
                default_index=default_index,
                lang=lang,
            )
        target_columns = [target_raw] if target_raw else []

    try:
        horizons_default = ",".join(str(h) for h in parse_horizons(args.horizons))
    except ValueError as exc:
        raise ValueError((f"--horizons is invalid: {args.horizons}. {exc}" if english else f"--horizons 输入无效：{args.horizons}。{exc}")) from exc
    horizons = parse_horizons(horizons_default) if args.yes else parse_horizons_prompt("Enter forecast horizons" if english else "请输入预测步长", horizons_default, lang=lang)

    if args.seq_len is not None:
        seq_len_default = require_positive_int(args.seq_len, "--seq-len", lang=lang)
    else:
        seq_len_default = DEFAULT_SEQ_LEN
    if args.yes:
        seq_len = seq_len_default
    else:
        seq_len = prompt_positive_int("Enter input window length seq_len" if english else "请输入输入窗口长度 seq_len", seq_len_default, lang=lang)

    if args.strategy and args.strategy != DEFAULT_STRATEGY:
        raise ValueError(f"--strategy currently supports only {DEFAULT_STRATEGY}." if english else f"--strategy 目前只支持 {DEFAULT_STRATEGY}。")
    strategy_name = DEFAULT_STRATEGY

    baseline_strategy = require_choice(
        args.baseline_strategy,
        BASELINE_STRATEGY_CHOICES,
        "--baseline-strategy",
        descriptions=descriptions_for("baseline_strategy", lang),
    recommended="auto",
        lang=lang,
    ) if args.yes else choose_described_option(
        "Select baseline strategy" if english else "请选择 baseline 策略",
        BASELINE_STRATEGY_CHOICES,
        descriptions_for("baseline_strategy", lang),
        default_index=BASELINE_STRATEGY_CHOICES.index(args.baseline_strategy)
        if args.baseline_strategy in BASELINE_STRATEGY_CHOICES
        else 1,
        lang=lang,
    )
    baseline_models = parse_model_list(args.baseline_models)
    if baseline_strategy == "manual" and not args.yes:
        baseline_models = prompt_manual_baselines(args.baseline_models, lang=lang)
    if baseline_strategy == "manual":
        baseline_models, missing_baseline_models = validate_manual_model_list(baseline_models)
        if missing_baseline_models:
            message = format_missing_manual_models(missing_baseline_models, lang=lang)
            raise ValueError(
                f"Unknown baseline model(s): {message}. Please correct --baseline-models."
                if english
                else f"未找到 baseline 模型：{message}。请修正 --baseline-models。"
            )
    if baseline_strategy == "manual" and not baseline_models:
        raise ValueError("--baseline-strategy manual requires at least one model through --baseline-models." if english else "--baseline-strategy manual 需要通过 --baseline-models 指定至少一个模型。")
    if args.yes:
        build_mode = bool(args.build_mode)
    else:
        build_mode = prompt_yes_no("Enable build mode" if english else "是否开启 build mode（极速闭环验证模式）", default=bool(args.build_mode), lang=lang)
    dataset_diagnosis_mode = require_choice(
        args.dataset_diagnosis_mode,
        DATASET_DIAGNOSIS_MODE_CHOICES,
        "--dataset-diagnosis-mode",
        descriptions=descriptions_for("dataset_diagnosis_mode", lang),
        recommended="required",
        lang=lang,
    ) if args.yes else choose_described_option(
        "Select dataset diagnosis mode" if english else "请选择数据集诊断模式",
        DATASET_DIAGNOSIS_INTERACTIVE_CHOICES,
        descriptions_for("dataset_diagnosis_mode", lang),
        default_index=DATASET_DIAGNOSIS_INTERACTIVE_CHOICES.index(args.dataset_diagnosis_mode)
        if args.dataset_diagnosis_mode in DATASET_DIAGNOSIS_INTERACTIVE_CHOICES
        else 0,
        lang=lang,
    )
    if args.yes:
        max_ablation_targets = require_non_negative_int(args.max_ablation_targets, "--max-ablation-targets", lang=lang)
    else:
        max_ablation_targets = prompt_non_negative_int(
            "Enter maximum baseline diagnosis ablations (0 skips ablation)" if english else "请输入 baseline 诊断最大消融数量（0 表示跳过消融）",
            args.max_ablation_targets,
            lang=lang,
        )

    return {
        "language": lang,
        "dataset_path": relative_to_root(dataset_path, root),
        "time_col": time_col,
        "target_columns": target_columns,
        "input_columns": "all_except_time",
        "task_mode": task_mode,
        "horizons": horizons,
        "input_chunk_length": seq_len,
        "strategy_name": strategy_name,
        "baseline_strategy": baseline_strategy,
        "baseline_models": baseline_models,
        "build_mode": build_mode,
        "dataset_diagnosis_mode": dataset_diagnosis_mode,
        "baseline_diagnosis_max_ablation_targets": max_ablation_targets,
        "agent_ablation": _normalize_agent_ablation_arg(getattr(args, "agent_ablation", "none")),
        "objective_metric": require_choice(
            objective_default, objective_choices, "--objective",
            descriptions=descriptions_for("objective", lang), recommended=task_default_objective, lang=lang,
        ) if args.yes else prompt_choice_text(
            "Select optimization metric" if english else "请选择优化指标",
            objective_choices,
            objective_default,
            descriptions=descriptions_for("objective", lang),
            lang=lang,
        ),
    }


def write_compiled_config(task_id: str, config: Dict[str, Any], base_dir: Optional[Path] = None) -> Path:
    return persist_compiled_config(task_id, config, base_dir or runtime_dir())


def print_summary(task_id: str, intent: Dict[str, Any], budget: str, max_rounds: int, run_now: bool) -> None:
    en = is_english(intent)
    targets = ", ".join(intent["target_columns"]) if intent["target_columns"] else ("all variables" if en else "全部变量")
    seq_len = intent.get("input_chunk_length", "")
    print("\nEvoCast Task Summary" if en else "\nEvoCast 任务配置摘要")
    print("-" * 60)
    if en:
        print(f"Task ID:        {task_id}")
        print(f"Language:       English")
        print(f"Dataset:        {intent['dataset_path']}")
        print(f"Task mode:      {intent['task_mode']}")
        print(f"Horizons:       {intent['horizons']}")
        print(f"Input window:   {seq_len}")
        print("label_len:      auto-derived")
        print(f"Time column:    {intent['time_col']}")
        print(f"Targets:        {targets}")
        print(f"Evaluation:     {intent['strategy_name']}")
        print(f"Objective:      {intent['objective_metric']}")
        print(f"Budget:         {budget}")
        print(f"Baseline mode:  {intent.get('baseline_strategy', 'auto')}")
    else:
        print(f"任务 ID：        {task_id}")
        print(f"语言：           中文")
        print(f"数据集：         {intent['dataset_path']}")
        print(f"预测模式：       {intent['task_mode']} - {MODE_DESCRIPTIONS.get(intent['task_mode'], '')}")
        print(f"预测步长：       {intent['horizons']}")
        print(f"输入窗口：       {seq_len}")
        print("label_len：      自动派生（不手动填写）")
        print(f"时间列：         {intent['time_col']}")
        print(f"目标变量：       {targets}")
        print(f"评估方式：       {intent['strategy_name']}")
        print(f"优化指标：       {intent['objective_metric']} - {OBJECTIVE_DESCRIPTIONS.get(intent['objective_metric'], '')}")
        print(f"实验规格：       {budget}")
        print(f"baseline 策略：  {intent.get('baseline_strategy', 'auto')} - {BASELINE_STRATEGY_DESCRIPTIONS.get(intent.get('baseline_strategy', 'auto'), '')}")
    if intent.get("baseline_models"):
        print((f"Baseline:       {', '.join(intent.get('baseline_models') or [])}") if en else f"指定 baseline：  {', '.join(intent.get('baseline_models') or [])}")
    print((f"Build mode:     {'yes' if intent.get('build_mode') else 'no'}") if en else f"Build mode：    {'是' if intent.get('build_mode') else '否'}")
    diagnosis_mode = str(intent.get("dataset_diagnosis_mode") or "required")
    diagnosis_descriptions = DATASET_DIAGNOSIS_MODE_DESCRIPTIONS_EN if en else DATASET_DIAGNOSIS_MODE_DESCRIPTIONS
    diagnosis_label = {
        "required": "normal",
        "reuse": "reuse-only",
        "skip": "skip",
    }.get(diagnosis_mode, diagnosis_mode) if en else {
        "required": "正常分析",
        "reuse": "只复用旧结果",
        "skip": "跳过分析",
    }.get(diagnosis_mode, diagnosis_mode)
    print((f"Dataset analysis: {diagnosis_label} - {diagnosis_descriptions.get(diagnosis_mode, '')}") if en else f"数据集分析：     {diagnosis_label} - {diagnosis_descriptions.get(diagnosis_mode, '')}")
    print((f"Max ablations:  {int(intent.get('baseline_diagnosis_max_ablation_targets') or 0)}") if en else f"最大消融数量：   {int(intent.get('baseline_diagnosis_max_ablation_targets') or 0)}")
    print((f"Max rounds:     {max_rounds}") if en else f"最大研究轮数：   {max_rounds}")
    print("Force full rounds: yes" if en else "强制跑满轮数：   是")
    print((f"Start now:      {'yes' if run_now else 'no'}") if en else f"是否立即运行：   {'是' if run_now else '否'}")
    print("-" * 60)


def init_task_from_wizard(
    *,
    task_id: str,
    config_path: Path,
    objective_metric: str,
    budget: str,
    metric_direction: str,
    target_value: Optional[float],
    max_rounds: int,
    max_debug_depth: int,
    api_config_name: str,
    intent: Dict[str, Any],
) -> Dict[str, Any]:
    return initialize_task(
        task_id=task_id,
        config_path=config_path,
        objective_metric=objective_metric,
        budget=budget,
        metric_direction=metric_direction,
        target_value=target_value,
        max_rounds=max_rounds,
        max_debug_depth=max_debug_depth,
        api_config_name=api_config_name,
        intent=intent,
        runtime_base=runtime_dir(),
    )


def run_baseline_diagnosis_before_agent(
    *,
    task_id: str,
    baseline: Dict[str, Any],
    objective_metric: str,
    budget: str,
    seed: int,
    max_targets: int | None = None,
    language: str = "zh",
) -> Dict[str, Any]:
    """Compatibility facade for the extracted baseline diagnosis service."""
    return _baseline_flow_service().run_diagnosis(
        task_id=task_id, baseline=baseline, objective_metric=objective_metric,
        budget=budget, seed=seed, max_targets=max_targets, language=language,
    )


def run_baseline_diagnosis_nonblocking_before_agent(
    *,
    task_id: str,
    baseline: Dict[str, Any],
    objective_metric: str,
    budget: str,
    seed: int,
    max_targets: int | None = None,
    language: str = "zh",
) -> Dict[str, Any]:
    """Compatibility facade for non-blocking diagnosis."""
    return _baseline_flow_service().run_diagnosis_nonblocking(
        diagnosis_runner=run_baseline_diagnosis_before_agent,
        task_id=task_id, baseline=baseline, objective_metric=objective_metric,
        budget=budget, seed=seed, max_targets=max_targets, language=language,
    )


def write_nonblocking_baseline_diagnosis(
    *,
    task_id: str,
    baseline: Dict[str, Any],
    objective_metric: str,
    reason: str,
    targets: Optional[List[Dict[str, Any]]] = None,
    results: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Compatibility facade for persisted partial diagnosis evidence."""
    return _baseline_flow_service().write_nonblocking_diagnosis(
        task_id=task_id, baseline=baseline, objective_metric=objective_metric,
        reason=reason, targets=targets, results=results,
    )


def _baseline_diagnosis_structure_has_content(value: Any) -> bool:
    return BaselineFlowService._structure_has_content(value)


def validate_global_args(args: argparse.Namespace) -> None:
    lang = normalize_language(str(getattr(args, "language", "zh") or "zh"))
    selected_task_mode = args.task_mode or DEFAULT_TASK_MODE
    selected_objective = args.objective or get_default_objective_metric(selected_task_mode)
    require_choice(
        selected_task_mode,
        MODE_OPTIONS,
        "--task-mode",
        descriptions=descriptions_for("mode", lang),
        recommended=DEFAULT_TASK_MODE,
        lang=lang,
    )
    if args.strategy and args.strategy != DEFAULT_STRATEGY:
        raise ValueError(f"--strategy currently supports only {DEFAULT_STRATEGY}." if is_english(lang) else f"--strategy 目前只支持 {DEFAULT_STRATEGY}。")
    require_choice(
        selected_objective,
        get_objective_choices_for_task_mode(selected_task_mode),
        "--objective",
        descriptions=descriptions_for("objective", lang),
        recommended=get_default_objective_metric(selected_task_mode),
        lang=lang,
    )
    require_choice(
        args.metric_direction,
        list(METRIC_DIRECTION_DESCRIPTIONS),
        "--metric-direction",
        descriptions=descriptions_for("metric_direction", lang),
        recommended="lower_is_better",
        lang=lang,
    )
    require_choice(
        args.baseline_strategy,
        BASELINE_STRATEGY_CHOICES,
        "--baseline-strategy",
        descriptions=descriptions_for("baseline_strategy", lang),
        recommended="auto",
        lang=lang,
    )
    require_choice(
        args.dataset_diagnosis_mode,
        DATASET_DIAGNOSIS_MODE_CHOICES,
        "--dataset-diagnosis-mode",
        descriptions=descriptions_for("dataset_diagnosis_mode", lang),
        recommended="required",
        lang=lang,
    )
    if args.seq_len is not None:
        require_positive_int(args.seq_len, "--seq-len", lang=lang)
    require_non_negative_int(args.max_rounds, "--max-rounds", lang=lang)
    require_non_negative_int(args.max_ablation_targets, "--max-ablation-targets", lang=lang)
    require_positive_int(args.max_debug_depth, "--max-debug-depth", lang=lang)


def run_wizard(args: argparse.Namespace) -> Dict[str, Any]:
    if args.yes:
        language = normalize_language(args.language)
    else:
        print("请选择语言 / Select language:")
        print("  - 1. 中文")
        print("  - 2. English")
        raw_language = _read_line("请输入序号 / Enter number [1]: ", lang="zh", allow_back=False).strip()
        language = "en" if raw_language == "2" else "zh"
    args.language = language
    en = is_english(language)
    print("EvoCast Wizard" if en else "EvoCast 启动向导")
    print("=" * 60)

    if getattr(args, "resume", False):
        return resume_existing_task(args)

    if args.yes:
        api_config_name = choose_api_config(args, lang=language)
        os.environ["EVOCAST_API_CONFIG"] = api_config_name
        print(f"[Wizard] API config: {api_config_name}" if en else f"[向导] 当前 API 配置：{api_config_name}")
        resumed = maybe_resume_interactive(args, lang=language)
        if resumed is not None:
            return resumed
        validate_global_args(args)
        intent = build_intent_from_answers(args)
        intent["research_intent"] = _research_intent_from_args(args)
        report = validate_intent(intent, base_dir=str(project_root()))
        print(format_validation_report(report, language))
        if report.has_errors():
            return {"status": "failed", "reason": "validation_failed", "errors": report.errors}
        objective_metric = intent["objective_metric"]
        task_id = str(args.task_id or "").strip() or generate_task_id(build_task_semantics_from_intent(intent))
        budget = normalize_budget(args.budget, str(runtime_dir()))
        max_rounds = require_non_negative_int(args.max_rounds, "--max-rounds", lang=language)
        max_debug_depth = require_positive_int(args.max_debug_depth, "--max-debug-depth", lang=language)
        run_now = True if args.start_run else False
        print_summary(task_id, intent, budget, max_rounds, run_now)
    else:
        setup_step = 0
        api_config_name = ""
        intent: Dict[str, Any] = {}
        objective_metric = ""
        task_id = ""
        budget = ""
        max_rounds = 0
        max_debug_depth = 0
        run_now = False
        while True:
            try:
                if setup_step == 0:
                    api_config_name = choose_api_config(args, lang=language)
                    os.environ["EVOCAST_API_CONFIG"] = api_config_name
                    print(f"[Wizard] API config: {api_config_name}" if en else f"[向导] 当前 API 配置：{api_config_name}")
                    setup_step += 1
                    continue
                if setup_step == 1:
                    resumed = maybe_resume_interactive(args, lang=language)
                    if resumed is not None:
                        return resumed
                    setup_step += 1
                    continue
                if setup_step == 2:
                    validate_global_args(args)
                    intent = build_intent_from_answers(args)
                    intent["research_intent"] = _research_intent_from_args(args)
                    report = validate_intent(intent, base_dir=str(project_root()))
                    print(format_validation_report(report, language))
                    if report.has_errors():
                        return {"status": "failed", "reason": "validation_failed", "errors": report.errors}
                    objective_metric = intent["objective_metric"]
                    task_id = str(args.task_id or "").strip() or generate_task_id(build_task_semantics_from_intent(intent))
                    budget = normalize_budget(args.budget, str(runtime_dir()))
                    setup_step += 1
                    continue
                if setup_step == 3:
                    max_rounds = prompt_non_negative_int("Enter maximum research rounds" if en else "请输入最大研究轮数", args.max_rounds, lang=language)
                    setup_step += 1
                    continue
                if setup_step == 4:
                    max_debug_depth = prompt_positive_int("Enter maximum consecutive debug rounds" if en else "请输入最大连续调试轮数", args.max_debug_depth, lang=language)
                    setup_step += 1
                    continue
                if setup_step == 5:
                    if args.start_run:
                        run_now = True
                    elif args.configure_only or args.dry_run:
                        run_now = False
                    else:
                        run_now = prompt_yes_no("Start the full automatic experiment loop now" if en else "是否现在启动完整自动实验循环", default=True, lang=language)
                    setup_step += 1
                    continue
                print_summary(task_id, intent, budget, max_rounds, run_now)
                if not args.dry_run and not prompt_yes_no("Confirm this configuration" if en else "确认使用以上配置", default=True, lang=language):
                    return {"status": "cancelled"}
                break
            except WizardBack:
                if setup_step <= 0:
                    print("[Wizard] Already at the first configurable step." if en else "[向导] 已经在第一个可配置步骤。")
                    continue
                setup_step -= 1
                print(_interactive_back_message(language))

    compiled = compile_config(intent, report, seed=args.seed, build_mode=bool(intent.get("build_mode")))

    if args.dry_run:
        print("[Wizard] Dry run complete: no files written and no experiment launched." if en else "[向导] 空运行完成：未写入文件，也未启动实验。")
        return {
            "status": "dry_run",
            "task_id": task_id,
            "intent": intent,
            "compiled_config": compiled,
        }

    config_path = write_compiled_config(task_id, compiled)
    print(f"[Wizard] Compiled config written: {config_path}" if en else f"[向导] 已写入编译后的配置：{config_path}")

    if not run_now:
        init_task_from_wizard(
            task_id=task_id,
            config_path=config_path,
            objective_metric=objective_metric,
            budget=budget,
            metric_direction=args.metric_direction,
            target_value=args.target_value,
            max_rounds=max_rounds,
            max_debug_depth=max_debug_depth,
            api_config_name=api_config_name,
            intent=intent,
        )
        print("[Wizard] Task configured. Re-run with --start-run when ready." if en else "[向导] 任务已配置完成。准备好后可重新运行并添加 --start-run 启动 v3 agent。")
        return {"status": "configured", "task_id": task_id, "config_path": str(config_path)}

    init_task_from_wizard(
        task_id=task_id,
        config_path=config_path,
        objective_metric=objective_metric,
        budget=budget,
        metric_direction=args.metric_direction,
        target_value=args.target_value,
        max_rounds=max_rounds,
        max_debug_depth=max_debug_depth,
        api_config_name=api_config_name,
        intent=intent,
    )
    # One terminal surface now covers diagnosis, baseline search, ablations,
    # and formal research.  The final HTML report remains post-run only.
    terminal_ui = TerminalUI(str(runtime_dir()), task_id).start()
    atexit.register(terminal_ui.stop)

    baseline_result = _baseline_flow_service().run(
        BaselineFlowRequest(
            task_id=task_id,
            objective_metric=objective_metric,
            budget=budget,
            metric_direction=args.metric_direction,
            baseline_strategy=str(intent.get("baseline_strategy") or "auto"),
            baseline_models=list(intent.get("baseline_models") or []),
            seed=args.seed,
            dataset_diagnosis_mode=str(intent.get("dataset_diagnosis_mode") or "required"),
            max_ablation_targets=int(intent.get("baseline_diagnosis_max_ablation_targets") or 0),
            language=language,
        )
    )
    if baseline_result.get("status") == "baseline_failed":
        baseline_summary = dict(baseline_result.get("baseline_summary") or {})
        print(
            f"[Wizard] Baseline did not complete; agent launch stopped. Summary: {json.dumps(baseline_summary, ensure_ascii=False, default=str)}"
            if en
            else f"[向导] baseline 未完成，已停止启动 agent。摘要：{json.dumps(baseline_summary, ensure_ascii=False, default=str)}"
        )
        return {
            "status": "baseline_failed",
            "task_id": task_id,
            "config_path": str(config_path),
            "baseline_summary": baseline_summary,
        }
    diagnosis_summary = dict(baseline_result.get("diagnosis_summary") or {})

    finalizer = _finalization_service()
    finalization_request = FinalizationRequest(
        task_id=task_id,
        review_report=_review_report_enabled(args),
        open_report=bool(args.open_report),
        strict_report=max_rounds == 0,
    )
    if max_rounds == 0:
        outcome = finalizer.finalize_zero_rounds(finalization_request)
        report = dict(outcome.get("report") or {})
        if report.get("status") == "ok":
            print(
                f"[Wizard] Review HTML generated: {report.get('html_path')}"
                if en
                else f"[向导] Review HTML 已生成：{report.get('html_path')}"
            )
        print(
            "[Wizard] max_rounds=0: baseline and baseline diagnosis complete; no research round launched."
            if en
            else "[向导] max_rounds=0：baseline 和 baseline 诊断消融已完成，不启动创新 round。"
        )
        return {"status": "baseline_diagnosis_completed", "task_id": task_id, "config_path": str(config_path)}

    sync_task_stage(str(runtime_dir()), task_id, stage="agent_launch", status="starting")
    try:
        final_progress = _research_loop_service().run(
            ResearchLoopRequest(
                task_id=task_id,
                objective_metric=objective_metric,
                api_config=api_config_name,
                max_rounds=max_rounds,
                language=language,
                idea_planner=str(
                    getattr(args, "idea_planner", "research_program_hypothesis_competition")
                    or "research_program_hypothesis_competition"
                ),
                agent_ablation=str(intent.get("agent_ablation") or "none"),
                diagnosis=dict(diagnosis_summary.get("diagnosis") or {}),
                explicit_contract=str(getattr(args, "build_contract", "") or "").strip(),
            )
        )
    except Exception as exc:
        failure = finalizer.record_agent_failure(
            FinalizationRequest(
                task_id=task_id,
                review_report=_review_report_enabled(args),
                strict_report=False,
            ),
            error=exc,
            progress=round_progress(str(runtime_dir()), task_id),
        )
        report = dict(failure.get("report") or {})
        if report.get("status") == "ok":
            print(
                f"[Wizard] Review HTML generated after task failure: {report.get('html_path')}"
                if en
                else f"[向导] 任务失败证据报告已生成：{report.get('html_path')}"
            )
        raise

    completion = finalizer.finalize_research(
        finalization_request,
        progress=final_progress,
        max_rounds=max_rounds,
    )
    report = dict(completion.get("report") or {})
    if report.get("status") == "ok":
        print(
            f"[Wizard] Review HTML generated: {report.get('html_path')}"
            if en
            else f"[向导] Review HTML 已生成：{report.get('html_path')}"
        )
    elif report and report.get("status") not in {"skipped", None}:
        print(
            f"[Wizard] Review HTML generation failed: {json.dumps(report, ensure_ascii=False, default=str)}"
            if en
            else f"[向导] Review HTML 生成失败：{json.dumps(report, ensure_ascii=False, default=str)}",
            file=sys.stderr,
        )
    return {"status": "started_v3", "task_id": task_id, "config_path": str(config_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    diagnosis_policy = baseline_diagnosis_policy()
    parser = ChineseArgumentParser(description="EvoCast 中文交互式启动向导。")
    parser.add_argument("--task-id", default="", help="显式指定任务 ID；并行实验时用于避免自动时间戳碰撞。")
    parser.add_argument("--resume", action="store_true", help="从已有任务的正式 Research 断点继续；不会重跑数据诊断、baseline 或 baseline 消融。")
    parser.add_argument("--language", choices=LANGUAGE_CHOICES, default="zh", help="向导与报告语言：zh 中文，en English。")
    parser.add_argument("--dataset", default="", help="数据集 CSV 路径或文件名，例如 ETTh1.csv。")
    parser.add_argument("--time-col", default="", help="时间列名；ETTh1 这类数据通常使用 date。")
    parser.add_argument("--task-mode", default=DEFAULT_TASK_MODE, help="预测模式：MM 全变量预测，MS 单目标预测，SS 单变量预测。")
    parser.add_argument("--target", default="", help="MS/SS 模式下的目标变量，例如 OT。")
    parser.add_argument("--horizons", default="96", help="预测步长，多个值用逗号分隔，例如 96,192。")
    parser.add_argument("--seq-len", type=int, default=None, help="输入窗口长度 seq_len；不提供时，交互模式下会明确询问。")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, help=argparse.SUPPRESS)
    parser.add_argument("--objective", default=DEFAULT_OBJECTIVE, help="优化指标，例如 mse、mae、rmse。")
    parser.add_argument("--metric-direction", default="lower_is_better", help="指标方向：误差类指标通常选择 lower_is_better。")
    parser.add_argument("--budget", default=DEFAULT_BUDGET, help=argparse.SUPPRESS)
    parser.add_argument(
        "--baseline-strategy",
        default="auto",
        choices=BASELINE_STRATEGY_CHOICES,
        help="baseline 前置策略：auto 自动搜索，manual 指定模型。研究任务必须先建立 baseline。",
    )
    parser.add_argument(
        "--baseline-models",
        default="",
        help="manual baseline 模型列表，多个用逗号或空格分隔，例如 Informer,iTransformer。",
    )
    parser.add_argument(
        "--dataset-diagnosis-mode",
        default="required",
        choices=DATASET_DIAGNOSIS_MODE_CHOICES,
        help="数据集分析模式：required 正常分析；skip 跳过分析直接测试 research；reuse 为高级模式，只复用已有分析结果。",
    )
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS, help="最大研究轮数；0 表示只跑 baseline 和 baseline 诊断消融，不进入创新 round。")
    parser.add_argument(
        "--max-ablation-targets",
        type=int,
        default=max(0, int(diagnosis_policy.get("max_ablation_targets", 3))),
        help="baseline 诊断阶段最多允许 LLM 选择多少个机制消融目标；0 表示跳过消融。",
    )
    parser.add_argument("--research-intent", default="", help="本任务的持久化研究意图；会写入 TaskConfig 和每个 BuildContract。")
    parser.add_argument("--research-intent-file", default="", help="从 UTF-8 文本文件读取持久化研究意图。")
    parser.add_argument("--build-contract", default="", help="显式 BuildContract JSON；新 Research 路径必须提供。")
    parser.add_argument(
        "--idea-planner",
        default="research_program_hypothesis_competition",
        choices=["research_program_hypothesis_competition", "scientist_critic", "none"],
        help="Research 前置 idea 生成器；research_program_hypothesis_competition 会在 BuildContract 前生成结构化研究方向，none 保持旧行为。",
    )
    parser.add_argument(
        "--agent-ablation",
        default="none",
        choices=AGENT_ABLATION_CHOICES,
        help="Agent 消融：A3 删除 planner 证据注入；A4 放开为全仓库任意写；A5 冻结 planner 历史证据；none 为正式主链。",
    )
    parser.add_argument("--max-debug-depth", type=int, default=3, help="最大连续调试轮数，必须是正整数。")
    parser.add_argument("--seed", type=int, default=baseline_seed(str(runtime_dir())), help="随机种子。")
    parser.add_argument(
        "--api-config",
        default=DEFAULT_API_CONFIG,
        choices=API_CONFIG_CHOICES,
        help="选择使用的 API YAML 配置，例如 providers/deepseek.yaml、providers/openai.yaml 或 providers/minimax.yaml。",
    )
    parser.add_argument("--target-value", type=float, default=None, help="希望达到的目标指标值；不设置则尽量寻找更优结果。")
    parser.add_argument(
        "--build-mode",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_BUILD_MODE,
        help=BUILD_MODE_DESCRIPTION,
    )
    parser.add_argument("--start-run", action="store_true", help="确认配置后立即启动 multi-agent 实验循环。")
    parser.add_argument("--configure-only", action="store_true", help="只写入任务配置，不启动实验。")
    parser.add_argument("--dry-run", action="store_true", help="只校验和编译到内存，不写文件、不启动实验。")
    parser.add_argument(
        "--open-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="实验结束生成 Review HTML 后，自动用系统默认浏览器打开。可用 --no-open-report 关闭。",
    )
    parser.add_argument(
        "--suppress-review-report",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--yes", action="store_true", help="非交互模式：接受自动推断值和默认值。")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    _configure_windows_safe_io()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = run_wizard(args)
    except KeyboardInterrupt:
        print("\n[向导] 已取消。")
        sys.exit(130)
    except Exception as exc:
        print(f"[向导] 错误：{exc}", file=sys.stderr)
        sys.exit(1)

    if result.get("status") == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
