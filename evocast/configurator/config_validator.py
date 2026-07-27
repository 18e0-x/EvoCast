"""Deterministic config validator.

Reads a CSV, checks column names, types, time parseability, horizon
constraints, and task-mode consistency. Returns a ValidationReport.
The LLM proposes — the validator decides.
"""

import csv
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ts_benchmark.data.utils import _parse_datetime_series
from evocast.domain.frequency_semantics import (
    canonical_frequency_label_from_pandas_alias,
    describe_horizon_span,
    is_subdaily_frequency,
    normalize_frequency_label,
    parse_frequency_seconds,
)
from evocast.domain.metric_semantics import validate_objective_metric_for_task_mode


@dataclass
class ValidationReport:
    """Structured validation result."""

    intent: Dict
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Discovered facts about the dataset
    column_names: List[str] = field(default_factory=list)
    row_count: int = 0
    time_col_index: Optional[int] = None
    target_indices: Dict[str, int] = field(default_factory=dict)
    numeric_target_indices: Dict[str, int] = field(default_factory=dict)
    numeric_feature_columns: List[str] = field(default_factory=list)
    dropped_non_numeric_columns: List[str] = field(default_factory=list)
    detected_frequency: Optional[str] = None
    detected_frequency_confidence: Optional[float] = None
    detected_frequency_method: Optional[str] = None
    detected_frequency_evidence: Optional[Dict[str, Any]] = None
    time_min: Optional[str] = None
    time_max: Optional[str] = None
    # Long-format CSV support (date, data, cols style)
    is_long_format: bool = False
    long_format_vars: List[str] = field(default_factory=list)

    # Inferred config hints
    inferred_horizon_max: Optional[int] = None
    suggested_input_chunk_length: Optional[int] = None

    def has_errors(self) -> bool:
        return len(self.errors) > 0


# ── Main entry point ──────────────────────────────────────────────────────


def validate_intent(intent: Dict, base_dir: Optional[str] = None) -> ValidationReport:
    """Validate a config intent against the actual CSV data.

    Args:
        intent: Dict with dataset_path, time_col, target_columns, etc.
        base_dir: Project root for resolving relative paths.

    Returns:
        ValidationReport — check .valid before compiling.
    """
    # Resolve dataset path
    dataset_path = intent.get("dataset_path", "")
    if base_dir and not os.path.isabs(dataset_path):
        dataset_path = os.path.join(base_dir, dataset_path)

    report = ValidationReport(intent=intent, valid=True)

    # 0. Check file existence — try dataset/forecasting/ as fallback
    if not dataset_path or not os.path.exists(dataset_path):
        basename = os.path.basename(dataset_path) if dataset_path else ""
        # Search in dataset/forecasting/ (base_dir is project root here)
        proj = base_dir or os.getcwd()
        alt_path = os.path.join(proj, "dataset", "forecasting", basename)
        if not os.path.exists(alt_path):
            alt_path = os.path.join(proj, "dataset", basename)
        if basename and os.path.exists(alt_path):
            dataset_path = alt_path
            report.warnings.append(f"Resolved dataset to: {dataset_path}")
        else:
            report.errors.append(f"Dataset not found: {dataset_path}")
            report.valid = False
            return report

    # 1. Read CSV header + sample rows
    try:
        with open(dataset_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            report.column_names = header
            sample_rows = []
            for i, row in enumerate(reader):
                sample_rows.append(row)
                if len(sample_rows) >= 100:
                    break
            # Count remaining rows for total
            row_count = len(sample_rows)
            for _ in reader:
                row_count += 1
            report.row_count = row_count
    except Exception as e:
        report.errors.append(f"Cannot read CSV: {e}")
        report.valid = False
        return report

    # Detect long-format CSV (date, data, cols) early so downstream checks can adapt
    _detect_long_format(report, header, sample_rows)

    # For long-format CSVs, the first 100 rows may only contain one variable
    # (data is grouped by variable). Do a second pass to discover all vars.
    if report.is_long_format and len(report.long_format_vars) < 2:
        _discover_all_long_format_vars(report, dataset_path)

    # 2. Time column
    time_col = intent.get("time_col", "")
    _validate_time_col(report, header, time_col, sample_rows)

    # 3. Wide-format numeric forecasting schema
    _inspect_wide_numeric_columns(report, dataset_path, time_col)

    # 4. Target columns
    target_columns = intent.get("target_columns", [])
    task_mode = intent.get("task_mode", "MS")
    _validate_target_columns(report, header, target_columns, task_mode)

    # 5. Input columns
    input_columns = intent.get("input_columns", "all_except_time")
    _validate_input_columns(report, header, time_col, target_columns, input_columns)

    # 6. Horizon
    horizons = intent.get("horizons", [])
    if not isinstance(horizons, list):
        horizons = [horizons]
    _validate_horizons(report, horizons)

    # 7. Task mode consistency
    _validate_task_mode(report, task_mode, target_columns)

    # 7b. Objective metric consistency with task mode
    _validate_objective_metric(report, intent.get("objective_metric"), task_mode)

    # 7c. Strategy consistency
    _validate_strategy_name(report, intent.get("strategy_name"))

    # 8. Detect frequency from time column
    if report.time_col_index is not None and sample_rows:
        _detect_frequency(report, dataset_path, time_col)

    # 9. Inferred hints
    _infer_hints(report, horizons)

    # 10. Surface horizon semantics for high-frequency datasets
    _warn_horizon_duration_semantics(report, horizons)

    report.valid = len(report.errors) == 0
    return report


# ── Individual checks ─────────────────────────────────────────────────────


def _validate_time_col(
    report: ValidationReport,
    header: List[str],
    time_col: str,
    sample_rows: List[List[str]],
) -> None:
    if not time_col:
        report.errors.append("time_col is required")
        return

    if time_col not in header:
        report.errors.append(
            f"time_col '{time_col}' not found in CSV columns: {header}"
        )
        return

    report.time_col_index = header.index(time_col)

    # Check parseability
    failed = 0
    times = []
    for row in sample_rows:
        val = row[report.time_col_index].strip()
        if not val:
            failed += 1
            continue
        try:
            parsed = _try_parse_time(val)
            if parsed:
                times.append(parsed)
            else:
                failed += 1
        except Exception:
            failed += 1

    if failed == len(sample_rows):
        report.errors.append(
            f"Cannot parse any values in time column '{time_col}'. "
            f"Sample: {[r[report.time_col_index] for r in sample_rows[:3]]}"
        )
    elif failed > 0:
        report.warnings.append(f"{failed}/{len(sample_rows)} time values failed to parse")

    if times:
        report.time_min = min(times).isoformat()
        report.time_max = max(times).isoformat()


def _validate_target_columns(
    report: ValidationReport,
    header: List[str],
    target_columns: List[str],
    task_mode: str,
) -> None:
    if task_mode == "MM":
        # Predict all columns — no specific target needed
        return

    if not target_columns:
        if task_mode in ("MS", "SS"):
            report.errors.append(f"target_columns is required for task_mode='{task_mode}'")
        return

    # Long-format CSV: target is a value in the 'cols' column, not a column name
    if report.is_long_format:
        for tc in target_columns:
            if tc not in report.long_format_vars:
                report.errors.append(
                    f"target_column '{tc}' not found in CSV 'cols' values. "
                    f"Available variables (sample): {report.long_format_vars[:20]}"
                )
        return

    for tc in target_columns:
        if tc not in header:
            report.errors.append(f"target_column '{tc}' not found in CSV columns")
        else:
            report.target_indices[tc] = header.index(tc)
            if report.numeric_feature_columns and tc in report.numeric_target_indices:
                continue
            if report.numeric_feature_columns and task_mode in ("MS", "SS"):
                allowed = ", ".join(report.numeric_feature_columns)
                report.errors.append(
                    f"target_column '{tc}' is not a forecasting-safe numeric column. "
                    f"Wide-format non-numeric columns are dropped before modeling. "
                    f"Choose one of: {allowed}"
                )


def _validate_input_columns(
    report: ValidationReport,
    header: List[str],
    time_col: str,
    target_columns: List[str],
    input_columns: Any,
) -> None:
    if isinstance(input_columns, list):
        missing = [c for c in input_columns if c not in header]
        if missing:
            report.errors.append(f"input_columns not in CSV: {missing}")
    elif input_columns == "all_except_time":
        pass  # compiler handles this
    elif isinstance(input_columns, str) and input_columns != "all_except_time":
        report.warnings.append(f"Unknown input_columns value: '{input_columns}'")
    else:
        pass  # default


def _validate_horizons(report: ValidationReport, horizons: List[int]) -> None:
    if not horizons:
        report.errors.append("horizons is required (e.g., [96, 192, 336, 720])")
        return

    for h in horizons:
        if not isinstance(h, int) or h <= 0:
            report.errors.append(f"Invalid horizon value: {h}")
        if report.row_count > 0 and h >= report.row_count:
            report.warnings.append(
                f"Horizon {h} >= row count {report.row_count} — test set may be too small"
            )


def _validate_task_mode(
    report: ValidationReport,
    task_mode: str,
    target_columns: List[str],
) -> None:
    valid_modes = {"MS", "MM", "SS"}
    if task_mode not in valid_modes:
        report.errors.append(f"task_mode must be one of {valid_modes}, got '{task_mode}'")

    if task_mode == "MS" and len(target_columns) != 1:
        report.errors.append(
            f"MS mode requires exactly 1 target column, got {len(target_columns)}"
        )

    if task_mode == "SS" and len(target_columns) > 1:
        report.warnings.append(
            f"SS mode with {len(target_columns)} targets — only the first will be used"
        )


def _validate_objective_metric(
    report: ValidationReport,
    objective_metric: Any,
    task_mode: str,
) -> None:
    try:
        metric = validate_objective_metric_for_task_mode(
            str(objective_metric or ""),
            task_mode,
        )
    except ValueError as exc:
        report.errors.append(str(exc))
        return

    if task_mode in {"MS", "SS"} and metric.endswith("_norm"):
        report.warnings.append(
            f"objective_metric '{metric}' uses ts_benchmark normalized error semantics: "
            "errors are scaled by the training-set standard deviation. "
            "For MS/SS tasks this can produce much smaller values than raw mse/mae. "
            "If you want directly interpretable raw-scale errors, prefer 'mse' or 'mae'."
        )


def _validate_strategy_name(
    report: ValidationReport,
    strategy_name: Any,
) -> None:
    strategy = str(strategy_name or "rolling_forecast").strip()
    if strategy != "rolling_forecast":
        report.errors.append(
            f"Unsupported strategy_name '{strategy}'. "
            "EvoCast now only supports rolling_forecast."
        )


def _detect_long_format(
    report: ValidationReport,
    header: List[str],
    sample_rows: List[List[str]],
) -> None:
    """Detect TFB long-format CSV: date/time + data + cols (variable name)."""
    header_lower = [h.lower().strip() for h in header]
    if header_lower[:3] == ["date", "data", "cols"] or header_lower[:3] == ["time", "data", "cols"]:
        report.is_long_format = True
        report.long_format_vars = _collect_long_format_vars(sample_rows, cols_idx=2)
    elif len(header) <= 5 and "data" in header_lower and "cols" in header_lower:
        # Heuristic: few columns + has 'data' + 'cols' → likely long format
        report.is_long_format = True
        try:
            cols_idx = header_lower.index("cols")
        except ValueError:
            cols_idx = header_lower.index("data") + 1 if "data" in header_lower else 2
        report.long_format_vars = _collect_long_format_vars(sample_rows, cols_idx=cols_idx)


def _collect_long_format_vars(
    sample_rows: List[List[str]],
    cols_idx: int,
    max_scan: int = 5000,
) -> List[str]:
    """Collect unique variable names from long-format rows.

    Scans up to max_scan rows to discover all variables — long-format CSVs
    typically group rows by variable, so the first 100 rows may only show
    the first variable.

    Order is preserved (first-seen wins) to match TFB's pd.unique() behaviour,
    which determines column indices after long→wide conversion.
    """
    vars_seen: dict = {}  # preserve insertion order
    for row in sample_rows[:max_scan]:
        if len(row) > cols_idx and row[cols_idx].strip():
            v = row[cols_idx].strip()
            if v not in vars_seen:
                vars_seen[v] = True
    return list(vars_seen.keys())


def _discover_all_long_format_vars(report: ValidationReport, dataset_path: str) -> None:
    """Second-pass scan to find all unique variable names in a long-format CSV.

    Long-format CSVs group rows by variable, so the initial 100-row sample
    typically only sees the first variable.  This scans the full file to
    collect every unique value from the 'cols' column, preserving appearance
    order to match TFB's pd.unique().
    """
    import csv as csv_module
    vars_seen: dict = {}
    try:
        with open(dataset_path, "r", encoding="utf-8-sig") as f:
            reader = csv_module.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) > 2 and row[2].strip():
                    v = row[2].strip()
                    if v not in vars_seen:
                        vars_seen[v] = True
        if vars_seen:
            report.long_format_vars = list(vars_seen.keys())
    except Exception:
        pass  # keep whatever we had from the initial sample


def _detect_frequency(report: ValidationReport, dataset_path: str, time_col: str) -> None:
    """Detect dataset frequency with infer_freq, robust fallback, and calendar checks."""
    index = _load_frequency_index(dataset_path, time_col)
    if index is None or len(index) < 2:
        report.detected_frequency = "other"
        report.detected_frequency_confidence = 0.0
        report.detected_frequency_method = "insufficient_time_points"
        report.detected_frequency_evidence = {
            "valid_timestamp_count": 0 if index is None else int(len(index)),
        }
        return

    inferred_alias = None
    inferred_frequency = None
    if len(index) >= 3:
        try:
            inferred_alias = pd.infer_freq(index)
        except ValueError:
            inferred_alias = None
        inferred_frequency = _map_pandas_frequency_alias(inferred_alias)

    if inferred_frequency:
        detected, confidence, method, evidence = _apply_calendar_semantics(
            index=index,
            candidate=inferred_frequency,
            base_confidence=0.98,
            method="pandas_infer_freq",
            evidence={
                "pandas_alias": inferred_alias,
                "valid_timestamp_count": int(len(index)),
            },
        )
    else:
        detected, confidence, method, evidence = _detect_frequency_from_deltas(index)

    report.detected_frequency = detected
    report.detected_frequency_confidence = round(float(confidence), 4)
    report.detected_frequency_method = method
    report.detected_frequency_evidence = evidence

    if detected == "other":
        report.warnings.append(
            "Frequency detection is inconclusive; falling back to 'other'. "
            "Please confirm the dataset granularity if metrics like MASE matter."
        )
    elif confidence < 0.75:
        report.warnings.append(
            f"Frequency detection confidence is low ({confidence:.2f}) for '{detected}'. "
            "Please confirm the inferred granularity."
        )


def _load_frequency_index(
    dataset_path: str,
    time_col: str,
    max_rows: int = 4096,
) -> Optional[pd.DatetimeIndex]:
    try:
        data = pd.read_csv(
            dataset_path,
            usecols=[time_col],
            nrows=max_rows,
            encoding="utf-8-sig",
        )
    except Exception:
        return None

    if time_col not in data.columns:
        return None

    parsed = _parse_datetime_series(data[time_col]).dropna()
    if parsed.empty:
        return None

    parsed = parsed.sort_values().drop_duplicates()
    if parsed.empty:
        return None

    return pd.DatetimeIndex(parsed)


def _map_pandas_frequency_alias(alias: Optional[str]) -> Optional[str]:
    return canonical_frequency_label_from_pandas_alias(alias)


def _detect_frequency_from_deltas(
    index: pd.DatetimeIndex,
) -> Tuple[str, float, str, Dict[str, Any]]:
    delta_seconds = (
        pd.Series(index)
        .diff()
        .dropna()
        .dt.total_seconds()
    )
    delta_seconds = delta_seconds[delta_seconds > 0]
    if delta_seconds.empty:
        return (
            "other",
            0.0,
            "delta_fallback_insufficient",
            {"valid_timestamp_count": int(len(index)), "positive_delta_count": 0},
        )

    dominant_seconds, dominant_share = _dominant_delta_signature(delta_seconds)
    median_seconds = float(delta_seconds.median())
    candidate_seconds = dominant_seconds or median_seconds
    best_label = normalize_frequency_label(
        None if candidate_seconds is None else f"{int(round(candidate_seconds))}s"
    )
    if not best_label or best_label == "other":
        best_label = "other"

    if candidate_seconds is None:
        best_score = 0.0
    else:
        tolerance = max(1.0, float(candidate_seconds) * 0.1)
        within = (delta_seconds - float(candidate_seconds)).abs() <= tolerance
        best_score = float(within.mean())

    if best_score < 0.6:
        return (
            "other",
            min(0.59, max(best_score, 0.0)),
            "delta_fallback",
            {
                "valid_timestamp_count": int(len(index)),
                "positive_delta_count": int(len(delta_seconds)),
                "median_delta_seconds": median_seconds,
                "dominant_delta_seconds": dominant_seconds,
                "dominant_delta_share": dominant_share,
                "canonical_candidate": best_label,
            },
        )

    detected, confidence, method, evidence = _apply_calendar_semantics(
        index=index,
        candidate=best_label,
        base_confidence=min(0.95, max(0.6, best_score)),
        method="delta_fallback",
        evidence={
            "valid_timestamp_count": int(len(index)),
            "positive_delta_count": int(len(delta_seconds)),
            "median_delta_seconds": median_seconds,
            "dominant_delta_seconds": dominant_seconds,
            "dominant_delta_share": dominant_share,
            "canonical_candidate": best_label,
        },
    )
    return detected, confidence, method, evidence


def _dominant_delta_signature(delta_seconds: pd.Series) -> Tuple[Optional[float], float]:
    rounded = delta_seconds.round().astype(int)
    if rounded.empty:
        return None, 0.0
    counter = Counter(rounded.tolist())
    dominant_value, dominant_count = counter.most_common(1)[0]
    return float(dominant_value), float(dominant_count / len(rounded))


def _apply_calendar_semantics(
    *,
    index: pd.DatetimeIndex,
    candidate: str,
    base_confidence: float,
    method: str,
    evidence: Dict[str, Any],
) -> Tuple[str, float, str, Dict[str, Any]]:
    calendar = _calendar_semantic_features(index)
    merged_evidence = dict(evidence)
    merged_evidence["calendar_features"] = calendar

    adjusted = candidate
    adjusted_method = method
    confidence = float(base_confidence)

    if candidate == "hourly" and calendar["mean_rows_per_day"] <= 1.2:
        adjusted = "daily"
        adjusted_method = method + "+calendar_semantics"
        confidence = max(confidence, 0.9)
        merged_evidence["calendar_override_reason"] = "one_row_per_day"

    if adjusted == "daily" and calendar["mean_rows_per_day"] <= 1.2:
        confidence = min(0.99, max(confidence, 0.92))

    if is_subdaily_frequency(adjusted):
        confidence = min(0.99, max(confidence, 0.92))

    if adjusted == "weekly" and calendar["unique_weekday_count"] == 1:
        confidence = min(0.99, max(confidence, 0.92))

    if adjusted == "monthly" and calendar["month_boundary_ratio"] >= 0.7:
        confidence = min(0.99, max(confidence, 0.9))

    if adjusted in {"quarterly", "yearly"} and calendar["month_boundary_ratio"] >= 0.7:
        confidence = min(0.99, max(confidence, 0.88))

    return adjusted, confidence, adjusted_method, merged_evidence


def _calendar_semantic_features(index: pd.DatetimeIndex) -> Dict[str, Any]:
    if len(index) == 0:
        return {
            "mean_rows_per_day": 0.0,
            "median_rows_per_day": 0.0,
            "midnight_ratio": 0.0,
            "unique_weekday_count": 0,
            "month_boundary_ratio": 0.0,
        }

    ts = pd.Series(index)
    normalized_days = ts.dt.normalize()
    rows_per_day = normalized_days.value_counts()
    month_boundary = (
        (ts.dt.day == 1)
        | (ts.dt.is_month_end)
    )
    midnight = (
        (ts.dt.hour == 0)
        & (ts.dt.minute == 0)
        & (ts.dt.second == 0)
    )
    return {
        "mean_rows_per_day": float(rows_per_day.mean()) if not rows_per_day.empty else 0.0,
        "median_rows_per_day": float(rows_per_day.median()) if not rows_per_day.empty else 0.0,
        "midnight_ratio": float(midnight.mean()) if len(ts) else 0.0,
        "unique_weekday_count": int(ts.dt.weekday.nunique()),
        "month_boundary_ratio": float(month_boundary.mean()) if len(ts) else 0.0,
    }


def _infer_hints(report: ValidationReport, horizons: List[int]) -> None:
    """Infer suggested config values from the data."""
    if horizons:
        report.inferred_horizon_max = max(horizons)
        report.suggested_input_chunk_length = max(horizons)


def _warn_horizon_duration_semantics(
    report: ValidationReport,
    horizons: List[int],
) -> None:
    freq = report.detected_frequency
    if not horizons or not freq:
        return

    valid_horizons = [int(h) for h in horizons if isinstance(h, int)]
    if not valid_horizons:
        return

    horizon = max(valid_horizons)
    span = describe_horizon_span(horizon, freq)
    if not span:
        return
    if is_subdaily_frequency(freq):
        report.warnings.append(
            f"Horizon {horizon} on frequency '{freq}' means {span} of forecast span, "
            "not that many days. Confirm this matches your intended evaluation window."
        )


def _inspect_wide_numeric_columns(
    report: ValidationReport,
    dataset_path: str,
    time_col: str,
) -> None:
    """Record the wide-format numeric forecasting schema used at runtime.

    The forecasting runtime consumes numeric matrices only. For wide-format
    CSVs, text/categorical columns are dropped before modeling. This helper
    mirrors that behavior so validation and compilation use the same column
    space as runtime.
    """
    if report.is_long_format:
        return

    try:
        data = pd.read_csv(dataset_path, nrows=256)
    except Exception:
        return

    if data.empty:
        return

    numeric_columns, dropped_columns = _extract_wide_numeric_columns(data, time_col)
    report.numeric_feature_columns = numeric_columns
    report.dropped_non_numeric_columns = dropped_columns
    report.numeric_target_indices = {
        col: idx for idx, col in enumerate(report.numeric_feature_columns)
    }

    if dropped_columns:
        report.warnings.append(
            "Non-numeric wide-format columns will be dropped before forecasting: "
            + ", ".join(dropped_columns)
        )
    if not numeric_columns:
        report.errors.append(
            "No forecasting-safe numeric columns remain after excluding the time column "
            "and dropping non-numeric wide-format columns."
        )


def get_wide_numeric_feature_columns(
    dataset_path: str,
    time_col: str,
    nrows: int = 256,
) -> Tuple[List[str], List[str]]:
    """Return runtime-compatible numeric forecasting columns for a wide CSV."""
    data = pd.read_csv(dataset_path, nrows=nrows)
    return _extract_wide_numeric_columns(data, time_col)


def _extract_wide_numeric_columns(
    data: pd.DataFrame,
    time_col: str,
) -> Tuple[List[str], List[str]]:
    numeric_columns: List[str] = []
    dropped_columns: List[str] = []

    for col in data.columns:
        if col == time_col:
            continue

        series = data[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(col)
            continue

        normalized = series
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            normalized = series.astype(str).str.replace(",", "", regex=False).str.strip()
            normalized = normalized.mask(series.isna(), other=pd.NA)
            normalized = normalized.replace({"": pd.NA})

        converted = pd.to_numeric(normalized, errors="coerce")
        non_empty_mask = normalized.notna() if hasattr(normalized, "notna") else series.notna()
        if non_empty_mask.any() and converted[non_empty_mask].notna().all():
            numeric_columns.append(col)
        else:
            dropped_columns.append(col)

    return numeric_columns, dropped_columns


def _try_parse_time(val: str) -> Optional[datetime]:
    """Try multiple datetime formats. Returns datetime or None."""
    from datetime import datetime as dt

    # Common formats in TFB datasets
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return dt.strptime(val.strip(), fmt)
        except ValueError:
            continue
    # Try pandas if available
    try:
        import pandas as pd
        ts = pd.to_datetime(val.strip())
        if ts is not pd.NaT:
            return ts.to_pydatetime()
    except Exception:
        pass
    return None


def format_report(report: ValidationReport) -> str:
    """Human-readable validation report."""
    lines = [
        "=" * 60,
        "Config Validation Report",
        "=" * 60,
        f"Valid: {report.valid}",
        f"Dataset: {report.intent.get('dataset_path', '?')}",
        f"Columns: {report.column_names}",
        f"Rows: {report.row_count}",
        f"Detected frequency: {report.detected_frequency or 'unknown'}",
        (
            f"Frequency confidence: {report.detected_frequency_confidence:.2f}"
            if report.detected_frequency_confidence is not None
            else "Frequency confidence: unknown"
        ),
        f"Frequency method: {report.detected_frequency_method or 'unknown'}",
        f"Time range: {report.time_min or '?'} → {report.time_max or '?'}",
        f"Suggested input_chunk_length: {report.suggested_input_chunk_length}",
        "",
    ]
    if report.errors:
        lines.append("ERRORS:")
        for e in report.errors:
            lines.append(f"  [ERR] {e}")
        lines.append("")
    if report.warnings:
        lines.append("WARNINGS:")
        for w in report.warnings:
            lines.append(f"  [WARN] {w}")
        lines.append("")
    if report.valid and not report.warnings:
        lines.append("[OK] All checks passed.")
    return "\n".join(lines)
