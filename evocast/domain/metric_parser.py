"""Deterministic TFB metric parser.

Reads TFB result records from paths returned by pipeline().
Uses ts_benchmark.recording to read compressed and uncompressed CSV record files.

TFB produces two record file formats:
  1. Record files (.csv.tar.gz): wide format with metric columns.
  2. Report files (test_report_*.csv): melted format with columns
     strategy_args, metric_name, <model_info_str>.

This parser handles both formats and keeps only EvoCast's canonical metrics.
"""

import json
import os
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from evocast.domain.metric_semantics import (
    PUBLIC_METRICS,
    canonicalize_metric_values,
    validate_public_metric_name,
)
from ts_benchmark.recording import read_record_file, find_record_files

# Known TFB metric columns (wide-format records).
KNOWN_METRIC_COLS = set(PUBLIC_METRICS)

# Columns in wide-format records that are NOT metrics.
NON_METRIC_COLS = {
    "model_name", "strategy_args", "model_params", "file_name",
    "fit_time", "inference_time", "actual_data", "inference_data",
    "log_info", "eval_res", "pred_data", "forecast_execution_path",
}


def _extract_log_errors(df: pd.DataFrame) -> List[str]:
    """Extract traceback/error text embedded in TFB record log columns."""
    errors: List[str] = []
    for col in ("log_info", "eval_res"):
        if col not in df.columns:
            continue
        for value in df[col].dropna():
            text = str(value)
            if "Traceback" in text or "RuntimeError" in text or "Exception" in text:
                errors.append(text[:3000])
    return errors


def _resolve_record_paths(log_paths: List[str], prefer_record: bool = True) -> List[str]:
    """Given the pipeline return paths, find all record files.

    Pipeline returns file paths (not directories). We collect them directly
    and also search for any adjacent report files.

    Args:
        log_paths: Paths returned by pipeline().
        prefer_record: If True, prefer .tar.gz record files over report files.

    Returns:
        Deduplicated list of absolute file paths.
    """
    record_files = []
    for p in log_paths:
        if os.path.isfile(p):
            record_files.append(p)
        elif os.path.isdir(p):
            record_files.extend(find_record_files(p))

    # Separate into record files (.tar.gz) and report files (.csv)
    records = [f for f in record_files if f.endswith(".tar.gz")]
    reports = [f for f in record_files if "test_report" in os.path.basename(f)]

    # Deduplicate: prefer record files for the same model
    if prefer_record and records:
        used = records
        # Only add report files that don't overlap with record files
        record_basenames = {os.path.basename(f).split(".csv")[0] for f in records}
        for rp in reports:
            base = os.path.basename(rp).replace("test_report.", "").split(".csv")[0]
            if base not in record_basenames:
                used.append(rp)
    else:
        # Deduplicate by basename
        seen = set()
        used = []
        for f in record_files:
            base = os.path.basename(f)
            if base not in seen:
                seen.add(base)
                used.append(f)

    return used


def parse_single_record(file_path: str) -> pd.DataFrame:
    """Parse a single TFB record file into a DataFrame.

    Delegates to ts_benchmark.recording.read_record_file which handles
    .csv and .csv.tar.gz (and other compressed formats).
    """
    return read_record_file(file_path)


def _is_wide_format(df: pd.DataFrame) -> bool:
    """Detect if a DataFrame is in TFB wide record format.

    Wide format has model_name, strategy_args, model_params columns
    plus individual metric columns.
    """
    cols = set(df.columns)
    return "model_name" in cols and "mse" in cols


def _is_melted_format(df: pd.DataFrame) -> bool:
    """Detect if a DataFrame is in TFB melted report format.

    Melted format has strategy_args, metric_name, and a model-info column.
    """
    cols = set(df.columns)
    return "metric_name" in cols


def _extract_wide_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Extract metrics from a wide-format TFB record DataFrame.

    Columns: model_name, strategy_args, model_params, mae, mse, ...,
    file_name, fit_time, inference_time, etc.

    Returns mean across all rows for each metric column.
    """
    # Find metric columns: numeric columns minus known non-metric columns
    metric_cols = []
    for col in df.columns:
        if col in NON_METRIC_COLS:
            continue
        if str(col).startswith("evocast_"):
            continue
        if col in KNOWN_METRIC_COLS:
            metric_cols.append(col)
        elif df[col].dtype in (np.float64, np.float32, np.int64, np.int32, float, int):
            # Check the column name looks like a metric
            metric_cols.append(col)

    if not metric_cols:
        return {}

    metrics = {}
    for col in metric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        val = series.mean()
        if hasattr(val, "item"):
            val = val.item()
        metrics[col] = float(val)

    return metrics


def _extract_runtime_stats(df: pd.DataFrame) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for col in ("fit_time", "inference_time"):
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue
        val = series.mean()
        if hasattr(val, "item"):
            val = val.item()
        stats[col] = float(val)
    return stats


def _extract_melted_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Extract metrics from a melted TFB report DataFrame.

    Columns: strategy_args, metric_name, <model_info_str>
    where <model_info_str> contains the metric values.

    If a 'metric_value' column is present, it is used as the authoritative source.
    Otherwise falls back to the first non-standard column.

    Groups by metric_name and computes the mean value.
    """
    # Prefer explicit metric_value column if present
    if "metric_value" in df.columns:
        value_col = "metric_value"
    else:
        value_cols = [c for c in df.columns if c not in ("strategy_args", "metric_name")]
        if not value_cols:
            return {}
        value_col = value_cols[0]

    metrics = {}
    for metric_name, group in df.groupby("metric_name"):
        series = pd.to_numeric(group[value_col], errors="coerce").dropna()
        if len(series) == 0:
            continue
        val = series.mean()
        if hasattr(val, "item"):
            val = val.item()
        if str(metric_name) in KNOWN_METRIC_COLS:
            metrics[str(metric_name)] = float(val)

    return metrics


def parse_log_paths(log_paths: List[str]) -> List[Dict[str, Any]]:
    """Parse all record files from pipeline return paths.

    Returns a list of parsed record dicts, each containing:
      - source_path: str
      - format: "wide" or "melted"
      - metrics: {metric_name: value}
      - model_name: str (if available)
      - parser_error: str (if the file failed to parse)
    """
    record_files = _resolve_record_paths(log_paths)
    results = []

    for fp in record_files:
        try:
            df = parse_single_record(fp)
            if _is_wide_format(df):
                metrics = _extract_wide_metrics(df)
                model = df["model_name"].iloc[0] if "model_name" in df.columns else "unknown"
                record = {
                    "source_path": fp,
                    "format": "wide",
                    "metrics": metrics,
                    "model_name": model,
                    "runtime_stats": _extract_runtime_stats(df),
                }
                log_errors = _extract_log_errors(df)
                if log_errors:
                    record["record_errors"] = log_errors
                results.append(record)
            elif _is_melted_format(df):
                metrics = _extract_melted_metrics(df)
                record = {
                    "source_path": fp,
                    "format": "melted",
                    "metrics": metrics,
                    "model_name": "unknown",
                    "runtime_stats": {},
                }
                log_errors = _extract_log_errors(df)
                if log_errors:
                    record["record_errors"] = log_errors
                results.append(record)
            else:
                results.append({
                    "source_path": fp,
                    "format": "unknown",
                    "metrics": {},
                    "model_name": "unknown",
                    "parser_error": f"Unrecognized record format, columns={list(df.columns)[:10]}",
                })
        except Exception as e:
            results.append({
                "source_path": fp,
                "format": "unknown",
                "metrics": {},
                "model_name": "unknown",
                "parser_error": f"{type(e).__name__}: {str(e)[:500]}",
            })

    return results


def extract_metrics(
    parsed_records: List[Dict[str, Any]],
    aggregate: str = "mean",
) -> Dict[str, float]:
    """Combine metrics from multiple parsed records.

    Args:
        parsed_records: Output from parse_log_paths().
        aggregate: Aggregation method ("mean", "median", or "first").

    Returns:
        Dict of metric_name -> metric_value.
    """
    # Prefer wide-format records (more complete) over melted reports
    wide = [r for r in parsed_records if r["format"] == "wide"]
    melted = [r for r in parsed_records if r["format"] == "melted"]

    # Use wide format if available
    sources = wide if wide else melted
    if not sources:
        return {}

    if aggregate == "first":
        return canonicalize_metric_values(dict(sources[0]["metrics"]))

    # Collect all values for each metric
    all_values: Dict[str, list] = {}
    for rec in sources:
        for k, v in rec["metrics"].items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                all_values.setdefault(k, []).append(v)

    result = {}
    for k, vals in all_values.items():
        if aggregate == "median":
            result[k] = float(np.median(vals))
        else:
            result[k] = float(np.mean(vals))

    return canonicalize_metric_values(result)


def parse_metrics_from_paths(
    log_paths: List[str],
    objective_metric: Optional[str] = None,
    aggregate: str = "mean",
) -> Dict:
    """Main entry point: parse metrics from pipeline return paths.

    Args:
        log_paths: The list of paths returned by pipeline().
        objective_metric: The primary metric to validate exists.
        aggregate: Aggregation method ("mean", "median", "first").

    Returns:
        Dict with keys:
          - metric_values: {metric_name: value}
          - metric_source_paths: list of record files used
          - model_name: str
          - status: "ok" | "warning" | "error"
          - warnings: list of warning strings
          - objective_metric_present: bool
    """
    result: Dict[str, Any] = {
        "metric_values": {},
        "metric_source_paths": [],
        "model_name": "unknown",
        "status": "ok",
        "warnings": [],
        "parser_errors": [],
        "record_errors": [],
        "objective_metric_present": False,
        "runtime_stats": {},
    }

    record_files = _resolve_record_paths(log_paths)
    result["metric_source_paths"] = record_files

    if not record_files:
        result["status"] = "error"
        result["warnings"].append("No TFB record files found from pipeline output")
        return result

    parsed = parse_log_paths(log_paths)
    if not parsed:
        result["status"] = "error"
        result["warnings"].append("Record files found but all failed to parse")
        return result

    # Collect parser errors from individual files
    parser_errors = [p["parser_error"] for p in parsed if "parser_error" in p]
    if parser_errors:
        result["parser_errors"] = parser_errors
        result["warnings"].append(
            f"{len(parser_errors)} record file(s) failed to parse"
        )

    # Only use successfully parsed records for metrics
    valid_parsed = [p for p in parsed if "parser_error" not in p]
    if not valid_parsed:
        result["status"] = "error"
        result["warnings"].append("All record files failed to parse")
        return result

    record_errors = []
    for p in valid_parsed:
        record_errors.extend(p.get("record_errors", []))
    if record_errors:
        result["record_errors"] = record_errors
        result["warnings"].append(
            f"{len(record_errors)} record file(s) contain embedded runtime errors"
        )

    result["model_name"] = valid_parsed[0].get("model_name", "unknown")
    metrics = extract_metrics(valid_parsed, aggregate=aggregate)
    result["metric_values"] = metrics
    runtime_stats: Dict[str, float] = {}
    runtime_keys = {"fit_time", "inference_time"}
    for key in runtime_keys:
        values = [
            p.get("runtime_stats", {}).get(key)
            for p in valid_parsed
            if p.get("runtime_stats", {}).get(key) is not None
        ]
        if values:
            result["runtime_stats"][key] = float(np.mean(values))

    if not metrics:
        result["status"] = "error"
        if record_errors:
            result["warnings"].append("No metrics found because the run failed inside TFB evaluation")
        else:
            result["warnings"].append("No metric columns found in record files")
        return result

    # Check objective metric — exact match only.
    if objective_metric:
        validate_public_metric_name(objective_metric)
        if objective_metric in metrics:
            result["objective_metric_present"] = True
        else:
            result["status"] = "error"
            result["warnings"].append(
                f"Objective metric '{objective_metric}' not found in records. "
                f"Available: {sorted(metrics.keys())}"
            )

    return result


def metrics_to_json(parsed: Dict, output_path: str) -> str:
    """Write parsed metrics to a JSON file.

    Returns the output path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    serializable = {
        "metric_values": parsed["metric_values"],
        "metric_source_paths": parsed["metric_source_paths"],
        "status": parsed["status"],
        "warnings": parsed["warnings"],
        "parser_errors": parsed.get("parser_errors", []),
        "record_errors": parsed.get("record_errors", []),
        "objective_metric_present": parsed["objective_metric_present"],
    }
    # Convert numpy types to native Python
    clean_metrics = {}
    for k, v in serializable["metric_values"].items():
        if hasattr(v, "item"):
            clean_metrics[k] = v.item()
        else:
            clean_metrics[k] = float(v)
    serializable["metric_values"] = clean_metrics

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
    return output_path
