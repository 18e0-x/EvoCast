"""Measured execution-cost views derived from the task cost ledger."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from evocast.state.cost_ledger import ledger_path, load_cost_events, record_execution_cost


SUCCESS_STATUS = "success"
OOM_STATUS = "oom"
TIMEOUT_STATUS = "timeout"
ERROR_STATUS = "error"
DRY_RUN_STATUS = "dry_run"


def observation_path(base_dir: str, task_id: str) -> str:
    return str(ledger_path(base_dir, task_id))


def load_cost_observations(base_dir: str, task_id: str) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for event in load_cost_events(base_dir, task_id):
        if event.get("kind") != "execution":
            continue
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        row = dict(details)
        row.setdefault("status", event.get("status"))
        row.setdefault("elapsed_seconds", event.get("elapsed_seconds"))
        row.setdefault("elapsed_seconds_total", event.get("elapsed_seconds"))
        row.setdefault("fit_time", event.get("fit_time_seconds"))
        row.setdefault("inference_time", event.get("inference_time_seconds"))
        row.setdefault("evaluation_stage", event.get("stage"))
        observations.append(row)
    return observations


def latest_observation_map(
    base_dir: str,
    task_id: str,
    *,
    include_tiers: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return the chronologically last observation per model.

    This relies on append-only JSONL order: later lines overwrite earlier ones.
    If callers ever rewrite or reorder the file, "latest" semantics no longer
    hold automatically.
    """
    rows = load_cost_observations(base_dir, task_id)
    allowed = set(include_tiers or [])
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if allowed and row.get("tier") not in allowed:
            continue
        model_key = row.get("model_key")
        if not model_key:
            continue
        latest[model_key] = row
    return latest


def successful_elapsed_values(
    observations: Iterable[Dict[str, Any]],
    *,
    min_elapsed_seconds: float = 1e-9,
) -> List[float]:
    values: List[float] = []
    for row in observations:
        if row.get("status") != SUCCESS_STATUS:
            continue
        elapsed = row.get("elapsed_seconds_total")
        if elapsed is None:
            elapsed = row.get("elapsed_seconds")
        try:
            value = float(elapsed)
        except (TypeError, ValueError):
            continue
        if value >= min_elapsed_seconds:
            values.append(value)
    values.sort()
    return values


def percentile(sorted_values: List[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q = min(1.0, max(0.0, q))
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def static_cost_penalty(cost_level: int) -> float:
    cost = int(cost_level) if cost_level is not None else 2
    cost = max(1, min(3, cost))
    return (cost - 1) / 2.0


def measured_cost_penalty(
    observation: Optional[Dict[str, Any]],
    success_elapsed_samples: List[float],
    *,
    fallback_cost_level: int = 2,
    min_success_samples: int = 3,
) -> float:
    if not observation:
        return static_cost_penalty(fallback_cost_level)

    status = str(observation.get("status", "")).lower()
    if status in {OOM_STATUS, TIMEOUT_STATUS}:
        return 1.0
    if status in {ERROR_STATUS, DRY_RUN_STATUS}:
        return static_cost_penalty(fallback_cost_level)
    if status != SUCCESS_STATUS:
        return static_cost_penalty(fallback_cost_level)

    if len(success_elapsed_samples) < min_success_samples:
        return static_cost_penalty(fallback_cost_level)

    elapsed = observation.get("elapsed_seconds_total")
    if elapsed is None:
        elapsed = observation.get("elapsed_seconds")
    try:
        t = float(elapsed)
    except (TypeError, ValueError):
        return static_cost_penalty(fallback_cost_level)

    p50 = percentile(success_elapsed_samples, 0.50)
    p95 = percentile(success_elapsed_samples, 0.95)
    if p50 is None or p95 is None:
        return static_cost_penalty(fallback_cost_level)
    if p95 <= p50 + 1e-9:
        return 0.0 if t <= p50 else 1.0
    return max(0.0, min(1.0, (t - p50) / (p95 - p50)))


def summarize_cost_state(
    observations: Iterable[Dict[str, Any]],
    *,
    cheap_tiers: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    tiers = set(cheap_tiers or ["cheap"])
    total_elapsed = 0.0
    cheap_elapsed = 0.0
    counts: Dict[str, int] = {}
    for row in observations:
        status = str(row.get("status", "")).lower() or "unknown"
        counts[status] = counts.get(status, 0) + 1
        try:
            elapsed = float(row.get("elapsed_seconds_total", row.get("elapsed_seconds", 0.0)) or 0.0)
        except (TypeError, ValueError):
            elapsed = 0.0
        total_elapsed += elapsed
        if row.get("tier") in tiers:
            cheap_elapsed += elapsed
    return {
        "status_counts": counts,
        "total_elapsed_seconds": total_elapsed,
        "cheap_elapsed_seconds": cheap_elapsed,
    }
