"""Task-local source of truth for time, token, and API-cost accounting."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Dict, Iterator, List, Tuple

from evocast.domain.knowledge_paths import task_knowledge_dir


LEDGER_FILE = "cost_ledger.jsonl"


def _now() -> str:
    return datetime.now().isoformat()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ledger_path(base_dir: str, task_id: str) -> Path:
    return task_knowledge_dir(base_dir, task_id) / LEDGER_FILE


def append_cost_event(base_dir: str, task_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
    """Append one immutable accounting event and return its normalized form."""
    path = ledger_path(base_dir, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event or {})
    payload.setdefault("schema_version", "cost_ledger_event_v1")
    payload.setdefault("event_id", uuid.uuid4().hex)
    payload.setdefault("task_id", task_id)
    payload.setdefault("recorded_at", _now())
    payload.setdefault("status", "completed")
    payload.setdefault("kind", "stage")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    return payload


def load_cost_events(base_dir: str, task_id: str) -> List[Dict[str, Any]]:
    path = ledger_path(base_dir, task_id)
    if not path.is_file():
        return []
    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def estimate_llm_cost(provider_config: Dict[str, Any], model: str, usage: Dict[str, Any] | None) -> Dict[str, Any]:
    """Calculate the configured provider price for one returned usage payload."""
    usage = usage if isinstance(usage, dict) else {}
    pricing = provider_config.get("pricing") if isinstance(provider_config.get("pricing"), dict) else {}
    models = pricing.get("models") if isinstance(pricing.get("models"), dict) else {}
    model_pricing = models.get(str(model or "").lower()) or models.get("default") or pricing
    tiers = [item for item in list(model_pricing.get("tiers") or pricing.get("tiers") or []) if isinstance(item, dict)]
    prompt_tokens = int(_number(usage.get("prompt_tokens")))
    tier = next(
        (item for item in tiers if item.get("max_input_tokens") is None or prompt_tokens <= int(_number(item.get("max_input_tokens")))),
        tiers[-1] if tiers else {},
    )
    cache_hit = int(_number(usage.get("prompt_cache_hit_tokens") or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")))
    cache_miss_raw = usage.get("prompt_cache_miss_tokens")
    cache_miss = int(_number(cache_miss_raw)) if cache_miss_raw is not None else max(prompt_tokens - cache_hit, 0)
    completion = int(_number(usage.get("completion_tokens")))
    multiplier = _number((pricing.get("service_tier_multiplier") or {}).get(pricing.get("service_tier")), 1.0)
    cache_hit_cost = cache_hit * _number(tier.get("cache_read")) * multiplier / 1_000_000
    cache_miss_cost = cache_miss * _number(tier.get("input")) * multiplier / 1_000_000
    output_cost = completion * _number(tier.get("output")) * multiplier / 1_000_000
    return {
        "currency": pricing.get("currency") or "USD",
        "cache_hit_cost": cache_hit_cost,
        "cache_miss_input_cost": cache_miss_cost,
        "output_cost": output_cost,
        "total_cost": cache_hit_cost + cache_miss_cost + output_cost,
    }


@contextmanager
def cost_span(
    base_dir: str,
    task_id: str,
    *,
    stage: str,
    round_id: str = "",
    attempt_id: str = "",
    parent_id: str = "",
    kind: str = "stage",
    extra: Dict[str, Any] | None = None,
) -> Iterator[str]:
    """Record one measured synchronous stage, including failures."""
    event_id = uuid.uuid4().hex
    started_at = _now()
    started = time.monotonic()
    try:
        yield event_id
    except BaseException as exc:
        append_cost_event(
            base_dir,
            task_id,
            {
                "event_id": event_id,
                "parent_id": parent_id,
                "round_id": round_id,
                "attempt_id": attempt_id,
                "stage": stage,
                "kind": kind,
                "status": "failed",
                "started_at": started_at,
                "finished_at": _now(),
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                **dict(extra or {}),
            },
        )
        raise
    else:
        append_cost_event(
            base_dir,
            task_id,
            {
                "event_id": event_id,
                "parent_id": parent_id,
                "round_id": round_id,
                "attempt_id": attempt_id,
                "stage": stage,
                "kind": kind,
                "status": "success",
                "started_at": started_at,
                "finished_at": _now(),
                "elapsed_seconds": round(time.monotonic() - started, 6),
                **dict(extra or {}),
            },
        )


def record_execution_cost(base_dir: str, task_id: str, observation: Dict[str, Any]) -> str:
    """Write a training, evaluation, smoke, or seed-eval measurement to the ledger."""
    payload = dict(observation or {})
    elapsed = payload.get("elapsed_seconds_total")
    if elapsed is None:
        elapsed = payload.get("elapsed_seconds")
    event = append_cost_event(
        base_dir,
        task_id,
        {
            "kind": "execution",
            "stage": str(payload.get("evaluation_stage") or payload.get("tier") or payload.get("budget") or "execution"),
            "round_id": str(payload.get("round_id") or ""),
            "attempt_id": str(payload.get("run_id") or payload.get("node_id") or ""),
            "status": str(payload.get("status") or "completed"),
            "elapsed_seconds": elapsed,
            "fit_time_seconds": payload.get("fit_time"),
            "inference_time_seconds": payload.get("inference_time"),
            "details": payload,
        },
    )
    return str(ledger_path(base_dir, task_id))


def tracked_stage(
    stage: str,
    identity: Callable[..., Tuple[str, str, str, str]],
    *,
    kind: str = "stage",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a live pipeline function with one measured ledger span."""
    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            base_dir, task_id, round_id, attempt_id = identity(*args, **kwargs)
            with cost_span(
                base_dir,
                task_id,
                stage=stage,
                round_id=round_id,
                attempt_id=attempt_id,
                kind=kind,
            ):
                return func(*args, **kwargs)
        return wrapped
    return decorate
