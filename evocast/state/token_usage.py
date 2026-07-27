"""Summarize per-call LLM/API token usage for a task run."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from evocast.domain.execution_ids import format_research_id
from evocast.domain.knowledge_paths import task_knowledge_dir, task_runs_dir
from evocast.harness.api_client import resolve_task_api_config_path
from evocast.state.cost_ledger import ledger_path, load_cost_events
from evocast.state.domain_store import load_task_config
_VARIANT_RE = re.compile(r"variant_(?P<variant>\d+)")
_REPAIR_RE = re.compile(r"repair(?P<attempt>\d+)$")


def _coerce_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _provider_config_path(task_id: str, base_dir: str) -> Path:
    task_config = load_task_config(base_dir, task_id)
    configured = task_config.get("api_config") or task_config.get("provider_config") or task_config.get("llm_provider_config")
    if not configured:
        root = Path(base_dir)
        runtime = root
        repo_like_root = runtime.parent if runtime.name == ".evocast" else root
        legacy_minimax = repo_like_root / "evocast" / "configs" / "providers" / "minimax.yaml"
        if legacy_minimax.is_file():
            return legacy_minimax
    return resolve_task_api_config_path(base_dir=base_dir, task_id=task_id)


def _load_pricing(task_id: str, base_dir: str) -> Dict[str, Any]:
    provider_config = _read_yaml(_provider_config_path(task_id, base_dir))
    pricing = provider_config.get("pricing") if isinstance(provider_config.get("pricing"), dict) else {}
    if not pricing:
        return {}
    result = dict(pricing)
    result["provider"] = provider_config.get("provider") or result.get("provider")
    result["configured_models"] = provider_config.get("models")
    return result


def _model_price_map(pricing: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    models = pricing.get("models") if isinstance(pricing.get("models"), dict) else {}
    return {str(name).lower(): dict(value) for name, value in models.items() if isinstance(value, dict)}


def _pricing_for_entry(entry: Dict[str, Any], pricing: Dict[str, Any]) -> Dict[str, Any]:
    model_prices = _model_price_map(pricing)
    if not model_prices:
        return pricing
    model = str(entry.get("model") or "").lower()
    if model in model_prices:
        selected = dict(model_prices[model])
    elif "default" in model_prices:
        selected = dict(model_prices["default"])
    else:
        return {}
    merged = {key: value for key, value in pricing.items() if key != "models"}
    merged.update(selected)
    merged.setdefault("model", entry.get("model"))
    merged.setdefault("provider", entry.get("provider") or pricing.get("provider"))
    return merged


def _pricing_tier(pricing: Dict[str, Any], prompt_tokens: int) -> Dict[str, Any]:
    tiers = [item for item in list(pricing.get("tiers") or []) if isinstance(item, dict)]
    if not tiers:
        return {}
    for tier in tiers:
        limit = tier.get("max_input_tokens")
        if limit is None or prompt_tokens <= _coerce_int(limit):
            return tier
    return tiers[-1]


def _entry_cost(entry: Dict[str, Any], pricing: Dict[str, Any]) -> Dict[str, Any]:
    if not pricing:
        return {}
    prompt_tokens = _coerce_int(entry.get("prompt_tokens"))
    cache_hit = _coerce_int(entry.get("prompt_cache_hit_tokens"))
    cache_miss = _coerce_int(entry.get("prompt_cache_miss_tokens"))
    output_tokens = _coerce_int(entry.get("completion_tokens"))
    tier = _pricing_tier(pricing, prompt_tokens)
    multiplier = _coerce_float((pricing.get("service_tier_multiplier") or {}).get(pricing.get("service_tier")), 1.0)
    cache_hit_cost = cache_hit * _coerce_float(tier.get("cache_read")) * multiplier / 1_000_000
    cache_miss_cost = cache_miss * _coerce_float(tier.get("input")) * multiplier / 1_000_000
    output_cost = output_tokens * _coerce_float(tier.get("output")) * multiplier / 1_000_000
    return {
        "currency": pricing.get("currency") or "USD",
        "cache_hit_cost": cache_hit_cost,
        "cache_miss_input_cost": cache_miss_cost,
        "output_cost": output_cost,
        "total_cost": cache_hit_cost + cache_miss_cost + output_cost,
    }


def _cost_summary(entries: List[Dict[str, Any]], pricing: Dict[str, Any]) -> Dict[str, Any]:
    if not pricing:
        return {}
    totals = {
        "cache_hit_cost": 0.0,
        "cache_miss_input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
    }
    by_model: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        entry_pricing = _pricing_for_entry(entry, pricing)
        cost = _entry_cost(entry, entry_pricing)
        entry["cost"] = cost
        for key in totals:
            totals[key] += _coerce_float(cost.get(key))
        model_key = str(entry.get("model") or entry_pricing.get("model") or "unknown")
        model_bucket = by_model.setdefault(
            model_key,
            {
                "model": model_key,
                "provider": entry.get("provider") or entry_pricing.get("provider") or pricing.get("provider"),
                "cache_hit_tokens": 0,
                "cache_miss_input_tokens": 0,
                "output_tokens": 0,
                "cache_hit_cost": 0.0,
                "cache_miss_input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
            },
        )
        model_bucket["cache_hit_tokens"] += _coerce_int(entry.get("prompt_cache_hit_tokens"))
        model_bucket["cache_miss_input_tokens"] += _coerce_int(entry.get("prompt_cache_miss_tokens"))
        model_bucket["output_tokens"] += _coerce_int(entry.get("completion_tokens"))
        for key in totals:
            model_bucket[key] += _coerce_float(cost.get(key))
    return {
        "currency": pricing.get("currency") or "USD",
        "unit": pricing.get("unit") or "per_1m_tokens",
        "provider": pricing.get("provider"),
        "configured_models": pricing.get("configured_models"),
        "service_tier": pricing.get("service_tier"),
        "source": pricing.get("source"),
        "source_checked_at": pricing.get("source_checked_at"),
        "cache_hit_tokens": sum(_coerce_int(item.get("prompt_cache_hit_tokens")) for item in entries),
        "cache_miss_input_tokens": sum(_coerce_int(item.get("prompt_cache_miss_tokens")) for item in entries),
        "output_tokens": sum(_coerce_int(item.get("completion_tokens")) for item in entries),
        "by_model": list(by_model.values()),
        **totals,
    }


def _blank_bucket() -> Dict[str, int]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "reasoning_tokens": 0,
    }


def _add_usage(bucket: Dict[str, int], usage: Dict[str, Any]) -> None:
    bucket["calls"] += 1
    bucket["prompt_tokens"] += _coerce_int(usage.get("prompt_tokens"))
    bucket["completion_tokens"] += _coerce_int(usage.get("completion_tokens"))
    bucket["total_tokens"] += _coerce_int(usage.get("total_tokens"))
    bucket["prompt_cache_hit_tokens"] += _coerce_int(
        usage.get("prompt_cache_hit_tokens")
        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    )
    prompt_tokens = _coerce_int(usage.get("prompt_tokens"))
    cache_hit = _coerce_int(
        usage.get("prompt_cache_hit_tokens")
        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    )
    cache_miss = usage.get("prompt_cache_miss_tokens")
    if cache_miss is None and prompt_tokens:
        cache_miss = max(prompt_tokens - cache_hit, 0)
    bucket["prompt_cache_miss_tokens"] += _coerce_int(cache_miss)
    bucket["reasoning_tokens"] += _coerce_int(
        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    )


def _bucket_to_dict(name: str, bucket: Dict[str, int]) -> Dict[str, Any]:
    data = {"name": name}
    data.update(bucket)
    return data


def _stage_family(stage_label: str) -> str:
    if stage_label.startswith("implementer_variant_"):
        if "_repair" in stage_label:
            return "repair"
        return "implementer"
    if stage_label.startswith("repair_"):
        return "repair"
    if stage_label.startswith("baseline_ablation_"):
        return "baseline_ablation"
    if stage_label.startswith("round_review"):
        return "round_review"
    return stage_label


def _extract_variant(stage_label: str) -> str | None:
    match = _VARIANT_RE.search(stage_label)
    if not match:
        return None
    return f"variant_{match.group('variant')}"


def _extract_repair(stage_label: str) -> str | None:
    match = _REPAIR_RE.search(stage_label)
    if not match:
        return None
    return f"repair{match.group('attempt')}"


def build_token_usage_summary(task_id: str, base_dir: str) -> Dict[str, Any]:
    totals = _blank_bucket()
    by_round: Dict[str, Dict[str, int]] = defaultdict(_blank_bucket)
    by_stage: Dict[str, Dict[str, int]] = defaultdict(_blank_bucket)
    by_variant: Dict[str, Dict[str, int]] = defaultdict(_blank_bucket)
    by_repair: Dict[str, Dict[str, int]] = defaultdict(_blank_bucket)
    entries: List[Dict[str, Any]] = []

    for payload in load_cost_events(base_dir, task_id):
        if payload.get("kind") != "llm_api":
            continue
        usage = payload.get("usage") or {}
        if not isinstance(usage, dict) or not usage:
            continue
        round_value = payload.get("round") if payload.get("round") is not None else payload.get("round_id")
        round_num = _coerce_int(round_value)
        stage_label = str(payload.get("stage") or "").strip()
        if not stage_label:
            continue
        family = _stage_family(stage_label)
        variant = _extract_variant(stage_label)
        repair = _extract_repair(stage_label)

        _add_usage(totals, usage)
        _add_usage(by_round[str(round_num)], usage)
        _add_usage(by_stage[family], usage)
        if variant:
            _add_usage(by_variant[variant], usage)
        if repair:
            round_label = format_research_id(round_num) if round_num > 0 else "task"
            repair_key = f"{round_label}_{variant or 'no_variant'}_{repair}"
            _add_usage(by_repair[repair_key], usage)

        entries.append({
            "event_id": payload.get("event_id"),
            "round": round_num,
            "stage_label": stage_label,
            "stage_family": family,
            "variant": variant,
            "repair_attempt": repair,
            "provider": payload.get("provider"),
            "model": payload.get("model"),
            "prompt_tokens": _coerce_int(usage.get("prompt_tokens")),
            "completion_tokens": _coerce_int(usage.get("completion_tokens")),
            "total_tokens": _coerce_int(usage.get("total_tokens")),
            "prompt_cache_hit_tokens": _coerce_int(
                usage.get("prompt_cache_hit_tokens")
                or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            ),
            "prompt_cache_miss_tokens": _coerce_int(
                usage.get("prompt_cache_miss_tokens")
                if usage.get("prompt_cache_miss_tokens") is not None
                else max(
                    _coerce_int(usage.get("prompt_tokens"))
                    - _coerce_int(
                        usage.get("prompt_cache_hit_tokens")
                        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    ),
                    0,
                )
            ),
            "reasoning_tokens": _coerce_int(
                (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            ),
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "status": payload.get("status"),
            "cost": payload.get("cost") if isinstance(payload.get("cost"), dict) else {},
        })

    entries.sort(key=lambda row: (row["round"], row["stage_label"], str(row.get("event_id") or "")))
    pricing = _load_pricing(task_id, base_dir)
    cost = _cost_summary(entries, pricing)
    return {
        "task_id": task_id,
        "generated_at": datetime.now().isoformat(),
        "ledger_path": str(ledger_path(base_dir, task_id)),
        "totals": totals,
        "cost": cost,
        "by_round": [_bucket_to_dict(name, by_round[name]) for name in sorted(by_round, key=lambda v: int(v))],
        "by_stage": [_bucket_to_dict(name, by_stage[name]) for name in sorted(by_stage)],
        "by_variant": [_bucket_to_dict(name, by_variant[name]) for name in sorted(by_variant)],
        "by_repair": [_bucket_to_dict(name, by_repair[name]) for name in sorted(by_repair)],
        "entries": entries,
    }


def write_token_usage_summary(task_id: str, base_dir: str) -> str:
    summary = build_token_usage_summary(task_id, base_dir)
    output_path = str(task_runs_dir(base_dir, task_id) / "logs" / "token_usage_summary.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return output_path
