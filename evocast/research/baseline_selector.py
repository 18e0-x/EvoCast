from __future__ import annotations

from typing import Any, Dict, List, Tuple

from evocast.research.baseline_candidate_curator import curate_baseline_candidates
from evocast.research.baseline_candidate_pool import build_baseline_candidate_pool


def select_baseline_candidates(
    *,
    registry: List[Dict[str, Any]],
    config_data: Dict[str, Any],
    candidate_count: int,
    registry_pool_size: int,
    initial_seeds: List[str] | None = None,
    preferred_families: List[str] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pool, pool_rejections = build_baseline_candidate_pool(
        registry=registry,
        config_data=config_data,
        max_pool_size=registry_pool_size,
    )
    if not pool:
        return [], {
            "strategy": "deterministic_curator",
            "candidate_count": max(1, int(candidate_count or 1)),
            "selected": [],
            "rejected": pool_rejections,
            "status": "empty_pool",
        }
    selected, report = curate_baseline_candidates(
        pool=pool,
        candidate_count=candidate_count,
        initial_seeds=initial_seeds,
        preferred_families=preferred_families,
    )
    report["status"] = "ok" if selected else "empty_selection"
    report["pool_size"] = len(pool)
    report["pool_rejected"] = pool_rejections
    return selected, report
