from __future__ import annotations

from typing import Any, Dict, List, Tuple


DEFAULT_PREFERRED_FAMILIES = ["linear", "transformer", "mlp", "cnn", "gnn", "others"]


def _model_key(row: Dict[str, Any]) -> str:
    return str((row.get("brief") or {}).get("model_key") or "")


def _family(row: Dict[str, Any]) -> str:
    return str((row.get("brief") or {}).get("family") or "unknown")


def curate_baseline_candidates(
    *,
    pool: List[Dict[str, Any]],
    candidate_count: int,
    initial_seeds: List[str] | None = None,
    preferred_families: List[str] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    limit = max(1, int(candidate_count or 1))
    # Kept as a compatibility argument for old policies. Automatic selection
    # is intentionally driven only by family and alphabetical model order.
    ignored_seeds = [str(item).strip() for item in list(initial_seeds or []) if str(item).strip()]
    families = [str(item).strip() for item in list(preferred_families or DEFAULT_PREFERRED_FAMILIES) if str(item).strip()]
    if not families:
        families = list(DEFAULT_PREFERRED_FAMILIES)

    selected: List[Dict[str, Any]] = []
    selected_keys: set[str] = set()
    selected_report: List[Dict[str, Any]] = []
    rejected_report: List[Dict[str, Any]] = []

    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for row in pool:
        family = _family(row)
        by_family.setdefault(family, []).append(row)
    for rows in by_family.values():
        rows.sort(key=lambda row: _model_key(row).casefold())

    # Round-robin gives every configured family its alphabetically first model
    # before any family receives its second model.
    round_index = 0
    while len(selected) < limit:
        added_in_round = False
        for family in families:
            rows = by_family.get(family, [])
            if round_index >= len(rows):
                continue
            row = rows[round_index]
            key = _model_key(row)
            if key in selected_keys:
                continue
            selected.append(row["spec"])
            selected_keys.add(key)
            selected_report.append(
                {
                    "model_key": key,
                    "reason": [
                        "selected_by_family_round_robin",
                        f"family={family}",
                        f"family_round={round_index + 1}",
                        "ordering=family_order,alphabetical_model_key",
                    ],
                    "brief": row["brief"],
                }
            )
            added_in_round = True
            if len(selected) >= limit:
                break
        if not added_in_round:
            break
        round_index += 1

    for row in pool:
        key = str(row["brief"].get("model_key") or "")
        if key in selected_keys:
            continue
        rejected_report.append(
            {
                "model_key": key,
                "reason": "not_selected_after_candidate_limit",
                "brief": row["brief"],
            }
        )

    report = {
        "strategy": "deterministic_curator",
        "candidate_count": limit,
        "preferred_families": families,
        "initial_seeds": [],
        "ignored_legacy_initial_seeds": ignored_seeds,
        "selection_order": "family_order_round_robin_then_alphabetical_model_key",
        "selected": selected_report,
        "rejected": rejected_report,
    }
    return selected, report
