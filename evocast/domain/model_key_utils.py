"""Utilities for resolving user-facing model names to registry keys."""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, Iterable, List, Tuple


def compact_model_key(value: str) -> str:
    """Normalize spelling/case separators without erasing meaningful letters."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def resolve_model_key(value: str, registry: Iterable[Dict[str, Any]]) -> Tuple[str, str]:
    """Resolve a user-provided model name to a canonical registry key.

    Returns ``(canonical_key, resolution_kind)``.  The kind is one of:
    ``exact``, ``compact``, ``fuzzy_unique``, or ``missing``.
    """
    raw = str(value or "").strip()
    if not raw:
        return "", "missing"

    specs = list(registry or [])
    exact_aliases: Dict[str, str] = {}
    compact_aliases: Dict[str, List[str]] = {}
    for spec in specs:
        key = str(spec.get("model_key") or "").strip()
        import_path = str(spec.get("import_path") or "").strip()
        if not key:
            continue
        aliases = {key, key.lower()}
        if import_path:
            aliases.update({import_path, import_path.rsplit(".", 1)[-1]})
        for alias in aliases:
            if alias:
                exact_aliases[alias] = key
                compact_aliases.setdefault(compact_model_key(alias), []).append(key)

    if raw in exact_aliases:
        return exact_aliases[raw], "exact"
    if raw.lower() in exact_aliases:
        return exact_aliases[raw.lower()], "exact"

    compact = compact_model_key(raw)
    compact_matches = sorted(set(compact_aliases.get(compact) or []))
    if len(compact_matches) == 1:
        return compact_matches[0], "compact"

    registry_compacts = sorted(compact_aliases)
    close = difflib.get_close_matches(compact, registry_compacts, n=3, cutoff=0.88)
    if close:
        best_keys = sorted(set(compact_aliases.get(close[0]) or []))
        second_score = difflib.SequenceMatcher(None, compact, close[1]).ratio() if len(close) > 1 else 0.0
        best_score = difflib.SequenceMatcher(None, compact, close[0]).ratio()
        if len(best_keys) == 1 and best_score >= 0.88 and best_score - second_score >= 0.04:
            return best_keys[0], "fuzzy_unique"

    return "", "missing"
