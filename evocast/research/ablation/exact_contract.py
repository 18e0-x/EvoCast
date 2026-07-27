from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from evocast.harness.permissions import normalize_repo_path


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _code_key(value: str) -> str:
    text = re.sub(r"\s+", "", str(value or ""))
    return text.strip()


def _norm(path: str) -> str:
    return normalize_repo_path(str(path or "").replace("\\", "/").strip())


def _patch_lines(patch_text: str, marker: str) -> str:
    lines: list[str] = []
    for line in str(patch_text or "").splitlines():
        if not line.startswith(marker):
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        lines.append(line[1:])
    return "\n".join(lines)


def _patch_result_lines(patch_text: str) -> str:
    lines: list[str] = []
    for line in str(patch_text or "").splitlines():
        if line.startswith(("+++", "---", "diff --git", "index ", "@@")):
            continue
        if line.startswith("+"):
            lines.append(line[1:])
        elif line.startswith(" "):
            lines.append(line[1:])
    return "\n".join(lines)


def _patch_preimage_lines(patch_text: str) -> str:
    lines: list[str] = []
    for line in str(patch_text or "").splitlines():
        if line.startswith(("+++", "---", "diff --git", "index ", "@@")):
            continue
        if line.startswith("-"):
            lines.append(line[1:])
        elif line.startswith(" "):
            lines.append(line[1:])
    return "\n".join(lines)


def _meaningful_anchor_lines(anchor: str) -> list[str]:
    result: list[str] = []
    for line in str(anchor or "").splitlines():
        stripped = line.strip()
        if len(stripped) >= 8 and stripped not in {"else:", "try:", "finally:"}:
            result.append(stripped)
    return result


def compile_exact_ablation_target(
    target: dict[str, Any],
    *,
    repo_dir: str | Path,
    source_checkout: str | Path | None = None,
) -> dict[str, Any]:
    """Return the formal source anchor used by BuildContract ablations."""
    spec = dict((target or {}).get("edit_spec") or {})
    target_file = _norm(str(spec.get("target_file") or ""))
    anchor_text = str(spec.get("anchor_text") or "")
    replacement = str(spec.get("replacement_pseudocode") or "")
    replacement_intent = str(
        spec.get("replacement_intent")
        or (target or {}).get("ablation_intent")
        or (target or {}).get("exact_edit_intent")
        or replacement
        or ""
    ).strip()
    shape_invariant = str(spec.get("shape_invariant_argument") or "").strip()

    errors: list[str] = []
    if not target_file:
        errors.append("missing_edit_spec.target_file")
    if not anchor_text.strip():
        errors.append("missing_edit_spec.anchor_text")
    if not replacement_intent:
        errors.append("missing_ablation_intent")

    repo_root = Path(repo_dir).resolve()
    checkout_root = Path(source_checkout).resolve() if source_checkout else None
    path = repo_root / target_file if target_file else None
    source_text = ""
    if path is not None:
        checkout_path = checkout_root / target_file if checkout_root is not None else None
        if checkout_path is not None and checkout_path.is_file():
            source_text = checkout_path.read_text(encoding="utf-8", errors="replace")
        elif not path.is_file():
            errors.append(f"target_file_not_found:{target_file}")
        else:
            source_text = path.read_text(encoding="utf-8", errors="replace")
        if source_text and anchor_text and anchor_text not in source_text:
            errors.append("anchor_text_not_found")

    if errors:
        raise ValueError("; ".join(errors))

    return {
        "schema_version": "intent_anchor_ablation_target_v1",
        "target_file": target_file,
        "anchor_text": anchor_text,
        "replacement_pseudocode": replacement,
        "replacement_is_hint": True,
        "ablation_intent": replacement_intent,
        "replacement_intent": replacement_intent,
        "shape_invariant_argument": shape_invariant,
        "anchor_occurrences": source_text.count(anchor_text),
    }


def audit_exact_patch_hit(
    *,
    patch_path: str | Path | None,
    exact_target: dict[str, Any],
) -> dict[str, Any]:
    expected_file = _norm(str((exact_target or {}).get("target_file") or ""))
    anchor = str((exact_target or {}).get("anchor_text") or "").strip()
    replacement = str((exact_target or {}).get("replacement_pseudocode") or "").strip()
    path = Path(patch_path) if patch_path else None
    if path is None or not path.is_file():
        return {
            "schema_version": "exact_patch_audit_v1",
            "passed": False,
            "reason": "missing_patch",
            "patch_path": str(path) if path else None,
            "expected_file": expected_file,
        }

    patch_text = path.read_text(encoding="utf-8", errors="replace")
    removed = _patch_lines(patch_text, "-")
    added = _patch_lines(patch_text, "+")
    preimage_context = _patch_preimage_lines(patch_text)
    result_context = _patch_result_lines(patch_text)
    file_hit = not expected_file or (
        f"diff --git a/{expected_file} b/{expected_file}" in patch_text
        or f"--- a/{expected_file}" in patch_text
        or f"+++ b/{expected_file}" in patch_text
    )
    anchor_key = _code_key(anchor)
    preimage_key = _code_key(preimage_context)
    anchor_line_hit = any(
        line in _compact_text(preimage_context)
        or line in _compact_text(removed)
        or _code_key(line) in preimage_key
        for line in _meaningful_anchor_lines(anchor)
    )
    anchor_area_hit = bool(
        anchor
        and (
            _compact_text(anchor) in _compact_text(preimage_context)
            or _compact_text(anchor) in _compact_text(removed)
            or (preimage_key and anchor_key in preimage_key)
            or anchor_line_hit
        )
    )
    added_hit = (
        None
        if not replacement
        else bool(
            _compact_text(replacement) in _compact_text(added)
            or _compact_text(replacement) in _compact_text(result_context)
            or _code_key(replacement) in _code_key(result_context)
        )
    )
    substantive_change = bool(_compact_text(removed) or _compact_text(added))
    passed = file_hit and anchor_area_hit and substantive_change
    return {
        "schema_version": "exact_patch_audit_v1",
        "passed": passed,
        "reason": "anchor_area_patch_hit" if passed else "anchor_area_patch_miss",
        "patch_path": str(path),
        "expected_file": expected_file,
        "file_hit": file_hit,
        "removed_anchor_hit": anchor_area_hit,
        "anchor_area_hit": anchor_area_hit,
        "substantive_change": substantive_change,
        "added_replacement_hit": added_hit,
        "expected_anchor": anchor,
        "expected_replacement": replacement,
    }


def exact_target_instruction(exact_target: dict[str, Any]) -> str:
    return (
        "Formal intent-anchor ablation target. Inspect and edit near anchor_text in the target file. "
        "Use replacement_pseudocode only as an optional hint; do not copy it literally when it would "
        "break runtime protocol, return structure, tensor shape, dtype/device, or state keys.\n"
        + json.dumps(exact_target, ensure_ascii=False, indent=2)
    )
