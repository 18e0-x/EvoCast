from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

ROUNDS_DIR = "rounds"
RESEARCH_ROUNDS_DIR = ROUNDS_DIR
ABLATIONS_DIR = ROUNDS_DIR

_RESEARCH_ID_RE = re.compile(r"^Research(\d{3})$")
_ABLATION_ID_RE = re.compile(r"^(?:Ablation|ablation)(\d{3})$")


def format_research_id(index: int) -> str:
    return f"Research{int(index):03d}"


def format_ablation_id(index: int) -> str:
    return f"Ablation{int(index):03d}"


def parse_research_id(value: object) -> Optional[int]:
    text = str(value or "").strip()
    match = _RESEARCH_ID_RE.match(text)
    if not match:
        return None
    return int(match.group(1))


def parse_ablation_id(value: object) -> Optional[int]:
    text = str(value or "").strip()
    match = _ABLATION_ID_RE.match(text)
    if not match:
        return None
    return int(match.group(1))


def research_round_filename(index: int) -> str:
    return "round.json"


def round_dirname(kind: str, index: int) -> str:
    text = str(kind or "").strip().lower()
    if text == "ablation":
        return format_ablation_id(index)
    return format_research_id(index)


def round_record_filename() -> str:
    return "round.json"


def is_research_workspace_dirname(value: object) -> bool:
    return parse_research_id(value) is not None


def is_ablation_dirname(value: object) -> bool:
    return parse_ablation_id(value) is not None


def normalize_path_text(value: object) -> str:
    return str(value or "").replace("\\", "/").strip()


def _round_sources_parts(value: object) -> list[str]:
    normalized = normalize_path_text(value)
    parts = [part for part in normalized.split("/") if part]
    if "round_sources" not in parts:
        return []
    return parts[parts.index("round_sources"):]


def is_research_workspace_path(value: object) -> bool:
    normalized = normalize_path_text(value)
    return f"/{ROUNDS_DIR}/Research" in f"/{normalized}" or (
        "/sandboxes/" in f"/{normalized}"
        and "/Research" in f"/{normalized}"
        and ("/variant/" in normalized or normalized.endswith("/variant"))
    )


def is_research_source_path(value: object) -> bool:
    normalized = normalize_path_text(value)
    parts = _round_sources_parts(normalized)
    return (
        len(parts) >= 4
        and parts[0] == "round_sources"
        and parse_research_id(parts[2]) is not None
        and (normalized.endswith("/round_entry.py") or normalized.endswith("/round_entry"))
    )


def is_ablation_source_path(value: object) -> bool:
    normalized = normalize_path_text(value)
    parts = _round_sources_parts(normalized)
    return (
        len(parts) >= 4
        and parts[0] == "round_sources"
        and parse_ablation_id(parts[2]) is not None
        and (normalized.endswith("/round_entry.py") or normalized.endswith("/round_entry"))
    )


def is_research_variant_entry_path(value: object) -> bool:
    normalized = normalize_path_text(value)
    return is_research_workspace_path(normalized) and normalized.endswith("/round_entry.py")


def is_ablation_workspace_path(value: object) -> bool:
    normalized = normalize_path_text(value)
    return (
        f"/{ROUNDS_DIR}/Ablation" in f"/{normalized}"
        and ("/workspace/" in normalized or normalized.endswith("/workspace"))
    ) or (
        "/sandboxes/" in f"/{normalized}"
        and ("/Ablation" in f"/{normalized}" or "/ablation" in f"/{normalized}")
        and ("/variant/" in normalized or normalized.endswith("/variant"))
    )


def is_ablation_variant_entry_path(value: object) -> bool:
    normalized = normalize_path_text(value)
    return (is_ablation_workspace_path(normalized) or is_ablation_source_path(normalized)) and normalized.endswith("/round_entry.py")


def extract_execution_label_from_variant_path(value: object) -> str:
    normalized = normalize_path_text(value)
    parts = [part for part in normalized.split("/") if part]
    for part in parts:
        if parse_research_id(part) is not None or parse_ablation_id(part) is not None:
            return part
    return ""


def extract_research_index_from_variant_path(value: object) -> Optional[int]:
    normalized = normalize_path_text(value)
    for part in [part for part in normalized.split("/") if part]:
        parsed = parse_research_id(part)
        if parsed is not None:
            return parsed
    return None


def extract_ablation_index_from_variant_path(value: object) -> Optional[int]:
    normalized = normalize_path_text(value)
    for part in [part for part in normalized.split("/") if part]:
        parsed = parse_ablation_id(part)
        if parsed is not None:
            return parsed
    return None


def is_research_workspace_root(path: Path) -> bool:
    name = path.name
    return name == ROUNDS_DIR
