from __future__ import annotations

import ast
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List

from evocast.harness.permissions import PROJECT_ROOT, normalize_repo_path
from evocast.harness.session import AgentSession
from evocast.state.runtime.store import load_runtime_state
from evocast.tools.model_structure import analyze_model_structure


INTERESTING_IMPORT_RE = re.compile(
    r"(AMS|Layer|Attention|Filter|Graph|Embedding|Block|Conv|decomp|Fourier|"
    r"Weight|Transformer|Encoder|Decoder|TwoStage|SelfAttention|Sparse|"
    r"Dispatcher|mask|GCN|Backbone)",
    re.IGNORECASE,
)

INTERESTING_SYMBOL_RE = re.compile(
    r"(forward|forecast|encoder|decomp|mix|gate|mask|period|fft|filter|graph|"
    r"attention|patch|merge|expert|dispatch|weight|learner|backbone|gcn)",
    re.IGNORECASE,
)

GENERIC_EXPANSION_FILES = {
    "Embed.py",
    "RevIN.py",
    "StandardNorm.py",
    "tools.py",
    "timefeatures.py",
    "metrics.py",
}

OPERATION_HINT_RE = re.compile(
    r"(res\.append\(.*\+|torch\.cat|torch\.stack|einsum|topk|top_k|gates?\b|"
    r"dispatcher|dispatch|combine|mask\b|rfft|irfft|fft|softmax|"
    r"period_weight|frequency|cut_freq|dominance_freq|router|merge|"
    r"glo\s*\+\s*loc|loc\s*\+\s*glo)",
    re.IGNORECASE,
)


def _repo_rel(path: str | Path) -> str:
    value = Path(path)
    if value.is_absolute():
        try:
            value = value.resolve().relative_to(PROJECT_ROOT)
        except Exception:
            return ""
    try:
        return normalize_repo_path(str(value).replace("\\", "/"))
    except Exception:
        return ""


def _repo_file(path: str) -> Path:
    return (PROJECT_ROOT / normalize_repo_path(path)).resolve()


def _read_text(path: str) -> str:
    try:
        target = _repo_file(path)
        if not target.is_file():
            return ""
        return target.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _add_path(paths: List[str], path: Any) -> None:
    rel = _repo_rel(str(path or ""))
    if (
        rel
        and rel.startswith("ts_benchmark/")
        and rel.endswith(".py")
        and _repo_file(rel).is_file()
        and rel not in paths
    ):
        paths.append(rel)


def _active_source_paths(analysis: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    for item in list(analysis.get("source_files") or []):
        if isinstance(item, dict):
            _add_path(paths, item.get("path"))
    for inner in list(analysis.get("inner_models") or []):
        if not isinstance(inner, dict):
            continue
        for item in list(inner.get("source_files") or []):
            if isinstance(item, dict):
                _add_path(paths, item.get("path"))

    has_inner = bool(list(analysis.get("inner_models") or []))
    for item in list(analysis.get("components") or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        constructor = str(item.get("constructor") or "")
        if name.startswith("model.") or not has_inner or INTERESTING_SYMBOL_RE.search(f"{name} {constructor}"):
            _add_path(paths, item.get("source_file"))

    for key in ("candidate_fit_points", "safe_fit_points", "risky_fit_points", "unknown_fit_points"):
        for item in list(analysis.get(key) or []):
            if not isinstance(item, dict):
                continue
            _add_path(paths, item.get("owner_source_file") or item.get("source_file"))
    return paths


def _resolve_import(module_path: str, node: ast.ImportFrom) -> str:
    if not node.module:
        return ""
    current = _repo_file(module_path).parent
    if node.level:
        base = current
        for _ in range(max(0, node.level - 1)):
            base = base.parent
        candidate = base / (node.module.replace(".", "/") + ".py")
    else:
        candidate = PROJECT_ROOT / (node.module.replace(".", "/") + ".py")
    if not candidate.is_file():
        return ""
    return _repo_rel(candidate)


def _import_expansion_paths(active_paths: List[str], *, max_paths: int = 18) -> tuple[List[str], List[str]]:
    expanded = list(active_paths)
    imported_symbols: List[str] = []
    queue = list(active_paths)
    visited: set[str] = set()
    while queue and len(expanded) < max_paths:
        path = queue.pop(0)
        if path in visited:
            continue
        visited.add(path)
        if Path(path).name in GENERIC_EXPANSION_FILES:
            continue
        text = _read_text(path)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            names = [alias.name for alias in node.names]
            if not any(INTERESTING_IMPORT_RE.search(name) for name in names):
                continue
            target = _resolve_import(path, node)
            if not target:
                continue
            for name in names:
                if name not in imported_symbols:
                    imported_symbols.append(name)
            if target not in expanded:
                expanded.append(target)
                queue.append(target)
                if len(expanded) >= max_paths:
                    break
    return expanded, imported_symbols


def _function_or_class_blocks(path: str, *, max_blocks: int = 14, max_block_lines: int = 190) -> List[Dict[str, Any]]:
    text = _read_text(path)
    if not text:
        return []
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [{"kind": "raw", "name": "top", "content": "\n".join(lines[:260])}]

    blocks: List[Dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name in {"__init__", "forward", "forecast"} or INTERESTING_SYMBOL_RE.search(node.name):
                start = int(getattr(node, "lineno", 1))
                end = int(getattr(node, "end_lineno", start))
                blocks.append(
                    {
                        "kind": "function",
                        "name": node.name,
                        "line_start": start,
                        "line_end": end,
                        "content": "\n".join(lines[start - 1 : min(end, start + max_block_lines)]),
                    }
                )
            continue
        if not isinstance(node, ast.ClassDef):
            continue
        chunks: List[str] = []
        class_start = int(getattr(node, "lineno", 1))
        header = lines[class_start - 1] if 0 <= class_start - 1 < len(lines) else f"class {node.name}:"
        chunks.append(header)
        include_whole_class = bool(INTERESTING_SYMBOL_RE.search(node.name))
        for child in node.body:
            if not isinstance(child, ast.FunctionDef):
                continue
            if include_whole_class or child.name in {"__init__", "forward", "forecast"} or INTERESTING_SYMBOL_RE.search(child.name):
                start = int(getattr(child, "lineno", 1))
                end = int(getattr(child, "end_lineno", start))
                chunks.append("\n".join(lines[start - 1 : min(end, start + max_block_lines)]))
        if len(chunks) > 1:
            blocks.append(
                {
                    "kind": "class",
                    "name": node.name,
                    "line_start": class_start,
                    "line_end": int(getattr(node, "end_lineno", class_start)),
                    "content": "\n\n".join(chunks)[:18000],
                }
            )
        if len(blocks) >= max_blocks:
            break
    if not blocks:
        blocks.append({"kind": "raw", "name": "top", "content": "\n".join(lines[:260])})
    return blocks[:max_blocks]


def _compact_components(analysis: Dict[str, Any], *, limit: int = 80) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    has_inner = bool(list(analysis.get("inner_models") or []))
    for item in list(analysis.get("components") or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        constructor = str(item.get("constructor") or "")
        hay = f"{name} {constructor}"
        if not (name.startswith("model.") or not has_inner or INTERESTING_SYMBOL_RE.search(hay)):
            continue
        rows.append(
            {
                "name": name,
                "constructor": constructor[:180],
                "kind": item.get("kind"),
                "source_file": _repo_rel(item.get("source_file") or ""),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _compact_fit_points(analysis: Dict[str, Any], *, limit: int = 24) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in list(analysis.get("candidate_fit_points") or []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "path": item.get("path") or item.get("component_path"),
                "mechanism_family": item.get("mechanism_family") or item.get("mechanism_type"),
                "mechanism_role": item.get("mechanism_role"),
                "owner_source_file": _repo_rel(item.get("owner_source_file") or item.get("source_file") or ""),
                "forward_reachable": bool(item.get("forward_reachable")),
                "patchable": bool(item.get("patchable")),
                "granularity": item.get("granularity"),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _operation_hints(path: str, *, context: int = 2, limit: int = 36) -> List[Dict[str, Any]]:
    text = _read_text(path)
    if not text:
        return []
    lines = text.splitlines()
    hints: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or not OPERATION_HINT_RE.search(stripped):
            continue
        start = max(0, index - context)
        end = min(len(lines), index + context + 1)
        snippet = "\n".join(lines[start:end]).strip()
        key = _canonical_code_text(snippet)
        if key in seen:
            continue
        seen.add(key)
        kind = "operation"
        lowered = stripped.lower()
        if "glo" in lowered and "loc" in lowered and "+" in stripped:
            kind = "branch_fusion"
        elif "cat" in lowered or "stack" in lowered or "merge" in lowered:
            kind = "merge_or_concat"
        elif "gate" in lowered or "dispatch" in lowered or "mask" in lowered or "topk" in lowered:
            kind = "routing_or_masking"
        elif "fft" in lowered or "frequency" in lowered or "cut_freq" in lowered:
            kind = "frequency_operation"
        hints.append(
            {
                "file": path,
                "line": index + 1,
                "kind": kind,
                "anchor": stripped[:260],
                "context": snippet[:1200],
            }
        )
        if len(hints) >= limit:
            break
    return hints


def _baseline_hyper_params(session: AgentSession) -> Dict[str, Any]:
    state = load_runtime_state(session.base_dir, session.task_id)
    baseline = state.baseline.to_dict() if state.baseline.candidate_id else state.current_best.to_dict()
    model_config = dict(baseline.get("model_config") or {})
    return dict(model_config.get("model_hyper_params") or {})


def build_mechanism_evidence_graph(
    session: AgentSession,
    *,
    model_key: str,
    max_paths: int = 18,
) -> Dict[str, Any]:
    analysis = analyze_model_structure(
        session,
        {
            "model_key": model_key,
            "run_shape_probe": False,
            "force_refresh": True,
        },
    )
    active_paths = _active_source_paths(analysis)
    expanded_paths, imported_symbols = _import_expansion_paths(active_paths, max_paths=max_paths)
    source_blocks = [
        {
            "path": path,
            "blocks": _function_or_class_blocks(path),
        }
        for path in expanded_paths
    ]
    operation_hints: List[Dict[str, Any]] = []
    for path in expanded_paths:
        operation_hints.extend(_operation_hints(path))
    return {
        "schema_version": "mechanism_evidence_graph_v0",
        "model_key": model_key,
        "analysis_status": analysis.get("status"),
        "active_source_files": active_paths,
        "expanded_source_files": expanded_paths,
        "interesting_imported_symbols": imported_symbols,
        "source_blocks": source_blocks,
        "facts": {
            "baseline_hyper_params": _baseline_hyper_params(session),
            "components": _compact_components(analysis),
            "fit_points": _compact_fit_points(analysis),
            "forward_called_components": list((analysis.get("forward") or {}).get("called_components") or [])[:60],
            "mechanism_candidates_count": len(list(analysis.get("mechanism_candidates") or [])),
            "candidate_fit_points_count": len(list(analysis.get("candidate_fit_points") or [])),
            "safe_fit_points_count": len(list(analysis.get("safe_fit_points") or [])),
            "operation_hints": operation_hints[:120],
        },
    }


def _canonical_code_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"\s*([:\[\]\(\)\{\},.=+\-*/])\s*", r"\1", text)
    text = re.sub(r",([\)\]\}])", r"\1", text)
    return text


def source_contains_anchor(path: str, anchor: str) -> bool:
    text = _read_text(path)
    needle = str(anchor or "")
    if not text or not needle:
        return False
    if "..." in needle or "…" in needle:
        return _source_contains_ordered_anchor_parts(text, needle)
    if resolve_source_anchor(path, needle):
        return True
    if needle in text:
        return True
    normalized_text = re.sub(r"\s+", " ", text)
    normalized_anchor = re.sub(r"\s+", " ", needle)
    if normalized_anchor and normalized_anchor in normalized_text:
        return True
    canonical_text = _canonical_code_text(text)
    canonical_anchor = _canonical_code_text(needle)
    if canonical_anchor and canonical_anchor in canonical_text:
        return True
    anchor_lines = [line.strip() for line in needle.splitlines() if line.strip()]
    if len(anchor_lines) > 1:
        return any(source_contains_anchor(path, line) for line in anchor_lines)
    if ":" in needle:
        suffix = needle.split(":", 1)[1].strip()
        if suffix and suffix != needle:
            return source_contains_anchor(path, suffix)
    return False


def resolve_source_anchor(path: str, anchor: str) -> str:
    """Return an exact source substring for a semantically matching anchor."""

    text = _read_text(path)
    needle = str(anchor or "").strip()
    if not text or not needle or "..." in needle or "…" in needle:
        return ""

    lines = text.splitlines()
    if "\n" not in needle:
        docstring_lines = _docstring_line_numbers(text)
        first_match = ""
        for line_no, line in enumerate(lines, start=1):
            if line.strip() == needle:
                if not first_match:
                    first_match = line
                if line_no not in docstring_lines:
                    return line
        if first_match:
            return first_match

    if needle in text:
        return needle

    needle_key = _ast_statement_key(needle)
    if not needle_key:
        needle_key = _canonical_code_text(needle)
    if not needle_key:
        return ""

    best: str = ""
    best_score = -1
    max_window = min(12, max(1, len([line for line in needle.splitlines() if line.strip()]) + 8))
    for start in range(len(lines)):
        for end in range(start + 1, min(len(lines), start + max_window) + 1):
            raw_snippet = "\n".join(lines[start:end])
            snippet = raw_snippet.strip()
            if not snippet:
                continue
            snippet_key = _ast_statement_key(snippet) or _canonical_code_text(snippet)
            if not snippet_key:
                continue
            if snippet_key == needle_key:
                score = 10_000 - (end - start)
            elif _canonical_code_text(needle) in _canonical_code_text(snippet):
                score = 1_000 - (end - start)
            else:
                continue
            if score > best_score:
                best = raw_snippet
                best_score = score
    return best


def _docstring_line_numbers(text: str) -> set[int]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not body or not isinstance(body, list):
            continue
        first = body[0]
        if not isinstance(first, ast.Expr) or not isinstance(getattr(first, "value", None), ast.Constant):
            continue
        if not isinstance(first.value.value, str):
            continue
        start = int(getattr(first, "lineno", 0) or 0)
        end = int(getattr(first, "end_lineno", start) or start)
        for line_no in range(start, end + 1):
            lines.add(line_no)
    return lines


def _ast_statement_key(value: str) -> str:
    text = textwrap.dedent(str(value or "")).strip()
    if not text:
        return ""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    if not tree.body:
        return ""
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


def _source_contains_ordered_anchor_parts(text: str, anchor: str) -> bool:
    raw_parts = [
        part.strip()
        for part in re.split(r"(?:\.\.\.|…)", str(anchor or ""))
        if part.strip()
    ]
    if not raw_parts:
        return False
    normalized_text = re.sub(r"\s+", " ", text)
    normalized_parts = [re.sub(r"\s+", " ", part).strip() for part in raw_parts]
    canonical_text = _canonical_code_text(text)
    canonical_parts = [_canonical_code_text(part) for part in raw_parts]

    def _ordered_contains(haystack: str, parts: List[str]) -> bool:
        position = 0
        for part in parts:
            if not part:
                continue
            index = haystack.find(part, position)
            if index < 0:
                return False
            position = index + len(part)
        return True

    if _ordered_contains(normalized_text, normalized_parts) or _ordered_contains(canonical_text, canonical_parts):
        return True

    significant_lines = [
        _canonical_code_text(line.strip())
        for part in raw_parts
        for line in part.splitlines()
        if len(line.strip()) >= 18 and line.strip() != "..."
    ]
    return _ordered_contains(canonical_text, significant_lines)


def anchor_overlap_score(anchor: str, evidence_anchors: Iterable[str]) -> float:
    anchor_tokens = {
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(anchor or ""))
        if len(token) > 2
    }
    if not anchor_tokens:
        return 0.0
    best = 0.0
    for evidence in evidence_anchors:
        evidence_tokens = {
            token
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(evidence or ""))
            if len(token) > 2
        }
        if not evidence_tokens:
            continue
        best = max(best, len(anchor_tokens & evidence_tokens) / max(1, len(anchor_tokens | evidence_tokens)))
    return best
