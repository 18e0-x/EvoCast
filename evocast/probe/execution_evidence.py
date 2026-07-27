"""Generic execution evidence helpers for evocast.

These utilities intentionally avoid model-specific rules.  They turn runtime
artifacts into structured facts that controller prompts and gates can inspect
before an agent repairs code or treats a variant as meaningful.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from evocast.domain.execution_ids import parse_research_id

_FRAME_RE = re.compile(r'^\s*File "([^"]+)", line (\d+), in ([^\n]+)\s*$')
_SHAPE_ERROR_RE = re.compile(
    r"(size|shape|dimension|dim|batch|mat1|mat2|batch1|batch2|broadcast|expand|cat|stack|bmm|mm|matmul)",
    re.IGNORECASE,
)


def repo_rel(path: str, project_root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(project_root).as_posix()
    except Exception:
        return str(path or "")


def canonical_source_target(path: Any) -> str:
    """Map a round workspace source file back to its canonical source path."""
    normalized = str(path or "").replace("\\", "/").strip()
    parts = [part for part in normalized.split("/") if part]
    if "sandboxes" in parts and "variant" in parts:
        variant_idx = parts.index("variant")
        source_parts = parts[variant_idx + 1:]
        if not source_parts or source_parts[-1] == "round_entry.py":
            return normalized
        return "/".join(source_parts)
    if "ts_benchmark" in parts:
        source_parts = parts[parts.index("ts_benchmark"):]
        if source_parts and source_parts[-1] != "round_entry.py":
            return "/".join(source_parts)
    workspace_idx = -1
    for candidate in ("workspace",):
        try:
            workspace_idx = parts.index(candidate)
            break
        except ValueError:
            continue
    if workspace_idx < 0:
        return normalized
    if workspace_idx + 2 >= len(parts):
        return normalized
    round_part = next((part for part in parts if parse_research_id(part) is not None), "")
    if parse_research_id(round_part) is None:
        return normalized
    source_parts = parts[workspace_idx + 1:]
    if not source_parts or source_parts[-1] == "round_entry.py":
        return normalized
    return "/".join(source_parts)


def _editable_source_frame(frame: Dict[str, Any]) -> bool:
    path = str(frame.get("repo_path") or frame.get("file") or "").replace("\\", "/")
    canonical = canonical_source_target(path)
    if not canonical.startswith("ts_benchmark/"):
        return False
    lowered = path.lower()
    return "site-packages" not in lowered and "dist-packages" not in lowered


def extract_traceback_evidence(traceback_text: str, *, project_root: Path | None = None) -> Dict[str, Any]:
    """Parse a Python traceback into stable, source-grounded failure evidence."""
    text = str(traceback_text or "")
    lines = text.splitlines()
    frames: List[Dict[str, Any]] = []
    for index, line in enumerate(lines):
        match = _FRAME_RE.match(line)
        if not match:
            continue
        path, lineno, function = match.groups()
        code_line = ""
        if index + 1 < len(lines):
            code_line = lines[index + 1].strip()
        frame = {
            "file": path,
            "line": int(lineno),
            "function": function.strip(),
            "code_line": code_line,
        }
        if project_root is not None:
            frame["repo_path"] = repo_rel(path, project_root)
            frame["in_project"] = bool(frame["repo_path"] and not Path(str(frame["repo_path"])).is_absolute())
        canonical = canonical_source_target(frame.get("repo_path") or path)
        frame["canonical_target_file"] = canonical
        frame["editable_source"] = _editable_source_frame(frame)
        frames.append(frame)

    innermost = frames[-1] if frames else {}
    project_frames = [frame for frame in frames if frame.get("in_project")]
    innermost_project = project_frames[-1] if project_frames else {}
    editable_frames = [frame for frame in frames if frame.get("editable_source")]
    innermost_editable = editable_frames[-1] if editable_frames else {}
    final_error = ""
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            final_error = stripped
            break
    primary_frame = innermost_editable or innermost_project or innermost
    code_line = str(primary_frame.get("code_line") or "")
    return {
        "status": "ok" if frames else "no_traceback_frames",
        "final_error": final_error,
        "frames": frames[-12:],
        "innermost_frame": innermost,
        "innermost_project_frame": innermost_project,
        "innermost_editable_frame": innermost_editable,
        "suspected_operator_line": code_line,
        "shape_error_likely": bool(_SHAPE_ERROR_RE.search(final_error + "\n" + code_line)),
        "repair_scope_hint": _repair_scope_hint(primary_frame),
    }


def _repair_scope_hint(frame: Dict[str, Any]) -> Dict[str, Any]:
    if not frame:
        return {"status": "unknown", "reason": "no traceback frame was available"}
    path = str(frame.get("repo_path") or frame.get("file") or "")
    canonical = canonical_source_target(path)
    line = frame.get("line")
    return {
        "status": "traceback_local",
        "primary_file": path,
        "canonical_target_file": canonical,
        "primary_line": line,
        "primary_function": frame.get("function"),
        "required_evidence": (
            "Diagnose this traceback frame, but exact_edits target_file must use "
            "canonical_target_file, not a workspace path."
        ),
    }


def source_defined_symbols(source: str) -> Dict[str, Any]:
    """Return top-level classes/functions in a variant source file."""
    try:
        tree = ast.parse(str(source or ""))
    except SyntaxError as exc:
        return {"status": "syntax_error", "error": f"{type(exc).__name__}: {exc}", "classes": [], "functions": []}
    classes = []
    functions = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append({"name": node.name, "line": int(getattr(node, "lineno", 0) or 0)})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({"name": node.name, "line": int(getattr(node, "lineno", 0) or 0)})
    return {"status": "ok", "classes": classes, "functions": functions}


def tensor_digest(value: Any) -> Dict[str, Any]:
    """Small numeric fingerprint for tensor-like outputs."""
    try:
        import torch
    except Exception:
        torch = None
    if torch is None or not torch.is_tensor(value):
        return {"kind": type(value).__name__}
    detached = value.detach().float().cpu()
    flat = detached.reshape(-1)
    sample = flat[: min(int(flat.numel()), 2048)].numpy().tobytes() if flat.numel() else b""
    return {
        "kind": "tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "mean": float(detached.mean().item()) if detached.numel() else 0.0,
        "std": float(detached.std(unbiased=False).item()) if detached.numel() else 0.0,
        "sha1_sample": hashlib.sha1(sample).hexdigest(),
    }


def compare_tensor_outputs(left: Any, right: Any) -> Dict[str, Any]:
    """Compare two tensor-like outputs without assuming a specific model family."""
    left_digest = tensor_digest(left)
    right_digest = tensor_digest(right)
    result = {
        "left": left_digest,
        "right": right_digest,
        "same_shape": left_digest.get("shape") == right_digest.get("shape"),
        "exact_equal": False,
        "max_abs_diff": None,
        "mean_abs_diff": None,
    }
    try:
        import torch

        if torch.is_tensor(left) and torch.is_tensor(right) and list(left.shape) == list(right.shape):
            diff = (left.detach().float().cpu() - right.detach().float().cpu()).abs()
            result["exact_equal"] = bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
            result["max_abs_diff"] = float(diff.max().item()) if diff.numel() else 0.0
            result["mean_abs_diff"] = float(diff.mean().item()) if diff.numel() else 0.0
    except Exception:
        pass
    return result


def collect_module_class_paths(root: Any, *, limit: int = 400) -> List[Dict[str, Any]]:
    """Collect named module class owners from a torch module-like object."""
    if root is None or not hasattr(root, "named_modules"):
        return []
    items: List[Dict[str, Any]] = []
    try:
        iterator: Iterable[Any] = root.named_modules()
    except Exception:
        return []
    for index, (name, module) in enumerate(iterator):
        if index >= limit:
            break
        cls = type(module)
        items.append(
            {
                "name": str(name or "<root>"),
                "class_path": f"{getattr(cls, '__module__', '')}.{getattr(cls, '__name__', '')}".strip("."),
            }
        )
    return items
