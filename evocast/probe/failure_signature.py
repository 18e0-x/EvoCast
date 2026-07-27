"""Structured failure signatures for research-round gating."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Optional


_FILE_LINE_RE = re.compile(r'File "([^"]+)", line (\d+), in ([^\n]+)')
_EXC_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):\s*(.+)")
_SHAPE_RE = re.compile(r"(size of tensor [ab].+|mat1 and mat2 shapes cannot be multiplied.+|mat1\s*$|shape '[^']+' is invalid.+)", re.I)


def _repo_rel(path: str) -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        return Path(path).resolve().relative_to(root).as_posix()
    except Exception:
        return str(path or "")


def failure_signature(
    *,
    error_type: str,
    message: str = "",
    traceback_text: str = "",
    variant_path: Optional[str] = None,
    stage: str = "",
) -> Dict[str, Any]:
    text = f"{traceback_text}\n{message}".replace("\r", "")
    frames = []
    for match in _FILE_LINE_RE.finditer(text):
        frames.append(
            {
                "file": _repo_rel(match.group(1)),
                "line": int(match.group(2)),
                "function": match.group(3).strip(),
            }
        )
    exception_class = ""
    exception_message = str(message or "").strip()
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        match = _EXC_RE.search(line)
        if match:
            exception_class = match.group(1)
            exception_message = match.group(2)[:500]
            break
    shape_issue = ""
    match = _SHAPE_RE.search(text)
    if match:
        shape_issue = match.group(1)[:300]
    failure_site = frames[-1] if frames else {}
    modified_site = next((frame for frame in reversed(frames) if variant_path and str(frame.get("file")) == variant_path), {})
    key_parts = [
        str(error_type or ""),
        str(exception_class or ""),
        str(shape_issue or exception_message[:160]),
        str((failure_site or {}).get("file") or ""),
        str((failure_site or {}).get("function") or ""),
        str((modified_site or {}).get("file") or ""),
        str((modified_site or {}).get("function") or ""),
    ]
    signature_id = hashlib.sha1("|".join(key_parts).encode("utf-8", errors="replace")).hexdigest()[:16]
    return {
        "signature_id": signature_id,
        "stage": stage,
        "error_type": error_type,
        "exception_class": exception_class,
        "exception_message": exception_message[:500],
        "shape_issue": shape_issue,
        "failure_site": failure_site,
        "modified_site": modified_site,
        "variant_path": variant_path,
        "frames_tail": frames[-8:],
    }
