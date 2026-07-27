"""Small atomic JSON/text write helpers with Windows-friendly retries."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    retries: int = 8,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Windows includes the entire temporary filename in MAX_PATH checks.
    # Artifact names such as editable_source_anchor_menu combined with a PID
    # and nanosecond suffix can exceed the usable path budget even though the
    # final target itself is valid. Keep atomic staging names deliberately
    # short; uniqueness is provided by mkstemp's random component.
    fd, tmp_name = tempfile.mkstemp(
        prefix=".tfba-",
        suffix=".tmp",
        dir=str(target.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        last_exc: PermissionError | None = None
        for attempt in range(max(1, int(retries))):
            try:
                os.replace(tmp_name, target)
                return
            except PermissionError as exc:
                last_exc = exc
                time.sleep(0.05 * (attempt + 1))
        if last_exc:
            raise last_exc
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    default: Any = str,
    trailing_newline: bool = False,
) -> None:
    text = json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii, default=default)
    if trailing_newline:
        text += "\n"
    atomic_write_text(path, text)
