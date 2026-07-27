from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Any, Dict


def open_html_report(path: object) -> Dict[str, Any]:
    """Open an HTML report with the system default browser."""
    report_path = Path(str(path or "")).expanduser()
    if not report_path.is_file():
        return {"status": "skipped", "reason": "report_not_found", "path": str(report_path)}
    try:
        opened = webbrowser.open(report_path.resolve().as_uri(), new=2)
    except Exception as exc:
        return {
            "status": "error",
            "reason": type(exc).__name__,
            "message": str(exc),
            "path": str(report_path),
        }
    return {"status": "ok" if opened else "error", "opened": bool(opened), "path": str(report_path)}
