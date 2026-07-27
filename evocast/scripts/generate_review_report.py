"""Generate a mandatory LLM-narrated HTML review report for a EvoCast task."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from evocast.domain.knowledge_paths import package_root, runtime_root
from evocast.harness.api_client import resolve_provider_config_path
from evocast.reports.browser_open import open_html_report
from evocast.reports.review_report import generate_review_report


def _configure_windows_safe_io() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _resolve_api_config(value: str) -> Optional[Path]:
    if not value:
        return None
    return resolve_provider_config_path(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a EvoCast HTML review report.")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--base-dir", default="", help="Runtime root; defaults to repo .evocast.")
    parser.add_argument("--api-config", default="", help="API config YAML, e.g. providers/deepseek.yaml.")
    parser.add_argument(
        "--open-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the generated HTML report in the system default browser.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    _configure_windows_safe_io()
    args = build_arg_parser().parse_args(argv)
    if args.api_config:
        os.environ["EVOCAST_API_CONFIG"] = args.api_config
    base_dir = str(runtime_root(args.base_dir or None))
    result = generate_review_report(
        task_id=args.task_id,
        base_dir=base_dir,
        api_config=_resolve_api_config(args.api_config),
    )
    if result.get("status") == "ok" and args.open_report:
        result["open_report"] = open_html_report(result.get("html_path"))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
