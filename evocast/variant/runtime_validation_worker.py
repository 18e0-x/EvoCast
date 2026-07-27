"""Isolated process entry point for candidate runtime validation."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime validation request must be a JSON object")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _run(operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    # Import only in this process: candidate modules and CUDA are never loaded
    # by the parent agent process.
    from evocast.variant.contract import (
        _probe_variant_behavior_delta_in_process,
        _validate_variant_runtime_contract_in_process,
    )

    if operation == "runtime_contract":
        return _validate_variant_runtime_contract_in_process(**payload)
    if operation == "behavior_delta":
        return _probe_variant_behavior_delta_in_process(**payload)
    raise ValueError(f"unsupported runtime validation operation: {operation}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EvoCast isolated candidate runtime validation worker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    try:
        request = _read_json(Path(args.input))
        operation = str(request.get("operation") or "")
        payload = request.get("payload") if isinstance(request.get("payload"), dict) else {}
        result = _run(operation, dict(payload))
        _write_json(output, result if isinstance(result, dict) else {"status": "error", "error_message": "worker returned non-object"})
        return 0
    except Exception as exc:
        _write_json(
            output,
            {
                "status": "failed",
                "stage": "runtime_validation_worker",
                "error_type": type(exc).__name__,
                "error_message": f"{type(exc).__name__}: {exc}",
                "error_traceback": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
