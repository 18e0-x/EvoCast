"""Run candidate-model runtime validation in a short-lived Python process."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict

from evocast.policy.agent_control_policy import runtime_validation_policy
from evocast.probe.execution_evidence import extract_traceback_evidence
from evocast.probe.failure_signature import failure_signature


_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_MODULE = "evocast.variant.runtime_validation_worker"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _worker_failure(
    *,
    operation: str,
    variant_path: str,
    error_type: str,
    error_message: str,
    traceback_text: str = "",
    returncode: int | None = None,
) -> Dict[str, Any]:
    evidence = extract_traceback_evidence(traceback_text) if traceback_text else {}
    return {
        "status": "failed" if operation == "runtime_contract" else "error",
        "stage": "runtime_validation_worker",
        "operation": operation,
        "error_type": error_type,
        "error_message": error_message,
        "error_traceback": traceback_text,
        "failure_evidence": evidence,
        "failure_signature": failure_signature(
            error_type=error_type,
            message=error_message,
            traceback_text=traceback_text,
            variant_path=variant_path,
            stage="runtime_validation_worker",
        ),
        "worker_returncode": returncode,
    }


def run_runtime_validation_worker(
    *,
    operation: str,
    payload: Dict[str, Any],
    base_dir: str | None = None,
) -> Dict[str, Any]:
    """Execute one candidate CUDA validation operation outside the agent process."""
    policy = runtime_validation_policy(base_dir)
    work_dir = Path(tempfile.mkdtemp(prefix="evocast_runtime_validation_"))
    input_path = work_dir / "request.json"
    output_path = work_dir / "result.json"
    variant_path = str(payload.get("variant_path") or payload.get("source_entry_file") or "")
    try:
        request = {"operation": operation, "payload": dict(payload or {})}
        input_path.write_text(json.dumps(request, ensure_ascii=False, default=str), encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONFAULTHANDLER"] = "1"
        if policy["cuda_launch_blocking"]:
            env["CUDA_LAUNCH_BLOCKING"] = "1"
        python_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(_REPO_ROOT) + (os.pathsep + python_path if python_path else "")
        try:
            completed = subprocess.run(
                [sys.executable, "-m", _WORKER_MODULE, "--input", str(input_path), "--output", str(output_path)],
                cwd=str(_REPO_ROOT),
                env=env,
                text=True,
                capture_output=True,
                timeout=policy["timeout_sec"],
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            trace = "\n".join(part for part in [str(exc.stdout or ""), str(exc.stderr or "")] if part)
            return _worker_failure(
                operation=operation,
                variant_path=variant_path,
                error_type="runtime_validation_timeout",
                error_message=f"runtime validation worker exceeded {policy['timeout_sec']} seconds",
                traceback_text=trace,
            )

        result = _read_json(output_path)
        if result:
            result["worker_returncode"] = int(completed.returncode)
            result["worker_stdout"] = str(completed.stdout or "")[-8000:]
            result["worker_stderr"] = str(completed.stderr or "")[-8000:]
            return result
        trace = "\n".join(part for part in [str(completed.stdout or ""), str(completed.stderr or "")] if part)
        return _worker_failure(
            operation=operation,
            variant_path=variant_path,
            error_type="runtime_validation_worker_crashed",
            error_message=f"runtime validation worker exited with code {completed.returncode} before writing result.json",
            traceback_text=trace,
            returncode=int(completed.returncode),
        )
    except Exception as exc:
        return _worker_failure(
            operation=operation,
            variant_path=variant_path,
            error_type="runtime_validation_launcher_error",
            error_message=f"{type(exc).__name__}: {exc}",
            traceback_text=traceback.format_exc(),
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
