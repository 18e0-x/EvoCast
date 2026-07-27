from __future__ import annotations

import json
import subprocess
from pathlib import Path

from evocast.variant import contract
from evocast.variant import runtime_validation


class _Completed:
    returncode = 0
    stdout = "worker stdout"
    stderr = ""


def test_runtime_validation_worker_receives_isolated_request_and_cuda_sync(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["env"] = kwargs["env"]
        request_index = command.index("--input") + 1
        output_index = command.index("--output") + 1
        request = json.loads(Path(command[request_index]).read_text(encoding="utf-8"))
        observed["request"] = request
        Path(command[output_index]).write_text(
            json.dumps({"status": "ok", "stage": "runtime_contract_probe"}),
            encoding="utf-8",
        )
        return _Completed()

    monkeypatch.setattr(runtime_validation.subprocess, "run", fake_run)
    result = runtime_validation.run_runtime_validation_worker(
        operation="runtime_contract",
        payload={"tfb_config": {}, "variant_entry": {}, "variant_path": "model.py", "seed": 2021},
        base_dir=str(tmp_path),
    )

    assert result["status"] == "ok"
    assert observed["command"][1:3] == ["-m", "evocast.variant.runtime_validation_worker"]
    assert observed["env"]["CUDA_LAUNCH_BLOCKING"] == "1"
    assert observed["env"]["PYTHONFAULTHANDLER"] == "1"
    assert observed["request"]["operation"] == "runtime_contract"


def test_runtime_validation_timeout_returns_structured_failure(monkeypatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=120)

    monkeypatch.setattr(runtime_validation.subprocess, "run", fake_run)
    result = runtime_validation.run_runtime_validation_worker(
        operation="runtime_contract",
        payload={"variant_path": "model.py"},
        base_dir=str(tmp_path),
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "runtime_validation_timeout"


def test_runtime_validation_worker_crash_returns_structured_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_validation.subprocess, "run", lambda *args, **kwargs: _Completed())
    result = runtime_validation.run_runtime_validation_worker(
        operation="runtime_contract",
        payload={"variant_path": "model.py"},
        base_dir=str(tmp_path),
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "runtime_validation_worker_crashed"


def test_public_contract_entry_uses_worker_not_in_process_probe(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_worker(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "worker_returncode": 0}

    monkeypatch.setattr(runtime_validation, "run_runtime_validation_worker", fake_worker)
    result = contract.validate_variant_runtime_contract(
        tfb_config={"data_config": {}},
        variant_entry={"model_name": "PatchTST"},
        variant_path="ts_benchmark/baselines/time_series_library/models/PatchTST.py",
    )

    assert result["status"] == "ok"
    assert calls[0]["operation"] == "runtime_contract"

    contract.probe_variant_behavior_delta(
        tfb_config={},
        baseline_entry={"model_name": "PatchTST"},
        variant_entry={"model_name": "PatchTST"},
    )
    assert calls[1]["operation"] == "behavior_delta"
