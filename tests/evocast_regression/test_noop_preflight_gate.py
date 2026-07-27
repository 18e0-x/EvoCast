from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evocast.tools.tfb_experiment as tfb_experiment
from evocast.state.runtime.store import sync_best_baseline
from evocast.harness.session import AgentSession
from evocast.state.domain_store import load_task_config, save_task_config

VALID_VARIANT_PATH = "round_sources/noop_preflight_task/Research001/round_entry.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _session(tmp_path: Path) -> AgentSession:
    base_dir = tmp_path / "evocast"
    task_id = "noop_preflight_task"
    task_dir = base_dir / "task_knowledge" / task_id
    compiled = {
        "data_config": {"task_semantics": {"task_mode": "MM"}},
        "model_config": {"recommend_model_hyper_params": {"input_chunk_length": 8, "output_chunk_length": 4}},
        "evaluation_config": {"strategy_args": {"horizon": 4, "seed": 2021}},
    }
    _write_json(task_dir / "compiled_config.json", compiled)
    _write_json(
        task_dir / "task_config.json",
        {"task_id": task_id, "objective_metric": "mse_norm", "budget": "unified"},
    )
    sync_best_baseline(
        str(base_dir),
        task_id,
        {
            "candidate_id": "baseline_fixture",
            "candidate_kind": "baseline",
            "display_name": "FixtureBaseline",
            "model_name": "FixtureBaseline",
            "import_path": "fixture.baseline.Model",
            "adapter": None,
            "metrics": {"mse_norm": 1.0},
            "objective_metric": "mse_norm",
            "model_config": {
                "model_name": "fixture.baseline.Model",
                "adapter": None,
                "model_hyper_params": {"input_chunk_length": 8, "output_chunk_length": 4},
            },
        },
    )
    session = AgentSession(task_id=task_id, base_dir=str(base_dir), client=object())
    session.ensure_dirs()
    return session


def _noop_delta() -> dict:
    return {
        "status": "ok",
        "stage": "behavior_delta_probe",
        "variant_path": VALID_VARIANT_PATH,
        "comparison": {"same_shape": True, "exact_equal": True, "max_abs_diff": 0.0},
        "suspected_noop": True,
        "reason": "baseline and variant probe outputs are identical or numerically indistinguishable",
    }


def _active_delta() -> dict:
    return {
        "status": "ok",
        "stage": "behavior_delta_probe",
        "variant_path": VALID_VARIANT_PATH,
        "comparison": {"same_shape": True, "exact_equal": False, "max_abs_diff": 0.25},
        "suspected_noop": False,
        "reason": "baseline and variant probe outputs differ",
    }


def test_run_experiment_rejects_noop_before_any_training(monkeypatch, tmp_path) -> None:
    session = _session(tmp_path)
    train_calls = []
    probe_calls = []

    monkeypatch.setattr(tfb_experiment, "validate_model_config", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(tfb_experiment, "validate_variant_runtime_contract", lambda *args, **kwargs: {"status": "ok"})

    def fake_probe(**kwargs):
        probe_calls.append(kwargs)
        return _noop_delta()

    def forbidden_execute(**kwargs):
        train_calls.append(kwargs)
        raise AssertionError("no-op preflight must reject before smoke/formal training")

    monkeypatch.setattr(tfb_experiment, "probe_variant_behavior_delta", fake_probe)
    monkeypatch.setattr(tfb_experiment, "_execute_variant", forbidden_execute)

    result = tfb_experiment.run_experiment(
        session,
        {"variant_path": VALID_VARIANT_PATH},
    )

    assert probe_calls, result
    assert train_calls == [], result
    assert result["status"] == "ok", result
    assert result["success"] is False, result
    assert result["parsed_status"] == "noop_variant_contract", result
    assert result["error_type"] == "noop_variant_contract", result
    assert result["metrics"] == {}, result
    assert result["log_paths"] == [], result
    assert result["smoke_precheck"] is None, result
    assert result["behavior_delta"]["suspected_noop"] is True, result
    assert result["failure_signature"]["error_type"] == "noop_variant_contract", result

    record = json.loads(Path(result["run_record_path"]).read_text(encoding="utf-8"))
    evidence = (record["config_provenance"] or {}).get("failure_evidence") or {}
    assert evidence["probe_policy"] == "same_input_forward_only_no_baseline_training", evidence
    assert evidence["repair_scope_hint"]["baseline_training_rerun_allowed"] is False, evidence


def test_run_experiment_binds_variant_path_to_smoke_and_formal_execution(monkeypatch, tmp_path) -> None:
    session = _session(tmp_path)
    task_config = load_task_config(session.base_dir, session.task_id)
    task_config["build_mode"] = True
    save_task_config(session.base_dir, session.task_id, task_config)
    execute_calls = []

    monkeypatch.setattr(tfb_experiment, "validate_model_config", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(tfb_experiment, "validate_variant_runtime_contract", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(tfb_experiment, "probe_variant_behavior_delta", lambda **kwargs: _active_delta())
    monkeypatch.setattr(
        tfb_experiment,
        "_auto_gate_successful_run",
        lambda **kwargs: {
            "decision": "reject",
            "gate": {"evaluation_stage": kwargs.get("evaluation_stage"), "evaluation_budget": "build_mode"},
        },
    )

    def fake_execute(**kwargs):
        execute_calls.append(kwargs)
        return (
            {"success": True, "log_paths": [], "elapsed_seconds": 0.01},
            {"status": "ok", "metric_values": {"mse_norm": 0.9}},
            {"mse_norm": 0.9},
            "success",
        )

    monkeypatch.setattr(tfb_experiment, "_execute_variant", fake_execute)

    result = tfb_experiment.run_experiment(
        session,
        {"variant_path": VALID_VARIANT_PATH},
    )

    assert result["success"] is True, result
    assert len(execute_calls) == 2, execute_calls
    assert execute_calls[0]["evaluation_budget"] == "smoke_precheck"
    assert execute_calls[1]["evaluation_budget"] == "build_mode"
    assert all(call["candidate_kind"] == "variant" for call in execute_calls)
    assert all(call["variant_entry"]["variant_path"] == VALID_VARIANT_PATH for call in execute_calls)


def test_run_experiment_contract_checks_source_checkout_before_training(monkeypatch, tmp_path) -> None:
    session = _session(tmp_path)
    source_checkout = tmp_path / "candidate_checkout"
    source_checkout.mkdir()
    contract_calls = []
    execute_calls = []

    monkeypatch.setattr(tfb_experiment, "validate_model_config", lambda *args, **kwargs: {"status": "ok"})

    def fake_contract(**kwargs):
        contract_calls.append(kwargs)
        return {
            "status": "failed",
            "error_type": "invalid_mechanism_contract",
            "error_message": "candidate forward failed",
            "error_traceback": "Traceback: candidate checkout failure",
        }

    def forbidden_execute(**kwargs):
        execute_calls.append(kwargs)
        raise AssertionError("source checkout must pass runtime contract before smoke/formal execution")

    monkeypatch.setattr(tfb_experiment, "validate_variant_runtime_contract", fake_contract)
    monkeypatch.setattr(tfb_experiment, "_execute_variant", forbidden_execute)

    result = tfb_experiment.run_experiment(
        session,
        {
            "source_checkout": str(source_checkout),
            "model_name": "fixture.baseline.Model",
        },
    )

    assert execute_calls == [], result
    assert contract_calls, result
    assert contract_calls[0]["source_checkout"] == str(source_checkout), contract_calls
    assert contract_calls[0]["variant_path"] == "config:fixture.baseline.Model", contract_calls
    assert result["success"] is False, result
    assert result["parsed_status"] == "contract_failed", result
    assert result["error_type"] == "invalid_mechanism_contract", result
    record = json.loads(Path(result["run_record_path"]).read_text(encoding="utf-8"))
    assert "candidate checkout failure" in record["run_result"]["error_traceback"], record


def test_run_experiment_smoke_and_formal_use_source_checkout(monkeypatch, tmp_path) -> None:
    session = _session(tmp_path)
    source_checkout = tmp_path / "candidate_checkout"
    source_checkout.mkdir()
    execute_calls = []

    monkeypatch.setattr(tfb_experiment, "validate_model_config", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(tfb_experiment, "validate_variant_runtime_contract", lambda **kwargs: {"status": "ok"})
    monkeypatch.setattr(tfb_experiment, "probe_variant_behavior_delta", lambda **kwargs: _active_delta())

    def fake_execute(**kwargs):
        execute_calls.append(kwargs)
        return (
            {"success": True, "log_paths": [], "elapsed_seconds": 0.01},
            {"status": "ok", "metric_values": {"mse_norm": 0.9}},
            {"mse_norm": 0.9},
            "success",
        )

    monkeypatch.setattr(tfb_experiment, "_execute_variant", fake_execute)

    result = tfb_experiment.run_experiment(
        session,
        {
            "source_checkout": str(source_checkout),
            "model_name": "fixture.baseline.Model",
        },
    )

    assert result["success"] is True, result
    assert len(execute_calls) == 2, execute_calls
    assert execute_calls[0]["evaluation_budget"] == "smoke_precheck"
    assert execute_calls[1]["evaluation_budget"] == "experiment"
    assert all(call["candidate_kind"] == "config" for call in execute_calls)
    assert all(call["source_checkout"] == str(source_checkout) for call in execute_calls)


def test_run_experiment_rejects_source_checkout_noop_before_training(monkeypatch, tmp_path) -> None:
    session = _session(tmp_path)
    source_checkout = tmp_path / "candidate_checkout"
    source_entry = source_checkout / "ts_benchmark" / "baselines" / "time_series_library" / "models" / "PatchTST.py"
    source_entry.parent.mkdir(parents=True)
    source_entry.write_text("class Model: pass\n", encoding="utf-8")
    probe_calls = []
    execute_calls = []

    monkeypatch.setattr(tfb_experiment, "validate_model_config", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(tfb_experiment, "validate_variant_runtime_contract", lambda **kwargs: {"status": "ok"})

    def fake_probe(**kwargs):
        probe_calls.append(kwargs)
        return _noop_delta()

    def forbidden_execute(**kwargs):
        execute_calls.append(kwargs)
        raise AssertionError("source checkout no-op must reject before smoke/formal training")

    monkeypatch.setattr(tfb_experiment, "probe_variant_behavior_delta", fake_probe)
    monkeypatch.setattr(tfb_experiment, "_execute_variant", forbidden_execute)

    result = tfb_experiment.run_experiment(
        session,
        {
            "source_checkout": str(source_checkout),
            "source_entry_file": str(source_entry),
            "model_name": "fixture.baseline.Model",
        },
    )

    assert probe_calls, result
    assert probe_calls[0]["source_checkout"] == str(source_checkout), probe_calls
    assert probe_calls[0]["source_entry_file"] == str(source_entry), probe_calls
    assert execute_calls == [], result
    assert result["success"] is False, result
    assert result["parsed_status"] == "noop_variant_contract", result
    assert result["error_type"] == "noop_variant_contract", result
    assert result["failure_signature"]["error_type"] == "noop_variant_contract", result


def test_execute_variant_smoke_precheck_limits_rolling_windows(monkeypatch, tmp_path) -> None:
    captured_overrides = []
    tfb_config = {
        "data_config": {"task_semantics": {"task_mode": "SS"}},
        "model_config": {"recommend_model_hyper_params": {}},
        "evaluation_config": {"strategy_args": {"horizon": 24, "num_rollings": 48000, "stride": 1}},
    }

    def fake_build_run_configs(_tfb_config, _model_entries, *, save_path, seed, override_eval_args):
        captured_overrides.append(dict(override_eval_args or {}))
        return {}, {"models": list(_model_entries)}, {"strategy_args": dict(override_eval_args or {})}

    monkeypatch.setattr(tfb_experiment, "build_run_configs", fake_build_run_configs)
    monkeypatch.setattr(
        tfb_experiment,
        "run_pipeline",
        lambda *_args, **_kwargs: {"success": True, "log_paths": [], "elapsed_seconds": 0.01},
    )

    tfb_experiment._execute_variant(
        base_dir=str(tmp_path),
        task_id="smoke_eval_policy",
        run_id="run_smoke",
        candidate_id="candidate",
        candidate_kind="config",
        tfb_config=tfb_config,
        variant_entry={"model_name": "fixture.baseline.Model", "model_hyper_params": {}},
        objective_metric="mse_norm",
        save_path="smoke_save",
        seed=2027,
        evaluation_budget="smoke_precheck",
        build_mode=False,
    )
    tfb_experiment._execute_variant(
        base_dir=str(tmp_path),
        task_id="smoke_eval_policy",
        run_id="run_formal",
        candidate_id="candidate",
        candidate_kind="config",
        tfb_config=tfb_config,
        variant_entry={"model_name": "fixture.baseline.Model", "model_hyper_params": {}},
        objective_metric="mse_norm",
        save_path="formal_save",
        seed=2027,
        evaluation_budget="experiment",
        build_mode=False,
    )

    assert captured_overrides[0]["save_true_pred"] is True
    assert captured_overrides[0]["num_rollings"] == 1
    assert captured_overrides[0]["stride"] == 24
    assert captured_overrides[1] == {"save_true_pred": True}


def test_variant_smoke_test_uses_source_checkout_and_source_entry_file(monkeypatch, tmp_path) -> None:
    session = _session(tmp_path)
    source_checkout = tmp_path / "candidate_checkout"
    source_entry = source_checkout / "ts_benchmark" / "baselines" / "time_series_library" / "models" / "PatchTST.py"
    source_entry.parent.mkdir(parents=True)
    source_entry.write_text("class PatchTST: pass\n", encoding="utf-8")
    contract_calls = []

    monkeypatch.setattr(tfb_experiment, "validate_model_config", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(tfb_experiment, "probe_variant_behavior_delta", lambda **kwargs: _active_delta())

    def fake_contract(**kwargs):
        contract_calls.append(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(tfb_experiment, "validate_variant_runtime_contract", fake_contract)

    result = tfb_experiment.run_variant_smoke_test(
        session,
        {
            "source_checkout": str(source_checkout),
            "source_entry_file": str(source_entry),
            "model_name": "fixture.baseline.Model",
        },
    )

    assert result["success"] is True, result
    assert contract_calls, result
    assert contract_calls[0]["source_checkout"] == str(source_checkout), contract_calls
    assert contract_calls[0]["variant_path"] == str(source_entry), contract_calls


def test_execute_variant_preserves_tfb_record_traceback_over_provenance_failure(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "records" / "result.csv"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("placeholder", encoding="utf-8")
    record_error = "RuntimeError: adaptive_avg_pool1d expected 2D or 3D tensor"

    monkeypatch.setattr(
        tfb_experiment,
        "build_run_configs",
        lambda *args, **kwargs: ({"data": True}, {"model": True}, {"eval": True}),
    )
    monkeypatch.setattr(
        tfb_experiment,
        "run_pipeline",
        lambda *args, **kwargs: {
            "success": True,
            "log_paths": [str(log_path)],
            "elapsed_seconds": 0.1,
        },
    )
    monkeypatch.setattr(
        tfb_experiment,
        "parse_metrics_from_paths",
        lambda *args, **kwargs: {
            "metric_values": {"mse_norm": 9.9},
            "record_errors": [record_error],
            "status": "ok",
            "warnings": [],
        },
    )
    provenance_calls = {"stamp": 0, "validate": 0}

    def forbidden_stamp(*args, **kwargs):
        provenance_calls["stamp"] += 1
        raise AssertionError("runtime-error result records must not be stamped as successful artifacts")

    def forbidden_validate(*args, **kwargs):
        provenance_calls["validate"] += 1
        raise AssertionError("runtime-error result records must not enter formal provenance validation")

    monkeypatch.setattr(tfb_experiment, "stamp_result_artifacts", forbidden_stamp)
    monkeypatch.setattr(tfb_experiment, "validate_result_artifact_provenance", forbidden_validate)

    run_result, parsed, metrics, label = tfb_experiment._execute_variant(
        base_dir=str(tmp_path),
        task_id="task",
        run_id="run",
        candidate_id="candidate",
        candidate_kind="config",
        tfb_config={"evaluation_config": {"strategy_args": {"strategy_name": "rolling_forecast"}}},
        variant_entry={"model_name": "fixture.baseline.Model", "model_hyper_params": {}},
        objective_metric="mse_norm",
        save_path=str(tmp_path / "save"),
        seed=2021,
        evaluation_budget="formal",
        build_mode=False,
        source_checkout=str(tmp_path),
    )

    assert run_result["success"] is False, run_result
    assert record_error in run_result["error_traceback"], run_result
    assert "formal forecast path is not batch_forecast" not in run_result["error_traceback"], run_result
    assert parsed["metric_values"] == {}, parsed
    assert metrics == {}, metrics
    assert label != "success", label
    assert run_result["artifact_provenance"]["validation"]["status"] == "skipped", run_result
    assert provenance_calls == {"stamp": 0, "validate": 0}


def test_variant_smoke_test_rejects_noop_with_failure_evidence(monkeypatch, tmp_path) -> None:
    session = _session(tmp_path)

    monkeypatch.setattr(tfb_experiment, "validate_model_config", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(tfb_experiment, "validate_variant_runtime_contract", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(tfb_experiment, "probe_variant_behavior_delta", lambda **kwargs: _noop_delta())

    result = tfb_experiment.run_variant_smoke_test(
        session,
        {"variant_path": VALID_VARIANT_PATH},
    )

    assert result["status"] == "failed", result
    assert result["success"] is False, result
    assert result["error_type"] == "noop_variant_contract", result
    assert result["behavior_delta"]["suspected_noop"] is True, result
    assert result["failure_signature"]["error_type"] == "noop_variant_contract", result
    assert result["failure_evidence"]["probe_policy"] == "same_input_forward_only_no_baseline_training", result
    assert result["failure_evidence"]["repair_scope_hint"]["baseline_training_rerun_allowed"] is False, result


def test_variant_smoke_test_accepts_objective_noop_when_additional_loss_is_proven(monkeypatch, tmp_path) -> None:
    session = _session(tmp_path)
    runtime_contract = {
        "status": "ok",
        "mechanism_probe": {
            "status": "ok",
            "cases": [
                {
                    "name": "full_model.train_backward",
                    "status": "ok",
                    "has_additional_loss": True,
                    "additional_loss_shape": [],
                }
            ],
        },
    }

    monkeypatch.setattr(tfb_experiment, "validate_model_config", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(tfb_experiment, "validate_variant_runtime_contract", lambda *args, **kwargs: runtime_contract)
    monkeypatch.setattr(tfb_experiment, "probe_variant_behavior_delta", lambda **kwargs: _noop_delta())

    result = tfb_experiment.run_variant_smoke_test(
        session,
        {"variant_path": VALID_VARIANT_PATH},
    )

    assert result["status"] == "ok", result
    assert result["success"] is True, result
    assert not result.get("error_type"), result
    assert result["behavior_delta"]["suspected_noop"] is True, result
    assert result["behavior_delta"]["objective_noop_allowed"] is True, result
