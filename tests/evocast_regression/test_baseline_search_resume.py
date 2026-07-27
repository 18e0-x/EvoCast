from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from evocast.research.dataset_profile import write_skipped_dataset_profile
from evocast.runners import baseline_search_runner
from evocast.state.runtime.resume_state import atomic_write_json, baseline_search_state_path
from evocast.state.runtime.trial_journal import append_node, create_node
from evocast.state.runtime.store import load_runtime_state
from evocast.state.domain_store import load_task_config, save_task_config


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _prepare_task(base_dir: Path, task_id: str) -> None:
    knowledge_dir = base_dir / "task_knowledge" / task_id
    compiled = {
        "data_config": {
            "feature_dict": {"if_univariate": False},
            "task_semantics": {"task_mode": "MM"},
        },
        "model_config": {"models": [], "recommend_model_hyper_params": {}},
        "evaluation_config": {"strategy_args": {"strategy_name": "rolling_forecast"}},
    }
    _write_json(knowledge_dir / "compiled_config.json", compiled)
    _write_json(
        knowledge_dir / "task_config.json",
        {
            "task_id": task_id,
            "config_path": str(knowledge_dir / "compiled_config.json"),
            "objective_metric": "mse_norm",
            "metric_direction": "lower_is_better",
        },
    )
    write_skipped_dataset_profile(task_id=task_id, base_dir=str(base_dir))


def _registry() -> list[dict]:
    return [
        {
            "model_key": "LinearFixture",
            "import_path": "pkg.LinearFixture",
            "family": "linear",
            "verified_import": True,
            "local_code": True,
            "supports_univariate": True,
            "supports_multivariate": True,
            "cost_level": 1,
            "reliability": 0,
        },
        {
            "model_key": "TransformerFixture",
            "import_path": "pkg.TransformerFixture",
            "family": "transformer",
            "verified_import": True,
            "local_code": True,
            "supports_univariate": True,
            "supports_multivariate": True,
            "cost_level": 1,
            "reliability": 0,
        },
    ]


def _mock_baseline_reference(monkeypatch) -> None:
    def _fake_write_initial_baseline_reference(**kwargs):
        path = Path(str(kwargs["base_dir"])) / "task_knowledge" / str(kwargs["task_id"]) / "baseline_reference.json"
        payload = {"status": "completed", "path": str(path)}
        _write_json(path, payload)
        return payload

    monkeypatch.setattr(baseline_search_runner, "write_initial_baseline_reference", _fake_write_initial_baseline_reference)


def _append_successful_baseline(base_dir: Path, task_id: str, model_key: str, node_id: str, value: float) -> None:
    node = create_node(
        task_id,
        node_id,
        action_type="baseline",
        model_name=model_key,
        model_config={"model_name": f"pkg.{model_key}", "model_hyper_params": {}},
        objective_metric="mse_norm",
        metrics={"mse_norm": value},
        status="success",
        error_type="success",
        artifact_paths=[str(base_dir / "result" / model_key / "artifact.csv")],
    )
    node["completed_at"] = datetime.now().isoformat()
    append_node(task_id, node, str(base_dir))


def test_baseline_search_resume_skips_successful_journal_nodes(tmp_path, monkeypatch) -> None:
    base_dir = tmp_path / "evocast"
    task_id = "baseline_resume_skip_success"
    _prepare_task(base_dir, task_id)
    _append_successful_baseline(base_dir, task_id, "LinearFixture", "baseline_001_LinearFixture", 0.4)

    monkeypatch.setattr(baseline_search_runner, "build_registry", lambda verify=True: _registry())
    monkeypatch.setattr(baseline_search_runner, "augment_registry_with_facts", lambda value: value)
    _mock_baseline_reference(monkeypatch)
    executed: list[str] = []

    def _fake_run_baseline_candidate(**kwargs):
        model_key = str(kwargs["spec"]["model_key"])
        executed.append(model_key)
        return {
            "status": "success",
            "objective_value": 0.3,
            "metrics": {"mse_norm": 0.3},
            "model_key": model_key,
            "family": str(kwargs["spec"].get("family") or ""),
            "node_id": str(kwargs["node_id"]),
            "model_config": {"model_name": str(kwargs["spec"].get("import_path") or ""), "model_hyper_params": {}},
            "run_result": {"log_paths": []},
        }

    monkeypatch.setattr(baseline_search_runner, "run_baseline_candidate", _fake_run_baseline_candidate)

    summary = baseline_search_runner.run_baseline_search(
        task_id=task_id,
        baseline_strategy="auto",
        objective_metric="mse_norm",
        metric_direction="lower_is_better",
        base_dir=str(base_dir),
    )

    assert executed == ["TransformerFixture"]
    assert summary["status"] == "completed"
    assert summary["total"] == 2
    assert summary["successes"] == 2
    state = load_runtime_state(str(base_dir), task_id).baseline_search_progress
    assert state["status"] == "completed"
    assert [item["model_key"] for item in state["candidate_order"]] == ["LinearFixture", "TransformerFixture"]


def test_build_mode_keeps_auto_baseline_candidate_selection_but_uses_smoke_budget(tmp_path, monkeypatch) -> None:
    base_dir = tmp_path / "evocast"
    task_id = "build_mode_auto_baseline_selection"
    _prepare_task(base_dir, task_id)
    task_config = load_task_config(str(base_dir), task_id)
    task_config["build_mode"] = True
    save_task_config(str(base_dir), task_id, task_config)

    monkeypatch.setattr(baseline_search_runner, "build_registry", lambda verify=True: _registry())
    monkeypatch.setattr(baseline_search_runner, "augment_registry_with_facts", lambda value: value)
    _mock_baseline_reference(monkeypatch)
    calls: list[dict] = []

    def _fake_run_baseline_candidate(**kwargs):
        calls.append(kwargs)
        model_key = str(kwargs["spec"]["model_key"])
        return {
            "status": "success", "objective_value": 0.5, "metrics": {"mse_norm": 0.5},
            "model_key": model_key, "family": str(kwargs["spec"].get("family") or ""),
            "node_id": str(kwargs["node_id"]),
            "model_config": {"model_name": str(kwargs["spec"].get("import_path") or ""), "model_hyper_params": {}},
            "run_result": {"log_paths": []},
        }

    monkeypatch.setattr(baseline_search_runner, "run_baseline_candidate", _fake_run_baseline_candidate)
    summary = baseline_search_runner.run_baseline_search(task_id=task_id, baseline_strategy="auto", base_dir=str(base_dir))

    assert [item["spec"]["model_key"] for item in calls] == ["LinearFixture", "TransformerFixture"]
    assert all(item["budget"] == "smoke_test" for item in calls)
    assert summary["training_budget"] == "smoke_test"
    assert summary["total"] == 2


def test_baseline_search_resume_reuses_candidate_snapshot(tmp_path, monkeypatch) -> None:
    base_dir = tmp_path / "evocast"
    task_id = "baseline_resume_snapshot"
    _prepare_task(base_dir, task_id)
    _append_successful_baseline(base_dir, task_id, "LinearFixture", "baseline_001_LinearFixture", 0.4)
    registry = _registry()
    candidate_order = [
        {"index": 1, "node_id": "baseline_001_LinearFixture", "model_key": "LinearFixture", "spec": registry[0]},
        {"index": 2, "node_id": "baseline_002_TransformerFixture", "model_key": "TransformerFixture", "spec": registry[1]},
    ]
    atomic_write_json(
        baseline_search_state_path(str(base_dir), task_id),
        {
            "status": "running",
            "strategy": "auto",
            "budget": "unified",
            "candidate_order": candidate_order,
            "selection_report": {"strategy": "from_previous_snapshot"},
            "run_results": [],
        },
    )

    monkeypatch.setattr(baseline_search_runner, "build_registry", lambda verify=True: registry)
    monkeypatch.setattr(baseline_search_runner, "augment_registry_with_facts", lambda value: value)
    monkeypatch.setattr(
        baseline_search_runner,
        "select_baseline_candidates",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("selection must not run during snapshot resume")),
    )
    _mock_baseline_reference(monkeypatch)
    executed: list[str] = []

    def _fake_run_baseline_candidate(**kwargs):
        model_key = str(kwargs["spec"]["model_key"])
        executed.append(model_key)
        return {
            "status": "success",
            "objective_value": 0.3,
            "metrics": {"mse_norm": 0.3},
            "model_key": model_key,
            "family": str(kwargs["spec"].get("family") or ""),
            "node_id": str(kwargs["node_id"]),
            "model_config": {"model_name": str(kwargs["spec"].get("import_path") or ""), "model_hyper_params": {}},
            "run_result": {"log_paths": []},
        }

    monkeypatch.setattr(baseline_search_runner, "run_baseline_candidate", _fake_run_baseline_candidate)

    summary = baseline_search_runner.run_baseline_search(
        task_id=task_id,
        baseline_strategy="auto",
        objective_metric="mse_norm",
        metric_direction="lower_is_better",
        base_dir=str(base_dir),
    )

    assert executed == ["TransformerFixture"]
    assert summary["selection_report"] == {"strategy": "from_previous_snapshot"}
    assert [item["model_key"] for item in summary["candidate_order"]] == ["LinearFixture", "TransformerFixture"]
