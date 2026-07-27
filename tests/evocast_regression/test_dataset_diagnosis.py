from __future__ import annotations

import json
from pathlib import Path

import evocast.research.dataset_profile as dataset_profile_module
from evocast.research.dataset_profile import (
    SCHEMA_VERSION,
    dataset_profile_path,
    ensure_dataset_profile,
    generate_dataset_profile,
    has_dataset_profile_or_intentional_skip,
    write_skipped_dataset_profile,
)
from evocast.runners import baseline_search_runner
from characteristics_extractor.Characteristics_Extractor import TimeSeriesProcessor


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_series_csv(path: Path, rows: int = 256) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["date,target,aux"]
    for idx in range(rows):
        target = (idx % 24) + 0.1 * idx
        aux = (idx % 12) * 0.5 + 0.05 * idx
        lines.append(f"2024-01-{(idx % 28) + 1:02d} {idx % 24:02d}:00:00,{target:.4f},{aux:.4f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _prepare_task(base_dir: Path, task_id: str, dataset_path: Path) -> None:
    knowledge_dir = base_dir / "task_knowledge" / task_id
    semantics = {
        "task_mode": "MM",
        "input_variable_topology": "multivariate",
        "prediction_target_selection": "all_targets",
        "target_columns": ["target", "aux"],
        "time_col": "date",
        "dataset_path": str(dataset_path),
        "frequency": "hourly",
        "strategy_name": "rolling_forecast",
        "objective_metric": "mse_norm",
        "horizons": [24],
        "input_chunk_length": 48,
    }
    compiled = {
        "data_config": {
            "feature_dict": {
                "if_univariate": False,
                "if_trend": True,
                "if_season": True,
                "canonical_freq": "hourly",
                "freq": "h",
            },
            "data_set_name": "large_forecast",
            "dataset_path": str(dataset_path),
            "data_name_list": [dataset_path.name],
            "time_col": "date",
            "target_columns": ["target", "aux"],
            "task_semantics": semantics,
            "scale": True,
        },
        "model_config": {
            "models": [],
            "recommend_model_hyper_params": {
                "input_chunk_length": 48,
                "output_chunk_length": 24,
            },
        },
        "evaluation_config": {
            "metrics": "all",
            "strategy_args": {
                "strategy_name": "rolling_forecast",
                "horizon": 24,
                "tv_ratio": 0.8,
                "train_ratio_in_tv": {"__default__": 0.875},
                "stride": 1,
                "num_rollings": 8,
                "seed": 2021,
                "deterministic": "efficient",
            },
        },
    }
    task_config = {
        "task_id": task_id,
        "config_path": str(knowledge_dir / "compiled_config.json"),
        "objective_metric": "mse_norm",
        "metric_direction": "lower_is_better",
        "budget": "unified",
        "max_rounds": 3,
        "force_full_rounds": True,
        "baseline_strategy": "manual",
        "baseline_models": ["Fixture"],
        "data_set_name": "large_forecast",
        "dataset_path": str(dataset_path),
        "horizon": 24,
        "seq_len": 48,
        "api_config": "",
        "task_semantics": semantics,
        "feature_dict": compiled["data_config"]["feature_dict"],
    }
    _write_json(knowledge_dir / "compiled_config.json", compiled)
    _write_json(knowledge_dir / "task_config.json", task_config)


def _mock_baseline_reference(monkeypatch) -> None:
    def _fake_write_initial_baseline_reference(**kwargs):
        base_dir = Path(str(kwargs["base_dir"]))
        task_id = str(kwargs["task_id"])
        path = base_dir / "task_knowledge" / task_id / "baseline_reference.json"
        payload = {
            "schema_version": "baseline_reference_v1",
            "task_id": task_id,
            "candidate_id": "baseline_001_Fixture",
            "node_id": "baseline_001_Fixture_baseline_reference",
            "candidate_kind": "baseline",
            "variant_path": None,
            "objective_metric": kwargs.get("objective_metric") or "mse_norm",
            "metric_stats": {
                "mse_norm": {
                    "mean": 0.5,
                    "std": 0.0,
                    "seed_count": 3,
                }
            },
            "source_clean": True,
            "generated_before_first_variant": True,
            "path": str(path),
            "result_path": str(path.with_name("baseline_reference_seed_eval.json")),
        }
        _write_json(path, payload)
        return payload

    monkeypatch.setattr(baseline_search_runner, "write_initial_baseline_reference", _fake_write_initial_baseline_reference)


def test_generate_dataset_profile_persists_formal_profile(tmp_path) -> None:
    base_dir = tmp_path / "evocast"
    task_id = "dataset_profile_basic"
    dataset_path = base_dir / "dataset" / "forecasting" / "fixture.csv"
    _write_series_csv(dataset_path)
    _prepare_task(base_dir, task_id, dataset_path)

    profile = generate_dataset_profile(
        task_id=task_id,
        base_dir=str(base_dir),
        client=None,
    )

    assert profile["schema_version"] == SCHEMA_VERSION
    assert profile["status"] == "ok"
    assert profile["characteristics_engine"] == "python"
    assert isinstance(profile["diagnostics"], list)
    assert set(profile["raw_characteristics"]) >= {
        "Correlation",
        "Transition",
        "Shifting",
        "Seasonality",
        "Trend",
        "Stationarity",
        "Short_term_jsd",
        "Long_term_jsd",
    }
    assert profile["llm_narrative"]["status"] == "deterministic"
    assert "数据集诊断" in profile["llm_narrative"]["dataset_summary"]
    assert "建议优先关注" in profile["llm_narrative"]["research_interpretation"]
    assert dataset_profile_path(str(base_dir), task_id).exists()
    assert ".evocast" not in str(dataset_profile_path(str(base_dir), task_id))
    assert "dataset_knowledge" in str(dataset_profile_path(str(base_dir), task_id))
    binding_path = base_dir / "dataset_knowledge" / "task_bindings" / f"{task_id}.json"
    assert binding_path.exists()
    assert (base_dir / "dataset_knowledge" / "registry.json").exists()


def test_dataset_characteristics_are_reused_across_tasks_and_invalidated_by_source_change(tmp_path, monkeypatch) -> None:
    base_dir = tmp_path / "evocast"
    dataset_path = base_dir / "dataset" / "forecasting" / "fixture.csv"
    _write_series_csv(dataset_path)
    _prepare_task(base_dir, "diagnosis_cache_first", dataset_path)
    _prepare_task(base_dir, "diagnosis_cache_second", dataset_path)

    original = dataset_profile_module._compute_raw_characteristics
    calls = {"count": 0}

    def _counted_compute(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(dataset_profile_module, "_compute_raw_characteristics", _counted_compute)

    first = generate_dataset_profile(task_id="diagnosis_cache_first", base_dir=str(base_dir), client=None)
    second = generate_dataset_profile(task_id="diagnosis_cache_second", base_dir=str(base_dir), client=None)

    assert first["characteristics_cache"]["hit"] is False
    assert second["characteristics_cache"]["hit"] is True
    assert first["characteristics_cache"]["key"] == second["characteristics_cache"]["key"]
    assert first["raw_characteristics"] == second["raw_characteristics"]
    assert calls["count"] == 1
    assert len(list((base_dir / "dataset_knowledge" / "datasets").glob("*/views/*/characteristics.json"))) == 1

    dataset_path.write_text(dataset_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _prepare_task(base_dir, "diagnosis_cache_after_source_change", dataset_path)
    changed = generate_dataset_profile(
        task_id="diagnosis_cache_after_source_change",
        base_dir=str(base_dir),
        client=None,
    )

    assert changed["characteristics_cache"]["hit"] is False
    assert changed["characteristics_cache"]["key"] != first["characteristics_cache"]["key"]
    assert calls["count"] == 2


def test_dataset_characteristics_artifact_respects_analysis_view_and_force_refresh(tmp_path, monkeypatch) -> None:
    base_dir = tmp_path / "evocast"
    dataset_path = base_dir / "dataset" / "forecasting" / "fixture.csv"
    _write_series_csv(dataset_path)
    _prepare_task(base_dir, "diagnosis_cache_full", dataset_path)
    _prepare_task(base_dir, "diagnosis_cache_target_only", dataset_path)
    _prepare_task(base_dir, "diagnosis_cache_refresh", dataset_path)

    target_task_path = base_dir / "task_knowledge" / "diagnosis_cache_target_only" / "task_config.json"
    target_task = json.loads(target_task_path.read_text(encoding="utf-8"))
    target_task["task_semantics"]["task_mode"] = "MS"
    target_task["task_semantics"]["target_columns"] = ["target"]
    target_task_path.write_text(json.dumps(target_task), encoding="utf-8")

    original = dataset_profile_module._compute_raw_characteristics
    calls = {"count": 0}

    def _counted_compute(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(dataset_profile_module, "_compute_raw_characteristics", _counted_compute)
    full = generate_dataset_profile(task_id="diagnosis_cache_full", base_dir=str(base_dir), client=None)
    target_only = generate_dataset_profile(task_id="diagnosis_cache_target_only", base_dir=str(base_dir), client=None)
    refreshed = generate_dataset_profile(
        task_id="diagnosis_cache_refresh",
        base_dir=str(base_dir),
        client=None,
        force_refresh=True,
    )

    assert full["characteristics_cache"]["key"] != target_only["characteristics_cache"]["key"]
    assert target_only["characteristics_cache"]["hit"] is False
    assert refreshed["characteristics_cache"]["hit"] is False
    assert calls["count"] == 3


def test_dataset_profile_reports_progress_stages(tmp_path) -> None:
    base_dir = tmp_path / "evocast"
    task_id = "dataset_profile_progress"
    dataset_path = base_dir / "dataset" / "forecasting" / "fixture.csv"
    _write_series_csv(dataset_path)
    _prepare_task(base_dir, task_id, dataset_path)
    messages: list[str] = []

    profile = generate_dataset_profile(
        task_id=task_id,
        base_dir=str(base_dir),
        client=None,
        progress=messages.append,
    )

    joined = "\n".join(messages)
    assert profile["status"] == "ok"
    assert "开始数据集诊断" in joined
    assert "CSV 读取完成" in joined
    assert "开始计算 Python 数据集特征" in joined
    assert "计算序列特征" in joined
    assert "LLM narrative 完成" in joined
    assert "写入 dataset_knowledge profile" in joined


def test_dataset_profile_reports_english_progress_when_requested(tmp_path) -> None:
    base_dir = tmp_path / "evocast"
    task_id = "dataset_profile_progress_en"
    dataset_path = base_dir / "dataset" / "forecasting" / "fixture.csv"
    _write_series_csv(dataset_path)
    _prepare_task(base_dir, task_id, dataset_path)
    messages: list[str] = []

    profile = generate_dataset_profile(
        task_id=task_id,
        base_dir=str(base_dir),
        client=None,
        progress=messages.append,
        language="en",
    )

    joined = "\n".join(messages)
    assert profile["status"] == "ok"
    assert "Starting dataset diagnosis" in joined
    assert "CSV read complete" in joined
    assert "Computing Python dataset characteristics" in joined
    assert "Computing series characteristics" in joined
    assert "LLM narrative complete" in joined
    assert "Writing dataset-knowledge profile" in joined
    assert "Dataset diagnosis found" in profile["llm_narrative"]["dataset_summary"]


def test_ensure_dataset_profile_regenerates_invalid_cache(tmp_path) -> None:
    base_dir = tmp_path / "evocast"
    task_id = "dataset_profile_regenerate"
    dataset_path = base_dir / "dataset" / "forecasting" / "fixture.csv"
    _write_series_csv(dataset_path)
    _prepare_task(base_dir, task_id, dataset_path)

    bad_cache = dataset_profile_path(str(base_dir), task_id)
    bad_cache.parent.mkdir(parents=True, exist_ok=True)
    bad_cache.write_text("{}", encoding="utf-8")

    profile = ensure_dataset_profile(task_id=task_id, base_dir=str(base_dir), client=None)

    assert profile["schema_version"] == SCHEMA_VERSION
    assert json.loads(bad_cache.read_text(encoding="utf-8"))["schema_version"] == SCHEMA_VERSION


def test_required_dataset_diagnosis_does_not_reuse_skipped_placeholder(tmp_path) -> None:
    base_dir = tmp_path / "evocast"
    task_id = "dataset_profile_required_after_skip"
    dataset_path = base_dir / "dataset" / "forecasting" / "fixture.csv"
    _write_series_csv(dataset_path)
    _prepare_task(base_dir, task_id, dataset_path)

    skipped = write_skipped_dataset_profile(task_id=task_id, base_dir=str(base_dir))
    assert skipped["status"] == "skipped"

    profile = ensure_dataset_profile(task_id=task_id, base_dir=str(base_dir), client=None)

    assert profile["schema_version"] == SCHEMA_VERSION
    assert profile["status"] == "ok"
    assert profile["characteristics_engine"] == "python"
    assert json.loads(dataset_profile_path(str(base_dir), task_id).read_text(encoding="utf-8"))["status"] == "ok"


def test_skipped_dataset_profile_unlocks_fast_research_path(tmp_path, monkeypatch) -> None:
    base_dir = tmp_path / "evocast"
    task_id = "baseline_accepts_skipped_dataset_profile"
    dataset_path = base_dir / "dataset" / "forecasting" / "fixture.csv"
    _write_series_csv(dataset_path)
    _prepare_task(base_dir, task_id, dataset_path)

    monkeypatch.setattr(
        baseline_search_runner,
        "build_registry",
        lambda verify=True: [{"model_key": "Fixture", "import_path": "pkg.Fixture", "family": "transformer"}],
    )
    monkeypatch.setattr(baseline_search_runner, "augment_registry_with_facts", lambda registry: registry)
    monkeypatch.setattr(baseline_search_runner, "resolve_model_key", lambda model_key, registry: ("Fixture", "exact"))
    monkeypatch.setattr(
        baseline_search_runner,
        "generate_model_entry",
        lambda spec: {"model_name": str(spec.get("import_path") or ""), "adapter": None, "model_hyper_params": {}},
    )

    def _fake_run_baseline_candidate(**kwargs):
        spec = kwargs["spec"]
        return {
            "status": "success",
            "objective_value": 0.5,
            "metrics": {"mse_norm": 0.5},
            "model_key": str(spec.get("model_key") or "Fixture"),
            "family": str(spec.get("family") or "transformer"),
            "node_id": str(kwargs.get("node_id") or "baseline_001_Fixture"),
            "elapsed_seconds": 0.1,
            "model_config": {"model_name": str(spec.get("import_path") or "pkg.Fixture")},
            "run_result": {"log_paths": []},
            "tags": [],
        }

    monkeypatch.setattr(baseline_search_runner, "run_baseline_candidate", _fake_run_baseline_candidate)
    _mock_baseline_reference(monkeypatch)

    try:
        baseline_search_runner.run_baseline_search(
            task_id=task_id,
            baseline_strategy="manual",
            manual_models=["Fixture"],
            objective_metric="mse_norm",
            metric_direction="lower_is_better",
            base_dir=str(base_dir),
        )
    except RuntimeError as exc:
        assert "DATASET_DIAGNOSIS_REQUIRED" in str(exc)
    else:
        raise AssertionError("baseline search must require completed dataset diagnosis")

    skipped = write_skipped_dataset_profile(task_id=task_id, base_dir=str(base_dir))
    assert skipped["status"] == "skipped"
    assert skipped["derived_claims"] == []
    assert skipped["diagnostics"][0]["message"].startswith("数据集诊断")
    assert skipped["llm_narrative"]["limitations"][0].startswith("数据集诊断")
    assert has_dataset_profile_or_intentional_skip(str(base_dir), task_id)

    summary = baseline_search_runner.run_baseline_search(
        task_id=task_id,
        baseline_strategy="manual",
        manual_models=["Fixture"],
        objective_metric="mse_norm",
        metric_direction="lower_is_better",
        base_dir=str(base_dir),
    )

    assert summary["status"] == "completed"
    assert summary["dataset_diagnosis"]["status"] == "skipped"
    assert summary["dataset_diagnosis"]["characteristics_engine"] == "skipped"
    assert dataset_profile_path(str(base_dir), task_id).exists()


def test_baseline_search_accepts_completed_dataset_profile(tmp_path, monkeypatch) -> None:
    base_dir = tmp_path / "evocast"
    task_id = "baseline_ensures_dataset_profile"
    dataset_path = base_dir / "dataset" / "forecasting" / "fixture.csv"
    _write_series_csv(dataset_path)
    _prepare_task(base_dir, task_id, dataset_path)

    monkeypatch.setattr(
        baseline_search_runner,
        "build_registry",
        lambda verify=True: [{"model_key": "Fixture", "import_path": "pkg.Fixture", "family": "transformer"}],
    )
    monkeypatch.setattr(baseline_search_runner, "augment_registry_with_facts", lambda registry: registry)
    monkeypatch.setattr(baseline_search_runner, "resolve_model_key", lambda model_key, registry: ("Fixture", "exact"))
    monkeypatch.setattr(
        baseline_search_runner,
        "generate_model_entry",
        lambda spec: {"model_name": str(spec.get("import_path") or ""), "adapter": None, "model_hyper_params": {}},
    )

    def _fake_run_baseline_candidate(**kwargs):
        spec = kwargs["spec"]
        return {
            "status": "success",
            "objective_value": 0.5,
            "metrics": {"mse_norm": 0.5},
            "model_key": str(spec.get("model_key") or "Fixture"),
            "family": str(spec.get("family") or "transformer"),
            "node_id": str(kwargs.get("node_id") or "baseline_001_Fixture"),
            "elapsed_seconds": 0.1,
            "model_config": {"model_name": str(spec.get("import_path") or "pkg.Fixture")},
            "run_result": {"log_paths": []},
            "tags": [],
        }

    monkeypatch.setattr(baseline_search_runner, "run_baseline_candidate", _fake_run_baseline_candidate)
    _mock_baseline_reference(monkeypatch)
    generate_dataset_profile(task_id=task_id, base_dir=str(base_dir), client=None)
    summary = baseline_search_runner.run_baseline_search(
        task_id=task_id,
        baseline_strategy="manual",
        manual_models=["Fixture"],
        objective_metric="mse_norm",
        metric_direction="lower_is_better",
        base_dir=str(base_dir),
    )

    assert summary["status"] == "completed"
    assert summary["dataset_diagnosis"]["status"] == "ok"
    assert summary["dataset_diagnosis"]["characteristics_engine"] == "python"


def test_auto_baseline_search_uses_deterministic_curator_report(tmp_path, monkeypatch) -> None:
    base_dir = tmp_path / "evocast"
    task_id = "baseline_auto_curator"
    dataset_path = base_dir / "dataset" / "forecasting" / "fixture.csv"
    _write_series_csv(dataset_path)
    _prepare_task(base_dir, task_id, dataset_path)
    policy_dir = base_dir / "configs" / "policies"
    policy_dir.mkdir(parents=True)
    (policy_dir / "experiment.yaml").write_text(
        """
baseline_search:
  candidate_count: 2
  registry_pool_size: 4
  preferred_families: [linear, transformer]
  initial_seeds: []
""".strip(),
        encoding="utf-8",
    )

    registry = [
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
            "sentinel": False,
        },
        {
            "model_key": "TransformerFixture",
            "import_path": "pkg.TransformerFixture",
            "family": "transformer",
            "verified_import": True,
            "local_code": True,
            "supports_univariate": True,
            "supports_multivariate": True,
            "cost_level": 2,
            "reliability": 0,
            "sentinel": False,
        },
        {
            "model_key": "UniOnlyFixture",
            "import_path": "pkg.UniOnlyFixture",
            "family": "mlp",
            "verified_import": True,
            "local_code": True,
            "supports_univariate": True,
            "supports_multivariate": False,
            "cost_level": 1,
            "reliability": 0,
            "sentinel": True,
        },
    ]
    monkeypatch.setattr(baseline_search_runner, "build_registry", lambda verify=True: registry)
    monkeypatch.setattr(baseline_search_runner, "augment_registry_with_facts", lambda value: value)
    monkeypatch.setattr(
        baseline_search_runner,
        "generate_model_entry",
        lambda spec: {"model_name": str(spec.get("import_path") or ""), "adapter": None, "model_hyper_params": {}},
    )

    executed: list[str] = []

    def _fake_run_baseline_candidate(**kwargs):
        spec = kwargs["spec"]
        model_key = str(spec.get("model_key") or "")
        executed.append(model_key)
        return {
            "status": "success",
            "objective_value": 0.5 + len(executed),
            "metrics": {"mse_norm": 0.5 + len(executed)},
            "model_key": model_key,
            "family": str(spec.get("family") or ""),
            "node_id": str(kwargs.get("node_id") or f"baseline_{len(executed):03d}_{model_key}"),
            "elapsed_seconds": 0.1,
            "model_config": {"model_name": str(spec.get("import_path") or "")},
            "run_result": {"log_paths": []},
            "tags": [],
        }

    monkeypatch.setattr(baseline_search_runner, "run_baseline_candidate", _fake_run_baseline_candidate)
    _mock_baseline_reference(monkeypatch)
    generate_dataset_profile(task_id=task_id, base_dir=str(base_dir), client=None)

    summary = baseline_search_runner.run_baseline_search(
        task_id=task_id,
        baseline_strategy="auto",
        objective_metric="mse_norm",
        metric_direction="lower_is_better",
        base_dir=str(base_dir),
    )

    assert summary["status"] == "completed"
    assert executed == ["LinearFixture", "TransformerFixture"]
    report_path = Path(summary["selection_report_path"])
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["strategy"] == "deterministic_curator"
    assert [item["model_key"] for item in report["selected"]] == ["LinearFixture", "TransformerFixture"]
    assert any(item["reason"] == "task_mode_not_supported" for item in report["pool_rejected"])


def test_baseline_search_requires_dataset_diagnosis_before_execution(tmp_path) -> None:
    base_dir = tmp_path / "evocast"
    task_id = "baseline_requires_dataset_diagnosis"
    dataset_path = base_dir / "dataset" / "forecasting" / "fixture.csv"
    _write_series_csv(dataset_path)
    _prepare_task(base_dir, task_id, dataset_path)

    assert has_dataset_profile_or_intentional_skip(str(base_dir), task_id) is False
    try:
        baseline_search_runner.run_baseline_search(
            task_id=task_id,
            baseline_strategy="manual",
            manual_models=["Fixture"],
            objective_metric="mse_norm",
            metric_direction="lower_is_better",
            base_dir=str(base_dir),
            dry_run=True,
        )
    except RuntimeError as exc:
        assert "DATASET_DIAGNOSIS_REQUIRED" in str(exc)
    else:
        raise AssertionError("baseline search must require dataset diagnosis before execution")


def test_intentional_dataset_diagnosis_skip_unblocks_baseline_search(tmp_path) -> None:
    base_dir = tmp_path / "evocast"
    task_id = "dataset_diagnosis_skip_unblocks_baseline"
    dataset_path = base_dir / "dataset" / "forecasting" / "fixture.csv"
    _write_series_csv(dataset_path)
    _prepare_task(base_dir, task_id, dataset_path)
    write_skipped_dataset_profile(task_id=task_id, base_dir=str(base_dir))

    assert has_dataset_profile_or_intentional_skip(str(base_dir), task_id) is True


def test_dataset_profile_failure_is_explicit(tmp_path) -> None:
    base_dir = tmp_path / "evocast"
    task_id = "dataset_profile_failed"
    missing_path = base_dir / "dataset" / "forecasting" / "missing.csv"
    _prepare_task(base_dir, task_id, missing_path)

    profile = generate_dataset_profile(task_id=task_id, base_dir=str(base_dir), client=None)

    assert profile["schema_version"] == SCHEMA_VERSION
    assert profile["status"] == "failed"
    assert profile["characteristics_engine"] == "python"
    assert any(item.get("severity") == "error" for item in profile["diagnostics"])


def test_pure_python_characteristics_extractor_writes_single_series_outputs(tmp_path) -> None:
    data_path = tmp_path / "single.csv"
    output_dir = tmp_path / "characteristics"
    lines = ["date,target"]
    for idx in range(160):
        lines.append(f"2024-01-{(idx % 28) + 1:02d},{idx % 24 + idx * 0.01:.4f}")
    data_path.write_text("\n".join(lines), encoding="utf-8")

    TimeSeriesProcessor(output_dir=str(output_dir)).process_path(str(data_path))

    assert (output_dir / "All_characteristics_single.csv").exists()
    tfb_path = output_dir / "TFB_characteristics_single.csv"
    assert tfb_path.exists()
    header = tfb_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == [
        "Correlation",
        "Transition",
        "Shifting",
        "Seasonality",
        "Trend",
        "Stationarity",
        "Short_term_jsd",
        "Long_term_jsd",
    ]
    assert not (output_dir / "mean_All_characteristics_single.csv").exists()


def test_pure_python_characteristics_extractor_writes_multivariate_mean_outputs(tmp_path) -> None:
    data_path = tmp_path / "multi.csv"
    output_dir = tmp_path / "characteristics"
    _write_series_csv(data_path, rows=180)

    TimeSeriesProcessor(output_dir=str(output_dir)).process_path(str(data_path))

    assert (output_dir / "All_characteristics_multi.csv").exists()
    assert (output_dir / "TFB_characteristics_multi.csv").exists()
    assert (output_dir / "mean_All_characteristics_multi.csv").exists()
    assert (output_dir / "mean_TFB_characteristics_multi.csv").exists()
