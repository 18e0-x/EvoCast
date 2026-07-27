from __future__ import annotations

import json
from pathlib import Path

from tests.evocast_regression.buildmode_acceptance import main
from tests.evocast_regression.buildmode_acceptance import _evaluate_acceptance
from evocast.research.dataset_profile import write_skipped_dataset_profile
from evocast.state.domain_store import load_task_config, save_round_record


def _sync_fixture_rounds_to_canonical(base_dir: Path, task_id: str) -> None:
    """Keep artifact fixtures as evidence while making RoundStore authoritative."""
    for path in sorted((base_dir / "task_knowledge" / task_id / "rounds").glob("Research*/round.json")):
        save_round_record(str(base_dir), task_id, json.loads(path.read_text(encoding="utf-8")))


def test_buildmode_acceptance_prepare_writes_formal_two_round_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "ETTh1.csv"
    dataset.write_text(
        "date,OT,HUFL\n"
        "2020-01-01 00:00:00,1.0,2.0\n"
        "2020-01-01 01:00:00,1.1,2.1\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "--task-id",
            "acceptance_contract",
            "--base-dir",
            str(tmp_path),
            "--build-mode",
            "true",
            "--skip-dataset-diagnosis",
            "true",
            "--skip-baseline-diagnosis",
            "true",
            "--ablation-targets",
            "0",
            "--rounds-per-attempt",
            "2",
            "--max-attempts",
            "1",
            "--dataset",
            str(dataset),
            "--baseline-model",
            "DUET",
            "--prepare-only",
        ]
    )

    assert rc == 0
    task_dir = tmp_path / "task_knowledge" / "acceptance_contract"
    task_config = load_task_config(str(tmp_path), "acceptance_contract")
    report = json.loads((task_dir / "buildmode_research_acceptance_report.json").read_text(encoding="utf-8"))

    assert task_config["build_mode"] is True
    assert task_config["dataset_diagnosis_mode"] == "skip"
    assert task_config["baseline_diagnosis_max_ablation_targets"] == 0
    assert task_config["max_rounds"] == 2
    assert task_config["force_full_rounds"] is True
    assert task_config["baseline_strategy"] == "manual"
    assert task_config["baseline_models"] == ["DUET"]
    assert task_config["acceptance_baseline_model"] == "DUET"
    assert report["status"] == "prepared"


def test_buildmode_acceptance_requires_formal_buildmode_stage(tmp_path: Path) -> None:
    task_id = "acceptance_eval"
    task_dir = tmp_path / "task_knowledge" / task_id
    research_dir = task_dir / "rounds"
    research_dir.mkdir(parents=True)
    (task_dir / "task_config.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "build_mode": True,
                "dataset_diagnosis_mode": "skip",
                "baseline_diagnosis_max_ablation_targets": 0,
                "max_rounds": 2,
                "force_full_rounds": True,
                "baseline_models": ["FiLM"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_skipped_dataset_profile(task_id=task_id, base_dir=str(tmp_path))
    for idx in (1, 2):
        rid = f"Research{idx:03d}"
        round_dir = research_dir / rid
        round_dir.mkdir(parents=True)
        (round_dir / "round.json").write_text(
            json.dumps(
                {
                    "round_id": idx,
                    "research_id": rid,
                    "status": "rejected",
                    "variant_path": f"round_sources/{task_id}/{rid}/round_entry.py",
                    "metrics": {"mse_norm": 0.9 + idx / 100},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (round_dir / "module_manifest.json").write_text(
            json.dumps({"internal_component_map": {"a": "M.a", "b": "M.b", "c": "M.c"}}, ensure_ascii=False),
            encoding="utf-8",
        )
        (round_dir / "module_validity_probe.json").write_text(
            json.dumps(
                {
                    "status": "module_valid",
                    "checks": {"module_registered": True},
                    "component_traces": {
                        "_runtime_probe": {"runtime_detected_components": {"a": "m.a", "b": "m.b", "c": "m.c"}}
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (round_dir / "run_experiment_result.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "parsed_status": "ok",
                    "evaluation_stage": "build_mode",
                    "evaluation_budget": "build_mode",
                    "build_mode": True,
                    "gate": {"evaluation_stage": "build_mode", "evaluation_budget": "build_mode"},
                    "run_result": {
                        "model_config": {
                            "models": [
                                {
                                    "model_name": "ts_benchmark.baselines.time_series_library.FiLM",
                                    "variant_path": f"round_sources/{task_id}/{rid}/round_entry.py",
                                }
                            ]
                        },
                        "artifact_provenance": {
                            "expected": {
                                "variant_path": f"/abs/round_sources/{task_id}/{rid}/round_entry.py",
                                "variant_source_sha256": f"source_sha_{idx}",
                                "model_entry_hash": f"entry_hash_{idx}",
                            },
                            "validation": {
                                "status": "ok",
                                "records": [{"prediction_hashes": [f"prediction_hash_{idx}"]}],
                            },
                        },
                    },
                    "metrics": {"mse_norm": 0.9 + idx / 100},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    _sync_fixture_rounds_to_canonical(tmp_path, task_id)
    result = _evaluate_acceptance(str(tmp_path), task_id, rounds_per_attempt=2, baseline_model="FiLM")

    assert result["status"] == "passed"
    assert result["checks"]["all_evaluation_stage_build_mode"] is True
    assert result["checks"]["all_gate_stage_build_mode"] is True
    assert result["checks"]["all_formal_model_config_variant_bound"] is True
    assert result["checks"]["all_formal_artifact_variant_bound"] is True
    assert result["checks"]["all_formal_artifact_provenance_ok"] is True


def test_buildmode_acceptance_rejects_metrics_without_formal_variant_binding(tmp_path: Path) -> None:
    task_id = "acceptance_missing_binding"
    task_dir = tmp_path / "task_knowledge" / task_id
    research_dir = task_dir / "rounds"
    research_dir.mkdir(parents=True)
    (task_dir / "task_config.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "build_mode": True,
                "dataset_diagnosis_mode": "skip",
                "baseline_diagnosis_max_ablation_targets": 0,
                "max_rounds": 2,
                "force_full_rounds": True,
                "baseline_models": ["FiLM"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_skipped_dataset_profile(task_id=task_id, base_dir=str(tmp_path))
    for idx in (1, 2):
        rid = f"Research{idx:03d}"
        round_dir = research_dir / rid
        round_dir.mkdir(parents=True)
        (round_dir / "round.json").write_text(
            json.dumps(
                {
                    "round_id": idx,
                    "research_id": rid,
                    "status": "rejected",
                    "variant_path": f"round_sources/{task_id}/{rid}/round_entry.py",
                    "metrics": {"mse_norm": 0.9 + idx / 100},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (round_dir / "module_manifest.json").write_text(
            json.dumps({"internal_component_map": {"a": "M.a", "b": "M.b", "c": "M.c"}}, ensure_ascii=False),
            encoding="utf-8",
        )
        (round_dir / "module_validity_probe.json").write_text(
            json.dumps({"status": "module_valid", "checks": {"module_registered": True}}, ensure_ascii=False),
            encoding="utf-8",
        )
        (round_dir / "run_experiment_result.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "parsed_status": "ok",
                    "evaluation_stage": "build_mode",
                    "evaluation_budget": "build_mode",
                    "build_mode": True,
                    "gate": {"evaluation_stage": "build_mode", "evaluation_budget": "build_mode"},
                    "run_result": {
                        "model_config": {
                            "models": [{"model_name": "ts_benchmark.baselines.time_series_library.FiLM"}]
                        },
                        "artifact_provenance": {"expected": {}, "validation": {"status": "ok", "records": []}},
                    },
                    "metrics": {"mse_norm": 0.9 + idx / 100},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    _sync_fixture_rounds_to_canonical(tmp_path, task_id)
    result = _evaluate_acceptance(str(tmp_path), task_id, rounds_per_attempt=2, baseline_model="FiLM")

    assert result["status"] == "failed"
    assert result["checks"]["all_entered_experiment"] is True
    assert result["checks"]["all_mse_norm_present"] is True
    assert result["checks"]["all_formal_model_config_variant_bound"] is False
    assert result["checks"]["all_formal_artifact_variant_bound"] is False


def test_buildmode_acceptance_treats_module_validity_mismatch_as_non_blocking_audit(tmp_path: Path) -> None:
    task_id = "acceptance_manifest_audit_only"
    task_dir = tmp_path / "task_knowledge" / task_id
    round_dir = task_dir / "rounds" / "Research001"
    round_dir.mkdir(parents=True)
    variant_path = f"round_sources/{task_id}/Research001/round_entry.py"
    (task_dir / "task_config.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "build_mode": True,
                "dataset_diagnosis_mode": "skip",
                "baseline_diagnosis_max_ablation_targets": 0,
                "max_rounds": 1,
                "force_full_rounds": True,
                "baseline_models": ["TimeKAN"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_skipped_dataset_profile(task_id=task_id, base_dir=str(tmp_path))
    (round_dir / "round.json").write_text(
        json.dumps(
            {
                "round_id": 1,
                "research_id": "Research001",
                "status": "rejected",
                "variant_path": variant_path,
                "metrics": {"mse_norm": 0.95},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (round_dir / "module_manifest.json").write_text(
        json.dumps({"internal_component_map": {"trend": "M.trend"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (round_dir / "module_validity_probe.json").write_text(
        json.dumps(
            {
                "status": "manifest_code_mismatch",
                "failure_kind": "manifest_code_mismatch",
                "checks": {"manifest_code_alignment": False},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (round_dir / "run_experiment_result.json").write_text(
        json.dumps(
            {
                "success": True,
                "parsed_status": "ok",
                "evaluation_stage": "build_mode",
                "evaluation_budget": "build_mode",
                "build_mode": True,
                "gate": {"evaluation_stage": "build_mode", "evaluation_budget": "build_mode"},
                "run_result": {
                    "model_config": {"models": [{"model_name": "TimeKAN", "variant_path": variant_path}]},
                    "artifact_provenance": {
                        "expected": {
                            "variant_path": "/abs/" + variant_path,
                            "variant_source_sha256": "source_sha",
                            "model_entry_hash": "entry_hash",
                        },
                        "validation": {"status": "ok", "records": [{"prediction_hashes": ["pred_hash"]}]},
                    },
                },
                "metrics": {"mse_norm": 0.95},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _sync_fixture_rounds_to_canonical(tmp_path, task_id)
    result = _evaluate_acceptance(str(tmp_path), task_id, rounds_per_attempt=1, baseline_model="TimeKAN")

    assert result["status"] == "passed"
    assert result["checks"]["all_module_valid"] is False
    assert result["checks"]["no_manifest_code_mismatch"] is False
    assert "all_module_valid" not in result["blocking_checks"]
    assert "no_manifest_code_mismatch" not in result["blocking_checks"]
