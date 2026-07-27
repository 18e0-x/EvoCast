from __future__ import annotations

import json
from pathlib import Path

from evocast.build.contract_compiler import build_ablation_contract
from evocast.build.source_snapshot import source_manifest
from evocast.harness.rounds import round_progress, start_round
from evocast.state.runtime.store import sync_best_baseline


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _task(base_dir: Path, task_id: str, *, build_mode: bool) -> None:
    _write_json(
        base_dir / "task_knowledge" / task_id / "task_config.json",
        {
            "task_id": task_id,
            "objective_metric": "mse_norm",
            "max_rounds": 2,
            "build_mode": build_mode,
            "research_intent": "Test whether temporal attention is necessary.",
        },
    )
    _write_json(
        base_dir / "task_knowledge" / task_id / "compiled_config.json",
        {
            "data_config": {"data_set_name": "fixture", "dataset_path": "fixture.csv", "target_columns": ["OT"]},
            "model_config": {"recommend_model_hyper_params": {"input_chunk_length": 96, "output_chunk_length": 24}},
            "evaluation_config": {"strategy_args": {"horizon": 24}},
        },
    )


def _baseline(repo: Path, *, source_files: list[str] | None = None) -> dict:
    manifest = source_manifest(repo)
    source_binding = {
        "entry_file": "baseline.py",
        "source_files": list(source_files or ["baseline.py"]),
        "verified": True,
    }
    return {
        "candidate_id": "baseline_fixture",
        "display_name": "FixtureModel",
        "model_name": "FixtureModel",
        "import_path": "fixture.Model",
        "metrics": {"mse_norm": 1.0},
        "source_binding": source_binding,
        "source_ref": {
            "kind": "source_snapshot",
            "candidate_snapshot_id": manifest["snapshot_id"],
            "base_snapshot_id": manifest["snapshot_id"],
            "source_checkout": str(repo),
            "source_manifest_hash": manifest["manifest_hash"],
            "source_binding": source_binding,
        },
    }


def _target(stage: str) -> dict:
    return {
        "ablation_id": "Ablation001",
        "target_id": "attention",
        "mechanism_name": "temporal attention",
        "diagnosis_question": "Does temporal attention affect the forecast?",
        "causal_variable": "attention weights route temporal context",
        "evidence_files": ["baseline.py"],
        "evidence_anchors": ["Baseline"],
        "evaluation_stage": stage,
        "exact_edit_intent": "replace attention with a shape-preserving path",
        "edit_spec": {
            "target_file": "baseline.py",
            "anchor_text": "class Baseline:\n    pass",
            "replacement_intent": "mark the ablated mechanism",
            "shape_invariant_argument": "the class remains importable",
        },
        "preserve_contract": {"input": "preserve input", "output": "preserve output", "task": "preserve task"},
    }


def test_ablation_contract_is_bound_to_verified_source_and_persisted_intent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "baseline.py").write_text("class Baseline:\n    pass\n", encoding="utf-8")
    base_dir = tmp_path / "runtime"
    task_id = "ablation_boundary"
    _task(base_dir, task_id, build_mode=False)
    baseline = _baseline(repo)
    sync_best_baseline(str(base_dir), task_id, baseline)

    contract = build_ablation_contract(
        base_dir=str(base_dir), task_id=task_id, target=_target("experiment"),
        baseline=baseline, objective_metric="mse_norm", repo_dir=repo, repair_budget=1,
    )

    assert contract.allowed_edit_files == ["baseline.py"]
    assert contract.allowed_new_file_roots == []
    assert contract.research_intent == "Test whether temporal attention is necessary."
    assert contract.metric_protocol["evaluation_stage"] == "experiment"
    assert contract.metric_protocol["model_config"]["model_hyper_params"]["num_epochs"] == 10


def test_ablation_contract_uses_smoke_budget_only_in_build_mode(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "baseline.py").write_text("class Baseline:\n    pass\n", encoding="utf-8")
    base_dir = tmp_path / "runtime"
    task_id = "ablation_build_mode"
    _task(base_dir, task_id, build_mode=True)
    baseline = _baseline(repo)
    sync_best_baseline(str(base_dir), task_id, baseline)

    contract = build_ablation_contract(
        base_dir=str(base_dir), task_id=task_id, target=_target("smoke"),
        baseline=baseline, objective_metric="mse_norm", repo_dir=repo, repair_budget=1,
    )

    hparams = contract.metric_protocol["model_config"]["model_hyper_params"]
    assert contract.metric_protocol["evaluation_stage"] == "smoke"
    assert (hparams["num_epochs"], hparams["batch_size"], hparams["patience"]) == (1, 1, 1)
    assert (hparams["max_train_batches"], hparams["max_val_batches"]) == (1, 1)


def test_ablation_contract_keeps_every_verified_source_binding_file_editable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "baseline.py").write_text("from layers import Layer\nclass Baseline:\n    pass\n", encoding="utf-8")
    (repo / "layers.py").write_text("class Layer:\n    pass\n", encoding="utf-8")
    base_dir = tmp_path / "runtime"
    task_id = "ablation_multiple_source_files"
    _task(base_dir, task_id, build_mode=False)
    baseline = _baseline(repo, source_files=["baseline.py", "layers.py"])
    sync_best_baseline(str(base_dir), task_id, baseline)

    contract = build_ablation_contract(
        base_dir=str(base_dir), task_id=task_id, target=_target("experiment"),
        baseline=baseline, objective_metric="mse_norm", repo_dir=repo, repair_budget=1,
    )

    assert contract.allowed_edit_files == ["baseline.py", "layers.py"]
    assert {"baseline.py", "layers.py"}.issubset(contract.likely_entrypoints)
    assert all(
        any(source_file in " ".join(command) for command in contract.internal_check_commands)
        for source_file in ("baseline.py", "layers.py")
    )


def test_ablation_artifacts_never_consume_formal_research_budget(tmp_path: Path) -> None:
    base_dir = tmp_path / "runtime"
    task_id = "ablation_budget_boundary"
    _task(base_dir, task_id, build_mode=False)
    _write_json(
        base_dir / "task_knowledge" / task_id / "rounds" / "Ablation001" / "ablation_record.json",
        {"ablation_id": "Ablation001", "round_scope": "baseline_diagnosis", "status": "failed"},
    )

    assert round_progress(str(base_dir), task_id)["research_rounds"] == 0
    record = start_round(
        base_dir=str(base_dir), task_id=task_id, fit_point="model.predictor",
        hypothesis="formal research after diagnostic ablation", evidence_source="test fixture",
    )
    assert record["research_id"] == "Research001"
