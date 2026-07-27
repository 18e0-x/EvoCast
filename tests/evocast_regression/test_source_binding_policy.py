from __future__ import annotations

import json
from pathlib import Path

from evocast.build.contract_compiler import build_research_contract
from evocast.build.source_snapshot import source_manifest
from evocast.domain.baseline_identity import source_binding_from_entry_file
from evocast.state.runtime.store import sync_best_baseline


def test_tsl_patchtst_binding_uses_entry_plus_direct_local_imports() -> None:
    binding = source_binding_from_entry_file(
        "ts_benchmark/baselines/time_series_library/models/PatchTST.py"
    )

    assert binding["binding_policy"] == "tsl_entry_plus_direct_local_imports_editable"
    assert binding["entry_file"] == "ts_benchmark/baselines/time_series_library/models/PatchTST.py"
    assert binding["source_files"] == [
        "ts_benchmark/baselines/time_series_library/models/PatchTST.py",
        "ts_benchmark/baselines/time_series_library/layers/Embed.py",
        "ts_benchmark/baselines/time_series_library/layers/SelfAttention_Family.py",
        "ts_benchmark/baselines/time_series_library/layers/Transformer_EncDec.py",
    ]
    assert "ts_benchmark/baselines/time_series_library/models/Autoformer.py" not in binding["source_files"]
    assert "ts_benchmark/baselines/time_series_library/models/DLinear.py" not in binding["source_files"]
    assert "ts_benchmark/baselines/time_series_library/models/MambaSimple.py" not in binding["source_files"]


def test_tsl_crossformer_binding_includes_direct_model_and_layer_imports() -> None:
    binding = source_binding_from_entry_file(
        "ts_benchmark/baselines/time_series_library/models/Crossformer.py"
    )

    assert binding["binding_policy"] == "tsl_entry_plus_direct_local_imports_editable"
    assert binding["source_files"] == [
        "ts_benchmark/baselines/time_series_library/models/Crossformer.py",
        "ts_benchmark/baselines/time_series_library/layers/Crossformer_EncDec.py",
        "ts_benchmark/baselines/time_series_library/layers/Embed.py",
        "ts_benchmark/baselines/time_series_library/layers/SelfAttention_Family.py",
        "ts_benchmark/baselines/time_series_library/models/PatchTST.py",
    ]
    assert "ts_benchmark/baselines/time_series_library/models/Transformer.py" not in binding["source_files"]


def test_non_tsl_binding_keeps_package_all_python_files() -> None:
    binding = source_binding_from_entry_file(
        "ts_benchmark/baselines/crosslinear/crosslinear.py"
    )

    assert binding["binding_policy"] == "package_all_python_files_editable"
    assert "ts_benchmark/baselines/crosslinear/crosslinear.py" in binding["source_files"]
    assert "ts_benchmark/baselines/crosslinear/model/crosslinear_model.py" in binding["source_files"]
    assert "ts_benchmark/baselines/crosslinear/__init__.py" in binding["source_files"]


def test_research_contract_uses_tsl_entry_plus_direct_imports_as_allowed_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source_root = repo / "ts_benchmark" / "baselines" / "time_series_library"
    models = source_root / "models"
    layers = source_root / "layers"
    models.mkdir(parents=True)
    layers.mkdir(parents=True)
    (models / "PatchTST.py").write_text(
        "from ..layers.Embed import PatchEmbedding\n"
        "from ..layers.SelfAttention_Family import FullAttention\n"
        "class PatchTST:\n    pass\n",
        encoding="utf-8",
    )
    (models / "Autoformer.py").write_text("class Autoformer:\n    pass\n", encoding="utf-8")
    (models / "MambaSimple.py").write_text("class MambaSimple:\n    pass\n", encoding="utf-8")
    (layers / "Embed.py").write_text("class PatchEmbedding:\n    pass\n", encoding="utf-8")
    (layers / "SelfAttention_Family.py").write_text("class FullAttention:\n    pass\n", encoding="utf-8")
    manifest = source_manifest(repo)
    binding = source_binding_from_entry_file(
        "ts_benchmark/baselines/time_series_library/models/PatchTST.py",
        repo_dir=repo,
    )
    runtime = tmp_path / "runtime"
    task_id = "tsl_contract_policy"
    baseline = {
        "candidate_id": "baseline_001_PatchTST",
        "display_name": "PatchTST",
        "import_path": "ts_benchmark.baselines.time_series_library.PatchTST",
        "source_binding": binding,
        "source_ref": {
            "kind": "source_snapshot",
            "candidate_snapshot_id": manifest["snapshot_id"],
            "base_snapshot_id": manifest["snapshot_id"],
            "source_checkout": str(repo),
            "source_manifest_hash": manifest["manifest_hash"],
            "source_binding": binding,
        },
        "metrics": {"mse_norm": 1.0},
        "objective_metric": "mse_norm",
        "model_config": {
            "model_name": "ts_benchmark.baselines.time_series_library.PatchTST",
            "model_hyper_params": {},
        },
    }
    sync_best_baseline(str(runtime), task_id, baseline)

    contract = build_research_contract(
        base_dir=str(runtime),
        task_id=task_id,
        baseline=baseline,
        objective_metric="mse_norm",
        repo_dir=repo,
        research_id="Research001",
    )

    assert contract.allowed_edit_files == [
        "ts_benchmark/baselines/time_series_library/models/PatchTST.py",
        "ts_benchmark/baselines/time_series_library/layers/Embed.py",
        "ts_benchmark/baselines/time_series_library/layers/SelfAttention_Family.py",
    ]
    assert contract.allowed_new_file_roots == [
        "ts_benchmark/baselines/time_series_library/models",
        "ts_benchmark/baselines/time_series_library/layers",
    ]
    assert "ts_benchmark/baselines/time_series_library/models/Autoformer.py" not in contract.allowed_edit_files
    assert "ts_benchmark/baselines/time_series_library/models/MambaSimple.py" not in contract.allowed_edit_files
    assert all(
        command[-1].count(path) == 1
        for command, path in zip(contract.internal_check_commands, contract.allowed_edit_files)
    )


def test_research_contract_refreshes_old_tsl_directory_wide_binding(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source_root = repo / "ts_benchmark" / "baselines" / "time_series_library"
    models = source_root / "models"
    layers = source_root / "layers"
    models.mkdir(parents=True)
    layers.mkdir(parents=True)
    (models / "PatchTST.py").write_text(
        "from ..layers.Embed import PatchEmbedding\n"
        "class PatchTST:\n    pass\n",
        encoding="utf-8",
    )
    (models / "Autoformer.py").write_text("class Autoformer:\n    pass\n", encoding="utf-8")
    (models / "MambaSimple.py").write_text("class MambaSimple:\n    pass\n", encoding="utf-8")
    (layers / "Embed.py").write_text("class PatchEmbedding:\n    pass\n", encoding="utf-8")
    manifest = source_manifest(repo)
    old_directory_wide_binding = {
        "schema_version": "source_binding_v1",
        "entry_file": "ts_benchmark/baselines/time_series_library/models/PatchTST.py",
        "source_root": "ts_benchmark/baselines/time_series_library/models",
        "source_files": [
            "ts_benchmark/baselines/time_series_library/models/PatchTST.py",
            "ts_benchmark/baselines/time_series_library/models/Autoformer.py",
            "ts_benchmark/baselines/time_series_library/models/MambaSimple.py",
        ],
        "core_files": [
            "ts_benchmark/baselines/time_series_library/models/Autoformer.py",
            "ts_benchmark/baselines/time_series_library/models/MambaSimple.py",
        ],
        "support_files": [],
    }
    runtime = tmp_path / "runtime"
    task_id = "tsl_contract_refreshes_old_binding"
    baseline = {
        "candidate_id": "baseline_001_PatchTST",
        "display_name": "PatchTST",
        "import_path": "ts_benchmark.baselines.time_series_library.PatchTST",
        "source_binding": old_directory_wide_binding,
        "source_ref": {
            "kind": "source_snapshot",
            "candidate_snapshot_id": manifest["snapshot_id"],
            "base_snapshot_id": manifest["snapshot_id"],
            "source_checkout": str(repo),
            "source_manifest_hash": manifest["manifest_hash"],
            "source_binding": old_directory_wide_binding,
        },
        "metrics": {"mse_norm": 1.0},
        "objective_metric": "mse_norm",
        "model_config": {
            "model_name": "ts_benchmark.baselines.time_series_library.PatchTST",
            "model_hyper_params": {},
        },
    }
    sync_best_baseline(str(runtime), task_id, baseline)

    contract = build_research_contract(
        base_dir=str(runtime),
        task_id=task_id,
        baseline=baseline,
        objective_metric="mse_norm",
        repo_dir=repo,
        research_id="Research001",
    )

    assert contract.allowed_edit_files == [
        "ts_benchmark/baselines/time_series_library/models/PatchTST.py",
        "ts_benchmark/baselines/time_series_library/layers/Embed.py",
    ]
    assert contract.allowed_new_file_roots == [
        "ts_benchmark/baselines/time_series_library/models",
        "ts_benchmark/baselines/time_series_library/layers",
    ]
    assert "ts_benchmark/baselines/time_series_library/models/Autoformer.py" not in contract.allowed_edit_files
    assert "ts_benchmark/baselines/time_series_library/models/MambaSimple.py" not in contract.allowed_edit_files


def test_initial_build_message_does_not_embed_full_contract(tmp_path: Path) -> None:
    from tests.evocast_regression.scripted_backend import ScriptedBackend
    from evocast.build.contract import BuildContract
    from evocast.build.orchestrator import ResearchBuildOrchestrator

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "model.py").write_text("VALUE = 'baseline'\n", encoding="utf-8")
    manifest = source_manifest(repo)
    contract = BuildContract(
        research_id="Research001",
        base_snapshot_id=manifest["snapshot_id"],
        semantic_goal="Edit model.",
        hypothesis="Marker edit.",
        target_model="Fixture",
        base_source_ref={
            "source_checkout": str(repo),
            "candidate_snapshot_id": manifest["snapshot_id"],
            "source_binding": {"entry_file": "model.py", "source_files": ["model.py"]},
        },
        allowed_edit_files=["model.py"],
    )
    message = ResearchBuildOrchestrator(
        base_dir=str(tmp_path / "runtime"),
        task_id="short_initial_message",
        repo_dir=repo,
        backend=ScriptedBackend([]),
    )._initial_message(contract)

    assert "<build_contract>" not in message
    assert json.dumps(contract.to_dict(), ensure_ascii=False) not in message
    assert "backend-provided execution_contract" in message


def test_research_contract_a4_expands_to_repo_wide_authority(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "model.py").write_text("VALUE = 'baseline'\n", encoding="utf-8")
    (repo / "pkg" / "helper.py").write_text("HELPER = 1\n", encoding="utf-8")
    (repo / "tests" / "test_fixture.py").write_text("def test_fixture():\n    assert True\n", encoding="utf-8")
    manifest = source_manifest(repo)
    binding = {
        "schema_version": "source_binding_v1",
        "entry_file": "model.py",
        "source_files": ["model.py"],
        "core_files": ["model.py"],
        "support_files": [],
        "verified": True,
    }
    runtime = tmp_path / "runtime"
    task_id = "repo_wide_a4_contract"
    (runtime / "task_knowledge" / task_id).mkdir(parents=True)
    (runtime / "task_knowledge" / task_id / "task_config.json").write_text(
        json.dumps({"agent_ablation": "A4"}, ensure_ascii=False),
        encoding="utf-8",
    )
    baseline = {
        "candidate_id": "baseline_001_Model",
        "display_name": "FixtureModel",
        "import_path": "model",
        "source_binding": binding,
        "source_ref": {
            "kind": "source_snapshot",
            "candidate_snapshot_id": manifest["snapshot_id"],
            "base_snapshot_id": manifest["snapshot_id"],
            "source_checkout": str(repo),
            "source_manifest_hash": manifest["manifest_hash"],
            "source_binding": binding,
        },
        "metrics": {"mse_norm": 1.0},
        "objective_metric": "mse_norm",
        "model_config": {"model_name": "model", "model_hyper_params": {}},
    }
    sync_best_baseline(str(runtime), task_id, baseline)

    contract = build_research_contract(
        base_dir=str(runtime),
        task_id=task_id,
        baseline=baseline,
        objective_metric="mse_norm",
        repo_dir=repo,
        research_id="Research001",
    )

    assert contract.execution_authority == "repo_wide"
    assert contract.allowed_new_file_roots == ["."]
    assert "model.py" in contract.allowed_edit_files
    assert "pkg/helper.py" in contract.allowed_edit_files
    assert "tests/test_fixture.py" in contract.allowed_edit_files
    assert contract.protected_globs == []
    assert contract.forbidden_globs == []
