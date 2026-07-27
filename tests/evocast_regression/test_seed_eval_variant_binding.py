from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from evocast.harness.session import AgentSession
from evocast.research.baseline_reference import write_initial_baseline_reference
from evocast.variant.import_isolation import purge_workspace_modules
from evocast.tools.tfb_seed_eval import _build_seed_eval_model_entry, run_seed_eval
from ts_benchmark.models.model_loader import get_models


def test_seed_eval_uses_baseline_runtime_import_for_workspace_variant(tmp_path: Path) -> None:
    base_dir = tmp_path / "repo"
    task_id = "seed_eval_variant_binding"
    knowledge_dir = base_dir / "task_knowledge" / task_id
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    (knowledge_dir / "task_config.json").write_text(
        json.dumps({"objective_metric": "mse_norm", "build_mode": True}, ensure_ascii=False),
        encoding="utf-8",
    )
    (knowledge_dir / "compiled_config.json").write_text(
        json.dumps({"model_config": {"recommend_model_hyper_params": {}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (knowledge_dir / "runtime_state.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "objective_metric": "mse_norm",
                "baseline": {
                    "candidate_id": "baseline_001_DUET",
                    "display_name": "DUET",
                    "import_path": "ts_benchmark.baselines.duet.models.duet_model.DUETModel",
                    "adapter": None,
                    "model_config": {
                        "model_name": "ts_benchmark.baselines.duet.models.duet_model.DUETModel",
                        "model_hyper_params": {"seq_len": 96, "pred_len": 96},
                    },
                    "metrics": {"mse_norm": 1.0},
                    "source": "baseline_search",
                },
                "current_best": {
                    "candidate_id": "baseline_001_DUET",
                    "display_name": "DUET",
                    "import_path": "ts_benchmark.baselines.duet.models.duet_model.DUETModel",
                    "adapter": None,
                    "model_config": {
                        "model_name": "ts_benchmark.baselines.duet.models.duet_model.DUETModel",
                        "model_hyper_params": {"seq_len": 96, "pred_len": 96},
                    },
                    "metrics": {"mse_norm": 1.0},
                    "source": "baseline_search",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    session = AgentSession(task_id=task_id, base_dir=str(base_dir), client=SimpleNamespace(api_available=False))
    session.ensure_dirs()
    variant_path = str(knowledge_dir / "rounds" / "Ablation001" / "workspace" / "round_entry.py")

    entry, provenance = _build_seed_eval_model_entry(
        session,
        {
            "variant_path": variant_path,
            "model_key": "DUET",
        },
    )

    assert entry["model_name"] == "ts_benchmark.baselines.duet.models.duet_model.DUETModel"
    assert entry["variant_path"] == variant_path
    assert not entry["model_name"].startswith("global.")
    assert provenance["display_model_name"] == variant_path


def test_model_loader_variant_path_takes_precedence_over_model_name(tmp_path: Path) -> None:
    workspace = tmp_path / "task_knowledge" / "loader_variant" / "rounds" / "Research001" / "workspace"
    workspace.mkdir(parents=True)
    variant_path = workspace / "round_entry.py"
    variant_path.write_text(
        "\n".join(
            [
                "class Model:",
                "    def __init__(self, **kwargs):",
                "        self.loaded_from_workspace = True",
                "        self.kwargs = kwargs",
            ]
        ),
        encoding="utf-8",
    )

    factories = get_models(
        {
            "models": [
                {
                    "model_name": "ts_benchmark.baselines.time_series_library.ETSformer",
                    "variant_path": str(variant_path),
                    "model_hyper_params": {"sentinel": 17},
                }
            ],
            "recommend_model_hyper_params": {},
        }
    )

    model = factories[0]()
    assert model.loaded_from_workspace is True
    assert model.kwargs["sentinel"] == 17


def test_import_isolation_purges_workspace_ts_benchmark_modules(tmp_path: Path) -> None:
    workspace = tmp_path / "task_knowledge" / "loader_variant" / "rounds" / "Research001" / "workspace"
    pkg = workspace / "ts_benchmark" / "baselines" / "fixture"
    pkg.mkdir(parents=True)
    for parent in [workspace / "ts_benchmark", workspace / "ts_benchmark" / "baselines", pkg]:
        (parent / "__init__.py").write_text("", encoding="utf-8")
    module_path = pkg / "model.py"
    module_path.write_text("SENTINEL = 'workspace'\n", encoding="utf-8")

    purge_workspace_modules(purge_ts_benchmark=True)
    sys.path.insert(0, str(workspace))
    try:
        import ts_benchmark.baselines.fixture.model as model  # type: ignore

        assert model.SENTINEL == "workspace"
        assert str(workspace) in str(getattr(model, "__file__", ""))
    finally:
        purge_workspace_modules(purge_ts_benchmark=True)

    assert "ts_benchmark.baselines.fixture.model" not in sys.modules
    assert str(workspace) not in sys.path


def test_initial_baseline_reference_uses_baseline_search_model_config(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / "repo"
    task_id = "build_mode_baseline_reference"
    knowledge_dir = base_dir / "task_knowledge" / task_id
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    (knowledge_dir / "task_config.json").write_text(
        json.dumps({"objective_metric": "mse_norm", "build_mode": True}, ensure_ascii=False),
        encoding="utf-8",
    )
    captured: dict = {}

    def _fake_run_seed_evaluation(**kwargs):
        captured.update(kwargs)
        return {
            "mean": 1.0,
            "std": 0.0,
            "seed_list": [2021, 2022, 2023],
            "mean_metrics": {"mse_norm": 1.0},
            "per_seed": [
                {"seed": 2021, "metrics": {"mse_norm": 1.0}, "objective_value": 1.0, "success": True},
                {"seed": 2022, "metrics": {"mse_norm": 1.1}, "objective_value": 1.1, "success": True},
                {"seed": 2023, "metrics": {"mse_norm": 0.9}, "objective_value": 0.9, "success": True},
            ],
            "valid_metric_seeds": 3,
            "successful_seeds": 3,
            "result_path": str(knowledge_dir / "seed_ref.json"),
        }

    monkeypatch.setattr("evocast.research.baseline_reference.run_seed_evaluation", _fake_run_seed_evaluation)

    result = write_initial_baseline_reference(
        task_id=task_id,
        base_dir=str(base_dir),
        config_path=str(knowledge_dir / "compiled_config.json"),
        baseline_record={
            "candidate_id": "baseline_001_DUET",
            "node_id": "baseline_001_DUET",
            "display_name": "DUET",
            "seed": 2022,
            "model_config": {
                "model_name": "ts_benchmark.baselines.duet.models.duet_model.DUETModel",
                "model_hyper_params": {
                    "batch_size": 1,
                    "num_epochs": 1,
                    "patience": 1,
                    "max_train_batches": 1,
                    "max_val_batches": 1,
                    "seq_len": 96,
                    "pred_len": 96,
                },
            },
            "metrics": {"mse_norm": 1.0},
        },
        objective_metric="mse_norm",
        num_seeds=3,
        base_seed=2021,
    )

    hparams = captured["model_config"]["model_hyper_params"]
    assert result["schema_version"] == "baseline_reference_v1"
    assert result["candidate_kind"] == "baseline"
    assert result["variant_path"] is None
    assert result["source_clean"] is True
    assert hparams["batch_size"] == 1
    assert hparams["num_epochs"] == 1
    assert hparams["patience"] == 1
    assert hparams["max_train_batches"] == 1
    assert hparams["max_val_batches"] == 1
    assert captured["seed_list"] == [2021, 2023]
    assert captured["seed_universe"] == [2021, 2022, 2023]
    assert captured["precomputed_seed_values"][0]["seed"] == 2022


def test_source_checkout_seed_eval_uses_current_best_reference(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / "repo"
    task_id = "source_checkout_seed_eval_reference"
    knowledge_dir = base_dir / "task_knowledge" / task_id
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    (knowledge_dir / "task_config.json").write_text(
        json.dumps({"objective_metric": "mse_norm", "build_mode": True}, ensure_ascii=False),
        encoding="utf-8",
    )
    (knowledge_dir / "compiled_config.json").write_text(
        json.dumps({"model_config": {"recommend_model_hyper_params": {}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    baseline = {
        "candidate_id": "baseline_001_PAttn",
        "display_name": "PAttn",
        "import_path": "ts_benchmark.baselines.time_series_library.PAttn",
        "adapter": "transformer_adapter",
        "model_config": {
            "model_name": "ts_benchmark.baselines.time_series_library.PAttn",
            "adapter": "transformer_adapter",
            "model_hyper_params": {"seq_len": 96, "pred_len": 96, "label_len": 48},
        },
        "metrics": {"mse_norm": 1.0967172877195497},
        "objective_metric": "mse_norm",
        "source": "baseline_search",
    }
    (knowledge_dir / "runtime_state.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "objective_metric": "mse_norm",
                "baseline": baseline,
                "current_best": baseline,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (knowledge_dir / "baseline_reference.json").write_text(
        json.dumps(
            {
                "schema_version": "baseline_reference_v1",
                "task_id": task_id,
                "candidate_id": "baseline_001_PAttn",
                "candidate_kind": "baseline",
                "variant_path": None,
                "source_clean": True,
                "generated_before_first_variant": True,
                "model_config_hash": "baseline-hash",
                "path": str(knowledge_dir / "baseline_reference.json"),
                "result_path": str(knowledge_dir / "baseline_seed_eval.json"),
                "metric_stats": {
                    "mse_norm": {
                        "mean": 1.145480185186837,
                        "std": 0.04621806870589411,
                        "seed_count": 3,
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    source_checkout = tmp_path / "candidate_checkout"
    source_checkout.mkdir()
    captured: dict = {}

    def _fake_run_seed_evaluation(**kwargs):
        captured.update(kwargs)
        return {
            "node_id": kwargs["node_id"],
            "mean": 1.1149437669403064,
            "std": 0.049763308968836115,
            "mean_metrics": {"mse_norm": 1.1149437669403064},
            "reference_mean": kwargs["reference_mean"],
            "reference_std": kwargs["reference_std"],
            "reference_seed_count": kwargs["reference_seed_count"],
            "current_best_mean": kwargs["reference_mean"],
            "current_best_std": kwargs["reference_std"],
            "current_best_seed_count": kwargs["reference_seed_count"],
            "successful_seeds": 3,
            "valid_metric_seeds": 3,
            "significance_decision": {"decision": "accept"},
            "promoted_to_current_best": False,
            "result_path": str(knowledge_dir / "candidate_seed_eval.json"),
        }

    monkeypatch.setattr("evocast.tools.tfb_seed_eval.run_seed_evaluation", _fake_run_seed_evaluation)
    session = AgentSession(task_id=task_id, base_dir=str(base_dir), client=SimpleNamespace(api_available=False))

    result = run_seed_eval(
        session,
        {
            "model_name": "ts_benchmark.baselines.time_series_library.PAttn",
            "source_checkout": str(source_checkout),
            "objective_metric": "mse_norm",
            "candidate_id": "Research002",
            "promote_on_accept": False,
        },
    )

    assert captured["reference_mean"] == 1.145480185186837
    assert captured["reference_std"] == 0.04621806870589411
    assert captured["reference_seed_count"] == 3
    assert captured["source_checkout"] == str(source_checkout)
    assert result["comparison_reference"]["kind"] == "current_best"
    assert result["comparison_reference"]["reference_mean"] == 1.145480185186837
    assert result["comparison_reference"]["seed_count"] == 3
    assert result["significance_decision"]["decision"] == "accept"
