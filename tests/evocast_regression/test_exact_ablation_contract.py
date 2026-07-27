from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from evocast.build.contract_compiler import build_ablation_contract
from evocast.build.result import BuildDecision, BuildOutcome
from evocast.build.source_snapshot import source_manifest
from evocast.harness.ablation_round import _outcome_record
from evocast.harness.session import AgentSession
from evocast.research.ablation.exact_contract import audit_exact_patch_hit, compile_exact_ablation_target
from evocast.state.runtime.store import sync_best_baseline


def _target() -> dict:
    return {
        "target_id": "q001",
        "ablation_id": "Ablation001",
        "target_kind": "mechanism_ablation",
        "mechanism_id": "m001",
        "mechanism_name": "demo branch",
        "diagnosis_question": "Does the demo branch help?",
        "causal_variable": "demo branch contribution",
        "exact_edit_intent": "replace branch contribution with zeros",
        "edit_spec": {
            "target_file": "ts_benchmark/baselines/demo/model.py",
            "anchor_text": "y = self.branch(x)",
            "replacement_intent": "remove branch contribution",
            "replacement_pseudocode": "y = torch.zeros_like(self.branch(x))",
            "shape_invariant_argument": "zeros_like preserves y shape",
        },
        "preserve_contract": {"input": "same", "output": "same", "task": "do not change training policy"},
        "expected_behavior_delta": "same-input forecast should change",
    }


def _write_contract(path: Path, exact_target: dict, evaluation_stage: str = "experiment") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"metric_protocol": {"exact_ablation_target": exact_target, "evaluation_stage": evaluation_stage}},
            indent=2,
        ),
        encoding="utf-8",
    )


def test_compile_exact_ablation_target_requires_existing_anchor(tmp_path: Path) -> None:
    source = tmp_path / "ts_benchmark" / "baselines" / "demo" / "model.py"
    source.parent.mkdir(parents=True)
    source.write_text("def forward(x):\n    y = self.branch(x)\n    return y\n", encoding="utf-8")

    exact = compile_exact_ablation_target(_target(), repo_dir=tmp_path)

    assert exact["target_file"] == "ts_benchmark/baselines/demo/model.py"
    assert exact["anchor_text"] == "y = self.branch(x)"
    assert exact["replacement_pseudocode"] == "y = torch.zeros_like(self.branch(x))"
    assert exact["replacement_is_hint"] is True
    assert exact["anchor_occurrences"] == 1


def test_compile_exact_ablation_target_accepts_intent_only_anchor(tmp_path: Path) -> None:
    source = tmp_path / "ts_benchmark" / "baselines" / "demo" / "model.py"
    source.parent.mkdir(parents=True)
    source.write_text("def forward(x):\n    y = self.branch(x)\n    return y\n", encoding="utf-8")
    target = _target()
    target["edit_spec"].pop("replacement_pseudocode")

    exact = compile_exact_ablation_target(target, repo_dir=tmp_path)

    assert exact["target_file"] == "ts_benchmark/baselines/demo/model.py"
    assert exact["replacement_pseudocode"] == ""
    assert exact["ablation_intent"] == "remove branch contribution"


def test_exact_patch_audit_accepts_anchor_area_change_without_literal_replacement(tmp_path: Path) -> None:
    exact = {
        "target_file": "ts_benchmark/baselines/demo/model.py",
        "anchor_text": "if enabled:\n    y = self.branch(x)\n    return y",
        "replacement_pseudocode": "",
    }
    patch = tmp_path / "agent.patch"
    patch.write_text(
        "\n".join(
            [
                "diff --git a/ts_benchmark/baselines/demo/model.py b/ts_benchmark/baselines/demo/model.py",
                "--- a/ts_benchmark/baselines/demo/model.py",
                "+++ b/ts_benchmark/baselines/demo/model.py",
                "@@ -1,4 +1,4 @@",
                " if enabled:",
                "-    y = self.branch(x)",
                "+    y = self.other_branch(x)",
                "     return y",
            ]
        ),
        encoding="utf-8",
    )

    audit = audit_exact_patch_hit(patch_path=patch, exact_target=exact)
    assert audit["passed"] is True
    assert audit["anchor_area_hit"] is True
    assert audit["added_replacement_hit"] is None


def test_no_direct_probe_for_pattn_imported_modules(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "ts_benchmark" / "baselines" / "time_series_library" / "models" / "PAttn.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import torch\n"
        "from ..layers.Transformer_EncDec import Encoder, EncoderLayer\n"
        "from ..layers.SelfAttention_Family import AttentionLayer, FullAttention\n"
        "class Model(torch.nn.Module):\n"
        "    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):\n"
        "        return x_enc\n",
        encoding="utf-8",
    )
    manifest = source_manifest(repo)
    runtime = tmp_path / "runtime"
    task_id = "pattn_no_probe"
    knowledge = runtime / "task_knowledge" / task_id
    knowledge.mkdir(parents=True)
    (knowledge / "compiled_config.json").write_text(
        json.dumps(
            {
                "data_config": {
                    "data_set_name": "large_forecast",
                    "dataset_path": "missing.csv",
                    "data_name_list": ["missing.csv"],
                    "time_col": "date",
                    "target_columns": ["OT"],
                },
                "model_config": {"recommend_model_hyper_params": {"input_chunk_length": 96, "output_chunk_length": 24, "norm": True}},
                "evaluation_config": {"strategy_args": {"horizon": 24}},
            }
        ),
        encoding="utf-8",
    )
    baseline = {
        "candidate_id": "baseline_pattn",
        "display_name": "PAttn",
        "model_name": "PAttn",
        "import_path": "ts_benchmark.baselines.time_series_library.PAttn",
        "adapter": "transformer_adapter",
        "source_binding": {
            "entry_file": "ts_benchmark/baselines/time_series_library/models/PAttn.py",
            "source_files": ["ts_benchmark/baselines/time_series_library/models/PAttn.py"],
        },
        "metrics": {"mse_norm": 1.0},
        "objective_metric": "mse_norm",
        "model_config": {
            "model_name": "ts_benchmark.baselines.time_series_library.PAttn",
            "adapter": "transformer_adapter",
            "model_hyper_params": {"lr": 0.003, "patch_len": 12, "batch_size": 7},
        },
        "source_ref": {
            "kind": "source_snapshot",
            "candidate_snapshot_id": manifest["snapshot_id"],
            "base_snapshot_id": manifest["snapshot_id"],
            "source_checkout": str(repo),
            "source_manifest_hash": manifest["manifest_hash"],
            "source_binding": {
                "entry_file": "ts_benchmark/baselines/time_series_library/models/PAttn.py",
                "source_files": ["ts_benchmark/baselines/time_series_library/models/PAttn.py"],
            },
        },
    }
    sync_best_baseline(str(runtime), task_id, baseline)
    target = _target()
    target["edit_spec"]["target_file"] = "ts_benchmark/baselines/time_series_library/models/PAttn.py"
    target["edit_spec"]["anchor_text"] = "return x_enc"

    contract = build_ablation_contract(
        base_dir=str(runtime),
        task_id=task_id,
        target=target,
        baseline=baseline,
        objective_metric="mse_norm",
        repo_dir=repo,
    )

    commands = json.dumps(contract.to_dict())
    assert "ABLATION_PROTOCOL_PROBE" not in commands
    assert "FullAttention.forward" not in commands
    hparams = contract.metric_protocol["model_config"]["model_hyper_params"]
    for key in ["d_model", "factor", "dropout", "n_heads", "d_ff", "activation", "seq_len", "pred_len", "horizon", "enc_in", "dec_in", "c_out"]:
        assert key in hparams
    assert hparams["lr"] == 0.0001
    assert hparams["patch_len"] == 12
    assert hparams["batch_size"] == 32
    assert hparams["num_epochs"] == 10


def test_gate_rejected_metric_with_exact_patch_hit_is_usable_evidence(tmp_path: Path) -> None:
    session = AgentSession(task_id="exact_record", base_dir=str(tmp_path), client=SimpleNamespace(api_available=False))
    session.ensure_dirs()
    exact = {
        "target_file": "ts_benchmark/baselines/demo/model.py",
        "anchor_text": "y = self.branch(x)",
        "replacement_pseudocode": "y = torch.zeros_like(self.branch(x))",
    }
    round_dir = tmp_path / "task_knowledge" / "exact_record" / "rounds" / "Research001"
    metric_dir = round_dir / "metric"
    metric_dir.mkdir(parents=True)
    (metric_dir / "metric_result.json").write_text(
        json.dumps({"result": {"success": True, "metrics": {"mse_norm": 1.2}, "variant_path": "unused.py"}}),
        encoding="utf-8",
    )
    patch = round_dir / "build_attempt_01" / "agent.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text(
        "diff --git a/ts_benchmark/baselines/demo/model.py b/ts_benchmark/baselines/demo/model.py\n"
        "--- a/ts_benchmark/baselines/demo/model.py\n"
        "+++ b/ts_benchmark/baselines/demo/model.py\n"
        "@@ -1 +1 @@\n"
        "-y = self.branch(x)\n"
        "+y = torch.zeros_like(self.branch(x))\n",
        encoding="utf-8",
    )
    contract_path = round_dir / "build_contract.json"
    _write_contract(contract_path, exact)
    outcome = BuildOutcome(
        status=BuildDecision.TERMINAL_REJECTED,
        research_id="Research001",
        round_id=1,
        round_dir=round_dir,
        patch_path=patch,
        summary="Metric completed and gate rejected the candidate.",
    )

    record = _outcome_record(
        session=session,
        target=_target(),
        outcome=outcome,
        contract_path=contract_path,
        variant_path="",
        objective_metric="mse_norm",
    )

    assert record["status"] == "success"
    assert record["usable_evidence_status"] == "usable_evidence"
    assert record["execution_status"] == "metric_completed"
    assert record["gate_decision"] == "TERMINAL_REJECTED"
    assert record["scientific_decision"] == "scientific_rejected"
    assert record["evaluation_stage"] == "experiment"


def test_metric_with_exact_patch_miss_is_failed_evidence(tmp_path: Path) -> None:
    session = AgentSession(task_id="exact_record_miss", base_dir=str(tmp_path), client=SimpleNamespace(api_available=False))
    session.ensure_dirs()
    exact = {
        "target_file": "ts_benchmark/baselines/demo/model.py",
        "anchor_text": "y = self.branch(x)",
        "replacement_pseudocode": "y = torch.zeros_like(self.branch(x))",
    }
    round_dir = tmp_path / "task_knowledge" / "exact_record_miss" / "rounds" / "Research001"
    metric_dir = round_dir / "metric"
    metric_dir.mkdir(parents=True)
    (metric_dir / "metric_result.json").write_text(
        json.dumps({"result": {"success": True, "metrics": {"mse_norm": 1.2}}}),
        encoding="utf-8",
    )
    patch = round_dir / "build_attempt_01" / "agent.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text(
        "diff --git a/ts_benchmark/baselines/demo/model.py b/ts_benchmark/baselines/demo/model.py\n"
        "--- a/ts_benchmark/baselines/demo/model.py\n"
        "+++ b/ts_benchmark/baselines/demo/model.py\n"
        "@@ -10 +10 @@\n"
        "-z = self.unrelated(x)\n"
        "+z = self.other_unrelated(x)\n",
        encoding="utf-8",
    )
    contract_path = round_dir / "build_contract.json"
    _write_contract(contract_path, exact)
    outcome = BuildOutcome(
        status=BuildDecision.ACCEPTED,
        research_id="Research001",
        round_id=1,
        round_dir=round_dir,
        patch_path=patch,
        summary="accepted",
    )

    record = _outcome_record(
        session=session,
        target=_target(),
        outcome=outcome,
        contract_path=contract_path,
        variant_path="",
        objective_metric="mse_norm",
    )

    assert record["status"] == "failed_ablation_round"
    assert record["usable_evidence_status"] == "failed_evidence"
    assert record["failure_type"] == "anchor_area_patch_miss"
