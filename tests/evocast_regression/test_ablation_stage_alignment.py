from __future__ import annotations

import json

from evocast.state.runtime.store import load_runtime_state
from evocast.state.domain_store import load_task_config, save_task_config
from evocast.scripts.wizard import run_baseline_diagnosis_before_agent
from evocast.tools.tfb_ablation import run_ablation
from evocast.tools.tfb_experiment import _runtime_model_name_for_variant
from tests.evocast_regression.support import create_dtaf_task


def test_run_ablation_rejects_stage_mismatch_before_execution(monkeypatch) -> None:
    session = create_dtaf_task("ablation_stage_mismatch", max_rounds=1)
    task_config = load_task_config(session.base_dir, session.task_id)
    task_config["build_mode"] = True
    save_task_config(session.base_dir, session.task_id, task_config)

    monkeypatch.setattr(
        "evocast.tools.tfb_ablation.run_ablation_round",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("run_ablation_round should not execute")),
    )

    result = run_ablation(
        session,
        {
            "target_id": "q001_attention",
            "target_kind": "mechanism_ablation",
            "target": {
                "target_id": "q001_attention",
                "target_kind": "mechanism_ablation",
                "mechanism_id": "m_attention",
                "mechanism_name": "temporal attention",
                "diagnosis_question": "Does temporal attention matter?",
                "causal_variable": "attention weights route temporal context",
                "evidence_files": ["ts_benchmark/baselines/dtaf/dtaf.py"],
                "evidence_anchors": ["temporal_attention"],
                "exact_edit_intent": "replace temporal attention with a shape-preserving non-attention path",
                "preserve_contract": {
                    "input": "preserve model input protocol",
                    "output": "preserve forecast output shape",
                    "task": "do not change dataset, target columns, horizon, metric, optimizer, scheduler, or training policy",
                },
                "expected_behavior_delta": "same-input forecast should change if this mechanism is active",
            },
            "model_key": "DTAF",
            "baseline_model": "DTAF",
            "baseline_metrics": {"mse_norm": 1.0},
            "budget": "unified",
            "smoke": False,
            "objective_metric": "mse_norm",
            "candidate_stage": "experiment",
        },
    )

    assert result["status"] == "failed"
    assert result["record"]["failure_type"] == "stage_mismatch"


def test_wizard_baseline_diagnosis_aligns_build_mode_to_smoke(monkeypatch) -> None:
    session = create_dtaf_task("baseline_diagnosis_smoke_align", max_rounds=1)
    task_config = load_task_config(session.base_dir, session.task_id)
    task_config["build_mode"] = True
    save_task_config(session.base_dir, session.task_id, task_config)
    baseline = load_runtime_state(session.base_dir, session.task_id).baseline.to_dict()
    captured_budgets: list[str] = []

    monkeypatch.setattr(
        "evocast.scripts.wizard.analyze_model_structure",
        lambda *_args, **_kwargs: {"safe_fit_points": [], "source_files": [{"path": "ts_benchmark/baselines/dtaf/dtaf.py"}]},
    )

    def _fake_generate_mechanism_ablation_plan(*_args, **kwargs):
        assert kwargs["evaluation_stage"] == "smoke"
        target = {
            "target_id": "q001_attention",
            "target_kind": "mechanism_ablation",
            "mechanism_id": "m_attention",
            "mechanism_name": "temporal attention",
            "diagnosis_question": "Does temporal attention matter?",
            "causal_variable": "attention weights route temporal context",
            "evidence_files": ["ts_benchmark/baselines/dtaf/dtaf.py"],
            "evidence_anchors": ["temporal_attention"],
            "exact_edit_intent": "replace temporal attention with a shape-preserving non-attention path",
            "preserve_contract": {
                "input": "preserve model input protocol",
                "output": "preserve forecast output shape",
                "task": "do not change dataset, target columns, horizon, metric, optimizer, scheduler, or training policy",
            },
            "expected_behavior_delta": "same-input forecast should change if this mechanism is active",
            "evaluation_stage": "smoke",
        }
        return {
            "plan": {
                "schema_version": "mechanism_ablation_plan_v1",
                "base_model": "DTAF",
                "objective_metric": "mse_norm",
                "evaluation_stage": "smoke",
                "targets": [target],
            },
            "review": {
                "schema_version": "mechanism_ablation_plan_review_v1",
                "status": "approved",
                "reviewed_targets": [target],
                "rejected_targets": [],
                "errors": [],
                "corrections": [],
            },
        }

    monkeypatch.setattr(
        "evocast.scripts.wizard.generate_mechanism_ablation_plan",
        _fake_generate_mechanism_ablation_plan,
    )
    monkeypatch.setattr("evocast.scripts.wizard.persist_ablation_plan", lambda *_args, **_kwargs: {})

    def _fake_run_ablation_round(_session, _target, **kwargs):
        captured_budgets.append(str(kwargs.get("budget")))
        return {
            "status": "ok",
            "record": {
                "target_id": _target["target_id"],
                "target_name": _target["target_id"],
                "mechanism_id": _target["mechanism_id"],
                "mechanism_name": _target["mechanism_name"],
                "exact_edit_intent": _target["exact_edit_intent"],
                "status": "success",
                "evaluation_stage": "smoke",
                "metrics": {"mse_norm": 1.2},
                "metric_delta": {"mse_norm": {"candidate": 1.2, "baseline": 1.0, "delta": 0.2, "relative_delta": 0.2}},
                "usable_evidence_status": "usable_evidence",
                "variant_path": "evocast/task_knowledge/fake/workspace/round_entry.py",
                "artifact_paths": [],
                "interpretation": {"recommended_action": "record_as_context"},
                "created_at": "2026-07-04T00:00:00",
            },
        }

    monkeypatch.setattr("evocast.scripts.wizard.run_ablation_round", _fake_run_ablation_round)
    monkeypatch.setattr(
        "evocast.scripts.wizard.read_ablation_results",
        lambda *_args, **_kwargs: {"count": 2, "ablations": [{"status": "success"}, {"status": "success"}]},
    )

    summary = run_baseline_diagnosis_before_agent(
        task_id=session.task_id,
        baseline=baseline,
        objective_metric="mse_norm",
        budget="unified",
        seed=2021,
        max_targets=2,
    )

    assert summary["status"] in {"completed", "partial"}
    assert captured_budgets == ["unified"]


def test_variant_runtime_model_name_uses_baseline_import_path() -> None:
    model_name = _runtime_model_name_for_variant(
        registry_entry={"model_name": "DTAF"},
        baseline={"import_path": "ts_benchmark.baselines.dtaf.DTAF"},
        fallback_model_name="invalid.workspace.display",
    )

    assert model_name == "ts_benchmark.baselines.dtaf.DTAF"
