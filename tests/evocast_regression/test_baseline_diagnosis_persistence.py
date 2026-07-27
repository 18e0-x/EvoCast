from __future__ import annotations

import json
from pathlib import Path

import pytest

from evocast.scripts import wizard
from evocast.domain.knowledge_paths import task_knowledge_dir
from evocast.harness.mechanism_ablation_diagnosis import (
    _normalize_active_path_target,
    _target_from_question,
    generate_mechanism_ablation_plan,
)
from evocast.research.ablation.task import validate_ablation_task
from evocast.state.runtime.store import load_runtime_state
from tests.evocast_regression.support import create_dtaf_task


def _target() -> dict:
    return {
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


def _patch_mechanism_plan(monkeypatch, target: dict) -> None:
    monkeypatch.setattr(
        wizard,
        "generate_mechanism_ablation_plan",
        lambda *_args, **_kwargs: {
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
        },
    )


def test_mechanism_target_filters_polluted_site_package_evidence_files() -> None:
    question = {
        "question_id": "q001_frequency",
        "mechanism_id": "m_frequency",
        "question": "Does frequency interpolation matter?",
        "counterfactual": "Replace interpolation with zeros.",
        "mapping_kind": "operation_edit",
        "edit_spec": {
            "target_file": "ts_benchmark/baselines/timekan/models/timekan_model.py",
            "anchor_text": "out_high = out_high + out_high_res",
            "replacement_intent": "Remove residual frequency addition.",
            "replacement_pseudocode": "out_high = out_high",
            "shape_invariant_argument": "out_high keeps the same shape.",
        },
    }
    mechanism = {
        "mechanism_id": "m_frequency",
        "name": "frequency interpolation",
        "claim": "frequency interpolation",
        "resolved_evidence_files": [
            ".venv/Lib/site-packages/sklearn/preprocessing/_data.py",
            "ts_benchmark/baselines/timekan/models/timekan_model.py",
        ],
    }

    target = _target_from_question(question, mechanism=mechanism, index=1, evaluation_stage="smoke")
    validation = validate_ablation_task(target, evaluation_stage="smoke")

    assert target["evidence_files"] == ["ts_benchmark/baselines/timekan/models/timekan_model.py"]
    assert validation["status"] == "ok"


def test_active_path_target_accepts_legacy_question_id_alias() -> None:
    target = _normalize_active_path_target(
        "Amplifier",
        {
            "question_id": "q001_frequency",
            "mechanism_id": "energy_block",
            "mechanism_name": "Energy block",
            "edit_spec": {
                "target_file": "ts_benchmark/baselines/amplifier/models/amplifier_model.py",
                "anchor_text": "out_fft = out_amplifier_fft - x_inverse_fft",
                "replacement_intent": "Bypass subtraction.",
            },
        },
        1,
        evaluation_stage="smoke",
    )

    assert target["target_id"] == "q001_frequency"
    assert target["target_kind"] == "mechanism_ablation"


def test_mechanism_ablation_plan_does_not_gate_on_observability(monkeypatch) -> None:
    session = create_dtaf_task("baseline_diag_no_observability_gate", max_rounds=1)
    source_file = "ts_benchmark/baselines/crosslinear/model/crosslinear_model.py"

    class FakeClient:
        api_available = True

        def call_json(self, **_kwargs):
            return {
                "ablation_questions": [
                    {
                        "target_id": "abl_corr_emb",
                        "target_kind": "mechanism_ablation",
                        "mechanism_id": "correlation_embedding",
                        "mechanism_name": "Correlation Embedding",
                        "diagnosis_question": "Does ablating correlation embedding change mse_norm?",
                        "causal_variable": "alpha-weighted correlation embedding branch",
                        "evidence_files": [source_file],
                        "evidence_anchors": [
                            "self.alpha * x_obj + (1 - self.alpha) * self.correlation_embedding(\n            x_enc\n        )"
                        ],
                        "edit_spec": {
                            "target_file": source_file,
                            "anchor_text": "self.alpha * x_obj + (1 - self.alpha) * self.correlation_embedding(\n            x_enc\n        )",
                            "replacement_intent": "Bypass correlation embedding",
                            "replacement_pseudocode": "x_obj = x_obj",
                            "shape_invariant_argument": "x_obj shape is preserved.",
                            "risk": "low",
                        },
                        "exact_edit_intent": "Bypass correlation embedding with x_obj = x_obj.",
                        "expected_behavior_delta": "mse_norm may change if correlation embedding matters.",
                    }
                ]
            }

    session.client = FakeClient()
    monkeypatch.setattr(
        "evocast.harness.mechanism_ablation_diagnosis.build_mechanism_evidence_graph",
        lambda *_args, **_kwargs: {"expanded_source_files": [source_file]},
    )

    result = generate_mechanism_ablation_plan(
        session,
        model_key="CrossLinear",
        objective_metric="mse_norm",
        evaluation_stage="smoke",
        analysis={},
        max_targets=1,
    )

    review = result["review"]
    assert review["status"] == "approved"
    assert [target["target_id"] for target in review["reviewed_targets"]] == ["abl_corr_emb"]
    assert review["rejected_targets"] == []
    assert "observability" not in review["reviewed_targets"][0]


def test_required_baseline_diagnosis_hard_fails_when_no_executable_targets(monkeypatch) -> None:
    session = create_dtaf_task("baseline_diag_no_executable_targets", max_rounds=1)
    baseline = {"model_name": "DTAF", "metrics": {"mse_norm": 1.0}, "candidate_id": "baseline_001_DTAF"}

    monkeypatch.setattr(wizard, "agent_dir", lambda: Path(session.base_dir))
    monkeypatch.setattr(wizard, "analyze_model_structure", lambda *_args, **_kwargs: {"source_files": [{"path": "ts_benchmark/baselines/dtaf/dtaf.py"}]})
    monkeypatch.setattr(
        wizard,
        "generate_mechanism_ablation_plan",
        lambda *_args, **_kwargs: {
            "plan": {
                "schema_version": "mechanism_ablation_plan_v1",
                "base_model": "DTAF",
                "objective_metric": "mse_norm",
                "evaluation_stage": "smoke",
                "targets": [],
            },
            "review": {
                "schema_version": "mechanism_ablation_plan_review_v1",
                "status": "rejected",
                "reviewed_targets": [],
                "rejected_targets": [],
                "errors": ["no_valid_mechanism_ablation_targets"],
                "corrections": [],
            },
        },
    )
    monkeypatch.setattr(wizard, "persist_ablation_plan", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(wizard, "run_ablation_round", lambda *_args, **_kwargs: pytest.fail("ablation should not run"))

    with pytest.raises(RuntimeError, match="baseline_diagnosis_ablation_targets_required_but_none_executable"):
        wizard.run_baseline_diagnosis_before_agent(
            task_id=session.task_id,
            baseline=baseline,
            objective_metric="mse_norm",
            budget="unified",
            seed=2021,
            max_targets=1,
        )

    diagnosis = load_runtime_state(session.base_dir, session.task_id).baseline_diagnosis
    assert diagnosis["status"] == "invalid_plan"
    assert diagnosis["plan_review"]["review_errors"] == ["no_valid_mechanism_ablation_targets"]


def test_nonblocking_baseline_diagnosis_converts_prelaunch_exception_to_evidence(monkeypatch) -> None:
    session = create_dtaf_task("baseline_diag_nonblocking_prelaunch", max_rounds=10)
    baseline = {"model_name": "DTAF", "metrics": {"mse_norm": 1.0}, "candidate_id": "baseline_001_DTAF"}

    monkeypatch.setattr(wizard, "runtime_dir", lambda: Path(session.base_dir))

    def fail_required_diagnosis(**_kwargs):
        raise RuntimeError("baseline_diagnosis_ablation_targets_required_but_none_executable")

    monkeypatch.setattr(wizard, "run_baseline_diagnosis_before_agent", fail_required_diagnosis)

    summary = wizard.run_baseline_diagnosis_nonblocking_before_agent(
        task_id=session.task_id,
        baseline=baseline,
        objective_metric="mse_norm",
        budget="unified",
        seed=2021,
        max_targets=1,
    )

    assert summary["status"] == "failed_non_blocking"
    assert "baseline_diagnosis_ablation_targets_required_but_none_executable" in summary["reason"]
    diagnosis_path = task_knowledge_dir(session.base_dir, session.task_id) / "baseline_diagnosis" / "diagnosis_summary.json"
    assert summary["diagnosis"]["baseline_diagnosis_path"] == str(diagnosis_path)
    diagnosis = load_runtime_state(session.base_dir, session.task_id).baseline_diagnosis
    assert diagnosis["failed_ablation_count"] == 1
    assert diagnosis["failed_ablations"][0]["action"] == "continue_agent_with_partial_evidence"


def test_write_nonblocking_baseline_diagnosis_preserves_existing_counts(monkeypatch, tmp_path: Path) -> None:
    task_id = "baseline_diag_persist"
    knowledge_dir = task_knowledge_dir(str(tmp_path), task_id)
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    diagnosis_path = knowledge_dir / "baseline_diagnosis.json"
    existing = {
        "schema_version": "baseline_diagnosis_v3",
        "task_id": task_id,
        "executed_ablation_count": 4,
        "failed_ablation_count": 2,
        "ablations": [{"target_id": "q001", "status": "success"}],
        "failed_ablations": [{"target": "q002", "failure_type": "x"}],
    }
    diagnosis_path.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(wizard, "runtime_dir", lambda: tmp_path)

    returned = wizard.write_nonblocking_baseline_diagnosis(
        task_id=task_id,
        baseline={"model_name": "DUET", "metrics": {"mse_norm": 1.0}},
        objective_metric="mse_norm",
        reason="partial",
        targets=[],
        results=[{"status": "error", "target_name": "q003", "error": "boom"}],
    )

    assert returned == knowledge_dir / "baseline_diagnosis" / "diagnosis_summary.json"
    reloaded = load_runtime_state(str(tmp_path), task_id).baseline_diagnosis
    assert reloaded["executed_ablation_count"] == 4
    assert reloaded["failed_ablation_count"] == 2


def test_baseline_diagnosis_uses_current_best_reference_metrics_after_seed_promotion(monkeypatch) -> None:
    session = create_dtaf_task("baseline_diag_current_best_reference", max_rounds=1)
    baseline = {"model_name": "DTAF", "metrics": {"mse_norm": 1.0}, "candidate_id": "baseline_001_DTAF"}
    target = _target()
    captured_references: list[dict] = []

    monkeypatch.setattr(wizard, "agent_dir", lambda: Path(session.base_dir))
    monkeypatch.setattr(wizard, "analyze_model_structure", lambda *_args, **_kwargs: {"source_files": [{"path": "ts_benchmark/baselines/dtaf/dtaf.py"}]})
    _patch_mechanism_plan(monkeypatch, target)
    monkeypatch.setattr(wizard, "persist_ablation_plan", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(wizard, "read_ablation_results", lambda *_args, **_kwargs: {"count": 1, "ablations": [{"status": "success"}]})

    def fake_run_ablation_round(_session, _target, **kwargs):
        captured_references.append(dict(kwargs.get("reference_metrics") or {}))
        return {
            "status": "ok",
            "record": {
                "target_id": target["target_id"],
                "target_name": target["target_id"],
                "mechanism_id": target["mechanism_id"],
                "mechanism_name": target["mechanism_name"],
                "exact_edit_intent": target["exact_edit_intent"],
                "status": "success",
                "evaluation_stage": "smoke",
                "metrics": {"mse_norm": 0.97},
                "metric_delta": {"mse_norm": {"candidate": 0.97, "baseline": 1.0, "delta": -0.03, "relative_delta": -0.03}},
                "usable_evidence_status": "usable_evidence",
                "variant_path": f"evocast/task_knowledge/{session.task_id}/rounds/Ablation001/workspace/round_entry.py",
                "artifact_paths": [],
                "interpretation": {"recommended_action": "run_seed_eval"},
                "created_at": "2026-07-05T00:00:00",
            },
        }

    monkeypatch.setattr(wizard, "run_ablation_round", fake_run_ablation_round)
    monkeypatch.setattr(
        wizard,
        "run_seed_eval",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "mean": 0.9,
            "reference_mean": 1.0,
            "reference_std": 0.1,
            "valid_metric_seeds": 3,
            "successful_seeds": 3,
            "significance_decision": {"decision": "accept"},
            "promoted_to_current_best": True,
        },
    )

    summary = wizard.run_baseline_diagnosis_before_agent(
        task_id=session.task_id,
        baseline=baseline,
        objective_metric="mse_norm",
        budget="unified",
        seed=2021,
        max_targets=1,
    )

    assert summary["status"] in {"completed", "partial"}
    assert captured_references == [{"mse_norm": 1.0}]


def test_baseline_diagnosis_zero_max_targets_skips_and_persists(monkeypatch) -> None:
    session = create_dtaf_task("baseline_diag_zero_max_targets", max_rounds=1)
    baseline = {"model_name": "DTAF", "metrics": {"mse_norm": 1.0}, "candidate_id": "baseline_001_DTAF"}

    monkeypatch.setattr(wizard, "analyze_model_structure", lambda *_args, **_kwargs: pytest.fail("analysis should be skipped"))
    monkeypatch.setattr(wizard, "generate_mechanism_ablation_plan", lambda *_args, **_kwargs: pytest.fail("LLM plan should be skipped"))
    monkeypatch.setattr(wizard, "run_ablation_round", lambda *_args, **_kwargs: pytest.fail("ablation should be skipped"))

    summary = wizard.run_baseline_diagnosis_before_agent(
        task_id=session.task_id,
        baseline=baseline,
        objective_metric="mse_norm",
        budget="unified",
        seed=2021,
        max_targets=0,
    )

    diagnosis = load_runtime_state(session.base_dir, session.task_id).baseline_diagnosis

    assert summary["status"] == "skipped"
    assert diagnosis["status"] == "skipped"
    assert diagnosis["planned_ablation_count"] == 0
    assert diagnosis["executed_ablation_count"] == 0


@pytest.mark.parametrize(
    ("decision", "candidate_mean", "expected_effect", "expected_action", "promoted_to_current_best"),
    [
        ("reject", 1.0936, "confirmed_essential", "keep_current_baseline", False),
        ("reject", 0.999956, "seed_eval_weak_or_uncertain", "keep_current_baseline", False),
        ("accept", 0.97, "confirmed_harmful_or_redundant", "promote_ablation_variant_to_current_best", True),
    ],
)
def test_baseline_diagnosis_persists_seed_eval_classification_semantics(
    monkeypatch,
    decision: str,
    candidate_mean: float,
    expected_effect: str,
    expected_action: str,
    promoted_to_current_best: bool,
) -> None:
    session = create_dtaf_task(f"baseline_diag_seed_eval_{decision}", max_rounds=1)
    baseline = {"model_name": "DTAF", "metrics": {"mse_norm": 1.0}, "candidate_id": "baseline_001_DTAF"}
    target = _target()

    monkeypatch.setattr(wizard, "agent_dir", lambda: Path(session.base_dir))
    monkeypatch.setattr(wizard, "analyze_model_structure", lambda *_args, **_kwargs: {"source_files": [{"path": "ts_benchmark/baselines/dtaf/dtaf.py"}]})
    _patch_mechanism_plan(monkeypatch, target)
    monkeypatch.setattr(wizard, "persist_ablation_plan", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(wizard, "read_ablation_results", lambda *_args, **_kwargs: {"count": 1, "ablations": [{"status": "success"}]})
    monkeypatch.setattr(
        wizard,
        "run_ablation_round",
        lambda _session, _target, **_kwargs: {
            "status": "ok",
            "record": {
                "target_id": target["target_id"],
                "target_name": target["target_id"],
                "mechanism_id": target["mechanism_id"],
                "mechanism_name": target["mechanism_name"],
                "exact_edit_intent": target["exact_edit_intent"],
                "status": "success",
                "evaluation_stage": "smoke",
                "metrics": {"mse_norm": 0.97},
                "metric_delta": {"mse_norm": {"candidate": 0.97, "baseline": 1.0, "delta": -0.03, "relative_delta": -0.03}},
                "usable_evidence_status": "usable_evidence",
                "variant_path": f"evocast/task_knowledge/{session.task_id}/rounds/Ablation001/workspace/round_entry.py",
                "artifact_paths": [],
                "interpretation": {"recommended_action": "run_seed_eval"},
                "created_at": "2026-07-05T00:00:00",
            },
        },
    )
    monkeypatch.setattr(
        wizard,
        "run_seed_eval",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "mean": candidate_mean,
            "reference_mean": 1.0,
            "reference_std": 0.1,
            "valid_metric_seeds": 3,
            "successful_seeds": 3,
            "significance_decision": {"decision": decision},
            "promoted_to_current_best": promoted_to_current_best,
        },
    )

    summary = wizard.run_baseline_diagnosis_before_agent(
        task_id=session.task_id,
        baseline=baseline,
        objective_metric="mse_norm",
        budget="unified",
        seed=2021,
        max_targets=1,
    )

    assert summary["status"] in {"completed", "partial"}
    diagnosis = load_runtime_state(session.base_dir, session.task_id).baseline_diagnosis
    policy = diagnosis["mechanism_development_policy"][0]
    ablation = diagnosis["ablations"][0]

    assert policy["module_effect"] == expected_effect
    assert policy["recommended_action"] == expected_action
    assert bool(ablation["seed_eval"]["promoted_to_current_best"]) == promoted_to_current_best
