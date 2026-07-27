from __future__ import annotations

from evocast.research.ablation.task import validate_ablation_task
from evocast.tools.tfb_ablation import run_ablation
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
    }


def test_legacy_exact_edit_strategy_is_rejected() -> None:
    validation = validate_ablation_task(
        {
            **_target(),
            "exact_edit_strategy": "wrap_module_zero_output",
        }
    )

    assert validation["status"] == "error"
    assert "forbidden_field:exact_edit_strategy" in validation["errors"]


def test_run_ablation_rejects_legacy_field_compatibility() -> None:
    session = create_dtaf_task("legacy_ablation_kind_reject", max_rounds=1)

    result = run_ablation(
        session,
        {
            "target": _target(),
            "target_id": "q001_attention",
            "target_kind": "mechanism_ablation",
            "model_key": "DTAF",
            "baseline_model": "DTAF",
            "baseline_metrics": {"mse_norm": 1.0},
            "ablation_kind": "zero_output_ablation",
            "objective_metric": "mse_norm",
        },
    )

    assert result["status"] == "failed"
    assert result["record"]["failure_type"] == "invalid_ablation_task"
    assert "forbidden_field:ablation_kind" in result["record"]["failure_reason"]
