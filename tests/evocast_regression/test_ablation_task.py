from __future__ import annotations

from evocast.research.ablation.task import validate_ablation_task


def _mechanism_task(**overrides):
    task = {
        "target_id": "q001_router",
        "target_kind": "mechanism_ablation",
        "mechanism_id": "m_router_attention",
        "mechanism_name": "route-based cross-dimension attention",
        "diagnosis_question": "Does route-based attention matter?",
        "causal_variable": "router vectors distribute messages across dimensions",
        "evidence_files": ["ts_benchmark/baselines/time_series_library/layers/SelfAttention_Family.py"],
        "evidence_anchors": ["dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer"],
        "exact_edit_intent": "replace route-based receiver attention with a shape-preserving non-routed baseline",
        "edit_spec": {
            "target_file": "ts_benchmark/baselines/time_series_library/layers/SelfAttention_Family.py",
            "anchor_text": "dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer",
            "replacement_intent": "replace route-based receiver attention with a shape-preserving non-routed baseline",
            "replacement_pseudocode": "dim_receive, attn = dim_send, None",
            "shape_invariant_argument": "dim_send preserves the receiver tensor shape expected by downstream code.",
            "risk": "medium",
        },
        "preserve_contract": {
            "input": "preserve model input protocol",
            "output": "preserve forecast output shape",
            "task": "do not change dataset, target columns, horizon, metric, optimizer, scheduler, or training policy",
        },
        "expected_behavior_delta": "same-input forecast should change if this mechanism is active",
    }
    task.update(overrides)
    return task


def test_mechanism_ablation_task_is_valid() -> None:
    validation = validate_ablation_task(_mechanism_task(), evaluation_stage="smoke")

    assert validation["status"] == "ok"
    assert validation["task"]["target_kind"] == "mechanism_ablation"


def test_mechanism_ablation_task_accepts_intent_only_anchor() -> None:
    task = _mechanism_task()
    task["edit_spec"] = {
        "target_file": "ts_benchmark/baselines/time_series_library/layers/SelfAttention_Family.py",
        "anchor_text": "dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer",
        "replacement_intent": "remove routed receiver attention while preserving return structure",
        "shape_invariant_argument": "the coding agent must preserve tensor shape and adapter protocol",
    }

    validation = validate_ablation_task(task, evaluation_stage="smoke")

    assert validation["status"] == "ok"


def test_preserve_contract_is_not_rejected_by_protocol_keywords() -> None:
    validation = validate_ablation_task(
        _mechanism_task(
            preserve_contract={
                "input": "x_enc tensor of shape (B, seq_len, dec_in) unchanged",
                "output": "forecast returns (B, pred_len, dec_in) tensor unchanged",
                "task": "no change to dataset, target columns, horizon, metric, optimizer, scheduler, or training policy",
            }
        ),
        evaluation_stage="experiment",
    )

    assert validation["status"] == "ok"


def test_component_and_hypothesis_ablation_are_rejected() -> None:
    component = validate_ablation_task(_mechanism_task(target_kind="component_ablation"))
    hypothesis = validate_ablation_task(_mechanism_task(target_kind="hypothesis_ablation"))

    assert component["status"] == "error"
    assert "unsupported_target_kind:component_ablation" in component["errors"]
    assert hypothesis["status"] == "error"
    assert "unsupported_target_kind:hypothesis_ablation" in hypothesis["errors"]


def test_legacy_component_fields_are_rejected() -> None:
    validation = validate_ablation_task(_mechanism_task(component_path="self.model.attn"))

    assert validation["status"] == "error"
    assert "forbidden_field:component_path" in validation["errors"]


def test_exact_edit_intent_cannot_change_task_semantics() -> None:
    validation = validate_ablation_task(
        _mechanism_task(exact_edit_intent="change the pred_len to make the ablation easier")
    )

    assert validation["status"] == "error"
    assert "exact_edit_intent_changes_task_or_training_semantics" in validation["errors"]
