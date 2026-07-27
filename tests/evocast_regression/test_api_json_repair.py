from __future__ import annotations

from evocast.harness.api_client import parse_json_object


def test_json_repair_closes_target_id_array_items():
    text = (
        '{"ablation_questions": ['
        '{"target_id": "eab_mask_bypass", "edit_spec": {"risk": "medium"}, '
        '{"target_id": "erb_subtraction_bypass", "edit_spec": {"risk": "medium"}], '
        '"preserve_contract": {"task": "fixed"}}'
    )

    parsed = parse_json_object(text, required_top_level_keys=["ablation_questions"])

    assert [item["target_id"] for item in parsed["ablation_questions"]] == [
        "eab_mask_bypass",
        "erb_subtraction_bypass",
    ]
    assert parsed["preserve_contract"]["task"] == "fixed"


def test_json_repair_preserves_legacy_question_id_array_items():
    text = (
        '{"questions": ['
        '{"question_id": "q001", "edit_spec": {"risk": "low"}, '
        '{"question_id": "q002", "edit_spec": {"risk": "high"}], '
        '"selection_rationale": "choose the first valid candidate"}'
    )

    parsed = parse_json_object(text, required_top_level_keys=["questions"])

    assert [item["question_id"] for item in parsed["questions"]] == ["q001", "q002"]
    assert parsed["selection_rationale"] == "choose the first valid candidate"
