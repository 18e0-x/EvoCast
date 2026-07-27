from __future__ import annotations

import json
from pathlib import Path

from evocast.domain.knowledge_paths import task_knowledge_dir
from evocast.research.idea_planner import _messages, load_research_evidence_bundle
from evocast.state.domain_store import save_runtime_payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_scientist_critic_prompt_defines_method_and_rejected_target_semantics() -> None:
    messages = _messages(
        evidence_bundle={
            "baseline_diagnosis": {
                "plan_review": {
                    "rejected_targets": [
                        {
                            "target_id": "abl_corr_emb",
                            "mechanism_id": "correlation_embedding",
                        }
                    ]
                }
            },
            "round_history_digest": {"items": []},
            "recent_round_status_table": [],
        },
        research_id="Research002",
        planner_architecture="scientist_critic",
    )

    system = messages[0]["content"]
    user = json.loads(messages[1]["content"])
    hard_requirements = "\n".join(user["hard_requirements"])
    combined = system + "\n" + user["architecture_instruction"] + "\n" + hard_requirements

    assert "Scientist phase" in combined
    assert "Critic phase" in combined
    assert "Selection phase" in combined
    assert "pre-execution rejected_targets" in combined
    assert "not mechanism evidence" in combined
    assert "Do not select the same ablation action" in combined
    assert "runtime failures as scientific evidence" in combined
    assert "Use round_history_digest as an explored-territory map" in combined
    assert "not as a template to extend" in combined
    assert "current_best is canonical_current_best" in combined or "canonical_current_best / runtime_state.current_best" in combined
    assert "Never infer current_best from a prior round's lower single-seed metric" in combined
    assert "history_usage" in json.dumps(user["required_json_shape"])


def test_research_program_hypothesis_competition_prompt_is_live_option() -> None:
    messages = _messages(
        evidence_bundle={
            "dataset_profile": {"Transition": 0.75},
            "baseline_diagnosis": {"plan_review": {"rejected_targets": []}},
            "round_history_digest": {
                "items": [
                    {"round_id": "Research001", "one_line": "在 head 前加入残差模块，mse_norm=1.0，未提升。"},
                    {"round_id": "Research002", "one_line": "替换 head 后 shape_error，未产出指标。"},
                ]
            },
            "recent_round_status_table": [
                {"round_id": "Research001", "status": "successful"},
                {"round_id": "Research002", "status": "infra_failed", "error_type": "shape_error"},
            ],
        },
        research_id="Research003",
        planner_architecture="research_program_hypothesis_competition",
    )

    system = messages[0]["content"]
    user = json.loads(messages[1]["content"])
    hard_requirements = "\n".join(user["hard_requirements"])
    combined = system + "\n" + user["architecture_instruction"] + "\n" + hard_requirements

    assert user["planner_architecture"] == "research_program_hypothesis_competition"
    assert "Research Program Director" in combined
    assert "supported, weakened, saturated, or unresolved" in combined
    assert "Hypothesis Competition scientist" in combined
    assert "competing explanations" in combined
    assert "distinguishes among competing explanations" in combined
    assert "terminal_display_title must be 8 words or fewer" in combined


def test_a3_removes_evidence_fields_from_planner_payload() -> None:
    messages = _messages(
        evidence_bundle={
            "task_config": {"agent_ablation": "A3"},
            "dataset_profile": {"Transition": 0.75},
            "baseline_diagnosis": {"plan_review": {"rejected_targets": [{"target_id": "x"}]}},
            "round_history_digest": {"items": [{"round_id": "Research001", "one_line": "digest"}]},
            "recent_round_status_table": [{"round_id": "Research001", "status": "rejected"}],
            "runtime_state": {"current_best": {"metric": 0.1}},
            "canonical_current_best": {"metric": 0.1},
            "latest_build_contract": {"research_id": "Research002"},
        },
        research_id="Research002",
        planner_architecture="research_program_hypothesis_competition",
    )

    user = json.loads(messages[1]["content"])
    hard_requirements = "\n".join(user["hard_requirements"])
    payload = json.loads(user["evidence_bundle_text"])

    assert user["agent_ablation"] == "A3"
    assert "Use dataset, ablation, and prior-round evidence together whenever available." not in hard_requirements
    assert "Use round_history_digest as an explored-territory map" not in hard_requirements
    assert "dataset_profile" not in payload
    assert "baseline_diagnosis" not in payload
    assert "round_history_digest" not in payload
    assert "recent_round_status_table" not in payload
    assert payload["canonical_current_best"]["metric"] == 0.1
    assert payload["runtime_state"]["current_best"]["metric"] == 0.1


def test_a5_freezes_history_evidence_but_keeps_live_current_best(tmp_path, monkeypatch) -> None:
    import evocast.research.idea_planner as planner_module

    task_id = "task_a5"
    root = task_knowledge_dir(str(tmp_path), task_id)
    _write_json(root / "task_config.json", {"agent_ablation": "A5"})
    _write_json(root / "runtime_state.json", {"current_best": {"metric": 0.42}})
    _write_json(root / "baseline_diagnosis.json", {"ablations": [{"id": "A1"}]})

    digest_state = {"items": [{"round_id": "Research001", "one_line": "initial digest"}]}
    status_state = [{"round_id": "Research001", "status": "rejected"}]

    monkeypatch.setattr(planner_module, "load_dataset_profile", lambda *_args, **_kwargs: {"profile": "initial"})
    monkeypatch.setattr(planner_module, "ensure_round_history_digest", lambda **_kwargs: digest_state)
    monkeypatch.setattr(planner_module, "round_status_table", lambda *_args, **_kwargs: status_state)

    evidence_first = load_research_evidence_bundle(
        str(tmp_path),
        task_id,
        api_config="providers/minimax.yaml",
        history_limit=20,
    )
    snapshot_path = root / "planner_evidence_snapshot_A5.json"

    assert evidence_first["planner_evidence_policy"]["agent_ablation"] == "A5"
    assert evidence_first["round_history_digest"]["items"][0]["round_id"] == "Research001"
    assert snapshot_path.is_file()

    digest_state = {"items": [{"round_id": "Research002", "one_line": "new digest"}]}
    status_state = [{"round_id": "Research002", "status": "successful"}]
    save_runtime_payload(str(tmp_path), task_id, {"current_best": {"metric": 0.31}})

    evidence_second = load_research_evidence_bundle(
        str(tmp_path),
        task_id,
        api_config="providers/minimax.yaml",
        history_limit=20,
    )

    assert evidence_second["round_history_digest"]["items"][0]["round_id"] == "Research001"
    assert evidence_second["recent_round_status_table"][0]["round_id"] == "Research001"
    assert evidence_second["runtime_state"]["current_best"]["metric"] == 0.31
    assert evidence_second["canonical_current_best"]["metric"] == 0.31
