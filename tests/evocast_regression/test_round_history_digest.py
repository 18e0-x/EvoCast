from __future__ import annotations

import json
from pathlib import Path

from evocast.domain.knowledge_paths import task_knowledge_dir
from evocast.research.idea_planner import load_research_evidence_bundle
from evocast.research.round_history_digest import ensure_round_history_digest, round_status_table
from evocast.state.domain_store import save_round_record, save_task_config


class FakeDigestClient:
    api_available = True

    def call_json(self, *_args, **_kwargs):
        return {
            "items": [
                {
                    "round_id": "Research001",
                    "one_line": "在 value+pos patch token 后加入跨变量残差路由，mse_norm=0.4675，未提升。",
                },
                {
                    "round_id": "Research002",
                    "one_line": "替换 alpha-mix 跨变量桥为 memory MoE，mse_norm=0.4539，未最终采纳。",
                },
            ]
        }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _round(root: Path, rid: str, *, status: str, mse_norm: float, idea: str) -> None:
    round_dir = root / "rounds" / rid
    _write_json(
        round_dir / "round.json",
        {
            "research_id": rid,
            "status": status,
            "display_idea": idea,
            "idea_summary": f"{idea} long implementation summary that should not be passed as recent_rounds.",
            "metrics": {"mse_norm": mse_norm, "mae_norm": 0.49},
        },
    )
    _write_json(
        root / "research_directions" / f"{rid}_research_program_hypothesis_competition.json",
        {
            "research_id": rid,
            "terminal_display_title": idea,
            "target_mechanism": f"{idea} mechanism",
            "target_code_region": "ts_benchmark/baselines/crosslinear/model/crosslinear_model.py",
        },
    )


def test_round_history_digest_is_generated_from_research_rounds(tmp_path, monkeypatch) -> None:
    import evocast.research.round_history_digest as digest_module

    monkeypatch.setattr(digest_module, "create_task_client", lambda **_kwargs: FakeDigestClient())
    root = task_knowledge_dir(str(tmp_path), "task_digest")
    _round(root, "Research001", status="rejected", mse_norm=0.467512, idea="Patch Router")
    _round(root, "Research002", status="scientific_rejected", mse_norm=0.45385, idea="Bridge MoE")

    digest = ensure_round_history_digest(
        base_dir=str(tmp_path),
        task_id="task_digest",
        api_config="providers/minimax.yaml",
        history_limit=20,
    )

    assert digest["schema_version"] == "round_history_digest_v3"
    assert [item["round_id"] for item in digest["items"]] == ["Research001", "Research002"]
    assert digest["items"][0]["one_line"].startswith("在 value+pos")
    assert (root / "round_history_digest.json").is_file()


def test_planner_history_includes_canonical_round_without_legacy_round_file(tmp_path, monkeypatch) -> None:
    import evocast.research.round_history_digest as digest_module

    task_id = "canonical_digest"
    save_task_config(str(tmp_path), task_id, {"task_id": task_id, "objective_metric": "mse_norm"})
    save_round_record(
        str(tmp_path),
        task_id,
        {
            "round_id": 1,
            "research_id": "Research001",
            "status": "completed",
            "proposal": {"innovation_name": "Canonical residual router"},
            "evaluation": {"status": "completed", "metrics": {"mse_norm": 0.42}},
            "decision": {"status": "reject", "reason": "did not beat baseline"},
        },
    )
    monkeypatch.setattr(digest_module, "create_task_client", lambda **_kwargs: FakeDigestClient())

    evidence = load_research_evidence_bundle(
        str(tmp_path), task_id, api_config="providers/minimax.yaml", history_limit=20,
    )

    assert evidence["round_history_digest"]["items"][0]["round_id"] == "Research001"
    assert evidence["recent_round_status_table"][0]["mse_norm"] == 0.42
    assert not (task_knowledge_dir(str(tmp_path), task_id) / "rounds" / "Research001" / "round.json").exists()


def test_planner_evidence_uses_digest_and_status_table_not_long_recent_rounds(tmp_path, monkeypatch) -> None:
    import evocast.research.round_history_digest as digest_module

    monkeypatch.setattr(digest_module, "create_task_client", lambda **_kwargs: FakeDigestClient())
    root = task_knowledge_dir(str(tmp_path), "task_planner_evidence")
    _round(root, "Research001", status="rejected", mse_norm=0.467512, idea="Patch Router")
    _round(root, "Research002", status="scientific_rejected", mse_norm=0.45385, idea="Bridge MoE")

    evidence = load_research_evidence_bundle(
        str(tmp_path),
        "task_planner_evidence",
        api_config="providers/minimax.yaml",
        history_limit=20,
    )

    assert "recent_rounds" not in evidence
    assert evidence["round_history_digest"]["items"][1]["round_id"] == "Research002"
    assert evidence["recent_round_status_table"] == round_status_table(
        str(tmp_path),
        "task_planner_evidence",
        limit=20,
    )
    assert "idea_summary" not in evidence["recent_round_status_table"][0]


def test_round_status_table_normalizes_marginal_gate_over_stale_completed_status(tmp_path) -> None:
    root = task_knowledge_dir(str(tmp_path), "task_marginal_status")
    _round(root, "Research004", status="completed", mse_norm=0.455358, idea="PSCVR")
    round_path = root / "rounds" / "Research004" / "round.json"
    record = json.loads(round_path.read_text(encoding="utf-8"))
    record["gate_decision"] = "marginal_no_seed_eval"
    record["gate_event"] = {
        "decision": "marginal_no_seed_eval",
        "relative_improvement": 0.004805,
        "threshold_met": False,
        "promoted_to_current_best": False,
    }
    round_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    table = round_status_table(str(tmp_path), "task_marginal_status", limit=20)

    assert table[0]["status"] == "marginal_no_seed_eval"
    assert table[0]["gate_decision"] == "marginal_no_seed_eval"
    assert table[0]["promoted_to_current_best"] is False


def test_round_history_digest_accepts_fact_or_string_items(tmp_path, monkeypatch) -> None:
    import evocast.research.round_history_digest as digest_module

    class FlexibleShapeClient:
        api_available = True

        def call_json(self, *_args, **_kwargs):
            return {
                "items": [
                    {"round_id": "Research001", "sentence": "在 head 前新增残差路径，mse_norm=0.5，未提升。"},
                    "Research002：替换跨变量桥，mse_norm=0.4，未最终采纳。",
                ]
            }

    monkeypatch.setattr(digest_module, "create_task_client", lambda **_kwargs: FlexibleShapeClient())
    root = task_knowledge_dir(str(tmp_path), "task_flexible_digest")
    _round(root, "Research001", status="rejected", mse_norm=0.5, idea="Residual")
    _round(root, "Research002", status="rejected", mse_norm=0.4, idea="Replacement")

    digest = ensure_round_history_digest(
        base_dir=str(tmp_path),
        task_id="task_flexible_digest",
        api_config="providers/minimax.yaml",
        history_limit=20,
    )

    assert digest["items"][0]["one_line"].startswith("在 head 前")
    assert digest["items"][1]["one_line"].startswith("Research002")


def test_round_history_digest_accepts_top_level_round_mapping(tmp_path, monkeypatch) -> None:
    import evocast.research.round_history_digest as digest_module

    class MappingClient:
        api_available = True

        def call_json(self, *_args, **_kwargs):
            return {
                "Research001": "在 head 前新增残差路径，mse_norm=0.5，未提升。",
                "Research002": "替换跨变量桥，mse_norm=0.4，未最终采纳。",
            }

    monkeypatch.setattr(digest_module, "create_task_client", lambda **_kwargs: MappingClient())
    root = task_knowledge_dir(str(tmp_path), "task_mapping_digest")
    _round(root, "Research001", status="rejected", mse_norm=0.5, idea="Residual")
    _round(root, "Research002", status="rejected", mse_norm=0.4, idea="Replacement")

    digest = ensure_round_history_digest(
        base_dir=str(tmp_path),
        task_id="task_mapping_digest",
        api_config="providers/minimax.yaml",
        history_limit=20,
    )

    assert digest["items"][0] == {
        "round_id": "Research001",
        "one_line": "在 head 前新增残差路径，mse_norm=0.5，未提升。",
    }


def test_round_history_digest_accepts_compressed_fact_items(tmp_path, monkeypatch) -> None:
    import evocast.research.round_history_digest as digest_module

    class CompressedFactClient:
        api_available = True

        def call_json(self, *_args, **_kwargs):
            return {
                "items": [
                    {
                        "round_id": "Research001",
                        "compressed_fact": "在 PAttn head 前加入路由残差，mse_norm=0.47619，未晋升。",
                    }
                ]
            }

    monkeypatch.setattr(digest_module, "create_task_client", lambda **_kwargs: CompressedFactClient())
    root = task_knowledge_dir(str(tmp_path), "task_compressed_fact_digest")
    _round(root, "Research001", status="rejected", mse_norm=0.47619, idea="Router Residual")

    digest = ensure_round_history_digest(
        base_dir=str(tmp_path),
        task_id="task_compressed_fact_digest",
        api_config="providers/minimax.yaml",
        history_limit=20,
    )

    assert digest["source"] == "llm"
    assert digest["items"][0] == {
        "round_id": "Research001",
        "one_line": "在 PAttn head 前加入路由残差，mse_norm=0.47619，未晋升。",
    }


def test_round_history_digest_falls_back_when_llm_payload_is_unusable(tmp_path, monkeypatch) -> None:
    import evocast.research.round_history_digest as digest_module

    class BadShapeClient:
        api_available = True

        def call_json(self, *_args, **_kwargs):
            return {"items": [{"round_id": "Research001", "unexpected": "no usable digest"}]}

    monkeypatch.setattr(digest_module, "create_task_client", lambda **_kwargs: BadShapeClient())
    root = task_knowledge_dir(str(tmp_path), "task_bad_shape_digest")
    _round(root, "Research001", status="rejected", mse_norm=0.47619, idea="Router Residual")

    digest = ensure_round_history_digest(
        base_dir=str(tmp_path),
        task_id="task_bad_shape_digest",
        api_config="providers/minimax.yaml",
        history_limit=20,
    )

    assert digest["source"] == "llm"
    assert digest["items"][0]["round_id"] == "Research001"
    assert "Research001" in digest["items"][0]["one_line"]
    assert "mse_norm=0.47619" in digest["items"][0]["one_line"]
    assert "promoted_to_current_best=false" in digest["items"][0]["one_line"]


def test_round_history_digest_falls_back_when_client_fails(tmp_path, monkeypatch) -> None:
    import evocast.research.round_history_digest as digest_module

    class FailingClient:
        api_available = True

        def call_json(self, *_args, **_kwargs):
            raise RuntimeError("json parse failed")

    monkeypatch.setattr(digest_module, "create_task_client", lambda **_kwargs: FailingClient())
    root = task_knowledge_dir(str(tmp_path), "task_client_failure_digest")
    _round(root, "Research001", status="rejected", mse_norm=0.47619, idea="Router Residual")

    digest = ensure_round_history_digest(
        base_dir=str(tmp_path),
        task_id="task_client_failure_digest",
        api_config="providers/minimax.yaml",
        history_limit=20,
    )

    assert digest["source"] == "mechanical_fallback"
    assert "json parse failed" in digest["fallback_reason"]
    assert digest["items"][0]["round_id"] == "Research001"
    assert "mse_norm=0.47619" in digest["items"][0]["one_line"]
