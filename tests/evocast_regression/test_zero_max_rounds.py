from __future__ import annotations

import json
from pathlib import Path

import pytest

from evocast.harness.rounds import close_current_round, round_progress, start_round
from evocast.reports.review_html import _final_architecture
from evocast.reports.view_model import build_report_view_model
from evocast.state.runtime.store import load_runtime_state, sync_task_stage


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_task_config(base_dir: Path, task_id: str, *, max_rounds: int) -> None:
    _write_json(
        base_dir / "task_knowledge" / task_id / "task_config.json",
        {
            "task_id": task_id,
            "objective_metric": "mse_norm",
            "force_full_rounds": True,
            "max_rounds": max_rounds,
        },
    )


def test_zero_max_rounds_rejects_formal_research_round_start(tmp_path: Path) -> None:
    base_dir = tmp_path / "runtime"
    task_id = "zero_max_rounds"
    _write_task_config(base_dir, task_id, max_rounds=0)

    with pytest.raises(RuntimeError, match="max_rounds=0"):
        start_round(
            base_dir=str(base_dir),
            task_id=task_id,
            fit_point="model.predictor",
            hypothesis="formal research round must not start",
            evidence_source="test fixture",
        )

    progress = round_progress(str(base_dir), task_id)
    assert progress["research_rounds"] == 0
    assert progress["terminal_rounds"] == 0


def test_zero_max_rounds_completion_state_is_current_runtime_contract(tmp_path: Path) -> None:
    base_dir = tmp_path / "runtime"
    task_id = "zero_max_rounds_finalize"
    _write_task_config(base_dir, task_id, max_rounds=0)

    sync_task_stage(
        str(base_dir),
        task_id,
        stage="completed",
        status="completed",
        extra={
            "research": {
                "finalize_reason": "max_rounds_zero_baseline_diagnosis_complete",
                "next_round": 1,
                "valid_trial_rounds": 0,
                "terminal_rounds": 0,
            }
        },
    )

    state = load_runtime_state(str(base_dir), task_id)
    assert state.task_status == "completed"
    assert state.current_stage == "completed"
    assert state.current_stage_status == "completed"
    assert state.research["finalize_reason"] == "max_rounds_zero_baseline_diagnosis_complete"


def test_zero_max_rounds_report_says_architecture_search_was_not_started() -> None:
    html = _final_architecture(
        {
            "task": {"language": "zh", "objective_metric": "mse_norm"},
            "final_state": {
                "architecture_search_status": "not_started",
                "final_best": {"candidate_id": "Informer", "display_name": "Informer", "metrics": {"mse_norm": 1.0}},
            },
        },
        {"final_architecture_summary": "未产生架构搜索结果：max_rounds=0。"},
    )
    assert "未产生架构搜索结果：max_rounds=0。" in html


def test_zero_max_rounds_replaces_conflicting_llm_architecture_narrative() -> None:
    view = build_report_view_model(
        fact_pack={
            "task": {"language": "zh", "objective_metric": "mse_norm"},
            "final_state": {"architecture_search_status": "not_started", "final_best": {}},
            "research_rounds": [],
        },
        narrative={
            "title": "fixture",
            "executive_summary": [{"text": "未找到更优架构。"}],
            "final_architecture_summary": "未找到更优架构。",
        },
    )

    narrative = view["narrative"]
    assert narrative["executive_summary"] == [{"text": "架构搜索未启动：max_rounds=0，仅完成 baseline 与诊断消融。", "evidence_refs": []}]
    assert narrative["final_architecture_summary"] == "未产生架构搜索结果：max_rounds=0。"


def test_task_terminal_state_after_infra_failed_last_round(tmp_path: Path) -> None:
    base_dir = tmp_path / "runtime"
    task_id = "infra_failed_terminal_task"
    _write_task_config(base_dir, task_id, max_rounds=1)
    start_round(
        base_dir=str(base_dir),
        task_id=task_id,
        fit_point="model.predictor",
        hypothesis="failed build still counts as a research round",
        evidence_source="test fixture",
    )
    close_current_round(
        base_dir=str(base_dir),
        task_id=task_id,
        status="infra_failed",
        reason="fixture infra failure",
        extra={"failure_stage": "build"},
    )
    sync_task_stage(
        str(base_dir),
        task_id,
        stage="completed",
        status="completed_with_failed_rounds",
        extra={"research": {"finalize_reason": "max_rounds_exhausted", "failed_rounds": 1}},
    )

    state = load_runtime_state(str(base_dir), task_id)
    assert state.task_status == "completed_with_failed_rounds"
    assert state.current_stage == "completed"
    assert state.current_stage_status == "completed_with_failed_rounds"


def test_baseline_diagnosis_ablation_records_do_not_consume_research_budget(tmp_path: Path) -> None:
    base_dir = tmp_path / "runtime"
    task_id = "baseline_diagnosis_budget"
    _write_task_config(base_dir, task_id, max_rounds=3)
    rounds_dir = base_dir / "task_knowledge" / task_id / "rounds"
    for round_id in range(1, 4):
        _write_json(
            rounds_dir / f"Ablation{round_id:03d}" / "ablation_record.json",
            {
                "ablation_index": round_id,
                "ablation_id": f"Ablation{round_id:03d}",
                "variant_path": f"round_sources/{task_id}/Ablation{round_id:03d}/round_entry.py",
                "status": "rejected",
                "usable_evidence_status": "failed_evidence",
            },
        )

    progress = round_progress(str(base_dir), task_id)
    assert progress["total_rounds"] == 0
    assert progress["terminal_rounds"] == 0
    assert progress["diagnostic_terminal_rounds"] == 0

    record = start_round(
        base_dir=str(base_dir),
        task_id=task_id,
        fit_point="model.predictor",
        hypothesis="formal research round after baseline diagnosis",
        evidence_source="test fixture",
    )
    assert record["round_id"] == 1
    assert record["counts_toward_research_budget"] is True

    close_current_round(
        base_dir=str(base_dir),
        task_id=task_id,
        status="proposal_failed",
        reason="formal terminal fixture",
    )
    progress = round_progress(str(base_dir), task_id)
    assert progress["research_rounds"] == 1
    assert progress["terminal_rounds"] == 1
    assert progress["diagnostic_terminal_rounds"] == 0
