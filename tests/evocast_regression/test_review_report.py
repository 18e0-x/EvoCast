from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from evocast.domain.knowledge_paths import task_knowledge_dir, task_runs_dir
from evocast.research.dataset_profile import dataset_profile_path
from evocast.reports.review_fact_pack import build_review_fact_pack
from evocast.reports.review_html import _promotion_outcome_status, _round_success_failed_counts, render_review_html
from evocast.reports.review_narrative import build_review_narrative
from evocast.reports.review_report import generate_review_report
from evocast.state.cost_ledger import append_cost_event
from evocast.state.domain_store import (
    domain_state_path,
    load_domain_state,
    load_round_record,
    save_round_record,
    save_runtime_payload,
    save_task_config,
)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_fact_pack_reads_baseline_facts_from_canonical_state_only(tmp_path: Path) -> None:
    task_id = "canonical_report"
    save_task_config(
        str(tmp_path), task_id,
        {"task_id": task_id, "objective_metric": "mse_norm", "research_intent": "Improve Informer."},
    )
    save_runtime_payload(
        str(tmp_path), task_id,
        {
            "objective_metric": "mse_norm",
            "baseline": {"candidate_id": "baseline_informer", "display_name": "Informer", "metrics": {"mse_norm": 0.25}},
            "current_best": {"candidate_id": "baseline_informer", "display_name": "Informer", "metrics": {"mse_norm": 0.25}},
            "baseline_search_progress": {"status": "completed", "cheap_completed": 1, "cheap_target": 1},
            "baseline_diagnosis": {"status": "completed", "selected_mechanisms": []},
            "research": {},
        },
    )

    fact_pack = build_review_fact_pack(task_id=task_id, base_dir=str(tmp_path))

    assert fact_pack["baseline"]["model"] == "Informer"
    assert fact_pack["baseline"]["metrics"] == {"mse_norm": 0.25}


def _fixture_task(base_dir: Path, *, language: str = "zh") -> str:
    task_id = "review_report_fixture"
    knowledge_dir = task_knowledge_dir(str(base_dir), task_id)
    _write_json(
        knowledge_dir / "task_config.json",
        {
            "task_id": task_id,
            "dataset_path": "dataset/forecasting/ETTh1.csv",
            "target_columns": ["OT"],
            "objective_metric": "mse_norm",
            "horizon": 96,
            "seq_len": 96,
            "budget": "unified",
            "build_mode": True,
            "language": language,
        },
    )
    _write_json(
        knowledge_dir / "compiled_config.json",
        {
            "data_config": {
                "dataset_path": "dataset/forecasting/ETTh1.csv",
                "data_name_list": ["ETTh1.csv"],
                "target_columns": ["OT"],
            },
            "evaluation_config": {"strategy_args": {"horizon": 96}},
            "model_config": {"recommend_model_hyper_params": {"seq_len": 96}},
        },
    )
    _write_json(
        knowledge_dir / "baseline_diagnosis.json",
        {
            "status": "completed",
            "baseline_model": "Autoformer",
            "baseline_metrics": {"mse_norm": 0.42, "mae_norm": 0.31, "wape": 9.9},
            "active_baseline_after_diagnosis": {
                "candidate_id": "baseline_autoformer",
                "display_name": "Autoformer",
                "metrics": {"mse_norm": 0.42, "mae_norm": 0.31, "wape": 9.9},
                "model_config": {"model_name": "Autoformer"},
            },
            "planned_ablation_count": 1,
            "executed_ablation_count": 1,
            "usable_ablation_count": 1,
            "failed_ablation_count": 0,
            "mechanism_understanding": {
                "schema_version": "active_path_summary_ablation_v1",
                "target_plan": {
                    "ablation_questions": [
                        {"mechanism_id": "series_decomposition", "mechanism_name": "series decomposition"}
                    ]
                },
            },
        },
    )
    _write_json(
        knowledge_dir / "baseline_search_state.json",
        {
            "status": "completed",
            "strategy": "auto",
            "candidate_order": [
                {"index": 1, "node_id": "baseline_001_Linear", "model_key": "Linear"},
                {"index": 2, "node_id": "baseline_002_Autoformer", "model_key": "Autoformer"},
                {"index": 3, "node_id": "baseline_003_TimesNet", "model_key": "TimesNet"},
            ],
            "run_results": [
                {"node_id": "baseline_001_Linear", "model_key": "Linear", "family": "linear", "status": "success", "metrics": {"mse_norm": 0.46}, "elapsed_seconds": 8.0},
                {"node_id": "baseline_002_Autoformer", "model_key": "Autoformer", "family": "transformer", "status": "success", "metrics": {"mse_norm": 0.42}, "elapsed_seconds": 12.0},
                {"node_id": "baseline_003_TimesNet", "model_key": "TimesNet", "family": "cnn", "status": "success", "metrics": {"mse_norm": 0.44}, "elapsed_seconds": 10.0},
            ],
        },
    )
    _write_json(
        dataset_profile_path(str(base_dir), task_id),
        {
            "schema_version": "dataset_profile_v1",
            "task_id": task_id,
            "status": "ok",
            "characteristics_engine": "python",
            "basic_facts": {
                "dataset_name": "ETTh1",
                "dataset_path": "dataset/forecasting/ETTh1.csv",
                "task_mode": "MS",
                "frequency": "hourly",
                "seq_len": 96,
                "horizon": 96,
                "num_variables": 7,
                "train_shape": [17420, 7],
                "missing_ratio": 0.0,
                "is_univariate": False,
            },
            "raw_characteristics": {
                "Correlation": 0.32,
                "Transition": 0.79,
                "Shifting": 0.09,
                "Seasonality": 0.61,
                "Trend": 0.81,
                "Stationarity": 0.006,
                "Short_term_jsd": 0.07,
                "Long_term_jsd": 0.04,
            },
            "derived_claims": [
                {
                    "claim": "strong_trend",
                    "confidence": 0.81,
                    "evidence": "Trend=0.81",
                    "research_implication": "Trend-aware mechanisms may be valuable.",
                }
            ],
            "research_implications": [
                {
                    "topic": "trend_aware_modeling",
                    "priority": "high",
                    "confidence": 0.81,
                    "reason": "Trend-aware mechanisms may be valuable.",
                }
            ],
            "llm_narrative": {
                "status": "ok",
                "dataset_summary": "ETTh1 shows strong trend evidence.",
                "research_interpretation": "Trend-aware modeling should be considered.",
                "suggested_opportunity_themes": [],
                "limitations": [],
            },
            "diagnostics": [],
        },
    )
    ablation_dir = knowledge_dir / "rounds" / "Ablation001"
    _write_json(
        ablation_dir / "ablation_record.json",
        {
            "target_id": "q001",
            "mechanism_id": "series_decomposition",
            "mechanism_name": "series decomposition",
            "status": "success",
            "usable_evidence_status": "usable_evidence",
            "exact_edit_intent": "Remove decomposition branch.",
            "metrics": {"mse_norm": 0.5, "mae_norm": 0.34, "wape": 8.8},
            "metric_delta": {"mse_norm": 0.08, "wape": -1.1},
        },
    )
    _write_json(
        ablation_dir / "exact_edits.json",
        {
            "edits": [
                {
                    "target_file": "ts_benchmark/baselines/autoformer/models/Autoformer.py",
                    "old_string": "seasonal, trend = self.decomp(x)",
                    "new_string": "seasonal, trend = x, 0",
                    "intent": "Disable decomposition.",
                }
            ]
        },
    )
    _write_json(ablation_dir / "experiment_result.json", {"status": "success", "success": True})
    seed_eval_detail = {
        "status": "ok",
        "valid_metric_seeds": 3,
        "successful_seeds": 3,
        "mean_metrics": {"mse_norm": 0.49, "mae_norm": 0.33},
        "per_seed": [
            {"seed": 2021, "metrics": {"mse_norm": 0.48, "mae_norm": 0.32}},
            {"seed": 2022, "metrics": {"mse_norm": 0.50, "mae_norm": 0.34}},
            {"seed": 2023, "metrics": {"mse_norm": 0.49, "mae_norm": 0.33}},
        ],
    }
    seed_eval_detail_path = knowledge_dir / "rounds" / "Ablation001" / "seed_eval_detail.json"
    _write_json(seed_eval_detail_path, seed_eval_detail)
    _write_json(
        ablation_dir / "seed_eval_result.json",
        {
            "status": "ok",
            "success": True,
            "valid_metric_seeds": 3,
            "successful_seeds": 3,
            "mean": 0.49,
            "std": 0.008164965809277268,
            "result_path": str(seed_eval_detail_path),
        },
    )
    _write_json(
        knowledge_dir / "rounds" / "Research001" / "round.json",
        {
            "research_id": "Research001",
            "round_id": 1,
            "status": "completed",
            "idea": "Tune decomposition interaction.",
            "metrics": {"mse_norm": 0.39},
            "seed_eval": {"status": "completed", "mean": 0.39, "std": 0.02, "valid_metric_seeds": 3},
            "decision": "accepted",
            "proposal": {
                "opportunity_id": "opp_autoformer_001_decomp",
                "research_question": "Can trend-aware decomposition refinement improve ETTh1 forecasting?",
                "mechanism_id": "series_decomposition",
                "primary_fit_point": "self.model.decomp",
                "linked_evidence": ["dataset_strong_trend", "ablation_series_decomposition", "structure_decomp"],
                "source_influence": {"dataset": 0.35, "ablation": 0.25, "model_structure": 0.3, "current_best": 0.1},
                "selection_reasons": ["aligned with trend evidence", "uses usable ablation evidence"],
                "risks": ["seed evidence remains limited"],
                "context_reasoning": {
                    "dataset": "Strong trend motivates decomposition refinement.",
                    "ablation": "Removing decomposition hurt metrics.",
                    "model_structure": "self.model.decomp is the source-grounded anchor.",
                },
                "autonomy_reasoning": "The gate design is selected inside the decomposition opportunity.",
                "why_not_merely_repeat_previous_work": "It refines the mechanism rather than repeating the removal ablation.",
            },
        },
    )
    append_cost_event(
        str(base_dir),
        task_id,
        {
            "kind": "llm_api",
            "stage": "designer",
            "round": 1,
            "status": "success",
            "cost": {},
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "total_tokens": 125,
                "prompt_cache_hit_tokens": 40,
                "prompt_cache_miss_tokens": 60,
                "completion_tokens_details": {"reasoning_tokens": 10},
            },
        },
    )
    append_cost_event(
        str(base_dir),
        task_id,
        {
            "kind": "llm_api",
            "stage": "ablation_repair",
            "round": 1,
            "status": "success",
            "cost": {},
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        },
    )
    (knowledge_dir / "candidates.jsonl").write_text(
        json.dumps(
            {
                "candidate_id": "Research001",
                "display_name": "Autoformer_research001",
                "metrics": {"mse_norm": 0.39},
                "is_current_best": True,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    # Explicit one-time migration models an upgrade boundary.  Runtime report
    # reads must not subsequently import edits to these legacy fixtures.
    domain_state_path(str(base_dir), task_id).unlink(missing_ok=True)
    load_domain_state(str(base_dir), task_id)
    return task_id


def test_review_status_uses_current_best_promotion_only() -> None:
    fact_pack = {
        "task_config": {"objective_metric": "mse_norm", "language": "zh"},
        "research_rounds": [
            {
                "id": "Research001",
                "status": "completed",
                "metrics": {"mse_norm": 0.9},
                "decision": "accept",
                "seed_eval": {"status": "completed", "significance_decision": {"decision": "accept"}, "promoted_to_current_best": False},
            },
            {
                "id": "Research002",
                "status": "rejected",
                "metrics": {"mse_norm": 0.8},
                "seed_eval": {"status": "completed", "significance_decision": {"decision": "accept"}, "promoted_to_current_best": True},
            },
        ],
    }

    assert _promotion_outcome_status(fact_pack["research_rounds"][0], lang="zh") == "已完成"
    assert _promotion_outcome_status(fact_pack["research_rounds"][1], lang="zh") == "已拒绝"
    assert _round_success_failed_counts(fact_pack) == (2, 0)


class FakeNarrativeClient:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def call_json(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(dict(kwargs))
        if kwargs.get("stage") == "review_report_round":
            payload = json.loads(kwargs["messages"][1]["content"])
            english = "English" in kwargs["messages"][0]["content"]
            return {
                "rounds": [
                    {
                        "id": item["id"],
                        "summary": (
                            "Change: recorded mechanism edit. Result: recorded metric and gate outcome. Conclusion: retained for review."
                            if english else "改动：围绕记录中的机制完成候选改动。结果：已记录本轮指标与门控结果。结论：该轮结果已纳入实验复盘。"
                        ),
                    }
                    for item in payload.get("rounds", [])
                ]
            }
        language = "Chinese"
        try:
            user_payload = json.loads(kwargs["messages"][1]["content"])
            language = str(user_payload.get("language") or "Chinese")
        except Exception:
            language = "Chinese"
        english = language == "English"
        if english:
            return {
                "title": "Autoformer Review Report",
                "executive_summary": [{"text": "Baseline, ablation, and one formal Research round completed.", "evidence_refs": ["fact_pack.baseline.metrics"]}],
                "timeline": [
                    {"id": "Ablation001", "title": "Ablation", "summary": "Disabled the decomposition branch.", "outcome": "usable", "evidence_refs": ["fact_pack.baseline_diagnosis.status"]}
                ],
                "baseline_summary": "Autoformer baseline has usable metrics.",
                "baseline_diagnosis_summary": "The ablation produced usable evidence.",
                "ablation_narratives": [
                    {
                        "id": "Ablation001",
                        "mechanism_display": "Series decomposition",
                        "what_changed": "Disabled series decomposition.",
                        "why_it_matters": "This is one of Autoformer's core mechanisms.",
                        "code_change": "The decomposition branch was disabled in Autoformer.py.",
                        "result_interpretation": "mse_norm became worse.",
                        "review_risk": "Only one mechanism is covered.",
                        "audit_trace": "See Ablation001 round artifacts and exact edit.",
                        "evidence_refs": ["baseline_diagnosis.ablations.0"],
                    }
                ],
                "research_round_narratives": [
                    {
                        "id": "Research001",
                        "idea": "Adjust decomposition interaction.",
                        "research_question": "Can a different decomposition interaction improve forecasting?",
                        "idea_source": "The ablation showed decomposition should not be removed outright.",
                        "implementation": "Generated one formal variant.",
                        "code_change": "The variant changed the decomposition interaction code path.",
                        "result_interpretation": "mse_norm improved over baseline.",
                        "relation_to_previous": "This follows the ablation by refining rather than removing the mechanism.",
                        "review_risk": "Only one fixture round is covered.",
                        "audit_trace": "See Research001 round artifacts.",
                        "evidence_refs": ["research_rounds.0"],
                    }
                ],
                "final_architecture_summary": "The final best candidate is Research001.",
                "review_checklist": ["Check the exact edit for Ablation001"],
                "risk_notes": ["Only one fixture ablation"],
            }
        return {
            "title": "Autoformer 实验复盘",
            "executive_summary": [{"text": "完成 baseline、消融和一个正式 Research。", "evidence_refs": ["fact_pack.baseline.metrics"]}],
            "timeline": [
                {"id": "Ablation001", "title": "消融", "summary": "关闭分解分支。", "outcome": "usable", "evidence_refs": ["fact_pack.baseline_diagnosis.status"]}
            ],
            "baseline_summary": "Autoformer baseline 有可用指标，不引用 fact_pack.baseline.metrics。",
            "baseline_diagnosis_summary": "消融产出了 usable evidence。",
            "ablation_narratives": [
                {
                    "id": "Ablation001",
                    "mechanism_display": "分解机制",
                    "what_changed": "关闭 series decomposition。",
                    "why_it_matters": "这是 Autoformer 的核心机制之一。",
                    "code_change": "在 Autoformer.py 中关闭分解分支。",
                    "result_interpretation": "mse_norm 变差。内部证据 fact_pack.baseline_diagnosis.ablations[0] 不应出现在 HTML。",
                    "review_risk": "只覆盖一个机制。",
                    "audit_trace": "查看 Ablation001 的轮次产物和 exact edit。",
                    "evidence_refs": ["baseline_diagnosis.ablations.0"],
                }
            ],
            "research_round_narratives": [
                {
                    "id": "Research001",
                    "idea": "调整分解交互。",
                    "research_question": "调整分解交互是否能改善预测？",
                    "idea_source": "消融说明不能直接移除分解机制，因此转向细化该机制。",
                    "implementation": "生成一个正式变体。",
                    "code_change": "该变体修改了分解交互相关代码路径。",
                    "result_interpretation": "mse_norm 优于 baseline。",
                    "relation_to_previous": "该轮承接消融结果，选择细化而不是移除分解机制。",
                    "review_risk": "只有一个 fixture research 轮次。",
                    "audit_trace": "查看 Research001 的轮次产物。",
                    "evidence_refs": ["research_rounds.0"],
                }
            ],
            "final_architecture_summary": "最终 best candidate 是 Research001。",
            "review_checklist": ["检查 Ablation001 的 exact edit"],
            "risk_notes": ["只有一个 fixture 消融"],
        }


class FailingClient:
    def call_json(self, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("narrative unavailable")


class MissingAblationNarrativeClient:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def call_json(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(dict(kwargs))
        if kwargs.get("stage") == "review_report_supplement":
            return {
                "ablation_narratives": [
                    {
                        "id": "Ablation001",
                        "mechanism_display": "分解机制",
                        "what_changed": "补充说明：该轮关闭了分解机制。",
                        "why_it_matters": "补充说明：它检验核心机制是否必要。",
                        "code_change": "补充说明：在模型代码中关闭对应分解路径。",
                        "result_interpretation": "补充说明：指标变差，说明该机制不能轻易移除。",
                        "review_risk": "补充说明：仍需结合更多数据验证。",
                        "audit_trace": "补充说明：查看 Ablation001 的轮次产物。",
                    }
                ],
                "research_round_narratives": [],
            }
        return {
            "title": "补写测试报告",
            "executive_summary": [{"text": "完成一次报告生成。"}],
            "timeline": [],
            "baseline_summary": "基线已完成。",
            "baseline_diagnosis_summary": "诊断已完成。",
            "ablation_narratives": [],
            "research_round_narratives": [
                {
                    "id": "Research001",
                    "idea": "调整分解交互。",
                    "research_question": "调整分解交互是否能改善预测？",
                    "idea_source": "该轮来自消融后的机制细化方向。",
                    "implementation": "完成一个正式变体。",
                    "code_change": "修改分解交互相关代码路径。",
                    "result_interpretation": "结果优于基线。",
                    "relation_to_previous": "承接前面的消融证据继续深化。",
                    "review_risk": "仍需更多轮次验证。",
                    "audit_trace": "查看 Research001 的轮次产物。",
                }
            ],
            "final_architecture_summary": "最终候选已记录。",
            "review_checklist": [],
            "risk_notes": [],
        }


class WrongLanguageNarrativeClient:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def call_json(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(dict(kwargs))
        if kwargs.get("stage") == "review_report_supplement":
            return {
                "ablation_narratives": [
                    {
                        "id": "Ablation001",
                        "mechanism_display": "分解机制",
                        "what_changed": "补写：该消融关闭分解分支。",
                        "why_it_matters": "补写：分解是当前基线的重要结构。",
                        "code_change": "补写：修改 Autoformer.py 中的分解路径。",
                        "result_interpretation": "补写：指标变差，说明不能直接移除该机制。",
                        "review_risk": "补写：仍需更多数据集确认。",
                        "audit_trace": "补写：查看 Ablation001 产物。",
                    }
                ],
                "research_round_narratives": [
                    {
                        "id": "Research001",
                        "idea": "补写：调整分解交互。",
                        "research_question": "补写：细化分解交互是否能改善预测？",
                        "idea_source": "补写：该方向来自消融后对分解机制的保留判断。",
                        "implementation": "补写：生成一个正式研究变体。",
                        "code_change": "补写：修改分解交互相关代码。",
                        "result_interpretation": "补写：结果优于基线。",
                        "relation_to_previous": "补写：承接消融证据继续深化。",
                        "review_risk": "补写：仍需进一步验证。",
                        "audit_trace": "补写：查看 Research001 产物。",
                    }
                ],
            }
        return {
            "title": "Wrong Language Report",
            "executive_summary": [{"text": "This English sentence must be repaired."}],
            "timeline": [],
            "baseline_summary": "This English baseline summary must be repaired.",
            "baseline_diagnosis_summary": "This English ablation summary must be repaired.",
            "ablation_narratives": [
                {
                    "id": "Ablation001",
                    "mechanism_display": "Series decomposition",
                    "what_changed": "Disabled series decomposition.",
                    "why_it_matters": "This is a core mechanism.",
                    "code_change": "Changed Autoformer.py.",
                    "result_interpretation": "The metric became worse.",
                    "review_risk": "Only one mechanism was tested.",
                    "audit_trace": "See Ablation001 artifacts.",
                }
            ],
            "research_round_narratives": [
                {
                    "id": "Research001",
                    "idea": "Adjust decomposition interaction.",
                    "research_question": "Can decomposition interaction help?",
                    "idea_source": "The ablation motivated this idea.",
                    "implementation": "Generated one variant.",
                    "code_change": "Changed decomposition interaction.",
                    "result_interpretation": "The metric improved.",
                    "relation_to_previous": "This follows the ablation.",
                    "review_risk": "Needs more validation.",
                    "audit_trace": "See Research001 artifacts.",
                }
            ],
            "final_architecture_summary": "Final candidate recorded.",
            "review_checklist": ["Check the artifacts."],
            "risk_notes": ["Needs review."],
        }


class SingleItemSupplementClient:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def call_json(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(dict(kwargs))
        if kwargs.get("stage") != "review_report_supplement":
            return {
                "title": "单项补写测试",
                "executive_summary": [{"text": "需要逐项补写。"}],
                "timeline": [],
                "baseline_summary": "基线已记录。",
                "baseline_diagnosis_summary": "诊断已记录。",
                "ablation_narratives": [],
                "research_round_narratives": [],
                "final_architecture_summary": "最终候选已记录。",
                "review_checklist": [],
                "risk_notes": [],
            }
        payload = json.loads(kwargs["messages"][1]["content"])
        missing = payload["missing_fields"]
        if missing.get("ablations"):
            aid = next(iter(missing["ablations"]))
            return {
                "id": aid,
                "mechanism_display": "分解机制",
                "what_changed": "逐项补写：关闭分解机制。",
                "why_it_matters": "逐项补写：检验核心机制。",
                "code_change": "逐项补写：修改 Autoformer.py。",
                "result_interpretation": "逐项补写：指标变差。",
                "review_risk": "逐项补写：仍需验证。",
                "audit_trace": "逐项补写：查看消融产物。",
            }
        rid = next(iter(missing["research_rounds"]))
        return {
            "id": rid,
            "idea": "逐项补写：调整分解交互。",
            "research_question": "逐项补写：该交互是否改善预测？",
            "idea_source": "逐项补写：来自消融证据。",
            "implementation": "逐项补写：生成正式变体。",
            "code_change": "逐项补写：修改分解交互代码。",
            "result_interpretation": "逐项补写：结果优于基线。",
            "relation_to_previous": "逐项补写：承接消融。",
            "review_risk": "逐项补写：仍需验证。",
            "audit_trace": "逐项补写：查看研究产物。",
        }


def test_review_fact_pack_indexes_ablation_and_research(tmp_path: Path) -> None:
    task_id = _fixture_task(tmp_path)
    fact_pack = build_review_fact_pack(task_id=task_id, base_dir=str(tmp_path))

    assert fact_pack["task"]["task_id"] == task_id
    assert fact_pack["task"]["language"] == "zh"
    assert fact_pack["baseline"]["model"] == "Autoformer"
    assert [row["model"] for row in fact_pack["baseline_search"]["rows"]] == ["Linear", "Autoformer", "TimesNet"]
    assert [row["rank"] for row in fact_pack["baseline_search"]["rows"]] == [3, 1, 2]
    assert fact_pack["dataset_diagnosis"]["status"] == "ok"
    assert fact_pack["dataset_diagnosis"]["characteristics_engine"] == "python"
    assert fact_pack["dataset_diagnosis"]["raw_characteristics"]["Trend"] == 0.81
    assert fact_pack["dataset_diagnosis"]["derived_claims"][0]["claim"] == "strong_trend"
    assert fact_pack["token_usage"]["totals"]["calls"] == 2
    assert fact_pack["token_usage"]["totals"]["total_tokens"] == 150
    assert "wape" not in fact_pack["baseline"]["metrics"]
    assert fact_pack["baseline_diagnosis"]["usable"] == 1
    assert fact_pack["baseline_diagnosis"]["ablations"][0]["id"] == "Ablation001"
    assert fact_pack["baseline_diagnosis"]["ablations"][0]["changed_files"] == [
        "ts_benchmark/baselines/autoformer/models/Autoformer.py"
    ]
    assert fact_pack["baseline_diagnosis"]["ablations"][0]["seed_eval"]["metric_stats"]["mse_norm"]["mean"] == 0.49
    assert fact_pack["research_rounds"][0]["id"] == "Research001"
    assert fact_pack["research_rounds"][0]["opportunity"]["opportunity_id"] == "opp_autoformer_001_decomp"
    assert fact_pack["research_rounds"][0]["opportunity"]["source_influence"]["dataset"] == 0.35


def test_review_fact_pack_uses_recorded_current_best_comparison_per_round(tmp_path: Path) -> None:
    task_id = _fixture_task(tmp_path)
    knowledge_dir = task_knowledge_dir(str(tmp_path), task_id)
    _write_json(
        knowledge_dir / "rounds" / "Research002" / "round.json",
        {
            "research_id": "Research002",
            "round_id": 2,
            "status": "rejected",
            "idea": "Post-current-best local variant.",
            "metrics": {"mse_norm": 0.40},
            "gate_decision": "reject",
            "gate_event": {
                "decision": "reject",
                "objective_metric": "mse_norm",
                "candidate_value": 0.40,
                "reference_kind": "current_best",
                "reference_value": 0.39,
                "current_best_value": 0.39,
                "relative_improvement": (0.39 - 0.40) / 0.39,
                "threshold_met": False,
            },
        },
    )
    _write_json(
        knowledge_dir / "rounds" / "Research003" / "round.json",
        {
            "research_id": "Research003",
            "round_id": 3,
            "status": "completed",
            "idea": "Seed-verified successor.",
            "metrics": {"mse_norm": 0.37},
            "seed_eval": {
                "status": "completed",
                "valid_metric_seeds": 3,
                "mean": 0.38,
                "std": 0.01,
                "significance_decision": {
                    "decision": "accept",
                    "variant_mean": 0.38,
                    "reference_kind": "current_best",
                    "reference_mean": 0.39,
                    "current_best_mean": 0.39,
                    "relative_improvement": (0.39 - 0.38) / 0.39,
                },
            },
        },
    )
    for research_id in ("Research002", "Research003"):
        payload = json.loads((knowledge_dir / "rounds" / research_id / "round.json").read_text(encoding="utf-8"))
        save_round_record(str(tmp_path), task_id, payload)
    fact_pack = build_review_fact_pack(task_id=task_id, base_dir=str(tmp_path))
    by_id = {item["id"]: item for item in fact_pack["research_rounds"]}
    single_seed_objective = by_id["Research002"]["metric_comparison"]["objective"]
    seed_eval_objective = by_id["Research003"]["metric_comparison"]["objective"]

    assert single_seed_objective["reference"] == 0.39
    assert single_seed_objective["reference_source"] == "gate_event"
    assert single_seed_objective["reference_local"] is True
    assert single_seed_objective["relative_improvement"] < 0
    assert seed_eval_objective["reference"] == 0.39
    assert seed_eval_objective["reference_source"] == "seed_eval_significance"
    assert seed_eval_objective["reference_local"] is True
    assert seed_eval_objective["relative_improvement"] == (0.39 - 0.38) / 0.39


def test_review_narrative_supplements_missing_ablation_fields(tmp_path: Path) -> None:
    task_id = _fixture_task(tmp_path)
    fact_pack = build_review_fact_pack(task_id=task_id, base_dir=str(tmp_path))
    client = MissingAblationNarrativeClient()

    narrative = build_review_narrative(client=client, fact_pack=fact_pack)

    assert len(client.calls) == 5
    assert narrative["ablation_narratives"] == []


def test_review_narrative_supplements_wrong_language_round_fields(tmp_path: Path) -> None:
    task_id = _fixture_task(tmp_path, language="zh")
    fact_pack = build_review_fact_pack(task_id=task_id, base_dir=str(tmp_path))
    client = WrongLanguageNarrativeClient()

    narrative = build_review_narrative(client=client, fact_pack=fact_pack)

    assert len(client.calls) == 5
    assert narrative["ablation_narratives"] == []
    assert narrative["research_round_narratives"] == []


def test_review_narrative_accepts_single_item_supplement_payload(tmp_path: Path) -> None:
    task_id = _fixture_task(tmp_path, language="zh")
    fact_pack = build_review_fact_pack(task_id=task_id, base_dir=str(tmp_path))
    client = SingleItemSupplementClient()

    narrative = build_review_narrative(client=client, fact_pack=fact_pack)

    assert narrative["ablation_narratives"] == []
    assert narrative["research_round_narratives"] == []
    assert len(client.calls) == 5


def test_token_usage_cost_uses_provider_pricing(tmp_path: Path) -> None:
    task_id = _fixture_task(tmp_path)
    provider_dir = tmp_path / "evocast" / "configs" / "providers"
    provider_dir.mkdir(parents=True, exist_ok=True)
    (provider_dir / "minimax.yaml").write_text(
        """
provider: minimax
models: MiniMax-M3
pricing:
  currency: USD
  unit: per_1m_tokens
  service_tier: standard
  source: https://platform.minimax.io/docs/guides/pricing-paygo
  tiers:
    - max_input_tokens: 512000
      input: 0.30
      output: 1.20
      cache_read: 0.06
  service_tier_multiplier:
    standard: 1.0
""",
        encoding="utf-8",
    )

    fact_pack = build_review_fact_pack(task_id=task_id, base_dir=str(tmp_path))
    cost = fact_pack["token_usage"]["cost"]

    assert cost["configured_models"] == "MiniMax-M3"
    assert cost["by_model"][0]["model"] == "unknown"
    assert cost["cache_hit_tokens"] == 40
    assert cost["cache_miss_input_tokens"] == 80
    assert cost["output_tokens"] == 30
    assert cost["cache_hit_cost"] > 0
    assert cost["cache_miss_input_cost"] > 0
    assert cost["output_cost"] > 0


def test_token_usage_cost_uses_actual_logged_model_pricing(tmp_path: Path) -> None:
    task_id = _fixture_task(tmp_path)
    knowledge_dir = task_knowledge_dir(str(tmp_path), task_id)
    task_config = json.loads((knowledge_dir / "task_config.json").read_text(encoding="utf-8"))
    task_config["api_config"] = "providers/deepseek.yaml"
    _write_json(knowledge_dir / "task_config.json", task_config)
    provider_dir = tmp_path / "evocast" / "configs" / "providers"
    provider_dir.mkdir(parents=True, exist_ok=True)
    (provider_dir / "deepseek.yaml").write_text(
        """
provider: deepseek
models: deepseek-v4-flash
pricing:
  currency: USD
  unit: per_1m_tokens
  source: https://api-docs.deepseek.com/quick_start/pricing
  models:
    deepseek-v4-flash:
      tiers:
        - max_input_tokens:
          input: 0.14
          output: 0.28
          cache_read: 0.0028
    deepseek-v4-pro:
      tiers:
        - max_input_tokens:
          input: 0.435
          output: 0.87
          cache_read: 0.003625
""",
        encoding="utf-8",
    )
    append_cost_event(
        str(tmp_path),
        task_id,
        {
            "kind": "llm_api",
            "stage": "planner",
            "round": 1,
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "status": "success",
            "usage": {
                "prompt_tokens": 1000000,
                "completion_tokens": 1000000,
                "total_tokens": 2000000,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 1000000,
            },
        },
    )
    append_cost_event(
        str(tmp_path),
        task_id,
        {
            "kind": "llm_api",
            "stage": "planner",
            "round": 2,
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "status": "success",
            "usage": {
                "prompt_tokens": 1000000,
                "completion_tokens": 1000000,
                "total_tokens": 2000000,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 1000000,
            },
        },
    )

    fact_pack = build_review_fact_pack(task_id=task_id, base_dir=str(tmp_path))
    cost = fact_pack["token_usage"]["cost"]

    assert round(cost["total_cost"], 6) == round((0.14 + 0.28) + (0.435 + 0.87), 6)
    by_model = {item["model"]: item for item in cost["by_model"]}
    assert round(by_model["deepseek-v4-flash"]["total_cost"], 6) == 0.42
    assert round(by_model["deepseek-v4-pro"]["total_cost"], 6) == 1.305


def test_token_usage_groups_research_and_ablation_log_labels(tmp_path: Path) -> None:
    task_id = _fixture_task(tmp_path)
    provider_dir = tmp_path / "evocast" / "configs" / "providers"
    provider_dir.mkdir(parents=True, exist_ok=True)
    (provider_dir / "minimax.yaml").write_text(
        """
provider: minimax
models: MiniMax-M3
pricing:
  currency: USD
  unit: per_1m_tokens
  tiers:
    - max_input_tokens:
      input: 0.30
      output: 1.20
      cache_read: 0.06
""",
        encoding="utf-8",
    )
    for round_num, total in [(1, 11), (2, 22)]:
        append_cost_event(
            str(tmp_path),
            task_id,
            {
                "kind": "llm_api",
                "stage": "implementer_workspace",
                "round": round_num,
                "provider": "minimax",
                "model": "MiniMax-M3",
                "status": "success",
                "usage": {
                    "prompt_tokens": total,
                    "completion_tokens": 0,
                    "total_tokens": total,
                    "prompt_cache_hit_tokens": 0,
                    "prompt_cache_miss_tokens": total,
                },
            },
        )

    fact_pack = build_review_fact_pack(task_id=task_id, base_dir=str(tmp_path))
    by_round = {item["name"]: item["total_tokens"] for item in fact_pack["token_usage"]["by_round"]}

    assert by_round["1"] >= 11
    assert by_round["2"] == 22


def test_review_html_attributes_round_tokens_from_exact_log_entries() -> None:
    fact_pack = {
        "task": {"language": "zh", "objective_metric": "mse_norm"},
        "baseline": {"metrics": {"mse_norm": 0.42}},
        "baseline_diagnosis": {
            "ablations": [
                {
                    "id": "Ablation001",
                    "target_name": "patch embedding",
                    "mechanism_name": "patch embedding",
                    "status": "rejected",
                    "metrics": {"mse_norm": 0.50},
                }
            ]
        },
        "research_rounds": [
            {
                "id": "Research001",
                "round_id": 1,
                "status": "rejected",
                "display_idea": "Spectral residual block",
                "metrics": {"mse_norm": 0.41},
            }
        ],
        "token_usage": {
            "totals": {"total_tokens": 300, "prompt_tokens": 250, "completion_tokens": 50},
            "cost": {"currency": "USD", "total_cost": 0.003},
            "by_round": [
                {"name": "1", "calls": 2, "total_tokens": 300, "prompt_tokens": 250, "completion_tokens": 50},
            ],
            "by_stage": [
                {"name": "implementer_workspace", "calls": 2, "total_tokens": 300, "prompt_tokens": 250, "completion_tokens": 50},
            ],
            "entries": [
                {
                    "file": "Ablation001_step01_implementer_workspace.meta.json",
                    "round": 1,
                    "stage_label": "implementer_workspace",
                    "calls": 1,
                    "prompt_tokens": 90,
                    "completion_tokens": 10,
                    "total_tokens": 100,
                    "reasoning_tokens": 0,
                    "cost": {"currency": "USD", "total_cost": 0.001},
                },
                {
                    "file": "Research001_step01_implementer_workspace.meta.json",
                    "round": 1,
                    "stage_label": "implementer_workspace",
                    "calls": 1,
                    "prompt_tokens": 160,
                    "completion_tokens": 40,
                    "total_tokens": 200,
                    "reasoning_tokens": 0,
                    "cost": {"currency": "USD", "total_cost": 0.002},
                },
            ],
        },
    }

    html = render_review_html(
        fact_pack=fact_pack,
        narrative={"title": "归因测试", "executive_summary": [], "review_checklist": [], "risk_notes": []},
    )

    assert "未归因" not in html
    assert "100" in html
    assert "200" in html
    assert "USD 0.001000" in html
    assert "USD 0.002000" in html


def test_failed_ablation_does_not_display_experiment_metrics(tmp_path: Path) -> None:
    task_id = _fixture_task(tmp_path)
    ablation_dir = task_knowledge_dir(str(tmp_path), task_id) / "rounds" / "Ablation001"
    _write_json(
        ablation_dir / "ablation_record.json",
        {
            "target_id": "q001",
            "mechanism_id": "series_decomposition",
            "mechanism_name": "series decomposition",
            "status": "failed_ablation_round",
            "usable_evidence_status": "failed_evidence",
            "exact_edit_intent": "Remove decomposition branch.",
            "metrics": {},
            "metric_delta": {},
        },
    )
    _write_json(
        ablation_dir / "final_validity.json",
        {
            "status": "failed_ablation_round",
            "validity_status": "invalid_ablation",
            "usable_evidence_status": "failed_evidence",
        },
    )
    _write_json(
        ablation_dir / "experiment_result.json",
        {
            "status": "ok",
            "success": True,
            "metrics": {"mse_norm": 0.5, "mae_norm": 0.34},
        },
    )

    canonical = load_round_record(str(tmp_path), task_id, "Ablation001")
    canonical.update({"status": "failed_ablation_round", "usable_evidence_status": "failed_evidence", "metrics": {}})
    save_round_record(str(tmp_path), task_id, canonical)

    fact_pack = build_review_fact_pack(task_id=task_id, base_dir=str(tmp_path))
    ablation = fact_pack["baseline_diagnosis"]["ablations"][0]
    assert ablation["metrics"] == {}

    html = render_review_html(
        fact_pack=fact_pack,
        narrative={"title": "失败消融报告", "executive_summary": [], "review_checklist": [], "risk_notes": []},
    )
    assert "failed_evidence" not in html
    assert '<td class="metric-value">0.5</td>' not in html
    assert "实现失败" in html
    assert "无机制结论" in html


def test_generate_review_report_requires_llm_narrative_and_writes_html(tmp_path: Path) -> None:
    task_id = _fixture_task(tmp_path)
    client = FakeNarrativeClient()

    result = generate_review_report(task_id=task_id, base_dir=str(tmp_path), client=client)

    assert result["status"] == "ok"
    assert client.calls
    assert client.calls[0]["stage"] == "review_report_round"
    assert client.calls[-1]["stage"] == "review_report"
    html_path = Path(result["html_path"])
    assert html_path.is_file()
    assert html_path.parent.name == "agent_reports"
    html = html_path.read_text(encoding="utf-8")
    assert '<html lang="zh-CN">' in html
    assert "运行总览" in html
    assert "轮次总览" in html
    assert "轮次记录" in html
    assert "指标看板" in html
    assert "Token 消耗" in html
    assert "总 tokens" in html
    assert "150" in html
    assert "数据集特征" in html
    assert "Baseline 寻优" in html
    assert "baseline-search-best" in html
    assert "baseline-search-second" in html
    assert "消融实验" in html
    assert "审计附录" in html
    assert "执行时间线" not in html
    assert "最慢 Top 5" not in html
    assert "最贵 Top 5" not in html
    assert "token Top 5" not in html
    assert "研究机会" not in html
    assert "证据来源影响" not in html
    html = html_path.read_text(encoding="utf-8")
    assert "EvoCast 运行审计报告" in html
    assert "数据集诊断" in html
    assert "审计附录" in html
    assert "Trend-aware mechanisms may be valuable." not in html
    assert "Tune decomposition interaction." not in html
    assert "Ablation001" in html
    assert "<th>mse_norm</th><th>mae_norm</th><th>rmse_norm</th><th>mse</th><th>mae</th><th>rmse</th>" in html
    assert "0.49 ± 0.00816497" in html
    assert "0.39 ± 0.02" in html
    assert html.count("<th>变化量</th>") >= 2
    assert html.count("<th>相对 current_best</th>") >= 2
    assert "<td class=\"metric-name\">mse_norm</td>" not in html
    assert "Code Changes" not in html
    assert "edit-card" not in html
    assert "wape" not in html
    assert "fact_pack" not in html
    assert "evidence_refs" not in html
    assert "Evidence:" not in html
    assert "LLM narrative JSON" not in html
    assert "Research001" in html
    assert "改动：围绕记录中的机制完成候选改动。" in html
    assert "结果：已记录本轮指标与门控结果。" in html
    assert "结论：该轮结果已纳入实验复盘。" in html
    assert Path(result["locale_validation_path"]).is_file()
    system_prompt = client.calls[-1]["messages"][0]["content"]
    user_payload = json.loads(client.calls[-1]["messages"][1]["content"])
    assert "Write user-facing text in Chinese" in system_prompt
    assert user_payload["language"] == "Chinese"


def test_generate_review_report_uses_english_when_task_language_is_en(tmp_path: Path) -> None:
    task_id = _fixture_task(tmp_path, language="en")
    client = FakeNarrativeClient()

    result = generate_review_report(task_id=task_id, base_dir=str(tmp_path), client=client)

    assert result["status"] == "ok"
    html = Path(result["html_path"]).read_text(encoding="utf-8")
    assert '<html lang="en">' in html
    assert "Autoformer Review Report" in html
    assert "Run Overview" in html
    assert "Round overview" in html
    assert "Round records" in html
    assert "Metric Dashboard" in html
    assert "Token Usage" in html
    assert "Dataset Characteristics" in html
    assert "Ablations" in html
    assert "Audit appendix" in html
    assert "Research Opportunity" not in html
    assert "Source Influence" not in html
    assert "数据集诊断" not in html
    system_prompt = client.calls[-1]["messages"][0]["content"]
    user_payload = json.loads(client.calls[-1]["messages"][1]["content"])
    assert "Write user-facing text in English" in system_prompt
    assert user_payload["language"] == "English"


def test_review_report_failure_writes_error_without_success_html(tmp_path: Path) -> None:
    task_id = _fixture_task(tmp_path)

    result = generate_review_report(task_id=task_id, base_dir=str(tmp_path), client=FailingClient())

    assert result["status"] == "error"
    assert Path(result["error_path"]).is_file()
    assert not (task_knowledge_dir(str(tmp_path), task_id) / "review_report.html").exists()


def test_review_html_escapes_narrative_text() -> None:
    html = render_review_html(
        fact_pack={
            "task": {"task_id": "x", "dataset": "<dataset>", "objective_metric": "mse_norm", "build_mode": True, "language": "en"},
            "baseline": {"model": "A", "metrics": {"mse_norm": 1.0}},
            "baseline_diagnosis": {"ablations": []},
            "research_rounds": [],
            "final_state": {"final_best": {}},
            "artifacts": [],
        },
        narrative={
            "title": "<script>alert(1)</script>",
            "executive_summary": [{"text": "<b>unsafe</b>", "evidence_refs": []}],
            "review_checklist": [],
            "risk_notes": [],
        },
    )

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "Run Overview" in html


def test_review_html_renders_module_research_panel() -> None:
    html = render_review_html(
        fact_pack={
            "task": {"task_id": "x", "dataset": "d", "objective_metric": "mse_norm", "build_mode": True, "language": "en"},
            "baseline": {"model": "PDF", "metrics": {"mse_norm": 1.0}},
            "baseline_diagnosis": {"status": "skipped", "planned": 0, "executed": 0, "usable": 0, "failed": 0, "ablations": []},
            "research_rounds": [
                {
                    "id": "Research001",
                    "status": "completed",
                    "idea": "state memory",
                    "architecture_module_spec": {
                        "module_name": "TemporalStateGate",
                        "module_family": "state_adaptive_embedding",
                        "exploration_scale": "module_level",
                        "internal_components": [{"name": "StateEncoder"}, {"name": "StateProjection"}, {"name": "GatingMul"}],
                        "controls": [],
                    },
                    "module_manifest": {
                        "module_name": "TemporalStateGate",
                        "module_family": "state_adaptive_embedding",
                    },
                    "module_validity": {
                        "status": "module_valid",
                        "component_traces": {
                            "StateEncoder": {"runtime_path": "model.state_gate.StateEncoder", "called": True, "output_norm": 1.25, "has_grad": True},
                        },
                    },
                    "module_experiment_decision": {
                        "metric_signal": "positive_single_seed",
                        "control_status": "not_required",
                        "promotion_decision": "needs_seed_eval",
                        "module_family_decision": "evolve_module_family",
                        "evaluation_budget": "build_mode",
                        "next_required": [],
                    },
                    "module_family_memory_reference": {
                        "referenced_family_ids": ["state_adaptive_convolution"],
                        "referenced_previous_decisions": ["evolve_module_family"],
                        "referenced_tested_modules": ["StateAdaptiveConv"],
                        "next_required_addressed": ["evolve state-aware embedding gate"],
                    },
                    "metrics": {"mse_norm": 0.9},
                }
            ],
            "final_state": {"final_best": {}},
            "artifacts": [],
        },
        narrative={"title": "Module Report", "executive_summary": [], "review_checklist": [], "risk_notes": []},
    )

    assert "Round records" in html
    assert "TemporalStateGate" in html
    assert "state_adaptive_embedding" in html
    assert "module_valid" in html
    assert "ablate_internal_components" not in html
    assert "Ablation switches" not in html
    assert "Round overview" in html
    assert "ablate use_state_gate" not in html


def test_review_html_uses_same_seed_baseline_for_quick_relative_delta() -> None:
    html = render_review_html(
        fact_pack={
            "task": {
                "task_id": "x",
                "dataset": "d",
                "objective_metric": "mse_norm",
                "build_mode": True,
                "language": "zh",
            },
            "baseline": {
                "model": "FiLM",
                "metrics": {"mse_norm": 0.966374454520606},
                "seed_eval": {
                    "status": "completed",
                    "metric_stats": {
                        "mse_norm": {
                            "mean": 0.9677921748670294,
                            "std": 0.018551058614950225,
                            "seed_count": 3,
                        }
                    },
                },
            },
            "baseline_diagnosis": {"status": "skipped", "planned": 0, "executed": 0, "usable": 0, "failed": 0, "ablations": []},
            "research_rounds": [
                {
                    "id": "Research001",
                    "status": "rejected",
                    "architecture_module_spec": {"module_name": "FrequencyGatingModule"},
                    "module_experiment_decision": {"promotion_decision": "reject", "evaluation_budget": "build_mode"},
                    "metrics": {"mse_norm": 1.02605437749731},
                }
            ],
            "final_state": {"final_best": {}},
            "artifacts": [],
        },
        narrative={"title": "报告", "executive_summary": [], "review_checklist": [], "risk_notes": []},
    )

    assert '<td class="metric-value">0.966374</td>' in html
    assert '<td class="metric-value">退化 6.18%</td>' in html
    assert '<td class="metric-value">-6.02%</td>' not in html


def test_chinese_report_replaces_raw_english_fact_prose(tmp_path: Path) -> None:
    task_id = _fixture_task(tmp_path, language="zh")
    result = generate_review_report(task_id=task_id, base_dir=str(tmp_path), client=FakeNarrativeClient())

    assert result["status"] == "ok"
    html = Path(result["html_path"]).read_text(encoding="utf-8")
    assert "The dataset" not in html
    assert "Trend-aware mechanisms may be valuable." not in html
    assert "Can trend-aware decomposition refinement improve" not in html
    assert "Strong trend motivates decomposition refinement." not in html
    assert "运行总览" in html
    assert "指标看板" in html
    assert "审计附录" in html
    validation = json.loads(Path(result["locale_validation_path"]).read_text(encoding="utf-8"))
    assert validation["language"] == "zh"
    assert validation["status"] == "fallback_applied"


def test_review_html_hides_internal_evidence_fields() -> None:
    html = render_review_html(
        fact_pack={
            "task": {"task_id": "x", "dataset": "d", "objective_metric": "mse_norm", "build_mode": True},
            "baseline": {"model": "A", "metrics": {"mse_norm": 1.0}},
            "baseline_diagnosis": {"status": "completed", "planned": 0, "executed": 0, "usable": 0, "failed": 0, "ablations": []},
            "research_rounds": [],
            "final_state": {"final_best": {}},
            "artifacts": [{"name": "api.parsed.json", "path": "runs/x/logs/api/api.parsed.json", "size_bytes": 10}],
        },
        narrative={
            "title": "报告",
            "executive_summary": [{"text": "根据 fact_pack.baseline.metrics 得出结论。", "evidence_refs": ["fact_pack.x"]}],
            "timeline": [{"id": "t1", "title": "阶段", "summary": "参考 fact_pack.baseline_diagnosis.status", "outcome": "完成", "evidence_refs": ["fact_pack.y"]}],
            "baseline_summary": "不展示 evidence_refs。",
            "baseline_diagnosis_summary": "",
            "review_checklist": ["不要展示 fact_pack.baseline.metrics"],
            "risk_notes": [],
        },
    )

    assert "fact_pack" not in html
    assert "evidence_refs" not in html
    assert "Evidence:" not in html
    assert "LLM narrative JSON" not in html
    assert "api.parsed.json" not in html


def test_review_report_reads_build_contract_style_round_artifacts(tmp_path: Path) -> None:
    task_id = "review_report_new_arch_fixture"
    knowledge_dir = task_knowledge_dir(str(tmp_path), task_id)
    _write_json(
        knowledge_dir / "task_config.json",
        {
            "task_id": task_id,
            "dataset_path": "dataset/forecasting/ETTh1.csv",
            "target_columns": ["OT"],
            "objective_metric": "mse_norm",
            "horizon": 96,
            "seq_len": 96,
            "build_mode": True,
            "language": "zh",
        },
    )
    _write_json(
        knowledge_dir / "compiled_config.json",
        {
            "data_config": {"dataset_path": "dataset/forecasting/ETTh1.csv", "target_columns": ["OT"]},
            "evaluation_config": {"strategy_args": {"horizon": 96}},
            "model_config": {"model_name": "CrossLinear", "model_hyper_params": {"seq_len": 96, "pred_len": 96}},
        },
    )
    _write_json(
        knowledge_dir / "baseline_diagnosis.json",
        {
            "status": "completed",
            "baseline_model": "CrossLinear",
            "baseline_metrics": {"mse_norm": 1.0, "mae_norm": 0.5},
            "active_baseline_after_diagnosis": {
                "candidate_id": "baseline_crosslinear",
                "display_name": "CrossLinear",
                "metrics": {"mse_norm": 1.0, "mae_norm": 0.5},
                "model_config": {"model_name": "CrossLinear"},
            },
            "planned_ablation_count": 1,
            "executed_ablation_count": 1,
            "usable_ablation_count": 1,
            "failed_ablation_count": 0,
        },
    )

    ablation_dir = knowledge_dir / "rounds" / "Ablation001"
    _write_json(
        ablation_dir / "round.json",
        {
            "research_id": "Ablation001",
            "status": "rejected",
            "display_idea": "CrossLinear | Ablate corr embed Conv1d",
            "mechanism_id": "abl_corr_embed",
            "mechanism_summary": "Remove the correlation embedding branch.",
            "hypothesis": "If correlation embedding matters, removing it should hurt.",
            "metrics": {"mse_norm": 1.12, "mae_norm": 0.61},
            "changed_files": ["ts_benchmark/baselines/crosslinear/model/crosslinear_model.py"],
        },
    )
    _write_json(
        ablation_dir / "build_contract.json",
        {
            "name": "Ablation001",
            "semantic_goal": "Ablate corr embed Conv1d.",
            "hypothesis": "Correlation embedding may be a core CrossLinear component.",
            "allowed_edit_files": ["ts_benchmark/baselines/crosslinear/model/crosslinear_model.py"],
            "metric_protocol": {"mechanism_id": "abl_corr_embed", "evaluation_stage": "build_mode"},
        },
    )
    _write_json(
        ablation_dir / "build_outcome.json",
        {
            "status": "completed",
            "candidate_snapshot_id": "snap-abc123",
            "candidate_checkout": "rounds/Ablation001/candidate_01/source",
            "patch_path": "build_attempt_01/agent.patch",
            "changed_files": ["ts_benchmark/baselines/crosslinear/model/crosslinear_model.py"],
        },
    )
    _write_json(ablation_dir / "metric_result.json", {"status": "ok", "metrics": {"mse_norm": 1.12, "mae_norm": 0.61}})

    research_dir = knowledge_dir / "rounds" / "Research001"
    direction = {
        "planner": "research_program_hypothesis_competition",
        "round_role": "explore",
        "terminal_display_title": "Add adaptive channel affine",
        "target_mechanism": "learnable_channel_affine",
        "target_code_region": "ts_benchmark/baselines/crosslinear/model/crosslinear_model.py::CrossLinear",
        "hypothesis": "A channel affine adapter can correct per-channel scale mismatch.",
        "what_this_round_will_teach": "Whether lightweight channel adaptation is useful.",
        "why_this_role_now": "Ablation suggests core linear path still leaves channel bias.",
        "failure_to_avoid": "Do not repeat rejected ablation removal.",
    }
    _write_json(
        research_dir / "round.json",
        {
            "research_id": "Research001",
            "round_id": 1,
            "status": "completed",
            "display_idea": "Add adaptive channel affine",
            "idea_summary": "Add a small channel affine adapter.",
            "mechanism_summary": "Learnable per-channel scale and bias after CrossLinear projection.",
            "metrics": {"mse_norm": 0.97, "mae_norm": 0.45},
            "changed_files": ["ts_benchmark/baselines/crosslinear/model/crosslinear_model.py"],
            "gate_decision": "accepted",
        },
    )
    _write_json(research_dir / "research_direction.json", direction)
    _write_json(
        research_dir / "build_contract.json",
        {
            "name": "Research001",
            "semantic_goal": "Implement adaptive channel affine.",
            "hypothesis": direction["hypothesis"],
            "allowed_edit_files": ["ts_benchmark/baselines/crosslinear/model/crosslinear_model.py"],
            "metric_protocol": {"research_direction": direction},
        },
    )
    _write_json(
        research_dir / "build_outcome.json",
        {
            "status": "completed",
            "candidate_snapshot_id": "snap-def456",
            "candidate_checkout": "rounds/Research001/candidate_01/source",
            "patch_path": "build_attempt_01/agent.patch",
            "changed_files": ["ts_benchmark/baselines/crosslinear/model/crosslinear_model.py"],
        },
    )
    _write_json(research_dir / "metric_result.json", {"status": "ok", "metrics": {"mse_norm": 0.97, "mae_norm": 0.45}})

    fact_pack = build_review_fact_pack(task_id=task_id, base_dir=str(tmp_path))
    html = render_review_html(
        fact_pack=fact_pack,
        narrative={"title": "报告", "executive_summary": [], "review_checklist": [], "risk_notes": []},
    )

    ablation = fact_pack["baseline_diagnosis"]["ablations"][0]
    research = fact_pack["research_rounds"][0]
    assert ablation["display_idea"] == "CrossLinear | Ablate corr embed Conv1d"
    assert ablation["metrics"]["mse_norm"] == 1.12
    assert research["display_idea"] == "Add adaptive channel affine"
    assert research["planner"] == "research_program_hypothesis_competition"
    assert research["round_role"] == "explore"
    assert research["opportunity"]["mechanism_id"] == "learnable_channel_affine"
    assert "crosslinear_model.py::CrossLinear" in research["target_code_region"]
    assert "CrossLinear | Ablate corr embed Conv1d" in html
    assert "Add adaptive channel affine" in html
    assert "Add adaptive channel affine" in html
    assert "改动位置" not in html
    assert "轮次 tokens" in html
    assert "轮次费用" in html
    assert "机制（）" not in html
    assert '<div class="fact-value"></div>' not in html
    assert "<td></td>" not in html
    assert "runtime_event" not in html
