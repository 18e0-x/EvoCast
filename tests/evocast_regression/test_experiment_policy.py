from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from evocast.policy.agent_control_policy import (
    build_mode_policy,
    context_policy,
    gate_policy,
    protocol_policy,
    round_control_policy,
)
from evocast.domain.config_paths import resolve_config_path
from evocast.harness.session import AgentSession
from evocast.harness.rounds import record_phase_transition, start_round
from evocast.build.contract import BuildContract
from evocast.policy.experiment_policy import (
    DEFAULT_POLICY,
    baseline_diagnosis_policy,
    baseline_search_policy,
    fixed_seed_list,
    baseline_seed,
    training_policy_for,
    normalize_budget,
)
from evocast.build.metric_runner import TFBExperimentMetricRunner
from evocast.domain.knowledge_paths import runtime_root
from evocast.build.contract_compiler import build_research_contract
from evocast.build.source_snapshot import source_manifest
from evocast.research.model_registry import build_registry
from evocast.runners.seed_runner import run_seed_evaluation
from evocast.scripts import wizard
from evocast.scripts.wizard import build_arg_parser, validate_global_args
from evocast.tools.tfb_ablation import run_ablation
from evocast.state.runtime.store import mark_task_interrupted, sync_best_baseline
from evocast.state.domain_store import load_task_config, save_build_contract, save_task_config


def _write_resume_task_config(tmp_path: Path, task_id: str) -> None:
    save_task_config(
        str(tmp_path),
        task_id,
        {
            "task_id": task_id,
            "language": "zh",
            "api_config": "providers/minimax.yaml",
            "objective_metric": "mse_norm",
            "max_rounds": 20,
            "build_mode": False,
            "research_intent": "Preserve a long-horizon seasonal signal.",
        },
    )


class _NoopTerminalUI:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def start(self):
        return self

    def stop(self) -> None:
        pass


def test_resume_after_research_skips_prelude_and_starts_next_research(monkeypatch, tmp_path: Path) -> None:
    task_id = "resume_after_research"
    _write_resume_task_config(tmp_path, task_id)
    records = [
        {"round_id": index, "research_id": f"Research{index:03d}", "status": "completed", "round_scope": "research"}
        for index in range(1, 7)
    ]
    generated: list[str] = []
    agent_calls: list[list[str]] = []
    progress_calls = 0

    monkeypatch.setattr(wizard, "runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(wizard, "TerminalUI", _NoopTerminalUI)
    monkeypatch.setattr(wizard.round_records, "list_rounds", lambda *_args: records)
    monkeypatch.setattr(wizard.round_records, "current_round", lambda *_args: None)
    monkeypatch.setattr(wizard.round_records, "_round_counts_toward_research_budget", lambda record: record["research_id"].startswith("Research"))

    def fake_progress(*_args):
        nonlocal progress_calls
        progress_calls += 1
        return {"research_open_rounds": 0, "terminal_rounds": 6 if progress_calls <= 2 else 20, "completed_rounds": 20}

    monkeypatch.setattr(wizard, "round_progress", fake_progress)
    monkeypatch.setattr(
        wizard,
        "_ensure_research_build_contract",
        lambda **kwargs: generated.append(kwargs["task_id"]) or "Research007",
    )
    monkeypatch.setattr(wizard, "run_agent_v3_main", lambda argv: agent_calls.append(argv) or 0)
    monkeypatch.setattr(wizard, "build_dashboard", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(wizard, "sync_task_stage", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wizard, "record_runtime_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wizard, "generate_review_report", lambda **_kwargs: {"status": "ok"})
    monkeypatch.setattr(wizard, "init_task_from_wizard", lambda **_kwargs: pytest.fail("resume must not initialize task"))
    monkeypatch.setattr(wizard, "_baseline_flow_service", lambda: pytest.fail("resume must not run baseline"))
    monkeypatch.setattr(wizard, "run_baseline_diagnosis_nonblocking_before_agent", lambda **_kwargs: pytest.fail("resume must not run ablation"))

    args = wizard.build_arg_parser().parse_args(["--resume", "--task-id", task_id, "--yes"])
    result = wizard.run_wizard(args)

    assert result["status"] == "resumed"
    assert generated == [task_id]
    assert agent_calls[0][agent_calls[0].index("--research-id") + 1] == "Research007"
    assert "providers/minimax.yaml" in agent_calls[0]


def test_interrupted_build_resumes_with_the_persisted_research_intent(monkeypatch, tmp_path: Path) -> None:
    task_id = "resume_persisted_intent"
    _write_resume_task_config(tmp_path, task_id)
    task_config = load_task_config(str(tmp_path), task_id)
    task_config["research_intent"] = "Preserve a long-horizon seasonal signal."
    save_task_config(str(tmp_path), task_id, task_config)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "model.py").write_text("class Model:\n    pass\n", encoding="utf-8")
    manifest = source_manifest(repo)
    source_binding = {"entry_file": "model.py", "source_files": ["model.py"], "verified": True}
    baseline = {
        "candidate_id": "baseline_fixture",
        "display_name": "FixtureModel",
        "model_name": "FixtureModel",
        "import_path": "fixture.Model",
        "metrics": {"mse_norm": 1.0},
        "source_binding": source_binding,
        "source_ref": {
            "kind": "source_snapshot",
            "candidate_snapshot_id": manifest["snapshot_id"],
            "base_snapshot_id": manifest["snapshot_id"],
            "source_checkout": str(repo),
            "source_manifest_hash": manifest["manifest_hash"],
            "source_binding": source_binding,
        },
    }
    sync_best_baseline(str(tmp_path), task_id, baseline)
    contract = build_research_contract(
        base_dir=str(tmp_path), task_id=task_id, baseline=baseline,
        objective_metric="mse_norm", repo_dir=repo, research_id="Research001",
    )
    record = start_round(
        base_dir=str(tmp_path), task_id=task_id, fit_point="fixture.Model",
        hypothesis="fixture hypothesis", evidence_source="fixture",
    )
    record_phase_transition(base_dir=str(tmp_path), task_id=task_id, phase="build")
    save_build_contract(str(tmp_path), task_id, str(record["research_id"]), contract.to_dict())
    calls: list[list[str]] = []
    monkeypatch.setattr(wizard, "runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(wizard, "run_agent_v3_main", lambda argv: calls.append(argv) or 0)

    resumed = wizard._resume_open_round(
        task_id,
        wizard._load_existing_task_config(task_id),
        argparse.Namespace(api_config="providers/minimax.yaml"),
    )

    assert resumed is True
    assert calls[0][calls[0].index("--research-id") + 1] == "Research001"


def test_fresh_canonical_buildmode_interrupt_then_wizard_resume(monkeypatch, tmp_path: Path) -> None:
    """A new canonical task resumes an interrupted BuildMode round without legacy files."""
    task_id = "fresh_canonical_buildmode_resume"
    save_task_config(
        str(tmp_path),
        task_id,
        {
            "task_id": task_id,
            "language": "zh",
            "api_config": "providers/minimax.yaml",
            "objective_metric": "mse_norm",
            "max_rounds": 1,
            "build_mode": True,
            "research_intent": "Preserve the long-horizon seasonal signal.",
        },
    )
    record = start_round(
        base_dir=str(tmp_path), task_id=task_id, fit_point="Informer",
        hypothesis="A small residual path improves the smoke metric.", evidence_source={"kind": "test"},
    )
    record_phase_transition(base_dir=str(tmp_path), task_id=task_id, phase="build")
    save_build_contract(str(tmp_path), task_id, str(record["research_id"]), {"research_id": record["research_id"]})
    mark_task_interrupted(str(tmp_path), task_id)

    calls: list[list[str]] = []
    monkeypatch.setattr(wizard, "runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(wizard, "TerminalUI", _NoopTerminalUI)
    monkeypatch.setattr(wizard, "run_agent_v3_main", lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(wizard, "build_dashboard", lambda *_args, **_kwargs: {})

    result = wizard.run_wizard(wizard.build_arg_parser().parse_args(["--resume", "--task-id", task_id, "--yes"]))

    assert result["status"] == "resumed"
    assert not (wizard.task_knowledge_dir(str(tmp_path), task_id) / "task_config.json").exists()
    assert wizard._load_existing_task_config(task_id)["research_intent"] == "Preserve the long-horizon seasonal signal."
    assert calls[0][calls[0].index("--research-id") + 1] == "Research001"


def test_resume_closes_late_open_ablation_without_replaying_it(monkeypatch, tmp_path: Path) -> None:
    task_id = "resume_late_ablation"
    _write_resume_task_config(tmp_path, task_id)
    formal = {"round_id": 6, "research_id": "Research006", "status": "completed", "round_scope": "research"}
    ablation = {"round_id": 4, "research_id": "Ablation004", "status": "round_started", "phase": "experiment", "round_scope": "baseline_diagnosis"}
    closed: list[dict] = []

    monkeypatch.setattr(wizard, "runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(wizard, "TerminalUI", _NoopTerminalUI)
    monkeypatch.setattr(wizard.round_records, "list_rounds", lambda *_args: [formal, ablation])
    monkeypatch.setattr(wizard.round_records, "current_round", lambda *_args: ablation)
    monkeypatch.setattr(wizard.round_records, "_round_counts_toward_research_budget", lambda record: record["research_id"].startswith("Research"))
    monkeypatch.setattr(wizard.round_records, "close_current_round", lambda **kwargs: closed.append(kwargs) or ablation)
    monkeypatch.setattr(wizard, "round_progress", lambda *_args: {"research_open_rounds": 0, "terminal_rounds": 20, "completed_rounds": 20})
    monkeypatch.setattr(wizard, "build_dashboard", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(wizard, "sync_task_stage", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wizard, "record_runtime_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wizard, "generate_review_report", lambda **_kwargs: {"status": "ok"})

    args = wizard.build_arg_parser().parse_args(["--resume", "--task-id", task_id, "--yes"])
    wizard.run_wizard(args)

    assert closed == [
        {
            "base_dir": str(tmp_path),
            "task_id": task_id,
            "research_id": "Ablation004",
            "status": "infra_failed",
            "reason": "resume_process_interrupted_at_experiment",
            "extra": {
                "terminal_reason": "resume_process_interrupted",
                "resume_preflight": True,
                "interrupted_phase": "experiment",
            },
        }
    ]


def test_baseline_diagnosis_limits_live_in_experiment_policy() -> None:
    assert DEFAULT_POLICY["baseline_diagnosis"]["max_ablation_targets"] == 3
    assert DEFAULT_POLICY["baseline_diagnosis"]["ablation_repair_attempts"] == 3


def test_baseline_search_policy_exposes_only_curator_fields(tmp_path: Path) -> None:
    policy_dir = tmp_path / "configs" / "policies"
    policy_dir.mkdir(parents=True)
    (policy_dir / "experiment.yaml").write_text(
        """
baseline_search:
  candidate_count: 4
  registry_pool_size: 12
  initial_seeds: [Linear]
  preferred_families: [linear, transformer]
  seed_count: 99
  required_tags: ["#frequency"]
  target_tag_coverage: 0.9
  alpha: 0.1
  beta: 0.2
""".strip(),
        encoding="utf-8",
    )

    policy = baseline_search_policy(str(tmp_path))

    assert policy == {
        "candidate_count": 4,
        "registry_pool_size": 12,
        "initial_seeds": ["Linear"],
        "preferred_families": ["linear", "transformer"],
    }


def test_default_baseline_preferred_families_cover_registry_families() -> None:
    registry_families = {str(item.get("family") or "") for item in build_registry(verify=False)}
    configured = set(DEFAULT_POLICY["baseline_search"]["preferred_families"])

    assert registry_families <= configured


def test_runtime_training_policy_uses_formal_experiment_yaml(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    policy_dir = repo / "evocast" / "configs" / "policies"
    policy_dir.mkdir(parents=True)
    (repo / "evocast" / "core").mkdir(parents=True)
    (repo / "evocast" / "harness").mkdir(parents=True)
    (repo / "ts_benchmark").mkdir()
    (policy_dir / "experiment.yaml").write_text(
        "training:\n  experiment:\n    batch_size: 32\n    num_epochs: 10\n    patience: 3\n    lr: 0.0001\n",
        encoding="utf-8",
    )

    assert training_policy_for("experiment", str(runtime_root(str(repo)))) == {
        "batch_size": 32,
        "num_epochs": 10,
        "patience": 3,
        "lr": 0.0001,
    }


def test_wizard_ablation_target_limit_defaults_to_three_and_allows_zero() -> None:
    parser = build_arg_parser()

    default_args = parser.parse_args([])
    assert default_args.max_ablation_targets == 3

    zero_args = parser.parse_args(["--max-ablation-targets", "0"])
    validate_global_args(zero_args)
    assert zero_args.max_ablation_targets == 0


def test_wizard_cli_contract_keeps_public_options_and_unknown_argument_rejection() -> None:
    parser = build_arg_parser()
    public_dests = {
        action.dest
        for action in parser._actions
        if action.dest != "help" and action.option_strings
    }

    assert {
        "task_id", "resume", "dataset", "max_rounds", "research_intent",
        "build_mode", "start_run", "configure_only", "dry_run", "yes",
    } <= public_dests
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--unknown-wizard-option"])
    assert exc.value.code == 2


def test_wizard_ablation_target_limit_rejects_negative() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--max-ablation-targets", "-1"])

    with pytest.raises(ValueError):
        validate_global_args(args)


def test_wizard_requires_a_baseline_strategy_and_rejects_skip() -> None:
    parser = build_arg_parser()
    assert parser.parse_args([]).baseline_strategy == "auto"
    with pytest.raises(SystemExit):
        parser.parse_args(["--baseline-strategy", "skip"])


def test_manual_baseline_validation_suggests_timebridge_for_typo() -> None:
    normalized, missing = wizard.validate_manual_model_list(["TIMEBRIGH"])

    assert normalized == []
    assert missing[0]["requested"] == "TIMEBRIGH"
    assert "TimeBridge" in missing[0]["suggestions"]


def test_prompt_manual_baselines_reprompts_unknown_model(monkeypatch, capsys) -> None:
    answers = iter(["TIMEBRIGH", "TimeBridge"])
    monkeypatch.setattr(wizard, "prompt_text", lambda *_args, **_kwargs: next(answers))

    models = wizard.prompt_manual_baselines(lang="zh")

    assert models == ["TimeBridge"]
    captured = capsys.readouterr()
    assert "未找到 baseline 模型" in captured.out
    assert "TimeBridge" in captured.out


def test_prompt_manual_baselines_lists_all_models_grouped_by_family(monkeypatch, capsys) -> None:
    registry = [
        {"model_key": "LinearA", "family": "linear"},
        {"model_key": "LinearB", "family": "linear"},
        {"model_key": "TransformerA", "family": "transformer"},
        {"model_key": "GnnA", "family": "gnn"},
    ]
    monkeypatch.setattr(wizard, "build_registry", lambda verify=False: registry)
    monkeypatch.setattr(
        wizard,
        "baseline_search_policy",
        lambda _base=None: {"preferred_families": ["linear", "transformer", "gnn"]},
    )
    monkeypatch.setattr(wizard, "prompt_text", lambda *_args, **_kwargs: "TransformerA")

    models = wizard.prompt_manual_baselines(lang="zh")

    assert models == ["TransformerA"]
    captured = capsys.readouterr()
    assert "可选 baseline 模型（按 family 分组，共 4 个）" in captured.out
    assert "linear（2）：LinearA, LinearB" in captured.out
    assert "transformer（1）：TransformerA" in captured.out
    assert "gnn（1）：GnnA" in captured.out


def test_wizard_read_line_back_command_raises(monkeypatch) -> None:
    monkeypatch.setattr(wizard.os, "name", "posix")
    monkeypatch.setattr("builtins.input", lambda _prompt: ":back")

    with pytest.raises(wizard.WizardBack):
        wizard._read_line("prompt: ", lang="zh")


def test_wizard_choose_option_propagates_back(monkeypatch) -> None:
    def _raise_back(*_args, **_kwargs):
        raise wizard.WizardBack()

    monkeypatch.setattr(wizard, "_read_line", _raise_back)

    with pytest.raises(wizard.WizardBack):
        wizard.choose_option("请选择：", ["a", "b"], lang="zh")


def test_wizard_resumable_tasks_are_sorted_by_recent_update(monkeypatch, tmp_path: Path) -> None:
    import evocast.services.task_discovery as discovery_service

    knowledge_root = tmp_path / "task_knowledge"
    old_task = knowledge_root / "old_but_active"
    recent_task = knowledge_root / "recent_interrupted"
    old_task.mkdir(parents=True)
    recent_task.mkdir(parents=True)
    (old_task / "task_config.json").write_text("{}", encoding="utf-8")
    (recent_task / "task_config.json").write_text("{}", encoding="utf-8")

    old_mtime = 1_000_000
    recent_mtime = 2_000_000
    import os

    os.utime(old_task / "task_config.json", (old_mtime, old_mtime))
    os.utime(recent_task / "task_config.json", (recent_mtime, recent_mtime))

    monkeypatch.setattr(discovery_service, "task_knowledge_root", lambda _base: knowledge_root)
    monkeypatch.setattr(
        wizard,
        "resume_summary",
        lambda _base, task_id: {
            "task_id": task_id,
            "has_task_config": True,
            "has_best_baseline": task_id == "old_but_active",
            "baseline_search_status": "running" if task_id == "old_but_active" else "completed",
            "completed_rounds": [1, 2, 3] if task_id == "old_but_active" else [],
        },
    )

    tasks = wizard.discover_resumable_tasks(tmp_path)

    assert [item["task_id"] for item in tasks] == ["recent_interrupted", "old_but_active"]


def test_config_path_does_not_resolve_legacy_flat_names() -> None:
    assert resolve_config_path("deepseek_api.yaml").as_posix().endswith("configs/deepseek_api.yaml")
    assert resolve_config_path("experiment_policy.yaml").as_posix().endswith("configs/experiment_policy.yaml")
    assert resolve_config_path("model_registry_overrides.yaml").as_posix().endswith("configs/model_registry_overrides.yaml")


def test_budget_accepts_only_current_names() -> None:
    assert normalize_budget("") == "unified"
    assert normalize_budget("unified") == "unified"
    assert normalize_budget("seed_eval") == "seed_eval"
    for value in ("quick", "standard", "full", "anything_else"):
        with pytest.raises(ValueError):
            normalize_budget(value)


def test_seed_eval_explicit_yaml_seed_list_is_authoritative(tmp_path: Path) -> None:
    policy_dir = tmp_path / "configs" / "policies"
    policy_dir.mkdir(parents=True)
    (policy_dir / "experiment.yaml").write_text(
        """
seed_eval:
  seeds: [11, 17]
""".strip(),
        encoding="utf-8",
    )

    assert fixed_seed_list(str(tmp_path)) == [11, 17]


def test_baseline_seed_uses_middle_configured_seed(tmp_path: Path) -> None:
    policy_dir = tmp_path / "configs" / "policies"
    policy_dir.mkdir(parents=True)
    (policy_dir / "experiment.yaml").write_text(
        "seed_eval:\n  seeds: [11, 17, 23]\n",
        encoding="utf-8",
    )

    assert baseline_seed(str(tmp_path)) == 17


def test_experiment_policy_resolves_from_repo_and_runtime_roots(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    policy_dir = repo / "evocast" / "configs" / "policies"
    policy_dir.mkdir(parents=True)
    (repo / "evocast" / "core").mkdir(parents=True)
    (repo / "evocast" / "harness").mkdir(parents=True)
    (repo / "ts_benchmark").mkdir()
    (policy_dir / "experiment.yaml").write_text(
        """
seed_eval:
  seeds: [31, 37, 41]
""".strip(),
        encoding="utf-8",
    )

    assert fixed_seed_list(str(repo)) == [31, 37, 41]
    assert fixed_seed_list(str(runtime_root(str(repo)))) == [31, 37, 41]


def test_build_metric_runner_default_seed_uses_policy_middle_seed(tmp_path: Path) -> None:
    policy_dir = tmp_path / "configs" / "policies"
    policy_dir.mkdir(parents=True)
    (policy_dir / "experiment.yaml").write_text(
        """
seed_eval:
  seeds: [43, 47, 53]
""".strip(),
        encoding="utf-8",
    )
    session = AgentSession(task_id="seed_policy_task", base_dir=str(tmp_path), client=None)  # type: ignore[arg-type]

    runner = TFBExperimentMetricRunner(session=session)

    assert runner.seed == 47


def test_seed_eval_legacy_count_fallback_does_not_force_three(tmp_path: Path) -> None:
    policy_dir = tmp_path / "configs" / "policies"
    policy_dir.mkdir(parents=True)
    (policy_dir / "experiment.yaml").write_text(
        """
seed_eval:
  seeds: []
  base_seed: 7
  num_seeds: 2
""".strip(),
        encoding="utf-8",
    )

    assert fixed_seed_list(str(tmp_path)) == [7, 8]


def test_baseline_diagnosis_yaml_controls_targets_and_repair_attempts(tmp_path: Path) -> None:
    policy_dir = tmp_path / "configs" / "policies"
    policy_dir.mkdir(parents=True)
    (policy_dir / "experiment.yaml").write_text(
        """
baseline_diagnosis:
  max_ablation_targets: 5
  ablation_repair_attempts: 4
  ablation_effect_threshold: 0.03
""".strip(),
        encoding="utf-8",
    )

    policy = baseline_diagnosis_policy(str(tmp_path))
    assert policy["max_ablation_targets"] == 5
    assert policy["ablation_repair_attempts"] == 4
    assert policy["ablation_effect_threshold"] == 0.03


def test_run_ablation_uses_yaml_repair_attempt_default(tmp_path: Path, monkeypatch) -> None:
    policy_dir = tmp_path / "configs" / "policies"
    policy_dir.mkdir(parents=True)
    (policy_dir / "experiment.yaml").write_text(
        """
baseline_diagnosis:
  ablation_repair_attempts: 4
  ablation_effect_threshold: 0.02
""".strip(),
        encoding="utf-8",
    )
    session = AgentSession(task_id="ablation_policy_task", base_dir=str(tmp_path), client=object())  # type: ignore[arg-type]
    session.ensure_dirs()
    captured: dict[str, int] = {}

    def _fake_run_ablation_round(*_args, **kwargs):
        captured["max_repair_attempts"] = int(kwargs["max_repair_attempts"])
        return {
            "status": "ok",
            "record": {
                "ablation_id": "ablation001",
                "ablation_index": 1,
                "target_id": "q001",
                "status": "success",
            },
        }

    monkeypatch.setattr("evocast.tools.tfb_ablation.run_ablation_round", _fake_run_ablation_round)

    result = run_ablation(
        session,
        {
            "target_id": "q001",
            "target_kind": "mechanism_ablation",
            "target": {
                "target_id": "q001",
                "target_kind": "mechanism_ablation",
                "mechanism_id": "m1",
                "mechanism_name": "fixture mechanism",
                "causal_variable": "fixture variable",
                "evidence_files": ["ts_benchmark/baselines/dtaf/dtaf.py"],
                "evidence_anchors": ["fixture"],
                "exact_edit_intent": "fixture edit",
            },
            "model_key": "DTAF",
            "baseline_metrics": {"mse_norm": 1.0},
            "budget": "unified",
            "objective_metric": "mse_norm",
        },
    )

    assert result["status"] == "ok"
    assert captured["max_repair_attempts"] == 4


def test_run_seed_evaluation_uses_explicit_nonconsecutive_seed_list(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / "repo"
    task_id = "seed_list_task"
    (base_dir / "runs" / task_id).mkdir(parents=True)
    captured_seeds: list[int] = []

    monkeypatch.setattr("evocast.runners.seed_runner.load_config_json", lambda _path: {})

    def _fake_build_run_configs(_tfb_config, _model_configs, *, save_path, seed, override_eval_args):
        captured_seeds.append(seed)
        return {"data": True}, {"model": True}, {"evaluation": True, "seed": seed, "save_path": save_path}

    def _fake_run_pipeline(_data_config, _model_config, evaluation_config, timeout, **_kwargs):
        seed = evaluation_config["seed"]
        artifact = base_dir / "runs" / task_id / f"seed_{seed}.json"
        artifact.write_text("{}", encoding="utf-8")
        return {"success": True, "log_paths": [str(artifact)]}

    def _fake_parse_metrics_from_paths(paths, objective_metric):
        seed = int(Path(paths[0]).stem.split("_")[-1])
        return {"metric_values": {objective_metric: float(seed)}}

    monkeypatch.setattr("evocast.runners.seed_runner.build_run_configs", _fake_build_run_configs)
    monkeypatch.setattr("evocast.runners.seed_runner.run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr("evocast.runners.seed_runner.parse_metrics_from_paths", _fake_parse_metrics_from_paths)
    monkeypatch.setattr("evocast.runners.seed_runner.stamp_result_artifacts", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "evocast.runners.seed_runner.validate_result_artifact_provenance",
        lambda *_args, **_kwargs: {"status": "ok"},
    )
    monkeypatch.setattr("evocast.runners.seed_runner.append_node", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("evocast.runners.seed_runner.record_runtime_event", lambda *_args, **_kwargs: None)

    result = run_seed_evaluation(
        task_id=task_id,
        node_id="node",
        model_config={"model_name": "fixture.Model"},
        config_path=str(tmp_path / "compiled.json"),
        objective_metric="mse_norm",
        seed_list=[5, 42],
        seed_universe=[5, 9, 42],
        precomputed_seed_values=[
            {
                "seed": 9,
                "objective_value": 9.0,
                "metrics": {"mse_norm": 9.0},
                "success": True,
                "source": "baseline_initial_run",
            }
        ],
        base_dir=str(base_dir),
        reference_mean=100.0,
        reference_std=0.0,
        reference_seed_count=3,
        promote_on_accept=False,
    )

    assert captured_seeds == [5, 42]
    assert result["seed_list"] == [5, 9, 42]
    assert result["num_seeds"] == 3
    assert [item["seed"] for item in result["per_seed"]] == [5, 9, 42]
    assert result["valid_metric_seeds"] == 3


def test_agent_control_uses_flat_proposal_rejection_limit(tmp_path: Path) -> None:
    policy_dir = tmp_path / "configs" / "policies"
    policy_dir.mkdir(parents=True)
    (policy_dir / "agent_control.yaml").write_text(
        "max_consecutive_proposal_rejections: 5\n",
        encoding="utf-8",
    )

    assert protocol_policy(str(tmp_path))["max_consecutive_rejections"] == 5


def test_agent_control_ignores_protocol_legacy_block(tmp_path: Path) -> None:
    policy_dir = tmp_path / "configs" / "policies"
    policy_dir.mkdir(parents=True)
    (policy_dir / "agent_control.yaml").write_text(
        """
protocol:
  max_consecutive_rejections: 4
""".strip(),
        encoding="utf-8",
    )

    assert protocol_policy(str(tmp_path))["max_consecutive_rejections"] == 3


def test_agent_control_default_context_strategy_is_compact() -> None:
    assert context_policy()["strategy"] == "proposal_compact_builder_minimal"


def test_agent_control_resolves_from_runtime_base_dir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    policy_dir = repo / "evocast" / "configs" / "policies"
    policy_dir.mkdir(parents=True)
    (policy_dir / "agent_control.yaml").write_text(
        """
round_control:
  max_repair_attempts: 7
gate:
  min_relative_improvement: 0.03
  min_seed_eval_relative_improvement: 0.02
  seed_eval_min_absolute_improvement: 0.0002
build_mode:
  force_smoke_experiment: true
  num_rollings: 2
""".strip(),
        encoding="utf-8",
    )

    runtime = repo / ".evocast"
    runtime.mkdir()

    assert round_control_policy(str(runtime))["max_repair_attempts"] == 7
    assert gate_policy(str(runtime))["min_relative_improvement"] == 0.03
    assert build_mode_policy(str(runtime))["num_rollings"] == 2
