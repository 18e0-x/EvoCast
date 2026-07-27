from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

from tests.evocast_regression.scripted_backend import ScriptedBackend
from evocast.build.contract_compiler import build_research_contract
from evocast.build.diff_audit import DiffAuditor
from evocast.build.metric_runner import TFBExperimentMetricRunner
from evocast.build.contract import BuildContract
from evocast.build.ledger import ResearchRoundLedger
from evocast.build.metadata_writer import ResearchMetadata
from evocast.build.orchestrator import ResearchBuildOrchestrator
from evocast.build.result import BuildDecision, VerificationResult
from evocast.build.source_snapshot import copy_source_tree, source_manifest
from evocast.harness.rounds import (
    FAILURE_KIND_RUNTIME,
    PHASE_BUILD,
    classify_failure_kind,
    current_round,
    record_experiment_run,
    record_gate_event,
    record_metric_completed_result,
    record_phase_transition,
    record_seed_eval_result,
    round_progress,
)
from evocast.harness.session import AgentSession
from evocast.state.runtime.terminal_ui import TerminalUI
from evocast.state.runtime.store import sync_best_baseline, sync_current_best
from evocast.state.domain_store import load_round_record


def _canonical_round_record(outcome) -> dict:
    return load_round_record(
        str(outcome.round_dir.parents[3]),
        str(outcome.round_dir.parents[1].name),
        str(outcome.research_id),
    )


class FakeMetadataWriter:
    def write(self, *, output_dir: Path, contract: BuildContract, patch_text: str = "", coding_summary: str = "", **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        protocol = contract.metric_protocol if isinstance(contract.metric_protocol, dict) else {}
        direction = protocol.get("research_direction") if isinstance(protocol.get("research_direction"), dict) else {}
        display = str(direction.get("terminal_display_title") or "Fixture marker edit")
        if "max_tool_turns" in str(coding_summary):
            idea_summary = "VALUE = 'improved' changes the fixture model marker."
        else:
            idea_summary = "The candidate changes the fixture model marker."
        metadata = ResearchMetadata(
            display_idea=display,
            idea_summary=idea_summary,
            mechanism_name="Fixture marker",
            changed_mechanism="model.py VALUE marker",
            expected_effect="Verifier detects the improved marker.",
        )
        (output_dir / "round_metadata.json").write_text(json.dumps(metadata.to_dict(), ensure_ascii=False), encoding="utf-8")
        return metadata


_RealResearchBuildOrchestrator = ResearchBuildOrchestrator


def ResearchBuildOrchestrator(**kwargs):
    kwargs.setdefault("metadata_writer", FakeMetadataWriter())
    return _RealResearchBuildOrchestrator(**kwargs)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "model.py").write_text("VALUE = 'baseline'\n", encoding="utf-8")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    return repo, source_manifest(repo)["snapshot_id"]


def _source_ref(repo: Path, *, entry_file: str = "model.py") -> dict:
    manifest = source_manifest(repo)
    binding = {"entry_file": entry_file, "source_files": [entry_file], "verified": True}
    return {
        "kind": "source_snapshot",
        "candidate_snapshot_id": manifest["snapshot_id"],
        "base_snapshot_id": manifest["snapshot_id"],
        "source_checkout": str(repo),
        "source_manifest_hash": manifest["manifest_hash"],
        "source_binding": binding,
    }


def test_orchestrator_patch_capture_uses_snapshot_diff_for_clean_workspace(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    orchestrator = ResearchBuildOrchestrator(
        base_dir=str(tmp_path / "runtime"),
        task_id="missing_stdout",
        repo_dir=repo,
        backend=ScriptedBackend([]),
    )
    contract = _contract(repo, base_snapshot_id, repair_budget=1)

    assert orchestrator._capture_patch(repo, contract) == ""
    assert orchestrator._changed_files(repo, contract) == []


def _contract(repo: Path, base_snapshot_id: str | None = None, *, repair_budget: int = 1) -> BuildContract:
    base_snapshot_id = base_snapshot_id or source_manifest(repo)["snapshot_id"]
    return BuildContract(
        research_id="Research001",
        base_snapshot_id=base_snapshot_id,
        semantic_goal="Change model.py so the fixture verifier can detect the candidate.",
        hypothesis="A visible candidate marker proves the build path works.",
        target_model="FixtureModel",
        base_source_ref=_source_ref(repo),
        allowed_edit_files=["model.py"],
        likely_entrypoints=["model.py"],
        discovery_hints=["Edit the smallest safe model surface."],
        protected_globs=["evocast/**"],
        forbidden_globs=["tests/**"],
        required_behavior=["model.py contains improved"],
        external_verification_commands=[
            [
                "python",
                "-c",
                "from pathlib import Path; assert 'improved' in Path('model.py').read_text()",
            ]
        ],
        timeout_seconds=30,
        repair_budget=repair_budget,
    )


def test_research_contract_does_not_use_agent_control_repair_budget(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"
    policy_dir = runtime / "configs" / "policies"
    policy_dir.mkdir(parents=True)
    (policy_dir / "agent_control.yaml").write_text(
        """
round_control:
  max_repair_attempts: 8
""".strip(),
        encoding="utf-8",
    )
    baseline = {
        "candidate_id": "baseline",
        "display_name": "FixtureModel",
        "import_path": "fixture.Model",
        "model_config": {"model_name": "fixture.Model", "model_hyper_params": {}},
        "source_binding": {
            "entry_file": "model.py",
            "source_files": ["model.py"],
            "source_hashes": {"model.py": "fixture"},
            "verified": True,
        },
        "source_ref": _source_ref(repo),
    }
    sync_best_baseline(str(runtime), "agent_control_repair_budget", baseline)
    contract = build_research_contract(
        base_dir=str(runtime),
        task_id="agent_control_repair_budget",
        baseline=baseline,
        objective_metric="mse_norm",
        repo_dir=repo,
        repair_budget=None,
    )

    assert contract.base_snapshot_id == base_snapshot_id
    assert contract.repair_budget == 0


def _variant_contract(repo: Path, base_snapshot_id: str | None = None, *, task_id: str = "build_variant_root") -> BuildContract:
    base_snapshot_id = base_snapshot_id or source_manifest(repo)["snapshot_id"]
    return BuildContract(
        research_id="Research001",
        base_snapshot_id=base_snapshot_id,
        semantic_goal="Edit the active model source.",
        hypothesis="The BuildContract source file is the formal write surface.",
        target_model="FixtureModel",
        base_source_ref=_source_ref(repo),
        allowed_edit_files=["model.py"],
        likely_entrypoints=["model.py"],
        required_behavior=["model.py contains improved"],
        external_verification_commands=[
            [
                "python",
                "-c",
                "from pathlib import Path; assert 'improved' in Path('model.py').read_text()",
            ]
        ],
        metric_protocol={"objective_metric": "mse_norm", "model_config": {"model_name": "FixtureModel"}},
        timeout_seconds=30,
        repair_budget=1,
    )


def _completed_metric_runner(_checkout: Path, _contract: BuildContract, output_dir: Path) -> VerificationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metric.json").write_text(json.dumps({"mse_norm": 0.1}), encoding="utf-8")
    return VerificationResult(
        status=BuildDecision.METRIC_COMPLETED,
        reason_code="fixture_metric_completed",
        summary="Fixture metric completed.",
        artifact_paths=[str(output_dir / "metric.json")],
        metrics={"mse_norm": 0.1},
        raw_payload={"result": {"success": True, "metrics": {"mse_norm": 0.1}}},
    )


def test_metric_completed_sync_overwrites_stale_experiment_failed_round(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    task_id = "metric_completed_sync"
    round_dir = runtime / "task_knowledge" / task_id / "rounds" / "Research002"
    round_dir.mkdir(parents=True)
    stale = {
        "round_id": 2,
        "research_index": 2,
        "research_id": "Research002",
        "round_scope": "research",
        "counts_toward_research_budget": True,
        "status": "experiment_failed",
        "phase": "closed",
        "failure_kind": "runtime_failure",
    }
    (round_dir / "round.json").write_text(json.dumps(stale), encoding="utf-8")

    record = record_metric_completed_result(
        base_dir=str(runtime),
        task_id=task_id,
        research_id="Research002",
        round_id=2,
        metrics={"mse_norm": 0.3142079437992507, "mae_norm": 0.3597151003428061},
        objective_metric="mse_norm",
        metric_result={
            "status": "METRIC_COMPLETED",
            "metrics": {"mse_norm": 0.3142079437992507},
            "raw_payload": {
                "result": {
                    "auto_gate": {
                        "decision": "reject",
                        "objective_metric": "mse_norm",
                        "candidate_metrics": {
                            "mse_norm": 0.3142079437992507,
                            "mae_norm": 0.3597151003428061,
                        },
                    }
                }
            },
        },
        candidate_snapshot_id="snapshot_002",
        patch_path=round_dir / "build_attempt_05" / "agent.patch",
        changed_files=["ts_benchmark/baselines/time_series_library/models/iTransformer.py"],
    )

    assert record["status"] == "rejected"
    assert record["phase"] == "closed"
    assert record["failure_kind"] == "scientific_failure"
    assert record["gate_decision"] == "reject"
    assert record["gate_event"]["decision"] == "reject"
    assert record["metrics"]["mse_norm"] == 0.3142079437992507
    assert record["candidate_metrics"]["mae_norm"] == 0.3597151003428061


def test_metric_completed_sync_preserves_marginal_no_seed_eval_as_not_promoted(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    task_id = "metric_completed_marginal"
    round_dir = runtime / "task_knowledge" / task_id / "rounds" / "Research004"
    round_dir.mkdir(parents=True)
    original = {
        "round_id": 4,
        "research_index": 4,
        "research_id": "Research004",
        "round_scope": "research",
        "counts_toward_research_budget": True,
        "status": "marginal_no_seed_eval",
        "phase": "experiment",
        "gate_decision": "marginal_no_seed_eval",
        "gate_event": {
            "decision": "marginal_no_seed_eval",
            "relative_improvement": 0.0048,
            "threshold_met": False,
            "candidate_metrics": {"mse_norm": 0.455357927188096, "mae_norm": 0.4863341332493084},
            "current_best_metrics": {"mse_norm": 0.4575566692098171, "mae_norm": 0.4882880552587458},
            "promoted_to_current_best": False,
        },
    }
    (round_dir / "round.json").write_text(json.dumps(original), encoding="utf-8")

    record = record_metric_completed_result(
        base_dir=str(runtime),
        task_id=task_id,
        research_id="Research004",
        round_id=4,
        metrics={"mse_norm": 0.455357927188096, "mae_norm": 0.4863341332493084},
        objective_metric="mse_norm",
        metric_result={"status": "METRIC_COMPLETED"},
    )

    assert record["status"] == "marginal_no_seed_eval"
    assert record["phase"] == "closed"
    assert record["failure_kind"] == "scientific_failure"
    assert record["gate_decision"] == "marginal_no_seed_eval"
    assert record["promoted_to_current_best"] is False
    assert "did not promote current_best" in record["close_reason"]
    assert record["metrics_source"] == "metric_completed"
    assert record["close_extra"]["metric_result"]["status"] == "METRIC_COMPLETED"


def test_metric_completed_without_objective_metric_enters_repair_failure(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"
    task_id = "metric_completed_without_metric"

    def turn(workspace: Path, _contract: BuildContract, _message: str, _timeout: int) -> dict:
        (workspace / "model.py").write_text("VALUE = 'improved'\n", encoding="utf-8")
        return {"status": "completed", "summary": "edited model source"}

    def bad_metric_runner(_checkout: Path, _contract: BuildContract, output_dir: Path) -> VerificationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        return VerificationResult(
            status=BuildDecision.METRIC_COMPLETED,
            reason_code="metric_completed",
            summary="Fixture metric claimed completion without metrics.",
            artifact_paths=[str(output_dir / "metric.json")],
            metrics={},
        )

    outcome = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id=task_id,
        repo_dir=repo,
        backend=ScriptedBackend([turn]),
        metric_runner=bad_metric_runner,
    ).run(_contract(repo, base_snapshot_id, repair_budget=0))

    assert outcome.status == BuildDecision.TERMINAL_REJECTED
    record = _canonical_round_record(outcome)
    assert record["status"] == "experiment_failed"
    assert record["close_extra"]["feedback"]["reason_code"] == "metric_completed_without_valid_objective_metric"
    assert round_progress(str(runtime), task_id)["terminal_rounds"] == 1


def test_build_orchestrator_accepts_candidate_from_clean_verifier_checkout(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"

    def turn(workspace: Path, _contract: BuildContract, _message: str, _timeout: int) -> dict:
        (workspace / "model.py").write_text("VALUE = 'improved'\n", encoding="utf-8")
        return {
            "status": "completed",
            "summary": "implemented a residual patch gate for FixtureModel",
            "display_idea": "FixtureModel residual patch gate",
            "idea_summary": "implemented a residual patch gate for FixtureModel with a documented config note",
            "mechanism_name": "residual patch gate",
            "changed_mechanism": "model.py fixture value",
            "expected_effect": "verifier detects improved candidate",
        }

    outcome = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id="build_accepts_candidate",
        repo_dir=repo,
        backend=ScriptedBackend([turn]),
        metric_runner=_completed_metric_runner,
    ).run(_contract(repo, base_snapshot_id))

    assert outcome.status == BuildDecision.ACCEPTED
    assert outcome.candidate_snapshot_id
    assert outcome.candidate_checkout and outcome.candidate_checkout.is_dir()
    assert outcome.patch_path and outcome.patch_path.is_file()
    record = _canonical_round_record(outcome)
    assert record["status"] == "completed"
    assert record["idea_summary"] == "The candidate changes the fixture model marker."
    assert record["mechanism_summary"] == record["idea_summary"]
    assert record["execution_summary"] == "implemented a residual patch gate for FixtureModel"
    assert record["display_idea"] == "Fixture marker edit"
    assert len(record["display_idea"].split()) <= 8
    assert record["candidate_snapshot_id"] == outcome.candidate_snapshot_id
    assert record["patch_path"] == str(outcome.patch_path)
    assert record["changed_files"] == outcome.changed_files
    assert record["build_outcome"]["status"] == BuildDecision.ACCEPTED.value
    from evocast.state.domain_store import load_build_contract
    assert load_build_contract(str(runtime), "build_accepts_candidate", outcome.research_id)["research_id"] == outcome.research_id
    assert (outcome.round_dir / "build_attempt_01" / "verifier" / "verifier.json").is_file()
    assert (outcome.round_dir / "metric" / "metric.json").is_file()
    assert round_progress(str(runtime), "build_accepts_candidate")["completed_rounds"] == 1


def test_scientist_critic_terminal_title_controls_display_idea(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"
    contract = replace(
        _contract(repo, base_snapshot_id),
        metric_protocol={
            "research_direction": {
                "planner": "scientist_critic",
                "terminal_display_title": "Patch Transition Probe",
                "idea_title": "A much longer title that should not reach terminal display",
            }
        }
    )

    def turn(workspace: Path, _contract: BuildContract, _message: str, _timeout: int) -> dict:
        (workspace / "model.py").write_text("VALUE = 'improved'\n", encoding="utf-8")
        return {
            "status": "completed",
            "summary": "implemented fixture change",
            "display_idea": "FixtureModel overly verbose agent generated display title",
            "idea_summary": "implemented a residual patch gate for FixtureModel",
        }

    outcome = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id="scientist_critic_display",
        repo_dir=repo,
        backend=ScriptedBackend([turn]),
        metric_runner=_completed_metric_runner,
    ).run(contract)

    assert outcome.status == BuildDecision.ACCEPTED
    record = _canonical_round_record(outcome)
    assert record["display_idea"] == "Patch Transition Probe"
    assert len(record["display_idea"].split()) <= 8


def test_build_contract_ablation_round_uses_ablation_label(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"
    task_id = "ablation_label"
    contract = BuildContract(
        research_id="Ablation001",
        base_snapshot_id=base_snapshot_id,
        semantic_goal="Ablate fixture behavior.",
        hypothesis="Removing fixture behavior changes the smoke metric.",
        target_model="FixtureModel",
        base_source_ref=_source_ref(repo),
        allowed_edit_files=["model.py"],
        failure_semantics={"round_kind": "baseline_diagnosis_ablation"},
    )

    opened = ResearchRoundLedger(base_dir=str(runtime), task_id=task_id).open_round(contract)

    assert opened.research_id == "Ablation001"
    assert opened.round_dir == runtime / "task_knowledge" / task_id / "rounds" / "Ablation001"
    assert load_round_record(str(runtime), "ablation_label", "Ablation001")["research_id"] == "Ablation001"
    assert not (runtime / "task_knowledge" / task_id / "rounds" / "Research001").exists()
    assert round_progress(str(runtime), task_id)["research_rounds"] == 0


def test_build_orchestrator_uses_patch_idea_when_tool_turn_budget_finishes_with_diff(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"
    task_id = "build_patch_idea_from_max_tool_turns"

    def turn(workspace: Path, _contract: BuildContract, _message: str, _timeout: int) -> dict:
        (workspace / "model.py").write_text("VALUE = 'improved'\n", encoding="utf-8")
        return {
            "status": "completed",
            "summary": "coding backend exceeded max_tool_turns=10; workspace has changes for external audit",
            "execution_summary": "coding backend exceeded max_tool_turns=10; workspace has changes for external audit",
        }

    outcome = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id=task_id,
        repo_dir=repo,
        backend=ScriptedBackend([turn]),
        metric_runner=_completed_metric_runner,
    ).run(_contract(repo, base_snapshot_id))

    assert outcome.status == BuildDecision.ACCEPTED
    record = _canonical_round_record(outcome)
    assert "max_tool_turns" in record["execution_summary"]
    assert "max_tool_turns" not in record["idea_summary"]
    assert "VALUE = 'improved'" in record["idea_summary"]

    ui = TerminalUI(str(runtime), task_id)
    row = {item["id"]: item for item in ui._result_rows(ui._task_view())}["Research001"]
    assert "max_tool_turns" not in row["what"]
    assert row["what"] == "FixtureModel | Fixture marker edit"


def test_build_orchestrator_ignores_generated_python_cache_in_candidate_patch(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"

    def turn(workspace: Path, _contract: BuildContract, _message: str, _timeout: int) -> dict:
        (workspace / "model.py").write_text("VALUE = 'improved'\n", encoding="utf-8")
        cache_dir = workspace / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "model.cpython-310.pyc").write_bytes(b"generated")
        return {"status": "completed", "summary": "implemented fixture change"}

    outcome = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id="build_ignores_pycache",
        repo_dir=repo,
        backend=ScriptedBackend([turn]),
        metric_runner=_completed_metric_runner,
    ).run(_contract(repo, base_snapshot_id))

    assert outcome.status == BuildDecision.ACCEPTED
    patch_text = outcome.patch_path.read_text(encoding="utf-8")
    assert "__pycache__" not in patch_text


def test_build_orchestrator_captures_allowed_source_file(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"
    task_id = "build_variant_root"

    def turn(workspace: Path, _contract: BuildContract, _message: str, _timeout: int) -> dict:
        (workspace / "model.py").write_text("VALUE = 'improved'\n", encoding="utf-8")
        return {"status": "completed", "summary": "edited model source"}

    outcome = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id=task_id,
        repo_dir=repo,
        backend=ScriptedBackend([turn]),
        metric_runner=_completed_metric_runner,
    ).run(_variant_contract(repo, base_snapshot_id, task_id=task_id))

    assert outcome.status == BuildDecision.ACCEPTED
    assert outcome.changed_files == ["model.py"]
    assert "model.py" in outcome.patch_path.read_text(encoding="utf-8")


def test_candidate_snapshot_from_agent_state_without_git_apply(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"

    def turn(workspace: Path, _contract: BuildContract, _message: str, _timeout: int) -> dict:
        (workspace / "model.py").write_text("IMPORT = 'changed'\nVALUE = 'improved body'\n", encoding="utf-8")
        return {"status": "completed", "summary": "edited import and body"}

    outcome = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id="build_no_git_apply",
        repo_dir=repo,
        backend=ScriptedBackend([turn]),
        metric_runner=_completed_metric_runner,
    ).run(_contract(repo, base_snapshot_id))

    assert outcome.status == BuildDecision.ACCEPTED
    assert outcome.patch_path and outcome.patch_path.is_file()
    assert outcome.candidate_snapshot_id
    assert outcome.candidate_checkout
    materialized = (outcome.candidate_checkout / "model.py").read_text(encoding="utf-8")
    assert "IMPORT = 'changed'" in materialized
    assert "VALUE = 'improved body'" in materialized


def test_smoke_failure_enters_repair_loop(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"
    messages: list[str] = []

    def first_turn(workspace: Path, _contract: BuildContract, message: str, _timeout: int) -> dict:
        messages.append(message)
        (workspace / "model.py").write_text("VALUE = 'improved but bad smoke'\n", encoding="utf-8")
        return {"status": "completed", "summary": "bad smoke candidate"}

    def repair_turn(workspace: Path, _contract: BuildContract, message: str, _timeout: int) -> dict:
        messages.append(message)
        (workspace / "model.py").write_text("VALUE = 'improved after smoke repair'\n", encoding="utf-8")
        return {"status": "completed", "summary": "repaired smoke candidate"}

    calls = 0

    def metric_runner(_checkout: Path, _contract: BuildContract, output_dir: Path) -> VerificationResult:
        nonlocal calls
        calls += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        if calls == 1:
            return VerificationResult(
                status=BuildDecision.REPAIR_REQUIRED,
                reason_code="shape_error",
                summary="RuntimeError: bad shape",
                repair_instructions=["Use traceback to repair shape."],
                raw_payload={"stage": "smoke", "error_type": "shape_error", "traceback": "Traceback..."},
            )
        return _completed_metric_runner(_checkout, _contract, output_dir)

    outcome = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id="build_smoke_repair",
        repo_dir=repo,
        backend=ScriptedBackend([first_turn, repair_turn]),
        metric_runner=metric_runner,
    ).run(_contract(repo, base_snapshot_id, repair_budget=1))

    assert outcome.status == BuildDecision.ACCEPTED
    assert calls == 2
    assert "evocast_feedback" in messages[1]
    assert "RuntimeError: bad shape" in messages[1]


def test_evaluator_error_is_runtime_failure_not_scientific_failure() -> None:
    assert (
        classify_failure_kind(
            "experiment_failed",
            error_type="evaluator_error",
            has_experiment_run=True,
        )
        == FAILURE_KIND_RUNTIME
    )


def test_metric_repair_exhausted_after_terminal_round_does_not_double_close(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"
    task_id = "metric_terminal_no_double_close"
    messages: list[str] = []

    def turn(workspace: Path, _contract: BuildContract, message: str, _timeout: int) -> dict:
        messages.append(message)
        (workspace / "model.py").write_text(
            f"VALUE = 'improved attempt {len(messages)}'\n",
            encoding="utf-8",
        )
        return {
            "status": "completed",
            "summary": "edited model source",
            "display_idea": "Fixture metric failure",
            "idea_summary": "fixture candidate for exhausted metric repair",
        }

    calls = 0

    def metric_runner(_checkout: Path, _contract: BuildContract, output_dir: Path) -> VerificationResult:
        nonlocal calls
        calls += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        record_experiment_run(
            base_dir=str(runtime),
            task_id=task_id,
            run_id=f"metric_fail_{calls}",
            variant_path=None,
            status="failed",
            error_type="evaluator_error",
            error_message="formal forecast path is not batch_forecast: []",
            stage="experiment",
        )
        return VerificationResult(
            status=BuildDecision.REPAIR_REQUIRED,
            reason_code="evaluator_error",
            summary="formal forecast path is not batch_forecast: []",
            repair_instructions=["canonical metric failed"],
            raw_payload={"stage": "canonical_metric", "error_type": "evaluator_error"},
        )

    outcome = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id=task_id,
        repo_dir=repo,
        backend=ScriptedBackend([turn, turn, turn]),
        metric_runner=metric_runner,
    ).run(_contract(repo, base_snapshot_id, repair_budget=2))

    assert outcome.status == BuildDecision.TERMINAL_REJECTED
    assert calls == 3
    assert current_round(str(runtime), task_id) is None
    round_record = _canonical_round_record(outcome)
    assert round_record["status"] == "experiment_failed"
    assert round_record["failure_kind"] == FAILURE_KIND_RUNTIME
    assert round_record["failure_analysis"]["terminal_reason"] == "variant_forge_attempts_exhausted"
    assert len(round_record["runs"]) == 3
    assert outcome.patch_path and outcome.patch_path.is_file()
    assert "evocast_feedback" in messages[1]


def test_build_orchestrator_recovers_stale_build_round_with_existing_diff(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"
    task_id = "build_recovers_stale_round"
    contract = _contract(repo, base_snapshot_id)
    ledger = ResearchRoundLedger(base_dir=str(runtime), task_id=task_id)
    opened = ledger.open_round(contract)
    agent_workspace = opened.round_dir / "agent_workspace"
    copy_source_tree(repo, agent_workspace, overwrite=True)
    (agent_workspace / "model.py").write_text("VALUE = 'improved from stale workspace'\n", encoding="utf-8")
    (opened.round_dir / "workspaces.json").write_text(
        json.dumps({"agent_workspace": str(agent_workspace)}),
        encoding="utf-8",
    )
    record_phase_transition(base_dir=str(runtime), task_id=task_id, phase=PHASE_BUILD)

    outcome = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id=task_id,
        repo_dir=repo,
        backend=ScriptedBackend([]),
        metric_runner=_completed_metric_runner,
    ).run(contract)

    assert outcome.status == BuildDecision.ACCEPTED
    assert outcome.changed_files == ["model.py"]
    assert "improved from stale workspace" in outcome.patch_path.read_text(encoding="utf-8")
    assert current_round(str(runtime), task_id) is None


def test_build_orchestrator_runs_seed_eval_for_positive_build_mode_gate(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"
    task_id = "build_seed_eval_gate"
    seed_calls: list[dict] = []

    def turn(workspace: Path, _contract: BuildContract, _message: str, _timeout: int) -> dict:
        (workspace / "model.py").write_text("VALUE = 'improved'\n", encoding="utf-8")
        return {"status": "completed", "summary": "edited model source"}

    def metric_runner(checkout: Path, contract: BuildContract, output_dir: Path) -> VerificationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"mse_norm": 0.9}
        record_experiment_run(
            base_dir=str(runtime),
            task_id=task_id,
            run_id="metric_run",
            variant_path=None,
            status="success",
            metrics=metrics,
            stage="experiment",
        )
        record_gate_event(
            base_dir=str(runtime),
            task_id=task_id,
            gate_event={
                "run_id": "metric_run",
                "source_checkout": str(checkout),
                "decision": "needs_seed_eval",
                "objective_metric": "mse_norm",
                "candidate_metrics": metrics,
                "candidate_value": 0.9,
                "baseline_value": 1.0,
                "gate": {"decision": "needs_seed_eval", "reason": "fixture requires seed eval"},
            },
        )
        (output_dir / "metric.json").write_text(json.dumps(metrics), encoding="utf-8")
        return VerificationResult(
            status=BuildDecision.METRIC_COMPLETED,
            reason_code="metric_completed",
            summary="Fixture metric completed.",
            artifact_paths=[str(output_dir / "metric.json")],
            metrics=metrics,
            raw_payload={"result": {"success": True, "metrics": metrics}},
        )

    def seed_eval_runner(args: dict) -> dict:
        seed_calls.append(dict(args))
        result = {
            "node_id": "seed_eval_metric_run",
            "mean": 0.88,
            "std": 0.01,
            "successful_seeds": 3,
            "significance_decision": {"decision": "accept"},
            "promoted_to_current_best": True,
            "result_path": str(runtime / "seed_eval.json"),
        }
        round_record = record_seed_eval_result(
            base_dir=str(runtime),
            task_id=task_id,
            variant_path=args.get("variant_path") or None,
            result=result,
        )
        return {**result, "round_record": round_record}

    outcome = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id=task_id,
        repo_dir=repo,
        backend=ScriptedBackend([turn]),
        metric_runner=metric_runner,
        seed_eval_runner=seed_eval_runner,
    ).run(_variant_contract(repo, base_snapshot_id, task_id=task_id))

    record = _canonical_round_record(outcome)
    assert outcome.status == BuildDecision.ACCEPTED
    assert len(seed_calls) == 1
    assert seed_calls[0]["source_checkout"]
    assert seed_calls[0]["promotion_metadata"]["changed_files"] == ["model.py"]
    assert record["status"] == "completed"
    assert record["gate_decision"] == "accept"
    assert record["seed_eval"]["status"] == "completed"
    events = (outcome.round_dir / "event_ledger.jsonl").read_text(encoding="utf-8")
    assert "seed_eval_completed" in events
    assert "round_completed" not in events
    assert current_round(str(runtime), task_id) is None


def test_build_orchestrator_records_failed_seed_eval_obligation(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"
    task_id = "build_seed_eval_failure_obligation"

    def turn(workspace: Path, _contract: BuildContract, _message: str, _timeout: int) -> dict:
        (workspace / "model.py").write_text("VALUE = 'improved'\n", encoding="utf-8")
        return {"status": "completed", "summary": "edited model source"}

    def metric_runner(checkout: Path, _contract: BuildContract, output_dir: Path) -> VerificationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"mse_norm": 0.9}
        record_experiment_run(
            base_dir=str(runtime),
            task_id=task_id,
            run_id="metric_run",
            variant_path=None,
            status="success",
            metrics=metrics,
            stage="experiment",
        )
        record_gate_event(
            base_dir=str(runtime),
            task_id=task_id,
            gate_event={
                "run_id": "metric_run",
                "source_checkout": str(checkout),
                "decision": "needs_seed_eval",
                "objective_metric": "mse_norm",
                "candidate_metrics": metrics,
                "candidate_value": 0.9,
                "baseline_value": 1.0,
                "gate": {"decision": "needs_seed_eval", "reason": "fixture requires seed eval"},
            },
        )
        return VerificationResult(
            status=BuildDecision.METRIC_COMPLETED,
            reason_code="metric_completed",
            summary="Fixture metric completed.",
            artifact_paths=[],
            metrics=metrics,
            raw_payload={"result": {"success": True, "metrics": metrics}},
        )

    def seed_eval_runner(_args: dict) -> dict:
        raise RuntimeError("fixture seed eval crash")

    outcome = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id=task_id,
        repo_dir=repo,
        backend=ScriptedBackend([turn]),
        metric_runner=metric_runner,
        seed_eval_runner=seed_eval_runner,
    ).run(_variant_contract(repo, base_snapshot_id, task_id=task_id))

    record = _canonical_round_record(outcome)
    assert outcome.status == BuildDecision.TERMINAL_REJECTED
    assert record["status"] == "experiment_failed"
    assert record["phase"] == "closed"
    assert record["gate_decision"] == "needs_seed_eval"
    assert record["seed_eval"]["status"] == "failed"
    assert record["seed_eval"]["error_type"] == "RuntimeError"
    assert "fixture seed eval crash" in record["seed_eval"]["error_message"]
    assert current_round(str(runtime), task_id) is None


def test_build_orchestrator_rejects_stale_contract_reuse_for_next_round(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"

    def turn(workspace: Path, _contract: BuildContract, _message: str, _timeout: int) -> dict:
        (workspace / "model.py").write_text("VALUE = 'improved'\n", encoding="utf-8")
        return {"status": "completed", "summary": "implemented a residual patch gate for FixtureModel"}

    orchestrator = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id="build_rejects_stale_contract",
        repo_dir=repo,
        backend=ScriptedBackend([turn]),
        metric_runner=_completed_metric_runner,
    )
    orchestrator.run(_contract(repo, base_snapshot_id))

    try:
        ResearchBuildOrchestrator(
            base_dir=str(runtime),
            task_id="build_rejects_stale_contract",
            repo_dir=repo,
            backend=ScriptedBackend([turn]),
            metric_runner=_completed_metric_runner,
        ).run(_contract(repo, base_snapshot_id))
    except RuntimeError as exc:
        assert "Generate a fresh BuildContract" in str(exc)
    else:
        raise AssertionError("stale Research001 BuildContract must not open Research002")


def test_build_orchestrator_requires_allowed_source_file(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"
    task_id = "build_variant_root_repair"

    def bad_turn(workspace: Path, _contract: BuildContract, _message: str, _timeout: int) -> dict:
        (workspace / "README.md").write_text("changed outside allowed source\n", encoding="utf-8")
        return {"status": "completed", "summary": "changed unrelated source"}

    outcome = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id=task_id,
        repo_dir=repo,
        backend=ScriptedBackend([bad_turn]),
    ).run(_variant_contract(repo, base_snapshot_id, task_id=task_id))

    assert outcome.status == BuildDecision.TERMINAL_REJECTED
    audit = json.loads((outcome.round_dir / "build_attempt_01" / "diff_audit.json").read_text(encoding="utf-8"))
    assert audit["findings"][0]["code"] == "outside_allowed_build_root"


def test_build_orchestrator_repairs_empty_patch_in_same_round(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"
    messages: list[str] = []

    def empty_turn(_workspace: Path, _contract: BuildContract, message: str, _timeout: int) -> dict:
        messages.append(message)
        return {"status": "completed", "summary": "no change yet"}

    def repaired_turn(workspace: Path, _contract: BuildContract, message: str, _timeout: int) -> dict:
        messages.append(message)
        (workspace / "model.py").write_text("VALUE = 'improved after repair'\n", encoding="utf-8")
        return {"status": "completed", "summary": "repaired"}

    outcome = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id="build_repairs_empty_patch",
        repo_dir=repo,
        backend=ScriptedBackend([empty_turn, repaired_turn]),
        metric_runner=_completed_metric_runner,
    ).run(_contract(repo, base_snapshot_id, repair_budget=1))

    assert outcome.status == BuildDecision.ACCEPTED
    assert outcome.repair_attempts == 1
    assert len(messages) == 2
    assert "evocast_feedback" in messages[1]
    record = _canonical_round_record(outcome)
    assert record["round_id"] == 1
    assert len(record.get("repair_history") or []) == 1
    assert current_round(str(runtime), "build_repairs_empty_patch") is None


def test_build_orchestrator_closes_failed_when_metric_runner_is_missing(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"

    def turn(workspace: Path, _contract: BuildContract, _message: str, _timeout: int) -> dict:
        (workspace / "model.py").write_text("VALUE = 'improved'\n", encoding="utf-8")
        return {"status": "completed", "summary": "implemented fixture change"}

    outcome = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id="build_missing_metric_runner",
        repo_dir=repo,
        backend=ScriptedBackend([turn]),
    ).run(_contract(repo, base_snapshot_id))

    assert outcome.status == BuildDecision.TERMINAL_REJECTED
    assert outcome.candidate_snapshot_id
    record = _canonical_round_record(outcome)
    assert record["status"] == "experiment_failed"
    assert record["candidate_snapshot_id"] == outcome.candidate_snapshot_id
    assert record["patch_path"] == str(outcome.patch_path)
    assert record["changed_files"] == outcome.changed_files
    assert record["build_outcome"]["status"] == BuildDecision.TERMINAL_REJECTED.value
    assert current_round(str(runtime), "build_missing_metric_runner") is None


def test_build_orchestrator_terminal_rejects_forbidden_test_changes(tmp_path: Path) -> None:
    repo, _base_snapshot_id = _repo(tmp_path)
    (repo / "tests").mkdir()
    (repo / "tests" / "fixture.py").write_text("VALUE = 'baseline'\n", encoding="utf-8")
    base_snapshot_id = source_manifest(repo)["snapshot_id"]
    runtime = tmp_path / "runtime"

    def bad_turn(workspace: Path, _contract: BuildContract, _message: str, _timeout: int) -> dict:
        (workspace / "tests" / "fixture.py").write_text("VALUE = 'modified'\n", encoding="utf-8")
        return {"status": "completed", "summary": "modified forbidden tests"}

    outcome = ResearchBuildOrchestrator(
        base_dir=str(runtime),
        task_id="build_rejects_forbidden",
        repo_dir=repo,
        backend=ScriptedBackend([bad_turn]),
    ).run(_contract(repo, base_snapshot_id, repair_budget=1))

    assert outcome.status == BuildDecision.TERMINAL_REJECTED
    assert outcome.candidate_snapshot_id is None
    record = _canonical_round_record(outcome)
    assert record["status"] == "implementation_rejected"
    assert record["candidate_snapshot_id"] is None
    assert record["patch_path"] == str(outcome.patch_path)
    assert record["changed_files"] == outcome.changed_files
    assert record["build_outcome"]["status"] == BuildDecision.TERMINAL_REJECTED.value
    audit = json.loads((outcome.round_dir / "build_attempt_01" / "diff_audit.json").read_text(encoding="utf-8"))
    assert audit["reason_code"] == "terminal_diff_violation"


def test_contract_compiler_uses_current_best_source_snapshot(tmp_path: Path) -> None:
    repo, _base_snapshot_id = _repo(tmp_path)
    (repo / "model.py").write_text("class Model:\n    def forward(self):\n        return 'best'\n", encoding="utf-8")
    best_ref = _source_ref(repo)
    best_snapshot_id = best_ref["candidate_snapshot_id"]
    runtime = tmp_path / "runtime"
    task_id = "compiler_current_best_source"
    baseline = {
        "candidate_id": "Baseline",
        "display_name": "FixtureModel",
        "import_path": "model",
        "source_binding": {
            "schema_version": "source_binding_v1",
            "entry_file": "model.py",
            "source_files": ["model.py"],
            "core_files": [],
            "support_files": [],
        },
        "metrics": {"mse_norm": 1.0},
        "objective_metric": "mse_norm",
        "model_config": {"model_name": "model", "model_hyper_params": {}},
    }
    sync_best_baseline(str(runtime), task_id, baseline)
    sync_current_best(
        str(runtime),
        task_id,
        {
            **baseline,
            "candidate_id": "Research002",
            "metrics": {"mse_norm": 0.9},
            "source_ref": best_ref,
        },
    )

    contract = build_research_contract(
        base_dir=str(runtime),
        task_id=task_id,
        baseline=baseline,
        objective_metric="mse_norm",
        repo_dir=repo,
        research_id="Research003",
    )

    assert contract.base_snapshot_id == best_snapshot_id
    assert contract.base_candidate_id == "Research002"
    assert contract.base_source_ref["candidate_snapshot_id"] == best_snapshot_id
    assert contract.allowed_edit_files == ["model.py"]
    assert "variant_path" not in contract.metric_protocol


def test_research_contract_embeds_scientist_critic_direction(tmp_path: Path) -> None:
    repo, _base_snapshot_id = _repo(tmp_path)
    runtime = tmp_path / "runtime"
    task_id = "compiler_scientist_critic_direction"
    baseline = {
        "candidate_id": "Baseline",
        "display_name": "FixtureModel",
        "import_path": "model",
        "source_binding": {
            "schema_version": "source_binding_v1",
            "entry_file": "model.py",
            "source_files": ["model.py"],
            "core_files": [],
            "support_files": [],
        },
        "metrics": {"mse_norm": 1.0},
        "objective_metric": "mse_norm",
        "model_config": {"model_name": "model", "model_hyper_params": {}},
        "source_ref": _source_ref(repo),
    }
    sync_best_baseline(str(runtime), task_id, baseline)
    direction = {
        "planner": "scientist_critic",
        "terminal_display_title": "State-Aware Gate",
        "round_role": "explore",
        "hypothesis": "A state-aware gate can test regime-sensitive forecasting.",
        "target_mechanism": "state gating",
        "target_code_region": "model.py forward",
        "expected_behavior_delta": "shape-preserving changed forecast",
        "failure_to_avoid": "do not change training policy",
        "what_this_round_will_teach": "whether state routing is useful",
    }

    contract = build_research_contract(
        base_dir=str(runtime),
        task_id=task_id,
        baseline=baseline,
        objective_metric="mse_norm",
        repo_dir=repo,
        research_id="Research001",
        research_direction=direction,
    )

    assert contract.hypothesis == direction["hypothesis"]
    assert contract.metric_protocol["research_direction"]["planner"] == "scientist_critic"
    assert "Scientist-Critic selected research direction" in "\n".join(contract.discovery_hints)
    assert "Implement the selected Scientist-Critic research direction" in "\n".join(contract.required_behavior)
    assert "State-Aware Gate" in "\n".join(contract.discovery_hints)


def test_contract_compiler_recovers_source_binding_for_promoted_source_patch(tmp_path: Path) -> None:
    repo, _base_snapshot_id = _repo(tmp_path)
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "model.py").write_text("class Model:\n    pass\n", encoding="utf-8")
    best_ref = _source_ref(repo, entry_file="pkg/model.py")
    best_ref["source_binding"] = {
        "schema_version": "source_binding_v1",
        "entry_file": "pkg/model.py",
        "source_files": ["pkg/model.py", "pkg/__init__.py"],
        "core_files": ["pkg/model.py"],
        "support_files": ["pkg/__init__.py"],
    }
    best_snapshot_id = best_ref["candidate_snapshot_id"]
    runtime = tmp_path / "runtime"
    task_id = "compiler_recovers_binding"
    binding = {
        "schema_version": "source_binding_v1",
        "entry_file": "pkg/model.py",
        "source_files": ["pkg/model.py", "pkg/__init__.py"],
        "core_files": ["pkg/model.py"],
        "support_files": ["pkg/__init__.py"],
    }
    binding_ref = runtime / "task_knowledge" / task_id / "model_bindings" / "baseline.json"
    binding_ref.parent.mkdir(parents=True)
    binding_ref.write_text(
        json.dumps({"source_binding": binding, "public_import_path": "pkg.model.Model"}),
        encoding="utf-8",
    )
    baseline = {
        "candidate_id": "baseline_001_Model",
        "display_name": "FixtureModel",
        "import_path": "pkg.model.Model",
        "model_binding_ref": str(binding_ref),
        "metrics": {"mse_norm": 1.0},
        "objective_metric": "mse_norm",
        "model_config": {"model_name": "pkg.model.Model", "model_hyper_params": {}},
    }
    sync_best_baseline(str(runtime), task_id, baseline)
    sync_current_best(
        str(runtime),
        task_id,
        {
            "candidate_id": "Research002",
            "display_name": "FixtureModel",
            "import_path": "pkg.model.Model",
            "candidate_kind": "source_patch",
            "candidate_snapshot_id": best_snapshot_id,
            "base_snapshot_id": best_snapshot_id,
            "changed_files": ["pkg/model.py"],
            "source_ref": {**best_ref, "changed_files": ["pkg/model.py"], "model_binding_ref": str(binding_ref)},
            "metrics": {"mse_norm": 0.9},
            "objective_metric": "mse_norm",
            "model_config": {"model_name": "pkg.model.Model", "model_hyper_params": {}},
        },
    )

    contract = build_research_contract(
        base_dir=str(runtime),
        task_id=task_id,
        baseline=baseline,
        objective_metric="mse_norm",
        repo_dir=repo,
        research_id="Research003",
    )

    assert contract.base_candidate_id == "Research002"
    assert contract.base_snapshot_id == best_snapshot_id
    assert contract.allowed_edit_files == ["pkg/model.py", "pkg/__init__.py"]
    assert contract.base_source_ref["source_binding"]["entry_file"] == "pkg/model.py"


def test_repo_wide_authority_allows_candidate_changes_outside_source_binding(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    orchestrator = ResearchBuildOrchestrator(
        base_dir=str(tmp_path / "runtime"),
        task_id="repo_wide_authority",
        repo_dir=repo,
        backend=ScriptedBackend([]),
    )
    contract = BuildContract(
        **{
            **_contract(repo, base_snapshot_id).to_dict(),
            "execution_authority": "repo_wide",
            "allowed_edit_files": ["model.py", "README.md", "tests/test_any.py"],
            "allowed_new_file_roots": ["."],
            "protected_globs": [],
            "forbidden_globs": [],
        }
    )

    orchestrator._validate_candidate_changed_files(
        ["tests/test_any.py", "README.md"],
        contract,
        base_checkout=repo,
    )


def test_repo_wide_authority_diff_audit_allows_test_path_changes(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    contract = BuildContract(
        **{
            **_contract(repo, base_snapshot_id).to_dict(),
            "execution_authority": "repo_wide",
            "allowed_edit_files": ["model.py", "tests/test_any.py"],
            "allowed_new_file_roots": ["."],
            "protected_globs": [],
            "forbidden_globs": [],
        }
    )

    result = DiffAuditor().audit(
        contract=contract,
        changed_files=["tests/test_any.py"],
        patch_text="diff --snapshot a/tests/test_any.py b/tests/test_any.py\n--- a/tests/test_any.py\n+++ b/tests/test_any.py\n",
        output_dir=tmp_path / "audit",
    )

    assert result.status == BuildDecision.ACCEPTED


def test_metric_runner_executes_from_candidate_checkout(monkeypatch, tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    checkout = tmp_path / "checkout"
    copy_source_tree(repo, checkout, overwrite=True)
    runtime = tmp_path / "runtime"
    session = AgentSession(task_id="metric_source_checkout", base_dir=str(runtime), client=None, dry_run=False)
    calls: list[dict] = []

    def fake_run_experiment(_session: AgentSession, args: dict) -> dict:
        calls.append(dict(args))
        return {"success": True, "metrics": {"mse_norm": 0.5}, "log_paths": []}

    monkeypatch.setattr("evocast.build.metric_runner.run_experiment", fake_run_experiment)
    contract = BuildContract(
        research_id="Research001",
        base_snapshot_id=base_snapshot_id,
        semantic_goal="edit model source",
        hypothesis="metric runner uses checkout source",
        target_model="FixtureModel",
        base_source_ref=_source_ref(repo),
        allowed_edit_files=["model.py"],
        metric_protocol={
            "objective_metric": "mse_norm",
            "model_config": {"model_name": "model", "adapter": "fixture", "model_hyper_params": {"x": 1}},
        },
    )

    result = TFBExperimentMetricRunner(session=session)(checkout, contract, tmp_path / "metric")

    assert result.status == BuildDecision.METRIC_COMPLETED
    assert result.metrics == {"mse_norm": 0.5}
    assert result.raw_payload["result"]["metrics"] == {"mse_norm": 0.5}
    assert calls[0]["source_checkout"] == str(checkout.resolve())
    assert calls[0]["source_entry_file"] == str((checkout / "model.py").resolve())
    assert calls[0]["model_name"] == "model"
    assert calls[0]["adapter"] == "fixture"
