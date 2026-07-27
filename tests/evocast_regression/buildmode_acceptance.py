"""Build-mode acceptance harness kept with its regression tests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from evocast.configurator.config_compiler import compile_config
from evocast.research.dataset_profile import load_dataset_profile, write_skipped_dataset_profile
from evocast.domain.knowledge_paths import runtime_root, task_knowledge_dir
from evocast.harness.rounds import round_progress
from evocast.scripts.init_task import init_task
from evocast.state.domain_store import domain_state_path, list_round_records, load_task_config, save_task_config


def _bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _default_intent(dataset_path: Path, *, seq_len: int, horizon: int, baseline_model: str) -> Dict[str, Any]:
    return {
        "dataset_path": str(dataset_path),
        "time_col": "date",
        "task_mode": "MM",
        "target_columns": [],
        "horizons": [int(horizon)],
        "input_chunk_length": int(seq_len),
        "strategy_name": "rolling_forecast",
        "objective_metric": "mse_norm",
        "frequency": "hourly",
        "baseline_strategy": "manual",
        "baseline_models": [str(baseline_model)],
        "build_mode": True,
        "dataset_diagnosis_mode": "skip",
        "baseline_diagnosis_max_ablation_targets": 0,
    }


def _ensure_task(
    *,
    base_dir: str,
    task_id: str,
    dataset_path: Path,
    seq_len: int,
    horizon: int,
    baseline_model: str,
    rounds_per_attempt: int,
    api_config: str,
) -> Dict[str, Any]:
    knowledge_dir = task_knowledge_dir(base_dir, task_id)
    if not domain_state_path(base_dir, task_id).is_file():
        compiled = compile_config(
            _default_intent(dataset_path, seq_len=seq_len, horizon=horizon, baseline_model=baseline_model),
            seed=2021,
            build_mode=True,
        )
        compiled_path = knowledge_dir / "compiled_config.json"
        _write_json(compiled_path, compiled)
        init_task(
            task_id=task_id,
            config_path=str(compiled_path),
            objective_metric="mse_norm",
            budget="unified",
            metric_direction="lower_is_better",
            max_rounds=int(rounds_per_attempt),
            max_debug_depth=3,
            api_config=api_config,
            baseline_strategy="manual",
            baseline_models=[str(baseline_model)],
            build_mode=True,
            dataset_diagnosis_mode="skip",
            baseline_diagnosis_max_ablation_targets=0,
            force_full_rounds=True,
            language="zh",
            base_dir=base_dir,
        )
    task_config = load_task_config(base_dir, task_id)
    task_config.update(
        {
            "build_mode": True,
            "dataset_diagnosis_mode": "skip",
            "baseline_diagnosis_max_ablation_targets": 0,
            "max_rounds": int(rounds_per_attempt),
            "force_full_rounds": True,
            "baseline_strategy": "manual",
            "baseline_models": [str(baseline_model)],
            "budget": "unified",
            "objective_metric": "mse_norm",
            "metric_direction": "lower_is_better",
            "acceptance_mode": "buildmode_research_acceptance",
            "acceptance_baseline_model": str(baseline_model),
            "acceptance_rounds_per_attempt": int(rounds_per_attempt),
        }
    )
    save_task_config(base_dir, task_id, task_config)
    write_skipped_dataset_profile(task_id=task_id, base_dir=base_dir, language="zh")
    return task_config


def _latest_json_by_prefix(directory: Path, prefix: str) -> Dict[str, Any]:
    files = sorted(directory.glob(f"{prefix}*.json"))
    if not files:
        return {}
    return _read_json(files[-1], {}) or {}


def _research_round_facts(base_dir: str, task_id: str) -> List[Dict[str, Any]]:
    knowledge_dir = task_knowledge_dir(base_dir, task_id)
    facts: List[Dict[str, Any]] = []
    records = [
        record
        for record in list_round_records(base_dir, task_id)
        if (
            bool(record.get("counts_toward_research_budget"))
            if record.get("counts_toward_research_budget") is not None
            else str(record.get("round_scope") or "research").strip().lower() == "research"
        )
    ]
    for record in sorted(records, key=lambda item: str(item.get("research_id") or "")):
        rid = str(record.get("research_id") or "")
        if not rid:
            continue
        artifact_dir = knowledge_dir / "rounds" / rid
        manifest = _latest_json_by_prefix(artifact_dir, "module_manifest")
        validity = _latest_json_by_prefix(artifact_dir, "module_validity_probe")
        experiment = _latest_json_by_prefix(artifact_dir, "run_experiment_result")
        gate = experiment.get("gate") if isinstance(experiment.get("gate"), dict) else {}
        run_record = _read_json(Path(str(experiment.get("run_record_path") or "")), {}) or {}
        run_result = experiment.get("run_result") if isinstance(experiment.get("run_result"), dict) else {}
        if not run_result and isinstance(run_record.get("run_result"), dict):
            run_result = run_record.get("run_result") or {}
        model_config = run_result.get("model_config") if isinstance(run_result.get("model_config"), dict) else {}
        model_entries = list(model_config.get("models") or []) if isinstance(model_config, dict) else []
        formal_model_entry = model_entries[0] if model_entries and isinstance(model_entries[0], dict) else {}
        artifact_provenance = (
            run_result.get("artifact_provenance")
            if isinstance(run_result.get("artifact_provenance"), dict)
            else {}
        )
        artifact_expected = (
            artifact_provenance.get("expected")
            if isinstance(artifact_provenance.get("expected"), dict)
            else {}
        )
        artifact_validation = (
            artifact_provenance.get("validation")
            if isinstance(artifact_provenance.get("validation"), dict)
            else {}
        )
        prediction_hashes: List[str] = []
        for record_item in artifact_validation.get("records") or []:
            if isinstance(record_item, dict):
                prediction_hashes.extend(str(item) for item in record_item.get("prediction_hashes") or [] if item)
        evaluation = record.get("evaluation") if isinstance(record.get("evaluation"), dict) else {}
        metrics = (
            record.get("metrics")
            or record.get("candidate_metrics")
            or evaluation.get("metrics")
            or experiment.get("metrics")
            or experiment.get("candidate_metrics")
            or {}
        )
        facts.append(
            {
                "id": rid,
                "status": record.get("status"),
                "variant_path": record.get("variant_path"),
                "variant_path_present": bool(record.get("variant_path")),
                "manifest_declared_components": list((manifest.get("internal_component_map") or {}).keys()),
                "runtime_detected_components": (
                    ((validity.get("component_traces") or {}).get("_runtime_probe") or {}).get("runtime_detected_components")
                    if isinstance(validity.get("component_traces"), dict)
                    else {}
                ),
                "module_validity_status": validity.get("status"),
                "module_validity_checks": validity.get("checks") or {},
                "failure_kind": validity.get("failure_kind"),
                "repair_target": validity.get("repair_target"),
                "experiment_success": bool(experiment.get("success")),
                "parsed_status": experiment.get("parsed_status") or experiment.get("status"),
                "evaluation_stage": experiment.get("evaluation_stage"),
                "evaluation_budget": experiment.get("evaluation_budget"),
                "gate_evaluation_stage": gate.get("evaluation_stage"),
                "gate_evaluation_budget": gate.get("evaluation_budget"),
                "build_mode": bool(experiment.get("build_mode")),
                "formal_model_config_variant_path": formal_model_entry.get("variant_path"),
                "formal_artifact_variant_path": artifact_expected.get("variant_path"),
                "formal_artifact_variant_source_sha256": artifact_expected.get("variant_source_sha256"),
                "formal_artifact_model_entry_hash": artifact_expected.get("model_entry_hash"),
                "formal_artifact_provenance_status": artifact_validation.get("status"),
                "formal_prediction_hashes": sorted(set(prediction_hashes)),
                "metrics": metrics,
                "mse_norm": metrics.get("mse_norm"),
            }
        )
    return facts


def _evaluate_acceptance(base_dir: str, task_id: str, *, rounds_per_attempt: int, baseline_model: str) -> Dict[str, Any]:
    knowledge_dir = task_knowledge_dir(base_dir, task_id)
    task_config = load_task_config(base_dir, task_id)
    dataset_profile = load_dataset_profile(base_dir, task_id) or {}
    ablation_root = knowledge_dir / "rounds"
    facts = _research_round_facts(base_dir, task_id)
    selected = facts[: int(rounds_per_attempt)]
    failed_kinds = [str(item.get("failure_kind") or "") for item in selected if item.get("failure_kind")]
    progress = round_progress(base_dir, task_id)
    checks = {
        "build_mode_enabled": bool(task_config.get("build_mode")),
        "baseline_model_matches": list(task_config.get("baseline_models") or []) == [str(baseline_model)],
        "dataset_diagnosis_skipped": str(task_config.get("dataset_diagnosis_mode") or "") == "skip"
        and str(dataset_profile.get("status") or "") == "skipped",
        "ablation_targets_zero": int(task_config.get("baseline_diagnosis_max_ablation_targets") or 0) == 0,
        "no_ablation_artifacts": not ablation_root.exists() or not any(ablation_root.glob("Ablation*")),
        "no_diagnostic_rounds": int(progress.get("diagnostic_rounds") or 0) == 0
        and int(progress.get("diagnostic_terminal_rounds") or 0) == 0,
        "research_round_count": len(selected) == int(rounds_per_attempt),
        "all_variants_present": all(item.get("variant_path_present") for item in selected),
        "all_module_valid": all(item.get("module_validity_status") == "module_valid" for item in selected),
        "all_module_validity_audited": all(bool(str(item.get("module_validity_status") or "").strip()) for item in selected),
        "all_entered_experiment": all(item.get("experiment_success") for item in selected),
        "all_parsed_status_ok": all(str(item.get("parsed_status") or "") == "ok" for item in selected),
        "all_evaluation_stage_build_mode": all(str(item.get("evaluation_stage") or "") == "build_mode" for item in selected),
        "all_gate_stage_build_mode": all(str(item.get("gate_evaluation_stage") or "") == "build_mode" for item in selected),
        "all_evaluation_budget_build_mode": all(str(item.get("evaluation_budget") or "") == "build_mode" for item in selected)
        and all(str(item.get("gate_evaluation_budget") or "") == "build_mode" for item in selected),
        "all_result_build_mode": all(bool(item.get("build_mode")) for item in selected),
        "all_mse_norm_present": all(isinstance(item.get("mse_norm"), (int, float)) for item in selected),
        "all_formal_model_config_variant_bound": all(
            bool(str(item.get("formal_model_config_variant_path") or "").strip()) for item in selected
        ),
        "all_formal_artifact_variant_bound": all(
            bool(str(item.get("formal_artifact_variant_path") or "").strip())
            and bool(str(item.get("formal_artifact_variant_source_sha256") or "").strip())
            and bool(str(item.get("formal_artifact_model_entry_hash") or "").strip())
            for item in selected
        ),
        "all_formal_artifact_provenance_ok": all(
            str(item.get("formal_artifact_provenance_status") or "") == "ok" for item in selected
        ),
        "all_formal_prediction_hash_present": all(
            bool(item.get("formal_prediction_hashes")) for item in selected
        ),
        "all_formal_model_entry_hash_unique": len(
            {
                str(item.get("formal_artifact_model_entry_hash") or "")
                for item in selected
                if str(item.get("formal_artifact_model_entry_hash") or "").strip()
            }
        )
        == len(selected),
        "no_manifest_code_mismatch": "manifest_code_mismatch" not in failed_kinds,
        "no_component_not_called": "component_not_called" not in failed_kinds,
        "no_component_no_gradient": "component_no_gradient" not in failed_kinds,
    }
    blocking_check_names = [
        key for key in checks
        if key not in {
            "all_module_valid",
            "no_manifest_code_mismatch",
            "no_component_not_called",
            "no_component_no_gradient",
        }
    ]
    return {
        "status": "passed" if all(bool(checks.get(key)) for key in blocking_check_names) else "failed",
        "task_id": task_id,
        "rounds_per_attempt": int(rounds_per_attempt),
        "checks": checks,
        "blocking_checks": blocking_check_names,
        "rounds": selected,
        "progress": progress,
        "created_at": datetime.now().isoformat(),
    }


def _run_agent(task_id: str, api_config: str, build_contract: str) -> int:
    cmd = [
        sys.executable,
        "-m",
        "evocast.scripts.run_agent_v3",
        "--task-id",
        task_id,
        "--api-config",
        api_config,
        "--build-contract",
        build_contract,
        "--build-mode",
    ]
    return subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[2])).returncode


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run formal Buildmode module-research acceptance.")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--build-mode", type=_bool_arg, default=True)
    parser.add_argument("--skip-dataset-diagnosis", type=_bool_arg, default=True)
    parser.add_argument("--skip-baseline-diagnosis", type=_bool_arg, default=True)
    parser.add_argument("--ablation-targets", type=int, default=0)
    parser.add_argument("--rounds-per-attempt", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--api-config", default="providers/deepseek.yaml")
    parser.add_argument("--build-contract", default="")
    parser.add_argument("--baseline-model", default="FiLM")
    parser.add_argument("--dataset", default="dataset/ETTh1.csv")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)

    if not args.build_mode:
        raise SystemExit("Buildmode acceptance requires --build-mode true.")
    if not args.skip_dataset_diagnosis:
        raise SystemExit("Buildmode acceptance requires --skip-dataset-diagnosis true.")
    if not args.skip_baseline_diagnosis:
        raise SystemExit("Buildmode acceptance requires --skip-baseline-diagnosis true.")
    if int(args.ablation_targets) != 0:
        raise SystemExit("Buildmode acceptance requires --ablation-targets 0.")
    if int(args.rounds_per_attempt) != 2:
        raise SystemExit("Buildmode acceptance runs in 2-round units.")

    base_dir = str(runtime_root(args.base_dir))
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(args.base_dir).resolve() / dataset_path
    if not dataset_path.is_file():
        raise SystemExit(f"dataset not found: {dataset_path}")

    reports: List[Dict[str, Any]] = []
    for attempt in range(1, int(args.max_attempts) + 1):
        attempt_task_id = args.task_id if attempt == 1 else f"{args.task_id}_attempt{attempt:02d}"
        _ensure_task(
            base_dir=base_dir,
            task_id=attempt_task_id,
            dataset_path=dataset_path,
            seq_len=int(args.seq_len),
            horizon=int(args.horizon),
            baseline_model=str(args.baseline_model),
            rounds_per_attempt=int(args.rounds_per_attempt),
            api_config=str(args.api_config),
        )
        if not args.prepare_only:
            if not str(args.build_contract or "").strip():
                raise SystemExit(
                    "Buildmode acceptance no longer launches the removed legacy agent loop. "
                    "Pass --build-contract to run the new BuildContract path."
                )
            rc = _run_agent(attempt_task_id, str(args.api_config), str(args.build_contract))
            if rc != 0:
                reports.append({"task_id": attempt_task_id, "status": "agent_failed", "returncode": rc})
            else:
                reports.append(_evaluate_acceptance(
                    base_dir,
                    attempt_task_id,
                    rounds_per_attempt=int(args.rounds_per_attempt),
                    baseline_model=str(args.baseline_model),
                ))
        else:
            reports.append({"task_id": attempt_task_id, "status": "prepared"})
        if reports[-1].get("status") == "passed":
            break

    summary = {
        "status": "passed" if any(item.get("status") == "passed" for item in reports) else ("prepared" if args.prepare_only else "failed"),
        "base_dir": base_dir,
        "reports": reports,
        "created_at": datetime.now().isoformat(),
    }
    out_path = task_knowledge_dir(base_dir, args.task_id) / "buildmode_research_acceptance_report.json"
    _write_json(out_path, summary)
    print(json.dumps({**summary, "report_path": str(out_path)}, indent=2, ensure_ascii=False, default=str))
    return 0 if summary["status"] in {"passed", "prepared"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
