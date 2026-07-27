from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evocast.build.contract import BuildContract
from evocast.build.result import BuildDecision, VerificationResult
from evocast.domain.atomic_io import atomic_write_json
from evocast.harness.session import AgentSession
from evocast.policy.experiment_policy import baseline_seed
from evocast.tools.tfb_experiment import run_experiment
from evocast.state.cost_ledger import tracked_stage


def _norm(path: str) -> str:
    return str(path or "").replace("\\", "/").strip().lstrip("./")


def _metric_model_entry(contract: BuildContract) -> dict[str, Any]:
    protocol = dict(contract.metric_protocol or {})
    baseline = dict(protocol.get("baseline") or {})
    entry = dict(protocol.get("model_config") or baseline.get("model_config") or {})
    model_name = str(entry.get("model_name") or baseline.get("import_path") or baseline.get("model_name") or contract.target_model or "").strip()
    if not model_name:
        raise ValueError("BuildContract.metric_protocol.model_config.model_name is required for metric execution")
    entry["model_name"] = model_name
    if "adapter" not in entry and baseline.get("adapter") is not None:
        entry["adapter"] = baseline.get("adapter")
    effective = entry.get("effective_model_hyper_params")
    if isinstance(effective, dict) and effective:
        entry["model_hyper_params"] = dict(effective)
    return entry


def _source_entry_file(checkout_dir: Path, contract: BuildContract) -> str:
    protocol = dict(contract.metric_protocol or {})
    baseline = dict(protocol.get("baseline") or {})
    source_ref = dict(contract.base_source_ref or {})
    source_binding = dict(source_ref.get("source_binding") or {})
    candidates = [
        protocol.get("source_entry_file"),
        protocol.get("entry_file"),
        baseline.get("entry_file"),
        source_binding.get("entry_file"),
    ]
    model_name = str(
        (protocol.get("model_config") or {}).get("model_name")
        or baseline.get("import_path")
        or baseline.get("model_name")
        or contract.target_model
        or ""
    ).strip()
    short_name = str(contract.target_model or model_name.rsplit(".", 1)[-1]).strip()
    if short_name:
        candidates.append(f"ts_benchmark/baselines/time_series_library/models/{short_name}.py")
    for item in candidates:
        text = _norm(str(item or ""))
        if not text:
            continue
        path = (checkout_dir / text).resolve()
        try:
            path.relative_to(checkout_dir.resolve())
        except ValueError:
            continue
        if path.is_file():
            return str(path)
    return ""


class TFBExperimentMetricRunner:
    def __init__(self, *, session: AgentSession, budget: str = "unified", seed: int | None = None) -> None:
        self.session = session
        self.budget = budget
        self.seed = int(seed if seed is not None else baseline_seed(session.base_dir))

    @tracked_stage(
        "canonical_metric",
        lambda self, checkout_dir, contract, output_dir: (
            self.session.base_dir,
            self.session.task_id,
            str(contract.research_id),
            str(output_dir.name),
        ),
    )
    def __call__(self, checkout_dir: Path, contract: BuildContract, output_dir: Path) -> VerificationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            model_entry = _metric_model_entry(contract)
            source_entry_file = _source_entry_file(checkout_dir, contract)
            objective_metric = str((contract.metric_protocol or {}).get("objective_metric") or "mse_norm")
            stage = str((contract.metric_protocol or {}).get("evaluation_stage") or "build_mode")
            result = run_experiment(
                self.session,
                {
                    "model_name": model_entry.get("model_name"),
                    "adapter": model_entry.get("adapter"),
                    "model_hyper_params": dict(model_entry.get("model_hyper_params") or {}),
                    "source_checkout": str(checkout_dir.resolve()),
                    "source_entry_file": source_entry_file,
                    "objective_metric": objective_metric,
                    "budget": self.budget,
                    "smoke": stage == "smoke",
                    "seed": self.seed,
                    "fit_point": contract.target_model,
                    "round_id": contract.research_id,
                    "orchestrator_owns_round_terminal": True,
                },
            )
            payload: dict[str, Any] = {
                "status": "success" if result.get("success") and result.get("metrics") else "failed",
                "source_checkout": str(checkout_dir.resolve()),
                "source_entry_file": source_entry_file,
                "model_entry": model_entry,
                "objective_metric": objective_metric,
                "result": result,
            }
            path = output_dir / "metric_result.json"
            atomic_write_json(path, payload, ensure_ascii=False, default=str)
            if result.get("success") and result.get("metrics"):
                metrics = dict(result.get("metrics") or {})
                return VerificationResult(
                    status=BuildDecision.METRIC_COMPLETED,
                    reason_code="metric_completed",
                    summary=f"Metric execution completed for {model_entry.get('model_name')} from candidate checkout.",
                    artifact_paths=[str(path), *list(result.get("log_paths") or [])],
                    metrics=metrics,
                    raw_payload=payload,
                )
            failure_payload = {
                "stage": stage,
                "error_type": result.get("error_type") or result.get("parsed_status") or "metric_failed",
                "error_message": result.get("error_message") or result.get("error") or "Metric execution failed or produced no metrics.",
                "traceback": result.get("error_traceback") or result.get("full_traceback_excerpt") or "",
                "failure_evidence": result.get("failure_evidence") or {},
                "source_checkout": str(checkout_dir.resolve()),
                "source_entry_file": source_entry_file,
                "artifact_provenance": result.get("artifact_provenance") or {},
                "parsed_status": result.get("parsed_status"),
                "changed_files": list((contract.metric_protocol or {}).get("changed_files") or []),
            }
            return VerificationResult(
                status=BuildDecision.REPAIR_REQUIRED,
                reason_code=str(failure_payload["error_type"]),
                summary=str(failure_payload["error_message"]),
                repair_instructions=[
                    "Repair the candidate source using the canonical smoke/build-mode metric failure.",
                    f"Failure stage: {stage}.",
                    f"Error type: {failure_payload['error_type']}.",
                    "Use the traceback and failure_evidence below as the primary repair target.",
                    "Do not repair provenance, evaluator, data, metric, task semantics, or training policy unless the traceback directly points there.",
                ],
                artifact_paths=[str(path), *list(result.get("log_paths") or [])],
                raw_payload=failure_payload,
            )
        except Exception as exc:
            path = output_dir / "metric_exception.json"
            atomic_write_json(
                path,
                {"status": "failed", "error": str(exc), "type": type(exc).__name__},
                ensure_ascii=False,
            )
            return VerificationResult(
                status=BuildDecision.REPAIR_REQUIRED,
                reason_code="metric_runner_exception",
                summary=f"{type(exc).__name__}: {exc}",
                repair_instructions=[
                    "Repair or retry using the metric runner exception artifact.",
                    "The round remains a failed dynamic research attempt if the repair budget is exhausted.",
                ],
                artifact_paths=[str(path)],
                raw_payload={"stage": "canonical_metric", "error_type": type(exc).__name__, "error_message": str(exc)},
            )
