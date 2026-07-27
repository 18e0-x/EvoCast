"""Pure TFB candidate execution and metric-artifact validation."""

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any

from evocast.domain.metric_parser import parse_metrics_from_paths
from evocast.domain.result_provenance import (
    build_result_provenance,
    stamp_result_artifacts,
    validate_result_artifact_provenance,
)
from evocast.policy.agent_control_policy import (
    build_mode_policy,
    execution_timeout_policy,
)
from evocast.policy.error_taxonomy import classify_from_result
from evocast.runners.tfb_pipeline_runner import build_run_configs, run_pipeline


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    return value


def resolved_path_text(path: str) -> str:
    return str(Path(str(path)).expanduser().resolve())


def require_variant_model_entry_binding(
    model_entry: dict[str, Any],
    variant_path: str,
    *,
    stage: str,
) -> None:
    expected = str(variant_path or "").strip()
    actual = str(model_entry.get("variant_path") or "").strip()
    if not expected:
        raise RuntimeError(f"{stage}: missing canonical variant_path")
    if not actual:
        raise RuntimeError(f"{stage}: model entry lost variant_path binding")
    if resolved_path_text(actual) != resolved_path_text(expected):
        raise RuntimeError(
            f"{stage}: model entry variant_path mismatch: {actual!r} != {expected!r}"
        )


def require_formal_model_config_binding(
    model_config: dict[str, Any],
    variant_path: str,
    *,
    stage: str,
) -> None:
    models = list(model_config.get("models") or [])
    if not models or not isinstance(models[0], dict):
        raise RuntimeError(f"{stage}: formal model_config has no model entry")
    require_variant_model_entry_binding(
        dict(models[0]),
        variant_path,
        stage=stage,
    )


def execute_variant(
    *,
    base_dir: str,
    task_id: str,
    run_id: str,
    candidate_id: str,
    candidate_kind: str,
    tfb_config: dict[str, Any],
    variant_entry: dict[str, Any],
    objective_metric: str,
    save_path: str,
    seed: int,
    evaluation_budget: str,
    build_mode: bool,
    source_checkout: str | None = None,
    source_entry_file: str | None = None,
    build_run_configs_fn=build_run_configs,
    run_pipeline_fn=run_pipeline,
    parse_metrics_fn=parse_metrics_from_paths,
    build_provenance_fn=build_result_provenance,
    stamp_artifacts_fn=stamp_result_artifacts,
    validate_provenance_fn=validate_result_artifact_provenance,
    classify_result_fn=classify_from_result,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
    """Execute one candidate without writing round, gate, or decision state."""

    variant_path = str(variant_entry.get("variant_path") or "").strip()
    provenance_variant_path = variant_path or str(source_entry_file or "").strip()
    result_provenance: dict[str, Any] = {}
    try:
        if candidate_kind == "variant":
            require_variant_model_entry_binding(
                variant_entry,
                variant_path,
                stage="execute_variant.model_entry",
            )
        result_provenance = build_provenance_fn(
            task_id=task_id,
            run_id=run_id,
            candidate_id=candidate_id,
            candidate_kind=candidate_kind,
            model_entry=variant_entry,
            evaluation_budget=evaluation_budget,
            build_mode=build_mode,
            variant_path=provenance_variant_path or None,
        )
        override_eval_args: dict[str, Any] = {"save_true_pred": True}
        if str(evaluation_budget or "").strip().lower() == "smoke_precheck":
            strategy_args = dict(
                (tfb_config.get("evaluation_config") or {}).get("strategy_args")
                or {}
            )
            horizon = int(strategy_args.get("horizon") or 1)
            override_eval_args.update(
                {
                    "num_rollings": int(
                        build_mode_policy(base_dir).get("num_rollings") or 1
                    ),
                    "stride": horizon,
                }
            )
        data_config, model_config, evaluation_config = build_run_configs_fn(
            tfb_config,
            [variant_entry],
            save_path=save_path,
            seed=seed,
            override_eval_args=override_eval_args,
        )
        if candidate_kind == "variant":
            require_formal_model_config_binding(
                model_config,
                variant_path,
                stage="execute_variant.model_config",
            )
        run_result = run_pipeline_fn(
            data_config,
            model_config,
            evaluation_config,
            timeout=int(execution_timeout_policy(base_dir)["metric_pipeline"]),
            source_checkout=source_checkout,
        )
    except Exception as exc:
        run_result = {
            "success": False,
            "log_paths": [],
            "error": exc,
            "error_traceback": traceback.format_exc(),
            "elapsed_seconds": 0,
        }

    log_paths = list(run_result.get("log_paths") or [])
    if log_paths:
        parsed = parse_metrics_fn(
            log_paths,
            objective_metric=objective_metric,
        )
        metrics = dict(parsed.get("metric_values") or {})
        record_errors = [
            str(item)
            for item in list(parsed.get("record_errors") or [])
            if str(item)
        ]
        if record_errors:
            run_result["success"] = False
            run_result["error"] = RuntimeError(record_errors[0])
            run_result["error_traceback"] = "\n\n".join(record_errors)
            parsed["metric_values"] = {}
            parsed["status"] = "error"
            parsed["error_type"] = "runtime_error"
            metrics = {}
            run_result["artifact_provenance"] = {
                "expected": result_provenance,
                "validation": {
                    "status": "skipped",
                    "reason": "runtime_error_record",
                    "message": (
                        "TFB result record contains a runtime error; provenance "
                        "validation applies only to successful metric artifacts."
                    ),
                },
            }
        else:
            stamped = stamp_artifacts_fn(log_paths, result_provenance)
            validation = validate_provenance_fn(
                log_paths,
                result_provenance,
                require_prediction_hash=bool(provenance_variant_path),
                require_batch_forecast=(
                    str(
                        (
                            (tfb_config.get("evaluation_config") or {}).get(
                                "strategy_args"
                            )
                            or {}
                        ).get("strategy_name")
                        or ""
                    )
                    == "rolling_forecast"
                ),
            )
            run_result["artifact_provenance"] = {
                "expected": result_provenance,
                "stamped_records": stamped,
                "validation": validation,
            }
            if validation.get("status") != "ok":
                message = "; ".join(
                    str(item) for item in validation.get("failures") or []
                )
                warnings = list(parsed.get("warnings") or [])
                warnings.append("result artifact provenance validation failed")
                parsed["warnings"] = warnings
                parsed["artifact_provenance"] = validation
                if run_result.get("success"):
                    run_result["success"] = False
                    run_result["error"] = RuntimeError(
                        "result artifact provenance validation failed: "
                        + message
                    )
                    run_result["error_traceback"] = json.dumps(
                        validation,
                        ensure_ascii=False,
                        default=str,
                    )
                    parsed["metric_values"] = {}
                    parsed["status"] = "error"
                    metrics = {}
    else:
        parsed = {"metric_values": {}, "status": "error", "warnings": []}
        metrics = {}

    if (
        not run_result.get("success")
        and run_result.get("artifact_provenance")
        and not parsed.get("record_errors")
    ):
        label_value = "evaluator_error"
    else:
        label = classify_result_fn(
            success=bool(run_result.get("success")),
            error=run_result.get("error"),
            metrics=metrics,
            objective_metric=objective_metric,
        )
        label_value = label.value
    if label_value == "metric_missing" and parsed.get("record_errors"):
        run_result["error_traceback"] = "\n\n".join(parsed.get("record_errors", []))
        run_result["error"] = RuntimeError(
            parsed.get("record_errors", ["TFB record contains runtime error"])[0]
        )
        label_value = classify_result_fn(
            False,
            error=run_result.get("error"),
        ).value
    return (
        _json_safe(run_result),
        _json_safe(parsed),
        _json_safe(metrics),
        label_value,
    )
