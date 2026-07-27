from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evocast.build.contract import BuildContract


def _short_text(value: Any, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _string_list(values: Any, *, max_items: int, max_chars: int) -> list[str]:
    result: list[str] = []
    for value in list(values or []):
        text = _short_text(value, max_chars=max_chars)
        if text:
            result.append(text)
        if len(result) >= max_items:
            break
    return result


def _dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _metric_value(metrics: dict[str, Any], objective_metric: str) -> Any:
    if objective_metric and objective_metric in metrics:
        return metrics.get(objective_metric)
    for key in ("mse_norm", "mae_norm", "mse", "mae"):
        if key in metrics:
            return metrics.get(key)
    return None


def _model_hyper_params(model_config: dict[str, Any]) -> dict[str, Any]:
    params = _dict(model_config.get("effective_model_hyper_params"))
    if not params:
        params = _dict(model_config.get("model_hyper_params"))
    keys = [
        "input_chunk_length",
        "output_chunk_length",
        "seq_len",
        "pred_len",
        "horizon",
        "label_len",
        "enc_in",
        "dec_in",
        "c_out",
        "d_model",
        "n_heads",
        "e_layers",
        "d_layers",
        "d_ff",
        "batch_size",
        "num_epochs",
        "patience",
        "lr",
        "num_workers",
        "task_name",
    ]
    return {key: params[key] for key in keys if key in params}


def _compact_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": model_config.get("model_name"),
        "model_key": model_config.get("model_key"),
        "adapter": model_config.get("adapter"),
        "model_hyper_params": _model_hyper_params(model_config),
    }


def _target_channel_count_from_task(semantics: dict[str, Any], metric_semantics: dict[str, Any], params: dict[str, Any]) -> int | None:
    target_columns = _first_present(semantics.get("target_columns"), metric_semantics.get("target_columns"))
    if isinstance(target_columns, list) and target_columns:
        return len(target_columns)
    target_channel = _first_present(semantics.get("target_channel"), metric_semantics.get("target_channel"))
    if isinstance(target_channel, list) and target_channel:
        count = 0
        input_channels = int(params.get("enc_in") or params.get("c_out") or 0)
        for item in target_channel:
            if isinstance(item, int) and not isinstance(item, bool):
                count += 1
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    start = int(item[0])
                    end = int(item[1])
                except Exception:
                    continue
                if start < 0 and input_channels:
                    start = max(0, input_channels + start)
                if end < 0 and input_channels:
                    end = max(0, input_channels + end)
                count += max(0, end - start)
        if count:
            return count
    return None


def _baseline_summary(metric_protocol: dict[str, Any]) -> dict[str, Any]:
    baseline = _dict(metric_protocol.get("baseline"))
    metric_semantics = _dict(baseline.get("metric_semantics"))
    metrics = _dict(_first_present(baseline.get("metrics"), metric_semantics.get("metrics")))
    objective_metric = str(_first_present(metric_protocol.get("objective_metric"), metric_semantics.get("objective_metric"), "mse_norm"))
    keep_metrics = {}
    for key in [objective_metric, "mse_norm", "mae_norm", "mse", "mae"]:
        if key and key in metrics:
            keep_metrics[key] = metrics[key]
    return {
        "candidate_id": baseline.get("candidate_id"),
        "model_name": baseline.get("model_name"),
        "objective_metric": objective_metric,
        "metrics": keep_metrics,
        "model_config": _compact_model_config(_dict(baseline.get("model_config"))),
    }


def _source_summary(contract: BuildContract, metric_protocol: dict[str, Any]) -> dict[str, Any]:
    source_binding = _dict(_first_present(metric_protocol.get("source_binding"), contract.base_source_ref.get("source_binding")))
    allowed_edit_files = (
        {"mode": "repo_wide", "count": len(contract.allowed_edit_files)}
        if str(contract.execution_authority or "") == "repo_wide"
        else list(contract.allowed_edit_files)
    )
    allowed_new_file_roots = (
        {"mode": "repo_wide"}
        if str(contract.execution_authority or "") == "repo_wide"
        else list(contract.allowed_new_file_roots)
    )
    return {
        "source_mode": contract.source_mode,
        "execution_authority": contract.execution_authority,
        "entry_file": source_binding.get("entry_file"),
        "source_root": source_binding.get("source_root"),
        "allowed_edit_files": allowed_edit_files,
        "allowed_new_file_roots": allowed_new_file_roots,
        "primary_edit_files": _primary_edit_files(contract, metric_protocol),
    }


def _source_checkout_path(contract: BuildContract, metric_protocol: dict[str, Any]) -> Path | None:
    raw = _first_present(
        _dict(contract.base_source_ref).get("source_checkout"),
        _dict(metric_protocol.get("base_source_ref")).get("source_checkout"),
    )
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_dir() else None


def _dedupe_preserve_order(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _research_direction(metric_protocol: dict[str, Any]) -> dict[str, Any]:
    return _dict(metric_protocol.get("research_direction"))


def _compact_surrounding_contract(contract: BuildContract, *, keep_research_direction: bool = True) -> dict[str, Any]:
    metric_protocol = _dict(contract.metric_protocol)
    payload: dict[str, Any] = {
        "schema": "evocast_build_contract_compact_v1",
        "research_id": contract.research_id,
        "target_model": contract.target_model,
        "semantic_goal": contract.semantic_goal,
        "hypothesis": contract.hypothesis,
        "source": _source_summary(contract, metric_protocol),
        "task_semantics": _task_semantics(contract, metric_protocol),
        "shape_contract": _shape_contract(contract, _task_semantics(contract, metric_protocol)),
        "baseline": _baseline_summary(metric_protocol),
        "discovery_hints": _string_list(contract.discovery_hints, max_items=12, max_chars=500),
        "required_behavior": _string_list(contract.required_behavior, max_items=8, max_chars=500),
        "forbidden_behavior": _string_list(contract.forbidden_behavior, max_items=8, max_chars=500),
        "implementation_constraints": _string_list(contract.implementation_constraints, max_items=10, max_chars=500),
        "internal_check_commands": _dedupe_preserve_order(list(contract.internal_check_commands)),
        "finish_metadata_requirements": [
            "display_idea",
            "idea_summary",
            "mechanism_name",
            "changed_mechanism",
            "expected_effect",
        ],
    }
    if keep_research_direction:
        payload["research_direction"] = _research_direction(metric_protocol)
    return payload


def build_dedup_conservative_message(contract: BuildContract) -> str:
    payload = _compact_surrounding_contract(contract, keep_research_direction=True)
    return (
        "Implement this EvoCast research build contract in the current workspace.\n"
        "This is a compacted contract: research semantics are preserved, duplicated audit/source snapshot fields are removed.\n"
        "Inspect the primary edit file first, make a concrete source edit, run checks, then finish.\n"
        "<compacted_build_contract>\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n</compacted_build_contract>\n"
    )


def build_dedup_guided_finish_message(contract: BuildContract) -> str:
    payload = _compact_surrounding_contract(contract, keep_research_direction=True)
    payload["execution_discipline"] = [
        "Read the primary edit file first and make the first source edit within the first four tool calls whenever the file exists.",
        "Do not inspect broad framework internals before the first edit; use supporting reads only for a concrete symbol or shape uncertainty.",
        "After editing, run internal checks immediately. If checks pass, finish; do not keep browsing for optional refinements.",
        "If max tool turns are limited, a small correct mechanism is preferable to unfinished exploration.",
    ]
    payload["timekan_runtime_shape_guard"] = [
        "At the start of TimeKANModeL.forecast, save the original tensor input, e.g. x_enc_raw = x_enc, before any multi-scale processing.",
        "If TimeKAN later converts x_enc into a list of multi-resolution tensors, do not call .size(), .shape, mean, std, FFT, or tensor ops on that list.",
        "Use x_enc_raw.size(0) for batch B and configs.enc_in/configs.c_out for channel count C; never infer C from d_model or a descriptor width.",
        "When adding a module around enc_out_list, remember enc_out_list is a list; operate on a selected tensor such as enc_out_list[0] or loop over its tensor elements.",
    ]
    return (
        "Implement this EvoCast research build contract in the current workspace.\n"
        "This is a compacted contract with preserved research semantics and explicit execution discipline for finishing within the tool budget.\n"
        "Inspect the primary edit file first, make a concrete source edit, run checks, then finish.\n"
        "<compacted_guided_build_contract>\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n</compacted_guided_build_contract>\n"
    )


def _timekan_anatomy_capsule() -> dict[str, Any]:
    return {
        "model": "TimeKAN",
        "primary_class": "TimeKANModeL",
        "forecast_flow": [
            "forecast(x_enc) receives a tensor shaped (B, seq_len, C).",
            "The original code immediately does x_enc = self.__multi_level_process_inputs(x_enc); after that x_enc is a list of tensors.",
            "Each level tensor x has shape (B, T_i, C); it is normalized and reshaped to (B*C, T_i, 1).",
            "enc_embedding maps each level to (B*C, T_i, d_model).",
            "res_blocks/add_blocks consume and return enc_out_list, a list of tensors.",
            "The head uses enc_out_list[0], applies predict_layer over time to get (B*C, pred_len, d_model), then projection_layer to (B*C, pred_len, 1).",
            "The final reshape is (B, c_out, pred_len).permute(0, 2, 1), so output stays (B, pred_len, c_out).",
        ],
        "safe_edit_points": [
            "Add new nn.Module classes before class TimeKANModeL(nn.Module), leaving the TimeKANModeL class header intact.",
            "Instantiate new modules in TimeKANModeL.__init__ after predict_layer/projection_layer definitions or near related layers.",
            "For head-side mechanisms, operate on the tensor after predict_layer before projection_layer, or on the final (B, pred_len, c_out) tensor before denorm.",
            "For encoder-side mechanisms, operate on each tensor in enc_out_list or on enc_out_list[0], never on the list object itself.",
        ],
        "shape_hazards": [
            "Save x_enc_raw = x_enc at the start of forecast if raw input statistics are needed; later x_enc is a list.",
            "Do not call .size(), .shape, mean, std, fft, permute, reshape, or projection modules on enc_out_list or x_enc after they became lists.",
            "Recover batch B from x_enc_raw.size(0) or from the per-level tensor before flattening; recover channel C from configs.enc_in/configs.c_out.",
            "Do not infer C from d_model. d_model is hidden width, not variable count.",
        ],
        "constructor_hazards": [
            "Do not duplicate or concatenate class headers such as 'class TimeKANModeL(...):class NewModule(...)'.",
            "Preserve TimeKANModeL.__init__, forecast, and forward signatures.",
            "Prefer plain nn.Linear, nn.LayerNorm, nn.GELU, nn.Dropout, nn.Parameter, torch.einsum, torch.softmax for new modules.",
            "If reusing local BasicConv, verify its current signature in the file before constructing it; otherwise use torch.nn.Conv1d directly.",
        ],
    }


def build_dedup_with_timekan_anatomy_message(contract: BuildContract) -> str:
    payload = _compact_surrounding_contract(contract, keep_research_direction=True)
    payload["timekan_source_anatomy"] = _timekan_anatomy_capsule()
    return (
        "Implement this EvoCast research build contract in the current workspace.\n"
        "This is a compacted contract with preserved research semantics plus a concise TimeKAN source-anatomy capsule.\n"
        "Use the anatomy capsule to avoid broad source browsing and to preserve tensor/list shape semantics.\n"
        "<compacted_anatomy_build_contract>\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n</compacted_anatomy_build_contract>\n"
    )


def _source_skeleton_for_file(path: Path, *, rel_path: str, max_lines: int = 90) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    selected: list[tuple[int, str]] = []
    seen: set[int] = set()

    def add(line_no: int) -> None:
        if line_no < 1 or line_no > len(lines) or line_no in seen:
            return
        text = lines[line_no - 1].rstrip()
        if text.strip():
            selected.append((line_no, text))
            seen.add(line_no)

    structural_prefixes = ("class ", "def ", "async def ")
    keywords = [
        "self.",
        "nn.",
        "ModuleList",
        "Sequential",
        "Linear",
        "Conv",
        "GRU",
        "LSTM",
        "Embedding",
        "forecast(",
        "forward(",
        ".size(",
        ".shape",
        ".reshape(",
        ".view(",
        ".permute(",
        ".transpose(",
        ".contiguous(",
        "torch.cat",
        "torch.stack",
        "torch.einsum",
        "softmax",
        "return ",
    ]

    for idx, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith(structural_prefixes):
            add(idx)
            continue
        if any(token in line for token in keywords):
            add(idx)
        if len(selected) >= max_lines:
            break

    return {
        "file": rel_path,
        "total_lines": len(lines),
        "skeleton_lines": [f"{line_no}: {text}" for line_no, text in selected[:max_lines]],
    }


def _source_skeleton_capsule(contract: BuildContract, metric_protocol: dict[str, Any]) -> dict[str, Any]:
    checkout = _source_checkout_path(contract, metric_protocol)
    primary_files = _primary_edit_files(contract, metric_protocol)
    files: list[dict[str, Any]] = []
    if checkout is not None:
        for rel_path in primary_files[:2]:
            path = checkout / rel_path
            if path.is_file() and path.suffix == ".py":
                files.append(_source_skeleton_for_file(path, rel_path=rel_path))
    return {
        "schema": "auto_source_skeleton_v1",
        "purpose": "Compact local source structure extracted from primary edit files. It is generated from source, not a hand-written model-specific hint.",
        "use": [
            "Use this skeleton to choose the first edit location before broad browsing.",
            "Read the full file only when exact anchors or nearby code are needed.",
            "Preserve method signatures and output shape contracts shown in the BuildContract.",
        ],
        "files": files,
    }


def build_dedup_with_source_skeleton_message(contract: BuildContract) -> str:
    metric_protocol = _dict(contract.metric_protocol)
    payload = _compact_surrounding_contract(contract, keep_research_direction=True)
    payload["auto_source_skeleton"] = _source_skeleton_capsule(contract, metric_protocol)
    return (
        "Implement this EvoCast research build contract in the current workspace.\n"
        "This is a compacted contract with preserved research semantics plus an automatically extracted source skeleton from primary edit files.\n"
        "The skeleton is navigational context only; the BuildContract semantics and shape contract remain authoritative.\n"
        "<compacted_source_skeleton_build_contract>\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n</compacted_source_skeleton_build_contract>\n"
    )


def build_research_full_surrounding_compact_message(contract: BuildContract) -> str:
    payload = _compact_surrounding_contract(contract, keep_research_direction=True)
    direction = payload.pop("research_direction", {})
    return (
        "Implement this EvoCast research round. The selected research_direction below is authoritative and preserved in full.\n"
        "Surrounding source snapshot, baseline snapshot, and repeated model-config audit fields have been compacted.\n"
        "<execution_context>\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n</execution_context>\n"
        "<research_direction_full>\n"
        + json.dumps(direction, ensure_ascii=False, indent=2)
        + "\n</research_direction_full>\n"
    )


def build_two_stage_research_message(contract: BuildContract) -> str:
    metric_protocol = _dict(contract.metric_protocol)
    summary = {
        "round_id": contract.research_id,
        "target_model": contract.target_model,
        "primary_edit_files": _primary_edit_files(contract, metric_protocol),
        "task_semantics": _task_semantics(contract, metric_protocol),
        "implementation_goal": _implementation_goal(contract, metric_protocol),
        "shape_contract": _shape_contract(contract, _task_semantics(contract, metric_protocol)),
        "execution_rules": [
            "Start from primary_edit_files.",
            "Preserve model class methods and runtime shape.",
            "A concrete source edit is required.",
            "Run internal checks before finish.",
        ],
    }
    payload = _compact_surrounding_contract(contract, keep_research_direction=False)
    return (
        "Implement this EvoCast research build. First follow the execution summary, then use the full research direction for mechanism details.\n"
        "<execution_summary>\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n</execution_summary>\n"
        "<research_direction_full>\n"
        + json.dumps(_research_direction(metric_protocol), ensure_ascii=False, indent=2)
        + "\n</research_direction_full>\n"
        "<compact_context>\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n</compact_context>\n"
    )


def build_natural_language_first_message(contract: BuildContract) -> str:
    metric_protocol = _dict(contract.metric_protocol)
    goal = _implementation_goal(contract, metric_protocol)
    task = _task_semantics(contract, metric_protocol)
    direction = _research_direction(metric_protocol)
    context = {
        "round_id": contract.research_id,
        "target_model": contract.target_model,
        "source": _source_summary(contract, metric_protocol),
        "task_semantics": task,
        "shape_contract": _shape_contract(contract, task),
        "avoid": _avoid_list(contract, metric_protocol),
        "internal_check_commands": _dedupe_preserve_order(list(contract.internal_check_commands)),
    }
    prose = "\n".join(
        [
            f"Round: {contract.research_id}. Target model: {contract.target_model}.",
            f"Mechanism to implement: {goal.get('mechanism_name')}",
            f"Target code region: {goal.get('target_code_region')}",
            f"Required change: {goal.get('required_change')}",
            f"Hypothesis: {direction.get('hypothesis') or contract.hypothesis}",
            f"Expected effect: {direction.get('expected_behavior_delta') or goal.get('expected_effect')}",
            f"Failure to avoid: {direction.get('failure_to_avoid') or ''}",
        ]
    )
    return (
        "Implement the following research mechanism. The prose is authoritative; the JSON context gives exact files and shape constraints.\n"
        "<research_mechanism>\n"
        + prose
        + "\n</research_mechanism>\n"
        "<execution_context>\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
        + "\n</execution_context>\n"
    )


def build_failure_table_message(contract: BuildContract) -> str:
    metric_protocol = _dict(contract.metric_protocol)
    direction = _research_direction(metric_protocol)
    failure_items: list[str] = []
    for value in list(contract.discovery_hints) + [direction.get("failure_to_avoid"), direction.get("why_not_other_options")]:
        text = _short_text(value, max_chars=240)
        if any(token in text.lower() for token in ["research", "ablation", "failed", "rejected", "do not", "avoid"]):
            failure_items.append(text)
    compact = _compact_surrounding_contract(contract, keep_research_direction=False)
    compact["prior_failure_memory"] = _dedupe_preserve_order(_string_list(failure_items, max_items=12, max_chars=240))
    compact["research_direction_core"] = {
        "hypothesis": direction.get("hypothesis"),
        "target_mechanism": direction.get("target_mechanism"),
        "target_code_region": direction.get("target_code_region"),
        "expected_behavior_delta": direction.get("expected_behavior_delta"),
        "terminal_display_title": direction.get("terminal_display_title"),
    }
    return (
        "Implement this research round. Prior failures are compressed into a short memory table; do not repair or recombine rejected mechanisms.\n"
        "<failure_aware_contract>\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
        + "\n</failure_aware_contract>\n"
    )


def build_contract_prompt_message(contract: BuildContract, architecture: str) -> str:
    key = str(architecture or "").strip().lower()
    if key == "dedup_conservative":
        return build_dedup_conservative_message(contract)
    if key == "dedup_guided_finish":
        return build_dedup_guided_finish_message(contract)
    if key == "dedup_with_timekan_anatomy":
        return build_dedup_with_timekan_anatomy_message(contract)
    if key == "dedup_with_source_skeleton":
        return build_dedup_with_source_skeleton_message(contract)
    if key == "research_full_surrounding_compact":
        return build_research_full_surrounding_compact_message(contract)
    if key == "two_stage_research":
        return build_two_stage_research_message(contract)
    if key == "natural_language_first":
        return build_natural_language_first_message(contract)
    if key == "failure_table":
        return build_failure_table_message(contract)
    raise ValueError(f"unknown BuildContract prompt architecture: {architecture}")


def _task_semantics(contract: BuildContract, metric_protocol: dict[str, Any]) -> dict[str, Any]:
    baseline = _dict(metric_protocol.get("baseline"))
    semantics = _dict(_first_present(metric_protocol.get("task_semantics"), baseline.get("task_semantics")))
    metric_semantics = _dict(baseline.get("metric_semantics"))
    model_config = _dict(_first_present(metric_protocol.get("model_config"), baseline.get("model_config")))
    params = _model_hyper_params(model_config)
    objective_metric = str(
        _first_present(metric_protocol.get("objective_metric"), metric_semantics.get("objective_metric"), "mse_norm")
    )
    metrics = _dict(_first_present(baseline.get("metrics"), metric_semantics.get("metrics")))
    dataset = _first_present(
        semantics.get("dataset_path"),
        metric_protocol.get("dataset_path"),
        metric_semantics.get("dataset_path"),
    )
    return {
        "dataset": Path(str(dataset)).name if dataset else "",
        "task_mode": _first_present(semantics.get("task_mode"), metric_semantics.get("task_mode")),
        "time_col": _first_present(semantics.get("time_col"), metric_semantics.get("time_col")),
        "frequency": _first_present(semantics.get("frequency"), semantics.get("freq"), metric_semantics.get("frequency")),
        "objective_metric": objective_metric,
        "baseline_value": _metric_value(metrics, objective_metric),
        "baseline_model": _first_present(
            baseline.get("model_name"),
            model_config.get("model_key"),
            model_config.get("model_name"),
            contract.target_model,
        ),
        "runtime_hyper_params": params,
        "channel_contract": {
            "input_channels": params.get("enc_in"),
            "raw_output_channels": params.get("c_out"),
            "target_channels": _target_channel_count_from_task(semantics, metric_semantics, params),
            "target_columns": _first_present(semantics.get("target_columns"), metric_semantics.get("target_columns")),
            "target_channel": _first_present(semantics.get("target_channel"), metric_semantics.get("target_channel")),
        },
    }


def _implementation_goal(contract: BuildContract, metric_protocol: dict[str, Any]) -> dict[str, Any]:
    direction = _dict(metric_protocol.get("research_direction"))
    target = _dict(metric_protocol.get("target"))
    exact = _dict(metric_protocol.get("exact_ablation_target"))
    return {
        "mechanism_name": _short_text(
            _first_present(direction.get("target_mechanism"), target.get("mechanism_name"), target.get("mechanism_id"), contract.target_model),
            max_chars=300,
        ),
        "target_code_region": _short_text(
            _first_present(direction.get("target_code_region"), target.get("target_code_region"), exact.get("target_region")),
            max_chars=300,
        ),
        "required_change": _short_text(
            _first_present(
                direction.get("expected_behavior_delta"),
                target.get("exact_edit_intent"),
                exact.get("replacement_intent"),
                contract.semantic_goal,
            ),
            max_chars=900,
        ),
        "hypothesis": _short_text(_first_present(direction.get("hypothesis"), contract.hypothesis), max_chars=900),
        "expected_effect": _short_text(
            _first_present(direction.get("expected_behavior_delta"), target.get("expected_effect"), contract.hypothesis),
            max_chars=700,
        ),
    }


def _avoid_list(contract: BuildContract, metric_protocol: dict[str, Any]) -> list[str]:
    direction = _dict(metric_protocol.get("research_direction"))
    values: list[Any] = []
    values.extend(contract.forbidden_behavior)
    values.extend(contract.implementation_constraints)
    for key in ("failure_to_avoid", "why_not_other_options", "avoid"):
        raw = direction.get(key)
        if isinstance(raw, list):
            values.extend(raw)
        elif raw:
            values.append(raw)
    values.extend(
        [
            "Do not change dataset, metric, train/validation/test semantics, horizon, input length, batch size, epoch count, patience, or optimizer schedule.",
            "Existing files may be modified only when listed in allowed_edit_files.",
            "New files may be created only under allowed_new_file_roots.",
        ]
    )
    return _string_list(values, max_items=10, max_chars=260)


def _primary_edit_files(contract: BuildContract, metric_protocol: dict[str, Any]) -> list[str]:
    direction = _dict(metric_protocol.get("research_direction"))
    target = _dict(metric_protocol.get("target"))
    exact = _dict(metric_protocol.get("exact_ablation_target"))
    haystack = " ".join(
        str(value or "")
        for value in [
            direction.get("target_code_region"),
            direction.get("hypothesis"),
            direction.get("expected_behavior_delta"),
            target.get("target_code_region"),
            target.get("evidence_files"),
            exact.get("target_file"),
            contract.semantic_goal,
            contract.hypothesis,
        ]
    ).replace("\\", "/")
    selected: list[str] = []
    for path in contract.allowed_edit_files:
        normalized = str(path or "").replace("\\", "/")
        if normalized and normalized in haystack:
            selected.append(normalized)
    if not selected:
        selected.extend(
            path
            for path in contract.allowed_edit_files
            if "/models/" in str(path).replace("\\", "/") or str(path).replace("\\", "/").endswith("_model.py")
        )
    if not selected and contract.allowed_edit_files:
        selected.append(str(contract.allowed_edit_files[0]).replace("\\", "/"))
    result: list[str] = []
    seen: set[str] = set()
    for path in selected:
        if path and path not in seen:
            result.append(path)
            seen.add(path)
        if len(result) >= 3:
            break
    return result


def _shape_contract(contract: BuildContract, task: dict[str, Any]) -> list[str]:
    params = _dict(task.get("runtime_hyper_params"))
    enc_in = params.get("enc_in")
    dec_in = params.get("dec_in")
    c_out = params.get("c_out")
    d_model = params.get("d_model")
    seq_len = _first_present(params.get("seq_len"), params.get("input_chunk_length"))
    pred_len = _first_present(params.get("pred_len"), params.get("horizon"), params.get("output_chunk_length"))
    channel_contract = _dict(task.get("channel_contract"))
    target_channels = channel_contract.get("target_channels")
    lines = [
        f"Preserve the runtime task semantics: task_mode={task.get('task_mode')}, seq_len={seq_len}, pred_len/horizon={pred_len}.",
        f"Preserve channel dimensions: enc_in={enc_in}, dec_in={dec_in}, c_out={c_out}.",
        f"Runtime probe input is x_enc with shape (batch=2, seq_len={seq_len}, channels={enc_in}); model output must be (2, {pred_len}, {c_out}).",
        "Preserve TimeKANModeL.__init__, TimeKANModeL.forecast, and TimeKANModeL.forward as callable methods; do not replace or duplicate class headers.",
        "When the model flattens batch and channels as B*C, recover B from x_enc.size(0) and C from configs.enc_in/c_out; never infer channels from descriptor width.",
    ]
    if str(task.get("task_mode") or "").upper() == "MS" and target_channels and c_out and target_channels != c_out:
        lines.append(
            f"This is an MS single-target/multi-input task: raw model channels stay c_out={c_out}, "
            f"and benchmark evaluation selects target_channels={target_channels} afterward. "
            "Do not force forward/_process to emit only the target channel unless the baseline already does so."
        )
    if d_model is not None and enc_in is not None:
        lines.append(f"Do not confuse channel count C={enc_in} with hidden/model dimension d_model={d_model}.")
    lines.extend(_string_list(contract.required_behavior, max_items=4, max_chars=220))
    return [line for line in lines if str(line).strip()]


def build_execution_contract_dict(contract: BuildContract) -> dict[str, Any]:
    metric_protocol = _dict(contract.metric_protocol)
    task = _task_semantics(contract, metric_protocol)
    primary_edit_files = _primary_edit_files(contract, metric_protocol)
    repo_wide = str(contract.execution_authority or "") == "repo_wide"
    return {
        "schema": "evocast_build_execution_contract_v1",
        "round_id": contract.research_id,
        "target_model": contract.target_model,
        "implementation_goal": _implementation_goal(contract, metric_protocol),
        "execution_authority": contract.execution_authority,
        "primary_edit_files": primary_edit_files,
        "allowed_edit_files": (
            {"mode": "repo_wide", "count": len(contract.allowed_edit_files)}
            if repo_wide
            else list(contract.allowed_edit_files)
        ),
        "allowed_new_file_roots": (
            {"mode": "repo_wide"}
            if repo_wide
            else list(contract.allowed_new_file_roots)
        ),
        "likely_entrypoints": _string_list(contract.likely_entrypoints, max_items=8, max_chars=220),
        "task_semantics": task,
        "shape_contract": _shape_contract(contract, task),
        "discovery_hints": _string_list(contract.discovery_hints, max_items=8, max_chars=260),
        "avoid": _avoid_list(contract, metric_protocol),
        "execution_workflow": [
            "A concrete source edit is required in this turn.",
            "Start with primary_edit_files. Read no more than two supporting files before the first edit unless the primary file is missing.",
            "After the first successful edit, run run_internal_checks, inspect the diff, and finish with metadata.",
            "If the exact mechanism is source-infeasible, finish with precise infeasibility metadata instead of broad exploration.",
        ],
        "internal_check_commands": list(contract.internal_check_commands),
        "finish_metadata_requirements": [
            "display_idea",
            "idea_summary",
            "mechanism_name",
            "changed_mechanism",
            "expected_effect",
        ],
    }


def build_execution_contract_message(contract: BuildContract) -> str:
    payload = build_execution_contract_dict(contract)
    repo_wide = str(contract.execution_authority or "") == "repo_wide"
    return (
        "Implement this EvoCast research build execution contract in the current workspace.\n"
        + (
            "Run fast internal checks while working. Execution authority is repo-wide for this ablation arm.\n"
            if repo_wide
            else "Run fast internal checks while working. Do not edit protected, metric, data, or test infrastructure.\n"
        )
        + 
        "The payload below is the complete and authoritative coding instruction for this build attempt. "
        "Do not search for a separate BuildContract file. Use this execution contract, inspect only the needed source files, then make a concrete edit.\n"
        "<build_execution_contract>\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n</build_execution_contract>\n"
    )


def build_full_contract_message(contract: BuildContract) -> str:
    repo_wide = str(contract.execution_authority or "") == "repo_wide"
    return (
        "Implement this EvoCast research build contract in the current workspace.\n"
        + (
            "Run fast internal checks while working. Execution authority is repo-wide for this ablation arm.\n"
            if repo_wide
            else "Run fast internal checks while working. Do not edit protected, metric, data, or test infrastructure.\n"
        )
        +
        "<build_contract>\n"
        + json.dumps(contract.to_dict(), ensure_ascii=False, indent=2)
        + "\n</build_contract>\n"
    )
