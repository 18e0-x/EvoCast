"""Shared baseline candidate runner for evocast.

Provides run_baseline_candidate(), the single entry point used by both
baseline_search.py to run a single model
through the TFB pipeline, parse metrics, classify errors, and record
a journal entry.

Usage:
    from evocast.runners.baseline_runner import run_baseline_candidate

    result = run_baseline_candidate(
        spec=spec,
        config_data=config_data,
        task_id=task_id,
        node_id="baseline_001_DLinear",
        objective_metric="mse",
        num_epochs=1,
        seed=2021,
        dry_run=False,
    )
"""

import math
import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from evocast.domain.metric_semantics import DEFAULT_OBJECTIVE_METRIC
from evocast.domain.metric_parser import parse_metrics_from_paths
from evocast.policy.error_taxonomy import classify_from_result, ErrorLabel
from evocast.policy.cost_profile import (
    SUCCESS_STATUS,
    OOM_STATUS,
    TIMEOUT_STATUS,
    ERROR_STATUS,
    DRY_RUN_STATUS,
)
from evocast.state.cost_ledger import record_execution_cost, tracked_stage
from evocast.domain.knowledge_paths import runs_root
from evocast.state.runtime.trial_journal import create_node, append_node
from evocast.policy.training_policy import apply_policy
from evocast.policy.experiment_policy import baseline_seed, task_build_mode
from evocast.domain.task_identity import compact_result_save_path
from evocast.domain.effective_model_config import resolve_effective_model_config
from evocast.research.baseline_knowledge import (
    build_candidate_signature,
    load_candidate_result,
    signature_hash,
    write_candidate_result,
)
from evocast.policy.model_hparam_compat import (
    apply_model_hparam_compatibility,
    validate_model_hparam_compatibility,
)
from evocast.runners.command_generator import generate_model_entry
from evocast.runners.tfb_pipeline_runner import (
    run_pipeline,
    build_run_configs,
)

_PATCH_COMPATIBILITY_TARGET = 8
_PATCH_MODEL_KEYS = {
    "PatchTST",
    "PatchMLP",
    "xPatch",
    "MultiPatchFormer",
    "PAttn",
    "PDF",
    "Crossformer",
    "CMoS",
    "Pathformer",
    "ModernTCN",
    "TimeFilter",
}
_INDEXED_PATCH_PARAM_MODEL_KEYS = {
    "PDF",
}

# Sources whose models accept transformer-style seq_len / pred_len / horizon /
# label_len parameters. New model sources are safe by default: they only get
# injected params if explicitly added here.
_SOURCES_USING_SEQ_PRED_PARAMS = {"time_series_library", "standalone"}


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _effective_seq_len(config_data: Dict, model_hparams: Dict[str, Any]) -> Optional[int]:
    for value in (
        model_hparams.get("seq_len"),
        config_data.get("data_config", {}).get("seq_len"),
        config_data.get("model_config", {}).get("recommend_model_hyper_params", {}).get("input_chunk_length"),
    ):
        parsed = _safe_int(value)
        if parsed is not None:
            return parsed
    return None


def _effective_pred_len(config_data: Dict, model_hparams: Dict[str, Any]) -> Optional[int]:
    for key in ("pred_len", "horizon"):
        parsed = _safe_int(model_hparams.get(key))
        if parsed is not None:
            return parsed
    for key in ("pred_len", "horizon"):
        parsed = _safe_int(config_data.get("data_config", {}).get(key))
        if parsed is not None:
            return parsed
    for value in (
        config_data.get("model_config", {}).get("recommend_model_hyper_params", {}).get("output_chunk_length"),
        config_data.get("evaluation_config", {}).get("strategy_args", {}).get("horizon"),
    ):
        parsed = _safe_int(value)
        if parsed is not None:
            return parsed
    return None


def _task_seq_len(config_data: Dict[str, Any]) -> Optional[int]:
    for value in (
        config_data.get("data_config", {}).get("seq_len"),
        config_data.get("model_config", {}).get("recommend_model_hyper_params", {}).get("input_chunk_length"),
    ):
        parsed = _safe_int(value)
        if parsed is not None:
            return parsed
    return None


def _task_pred_len(config_data: Dict[str, Any]) -> Optional[int]:
    for value in (
        config_data.get("data_config", {}).get("pred_len"),
        config_data.get("data_config", {}).get("horizon"),
        config_data.get("model_config", {}).get("recommend_model_hyper_params", {}).get("output_chunk_length"),
        config_data.get("evaluation_config", {}).get("strategy_args", {}).get("horizon"),
    ):
        parsed = _safe_int(value)
        if parsed is not None:
            return parsed
    return None


def _positive_divisors(value: int) -> List[int]:
    divisors: List[int] = []
    for candidate in range(1, int(math.sqrt(value)) + 1):
        if value % candidate != 0:
            continue
        divisors.append(candidate)
        mirror = value // candidate
        if mirror != candidate:
            divisors.append(mirror)
    return sorted(divisors)


def _choose_divisor_close_to_target(
    value: int,
    *,
    target: int,
    min_value: int = 1,
) -> Optional[int]:
    candidates = [d for d in _positive_divisors(value) if d >= min_value]
    if not candidates:
        return None
    return min(candidates, key=lambda d: (abs(d - target), -d))


def _filter_patch_scales(seq_len: int, patch_scales: List[int]) -> List[int]:
    valid: List[int] = []
    for scale in patch_scales:
        scale_int = _safe_int(scale)
        if scale_int is None or scale_int <= 1 or scale_int > seq_len:
            continue
        patch_step = max(1, scale_int // 2)
        patch_num = int((seq_len - scale_int) / patch_step + 1)
        if patch_num >= 1:
            valid.append(scale_int)
    deduped = sorted(set(valid), reverse=True)
    if len(deduped) >= 3:
        return deduped[:4]
    fallback = [seq_len, max(2, seq_len // 2), max(2, seq_len // 4), max(2, seq_len // 8)]
    expanded: List[int] = []
    for candidate in fallback:
        if candidate > seq_len:
            continue
        patch_step = max(1, candidate // 2)
        patch_num = int((seq_len - candidate) / patch_step + 1)
        if patch_num >= 1:
            expanded.append(candidate)
    return sorted(set(deduped + expanded), reverse=True)


def _is_patch_like_model(spec: Dict[str, Any], hparams: Dict[str, Any]) -> bool:
    model_key = str(spec.get("model_key", ""))
    model_key_lower = model_key.lower()
    import_path = str(spec.get("import_path", "")).lower()
    tags = {str(tag).lower() for tag in list(spec.get("tags", []) or [])}
    return bool(
        "patch_len" in hparams
        or "patch_size" in hparams
        or "patch_size_list" in hparams
        or "#patch" in tags
        or model_key in _PATCH_MODEL_KEYS
        or "patch" in model_key_lower
        or "patch" in import_path
    )


def _uses_indexed_patch_params(spec: Dict[str, Any], hparams: Dict[str, Any]) -> bool:
    model_key = str(spec.get("model_key", ""))
    return bool(
        model_key in _INDEXED_PATCH_PARAM_MODEL_KEYS
        or isinstance(hparams.get("patch_len"), list)
        or isinstance(hparams.get("stride"), list)
        or isinstance(hparams.get("period"), list)
    )


def _indexed_param_count(hparams: Dict[str, Any]) -> int:
    for key in ("period", "patch_len", "stride"):
        value = hparams.get(key)
        if isinstance(value, list) and value:
            return len(value)
    return 1


def _repeat_indexed_param(value: int, count: int) -> List[int]:
    return [value for _ in range(max(1, count))]


def _patch_compatibility_anchor(seq_len: Optional[int]) -> int:
    if seq_len is None:
        return _PATCH_COMPATIBILITY_TARGET
    return max(2, min(_PATCH_COMPATIBILITY_TARGET, seq_len))


def _default_patch_scale_profile(anchor: int) -> List[int]:
    if anchor >= 8:
        return [8, 6, 4, 2]
    if anchor >= 6:
        return [6, 4, 2, 2]
    if anchor >= 4:
        return [4, 2, 2, 2]
    return [2, 2, 2, 2]


def _align_model_hparams_to_task(
    spec: Dict[str, Any],
    config_data: Dict[str, Any],
    entry: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    model_key = str(spec.get("model_key", ""))
    hparams = dict(entry.get("model_hyper_params", {}) or {})
    notes: List[str] = []
    seq_len = _task_seq_len(config_data) or _effective_seq_len(config_data, hparams)
    pred_len = _task_pred_len(config_data) or _effective_pred_len(config_data, hparams)

    patch_anchor = _patch_compatibility_anchor(seq_len)
    if model_key == "PatchMLP":
        default_scales = _default_patch_scale_profile(patch_anchor)
        patch_scales = _filter_patch_scales(
            seq_len or patch_anchor,
            default_scales,
        )
        while len(patch_scales) < 4:
            patch_scales.append(patch_scales[-1] if patch_scales else max(2, patch_anchor))
        patch_scales = patch_scales[:4]
        if hparams.get("patch_len") != patch_scales:
            hparams["patch_len"] = patch_scales
            notes.append(f"default patch_len={patch_scales}")
    elif _uses_indexed_patch_params(spec, hparams):
        indexed_count = _indexed_param_count(hparams)
        patch_len = _repeat_indexed_param(patch_anchor, indexed_count)
        if hparams.get("patch_len") != patch_len:
            hparams["patch_len"] = patch_len
            notes.append(f"default indexed patch_len={patch_len}")
        stride_anchor = max(1, patch_anchor // 2)
        if "stride" in hparams:
            stride = _repeat_indexed_param(stride_anchor, indexed_count)
            if hparams.get("stride") != stride:
                hparams["stride"] = stride
                notes.append(f"default indexed stride={stride}")
    elif _is_patch_like_model(spec, hparams):
        if _safe_int(hparams.get("patch_len")) != patch_anchor:
            hparams["patch_len"] = patch_anchor
            notes.append(f"default patch_len={patch_anchor}")
        stride_anchor = max(1, patch_anchor // 2)
        if "stride" in hparams and _safe_int(hparams.get("stride")) != stride_anchor:
            hparams["stride"] = stride_anchor
            notes.append(f"default stride={stride_anchor}")
        if "patch_size" in hparams and _safe_int(hparams.get("patch_size")) != patch_anchor:
            hparams["patch_size"] = patch_anchor
            notes.append(f"default patch_size={patch_anchor}")
        if "patch_stride" in hparams and _safe_int(hparams.get("patch_stride")) != stride_anchor:
            hparams["patch_stride"] = stride_anchor
            notes.append(f"default patch_stride={stride_anchor}")
        if "patch_size_list" in hparams:
            scale_profile = _filter_patch_scales(seq_len or patch_anchor, _default_patch_scale_profile(patch_anchor))
            while len(scale_profile) < 4:
                scale_profile.append(scale_profile[-1] if scale_profile else max(2, patch_anchor))
            scale_profile = scale_profile[:4]
            layer_count = len(hparams.get("patch_size_list", []) or []) or 1
            patch_size_list = [list(scale_profile) for _ in range(layer_count)]
            if hparams.get("patch_size_list") != patch_size_list:
                hparams["patch_size_list"] = patch_size_list
                notes.append(f"default patch_size_list={patch_size_list}")

    if model_key == "CMoS" and seq_len is not None and pred_len is not None:
        target = _safe_int(hparams.get("patch_len")) or min(seq_len, pred_len)
        aligned_patch = _choose_divisor_close_to_target(math.gcd(seq_len, pred_len), target=target, min_value=2)
        if aligned_patch is not None and aligned_patch != _safe_int(hparams.get("patch_len")):
            hparams["patch_len"] = aligned_patch
            notes.append(f"adapted patch_len={aligned_patch}")
        stride_value = _safe_int(hparams.get("stride"))
        expected_stride = max(1, aligned_patch // 2) if aligned_patch is not None else None
        if expected_stride is not None and stride_value != expected_stride:
            hparams["stride"] = max(1, aligned_patch // 2)
            notes.append(f"adapted stride={hparams['stride']}")
        kernel_size = _safe_int(hparams.get("kernel_size"))
        conv_stride = _safe_int(hparams.get("conv_stride"))
        if kernel_size is None or kernel_size > seq_len:
            hparams["kernel_size"] = max(2, min(seq_len, aligned_patch))
            notes.append(f"adapted kernel_size={hparams['kernel_size']}")
        if conv_stride is None or conv_stride < 1 or ((seq_len - _safe_int(hparams.get('kernel_size'))) // conv_stride + 1) < 1:
            hparams["conv_stride"] = max(1, min(_safe_int(hparams.get("kernel_size")) or 1, seq_len))
            notes.append(f"adapted conv_stride={hparams['conv_stride']}")

    # ── Task-dimension → model param injection ──────────────────────────
    # Only inject seq_len / pred_len / horizon / label_len for model families
    # that explicitly declare they accept them. Other sources use their own
    # native conventions and are safe by default.
    if spec.get("source") in _SOURCES_USING_SEQ_PRED_PARAMS:
        if seq_len is not None:
            hparams["seq_len"] = seq_len
            if model_key == "PatchMLP":
                hparams["label_len"] = seq_len
            else:
                hparams.setdefault("label_len", max(1, seq_len // 2))
        if pred_len is not None:
            hparams["pred_len"] = pred_len
            hparams["horizon"] = pred_len

    hparams, compatibility_notes = apply_model_hparam_compatibility(model_key, hparams, config_data)
    notes.extend(compatibility_notes)
    entry["model_hyper_params"] = hparams
    return entry, notes


def align_model_hparams_to_task(
    spec: Dict[str, Any],
    config_data: Dict[str, Any],
    model_hyper_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Public wrapper so other runtime paths reuse baseline compatibility fixes."""
    entry = {"model_hyper_params": dict(model_hyper_params or {})}
    aligned_entry, notes = _align_model_hparams_to_task(spec, config_data or {}, entry)
    return dict(aligned_entry.get("model_hyper_params", {}) or {}), notes


def _validate_model_constraints(
    spec: Dict[str, Any],
    config_data: Dict[str, Any],
    entry: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    model_key = str(spec.get("model_key", ""))
    hparams = entry.get("model_hyper_params", {}) or {}
    seq_len = _task_seq_len(config_data) or _effective_seq_len(config_data, hparams)
    pred_len = _task_pred_len(config_data) or _effective_pred_len(config_data, hparams)

    compatibility_error = validate_model_hparam_compatibility(model_key, hparams, config_data)
    if compatibility_error:
        return {
            "error_type": ErrorLabel.SHAPE_CONSTRAINT_VIOLATION.value,
            "error_message": compatibility_error,
        }

    required_hparams = {
        "TimeMixer": ["down_sampling_window"],
        "MambaSimple": ["expand", "d_conv"],
    }
    for required in required_hparams.get(model_key, []):
        if hparams.get(required) is None:
            return {
                "error_type": ErrorLabel.MISSING_REQUIRED_HPARAM.value,
                "error_message": f"{model_key} requires hyperparameter '{required}' but it is missing.",
            }

    if model_key == "SparseTSF":
        period_len = hparams.get("period_len", 24)
        try:
            period_len = int(period_len) if period_len is not None else None
        except (TypeError, ValueError):
            period_len = None
        if period_len and seq_len and seq_len % period_len != 0:
            return {
                "error_type": ErrorLabel.SHAPE_CONSTRAINT_VIOLATION.value,
                "error_message": (
                    f"{model_key} requires seq_len divisible by period_len; "
                    f"got seq_len={seq_len}, period_len={period_len}."
                ),
            }
        if period_len and pred_len and pred_len % period_len != 0:
            return {
                "error_type": ErrorLabel.SHAPE_CONSTRAINT_VIOLATION.value,
                "error_message": (
                    f"{model_key} requires pred_len divisible by period_len; "
                    f"got pred_len={pred_len}, period_len={period_len}."
                ),
            }

    if model_key == "CMoS":
        patch_len = hparams.get("patch_len", 16)
        try:
            patch_len = int(patch_len) if patch_len is not None else None
        except (TypeError, ValueError):
            patch_len = None
        if patch_len and seq_len and seq_len % patch_len != 0:
            return {
                "error_type": ErrorLabel.SHAPE_CONSTRAINT_VIOLATION.value,
                "error_message": (
                    f"{model_key} requires seq_len divisible by patch_len; "
                    f"got seq_len={seq_len}, patch_len={patch_len}."
                ),
            }
        if patch_len and pred_len and pred_len % patch_len != 0:
            return {
                "error_type": ErrorLabel.SHAPE_CONSTRAINT_VIOLATION.value,
                "error_message": (
                    f"{model_key} requires pred_len divisible by patch_len; "
                    f"got pred_len={pred_len}, patch_len={patch_len}."
                ),
            }

    if model_key == "PatchMLP":
        patch_scales = hparams.get("patch_len", [48, 24, 12, 6])
        if not isinstance(patch_scales, list):
            patch_scales = [patch_scales]
        valid_scales = _filter_patch_scales(seq_len or 0, patch_scales) if seq_len else []
        if not valid_scales:
            return {
                "error_type": ErrorLabel.SHAPE_CONSTRAINT_VIOLATION.value,
                "error_message": (
                    f"{model_key} has no valid patch scales for seq_len={seq_len}. "
                    f"Current patch_len={patch_scales}."
                ),
            }
        for scale in valid_scales:
            step = max(1, scale // 2)
            patch_num = int(((seq_len or 0) - scale) / step + 1)
            if patch_num <= 0:
                return {
                    "error_type": ErrorLabel.SHAPE_CONSTRAINT_VIOLATION.value,
                    "error_message": (
                        f"{model_key} computed invalid patch_num={patch_num} "
                        f"for seq_len={seq_len}, patch_len={scale}, patch_step={step}."
                    ),
                }

    if _is_patch_like_model(spec, hparams):
        patch_len = _safe_int(hparams.get("patch_len"))
        if patch_len is not None and seq_len is not None and patch_len > seq_len:
            return {
                "error_type": ErrorLabel.SHAPE_CONSTRAINT_VIOLATION.value,
                "error_message": (
                    f"{model_key} requires patch_len <= seq_len; "
                    f"got seq_len={seq_len}, patch_len={patch_len}."
                ),
            }
        patch_size = _safe_int(hparams.get("patch_size"))
        if patch_size is not None and seq_len is not None and patch_size > seq_len:
            return {
                "error_type": ErrorLabel.SHAPE_CONSTRAINT_VIOLATION.value,
                "error_message": (
                    f"{model_key} requires patch_size <= seq_len; "
                    f"got seq_len={seq_len}, patch_size={patch_size}."
                ),
            }
        patch_size_list = hparams.get("patch_size_list")
        if isinstance(patch_size_list, list) and seq_len is not None:
            flat_sizes: List[int] = []
            for item in patch_size_list:
                if isinstance(item, list):
                    flat_sizes.extend(v for v in (_safe_int(x) for x in item) if v is not None)
                else:
                    value = _safe_int(item)
                    if value is not None:
                        flat_sizes.append(value)
            if flat_sizes and any(size > seq_len for size in flat_sizes):
                return {
                    "error_type": ErrorLabel.SHAPE_CONSTRAINT_VIOLATION.value,
                    "error_message": (
                        f"{model_key} requires every patch_size_list element <= seq_len; "
                        f"got seq_len={seq_len}, patch_size_list={patch_size_list}."
                    ),
                }

    return None


def _preflight_candidate_runtime(
    config_data: Dict[str, Any],
    entry: Dict[str, Any],
    *,
    seed: int,
) -> Optional[Dict[str, str]]:
    """Construct the real TFB adapter/model on the available device.

    Registry importability is insufficient: some third-party models import on
    CPU but hard-code CUDA allocation in ``__init__``.  This runs the same
    model-loader and adapter construction path as the benchmark before a
    baseline reserves a pipeline run.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        from ts_benchmark.models.model_loader import get_models

        _, model_config, _ = build_run_configs(
            config_data,
            [entry],
            save_path="EvoCast_candidate_preflight",
            seed=seed,
        )
        factory = get_models(model_config)[0]
        model = factory()
        if hasattr(model, "device"):
            model.device = device
        inner = getattr(model, "model", None)
        if inner is None and hasattr(model, "_init_model"):
            inner = model._init_model()
        target = inner if inner is not None else model
        if hasattr(target, "to"):
            target.to(device)
        return None
    except Exception as exc:
        model_name = str(entry.get("model_name") or "candidate")
        error_text = f"{type(exc).__name__}: {exc}"
        lowered_error = error_text.lower()
        device_name = (
            "cpu"
            if "cuda" in lowered_error and ("not compiled" in lowered_error or "not available" in lowered_error)
            else str(device)
        )
        return {
            "error_type": ErrorLabel.ENVIRONMENT_INCOMPATIBLE.value,
            "error_message": (
                f"{model_name} cannot be instantiated on the current {device_name} environment: "
                f"{error_text}"
            ),
            "traceback": traceback.format_exc(),
        }


def _observation_status_from_label(error_label: ErrorLabel, dry_run: bool) -> str:
    if dry_run:
        return DRY_RUN_STATUS
    if error_label == ErrorLabel.SUCCESS:
        return SUCCESS_STATUS
    if error_label == ErrorLabel.OOM:
        return OOM_STATUS
    if error_label == ErrorLabel.TIMEOUT:
        return TIMEOUT_STATUS
    return ERROR_STATUS


def _refine_error_type_from_record_errors(
    current_error_type: str,
    parsed: Dict[str, Any],
) -> tuple[str, str]:
    record_errors = list(parsed.get("record_errors", []) or [])
    if not record_errors:
        return current_error_type, ""
    combined = "\n\n".join(str(item) for item in record_errors if item)
    lowered = combined.lower()

    if current_error_type == ErrorLabel.METRIC_MISSING.value:
        if "requires hyperparameter" in lowered or "has no attribute" in lowered:
            return ErrorLabel.MISSING_REQUIRED_HPARAM.value, combined
        if "shape " in lowered and "invalid for input of size" in lowered:
            return ErrorLabel.SHAPE_CONSTRAINT_VIOLATION.value, combined
        if "traceback (most recent call last)" in lowered or "runtimeerror:" in lowered:
            return ErrorLabel.RUNTIME_ERROR.value, combined

    return current_error_type, combined


def _extract_runtime_breakdown(parsed: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    runtime_stats = parsed.get("runtime_stats", {}) if isinstance(parsed, dict) else {}
    fit_time = runtime_stats.get("fit_time", metrics.get("fit_time"))
    inference_time = runtime_stats.get("inference_time", metrics.get("inference_time"))
    try:
        fit_time_val = float(fit_time) if fit_time is not None else None
    except (TypeError, ValueError):
        fit_time_val = None
    try:
        inference_time_val = float(inference_time) if inference_time is not None else None
    except (TypeError, ValueError):
        inference_time_val = None
    return {
        "fit_time": fit_time_val,
        "inference_time": inference_time_val,
        "metric_source_paths": list(parsed.get("metric_source_paths", []) or []),
    }


@tracked_stage(
    "baseline_candidate",
    lambda spec, config_data, task_id, node_id, objective_metric=DEFAULT_OBJECTIVE_METRIC, num_epochs=None, budget="unified", tier=None, seed=None, base_dir=None, dry_run=False, pipeline_timeout=600: (
        str(base_dir or os.path.join(os.path.dirname(__file__), "..")),
        str(task_id),
        "",
        str(node_id),
    ),
)
def run_baseline_candidate(
    spec: Dict,
    config_data: Dict,
    task_id: str,
    node_id: str,
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC,
    num_epochs: Optional[int] = None,
    budget: str = "unified",
    tier: Optional[str] = None,
    seed: int | None = None,
    base_dir: Optional[str] = None,
    dry_run: bool = False,
    pipeline_timeout: float = 600,
) -> Dict:
    """Run a single baseline candidate through the TFB pipeline.

    Training hyperparams come from the unified training policy, merged on
    top of the model's default hyperparams.  An explicit ``num_epochs``
    argument still works as an override for callers that need it (e.g.
    baseline search uses tier-based epoch counts).
    """
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "..")

    seed = int(seed if seed is not None else baseline_seed(base_dir))

    model_key = spec["model_key"]

    # Build model entry from spec defaults, then apply training policy
    entry = generate_model_entry(spec)
    entry, adaptation_notes = _align_model_hparams_to_task(spec, config_data, entry)
    entry["model_hyper_params"] = apply_policy(
        entry["model_hyper_params"], budget=budget, base_dir=base_dir,
    )
    entry, post_policy_notes = _align_model_hparams_to_task(spec, config_data, entry)
    adaptation_notes.extend(post_policy_notes)
    # Allow explicit num_epochs override (beam-search tier mapping)
    if num_epochs is not None:
        entry["model_hyper_params"]["num_epochs"] = num_epochs
    resolved = resolve_effective_model_config(
        config_data=config_data,
        base_dir=base_dir,
        task_id=task_id,
        model_entry={
            **entry,
            "model_key": model_key,
            "model_name": entry.get("model_name") or spec.get("import_path") or model_key,
            "adapter": entry.get("adapter") if entry.get("adapter") is not None else spec.get("adapter"),
        },
        explicit_model_hyper_params=dict(entry.get("model_hyper_params") or {}),
        requested_budget=budget,
        smoke=False,
    )
    entry = resolved.entry
    if resolved.compatibility_notes:
        adaptation_notes.extend(resolved.compatibility_notes)

    formal_baseline_knowledge_enabled = (
        not dry_run
        and not bool(task_build_mode(base_dir, task_id))
        and str(budget or "").strip().lower() not in {"smoke", "smoke_test", "build_mode"}
    )
    baseline_signature: Dict[str, Any] = {}
    if formal_baseline_knowledge_enabled:
        baseline_signature = build_candidate_signature(
            task_id=task_id,
            base_dir=base_dir,
            config_data=config_data,
            model_key=model_key,
            model_entry=entry,
            objective_metric=objective_metric,
            budget=budget,
            seed=seed,
        )
        cached_result = load_candidate_result(base_dir, baseline_signature)
        if cached_result:
            reused = dict(cached_result)
            reused["node_id"] = node_id
            reused["model_key"] = model_key
            reused["objective_metric"] = objective_metric
            reused["objective_value"] = (reused.get("metrics") or {}).get(objective_metric)
            reused["tier"] = tier or reused.get("tier")
            reused["budget"] = budget
            reused["seed"] = seed
            reused["model_config"] = entry
            reused["elapsed_seconds"] = 0.0
            reused["cost_status"] = "reused"
            reused.setdefault("run_result", {})["success"] = True
            reused.setdefault("run_result", {})["elapsed_seconds"] = 0.0
            reused.setdefault("run_result", {})["baseline_knowledge_reused"] = True
            return reused

    # Create journal node
    node = create_node(
        task_id=task_id,
        node_id=node_id,
        action_type="baseline",
        model_name=model_key,
        model_config=entry,
        objective_metric=objective_metric,
        status="running",
    )
    node["seed"] = seed
    if adaptation_notes:
        node["llm_summary"] = "Adaptive baseline compatibility: " + "; ".join(dict.fromkeys(adaptation_notes))

    constraint_error = _validate_model_constraints(spec, config_data, entry)
    if constraint_error is None and not dry_run:
        constraint_error = _preflight_candidate_runtime(config_data, entry, seed=seed)
    if constraint_error is not None:
        node["status"] = "failed"
        node["error_type"] = constraint_error["error_type"]
        node["error_message"] = constraint_error["error_message"][:2000]
        node["metrics"] = {}
        node["completed_at"] = datetime.now().isoformat()
        append_node(task_id, node, str(runs_root(base_dir)))
        model_hparams = entry.get("model_hyper_params", {})
        strategy_args = (
            config_data.get("evaluation_config", {}).get("strategy_args", {})
            if isinstance(config_data.get("evaluation_config", {}), dict)
            else {}
        )
        record_execution_cost(
            base_dir,
            task_id,
            {
                "model_key": model_key,
                "node_id": node_id,
                "tier": tier or budget,
                "budget": budget,
                "status": ERROR_STATUS,
                "error_type": node["error_type"],
                "elapsed_seconds_total": 0.0,
                "fit_time": None,
                "inference_time": None,
                "objective_metric": objective_metric,
                "objective_value": None,
                "seq_len": _effective_seq_len(config_data, model_hparams),
                "pred_len": _effective_pred_len(config_data, model_hparams),
                "batch_size": model_hparams.get("batch_size"),
                "num_epochs": model_hparams.get("num_epochs"),
                "horizon": strategy_args.get("horizon"),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "metric_source_paths": [],
            },
        )
        return {
            "model_key": model_key,
            "node_id": node_id,
            "status": "failed",
            "objective_metric": objective_metric,
            "objective_value": None,
            "metrics": {},
            "error_type": node["error_type"],
            "error_message": node["error_message"],
            "elapsed_seconds": 0.0,
            "fit_time": None,
            "inference_time": None,
            "cost_status": ERROR_STATUS,
            "run_result": {
                "success": False,
                "log_paths": [],
                "error": node["error_message"],
                "error_traceback": "",
                "elapsed_seconds": 0.0,
            },
            "tags": spec.get("tags", []),
            "family": spec.get("family", "unknown"),
            "import_path": spec.get("import_path", ""),
            "adapter": entry.get("adapter"),
            "model_config": entry,
            "primary_group": spec.get("primary_group", ""),
            "cost_level": spec.get("cost_level", 2),
            "reliability": spec.get("reliability", 1),
            "sentinel": spec.get("sentinel", False),
            "seed": seed,
        }

    save_path = compact_result_save_path(task_id, node_id)

    if dry_run:
        run_result = {
            "success": True,
            "log_paths": [],
            "error": None,
            "error_traceback": "",
            "elapsed_seconds": 0,
        }
        # Synthesise a dummy metric so beam expansion / tournament logic
        # can be exercised.  Mark the node so downstream code knows it is
        # not a real result.
        metrics = {"_dry_run": 1.0}
        parsed = {"metric_values": metrics, "status": "ok", "warnings": [], "parser_errors": []}
    else:
        t_start = time.time()
        data_config, model_config, evaluation_config = build_run_configs(
            config_data, [entry], save_path=save_path, seed=seed,
        )
        run_result = run_pipeline(
            data_config, model_config, evaluation_config,
            timeout=pipeline_timeout,
        )

        # Parse metrics
        if run_result["success"] and run_result["log_paths"]:
            parsed = parse_metrics_from_paths(
                run_result["log_paths"],
                objective_metric=objective_metric,
            )
            metrics = parsed["metric_values"]
            if objective_metric and not parsed.get("objective_metric_present"):
                parsed["status"] = "error"
        else:
            parsed = {"metric_values": {}, "status": "error", "warnings": [], "parser_errors": []}
            metrics = {}

    # Classify error
    error_label = classify_from_result(
        success=run_result["success"],
        error=run_result.get("error"),
        metrics=metrics,
        objective_metric=objective_metric,
    )

    # Update journal node
    node["status"] = "success" if error_label == ErrorLabel.SUCCESS else "failed"
    node["error_type"] = error_label.value
    node["error_message"] = str(run_result.get("error", ""))[:2000]
    node["metrics"] = metrics
    node["completed_at"] = datetime.now().isoformat()
    node["artifact_paths"] = run_result.get("log_paths", [])

    refined_error_type, record_trace = _refine_error_type_from_record_errors(
        node["error_type"],
        parsed,
    )
    node["error_type"] = refined_error_type
    if record_trace:
        if node["error_message"]:
            node["error_message"] = f"{node['error_message']}\n\n{record_trace}"[:2000]
        else:
            node["error_message"] = record_trace[:2000]

    if dry_run:
        node["status"] = "dry_run"
        node["error_type"] = "dry_run"

    append_node(task_id, node, str(runs_root(base_dir)))

    obj_val = metrics.get(objective_metric)
    runtime_breakdown = _extract_runtime_breakdown(parsed, metrics)
    observation_status = _observation_status_from_label(error_label, dry_run)
    model_hparams = entry.get("model_hyper_params", {})
    strategy_args = (
        config_data.get("evaluation_config", {}).get("strategy_args", {})
        if isinstance(config_data.get("evaluation_config", {}), dict)
        else {}
    )
    record_execution_cost(
        base_dir,
        task_id,
        {
            "model_key": model_key,
            "node_id": node_id,
            "tier": tier or budget,
            "budget": budget,
            "status": observation_status,
            "error_type": node["error_type"],
            "elapsed_seconds_total": float(run_result.get("elapsed_seconds", 0) or 0),
            "fit_time": runtime_breakdown["fit_time"],
            "inference_time": runtime_breakdown["inference_time"],
            "objective_metric": objective_metric,
            "objective_value": obj_val,
            "seq_len": _effective_seq_len(config_data, model_hparams),
            "pred_len": _effective_pred_len(config_data, model_hparams),
            "batch_size": model_hparams.get("batch_size"),
            "num_epochs": model_hparams.get("num_epochs"),
            "horizon": strategy_args.get("horizon"),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "metric_source_paths": runtime_breakdown["metric_source_paths"],
        },
    )

    result_payload = {
        "model_key": model_key,
        "node_id": node_id,
        "status": node["status"],
        "objective_metric": objective_metric,
        "objective_value": obj_val,
        "metrics": metrics,
        "error_type": node["error_type"],
        "error_message": node.get("error_message", ""),
        "elapsed_seconds": run_result.get("elapsed_seconds", 0),
        "fit_time": runtime_breakdown["fit_time"],
        "inference_time": runtime_breakdown["inference_time"],
        "cost_status": observation_status,
        "run_result": run_result,
        "tags": spec.get("tags", []),
        "family": spec.get("family", "unknown"),
        "import_path": spec.get("import_path", ""),
        "adapter": entry.get("adapter"),
        "model_config": entry,
        "primary_group": spec.get("primary_group", ""),
        "cost_level": spec.get("cost_level", 2),
        "reliability": spec.get("reliability", 1),
        "sentinel": spec.get("sentinel", False),
        "seed": seed,
    }
    if formal_baseline_knowledge_enabled and result_payload.get("status") == "success":
        knowledge_payload = write_candidate_result(
            base_dir=base_dir,
            task_id=task_id,
            signature=baseline_signature,
            result=result_payload,
        )
        if knowledge_payload:
            result_payload["baseline_knowledge"] = {
                "reused": False,
                "signature_hash": signature_hash(baseline_signature),
                "path": str(knowledge_payload.get("path") or ""),
            }
    return result_payload
