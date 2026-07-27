"""Model-specific hyperparameter compatibility rules for baseline execution.

These rules repair internal model defaults that are invalid for a model's own
constructor.  They must not change task semantics such as dataset, horizon,
target columns, or evaluation policy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _task_seq_len(config_data: Dict[str, Any], hparams: Dict[str, Any]) -> Optional[int]:
    model_config = config_data.get("model_config", {}) if isinstance(config_data, dict) else {}
    recommended = model_config.get("recommend_model_hyper_params", {}) if isinstance(model_config, dict) else {}
    data_config = config_data.get("data_config", {}) if isinstance(config_data, dict) else {}
    strategy_args = (
        (config_data.get("evaluation_config", {}) or {}).get("strategy_args", {})
        if isinstance(config_data.get("evaluation_config", {}), dict)
        else {}
    )
    for value in (
        hparams.get("seq_len"),
        hparams.get("input_chunk_length"),
        data_config.get("seq_len") if isinstance(data_config, dict) else None,
        recommended.get("input_chunk_length") if isinstance(recommended, dict) else None,
        strategy_args.get("seq_len") if isinstance(strategy_args, dict) else None,
    ):
        parsed = _safe_int(value)
        if parsed is not None:
            return parsed
    return None


def _task_pred_len(config_data: Dict[str, Any], hparams: Dict[str, Any]) -> Optional[int]:
    model_config = config_data.get("model_config", {}) if isinstance(config_data, dict) else {}
    recommended = model_config.get("recommend_model_hyper_params", {}) if isinstance(model_config, dict) else {}
    data_config = config_data.get("data_config", {}) if isinstance(config_data, dict) else {}
    strategy_args = (
        (config_data.get("evaluation_config", {}) or {}).get("strategy_args", {})
        if isinstance(config_data.get("evaluation_config", {}), dict)
        else {}
    )
    for value in (
        hparams.get("pred_len"),
        hparams.get("horizon"),
        hparams.get("output_chunk_length"),
        data_config.get("horizon") if isinstance(data_config, dict) else None,
        recommended.get("output_chunk_length") if isinstance(recommended, dict) else None,
        strategy_args.get("horizon") if isinstance(strategy_args, dict) else None,
    ):
        parsed = _safe_int(value)
        if parsed is not None:
            return parsed
    return None


def _nearest_divisor(value: int, preferred: int) -> int:
    divisors = [candidate for candidate in range(1, value + 1) if value % candidate == 0]
    if not divisors:
        return 1
    return min(divisors, key=lambda item: (abs(item - preferred), -item))


def _align_equal_layers(model_key: str, hparams: Dict[str, Any], notes: List[str]) -> None:
    if model_key != "ETSformer":
        return
    e_layers = _safe_int(hparams.get("e_layers"))
    d_layers = _safe_int(hparams.get("d_layers"))
    if e_layers is None and d_layers is None:
        hparams["e_layers"] = 2
        hparams["d_layers"] = 2
        notes.append("default ETSformer e_layers=d_layers=2")
        return
    if e_layers is None and d_layers is not None:
        hparams["e_layers"] = d_layers
        notes.append(f"aligned ETSformer e_layers={d_layers}")
        return
    if d_layers is None and e_layers is not None:
        hparams["d_layers"] = e_layers
        notes.append(f"aligned ETSformer d_layers={e_layers}")
        return
    if e_layers != d_layers:
        hparams["d_layers"] = e_layers
        notes.append(f"aligned ETSformer d_layers={e_layers} to match e_layers")


def _align_attention_heads(model_key: str, hparams: Dict[str, Any], notes: List[str]) -> None:
    if model_key != "TemporalFusionTransformer":
        return
    d_model = _safe_int(hparams.get("d_model"))
    n_heads = _safe_int(hparams.get("n_heads"))
    if d_model is None:
        return
    if n_heads is None or n_heads < 1:
        hparams["n_heads"] = 1
        notes.append("default TemporalFusionTransformer n_heads=1")
        return
    if d_model % n_heads != 0:
        aligned = _nearest_divisor(d_model, n_heads)
        hparams["n_heads"] = aligned
        notes.append(
            f"aligned TemporalFusionTransformer n_heads={aligned} "
            f"so d_model={d_model} is divisible"
        )


def _align_mamba_simple(model_key: str, hparams: Dict[str, Any], notes: List[str]) -> None:
    if model_key != "MambaSimple":
        return
    if _safe_int(hparams.get("expand")) is None:
        hparams["expand"] = 2
        notes.append("default MambaSimple expand=2")
    if _safe_int(hparams.get("d_conv")) is None:
        hparams["d_conv"] = 4
        notes.append("default MambaSimple d_conv=4")


def _align_time_mixer(model_key: str, hparams: Dict[str, Any], config_data: Dict[str, Any], notes: List[str]) -> None:
    if model_key != "TimeMixer":
        return
    if _safe_int(hparams.get("down_sampling_layers")) is None:
        hparams["down_sampling_layers"] = 1
        notes.append("default TimeMixer down_sampling_layers=1")
    if _safe_int(hparams.get("channel_independence")) is None:
        hparams["channel_independence"] = 0
        notes.append("default TimeMixer channel_independence=0")
    if not hparams.get("decomp_method"):
        hparams["decomp_method"] = "moving_avg"
        notes.append("default TimeMixer decomp_method=moving_avg")
    if _safe_int(hparams.get("moving_avg")) is None:
        hparams["moving_avg"] = 25
        notes.append("default TimeMixer moving_avg=25")
    if _safe_int(hparams.get("top_k")) is None:
        hparams["top_k"] = 5
        notes.append("default TimeMixer top_k=5")
    seq_len = _task_seq_len(config_data, hparams)
    if seq_len is not None:
        window = _safe_int(hparams.get("down_sampling_window"))
        layers = _safe_int(hparams.get("down_sampling_layers")) or 1
        if window is None or window < 2:
            hparams["down_sampling_window"] = 2
            notes.append("default TimeMixer down_sampling_window=2")
        elif seq_len // (window**layers) < 1:
            hparams["down_sampling_window"] = 2
            notes.append("adjusted TimeMixer down_sampling_window=2")


def _align_window_params_to_seq_len(hparams: Dict[str, Any], config_data: Dict[str, Any], notes: List[str]) -> None:
    """Keep generic patch/window constructor parameters within the task input window.

    This is model-agnostic compatibility, not a research hyperparameter change:
    if a model default asks for a patch/window longer than the configured
    seq_len, the model cannot even instantiate its own masks/patches.  The
    executable task window is the authority.
    """
    seq_len = _task_seq_len(config_data, hparams)
    if seq_len is None or seq_len <= 0:
        return
    for key in ("patch_len", "patch_size", "window_size"):
        value = hparams.get(key)
        if isinstance(value, list):
            cleaned = []
            changed = False
            for item in value:
                parsed = _safe_int(item)
                if parsed is None:
                    cleaned.append(item)
                    continue
                aligned = max(1, min(parsed, seq_len))
                cleaned.append(aligned)
                changed = changed or aligned != parsed
            if changed:
                hparams[key] = cleaned or [seq_len]
                notes.append(f"aligned {key} list entries to seq_len={seq_len}")
            continue
        parsed = _safe_int(value)
        if parsed is not None and parsed > seq_len:
            hparams[key] = seq_len
            notes.append(f"aligned {key}={seq_len} because task seq_len={seq_len}")


def apply_model_hparam_compatibility(
    model_key: str,
    hparams: Dict[str, Any],
    config_data: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Return hparams adjusted for known model-internal constraints."""
    aligned = dict(hparams or {})
    notes: List[str] = []
    config = dict(config_data or {})
    key = str(model_key or "")
    _align_equal_layers(key, aligned, notes)
    _align_attention_heads(key, aligned, notes)
    _align_mamba_simple(key, aligned, notes)
    _align_time_mixer(key, aligned, config, notes)
    _align_window_params_to_seq_len(aligned, config, notes)
    return aligned, notes


def validate_model_hparam_compatibility(
    model_key: str,
    hparams: Dict[str, Any],
    config_data: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return a precise compatibility error, or None when the config is valid."""
    key = str(model_key or "")
    values = dict(hparams or {})
    config = dict(config_data or {})

    if key == "ETSformer":
        e_layers = _safe_int(values.get("e_layers"))
        d_layers = _safe_int(values.get("d_layers"))
        if e_layers is not None and d_layers is not None and e_layers != d_layers:
            return (
                "ETSformer requires e_layers == d_layers; "
                f"got e_layers={e_layers}, d_layers={d_layers}."
            )

    if key == "TemporalFusionTransformer":
        d_model = _safe_int(values.get("d_model"))
        n_heads = _safe_int(values.get("n_heads"))
        if d_model is not None and n_heads is not None and (n_heads < 1 or d_model % n_heads != 0):
            return (
                "TemporalFusionTransformer requires d_model divisible by n_heads; "
                f"got d_model={d_model}, n_heads={n_heads}."
            )

    if key == "LightTS":
        seq_len = _task_seq_len(config, values)
        pred_len = _task_pred_len(config, values)
        if seq_len is not None:
            chunk_size = min(value for value in (pred_len, seq_len, 24) if value is not None)
            if chunk_size <= 0 or seq_len % chunk_size != 0:
                return (
                    "LightTS requires seq_len divisible by its constructor chunk_size. "
                    f"With seq_len={seq_len}, pred_len={pred_len}, chunk_size={chunk_size}; "
                    "this constraint is not controlled by model_hyper_params."
                )

    seq_len = _task_seq_len(config, values)
    if seq_len is not None and seq_len > 0:
        for param in ("patch_len", "patch_size", "window_size"):
            parsed = _safe_int(values.get(param))
            if parsed is not None and parsed > seq_len:
                return f"{param}={parsed} exceeds task seq_len={seq_len}; apply compatibility before runtime."

    return None
