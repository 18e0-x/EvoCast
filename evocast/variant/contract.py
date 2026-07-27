"""Executable runtime contract probes for generated variants."""

from __future__ import annotations

import json
import traceback
import inspect
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from evocast.probe.execution_evidence import (
    canonical_source_target,
    collect_module_class_paths,
    compare_tensor_outputs,
    extract_traceback_evidence,
    source_defined_symbols,
)
from evocast.probe.failure_signature import failure_signature
from evocast.research.mechanism.probe import MechanismProbeError, run_mechanism_probe
from evocast.research.objective_boundary import normalize_tfb_process_output
from evocast.probe.tensor_trace import RuntimeFactTracer
from evocast.policy.model_hparam_compat import apply_model_hparam_compatibility
from evocast.variant.import_isolation import model_execution_import_context
from evocast.variant.workspace_loader import is_workspace_variant_path
from evocast.runners.tfb_pipeline_runner import build_run_configs

EDITED_SOURCE_FILES_MARKER = "# edited_source_files="


def _edited_source_files_from_variant_source(source: str) -> List[str]:
    for line in source.splitlines()[:12]:
        if line.startswith(EDITED_SOURCE_FILES_MARKER):
            raw = line.split("=", 1)[-1].strip()
            try:
                payload = json.loads(raw)
            except Exception:
                return []
            if isinstance(payload, list):
                return [str(item).replace("\\", "/") for item in payload if str(item).strip()]
    return []


def _source_file_for_lineno(source: str, lineno: int) -> str:
    current = ""
    for idx, line in enumerate(source.splitlines(), start=1):
        if line.startswith("# ==== BEGIN SOURCE FILE: ") and line.endswith(" ===="):
            current = line[len("# ==== BEGIN SOURCE FILE: ") : -len(" ====")].strip()
        elif line.startswith("# ==== END SOURCE FILE: "):
            current = ""
        if idx == lineno:
            return current.replace("\\", "/")
    return ""


def _source_line_for_variant_lineno(source: str, lineno: int) -> int:
    current_source_line = 0
    in_source_block = False
    for idx, line in enumerate(source.splitlines(), start=1):
        if line.startswith("# ==== BEGIN SOURCE FILE: ") and line.endswith(" ===="):
            in_source_block = True
            current_source_line = 0
            continue
        if line.startswith("# ==== END SOURCE FILE: "):
            in_source_block = False
            current_source_line = 0
            continue
        if in_source_block:
            current_source_line += 1
        if idx == lineno:
            return current_source_line if in_source_block else 0
    return 0


def _remap_traceback_evidence_to_source_files(
    evidence: Dict[str, Any],
    variant_source: str,
    variant_path: str,
) -> Dict[str, Any]:
    if not isinstance(evidence, dict) or not variant_source:
        return evidence
    variant_rel = str(variant_path or "").replace("\\", "/").strip()
    frames = list(evidence.get("frames") or [])
    remapped_frames: List[Dict[str, Any]] = []
    for frame in frames:
        item = dict(frame or {})
        frame_file = str(item.get("repo_path") or item.get("file") or "").replace("\\", "/")
        if frame_file.endswith(variant_rel):
            mapped_file = _source_file_for_lineno(variant_source, int(item.get("line") or 0))
            mapped_line = _source_line_for_variant_lineno(variant_source, int(item.get("line") or 0))
            if mapped_file:
                item["assembled_variant_file"] = item.get("file")
                item["assembled_variant_line"] = item.get("line")
                item["file"] = mapped_file
                item["repo_path"] = mapped_file
                item["line"] = mapped_line or item.get("line")
        remapped_frames.append(item)
    if not remapped_frames:
        return evidence
    remapped = dict(evidence)
    remapped["frames"] = remapped_frames
    remapped["innermost_frame"] = remapped_frames[-1]
    project_frames = [frame for frame in remapped_frames if str(frame.get("repo_path") or "").strip()]
    remapped["innermost_project_frame"] = project_frames[-1] if project_frames else remapped_frames[-1]
    editable_frames = [
        frame for frame in remapped_frames
        if str(canonical_source_target(frame.get("repo_path") or frame.get("file") or "")).startswith("ts_benchmark/")
        and "site-packages" not in str(frame.get("file") or "").replace("\\", "/").lower()
        and "dist-packages" not in str(frame.get("file") or "").replace("\\", "/").lower()
    ]
    remapped["innermost_editable_frame"] = editable_frames[-1] if editable_frames else {}
    innermost = dict(
        remapped.get("innermost_editable_frame")
        or remapped.get("innermost_project_frame")
        or remapped.get("innermost_frame")
        or {}
    )
    if innermost:
        primary_file = str(innermost.get("repo_path") or innermost.get("file") or "")
        remapped["repair_scope_hint"] = {
            "status": "traceback_local",
            "primary_file": primary_file,
            "canonical_target_file": canonical_source_target(primary_file),
            "primary_line": innermost.get("line"),
            "primary_function": innermost.get("function"),
            "required_evidence": (
                "Diagnose this traceback frame, but exact_edits target_file must use "
                "canonical_target_file, not a workspace path."
            ),
        }
    return remapped


def _probe_device(torch_module: Any):
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if getattr(torch_module.backends, "mps", None) is not None and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def _move_tensor_tree(value: Any, device: Any, torch_module: Any) -> Any:
    """Move tensor attributes hidden in simple Python containers to the probe device."""
    if torch_module.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tensor_tree(item, device, torch_module) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensor_tree(item, device, torch_module) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree(item, device, torch_module) for item in value)
    if isinstance(value, set):
        return {_move_tensor_tree(item, device, torch_module) for item in value}
    return value


def _align_module_device(root: Any, device: Any, torch_module: Any) -> None:
    if root is None:
        return
    if hasattr(root, "to"):
        root.to(device)
    modules = [root]
    if hasattr(root, "modules"):
        try:
            modules.extend(list(root.modules()))
        except Exception:
            pass
    for module in modules:
        if hasattr(module, "device"):
            try:
                module.device = device
            except Exception:
                pass
        attributes = getattr(module, "__dict__", {}) or {}
        if not isinstance(attributes, dict):
            continue
        for name, value in attributes.items():
            if name.startswith("_"):
                continue
            try:
                moved = _move_tensor_tree(value, device, torch_module)
            except Exception:
                continue
            if moved is not value:
                try:
                    setattr(module, name, moved)
                except Exception:
                    pass


def _int_hparam(hparams: Dict[str, Any], *keys: str, default: int) -> int:
    for key in keys:
        value = hparams.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return int(default)


def _dataset_value_columns(tfb_config: Dict[str, Any]) -> List[str]:
    data_config = dict(tfb_config.get("data_config") or {})
    semantics = dict(data_config.get("task_semantics") or {})
    dataset_path = str(semantics.get("dataset_path") or data_config.get("dataset_path") or "").strip()
    time_col = str(semantics.get("time_col") or data_config.get("time_col") or "date")
    if not dataset_path:
        return []
    path = Path(dataset_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return []
    try:
        header = path.read_text(encoding="utf-8", errors="replace").splitlines()[0].split(",")
    except Exception:
        return []
    return [col.strip() for col in header if col.strip() and col.strip() != time_col]


def _target_value_columns(tfb_config: Dict[str, Any]) -> List[str]:
    data_config = dict(tfb_config.get("data_config") or {})
    semantics = dict(data_config.get("task_semantics") or {})
    return [
        str(col)
        for col in list(semantics.get("target_columns") or data_config.get("target_columns") or [])
        if str(col).strip()
    ]


def _target_channel_count(tfb_config: Dict[str, Any], *, input_channels: int, fallback: int = 1) -> int:
    target_columns = _target_value_columns(tfb_config)
    if target_columns:
        return max(1, len(target_columns))
    strategy_args = dict((tfb_config.get("evaluation_config") or {}).get("strategy_args") or {})
    target_channel = strategy_args.get("target_channel")
    if target_channel is None:
        return max(1, int(input_channels or fallback or 1))
    if isinstance(target_channel, list):
        count = 0
        for item in target_channel:
            if isinstance(item, int) and not isinstance(item, bool):
                count += 1
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    start = int(item[0])
                    end = int(item[1])
                except Exception:
                    continue
                if start < 0:
                    start = max(0, input_channels + start)
                if end < 0:
                    end = max(0, input_channels + end)
                count += max(0, min(input_channels, end) - max(0, start))
        if count > 0:
            return count
    return max(1, int(fallback or 1))


def _runtime_channels_from_config(tfb_config: Dict[str, Any], fallback: int = 1) -> int:
    columns = _dataset_value_columns(tfb_config)
    if columns:
        return max(1, len(columns))
    return max(1, int(fallback))


def _complete_channel_hparams(hparams: Dict[str, Any], tfb_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Fill the channel aliases normally injected by the forecasting data path."""
    result = dict(hparams or {})
    fallback = _int_hparam(result, "c_out", "enc_in", "dec_in", default=1)
    channel = _runtime_channels_from_config(tfb_config or {}, fallback=fallback)
    result["enc_in"] = channel
    result["dec_in"] = channel
    result["c_out"] = channel
    return result


def _resolve_channel_contract(tfb_config: Dict[str, Any], hparams: Dict[str, Any]) -> Dict[str, Any]:
    data_config = dict(tfb_config.get("data_config") or {})
    semantics = dict(data_config.get("task_semantics") or {})
    strategy_args = dict((tfb_config.get("evaluation_config") or {}).get("strategy_args") or {})
    fallback = _int_hparam(hparams, "c_out", "enc_in", "dec_in", default=1) or 1
    input_channels = _runtime_channels_from_config(tfb_config, fallback=fallback)
    raw_output_channels = max(1, _int_hparam(hparams, "c_out", "enc_in", "dec_in", default=input_channels) or input_channels)
    target_channels = _target_channel_count(tfb_config, input_channels=input_channels, fallback=raw_output_channels)
    target_channels = min(max(1, target_channels), max(input_channels, raw_output_channels))
    task_mode = str(semantics.get("task_mode") or data_config.get("task_mode") or "").upper()
    needs_target_slice = target_channels < raw_output_channels
    return {
        "task_mode": task_mode,
        "input_channels": input_channels,
        "raw_output_channels": raw_output_channels,
        "target_channels": target_channels,
        "target_columns": _target_value_columns(tfb_config),
        "target_channel": strategy_args.get("target_channel"),
        "needs_target_slice": needs_target_slice,
    }


def _complete_task_hparams(hparams: Dict[str, Any], tfb_config: Dict[str, Any]) -> Dict[str, Any]:
    """Fill task/shape aliases required by heterogeneous TFB adapters."""
    result = _complete_channel_hparams(hparams, tfb_config)
    model_hp = dict((tfb_config.get("model_config") or {}).get("recommend_model_hyper_params") or {})
    strategy_args = dict((tfb_config.get("evaluation_config") or {}).get("strategy_args") or {})
    semantics = dict((tfb_config.get("data_config") or {}).get("task_semantics") or {})
    horizons = list(semantics.get("horizons") or [])

    seq_len = _int_hparam(
        {**model_hp, **result},
        "seq_len",
        "input_chunk_length",
        default=int(semantics.get("input_chunk_length") or strategy_args.get("horizon") or 96),
    )
    horizon = _int_hparam(
        {**model_hp, **strategy_args, **result},
        "horizon",
        "pred_len",
        "output_chunk_length",
        default=int((horizons[0] if horizons else None) or strategy_args.get("horizon") or seq_len),
    )
    result.setdefault("seq_len", seq_len)
    result.setdefault("input_chunk_length", seq_len)
    result.setdefault("horizon", horizon)
    result.setdefault("pred_len", horizon)
    result.setdefault("output_chunk_length", horizon)
    result.setdefault("label_len", max(1, seq_len // 2))
    result.setdefault("norm", model_hp.get("norm", True))
    return result


def _complete_compatible_task_hparams(
    model_name: str,
    hparams: Dict[str, Any],
    tfb_config: Dict[str, Any],
) -> tuple[Dict[str, Any], List[str]]:
    completed = _complete_task_hparams(hparams, tfb_config)
    model_key = str(model_name or "").rsplit(".", 1)[-1]
    if not model_key:
        return completed, []
    return apply_model_hparam_compatibility(model_key, completed, tfb_config)


def _model_default_hparams(model_cls: Any) -> Dict[str, Any]:
    """Return adapter-level MODEL_HYPER_PARAMS for the loaded model class.

    TFB adapters often keep executable constructor defaults in a module-level
    MODEL_HYPER_PARAMS dict and only receive sparse overrides from the runner.
    Runtime probes must see those defaults before compatibility is applied;
    otherwise hidden defaults such as patch_len > seq_len bypass the universal
    patch/window safety pass.
    """
    try:
        module = sys.modules.get(getattr(model_cls, "__module__", ""))
        defaults = getattr(module, "MODEL_HYPER_PARAMS", {}) if module is not None else {}
        return dict(defaults or {}) if isinstance(defaults, dict) else {}
    except Exception:
        return {}


def _merge_source_defaults(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults or {})
    merged.update(dict(overrides or {}))
    return merged


def _shape_tree(value: Any) -> Any:
    """Return a JSON-friendly shape tree for tensors nested in simple containers."""
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(value):
        return {"kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_shape_tree(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_shape_tree(item) for item in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": {str(key): _shape_tree(item) for key, item in value.items()}}
    return {"kind": type(value).__name__}


def _tensor_shape_tree(value: Any) -> Any:
    """Shape-only tree used for strict baseline/variant contract comparisons."""
    shaped = _shape_tree(value)
    if isinstance(shaped, dict) and shaped.get("kind") == "tensor":
        return {"kind": "tensor", "shape": shaped.get("shape")}
    if isinstance(shaped, dict) and "items" in shaped:
        return {"kind": shaped.get("kind"), "items": shaped.get("items")}
    return shaped


def _normalize_fit_point_path(path: str) -> Tuple[str, str]:
    raw = str(path or "").strip()
    if raw.startswith("self."):
        raw = raw[len("self.") :]
    if raw.startswith("inner."):
        raw = raw[len("inner.") :]
    if raw.startswith("model."):
        return "inner", raw[len("model.") :]
    return "outer", raw


def _resolve_child(root: Any, dotted_path: str) -> Any:
    obj = root
    for part in [item for item in str(dotted_path or "").split(".") if item]:
        obj = getattr(obj, part)
    return obj


def _target_module(root_model: Any, fit_point: str) -> Tuple[Any, Dict[str, Any]]:
    scope, child_path = _normalize_fit_point_path(fit_point)
    root = root_model
    if scope == "inner":
        root = getattr(root_model, "model", None)
        if root is None:
            raise AttributeError("model has no inner self.model for fit_point " + str(fit_point))
    target = _resolve_child(root, child_path)
    info = {
        "scope": scope,
        "path": child_path,
        "module_type": type(target).__name__,
        "module_class": f"{type(target).__module__}.{type(target).__name__}",
        "forward_signature": "",
    }
    forward = getattr(target, "forward", None)
    if callable(forward):
        try:
            info["forward_signature"] = str(inspect.signature(forward))
        except Exception:
            info["forward_signature"] = "(signature unavailable)"
    return target, info


def _instantiate_probe_model(
    *,
    tfb_config: Dict[str, Any],
    model_entry: Dict[str, Any],
    seed: int,
    variant_path: str = "",
) -> Tuple[Any, Dict[str, Any], Any, Dict[str, Any]]:
    import torch

    normalized_entry = dict(model_entry)
    model_name = str(normalized_entry.get("model_name") or "")
    if not variant_path:
        # A prior variant load can leave workspace-shadowed ts_benchmark
        # modules in sys.modules.  Baseline probes must import the repository
        # package, otherwise package __init__ re-exports resolve to the edited
        # workspace copy or to an incomplete workspace marker.
        for loaded_name in list(sys.modules):
            if loaded_name == "ts_benchmark" or loaded_name.startswith("ts_benchmark."):
                sys.modules.pop(loaded_name, None)
        importlib.invalidate_caches()
    compatibility_model_key = str(
        normalized_entry.get("model_key")
        or normalized_entry.get("source_model_key")
        or normalized_entry.get("base_model_key")
        or model_name
    )
    compatible_hparams, _compatibility_notes = _complete_compatible_task_hparams(
        compatibility_model_key,
        dict(model_entry.get("model_hyper_params") or {}),
        tfb_config,
    )
    normalized_entry["model_hyper_params"] = compatible_hparams
    if variant_path:
        from ts_benchmark.baselines import ADAPTER
        from evocast.variant.workspace_loader import load_model_class

        model_cls = load_model_class(variant_path=variant_path, model_name=model_name)
        source_defaults = _model_default_hparams(model_cls)
        if source_defaults:
            compatible_hparams, _compatibility_notes = _complete_compatible_task_hparams(
                compatibility_model_key,
                _merge_source_defaults(source_defaults, model_entry.get("model_hyper_params") or {}),
                tfb_config,
            )
            normalized_entry["model_hyper_params"] = compatible_hparams
        adapter_name = str(normalized_entry.get("adapter") or "").strip()
        if adapter_name:
            adapter_path = str(ADAPTER.get(adapter_name) or "").strip()
            if not adapter_path:
                raise ValueError(f"Unknown adapter {adapter_name}")
            module_name, attr_name = adapter_path.rsplit(".", 1)
            adapter_fn = getattr(importlib.import_module(module_name), attr_name)
            model_info = adapter_fn(model_cls)
            model_factory = model_info.get("model_factory") if isinstance(model_info, dict) else model_info
            if not callable(model_factory):
                raise TypeError(f"adapter {adapter_name} did not return a callable model_factory")
            model = model_factory(**dict(normalized_entry.get("model_hyper_params") or {}))
        else:
            model = model_cls(**dict(normalized_entry.get("model_hyper_params") or {}))
        factory = SimpleNamespace(model_hyper_params=dict(normalized_entry.get("model_hyper_params") or {}))
    else:
        from ts_benchmark.models.model_loader import get_models

        _, model_config, _ = build_run_configs(
            tfb_config,
            [normalized_entry],
            save_path="EvoCast_contract_probe",
            seed=seed,
        )
        factory = get_models(model_config)[0]
        model = factory()
    device = _probe_device(torch)
    if hasattr(model, "device"):
        model.device = device
    if not hasattr(model, "model") and hasattr(model, "_init_model"):
        model.model = model._init_model()
    _align_module_device(model, device, torch)
    if hasattr(model, "model"):
        _align_module_device(model.model, device, torch)
    if hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()
    hparams, _ = _complete_compatible_task_hparams(
        compatibility_model_key,
        dict(getattr(factory, "model_hyper_params", {}) if factory is not None else normalized_entry.get("model_hyper_params") or {}),
        tfb_config,
    )
    return model, hparams, device, normalized_entry


def _probe_inputs(hparams: Dict[str, Any], device: Any, torch_module: Any) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
    seq_len = _int_hparam(hparams, "seq_len", "input_chunk_length", default=96)
    horizon = _int_hparam(hparams, "horizon", "pred_len", "output_chunk_length", default=96)
    label_len = _int_hparam(hparams, "label_len", default=max(1, seq_len // 2))
    channels = _int_hparam(hparams, "enc_in", "c_out", "dec_in", default=1)
    batch = 2
    target_len = label_len + horizon
    dtype = torch_module.float32
    pi = 3.141592653589793
    time_x = torch_module.linspace(-1.0, 1.0, steps=seq_len, device=device, dtype=dtype).view(1, seq_len, 1)
    time_y = torch_module.linspace(0.75, -0.75, steps=target_len, device=device, dtype=dtype).view(1, target_len, 1)
    batch_scale = torch_module.arange(1, batch + 1, device=device, dtype=dtype).view(batch, 1, 1)
    channel_scale = torch_module.arange(1, channels + 1, device=device, dtype=dtype).view(1, 1, channels)
    x = batch_scale * channel_scale * (
        time_x + 0.1 * torch_module.sin(time_x * (3.0 * pi))
    )
    target = batch_scale * channel_scale * (
        time_y + 0.05 * torch_module.cos(time_y * (2.0 * pi))
    )
    input_mark = torch_module.zeros(batch, seq_len, 4, device=device)
    target_mark = torch_module.zeros(batch, target_len, 4, device=device)
    return x, target, input_mark, target_mark, {
        "input_shape": list(x.shape),
        "target_shape": list(target.shape),
        "input_mark_shape": list(input_mark.shape),
        "target_mark_shape": list(target_mark.shape),
        "expected_output_shape": [batch, horizon, channels],
        "probe_signal": "deterministic_nonzero_wave",
    }


def _run_probe(model: Any, x: Any, target: Any, input_mark: Any, target_mark: Any) -> Any:
    if hasattr(model, "_process"):
        out = model._process(x, target, input_mark, target_mark)
        normalized = normalize_tfb_process_output(out)
        if normalized.get("nested_output_dict"):
            raise RuntimeError(str(normalized.get("error_message") or "objective_boundary_mismatch"))
        return normalized.get("prediction")
    if hasattr(model, "forward"):
        return model(x)
    raise RuntimeError("model has no _process or forward method for contract probe")


def _read_variant_source(variant_path: str) -> str:
    if not str(variant_path or "").strip():
        return ""
    try:
        from pathlib import Path

        path = Path(str(variant_path))
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _required_adapter_state_attrs_from_source(source: str) -> List[str]:
    for line in str(source or "").splitlines():
        if not line.startswith("# required_adapter_state_attrs="):
            continue
        raw = line.split("=", 1)[1] if "=" in line else ""
        return [item.strip() for item in raw.split(",") if item.strip()]
    return []


def _variant_module_from_path(variant_path: str) -> str:
    """Derive Python module name from variant file path.

    Delegates to workspace_loader for canonical naming.
    """
    from evocast.variant.workspace_loader import workspace_module_name
    return workspace_module_name(variant_path)


def _is_research_variant_module(module_name: str) -> bool:
    """True for generated workspace variant modules."""
    name = str(module_name or "")
    return name.startswith("evocast_workspace")


def _purge_research_variant_module(module_name: str) -> None:
    if not _is_research_variant_module(module_name):
        return
    importlib.invalidate_caches()
    for loaded_name in list(sys.modules):
        if loaded_name == module_name or loaded_name.startswith(module_name + "."):
            sys.modules.pop(loaded_name, None)
    rel_path = Path(*str(module_name or "").split(".")).with_suffix(".py")
    pycache = rel_path.parent / "__pycache__"
    if pycache.exists():
        stem = rel_path.stem
        for cached in pycache.glob(f"{stem}*.pyc"):
            try:
                cached.unlink()
            except OSError:
                pass


def _purge_variant_runtime_imports() -> None:
    importlib.invalidate_caches()
    for loaded_name in list(sys.modules):
        if (
            loaded_name.startswith("evocast_workspace")
            or loaded_name == "ts_benchmark"
            or loaded_name.startswith("ts_benchmark.")
        ):
            sys.modules.pop(loaded_name, None)


def _assert_variant_source_static_contract(source: str, variant_path: str) -> None:
    if not str(source or "").strip():
        return
    try:
        import ast
    except Exception:
        return
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Name)
            and func.value.id in {"config", "configs"}
        ):
            continue
        raise RuntimeError(
            "variant uses config.get(...)/configs.get(...), but TFB model configs are attribute objects. "
            "Use existing config attributes or getattr(configs, 'field', default) for optional architecture-only fields. "
            f"variant_path={variant_path}"
        )


def _class_owner(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__name__}"


def _callable_owner(value: Any) -> str:
    if value is None:
        return ""
    return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', '')}".strip(".")


def _entry_class_from_model_name(model_name: str) -> Any:
    raw = str(model_name or "")
    if raw.startswith("global."):
        raw = raw[len("global.") :]
    if "." not in raw:
        return None
    module_name, attr = raw.rsplit(".", 1)
    try:
        _purge_research_variant_module(module_name)
        module = importlib.import_module(module_name)
        return getattr(module, attr, None)
    except Exception:
        return None


def _entry_class_from_variant_path(variant_path: str) -> Any:
    module_name = _variant_module_from_path(variant_path)
    module = sys.modules.get(module_name)
    if module is not None:
        return getattr(module, "Model", None)
    return None


def _factory_inner_model_class(factory: Any, model: Any) -> Any:
    direct = getattr(factory, "model_class", None)
    if direct is not None:
        return direct
    model_class = getattr(model, "model_class", None)
    if model_class is not None:
        return model_class
    closure = getattr(getattr(factory, "model_factory", None), "__closure__", None) or []
    for cell in closure:
        try:
            value = cell.cell_contents
        except Exception:
            continue
        if isinstance(value, type):
            return value
    return None


def _source_entrypoint_context(
    *,
    factory: Any,
    model: Any,
    variant_entry: Dict[str, Any],
    variant_path: str,
) -> Dict[str, Any]:
    source = _read_variant_source(variant_path)
    variant_module = _variant_module_from_path(variant_path)
    defined_symbols = source_defined_symbols(source)
    edited_source_files = set(_edited_source_files_from_variant_source(source))
    entry_class = (
        _entry_class_from_variant_path(variant_path)
        if is_workspace_variant_path(variant_path)
        else _entry_class_from_model_name(str(variant_entry.get("model_name") or ""))
    )
    factory_model_class = _factory_inner_model_class(factory, model)
    inner_model = getattr(model, "model", None)
    try:
        factory_model_source = str(inspect.getsourcefile(factory_model_class) or "").replace("\\", "/") if factory_model_class is not None else ""
    except Exception:
        factory_model_source = ""
    try:
        inner_model_source = str(inspect.getsourcefile(type(inner_model)) or "").replace("\\", "/") if inner_model is not None else ""
    except Exception:
        inner_model_source = ""
    workspace_root = str(Path(variant_path).resolve().parent).replace("\\", "/")
    requires_workspace_binding = is_workspace_variant_path(variant_path)
    workspace_source_bound = bool(
        workspace_root
        and (
            (factory_model_source and factory_model_source.startswith(workspace_root + "/"))
            or (inner_model_source and inner_model_source.startswith(workspace_root + "/"))
        )
    )
    # P0-fix: actual_root must be the OUTER model (wrapper) for identity checks.
    # The variant IS the wrapper (Model(HDMixer), Model(DTAF), etc.), not the inner
    # model.  Using inner_model as actual_root causes smoke activation_check to
    # resolve actual_runtime_root_class to the baseline inner model class (e.g.
    # HDMixerModel) instead of the variant's Model class — which makes
    # variant_module_loaded always False for wrapper+inner-model families.
    # The inner model is still tracked separately via inner_model_class.
    actual_root = model
    variant_wraps_inner = inner_model is not None
    runtime_modules = collect_module_class_paths(model)
    runtime_inner_modules = collect_module_class_paths(inner_model)
    variant_runtime_modules = [
        item for item in runtime_modules + runtime_inner_modules
        if str(item.get("class_path") or "").startswith(variant_module + ".")
    ]
    defined_class_names: set[str] = set()
    for item in list(defined_symbols.get("classes") or []):
        class_name = str(item.get("name") or "")
        if not class_name or class_name == "Model":
            continue
        if edited_source_files:
            source_file = _source_file_for_lineno(source, int(item.get("lineno") or 0))
            if source_file and source_file not in edited_source_files:
                continue
        defined_class_names.add(class_name)
    bound_variant_class_names = {
        str(item.get("class_path") or "").rsplit(".", 1)[-1]
        for item in variant_runtime_modules
    }
    allowed_unbound_definitions: set[str] = set()
    if entry_class is not None:
        entry_name = str(getattr(entry_class, "__name__", "") or "").strip()
        if entry_name:
            allowed_unbound_definitions.add(entry_name)
    entry_init_owner = _callable_owner(getattr(entry_class, "__init__", None)) if entry_class is not None else ""
    if entry_init_owner and "." in entry_init_owner:
        wrapper_name = entry_init_owner.rsplit(".", 2)[-2]
        if wrapper_name:
            allowed_unbound_definitions.add(wrapper_name)
    ineffective_local_definitions = sorted(
        defined_class_names - bound_variant_class_names - allowed_unbound_definitions
    )
    context = {
        "variant_module": variant_module,
        "defined_symbols": defined_symbols,
        "entry_class": f"{getattr(entry_class, '__module__', '')}.{getattr(entry_class, '__name__', '')}".strip(".")
        if entry_class is not None
        else "",
        "entry_mro": [
            f"{getattr(cls, '__module__', '')}.{getattr(cls, '__name__', '')}".strip(".")
            for cls in list(getattr(entry_class, "__mro__", []) or [])[:6]
        ]
        if entry_class is not None
        else [],
        "entry_init_owner": _callable_owner(getattr(entry_class, "__init__", None)) if entry_class is not None else "",
        "entry_init_model_owner": _callable_owner(getattr(entry_class, "_init_model", None)) if entry_class is not None else "",
        "factory_model_class": f"{getattr(factory_model_class, '__module__', '')}.{getattr(factory_model_class, '__name__', '')}".strip(".")
        if factory_model_class is not None
        else "",
        "factory_model_source": factory_model_source,
        "inner_model_source": inner_model_source,
        "workspace_root": workspace_root,
        "requires_workspace_binding": requires_workspace_binding,
        "workspace_source_bound": workspace_source_bound,
        "adapter_model_class_matches_entry": bool(factory_model_class is entry_class) if entry_class is not None else None,
        "outer_model_class": _class_owner(model),
        "inner_model_class": _class_owner(inner_model) if inner_model is not None else "",
        "actual_runtime_root_class": _class_owner(actual_root),
        "actual_runtime_root_init_owner": _callable_owner(getattr(type(actual_root), "__init__", None)),
        "runtime_variant_modules": variant_runtime_modules[:80],
        "edited_source_files": sorted(edited_source_files),
        "ineffective_local_definitions": ineffective_local_definitions,
        "binding_warnings": [],
        "has_bundled_inner_alias": "_EVOCAST_VARIANT_INNER_MODEL" in source,
        "uses_wrapper_entrypoint_text": "_EVOCAST_VARIANT_INNER_MODEL" in source and "def _init_model" in source,
        "transformer_inner_entrypoint_text": "transformer_adapter instantiates this inner nn.Module directly" in source,
    }
    if ineffective_local_definitions:
        context["binding_warnings"].append(
            "Variant defines local classes that were not observed in the instantiated runtime module tree: "
            + ", ".join(ineffective_local_definitions[:8])
        )
    if (
        defined_class_names
        and not variant_runtime_modules
        and not str(context.get("actual_runtime_root_class") or "").startswith(variant_module + ".")
    ):
        context["binding_warnings"].append(
            "No non-entry variant-defined module was bound into the runtime model; edits may be unreachable."
        )
    return context


def _validate_source_entrypoint_effectiveness(context: Dict[str, Any]) -> None:
    entry_class = str(context.get("entry_class") or "")
    factory_model_class = str(context.get("factory_model_class") or "")
    if factory_model_class and entry_class and factory_model_class != entry_class:
        raise RuntimeError(
            "source variant adapter did not receive the exported Model class: "
            f"factory_model_class={factory_model_class}, entry_class={entry_class}"
        )
    if context.get("requires_workspace_binding") and not context.get("workspace_source_bound"):
        raise RuntimeError(
            "workspace variant source was not bound into the runtime model: "
            f"workspace_root={context.get('workspace_root')}, "
            f"factory_model_source={context.get('factory_model_source')}, "
            f"inner_model_source={context.get('inner_model_source')}"
        )
    entry_module = entry_class.rsplit(".", 1)[0] if "." in entry_class else entry_class
    if context.get("has_bundled_inner_alias") and not str(context.get("inner_model_class") or "").startswith(entry_module):
        raise RuntimeError(
            "source variant bundled inner model was not instantiated; "
            f"inner_model_class={context.get('inner_model_class')}, entry_class={entry_class}"
        )
    if (
        context.get("uses_wrapper_entrypoint_text")
        and str(context.get("actual_runtime_root_class") or "") == entry_class
        and "time_series_library.models." in str(context.get("entry_init_owner") or "")
    ):
        raise RuntimeError(
            "source variant defines a wrapper _init_model entrypoint, but runtime instantiated "
            "the exported Model as a raw transformer inner module; bundled edits are unreachable"
        )


def _validate_adapter_state_contract(model: Any, source: str) -> Dict[str, Any]:
    required = _required_adapter_state_attrs_from_source(source)
    if not required:
        return {"required_attrs": [], "missing_attrs": [], "status": "skipped"}
    missing = [attr for attr in required if not hasattr(model, attr)]
    if missing:
        raise RuntimeError(
            "adapter_state_contract_violation: source variant wrapper did not preserve required "
            "adapter state attributes: " + ", ".join(missing)
        )
    return {"required_attrs": required, "missing_attrs": [], "status": "ok"}


def _probe_target_io(model: Any, target_module: Any, inputs: Tuple[Any, Any, Any, Any], torch_module: Any) -> Dict[str, Any]:
    observed: Dict[str, Any] = {"called": False, "input": None, "output": None}

    def pre_hook(_module: Any, args: Tuple[Any, ...]) -> None:
        observed["called"] = True
        observed["input"] = _tensor_shape_tree(args)

    def hook(_module: Any, _args: Tuple[Any, ...], output: Any) -> None:
        observed["output"] = _tensor_shape_tree(output)

    handles = [
        target_module.register_forward_pre_hook(pre_hook),
        target_module.register_forward_hook(hook),
    ]
    try:
        with torch_module.no_grad():
            output = _run_probe(model, *inputs)
        if isinstance(output, tuple):
            output = output[0]
        observed["model_output"] = _tensor_shape_tree(output)
    finally:
        for handle in handles:
            handle.remove()
    return observed


# ── P1-2: Target hook delta capture ───────────────────────────────────────────

def _capture_target_hook_delta(
    baseline_model: Any,
    variant_model: Any,
    fit_point: str,
    baseline_inputs: tuple,
    variant_inputs: tuple,
    device: Any,
    variant_device: Any,
) -> Dict[str, Any] | None:
    """Capture intermediate outputs at the fit_point hook for both models.

    Returns None if the hook cannot be registered (fit_point not found or
    model structure doesn't support dotted-path access).
    """
    import torch

    if not fit_point:
        return None
    target_path = fit_point
    if target_path.startswith("self."):
        target_path = target_path[5:]
    if target_path.startswith("inner."):
        target_path = target_path[6:]
    if not target_path:
        return None

    baseline_hook_out = {}
    variant_hook_out = {}

    def _hook_fn(outputs: dict):
        def hook(module, input, output):
            outputs["value"] = output
        return hook

    try:
        # Navigate dotted path on baseline
        baseline_target = baseline_model
        for part in target_path.split("."):
            baseline_target = getattr(baseline_target, part, None)
            if baseline_target is None:
                return None
        b_handle = baseline_target.register_forward_hook(_hook_fn(baseline_hook_out))

        # Navigate dotted path on variant
        variant_target = variant_model
        for part in target_path.split("."):
            variant_target = getattr(variant_target, part, None)
            if variant_target is None:
                b_handle.remove()
                return None
        v_handle = variant_target.register_forward_hook(_hook_fn(variant_hook_out))

        with torch.no_grad():
            _run_probe(baseline_model, *baseline_inputs)
            _run_probe(variant_model, *variant_inputs)

        b_handle.remove()
        v_handle.remove()

        b_val = baseline_hook_out.get("value")
        v_val = variant_hook_out.get("value")
        if b_val is None or v_val is None:
            return None

        # Compare hook outputs
        if isinstance(b_val, tuple):
            b_val = b_val[0]
        if isinstance(v_val, tuple):
            v_val = v_val[0]

        hook_comparison = compare_tensor_outputs(b_val, v_val)
        return {
            "fit_point": fit_point,
            "comparison": hook_comparison,
            "exact_equal": bool(hook_comparison.get("exact_equal")),
        }
    except Exception:
        return None


def _probe_variant_behavior_delta_in_process(
    *,
    tfb_config: Dict[str, Any],
    baseline_entry: Dict[str, Any],
    variant_entry: Dict[str, Any],
    variant_path: str = "",
    source_checkout: str | Path | None = None,
    source_entry_file: str = "",
    seed: int = 2021,
    fit_point: str = "",
) -> Dict[str, Any]:
    """Run a deterministic same-input baseline/variant probe and compare outputs.

    P1-2: Two-layer delta detection.
    - model_output_delta: compares final model outputs (existing behavior)
    - target_hook_delta: compares intermediate outputs at the fit_point hook

    If the target hook is available, it is the primary proof that the intended
    mechanism changed. Final outputs may remain nearly identical for localized
    edits that are compensated downstream, so hook-level delta is sufficient to
    avoid classifying the variant as an immediate noop. When the hook is
    unavailable, fall back to final-output comparison.
    """
    try:
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        baseline_model, baseline_hparams, device, _ = _instantiate_probe_model(
            tfb_config=tfb_config,
            model_entry=baseline_entry,
            seed=seed,
        )
        torch.manual_seed(seed)
        if source_checkout:
            with model_execution_import_context(source_checkout=source_checkout):
                variant_model, _variant_hparams, variant_device, _ = _instantiate_probe_model(
                    tfb_config=tfb_config,
                    model_entry=variant_entry,
                    seed=seed,
                    variant_path="",
                )
        else:
            variant_model, _variant_hparams, variant_device, _ = _instantiate_probe_model(
                tfb_config=tfb_config,
                model_entry=variant_entry,
                seed=seed,
                variant_path=variant_path,
            )
        inputs = _probe_inputs(baseline_hparams, device, torch)[:4]
        variant_inputs = tuple(item.to(variant_device) for item in inputs)

        # ── P1-2: target_hook_delta — capture intermediate output at fit_point ──
        target_hook_delta = _capture_target_hook_delta(
            baseline_model, variant_model,
            fit_point, inputs, variant_inputs,
            device, variant_device,
        )

        # ── model_output_delta — compare final outputs ──
        with torch.no_grad():
            baseline_output = _run_probe(baseline_model, *inputs)
            variant_output = _run_probe(variant_model, *variant_inputs)
        if isinstance(baseline_output, tuple):
            baseline_output = baseline_output[0]
        if isinstance(variant_output, tuple):
            variant_output = variant_output[0]
        model_comparison = compare_tensor_outputs(baseline_output, variant_output)
        max_abs_diff = float(model_comparison.get("max_abs_diff") or 0.0)
        mean_abs_diff = float(model_comparison.get("mean_abs_diff") or 0.0)
        model_output_identical = bool(
            model_comparison.get("same_shape")
            and (
                model_comparison.get("exact_equal")
                or (max_abs_diff <= 1e-6 and mean_abs_diff <= 1e-7)
            )
        )

        target_hook_identical = bool(
            (target_hook_delta or {}).get("exact_equal", True)
        ) if target_hook_delta else None  # None = hook unavailable, fall through
        if target_hook_identical is None:
            suspected_noop = model_output_identical
        else:
            suspected_noop = target_hook_identical

        reason_parts = []
        if model_output_identical:
            reason_parts.append("model output identical to baseline")
        if target_hook_identical:
            reason_parts.append(f"target hook at '{fit_point}' identical to baseline")
        elif target_hook_identical is None:
            reason_parts.append(f"target hook at '{fit_point}' unavailable")
        elif model_output_identical:
            reason_parts.append(f"target hook at '{fit_point}' changed")

        return {
            "status": "ok",
            "stage": "behavior_delta_probe",
            "variant_path": source_entry_file or variant_path,
            "source_checkout": str(source_checkout or "") or None,
            "comparison": model_comparison,
            "target_hook_delta": target_hook_delta,
            "suspected_noop": suspected_noop,
            "reason": (
                "; ".join(reason_parts)
                if suspected_noop
                else "baseline and variant probe outputs differ"
            ),
        }
    except Exception as exc:
        tb = traceback.format_exc()
        return {
            "status": "error",
            "stage": "behavior_delta_probe",
            "variant_path": source_entry_file or variant_path,
            "source_checkout": str(source_checkout or "") or None,
            "error_type": f"{type(exc).__name__}",
            "error_message": f"{type(exc).__name__}: {exc}",
            "error_traceback": tb,
            "failure_evidence": extract_traceback_evidence(tb),
        }


def validate_fitpoint_runtime_contract(
    *,
    tfb_config: Dict[str, Any],
    baseline_entry: Dict[str, Any],
    variant_entry: Dict[str, Any],
    variant_path: str,
    fit_point: str,
    seed: int = 2021,
) -> Dict[str, Any]:
    """Compare baseline and variant target-module IO for shape-preserving inner edits.

    This probe is intentionally stricter than the whole-model runtime contract:
    generated inner-model adapters must preserve the selected fit point's call
    signature and tensor shape tree unless the harness has a specialized
    transformation for that model family.
    """
    if not str(fit_point or "").strip():
        return {"status": "skipped", "reason": "missing fit_point"}
    probe_context: Dict[str, Any] = {"fit_point": fit_point}
    try:
        import torch

        baseline_model, baseline_hparams, device, _ = _instantiate_probe_model(
            tfb_config=tfb_config,
            model_entry=baseline_entry,
            seed=seed,
        )
        variant_model, variant_hparams, variant_device, _ = _instantiate_probe_model(
            tfb_config=tfb_config,
            model_entry=variant_entry,
            seed=seed,
            variant_path=variant_path,
        )
        baseline_target, baseline_target_info = _target_module(baseline_model, fit_point)
        variant_target, variant_target_info = _target_module(variant_model, fit_point)
        inputs = _probe_inputs(baseline_hparams, device, torch)[:4]
        variant_inputs = tuple(item.to(variant_device) for item in inputs)
        probe_context.update(
            {
                "baseline_target": baseline_target_info,
                "variant_target": variant_target_info,
                "probe_input": _probe_inputs(baseline_hparams, device, torch)[4],
            }
        )
        baseline_observed = _probe_target_io(baseline_model, baseline_target, inputs, torch)
        variant_observed = _probe_target_io(variant_model, variant_target, variant_inputs, torch)
        probe_context["baseline_observed"] = baseline_observed
        probe_context["variant_observed"] = variant_observed

        errors: List[str] = []
        if not baseline_observed.get("called"):
            errors.append("baseline target module was not called during the probe")
        if not variant_observed.get("called"):
            errors.append("variant target module was not called during the probe")
        signature_warning = ""
        if baseline_target_info.get("forward_signature") != variant_target_info.get("forward_signature"):
            signature_warning = (
                "target forward signature changed: "
                f"{baseline_target_info.get('forward_signature')} -> {variant_target_info.get('forward_signature')}"
            )
        if baseline_observed.get("input") != variant_observed.get("input"):
            errors.append("target input shape tree changed")
        if baseline_observed.get("output") != variant_observed.get("output"):
            errors.append("target output shape tree changed")
        if baseline_observed.get("model_output") != variant_observed.get("model_output"):
            errors.append("whole-model output shape tree changed")
        if errors:
            raise RuntimeError("; ".join(errors))
        return {
            "status": "ok",
            "stage": "fitpoint_runtime_probe",
            **probe_context,
            "warnings": [signature_warning] if signature_warning else [],
        }
    except Exception as exc:
        tb = traceback.format_exc()
        return {
            "status": "failed",
            "stage": "fitpoint_runtime_probe",
            **probe_context,
            "error_type": "invalid_fitpoint_contract",
            "error_message": f"{type(exc).__name__}: {exc}",
            "error_traceback": tb,
            "failure_evidence": extract_traceback_evidence(tb),
            "failure_signature": failure_signature(
                error_type="invalid_fitpoint_contract",
                message=f"{type(exc).__name__}: {exc}",
                traceback_text=tb,
                variant_path=variant_path,
                stage="fitpoint_runtime_probe",
            ),
            "model_name": variant_entry.get("model_name"),
        }


def _validate_variant_runtime_contract_in_process(
    *,
    tfb_config: Dict[str, Any],
    variant_entry: Dict[str, Any],
    variant_path: str,
    source_checkout: str | Path | None = None,
    seed: int = 2021,
) -> Dict[str, Any]:
    """Instantiate a model entry and run a conservative adapter-level probe."""
    if source_checkout:
        from evocast.variant.import_isolation import model_execution_import_context

        with model_execution_import_context(source_checkout=source_checkout):
            return _validate_variant_runtime_contract_in_process(
                tfb_config=tfb_config,
                variant_entry=variant_entry,
                variant_path=variant_path,
                source_checkout=None,
                seed=seed,
            )

    probe_context: Dict[str, Any] = {}
    source_entrypoint_context: Dict[str, Any] = {}
    runtime_fact_pack: Dict[str, Any] = {}
    fact_tracer: RuntimeFactTracer | None = None
    try:
        import torch

        _purge_variant_runtime_imports()
        from ts_benchmark.models.model_loader import get_models
        variant_source = _read_variant_source(variant_path)
        _assert_variant_source_static_contract(variant_source, variant_path)
        module_name = _variant_module_from_path(variant_path)
        _purge_research_variant_module(module_name)
        normalized_entry = dict(variant_entry)
        if is_workspace_variant_path(variant_path):
            normalized_entry["variant_path"] = variant_path
            try:
                from evocast.variant.workspace_loader import load_model_class

                source_defaults = _model_default_hparams(
                    load_model_class(variant_path=variant_path, model_name=str(normalized_entry.get("model_name") or ""))
                )
            except Exception:
                source_defaults = {}
        else:
            source_defaults = {}
        compatibility_model_key = str(
            normalized_entry.get("model_key")
            or normalized_entry.get("source_model_key")
            or normalized_entry.get("base_model_key")
            or normalized_entry.get("model_name")
            or ""
        )
        compatible_hparams, compatibility_notes = _complete_compatible_task_hparams(
            compatibility_model_key,
            _merge_source_defaults(source_defaults, variant_entry.get("model_hyper_params") or {}),
            tfb_config,
        )
        normalized_entry["model_hyper_params"] = compatible_hparams
        _, model_config, _ = build_run_configs(
            tfb_config,
            [normalized_entry],
            save_path="EvoCast_contract_probe",
            seed=seed,
        )
        factory = get_models(model_config)[0]
        model = factory()
        device = _probe_device(torch)
        if normalized_entry.get("adapter") is None:
            missing_protocol = [
                name for name in ("forecast_fit", "forecast", "batch_forecast")
                if not hasattr(model, name)
            ]
            if missing_protocol:
                raise RuntimeError(
                    "model does not implement TFB forecasting protocol: "
                    + ", ".join(missing_protocol)
                    + ". Standalone research variants must subclass the baseline adapter "
                    "or be run through the correct adapter."
                )
        if hasattr(model, "device"):
            model.device = device
        if not hasattr(model, "model") and hasattr(model, "_init_model"):
            model.model = model._init_model()
        source_entrypoint_context = _source_entrypoint_context(
            factory=factory,
            model=model,
            variant_entry=normalized_entry,
            variant_path=variant_path,
        )
        source_entrypoint_context["model_hparam_compatibility_notes"] = compatibility_notes
        _validate_source_entrypoint_effectiveness(source_entrypoint_context)
        adapter_state_contract = _validate_adapter_state_contract(model, variant_source)
        source_entrypoint_context["adapter_state_contract"] = adapter_state_contract
        _align_module_device(model, device, torch)
        if hasattr(model, "model"):
            _align_module_device(model.model, device, torch)
        if hasattr(model, "model") and hasattr(model.model, "eval"):
            model.model.eval()

        hparams = _complete_task_hparams(
            dict(getattr(factory, "model_hyper_params", {}) or normalized_entry.get("model_hyper_params") or {}),
            tfb_config,
        )
        seq_len = _int_hparam(hparams, "seq_len", "input_chunk_length", default=96)
        horizon = _int_hparam(hparams, "horizon", "pred_len", "output_chunk_length", default=96)
        label_len = _int_hparam(hparams, "label_len", default=max(1, seq_len // 2))
        channel_contract = _resolve_channel_contract(tfb_config, hparams)
        input_channels = int(channel_contract["input_channels"])
        raw_output_channels = int(channel_contract["raw_output_channels"])
        target_channels = int(channel_contract["target_channels"])
        batch = 2

        mechanism_input_context = {
            "stage": "mechanism_probe",
            "task_shape": {
                "batch": batch,
                "seq_len": seq_len,
                "horizon": horizon,
                "channels": input_channels,
            },
            "channel_contract": channel_contract,
            "source_entrypoint": source_entrypoint_context,
        }
        mechanism_tracer = RuntimeFactTracer(model)
        mechanism_tracer.attach()
        try:
            mechanism_probe = run_mechanism_probe(
                model=model,
                variant_path=variant_path,
                source_entrypoint=source_entrypoint_context,
                task_shape=mechanism_input_context["task_shape"],
                variant_source=variant_source,
            )
        except Exception as mechanism_exc:
            runtime_fact_pack = mechanism_tracer.build(
                exception=mechanism_exc,
                input_context=mechanism_input_context,
            )
            raise
        finally:
            mechanism_tracer.close()
        probe_context["mechanism_probe"] = mechanism_probe
        if mechanism_probe.get("status") != "ok":
            mechanism_failures = [
                item for item in list(mechanism_probe.get("failures") or [])
                if isinstance(item, dict)
            ]
            first_mechanism_failure = mechanism_failures[0] if mechanism_failures else {}
            runtime_fact_pack = mechanism_tracer.build(
                input_context=mechanism_input_context,
                failure_traceback_text=str(first_mechanism_failure.get("traceback") or ""),
                failure_message=str(mechanism_probe.get("error_message") or first_mechanism_failure.get("error_message") or ""),
            )
            raise MechanismProbeError(mechanism_probe)

        target_len = label_len + horizon
        x = torch.zeros(batch, seq_len, input_channels, device=device)
        target = torch.zeros(batch, target_len, input_channels, device=device)
        input_mark = torch.zeros(batch, seq_len, 4, device=device)
        target_mark = torch.zeros(batch, target_len, 4, device=device)
        probe_context = {
            "input_shape": list(x.shape),
            "target_shape": list(target.shape),
            "input_mark_shape": list(input_mark.shape),
            "target_mark_shape": list(target_mark.shape),
            "expected_output_shape": [batch, horizon, raw_output_channels],
            "expected_eval_output_shape": [batch, horizon, target_channels],
            "channel_contract": channel_contract,
            "device": str(device),
            "source_entrypoint": source_entrypoint_context,
        }

        fact_tracer = RuntimeFactTracer(model)
        fact_tracer.attach()
        try:
            with torch.no_grad():
                if hasattr(model, "_process"):
                    out = model._process(x, target, input_mark, target_mark)
                    normalized = normalize_tfb_process_output(out)
                    if normalized.get("nested_output_dict"):
                        raise RuntimeError(str(normalized.get("error_message") or "objective_boundary_mismatch"))
                    output = normalized.get("prediction")
                    probe_context["objective_boundary"] = {
                        "has_additional_loss": bool(normalized.get("has_additional_loss")),
                        "additional_loss_shape": (
                            list(normalized.get("additional_loss").shape)
                            if hasattr(normalized.get("additional_loss"), "shape")
                            else None
                        ),
                    }
                elif hasattr(model, "forward"):
                    output = model(x)
                else:
                    raise RuntimeError("model has no _process or forward method for contract probe")
        except Exception as probe_exc:
            runtime_fact_pack = fact_tracer.build(
                exception=probe_exc,
                input_context=probe_context,
            )
            raise
        finally:
            fact_tracer.close()
        if isinstance(output, tuple):
            output = output[0]
        runtime_fact_pack = fact_tracer.build(
            output=output,
            input_context=probe_context,
        )
        if not hasattr(output, "shape"):
            raise RuntimeError(f"contract probe output has no shape: {type(output).__name__}")
        shape = list(output.shape)
        raw_expected = [batch, horizon, raw_output_channels]
        eval_expected = [batch, horizon, target_channels]
        slice_contract = {
            "status": "exact",
            "reason": "",
            "slice": None,
        }
        if len(shape) != 3 or shape[0] != batch:
            raise RuntimeError(f"contract probe output shape {shape} != expected batch contract {raw_expected}")
        if shape[2] not in {raw_output_channels, target_channels}:
            raise RuntimeError(
                f"contract probe output shape {shape} != expected raw/eval channel contract "
                f"{raw_expected} or {eval_expected}"
            )
        if shape[1] < horizon:
            raise RuntimeError(f"contract probe output shape {shape} is shorter than required horizon {horizon}")
        accepted_shape = [batch, horizon, target_channels]
        if shape[1] > horizon:
            slice_contract = {
                "status": "horizon_sliceable",
                "reason": "adapter/training code consumes output[:, -horizon:, :series_dim]",
                "slice": f"output[:, -{horizon}:, :{target_channels}]",
            }
        elif shape[2] == raw_output_channels and target_channels < raw_output_channels:
            slice_contract = {
                "status": "target_channel_sliceable",
                "reason": "MS/single-target evaluation consumes only the configured target channel(s) after the raw model output.",
                "slice": f"output[:, -{horizon}:, :{target_channels}]",
            }
        elif shape[2] == target_channels:
            slice_contract = {
                "status": "already_target_shaped",
                "reason": "raw model output already matches the evaluation target channel count.",
                "slice": None,
            }
        return {
            "status": "ok",
            "stage": "runtime_contract_probe",
            "input_shape": list(x.shape),
            "target_shape": list(target.shape),
            "output_shape": shape,
            "expected_output_shape": raw_expected,
            "expected_eval_output_shape": eval_expected,
            "accepted_output_shape": accepted_shape,
            "output_slice_contract": slice_contract,
            "channel_contract": {
                **channel_contract,
                "output_slice_contract": slice_contract,
            },
            "device": str(device),
            "model_name": variant_entry.get("model_name"),
            "source_entrypoint": source_entrypoint_context,
            "mechanism_probe": mechanism_probe,
            "runtime_fact_pack": runtime_fact_pack,
        }


    except Exception as exc:
        tb = traceback.format_exc()
        if not runtime_fact_pack and fact_tracer is not None:
            try:
                runtime_fact_pack = fact_tracer.build(
                    exception=exc,
                    input_context=probe_context,
                )
            finally:
                fact_tracer.close()
        mechanism_probe_result = exc.result if isinstance(exc, MechanismProbeError) else probe_context.get("mechanism_probe")
        wrapper_failure_evidence = _remap_traceback_evidence_to_source_files(
            extract_traceback_evidence(tb),
            variant_source,
            variant_path,
        )
        failure_evidence = wrapper_failure_evidence
        if isinstance(mechanism_probe_result, dict):
            mechanism_failures = [
                item for item in list(mechanism_probe_result.get("failures") or [])
                if isinstance(item, dict)
            ]
            mechanism_tb = str((mechanism_failures[0] if mechanism_failures else {}).get("traceback") or "")
            if mechanism_tb:
                mechanism_evidence = _remap_traceback_evidence_to_source_files(
                    extract_traceback_evidence(mechanism_tb),
                    variant_source,
                    variant_path,
                )
                mechanism_hint = mechanism_evidence.get("repair_scope_hint") if isinstance(mechanism_evidence, dict) else {}
                mechanism_target = str((mechanism_hint or {}).get("canonical_target_file") or "")
                if mechanism_target.startswith("ts_benchmark/"):
                    failure_evidence = dict(mechanism_evidence)
                    failure_evidence["wrapper_failure_evidence"] = wrapper_failure_evidence
                    failure_evidence["evidence_authority"] = "mechanism_probe_inner_traceback"
        error_type = "invalid_mechanism_contract" if isinstance(exc, MechanismProbeError) else "invalid_variant_contract"
        failure_chain = (
            dict(mechanism_probe_result.get("failure_chain") or {})
            if isinstance(mechanism_probe_result, dict)
            else {}
        )
        return {
            "status": "failed",
            "stage": "runtime_contract_probe",
            **probe_context,
            "error_type": error_type,
            "error_message": f"{type(exc).__name__}: {exc}",
            "error_traceback": tb,
            "failure_evidence": failure_evidence,
            "failure_chain": failure_chain,
            "failure_signature": failure_signature(
                error_type=error_type,
                message=f"{type(exc).__name__}: {exc}",
                traceback_text=tb,
                variant_path=variant_path,
                stage="runtime_contract_probe",
            ),
            "model_name": variant_entry.get("model_name"),
            "source_entrypoint": source_entrypoint_context,
            "runtime_fact_pack": runtime_fact_pack,
        }


def probe_variant_behavior_delta(
    *,
    tfb_config: Dict[str, Any],
    baseline_entry: Dict[str, Any],
    variant_entry: Dict[str, Any],
    variant_path: str = "",
    source_checkout: str | Path | None = None,
    source_entry_file: str = "",
    seed: int = 2021,
    fit_point: str = "",
    base_dir: str | None = None,
) -> Dict[str, Any]:
    """Compare candidate behavior in an isolated process with its own CUDA context."""
    from evocast.variant.runtime_validation import run_runtime_validation_worker

    return run_runtime_validation_worker(
        operation="behavior_delta",
        payload={
            "tfb_config": tfb_config,
            "baseline_entry": baseline_entry,
            "variant_entry": variant_entry,
            "variant_path": variant_path,
            "source_checkout": str(source_checkout) if source_checkout else None,
            "source_entry_file": source_entry_file,
            "seed": seed,
            "fit_point": fit_point,
        },
        base_dir=base_dir,
    )


def validate_variant_runtime_contract(
    *,
    tfb_config: Dict[str, Any],
    variant_entry: Dict[str, Any],
    variant_path: str,
    source_checkout: str | Path | None = None,
    seed: int = 2021,
    base_dir: str | None = None,
) -> Dict[str, Any]:
    """Validate candidate import, CUDA execution, and mechanism behavior in a worker."""
    from evocast.variant.runtime_validation import run_runtime_validation_worker

    return run_runtime_validation_worker(
        operation="runtime_contract",
        payload={
            "tfb_config": tfb_config,
            "variant_entry": variant_entry,
            "variant_path": variant_path,
            "source_checkout": str(source_checkout) if source_checkout else None,
            "seed": seed,
        },
        base_dir=base_dir,
    )
