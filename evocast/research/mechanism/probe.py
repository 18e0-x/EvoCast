"""Mechanism-level probes for exact-edit variants.

These probes run before expensive smoke/experiment execution.  They focus on
the failure class that full-model shape checks often miss: mechanism-local
branch and operator tensor contracts.
"""

from __future__ import annotations

import importlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from evocast.probe.failure_chain import build_failure_chain
from evocast.research.objective_boundary import normalize_tfb_process_output


class MechanismProbeError(RuntimeError):
    def __init__(self, result: Dict[str, Any]):
        self.result = result
        message = str(result.get("error_message") or "mechanism probe failed")
        super().__init__(message)


@dataclass(frozen=True)
class MechanismTaskShape:
    batch: int
    seq_len: int
    horizon: int
    channels: int


def _module_name_from_source(source_path: str) -> str:
    normalized = str(source_path or "").replace("\\", "/")
    marker = "/ts_benchmark/"
    if marker not in normalized:
        return ""
    rel = "ts_benchmark/" + normalized.split(marker, 1)[1]
    if rel.endswith(".py"):
        rel = rel[:-3]
    return rel.replace("/", ".")


def _load_module_from_source(source_path: str) -> Any:
    module_name = _module_name_from_source(source_path)
    if not module_name:
        return None
    return importlib.import_module(module_name)


def _case_ok(name: str, **payload: Any) -> Dict[str, Any]:
    return {"name": name, "status": "ok", **payload}


def _case_failed(name: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "name": name,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def _probe_conv1d_fft(torch: Any, module: Any, shape: MechanismTaskShape) -> List[Dict[str, Any]]:
    func = getattr(module, "conv1d_fft", None)
    if not callable(func):
        return []
    h = max(1, min(4, int(shape.channels or 1)))
    d = max(1, 8 if h > 1 else 1)
    cases: List[Dict[str, Any]] = []
    try:
        f = torch.zeros(shape.batch, shape.seq_len, h, d)
        g = torch.zeros(1, shape.seq_len, h, 1)
        out = func(f, g, dim=1)
        cases.append(_case_ok("operator.conv1d_fft.time_dim", output_shape=list(out.shape)))
    except Exception as exc:
        cases.append(_case_failed("operator.conv1d_fft.time_dim", exc))
    return cases


def _probe_exponential_smoothing(torch: Any, module: Any, shape: MechanismTaskShape) -> List[Dict[str, Any]]:
    cls = getattr(module, "ExponentialSmoothing", None)
    if not isinstance(cls, type):
        return []
    cases: List[Dict[str, Any]] = []
    specs = [
        ("ExponentialSmoothing.no_aux", max(1, min(4, shape.channels)), 8, False),
        ("ExponentialSmoothing.aux_single_channel", 1, 1, True),
        ("ExponentialSmoothing.aux_real_channels", max(1, shape.channels), 1, True),
    ]
    for name, heads, dim, aux in specs:
        try:
            layer = cls(dim, heads, aux=aux).eval()
            values = torch.zeros(shape.batch, shape.seq_len, heads, dim)
            aux_values = torch.zeros(shape.batch, shape.seq_len, heads, dim) if aux else None
            with torch.no_grad():
                out = layer(values, aux_values=aux_values)
            cases.append(_case_ok(name, input_shape=list(values.shape), output_shape=list(out.shape)))
        except Exception as exc:
            cases.append(_case_failed(name, exc))
    return cases


def _probe_full_model_train_step(torch: Any, model: Any, shape: MechanismTaskShape) -> Dict[str, Any]:
    try:
        train_target = model if hasattr(model, "train") else getattr(model, "model", model)
        try:
            device = next(train_target.parameters()).device
        except Exception:
            device = torch.device("cpu")
        if hasattr(train_target, "train"):
            train_target.train()
        x = torch.zeros(shape.batch, shape.seq_len, shape.channels, device=device)
        target_len = max(shape.seq_len // 2, 1) + shape.horizon
        target = torch.zeros(shape.batch, target_len, shape.channels, device=device)
        input_mark = torch.zeros(shape.batch, shape.seq_len, 4, device=device)
        target_mark = torch.zeros(shape.batch, target_len, 4, device=device)
        if hasattr(model, "_process"):
            out = model._process(x, target, input_mark, target_mark)
            normalized = normalize_tfb_process_output(out)
            if normalized.get("nested_output_dict"):
                return {
                    "name": "full_model.train_backward",
                    "status": "failed",
                    "error_type": "objective_boundary_mismatch",
                    "error_message": str(normalized.get("error_message") or "nested objective output dict"),
                    "raw_output_kind": normalized.get("raw_kind"),
                }
            output = normalized.get("prediction")
            additional_loss = normalized.get("additional_loss")
        else:
            output = model(x, input_mark, target, target_mark)
            additional_loss = None
        if isinstance(output, tuple):
            output = output[0]
        target_out = torch.zeros_like(output)
        loss = torch.nn.functional.mse_loss(output, target_out)
        additional_loss_value = None
        if additional_loss is not None:
            if not hasattr(additional_loss, "shape"):
                raise TypeError(f"additional_loss is not a Tensor: {type(additional_loss).__name__}")
            if additional_loss.ndim > 0 and additional_loss.numel() != 1:
                raise ValueError(f"additional_loss must be scalar-like, got shape={list(additional_loss.shape)}")
            additional_loss_value = float(additional_loss.detach().reshape(-1)[0])
            loss = loss + additional_loss.reshape(())
        loss.backward()
        return _case_ok(
            "full_model.train_backward",
            output_shape=list(output.shape),
            loss=float(loss.detach()),
            has_additional_loss=additional_loss is not None,
            additional_loss_value=additional_loss_value,
        )
    except Exception as exc:
        return _case_failed("full_model.train_backward", exc)
    finally:
        try:
            train_target = model if hasattr(model, "eval") else getattr(model, "model", model)
            if hasattr(train_target, "eval"):
                train_target.eval()
        except Exception:
            pass


def run_mechanism_probe(
    *,
    model: Any,
    variant_path: str,
    source_entrypoint: Dict[str, Any],
    task_shape: Dict[str, Any],
    variant_source: str = "",
) -> Dict[str, Any]:
    import torch

    shape = MechanismTaskShape(
        batch=int(task_shape.get("batch") or 2),
        seq_len=int(task_shape.get("seq_len") or 96),
        horizon=int(task_shape.get("horizon") or 96),
        channels=max(1, int(task_shape.get("channels") or 1)),
    )
    cases: List[Dict[str, Any]] = []

    source_files = [
        str(source_entrypoint.get("factory_model_source") or ""),
        str(source_entrypoint.get("inner_model_source") or ""),
    ]
    workspace_root = str(source_entrypoint.get("workspace_root") or "")
    if workspace_root:
        layer_path = Path(workspace_root) / "ts_benchmark" / "baselines" / "time_series_library" / "layers" / "ETSformer_EncDec.py"
        if layer_path.exists():
            source_files.append(str(layer_path))

    loaded_modules = []
    for source_path in source_files:
        module = _load_module_from_source(source_path)
        if module is not None and module not in loaded_modules:
            loaded_modules.append(module)

    for module in loaded_modules:
        cases.extend(_probe_conv1d_fft(torch, module, shape))
        cases.extend(_probe_exponential_smoothing(torch, module, shape))
    cases.append(_probe_full_model_train_step(torch, model, shape))

    failures = [case for case in cases if case.get("status") != "ok"]
    result = {
        "status": "failed" if failures else "ok",
        "stage": "mechanism_probe",
        "variant_path": variant_path,
        "task_shape": {
            "batch": shape.batch,
            "seq_len": shape.seq_len,
            "horizon": shape.horizon,
            "channels": shape.channels,
        },
        "case_count": len(cases),
        "cases": cases,
        "failures": failures,
    }
    if failures:
        first = failures[0]
        result["error_type"] = str(first.get("error_type") or "invalid_mechanism_contract")
        result["error_message"] = str(first.get("error_message") or "mechanism probe failed")
        result["failure_chain"] = build_failure_chain(
            failed_gate="mechanism_probe",
            primary_failure=(
                "objective_boundary_mismatch"
                if str(first.get("error_type") or "") == "objective_boundary_mismatch"
                else "invalid_tensor_contract"
            ),
            first_failure=str(first.get("name") or "mechanism_probe_failed"),
            steps=["workspace_bound", "mechanism_probe"],
            evidence=first,
        )
    return result
