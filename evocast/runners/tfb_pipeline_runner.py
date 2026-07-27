"""TFB pipeline runner.

Wraps ts_benchmark.pipeline.pipeline() with proper lifecycle management:
- Assembles data_config, model_config, evaluation_config
- Normalizes data_set_name (string -> list)
- Initializes and closes ParallelBackend
- Captures returned paths, stdout/stderr, exceptions
- Returns a structured run result

Owns the in-process TFB lifecycle used by EvoCast.
"""

import json
import gc
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ts_benchmark.common.constant import CONFIG_PATH, THIRD_PARTY_PATH
from ts_benchmark.pipeline import pipeline
from ts_benchmark.utils.parallel import ParallelBackend
from ts_benchmark.utils.parallel import _HAS_RAY
from evocast.policy.model_hparam_compat import apply_model_hparam_compatibility
from evocast.policy.training_policy import load_policy
from evocast.variant.import_isolation import collect_variant_paths, model_execution_import_context

logger = logging.getLogger(__name__)

_PIPELINE_WORKER_ENV = "EVOCAST_PIPELINE_WORKER"

def _default_evocast_num_workers() -> int:
    return 0 if sys.platform == "win32" else 1


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _runtime_channels(config_data: Dict[str, Any], hparams: Dict[str, Any]) -> int:
    data_config = config_data.get("data_config", {}) if isinstance(config_data, dict) else {}
    runtime_columns = _runtime_numeric_columns(config_data)
    if runtime_columns:
        return len(runtime_columns)
    columns = (
        data_config.get("target_columns")
        or data_config.get("target_cols")
        or data_config.get("value_columns")
        or data_config.get("columns")
        or []
    )
    if isinstance(columns, list) and columns:
        return max(1, len(columns))
    return max(1, _safe_int(hparams.get("c_out") or hparams.get("enc_in") or hparams.get("dec_in"), 1) or 1)


def _runtime_numeric_columns(config_data: Dict[str, Any]) -> List[str]:
    """Return the numeric wide-data columns consumed by the forecasting runtime."""
    data_config = config_data.get("data_config", {}) if isinstance(config_data, dict) else {}
    semantics = data_config.get("task_semantics", {}) if isinstance(data_config.get("task_semantics", {}), dict) else {}
    time_col = str(data_config.get("time_col") or semantics.get("time_col") or "date")
    candidates = [data_config.get("dataset_path")]
    candidates.extend(list(data_config.get("data_name_list") or []))
    for candidate in candidates:
        resolved = _resolve_dataset_path(str(candidate or ""))
        if not resolved:
            continue
        try:
            frame = pd.read_csv(resolved, nrows=256)
        except Exception:
            continue
        columns: List[str] = []
        for name in frame.columns:
            if name == time_col:
                continue
            series = frame[name]
            if pd.api.types.is_numeric_dtype(series):
                columns.append(str(name))
                continue
            normalized = series.astype(str).str.replace(",", "", regex=False).str.strip()
            normalized = normalized.mask(series.isna(), other=pd.NA).replace({"": pd.NA})
            converted = pd.to_numeric(normalized, errors="coerce")
            non_empty = normalized.notna()
            if non_empty.any() and converted[non_empty].notna().all():
                columns.append(str(name))
        if columns:
            return columns
    return []


def _complete_runtime_shape_hparams(config_data: Dict[str, Any], hparams: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(hparams or {})
    model_config = config_data.get("model_config", {}) if isinstance(config_data, dict) else {}
    recommended = model_config.get("recommend_model_hyper_params", {}) if isinstance(model_config, dict) else {}
    data_config = config_data.get("data_config", {}) if isinstance(config_data, dict) else {}
    semantics = data_config.get("task_semantics", {}) if isinstance(data_config.get("task_semantics", {}), dict) else {}
    strategy_args = (
        (config_data.get("evaluation_config", {}) or {}).get("strategy_args", {})
        if isinstance(config_data.get("evaluation_config", {}), dict)
        else {}
    )
    seq_len = _safe_int(
        result.get("seq_len")
        or result.get("input_chunk_length")
        or recommended.get("input_chunk_length")
        or data_config.get("seq_len")
        or semantics.get("input_chunk_length")
        or strategy_args.get("seq_len"),
        96,
    ) or 96
    horizon = _safe_int(
        result.get("horizon")
        or result.get("pred_len")
        or result.get("output_chunk_length")
        or recommended.get("output_chunk_length")
        or data_config.get("horizon")
        or strategy_args.get("horizon"),
        seq_len,
    ) or seq_len
    channel = _runtime_channels(config_data, result)
    result.setdefault("seq_len", seq_len)
    result.setdefault("input_chunk_length", seq_len)
    result.setdefault("horizon", horizon)
    result.setdefault("pred_len", horizon)
    result.setdefault("output_chunk_length", horizon)
    result.setdefault("label_len", max(1, seq_len // 2))
    result["enc_in"] = channel
    result["dec_in"] = channel
    result["c_out"] = channel
    return result


def _entry_source_default_hparams(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort adapter MODEL_HYPER_PARAMS defaults for workspace variants."""
    variant_path = str((entry or {}).get("variant_path") or "")
    model_name = str((entry or {}).get("model_name") or "")
    if not variant_path or not model_name:
        return {}
    try:
        from evocast.variant.workspace_loader import load_model_class

        model_cls = load_model_class(variant_path=variant_path, model_name=model_name)
        module = sys.modules.get(getattr(model_cls, "__module__", ""))
        defaults = getattr(module, "MODEL_HYPER_PARAMS", {}) if module is not None else {}
        return dict(defaults or {}) if isinstance(defaults, dict) else {}
    except Exception:
        return {}


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_dataset_path(dataset_path: str) -> Optional[str]:
    if not dataset_path:
        return None
    if os.path.isabs(dataset_path) and os.path.exists(dataset_path):
        return dataset_path
    root = _project_root()
    candidates = [
        os.path.join(root, dataset_path),
        os.path.join(root, "dataset", "forecasting", os.path.basename(dataset_path)),
        os.path.join(root, "dataset", os.path.basename(dataset_path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def resolve_dataset_path(dataset_path: str) -> Optional[str]:
    """Resolve a dataset path without copying or mutating files."""
    return _resolve_dataset_path(dataset_path)


def _normalize_dataset_metadata(config_data: Dict) -> Dict:
    """Normalize dataset metadata without silently copying data files."""
    config = dict(config_data)
    data_config = dict(config.get("data_config", {}) or {})
    dataset_path = data_config.get("dataset_path", "")
    resolved = _resolve_dataset_path(str(dataset_path))
    if not resolved:
        names = list(data_config.get("data_name_list", []) or [])
        if names:
            resolved = _resolve_dataset_path(str(names[0]))
    if not resolved:
        config["data_config"] = data_config
        return config

    basename = os.path.basename(resolved)
    data_names = list(data_config.get("data_name_list", []) or [])
    if basename not in data_names:
        data_names = [basename]
    data_config["data_name_list"] = data_names
    data_config["dataset_path"] = resolved
    config["data_config"] = data_config
    return config


def preflight_forecasting_datasets(config_data: Dict, project_root: Optional[str] = None) -> List[str]:
    """Return missing forecasting dataset names/paths for this config.

    The large_forecast loader resolves data_name_list under dataset/forecasting/.
    This check fails before a baseline batch starts, so environmental issues do not
    become many low-information trial failures.
    """
    root = os.path.abspath(project_root or _project_root())
    data_config = config_data.get("data_config", {}) or {}
    data_set_name = data_config.get("data_set_name", [])
    if isinstance(data_set_name, str):
        data_sets = [data_set_name]
    else:
        data_sets = list(data_set_name or [])

    names = list(data_config.get("data_name_list", []) or [])
    dataset_path = data_config.get("dataset_path")
    missing: List[str] = []

    if dataset_path and not _resolve_dataset_path(str(dataset_path)):
        missing.append(str(dataset_path))

    if "large_forecast" not in data_sets:
        return missing

    for name in names:
        text = str(name)
        candidates = []
        if os.path.isabs(text) or os.path.dirname(text):
            candidates.append(text)
        candidates.extend([
            os.path.join(root, "dataset", "forecasting", os.path.basename(text)),
            os.path.join(root, "dataset", text),
            os.path.join(root, text),
        ])
        if not any(os.path.exists(candidate) for candidate in candidates):
            missing.append(text)

    return sorted(set(missing))


def _init_worker(env: Dict) -> None:
    """Initialize a TFB worker process."""
    sys.path.insert(0, THIRD_PARTY_PATH)
    import torch
    torch.set_num_threads(1)


def normalize_data_set_name(data_config: Dict) -> Dict:
    """Ensure data_set_name is a list.

    Some config JSON files store data_set_name as a string (e.g., "large_forecast"),
    while the CLI passes it as a list. The pipeline already handles this in the
    pipeline() function, but we normalize it here for consistency.

    The pipeline code already does:
        if isinstance(dataset_name_list, str):
            dataset_name_list = [dataset_name_list]

    We mirror that logic so callers get a normalized config.
    """
    config = dict(data_config)
    ds_name = config.get("data_set_name", ["small_forecast"])
    if isinstance(ds_name, str):
        config["data_set_name"] = [ds_name]
    elif ds_name is None:
        config["data_set_name"] = ["small_forecast"]
    return config


def load_config_json(config_path: str) -> Dict:
    """Load a TFB config JSON file.

    The config_path can be:
    - An absolute path
    - A relative path from the repo root
    - A filename in config/ (e.g., "fixed_forecast_config_daily.json")
    """
    if os.path.isabs(config_path) and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return _normalize_dataset_metadata(normalize_strategy_name(json.load(f)))

    # Try relative to CONFIG_PATH
    config_file = os.path.join(CONFIG_PATH, config_path)
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            return _normalize_dataset_metadata(normalize_strategy_name(json.load(f)))

    # Try as-is
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return _normalize_dataset_metadata(normalize_strategy_name(json.load(f)))

    raise FileNotFoundError(f"Config file not found: {config_path}")


def normalize_strategy_name(config_data: Dict) -> Dict:
    """Normalize display labels into TFB strategy registry keys."""
    strategy_args = (
        config_data.get("evaluation_config", {})
        .get("strategy_args", {})
    )
    value = strategy_args.get("strategy_name")
    valid = {"rolling_forecast", "fixed_forecast"}
    if isinstance(value, str) and value not in valid:
        first_token = value.split(" ", 1)[0].strip()
        if first_token in valid:
            strategy_args["strategy_name"] = first_token
    return config_data


def build_run_configs(
    config_data: Dict,
    model_entries: List[Dict],
    save_path: str = "EvoCast_test",
    seed: Optional[int] = None,
    override_eval_args: Optional[Dict] = None,
) -> tuple:
    """Build data_config, model_config, evaluation_config from a TFB config JSON.

    Args:
        config_data: Loaded TFB config JSON.
        model_entries: List of model dicts with "model_name", optional "adapter",
                       optional "model_hyper_params".
        save_path: Directory under result/ for outputs.
        seed: Override random seed.
        override_eval_args: Override evaluation strategy args.

    Returns:
        (data_config, model_config, evaluation_config)
    """
    data_config = normalize_data_set_name(deepcopy(config_data["data_config"]))

    safe_model_entries = []
    for entry in model_entries:
        safe_entry = deepcopy(entry or {})
        source_defaults = _entry_source_default_hparams(safe_entry)
        merged_hparams = dict(source_defaults)
        merged_hparams.update(dict(safe_entry.get("model_hyper_params") or {}))
        hparams = _complete_runtime_shape_hparams(config_data, merged_hparams)
        model_name = str(
            safe_entry.get("model_key")
            or safe_entry.get("source_model_key")
            or safe_entry.get("base_model_key")
            or safe_entry.get("model_name")
            or ""
        )
        hparams, compatibility_notes = apply_model_hparam_compatibility(
            model_name.rsplit(".", 1)[-1],
            hparams,
            config_data,
        )
        default_num_workers = _default_evocast_num_workers()
        if hparams.get("num_workers") != default_num_workers:
            hparams["num_workers"] = default_num_workers
        if compatibility_notes:
            safe_entry["model_hparam_compatibility_notes"] = compatibility_notes
        safe_entry["model_hyper_params"] = hparams
        safe_model_entries.append(safe_entry)

    model_config = deepcopy(config_data.get("model_config", {}) or {})
    model_config = {
        "models": safe_model_entries,
        "recommend_model_hyper_params": deepcopy(model_config.get(
            "recommend_model_hyper_params", {}
        )),
    }

    evaluation_config = deepcopy(config_data["evaluation_config"])
    evaluation_config["save_path"] = save_path

    strategy_args = dict(evaluation_config.get("strategy_args") or {})
    if seed is not None:
        strategy_args["seed"] = seed
    if override_eval_args:
        strategy_args.update(override_eval_args)
    evaluation_config["strategy_args"] = strategy_args

    # Ensure deterministic is set
    if "deterministic" not in strategy_args:
        strategy_args["deterministic"] = "efficient"

    return data_config, model_config, evaluation_config


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _kill_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(process.pid, 15)
    except Exception:
        try:
            process.terminate()
        except Exception:
            pass
    try:
        process.wait(timeout=5)
        return
    except Exception:
        pass
    try:
        os.killpg(process.pid, 9)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _pipeline_timeout_result(
    *,
    data_config: Dict,
    model_config: Dict,
    evaluation_config: Dict,
    timeout: float,
    elapsed: float,
) -> Dict[str, Any]:
    message = (
        f"TFB pipeline hard timeout: single pipeline run exceeded "
        f"{float(timeout):.1f}s"
    )
    return {
        "success": False,
        "log_paths": [],
        "error": TimeoutError(message),
        "error_type": "timeout",
        "error_traceback": message,
        "elapsed_seconds": float(elapsed),
        "timeout_seconds": float(timeout),
        "timeout_scope": "single_pipeline_run",
        "data_config": data_config,
        "model_config": model_config,
        "evaluation_config": evaluation_config,
        "started_at": datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat(),
    }


def _restore_error_object(result: Dict[str, Any]) -> Dict[str, Any]:
    if bool(result.get("success")):
        return result
    error = result.get("error")
    if isinstance(error, BaseException):
        return result
    message = str(error or result.get("error_traceback") or result.get("error_type") or "pipeline failed")
    if str(result.get("error_type") or "").lower() == "timeout" or "timeout" in message.lower():
        result["error"] = TimeoutError(message)
    else:
        result["error"] = RuntimeError(message)
    return result


def _run_pipeline_hard_timeout(
    *,
    data_config: Dict,
    model_config: Dict,
    evaluation_config: Dict,
    backend: str,
    n_workers: Optional[int],
    n_cpus: Optional[int],
    gpu_devices: Optional[List[int]],
    timeout: float,
    max_tasks_per_child: int,
    source_checkout: Optional[str],
) -> Dict[str, Any]:
    timeout_seconds = float(timeout or 0)
    if timeout_seconds <= 0:
        return _run_pipeline_in_process(
            data_config=data_config,
            model_config=model_config,
            evaluation_config=evaluation_config,
            backend=backend,
            n_workers=n_workers,
            n_cpus=n_cpus,
            gpu_devices=gpu_devices,
            timeout=timeout,
            max_tasks_per_child=max_tasks_per_child,
            source_checkout=source_checkout,
        )

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="evocast_pipeline_") as tmp:
        input_path = os.path.join(tmp, "input.json")
        output_path = os.path.join(tmp, "output.json")
        payload = {
            "data_config": data_config,
            "model_config": model_config,
            "evaluation_config": evaluation_config,
            "backend": backend,
            "n_workers": n_workers,
            "n_cpus": n_cpus,
            "gpu_devices": gpu_devices,
            "timeout": timeout,
            "max_tasks_per_child": max_tasks_per_child,
            "source_checkout": source_checkout,
        }
        with open(input_path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, ensure_ascii=False, default=str)

        env = dict(os.environ)
        env[_PIPELINE_WORKER_ENV] = "1"
        popen_kwargs: Dict[str, Any] = {
            "cwd": _project_root(),
            "env": env,
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        else:
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "evocast.runners.tfb_pipeline_runner",
                "--pipeline-worker",
                input_path,
                output_path,
            ],
            **popen_kwargs,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            elapsed = time.time() - started
            _kill_process_tree(process)
            return _pipeline_timeout_result(
                data_config=data_config,
                model_config=model_config,
                evaluation_config=evaluation_config,
                timeout=timeout_seconds,
                elapsed=elapsed,
            )

        elapsed = time.time() - started
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as handle:
                result = json.load(handle)
            result.setdefault("elapsed_seconds", elapsed)
            result.setdefault("timeout_seconds", timeout_seconds)
            result.setdefault("timeout_scope", "single_pipeline_run")
            return _restore_error_object(result)
        return {
            "success": False,
            "log_paths": [],
            "error": RuntimeError(f"pipeline worker exited with code {returncode} without writing output"),
            "error_type": "pipeline_worker_failed",
            "error_traceback": f"pipeline worker exited with code {returncode} without writing output",
            "elapsed_seconds": elapsed,
            "timeout_seconds": timeout_seconds,
            "timeout_scope": "single_pipeline_run",
            "data_config": data_config,
            "model_config": model_config,
            "evaluation_config": evaluation_config,
            "started_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
        }


def _run_pipeline_in_process(
    *,
    data_config: Dict,
    model_config: Dict,
    evaluation_config: Dict,
    backend: str = "auto",
    n_workers: Optional[int] = None,
    n_cpus: Optional[int] = None,
    gpu_devices: Optional[List[int]] = None,
    timeout: float = 600,
    max_tasks_per_child: int = 100,
    source_checkout: Optional[str] = None,
) -> Dict:
    """Run the actual TFB pipeline in the current process."""
    import torch
    torch.set_num_threads(3)

    # Auto-detect backend
    if backend == "auto":
        # Windows uses spawn-based multiprocessing, which is fragile for the
        # long in-process agent loop. Keep experiments single-process by default.
        if sys.platform == "win32":
            backend = "sequential"
        elif _HAS_RAY:
            backend = "ray"
        else:
            backend = "sequential"

    # Auto-detect GPU devices
    if gpu_devices is None:
        gpu_devices = [0] if torch.cuda.is_available() else []

    if n_workers is None:
        n_workers = os.cpu_count() or 4
    if n_cpus is None:
        n_cpus = os.cpu_count() or 4

    result: Dict[str, Any] = {
        "success": False,
        "log_paths": [],
        "error": None,
        "error_type": None,
        "error_traceback": None,
        "elapsed_seconds": 0.0,
        "timeout_seconds": float(timeout or 0),
        "timeout_scope": "single_pipeline_run",
        "data_config": data_config,
        "model_config": model_config,
        "evaluation_config": evaluation_config,
        "started_at": datetime.now().isoformat(),
    }

    ParallelBackend().init(
        backend=backend,
        n_workers=n_workers,
        n_cpus=n_cpus,
        gpu_devices=gpu_devices,
        default_timeout=timeout,
        max_tasks_per_child=max_tasks_per_child,
        worker_initializers=[_init_worker],
    )

    t_start = time.time()
    try:
        with model_execution_import_context(collect_variant_paths(model_config), source_checkout=source_checkout):
            log_paths = pipeline(data_config, model_config, evaluation_config)
        result["log_paths"] = list(log_paths) if log_paths else []
        result["success"] = True
    except Exception as exc:
        result["error"] = exc
        result["error_type"] = type(exc).__name__
        result["error_traceback"] = traceback.format_exc()
        logger.error(f"Pipeline failed: {exc}")
    finally:
        try:
            ParallelBackend().close(force=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result["elapsed_seconds"] = time.time() - t_start
    result["completed_at"] = datetime.now().isoformat()

    return result


def run_pipeline(
    data_config: Dict,
    model_config: Dict,
    evaluation_config: Dict,
    backend: str = "auto",
    n_workers: Optional[int] = None,
    n_cpus: Optional[int] = None,
    gpu_devices: Optional[List[int]] = None,
    timeout: float = 600,
    max_tasks_per_child: int = 100,
    source_checkout: Optional[str] = None,
) -> Dict:
    """Run the TFB benchmark pipeline and return a structured result.

    This is the primary entry point for running TFB experiments.

    Args:
        data_config: Normalized data configuration.
        model_config: Model configuration with "models" list.
        evaluation_config: Evaluation configuration.
        backend: "sequential", "ray", or "auto" (auto-detects best available).
        n_workers: Number of workers.
        n_cpus: Number of CPUs.
        gpu_devices: GPU device indices. Auto-detected if None.
        timeout: Timeout per task in seconds.
        max_tasks_per_child: Max tasks per worker child process.

    Returns:
        Dict with:
          - success: bool
          - log_paths: List[str] from pipeline() or empty
          - stdout_captured: str
          - stderr_captured: str
          - error: Exception or None
          - error_traceback: str or None
          - elapsed_seconds: float
          - data_config: normalized config used
          - model_config: config used
          - evaluation_config: config used
    """
    if os.environ.get(_PIPELINE_WORKER_ENV) == "1":
        return _run_pipeline_in_process(
            data_config=data_config,
            model_config=model_config,
            evaluation_config=evaluation_config,
            backend=backend,
            n_workers=n_workers,
            n_cpus=n_cpus,
            gpu_devices=gpu_devices,
            timeout=timeout,
            max_tasks_per_child=max_tasks_per_child,
            source_checkout=source_checkout,
        )
    return _run_pipeline_hard_timeout(
        data_config=data_config,
        model_config=model_config,
        evaluation_config=evaluation_config,
        backend=backend,
        n_workers=n_workers,
        n_cpus=n_cpus,
        gpu_devices=gpu_devices,
        timeout=timeout,
        max_tasks_per_child=max_tasks_per_child,
        source_checkout=source_checkout,
    )


def run_single_model(
    config_path: str,
    model_name: str,
    adapter: Optional[str] = None,
    model_hyper_params: Optional[Dict] = None,
    save_path: str = "EvoCast_test",
    seed: Optional[int] = None,
    override_eval_args: Optional[Dict] = None,
    backend: str = "sequential",
    n_workers: Optional[int] = None,
    timeout: float = 600,
) -> Dict:
    """Convenience function: run one model from a config file.

    Args:
        config_path: Path to TFB config JSON.
        model_name: Model import path (e.g., "time_series_library.DLinear").
        adapter: Optional adapter name (e.g., "transformer_adapter").
        model_hyper_params: Optional model hyperparameter overrides.
        save_path: Output subdirectory under result/.
        seed: Random seed.
        override_eval_args: Override evaluation args.
        backend: "sequential" or "ray".
        n_workers: Worker count.
        timeout: Task timeout.

    Returns:
        Structured run result dict.
    """
    config_data = load_config_json(config_path)

    model_entry = {"model_name": model_name}
    if adapter:
        model_entry["adapter"] = adapter
    if model_hyper_params:
        model_entry["model_hyper_params"] = model_hyper_params
    else:
        model_entry["model_hyper_params"] = {}

    data_config, model_config, evaluation_config = build_run_configs(
        config_data,
        [model_entry],
        save_path=save_path,
        seed=seed,
        override_eval_args=override_eval_args,
    )

    return run_pipeline(
        data_config,
        model_config,
        evaluation_config,
        backend=backend,
        n_workers=n_workers,
        timeout=timeout,
    )


# ─── Smoke run mode ─────────────────────────────────────────────────────

def run_smoke(
    config_path: str,
    model_name: str,
    adapter: str = "transformer_adapter",
    model_hyper_params: Optional[Dict] = None,
    save_path: str = "EvoCast_smoke",
    **kwargs,
) -> Dict:
    """Run a minimal smoke test with the build-mode fast training schedule."""
    hp = dict(model_hyper_params or {})
    hp.update(load_policy("smoke_test"))

    return run_single_model(
        config_path=config_path,
        model_name=model_name,
        adapter=adapter,
        model_hyper_params=hp,
        save_path=save_path,
        **kwargs,
    )


def _pipeline_worker_main(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result = _run_pipeline_in_process(
        data_config=dict(payload.get("data_config") or {}),
        model_config=dict(payload.get("model_config") or {}),
        evaluation_config=dict(payload.get("evaluation_config") or {}),
        backend=str(payload.get("backend") or "auto"),
        n_workers=payload.get("n_workers"),
        n_cpus=payload.get("n_cpus"),
        gpu_devices=payload.get("gpu_devices"),
        timeout=float(payload.get("timeout") or 0),
        max_tasks_per_child=int(payload.get("max_tasks_per_child") or 100),
        source_checkout=payload.get("source_checkout"),
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(result), handle, ensure_ascii=False, default=str)


def main(argv: Optional[List[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 3 and args[0] == "--pipeline-worker":
        _pipeline_worker_main(args[1], args[2])
        return
    raise SystemExit(
        "Usage: python -m evocast.runners.tfb_pipeline_runner "
        "--pipeline-worker <input.json> <output.json>"
    )


if __name__ == "__main__":
    main()
