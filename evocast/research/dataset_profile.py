from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy, norm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

from evocast.domain.atomic_io import atomic_write_json
from evocast.domain.i18n import localize_code
from evocast.domain.knowledge_paths import (
    dataset_record_dir,
    dataset_registry_path,
    dataset_task_binding_path,
    dataset_view_dir,
    task_knowledge_dir,
)
from evocast.harness.api_client import ProviderAPIError, create_task_client
from evocast.runners.tfb_pipeline_runner import resolve_dataset_path
from evocast.state.cost_ledger import tracked_stage
from evocast.state.domain_store import load_task_config

SCHEMA_VERSION = "dataset_profile_v1"
CHARACTERISTICS_SCHEMA_VERSION = "dataset_characteristics_v1"
CHARACTERISTICS_ANALYSIS_VERSION = "python_characteristics_v1"
DATASET_DIAGNOSIS_MODES = {"required", "reuse", "skip"}
CHARACTERISTIC_KEYS = [
    "Correlation",
    "Transition",
    "Shifting",
    "Seasonality",
    "Trend",
    "Stationarity",
    "Short_term_jsd",
    "Long_term_jsd",
]
DEFAULT_PERIODS = [4, 7, 12, 24, 48, 52, 96, 144, 168, 336, 672, 1008, 1440]
MAX_ANALYSIS_POINTS = 4096
BUILD_MODE_MAX_ANALYSIS_POINTS = 1024
BUILD_MODE_MAX_STL_PERIODS = 3
MIN_SERIES_LENGTH = 32
ProgressCallback = Callable[[str], None]


def _is_english(language: str) -> bool:
    return str(language or "").strip().lower() in {"en", "english"}


def _emit_progress(
    progress: ProgressCallback | None,
    message: str,
    *,
    language: str = "zh",
    english_message: str | None = None,
) -> None:
    if progress is not None:
        progress(str(english_message if _is_english(language) and english_message is not None else message))


def dataset_profile_path(base_dir: str, task_id: str) -> Path:
    binding = load_dataset_task_binding(base_dir, task_id)
    bound_path = str(binding.get("dataset_profile_path") or "").strip()
    if bound_path:
        return Path(bound_path)
    task_config, data_config, semantics = _task_dataset_inputs(base_dir, task_id)
    dataset_path = _resolve_dataset_path(task_config=task_config, data_config=data_config)
    dataset_id = _dataset_id(dataset_path, task_config=task_config, data_config=data_config, task_id=task_id)
    view_id = _dataset_view_id(
        dataset_path=dataset_path,
        task_config=task_config,
        data_config=data_config,
        semantics=semantics,
    )
    return dataset_view_dir(base_dir, dataset_id, view_id) / "dataset_profile.json"


def _json_digest(payload: Dict[str, Any]) -> str:
    import hashlib

    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_slug(value: Any, *, default: str = "dataset") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    return text or default


def _dataset_source_identity(path: Path) -> Dict[str, Any]:
    """Fast source identity used to invalidate a shared computation cache.

    The analysis cache must be inexpensive to check; hashing a large CSV before
    every task would itself duplicate a substantial part of the diagnosis I/O.
    File size plus nanosecond mtime is therefore the live invalidation boundary.
    """
    stat = path.stat()
    return {
        "resolved_path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _task_dataset_inputs(base_dir: str, task_id: str) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    knowledge_dir = task_knowledge_dir(base_dir, task_id)
    task_config = load_task_config(base_dir, task_id)
    compiled_config = _read_json(knowledge_dir / "compiled_config.json")
    data_config = dict(compiled_config.get("data_config") or {})
    semantics = dict(task_config.get("task_semantics") or data_config.get("task_semantics") or {})
    return task_config, data_config, semantics


def _dataset_id(
    dataset_path: Path | None,
    *,
    task_config: Dict[str, Any],
    data_config: Dict[str, Any],
    task_id: str,
) -> str:
    if dataset_path is None:
        return f"unresolved_dataset__{_safe_slug(task_id, default='task')}"
    dataset_name = (
        str(task_config.get("data_set_name") or data_config.get("data_set_name") or "").strip()
        or dataset_path.stem
    )
    digest = _json_digest(_dataset_source_identity(dataset_path))[:12]
    return f"{_safe_slug(dataset_name)}__{digest}"


def _dataset_view_descriptor(
    *,
    dataset_path: Path | None,
    task_config: Dict[str, Any],
    data_config: Dict[str, Any],
    semantics: Dict[str, Any],
) -> Dict[str, Any]:
    requested_targets = _target_columns_for_profile(
        task_config=task_config,
        data_config=data_config,
        semantics=semantics,
    )
    return {
        "schema_version": CHARACTERISTICS_SCHEMA_VERSION,
        "analysis_version": CHARACTERISTICS_ANALYSIS_VERSION,
        "task_mode": str(semantics.get("task_mode") or task_config.get("task_mode") or "").upper(),
        "input_variable_topology": str(semantics.get("input_variable_topology") or ""),
        "prediction_target_selection": str(semantics.get("prediction_target_selection") or ""),
        "requested_target_columns": [str(item).strip().lower() for item in requested_targets],
        "seq_len": task_config.get("seq_len") or data_config.get("seq_len"),
        "horizon": task_config.get("horizon") or data_config.get("horizon"),
        "frequency": str(semantics.get("frequency") or data_config.get("freq") or ""),
        "max_points": _analysis_points_limit(task_config),
        "max_stl_periods": _stl_period_limit(task_config),
        "normalization_version": "numeric_frame_long_wide_v1",
        "source_available": dataset_path is not None,
    }


def _dataset_view_id(
    *,
    dataset_path: Path | None,
    task_config: Dict[str, Any],
    data_config: Dict[str, Any],
    semantics: Dict[str, Any],
) -> str:
    descriptor = _dataset_view_descriptor(
        dataset_path=dataset_path,
        task_config=task_config,
        data_config=data_config,
        semantics=semantics,
    )
    task_mode = _safe_slug(descriptor.get("task_mode") or "task", default="task").lower()
    targets = descriptor.get("requested_target_columns") or []
    target_slug = "all_targets" if not targets else _safe_slug("_".join(str(item) for item in targets), default="targets").lower()
    seq = _safe_slug(descriptor.get("seq_len") or "seq", default="seq")
    horizon = _safe_slug(descriptor.get("horizon") or "pred", default="pred")
    freq = _safe_slug(descriptor.get("frequency") or "freq", default="freq").lower()
    return f"{task_mode}_{target_slug}__seq{seq}_pred{horizon}__{freq}__{_json_digest(descriptor)[:10]}"


def _analysis_view_descriptor(
    *,
    dataset_path: Path,
    task_config: Dict[str, Any],
    data_config: Dict[str, Any],
    semantics: Dict[str, Any],
) -> Dict[str, Any]:
    descriptor = _dataset_view_descriptor(
        dataset_path=dataset_path,
        task_config=task_config,
        data_config=data_config,
        semantics=semantics,
    )
    descriptor["source"] = _dataset_source_identity(dataset_path)
    return descriptor


def _characteristics_cache_path(base_dir: str, dataset_id: str, view_id: str) -> Path:
    return dataset_view_dir(base_dir, dataset_id, view_id) / "characteristics.json"


def _load_characteristics_cache(path: Path, descriptor: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("schema_version") != CHARACTERISTICS_SCHEMA_VERSION:
        return {}
    if payload.get("analysis_view") != descriptor:
        return {}
    if not isinstance(payload.get("raw_characteristics"), dict) or not isinstance(payload.get("basic_stats"), dict):
        return {}
    return payload


def _write_characteristics_cache(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_json(path, payload, ensure_ascii=False, default=str)


def load_dataset_task_binding(base_dir: str, task_id: str) -> Dict[str, Any]:
    path = dataset_task_binding_path(base_dir, task_id)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_dataset_task_binding(
    *,
    base_dir: str,
    task_id: str,
    dataset_id: str,
    view_id: str,
    profile_path: Path,
    dataset_path: Path | None,
    view_descriptor: Dict[str, Any],
) -> Dict[str, Any]:
    binding = {
        "schema_version": "dataset_task_binding_v1",
        "task_id": task_id,
        "dataset_id": dataset_id,
        "view_id": view_id,
        "dataset_profile_path": str(profile_path),
        "dataset_path": str(dataset_path) if dataset_path is not None else "",
        "view_descriptor": view_descriptor,
        "updated_at": datetime.now().isoformat(),
    }
    path = dataset_task_binding_path(base_dir, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(binding, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return binding


def _write_dataset_manifest(
    *,
    base_dir: str,
    dataset_id: str,
    dataset_path: Path | None,
    task_config: Dict[str, Any],
    data_config: Dict[str, Any],
) -> None:
    record_dir = dataset_record_dir(base_dir, dataset_id)
    record_dir.mkdir(parents=True, exist_ok=True)
    source_identity = _dataset_source_identity(dataset_path) if dataset_path is not None else {}
    dataset_name = str(task_config.get("data_set_name") or data_config.get("data_set_name") or "").strip()
    manifest = {
        "schema_version": "dataset_manifest_v1",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name or (dataset_path.stem if dataset_path is not None else ""),
        "dataset_path": str(dataset_path) if dataset_path is not None else "",
        "source_identity": source_identity,
        "updated_at": datetime.now().isoformat(),
    }
    (record_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (record_dir / "source_identity.json").write_text(
        json.dumps(source_identity, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _update_dataset_registry(
    *,
    base_dir: str,
    dataset_id: str,
    view_id: str,
    dataset_path: Path | None,
    profile_path: Path,
) -> None:
    path = dataset_registry_path(base_dir)
    try:
        registry = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        registry = {}
    if not isinstance(registry, dict):
        registry = {}
    datasets = dict(registry.get("datasets") or {})
    record = dict(datasets.get(dataset_id) or {})
    views = dict(record.get("views") or {})
    views[view_id] = {
        "dataset_profile_path": str(profile_path),
        "updated_at": datetime.now().isoformat(),
    }
    record.update(
        {
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_path) if dataset_path is not None else "",
            "views": views,
        }
    )
    datasets[dataset_id] = record
    registry = {
        "schema_version": "dataset_knowledge_registry_v1",
        "updated_at": datetime.now().isoformat(),
        "datasets": datasets,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_dataset_profile(base_dir: str, task_id: str) -> Dict[str, Any]:
    path = dataset_profile_path(base_dir, task_id)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def has_dataset_profile(base_dir: str, task_id: str) -> bool:
    payload = load_dataset_profile(base_dir, task_id)
    return bool(payload.get("schema_version") == SCHEMA_VERSION and payload.get("status") == "ok")


def has_dataset_profile_or_intentional_skip(base_dir: str, task_id: str) -> bool:
    payload = load_dataset_profile(base_dir, task_id)
    return bool(
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("status") in {"ok", "skipped"}
        and (payload.get("status") != "skipped" or payload.get("skip_reason"))
    )


def dataset_diagnosis_mode(base_dir: str, task_id: str) -> str:
    task_config = load_task_config(base_dir, task_id)
    mode = str(task_config.get("dataset_diagnosis_mode") or "required").strip().lower()
    return mode if mode in DATASET_DIAGNOSIS_MODES else "required"


def write_skipped_dataset_profile(
    *,
    task_id: str,
    base_dir: str,
    reason: str = "user_requested_research_fast_path",
    language: str = "zh",
    progress: ProgressCallback | None = None,
) -> Dict[str, Any]:
    knowledge_dir = task_knowledge_dir(base_dir, task_id)
    task_config = load_task_config(base_dir, task_id)
    compiled_config = _read_json(knowledge_dir / "compiled_config.json")
    data_config = dict(compiled_config.get("data_config") or {})
    feature_dict = dict(task_config.get("feature_dict") or data_config.get("feature_dict") or {})
    semantics = dict(task_config.get("task_semantics") or data_config.get("task_semantics") or {})
    basic_facts = _basic_facts(task_config=task_config, data_config=data_config, semantics=semantics, feature_dict=feature_dict)
    dataset_path = _resolve_dataset_path(task_config=task_config, data_config=data_config)
    dataset_id = _dataset_id(dataset_path, task_config=task_config, data_config=data_config, task_id=task_id)
    view_id = _dataset_view_id(
        dataset_path=dataset_path,
        task_config=task_config,
        data_config=data_config,
        semantics=semantics,
    )
    view_descriptor = _dataset_view_descriptor(
        dataset_path=dataset_path,
        task_config=task_config,
        data_config=data_config,
        semantics=semantics,
    )
    path = dataset_view_dir(base_dir, dataset_id, view_id) / "dataset_profile.json"
    if dataset_path:
        basic_facts["dataset_path"] = str(dataset_path)
    skipped_message = (
        "Dataset diagnosis was intentionally skipped for research-loop fast path."
        if _is_english(language)
        else "数据集诊断已按配置跳过，用于快速进入 research 流程。"
    )
    skipped_limitation = (
        "Dataset diagnosis was skipped by configuration; downstream research should not treat dataset-level evidence as observed."
        if _is_english(language)
        else "数据集诊断已按配置跳过；后续研究不能把数据集级证据当作已观测事实。"
    )
    diagnostics = [
        {
            "severity": "info",
            "code": "DATASET_DIAGNOSIS_SKIPPED",
            "message": skipped_message,
        }
    ]
    narrative_summary = (
        "Dataset diagnosis was intentionally skipped; no dataset-derived claims are available."
        if _is_english(language)
        else "数据集诊断已按用户要求跳过；本次没有数据集派生证据。"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "created_at": datetime.now().isoformat(),
        "status": "skipped",
        "skip_reason": str(reason or "user_requested_research_fast_path"),
        "characteristics_engine": "skipped",
        "basic_facts": basic_facts,
        "raw_characteristics": {key: None for key in CHARACTERISTIC_KEYS},
        "derived_claims": [],
        "research_implications": [],
        "llm_narrative": {
            "status": "skipped",
            "dataset_summary": narrative_summary,
            "research_interpretation": narrative_summary,
            "limitations": [
                skipped_limitation
            ],
        },
        "diagnostics": diagnostics,
        "source_artifacts": [{"kind": "dataset_csv", "path": str(dataset_path)}] if dataset_path else [],
        "dataset_knowledge": {
            "dataset_id": dataset_id,
            "view_id": view_id,
            "view_descriptor": view_descriptor,
            "dataset_profile_path": str(path),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _emit_progress(
        progress,
        f"写入跳过占位 dataset_knowledge profile: {path}",
        language=language,
        english_message=f"Writing skipped dataset-knowledge profile: {path}",
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_dataset_manifest(
        base_dir=base_dir,
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        task_config=task_config,
        data_config=data_config,
    )
    _write_dataset_task_binding(
        base_dir=base_dir,
        task_id=task_id,
        dataset_id=dataset_id,
        view_id=view_id,
        profile_path=path,
        dataset_path=dataset_path,
        view_descriptor=view_descriptor,
    )
    _update_dataset_registry(
        base_dir=base_dir,
        dataset_id=dataset_id,
        view_id=view_id,
        dataset_path=dataset_path,
        profile_path=path,
    )
    return payload


@tracked_stage(
    "dataset_diagnosis",
    lambda *args, **kwargs: (str(kwargs["base_dir"]), str(kwargs["task_id"]), "", ""),
)
def generate_dataset_profile(
    *,
    task_id: str,
    base_dir: str,
    client: Any | None = None,
    force_refresh: bool = False,
    progress: ProgressCallback | None = None,
    language: str = "zh",
) -> Dict[str, Any]:
    started_at = time.monotonic()
    existing = load_dataset_profile(base_dir, task_id)
    if existing.get("schema_version") == SCHEMA_VERSION and existing.get("status") == "ok" and not force_refresh:
        task_config, data_config, semantics = _task_dataset_inputs(base_dir, task_id)
        dataset_path = _resolve_dataset_path(task_config=task_config, data_config=data_config)
        dataset_id = str((existing.get("dataset_knowledge") or {}).get("dataset_id") or "")
        view_id = str((existing.get("dataset_knowledge") or {}).get("view_id") or "")
        if not dataset_id:
            dataset_id = _dataset_id(dataset_path, task_config=task_config, data_config=data_config, task_id=task_id)
        if not view_id:
            view_id = _dataset_view_id(
                dataset_path=dataset_path,
                task_config=task_config,
                data_config=data_config,
                semantics=semantics,
            )
        view_descriptor = _dataset_view_descriptor(
            dataset_path=dataset_path,
            task_config=task_config,
            data_config=data_config,
            semantics=semantics,
        )
        profile_path = dataset_profile_path(base_dir, task_id)
        _write_dataset_task_binding(
            base_dir=base_dir,
            task_id=task_id,
            dataset_id=dataset_id,
            view_id=view_id,
            profile_path=profile_path,
            dataset_path=dataset_path,
            view_descriptor=view_descriptor,
        )
        _write_dataset_manifest(
            base_dir=base_dir,
            dataset_id=dataset_id,
            dataset_path=dataset_path,
            task_config=task_config,
            data_config=data_config,
        )
        _update_dataset_registry(
            base_dir=base_dir,
            dataset_id=dataset_id,
            view_id=view_id,
            dataset_path=dataset_path,
            profile_path=profile_path,
        )
        reused = dict(existing)
        reused_cache = dict(reused.get("characteristics_cache") or {})
        if reused_cache:
            reused_cache["hit"] = True
            reused["characteristics_cache"] = reused_cache
        _emit_progress(progress, f"复用已存在的 dataset profile: {profile_path}", language=language, english_message=f"Reusing existing dataset profile: {profile_path}")
        return reused

    _emit_progress(progress, f"开始数据集诊断 task_id={task_id}", language=language, english_message=f"Starting dataset diagnosis task_id={task_id}")
    knowledge_dir = task_knowledge_dir(base_dir, task_id)
    _emit_progress(progress, "读取 canonical 任务配置", language=language, english_message="Reading canonical task config")
    task_config = load_task_config(base_dir, task_id)
    _emit_progress(progress, f"读取编译配置: {knowledge_dir / 'compiled_config.json'}", language=language, english_message=f"Reading compiled config: {knowledge_dir / 'compiled_config.json'}")
    compiled_config = _read_json(knowledge_dir / "compiled_config.json")
    data_config = dict(compiled_config.get("data_config") or {})
    feature_dict = dict(task_config.get("feature_dict") or data_config.get("feature_dict") or {})
    semantics = dict(task_config.get("task_semantics") or data_config.get("task_semantics") or {})
    basic_facts = _basic_facts(task_config=task_config, data_config=data_config, semantics=semantics, feature_dict=feature_dict)
    diagnostics: List[Dict[str, Any]] = []
    artifacts: List[Dict[str, Any]] = []

    dataset_path = _resolve_dataset_path(task_config=task_config, data_config=data_config)
    dataset_id = _dataset_id(dataset_path, task_config=task_config, data_config=data_config, task_id=task_id)
    view_id = _dataset_view_id(
        dataset_path=dataset_path,
        task_config=task_config,
        data_config=data_config,
        semantics=semantics,
    )
    view_descriptor = _dataset_view_descriptor(
        dataset_path=dataset_path,
        task_config=task_config,
        data_config=data_config,
        semantics=semantics,
    )
    path = dataset_view_dir(base_dir, dataset_id, view_id) / "dataset_profile.json"
    if dataset_path:
        _emit_progress(progress, f"解析数据集路径完成: {dataset_path}", language=language, english_message=f"Resolved dataset path: {dataset_path}")
        artifacts.append({"kind": "dataset_csv", "path": str(dataset_path)})
        basic_facts["dataset_path"] = str(dataset_path)
    else:
        diagnostics.append(
            {
                "severity": "error",
                "code": "DATASET_PATH_UNRESOLVED",
                "message": "No resolvable dataset path was found from task_config or compiled_config.",
            }
        )

    raw_characteristics = {key: None for key in CHARACTERISTIC_KEYS}
    status = "failed" if dataset_path is None else "ok"
    frame: Optional[pd.DataFrame] = None
    characteristics_cache: Dict[str, Any] = {
        "hit": False,
        "key": None,
        "path": None,
        "computed_at": None,
    }
    if dataset_path is not None:
        descriptor = _analysis_view_descriptor(
            dataset_path=dataset_path,
            task_config=task_config,
            data_config=data_config,
            semantics=semantics,
        )
        cache_path = _characteristics_cache_path(base_dir, dataset_id, view_id)
        cache_key = f"{dataset_id}/{view_id}"
        cached = {} if force_refresh else _load_characteristics_cache(cache_path, descriptor)
        characteristics_cache.update({"key": cache_key, "path": str(cache_path)})
        if cached:
            characteristics_cache.update({"hit": True, "computed_at": cached.get("computed_at")})
            basic_facts.update(dict(cached.get("basic_stats") or {}))
            raw_characteristics = dict(cached.get("raw_characteristics") or raw_characteristics)
            cached_frame = dict(cached.get("analysis_frame") or {})
            artifacts.append({"kind": "analysis_frame", **cached_frame})
            artifacts.append({"kind": "dataset_characteristics", "path": str(cache_path), "hit": True})
            _emit_progress(
                progress,
                f"命中 dataset_knowledge 特征产物: key={cache_key}, computed_at={cached.get('computed_at')}",
                language=language,
                english_message=f"Dataset-knowledge characteristics hit: key={cache_key}, computed_at={cached.get('computed_at')}",
            )
        else:
            try:
                _emit_progress(progress, "dataset_knowledge 特征产物未命中；读取并整理 CSV，识别 wide/long 数据格式", language=language, english_message="Dataset-knowledge characteristics miss; reading and normalizing CSV; detecting wide/long data format")
                frame = _read_dataset_frame(
                    dataset_path,
                    task_config=task_config,
                    data_config=data_config,
                    semantics=semantics,
                    progress=progress,
                    language=language,
                )
            except Exception as exc:
                diagnostics.append(
                    {
                        "severity": "error",
                        "code": "DATASET_READ_FAILED",
                        "message": f"Failed to read dataset CSV: {type(exc).__name__}: {exc}",
                    }
                )
                status = "failed"
            else:
                _emit_progress(progress, f"数据表整理完成: rows={int(frame.shape[0])}, variables={int(frame.shape[1])}", language=language, english_message=f"Analysis table ready: rows={int(frame.shape[0])}, variables={int(frame.shape[1])}")
                frame_stats = _frame_basic_stats(frame)
                frame_artifact = {"rows": int(frame.shape[0]), "columns": list(frame.columns)}
                artifacts.append({"kind": "analysis_frame", **frame_artifact})
                basic_facts.update(frame_stats)
                try:
                    _emit_progress(progress, "开始计算 Python 数据集特征", language=language, english_message="Computing Python dataset characteristics")
                    raw_characteristics = _compute_raw_characteristics(
                        frame,
                        max_points=_analysis_points_limit(task_config),
                        max_stl_periods=_stl_period_limit(task_config),
                        progress=progress,
                        language=language,
                    )
                    computed_at = datetime.now().isoformat()
                    try:
                        _write_characteristics_cache(
                            cache_path,
                            {
                                "schema_version": CHARACTERISTICS_SCHEMA_VERSION,
                                "analysis_view": descriptor,
                                "computed_at": computed_at,
                                "basic_stats": frame_stats,
                                "analysis_frame": frame_artifact,
                                "raw_characteristics": raw_characteristics,
                            },
                        )
                    except Exception as cache_exc:
                        diagnostics.append(
                            {
                                "severity": "warning",
                                "code": "CHARACTERISTICS_CACHE_WRITE_FAILED",
                                "message": f"Dataset characteristics were computed but cache write failed: {type(cache_exc).__name__}: {cache_exc}",
                            }
                        )
                    else:
                        characteristics_cache.update({"computed_at": computed_at})
                        artifacts.append({"kind": "dataset_characteristics", "path": str(cache_path), "hit": False})
                        (path.parent / "analysis_frame.json").write_text(
                            json.dumps(frame_artifact, ensure_ascii=False, indent=2, default=str),
                            encoding="utf-8",
                        )
                    _emit_progress(progress, f"Python 特征计算完成: {raw_characteristics}", language=language, english_message=f"Python characteristics complete: {raw_characteristics}")
                except Exception as exc:
                    diagnostics.append(
                        {
                            "severity": "error",
                            "code": "PYTHON_CHARACTERISTICS_FAILED",
                            "message": f"Python dataset characteristics failed: {type(exc).__name__}: {exc}",
                        }
                    )
                    status = "failed"

    _emit_progress(progress, "根据原始特征生成 derived_claims", language=language, english_message="Generating derived_claims from raw characteristics")
    derived_claims = _derive_claims(raw_characteristics, basic_facts, feature_dict)
    _emit_progress(progress, f"derived_claims 完成: count={len(derived_claims)}, claims={[item.get('claim') for item in derived_claims]}", language=language, english_message=f"derived_claims complete: count={len(derived_claims)}, claims={[item.get('claim') for item in derived_claims]}")
    _emit_progress(progress, "生成 research_implications", language=language, english_message="Generating research_implications")
    research_implications = _build_research_implications(derived_claims)
    _emit_progress(progress, f"research_implications 完成: count={len(research_implications)}", language=language, english_message=f"research_implications complete: count={len(research_implications)}")
    _emit_progress(progress, "生成 LLM narrative", language=language, english_message="Generating LLM narrative")
    llm_narrative = _build_llm_narrative(
        task_id=task_id,
        task_config=task_config,
        basic_facts=basic_facts,
        raw_characteristics=raw_characteristics,
        derived_claims=derived_claims,
        research_implications=research_implications,
        diagnostics=diagnostics,
        client=client,
        progress=progress,
        language=language,
    )
    _emit_progress(progress, f"LLM narrative 完成: status={llm_narrative.get('status')}", language=language, english_message=f"LLM narrative complete: status={llm_narrative.get('status')}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "created_at": datetime.now().isoformat(),
        "status": status,
        "characteristics_engine": "python",
        "characteristics_cache": characteristics_cache,
        "basic_facts": basic_facts,
        "raw_characteristics": raw_characteristics,
        "derived_claims": derived_claims,
        "research_implications": research_implications,
        "llm_narrative": llm_narrative,
        "diagnostics": diagnostics,
        "source_artifacts": artifacts,
        "dataset_knowledge": {
            "dataset_id": dataset_id,
            "view_id": view_id,
            "view_descriptor": view_descriptor,
            "dataset_profile_path": str(path),
            "characteristics_path": str(_characteristics_cache_path(base_dir, dataset_id, view_id)),
            "analysis_frame_path": str(path.parent / "analysis_frame.json"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress, f"写入 dataset_knowledge profile: {path}", language=language, english_message=f"Writing dataset-knowledge profile: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_dataset_manifest(
        base_dir=base_dir,
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        task_config=task_config,
        data_config=data_config,
    )
    _write_dataset_task_binding(
        base_dir=base_dir,
        task_id=task_id,
        dataset_id=dataset_id,
        view_id=view_id,
        profile_path=path,
        dataset_path=dataset_path,
        view_descriptor=view_descriptor,
    )
    _update_dataset_registry(
        base_dir=base_dir,
        dataset_id=dataset_id,
        view_id=view_id,
        dataset_path=dataset_path,
        profile_path=path,
    )
    _emit_progress(progress, f"数据集诊断完成: status={status}, elapsed={time.monotonic() - started_at:.1f}s", language=language, english_message=f"Dataset diagnosis complete: status={status}, elapsed={time.monotonic() - started_at:.1f}s")
    return payload


def ensure_dataset_profile(
    *,
    task_id: str,
    base_dir: str,
    client: Any | None = None,
    force_refresh: bool = False,
    progress: ProgressCallback | None = None,
    language: str = "zh",
) -> Dict[str, Any]:
    existing = load_dataset_profile(base_dir, task_id)
    mode = dataset_diagnosis_mode(base_dir, task_id)
    if existing.get("schema_version") == SCHEMA_VERSION and existing.get("status") == "ok" and not force_refresh:
        _emit_progress(progress, f"复用已存在的 dataset profile: {dataset_profile_path(base_dir, task_id)}", language=language, english_message=f"Reusing existing dataset profile: {dataset_profile_path(base_dir, task_id)}")
        return existing
    if mode == "skip" and not force_refresh:
        return write_skipped_dataset_profile(
            task_id=task_id,
            base_dir=base_dir,
            language=language,
            progress=progress,
        )
    if mode == "reuse" and not force_refresh:
        raise RuntimeError(
            "DATASET_DIAGNOSIS_REUSE_MISSING: dataset_diagnosis_mode=reuse requires an existing dataset profile."
        )
    effective_client = client
    if effective_client is None:
        try:
            effective_client = create_task_client(base_dir=base_dir, task_id=task_id)
        except Exception as exc:
            _emit_progress(
                progress,
                f"LLM client 创建失败，使用程序生成的 narrative: {type(exc).__name__}: {exc}",
                language=language,
                english_message=f"LLM client creation failed; using program-generated narrative: {type(exc).__name__}: {exc}",
            )
            effective_client = None
    return generate_dataset_profile(
        task_id=task_id,
        base_dir=base_dir,
        client=effective_client,
        force_refresh=force_refresh,
        progress=progress,
        language=language,
    )


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _basic_facts(*, task_config: Dict[str, Any], data_config: Dict[str, Any], semantics: Dict[str, Any], feature_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_name": str(task_config.get("data_set_name") or data_config.get("data_set_name") or ""),
        "dataset_path": str(task_config.get("dataset_path") or data_config.get("dataset_path") or ""),
        "task_mode": str(semantics.get("task_mode") or ""),
        "frequency": str(semantics.get("frequency") or feature_dict.get("canonical_freq") or feature_dict.get("freq") or ""),
        "seq_len": task_config.get("seq_len"),
        "horizon": task_config.get("horizon"),
        "num_variables": 1 if feature_dict.get("if_univariate") else None,
        "train_shape": None,
        "missing_ratio": None,
        "scale": bool(data_config.get("scale", True)),
        "is_univariate": bool(feature_dict.get("if_univariate")) if feature_dict.get("if_univariate") is not None else None,
        "has_trend_flag": feature_dict.get("if_trend"),
        "has_seasonality_flag": feature_dict.get("if_season"),
    }


def _resolve_dataset_path(*, task_config: Dict[str, Any], data_config: Dict[str, Any]) -> Optional[Path]:
    candidates = [
        str(task_config.get("dataset_path") or "").strip(),
        str(data_config.get("dataset_path") or "").strip(),
    ]
    for name in list(data_config.get("data_name_list") or []):
        text = str(name or "").strip()
        if text:
            candidates.append(text)
    for text in candidates:
        if not text:
            continue
        resolved = resolve_dataset_path(text)
        if resolved:
            path = Path(resolved)
            if path.exists():
                return path
    return None


def _target_columns_for_profile(
    *,
    task_config: Dict[str, Any],
    data_config: Dict[str, Any],
    semantics: Dict[str, Any],
) -> List[str]:
    task_mode = str(semantics.get("task_mode") or task_config.get("task_mode") or "").upper()
    if task_mode not in {"MS", "SS"}:
        return []
    raw_targets = (
        task_config.get("target_columns")
        or semantics.get("target_columns")
        or data_config.get("target_columns")
        or []
    )
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    return [str(item).strip() for item in list(raw_targets or []) if str(item).strip()]


def _filter_profile_targets(
    frame: pd.DataFrame,
    *,
    task_config: Dict[str, Any],
    data_config: Dict[str, Any],
    semantics: Dict[str, Any],
    progress: ProgressCallback | None = None,
    language: str = "zh",
) -> pd.DataFrame:
    targets = _target_columns_for_profile(task_config=task_config, data_config=data_config, semantics=semantics)
    if not targets or frame.empty:
        return frame
    lookup = {str(col).strip().lower(): col for col in frame.columns}
    selected = [lookup[target.lower()] for target in targets if target.lower() in lookup]
    if not selected:
        _emit_progress(
            progress,
            f"目标列诊断裁剪未命中: target_columns={targets}; 保留全部数值列",
            language=language,
            english_message=f"Target-column profile filter missed target_columns={targets}; keeping all numeric columns",
        )
        return frame
    result = frame[selected]
    _emit_progress(
        progress,
        f"按任务目标裁剪诊断列: target_columns={targets}, profile_columns={list(result.columns)}",
        language=language,
        english_message=f"Filtered profile columns by task targets: target_columns={targets}, profile_columns={list(result.columns)}",
    )
    return result


def _analysis_points_limit(task_config: Dict[str, Any]) -> int:
    if bool(task_config.get("build_mode")):
        return BUILD_MODE_MAX_ANALYSIS_POINTS
    return MAX_ANALYSIS_POINTS


def _stl_period_limit(task_config: Dict[str, Any]) -> int:
    if bool(task_config.get("build_mode")):
        return BUILD_MODE_MAX_STL_PERIODS
    return 8


def _read_dataset_frame(
    path: Path,
    *,
    task_config: Dict[str, Any] | None = None,
    data_config: Dict[str, Any] | None = None,
    semantics: Dict[str, Any] | None = None,
    progress: ProgressCallback | None = None,
    language: str = "zh",
) -> pd.DataFrame:
    task_config = dict(task_config or {})
    data_config = dict(data_config or {})
    semantics = dict(semantics or {})
    read_started = time.monotonic()
    data = pd.read_csv(path)
    _emit_progress(
        progress,
        f"CSV 读取完成: rows={int(data.shape[0])}, columns={int(data.shape[1])}, elapsed={time.monotonic() - read_started:.1f}s",
        language=language,
        english_message=f"CSV read complete: rows={int(data.shape[0])}, columns={int(data.shape[1])}, elapsed={time.monotonic() - read_started:.1f}s",
    )
    lowered = [str(col).strip().lower() for col in data.columns]
    if "cols" in lowered and "data" in lowered:
        _emit_progress(progress, "检测到 long-format 数据: 使用 cols/data 还原为多变量宽表", language=language, english_message="Detected long-format data: reconstructing multivariate wide table from cols/data")
        cols_col = data.columns[lowered.index("cols")]
        data_col = data.columns[lowered.index("data")]
        time_col = None
        for candidate in ("date", "time", "timestamp", "datetime"):
            if candidate in lowered:
                time_col = data.columns[lowered.index(candidate)]
                break
        if time_col is not None:
            pivot_started = time.monotonic()
            _emit_progress(progress, f"使用时间列 {time_col} 构建 pivot 表", language=language, english_message=f"Building pivot table with time column {time_col}")
            frame = data[[time_col, cols_col, data_col]].copy()
            frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
            frame = frame.dropna(subset=[time_col])
            pivot = frame.pivot_table(index=time_col, columns=cols_col, values=data_col, aggfunc="first")
            pivot = pivot.sort_index()
            result = _coerce_numeric_frame(pivot.reset_index(drop=True), progress=progress, language=language)
            _emit_progress(
                progress,
                f"long-format pivot 完成: rows={int(result.shape[0])}, variables={int(result.shape[1])}, elapsed={time.monotonic() - pivot_started:.1f}s",
                language=language,
                english_message=f"long-format pivot complete: rows={int(result.shape[0])}, variables={int(result.shape[1])}, elapsed={time.monotonic() - pivot_started:.1f}s",
            )
            return _filter_profile_targets(
                result,
                task_config=task_config,
                data_config=data_config,
                semantics=semantics,
                progress=progress,
                language=language,
            )
        pivot_started = time.monotonic()
        _emit_progress(progress, "未找到时间列，按每个 cols 的出现顺序构建 pivot 表", language=language, english_message="No time column found; building pivot table by each cols occurrence order")
        frame = data[[cols_col, data_col]].copy()
        frame["_order"] = frame.groupby(cols_col).cumcount()
        pivot = frame.pivot(index="_order", columns=cols_col, values=data_col).sort_index()
        result = _coerce_numeric_frame(pivot, progress=progress, language=language)
        _emit_progress(
            progress,
            f"long-format pivot 完成: rows={int(result.shape[0])}, variables={int(result.shape[1])}, elapsed={time.monotonic() - pivot_started:.1f}s",
            language=language,
            english_message=f"long-format pivot complete: rows={int(result.shape[0])}, variables={int(result.shape[1])}, elapsed={time.monotonic() - pivot_started:.1f}s",
        )
        return _filter_profile_targets(
            result,
            task_config=task_config,
            data_config=data_config,
            semantics=semantics,
            progress=progress,
            language=language,
        )
    _emit_progress(progress, "检测到 wide-format 数据: 直接筛选数值列", language=language, english_message="Detected wide-format data: selecting numeric columns directly")
    numeric = _coerce_numeric_frame(data, progress=progress, language=language)
    if numeric.empty:
        raise ValueError(f"No numeric forecasting columns found in {path}")
    return _filter_profile_targets(
        numeric,
        task_config=task_config,
        data_config=data_config,
        semantics=semantics,
        progress=progress,
        language=language,
    )


def _coerce_numeric_frame(frame: pd.DataFrame, *, progress: ProgressCallback | None = None, language: str = "zh") -> pd.DataFrame:
    numeric_cols: List[str] = []
    for col in frame.columns:
        lowered = str(col).strip().lower()
        if lowered in {"date", "time", "timestamp", "datetime"}:
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        if series.notna().sum() >= max(8, int(len(series) * 0.5)):
            numeric_cols.append(col)
    if not numeric_cols:
        return pd.DataFrame()
    result = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    result = result.dropna(axis=1, how="all")
    _emit_progress(progress, f"数值列筛选完成: numeric_columns={len(result.columns)}", language=language, english_message=f"Numeric column selection complete: numeric_columns={len(result.columns)}")
    return result


def _frame_basic_stats(frame: pd.DataFrame) -> Dict[str, Any]:
    total = int(frame.shape[0] * frame.shape[1]) if not frame.empty else 0
    missing = int(frame.isna().sum().sum()) if not frame.empty else 0
    return {
        "num_variables": int(frame.shape[1]),
        "train_shape": [int(frame.shape[0]), int(frame.shape[1])],
        "missing_ratio": float(missing / total) if total > 0 else 0.0,
    }


def _compute_raw_characteristics(
    frame: pd.DataFrame,
    *,
    max_points: int = MAX_ANALYSIS_POINTS,
    max_stl_periods: int = 8,
    progress: ProgressCallback | None = None,
    language: str = "zh",
) -> Dict[str, Any]:
    clean = frame.copy()
    if clean.empty:
        raise ValueError("Dataset frame is empty.")
    max_points = max(MIN_SERIES_LENGTH, int(max_points or MAX_ANALYSIS_POINTS))
    if clean.shape[0] > max_points:
        _emit_progress(progress, f"特征计算截断到最近 {max_points} 个时间点: original_rows={int(clean.shape[0])}", language=language, english_message=f"Truncated characteristic computation to latest {max_points} time points: original_rows={int(clean.shape[0])}")
        clean = clean.iloc[-max_points:, :]
    per_series: List[Dict[str, float]] = []
    columns = list(clean.columns)
    total = len(columns)
    for index, col in enumerate(columns, start=1):
        series = clean[col].dropna().astype(float).to_numpy()
        if series.size < MIN_SERIES_LENGTH:
            _emit_progress(progress, f"跳过特征计算 {index}/{total}: column={col}, valid_points={series.size} < {MIN_SERIES_LENGTH}", language=language, english_message=f"Skipping characteristic computation {index}/{total}: column={col}, valid_points={series.size} < {MIN_SERIES_LENGTH}")
            continue
        series_started = time.monotonic()
        _emit_progress(progress, f"计算序列特征 {index}/{total}: column={col}, points={series.size}", language=language, english_message=f"Computing series characteristics {index}/{total}: column={col}, points={series.size}")
        per_series.append(_series_characteristics(series, max_stl_periods=max_stl_periods, progress=progress, label=str(col), language=language))
        _emit_progress(progress, f"序列特征完成 {index}/{total}: column={col}, elapsed={time.monotonic() - series_started:.1f}s", language=language, english_message=f"Series characteristics complete {index}/{total}: column={col}, elapsed={time.monotonic() - series_started:.1f}s")
    if not per_series:
        raise ValueError("No series with enough points for dataset diagnosis.")
    aggregate: Dict[str, Any] = {}
    _emit_progress(progress, "聚合逐变量特征", language=language, english_message="Aggregating per-variable characteristics")
    for key in ("Transition", "Shifting", "Seasonality", "Trend", "Stationarity", "Short_term_jsd", "Long_term_jsd"):
        values = [float(item[key]) for item in per_series if isinstance(item.get(key), (int, float)) and math.isfinite(float(item[key]))]
        aggregate[key] = float(np.mean(values)) if values else None
    _emit_progress(progress, "计算变量相关性", language=language, english_message="Computing variable correlation")
    aggregate["Correlation"] = _dataset_correlation(clean)
    return aggregate


def _series_characteristics(
    series: np.ndarray,
    *,
    max_stl_periods: int = 8,
    progress: ProgressCallback | None = None,
    label: str = "",
    language: str = "zh",
) -> Dict[str, float]:
    series = np.asarray(series, dtype=float)
    if series.size > MAX_ANALYSIS_POINTS:
        series = series[-MAX_ANALYSIS_POINTS:]
    _emit_progress(progress, f"  [{label}] FFT 候选周期识别", language=language, english_message=f"  [{label}] FFT candidate period detection")
    periods = _candidate_periods(series)
    _emit_progress(progress, f"  [{label}] STL 强度计算: candidate_periods={periods[:8]}", language=language, english_message=f"  [{label}] STL strength computation: candidate_periods={periods[:8]}")
    seasonality, trend = _best_stl_strength(series, periods, max_periods=max_stl_periods)
    _emit_progress(progress, f"  [{label}] ADF 平稳性检验", language=language, english_message=f"  [{label}] ADF stationarity test")
    adf_p = _safe_adf_pvalue(series)
    _emit_progress(progress, f"  [{label}] Transition/Shifting/JSD 统计", language=language, english_message=f"  [{label}] Transition/Shifting/JSD statistics")
    return {
        "Transition": _transition_score(series),
        "Shifting": _shift_score(series),
        "Seasonality": seasonality,
        "Trend": trend,
        "Stationarity": adf_p,
        "Short_term_jsd": _window_jsd(series, 30),
        "Long_term_jsd": _window_jsd(series, 336),
    }


def _candidate_periods(series: np.ndarray) -> List[int]:
    centered = series - float(np.nanmean(series))
    if centered.size < 8:
        return DEFAULT_PERIODS[:3]
    fft = np.fft.rfft(centered)
    amps = np.abs(fft)[1:]
    freqs = np.fft.rfftfreq(centered.size)[1:]
    periods: List[int] = []
    if amps.size:
        order = np.argsort(amps)[::-1]
        for idx in order[:16]:
            freq = freqs[idx]
            if freq <= 0:
                continue
            period = int(round(1.0 / freq))
            period = _adjust_period(period)
            if period >= 4 and period not in periods:
                periods.append(period)
    for default in DEFAULT_PERIODS:
        if default not in periods:
            periods.append(default)
    return periods


def _adjust_period(period_value: int) -> int:
    for target, tolerance in (
        (4, 1), (7, 1), (12, 2), (24, 3), (48, 4), (52, 2), (96, 10), (144, 10), (168, 10), (336, 50), (672, 20), (1008, 20), (1440, 20)
    ):
        if abs(period_value - target) <= tolerance:
            return target
    return period_value


def _best_stl_strength(series: np.ndarray, periods: Iterable[int], *, max_periods: int = 8) -> tuple[float, float]:
    best_seasonality = 0.0
    best_trend = 0.0
    upper = max(int(series.size / 3), 12)
    values = pd.Series(series)
    checked = 0
    for period in periods:
        if period < 4 or period >= upper:
            continue
        if checked >= max(1, int(max_periods or 1)):
            break
        try:
            result = STL(values, period=int(period), robust=True).fit()
        except Exception:
            continue
        checked += 1
        resid = result.resid.to_numpy()
        detrend = (values - result.trend).to_numpy()
        deseasonal = (values - result.seasonal).to_numpy()
        deseasonal_var = float(np.nanvar(deseasonal))
        detrend_var = float(np.nanvar(detrend))
        resid_var = float(np.nanvar(resid))
        trend_strength = 0.0 if deseasonal_var == 0 else max(0.0, 1.0 - resid_var / deseasonal_var)
        seasonal_strength = 0.0 if detrend_var == 0 else max(0.0, 1.0 - resid_var / detrend_var)
        best_seasonality = max(best_seasonality, float(np.clip(seasonal_strength, 0.0, 1.0)))
        best_trend = max(best_trend, float(np.clip(trend_strength, 0.0, 1.0)))
    return best_seasonality, best_trend


def _safe_adf_pvalue(series: np.ndarray) -> float:
    try:
        return float(adfuller(series, autolag="AIC")[1])
    except Exception:
        return 1.0


def _transition_score(series: np.ndarray) -> float:
    values = np.asarray(series, dtype=float)
    if values.size < 8:
        return 0.0
    q1, q2 = np.nanquantile(values, [1 / 3, 2 / 3])
    symbols = np.where(values <= q1, 0, np.where(values <= q2, 1, 2))
    matrix = np.zeros((3, 3), dtype=float)
    for left, right in zip(symbols[:-1], symbols[1:]):
        matrix[int(left), int(right)] += 1.0
    total = float(matrix.sum())
    if total <= 0:
        return 0.0
    matrix /= total
    return float(np.clip(np.trace(matrix), 0.0, 1.0))


def _shift_score(series: np.ndarray) -> float:
    values = np.asarray(series, dtype=float)
    if values.size < 64:
        return 0.0
    window = max(16, min(128, values.size // 8))
    if window <= 4:
        return 0.0
    scores: List[float] = []
    for start in range(0, values.size - 2 * window + 1, window):
        left = values[start : start + window]
        right = values[start + window : start + 2 * window]
        if np.nanstd(left) == 0 and np.nanstd(right) == 0:
            scores.append(0.0)
            continue
        scores.append(_distribution_distance(left, right))
    return float(np.clip(np.mean(scores) if scores else 0.0, 0.0, 1.0))


def _distribution_distance(left: np.ndarray, right: np.ndarray) -> float:
    values = np.concatenate([left, right])
    bins = min(24, max(8, int(np.sqrt(values.size))))
    hist_l, edges = np.histogram(left, bins=bins, density=True)
    hist_r, _ = np.histogram(right, bins=edges, density=True)
    return float(np.clip(_js_divergence(hist_l, hist_r), 0.0, 1.0))


def _window_jsd(series: np.ndarray, window_size: int) -> float:
    values = np.asarray(series, dtype=float)
    if values.size < max(window_size, 8):
        return 0.0
    jsd_list: List[float] = []
    num_windows = values.size // window_size
    for idx in range(num_windows):
        window = values[idx * window_size : (idx + 1) * window_size]
        sigma = float(np.nanstd(window))
        if sigma == 0:
            jsd_list.append(0.0)
            continue
        hist, edges = np.histogram(window, bins=min(24, max(8, int(np.sqrt(window_size)))), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        pdf = norm.pdf(centers, float(np.nanmean(window)), sigma)
        jsd_list.append(_js_divergence(hist, pdf))
    return float(np.clip(np.mean(jsd_list) if jsd_list else 0.0, 0.0, 1.0))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (entropy(p, m) + entropy(q, m)))


def _dataset_correlation(frame: pd.DataFrame) -> Optional[float]:
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if numeric.shape[1] <= 1:
        return None
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    values = upper.stack().to_numpy(dtype=float)
    if values.size == 0:
        return None
    return float(np.clip(np.nanmean(values), 0.0, 1.0))


def _derive_claims(raw: Dict[str, Any], basic_facts: Dict[str, Any], feature_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []

    def add(claim: str, confidence: float, evidence: str, implication: str, profiles: List[str]) -> None:
        claims.append(
            {
                "claim": claim,
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
                "evidence": evidence,
                "research_implication": implication,
                "implication_code": _topic_for_claim(claim),
                "affected_mechanism_profiles": profiles,
            }
        )

    seasonality = _float_or_none(raw.get("Seasonality"))
    trend = _float_or_none(raw.get("Trend"))
    stationarity = _float_or_none(raw.get("Stationarity"))
    shifting = _float_or_none(raw.get("Shifting"))
    correlation = _float_or_none(raw.get("Correlation"))
    transition = _float_or_none(raw.get("Transition"))
    short_jsd = _float_or_none(raw.get("Short_term_jsd"))
    long_jsd = _float_or_none(raw.get("Long_term_jsd"))

    if seasonality is not None and seasonality >= 0.85:
        add("strong_seasonality", seasonality, f"Seasonality={seasonality:.4f}", "Seasonal/frequency/decomposition mechanisms may be valuable.", ["frequency_modeling", "seasonal_trend_decomposition", "multi_scale_temporal_modeling"])
    elif feature_dict.get("if_season") is True:
        add("seasonality_flag_present", 0.6, "feature_dict.if_season=true", "Seasonal mechanisms are worth keeping in the opportunity pool.", ["frequency_modeling", "seasonal_trend_decomposition"])

    if trend is not None and trend >= 0.70:
        add("strong_trend", trend, f"Trend={trend:.4f}", "Trend-aware branches or decomposition may be valuable.", ["trend_path", "seasonal_trend_decomposition", "adaptive_trend_correction"])
    elif feature_dict.get("if_trend") is True:
        add("trend_flag_present", 0.6, "feature_dict.if_trend=true", "Trend-aware mechanisms should stay in consideration.", ["trend_path", "seasonal_trend_decomposition"])

    if stationarity is not None and stationarity > 0.05:
        add("nonstationary_behavior", min(1.0, stationarity), f"ADF p-value={stationarity:.4f}", "Normalization and distribution-adaptive mechanisms may be valuable.", ["normalization", "nonstationary_transform", "distribution_shift_handling"])

    if shifting is not None and shifting >= 0.60:
        add("distribution_shift", shifting, f"Shifting={shifting:.4f}", "Adaptive scaling, robust normalization, or time-conditioned modules may be valuable.", ["distribution_shift_handling", "adaptive_scaling", "normalization"])

    if correlation is not None and correlation >= 0.65 and int(basic_facts.get("num_variables") or 0) > 1:
        add("strong_cross_variable_correlation", correlation, f"Correlation={correlation:.4f}", "Cross-variable mixing and shared latent mechanisms may be valuable.", ["channel_mixing", "cross_variable_attention", "graph_relation_modeling"])

    if transition is not None and transition >= 0.60:
        add("structured_state_transition", transition, f"Transition={transition:.4f}", "State-aware routing or regime-sensitive mechanisms may be valuable.", ["routing", "regime_switching", "temporal_state_modeling"])

    if short_jsd is not None and short_jsd >= 0.12:
        add("local_non_gaussianity", min(1.0, short_jsd * 2.5), f"Short_term_jsd={short_jsd:.4f}", "Short-horizon robust filtering or local adaptive mechanisms may be valuable.", ["robust_filtering", "local_adaptive_gating", "noise_robust_modeling"])

    if long_jsd is not None and long_jsd >= 0.12:
        add("global_distribution_complexity", min(1.0, long_jsd * 2.5), f"Long_term_jsd={long_jsd:.4f}", "Distribution-aware or heavy-tail-robust mechanisms may be valuable.", ["distribution_shift_handling", "robust_representation", "adaptive_normalization"])

    return claims


def _build_research_implications(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    implications: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for claim in claims:
        topic = _topic_for_claim(str(claim.get("claim") or ""))
        if not topic or topic in seen:
            continue
        seen.add(topic)
        confidence = float(claim.get("confidence") or 0.0)
        implications.append(
            {
                "topic": topic,
                "priority": "high" if confidence >= 0.8 else "medium",
                "confidence": confidence,
                "reason": str(claim.get("research_implication") or ""),
                "reason_code": str(claim.get("implication_code") or topic),
                "mechanism_profiles": list(claim.get("affected_mechanism_profiles") or []),
                "planner_effect": "increase_priority",
                "supporting_claims": [str(claim.get("claim") or "")],
            }
        )
    return implications


def _topic_for_claim(claim: str) -> str:
    mapping = {
        "strong_seasonality": "frequency_or_periodic_modeling",
        "seasonality_flag_present": "frequency_or_periodic_modeling",
        "strong_trend": "trend_aware_modeling",
        "trend_flag_present": "trend_aware_modeling",
        "nonstationary_behavior": "nonstationary_adaptation",
        "distribution_shift": "nonstationary_adaptation",
        "strong_cross_variable_correlation": "cross_variable_modeling",
        "structured_state_transition": "state_sensitive_temporal_modeling",
        "local_non_gaussianity": "local_robustness",
        "global_distribution_complexity": "distribution_robustness",
    }
    return mapping.get(claim, "")


def _build_llm_narrative(
    *,
    task_id: str,
    task_config: Dict[str, Any],
    basic_facts: Dict[str, Any],
    raw_characteristics: Dict[str, Any],
    derived_claims: List[Dict[str, Any]],
    research_implications: List[Dict[str, Any]],
    diagnostics: List[Dict[str, Any]],
    client: Any | None,
    progress: ProgressCallback | None = None,
    language: str = "zh",
) -> Dict[str, Any]:
    deterministic = _deterministic_narrative(
        basic_facts,
        derived_claims,
        research_implications,
        diagnostics,
        language=language,
    )
    if client is None or not getattr(client, "api_available", False):
        _emit_progress(progress, "LLM client 不可用，使用程序生成的 narrative", language=language, english_message="LLM client unavailable; using program-generated narrative")
        diagnostics.append(
            {
                "severity": "info",
                "code": "LLM_CLIENT_UNAVAILABLE",
                "message": "No available LLM client was attached for dataset narrative generation.",
            }
        )
        deterministic["status"] = "deterministic"
        return deterministic
    prompt = {
        "task": "Generate concise JSON narrative for dataset diagnosis before baseline search.",
        "output_language": "English" if _is_english(language) else "Chinese",
        "requirements": {
            "facts_only": True,
            "do_not_override_raw_metrics": True,
            "length": "short",
            "human_facing_language": "English" if _is_english(language) else "Chinese",
        },
        "input": {
            "task_id": task_id,
            "objective_metric": task_config.get("objective_metric"),
            "basic_facts": basic_facts,
            "raw_characteristics": raw_characteristics,
            "derived_claims": derived_claims,
            "research_implications": research_implications,
            "diagnostics": diagnostics,
        },
        "output_schema": {
            "dataset_summary": "string",
            "research_interpretation": "string",
            "evidence_conflicts": [{"conflict": "string", "interpretation": "string"}],
            "suggested_opportunity_themes": [{"theme": "string", "reason": "string"}],
            "limitations": ["string"],
        },
    }
    try:
        llm_started = time.monotonic()
        _emit_progress(progress, "请求 LLM 生成 dataset narrative", language=language, english_message="Requesting LLM dataset narrative")
        payload = client.call_json(
            stage="review_report",
            round_num=0,
            stage_label="dataset_diagnosis",
            execution_label="dataset_diagnosis",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are generating the narrative layer for EvoCast dataset diagnosis. "
                        "Use only the provided facts. Return valid JSON only. "
                        + (
                            "Write every human-facing text field in English."
                            if _is_english(language)
                            else "Write every human-facing text field in Chinese. Do not leave full English sentences."
                        )
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, indent=2, default=str)},
            ],
        )
    except (ProviderAPIError, Exception) as exc:
        _emit_progress(progress, f"LLM narrative 请求失败: {type(exc).__name__}: {exc}", language=language, english_message=f"LLM narrative request failed: {type(exc).__name__}: {exc}")
        diagnostics.append(
            {
                "severity": "warning",
                "code": "LLM_NARRATIVE_FAILED",
                "message": f"Dataset narrative generation failed: {type(exc).__name__}: {exc}",
            }
        )
        deterministic["status"] = "deterministic"
        return deterministic
    _emit_progress(progress, f"LLM narrative 请求完成: elapsed={time.monotonic() - llm_started:.1f}s", language=language, english_message=f"LLM narrative request complete: elapsed={time.monotonic() - llm_started:.1f}s")
    result = {
        "status": "ok",
        "dataset_summary": str(payload.get("dataset_summary") or deterministic.get("dataset_summary") or ""),
        "research_interpretation": str(payload.get("research_interpretation") or deterministic.get("research_interpretation") or ""),
        "evidence_conflicts": [item for item in list(payload.get("evidence_conflicts") or []) if isinstance(item, dict)],
        "suggested_opportunity_themes": [item for item in list(payload.get("suggested_opportunity_themes") or []) if isinstance(item, dict)],
        "limitations": [str(item) for item in list(payload.get("limitations") or []) if str(item).strip()],
    }
    return result


def _deterministic_narrative(
    basic_facts: Dict[str, Any],
    derived_claims: List[Dict[str, Any]],
    research_implications: List[Dict[str, Any]],
    diagnostics: List[Dict[str, Any]],
    *,
    language: str = "zh",
) -> Dict[str, Any]:
    claims = [str(item.get("claim") or "") for item in derived_claims]
    if _is_english(language):
        summary_parts = []
        if int(basic_facts.get("num_variables") or 0) > 1:
            summary_parts.append(f"{int(basic_facts.get('num_variables') or 0)} variables")
        if "strong_seasonality" in claims:
            summary_parts.append("strong seasonality")
        if "strong_trend" in claims:
            summary_parts.append("strong trend")
        if "distribution_shift" in claims or "nonstationary_behavior" in claims:
            summary_parts.append("distribution drift or non-stationarity")
        if "strong_cross_variable_correlation" in claims:
            summary_parts.append("cross-variable correlation")
        summary = "Dataset diagnosis found " + (", ".join(summary_parts) if summary_parts else "limited structured evidence from the current profile.")
        implications = "; ".join(str(item.get("topic") or "") for item in research_implications[:3] if str(item.get("topic") or "").strip())
        interpretation = f"Promising opportunity themes include {implications}." if implications else "Use this profile as a weak task prior; no dominant dataset-driven research theme was detected."
        limitations = ["Dataset diagnosis is a task prior, not model performance evidence."]
    else:
        traits = [localize_code(claim, "claim", language) for claim in claims[:4]]
        summary = f"数据集诊断覆盖 {int(basic_facts.get('num_variables') or 0)} 个变量"
        summary += "，主要特征包括" + "、".join(traits) + "。" if traits else "，当前未识别出占主导地位的结构特征。"
        topics = [localize_code(item.get("topic"), "topic", language) for item in research_implications[:3] if str(item.get("topic") or "").strip()]
        interpretation = "建议优先关注：" + "、".join(topics) + "。" if topics else "该诊断仅作为研究先验，不单独构成模型性能结论。"
        limitations = ["数据集诊断仅提供任务先验，不等同于模型性能证据。"]
    for diagnostic in diagnostics[:2]:
        limitations.append(str(diagnostic.get("message") or ""))
    return {
        "status": "deterministic",
        "dataset_summary": summary,
        "research_interpretation": interpretation,
        "evidence_conflicts": [],
        "suggested_opportunity_themes": [
            {"theme": str(item.get("topic") or ""), "reason": str(item.get("reason") or "")}
            for item in research_implications[:3]
            if str(item.get("topic") or "").strip()
        ],
        "limitations": limitations,
    }


def _float_or_none(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None
