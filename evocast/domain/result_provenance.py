from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ts_benchmark.recording import read_record_file
from ts_benchmark.utils.compress import compress


PROVENANCE_PREFIX = "evocast_"


class ResultProvenanceError(RuntimeError):
    """Raised when a formal result artifact is not bound to the requested source."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_resolve(path: str | os.PathLike[str]) -> Path:
    return Path(str(path)).expanduser().resolve()


def workspace_root_for_variant(variant_path: str | None) -> str:
    if not str(variant_path or "").strip():
        return ""
    return str(_safe_resolve(str(variant_path)).parent)


def variant_source_sha256(variant_path: str | None) -> str:
    """Hash the executable variant source tree rooted at the round entry folder."""

    if not str(variant_path or "").strip():
        return ""
    entry = _safe_resolve(str(variant_path))
    if not entry.exists():
        raise FileNotFoundError(f"variant_path not found: {entry}")
    root = entry.parent
    h = hashlib.sha256()
    files = [
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and path.is_file()
    ]
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        h.update(rel)
        h.update(b"\0")
        h.update(_sha256_bytes(data).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def model_entry_hash(model_entry: dict[str, Any], *, variant_path: str | None = None) -> str:
    payload = dict(model_entry or {})
    normalized_variant = str(variant_path or payload.get("variant_path") or "").strip()
    source_sha = variant_source_sha256(normalized_variant) if normalized_variant else ""
    payload["variant_path"] = str(_safe_resolve(normalized_variant)) if normalized_variant else None
    payload["variant_source_sha256"] = source_sha or None
    text = json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_result_provenance(
    *,
    task_id: str,
    run_id: str,
    candidate_id: str,
    candidate_kind: str,
    model_entry: dict[str, Any],
    evaluation_budget: str,
    build_mode: bool,
    variant_path: str | None = None,
) -> dict[str, Any]:
    normalized_variant = str(variant_path or model_entry.get("variant_path") or "").strip()
    if str(candidate_kind or "").strip().lower() == "variant" and not normalized_variant:
        raise ResultProvenanceError("variant result provenance requires variant_path")
    source_sha = variant_source_sha256(normalized_variant) if normalized_variant else ""
    if str(candidate_kind or "").strip().lower() == "variant" and not source_sha:
        raise ResultProvenanceError("variant result provenance requires variant_source_sha256")
    entry_hash = model_entry_hash(model_entry, variant_path=normalized_variant or None)
    workspace_root = workspace_root_for_variant(normalized_variant) if normalized_variant else ""
    return {
        "task_id": task_id,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "candidate_kind": candidate_kind,
        "variant_path": str(_safe_resolve(normalized_variant)) if normalized_variant else "",
        "variant_source_sha256": source_sha,
        "workspace_root": workspace_root,
        "model_entry_hash": entry_hash,
        "evaluation_budget": evaluation_budget,
        "build_mode": bool(build_mode),
    }


def _stringify_shape(shape: Any) -> str:
    try:
        return json.dumps(list(shape), ensure_ascii=False)
    except Exception:
        return ""


def _numeric_array(value: Any) -> np.ndarray:
    chunks: list[np.ndarray] = []

    def visit(item: Any) -> None:
        if isinstance(item, pd.DataFrame):
            chunks.append(item.to_numpy(dtype=float, copy=False).reshape(-1))
        elif isinstance(item, pd.Series):
            chunks.append(item.to_numpy(dtype=float, copy=False).reshape(-1))
        elif isinstance(item, np.ndarray):
            chunks.append(item.astype(float, copy=False).reshape(-1))
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            try:
                arr = np.asarray(item, dtype=float).reshape(-1)
            except Exception:
                return
            chunks.append(arr)

    visit(value)
    if not chunks:
        return np.asarray([], dtype=float)
    return np.concatenate(chunks)


def _shape_of(value: Any) -> Any:
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [_shape_of(item) for item in value[:5]]
    return list(np.asarray(value).shape)


def fingerprint_encoded_column(value: Any) -> dict[str, Any]:
    if value is None:
        return {"hash": "", "shape": "", "mean": np.nan, "std": np.nan}
    try:
        if pd.isna(value):
            return {"hash": "", "shape": "", "mean": np.nan, "std": np.nan}
    except Exception:
        pass
    text = str(value)
    if not text or text.lower() == "nan":
        return {"hash": "", "shape": "", "mean": np.nan, "std": np.nan}
    raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    try:
        decoded = pickle.loads(base64.b64decode(text.encode("utf-8")))
        arr = _numeric_array(decoded)
        if arr.size:
            return {
                "hash": raw_hash,
                "shape": _stringify_shape(_shape_of(decoded)),
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr)),
            }
    except Exception:
        pass
    return {"hash": raw_hash, "shape": "", "mean": np.nan, "std": np.nan}


def _record_paths(log_paths: list[str]) -> list[str]:
    paths: list[str] = []
    for raw in log_paths:
        path = Path(str(raw))
        if path.is_dir():
            paths.extend(str(item) for item in path.rglob("*.csv"))
            paths.extend(str(item) for item in path.rglob("*.tar.gz"))
        elif path.is_file() and (str(path).endswith(".csv") or str(path).endswith(".tar.gz")):
            paths.append(str(path))
    return sorted(set(paths))


def _write_record_file_exact(path: str, df: pd.DataFrame) -> None:
    if str(path).endswith(".tar.gz"):
        csv_name = Path(str(path)[:-7]).name
        data = {csv_name: df.to_csv(index=False)}
        Path(path).write_bytes(compress(data, method="gz"))
    else:
        df.to_csv(path, index=False)


def stamp_result_artifacts(
    log_paths: list[str],
    provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    """Write EvoCast provenance columns into formal result artifacts."""

    stamped: list[dict[str, Any]] = []
    for path in _record_paths(log_paths):
        try:
            df = read_record_file(path)
        except Exception:
            continue
        if df.empty or "model_name" not in df.columns:
            continue

        prediction_hashes: list[str] = []
        prediction_shapes: list[str] = []
        prediction_means: list[float] = []
        prediction_stds: list[float] = []
        actual_hashes: list[str] = []
        actual_shapes: list[str] = []
        for _, row in df.iterrows():
            pred = fingerprint_encoded_column(row.get("inference_data"))
            actual = fingerprint_encoded_column(row.get("actual_data"))
            prediction_hashes.append(pred["hash"])
            prediction_shapes.append(pred["shape"])
            prediction_means.append(pred["mean"])
            prediction_stds.append(pred["std"])
            actual_hashes.append(actual["hash"])
            actual_shapes.append(actual["shape"])

        for key, value in provenance.items():
            df[f"{PROVENANCE_PREFIX}{key}"] = value
        df[f"{PROVENANCE_PREFIX}prediction_hash"] = prediction_hashes
        df[f"{PROVENANCE_PREFIX}prediction_shape"] = prediction_shapes
        df[f"{PROVENANCE_PREFIX}prediction_mean"] = prediction_means
        df[f"{PROVENANCE_PREFIX}prediction_std"] = prediction_stds
        df[f"{PROVENANCE_PREFIX}actual_hash"] = actual_hashes
        df[f"{PROVENANCE_PREFIX}actual_shape"] = actual_shapes
        if "forecast_execution_path" in df.columns:
            df[f"{PROVENANCE_PREFIX}forecast_execution_path"] = df["forecast_execution_path"].astype(str)

        _write_record_file_exact(path, df)
        stamped.append(
            {
                "path": path,
                "rows": int(len(df)),
                "prediction_hashes": sorted({item for item in prediction_hashes if item}),
                "forecast_execution_paths": (
                    sorted(set(df["forecast_execution_path"].dropna().astype(str)))
                    if "forecast_execution_path" in df.columns
                    else []
                ),
            }
        )
    return stamped


def validate_result_artifact_provenance(
    log_paths: list[str],
    expected: dict[str, Any],
    *,
    require_prediction_hash: bool = False,
    require_batch_forecast: bool = False,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    expected_kind = str((expected or {}).get("candidate_kind") or "").strip().lower()
    if expected_kind == "variant":
        if not str((expected or {}).get("variant_path") or "").strip():
            failures.append("variant result expected provenance is missing variant_path")
        if not str((expected or {}).get("variant_source_sha256") or "").strip():
            failures.append("variant result expected provenance is missing variant_source_sha256")
        if not str((expected or {}).get("model_entry_hash") or "").strip():
            failures.append("variant result expected provenance is missing model_entry_hash")
    paths = _record_paths(log_paths)
    if not paths:
        failures.append("no result record files found")
    for path in paths:
        try:
            df = read_record_file(path)
        except Exception as exc:
            failures.append(f"{path}: cannot read record ({type(exc).__name__}: {exc})")
            continue
        if df.empty or "model_name" not in df.columns:
            continue
        missing = [f"{PROVENANCE_PREFIX}{key}" for key in expected if f"{PROVENANCE_PREFIX}{key}" not in df.columns]
        if missing:
            failures.append(f"{path}: missing provenance columns {missing}")
            continue
        for key, value in expected.items():
            col = f"{PROVENANCE_PREFIX}{key}"
            actual_values = {str(item) for item in df[col].dropna().astype(str)}
            expected_value = str(value)
            if expected_value == "" and not actual_values:
                continue
            if actual_values != {expected_value}:
                failures.append(f"{path}: {col} expected {expected_value!r}, got {sorted(actual_values)!r}")
        pred_col = f"{PROVENANCE_PREFIX}prediction_hash"
        prediction_hashes = sorted({str(item) for item in df.get(pred_col, pd.Series(dtype=str)).dropna().astype(str) if str(item)})
        if require_prediction_hash and not prediction_hashes:
            failures.append(f"{path}: missing prediction fingerprint")
        exec_col = f"{PROVENANCE_PREFIX}forecast_execution_path"
        execution_paths = sorted({str(item) for item in df.get(exec_col, pd.Series(dtype=str)).dropna().astype(str) if str(item)})
        if require_batch_forecast and "batch_forecast" not in execution_paths:
            failures.append(f"{path}: formal forecast path is not batch_forecast: {execution_paths}")
        records.append(
            {
                "path": path,
                "rows": int(len(df)),
                "prediction_hashes": prediction_hashes,
                "forecast_execution_paths": execution_paths,
            }
        )
    return {
        "status": "ok" if not failures and records else "failed",
        "records": records,
        "failures": failures or ([] if records else ["no TFB wide result records found"]),
    }
