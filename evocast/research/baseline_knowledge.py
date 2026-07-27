from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from evocast.domain.atomic_io import atomic_write_json
from evocast.domain.evaluation_signature import build_evaluation_signature
from evocast.domain.knowledge_paths import (
    baseline_evaluation_dir,
    baseline_knowledge_registry_path,
    baseline_task_binding_path,
    dataset_task_binding_path,
    task_knowledge_dir,
)
from evocast.domain.result_provenance import model_entry_hash


SCHEMA_VERSION = "baseline_knowledge_v1"
CANDIDATE_SCHEMA_VERSION = "baseline_candidate_result_v1"
REFERENCE_SCHEMA_VERSION = "baseline_reference_result_v1"


def _now() -> str:
    return datetime.now().isoformat()


def _read_json(path: Path, default: Any) -> Any:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return payload if isinstance(payload, type(default)) else default


def _stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dataset_binding(base_dir: str, task_id: str) -> Dict[str, Any]:
    payload = _read_json(dataset_task_binding_path(base_dir, task_id), {})
    return payload if isinstance(payload, dict) else {}


def _task_signature(
    *,
    task_id: str,
    base_dir: str,
    config_data: Dict[str, Any],
    objective_metric: str,
) -> Dict[str, Any]:
    binding = _dataset_binding(base_dir, task_id)
    data_config = dict(config_data.get("data_config") or {})
    task_config = {
        "objective_metric": objective_metric,
        "task_semantics": dict(data_config.get("task_semantics") or {}),
    }
    return {
        "evaluation_signature": build_evaluation_signature(
            task_config=task_config,
            compiled_config=config_data,
            objective_metric=objective_metric,
        ),
        "dataset_binding": {
            "dataset_id": str(binding.get("dataset_id") or ""),
            "view_id": str(binding.get("view_id") or ""),
            "dataset_path": str(binding.get("dataset_path") or ""),
            "view_descriptor": dict(binding.get("view_descriptor") or {}),
        },
    }


def build_candidate_signature(
    *,
    task_id: str,
    base_dir: str,
    config_data: Dict[str, Any],
    model_key: str,
    model_entry: Dict[str, Any],
    objective_metric: str,
    budget: str,
    seed: int,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "baseline_candidate",
        **_task_signature(task_id=task_id, base_dir=base_dir, config_data=config_data, objective_metric=objective_metric),
        "model": {
            "model_key": str(model_key),
            "model_name": str(model_entry.get("model_name") or ""),
            "adapter": model_entry.get("adapter"),
            "model_entry_hash": model_entry_hash(dict(model_entry), variant_path=None),
            "model_entry": deepcopy(model_entry),
        },
        "training": {
            "budget": str(budget),
            "build_mode": False,
            "seed": int(seed),
        },
    }


def build_reference_signature(
    *,
    task_id: str,
    base_dir: str,
    config_data: Dict[str, Any],
    baseline_record: Dict[str, Any],
    objective_metric: str,
    seed_list: list[int],
) -> Dict[str, Any]:
    model_config = dict(baseline_record.get("model_config") or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "baseline_reference",
        **_task_signature(task_id=task_id, base_dir=base_dir, config_data=config_data, objective_metric=objective_metric),
        "model": {
            "model_key": str(baseline_record.get("model_name") or baseline_record.get("display_name") or ""),
            "model_name": str(model_config.get("model_name") or ""),
            "adapter": baseline_record.get("adapter") or model_config.get("adapter"),
            "model_entry_hash": model_entry_hash(model_config, variant_path=None),
            "model_entry": deepcopy(model_config),
        },
        "training": {
            "build_mode": False,
            "seed_list": [int(seed) for seed in seed_list],
        },
    }


def signature_hash(signature: Dict[str, Any]) -> str:
    return _stable_hash(signature)


def _candidate_path(base_dir: str, sig_hash: str) -> Path:
    return baseline_evaluation_dir(base_dir, sig_hash) / "baseline_candidate.json"


def _reference_path(base_dir: str, sig_hash: str) -> Path:
    return baseline_evaluation_dir(base_dir, sig_hash) / "seed_eval_reference.json"


def _task_baseline_reference_path(base_dir: str, task_id: str) -> Path:
    return task_knowledge_dir(base_dir, task_id) / "baseline_reference.json"


def _valid_baseline_reference(payload: Dict[str, Any], *, objective_metric: str) -> bool:
    stats = ((payload.get("metric_stats") or {}).get(objective_metric) or {})
    return bool(
        payload.get("schema_version") == "baseline_reference_v1"
        and payload.get("candidate_kind") == "baseline"
        and not payload.get("variant_path")
        and payload.get("source_clean") is True
        and int(stats.get("seed_count") or 0) >= 1
        and isinstance(stats.get("mean"), (int, float))
    )


def load_candidate_result(base_dir: str, signature: Dict[str, Any]) -> Dict[str, Any]:
    sig_hash = signature_hash(signature)
    payload = _read_json(_candidate_path(base_dir, sig_hash), {})
    if not payload:
        return {}
    if payload.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        return {}
    if payload.get("signature_hash") != sig_hash or payload.get("signature") != signature:
        return {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    if result.get("status") != "success" or not isinstance(result.get("metrics"), dict):
        return {}
    reused = deepcopy(result)
    reused["baseline_knowledge"] = {
        "reused": True,
        "signature_hash": sig_hash,
        "path": str(_candidate_path(base_dir, sig_hash)),
        "origin_task_id": payload.get("origin_task_id"),
        "origin_node_id": payload.get("origin_node_id"),
    }
    return reused


def write_candidate_result(
    *,
    base_dir: str,
    task_id: str,
    signature: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    if result.get("status") != "success" or not isinstance(result.get("metrics"), dict):
        return {}
    sig_hash = signature_hash(signature)
    path = _candidate_path(base_dir, sig_hash)
    payload = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "signature_hash": sig_hash,
        "signature": signature,
        "origin_task_id": task_id,
        "origin_node_id": result.get("node_id"),
        "path": str(path),
        "result": deepcopy(result),
        "created_at": _now(),
    }
    atomic_write_json(path, payload, ensure_ascii=False)
    _update_registry(base_dir, sig_hash, "baseline_candidate", path, task_id)
    return payload


def load_reference_result(base_dir: str, signature: Dict[str, Any], *, objective_metric: str) -> Dict[str, Any]:
    sig_hash = signature_hash(signature)
    payload = _read_json(_reference_path(base_dir, sig_hash), {})
    if not payload:
        return {}
    if payload.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        return {}
    if payload.get("signature_hash") != sig_hash or payload.get("signature") != signature:
        return {}
    reference = payload.get("baseline_reference") if isinstance(payload.get("baseline_reference"), dict) else {}
    stats = ((reference.get("metric_stats") or {}).get(objective_metric) or {})
    if reference.get("schema_version") != "baseline_reference_v1" or not isinstance(stats.get("mean"), (int, float)):
        return {}
    return deepcopy(payload)


def write_reference_result(
    *,
    base_dir: str,
    task_id: str,
    signature: Dict[str, Any],
    baseline_reference: Dict[str, Any],
) -> Dict[str, Any]:
    sig_hash = signature_hash(signature)
    payload = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "signature_hash": sig_hash,
        "signature": signature,
        "origin_task_id": task_id,
        "origin_reference_path": baseline_reference.get("path"),
        "baseline_reference": deepcopy(baseline_reference),
        "created_at": _now(),
    }
    path = _reference_path(base_dir, sig_hash)
    atomic_write_json(path, payload, ensure_ascii=False)
    _update_registry(base_dir, sig_hash, "baseline_reference", path, task_id)
    return payload


def bind_reference_to_task(
    *,
    base_dir: str,
    task_id: str,
    cached_payload: Dict[str, Any],
    objective_metric: str,
) -> Dict[str, Any]:
    reference = deepcopy(cached_payload.get("baseline_reference") or {})
    sig_hash = str(cached_payload.get("signature_hash") or "")
    path = _task_baseline_reference_path(base_dir, task_id)
    reference["task_id"] = task_id
    reference["path"] = str(path)
    reference["source"] = "baseline_knowledge_reuse"
    reference["origin_task_id"] = cached_payload.get("origin_task_id")
    reference["baseline_knowledge_path"] = str(_reference_path(base_dir, sig_hash))
    reference["signature_hash"] = sig_hash
    reference["reused_at"] = _now()
    atomic_write_json(path, reference, ensure_ascii=False)
    if not _valid_baseline_reference(reference, objective_metric=objective_metric):
        raise RuntimeError("BASELINE_KNOWLEDGE_REUSE_INVALID: reused baseline_reference did not validate after binding.")
    _write_task_binding(base_dir=base_dir, task_id=task_id, signature_hash=sig_hash, reference_path=path)
    return reference


def _write_task_binding(*, base_dir: str, task_id: str, signature_hash: str, reference_path: Path) -> None:
    payload = {
        "schema_version": "baseline_task_binding_v1",
        "task_id": task_id,
        "signature_hash": signature_hash,
        "baseline_reference_path": str(reference_path),
        "updated_at": _now(),
    }
    atomic_write_json(baseline_task_binding_path(base_dir, task_id), payload, ensure_ascii=False)


def _update_registry(base_dir: str, signature_hash_value: str, kind: str, path: Path, task_id: str) -> None:
    registry_path = baseline_knowledge_registry_path(base_dir)
    registry = _read_json(registry_path, {})
    records = dict(registry.get("evaluations") or {})
    record = dict(records.get(signature_hash_value) or {})
    kinds = dict(record.get("artifacts") or {})
    kinds[kind] = str(path)
    origins = list(record.get("origin_task_ids") or [])
    if task_id not in origins:
        origins.append(task_id)
    records[signature_hash_value] = {
        "signature_hash": signature_hash_value,
        "artifacts": kinds,
        "origin_task_ids": origins,
        "updated_at": _now(),
    }
    atomic_write_json(
        registry_path,
        {"schema_version": "baseline_knowledge_registry_v1", "updated_at": _now(), "evaluations": records},
        ensure_ascii=False,
    )
