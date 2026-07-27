from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from evocast.domain.config_paths import configs_dir, resolve_config_path
from evocast.domain.knowledge_paths import agent_base_dir, runtime_root, task_knowledge_dir
from evocast.state.domain_store import load_task_config


def _profiles_path(base_dir: str | None = None) -> Path:
    if base_dir is None:
        return resolve_config_path("policies/baseline_profiles.yaml", config_root=configs_dir())
    base = Path(base_dir)
    source_base = base.parent if base.name == ".evocast" else base
    candidates = [
        base / "configs" / "policies" / "baseline_profiles.yaml",
        source_base / "configs" / "policies" / "baseline_profiles.yaml",
        source_base / "evocast" / "configs" / "policies" / "baseline_profiles.yaml",
        agent_base_dir(str(source_base)) / "configs" / "policies" / "baseline_profiles.yaml",
        configs_dir() / "policies" / "baseline_profiles.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_baseline_profiles(base_dir: str | None = None) -> list[dict[str, Any]]:
    path = _profiles_path(base_dir)
    if not path.exists():
        return []
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    profiles = payload.get("profiles") or []
    if not isinstance(profiles, list):
        return []
    return [dict(item or {}) for item in profiles if isinstance(item, dict)]


def _task_context(base_dir: str, task_id: str) -> dict[str, Any]:
    runtime = runtime_root(base_dir)
    knowledge_dir = task_knowledge_dir(str(runtime), task_id)
    compiled_path = knowledge_dir / "compiled_config.json"
    task_config = load_task_config(str(runtime), task_id)
    compiled = {}
    if compiled_path.exists():
        try:
            compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
        except Exception:
            compiled = {}
    data_config = dict(compiled.get("data_config") or {})
    task_semantics = dict(task_config.get("task_semantics") or {})
    dataset_path = str(task_config.get("dataset_path") or data_config.get("dataset_path") or "").strip()
    dataset_basename = Path(dataset_path).name if dataset_path else ""
    return {
        "dataset_path": dataset_path,
        "dataset_basename": dataset_basename,
        "task_mode": str(task_semantics.get("task_mode") or task_config.get("task_mode") or "").strip(),
        "seq_len": task_config.get("seq_len"),
        "horizon": task_config.get("horizon"),
    }


def _normalized_model_key(model_key_or_name: str) -> str:
    text = str(model_key_or_name or "").strip()
    if not text:
        return ""
    return text.rsplit(".", 1)[-1]


def _matches(profile: dict[str, Any], context: dict[str, Any], model_key_or_name: str) -> bool:
    match = dict(profile.get("match") or {})
    if not match:
        return False
    if str(match.get("dataset_basename") or "").strip() != str(context.get("dataset_basename") or "").strip():
        return False
    if str(match.get("task_mode") or "").strip() != str(context.get("task_mode") or "").strip():
        return False
    if int(match.get("seq_len") or -1) != int(context.get("seq_len") or -2):
        return False
    if int(match.get("horizon") or -1) != int(context.get("horizon") or -2):
        return False
    return _normalized_model_key(match.get("model_key")) == _normalized_model_key(model_key_or_name)


def resolve_baseline_profile(
    *,
    base_dir: str,
    task_id: str,
    model_key_or_name: str,
) -> Dict[str, Any]:
    context = _task_context(base_dir, task_id)
    for profile in load_baseline_profiles(base_dir):
        if not _matches(profile, context, model_key_or_name):
            continue
        return {
            "profile_id": str(profile.get("profile_id") or ""),
            "description": str(profile.get("description") or ""),
            "model_hyper_params": dict(profile.get("model_hyper_params") or {}),
            "match": dict(profile.get("match") or {}),
        }
    return {}
