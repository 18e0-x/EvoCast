from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from evocast.domain.config_paths import configs_dir, resolve_config_path
from evocast.domain.knowledge_paths import agent_base_dir

DEFAULT_AGENT_CONTROL_POLICY: Dict[str, Any] = {
    "max_consecutive_proposal_rejections": 3,
    "round_control": {
        "max_repair_attempts": 6,
        "repeated_failure_signature_limit": 3,
        "critical_variant_write_limit": 3,
    },
    "execution_timeout": {
        "metric_pipeline": 1500,
        "seed_eval_pipeline": 1500,
        "build_mode_baseline_pipeline": 900,
    },
    "runtime_validation": {
        "timeout_sec": 120,
        "cuda_launch_blocking": True,
    },
    "build_mode": {
        "force_smoke_experiment": True,
        "num_rollings": 1,
    },
    "gate": {
        "min_relative_improvement": 0.01,
        "min_seed_eval_relative_improvement": 0.005,
        "seed_eval_min_absolute_improvement": 0.0001,
    },
    "promotion": {
        "require_seed_eval_for_current_best": True,
        "require_seed_eval_for_baseline_update": True,
        "allow_control_fallback_promotion": False,
    },
    "context": {
        "strategy": "proposal_compact_builder_minimal",
    },
    "module_validity": {
        "mode": "audit_only",
    },
}

REQUIRED_GATE_KEYS = (
    "min_relative_improvement",
    "min_seed_eval_relative_improvement",
    "seed_eval_min_absolute_improvement",
)


def _policy_path(base_dir: Optional[str] = None) -> Path:
    if base_dir is None:
        return resolve_config_path("policies/agent_control.yaml", config_root=configs_dir())
    base = Path(base_dir)
    source_base = base.parent if base.name == ".evocast" else base
    candidates = [
        base / "configs" / "policies" / "agent_control.yaml",
        source_base / "configs" / "policies" / "agent_control.yaml",
        source_base / "evocast" / "configs" / "policies" / "agent_control.yaml",
        agent_base_dir(str(source_base)) / "configs" / "policies" / "agent_control.yaml",
        configs_dir() / "policies" / "agent_control.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def load_agent_control_policy(base_dir: Optional[str] = None) -> Dict[str, Any]:
    path = _policy_path(base_dir)
    if not path.exists():
        raise FileNotFoundError(f"agent control policy YAML not found: {path}")
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise RuntimeError(f"failed to load agent control policy YAML: {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"agent control policy YAML must contain a mapping: {path}")
    return _deep_merge(DEFAULT_AGENT_CONTROL_POLICY, loaded if isinstance(loaded, dict) else {})


def round_control_policy(base_dir: Optional[str] = None) -> Dict[str, Any]:
    return dict(load_agent_control_policy(base_dir).get("round_control") or {})


def research_mode_policy(base_dir: Optional[str] = None) -> Dict[str, Any]:
    return dict(load_agent_control_policy(base_dir).get("research_mode") or {})


def open_research_enabled(base_dir: Optional[str] = None) -> bool:
    policy = research_mode_policy(base_dir)
    return str(policy.get("mode") or "").strip().lower() == "open_research"


def protocol_policy(base_dir: Optional[str] = None) -> Dict[str, Any]:
    policy = load_agent_control_policy(base_dir)
    proposal_limit = policy.get("max_consecutive_proposal_rejections")
    return {"max_consecutive_rejections": proposal_limit}


def context_policy(base_dir: Optional[str] = None) -> Dict[str, Any]:
    return dict(load_agent_control_policy(base_dir).get("context") or {})


def module_validity_policy(base_dir: Optional[str] = None) -> Dict[str, Any]:
    return dict(load_agent_control_policy(base_dir).get("module_validity") or {})


def execution_timeout_policy(base_dir: Optional[str] = None) -> Dict[str, int]:
    policy = dict(load_agent_control_policy(base_dir).get("execution_timeout") or {})
    return {
        "metric_pipeline": positive_int(policy.get("metric_pipeline"), 600),
        "seed_eval_pipeline": positive_int(policy.get("seed_eval_pipeline"), 600),
        "build_mode_baseline_pipeline": positive_int(policy.get("build_mode_baseline_pipeline"), 900),
    }


def runtime_validation_policy(base_dir: Optional[str] = None) -> Dict[str, Any]:
    policy = dict(load_agent_control_policy(base_dir).get("runtime_validation") or {})
    return {
        "timeout_sec": positive_int(policy.get("timeout_sec"), 120),
        "cuda_launch_blocking": bool(policy.get("cuda_launch_blocking", True)),
    }


def build_mode_policy(base_dir: Optional[str] = None) -> Dict[str, Any]:
    policy = dict(load_agent_control_policy(base_dir).get("build_mode") or {})
    return {
        "force_smoke_experiment": bool(policy.get("force_smoke_experiment", True)),
        "num_rollings": positive_int(policy.get("num_rollings"), 1),
    }


def gate_policy(base_dir: Optional[str] = None) -> Dict[str, float]:
    gate = dict(load_agent_control_policy(base_dir).get("gate") or {})
    missing = [key for key in REQUIRED_GATE_KEYS if key not in gate]
    if missing:
        raise KeyError(f"agent_control.yaml missing required gate keys: {missing}")
    return {key: float(gate[key]) for key in REQUIRED_GATE_KEYS}


def promotion_policy(base_dir: Optional[str] = None) -> Dict[str, bool]:
    policy = dict(load_agent_control_policy(base_dir).get("promotion") or {})
    return {
        "require_seed_eval_for_current_best": bool(policy.get("require_seed_eval_for_current_best", True)),
        "require_seed_eval_for_baseline_update": bool(policy.get("require_seed_eval_for_baseline_update", True)),
        "allow_control_fallback_promotion": bool(policy.get("allow_control_fallback_promotion", False)),
    }


def research_repair_budget(base_dir: Optional[str] = None) -> int:
    return positive_int(round_control_policy(base_dir).get("max_repair_attempts"), 6, minimum=0)


def positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    return max(int(minimum), parsed)


def positive_float(value: Any, default: float, *, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    return max(float(minimum), parsed)
