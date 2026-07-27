from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from evocast.domain.config_paths import configs_dir, resolve_config_path
from evocast.domain.knowledge_paths import agent_base_dir, task_knowledge_dir
from evocast.state.domain_store import load_task_config


DEFAULT_POLICY: Dict[str, Any] = {
    "training": {
        "smoke_test": {
            "num_epochs": 1,
            "batch_size": 1,
            "max_train_batches": 1,
            "max_val_batches": 1,
            "patience": 1,
            "lr": 0.0001,
        },
        "experiment": {
            "num_epochs": 10,
            "batch_size": 32,
            "patience": 3,
            "lr": 0.0001,
        },
    },
    "baseline_search": {
        "candidate_count": 9,
        "registry_pool_size": 36,
        "initial_seeds": [],
        "preferred_families": ["linear", "transformer", "mlp", "cnn", "gnn", "others"],
    },
    "baseline_diagnosis": {
        "max_ablation_targets": 3,
        "ablation_repair_attempts": 3,
    },
    "seed_eval": {
        "seeds": [2021, 2022, 2023],
    },
    "guardrails": {
        "training_blacklist": [
            "lr",
            "learning_rate",
            "batch_size",
            "num_epochs",
            "epoch",
            "epochs",
            "patience",
            "optimizer",
            "scheduler",
            "warmup",
            "warmup_epochs",
            "warmup_steps",
            "weight_decay",
            "grad_clip",
            "gradient_clipping",
            "label_smoothing",
        ],
        "training_allowlist": ["loss"],
    },
}

UNIFIED_BUDGET = "unified"


def _policy_path(base_dir: Optional[str] = None) -> Path:
    if base_dir is None:
        return resolve_config_path("policies/experiment.yaml", config_root=configs_dir())
    base = Path(base_dir)
    source_base = base.parent if base.name == ".evocast" else base
    candidates = [
        base / "configs" / "policies" / "experiment.yaml",
        source_base / "configs" / "policies" / "experiment.yaml",
        source_base / "evocast" / "configs" / "policies" / "experiment.yaml",
        agent_base_dir(str(source_base)) / "configs" / "policies" / "experiment.yaml",
        configs_dir() / "policies" / "experiment.yaml",
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


def load_experiment_policy(base_dir: Optional[str] = None) -> Dict[str, Any]:
    path = _policy_path(base_dir)
    if not path.exists():
        return dict(DEFAULT_POLICY)
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        loaded = {}
    return _deep_merge(DEFAULT_POLICY, loaded if isinstance(loaded, dict) else {})


def normalize_budget(value: Any, base_dir: Optional[str] = None) -> str:
    del base_dir
    text = str(value or UNIFIED_BUDGET).strip().lower()
    if text in {"", UNIFIED_BUDGET}:
        return UNIFIED_BUDGET
    if text == "seed_eval":
        return "seed_eval"
    raise ValueError(f"Unsupported budget: {value}. Use 'unified' or 'seed_eval'.")


def training_policy_for(stage: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    policy = load_experiment_policy(base_dir)
    training = dict(policy.get("training") or {})
    key = "smoke_test" if str(stage or "").strip().lower() in {"smoke", "smoke_test"} else "experiment"
    return dict(training.get(key) or {})


def baseline_search_policy(base_dir: Optional[str] = None) -> Dict[str, Any]:
    raw = dict(load_experiment_policy(base_dir).get("baseline_search") or {})
    allowed = {"candidate_count", "registry_pool_size", "initial_seeds", "preferred_families"}
    return {key: value for key, value in raw.items() if key in allowed}


def baseline_diagnosis_policy(base_dir: Optional[str] = None) -> Dict[str, Any]:
    policy = dict(load_experiment_policy(base_dir).get("baseline_diagnosis") or {})
    if "ablation_effect_threshold" not in policy:
        raise KeyError("experiment.yaml missing required baseline_diagnosis.ablation_effect_threshold")
    return policy


def seed_eval_policy(base_dir: Optional[str] = None) -> Dict[str, Any]:
    return dict(load_experiment_policy(base_dir).get("seed_eval") or {})


def training_budget_for_task(
    *,
    base_dir: Optional[str],
    task_id: str,
    requested_budget: Any = "unified",
    smoke: bool = False,
) -> str:
    """Return the effective training budget for a task execution path.

    Build mode is a workflow-level fast validation mode.  Any training run
    inside that task must use the smoke_test policy so baseline, ablation,
    research, and seed-reference runs remain comparable.
    """
    if smoke or task_build_mode(base_dir, task_id):
        return "smoke_test"
    return normalize_budget(requested_budget, base_dir)


def training_guardrails(base_dir: Optional[str] = None) -> Dict[str, Set[str]]:
    guardrails = dict(load_experiment_policy(base_dir).get("guardrails") or {})
    return {
        "training_blacklist": {str(item) for item in list(guardrails.get("training_blacklist") or [])},
        "training_allowlist": {str(item) for item in list(guardrails.get("training_allowlist") or [])},
    }


def fixed_seed_list(base_dir: Optional[str] = None) -> List[int]:
    policy = seed_eval_policy(base_dir)
    explicit = policy.get("seeds")
    if isinstance(explicit, list) and explicit:
        seeds: List[int] = []
        for item in explicit:
            seeds.append(int(item))
        return seeds
    base_seed = int(policy.get("base_seed") or 2021)
    num_seeds = max(1, int(policy.get("num_seeds") or 3))
    return [base_seed + i for i in range(num_seeds)]


def baseline_seed(base_dir: Optional[str] = None) -> int:
    """Return the middle configured seed for the initial baseline run."""
    seeds = fixed_seed_list(base_dir)
    return seeds[len(seeds) // 2] if seeds else 2021


def task_build_mode(base_dir: Optional[str], task_id: str) -> bool:
    if not base_dir or not task_id:
        return False
    return bool(load_task_config(base_dir, task_id).get("build_mode"))
