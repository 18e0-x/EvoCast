"""Training policy compatibility wrapper.

The single source of truth is now ``configs/policies/experiment.yaml``.  This
module keeps the old function names, while historical budget strings are
normalized to the single unified experiment policy.
"""

from typing import Dict, Iterable, List, Optional, Set

from evocast.policy.experiment_policy import normalize_budget, training_guardrails, training_policy_for


DEFAULT_TRAINING_BLACKLIST = {
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
}

DEFAULT_TRAINING_ALLOWLIST = {"loss"}


def load_policy(budget: str = "unified", base_dir: Optional[str] = None) -> Dict:
    """Load training hyperparams for the unified policy.

    ``smoke`` and ``smoke_test`` return the mandatory epoch1/batch1 smoke
    policy; every other budget returns the single experiment policy.
    """
    text = str(budget or "").strip().lower()
    if text in {"smoke", "smoke_test"}:
        return training_policy_for("smoke_test", base_dir)
    normalize_budget(text, base_dir)
    return training_policy_for("experiment", base_dir)


def load_training_guardrails(base_dir: Optional[str] = None) -> Dict[str, Set[str]]:
    """Load central training-param ownership guardrails from the unified spec."""
    loaded = training_guardrails(base_dir)
    return {
        "training_blacklist": set(loaded.get("training_blacklist") or DEFAULT_TRAINING_BLACKLIST),
        "training_allowlist": set(loaded.get("training_allowlist") or DEFAULT_TRAINING_ALLOWLIST),
    }


def _normalize_key(key: str) -> str:
    return str(key).strip()


def _lower_map(values: Iterable[str]) -> Set[str]:
    return {_normalize_key(v).lower() for v in values}


def validate_hyper_param_changes(
    changes: Optional[Dict],
    base_model_hyper_params: Optional[Iterable[str]] = None,
    training_blacklist: Optional[Iterable[str]] = None,
    training_allowlist: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return policy violations for agent-proposed hyperparameter changes."""
    if not changes:
        return []

    base_keys = _lower_map(base_model_hyper_params or [])
    blacklist = _lower_map(training_blacklist or DEFAULT_TRAINING_BLACKLIST)
    allowlist = _lower_map(training_allowlist or DEFAULT_TRAINING_ALLOWLIST)

    violations: List[str] = []
    for raw_key in changes:
        key = _normalize_key(raw_key)
        key_l = key.lower()
        if key_l in blacklist and key_l not in allowlist:
            violations.append(f"training parameter '{key}' is centrally owned and cannot be changed")
        elif key_l in base_keys and key_l not in allowlist:
            violations.append(
                f"base model config parameter '{key}' cannot be overridden by the agent; "
                "architecture capacity or geometry changes must live in variant source code"
            )
    return violations


def filter_allowed_hyper_param_changes(
    changes: Optional[Dict],
    base_model_hyper_params: Optional[Iterable[str]] = None,
    training_blacklist: Optional[Iterable[str]] = None,
    training_allowlist: Optional[Iterable[str]] = None,
) -> Dict:
    """Drop disallowed hyperparameter changes while preserving allowed keys."""
    if not changes:
        return {}
    base_keys = _lower_map(base_model_hyper_params or [])
    blacklist = _lower_map(training_blacklist or DEFAULT_TRAINING_BLACKLIST)
    allowlist = _lower_map(training_allowlist or DEFAULT_TRAINING_ALLOWLIST)
    kept = {}
    for key, value in changes.items():
        key_l = _normalize_key(key).lower()
        if key_l in allowlist:
            kept[key] = value
        elif key_l in blacklist or key_l in base_keys:
            continue
        else:
            kept[key] = value
    return kept


def apply_policy(
    base_hp: Optional[Dict] = None,
    budget: str = "unified",
    base_dir: Optional[str] = None,
) -> Dict:
    """Merge training-policy HPs into a copy of base_hp.

    Policy values override base_hp keys (policy is the authority on
    budget-dependent settings like num_epochs, patience, batch_size, lr).
    """
    hp = dict(base_hp or {})
    policy = load_policy(budget, base_dir)
    hp.update(policy)
    return hp
