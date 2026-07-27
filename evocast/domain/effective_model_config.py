from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evocast.policy.experiment_policy import training_budget_for_task
from evocast.policy.baseline_profiles import resolve_baseline_profile
from evocast.policy.model_hparam_compat import apply_model_hparam_compatibility
from evocast.policy.training_policy import apply_policy
from evocast.research.model_registry import build_registry
from evocast.runners.tfb_pipeline_runner import _complete_runtime_shape_hparams


FORCE_EXPLICIT_HPARAM_KEYS = {
    "patch_len",
    "seq_len",
    "pred_len",
    "horizon",
    "d_model",
    "n_heads",
    "dropout",
}


@dataclass(frozen=True)
class EffectiveModelConfig:
    entry: dict[str, Any]
    explicit_model_hyper_params: dict[str, Any] = field(default_factory=dict)
    effective_model_hyper_params: dict[str, Any] = field(default_factory=dict)
    merged_model_hyper_params_before_policy: dict[str, Any] = field(default_factory=dict)
    policy_budget: str = "unified"
    compatibility_notes: list[str] = field(default_factory=list)


def registry_model_entry(model_key_or_name: str) -> dict[str, Any]:
    key = str(model_key_or_name or "").strip()
    if not key:
        return {}
    for spec in build_registry(verify=False):
        if key in {str(spec.get("model_key") or ""), str(spec.get("import_path") or ""), str(spec.get("model_name") or "")}:
            return {
                "model_key": str(spec.get("model_key") or key),
                "model_name": str(spec.get("import_path") or key),
                "adapter": spec.get("adapter"),
                "default_hyper_params": dict(spec.get("default_hyper_params") or {}),
                "source": str(spec.get("source") or ""),
            }
    return {}


def adapter_default_hparams(adapter: str | None) -> dict[str, Any]:
    if str(adapter or "") != "transformer_adapter":
        return {}
    from ts_benchmark.baselines.time_series_library.adapters_for_transformers import MODEL_HYPER_PARAMS

    return dict(MODEL_HYPER_PARAMS or {})


def _model_key(entry: dict[str, Any], registry_entry: dict[str, Any], fallback: str = "") -> str:
    return str(
        entry.get("model_key")
        or entry.get("source_model_key")
        or registry_entry.get("model_key")
        or entry.get("model_name")
        or fallback
        or ""
    ).rsplit(".", 1)[-1]


def _forced_explicit_hparams(explicit: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dict(explicit or {}).items() if key in FORCE_EXPLICIT_HPARAM_KEYS}


def resolve_effective_model_config(
    *,
    config_data: dict[str, Any],
    base_dir: str,
    task_id: str,
    model_entry: dict[str, Any],
    baseline_model_config: dict[str, Any] | None = None,
    explicit_model_hyper_params: dict[str, Any] | None = None,
    requested_budget: str = "unified",
    smoke: bool = False,
    apply_training_policy: bool = True,
) -> EffectiveModelConfig:
    entry = dict(model_entry or {})
    registry_entry = registry_model_entry(str(entry.get("model_key") or entry.get("model_name") or ""))
    baseline = dict(baseline_model_config or {})
    adapter = entry.get("adapter")
    if adapter is None:
        adapter = baseline.get("adapter") if baseline.get("adapter") is not None else registry_entry.get("adapter")
    model_name = str(entry.get("model_name") or baseline.get("model_name") or registry_entry.get("model_name") or "").strip()
    explicit = dict(explicit_model_hyper_params or entry.get("model_hyper_params") or {})

    merged = dict(registry_entry.get("default_hyper_params") or {})
    merged.update(adapter_default_hparams(str(adapter or "")))
    merged.update(dict((config_data.get("model_config") or {}).get("recommend_model_hyper_params") or {}))
    merged.update(dict(baseline.get("model_hyper_params") or {}))
    merged.update(explicit)
    baseline_profile = resolve_baseline_profile(
        base_dir=base_dir,
        task_id=task_id,
        model_key_or_name=_model_key(entry, registry_entry, model_name),
    )
    profile_hparams = dict(baseline_profile.get("model_hyper_params") or {})
    if profile_hparams:
        merged.update(profile_hparams)
    merged_before_policy = dict(merged)

    budget = training_budget_for_task(
        base_dir=base_dir,
        task_id=task_id,
        requested_budget=requested_budget,
        smoke=smoke,
    )
    effective = apply_policy(merged, budget=budget, base_dir=base_dir) if apply_training_policy else dict(merged)
    if profile_hparams:
        effective.update(profile_hparams)
    effective.update(_forced_explicit_hparams(explicit))
    effective = _complete_runtime_shape_hparams(config_data, effective)
    if profile_hparams:
        effective.update(profile_hparams)
    effective.update(_forced_explicit_hparams(explicit))
    effective, compatibility_notes = apply_model_hparam_compatibility(
        _model_key(entry, registry_entry, model_name),
        effective,
        config_data,
    )
    effective.update(_forced_explicit_hparams(explicit))

    resolved_entry = {
        **entry,
        "model_key": str(entry.get("model_key") or registry_entry.get("model_key") or "").strip(),
        "model_name": model_name,
        "adapter": adapter,
        "model_hyper_params": effective,
        "explicit_model_hyper_params": explicit,
        "effective_model_hyper_params": effective,
        "baseline_profile_id": baseline_profile.get("profile_id") if baseline_profile else "",
        "baseline_profile_description": baseline_profile.get("description") if baseline_profile else "",
    }
    if entry.get("variant_path"):
        resolved_entry["variant_path"] = entry["variant_path"]
    return EffectiveModelConfig(
        entry=resolved_entry,
        explicit_model_hyper_params=explicit,
        effective_model_hyper_params=effective,
        merged_model_hyper_params_before_policy=merged_before_policy,
        policy_budget=budget,
        compatibility_notes=list(compatibility_notes or []),
    )
