"""Command generator for evocast.

Generates valid TFB model_config entries from model registry specs.
Handles the mapping from registry spec to TFB pipeline model config format.
"""

import importlib
from typing import Any, Dict, List, Optional


_KNOWN_HPARAM_SOURCES = {
    "ts_benchmark.baselines.sparsetsf.SparseTSF": "ts_benchmark.baselines.sparsetsf.sparsetsf",
    "ts_benchmark.baselines.cmos.CMoS": "ts_benchmark.baselines.cmos.cmos",
    "ts_benchmark.baselines.patchmlp.PatchMLP": "ts_benchmark.baselines.patchmlp.patchmlp",
    "ts_benchmark.baselines.pathformer.Pathformer": "ts_benchmark.baselines.pathformer.pathformer",
    "ts_benchmark.baselines.moderntcn.ModernTCN": "ts_benchmark.baselines.moderntcn.moderntcn",
    "ts_benchmark.baselines.timefilter.TimeFilter": "ts_benchmark.baselines.timefilter.timefilter",
    "ts_benchmark.baselines.xpatch.xPatch": "ts_benchmark.baselines.xpatch.xpatch",
    "ts_benchmark.baselines.time_series_library.MambaSimple": "ts_benchmark.baselines.time_series_library.models.MambaSimple",
    "ts_benchmark.baselines.time_series_library.TimeMixer": "ts_benchmark.baselines.time_series_library.models.TimeMixer",
}


def _load_default_hyper_params(spec: Dict) -> Dict[str, Any]:
    defaults = dict(spec.get("default_hyper_params", {}) or {})
    if defaults:
        return defaults

    source_module = _KNOWN_HPARAM_SOURCES.get(spec.get("import_path", ""))
    if not source_module:
        import_path = str(spec.get("import_path") or "")
        if "." not in import_path:
            return {}
        try:
            module_name, attr = import_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, attr, None)
            source_module = getattr(cls, "__module__", module_name) if cls is not None else module_name
        except Exception:
            return {}
    try:
        module = importlib.import_module(source_module)
    except Exception:
        return {}
    loaded = getattr(module, "MODEL_HYPER_PARAMS", None)
    if not isinstance(loaded, dict):
        return {}
    return dict(loaded)


def generate_model_entry(
    spec: Dict,
    hyper_param_overrides: Optional[Dict] = None,
    adapter: Optional[str] = None,
) -> Dict:
    """Generate a single model_config entry for TFB's pipeline.

    Args:
        spec: Model registry spec dict.
        hyper_param_overrides: Override specific hyperparams.
        adapter: Override the adapter.

    Returns:
        Dict with "model_name", "adapter", "model_hyper_params".
    """
    entry = {
        "model_name": spec["import_path"],
        "model_hyper_params": _load_default_hyper_params(spec),
    }
    adapter_name = adapter if adapter is not None else spec.get("adapter")
    if adapter_name is not None:
        entry["adapter"] = adapter_name

    if hyper_param_overrides:
        entry["model_hyper_params"].update(hyper_param_overrides)

    return entry


def generate_variant_entry(
    variant_module: str,
    adapter: str = "transformer_adapter",
    hyper_params: Optional[Dict] = None,
    use_global_prefix: bool = True,
) -> Dict:
    """Generate a model_config entry for a variant.

    Workspace variants must be carried by variant_path.  The model_name remains
    the baseline import path used by TFB's adapter/config machinery.
    """
    model_name = str(variant_module or "").strip()
    if not model_name:
        raise ValueError("generate_variant_entry requires a baseline model_name")

    entry = {
        "model_name": model_name,
        "model_hyper_params": dict(hyper_params or {}),
    }
    if adapter is not None:
        entry["adapter"] = adapter
    return entry


def generate_model_config(
    entries: List[Dict],
    recommend_hyper_params: Optional[Dict] = None,
) -> Dict:
    """Generate a complete model_config for TFB's pipeline.

    Args:
        entries: List of model entry dicts.
        recommend_hyper_params: Globally recommended hyperparams.

    Returns:
        Full model_config dict.
    """
    return {
        "models": list(entries),
        "recommend_model_hyper_params": dict(recommend_hyper_params or {}),
    }


def spec_to_model_config(
    specs: List[Dict],
    budget: str = "unified",
    recommend_hyper_params: Optional[Dict] = None,
) -> Dict:
    """Convert a list of registry specs to a full model_config.

    Training hyperparams come from the unified training policy.
    """
    from evocast.policy.training_policy import apply_policy

    entries = []
    for spec in specs:
        hp = apply_policy(dict(spec.get("default_hyper_params", {})), budget=budget)
        entries.append(generate_model_entry(spec, hyper_param_overrides=hp))

    if recommend_hyper_params is None:
        recommend_hyper_params = {
            "input_chunk_length": 17,
            "output_chunk_length": 14,
            "norm": False,
        }

    return generate_model_config(entries, recommend_hyper_params)
