"""Model registry for evocast.

Discovers importable TFB baseline models, merges manual overrides,
and provides searchable model specs.

Each model spec includes:
- model_key: short key name
- import_path: fully qualified import path
- adapter: which TFB adapter to use
- family: model family (transformer, linear, mlp, etc.)
- supports_univariate / supports_multivariate
- default_hyper_params: known good defaults
- budget_profile: historical metadata; training epochs now come from configs/policies/experiment.yaml
- fit_points: modifiable architectural components
"""

import importlib
import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from evocast.domain.config_paths import resolve_config_path
from evocast.domain.metric_semantics import DEFAULT_OBJECTIVE_METRIC

# Auto-discovered model name -> manually assigned family
# (inferred from architecture when possible)
MODEL_FAMILIES = {
    "Autoformer": "transformer",
    "Crossformer": "transformer",
    "DLinear": "linear",
    "ETSformer": "transformer",
    "FEDformer": "transformer",
    "FiLM": "others",
    "Informer": "transformer",
    "iTransformer": "transformer",
    "Koopa": "others",
    "LightTS": "mlp",
    "Linear": "linear",
    "MICN": "cnn",
    "NLinear": "linear",
    "Nonstationary_Transformer": "transformer",
    "PatchTST": "transformer",
    "Pyraformer": "transformer",
    "Reformer": "transformer",
    "TimesNet": "cnn",
    "Transformer": "transformer",
    "Triformer": "transformer",
    "TimeMixer": "mlp",
    "FreTS": "mlp",
    "SegRNN": "others",
    "TemporalFusionTransformer": "transformer",
    "TiDE": "mlp",
    "TSMixer": "mlp",
    "SCINet": "cnn",
    "MultiPatchFormer": "transformer",
    "PAttn": "transformer",
    "KANAD": "others",
    "WPMixer": "mlp",
    "MSGNet": "gnn",
    "MambaSimple": "others",
    "TimeFilter": "gnn",
    "Amplifier": "others",
    "CMoS": "linear",
    "CrossLinear": "linear",
    "DTAF": "transformer",
    "DUET": "transformer",
    "FITS": "others",
    "HDMixer": "mlp",
    "ModernTCN": "cnn",
    "PatchMLP": "mlp",
    "Pathformer": "transformer",
    "PDF": "transformer",
    "VAR_model": "others",
    "SparseTSF": "linear",
    "SRSNet": "others",
    "TimeBase": "linear",
    "TimeBridge": "transformer",
    "TimeKAN": "others",
    "TimePerceiver": "transformer",
    "xPatch": "others",
}

# Exact or conceptual duplicates that should not be part of the canonical
# tournament registry. Keep their source code available, but do not select them.
TSL_EXCLUDED_MODELS = {
    # Prefer the standalone forecasting adapter because it preserves TimeFilter's
    # extra MoE loss in DeepForecastingModelBase._process.
    "TimeFilter",
}

STANDALONE_FORECASTING_MODELS = {
    "Amplifier": "ts_benchmark.baselines.amplifier.Amplifier",
    "CMoS": "ts_benchmark.baselines.cmos.CMoS",
    "CrossLinear": "ts_benchmark.baselines.crosslinear.CrossLinear",
    "DTAF": "ts_benchmark.baselines.dtaf.DTAF",
    "DUET": "ts_benchmark.baselines.duet.DUET",
    "FITS": "ts_benchmark.baselines.fits.FITS",
    "HDMixer": "ts_benchmark.baselines.hdmixer.HDMixer",
    "ModernTCN": "ts_benchmark.baselines.moderntcn.ModernTCN",
    "PatchMLP": "ts_benchmark.baselines.patchmlp.PatchMLP",
    "Pathformer": "ts_benchmark.baselines.pathformer.Pathformer",
    "PDF": "ts_benchmark.baselines.pdf.PDF",
    "SparseTSF": "ts_benchmark.baselines.sparsetsf.SparseTSF",
    "SRSNet": "ts_benchmark.baselines.srsnet.SRSNet",
    "TimeBase": "ts_benchmark.baselines.timebase.TimeBase",
    "TimeBridge": "ts_benchmark.baselines.timebridge.TimeBridge",
    "TimeFilter": "ts_benchmark.baselines.timefilter.TimeFilter",
    "TimeKAN": "ts_benchmark.baselines.timekan.TimeKAN",
    "TimePerceiver": "ts_benchmark.baselines.timeperceiver.TimePerceiver",
    "xPatch": "ts_benchmark.baselines.xpatch.xPatch",
}

# Model architecture defaults. Training HPs (batch_size, lr, num_epochs,
# patience, smoke, seed-eval seeds) live in configs/policies/experiment.yaml.
DEFAULT_HYPER_PARAMS = {
    "num_workers": 0,
    "loss": "MSE",
}

# Fit points for each model family.
FAMILY_FIT_POINTS = {
    "transformer": [
        "temporal_embedding", "attention_mechanism", "feedforward",
        "normalization", "output_head", "encoder_architecture",
        "loss_function",
    ],
    "linear": [
        "normalization", "output_head", "feedforward",
        "loss_function",
    ],
    "mlp": [
        "normalization", "output_head", "feedforward",
        "loss_function",
    ],
    "cnn": [
        "temporal_embedding", "feedforward", "normalization",
        "output_head", "loss_function",
    ],
    "rnn": [
        "temporal_embedding", "feedforward", "normalization",
        "output_head", "loss_function",
    ],
    "gnn": [
        "variable_embedding", "feedforward", "normalization",
        "output_head", "loss_function",
    ],
    "ssm": [
        "temporal_embedding", "feedforward", "normalization",
        "output_head", "loss_function",
    ],
    "others": [
        "temporal_embedding", "feedforward", "normalization",
        "output_head", "loss_function",
    ],
}

# ── Tag system for beam-search-driven model selection ──────────────────
# Each model carries 2-5 tags drawn from the five categories below.
# Tags are NOT mutually exclusive — a model can be both #transformer and
# #frequency.  The tag set is used by downstream seed-selection and
# beam-expansion logic to measure architectural similarity between models.

TAG_DEFINITIONS = {
    "backbone": {
        "#transformer": "Transformer / self-attention backbone",
        "#mlp": "Multilayer perceptron backbone",
        "#cnn": "Convolutional neural network backbone",
        "#rnn": "Recurrent neural network backbone",
        "#ssm": "State-space model backbone (Mamba / S4 family)",
        "#gnn": "Graph neural network backbone",
        "#linear": "Single or stacked linear-layer backbone",
        "#statistical": "Classical statistical estimator",
    },
    "input_processing": {
        "#patch": "Sequence → overlapping patch tokens (unfold / conv1d embedding)",
        "#period_reshape": "1D sequence → 2D matrix by period folding",
        "#point_wise": "Raw time-step tokens, no patching or reshaping",
    },
    "feature_space": {
        "#frequency": "Operates in frequency / spectral domain (FFT, wavelet)",
        "#channel": "Explicit cross-variable / cross-channel modelling",
        "#decomposition": "Decomposes signal into trend / season / residual branches",
    },
    "modelling_strategy": {
        "#attention": "Self-attention or cross-attention mechanism",
        "#mixing": "Token / feature mixing (MLP-Mixer style)",
        "#convolution": "Convolution-based local-pattern extraction",
        "#state_space": "Recurrent dynamics or state-space evolution",
        "#koopman": "Koopman operator theory (dynamics in lifted linear space)",
    },
    "special_paradigm": {
        "#generative": "Probabilistic / generative formulation (diffusion, VAE-like)",
        "#pretrained": "Pre-training + fine-tuning paradigm (foundation model)",
    },
}

# Forecasting-model architecture tag catalog for runnable local-code models.
MODEL_TAGS: Dict[str, List[str]] = {
    # ── 1. Linear / Simple ──────────────────────────────────────────
    "DLinear":       ["#linear", "#decomposition", "#point_wise"],
    "NLinear":       ["#linear", "#point_wise"],
    "Linear":        ["#linear", "#point_wise"],
    "CrossLinear":   ["#linear", "#channel", "#point_wise"],
    "SparseTSF":     ["#linear", "#period_reshape"],

    # ── 2. Patch / Tokenisation ─────────────────────────────────────
    "PatchTST":      ["#transformer", "#patch", "#attention"],
    "PatchMLP":      ["#mlp", "#patch", "#mixing"],
    "xPatch":        ["#mlp", "#cnn", "#patch", "#mixing"],
    "MultiPatchFormer": ["#transformer", "#patch", "#attention"],
    "PAttn":         ["#transformer", "#patch", "#attention"],
    "PDF":           ["#transformer", "#period_reshape", "#patch", "#generative"],

    # ── 3. Frequency / Spectral ─────────────────────────────────────
    "FITS":          ["#linear", "#frequency", "#point_wise"],
    "FreTS":         ["#mlp", "#frequency", "#point_wise"],
    "FEDformer":     ["#transformer", "#frequency", "#attention"],
    "FiLM":          ["#frequency", "#state_space", "#mixing"],
    "WPMixer":       ["#mlp", "#frequency", "#mixing"],
    "TimeFilter":    ["#gnn", "#channel"],

    # ── 4. Channel / Cross-variable ─────────────────────────────────
    "iTransformer":  ["#transformer", "#channel", "#attention"],
    "Crossformer":   ["#transformer", "#patch", "#channel", "#attention"],
    "DUET":          ["#transformer", "#channel", "#attention"],
    "MSGNet":        ["#gnn", "#channel", "#decomposition"],
    "CMoS":          ["#linear", "#cnn", "#channel"],

    # ── 5. General Transformer / Attention ──────────────────────────
    "Autoformer":    ["#transformer", "#decomposition", "#attention"],
    "Transformer":   ["#transformer", "#attention", "#point_wise"],
    "ETSformer":     ["#transformer", "#decomposition", "#attention"],
    "Informer":      ["#transformer", "#attention", "#point_wise"],
    "Nonstationary_Transformer": ["#transformer", "#attention", "#point_wise"],
    "Reformer":      ["#transformer", "#attention", "#point_wise"],
    "Pyraformer":    ["#transformer", "#attention", "#point_wise"],
    "Triformer":     ["#transformer", "#attention", "#point_wise"],
    "TemporalFusionTransformer": ["#transformer", "#attention", "#point_wise"],
    "Pathformer":    ["#transformer", "#patch", "#attention"],
    "DTAF":          ["#transformer", "#attention", "#point_wise"],
    "TimeBridge":    ["#transformer", "#attention", "#point_wise"],
    "TimePerceiver": ["#transformer", "#attention", "#point_wise"],

    # ── 6. CNN / Local Pattern ──────────────────────────────────────
    "TimesNet":      ["#cnn", "#period_reshape", "#convolution"],
    "ModernTCN":     ["#cnn", "#convolution", "#point_wise"],
    "MICN":          ["#cnn", "#convolution", "#point_wise"],
    "SCINet":        ["#cnn", "#convolution", "#point_wise"],

    # ── 7. MLP / Mixer ──────────────────────────────────────────────
    "TSMixer":       ["#mlp", "#mixing", "#point_wise"],
    "TiDE":          ["#mlp", "#mixing", "#point_wise"],
    "LightTS":       ["#mlp", "#mixing", "#point_wise"],
    "TimeMixer":     ["#mlp", "#mixing", "#point_wise"],
    "KANAD":         ["#mlp", "#mixing", "#point_wise"],
    "Amplifier":     ["#frequency", "#decomposition", "#mixing"],
    "HDMixer":       ["#mlp", "#decomposition", "#mixing"],
    "TimeKAN":       ["#frequency", "#mlp", "#mixing"],

    # ── 8. Dynamics / RNN / SSM / Koopman ───────────────────────────
    "SegRNN":        ["#rnn", "#state_space"],
    "MambaSimple":   ["#ssm", "#state_space", "#point_wise"],
    "SRSNet":        ["#patch", "#mlp", "#mixing"],
    "Koopa":         ["#koopman", "#frequency", "#decomposition"],

    # ── 9. Statistical / Classical ──────────────────────────────────
    "VAR_model":               ["#statistical", "#channel", "#point_wise"],

    # ── 10. Foundation / Pretraining ────────────────────────────────
    "TimeBase":      ["#linear", "#point_wise"],
}


def _check_import(path: str) -> bool:
    """Check if a Python module can be imported."""
    try:
        importlib.import_module(path)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _check_class(path: str) -> bool:
    """Verify that both the module AND the class/attribute exist.

    The path format is ``package.module.ClassName``.
    Returns True only if the module imports AND getattr(mod, class_name) succeeds.
    """
    parts = path.rsplit(".", 1)
    if len(parts) != 2:
        return False
    pkg_path, class_name = parts
    try:
        mod = importlib.import_module(pkg_path)
        target = getattr(mod, class_name)
        return target is not None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


def _make_spec(
    model_key: str,
    import_path: str,
    adapter: Optional[str],
    source: str,
    local_code: bool,
) -> Dict:
    family = MODEL_FAMILIES.get(model_key, "unknown")
    fit_points = FAMILY_FIT_POINTS.get(family, ["normalization", "output_head"])
    tags = MODEL_TAGS.get(model_key, [])
    spec = {
        "model_key": model_key,
        "import_path": import_path,
        "adapter": adapter,
        "family": family,
        "tags": tags,
        "supports_univariate": True,
        "supports_multivariate": True,
        "default_hyper_params": dict(DEFAULT_HYPER_PARAMS) if adapter else {},
        "required_hyper_params": ["seq_len", "horizon"] if adapter else [],
        "budget_profile": {
            "unified_epochs": 10,
        },
        "fit_points": fit_points if local_code else [],
        "source": source,
        "local_code": local_code,
        "canonical": True,
        "verified_import": False,
    }
    return spec


def _discover_time_series_library_models() -> List[Dict]:
    """Auto-discover models from ts_benchmark.baselines.time_series_library.

    Reads the __all__ list and __init__.py to find all exported models.
    Returns list of model spec dicts.
    """
    models = []
    try:
        from ts_benchmark.baselines.time_series_library import __all__ as tsl_models
    except ImportError:
        init_path = os.path.join(
            os.getcwd(),
            "ts_benchmark",
            "baselines",
            "time_series_library",
            "__init__.py",
        )
        try:
            with open(init_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            tsl_models = []
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        tsl_models = ast.literal_eval(node.value)
                        break
        except Exception:
            return models

    for model_name in tsl_models:
        if model_name in TSL_EXCLUDED_MODELS:
            continue
        import_path = f"ts_benchmark.baselines.time_series_library.{model_name}"
        models.append(_make_spec(
            model_key=model_name,
            import_path=import_path,
            adapter="transformer_adapter",
            source="time_series_library",
            local_code=True,
        ))

    return models


def _discover_standalone_forecasting_models() -> List[Dict]:
    """Return canonical standalone forecasting baselines.

    These classes already implement TFB's ModelBase protocol, so they must not
    be wrapped by transformer_adapter.
    """
    return [
        _make_spec(
            model_key=model_key,
            import_path=import_path,
            adapter=None,
            source="standalone",
            local_code=True,
        )
        for model_key, import_path in STANDALONE_FORECASTING_MODELS.items()
    ]


def verify_model_registry(models: List[Dict]) -> List[Dict]:
    """Verify that each model in the registry can be imported and its class exists."""
    for spec in models:
        import_path = spec["import_path"]
        spec["verified_import"] = _check_class(import_path)
    return models


def _default_override_path() -> str:
    return str(resolve_config_path("registry/model_overrides.yaml"))


def load_override_config(path: Optional[str] = None) -> Dict:
    """Load manual model registry overrides from a YAML/JSON config."""
    if path is None:
        path = _default_override_path()
    else:
        raw_path = Path(path)
        path = str(raw_path if raw_path.exists() else resolve_config_path(path))
    if not os.path.exists(path):
        return {}

    ext = os.path.splitext(path)[1]
    with open(path, "r", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            try:
                import yaml
                return yaml.safe_load(f) or {}
            except ImportError:
                return {}
        else:
            return json.load(f)


def _normalize_override_models(raw_models: Any) -> List[Dict[str, Any]]:
    """Normalize keyed YAML mapping overrides."""
    if isinstance(raw_models, dict):
        normalized = []
        for model_key, payload in raw_models.items():
            if payload is None:
                payload = {}
            if not isinstance(payload, dict):
                continue
            item = dict(payload)
            item.setdefault("model_key", str(model_key))
            normalized.append(item)
        return normalized
    if isinstance(raw_models, list):
        raise ValueError("registry/model_overrides.yaml models must be a mapping keyed by model_key")
    return []


def build_registry(
    overrides_path: Optional[str] = None,
    verify: bool = True,
) -> List[Dict]:
    """Build the full model registry.

    1. Auto-discover canonical local TSL models.
    2. Add canonical standalone forecasting baselines.
    3. Apply manual overrides from config.
    4. Keep only local-code models that can support the research/edit loop.
    5. Verify importability (optional).

    Returns:
        List of model spec dicts.
    """
    models = []
    models.extend(_discover_time_series_library_models())
    models.extend(_discover_standalone_forecasting_models())

    # Apply overrides
    overrides = load_override_config(overrides_path)
    override_models = _normalize_override_models(overrides.get("models", {}))
    override_map = {m["model_key"]: m for m in override_models}

    for spec in models:
        key = spec["model_key"]
        if key in override_map:
            spec.update(override_map[key])

    # Add any extra models from overrides not in auto-discovered
    existing_keys = {m["model_key"] for m in models}
    for m in override_models:
        if m["model_key"] not in existing_keys:
            models.append(m)

    # Drop explicit exclusions from config after overrides have been applied.
    exclude_models = set(overrides.get("exclude_models", []))
    if exclude_models:
        models = [m for m in models if m["model_key"] not in exclude_models]

    models = [m for m in models if bool(m.get("local_code"))]

    if verify:
        models = verify_model_registry(models)

    return models


def get_verified_models(registry: List[Dict]) -> List[Dict]:
    """Filter registry to only verified-importable models."""
    return [m for m in registry if m.get("verified_import", False)]


def get_models_by_family(registry: List[Dict], family: str) -> List[Dict]:
    """Filter registry by model family."""
    return [m for m in registry if m.get("family") == family]


def get_models_by_tag(registry: List[Dict], tag: str) -> List[Dict]:
    """Filter registry to models carrying a specific tag."""
    return [m for m in registry if tag in m.get("tags", [])]


def get_model_tags(model_key: str) -> List[str]:
    """Return the tag list for a model key, or empty list if unknown."""
    return MODEL_TAGS.get(model_key, [])


def get_all_tags() -> List[str]:
    """Return every tag defined across all five categories (flat list)."""
    tags: List[str] = []
    for category in TAG_DEFINITIONS.values():
        tags.extend(category.keys())
    return tags


def compute_tag_coverage(model_keys: List[str]) -> Dict:
    """Compute how many and which tags a set of model keys covers.

    Returns a dict with:
      - covered: set of tags that appear at least once
      - missing: set of all defined tags minus covered
      - coverage_ratio: float in [0, 1]
    """
    all_tags = set(get_all_tags())
    covered: Set[str] = set()
    for key in model_keys:
        covered.update(MODEL_TAGS.get(key, []))
    missing = all_tags - covered
    return {
        "covered": sorted(covered),
        "missing": sorted(missing),
        "total_tags": len(all_tags),
        "covered_count": len(covered),
        "coverage_ratio": len(covered) / len(all_tags) if all_tags else 0.0,
    }


# Tag weights for weighted Jaccard similarity.
# Higher weight → more informative / rarer tag.
# Tier 1 (0.3): trivial default, no signal — #point_wise
# Tier 2 (0.5): common backbone — #transformer, #attention, #mlp
# Tier 3 (1.5-2.0): moderately rare — #cnn, #rnn, #linear, #convolution, #mixing,
#                    #statistical, #frequency, #patch, #channel, #decomposition,
#                    #state_space
# Tier 4 (3.0): rare high-signal — #generative, #pretrained, #koopman,
#                #period_reshape, #gnn, #ssm
TAG_WEIGHT: Dict[str, float] = {
    "#point_wise": 0.3,
    "#transformer": 0.5, "#attention": 0.5, "#mlp": 0.5,
    "#cnn": 1.5, "#rnn": 1.5, "#linear": 1.5, "#convolution": 1.5,
    "#mixing": 1.5, "#statistical": 1.5,
    "#frequency": 2.0, "#patch": 2.0, "#channel": 2.0, "#decomposition": 2.0,
    "#state_space": 2.0,
    "#generative": 3.0, "#pretrained": 3.0, "#koopman": 3.0,
    "#period_reshape": 3.0, "#gnn": 3.0, "#ssm": 3.0,
}


def weighted_jaccard(tags_a: List[str], tags_b: List[str]) -> float:
    """Weighted Jaccard similarity between two tag sets.

    Each tag contributes its weight from TAG_WEIGHT.  High-frequency /
    low-information tags (#point_wise, #transformer) are down-weighted;
    rare high-signal tags (#koopman, #generative) dominate the numerator.

    Returns a float in [0, 1].
    """
    set_a = set(tags_a)
    set_b = set(tags_b)
    if not set_a and not set_b:
        return 1.0

    intersection = set_a & set_b
    union = set_a | set_b

    w_inter = sum(TAG_WEIGHT.get(t, 1.0) for t in intersection)
    w_union = sum(TAG_WEIGHT.get(t, 1.0) for t in union)

    if w_union == 0:
        return 0.0
    return w_inter / w_union


def jaccard_similarity(tags_a: List[str], tags_b: List[str]) -> float:
    """Unweighted Jaccard coefficient between two tag sets.

    0 = totally different, 1 = identical.
    """
    set_a, set_b = set(tags_a), set(tags_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_tag_scores(
    run_results: List[Dict],
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC,
    shrinkage_k: float = 2.0,
    model_tag_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict]:
    """Compute shrinkage-estimated tag scores from tournament run results.

    For each tag, computes:
      - observed: 1 − median_rank(models_with_tag) / total_models
      - n_models: number of successful models carrying this tag
      - shrinkage_lambda: n / (n + k)
      - score: λ × observed + (1 − λ) × 0.5

    Low-sample tags are pulled toward the uninformatable prior (0.5).

    Args:
        run_results: List of dicts, each with keys:
                     "model_key", "objective_value", "status", "tier"
        objective_metric: Metric name (for reference, not used in computation).
        shrinkage_k: Regularisation strength (default 2.0).

    Returns:
        Dict mapping tag → {score, n_models, observed, lambda, models}.
    """
    # Only use successful results with a valid objective_value
    valid = [
        r for r in run_results
        if r.get("status") == "success" and r.get("objective_value") is not None
    ]
    if not valid:
        return {}

    # Sort by objective value ascending (lower is better)
    valid_sorted = sorted(valid, key=lambda r: r["objective_value"])
    total = len(valid_sorted)

    # Assign ranks (1 = best)
    rank_map = {r["model_key"]: idx + 1 for idx, r in enumerate(valid_sorted)}

    # Collect all tags and the models that carry them
    if model_tag_map is None:
        model_tag_map = MODEL_TAGS
    tag_models: Dict[str, List[str]] = {}
    for r in valid:
        for tag in model_tag_map.get(r["model_key"], []):
            tag_models.setdefault(tag, []).append(r["model_key"])

    scores = {}
    for tag, models in tag_models.items():
        n = len(models)
        ranks = [rank_map[m] for m in models if m in rank_map]
        if not ranks:
            continue

        sorted_ranks = sorted(ranks)
        mid = len(sorted_ranks) // 2
        if len(sorted_ranks) % 2 == 0:
            median_rank = (sorted_ranks[mid - 1] + sorted_ranks[mid]) / 2.0
        else:
            median_rank = float(sorted_ranks[mid])

        observed = 1.0 - median_rank / total
        lam = n / (n + shrinkage_k)
        score = lam * observed + (1.0 - lam) * 0.5

        scores[tag] = {
            "score": round(score, 4),
            "n_models": n,
            "observed": round(observed, 4),
            "lambda": round(lam, 4),
            "models": sorted(models),
        }

    return scores


def snapshot_registry(registry: List[Dict], output_path: str) -> str:
    """Write a registry snapshot to JSON. Returns the path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_models": len(registry),
            "verified_models": len(get_verified_models(registry)),
            "models": registry,
        }, f, indent=2, ensure_ascii=False, default=str)
    return output_path
