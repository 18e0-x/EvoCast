"""Code-grounded baseline facts compatibility layer.

Loads the generated code-card artifact set and exposes helpers for:
- mapping code facts to selection tags
- overriding registry tags/family/cost/risk conservatively
- computing coverage over code-grounded tags
"""

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Set

from evocast.research.model_registry import TAG_WEIGHT, get_all_tags


DEFAULT_FACTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "context", "code_cards")
)
DEFAULT_INDEX_PATH = os.path.join(DEFAULT_FACTS_DIR, "index.json")
LEGACY_FACTS_PATH = os.path.join(DEFAULT_FACTS_DIR, "baseline_code_facts.json")


def _candidate_fact_paths(path: Optional[str] = None) -> List[str]:
    candidates: List[str] = []
    if path:
        candidates.append(os.path.abspath(path))
    candidates.extend([
        DEFAULT_FACTS_DIR,
        DEFAULT_INDEX_PATH,
        os.path.abspath(os.path.join(os.getcwd(), "evocast", "context", "code_cards")),
        os.path.abspath(os.path.join(os.getcwd(), "..", "evocast", "context", "code_cards")),
        os.path.abspath(os.path.join(os.getcwd(), "evocast", "context", "code_cards", "index.json")),
        os.path.abspath(os.path.join(os.getcwd(), "..", "evocast", "context", "code_cards", "index.json")),
        LEGACY_FACTS_PATH,
        os.path.abspath(os.path.join(os.getcwd(), "evocast", "context", "code_cards", "baseline_code_facts.json")),
        os.path.abspath(os.path.join(os.getcwd(), "..", "evocast", "context", "code_cards", "baseline_code_facts.json")),
        os.path.abspath(os.path.join(os.getcwd(), "research_automation", "code_cards", "baseline_code_facts.json")),
        os.path.abspath(os.path.join(os.getcwd(), "..", "research_automation", "code_cards", "baseline_code_facts.json")),
        os.path.abspath(os.path.join(os.getcwd(), "result", "baseline_code_facts.json")),
        os.path.abspath(os.path.join(os.getcwd(), "..", "result", "baseline_code_facts.json")),
    ])
    seen = set()
    ordered: List[str] = []
    for item in candidates:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _canonical_backbone_tag(backbone: str) -> Optional[str]:
    mapping = {
        "transformer": "#transformer",
        "mlp": "#mlp",
        "cnn": "#cnn",
        "rnn": "#rnn",
        "ssm": "#ssm",
        "gnn": "#gnn",
        "linear": "#linear",
        "statistical": "#statistical",
        "mixed": None,
    }
    return mapping.get(backbone)


def _facts_to_tags(facts: Dict) -> List[str]:
    tags: Set[str] = set()

    backbone_tag = _canonical_backbone_tag(facts.get("backbone", ""))
    if backbone_tag:
        tags.add(backbone_tag)

    tokenization = facts.get("tokenization")
    if tokenization == "patch":
        tags.add("#patch")
    elif tokenization == "period_reshape":
        tags.add("#period_reshape")
    elif tokenization in {"pointwise", "adapter_defined"}:
        tags.add("#point_wise")
    elif tokenization == "variate_token":
        tags.update({"#point_wise", "#channel"})
    elif tokenization == "basis_transform":
        tags.add("#point_wise")

    if facts.get("explicit_attention"):
        tags.add("#attention")
    if facts.get("explicit_convolution"):
        tags.add("#convolution")
    if facts.get("explicit_decomposition"):
        tags.add("#decomposition")
    if facts.get("explicit_frequency_path"):
        tags.add("#frequency")
    if facts.get("explicit_cross_variable_modeling"):
        tags.add("#channel")
    if facts.get("explicit_recurrence_or_ssm"):
        tags.add("#state_space")

    mechanisms = set(facts.get("main_mechanisms", []))
    if "cross_variable" in mechanisms:
        tags.add("#channel")
    if "frequency" in mechanisms:
        tags.add("#frequency")
    if "decomposition" in mechanisms:
        tags.add("#decomposition")
    if "attention" in mechanisms:
        tags.add("#attention")
    if "convolution" in mechanisms:
        tags.add("#convolution")
    if "recurrence_or_ssm" in mechanisms:
        tags.add("#state_space")
    if "mixing" in mechanisms:
        tags.add("#mixing")
    if "pretrained" in mechanisms:
        tags.add("#pretrained")
    if "tokenization:basis_transform" in mechanisms:
        tags.add("#point_wise")
    if "tokenization:period_reshape" in mechanisms and "#patch" in tags:
        tags.add("#generative")

    all_tags = set(get_all_tags())
    return sorted(t for t in tags if t in all_tags)


def _load_index_bundle(index_path: str) -> Dict[str, Dict]:
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
    except Exception:
        return {}

    base_dir = os.path.dirname(index_path)
    result: Dict[str, Dict] = {}
    for item in index_data.get("models", []):
        model_key = item.get("model_key")
        rel_path = item.get("path")
        if not model_key or not rel_path:
            continue
        abs_path = os.path.abspath(os.path.join(base_dir, rel_path))
        if not os.path.exists(abs_path):
            continue
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                model_data = json.load(f)
        except Exception:
            continue
        if model_data.get("model_key") == model_key:
            result[model_key] = model_data
    return result


def _load_aggregate_bundle(path: str) -> Dict[str, Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    models = data.get("models", [])
    return {m["model_key"]: m for m in models if "model_key" in m}


@lru_cache(maxsize=1)
def load_baseline_facts(path: Optional[str] = None) -> Dict[str, Dict]:
    for target in _candidate_fact_paths(path):
        if not os.path.exists(target):
            continue
        if os.path.isdir(target):
            index_path = os.path.join(target, "index.json")
            if os.path.exists(index_path):
                bundle = _load_index_bundle(index_path)
                if bundle:
                    return bundle
            aggregate_path = os.path.join(target, "baseline_code_facts.json")
            if os.path.exists(aggregate_path):
                bundle = _load_aggregate_bundle(aggregate_path)
                if bundle:
                    return bundle
            continue
        if os.path.basename(target).lower() == "index.json":
            bundle = _load_index_bundle(target)
            if bundle:
                return bundle
            continue
        bundle = _load_aggregate_bundle(target)
        if bundle:
            return bundle
    return {}


def get_selection_tags(model_key: str, fallback_tags: Optional[List[str]] = None) -> List[str]:
    facts = load_baseline_facts().get(model_key)
    if not facts:
        return list(fallback_tags or [])
    mapped = _facts_to_tags(facts)
    if mapped:
        return mapped
    return list(fallback_tags or [])


def augment_registry_with_facts(registry: List[Dict]) -> List[Dict]:
    facts_map = load_baseline_facts()
    if not facts_map:
        return registry

    augmented: List[Dict] = []
    for spec in registry:
        spec = dict(spec)
        facts = facts_map.get(spec["model_key"])
        spec["declared_tags"] = list(spec.get("tags", []))
        spec["selection_tags"] = get_selection_tags(spec["model_key"], spec.get("tags", []))
        spec["tag_source"] = "facts" if facts else "registry"
        if facts:
            spec["facts_confidence"] = facts.get("confidence", "low")
            spec["facts_analysis_route"] = facts.get("analysis_route", "unknown")
            spec["facts_tokenization"] = facts.get("tokenization", "unknown")
            spec["facts_backbone"] = facts.get("backbone", "unknown")
            # Family is an explicit human-maintained taxonomy in
            # MODEL_FAMILIES. Code facts remain evidence/reporting metadata and
            # must not silently change automatic baseline selection.
        else:
            spec["facts_confidence"] = None
            spec["facts_analysis_route"] = None
            spec["facts_tokenization"] = None
            spec["facts_backbone"] = None
        augmented.append(spec)
    return augmented


def compute_selection_tag_coverage(model_keys: List[str], spec_map: Dict[str, Dict]) -> Dict:
    all_tags = set(get_all_tags())
    covered: Set[str] = set()
    for key in model_keys:
        covered.update(spec_map.get(key, {}).get("selection_tags", []))
    missing = all_tags - covered
    total_weight = sum(TAG_WEIGHT.get(tag, 1.0) for tag in all_tags)
    covered_weight = sum(TAG_WEIGHT.get(tag, 1.0) for tag in covered)
    return {
        "covered": sorted(covered),
        "missing": sorted(missing),
        "total_tags": len(all_tags),
        "covered_count": len(covered),
        "coverage_ratio": len(covered) / len(all_tags) if all_tags else 0.0,
        "weighted_coverage": covered_weight / total_weight if total_weight > 0 else 0.0,
    }
