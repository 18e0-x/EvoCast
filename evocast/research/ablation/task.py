from __future__ import annotations

import re
from typing import Any, Dict, List

from evocast.harness.permissions import normalize_repo_path


TARGET_KIND = "mechanism_ablation"

FORBIDDEN_INPUT_FIELDS = {
    "ablation_kind",
    "fallback_ablation_kind",
    "component_path",
    "canonical_component_path",
    "display_component_path",
    "minimal_counterfactual",
    "mechanism_family",
    "exact_edit_strategy",
    "target_files",
    "target_classes",
    "edit_anchors",
    "owner_class",
    "owner_source_file",
    "local_component_path",
}

FORBIDDEN_INTENT_TOKENS = {
    "dataset",
    "dataloader",
    "target column",
    "target_columns",
    "metric",
    "optimizer",
    "scheduler",
    "training policy",
}

FORBIDDEN_TASK_CHANGE_RE = re.compile(
    r"\b(change|set|modify|alter|replace)\s+(the\s+)?(horizon|pred_len|seq_len|target|metric|dataset)\b",
    re.IGNORECASE,
)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _string_list(values: Any) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for item in _as_list(values):
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _normalize_evidence_files(values: Any) -> List[str]:
    files: List[str] = []
    for path in _string_list(values):
        normalized = normalize_repo_path(path)
        if normalized and normalized not in files:
            files.append(normalized)
    return files


def _normalize_preserve_contract(value: Any) -> Dict[str, Any]:
    contract = dict(value) if isinstance(value, dict) else {}
    contract.setdefault("input", "preserve model input protocol")
    contract.setdefault("output", "preserve forecast output shape")
    contract.setdefault(
        "task",
        "do not change dataset, target columns, horizon, metric, optimizer, scheduler, or training policy",
    )
    return contract


def _intent_changes_forbidden_semantics(text: str) -> bool:
    lowered = str(text or "").lower()
    if any(token in lowered for token in FORBIDDEN_INTENT_TOKENS):
        return True
    return bool(FORBIDDEN_TASK_CHANGE_RE.search(str(text or "")))


def normalize_ablation_task(task: Dict[str, Any], *, evaluation_stage: str = "") -> Dict[str, Any]:
    raw = dict(task or {})
    normalized = dict(raw)
    normalized["target_kind"] = str(raw.get("target_kind") or "").strip()
    normalized["target_id"] = str(raw.get("target_id") or "").strip()
    normalized["mechanism_id"] = str(raw.get("mechanism_id") or "").strip()
    normalized["mechanism_name"] = str(raw.get("mechanism_name") or raw.get("name") or "").strip()
    normalized["diagnosis_question"] = str(raw.get("diagnosis_question") or raw.get("question") or "").strip()
    normalized["causal_variable"] = str(raw.get("causal_variable") or "").strip()
    raw_spec = dict(raw.get("edit_spec") or {}) if isinstance(raw.get("edit_spec"), dict) else {}
    evidence_files = _normalize_evidence_files(raw.get("evidence_files"))
    spec_target_raw = str(raw_spec.get("target_file") or "").strip()
    spec_target_file = normalize_repo_path(spec_target_raw) if spec_target_raw else ""
    if spec_target_file and spec_target_file not in evidence_files:
        evidence_files.insert(0, spec_target_file)
    normalized["evidence_files"] = evidence_files
    normalized["evidence_anchors"] = _string_list(raw.get("evidence_anchors"))
    normalized["runtime_paths"] = _string_list(raw.get("runtime_paths"))
    normalized["owner_symbols"] = _string_list(raw.get("owner_symbols"))
    normalized["operation_hints"] = _string_list(raw.get("operation_hints"))
    normalized["exact_edit_intent"] = str(raw.get("exact_edit_intent") or "").strip()
    normalized["edit_spec"] = raw_spec
    normalized["preserve_contract"] = _normalize_preserve_contract(raw.get("preserve_contract"))
    normalized["expected_behavior_delta"] = str(
        raw.get("expected_behavior_delta") or "same-input forecast should change if this mechanism is active"
    ).strip()
    normalized["risk"] = str(raw.get("risk") or "medium").strip() or "medium"
    normalized["evaluation_stage"] = str(raw.get("evaluation_stage") or evaluation_stage or "experiment").strip()
    if "granularity" not in normalized:
        normalized["granularity"] = "mechanism"
    return normalized


def validate_ablation_task(task: Dict[str, Any], *, evaluation_stage: str = "") -> Dict[str, Any]:
    normalized = normalize_ablation_task(task, evaluation_stage=evaluation_stage)
    errors: List[str] = []

    for field in sorted(FORBIDDEN_INPUT_FIELDS):
        if field in (task or {}):
            errors.append(f"forbidden_field:{field}")
    if not normalized["target_id"]:
        errors.append("missing_target_id")
    if normalized["target_kind"] != TARGET_KIND:
        errors.append(f"unsupported_target_kind:{normalized['target_kind'] or '<missing>'}")
    if not normalized["mechanism_id"]:
        errors.append("missing_mechanism_id")
    if not normalized["mechanism_name"]:
        errors.append("missing_mechanism_name")
    if not normalized["diagnosis_question"]:
        errors.append("missing_diagnosis_question")
    if not normalized["causal_variable"]:
        errors.append("missing_causal_variable")
    if not normalized["evidence_files"]:
        errors.append("missing_evidence_files")
    else:
        for path in normalized["evidence_files"]:
            if not (path.startswith("ts_benchmark/") and path.endswith(".py")):
                errors.append(f"invalid_evidence_file:{path}")
    if not normalized["evidence_anchors"]:
        errors.append("missing_evidence_anchors")
    if not normalized["exact_edit_intent"]:
        errors.append("missing_exact_edit_intent")
    elif _intent_changes_forbidden_semantics(normalized["exact_edit_intent"]):
        errors.append("exact_edit_intent_changes_task_or_training_semantics")
    edit_spec = dict(normalized.get("edit_spec") or {})
    if not str(edit_spec.get("target_file") or "").strip():
        errors.append("missing_edit_spec.target_file")
    if not str(edit_spec.get("anchor_text") or "").strip():
        errors.append("missing_edit_spec.anchor_text")
    if not (
        str(edit_spec.get("replacement_intent") or "").strip()
        or str(edit_spec.get("replacement_pseudocode") or "").strip()
        or normalized["exact_edit_intent"]
    ):
        errors.append("missing_ablation_intent")
    if not normalized["expected_behavior_delta"]:
        errors.append("missing_expected_behavior_delta")
    return {
        "status": "ok" if not errors else "error",
        "task": normalized,
        "errors": sorted(set(errors)),
    }
