from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from evocast.research.ablation.task import validate_ablation_task
from evocast.policy.experiment_policy import baseline_diagnosis_policy
from evocast.research.mechanism.evidence_graph import (
    build_mechanism_evidence_graph,
    resolve_source_anchor,
    source_contains_anchor,
)
from evocast.harness.session import AgentSession


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _string_list(value: Any) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for item in _as_list(value):
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _is_executable_repo_python_path(path: str) -> bool:
    text = str(path or "").replace("\\", "/").strip()
    return bool(text.startswith("ts_benchmark/") and text.endswith(".py"))


def _evidence_files(mechanism: Dict[str, Any], question: Dict[str, Any]) -> List[str]:
    files: List[str] = []
    spec = dict(question.get("edit_spec") or {})
    for path in _string_list(spec.get("target_file")):
        if _is_executable_repo_python_path(path) and path not in files:
            files.append(path)
    for item in _as_list(mechanism.get("evidence")):
        if isinstance(item, dict):
            for path in _string_list(item.get("file") or item.get("path")):
                if _is_executable_repo_python_path(path) and path not in files:
                    files.append(path)
    for path in _string_list(mechanism.get("resolved_evidence_files")):
        if _is_executable_repo_python_path(path) and path not in files:
            files.append(path)
    return files


def _evidence_anchors(mechanism: Dict[str, Any], question: Dict[str, Any]) -> List[str]:
    anchors: List[str] = []
    spec = dict(question.get("edit_spec") or {})
    for anchor in _string_list(spec.get("anchor_text")):
        if anchor not in anchors:
            anchors.append(anchor)
    for item in _as_list(mechanism.get("evidence")):
        if isinstance(item, dict):
            candidates = item.get("anchors") or item.get("anchor") or item.get("code") or item.get("snippet")
        else:
            candidates = item
        for anchor in _string_list(candidates):
            if anchor not in anchors:
                anchors.append(anchor)
    return anchors


def _target_from_question(
    question: Dict[str, Any],
    *,
    mechanism: Dict[str, Any],
    index: int,
    evaluation_stage: str,
) -> Dict[str, Any]:
    spec = dict(question.get("edit_spec") or {})
    mechanism_id = str(question.get("mechanism_id") or mechanism.get("mechanism_id") or "").strip()
    question_id = str(question.get("question_id") or "").strip()
    target_id = question_id or f"mechanism_{index:03d}_{mechanism_id or 'target'}"
    counterfactual = str(question.get("counterfactual") or mechanism.get("counterfactual") or "").strip()
    replacement_intent = str(spec.get("replacement_intent") or "").strip()
    replacement_pseudocode = str(spec.get("replacement_pseudocode") or "").strip()
    exact_edit_intent = replacement_intent or counterfactual or replacement_pseudocode
    if replacement_pseudocode and replacement_pseudocode not in exact_edit_intent:
        exact_edit_intent = f"{exact_edit_intent}. Concrete replacement: {replacement_pseudocode}".strip()
    return {
        "target_id": target_id,
        "target_kind": "mechanism_ablation",
        "mechanism_id": mechanism_id,
        "mechanism_name": str(mechanism.get("name") or mechanism_id).strip(),
        "diagnosis_question": str(question.get("question") or "").strip(),
        "causal_variable": str(
            mechanism.get("causal_variable")
            or mechanism.get("claim")
            or question.get("counterfactual")
            or mechanism.get("name")
            or ""
        ).strip(),
        "evidence_files": _evidence_files(mechanism, question),
        "evidence_anchors": _evidence_anchors(mechanism, question),
        "runtime_paths": _string_list(mechanism.get("runtime_paths")),
        "owner_symbols": _string_list(mechanism.get("owner_symbols") or mechanism.get("chain_position")),
        "operation_hints": _string_list(mechanism.get("operation_hints") or question.get("mapping_kind")),
        "exact_edit_intent": exact_edit_intent,
        "edit_spec": spec,
        "preserve_contract": {
            "input": "preserve model input protocol",
            "output": "preserve forecast output shape",
            "task": "do not change dataset, target columns, horizon, metric, optimizer, scheduler, or training policy",
            "shape_invariant_argument": str(spec.get("shape_invariant_argument") or "").strip(),
        },
        "expected_behavior_delta": "same-input forecast should change if this mechanism is active",
        "risk": str(spec.get("risk") or "medium").strip() or "medium",
        "evaluation_stage": evaluation_stage,
        "granularity": "mechanism",
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return str(path)


def write_mechanism_diagnosis_artifacts(session: AgentSession, payload: Dict[str, Any]) -> Dict[str, str]:
    root = Path(session.knowledge_dir) / "baseline_diagnosis"
    paths = {
        "active_path_evidence_graph_path": _write_json(root / "active_path_evidence_graph.json", dict(payload.get("evidence_graph") or {})),
        "active_path_target_plan_path": _write_json(root / "active_path_target_plan.json", dict(payload.get("target_plan") or {})),
        "mechanism_ablation_plan_path": _write_json(root / "mechanism_ablation_plan.json", dict(payload.get("plan") or {})),
        "mechanism_ablation_plan_review_path": _write_json(root / "mechanism_ablation_plan_review.json", dict(payload.get("review") or {})),
    }
    return paths


def _slug(value: Any) -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or "target"


def _active_path_payload(evidence_graph: Dict[str, Any]) -> Dict[str, Any]:
    facts = dict(evidence_graph.get("facts") or {})
    blocks = []
    for item in list(evidence_graph.get("source_blocks") or [])[:14]:
        if not isinstance(item, dict):
            continue
        blocks.append(
            {
                "path": str(item.get("path") or "").replace("\\", "/"),
                "blocks": [
                    {
                        "name": block.get("name"),
                        "kind": block.get("kind"),
                        "line_start": block.get("line_start"),
                        "content": str(block.get("content") or "")[:9000],
                    }
                    for block in list(item.get("blocks") or [])[:8]
                    if isinstance(block, dict)
                ],
            }
        )
    return {
        "model_key": evidence_graph.get("model_key"),
        "active_source_files": list(evidence_graph.get("active_source_files") or []),
        "expanded_source_files": list(evidence_graph.get("expanded_source_files") or []),
        "baseline_hyper_params": facts.get("baseline_hyper_params") or {},
        "forward_called_components": list(facts.get("forward_called_components") or [])[:80],
        "components": list(facts.get("components") or [])[:80],
        "operation_hints": list(facts.get("operation_hints") or [])[:120],
        "source_blocks": blocks,
    }


def _target_schema() -> Dict[str, Any]:
    return {
        "required": ["ablation_questions"],
        "ablation_questions": [
            {
                "target_id": "short_unique_id",
                "target_kind": "mechanism_ablation",
                "mechanism_id": "mechanism_id",
                "mechanism_name": "human readable active mechanism",
                "diagnosis_question": "Does ablating ... change mse_norm?",
                "causal_variable": "precise source-level mechanism being ablated",
                "evidence_files": ["ts_benchmark/...py"],
                "evidence_anchors": ["exact source substring"],
                "runtime_paths": ["active forward/forecast call path"],
                "owner_symbols": ["forward", "forecast"],
                "operation_hints": ["exact_source_ablation"],
                "exact_edit_intent": "Intent-level source ablation to implement near the anchor while preserving runtime protocol.",
                "edit_spec": {
                    "target_file": "ts_benchmark/...py",
                    "anchor_text": "exact source substring",
                    "replacement_intent": "what mechanism is removed or bypassed while preserving runtime protocol",
                    "replacement_pseudocode": "optional implementation hint; leave empty if unsafe or if the coding agent should derive the exact patch from source",
                    "shape_invariant_argument": "why tensor shapes stay unchanged",
                    "risk": "low|medium|high",
                },
                "preserve_contract": {
                    "input": "preserve model input protocol",
                    "output": "preserve forecast output shape",
                    "task": "do not change dataset, target columns, horizon, metric, optimizer, scheduler, or training policy",
                },
                "expected_behavior_delta": "why predictions or metric should change",
                "risk": "low|medium|high",
            }
        ],
    }


def _active_path_summary_messages(model_key: str, evidence_graph: Dict[str, Any], max_targets: int) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Return exactly one JSON object. No markdown. No commentary. "
                "Do not propose dataset, metric, optimizer, scheduler, horizon, seq_len, or target-column changes."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "generate_active_path_mechanism_ablation_targets",
                    "model": model_key,
                    "architecture": "active_path_summary",
                    "max_targets": max_targets,
                    "quality_requirements": [
                        "target must be on the active forward/forecast path for current hyperparameters",
                        "target must test a core model innovation or mechanism",
                        "target_file must be a ts_benchmark Python file",
                        "anchor_text must be an exact source substring",
                        "replacement_pseudocode is optional and only a hint; prefer replacement_intent when exact code may break runtime protocol",
                        "state the runtime protocol that the coding agent must preserve, such as return structure, tensor shape, dtype/device, or state keys",
                        "do not use semantic identity replacements such as x = x, output = output, or a full forward method that only returns its input",
                        "prefer small local anchors from operation_hints over replacing whole methods or whole classes",
                        "do not target support-only normalization, output scaling, dropout, or activation unless it is the named core innovation of this model",
                        "do not use generic amplitude scaling unless it directly removes the named mechanism",
                    ],
                    "output_schema": _target_schema(),
                    "evidence": _active_path_payload(evidence_graph),
                    "instruction": (
                        "Generate ablation targets for the currently active runtime path. "
                        "Use active_source_files, current hyperparameters, forward-called components, operation_hints, and source blocks. "
                        "Reject dead branches internally and output only targets that execute under the shown config."
                    ),
                },
                ensure_ascii=False,
                default=str,
            ),
        },
    ]


def _normalize_active_path_target(model_key: str, item: Dict[str, Any], index: int, *, evaluation_stage: str) -> Dict[str, Any]:
    target = dict(item or {})
    spec = dict(target.get("edit_spec") or {})
    target_id = _slug(target.get("target_id") or target.get("question_id") or f"{model_key}_{index:03d}_{target.get('mechanism_name')}")
    target["target_id"] = target_id
    target["target_kind"] = "mechanism_ablation"
    target["mechanism_id"] = _slug(target.get("mechanism_id") or target_id)
    target["mechanism_name"] = str(target.get("mechanism_name") or target_id).strip()
    target["diagnosis_question"] = str(
        target.get("diagnosis_question")
        or f"Does ablating {target['mechanism_name']} in {model_key} change mse_norm?"
    ).strip()
    target["causal_variable"] = str(target.get("causal_variable") or target["mechanism_name"]).strip()
    target["evidence_files"] = [
        str(path).replace("\\", "/")
        for path in _as_list(target.get("evidence_files") or spec.get("target_file") or target.get("target_file"))
        if str(path).strip()
    ]
    target["evidence_anchors"] = [
        str(anchor)
        for anchor in _as_list(target.get("evidence_anchors") or spec.get("anchor_text") or target.get("anchor_text"))
        if str(anchor).strip()
    ]
    target["runtime_paths"] = [str(path) for path in _as_list(target.get("runtime_paths")) if str(path).strip()]
    target["owner_symbols"] = [str(value) for value in _as_list(target.get("owner_symbols") or ["forward", "forecast"]) if str(value).strip()]
    target["operation_hints"] = [str(value) for value in _as_list(target.get("operation_hints") or ["exact_source_ablation"]) if str(value).strip()]
    spec["target_file"] = str(spec.get("target_file") or target.get("target_file") or (target["evidence_files"][0] if target["evidence_files"] else "")).replace("\\", "/")
    spec["anchor_text"] = str(spec.get("anchor_text") or target.get("anchor_text") or (target["evidence_anchors"][0] if target["evidence_anchors"] else "")).strip()
    spec["replacement_pseudocode"] = str(spec.get("replacement_pseudocode") or target.get("replacement_pseudocode") or "").strip()
    spec["replacement_intent"] = str(spec.get("replacement_intent") or target.get("why_core") or f"Ablate {target['causal_variable']}").strip()
    spec["shape_invariant_argument"] = str(spec.get("shape_invariant_argument") or target.get("why_shape_safe") or "").strip()
    spec["risk"] = str(spec.get("risk") or target.get("risk") or "medium").strip()
    target["edit_spec"] = spec
    target["exact_edit_intent"] = str(
        target.get("exact_edit_intent")
        or spec.get("replacement_intent")
        or f"Ablate {target['causal_variable']} near anchor {spec['anchor_text']}."
    ).strip()
    target["preserve_contract"] = dict(target.get("preserve_contract") or {})
    target["preserve_contract"].setdefault("input", "preserve model input protocol")
    target["preserve_contract"].setdefault("output", "preserve forecast output shape")
    target["preserve_contract"].setdefault(
        "task",
        "do not change dataset, target columns, horizon, metric, optimizer, scheduler, or training policy",
    )
    target["preserve_contract"].setdefault("shape_invariant_argument", spec["shape_invariant_argument"])
    target["expected_behavior_delta"] = str(target.get("expected_behavior_delta") or "same-input forecast should change if this mechanism is active").strip()
    target["risk"] = str(target.get("risk") or spec["risk"] or "medium").strip()
    target["evaluation_stage"] = evaluation_stage
    target["granularity"] = "mechanism"
    return target


def generate_mechanism_ablation_plan(
    session: AgentSession,
    *,
    model_key: str,
    objective_metric: str,
    evaluation_stage: str,
    analysis: Dict[str, Any],
    max_targets: int,
) -> Dict[str, Any]:
    analysis = dict(analysis or {})
    policy = baseline_diagnosis_policy(session.base_dir)
    desired_count = max(
        0,
        int(max_targets if max_targets is not None else policy.get("max_ablation_targets", 3)),
    )
    evidence_graph = build_mechanism_evidence_graph(session, model_key=model_key)
    target_plan: Dict[str, Any]
    if desired_count <= 0:
        target_plan = {
            "schema_version": "active_path_ablation_target_plan_v1",
            "status": "skipped",
            "reason": "max_targets_zero",
            "model_key": model_key,
            "ablation_questions": [],
            "created_at": datetime.now().isoformat(),
        }
    else:
        client = getattr(session, "client", None)
        if client is None or not getattr(client, "api_available", False):
            target_plan = {
                "schema_version": "active_path_ablation_target_plan_v1",
                "status": "error",
                "error": "api_unavailable",
                "model_key": model_key,
                "ablation_questions": [],
                "created_at": datetime.now().isoformat(),
            }
        else:
            try:
                raw_plan = client.call_json(
                    stage="active_path_ablation_target_plan",
                    round_num=0,
                    stage_label=f"active_path_summary_targets_{model_key}",
                    execution_label=f"active_path_summary_{model_key}",
                    messages=_active_path_summary_messages(model_key, evidence_graph, desired_count),
                    schema_hint=json.dumps(_target_schema(), ensure_ascii=False),
                    stream_override=False,
                )
                target_plan = {
                    "schema_version": "active_path_ablation_target_plan_v1",
                    "status": "ok",
                    "model_key": model_key,
                    "ablation_questions": list(
                        raw_plan.get("ablation_questions")
                        or raw_plan.get("questions")
                        or raw_plan.get("items")
                        or []
                    ),
                    "raw_response": raw_plan,
                    "created_at": datetime.now().isoformat(),
                }
            except Exception as exc:
                target_plan = {
                    "schema_version": "active_path_ablation_target_plan_v1",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "model_key": model_key,
                    "ablation_questions": [],
                    "created_at": datetime.now().isoformat(),
                }
    targets: List[Dict[str, Any]] = []
    rejected_targets: List[Dict[str, Any]] = []
    active_files = set(str(path).replace("\\", "/") for path in list(evidence_graph.get("expanded_source_files") or []))
    for index, item in enumerate(list(target_plan.get("ablation_questions") or []), start=1):
        if not isinstance(item, dict):
            continue
        target = _normalize_active_path_target(model_key, item, index, evaluation_stage=evaluation_stage)
        spec = dict(target.get("edit_spec") or {})
        target_file = str(spec.get("target_file") or "").replace("\\", "/")
        anchor = str(spec.get("anchor_text") or "")
        replacement = str(spec.get("replacement_pseudocode") or "")
        resolved_anchor = resolve_source_anchor(target_file, anchor) if target_file else ""
        if resolved_anchor:
            spec["anchor_text"] = resolved_anchor
            if replacement and "\n" not in replacement and "\n" not in resolved_anchor:
                indent = resolved_anchor[: len(resolved_anchor) - len(resolved_anchor.lstrip())]
                if indent and not replacement.startswith(indent):
                    spec["replacement_pseudocode"] = indent + replacement.lstrip()
            target["edit_spec"] = spec
            target["evidence_anchors"] = [resolved_anchor]
            anchor = resolved_anchor
        validation = validate_ablation_task(target, evaluation_stage=evaluation_stage)
        if validation.get("status") == "ok":
            normalized_target = dict(validation.get("task") or target)
            anchor_exists = bool(target_file and source_contains_anchor(target_file, anchor))
            target_file_active = bool(target_file and target_file in active_files)
            normalized_target["active_path_review"] = {
                "anchor_exists": anchor_exists,
                "target_file_active": target_file_active,
                "target_file": target_file,
                "anchor_text": anchor,
            }
            if anchor_exists and target_file_active:
                targets.append(normalized_target)
            else:
                errors: List[str] = []
                if not anchor_exists:
                    errors.append("anchor_not_found")
                if not target_file_active:
                    errors.append("target_file_not_in_active_path")
                rejected_targets.append(
                    {
                        "target": normalized_target,
                        "errors": sorted(set(errors)),
                    }
                )
        else:
            rejected_targets.append({"target": target, "errors": list(validation.get("errors") or [])})

    plan = {
        "schema_version": "mechanism_ablation_plan_v1",
        "planner_source": "active_path_summary",
        "base_model": model_key,
        "model_key": model_key,
        "objective_metric": objective_metric,
        "evaluation_stage": evaluation_stage,
        "targets": targets,
        "created_at": datetime.now().isoformat(),
        "active_path_target_plan_status": target_plan.get("status"),
    }
    review_errors: List[str] = []
    if str(target_plan.get("status") or "") == "skipped":
        review_errors = []
    elif str(target_plan.get("status") or "") != "ok":
        review_errors.append(f"active_path_target_plan_status:{target_plan.get('status')}")
    if not targets and str(target_plan.get("status") or "") != "skipped":
        review_errors.append("no_valid_active_path_ablation_targets")
    review = {
        "schema_version": "mechanism_ablation_plan_review_v1",
        "status": (
            "skipped"
            if str(target_plan.get("status") or "") == "skipped"
            else ("approved" if targets and not review_errors else "rejected")
        ),
        "reviewed_targets": targets,
        "rejected_targets": rejected_targets,
        "errors": review_errors,
        "corrections": [],
        "evaluation_stage": evaluation_stage,
        "created_at": datetime.now().isoformat(),
    }
    payload = {
        "status": review["status"],
        "model_key": model_key,
        "evidence_graph": evidence_graph,
        "target_plan": target_plan,
        "plan": plan,
        "review": review,
    }
    payload["artifact_paths"] = write_mechanism_diagnosis_artifacts(session, payload)
    return payload
