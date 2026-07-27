from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from evocast.domain.knowledge_paths import task_knowledge_dir
from evocast.state.domain_store import load_runtime_payload, load_task_config
from evocast.harness.api_client import create_task_client
from evocast.research.dataset_profile import load_dataset_profile
from evocast.research.round_history_digest import ensure_round_history_digest, round_status_table


MAX_EVIDENCE_CHARS = 60000
AGENT_ABLATION_CHOICES = {"none", "A3", "A4", "A5"}
DEFAULT_AGENT_ABLATION = "none"

SCIENTIST_CRITIC_PROMPT = (
    "Use a three-phase scientist_critic method. Scientist phase: propose "
    "several standalone module-level architecture ideas from dataset facts, "
    "executed ablation evidence, prior research rounds, failure memory, model "
    "structure, and feasible code regions. Prior Research rounds are evidence "
    "only; do not propose to repair, ablate, isolate, combine, deepen, simplify, "
    "or slightly modify a previous Research mechanism. Focus on explore, replace, "
    "or invent roles. Critic phase: reject candidates that are shallow renames, "
    "pure config changes, metric chasing, redundant with prior rounds, local "
    "bias/gate/dropout/norm tweaks, wrappers without a new information path, "
    "or unsupported by the evidence; treat pre-execution rejected ablation "
    "targets as invalid target specs rather than scientific evidence, and do "
    "not reuse the same rejected ablation action as a Research idea. Selection "
    "phase: choose the candidate that introduces the clearest new module-level "
    "information flow while preserving the task protocol; return only the "
    "selected idea and the decisive critique."
)

RESEARCH_PROGRAM_HYPOTHESIS_COMPETITION_PROMPT = (
    "Use a two-stage research strategy. First act as a Research Program "
    "Director: infer which broad architectural needs are suggested by dataset "
    "facts, ablation evidence, prior rounds, failure memory, and model structure. "
    "Treat prior Research rounds as evidence only, not as mechanisms to repair, "
    "ablate, isolate, combine, deepen, simplify, or slightly modify. Then act as "
    "a Hypothesis Competition scientist: generate multiple standalone module-level "
    "architecture ideas that can be implemented from the active baseline source; "
    "each idea distinguishes among competing explanations. "
    "Prefer ideas with a named module, a clear insertion point, and a changed "
    "information flow. Avoid local follow-up experiments on previous variants. "
    "Select the module-level idea that best opens a new architectural direction "
    "while preserving dataset, horizon, metric, optimizer, scheduler, epoch, "
    "batch size, and learning rate."
)

PLANNER_ARCHITECTURE_PROMPTS = {
    "scientist_critic": SCIENTIST_CRITIC_PROMPT,
    "research_program_hypothesis_competition": RESEARCH_PROGRAM_HYPOTHESIS_COMPETITION_PROMPT,
}

IDEA_SCHEMA_HINT = json.dumps(
    {
        "required": [
            "idea_title",
            "decision_mode",
            "hypothesis",
            "evidence_trace",
            "target_mechanism",
            "target_code_region",
            "expected_behavior_delta",
            "failure_to_avoid",
            "why_not_other_options",
            "novelty_relation_to_history",
            "research_program_position",
            "terminal_display_title",
            "round_role",
            "why_this_role_now",
            "why_not_other_roles",
            "what_this_round_will_teach",
            "future_agenda_update",
            "history_usage",
        ]
    },
    ensure_ascii=False,
)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _compact_text(value: Any, *, max_chars: int = MAX_EVIDENCE_CHARS) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n...<truncated for scientist_critic planner>...\n" + text[-tail:]


def _latest_build_contract(root: Path) -> Dict[str, Any]:
    candidates = sorted(
        (root / "build_contracts").glob("*.json"),
        key=lambda item: item.stat().st_mtime if item.exists() else 0,
    )
    if not candidates:
        return {}
    payload = _read_json(candidates[-1], {})
    if isinstance(payload, dict):
        payload["_artifact_path"] = str(candidates[-1])
        return payload
    return {}


ROUND_HISTORY_GUIDANCE = (
    "Use round_history_digest as an explored-territory map, not as a mechanism library. Treat each one-line item as "
    "evidence about what has already been tried, not as a template to extend.\n\n"
    "Treat the current_best mechanism as evidence about the currently strongest error-correction path. Do not use "
    "current_best as a template for local continuations. If later rounds after current_best tested nearby heads, gates, "
    "routers, memory modules, residual refiners, boundary modules, or other small extensions and failed to replace "
    "current_best, compress those rounds mentally into one saturated region.\n\n"
    "Before naming a module, first identify the forecast-error source that remains unresolved by current_best. A valid "
    "candidate should attack that error source with a distinct information-flow path. A strong candidate usually differs "
    "from prior rounds in more than one aspect, such as where it enters the model, what operation it performs, what "
    "signal it uses, or how it affects prediction.\n\n"
    "Avoid candidates whose main novelty is a local variant of current_best, another router/gate/MoE/memory/head/"
    "frequency/decomposition module because similar families appeared in history, moving a previous mechanism to a "
    "nearby insertion point, or a higher-risk encoder/attention rewrite chosen only because boundary slots look "
    "saturated.\n\n"
    "Prefer a low-intrusion, shape-preserving module only when it has a clearly different error source and information "
    "path. If the available candidates are mostly local continuations of saturated historical families, select the least "
    "local module-level candidate that tests a genuinely different forecast-error hypothesis."
)


def _normalize_agent_ablation(value: Any) -> str:
    text = str(value or DEFAULT_AGENT_ABLATION).strip()
    if not text:
        return DEFAULT_AGENT_ABLATION
    if text.lower() == "none":
        return "none"
    upper = text.upper()
    if upper in AGENT_ABLATION_CHOICES:
        return upper
    raise ValueError(f"Unsupported agent_ablation: {value}. Use one of: {sorted(AGENT_ABLATION_CHOICES)}")


def _task_config(root: Path) -> Dict[str, Any]:
    return load_task_config(str(root.parents[1]), root.name)


def _frozen_planner_snapshot_path(root: Path) -> Path:
    return root / "planner_evidence_snapshot_A5.json"


def _live_evidence_bundle(
    *,
    base_dir: str,
    task_id: str,
    root: Path,
    api_config: str,
    history_limit: int,
    task_config: Dict[str, Any],
) -> Dict[str, Any]:
    runtime_state = load_runtime_payload(base_dir, task_id)
    canonical_current_best = runtime_state.get("current_best") if isinstance(runtime_state.get("current_best"), dict) else {}
    return {
        "task_id": task_id,
        "knowledge_dir": str(root),
        "task_config": task_config,
        "dataset_profile": load_dataset_profile(base_dir, task_id),
        "baseline_diagnosis": dict(runtime_state.get("baseline_diagnosis") or {}),
        "runtime_state": runtime_state,
        "canonical_current_best": canonical_current_best,
        "latest_build_contract": _latest_build_contract(root),
        "round_history_digest": ensure_round_history_digest(
            base_dir=base_dir,
            task_id=task_id,
            api_config=api_config,
            history_limit=history_limit,
        ),
        "recent_round_status_table": round_status_table(base_dir, task_id, limit=history_limit),
    }


def _frozen_evidence_keys(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_profile": bundle.get("dataset_profile", {}),
        "baseline_diagnosis": bundle.get("baseline_diagnosis", {}),
        "round_history_digest": bundle.get("round_history_digest", {}),
        "recent_round_status_table": bundle.get("recent_round_status_table", []),
    }


def _load_or_create_frozen_snapshot(*, root: Path, live_bundle: Dict[str, Any]) -> Dict[str, Any]:
    path = _frozen_planner_snapshot_path(root)
    existing = _read_json(path, {})
    if isinstance(existing, dict) and existing.get("schema_version") == "planner_evidence_snapshot_A5_v1":
        frozen = existing.get("frozen_evidence")
        if isinstance(frozen, dict):
            return frozen
    frozen = _frozen_evidence_keys(live_bundle)
    _write_json(
        path,
        {
            "schema_version": "planner_evidence_snapshot_A5_v1",
            "frozen_evidence": frozen,
        },
    )
    return frozen


def _planner_payload_bundle(evidence_bundle: Dict[str, Any], *, agent_ablation: str) -> Dict[str, Any]:
    payload = dict(evidence_bundle or {})
    if agent_ablation == "A3":
        for key in (
            "dataset_profile",
            "baseline_diagnosis",
            "round_history_digest",
            "recent_round_status_table",
        ):
            payload.pop(key, None)
    return payload


def _hard_requirements(agent_ablation: str) -> List[str]:
    requirements = [
        "Follow the architecture_instruction internally before selecting one idea.",
        "Generate multiple standalone module-level candidates or competing explanations before selecting one internally.",
        "Critique candidates for evidence grounding, novelty, module-level substance, actionability, redundancy, role fit, and failure-memory awareness.",
        "Choose one candidate because it best opens a new architectural direction with a clear module-level information-flow change.",
        "For research_program_hypothesis_competition: explicitly reason internally about supported, weakened, saturated, or unresolved interpretations, including challenged interpretations when the evidence conflicts.",
        "For research_program_hypothesis_competition: select a standalone module-level experiment, not the nearest local variant.",
        "Prior Research rounds are evidence only; do not use them as mechanisms to repair, ablate, isolate, combine, deepen, simplify, or slightly modify.",
        "Build the idea from the active baseline source, not from prior unpromoted variants.",
        "The only source of current_best is canonical_current_best / runtime_state.current_best. Never infer current_best from a prior round's lower single-seed metric, completed execution status, or digest wording.",
        "Do not cite build, import, shape, evaluator, provenance, or runtime failures as scientific evidence about model quality; treat them as engineering or infrastructure failure memory.",
        "Scientific rejection with a completed metric is valid evidence that the hypothesis was not supported under that protocol.",
        "Do not judge idea quality by expected metric improvement alone.",
        "terminal_display_title must be 8 words or fewer.",
        "Choose round_role from explore, replace, invent.",
        "Prefer standalone module-level architecture ideas with a named module, clear insertion point, and changed information flow.",
        "Reject local bias-only, gate-only, dropout-only, norm-only, activation-only, or parameter-only tweaks.",
        "Reject wrapper modules that do not create a new information path.",
        "The idea must be actionable for BuildContract/hints integration.",
        "Identify what new architectural behavior the idea would test if executed.",
        "Do not propose dataset, metric, optimizer, scheduler, epoch, batch_size, or lr changes.",
        "Do not present a pure configuration fix as a research idea.",
    ]
    if agent_ablation != "A3":
        requirements.extend(
            [
                "Use dataset, ablation, and prior-round evidence together whenever available.",
                "Separate evidence semantics: executed ablation rounds are mechanism evidence; pre-execution rejected_targets in baseline_diagnosis are invalid or unrepaired target specs, not mechanism evidence.",
                "Do not select the same ablation action from baseline_diagnosis.review.rejected_targets or plan_review.rejected_targets as a Research idea.",
                ROUND_HISTORY_GUIDANCE,
            ]
        )
    return requirements


def load_research_evidence_bundle(
    base_dir: str,
    task_id: str,
    *,
    api_config: str,
    history_limit: int = 20,
) -> Dict[str, Any]:
    root = task_knowledge_dir(base_dir, task_id)
    task_config = _task_config(root)
    agent_ablation = _normalize_agent_ablation(task_config.get("agent_ablation"))
    live_bundle = _live_evidence_bundle(
        base_dir=base_dir,
        task_id=task_id,
        root=root,
        api_config=api_config,
        history_limit=history_limit,
        task_config=task_config,
    )
    if agent_ablation == "A5":
        frozen = _load_or_create_frozen_snapshot(root=root, live_bundle=live_bundle)
        merged = dict(live_bundle)
        merged.update(frozen)
        merged["planner_evidence_policy"] = {
            "agent_ablation": agent_ablation,
            "frozen_keys": sorted(frozen.keys()),
            "live_keys": ["runtime_state", "canonical_current_best", "latest_build_contract", "task_config"],
        }
        return merged
    live_bundle["planner_evidence_policy"] = {"agent_ablation": agent_ablation}
    return live_bundle


def _word_count(text: Any) -> int:
    return len(str(text or "").strip().split())


def _normalize_title(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(payload or {})
    title = str(result.get("terminal_display_title") or result.get("idea_title") or "").strip()
    if _word_count(title) <= 8:
        result["terminal_display_title"] = title
        return result
    result["terminal_display_title_original"] = title
    result["terminal_display_title"] = " ".join(title.split()[:8])
    result["terminal_display_title_warning"] = "planner_returned_over_8_words_local_truncation_applied"
    return result


def _planner_architecture(value: str | None) -> str:
    key = str(value or "research_program_hypothesis_competition").strip().lower()
    if key not in PLANNER_ARCHITECTURE_PROMPTS:
        allowed = ", ".join(sorted(PLANNER_ARCHITECTURE_PROMPTS))
        raise ValueError(f"Unsupported idea planner architecture: {key}. Use one of: {allowed}")
    return key


def _messages(
    *,
    evidence_bundle: Dict[str, Any],
    research_id: str,
    planner_architecture: str = "research_program_hypothesis_competition",
) -> List[Dict[str, str]]:
    architecture = _planner_architecture(planner_architecture)
    agent_ablation = _normalize_agent_ablation(
        (evidence_bundle.get("task_config") if isinstance(evidence_bundle.get("task_config"), dict) else {}).get("agent_ablation")
        or (evidence_bundle.get("planner_evidence_policy") if isinstance(evidence_bundle.get("planner_evidence_policy"), dict) else {}).get("agent_ablation")
    )
    architecture_instruction = PLANNER_ARCHITECTURE_PROMPTS[architecture]
    system = (
        f"You are an autonomous time-series model research planner using the concrete {architecture} method. "
        "Your task is not to implement code and not to optimize a stochastic metric directly. "
        "Choose the next research idea from dataset analysis, ablation evidence, prior rounds, "
        "failure memory, model structure, and task constraints. Do not use fixed numeric source weights. "
        "Internally follow the named method before selecting exactly one idea and returning it as JSON."
    )
    user = {
        "planner_architecture": architecture,
        "agent_ablation": agent_ablation,
        "architecture_instruction": architecture_instruction,
        "research_id": research_id,
        "hard_requirements": _hard_requirements(agent_ablation),
        "required_json_shape": json.loads(IDEA_SCHEMA_HINT),
        "evidence_bundle_text": _compact_text(_planner_payload_bundle(evidence_bundle, agent_ablation=agent_ablation)),
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def run_scientist_critic_planner(
    *,
    base_dir: str,
    task_id: str,
    research_id: str,
    api_config: str,
    history_limit: int = 20,
    planner_architecture: str = "research_program_hypothesis_competition",
) -> Dict[str, Any]:
    architecture = _planner_architecture(planner_architecture)
    evidence = load_research_evidence_bundle(base_dir, task_id, api_config=api_config, history_limit=history_limit)
    client = create_task_client(
        base_dir=base_dir,
        task_id=f"{task_id}_{architecture}_{research_id}",
        explicit_config=api_config,
    )
    if not client.api_available:
        raise RuntimeError("scientist_critic_planner_requires_real_api_key")
    idea = client.call_json(
        "planner",
        int(str(research_id).replace("Research", "") or 0),
        _messages(evidence_bundle=evidence, research_id=research_id, planner_architecture=architecture),
        schema_hint=IDEA_SCHEMA_HINT,
        execution_label=f"{architecture}_research_direction",
        require_all_top_level_keys=True,
        stream_override=False,
    )
    idea = _normalize_title(idea)
    idea = {
        "schema_version": "research_direction_v1",
        "planner": architecture,
        "research_id": research_id,
        **idea,
    }
    out = task_knowledge_dir(base_dir, task_id) / "research_directions" / f"{research_id}_{architecture}.json"
    _write_json(out, idea)
    return {**idea, "artifact_path": str(out)}
