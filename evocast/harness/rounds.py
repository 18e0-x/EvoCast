"""Research round state tracking for v3 sessions.

The controller keeps an explicit current round pointer and enforces the core
invariant: one source-grounded hypothesis, one variant, one evaluation lineage.

P0-2/P0-4 failure classification (4 types, ordered by priority):
  implementation_failure  – code didn't compile / import / pass static checks
  activation_failure      – code wrote but didn't hook into forward path (noop)
  validation_failure      – smoke / behavior probe / contract check failed
  scientific_failure      – ran experiment but didn't beat baseline

Only scientific_failure enters failed_fit_points as research knowledge.

P0-2 phase state machine (DESIGN → BUILD → VERIFY → EXPERIMENT).
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from evocast.domain.atomic_io import atomic_write_json
from evocast.domain import round_semantics as semantics
from evocast.domain import round_transitions
from evocast.policy.agent_control_policy import positive_int, round_control_policy
from evocast.domain.execution_ids import (
    RESEARCH_ROUNDS_DIR,
    format_ablation_id,
    format_research_id,
    parse_ablation_id,
    parse_research_id,
    round_record_filename,
)
from evocast.domain.knowledge_paths import task_knowledge_dir
from evocast.state.runtime.store import load_runtime_state, publish_runtime_event, update_runtime_state
from evocast.state.domain_store import load_round_record, list_round_records, load_task_config, save_round_record
from evocast.domain.state_model import Decision, EvaluationResult
from evocast.state.round_projection import (
    gate_review_obligations as project_gate_review_obligations,
    project_round_progress,
    seed_eval_obligations as project_seed_eval_obligations,
)

FAILURE_KIND_IMPLEMENTATION = semantics.FAILURE_KIND_IMPLEMENTATION
FAILURE_KIND_ACTIVATION = semantics.FAILURE_KIND_ACTIVATION
FAILURE_KIND_VALIDATION = semantics.FAILURE_KIND_VALIDATION
FAILURE_KIND_RUNTIME = semantics.FAILURE_KIND_RUNTIME
FAILURE_KIND_SCIENTIFIC = semantics.FAILURE_KIND_SCIENTIFIC
FAILURE_KIND_IMPL_INVALID = semantics.FAILURE_KIND_IMPL_INVALID
FAILURE_KIND_IMPL_UNFAITHFUL = semantics.FAILURE_KIND_IMPL_UNFAITHFUL
FAILURE_KIND_IMPL_INACTIVE = semantics.FAILURE_KIND_IMPL_INACTIVE
FAILURE_KIND_RUNTIME_IMPORT = semantics.FAILURE_KIND_RUNTIME_IMPORT
FAILURE_KIND_RUNTIME_SHAPE = semantics.FAILURE_KIND_RUNTIME_SHAPE
FAILURE_KIND_RUNTIME_CUDA = semantics.FAILURE_KIND_RUNTIME_CUDA
MAX_CONSECUTIVE_PIPELINE_FAILURES = semantics.MAX_CONSECUTIVE_PIPELINE_FAILURES
_PIPELINE_FAILURE_KINDS = semantics._PIPELINE_FAILURE_KINDS

def classify_failure_kind(status: str, error_type: str = "", has_experiment_run: bool = False) -> str:
    """Return the canonical failure kind for a round status/error_type pair.

    Returns one of: implementation_failure, activation_failure, validation_failure, runtime_failure,
    scientific_failure.  Only scientific_failure means "we ran the experiment and it
    didn't work"; the other three are engineering failures.
    """
    return semantics.classify_failure_kind(status, error_type, has_experiment_run)

PHASE_DESIGN = semantics.PHASE_DESIGN
PHASE_BUILD = semantics.PHASE_BUILD
PHASE_VERIFY = semantics.PHASE_VERIFY
PHASE_EXPERIMENT = semantics.PHASE_EXPERIMENT
PHASE_CLOSED = semantics.PHASE_CLOSED

def _next_phase(current: str) -> str:
    """Return the next phase after *current*, or PHASE_CLOSED if terminal."""
    return semantics.next_phase(current)


def _now() -> str:
    return datetime.now().isoformat()


def rounds_dir(base_dir: str, task_id: str) -> Path:
    path = task_knowledge_dir(base_dir, task_id) / RESEARCH_ROUNDS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _round_path(base_dir: str, task_id: str, round_id: int) -> Path:
    path = rounds_dir(base_dir, task_id) / format_research_id(int(round_id))
    path.mkdir(parents=True, exist_ok=True)
    return path / round_record_filename()


def _round_path_for_label(base_dir: str, task_id: str, label: str) -> Path:
    path = rounds_dir(base_dir, task_id) / str(label)
    path.mkdir(parents=True, exist_ok=True)
    return path / round_record_filename()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_json(path, payload, ensure_ascii=False, default=str)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_rounds(base_dir: str, task_id: str) -> List[Dict[str, Any]]:
    return list_round_records(base_dir, task_id)


def _next_round_id(base_dir: str, task_id: str) -> int:
    existing = [record for record in list_rounds(base_dir, task_id) if _round_counts_toward_research_budget(record)]
    if not existing:
        return 1
    return max(int(item.get("round_id") or 0) for item in existing) + 1


def _next_ablation_id(base_dir: str, task_id: str) -> int:
    indices = []
    for path in rounds_dir(base_dir, task_id).glob("Ablation*"):
        parsed = parse_ablation_id(path.name)
        if parsed is not None:
            indices.append(parsed)
    for record in list_rounds(base_dir, task_id):
        if _infer_round_scope(record) == ROUND_SCOPE_BASELINE_DIAGNOSIS:
            indices.append(int(record.get("round_id") or 0))
    return max(indices or [0]) + 1


def _task_config(base_dir: str, task_id: str) -> Dict[str, Any]:
    return load_task_config(base_dir, task_id)


def _enforce_max_rounds(base_dir: str, task_id: str) -> None:
    config = _task_config(base_dir, task_id)
    if "max_rounds" not in config:
        return
    max_rounds = int(config.get("max_rounds") or 0)
    if max_rounds == 0:
        raise RuntimeError(f"max_rounds=0 for task {task_id}: no formal Research round may start")
    if max_rounds < 0:
        return
    progress = round_progress(base_dir, task_id)
    if int(progress.get("terminal_rounds") or 0) >= max_rounds:
        raise RuntimeError(
            f"max_rounds reached for task {task_id}: "
            f"{progress.get('terminal_rounds')}/{max_rounds} terminal rounds"
        )


def ensure_research_capacity(base_dir: str, task_id: str, fit_point: Any) -> None:
    """Reject a design before Builder work when no formal round can start."""
    _enforce_max_rounds(base_dir, task_id)
    _enforce_seed_eval_obligations(base_dir, task_id)
    enforce_gate_review_obligations_for_fit_point(base_dir, task_id, fit_point)


TERMINAL_ROUND_STATUSES = semantics.TERMINAL_ROUND_STATUSES
INFRA_FAILURE_STATUSES = semantics.INFRA_FAILURE_STATUSES
ROUND_OPEN_STATUSES = semantics.ROUND_OPEN_STATUSES
CRITICAL_VARIANT_WRITE_ERRORS = semantics.CRITICAL_VARIANT_WRITE_ERRORS
GATE_RESOLUTION_DECISIONS = semantics.GATE_RESOLUTION_DECISIONS
ROUND_SCOPE_RESEARCH = semantics.ROUND_SCOPE_RESEARCH
ROUND_SCOPE_BASELINE_DIAGNOSIS = semantics.ROUND_SCOPE_BASELINE_DIAGNOSIS


def _is_ablation_variant_path(value: Any) -> bool:
    return semantics.is_ablation_variant_path(value)


def _infer_round_scope(record: Dict[str, Any]) -> str:
    return semantics.infer_round_scope(record)


def _round_counts_toward_research_budget(record: Dict[str, Any]) -> bool:
    """Whether this idea-scoped Research round consumes one budget slot."""
    return semantics.counts_toward_research_budget(record)


def _apply_round_budget_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    return semantics.apply_round_budget_metadata(record)


def seed_eval_obligations(base_dir: str, task_id: str) -> List[Dict[str, Any]]:
    """Return open gate rounds requiring seed evaluation."""
    return project_seed_eval_obligations(list_rounds(base_dir, task_id))


def gate_review_obligations(base_dir: str, task_id: str) -> List[Dict[str, Any]]:
    """Return unresolved needs_review/needs_seed_eval gate events for same-fit-point locking."""
    return project_gate_review_obligations(list_rounds(base_dir, task_id))


def _enforce_seed_eval_obligations(base_dir: str, task_id: str) -> None:
    obligations = seed_eval_obligations(base_dir, task_id)
    if not obligations:
        return
    first = obligations[0]
    raise RuntimeError(
        "NEEDS_SEED_EVAL_OBLIGATION: round "
        f"{first.get('round_id')} for {first.get('variant_path') or first.get('candidate_name')} "
        "requires run_seed_eval before starting another research round"
    )


def enforce_gate_review_obligations_for_fit_point(base_dir: str, task_id: str, fit_point: Any) -> None:
    """Block repeated same-fit-point exploration while a seed-eval/review gate is unresolved."""
    requested = str(fit_point or "").strip().lower()
    if not requested:
        return
    for item in gate_review_obligations(base_dir, task_id):
        pending = str(item.get("fit_point") or "").strip().lower()
        if pending and pending == requested:
            raise RuntimeError(
                "GATE_REVIEW_OBLIGATION: round "
                f"{item.get('round_id')} at fit_point={item.get('fit_point')} has "
                f"decision={item.get('decision')} and requires run_seed_eval "
                "before starting another round at the same fit_point"
            )


def _has_experiment_run(record: Dict[str, Any]) -> bool:
    return any(str(run.get("stage") or "") == "experiment" for run in list(record.get("runs") or []))


def _record_counts_as_completed_round(record: Dict[str, Any]) -> bool:
    return semantics.counts_as_completed_round(record)


def _successful_experiment_runs(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return semantics.successful_experiment_runs(record)


def _gate_has_result_delta(record: Dict[str, Any]) -> bool:
    return semantics.gate_has_result_delta(record)


def _record_counts_as_valid_trial(record: Dict[str, Any]) -> bool:
    return semantics.counts_as_valid_trial(record)


def _latest_open_round(base_dir: str, task_id: str) -> Optional[Dict[str, Any]]:
    for record in reversed(list_rounds(base_dir, task_id)):
        if record.get("status") not in TERMINAL_ROUND_STATUSES:
            return record
    return None


def _current_round_id(base_dir: str, task_id: str) -> Optional[int]:
    state = load_runtime_state(base_dir, task_id, auto_migrate=False)
    value = (state.research or {}).get("current_round_id")
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _current_round_label(base_dir: str, task_id: str) -> str:
    state = load_runtime_state(base_dir, task_id, auto_migrate=False)
    return str((state.research or {}).get("current_research_id") or "").strip()


def _set_current_round_id(base_dir: str, task_id: str, round_id: Optional[int], *, label: str | None = None) -> None:
    def _apply(state):
        if round_id is None:
            state.research["current_round_id"] = None
            state.research["current_research_id"] = None
            state.research["current_round_stage"] = None
            return
        current_label = str(label or format_research_id(int(round_id)))
        state.research["current_round_id"] = int(round_id)
        state.research["current_research_id"] = current_label
        state.research["current_round"] = int(round_id)
        state.research["current_round_stage"] = "active"
        if parse_research_id(current_label) is not None:
            next_round = max(int(state.research.get("next_round") or 1), int(round_id) + 1)
            state.research["next_round"] = next_round
            state.research["next_research_id"] = format_research_id(next_round)

    update_runtime_state(base_dir, task_id, _apply)


def _get_round(base_dir: str, task_id: str, round_id: int) -> Dict[str, Any]:
    return load_round_record(base_dir, task_id, format_research_id(int(round_id)))


def _get_round_by_label(base_dir: str, task_id: str, label: str) -> Dict[str, Any]:
    if not label:
        return {}
    return load_round_record(base_dir, task_id, label)


def current_round(base_dir: str, task_id: str) -> Optional[Dict[str, Any]]:
    label = _current_round_label(base_dir, task_id)
    if label:
        record = _get_round_by_label(base_dir, task_id, label)
        if record and record.get("status") not in TERMINAL_ROUND_STATUSES:
            return record
        _set_current_round_id(base_dir, task_id, None)
    round_id = _current_round_id(base_dir, task_id)
    if round_id is not None:
        record = _get_round(base_dir, task_id, round_id)
        if record and record.get("status") not in TERMINAL_ROUND_STATUSES:
            return record
        _set_current_round_id(base_dir, task_id, None)
    return _latest_open_round(base_dir, task_id)


def _save_round(base_dir: str, task_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    _apply_round_budget_metadata(record)
    round_id = int(record["round_id"])
    record.setdefault("research_index", round_id)
    scope = _infer_round_scope(record)
    default_label = format_ablation_id(round_id) if scope == ROUND_SCOPE_BASELINE_DIAGNOSIS else format_research_id(round_id)
    record.setdefault("research_id", default_label)
    round_label = str(record.get("research_id") or default_label)
    terminal = record.get("status") in TERMINAL_ROUND_STATUSES
    counts_as_completed = _record_counts_as_completed_round(record)
    # A terminal attempt emits exactly one outcome event.  Gate persistence,
    # close helpers and report refreshes may all save the same record; none of
    # them may manufacture a second historical experiment.
    emit_research_outcome = bool(
        terminal
        and counts_as_completed
        and not bool(record.get("research_outcome_recorded"))
    )
    if emit_research_outcome:
        record["research_outcome_recorded"] = True
        record["research_outcome_recorded_at"] = _now()
    if terminal and record.get("status") != "completed" and not record.get("failure_kind"):
        record["failure_kind"] = _resolve_failure_kind(record)
    # This is the one live round-state writer.  Compatibility fields remain
    # evidence for historical artifacts, while canonical evaluation/decision
    # are refreshed here before persistence.
    record["evaluation"] = EvaluationResult.from_round(record).to_dict()
    record["decision"] = Decision.from_round(record).to_dict()
    record["updated_at"] = _now()
    record = save_round_record(base_dir, task_id, record)
    if terminal:
        current_label = _current_round_label(base_dir, task_id)
        if current_label == round_label or (not current_label and _current_round_id(base_dir, task_id) == round_id):
            _set_current_round_id(base_dir, task_id, None)

    def _apply_completion_membership(state):
        raw_completed = state.research.get("completed_rounds")
        if isinstance(raw_completed, list):
            completed = [int(item) for item in raw_completed if int(item or 0) > 0]
        else:
            completed = [
                int(item.get("round_id") or 0)
                for item in list_rounds(base_dir, task_id)
                if _record_counts_as_completed_round(item)
            ]
        if terminal and counts_as_completed and round_id not in completed:
            completed.append(round_id)
        if (not terminal or not counts_as_completed) and round_id in completed:
            completed = [item for item in completed if int(item) != round_id]
        state.research["completed_rounds"] = sorted(completed)

    update_runtime_state(base_dir, task_id, _apply_completion_membership)

    if terminal:
        def _apply(state):
            state.research["current_round_id"] = None
            state.research["current_research_id"] = None
            state.research["current_round"] = None
            state.research["current_round_stage"] = None
            if _round_counts_toward_research_budget(record):
                state.research["next_round"] = max(int(state.research.get("next_round") or 1), round_id + 1)
                state.research["next_research_id"] = format_research_id(
                    max(int(state.research.get("next_round") or 1), round_id + 1)
                )
            if record.get("status") == "completed" and counts_as_completed:
                state.research["consecutive_no_improvement"] = 0
            elif counts_as_completed:
                # P2-2: Only count scientific failures toward "no improvement".
                # Implementation / activation / validation failures mean the
                # experiment never ran or wasn't meaningful — they don't
                # tell us anything about scientific progress, so reset.
                failure_kind = record.get("failure_kind") or _resolve_failure_kind(record)
                if failure_kind == FAILURE_KIND_SCIENTIFIC:
                    state.research["consecutive_no_improvement"] = int(
                        state.research.get("consecutive_no_improvement") or 0
                    ) + 1
                else:
                    state.research["consecutive_no_improvement"] = 0

        update_runtime_state(base_dir, task_id, _apply)
        # ── P2: Track consecutive pipeline failures ──────────────────────────
        # Pipeline failures (implementation/activation/validation) that repeat
        # indicate a structural problem — block the pipeline before wasting
        # more rounds.
        if counts_as_completed:
            failure_kind = record.get("failure_kind") or _resolve_failure_kind(record)

            def _track_pipeline_blocked(state):
                blocked = bool(state.research.get("pipeline_blocked"))
                if blocked:
                    return  # already blocked — don't reset
                if failure_kind in _PIPELINE_FAILURE_KINDS:
                    count = int(state.research.get("consecutive_pipeline_failures") or 0) + 1
                    state.research["consecutive_pipeline_failures"] = count
                    if count >= MAX_CONSECUTIVE_PIPELINE_FAILURES:
                        state.research["pipeline_blocked"] = True
                        state.research["pipeline_blocked_reason"] = (
                            f"{count} consecutive pipeline failures "
                            f"(last: {failure_kind} in round {record.get('round_id')})"
                        )
                else:
                    state.research["consecutive_pipeline_failures"] = 0

            update_runtime_state(base_dir, task_id, _track_pipeline_blocked)

    publish_runtime_event(
        base_dir,
        task_id,
        "round_state",
        {
            "round_id": record.get("research_id") or format_research_id(round_id),
            "status": record.get("status"),
            "phase": record.get("phase"),
            "fit_point": record.get("fit_point"),
            "operation_family": record.get("operation_family"),
            "terminal": terminal,
        },
    )

    return record


def record_build_outcome(
    *,
    base_dir: str,
    task_id: str,
    research_id: str,
    round_id: int,
    candidate_snapshot_id: str | None,
    patch_path: Any,
    changed_files: List[str],
    status: Any,
) -> Dict[str, Any]:
    record = _get_round_by_label(base_dir, task_id, research_id)
    if not record:
        raise RuntimeError(f"round.json missing for {research_id}")
    if int(record.get("round_id") or 0) != int(round_id):
        raise RuntimeError(
            f"round.json round_id mismatch for {research_id}: "
            f"record={record.get('round_id')}, outcome={round_id}"
        )
    record["candidate_snapshot_id"] = candidate_snapshot_id
    record["patch_path"] = str(patch_path) if patch_path else None
    record["changed_files"] = list(changed_files or [])
    record["build_outcome"] = {"status": str(getattr(status, "value", status))}
    return _save_round(base_dir, task_id, record)


def start_round(
    *,
    base_dir: str,
    task_id: str,
    fit_point: Any,
    hypothesis: Any,
    evidence_source: Any,
    model_family: Any = None,
    operation_family: Any = None,
    research_direction: Any = None,
    research_intent: Any = None,
    borrowed_from: Any = None,
    mechanism_summary: Any = None,
    expected_effect: Any = None,
    shape_proof: Any = None,
    tensor_ledger: Any = None,
    task_semantics_unchanged: Any = None,
    training_policy_unchanged: Any = None,
    model_structure_cache: Any = None,
    proposal: Optional[Dict[str, Any]] = None,
    round_scope: Optional[str] = None,
    counts_toward_research_budget: Optional[bool] = None,
) -> Dict[str, Any]:
    requested_budget_record = {
        "round_scope": round_scope or ROUND_SCOPE_RESEARCH,
        "counts_toward_research_budget": counts_toward_research_budget,
    }
    if _round_counts_toward_research_budget(requested_budget_record):
        _enforce_max_rounds(base_dir, task_id)
        _enforce_seed_eval_obligations(base_dir, task_id)
        enforce_gate_review_obligations_for_fit_point(base_dir, task_id, fit_point)
    existing = current_round(base_dir, task_id)
    if existing and existing.get("status") not in TERMINAL_ROUND_STATUSES:
        existing_label = str(existing.get("research_id") or format_research_id(int(existing.get("round_id") or 0)))
        raise RuntimeError(
            f"{existing_label} is still open; close it before starting a new research round"
        )
    requested_scope = round_scope or ROUND_SCOPE_RESEARCH
    is_diagnostic_ablation = requested_scope == ROUND_SCOPE_BASELINE_DIAGNOSIS
    next_round_id = _next_ablation_id(base_dir, task_id) if is_diagnostic_ablation else _next_round_id(base_dir, task_id)
    execution_label = format_ablation_id(next_round_id) if is_diagnostic_ablation else format_research_id(next_round_id)
    record = {
        "round_id": next_round_id,
        "research_index": next_round_id,
        "research_id": execution_label,
        **({"ablation_id": execution_label, "ablation_index": next_round_id} if is_diagnostic_ablation else {}),
        "base_dir": base_dir,
        "variant_path": None,
        "fit_point": fit_point,
        "hypothesis": hypothesis,
        "evidence_source": evidence_source,
        "model_family": model_family,
        "operation_family": operation_family,
        "research_direction": research_direction,
        "research_intent": research_intent or hypothesis,
        "borrowed_from": list(borrowed_from or []),
        "mechanism_summary": mechanism_summary,
        "expected_effect": expected_effect,
        "shape_proof": shape_proof,
        "tensor_ledger": tensor_ledger,
        "task_semantics_unchanged": bool(task_semantics_unchanged) if task_semantics_unchanged is not None else None,
        "training_policy_unchanged": bool(training_policy_unchanged) if training_policy_unchanged is not None else None,
        "model_structure_cache": model_structure_cache,
        "proposal": dict(proposal or {}),
        "round_scope": requested_scope,
        "counts_toward_research_budget": counts_toward_research_budget,
        "concrete_variant_attempt": {"committed_at": _now(), "source": "direct_start"},
        "runs": [],
        "repair_count": 0,
        "repair_history": [],                     # P1-3: prevent repeating same fix
        "gate_decision": None,
        "status": "round_started",
        "phase": PHASE_DESIGN,                    # P0-2: phase state machine
        "failure_kind": None,                     # P0-4: canonical failure kind
        "created_at": _now(),
        "updated_at": _now(),
    }
    saved = _save_round(base_dir, task_id, record)
    _set_current_round_id(base_dir, task_id, int(saved["round_id"]), label=str(saved.get("research_id") or execution_label))
    return saved


def record_phase_transition(
    *,
    base_dir: str,
    task_id: str,
    phase: str,
) -> Optional[Dict[str, Any]]:
    """Advance the current round to *phase* (DESIGN → BUILD → VERIFY → EXPERIMENT)."""
    record = current_round(base_dir, task_id)
    if not record:
        return None
    record["phase"] = str(phase or PHASE_DESIGN).strip()
    return _save_round(base_dir, task_id, record)


def record_round_display_metadata(
    *,
    base_dir: str,
    task_id: str,
    display_idea: Any = None,
    idea_summary: Any = None,
    mechanism_summary: Any = None,
    execution_summary: Any = None,
    mechanism_name: Any = None,
    changed_mechanism: Any = None,
    expected_effect: Any = None,
) -> Optional[Dict[str, Any]]:
    """Attach user-facing research idea text to the current round."""
    record = current_round(base_dir, task_id)
    if not record:
        return None
    if display_idea is not None and str(display_idea).strip():
        record["display_idea"] = str(display_idea).strip()
    if idea_summary is not None and str(idea_summary).strip():
        record["idea_summary"] = str(idea_summary).strip()
    if mechanism_summary is not None and str(mechanism_summary).strip():
        record["mechanism_summary"] = str(mechanism_summary).strip()
    if execution_summary is not None and str(execution_summary).strip():
        record["execution_summary"] = str(execution_summary).strip()
    if mechanism_name is not None and str(mechanism_name).strip():
        record["mechanism_name"] = str(mechanism_name).strip()
    if changed_mechanism is not None and str(changed_mechanism).strip():
        record["changed_mechanism"] = str(changed_mechanism).strip()
    if expected_effect is not None and str(expected_effect).strip():
        record["expected_effect"] = str(expected_effect).strip()
    return _save_round(base_dir, task_id, record)


def record_repair_attempt(
    *,
    base_dir: str,
    task_id: str,
    repair_entry: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Add an entry to the round's repair_history with failure_kind and attempted fix."""
    record = current_round(base_dir, task_id)
    if not record:
        return None
    history = list(record.get("repair_history") or [])
    history.append(dict(repair_entry or {}))
    record["repair_history"] = history[-20:]  # keep bounded
    record["repair_count"] = len(record["repair_history"])
    return _save_round(base_dir, task_id, record)


def _resolve_failure_kind(record: Dict[str, Any]) -> str:
    return semantics.resolve_failure_kind(record)


def is_pipeline_blocked(base_dir: str, task_id: str) -> bool:
    """P2: Return True when consecutive pipeline failures block further rounds."""
    state = load_runtime_state(base_dir, task_id)
    return bool(state.research.get("pipeline_blocked"))


def check_pipeline_blocked(base_dir: str, task_id: str) -> Optional[str]:
    """P2: Return reason string if pipeline is blocked, None otherwise."""
    state = load_runtime_state(base_dir, task_id)
    if state.research.get("pipeline_blocked"):
        return str(state.research.get("pipeline_blocked_reason") or "pipeline_blocked")
    return None


def close_current_round(
    *,
    base_dir: str,
    task_id: str,
    status: str,
    reason: str = "",
    extra: Optional[Dict[str, Any]] = None,
    round_id: Optional[int] = None,
    research_id: Optional[str] = None,
) -> Dict[str, Any]:
    label = str(research_id or "").strip()
    if label:
        record = _get_round_by_label(base_dir, task_id, label)
    elif round_id is not None:
        record = _get_round(base_dir, task_id, int(round_id))
    else:
        record = current_round(base_dir, task_id)
    if not record:
        raise RuntimeError("no open round to close")
    if record.get("status") in TERMINAL_ROUND_STATUSES:
        return record
    if record.get("gate_decision") in GATE_RESOLUTION_DECISIONS:
        seed_eval = dict(record.get("seed_eval") or {})
        if seed_eval.get("status") not in {"completed", "failed"}:
            raise RuntimeError(
                "NEEDS_SEED_EVAL_OBLIGATION: closing a BuildContract research round requires run_seed_eval "
                "for a needs_review/needs_seed_eval gate"
            )
    normalized = str(status or "").strip()
    if normalized not in TERMINAL_ROUND_STATUSES:
        raise RuntimeError(f"close round requires terminal status, got {status!r}")
    record["status"] = normalized
    record["phase"] = PHASE_CLOSED
    record["close_reason"] = reason
    record["closed_at"] = _now()
    # P0-4: classify failure kind on close
    if normalized != "completed":
        record["failure_kind"] = _resolve_failure_kind(record)
    if extra:
        record.setdefault("close_extra", {}).update(dict(extra or {}))
        terminal_reason = str((extra or {}).get("terminal_reason") or "").strip()
        if terminal_reason:
            analysis = dict(record.get("failure_analysis") or {})
            analysis["terminal_reason"] = terminal_reason
            if "max_repair_attempts" not in analysis and (extra or {}).get("repair_budget") is not None:
                analysis["max_repair_attempts"] = int((extra or {}).get("repair_budget") or 0)
            record["failure_analysis"] = analysis
    return _save_round(base_dir, task_id, record)


def record_metric_completed_result(
    *,
    base_dir: str,
    task_id: str,
    metrics: Dict[str, Any],
    objective_metric: str,
    metric_result: Optional[Dict[str, Any]] = None,
    round_id: Optional[int] = None,
    research_id: Optional[str] = None,
    candidate_snapshot_id: Optional[str] = None,
    patch_path: Optional[Any] = None,
    changed_files: Optional[List[str]] = None,
    source_ref: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Synchronize a completed metric result onto its formal round record.

    This intentionally addresses rounds by ``research_id``/``round_id`` instead
    of using ``current_round()``.  A prior probe/experiment failure may already
    have put the record in a terminal state; once the canonical metric runner
    returns a valid metric, the round is no longer a build/runtime failure and
    the metric evidence must be the owner of the displayed status.
    """
    label = str(research_id or "").strip()
    if label:
        record = _get_round_by_label(base_dir, task_id, label)
        if not record:
            raise RuntimeError(f"missing explicit research round record: {label}")
    elif round_id is not None:
        record = _get_round(base_dir, task_id, int(round_id))
        if not record:
            raise RuntimeError(f"missing explicit round record: {int(round_id)}")
    else:
        record = current_round(base_dir, task_id) or {}
        if not record:
            raise RuntimeError("no round record found for completed metric result")

    if round_id is not None and int(record.get("round_id") or 0) != int(round_id):
        raise RuntimeError(
            f"metric result round_id mismatch: expected {round_id}, got {record.get('round_id')}"
        )
    if label and str(record.get("research_id") or "") != label:
        raise RuntimeError(
            f"metric result research_id mismatch: expected {label}, got {record.get('research_id')}"
        )

    normalized_metrics = dict(metrics or {})
    objective = str(objective_metric or "").strip()
    objective_value = normalized_metrics.get(objective)
    if not (
        objective
        and isinstance(objective_value, (int, float))
        and not isinstance(objective_value, bool)
        and math.isfinite(float(objective_value))
    ):
        raise RuntimeError(f"completed metric result missing valid objective metric: {objective or '<empty>'}")

    record["metrics"] = normalized_metrics
    record["candidate_metrics"] = normalized_metrics
    record["metrics_source"] = "metric_completed"
    record["objective_metric"] = objective
    if candidate_snapshot_id:
        record["candidate_snapshot_id"] = str(candidate_snapshot_id)
    if patch_path is not None:
        record["patch_path"] = str(patch_path)
    if changed_files is not None:
        record["changed_files"] = list(changed_files)
    if source_ref:
        record["source_ref"] = dict(source_ref)

    extra = record.setdefault("close_extra", {})
    if metric_result:
        extra["metric_result"] = dict(metric_result)
    if candidate_snapshot_id:
        extra["candidate_snapshot_id"] = str(candidate_snapshot_id)
    if patch_path is not None:
        extra["patch_path"] = str(patch_path)
    if changed_files is not None:
        extra["changed_files"] = list(changed_files)
    if source_ref:
        extra["source_ref"] = dict(source_ref)

    gate_decision = str(record.get("gate_decision") or "").strip()
    gate_event = dict(record.get("gate_event") or {})
    metric_payload = dict(metric_result or {})
    raw_payload = dict(metric_payload.get("raw_payload") or {})
    raw_result = dict(raw_payload.get("result") or {})
    metric_gate_event = (
        metric_payload.get("gate_event")
        or metric_payload.get("auto_gate")
        or raw_payload.get("gate_event")
        or raw_payload.get("auto_gate")
        or raw_result.get("gate_event")
        or raw_result.get("auto_gate")
    )
    if not gate_event and isinstance(metric_gate_event, dict):
        gate_event = dict(metric_gate_event)
        record["gate_event"] = gate_event
    metric_gate = (
        metric_payload.get("gate")
        or raw_payload.get("gate")
        or raw_result.get("gate")
        or gate_event.get("gate")
    )
    if metric_gate and "gate" not in record and isinstance(metric_gate, dict):
        record["gate"] = dict(metric_gate)
    if not gate_decision:
        gate_decision = str(gate_event.get("decision") or "").strip()
    if not gate_decision and isinstance(metric_gate, dict):
        gate_decision = str(metric_gate.get("decision") or "").strip()
    if gate_decision:
        record["gate_decision"] = gate_decision

    promoted = bool(
        record.get("promoted_to_current_best") is True
        or (dict(record.get("seed_eval") or {}).get("promoted_to_current_best") is True)
        or gate_event.get("promoted_to_current_best") is True
    )
    record["promoted_to_current_best"] = promoted

    if gate_decision in {"reject", "rejected"}:
        record["status"] = "rejected"
        record["phase"] = PHASE_CLOSED
        record["close_reason"] = "metric completed and gate rejected the candidate"
        record["failure_kind"] = FAILURE_KIND_SCIENTIFIC
        record["closed_at"] = _now()
    elif gate_decision == "marginal_no_seed_eval":
        record["status"] = "marginal_no_seed_eval"
        record["phase"] = PHASE_CLOSED
        record["close_reason"] = "single-seed improvement below seed-eval threshold; candidate did not promote current_best"
        record["failure_kind"] = FAILURE_KIND_SCIENTIFIC
        record["closed_at"] = _now()
    elif gate_decision in GATE_RESOLUTION_DECISIONS:
        record["status"] = "gate_checked"
        record["phase"] = PHASE_EXPERIMENT
        record["close_reason"] = ""
        record.pop("closed_at", None)
        record.pop("failure_kind", None)
    elif gate_decision == "accept" and not promoted:
        record["status"] = "gate_checked"
        record["phase"] = PHASE_EXPERIMENT
        record["close_reason"] = "metric completed and gate accepted single-seed candidate; current_best promotion still requires seed evaluation"
        record.pop("closed_at", None)
        record.pop("failure_kind", None)
    else:
        record["status"] = "completed"
        record["phase"] = PHASE_CLOSED
        record["close_reason"] = (
            "metric completed and gate accepted the candidate"
            if gate_decision == "accept"
            else "metric completed"
        )
        record.pop("failure_kind", None)
        record["closed_at"] = _now()

    return _save_round(base_dir, task_id, record)


def commit_current_round_to_variant_attempt(
    *,
    base_dir: str,
    task_id: str,
    attempt: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    del base_dir, task_id, attempt
    raise RuntimeError(
        "legacy pending-build Research commit has been removed; "
        "use the new BuildContract ledger when reintroducing research rounds"
    )

def record_seed_eval_result(
    *,
    base_dir: str,
    task_id: str,
    variant_path: Optional[str],
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Attach seed-eval evidence to the matching gate-review round and close it."""
    node_id = str(result.get("node_id") or "").strip()
    if node_id:
        for existing in reversed(list_rounds(base_dir, task_id)):
            seed_eval = dict(existing.get("seed_eval") or {})
            if str(seed_eval.get("node_id") or "") == node_id:
                return existing
    canonical_variant_path = _canonical_variant_path_text(variant_path)
    candidates = []
    for record in list_rounds(base_dir, task_id):
        if record.get("gate_decision") not in GATE_RESOLUTION_DECISIONS:
            continue
        if record.get("status") in TERMINAL_ROUND_STATUSES:
            continue
        if canonical_variant_path and not _same_path_text(record.get("variant_path"), canonical_variant_path):
            continue
        candidates.append(record)
    if not candidates:
        return None
    record = candidates[-1]
    record = round_transitions.apply_seed_result(
        record,
        result,
        recorded_at=_now(),
    )
    return _save_round(base_dir, task_id, record)


def record_seed_eval_failure(
    *,
    base_dir: str,
    task_id: str,
    variant_path: Optional[str],
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Attach failed seed-eval evidence to the matching gate-review round and close it."""
    node_id = str(result.get("node_id") or "").strip()
    if node_id:
        for existing in reversed(list_rounds(base_dir, task_id)):
            seed_eval = dict(existing.get("seed_eval") or {})
            if str(seed_eval.get("node_id") or "") == node_id:
                return existing
    canonical_variant_path = _canonical_variant_path_text(variant_path)
    candidates = []
    for record in list_rounds(base_dir, task_id):
        if record.get("gate_decision") not in GATE_RESOLUTION_DECISIONS:
            continue
        if record.get("status") in TERMINAL_ROUND_STATUSES:
            continue
        if canonical_variant_path and not _same_path_text(record.get("variant_path"), canonical_variant_path):
            continue
        candidates.append(record)
    if not candidates:
        return None
    record = candidates[-1]
    record = round_transitions.apply_seed_failure(
        record,
        result,
        recorded_at=_now(),
    )
    return _save_round(base_dir, task_id, record)


def record_variant_written(
    *,
    base_dir: str,
    task_id: str,
    variant_path: str,
    fit_point: Any = None,
    hypothesis: Any = None,
    evidence_source: Any = None,
    model_family: Any = None,
    operation_family: Any = None,
    research_direction: Any = None,
    research_intent: Any = None,
    borrowed_from: Any = None,
    mechanism_summary: Any = None,
    expected_effect: Any = None,
    shape_proof: Any = None,
    tensor_ledger: Any = None,
    task_semantics_unchanged: Any = None,
    training_policy_unchanged: Any = None,
    model_structure_cache: Any = None,
) -> Dict[str, Any]:
    canonical_variant_path = _canonical_variant_path_text(variant_path)
    variant_scope = ROUND_SCOPE_BASELINE_DIAGNOSIS if _is_ablation_variant_path(canonical_variant_path) else ROUND_SCOPE_RESEARCH
    variant_counts_toward_budget = variant_scope == ROUND_SCOPE_RESEARCH
    if variant_counts_toward_budget:
        _enforce_max_rounds(base_dir, task_id)
    record = current_round(base_dir, task_id)
    if record:
        existing_variant = record.get("variant_path")
        orchestrated_repair_active = (record.get("orchestrated_repair") or {}).get("active")
        if existing_variant and not _same_path_text(existing_variant, canonical_variant_path) and not orchestrated_repair_active:
            raise RuntimeError(
                "current round already has a different variant; "
                "close the round before writing a new variant"
            )
        record["variant_path"] = canonical_variant_path
        record["fit_point"] = fit_point or record.get("fit_point")
        record["hypothesis"] = hypothesis or record.get("hypothesis")
        record["evidence_source"] = evidence_source or record.get("evidence_source")
        record["model_family"] = model_family or record.get("model_family")
        record["operation_family"] = operation_family or record.get("operation_family")
        record["research_direction"] = research_direction or record.get("research_direction")
        record["research_intent"] = research_intent or record.get("research_intent") or record.get("hypothesis")
        if borrowed_from is not None:
            record["borrowed_from"] = list(borrowed_from or [])
        record["mechanism_summary"] = mechanism_summary or record.get("mechanism_summary")
        record["expected_effect"] = expected_effect or record.get("expected_effect")
        record["shape_proof"] = shape_proof or record.get("shape_proof")
        record["tensor_ledger"] = tensor_ledger or record.get("tensor_ledger")
        if task_semantics_unchanged is not None:
            record["task_semantics_unchanged"] = bool(task_semantics_unchanged)
        if training_policy_unchanged is not None:
            record["training_policy_unchanged"] = bool(training_policy_unchanged)
        record["model_structure_cache"] = model_structure_cache or record.get("model_structure_cache")
        record["round_scope"] = variant_scope
        record["counts_toward_research_budget"] = variant_counts_toward_budget
        record["status"] = "variant_written"
        saved = _save_round(base_dir, task_id, record)
        _set_current_round_id(base_dir, task_id, int(saved["round_id"]), label=str(saved.get("research_id") or ""))
        return saved
    raise RuntimeError(
        "variant_written requires a committed ExperimentAttempt; "
        "use the BuildContract coding-agent worktree as the sole workspace-write boundary"
    )


def _failure_analysis(
    *,
    run: Dict[str, Any],
    failures: List[Dict[str, Any]],
    repair_count: int,
    max_repair_attempts: int,
    terminal_reason: str = "",
) -> Dict[str, Any]:
    signature = dict(run.get("failure_signature") or {})
    signature_id = str(signature.get("signature_id") or "")
    same_signature_count = 0
    if signature_id:
        same_signature_count = sum(
            1
            for item in failures
            if str(((item.get("failure_signature") or {}).get("signature_id") or "")) == signature_id
        )
    action_contract = dict(run.get("action_contract") or {})
    next_required_action = "start_next_research_round" if terminal_reason else "failure_analysis_then_exact_edits"
    if not terminal_reason and str(run.get("stage") or "") == "variant_write" and not str(run.get("variant_path") or "").strip():
        next_required_action = "rewrite_variant_from_round_metadata"
    return {
        "latest_run_id": run.get("run_id"),
        "error_type": run.get("error_type"),
        "failure_signature": signature,
        "failure_evidence": run.get("failure_evidence") or {},
        "same_signature_count": same_signature_count,
        "repair_count": repair_count,
        "max_repair_attempts": max_repair_attempts,
        "repair_budget_remaining": max(0, max_repair_attempts - repair_count),
        "terminal_reason": terminal_reason,
        "next_required_action": next_required_action,
        "action_contract": action_contract,
        "required_wrapper_paths": action_contract.get("required_wrapper_paths") or [],
        "minimal_template": action_contract.get("minimal_template") or "",
    }


def _terminal_failure_status(record: Dict[str, Any], latest_run: Dict[str, Any]) -> str:
    stage = str(latest_run.get("stage") or "")
    if stage == "experiment" or _has_experiment_run(record):
        return "experiment_failed"
    if stage == "runtime_probe":
        return "runtime_probe_failed"
    if stage == "variant_write":
        error_type = str(latest_run.get("error_type") or "")
        if error_type in {"static_contract_error", "syntax_error", "import_error", "compile_error"}:
            return "compile_failed"
        if error_type in CRITICAL_VARIANT_WRITE_ERRORS:
            return "write_failed"
        return "edit_apply_failed"
    if stage == "smoke":
        error_type = str(latest_run.get("error_type") or "")
        if error_type in {"infra_error", "framework_protocol_error", "session_contract_error"}:
            return "infra_failed"
        return "smoke_failed"
    return "runtime_probe_failed"


def _round_limits(base_dir: str) -> Dict[str, int]:
    policy = round_control_policy(base_dir)
    return {
        "max_repair_attempts": positive_int(policy.get("max_repair_attempts"), 2),
        "repeated_failure_signature_limit": positive_int(policy.get("repeated_failure_signature_limit"), 2),
        "critical_variant_write_limit": positive_int(policy.get("critical_variant_write_limit"), 2),
    }


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _same_path_text(left: Any, right: Any) -> bool:
    return _canonical_variant_path_text(left) == _canonical_variant_path_text(right)


def _canonical_variant_path_text(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/").lstrip("./")
    if not text:
        return ""
    marker = "/round_sources/"
    lowered = text.lower()
    if lowered.startswith("round_sources/"):
        return text
    index = lowered.find(marker)
    if index >= 0:
        return text[index + 1 :]
    return text


def _large_variant_diff_count(run: Dict[str, Any]) -> Optional[int]:
    if str(run.get("stage") or "") != "variant_write":
        return None
    if str(run.get("error_type") or "") != "VARIANT_DIFF_TOO_LARGE":
        return None
    review = dict(run.get("validation_review") or {})
    candidates = [
        review,
        dict(review.get("variant_review") or {}),
    ]
    for item in candidates:
        count = _as_int(item.get("changed_line_count"))
        if count is not None:
            return count
    return None


def _large_variant_diff_escalating(failures: List[Dict[str, Any]]) -> bool:
    counts = [count for count in (_large_variant_diff_count(item) for item in failures) if count is not None]
    if len(counts) < 3:
        return False
    recent = counts[-3:]
    return recent[0] < recent[1] < recent[2]


def _apply_failure_transition(
    record: Dict[str, Any],
    latest_run: Dict[str, Any],
    *,
    base_dir: str,
    allow_terminal_transition: bool = True,
) -> Dict[str, Any]:
    limits = _round_limits(base_dir)
    runs = list(record.get("runs") or [])
    failures = [item for item in runs if item.get("status") == "failed"]
    repair_count = max(0, len(failures) - 1)
    record["repair_count"] = repair_count

    terminal_reason = ""
    signature = dict(latest_run.get("failure_signature") or {})
    signature_id = str(signature.get("signature_id") or "")
    if _large_variant_diff_escalating(failures):
        terminal_reason = "variant_diff_escalating"
    if not terminal_reason and signature_id:
        same_signature_count = sum(
            1
            for item in failures
            if str(((item.get("failure_signature") or {}).get("signature_id") or "")) == signature_id
        )
        if same_signature_count >= limits["repeated_failure_signature_limit"]:
            terminal_reason = "repeated_failure_signature"
    if not terminal_reason and repair_count >= limits["max_repair_attempts"]:
        terminal_reason = "repair_budget_exhausted"
    if not terminal_reason and str(latest_run.get("stage") or "") == "variant_write":
        critical_count = sum(
            1
            for item in failures
            if str(item.get("stage") or "") == "variant_write"
            and str(item.get("error_type") or "") in CRITICAL_VARIANT_WRITE_ERRORS
        )
        if critical_count >= limits["critical_variant_write_limit"]:
            terminal_reason = "repeated_variant_write_block"

    record["failure_analysis"] = _failure_analysis(
        run=latest_run,
        failures=failures,
        repair_count=repair_count,
        max_repair_attempts=limits["max_repair_attempts"],
        terminal_reason=terminal_reason,
    )
    if terminal_reason and allow_terminal_transition:
        record["status"] = _terminal_failure_status(record, latest_run)
    elif str(latest_run.get("stage") or "") == "variant_write" and not str(record.get("variant_path") or "").strip():
        record["status"] = "variant_write_blocked"
    else:
        record["status"] = "repair_needed"
    return record


def record_variant_write_failure(
    *,
    base_dir: str,
    task_id: str,
    variant_path: Optional[str],
    error_type: Any = None,
    error_message: str = "",
    failure_signature: Optional[Dict[str, Any]] = None,
    action_contract: Optional[Dict[str, Any]] = None,
    validation_review: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    record = current_round(base_dir, task_id)
    if not record:
        return None
    if variant_path and record.get("variant_path") and not _same_path_text(record.get("variant_path"), variant_path):
        raise RuntimeError(
            f"variant write failure path {variant_path} does not match current round variant {record.get('variant_path')}"
        )
    if variant_path and not record.get("variant_path"):
        record["variant_path"] = _canonical_variant_path_text(variant_path)
    runs = list(record.get("runs") or [])
    run = {
        "run_id": f"variant_write_{len(runs) + 1}",
        "variant_path": _canonical_variant_path_text(variant_path),
        "status": "failed",
        "error_type": error_type or "static_contract_error",
        "error_message": str(error_message or "")[:1000],
        "failure_signature": dict(failure_signature or {}),
        "action_contract": dict(action_contract or {}),
        "validation_review": dict(validation_review or {}),
        "stage": "variant_write",
        "created_at": _now(),
    }
    runs.append(run)
    record["runs"] = runs
    record = _apply_failure_transition(record, run, base_dir=base_dir)
    # Keep the round repairable when the write failure already identified a
    # concrete candidate variant path. This is the orchestrator's signal that
    # the exact-edits write path can still be revised in-place without forcing
    # the controller into the next-round policy gate.
    if (
        str(latest_variant := (variant_path or record.get("variant_path") or "")).strip()
        and record.get("status") == "variant_write_blocked"
    ):
        record["variant_path"] = latest_variant
        record["status"] = "repair_needed"
    return _save_round(base_dir, task_id, record)


def record_experiment_run(
    *,
    base_dir: str,
    task_id: str,
    run_id: str,
    variant_path: Optional[str],
    status: str,
    error_type: Any = None,
    metrics: Optional[Dict[str, Any]] = None,
    failure_signature: Optional[Dict[str, Any]] = None,
    failure_evidence: Optional[Dict[str, Any]] = None,
    error_message: str = "",
    stage: str = "experiment",
    allow_terminal_transition: bool = True,
) -> Optional[Dict[str, Any]]:
    run_label = str(run_id or "").strip()
    if run_label:
        for existing in reversed(list_rounds(base_dir, task_id)):
            if any(
                str(item.get("run_id") or "") == run_label
                for item in list(existing.get("runs") or [])
            ):
                return existing
    record = current_round(base_dir, task_id)
    if not record:
        return None
    if variant_path and record.get("variant_path") and not _same_path_text(record.get("variant_path"), variant_path):
        raise RuntimeError(
            f"run variant {variant_path} does not match current round variant {record.get('variant_path')}; close/start a round first"
        )
    if variant_path and not record.get("variant_path"):
        record["variant_path"] = _canonical_variant_path_text(variant_path)
    run = {
        "run_id": run_id,
        "variant_path": _canonical_variant_path_text(variant_path),
        "status": status,
        "error_type": error_type,
        "metrics": metrics or {},
        "failure_signature": dict(failure_signature or {}),
        "failure_evidence": dict(failure_evidence or {}),
        "error_message": str(error_message or "")[:1000],
        "stage": str(stage or "experiment"),
        "created_at": _now(),
    }
    record = round_transitions.apply_experiment_attempt(
        record,
        run,
        failure_transition=lambda current, failed: _apply_failure_transition(
            current,
            failed,
            base_dir=base_dir,
            allow_terminal_transition=allow_terminal_transition,
        ),
    )
    return _save_round(base_dir, task_id, record)


def record_gate_event(
    *,
    base_dir: str,
    task_id: str,
    gate_event: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    run_id = str(gate_event.get("run_id") or "").strip()
    if run_id:
        for existing in reversed(list_rounds(base_dir, task_id)):
            existing_gate = dict(existing.get("gate_event") or {})
            if str(existing_gate.get("run_id") or "") == run_id:
                return existing
    record = current_round(base_dir, task_id)
    if not record:
        return None
    canonical_gate_event = dict(gate_event or {})
    if canonical_gate_event.get("variant_path"):
        canonical_gate_event["variant_path"] = _canonical_variant_path_text(canonical_gate_event.get("variant_path"))
    record = round_transitions.apply_gate_event(record, canonical_gate_event)
    saved = _save_round(base_dir, task_id, record)
    return saved


def round_progress(base_dir: str, task_id: str) -> Dict[str, Any]:
    records = list_rounds(base_dir, task_id)
    return project_round_progress(
        records,
        current_round_id=_current_round_id(base_dir, task_id),
        current_research_id=_current_round_label(base_dir, task_id),
    )


def latest_round_needing_progress(base_dir: str, task_id: str) -> Optional[Dict[str, Any]]:
    record = current_round(base_dir, task_id)
    if not record:
        return None
    if record.get("status") == "repair_needed" and record.get("gate_decision") != "accept":
        return record
    if int(record.get("repair_count") or 0) >= 1 and record.get("gate_decision") != "accept":
        return record
    runs = list(record.get("runs") or [])
    failed_runs = [item for item in runs if item.get("status") == "failed"]
    if len(failed_runs) >= 2 and record.get("gate_decision") != "accept":
        return record
    return None
