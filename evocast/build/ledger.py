from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from evocast.build.contract import BuildContract
from evocast.domain.atomic_io import atomic_write_json
from evocast.domain.execution_ids import format_ablation_id, format_research_id
from evocast.domain.knowledge_paths import task_knowledge_dir
from evocast.harness import rounds as round_records
from evocast.state.domain_store import save_build_contract


def _now() -> str:
    return datetime.now().isoformat()


@dataclass(frozen=True)
class OpenRound:
    round_id: int
    research_id: str
    round_dir: Path
    record: dict[str, Any]


class ResearchRoundLedger:
    """Authoritative ledger for committed BuildContract-based research rounds."""

    def __init__(self, *, base_dir: str, task_id: str) -> None:
        self.base_dir = base_dir
        self.task_id = task_id
        self._round_labels: dict[int, str] = {}

    @property
    def task_dir(self) -> Path:
        return task_knowledge_dir(self.base_dir, self.task_id)

    @property
    def rounds_dir(self) -> Path:
        path = self.task_dir / "rounds"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def round_dir(self, round_id: int) -> Path:
        label = self._round_labels.get(int(round_id)) or format_research_id(round_id)
        path = self.rounds_dir / label
        path.mkdir(parents=True, exist_ok=True)
        return path

    def event_path(self, round_id: int) -> Path:
        return self.round_dir(round_id) / "event_ledger.jsonl"

    def append_event(self, round_id: int, event_type: str, payload: dict[str, Any]) -> Path:
        path = self.event_path(round_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": _now(),
            "event_type": event_type,
            "payload": payload,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return path

    def open_round(self, contract: BuildContract) -> OpenRound:
        contract.validate()
        failure_semantics = dict(contract.failure_semantics or {})
        round_kind = str(failure_semantics.get("round_kind") or "").strip()
        is_baseline_diagnosis_ablation = round_kind == "baseline_diagnosis_ablation"
        existing = round_records.list_rounds(self.base_dir, self.task_id)
        if is_baseline_diagnosis_ablation:
            existing_ablation_ids = [
                int(record.get("round_id") or 0)
                for record in existing
                if str(record.get("round_scope") or "") == round_records.ROUND_SCOPE_BASELINE_DIAGNOSIS
            ]
            expected_next = max(existing_ablation_ids or [0]) + 1
            expected_research_id = format_ablation_id(expected_next)
        else:
            existing_research_ids = [
                int(record.get("round_id") or 0)
                for record in existing
                if round_records._round_counts_toward_research_budget(record)
            ]
            expected_next = max(existing_research_ids or [0]) + 1
            expected_research_id = format_research_id(expected_next)
        if contract.research_id != expected_research_id:
            raise RuntimeError(
                "BuildContract research_id does not match the next execution round: "
                f"contract={contract.research_id}, expected={expected_research_id}. "
                "Generate a fresh BuildContract for each execution round."
            )
        round_scope = (
            round_records.ROUND_SCOPE_BASELINE_DIAGNOSIS
            if is_baseline_diagnosis_ablation
            else round_records.ROUND_SCOPE_RESEARCH
        )
        record = round_records.start_round(
            base_dir=self.base_dir,
            task_id=self.task_id,
            fit_point=contract.target_model,
            hypothesis=contract.hypothesis,
            evidence_source={"kind": "build_contract", "semantic_goal": contract.semantic_goal},
            model_family=contract.target_model,
            research_direction=contract.semantic_goal,
            research_intent=contract.research_intent or contract.semantic_goal,
            proposal={"build_contract": contract.to_dict()},
            round_scope=round_scope,
            counts_toward_research_budget=not is_baseline_diagnosis_ablation,
        )
        round_id = int(record["round_id"])
        self._round_labels[round_id] = str(record.get("research_id") or expected_research_id)
        round_dir = self.round_dir(round_id)
        # The contract is domain state; the event ledger references its stable
        # canonical identity instead of creating a second JSON authority.
        save_build_contract(self.base_dir, self.task_id, record["research_id"], contract.to_dict())
        self.append_event(round_id, "round_opened", {"research_id": record["research_id"]})
        return OpenRound(
            round_id=round_id,
            research_id=str(record.get("research_id") or format_research_id(round_id)),
            round_dir=round_dir,
            record=record,
        )

    def record_artifact(self, round_id: int, name: str, payload: dict[str, Any]) -> Path:
        path = self.round_dir(round_id) / name
        atomic_write_json(path, payload, ensure_ascii=False, default=str)
        self.append_event(round_id, "artifact_recorded", {"name": name, "path": str(path)})
        return path

    def record_agent_turn(self, round_id: int, attempt_index: int, payload: dict[str, Any]) -> None:
        self.append_event(
            round_id,
            "agent_turn",
            {"attempt_index": attempt_index, **dict(payload or {})},
        )

    def record_repair_feedback(self, round_id: int, attempt_index: int, payload: dict[str, Any]) -> None:
        round_records.record_repair_attempt(
            base_dir=self.base_dir,
            task_id=self.task_id,
            repair_entry={"attempt_index": attempt_index, **dict(payload or {})},
        )
        self.append_event(
            round_id,
            "repair_feedback",
            {"attempt_index": attempt_index, **dict(payload or {})},
        )

    def _research_id_for_round(self, round_id: int) -> str:
        round_id = int(round_id)
        label = self._round_labels.get(round_id)
        if label:
            return label
        for record in round_records.list_rounds(self.base_dir, self.task_id):
            if int(record.get("round_id") or 0) == round_id:
                label = str(record.get("research_id") or "").strip()
                if label:
                    self._round_labels[round_id] = label
                    return label
        return format_research_id(round_id)

    def close_failed(self, round_id: int, status: str, reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        research_id = self._research_id_for_round(round_id)
        record = round_records.close_current_round(
            base_dir=self.base_dir,
            task_id=self.task_id,
            round_id=round_id,
            research_id=research_id,
            status=status,
            reason=reason,
            extra=extra or {},
        )
        self.append_event(round_id, "round_failed", {"status": status, "reason": reason, "extra": extra or {}})
        return record

    def close_completed(self, round_id: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        research_id = self._research_id_for_round(round_id)
        record = round_records.close_current_round(
            base_dir=self.base_dir,
            task_id=self.task_id,
            round_id=round_id,
            research_id=research_id,
            status="completed",
            reason="accepted_candidate_verified",
            extra=extra or {},
        )
        self.append_event(round_id, "round_completed", {"extra": extra or {}})
        return record
