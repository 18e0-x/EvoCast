from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evocast.domain.atomic_io import atomic_write_json


def _strings(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values] if values.strip() else []
    result: list[str] = []
    for value in list(values or []):
        text = str(value or "").strip()
        if text:
            result.append(text)
    return result


def _commands(values: Any) -> list[list[str]]:
    result: list[list[str]] = []
    for command in list(values or []):
        if isinstance(command, str):
            parts = [part for part in command.split(" ") if part]
        else:
            parts = [str(part) for part in list(command or []) if str(part)]
        if parts:
            result.append(parts)
    return result


@dataclass(frozen=True)
class BuildContract:
    research_id: str
    base_snapshot_id: str
    semantic_goal: str
    hypothesis: str
    target_model: str
    research_intent: str = ""
    base_candidate_id: str = ""
    base_source_ref: dict[str, Any] = field(default_factory=dict)
    source_mode: str = "patch_current_best"
    execution_authority: str = "source_binding"
    allowed_edit_files: list[str] = field(default_factory=list)
    likely_entrypoints: list[str] = field(default_factory=list)
    discovery_hints: list[str] = field(default_factory=list)
    protected_files: list[str] = field(default_factory=list)
    protected_globs: list[str] = field(default_factory=list)
    high_risk_globs: list[str] = field(default_factory=list)
    allowed_new_file_roots: list[str] = field(default_factory=list)
    forbidden_files: list[str] = field(default_factory=list)
    forbidden_globs: list[str] = field(default_factory=list)
    required_behavior: list[str] = field(default_factory=list)
    forbidden_behavior: list[str] = field(default_factory=list)
    internal_check_commands: list[list[str]] = field(default_factory=list)
    implementation_constraints: list[str] = field(default_factory=list)
    external_verification_commands: list[list[str]] = field(default_factory=list)
    metric_protocol: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 900
    repair_budget: int = 6
    failure_semantics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BuildContract":
        data = dict(payload or {})
        timeout = int(data.get("timeout_seconds") or 900)
        repair_budget = int(data.get("repair_budget", 6))
        return cls(
            research_id=str(data.get("research_id") or "").strip(),
            base_snapshot_id=str(data.get("base_snapshot_id") or "").strip(),
            semantic_goal=str(data.get("semantic_goal") or "").strip(),
            hypothesis=str(data.get("hypothesis") or "").strip(),
            target_model=str(data.get("target_model") or "").strip(),
            research_intent=str(data.get("research_intent") or "").strip(),
            base_candidate_id=str(data.get("base_candidate_id") or "").strip(),
            base_source_ref=dict(data.get("base_source_ref") or {}),
            source_mode=str(data.get("source_mode") or "patch_current_best").strip(),
            execution_authority=str(data.get("execution_authority") or "source_binding").strip(),
            allowed_edit_files=_strings(data.get("allowed_edit_files")),
            likely_entrypoints=_strings(data.get("likely_entrypoints")),
            discovery_hints=_strings(data.get("discovery_hints")),
            protected_files=_strings(data.get("protected_files")),
            protected_globs=_strings(data.get("protected_globs")),
            high_risk_globs=_strings(data.get("high_risk_globs")),
            allowed_new_file_roots=_strings(data.get("allowed_new_file_roots")),
            forbidden_files=_strings(data.get("forbidden_files")),
            forbidden_globs=_strings(data.get("forbidden_globs")),
            required_behavior=_strings(data.get("required_behavior")),
            forbidden_behavior=_strings(data.get("forbidden_behavior")),
            internal_check_commands=_commands(data.get("internal_check_commands")),
            implementation_constraints=_strings(data.get("implementation_constraints")),
            external_verification_commands=_commands(data.get("external_verification_commands")),
            metric_protocol=dict(data.get("metric_protocol") or {}),
            timeout_seconds=max(timeout, 1),
            repair_budget=max(repair_budget, 0),
            failure_semantics=dict(data.get("failure_semantics") or {}),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "BuildContract":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        missing = [
            name
            for name in ("research_id", "base_snapshot_id", "semantic_goal", "hypothesis", "target_model")
            if not str(getattr(self, name) or "").strip()
        ]
        if missing:
            raise ValueError(f"BuildContract missing required fields: {', '.join(missing)}")
        if self.timeout_seconds <= 0:
            raise ValueError("BuildContract.timeout_seconds must be positive")
        if self.repair_budget < 0:
            raise ValueError("BuildContract.repair_budget must be non-negative")
        if self.execution_authority not in {"source_binding", "repo_wide"}:
            raise ValueError("BuildContract.execution_authority must be source_binding or repo_wide")
        if self.source_mode == "patch_current_best" and not self.allowed_edit_files:
            raise ValueError("BuildContract.allowed_edit_files is required for source edits")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target, self.to_dict(), ensure_ascii=False)
        return target
