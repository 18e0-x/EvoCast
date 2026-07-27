from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from evocast.domain.execution_ids import format_research_id


def _now() -> str:
    return datetime.now().isoformat()


@dataclass
class Candidate:
    candidate_id: str = ""
    candidate_kind: str = ""
    display_name: str = ""
    import_path: str = ""
    adapter: Optional[str] = None
    source: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    node_id: Optional[str] = None
    tier: str = ""
    model_config: Dict[str, Any] = field(default_factory=dict)
    parent_candidate_id: Optional[str] = None
    scientific_status: str = ""
    engineering_status: str = ""
    objective_metric: str = ""
    best_artifact_paths: List[str] = field(default_factory=list)
    family: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metric_semantics: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "Candidate":
        data = dict(payload or {})
        display_name = str(
            data.get("display_name")
            or data.get("best_model_name")
            or data.get("model_name")
            or data.get("best_model")
            or ""
        )
        node_id = data.get("node_id") or data.get("best_node_id")
        metrics = dict(data.get("metrics") or data.get("best_metrics") or {})
        candidate_id = str(data.get("candidate_id") or node_id or display_name or "")
        candidate_kind = str(data.get("candidate_kind") or "").strip()
        if not candidate_kind:
            source = str(data.get("source") or "")
            if source.startswith("baseline") or "baseline" in source or source.startswith("manual_"):
                candidate_kind = "baseline"
            else:
                candidate_kind = "variant"
        model_config = dict(data.get("model_config") or {})
        extras = dict(data)
        for key in (
            "candidate_id",
            "candidate_kind",
            "display_name",
            "best_model_name",
            "model_name",
            "best_model",
            "import_path",
            "adapter",
            "source",
            "metrics",
            "best_metrics",
            "node_id",
            "best_node_id",
            "tier",
            "best_tier",
            "model_config",
            "parent_candidate_id",
            "scientific_status",
            "engineering_status",
            "objective_metric",
            "best_artifact_paths",
            "artifact_paths",
            "family",
            "tags",
            "metric_semantics",
        ):
            extras.pop(key, None)
        return cls(
            candidate_id=candidate_id,
            candidate_kind=candidate_kind,
            display_name=display_name,
            import_path=str(data.get("import_path") or ""),
            adapter=data.get("adapter"),
            source=str(data.get("source") or ""),
            metrics=metrics,
            node_id=node_id,
            tier=str(data.get("tier") or data.get("best_tier") or ""),
            model_config=model_config,
            parent_candidate_id=data.get("parent_candidate_id"),
            scientific_status=str(data.get("scientific_status") or ""),
            engineering_status=str(data.get("engineering_status") or ""),
            objective_metric=str(data.get("objective_metric") or ""),
            best_artifact_paths=list(data.get("best_artifact_paths") or data.get("artifact_paths") or []),
            family=data.get("family"),
            tags=list(data.get("tags") or []),
            metric_semantics=dict(data.get("metric_semantics") or {}),
            extras=extras,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "display_name": self.display_name,
            "model_name": self.display_name,
            "import_path": self.import_path,
            "adapter": self.adapter,
            "source": self.source,
            "metrics": dict(self.metrics or {}),
            "node_id": self.node_id,
            "tier": self.tier,
            "model_config": dict(self.model_config or {}),
            "parent_candidate_id": self.parent_candidate_id,
            "scientific_status": self.scientific_status,
            "engineering_status": self.engineering_status,
            "objective_metric": self.objective_metric,
            "best_artifact_paths": list(self.best_artifact_paths or []),
            "family": self.family,
            "tags": list(self.tags or []),
            "metric_semantics": dict(self.metric_semantics or {}),
        }
        payload.update(dict(self.extras or {}))
        return payload

    @property
    def model_name(self) -> str:
        return self.display_name

    def metric_value(self, metric_name: str) -> Any:
        return (self.metrics or {}).get(metric_name)


BaselineIdentity = Candidate


@dataclass
class ResearchRoundState:
    round_number: int
    round_id: str
    current_stage: str = ""
    status: str = ""
    action: str = ""
    best_variant_id: Optional[str] = None
    best_variant_metric: Optional[float] = None
    degradation_count: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> Optional["ResearchRoundState"]:
        if not payload:
            return None
        data = dict(payload)
        return cls(
            round_number=int(data.get("round_number", 0) or 0),
            round_id=str(data.get("round_id") or format_research_id(int(data.get("round_number", 0) or 0))),
            current_stage=str(data.get("current_stage") or ""),
            status=str(data.get("status") or ""),
            action=str(data.get("action") or ""),
            best_variant_id=data.get("best_variant_id"),
            best_variant_metric=data.get("best_variant_metric"),
            degradation_count=int(data.get("degradation_count", 0) or 0),
            payload=dict(data.get("payload") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeState:
    task_id: str
    mode: str = "exploration"
    objective_metric: str = "mse"
    current_stage: str = ""
    current_stage_status: str = ""
    task_status: str = "created"
    baseline: Candidate = field(default_factory=Candidate)
    current_best: Candidate = field(default_factory=Candidate)
    provisional_best: Optional[Candidate] = None  # single-seed improved, NOT yet multi-seed verified
    executor_config_snapshot: Dict[str, Any] = field(default_factory=dict)
    artifact_bindings: Dict[str, str] = field(default_factory=dict)
    baseline_search_progress: Dict[str, Any] = field(default_factory=dict)
    baseline_diagnosis: Dict[str, Any] = field(default_factory=dict)
    research: Dict[str, Any] = field(default_factory=lambda: {
        "completed_rounds": [],
        "pending_round": None,
        "next_round": 1,
        "next_research_id": format_research_id(1),
        "current_round": None,
        "current_research_id": None,
        "current_round_stage": None,
        "consecutive_debugs": 0,
        "consecutive_no_improvement": 0,
        # P2: Pipeline blocked detection
        "consecutive_pipeline_failures": 0,
        "pipeline_blocked": False,
        "pipeline_blocked_reason": "",
        "last_action": "",
    })
    current_round: Optional[ResearchRoundState] = None
    export: Dict[str, Any] = field(default_factory=dict)
    consistency_warnings: List[str] = field(default_factory=list)
    degradation_chain: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @classmethod
    def empty(cls, task_id: str, objective_metric: str = "mse", mode: str = "exploration") -> "RuntimeState":
        return cls(task_id=task_id, objective_metric=objective_metric, mode=mode)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "RuntimeState":
        data = dict(payload or {})
        task_id = str(data.get("task_id") or "")
        state = cls(
            task_id=task_id,
            mode=str(data.get("mode") or "exploration"),
            objective_metric=str(data.get("objective_metric") or "mse"),
            current_stage=str(data.get("current_stage") or ""),
            current_stage_status=str(data.get("current_stage_status") or ""),
            task_status=str(data.get("task_status") or data.get("status") or "created"),
            baseline=Candidate.from_dict(data.get("baseline")),
            current_best=Candidate.from_dict(data.get("current_best")),
            provisional_best=Candidate.from_dict(data.get("provisional_best")) if data.get("provisional_best") else None,
            executor_config_snapshot=dict(data.get("executor_config_snapshot") or {}),
            artifact_bindings=dict(data.get("artifact_bindings") or {}),
            baseline_search_progress=dict(data.get("baseline_search_progress") or {}),
            baseline_diagnosis=dict(data.get("baseline_diagnosis") or {}),
            research=dict(data.get("research") or {}),
            current_round=ResearchRoundState.from_dict(data.get("current_round")),
            export=dict(data.get("export") or {}),
            consistency_warnings=list(data.get("consistency_warnings", []) or []),
            degradation_chain=list(data.get("degradation_chain", []) or []),
            created_at=str(data.get("created_at") or _now()),
            updated_at=str(data.get("updated_at") or _now()),
        )
        if "research" not in data:
            state.research = cls.empty(task_id).research
        else:
            default_research = cls.empty(task_id).research
            merged = dict(default_research)
            merged.update(dict(data.get("research") or {}))
            state.research = merged
        return state

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["baseline"] = self.baseline.to_dict()
        payload["current_best"] = self.current_best.to_dict()
        payload["provisional_best"] = self.provisional_best.to_dict() if self.provisional_best else None
        if self.current_round is None:
            payload["current_round"] = None
        return payload

    def touch(self) -> None:
        self.updated_at = _now()
