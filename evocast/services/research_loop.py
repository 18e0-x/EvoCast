"""Shared BuildContract creation and formal Research execution loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

from evocast.build.contract import BuildContract
from evocast.build.contract_compiler import build_research_contract, next_research_id
from evocast.research.idea_planner import run_scientist_critic_planner
from evocast.state.domain_store import load_task_config, save_build_contract, save_task_config
from evocast.state.runtime.store import load_runtime_state


@dataclass(frozen=True)
class ResearchContractRequest:
    task_id: str
    objective_metric: str
    api_config: str
    language: str = "zh"
    idea_planner: str = "research_program_hypothesis_competition"
    diagnosis: Dict[str, Any] | None = None
    explicit_contract: str = ""


class ResearchContractService:
    """Create or import one canonical BuildContract for the next round."""

    def __init__(
        self,
        *,
        base_dir: str,
        project_root: Path,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.project_root = project_root
        self.progress = progress or (lambda _message: None)

    @staticmethod
    def _english(language: str) -> bool:
        return str(language or "").strip().lower() in {"en", "english"}

    def _import_explicit(self, request: ResearchContractRequest) -> str:
        path = Path(request.explicit_contract).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        if not path.is_file():
            raise FileNotFoundError(f"--build-contract not found: {path}")
        contract = BuildContract.from_json_file(path)
        contract.validate()
        save_build_contract(self.base_dir, request.task_id, contract.research_id, contract.to_dict())
        return contract.research_id

    def create(self, request: ResearchContractRequest) -> str:
        if request.explicit_contract:
            return self._import_explicit(request)

        state = load_runtime_state(self.base_dir, request.task_id, auto_migrate=False)
        baseline = state.baseline.to_dict() if state.baseline.candidate_id else {}
        if not baseline:
            raise RuntimeError("BuildContract generation requires an active baseline in runtime_state.")

        planner = str(request.idea_planner or "research_program_hypothesis_competition").strip().lower()
        research_id = next_research_id(self.base_dir, request.task_id)
        direction: Dict[str, Any] = {}
        if planner not in {"", "none", "off", "disabled"}:
            if planner not in {"scientist_critic", "research_program_hypothesis_competition"}:
                raise ValueError(
                    f"Unsupported --idea-planner: {planner}. "
                    "Use research_program_hypothesis_competition, scientist_critic, or none."
                )
            label = (
                "Research-Program Hypothesis-Competition"
                if planner == "research_program_hypothesis_competition"
                else "Scientist-Critic"
            )
            self.progress(
                f"[Wizard] Running {label} planner for {research_id}."
                if self._english(request.language)
                else f"[向导] 正在为 {research_id} 运行 {label} idea planner。"
            )
            direction = run_scientist_critic_planner(
                base_dir=self.base_dir,
                task_id=request.task_id,
                research_id=research_id,
                api_config=request.api_config,
                planner_architecture=planner,
            )
            self.progress(
                f"[Wizard] {label} idea: {direction.get('terminal_display_title')}"
                if self._english(request.language)
                else f"[向导] {label} idea：{direction.get('terminal_display_title')}"
            )

        contract = build_research_contract(
            base_dir=self.base_dir,
            task_id=request.task_id,
            baseline=baseline,
            objective_metric=request.objective_metric,
            api_config=request.api_config,
            diagnosis=dict(request.diagnosis or {}),
            research_id=research_id,
            research_direction=direction,
        )
        save_build_contract(self.base_dir, request.task_id, contract.research_id, contract.to_dict())
        task_config = load_task_config(self.base_dir, request.task_id)
        task_config["build_contract_research_id"] = contract.research_id
        save_task_config(self.base_dir, request.task_id, task_config)
        self.progress(
            f"[Wizard] BuildContract generated: {contract.research_id}"
            if self._english(request.language)
            else f"[向导] BuildContract 已生成：{contract.research_id}"
        )
        return contract.research_id


@dataclass(frozen=True)
class ResearchLoopRequest:
    task_id: str
    objective_metric: str
    api_config: str
    max_rounds: int
    language: str = "zh"
    idea_planner: str = "research_program_hypothesis_competition"
    agent_ablation: str = "none"
    diagnosis: Dict[str, Any] | None = None
    explicit_contract: str = ""


@dataclass
class ResearchLoopDependencies:
    round_progress: Callable[[str, str], Dict[str, Any]]
    create_contract: Callable[[ResearchContractRequest], str]
    run_agent: Callable[[str, str, str], Any]


class ResearchLoopService:
    """Run formal Research rounds until budget exhaustion or a durable open round."""

    def __init__(self, *, base_dir: str, dependencies: ResearchLoopDependencies) -> None:
        self.base_dir = base_dir
        self.d = dependencies

    def run(self, request: ResearchLoopRequest) -> Dict[str, Any]:
        while True:
            progress = self.d.round_progress(self.base_dir, request.task_id)
            if int(progress.get("terminal_rounds") or 0) >= int(request.max_rounds):
                return progress
            if int(progress.get("research_open_rounds") or 0) > 0:
                return progress
            if request.explicit_contract and int(progress.get("terminal_rounds") or 0) > 0:
                raise RuntimeError(
                    "A fresh BuildContract is required for each Research round; "
                    "explicit --build-contract can launch only its matching round."
                )
            research_id = self.d.create_contract(
                ResearchContractRequest(
                    task_id=request.task_id,
                    objective_metric=request.objective_metric,
                    api_config=request.api_config,
                    language=request.language,
                    idea_planner=request.idea_planner,
                    diagnosis=dict(request.diagnosis or {}),
                    explicit_contract=request.explicit_contract,
                )
            )
            self.d.run_agent(request.task_id, request.api_config, research_id)
            after = self.d.round_progress(self.base_dir, request.task_id)
            if int(after.get("terminal_rounds") or 0) >= int(request.max_rounds):
                return after
            if int(after.get("research_open_rounds") or 0) > 0:
                return after
