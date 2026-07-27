"""BuildContract-based research workflow entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evocast.build.backends.variant_forge_backend import VariantForgeBackend
from evocast.build.contract import BuildContract
from evocast.build.metadata_writer import ResearchMetadataWriter
from evocast.build.metric_runner import TFBExperimentMetricRunner
from evocast.build.orchestrator import ResearchBuildOrchestrator
from evocast.build.result import BuildDecision
from evocast.domain.knowledge_paths import repo_root
from evocast.harness.session import AgentSession
from evocast.harness.api_client import ProviderClient, resolve_provider_config_path
from evocast.state.runtime.store import sync_task_stage
from evocast.state.domain_store import load_build_contract
from evocast.tools.tfb_seed_eval import run_seed_eval

DISABLED_MESSAGE = (
    "Pass --build-contract to run the BuildContract/CodingAgentBackend research path."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a BuildContract-based EvoCast research build.")
    parser.add_argument("--task-id", default="", help="Task id for runtime artifacts.")
    parser.add_argument("--base-dir", default=str(repo_root()), help="Runtime base directory or repo root.")
    parser.add_argument("--repo-dir", default=str(repo_root()), help="EvoCast source checkout root used for source snapshots.")
    parser.add_argument("--build-contract", default="", help="Legacy compatibility path to build_contract.json.")
    parser.add_argument("--research-id", default="", help="Canonical BuildContract identity for this task.")
    parser.add_argument("--backend", choices=["variant-forge"], default="variant-forge")
    parser.add_argument("--api-config", default="providers/minimax.yaml")
    parser.add_argument("--validate-contract-only", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.build_contract) == bool(args.research_id):
        raise RuntimeError(DISABLED_MESSAGE)
    if not args.task_id:
        raise RuntimeError("--task-id is required for the BuildContract research path")

    contract = (
        BuildContract.from_dict(load_build_contract(args.base_dir, args.task_id, args.research_id))
        if args.research_id
        else BuildContract.from_json_file(args.build_contract)
    )
    contract.validate()
    if args.validate_contract_only:
        print(json.dumps({"status": "valid", "contract": contract.to_dict()}, ensure_ascii=False, indent=2))
        return 0

    config_path = resolve_provider_config_path(args.api_config)
    client = ProviderClient(config_path=config_path, task_id=args.task_id)
    backend = VariantForgeBackend(client=client, user_prompt=contract.research_intent)
    session = AgentSession(task_id=args.task_id, base_dir=args.base_dir, client=client)
    session.ensure_dirs()
    orchestrator = ResearchBuildOrchestrator(
        base_dir=args.base_dir,
        task_id=args.task_id,
        repo_dir=Path(args.repo_dir),
        backend=backend,
        metric_runner=TFBExperimentMetricRunner(session=session),
        seed_eval_runner=lambda seed_args: run_seed_eval(session, seed_args),
        metadata_writer=ResearchMetadataWriter(client=client),
    )
    try:
        outcome = orchestrator.run(contract)
    except Exception as exc:
        sync_task_stage(
            args.base_dir,
            args.task_id,
            stage="build_orchestrator",
            status="infra_failed",
            extra={"research": {"error_type": type(exc).__name__, "error_message": str(exc)}},
        )
        raise
    if outcome.status in {BuildDecision.TERMINAL_REJECTED, BuildDecision.REPAIR_REQUIRED}:
        sync_task_stage(
            args.base_dir,
            args.task_id,
            stage="research_round",
            status="experiment_failed",
            extra={"research": {"last_build_outcome": outcome.to_dict()}},
        )
    else:
        sync_task_stage(
            args.base_dir,
            args.task_id,
            stage="research_round",
            status="completed",
            extra={"research": {"last_build_outcome": outcome.to_dict()}},
        )
    print(json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
