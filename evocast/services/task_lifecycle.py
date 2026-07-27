"""Task-creation services independent of the wizard interaction layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from evocast.domain.knowledge_paths import task_knowledge_dir
from evocast.scripts.init_task import init_task


def write_compiled_config(task_id: str, config: Dict[str, Any], base_dir: Path) -> Path:
    """Persist the compiled task protocol before a task is initialized."""
    knowledge_dir = task_knowledge_dir(str(base_dir), task_id)
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    output_path = knowledge_dir / "compiled_config.json"
    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def initialize_task(
    *,
    task_id: str,
    config_path: Path,
    objective_metric: str,
    budget: str,
    metric_direction: str,
    target_value: Optional[float],
    max_rounds: int,
    max_debug_depth: int,
    api_config_name: str,
    intent: Dict[str, Any],
    runtime_base: Path,
) -> Dict[str, Any]:
    """Create a canonical TaskConfig from wizard-approved task semantics."""
    return init_task(
        task_id=task_id,
        config_path=str(config_path),
        objective_metric=objective_metric,
        budget=budget,
        metric_direction=metric_direction,
        target_value=target_value,
        max_rounds=max_rounds,
        max_debug_depth=max_debug_depth,
        api_config=api_config_name,
        baseline_strategy=str(intent.get("baseline_strategy") or "auto"),
        baseline_models=list(intent.get("baseline_models") or []),
        build_mode=bool(intent.get("build_mode")),
        dataset_diagnosis_mode=str(intent.get("dataset_diagnosis_mode") or "required"),
        baseline_diagnosis_max_ablation_targets=int(intent.get("baseline_diagnosis_max_ablation_targets") or 0),
        agent_ablation=str(intent.get("agent_ablation") or "none"),
        research_intent=str(intent.get("research_intent") or ""),
        force_full_rounds=True,
        language=str(intent.get("language") or "zh"),
        base_dir=str(runtime_base),
    )
