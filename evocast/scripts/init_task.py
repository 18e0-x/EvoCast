"""Task initialization helpers for evocast.

Task creation is intentionally centralized in the wizard/configurator flow.
This module remains as an internal helper that materializes task directories
after a compiled_config.json has already been produced.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

from evocast.runners.tfb_pipeline_runner import load_config_json
from evocast.domain.knowledge_paths import runtime_root, task_knowledge_dir, task_runs_dir
from evocast.domain.metric_semantics import DEFAULT_OBJECTIVE_METRIC, validate_objective_metric_for_task_mode
from evocast.domain.evaluation_signature import build_evaluation_signature
from evocast.domain.task_identity import validate_compiled_config_semantics
from evocast.policy.experiment_policy import baseline_diagnosis_policy
from evocast.state.domain_store import save_task_config


def init_task(
    task_id: str,
    config_path: str,
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC,
    budget: str = "unified",
    metric_direction: str = "lower_is_better",
    target_value: float | None = None,
    max_rounds: int = 20,
    max_debug_depth: int = 3,
    api_config: str = "",
    baseline_strategy: str = "auto",
    baseline_models: list[str] | None = None,
    build_mode: bool = False,
    dataset_diagnosis_mode: str = "required",
    baseline_diagnosis_max_ablation_targets: int | None = None,
    force_full_rounds: bool = True,
    language: str = "zh",
    agent_ablation: str = "none",
    research_intent: str = "",
    base_dir: str | None = None,
) -> dict:
    """Initialize a EvoCast task.

    Reads the TFB dataset config, normalizes key fields, and writes
    task_config.json under .evocast/task_knowledge/<task_id>/.
    """
    if base_dir is None:
        base_dir = str(runtime_root())
    else:
        base_dir = str(runtime_root(base_dir))
    diagnosis_policy = baseline_diagnosis_policy(base_dir)
    if baseline_diagnosis_max_ablation_targets is None:
        baseline_diagnosis_max_ablation_targets = int(diagnosis_policy.get("max_ablation_targets", 3))
    normalized_baseline_strategy = str(baseline_strategy or "auto").strip().lower()
    if normalized_baseline_strategy not in {"auto", "manual"}:
        raise ValueError(
            "baseline_strategy must be 'auto' or 'manual'; research tasks cannot skip baseline establishment."
        )

    knowledge_dir = str(task_knowledge_dir(base_dir, task_id))
    runs_dir = str(task_runs_dir(base_dir, task_id))
    os.makedirs(knowledge_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(os.path.join(runs_dir, "logs", "api"), exist_ok=True)
    os.makedirs(os.path.join(runs_dir, "logs", "experiments"), exist_ok=True)
    os.makedirs(os.path.join(runs_dir, "variants"), exist_ok=True)
    os.makedirs(os.path.join(runs_dir, "final_model"), exist_ok=True)

    compiled_config_out = os.path.join(knowledge_dir, "compiled_config.json")
    src_config_path = os.path.abspath(config_path)
    dst_config_path = os.path.abspath(compiled_config_out)
    if src_config_path != dst_config_path:
        shutil.copyfile(src_config_path, dst_config_path)

    tfb_config = load_config_json(compiled_config_out)
    task_semantics = validate_compiled_config_semantics(tfb_config)
    resolved_objective_metric = validate_objective_metric_for_task_mode(
        str(objective_metric or task_semantics.get("objective_metric") or DEFAULT_OBJECTIVE_METRIC),
        str(task_semantics.get("task_mode") or ""),
    )

    data_config = tfb_config.get("data_config", {})
    model_config = tfb_config.get("model_config", {})
    evaluation_config = tfb_config.get("evaluation_config", {})
    feature_dict = data_config.get("feature_dict", {})
    strategy_args = evaluation_config.get("strategy_args", {})
    recommended_hp = model_config.get("recommend_model_hyper_params", {})
    horizon = data_config.get("horizon", strategy_args.get("horizon"))
    seq_len = data_config.get("seq_len", recommended_hp.get("input_chunk_length"))

    task_config = {
        "task_id": task_id,
        "config_path": dst_config_path,
        "objective_metric": resolved_objective_metric,
        "metric_direction": metric_direction,
        "budget": budget,
        "target_value": target_value,
        "max_rounds": max_rounds,
        "force_full_rounds": bool(force_full_rounds),
        "language": "en" if str(language or "").lower() in {"en", "english"} else "zh",
        "max_debug_depth": max_debug_depth,
        "api_config": api_config or "",
        "baseline_strategy": normalized_baseline_strategy,
        "baseline_models": list(baseline_models or []),
        "build_mode": bool(build_mode),
        "dataset_diagnosis_mode": str(dataset_diagnosis_mode or "required").strip().lower()
        if str(dataset_diagnosis_mode or "required").strip().lower() in {"required", "reuse", "skip"}
        else "required",
        "baseline_diagnosis_max_ablation_targets": max(0, int(baseline_diagnosis_max_ablation_targets)),
        "agent_ablation": str(agent_ablation or "none"),
        "research_intent": str(research_intent or "").strip(),
        "data_set_name": data_config.get("data_set_name", ""),
        "dataset_path": data_config.get("dataset_path", ""),
        "horizon": horizon,
        "seq_len": seq_len,
        "task_semantics": task_semantics,
        "feature_dict": {
            "if_univariate": feature_dict.get("if_univariate"),
            "if_trend": feature_dict.get("if_trend"),
            "has_timestamp": feature_dict.get("has_timestamp"),
            "if_season": feature_dict.get("if_season"),
            "freq": feature_dict.get("freq"),
            "canonical_freq": feature_dict.get("canonical_freq"),
        },
        "created_at": datetime.now().isoformat(),
    }
    task_config["evaluation_signature"] = build_evaluation_signature(
        task_config=task_config,
        compiled_config=tfb_config,
        objective_metric=resolved_objective_metric,
    )

    # ``domain_state.json`` is the sole writable task-definition store.  The
    # old task_config.json remains readable through the compatibility layer but
    # is deliberately not recreated for newly created tasks.
    task_config = save_task_config(base_dir, task_id, task_config)
    config_out = str(task_knowledge_dir(base_dir, task_id) / "domain_state.json")

    print(f"[init_task] Created task '{task_id}'")
    print(f"  Config:       {dst_config_path}")
    print(f"  Objective:    {resolved_objective_metric} ({metric_direction})")
    print(f"  Budget:       {budget}")
    if api_config:
        print(f"  API config:   {api_config}")
    print(f"  Language:     {task_config['language']}")
    print(f"  Build mode:   {'on' if build_mode else 'off'}")
    print(f"  Horizon:      {task_config['horizon']}")
    print(f"  Semantics:    {task_semantics['task_mode']} / {task_semantics['input_variable_topology']} / {task_semantics['prediction_target_selection']}")
    print(f"  Knowledge:    {knowledge_dir}")
    print(f"  Runs:         {runs_dir}")
    print(f"  Task config:  {config_out}")

    return task_config


def main():
    print(
        "[init_task] Direct CLI initialization is disabled. "
        "Create tasks through `python -m evocast.scripts.wizard` so task semantics and task_id stay canonical.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
