"""Model pack export CLI for evocast.

Usage:
  python -m evocast.scripts.export_model_pack --task-id <id>

Exports the final model pack under evocast/runs/<task_id>/final_model/:
  - variant.py
  - best_config.yaml
  - metrics.json
  - comparison_table.csv
  - trial_journal.jsonl (copy)
  - model_card.md
  - reproduction_commands.md
  - protected_path_report.json
"""

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from evocast.domain.formal_source import copy_formal_source_tree
from evocast.domain.knowledge_paths import runtime_root, task_knowledge_dir, task_runs_dir
from evocast.domain.knowledge_paths import repo_root as project_repo_root
from evocast.variant.protected_paths import generate_protected_path_report
from evocast.state.token_usage import write_token_usage_summary
from evocast.domain.metric_semantics import DEFAULT_OBJECTIVE_METRIC
from evocast.state.runtime.store import load_runtime_state, runtime_events_path, runtime_state_path
from evocast.state.domain_store import load_task_config
from evocast.templates.model_card import render_model_card


def _is_research_node(node: dict) -> bool:
    return str(node.get("action_type") or "") not in {"baseline", "ablation", "seed_eval"}


def _workspace_root_from_variant_path(variant_path: str) -> str:
    normalized = str(variant_path or "").replace("\\", "/").strip()
    if not normalized:
        return ""
    ws_parts = normalized.split("/")
    try:
        if "round_sources" in ws_parts:
            idx = ws_parts.index("round_sources")
            if idx + 2 < len(ws_parts):
                return "/".join(ws_parts[: idx + 3])
        if "sandboxes" in ws_parts and "variant" in ws_parts:
            variant_idx = ws_parts.index("variant")
            return "/".join(ws_parts[: variant_idx + 1])
        if "rounds" in ws_parts and "workspace" in ws_parts:
            workspace_idx = ws_parts.index("workspace")
            return "/".join(ws_parts[: workspace_idx + 1])
    except (ValueError, IndexError):
        return ""
    path = Path(normalized)
    if path.name == "round_entry.py":
        return str(path.parent)
    return ""


def _copy_workspace_source(ws_root: str, final_dir: str, export_files: list[str]) -> dict:
    source_record = {"kind": "round_sources", "workspace_root": ws_root, "copied_files": []}
    if not ws_root or not os.path.isdir(ws_root):
        return source_record
    entry_src = os.path.join(ws_root, "round_entry.py")
    if os.path.exists(entry_src):
        dst = os.path.join(final_dir, "variant.py")
        shutil.copy2(entry_src, dst)
        export_files.append(dst)
        source_record["entry_export"] = dst
    source_dir = os.path.join(final_dir, "source")
    for root, _dirs, files in os.walk(ws_root):
        if "__pycache__" in Path(root).parts:
            continue
        for fname in files:
            if not fname.endswith(".py") or fname == "round_entry.py":
                continue
            src = os.path.join(root, fname)
            rel = os.path.relpath(src, ws_root)
            dst = os.path.join(source_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            export_files.append(dst)
            source_record["copied_files"].append(os.path.join("source", rel).replace("\\", "/"))
    manifest_src = os.path.join(ws_root, "manifest.json")
    if os.path.exists(manifest_src):
        dst = os.path.join(final_dir, "manifest.json")
        shutil.copy2(manifest_src, dst)
        export_files.append(dst)
        source_record["manifest_export"] = dst
    return source_record


def _copy_build_contract_source(selected_node: dict, final_dir: str, export_files: list[str]) -> dict:
    source_ref = dict(selected_node.get("source_ref") or {})
    checkout = str(
        selected_node.get("source_checkout")
        or source_ref.get("source_checkout")
        or ""
    ).strip()
    record = {
        "kind": "source_snapshot",
        "source_checkout": checkout,
        "candidate_snapshot_id": selected_node.get("candidate_snapshot_id") or source_ref.get("candidate_snapshot_id"),
        "base_snapshot_id": selected_node.get("base_snapshot_id") or source_ref.get("base_snapshot_id"),
        "patch_path": selected_node.get("patch_path") or source_ref.get("patch_path"),
        "changed_files": list(selected_node.get("changed_files") or source_ref.get("changed_files") or []),
        "copied_files": [],
    }
    if checkout and Path(checkout).is_dir() and (Path(checkout) / "ts_benchmark").is_dir():
        destination = Path(final_dir) / "source_checkout"
        copied = copy_formal_source_tree(
            repo_root=Path(checkout),
            source_root=Path(checkout) / "ts_benchmark",
            destination_root=destination,
            overwrite=True,
        )
        record["copied_files"] = [str(Path("source_checkout") / item["path"]).replace("\\", "/") for item in copied]
        export_files.extend(str(destination / item["path"]) for item in copied)
    patch_path = str(record.get("patch_path") or "")
    if patch_path and os.path.exists(patch_path):
        dst = os.path.join(final_dir, "candidate.patch")
        shutil.copy2(patch_path, dst)
        export_files.append(dst)
        record["patch_export"] = dst
    return record


def _export_candidate_source(selected_node: dict, best_variant_path: str, final_dir: str, export_files: list[str]) -> dict:
    if best_variant_path:
        ws_root = _workspace_root_from_variant_path(best_variant_path)
        if ws_root and os.path.isdir(ws_root):
            return _copy_workspace_source(ws_root, final_dir, export_files)
    source_ref = dict(selected_node.get("source_ref") or {})
    if source_ref or selected_node.get("candidate_snapshot_id") or selected_node.get("source_checkout"):
        return _copy_build_contract_source(selected_node, final_dir, export_files)
    return {"kind": "baseline_or_unavailable", "copied_files": []}


def export_pack(
    task_id: str,
    base_dir: str | None = None,
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC,
) -> dict:
    """Export the final model pack for a task.

    Reads the best known result from the trial journal and assembles
    the complete export package.
    """
    if base_dir is None:
        base_dir = str(runtime_root())

    knowledge_dir = str(task_knowledge_dir(base_dir, task_id))
    runs_dir = str(task_runs_dir(base_dir, task_id))
    final_dir = os.path.join(runs_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)

    export_files: list[str] = []
    target_reached = False
    final_metrics: dict = {}
    best_variant_path = ""
    best_model_name = ""
    best_research_node = None
    best_research_value = None

    task_config = load_task_config(base_dir, task_id)

    runtime_state = load_runtime_state(base_dir, task_id)

    # Load best baseline
    best_baseline = runtime_state.baseline.to_dict() if runtime_state.baseline.model_name else {}

    # Find best successful variant from journal
    journal_path = os.path.join(runs_dir, "trial_journal.jsonl")
    best_node = None
    best_value = None

    if os.path.exists(journal_path):
        nodes = []
        with open(journal_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    node = json.loads(line)
                    nodes.append(node)
                    metrics = node.get("metrics", {})
                    val = metrics.get(objective_metric)
                    if val is not None and node.get("status") == "success":
                        if best_value is None or val < best_value:
                            best_value = val
                            best_node = node
                        if _is_research_node(node) and (best_research_value is None or val < best_research_value):
                            best_research_value = val
                            best_research_node = node

        # Copy journal
        dest_journal = os.path.join(final_dir, "trial_journal.jsonl")
        shutil.copy2(journal_path, dest_journal)
        export_files.append(dest_journal)
    else:
        nodes = []

    # Determine the best model
    current_best = runtime_state.current_best.to_dict() if runtime_state.current_best.candidate_id else {}
    selected_node = None
    selected_value = None
    if current_best:
        selected_node = dict(current_best)
        selected_node["node_id"] = selected_node.get("node_id") or current_best.get("candidate_id")
        selected_node["model_name"] = current_best.get("display_name") or current_best.get("model_name")
        selected_node["variant_path"] = (
            current_best.get("variant_path")
            or dict(current_best.get("model_config") or {}).get("variant_path")
            or ""
        )
        selected_node["metrics"] = dict(current_best.get("metrics") or {})
        selected_node["fit_points"] = list(current_best.get("fit_points") or [])
        selected_node["model_config"] = dict(current_best.get("model_config") or {})
        selected_node["action_type"] = str(current_best.get("tier") or current_best.get("source") or "runtime_state")
        selected_node["status"] = "success"
        selected_value = (selected_node.get("metrics") or {}).get(objective_metric)
        selected_node_id = str(selected_node.get("node_id") or "")
        journal_match = next((node for node in nodes if str(node.get("node_id") or "") == selected_node_id), None)
        if journal_match:
            merged = dict(journal_match)
            merged["model_name"] = selected_node.get("model_name") or merged.get("model_name")
            merged["metrics"] = dict(selected_node.get("metrics") or merged.get("metrics") or {})
            merged["model_config"] = dict(selected_node.get("model_config") or merged.get("model_config") or {})
            selected_node = merged
    else:
        selected_node = best_research_node or best_node
        selected_value = best_research_value if best_research_node is not None else best_value
    if selected_node:
        best_model_name = selected_node.get("model_name", "unknown")
        best_variant_path = selected_node.get("variant_path", "")
        final_metrics = selected_node.get("metrics", {})

        source_export = _export_candidate_source(selected_node, best_variant_path, final_dir, export_files)

        # Write metrics.json
        metrics_path = os.path.join(final_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "task_id": task_id,
                "objective_metric": objective_metric,
                "model_name": best_model_name,
                "metrics": final_metrics,
                "node_id": selected_node.get("node_id"),
                "best_overall_node_id": best_node.get("node_id") if best_node else None,
                "best_research_node_id": best_research_node.get("node_id") if best_research_node else None,
                "variant_path": best_variant_path,
                "source_ref": selected_node.get("source_ref") or {},
                "source_export": source_export,
                "export_source": "research_variant" if best_research_node is not None else "best_overall",
            }, f, indent=2, ensure_ascii=False)
        export_files.append(metrics_path)

        # Check target
        target = task_config.get("target_value")
        if target is not None and selected_value is not None:
            direction = task_config.get("metric_direction", "lower_is_better")
            if direction == "lower_is_better":
                target_reached = selected_value <= target
            else:
                target_reached = selected_value >= target
    else:
        # Fall back to best baseline — only valid if baseline succeeded
        bl_name = best_baseline.get("model_name")
        if not bl_name:
            print(
                f"[export] FATAL: No successful variant or baseline found for task '{task_id}'.",
                file=sys.stderr,
            )
            print(
                "[export] At least one success (baseline or variant) is required.",
                file=sys.stderr,
            )
            sys.exit(1)
        best_model_name = bl_name
        final_metrics = best_baseline.get("metrics", {})

        metrics_path = os.path.join(final_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "task_id": task_id,
                "objective_metric": objective_metric,
                "model_name": best_model_name,
                "metrics": final_metrics,
                "source": "baseline",
            }, f, indent=2, ensure_ascii=False)
        export_files.append(metrics_path)

    # Build comparison table
    comparison_path = os.path.join(final_dir, "comparison_table.csv")
    _build_comparison_table(nodes, best_baseline, objective_metric, comparison_path)
    export_files.append(comparison_path)

    # Write best_config.yaml
    best_config_path = os.path.join(final_dir, "best_config.yaml")
    _write_best_config(selected_node or best_baseline, task_config, best_config_path)
    export_files.append(best_config_path)

    # Render model card
    model_card_path = os.path.join(final_dir, "model_card.md")
    baseline_value = best_baseline.get("metrics", {}).get(objective_metric)
    model_card = render_model_card(
        variant_name=best_model_name,
        dataset=task_config.get("data_set_name", str(task_config.get("config_path", ""))),
        horizon=task_config.get("horizon", "?"),
        objective_metric=objective_metric,
        metric_direction=task_config.get("metric_direction", "lower_is_better"),
        baseline_model=best_baseline.get("model_name", "None"),
        baseline_value=baseline_value,
        variant_path=best_variant_path,
        fit_points=selected_node.get("fit_points", []) if selected_node else [],
        final_value=selected_value,
        num_seeds=sum(1 for n in nodes if n.get("action_type") == "seed_eval"),
        target_reached=target_reached,
        export_source="research_variant" if best_research_node is not None else ("best_overall" if best_node is not None else "baseline"),
        best_overall_variant_path=best_node.get("variant_path", "") if best_node else "",
        best_overall_value=best_value,
    )
    with open(model_card_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    export_files.append(model_card_path)

    # Write reproduction commands
    repro_path = os.path.join(final_dir, "reproduction_commands.md")
    _write_reproduction_commands(task_id, best_model_name, task_config, repro_path)
    export_files.append(repro_path)

    # Write protected path report
    report_path = os.path.join(final_dir, "protected_path_report.json")
    report = generate_protected_path_report(export_files)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    export_files.append(report_path)

    token_summary_path = write_token_usage_summary(task_id, base_dir)
    dest_token_summary = os.path.join(final_dir, "token_usage_summary.json")
    shutil.copy2(token_summary_path, dest_token_summary)
    export_files.append(dest_token_summary)

    for extra_path in (runtime_state_path(base_dir, task_id), runtime_events_path(base_dir, task_id)):
        if os.path.exists(extra_path):
            dest_path = os.path.join(final_dir, os.path.basename(extra_path))
            shutil.copy2(extra_path, dest_path)
            export_files.append(dest_path)

    print(f"[export] Final model pack exported to: {final_dir}")
    for f in export_files:
        print(f"  {os.path.basename(f)}")

    return {
        "task_id": task_id,
        "export_dir": final_dir,
        "best_model": best_model_name,
        "best_value": selected_value,
        "target_reached": target_reached,
        "files": [os.path.basename(f) for f in export_files],
    }


def _build_comparison_table(
    nodes: list[dict],
    best_baseline: dict,
    objective_metric: str,
    output_path: str,
):
    """Build a comparison table CSV of all runs."""
    rows: list[dict] = []

    # Add baseline
    bl_metrics = best_baseline.get("metrics", {})
    if bl_metrics:
        rows.append({
            "model": best_baseline.get("model_name", "baseline"),
            "type": "baseline",
            "status": "success",
            objective_metric: bl_metrics.get(objective_metric, "N/A"),
        })

    # Add all nodes
    for node in nodes:
        metrics = node.get("metrics", {})
        val = metrics.get(objective_metric)
        rows.append({
            "model": node.get("model_name", node.get("node_id", "")),
            "type": node.get("action_type", "variant"),
            "status": node.get("status", "unknown"),
            objective_metric: f"{val:.6f}" if val is not None else "N/A",
        })

    if not rows:
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_best_config(
    best_entry: dict | None,
    task_config: dict,
    output_path: str,
):
    """Write the best model config as YAML."""
    try:
        import yaml
    except ImportError:
        # Fallback to JSON
        config_data = {
            "task_id": task_config.get("task_id", ""),
            "model_name": best_entry.get("model_name", "") if best_entry else "",
            "model_config": best_entry.get("model_config", {}) if best_entry else {},
            "created_at": datetime.now().isoformat(),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        return

    config_data = {
        "task_id": task_config.get("task_id", ""),
        "model_name": best_entry.get("model_name", "") if best_entry else "",
        "model_config": best_entry.get("model_config", {}) if best_entry else {},
        "created_at": datetime.now().isoformat(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)


def _write_reproduction_commands(
    task_id: str,
    model_name: str,
    task_config: dict,
    output_path: str,
):
    """Write reproduction command instructions."""
    semantics = ((task_config.get("task_semantics") or {}) if isinstance(task_config, dict) else {})
    dataset = task_config.get("data_set_name") or task_config.get("dataset") or "<dataset.csv>"
    task_mode = semantics.get("task_mode") or task_config.get("task_mode") or "<MM|MS|SS>"
    horizons = semantics.get("horizons") or [task_config.get("horizon")] if task_config.get("horizon") is not None else []
    input_chunk_length = semantics.get("input_chunk_length") or task_config.get("input_chunk_length")
    target_columns = semantics.get("target_columns") or task_config.get("target_columns") or []
    objective_metric = task_config.get("objective_metric") or task_config.get("objective") or DEFAULT_OBJECTIVE_METRIC
    horizon_arg = ",".join(str(v) for v in horizons if v is not None) or "<horizon>"
    target_arg = target_columns[0] if len(target_columns) == 1 else "all"
    seq_len_arg = input_chunk_length if input_chunk_length is not None else "<seq_len>"
    lines = [
        "# Reproduction Commands",
        "",
        f"## Task: {task_id}",
        f"## Best Model: {model_name}",
        "",
        "## Commands to reproduce",
        "",
        "Task IDs are generated automatically from task semantics and timestamp.",
        "Start from the wizard, note the generated task_id, then reuse that task_id in later commands.",
        "",
        "```bash",
        "# 1. Compile the task through the wizard only",
        f"python -m evocast --dataset {dataset} --task-mode {task_mode} --target {target_arg} --horizons {horizon_arg} --seq-len {seq_len_arg} --objective {objective_metric} --configure-only --yes",
        "",
        "# 2. Build model registry for the generated task",
        "python -m evocast.scripts.build_model_registry --task-id <generated_task_id>",
        "",
        "# 3. Run the BuildContract research path",
        "python -m evocast.scripts.run_agent_v3 --task-id <generated_task_id> --build-contract <build_contract.json>",
        "",
        "# 4. Export final model pack",
        "python -m evocast.scripts.export_model_pack --task-id <generated_task_id>",
        "```",
        "",
        "## Final model location",
        f"`evocast/runs/{task_id}/final_model/`",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export the final model pack for a task.",
    )
    parser.add_argument(
        "--task-id", required=True,
        help="Task identifier.",
    )
    parser.add_argument(
        "--objective", default=DEFAULT_OBJECTIVE_METRIC,
        help="Objective metric (default: mse_norm).",
    )

    args = parser.parse_args()

    try:
        result = export_pack(
            task_id=args.task_id,
            objective_metric=args.objective,
        )
        print(f"\n[export] Done. Target reached: {result['target_reached']}")
        if not result["target_reached"]:
            print("[export] Target was not reached. This is the best verified result.")
    except Exception as e:
        print(f"[export_model_pack] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
