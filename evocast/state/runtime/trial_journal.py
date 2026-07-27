"""Trial journal for evocast.

Stores every attempted baseline and variant as a node in a JSONL file.
The journal is the source of truth for all runs.

Schema follows agent_plan.md Section 7.
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

from evocast.domain.knowledge_paths import runs_root, task_knowledge_dir


DEFAULT_JOURNAL_SCHEMA = {
    "node_id": "",
    "parent_id": None,
    "task_id": "",
    "action_type": "draft",          # draft, debug, improve, seed_eval, baseline
    "model_name": "",
    "variant_path": None,
    "model_config": {},
    "data_config": {},
    "evaluation_config": {},
    "objective_metric": "",
    "metrics": {},
    "status": "pending",             # pending, running, success, failed
    "error_type": None,
    "error_message": None,
    "traceback_path": None,
    "artifact_paths": [],
    "fit_points": [],
    "llm_summary": "",
    "gate_decision": {},
    "review": {},
    "created_at": "",
    "completed_at": None,
}


def _journal_path(task_id: str, base_dir: str = None) -> str:
    if base_dir is None:
        base_dir = str(runs_root())
    base_path = os.path.abspath(str(base_dir))
    if os.path.basename(base_path) == "runs":
        task_dir = os.path.join(base_path, task_id)
    else:
        task_dir = str(runs_root(base_path) / task_id)
    os.makedirs(task_dir, exist_ok=True)
    return os.path.join(task_dir, "trial_journal.jsonl")


def _task_knowledge_path(task_id: str, base_dir: str = None) -> str:
    if base_dir is None:
        task_dir = str(task_knowledge_dir(None, task_id))
        os.makedirs(task_dir, exist_ok=True)
        return task_dir
    base_name = os.path.basename(os.path.abspath(str(base_dir)))
    if base_name == "runs":
        task_dir = str(task_knowledge_dir(os.path.dirname(os.path.abspath(str(base_dir))), task_id))
    else:
        task_dir = str(task_knowledge_dir(str(base_dir), task_id))
    os.makedirs(task_dir, exist_ok=True)
    return task_dir


def create_node(task_id: str, node_id: str, **overrides) -> Dict:
    """Create a new journal node with defaults filled in."""
    node = dict(DEFAULT_JOURNAL_SCHEMA)
    node["node_id"] = node_id
    node["task_id"] = task_id
    node["created_at"] = datetime.now().isoformat()
    for k, v in overrides.items():
        if k in node:
            node[k] = v
    return node


def append_node(task_id: str, node: Dict, base_dir: str = None) -> str:
    """Append a node to the trial journal JSONL file.

    Returns the path to the journal file.
    """
    path = _journal_path(task_id, base_dir)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(node, ensure_ascii=False) + "\n")
    _archive_journal_path(task_id, path, base_dir)
    return path


def replace_node(task_id: str, node: Dict, base_dir: str = None) -> str:
    """Replace an existing journal node by node_id.

    Raises:
        KeyError: when the target node_id does not exist in the journal.
    """
    path = _journal_path(task_id, base_dir)
    nodes = read_journal(task_id, base_dir)
    target_id = node.get("node_id")
    replaced = False
    for index, existing in enumerate(nodes):
        if existing.get("node_id") == target_id:
            nodes[index] = node
            replaced = True
            break
    if not replaced:
        raise KeyError(f"journal node '{target_id}' not found for task '{task_id}'")
    with open(path, "w", encoding="utf-8") as f:
        for item in nodes:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    _archive_journal_path(task_id, path, base_dir)
    return path


def _archive_journal_path(task_id: str, journal_path: str, base_dir: str = None) -> None:
    """Copy the live journal to task_knowledge so runs/ cleanup does not erase memory."""
    try:
        runs_dir = os.path.abspath(base_dir) if base_dir else os.path.abspath(
            str(runs_root())
        )
        tfb_base = os.path.dirname(runs_dir) if os.path.basename(runs_dir) == "runs" else runs_dir
        archive_dir = str(task_knowledge_dir(tfb_base, task_id))
        os.makedirs(archive_dir, exist_ok=True)
        shutil.copy2(journal_path, os.path.join(archive_dir, "archived_trial_journal.jsonl"))
    except Exception:
        # Journaling must never fail because archival failed.
        return


def read_journal(task_id: str, base_dir: str = None) -> List[Dict]:
    """Read all nodes from a trial journal."""
    path = _journal_path(task_id, base_dir)
    if not os.path.exists(path):
        return []
    nodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    nodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return nodes


def latest_nodes_by_id(nodes: List[Dict]) -> List[Dict]:
    """Return the latest record for each node_id while preserving latest order."""
    latest = {}
    order = []
    for node in nodes:
        node_id = node.get("node_id")
        if not node_id:
            order.append(id(node))
            latest[id(node)] = node
            continue
        if node_id not in latest:
            order.append(node_id)
        latest[node_id] = node
    return [latest[key] for key in order if key in latest]


def get_latest_node(task_id: str, base_dir: str = None) -> Optional[Dict]:
    """Get the most recent node from the journal."""
    nodes = read_journal(task_id, base_dir)
    return nodes[-1] if nodes else None


def get_successful_nodes(task_id: str, base_dir: str = None) -> List[Dict]:
    """Get all nodes with status 'success'."""
    return [n for n in latest_nodes_by_id(read_journal(task_id, base_dir)) if n.get("status") == "success"]


def get_failed_nodes(task_id: str, base_dir: str = None) -> List[Dict]:
    """Get all nodes with status 'failed'."""
    return [n for n in latest_nodes_by_id(read_journal(task_id, base_dir)) if n.get("status") == "failed"]


def get_nodes_by_error(task_id: str, error_type: str, base_dir: str = None) -> List[Dict]:
    """Get all nodes with a specific error type."""
    return [n for n in latest_nodes_by_id(read_journal(task_id, base_dir)) if n.get("error_type") == error_type]


def journal_summary(task_id: str, base_dir: str = None) -> Dict:
    """Produce a summary of the journal for context building."""
    raw_nodes = read_journal(task_id, base_dir)
    nodes = latest_nodes_by_id(raw_nodes)
    successes = [n for n in nodes if n.get("status") == "success"]
    experiment_successes = [
        n for n in successes
        if n.get("action_type") not in {"gate"}
    ]
    failures = [n for n in nodes if n.get("status") == "failed"]

    error_counts = {}
    fit_points_tried = set()
    for n in nodes:
        et = n.get("error_type")
        if et and n.get("status") == "failed":
            error_counts[et] = error_counts.get(et, 0) + 1
        for fp in n.get("fit_points", []):
            fit_points_tried.add(fp)

    best = None
    best_value = None
    for n in experiment_successes:
        metrics = n.get("metrics", {})
        obj = n.get("objective_metric", "")
        if obj and obj in metrics:
            v = metrics[obj]
            if best is None or v < best_value:
                best = n
                best_value = v

    return {
        "total_nodes": len(nodes),
        "raw_total_nodes": len(raw_nodes),
        "successes": len(successes),
        "failures": len(failures),
        "error_counts": error_counts,
        "fit_points_tried": sorted(fit_points_tried),
        "best_node_id": best["node_id"] if best else None,
        "best_metric_value": best_value,
        "best_objective": best.get("objective_metric") if best else None,
    }
