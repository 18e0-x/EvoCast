"""Protected path validator.

Enforces the write boundary defined in agent_plan.md Section 3 and 4.
Any generated file or patch must not modify protected TFB files.
"""

from typing import List, Dict, Set

from evocast.domain.execution_ids import is_ablation_source_path, is_research_source_path

# Files and directories the agent must NOT automatically modify.
PROTECTED_FILES: Set[str] = {
    "ts_benchmark/pipeline.py",
    "ts_benchmark/models/model_loader.py",
    "ts_benchmark/recording.py",
}

PROTECTED_DIRS: Set[str] = {
    "ts_benchmark/evaluation",
    "ts_benchmark/baselines/time_series_library",
    "config",
}

# Paths the agent is allowed to write to.
# Runtime state lives under .evocast; candidate source lives under round_sources.
ALLOWED_DIRS: Set[str] = {
    ".evocast",
    "evocast",
    "round_sources",
    "result",
}

# Candidate source prefix for task-local variant files.
WORKSPACE_VARIANT_PREFIX = "round_sources/"


def normalize_path(path: str) -> str:
    """Normalize a filesystem path to use forward slashes, relative to repo root."""
    return path.replace("\\", "/").rstrip("/")


def is_protected(file_path: str) -> bool:
    """Check if a single file path falls under a protected area."""
    p = normalize_path(file_path)
    if p in PROTECTED_FILES:
        return True
    for d in PROTECTED_DIRS:
        if p.startswith(d + "/") or p == d:
            return True
    return False


def is_allowed(file_path: str) -> bool:
    """Check if a single file path falls under an allowed area."""
    p = normalize_path(file_path)
    for d in ALLOWED_DIRS:
        if p.startswith(d + "/") or p == d:
            return True
    return False


def check_file_paths(file_paths: List[str]) -> Dict:
    """Validate a list of file paths.

    A path is only valid if it is EXPLICITLY under an allowed directory.
    It's not enough to simply not be protected — the path must be in
    .evocast/**, round_sources/**, evocast/**, or result/**.

    Returns a dict with:
      - ok: bool
      - violations: list of (path, reason) tuples for rejected paths
      - allowed: list of paths that pass both checks
    """
    violations = []
    allowed = []
    for fp in file_paths:
        if is_protected(fp):
            violations.append((fp, "protected"))
        elif not is_allowed(fp):
            violations.append((fp, "not_in_allowed_dirs"))
        else:
            allowed.append(fp)
    return {
        "ok": len(violations) == 0,
        "violations": violations,
        "allowed": allowed,
    }


def validate_variant_path(variant_path: str) -> Dict:
    """Validate that a variant file is in the correct location.

    Variant files MUST be under a Git-visible candidate source workspace:
      round_sources/<task_id>/ResearchNNN/round_entry.py
      round_sources/<task_id>/AblationNNN/round_entry.py

    """
    p = normalize_path(variant_path)
    if is_research_source_path(p) or is_ablation_source_path(p):
        pass
    else:
        return {
            "ok": False,
            "error": (
                f"Variant path must be under a task workspace "
                f"(round_sources/<task_id>/<ResearchNNN>/round_entry.py), "
                f"got: {p}"
            ),
        }
    if is_protected(p):
        return {
            "ok": False,
            "error": f"Variant path conflicts with protected path: {p}",
        }
    return {"ok": True, "error": None}


def is_workspace_variant_path(variant_path: str) -> bool:
    """Return True if *variant_path* points to a task-local workspace.

    Delegates to the canonical implementation in workspace_loader.
    """
    from evocast.variant.workspace_loader import is_workspace_variant_path as _impl
    return _impl(variant_path)


def generate_protected_path_report(generated_files: List[str]) -> Dict:
    """Generate a full protected path compliance report.

    Returns a dict suitable for writing to protected_path_report.json.
    """
    check = check_file_paths(generated_files)
    return {
        "timestamp": "",
        "generated_files": generated_files,
        "protected_violations": check["violations"],
        "all_clear": check["ok"],
        "protected_files_list": sorted(PROTECTED_FILES),
        "protected_dirs_list": sorted(PROTECTED_DIRS),
        "allowed_dirs_list": sorted(ALLOWED_DIRS),
    }
