"""Model registry builder CLI for evocast.

Usage:
  python -m evocast.scripts.build_model_registry --task-id <id>
  python -m evocast.scripts.build_model_registry --task-id <id> --overrides configs/registry/model_overrides.yaml
  python -m evocast.scripts.build_model_registry --task-id <id> --no-verify

Builds the model registry, verifies imports, applies overrides, and
writes a snapshot to .evocast/task_knowledge/<task_id>/model_registry_snapshot.json.
"""

import argparse
import json
import os
import sys

from evocast.domain.knowledge_paths import runtime_root, task_knowledge_dir
from evocast.research.model_registry import (
    build_registry,
    get_verified_models,
    snapshot_registry,
)


def build(task_id: str, overrides_path: str | None = None, verify: bool = True, base_dir: str | None = None) -> dict:
    if base_dir is None:
        base_dir = str(runtime_root())

    knowledge_dir = str(task_knowledge_dir(base_dir, task_id))
    os.makedirs(knowledge_dir, exist_ok=True)

    print(f"[build_model_registry] Building registry...")
    if overrides_path:
        print(f"  Overrides: {overrides_path}")

    registry = build_registry(overrides_path=overrides_path, verify=verify)
    verified = get_verified_models(registry)

    print(f"  Total models:    {len(registry)}")
    print(f"  Verified:        {len(verified)}")
    print(f"  Not importable:  {len(registry) - len(verified)}")

    families: dict[str, int] = {}
    for m in verified:
        fam = m.get("family", "unknown")
        families[fam] = families.get(fam, 0) + 1
    print(f"  Families:        {families}")

    output = os.path.join(knowledge_dir, "model_registry_snapshot.json")
    snapshot_registry(registry, output)
    print(f"  Snapshot:        {output}")

    return {
        "task_id": task_id,
        "total_models": len(registry),
        "verified_models": len(verified),
        "families": families,
        "snapshot_path": output,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build the TFB model registry for a task.",
    )
    parser.add_argument(
        "--task-id", required=True,
        help="Task identifier.",
    )
    parser.add_argument(
        "--overrides",
        help="Path to model registry overrides YAML/JSON.",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip import verification.",
    )

    args = parser.parse_args()

    try:
        build(
            task_id=args.task_id,
            overrides_path=args.overrides,
            verify=not args.no_verify,
        )
    except Exception as e:
        print(f"[build_model_registry] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
