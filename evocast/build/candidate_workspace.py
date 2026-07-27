"""Contract-scoped source workspace, patch, and candidate snapshot operations."""

from __future__ import annotations

import fnmatch
from pathlib import Path

from evocast.build.contract import BuildContract
from evocast.build.source_snapshot import (
    CandidateSnapshot,
    changed_files_between,
    copy_source_tree,
    materialize_candidate_snapshot,
    source_manifest,
    unified_diff_between,
)


class CandidateWorkspaceService:
    """Own source materialization without making round or evaluation decisions."""

    def base_checkout(self, contract: BuildContract) -> Path:
        checkout = str((contract.base_source_ref or {}).get("source_checkout") or "").strip()
        if not checkout:
            raise RuntimeError("BuildContract.base_source_ref.source_checkout is required")
        path = Path(checkout).resolve()
        if not path.is_dir():
            raise RuntimeError(f"source checkout does not exist: {path}")
        return path

    def create_from_snapshot(self, path: Path, contract: BuildContract) -> None:
        if path.exists():
            raise RuntimeError(f"workspace path already exists: {path}")
        copy_source_tree(self.base_checkout(contract), path, overwrite=True)

    def copy_candidate(self, snapshot: CandidateSnapshot, destination: Path) -> None:
        copy_source_tree(snapshot.source_checkout, destination, overwrite=True)

    def capture_patch(self, workspace: Path, contract: BuildContract) -> str:
        changed = self.changed_files(workspace, contract)
        return unified_diff_between(self.base_checkout(contract), workspace, changed)

    def changed_files(self, workspace: Path, contract: BuildContract) -> list[str]:
        return changed_files_between(source_manifest(self.base_checkout(contract)), workspace)

    def materialize_candidate(
        self,
        *,
        agent_workspace: Path,
        candidate_dir: Path,
        changed_files: list[str],
        contract: BuildContract,
    ) -> CandidateSnapshot:
        changed = [self.normalize_path(path) for path in changed_files if self.normalize_path(path)]
        if not changed:
            raise RuntimeError("candidate final state has no changed files")
        base_checkout = self.base_checkout(contract)
        self.validate_changed_files(changed, contract, base_checkout=base_checkout)
        snapshot = materialize_candidate_snapshot(
            base_checkout=base_checkout,
            agent_workspace=agent_workspace,
            destination_dir=candidate_dir,
            changed_files=changed,
        )
        self.validate_changed_files(
            list(snapshot.changed_files),
            contract,
            base_checkout=base_checkout,
        )
        return snapshot

    @staticmethod
    def normalize_path(path: str) -> str:
        return str(path or "").replace("\\", "/").strip().lstrip("./")

    @classmethod
    def matches(cls, path: str, patterns: list[str]) -> bool:
        value = cls.normalize_path(path)
        for pattern in patterns:
            normalized = cls.normalize_path(pattern)
            if value == normalized or fnmatch.fnmatch(value, normalized):
                return True
        return False

    def validate_changed_files(
        self,
        changed_files: list[str],
        contract: BuildContract,
        *,
        base_checkout: Path | None = None,
    ) -> None:
        changed = [self.normalize_path(path) for path in changed_files if self.normalize_path(path)]
        if not changed:
            raise RuntimeError("candidate changed_files is empty")
        allowed_files = {self.normalize_path(path) for path in contract.allowed_edit_files}
        allowed_roots = [self.normalize_path(root) for root in contract.allowed_new_file_roots]
        forbidden_files = {
            self.normalize_path(path)
            for path in [*contract.forbidden_files, *contract.protected_files]
        }
        repo_wide = str(contract.execution_authority or "") == "repo_wide"
        forbidden_globs = [
            *list(contract.forbidden_globs),
            *list(contract.protected_globs),
            "**/__pycache__/**",
            "__pycache__/**",
            "*.pyc",
            "*.pyo",
            *([] if repo_wide else ["result/**", "dataset/**", "datasets/**"]),
            ".evocast/**",
        ]
        for path in changed:
            if path.startswith("../") or "/../" in f"/{path}":
                raise RuntimeError(f"unsafe changed path: {path}")
            exists_in_base = bool(base_checkout and (base_checkout / path).exists())
            allowed = path in allowed_files or (
                not exists_in_base
                and any(
                    root in {"", "."} or path.startswith(root.rstrip("/") + "/")
                    for root in allowed_roots
                )
            )
            if not allowed:
                raise RuntimeError(f"changed file is outside allowed edit surface: {path}")
            if path in forbidden_files or self.matches(path, forbidden_globs):
                raise RuntimeError(
                    f"changed file matches forbidden/protected/generated surface: {path}"
                )
