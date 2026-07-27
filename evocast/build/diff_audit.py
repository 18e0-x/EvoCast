from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evocast.build.contract import BuildContract
from evocast.build.result import BuildDecision, VerificationResult
from evocast.domain.atomic_io import atomic_write_json


DEFAULT_TERMINAL_GLOBS = [
    "data/**",
    "datasets/**",
    "ts_benchmark/datasets/**",
    "evocast/policy/**",
    "evocast/harness/rounds.py",
    "evocast/runners/*metric*",
    "evocast/domain/metric*.py",
    "tests/**",
]

NETWORK_PATTERNS = [
    "import requests",
    "from requests",
    "urllib.request",
    "socket.",
    "http.client",
    "aiohttp",
]


@dataclass(frozen=True)
class DiffAuditResult:
    status: BuildDecision
    reason_code: str
    summary: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    repair_instructions: list[str] = field(default_factory=list)

    def to_verification_result(self, artifact_paths: list[str] | None = None) -> VerificationResult:
        return VerificationResult(
            status=self.status,
            reason_code=self.reason_code,
            summary=self.summary,
            repair_instructions=list(self.repair_instructions),
            artifact_paths=list(artifact_paths or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason_code": self.reason_code,
            "summary": self.summary,
            "findings": list(self.findings),
            "repair_instructions": list(self.repair_instructions),
        }


def _norm(path: str) -> str:
    return str(path or "").replace("\\", "/").strip().lstrip("./")


def _matches(path: str, patterns: list[str]) -> bool:
    value = _norm(path)
    for pattern in patterns:
        pat = _norm(pattern)
        if value == pat or fnmatch.fnmatch(value, pat):
            return True
    return False


def _root_allows(path: str, roots: list[str]) -> bool:
    value = _norm(path)
    for root in roots:
        normalized = _norm(root)
        if normalized in {"", "."}:
            return True
        if value.startswith(normalized.rstrip("/") + "/"):
            return True
    return False


class DiffAuditor:
    def __init__(self, *, default_terminal_globs: list[str] | None = None) -> None:
        self.default_terminal_globs = list(default_terminal_globs or DEFAULT_TERMINAL_GLOBS)

    def audit(
        self,
        *,
        contract: BuildContract,
        changed_files: list[str],
        patch_text: str,
        output_dir: Path,
        high_risk_rationale: dict[str, str] | None = None,
    ) -> DiffAuditResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        changed = [_norm(path) for path in changed_files if _norm(path)]
        findings: list[dict[str, Any]] = []
        repair: list[str] = []
        terminal = False

        if not patch_text.strip() or not changed:
            findings.append({"severity": "repair", "code": "empty_patch"})
            repair.append("Produce a concrete code change that implements the build contract.")
        elif "diff --git" in patch_text and "\n--- " not in patch_text and "\nGIT binary patch" not in patch_text:
            findings.append({"severity": "repair", "code": "non_applicable_patch"})
            repair.append("Produce an auditable patch with concrete file content hunks.")

        allowed_files = {_norm(path) for path in contract.allowed_edit_files if _norm(path)}
        allowed_files_casefold = {path.casefold() for path in allowed_files}
        allowed_roots = [_norm(root) for root in contract.allowed_new_file_roots]
        allowed_roots_casefold = [root.casefold() for root in allowed_roots]
        if allowed_files or allowed_roots:
            for path in changed:
                path_casefold = path.casefold()
                if (
                    path not in allowed_files
                    and path_casefold not in allowed_files_casefold
                    and not _root_allows(path, allowed_roots)
                    and not any(root in {"", "."} or path_casefold.startswith(root.rstrip("/").casefold() + "/") for root in allowed_roots_casefold)
                ):
                    findings.append({"severity": "repair", "code": "outside_allowed_build_root", "path": path})
                    repair.append(
                        "Move the implementation into the BuildContract allowed source files: "
                        + ", ".join(sorted(allowed_files) + allowed_roots)
                    )

        terminal_globs = ([] if str(contract.execution_authority or "") == "repo_wide" else list(self.default_terminal_globs)) + list(contract.forbidden_globs)
        for path in changed:
            if path in {_norm(item) for item in contract.forbidden_files} or _matches(path, terminal_globs):
                terminal = True
                findings.append({"severity": "terminal", "code": "forbidden_path", "path": path})
            elif path in {_norm(item) for item in contract.protected_files} or _matches(path, contract.protected_globs):
                findings.append({"severity": "repair", "code": "protected_path", "path": path})
                repair.append(f"Restore protected file or provide an implementation that avoids modifying {path}.")
            elif _matches(path, contract.high_risk_globs):
                rationale = str((high_risk_rationale or {}).get(path) or "").strip()
                if not rationale:
                    findings.append({"severity": "repair", "code": "missing_high_risk_rationale", "path": path})
                    repair.append(f"Explain and justify the high-risk change to {path}, or move the implementation elsewhere.")

        patch_lower = patch_text.lower()
        for pattern in NETWORK_PATTERNS:
            if pattern in patch_lower:
                terminal = True
                findings.append({"severity": "terminal", "code": "network_side_effect", "pattern": pattern})

        if terminal:
            result = DiffAuditResult(
                status=BuildDecision.TERMINAL_REJECTED,
                reason_code="terminal_diff_violation",
                summary="Diff touched forbidden infrastructure, metric, test, data, or network surfaces.",
                findings=findings,
                repair_instructions=repair,
            )
        elif findings:
            result = DiffAuditResult(
                status=BuildDecision.REPAIR_REQUIRED,
                reason_code="repairable_diff_violation",
                summary="Diff requires repair before external verification.",
                findings=findings,
                repair_instructions=repair,
            )
        else:
            result = DiffAuditResult(
                status=BuildDecision.ACCEPTED,
                reason_code="diff_accepted",
                summary="Diff passed EvoCast audit.",
                findings=[],
                repair_instructions=[],
            )

        atomic_write_json(output_dir / "diff_audit.json", result.to_dict(), ensure_ascii=False)
        atomic_write_json(output_dir / "changed_files.json", {"changed_files": changed}, ensure_ascii=False)
        (output_dir / "agent.patch").write_text(patch_text, encoding="utf-8")
        return result

    @staticmethod
    def read_audit(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
