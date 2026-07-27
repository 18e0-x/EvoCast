from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evocast.build.contract import BuildContract
from evocast.build.result import BuildDecision, VerificationResult
from evocast.domain.atomic_io import atomic_write_json


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout[-8000:],
            "stderr": self.stderr[-8000:],
            "timed_out": self.timed_out,
        }


class ExternalVerifier:
    """Run authoritative checks from a clean candidate checkout."""

    def __init__(self, *, timeout_seconds: int | None = None) -> None:
        self.timeout_seconds = timeout_seconds

    def verify(
        self,
        *,
        contract: BuildContract,
        checkout_dir: Path,
        output_dir: Path,
    ) -> VerificationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        commands = list(contract.external_verification_commands)
        if not commands:
            payload = {
                "status": BuildDecision.ACCEPTED.value,
                "reason_code": "no_external_commands",
                "summary": "No external verification commands were specified.",
                "commands": [],
            }
            path = output_dir / "verifier.json"
            atomic_write_json(path, payload, ensure_ascii=False)
            return VerificationResult(
                status=BuildDecision.ACCEPTED,
                reason_code="no_external_commands",
                summary="No external verification commands were specified.",
                artifact_paths=[str(path)],
            )

        results: list[CommandResult] = []
        timeout = self.timeout_seconds or contract.timeout_seconds
        for index, command in enumerate(commands, start=1):
            command_dir = output_dir / f"command_{index:02d}.json"
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(checkout_dir),
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                )
                result = CommandResult(
                    command=list(command),
                    returncode=int(completed.returncode),
                    stdout=completed.stdout or "",
                    stderr=completed.stderr or "",
                )
            except subprocess.TimeoutExpired as exc:
                result = CommandResult(
                    command=list(command),
                    returncode=124,
                    stdout=str(exc.stdout or ""),
                    stderr=str(exc.stderr or ""),
                    timed_out=True,
                )
            results.append(result)
            atomic_write_json(command_dir, result.to_dict(), ensure_ascii=False)
            if result.timed_out:
                return self._write_summary(
                    output_dir=output_dir,
                    status=BuildDecision.TERMINAL_REJECTED,
                    reason_code="verifier_timeout",
                    summary=f"External verifier timed out: {' '.join(command)}",
                    repair_instructions=[],
                    results=results,
                )
            if result.returncode != 0:
                return self._write_summary(
                    output_dir=output_dir,
                    status=BuildDecision.REPAIR_REQUIRED,
                    reason_code="verifier_command_failed",
                    summary=f"External verifier command failed: {' '.join(command)}",
                    repair_instructions=[
                        "Use the verifier stdout/stderr artifact to repair the implementation in the same agent thread.",
                    ],
                    results=results,
                )

        return self._write_summary(
            output_dir=output_dir,
            status=BuildDecision.ACCEPTED,
            reason_code="external_verifier_passed",
            summary="All external verification commands passed.",
            repair_instructions=[],
            results=results,
        )

    def _write_summary(
        self,
        *,
        output_dir: Path,
        status: BuildDecision,
        reason_code: str,
        summary: str,
        repair_instructions: list[str],
        results: list[CommandResult],
    ) -> VerificationResult:
        path = output_dir / "verifier.json"
        atomic_write_json(
            path,
            {
                "status": status.value,
                "reason_code": reason_code,
                "summary": summary,
                "repair_instructions": list(repair_instructions),
                "commands": [item.to_dict() for item in results],
            },
            ensure_ascii=False,
        )
        artifacts = [str(path)] + [str(output_dir / f"command_{idx:02d}.json") for idx in range(1, len(results) + 1)]
        return VerificationResult(
            status=status,
            reason_code=reason_code,
            summary=summary,
            repair_instructions=list(repair_instructions),
            artifact_paths=artifacts,
        )
