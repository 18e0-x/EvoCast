"""Structured failure-chain records for variant validation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def build_failure_chain(
    *,
    failed_gate: str,
    primary_failure: str,
    first_failure: str = "",
    steps: Iterable[str] = (),
    evidence: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    chain_steps: List[str] = [str(step) for step in steps if str(step).strip()]
    if failed_gate and failed_gate not in chain_steps:
        chain_steps.append(failed_gate)
    return {
        "failed_gate": str(failed_gate or ""),
        "primary_failure": str(primary_failure or ""),
        "first_failure": str(first_failure or primary_failure or ""),
        "failure_chain": chain_steps,
        "evidence": dict(evidence or {}),
    }
