"""VariantForge backend for contract-constrained research implementation."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from evocast.build.backends.base import AgentRunResult
from evocast.build.contract import BuildContract
from evocast.build.contract_prompt import build_execution_contract_dict
from evocast.domain.atomic_io import atomic_write_json
from evocast.harness.api_client import ProviderClient
from evocast.state.cost_ledger import tracked_stage


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload, ensure_ascii=False)


class VariantForgeBackend:
    """Generate, validate, and repair complete-file research variants."""

    def __init__(
        self,
        *,
        client: ProviderClient,
        max_attempts: int = 5,
        max_file_chars: int = 28000,
        user_prompt: str = "",
    ) -> None:
        self.client = client
        self.max_attempts = max(1, int(max_attempts))
        self.max_file_chars = max(4000, int(max_file_chars))
        self.user_prompt = str(user_prompt or "").strip()
        self.sessions: dict[str, dict[str, Any]] = {}

    def start(self, workspace: Path, contract: BuildContract) -> str:
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = {
            "workspace": Path(workspace).resolve(),
            "contract": contract,
            "attempts": [],
        }
        return session_id

    def interrupt(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    @staticmethod
    def _round_num(contract: BuildContract) -> int:
        digits = "".join(char for char in str(contract.research_id or "") if char.isdigit())
        return int(digits or 1)

    def _read_allowed_sources(self, workspace: Path, contract: BuildContract) -> dict[str, str]:
        sources: dict[str, str] = {}
        prompt_files = contract.allowed_edit_files
        if str(contract.execution_authority or "") == "repo_wide":
            prompt_files = list(dict.fromkeys([*contract.likely_entrypoints, *list((contract.base_source_ref or {}).get("source_binding", {}).get("source_files") or [])]))
        for rel in prompt_files:
            normalized = str(rel).replace("\\", "/").strip().lstrip("./")
            path = workspace / normalized
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) > self.max_file_chars:
                half = self.max_file_chars // 2
                text = (
                    text[:half]
                    + "\n\n# ... [source truncated in prompt; preserve omitted code when editing] ...\n\n"
                    + text[-half:]
                )
            sources[normalized] = text
        return sources

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are VariantForge, EvoCast's contract-constrained implementation backend. "
            "Return exactly one JSON object. No markdown. No commentary. "
            "You must produce complete replacement content for the edited allowed file. "
            "Preserve the original public model interface, class names, method signatures, tensor shape contract, task semantics, "
            "input length, prediction horizon, target selection, and training hyperparameters. "
            "The implementation must activate the proposed mechanism in __init__ and the effective forecast/forward path. "
            "Do not merely add unused classes, helpers, or dead branches. "
            "If you add a module, instantiate it and call it on the live tensor path before the final prediction head. "
            "Do not edit data, metrics, runner, config, or evaluation code."
        )

    def _user_payload(
        self,
        *,
        contract: BuildContract,
        source_files: dict[str, str],
        feedback: list[dict[str, Any]],
        repair_message: str,
    ) -> dict[str, Any]:
        repo_wide = str(contract.execution_authority or "") == "repo_wide"
        return {
            "task": "Generate a complete-file implementation for this EvoCast BuildContract.",
            "user_research_intent": self.user_prompt,
            "required_output_schema": {
                "files": {"path/to/allowed_file.py": "complete replacement UTF-8 source code"},
                "metadata": {
                    "display_idea": "short title",
                    "idea_summary": "what was implemented",
                    "mechanism_name": "mechanism name",
                    "changed_mechanism": "exact live computation changed",
                    "expected_effect": "expected forecasting effect",
                },
                "activation_evidence": [
                    "Short evidence that the new mechanism is instantiated and called in the live forecast/forward path"
                ],
            },
            "path_policy": {
                "existing_files": (
                    "May modify any repository file inside the candidate workspace."
                    if repo_wide
                    else "May modify only paths listed in execution_contract.allowed_edit_files."
                ),
                "new_files": (
                    "May create new files anywhere inside the candidate workspace repository."
                    if repo_wide
                    else "May create new source files only under execution_contract.allowed_new_file_roots."
                ),
            },
            "execution_contract": build_execution_contract_dict(contract),
            "source_files": source_files,
            "previous_failed_attempts": feedback[-3:],
            "repair_message_from_orchestrator": repair_message,
        }

    @staticmethod
    def _root_allows(rel: str, roots: list[str]) -> bool:
        for root in roots:
            normalized = str(root).replace("\\", "/").strip().lstrip("./")
            if normalized in {"", "."}:
                return True
            prefix = normalized.rstrip("/") + "/"
            if rel.startswith(prefix):
                return True
        return False

    @staticmethod
    def _validate_response(payload: dict[str, Any], contract: BuildContract, workspace: Path) -> tuple[dict[str, str], dict[str, Any]]:
        files = payload.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("response.files must contain at least one edited file")
        allowed = {str(path).replace("\\", "/").strip().lstrip("./") for path in contract.allowed_edit_files}
        allowed_roots = [
            str(path).replace("\\", "/").strip().lstrip("./").rstrip("/") + "/"
            for path in contract.allowed_new_file_roots
            if str(path or "").strip()
        ]
        normalized: dict[str, str] = {}
        for raw_path, raw_content in files.items():
            rel = str(raw_path or "").replace("\\", "/").strip().lstrip("./")
            if not rel or rel.startswith("../") or "/../" in f"/{rel}":
                raise ValueError(f"unsafe edited file path: {rel}")
            path_exists = (workspace / rel).exists()
            if path_exists and rel not in allowed:
                raise ValueError(f"existing edited file is not in allowed_edit_files: {rel}")
            if not path_exists and rel not in allowed and not VariantForgeBackend._root_allows(rel, allowed_roots):
                raise ValueError(f"new file is not in allowed_new_file_roots: {rel}")
            content = str(raw_content or "")
            if not content.strip():
                raise ValueError(f"empty content for {rel}")
            normalized[rel] = content
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        return normalized, metadata

    @staticmethod
    def _activation_summary(payload: dict[str, Any]) -> str:
        evidence = [str(item) for item in (payload.get("activation_evidence") or [])]
        return "; ".join(evidence)[:1200]

    @staticmethod
    def _run_syntax_checks(workspace: Path, files: dict[str, str]) -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []
        for rel in files:
            if not rel.endswith(".py"):
                continue
            proc = subprocess.run(
                [sys.executable, "-m", "py_compile", str(workspace / rel)],
                cwd=str(workspace),
                text=True,
                capture_output=True,
                timeout=60,
            )
            checks.append(
                {
                    "command": [sys.executable, "-m", "py_compile", rel],
                    "returncode": proc.returncode,
                    "stdout": proc.stdout[-4000:],
                    "stderr": proc.stderr[-4000:],
                    "ok": proc.returncode == 0,
                }
            )
        return checks

    @tracked_stage(
        "build_coding_agent_turn",
        lambda self, session_id, message, timeout_seconds: (
            str(getattr(self.client, "base_dir", Path(self.sessions[session_id]["workspace"]).parent)),
            str(getattr(self.client, "task_id", "variant_forge_backend_test")),
            str(self.sessions.get(session_id, {}).get("contract").research_id if self.sessions.get(session_id, {}).get("contract") else ""),
            str(session_id),
        ),
    )
    def run_turn(self, session_id: str, message: str, timeout_seconds: int) -> AgentRunResult:
        if session_id not in self.sessions:
            raise RuntimeError(f"unknown VariantForge backend session: {session_id}")
        state = self.sessions[session_id]
        workspace = Path(state["workspace"])
        contract: BuildContract = state["contract"]
        feedback: list[dict[str, Any]] = list(state.get("attempts") or [])
        loop_index = len(feedback) + 1
        transcript_dir = workspace.parent / "agent_transcripts" / f"variant_forge_{loop_index:02d}"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        last_error = ""
        source_files = self._read_allowed_sources(workspace, contract)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    self._user_payload(
                        contract=contract,
                        source_files=source_files,
                        feedback=feedback,
                        repair_message=message,
                    ),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
            },
        ]
        try:
            payload = self.client.call_json(
                stage="build_coding_agent",
                round_num=self._round_num(contract),
                messages=messages,
                schema_hint=json.dumps({"required": ["files", "metadata", "activation_evidence"]}),
                require_all_top_level_keys=True,
                execution_label=f"{contract.research_id}_variant_forge_attempt{loop_index:02d}",
                stream_override=False,
            )
            files, metadata = self._validate_response(payload, contract, workspace)
            for rel, content in files.items():
                path = workspace / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            checks = self._run_syntax_checks(workspace, files)
            attempt_record = {
                "loop": loop_index,
                "status": "generated",
                "files": sorted(files),
                "metadata": metadata,
                "activation_evidence": payload.get("activation_evidence"),
                "internal_checks": checks,
            }
            feedback.append(attempt_record)
            state["attempts"] = feedback
            if all(item.get("ok") for item in checks):
                transcript_path = transcript_dir / "transcript.json"
                _write_json(
                    transcript_path,
                    {
                        "backend": "variant_forge",
                        "messages": messages,
                        "attempts": feedback,
                        "final_payload": payload,
                    },
                )
                summary = self._activation_summary(payload)
                default_summary = "VariantForge implementation generated and syntax checks passed."
                return AgentRunResult(
                    session_id=session_id,
                    turn_id=f"variant-forge-{loop_index:02d}",
                    status="completed",
                    summary=summary or default_summary,
                    execution_summary=summary or default_summary,
                    display_idea=str(metadata.get("display_idea") or ""),
                    idea_summary=str(metadata.get("idea_summary") or ""),
                    mechanism_name=str(metadata.get("mechanism_name") or ""),
                    changed_mechanism=str(metadata.get("changed_mechanism") or ""),
                    expected_effect=str(metadata.get("expected_effect") or ""),
                    changed_files=sorted(files),
                    transcript_path=transcript_path,
                    internal_checks=checks,
                )
            last_error = "Syntax/internal checks failed: " + json.dumps(checks, ensure_ascii=False)
            feedback[-1]["status"] = "syntax_failed"
            state["attempts"] = feedback
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            feedback.append({"loop": loop_index, "status": "error", "error": last_error})
            state["attempts"] = feedback
        transcript_path = transcript_dir / "transcript.json"
        _write_json(
            transcript_path,
            {
                "backend": "variant_forge",
                "messages": messages,
                "attempts": feedback,
                "final_error": last_error,
            },
        )
        return AgentRunResult(
            session_id=session_id,
            turn_id=f"variant-forge-{loop_index:02d}",
            status="failed",
            summary=last_error or "VariantForge failed to produce valid code.",
            error=last_error or "VariantForge failed to produce valid code.",
            transcript_path=transcript_path,
        )
