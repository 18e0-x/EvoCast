from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evocast.build.contract import BuildContract
from evocast.domain.atomic_io import atomic_write_json
from evocast.harness.api_client import ProviderClient


REQUIRED_METADATA_FIELDS = (
    "display_idea",
    "idea_summary",
    "mechanism_name",
    "changed_mechanism",
    "expected_effect",
)


@dataclass(frozen=True)
class ResearchMetadata:
    display_idea: str
    idea_summary: str
    mechanism_name: str
    changed_mechanism: str
    expected_effect: str

    def to_dict(self) -> dict[str, str]:
        return {
            "display_idea": self.display_idea,
            "idea_summary": self.idea_summary,
            "mechanism_name": self.mechanism_name,
            "changed_mechanism": self.changed_mechanism,
            "expected_effect": self.expected_effect,
        }


class ResearchMetadataWriter:
    """LLM-owned writer for formal research round metadata."""

    def __init__(
        self,
        *,
        client: ProviderClient,
        stage: str = "round_metadata",
        timeout_seconds: int = 90,
        max_tokens: int = 1600,
        max_patch_chars: int = 24000,
    ) -> None:
        self.client = client
        self.stage = stage
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.max_tokens = max(1, int(max_tokens))
        self.max_patch_chars = max(1000, int(max_patch_chars))

    def write(
        self,
        *,
        round_num: int,
        contract: BuildContract,
        patch_text: str,
        changed_files: list[str],
        coding_summary: str,
        output_dir: Path,
    ) -> ResearchMetadata:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = self._request_metadata(
            round_num=round_num,
            contract=contract,
            patch_text=patch_text,
            changed_files=changed_files,
            coding_summary=coding_summary,
        )
        metadata = self._validate(payload)
        atomic_write_json(output_dir / "round_metadata.json", metadata.to_dict(), ensure_ascii=False)
        return metadata

    def _request_metadata(
        self,
        *,
        round_num: int,
        contract: BuildContract,
        patch_text: str,
        changed_files: list[str],
        coding_summary: str,
    ) -> dict[str, Any]:
        protocol = contract.metric_protocol if isinstance(contract.metric_protocol, dict) else {}
        direction = protocol.get("research_direction") if isinstance(protocol.get("research_direction"), dict) else {}
        target = protocol.get("target") if isinstance(protocol.get("target"), dict) else {}
        exact_target = protocol.get("exact_ablation_target") if isinstance(protocol.get("exact_ablation_target"), dict) else {}
        patch_excerpt = str(patch_text or "")
        if len(patch_excerpt) > self.max_patch_chars:
            patch_excerpt = patch_excerpt[: self.max_patch_chars] + "\n...[patch truncated for metadata writing]"
        messages = [
            {
                "role": "system",
                "content": (
                    "You write the formal metadata for one EvoCast research build round. "
                    "Use only the provided contract fields, selected research direction, coding summary, changed files, and patch. "
                    "Return exactly one JSON object. Do not include markdown or commentary. "
                    "Do not invent training results, metric values, or evidence that is not present. "
                    "The metadata must describe the implemented research mechanism, not the tool process."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "Generate formal LLM-owned research metadata for the implemented source diff.",
                        "required_output_schema": {
                            "display_idea": "extremely short terminal title; eight English words or fewer; concise Chinese title is also allowed",
                            "idea_summary": "one or two sentences describing the concrete implemented idea",
                            "mechanism_name": "human-readable name of the mechanism being changed or introduced",
                            "changed_mechanism": "specific model component or computation changed by this diff",
                            "expected_effect": "expected forecasting behavior or metric effect implied by the idea",
                        },
                        "display_rule": (
                            "If contract_context.research_direction.terminal_display_title is present, "
                            "use it exactly as display_idea unless the patch proves that selected direction was infeasible."
                        ),
                        "contract_context": {
                            "research_id": contract.research_id,
                            "target_model": contract.target_model,
                            "semantic_goal": contract.semantic_goal,
                            "hypothesis": contract.hypothesis,
                            "allowed_edit_files": list(contract.allowed_edit_files or []),
                            "metric_protocol_target": target,
                            "exact_ablation_target": exact_target,
                            "research_direction": direction,
                        },
                        "implementation_context": {
                            "coding_summary": coding_summary,
                            "changed_files": list(changed_files),
                            "patch_excerpt": patch_excerpt,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
            },
        ]
        return self.client.call_json(
            stage=self.stage,
            round_num=int(round_num),
            messages=messages,
            schema_hint=json.dumps({"required": list(REQUIRED_METADATA_FIELDS)}),
            require_all_top_level_keys=True,
            timeout_sec_override=self.timeout_seconds,
            max_tokens_override=self.max_tokens,
            execution_label=f"{contract.research_id}_metadata",
            stream_override=False,
        )

    @staticmethod
    def _validate(payload: dict[str, Any]) -> ResearchMetadata:
        values = {field: " ".join(str(payload.get(field) or "").split()) for field in REQUIRED_METADATA_FIELDS}
        missing = [field for field, value in values.items() if not value]
        if missing:
            raise ValueError(f"metadata missing required field(s): {', '.join(missing)}")
        display_words = values["display_idea"].split()
        if len(display_words) > 8 and not any("\u4e00" <= char <= "\u9fff" for char in values["display_idea"]):
            raise ValueError("metadata display_idea exceeds eight words")
        return ResearchMetadata(**values)
