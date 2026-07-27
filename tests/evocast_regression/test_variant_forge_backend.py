from __future__ import annotations

import json
from pathlib import Path

import pytest

from evocast.build.backends.variant_forge_backend import VariantForgeBackend
from evocast.build.contract import BuildContract
from evocast.scripts.run_agent_v3 import main as run_agent_main
from evocast.build.source_snapshot import source_manifest


class FakeJsonClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def call_json(self, **kwargs):
        self.calls.append(dict(kwargs))
        if not self.payloads:
            raise RuntimeError("no fake payload left")
        return self.payloads.pop(0)


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "model.py").write_text("VALUE = 'baseline'\n", encoding="utf-8")
    (repo / "other.py").write_text("VALUE = 'other'\n", encoding="utf-8")
    return repo, source_manifest(repo)["snapshot_id"]


def _contract(repo: Path, base_snapshot_id: str) -> BuildContract:
    return BuildContract(
        research_id="Research001",
        base_snapshot_id=base_snapshot_id,
        semantic_goal="Change the fixture model value.",
        hypothesis="A complete-file edit can implement the fixture change.",
        target_model="FixtureModel",
        base_source_ref={
            "kind": "source_snapshot",
            "candidate_snapshot_id": base_snapshot_id,
            "base_snapshot_id": base_snapshot_id,
            "source_checkout": str(repo),
            "source_manifest_hash": source_manifest(repo)["manifest_hash"],
            "source_binding": {"entry_file": "model.py", "source_files": ["model.py"], "verified": True},
        },
        allowed_edit_files=["model.py"],
        allowed_new_file_roots=["helpers"],
        likely_entrypoints=["model.py"],
        required_behavior=["model.py contains improved"],
        timeout_seconds=30,
        repair_budget=1,
    )


def test_variant_forge_backend_writes_complete_allowed_file_and_records_transcript(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    workspace = tmp_path / "round" / "agent_workspace"
    workspace.mkdir(parents=True)
    (workspace / "model.py").write_text("VALUE = 'baseline'\n", encoding="utf-8")
    client = FakeJsonClient(
        [
            {
                "files": {"model.py": "VALUE = 'improved'\n"},
                "metadata": {
                    "display_idea": "Fixture edit",
                    "idea_summary": "Changed fixture VALUE.",
                    "mechanism_name": "Fixture value",
                    "changed_mechanism": "model.py VALUE assignment",
                    "expected_effect": "Verifier sees improved marker.",
                },
                "activation_evidence": ["model.py is replaced with the improved live value"],
            }
        ]
    )
    backend = VariantForgeBackend(client=client, user_prompt="Prioritize a stable long-horizon signal.")
    session_id = backend.start(workspace, _contract(repo, base_snapshot_id))

    result = backend.run_turn(session_id, "implement", 30)

    assert result.status == "completed"
    assert result.changed_files == ["model.py"]
    assert result.display_idea == "Fixture edit"
    assert "improved" in (workspace / "model.py").read_text(encoding="utf-8")
    assert result.transcript_path and result.transcript_path.is_file()
    transcript = json.loads(result.transcript_path.read_text(encoding="utf-8"))
    assert transcript["backend"] == "variant_forge"
    assert transcript["attempts"][0]["internal_checks"][0]["ok"] is True
    assert client.calls[0]["stage"] == "build_coding_agent"
    assert "timeout_sec_override" not in client.calls[0]
    assert "max_tokens_override" not in client.calls[0]
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert payload["user_research_intent"] == "Prioritize a stable long-horizon signal."


def test_run_agent_cli_rejects_unknown_arguments() -> None:
    with pytest.raises(SystemExit) as error:
        run_agent_main(["--not-a-real-option"])
    assert error.value.code == 2


def test_run_agent_cli_exposes_variant_forge_backend(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        run_agent_main(["--help"])

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--backend {variant-forge}" in help_text


def test_variant_forge_backend_rejects_file_outside_contract(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    workspace = tmp_path / "round" / "agent_workspace"
    workspace.mkdir(parents=True)
    (workspace / "model.py").write_text("VALUE = 'baseline'\n", encoding="utf-8")
    client = FakeJsonClient(
        [
            {
                "files": {"README.md": "changed\n"},
                "metadata": {},
                "activation_evidence": [],
            }
        ]
    )
    backend = VariantForgeBackend(client=client, max_attempts=1)
    session_id = backend.start(workspace, _contract(repo, base_snapshot_id))

    result = backend.run_turn(session_id, "implement", 30)

    assert result.status == "failed"
    assert "not in allowed_new_file_roots" in str(result.error)
    assert (workspace / "model.py").read_text(encoding="utf-8") == "VALUE = 'baseline'\n"


def test_variant_forge_backend_writes_new_file_under_allowed_new_file_root(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    workspace = tmp_path / "round" / "agent_workspace"
    workspace.mkdir(parents=True)
    (workspace / "model.py").write_text("from helpers.helper import VALUE\n", encoding="utf-8")
    client = FakeJsonClient(
        [
            {
                "files": {
                    "model.py": "from helpers.helper import VALUE\n",
                    "helpers/helper.py": "VALUE = 'improved'\n",
                },
                "metadata": {
                    "display_idea": "New helper",
                    "idea_summary": "Added helper module.",
                    "mechanism_name": "Helper",
                    "changed_mechanism": "model imports helper",
                    "expected_effect": "Verifier sees helper.",
                },
                "activation_evidence": ["helper.py is imported by model.py"],
            }
        ]
    )
    backend = VariantForgeBackend(client=client)
    session_id = backend.start(workspace, _contract(repo, base_snapshot_id))

    result = backend.run_turn(session_id, "implement", 30)

    assert result.status == "completed"
    assert result.changed_files == ["helpers/helper.py", "model.py"]
    assert (workspace / "helpers" / "helper.py").read_text(encoding="utf-8") == "VALUE = 'improved'\n"


def test_variant_forge_backend_rejects_existing_file_under_new_file_root_when_not_allowed(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    workspace = tmp_path / "round" / "agent_workspace"
    workspace.mkdir(parents=True)
    (workspace / "model.py").write_text("VALUE = 'baseline'\n", encoding="utf-8")
    (workspace / "other.py").write_text("VALUE = 'other'\n", encoding="utf-8")
    client = FakeJsonClient(
        [
            {
                "files": {"other.py": "VALUE = 'changed'\n"},
                "metadata": {},
                "activation_evidence": [],
            }
        ]
    )
    backend = VariantForgeBackend(client=client, max_attempts=1)
    session_id = backend.start(workspace, _contract(repo, base_snapshot_id))

    result = backend.run_turn(session_id, "implement", 30)

    assert result.status == "failed"
    assert "existing edited file is not in allowed_edit_files" in str(result.error)
    assert (workspace / "other.py").read_text(encoding="utf-8") == "VALUE = 'other'\n"


def test_variant_forge_backend_repo_wide_authority_allows_any_repo_file_and_top_level_new_file(tmp_path: Path) -> None:
    repo, base_snapshot_id = _repo(tmp_path)
    workspace = tmp_path / "round" / "agent_workspace"
    workspace.mkdir(parents=True)
    (workspace / "model.py").write_text("VALUE = 'baseline'\n", encoding="utf-8")
    (workspace / "other.py").write_text("VALUE = 'other'\n", encoding="utf-8")
    client = FakeJsonClient(
        [
            {
                "files": {
                    "other.py": "VALUE = 'repo-wide'\n",
                    "NEWFILE.py": "TOKEN = 'new'\n",
                },
                "metadata": {
                    "display_idea": "Repo wide",
                    "idea_summary": "Edited arbitrary repo file and added a top-level file.",
                    "mechanism_name": "Repo wide authority",
                    "changed_mechanism": "other.py and NEWFILE.py",
                    "expected_effect": "Authority no longer restricts edit surface.",
                },
                "activation_evidence": ["repo-wide authority permits arbitrary workspace file edits"],
            }
        ]
    )
    contract = _contract(repo, base_snapshot_id)
    contract = BuildContract(**{**contract.to_dict(), "execution_authority": "repo_wide", "allowed_edit_files": ["model.py", "other.py"], "allowed_new_file_roots": ["."]})
    backend = VariantForgeBackend(client=client)
    session_id = backend.start(workspace, contract)

    result = backend.run_turn(session_id, "implement", 30)

    assert result.status == "completed"
    assert result.changed_files == ["NEWFILE.py", "other.py"]
    assert (workspace / "other.py").read_text(encoding="utf-8") == "VALUE = 'repo-wide'\n"
    assert (workspace / "NEWFILE.py").read_text(encoding="utf-8") == "TOKEN = 'new'\n"
