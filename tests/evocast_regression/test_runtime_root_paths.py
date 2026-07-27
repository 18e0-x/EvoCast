from __future__ import annotations

from types import SimpleNamespace

from evocast.domain.knowledge_paths import runtime_root, shared_knowledge_dir, task_knowledge_dir, task_runs_dir
from evocast.harness.session import AgentSession


def test_repo_like_base_dir_writes_only_under_dot_evocast(tmp_path):
    repo = tmp_path / "repo"
    (repo / "evocast" / "core").mkdir(parents=True)
    (repo / "evocast" / "harness").mkdir(parents=True)
    (repo / "ts_benchmark").mkdir()

    session = AgentSession(task_id="path_guard", base_dir=str(repo), client=SimpleNamespace())
    session.ensure_dirs()

    expected_runtime = repo / ".evocast"
    assert runtime_root(str(repo)) == expected_runtime
    assert session.base_dir == str(expected_runtime)
    assert task_knowledge_dir(str(repo), "path_guard") == expected_runtime / "task_knowledge" / "path_guard"
    assert task_runs_dir(str(repo), "path_guard") == expected_runtime / "runs" / "path_guard"
    assert shared_knowledge_dir(str(repo)) == expected_runtime / "knowledge"
    assert (expected_runtime / "task_knowledge" / "path_guard").is_dir()
    assert (expected_runtime / "runs" / "path_guard").is_dir()
    assert not (repo / "task_knowledge").exists()
    assert not (repo / "runs").exists()
    assert not (repo / "knowledge").exists()
