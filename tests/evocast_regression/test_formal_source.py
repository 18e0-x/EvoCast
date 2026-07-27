from __future__ import annotations

from pathlib import Path

from evocast.domain.baseline_identity import create_immutable_baseline_snapshot, resolve_and_verify_model_binding
from evocast.domain.formal_source import copy_formal_source_tree, formal_source_fingerprint


def test_formal_source_excludes_generated_candidate_artifacts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    formal = repo / "ts_benchmark" / "baselines" / "timefilter"
    generated = repo / "ts_benchmark" / "baselines" / "research_variants"
    cache = formal / "__pycache__"
    formal.mkdir(parents=True)
    generated.mkdir(parents=True)
    cache.mkdir(parents=True)
    (formal / "timefilter.py").write_text("class TimeFilter:\n    pass\n", encoding="utf-8")
    (generated / "old_variant.py").write_text("BROKEN = True\n", encoding="utf-8")
    (cache / "timefilter.cpython-312.pyc").write_bytes(b"cache")

    before = formal_source_fingerprint(repo)
    (generated / "old_variant.py").write_text("BROKEN = False\n", encoding="utf-8")
    (generated / "new_variant.py").write_text("MORE = True\n", encoding="utf-8")
    after = formal_source_fingerprint(repo)

    assert before == after

    dst = tmp_path / "dst"
    copied = copy_formal_source_tree(
        repo_root=repo,
        source_root=repo / "ts_benchmark",
        destination_root=dst,
    )
    copied_paths = {item["path"] for item in copied}

    assert "ts_benchmark/baselines/timefilter/timefilter.py" in copied_paths
    assert not (dst / "ts_benchmark" / "baselines" / "research_variants").exists()
    assert not list(dst.rglob("*.pyc"))


def test_immutable_baseline_snapshot_excludes_legacy_research_variants(tmp_path: Path) -> None:
    binding = resolve_and_verify_model_binding(
        model_key="TimeFilter",
        public_import_path="ts_benchmark.baselines.timefilter.TimeFilter",
    )
    snapshot = create_immutable_baseline_snapshot(str(tmp_path), "formal_snapshot", binding)
    source_root = Path(snapshot.source_root)

    assert source_root.name == "ts_benchmark"
    assert source_root.is_dir()
    assert (source_root / "baselines" / "timefilter" / "timefilter.py").is_file()
    assert (source_root / "baselines" / "timefilter" / "models" / "TimeFilter.py").is_file()
    assert (source_root / "baselines" / "timefilter" / "layers" / "TimeFilter_layers.py").is_file()
    assert not (source_root / "baselines" / "research_variants").exists()
    assert not list(source_root.rglob("*.pyc"))
    assert formal_source_fingerprint(Path(__file__).resolve().parents[2], source_root) == snapshot.source_fingerprint
