from __future__ import annotations

from evocast.tools.tfb_experiment import _eligible_for_promotion, _evaluation_stage


def test_buildmode_research_experiment_uses_buildmode_stage() -> None:
    assert _evaluation_stage("unified", False, build_mode=True) == "build_mode"
    assert _eligible_for_promotion("build_mode") is True


def test_explicit_smoke_stays_smoke_inside_buildmode_task() -> None:
    assert _evaluation_stage("unified", True, build_mode=True) == "smoke"
