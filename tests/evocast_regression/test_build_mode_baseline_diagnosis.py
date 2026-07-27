from pathlib import Path

import yaml

from evocast.policy.experiment_policy import DEFAULT_POLICY


ROOT = Path(__file__).resolve().parents[2]


def test_build_mode_does_not_skip_baseline_diagnosis_policy():
    policy_path = ROOT / "evocast" / "configs" / "policies" / "experiment.yaml"
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))

    assert "skip_baseline_diagnosis" not in dict(policy.get("build_mode") or {})
    assert "skip_baseline_diagnosis" not in dict(DEFAULT_POLICY.get("build_mode") or {})


def test_wizard_has_no_build_mode_baseline_diagnosis_skip_path():
    wizard_source = (ROOT / "evocast" / "scripts" / "wizard.py").read_text(encoding="utf-8")

    assert "build_mode_skip_baseline_diagnosis" not in wizard_source
    assert "skipped_build_mode" not in wizard_source
    assert "run_baseline_diagnosis_before_agent(" in wizard_source
