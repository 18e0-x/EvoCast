from __future__ import annotations

from evocast.policy.error_taxonomy import ErrorLabel
from evocast.runners import baseline_runner


def test_runtime_preflight_rejects_a_model_that_hard_codes_cuda(monkeypatch):
    class CpuIncompatibleAdapter:
        def _init_model(self):
            raise AssertionError("Torch not compiled with CUDA enabled")

    monkeypatch.setattr(
        baseline_runner,
        "build_run_configs",
        lambda config_data, entries, save_path, seed: ({}, {"models": entries}, {}),
    )
    from ts_benchmark.models import model_loader

    monkeypatch.setattr(model_loader, "get_models", lambda config: [lambda: CpuIncompatibleAdapter()])

    result = baseline_runner._preflight_candidate_runtime(
        {},
        {"model_name": "Triformer", "model_hyper_params": {}},
        seed=2021,
    )

    assert result is not None
    assert result["error_type"] == ErrorLabel.ENVIRONMENT_INCOMPATIBLE.value
    assert "current cpu environment" in result["error_message"]
    assert "Torch not compiled with CUDA enabled" in result["error_message"]
