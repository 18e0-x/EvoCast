from __future__ import annotations

import torch

from evocast.variant.contract import _instantiate_probe_model, _probe_variant_behavior_delta_in_process


def test_behavior_delta_accepts_changed_target_hook_even_if_final_output_matches(monkeypatch) -> None:
    dummy_model = object()

    monkeypatch.setattr(
        "evocast.variant.contract._instantiate_probe_model",
        lambda **_kwargs: (dummy_model, {"seq_len": 8, "pred_len": 4, "enc_in": 1, "dec_in": 1, "c_out": 1}, torch.device("cpu"), {}),
    )
    monkeypatch.setattr(
        "evocast.variant.contract._probe_inputs",
        lambda *_args, **_kwargs: (
            torch.ones(1, 8, 1),
            torch.ones(1, 4, 1),
            torch.ones(1, 8, 1),
            torch.ones(1, 4, 1),
            {},
        ),
    )
    monkeypatch.setattr("evocast.variant.contract._run_probe", lambda *_args, **_kwargs: torch.zeros(1, 4, 1))
    monkeypatch.setattr(
        "evocast.variant.contract._capture_target_hook_delta",
        lambda *_args, **kwargs: {
            "fit_point": kwargs.get("fit_point") if "fit_point" in kwargs else "self.model.cluster.gate.distribution_fit",
            "comparison": {"same_shape": True, "exact_equal": False, "max_abs_diff": 0.25, "mean_abs_diff": 0.1},
            "exact_equal": False,
        },
    )

    result = _probe_variant_behavior_delta_in_process(
        tfb_config={},
        baseline_entry={"model_name": "baseline.Model", "model_hyper_params": {}},
        variant_entry={"model_name": "variant.Model", "model_hyper_params": {}},
        variant_path="workspace/round_entry.py",
        seed=2021,
        fit_point="self.model.cluster.gate.distribution_fit",
    )

    assert result["status"] == "ok"
    assert result["suspected_noop"] is False
    assert result["target_hook_delta"]["exact_equal"] is False
    assert result["reason"] == "baseline and variant probe outputs differ"


def test_variant_behavior_delta_wraps_workspace_model_with_adapter(monkeypatch) -> None:
    captured = {}

    class DummyInnerModel:
        pass

    class DummyWrappedModel:
        def __init__(self):
            self.model = torch.nn.Identity()

    monkeypatch.setattr(
        "evocast.variant.workspace_loader.load_model_class",
        lambda **_kwargs: DummyInnerModel,
    )
    monkeypatch.setattr(
        "ts_benchmark.baselines.time_series_library.adapters_for_transformers.transformer_adapter",
        lambda model_info: {
            "model_factory": lambda **kwargs: (captured.update({"kwargs": dict(kwargs)}) or DummyWrappedModel())
        },
    )

    model, _hparams, _device, _entry = _instantiate_probe_model(
        tfb_config={},
        model_entry={
            "model_name": "ts_benchmark.baselines.time_series_library.models.Crossformer.Crossformer",
            "adapter": "transformer_adapter",
            "model_hyper_params": {"num_workers": 0, "seq_len": 8, "pred_len": 4, "enc_in": 1, "dec_in": 1, "c_out": 1},
        },
        seed=2021,
        variant_path="evocast/task_knowledge/demo/rounds/Ablation001/workspace/round_entry.py",
    )

    assert model is not None
    assert captured["kwargs"]["num_workers"] == 0
