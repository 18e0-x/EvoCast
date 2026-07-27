from __future__ import annotations

from evocast.runners.tfb_pipeline_runner import run_smoke


def test_run_smoke_forces_epoch1_batch1_patience1(monkeypatch) -> None:
    captured: dict = {}

    def _fake_run_single_model(**kwargs):
        captured.update(kwargs)
        return {"success": True}

    monkeypatch.setattr("evocast.runners.tfb_pipeline_runner.run_single_model", _fake_run_single_model)

    result = run_smoke(
        config_path="compiled_config.json",
        model_name="DTAF",
        model_hyper_params={
            "batch_size": 64,
            "num_epochs": 10,
            "patience": 5,
        },
    )

    hparams = captured["model_hyper_params"]
    assert result["success"] is True
    assert hparams["batch_size"] == 1
    assert hparams["num_epochs"] == 1
    assert hparams["patience"] == 1
    assert hparams["max_train_batches"] == 1
    assert hparams["max_val_batches"] == 1
