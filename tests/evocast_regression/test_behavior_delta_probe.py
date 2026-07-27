from __future__ import annotations

import torch

from evocast.variant.contract import _probe_inputs, _run_probe


class _TailForecast(torch.nn.Module):
    def __init__(self, horizon: int, scale: float) -> None:
        super().__init__()
        self.horizon = horizon
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -self.horizon :, :] * self.scale


def test_behavior_delta_probe_uses_non_degenerate_signal() -> None:
    hparams = {
        "seq_len": 12,
        "label_len": 6,
        "horizon": 4,
        "enc_in": 3,
    }
    device = torch.device("cpu")
    x, target, input_mark, target_mark, meta = _probe_inputs(hparams, device, torch)

    assert meta["probe_signal"] == "deterministic_nonzero_wave"
    assert meta["input_shape"] == [2, 12, 3]
    assert meta["target_shape"] == [2, 10, 3]
    assert float(x.abs().sum().item()) > 0.0
    assert float(target.abs().sum().item()) > 0.0

    baseline_output = _run_probe(_TailForecast(horizon=4, scale=1.0), x, target, input_mark, target_mark)
    variant_output = _run_probe(_TailForecast(horizon=4, scale=1.5), x, target, input_mark, target_mark)

    assert baseline_output.shape == variant_output.shape == (2, 4, 3)
    assert not torch.equal(baseline_output, variant_output)
    assert float((baseline_output - variant_output).abs().max().item()) > 0.0
