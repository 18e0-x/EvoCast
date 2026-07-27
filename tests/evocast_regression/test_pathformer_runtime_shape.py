from __future__ import annotations

import torch

from ts_benchmark.baselines.pathformer.pathformer import Pathformer


def test_pathformer_binds_num_nodes_to_the_inferred_runtime_channels() -> None:
    """ETTh1-like seven-variable input must not retain Pathformer's 21-node default."""
    adapter = Pathformer(
        seq_len=96,
        pred_len=96,
        horizon=96,
        d_model=4,
        d_ff=64,
        k=2,
        layer_nums=3,
        num_experts_list=[4, 4, 4],
        patch_size_list=[[8, 6, 4, 2]] * 3,
        num_nodes=21,
        enc_in=1,
        dec_in=1,
        c_out=1,
    )
    # This is the value set by DeepForecastingModelBase after inspecting the
    # real multi-variate training dataframe and before calling _init_model.
    adapter.config.enc_in = adapter.config.dec_in = adapter.config.c_out = 7

    model = adapter._init_model()
    output, _ = model(torch.randn(1, 96, 7))

    assert adapter.config.num_nodes == 7
    assert model.AMS_lists[0].start_linear.in_features == 7
    assert tuple(output.shape) == (1, 96, 7)
