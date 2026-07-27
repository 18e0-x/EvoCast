from __future__ import annotations

from pathlib import Path

import torch

from evocast.research.mechanism.probe import run_mechanism_probe
from evocast.variant.import_isolation import model_execution_import_context


class _ProbeModel(torch.nn.Module):
    def __init__(self, horizon: int = 2) -> None:
        super().__init__()
        self.horizon = horizon
        self.projection = torch.nn.Linear(3, 3)

    def _process(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        input_mark: torch.Tensor,
        target_mark: torch.Tensor,
    ) -> torch.Tensor:
        del target, input_mark, target_mark
        return self.projection(x[:, -self.horizon :, :])


def _write_workspace_module(workspace: Path, source: str) -> Path:
    module_path = workspace / "ts_benchmark" / "mechanism_fixture.py"
    module_path.parent.mkdir(parents=True)
    (module_path.parent / "__init__.py").write_text("", encoding="utf-8")
    module_path.write_text(source, encoding="utf-8")
    return module_path


def _run_workspace_probe(workspace: Path, module_path: Path) -> dict:
    with model_execution_import_context(source_checkout=workspace):
        return run_mechanism_probe(
            model=_ProbeModel(),
            variant_path=str(workspace / "round_entry.py"),
            source_entrypoint={
                "factory_model_source": str(module_path),
                "inner_model_source": "",
                "workspace_root": str(workspace),
            },
            task_shape={"batch": 2, "seq_len": 8, "horizon": 2, "channels": 3},
        )


def test_mechanism_probe_isolates_sequential_workspace_variants(tmp_path: Path) -> None:
    research_workspace = tmp_path / "task_knowledge" / "fixture" / "rounds" / "Research003" / "workspace"
    ablation_workspace = tmp_path / "task_knowledge" / "fixture" / "rounds" / "Ablation001" / "workspace"
    research_module = _write_workspace_module(
        research_workspace,
        """
import torch

class ExponentialSmoothing(torch.nn.Module):
    def __init__(self, dim, heads, aux=False):
        super().__init__()
        self.aux = aux
        self.heads = heads

    def forward(self, values, aux_values=None):
        if self.aux and self.heads > 1:
            raise RuntimeError("real-channel auxiliary path is invalid")
        return values
""".strip()
        + "\n",
    )
    ablation_module = _write_workspace_module(
        ablation_workspace,
        """
def conv1d_fft(f, g, dim=1):
    raise RuntimeError("time-dimension FFT contract is invalid")
""".strip()
        + "\n",
    )

    research = _run_workspace_probe(research_workspace, research_module)
    ablation = _run_workspace_probe(ablation_workspace, ablation_module)

    assert research["status"] == "failed"
    assert ablation["status"] == "failed"
    assert research["failure_chain"]["first_failure"] == "ExponentialSmoothing.aux_real_channels"
    assert ablation["failure_chain"]["first_failure"] == "operator.conv1d_fft.time_dim"
    assert "Research003" in research["variant_path"]
    assert "Ablation001" in ablation["variant_path"]
