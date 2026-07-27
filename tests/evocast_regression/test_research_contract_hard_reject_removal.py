import subprocess

from evocast.build.contract_compiler import model_entry_contract_command
from evocast.policy.model_contract import validate_model_config


def test_unschematized_hparams_are_not_rejected_by_global_type_tables():
    result = validate_model_config(
        {
            "model_name": "",
            "model_hyper_params": {
                "patch_len": [8],
                "stride": [4],
                "seq_len": "already-validated-by-baseline",
            },
        },
        require_import=False,
    )

    assert result["status"] == "ok"
    assert result["errors"] == []


def test_source_entry_command_accepts_adapter_without_model_forward(tmp_path):
    source = tmp_path / "PDF.py"
    source.write_text(
        "class PDF:\n"
        "    def _init_model(self):\n"
        "        return object()\n",
        encoding="utf-8",
    )

    command = model_entry_contract_command(str(source))
    completed = subprocess.run(command, text=True, capture_output=True, check=False)

    assert completed.returncode == 0
    assert "SOURCE_PARSE_OK" in completed.stdout
