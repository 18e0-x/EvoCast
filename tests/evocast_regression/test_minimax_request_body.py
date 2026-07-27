from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evocast.harness.api_client import ProviderClient


def _write_minimax_config(path: Path, *, active_path_thinking: bool | None = None) -> None:
    lines = [
        'provider: "minimax"',
        'base_url: "https://example.invalid/v1"',
        'chat_completions_path: "/chat/completions"',
        'api_key_env: "SYNTHETIC_MINIMAX_KEY"',
        'token_param: "max_completion_tokens"',
        "minimax_reasoning_split: true",
        "stream: false",
        "max_retries: 0",
        'models: "MiniMax-M3"',
        "thinking:",
        "  default: true",
    ]
    if active_path_thinking is not None:
        lines.append(f"  active_path_ablation_target_plan: {str(active_path_thinking).lower()}")
    lines.extend(
        [
            "effort:",
            '  default: "medium"',
            "max_tokens:",
            "  default: 1000",
            "  active_path_ablation_target_plan: 3000",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_minimax_thinking_false_maps_to_disabled_request_body(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "minimax.yaml"
    _write_minimax_config(config_path, active_path_thinking=False)
    monkeypatch.setenv("SYNTHETIC_MINIMAX_KEY", "test-key")

    client = ProviderClient(config_path=config_path, task_id="minimax_thinking_body")
    body = client.build_request_body(
        stage="active_path_ablation_target_plan",
        messages=[{"role": "user", "content": "x"}],
        stream=False,
        expect_json=True,
    )

    assert body["thinking"] == {"type": "disabled"}
    assert body["reasoning_split"] is True
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_completion_tokens"] == 3000


def test_minimax_thinking_true_maps_to_adaptive_request_body(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "minimax.yaml"
    _write_minimax_config(config_path)
    monkeypatch.setenv("SYNTHETIC_MINIMAX_KEY", "test-key")

    client = ProviderClient(config_path=config_path, task_id="minimax_thinking_body")
    body = client.build_request_body(
        stage="proposal",
        messages=[{"role": "user", "content": "x"}],
        stream=False,
        expect_json=False,
    )

    assert body["thinking"] == {"type": "adaptive", "reasoning_effort": "medium"}
    assert body["reasoning_split"] is True
