from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

from evocast.harness.api_client import ProviderClient
from evocast.state.cost_ledger import append_cost_event, load_cost_events
from evocast.state.stage_timing import build_stage_timing_summary


def _config(path: Path) -> None:
    path.write_text(
        """
provider: minimax
base_url: https://example.invalid/v1
chat_completions_path: /chat/completions
api_key_env: COST_LEDGER_TEST_KEY
stream: false
max_retries: 1
models: MiniMax-M3
thinking: {default: false}
effort: {default: medium}
max_tokens: {default: 1000}
pricing:
  currency: USD
  unit: per_1m_tokens
  tiers:
    - max_input_tokens:
      input: 0.3
      output: 1.2
      cache_read: 0.06
""".strip()
        + "\n",
        encoding="utf-8",
    )


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "total_tokens": 125,
                    "prompt_cache_hit_tokens": 20,
                    "prompt_cache_miss_tokens": 80,
                },
                "choices": [{"message": {"content": "ok"}}],
            }
        ).encode("utf-8")


def test_each_http_attempt_is_a_distinct_ledger_event(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "minimax.yaml"
    _config(config_path)
    monkeypatch.setenv("COST_LEDGER_TEST_KEY", "test-key")
    calls = {"count": 0}

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.HTTPError(
                url="https://example.invalid",
                code=529,
                msg="busy",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"busy"}'),
            )
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("evocast.harness.api_client.time.sleep", lambda _: None)
    client = ProviderClient(config_path=config_path, task_id="ledger_task", base_dir=str(tmp_path))

    assert client.call_text(stage="proposal", round_num=2, messages=[{"role": "user", "content": "x"}]) == "ok"
    events = [item for item in load_cost_events(str(tmp_path), "ledger_task") if item.get("kind") == "llm_api"]

    assert len(events) == 2
    assert {item["status"] for item in events} == {"failed", "success"}
    assert len({item["event_id"] for item in events}) == 2
    assert len({item["logical_call_id"] for item in events}) == 1
    assert [item["transport_attempt"] for item in events] == [1, 2]
    success = next(item for item in events if item["status"] == "success")
    assert success["usage"]["total_tokens"] == 125
    assert success["cost"]["total_cost"] > 0
    assert success["elapsed_seconds"] >= 0


def test_stage_timing_reads_only_the_unified_ledger(tmp_path: Path) -> None:
    append_cost_event(
        str(tmp_path),
        "ledger_stage_task",
        {
            "kind": "stage",
            "stage": "dataset_diagnosis",
            "status": "success",
            "elapsed_seconds": 1.25,
        },
    )
    append_cost_event(
        str(tmp_path),
        "ledger_stage_task",
        {
            "kind": "execution",
            "stage": "formal_experiment",
            "status": "success",
            "elapsed_seconds": 2.5,
            "details": {"model_key": "PatchTST"},
        },
    )

    summary = build_stage_timing_summary("ledger_stage_task", str(tmp_path))

    assert summary["ledger_path"].endswith("cost_ledger.jsonl")
    assert summary["totals"]["execution_elapsed_seconds"] == 2.5
    assert {row["name"] for row in summary["rows"]} == {"dataset_diagnosis", "formal_experiment"}
