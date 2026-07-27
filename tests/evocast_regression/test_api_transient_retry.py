from __future__ import annotations

import io
import urllib.error

from evocast.harness.api_client import ProviderAPIError, ProviderClient


class RetryProbeClient(ProviderClient):
    def __init__(self, statuses: list[int]):
        self.statuses = list(statuses)
        self.attempts = 0

    def _request(self, stage, round_num, body, timeout_sec, attempt, execution_label=None, logical_call_id=None):
        self.attempts += 1
        status = self.statuses[min(self.attempts - 1, len(self.statuses) - 1)]
        raise urllib.error.HTTPError(
            url="https://example.invalid",
            code=status,
            msg="synthetic",
            hdrs=None,
            fp=io.BytesIO(b'{"type":"error","error":{"type":"overloaded_error"}}'),
        )


def expect_attempts(statuses: list[int], *, retries: int, expected_attempts: int) -> None:
    client = RetryProbeClient(statuses)
    try:
        client.request_message_with_retries(
            stage="proposal",
            round_num=1,
            body={"model": "synthetic"},
            timeout_sec=1,
            max_retries=retries,
        )
    except ProviderAPIError:
        pass
    if client.attempts != expected_attempts:
        raise RuntimeError(
            f"expected {expected_attempts} attempts for statuses={statuses}, got {client.attempts}"
        )


def main() -> int:
    expect_attempts([529, 529], retries=1, expected_attempts=2)
    expect_attempts([502, 502], retries=1, expected_attempts=2)
    expect_attempts([400, 400], retries=3, expected_attempts=1)
    print({"status": "ok", "checks": ["529 retried", "502 retried", "400 not retried"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
