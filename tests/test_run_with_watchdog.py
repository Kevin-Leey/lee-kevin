from __future__ import annotations

import sys
from pathlib import Path

import tools.run_with_watchdog as watchdog


def test_retry_limit_is_bounded_and_event_log_hides_command(
    tmp_path: Path, monkeypatch
) -> None:
    event_log = tmp_path / "events.log"
    calls: list[list[str]] = []
    secret_argument = "secret-must-not-be-logged"

    def fake_run_once(command, _idle, _grace, _report):
        calls.append(list(command))
        return 1, False

    monkeypatch.setattr(watchdog, "_run_once", fake_run_once)
    monkeypatch.setattr(watchdog.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_with_watchdog.py",
            "--attempts",
            "2",
            "--backoff",
            "0",
            "--max-backoff",
            "0",
            "--event-log",
            str(event_log),
            "--",
            "remote-client",
            "--api-key",
            secret_argument,
        ],
    )

    assert watchdog.main() == 1
    assert len(calls) == 2
    log_text = event_log.read_text(encoding="utf-8")
    assert "circuit breaker opened after 2 attempts" in log_text
    assert secret_argument not in log_text
    assert "remote-client" not in log_text
