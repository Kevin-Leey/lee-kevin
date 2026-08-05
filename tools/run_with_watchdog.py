"""Run a command with idle-timeout detection and bounded automatic retries.

This wrapper is intended for network-bound CLIs such as image generation or
remote-model probes. Pass credentials through environment variables; the
wrapper deliberately never prints the child command line.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Callable, Sequence


@dataclass(frozen=True)
class StreamEvent:
    stream: str
    line: str | None


def _forward_stream(
    pipe: IO[str], stream_name: str, events: queue.Queue[StreamEvent]
) -> None:
    try:
        for line in iter(pipe.readline, ""):
            events.put(StreamEvent(stream_name, line))
    finally:
        pipe.close()
        events.put(StreamEvent(stream_name, None))


def _stop_process(process: subprocess.Popen[str], grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_once(
    command: Sequence[str],
    idle_timeout: float,
    terminate_grace: float,
    report: Callable[[str], None],
) -> tuple[int, bool]:
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=os.environ.copy(),
    )
    assert process.stdout is not None
    assert process.stderr is not None

    events: queue.Queue[StreamEvent] = queue.Queue()
    threads = [
        threading.Thread(
            target=_forward_stream,
            args=(process.stdout, "stdout", events),
            daemon=True,
        ),
        threading.Thread(
            target=_forward_stream,
            args=(process.stderr, "stderr", events),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    last_activity = time.monotonic()
    closed_streams = 0
    timed_out = False
    while True:
        try:
            event = events.get(timeout=1.0)
        except queue.Empty:
            event = None

        if event is not None:
            if event.line is None:
                closed_streams += 1
            else:
                last_activity = time.monotonic()
                target = sys.stdout if event.stream == "stdout" else sys.stderr
                target.write(event.line)
                target.flush()

        if process.poll() is not None and closed_streams >= 2 and events.empty():
            break

        if time.monotonic() - last_activity >= idle_timeout:
            timed_out = True
            report(
                f"no child output for {idle_timeout:.1f}s; terminating attempt"
            )
            _stop_process(process, terminate_grace)
            break

    for thread in threads:
        thread.join(timeout=terminate_grace)
    return (124 if timed_out else int(process.returncode or 0), timed_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restart a command after prolonged silence or failure."
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=180.0,
        help="Seconds without stdout/stderr before restart (default: 180).",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=8,
        help="Maximum attempts before the circuit breaker opens (default: 8).",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=5.0,
        help="Initial retry delay in seconds (default: 5).",
    )
    parser.add_argument(
        "--max-backoff",
        type=float,
        default=60.0,
        help="Maximum retry delay in seconds (default: 60).",
    )
    parser.add_argument(
        "--terminate-grace",
        type=float,
        default=5.0,
        help="Grace period before killing a stalled child (default: 5).",
    )
    parser.add_argument(
        "--event-log",
        type=Path,
        help=(
            "Optional sanitized watchdog event log. Child output and the child "
            "command line are not copied into this file."
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command following '--'.",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a child command is required after '--'")
    if args.idle_timeout <= 0 or args.backoff < 0 or args.max_backoff < 0:
        parser.error("timeouts and backoff values must be nonnegative")
    if args.attempts <= 0:
        parser.error("--attempts must be positive")
    return args


def _event_reporter(path: Path | None) -> Callable[[str], None]:
    resolved = path.resolve() if path else None
    if resolved:
        resolved.parent.mkdir(parents=True, exist_ok=True)

    def report(message: str) -> None:
        line = f"[watchdog] {message}"
        print(line, file=sys.stderr, flush=True)
        if resolved:
            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            with resolved.open("a", encoding="utf-8") as fh:
                fh.write(f"{timestamp} {line}\n")

    return report


def main() -> int:
    args = parse_args()
    report = _event_reporter(args.event_log)
    attempt = 0
    last_code = 1
    try:
        while attempt < args.attempts:
            attempt += 1
            report(f"starting attempt {attempt}/{args.attempts}")
            last_code, _ = _run_once(
                args.command,
                args.idle_timeout,
                args.terminate_grace,
                report,
            )
            if last_code == 0:
                report("child completed successfully")
                return 0
            if attempt >= args.attempts:
                break
            delay = min(args.max_backoff, args.backoff * (2 ** (attempt - 1)))
            report(
                f"child exited with code {last_code}; retrying in {delay:.1f}s"
            )
            time.sleep(delay)
    except KeyboardInterrupt:
        report("interrupted")
        return 130

    report(
        f"circuit breaker opened after {args.attempts} attempts; "
        f"last code {last_code}"
    )
    return last_code


if __name__ == "__main__":
    raise SystemExit(main())
