"""Resume a Codex goal when its local rollout tree stops making progress.

The watchdog observes every local Codex rollout whose session metadata belongs
to one goal, including subagents.  If none of those rollouts changes for the
configured idle interval, it can invoke ``codex exec resume`` for an active
goal and, when explicitly enabled, a paused goal.  A blocked goal remains under
observation but is never resumed.  Failed or no-progress resume attempts are
retried with bounded exponential backoff and a hard circuit breaker.

Run this as a separate process.  It does not bypass a provider rate limit; it
cannot force the remote goal service to recover, distinguish a network pause
from an intentional pause, or make a silent long-running operation observable.
The state and event logs contain no credentials or full Codex command line.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PROMPT = (
    "Continue the active goal from its latest checkpoint. Inspect the current "
    "workspace before acting, preserve user changes, and complete the remaining "
    "work. The primary agent must write the paper; subagents may only audit."
)
RETRY_AFTER_RE = re.compile(r"retry(?:ing)?\s+after\s+(\d+(?:\.\d+)?)", re.I)


@dataclass
class WatchdogState:
    session_id: str
    status: str
    checked_at: str
    last_session_activity: str | None
    idle_seconds: float
    resume_attempts: int
    activity_log: str | None
    tracked_rollouts: int
    last_exit_code: int | None = None
    watchdog_status: str = "monitoring"
    max_resume_attempts: int = 0
    resume_paused: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    return Path(configured) if configured else Path.home() / ".codex"


def _session_goal_id(candidate: Path) -> str | None:
    try:
        with candidate.open("r", encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline()
        record = json.loads(first_line)
    except (json.JSONDecodeError, OSError):
        return None
    if record.get("type") != "session_meta":
        return None
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    goal_id = payload.get("session_id")
    return goal_id if isinstance(goal_id, str) else None


def _find_session_logs(codex_home: Path, session_id: str) -> list[Path]:
    matches: list[Path] = []
    for candidate in (codex_home / "sessions").rglob("*.jsonl"):
        if _session_goal_id(candidate) == session_id:
            matches.append(candidate)
    if not matches:
        raise FileNotFoundError(f"no local rollout log found for {session_id}")
    return sorted(matches)


def _latest_session_activity(session_logs: list[Path]) -> tuple[Path, float]:
    available: list[tuple[float, Path]] = []
    for session_log in session_logs:
        try:
            available.append((session_log.stat().st_mtime, session_log))
        except OSError:
            continue
    if not available:
        raise FileNotFoundError("no readable local rollout log remains")
    modified, session_log = max(
        available, key=lambda item: (item[0], str(item[1]))
    )
    return session_log, modified


def _find_session_log(codex_home: Path, session_id: str) -> Path:
    session_log, _ = _latest_session_activity(
        _find_session_logs(codex_home, session_id)
    )
    return session_log


def _goal_status(codex_home: Path, session_id: str) -> str | None:
    database = codex_home / "goals_1.sqlite"
    if not database.exists():
        return None
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=2)
        try:
            row = connection.execute(
                "SELECT status FROM thread_goals WHERE thread_id = ?", (session_id,)
            ).fetchone()
        finally:
            connection.close()
    except (sqlite3.Error, OSError):
        return None
    return None if row is None else str(row[0])


def _atomic_json(path: Path, state: WatchdogState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(state), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{_utc_now()} {message}\n")


def _acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size < 1:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(descriptor)
        raise RuntimeError(f"watchdog already owns lock: {path}") from exc

    payload = f"{os.getpid()}\n".encode("ascii")
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.write(descriptor, payload)
    os.ftruncate(descriptor, len(payload))
    os.fsync(descriptor)
    return descriptor


def _release_lock(descriptor: int, path: Path) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
        try:
            path.unlink()
        except OSError:
            # A newly started watchdog may already have the path open.
            pass


def _resume_once(
    executable: str,
    session_id: str,
    workspace: Path,
    prompt: str,
    output_log: Path,
    child_idle_timeout: float,
) -> tuple[int, str]:
    command = [
        executable,
        "-C",
        str(workspace),
        "-a",
        "never",
        "-s",
        "workspace-write",
        "exec",
        "resume",
        "--skip-git-repo-check",
        session_id,
        prompt,
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=os.environ.copy(),
    )
    assert process.stdout is not None
    events: queue.Queue[str | None] = queue.Queue()

    def forward_output() -> None:
        try:
            for output_line in iter(process.stdout.readline, ""):
                events.put(output_line)
        finally:
            process.stdout.close()
            events.put(None)

    reader = threading.Thread(target=forward_output, daemon=True)
    reader.start()
    tail: list[str] = []
    last_output = time.monotonic()
    stream_closed = False
    with output_log.open("a", encoding="utf-8") as fh:
        while True:
            try:
                line = events.get(timeout=0.5)
            except queue.Empty:
                line = ""
            if line is None:
                stream_closed = True
            elif line:
                last_output = time.monotonic()
                fh.write(line)
                fh.flush()
                tail.append(line.rstrip())
                tail = tail[-30:]
            if process.poll() is not None and stream_closed and events.empty():
                break
            if time.monotonic() - last_output >= child_idle_timeout:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                tail.append(f"resume subprocess idle for {child_idle_timeout:.1f}s")
                reader.join(timeout=2)
                return 124, "\n".join(tail)
    reader.join(timeout=2)
    return int(process.returncode or 0), "\n".join(tail)


def _retry_delay(output: str, attempt: int, base: float, ceiling: float) -> float:
    match = RETRY_AFTER_RE.search(output)
    if match:
        return min(ceiling, max(base, float(match.group(1))))
    return min(ceiling, base * (2 ** max(0, attempt - 1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume an active or explicitly opted-in paused Codex goal after "
            "session inactivity."
        )
    )
    parser.add_argument("--session-id", required=True, help="Codex session UUID.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--idle-timeout", type=float, default=180.0)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--child-idle-timeout", type=float, default=180.0)
    parser.add_argument("--retry-backoff", type=float, default=15.0)
    parser.add_argument("--max-backoff", type=float, default=300.0)
    parser.add_argument(
        "--max-resume-attempts",
        type=int,
        default=8,
        help=(
            "Stop after this many consecutive resume attempts without observed "
            "rollout progress (default: 8)."
        ),
    )
    parser.add_argument(
        "--resume-paused",
        action="store_true",
        help=(
            "Treat a stale paused goal as resumable. A local watchdog cannot "
            "distinguish an intentional pause from a network-induced pause."
        ),
    )
    parser.add_argument(
        "--allow-unknown-status",
        action="store_true",
        help="Monitor even when the local goal database cannot provide a status.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--codex", default="codex", help="Codex CLI executable.")
    parser.add_argument("--session-log", type=Path)
    parser.add_argument("--state-dir", type=Path, default=Path(".agents/watchdog"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report a stale session without invoking Codex.",
    )
    parser.add_argument(
        "--once", action="store_true", help="Perform one status check and exit."
    )
    args = parser.parse_args()
    positive = (args.idle_timeout, args.poll_interval, args.child_idle_timeout)
    if any(value <= 0 for value in positive):
        parser.error("idle and polling intervals must be positive")
    if args.retry_backoff < 0 or args.max_backoff < 0:
        parser.error("retry delays must be nonnegative")
    if args.max_resume_attempts <= 0:
        parser.error("--max-resume-attempts must be positive")
    return args


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    state_dir = (workspace / args.state_dir).resolve()
    state_path = state_dir / f"{args.session_id}.json"
    event_log = state_dir / f"{args.session_id}.log"
    resume_output = state_dir / f"{args.session_id}.resume.log"
    lock_path = state_dir / f"{args.session_id}.lock"
    lock_fd = _acquire_lock(lock_path)
    attempts = 0
    last_exit_code: int | None = None
    try:
        codex_home = _codex_home()
        fixed_session_log = args.session_log.resolve() if args.session_log else None
        session_logs = (
            [fixed_session_log]
            if fixed_session_log
            else _find_session_logs(codex_home, args.session_id)
        )
        tracked_logs = set(session_logs)
        session_log, initial_modified = _latest_session_activity(session_logs)
        last_progress_marker = (initial_modified, str(session_log))
        _append_log(
            event_log,
            f"watching {len(session_logs)} associated rollout(s); latest {session_log}",
        )
        while True:
            if fixed_session_log is None:
                session_logs = _find_session_logs(codex_home, args.session_id)
                current_logs = set(session_logs)
                if current_logs != tracked_logs:
                    added = len(current_logs - tracked_logs)
                    removed = len(tracked_logs - current_logs)
                    _append_log(
                        event_log,
                        "tracking rollout set changed: "
                        f"{len(current_logs)} total, {added} added, {removed} removed",
                    )
                    tracked_logs = current_logs
            session_log, modified = _latest_session_activity(session_logs)
            status = _goal_status(codex_home, args.session_id)
            idle = max(0.0, time.time() - modified)
            progress_marker = (modified, str(session_log))
            if progress_marker > last_progress_marker:
                if attempts:
                    _append_log(
                        event_log,
                        "new rollout activity observed; clearing consecutive "
                        f"resume count {attempts}",
                    )
                attempts = 0
                last_exit_code = None
                last_progress_marker = progress_marker
            state = WatchdogState(
                session_id=args.session_id,
                status=status or "unknown",
                checked_at=_utc_now(),
                last_session_activity=datetime.fromtimestamp(
                    modified, timezone.utc
                ).isoformat(timespec="seconds"),
                idle_seconds=round(idle, 3),
                resume_attempts=attempts,
                activity_log=str(session_log),
                tracked_rollouts=len(session_logs),
                last_exit_code=last_exit_code,
                max_resume_attempts=args.max_resume_attempts,
                resume_paused=args.resume_paused,
            )
            _atomic_json(state_path, state)

            if status is None and not args.allow_unknown_status:
                state.watchdog_status = "status_unknown"
                _atomic_json(state_path, state)
                _append_log(
                    event_log,
                    "goal status is unavailable; watchdog exiting fail-closed",
                )
                return 3
            if status == "paused" and not args.resume_paused:
                state.watchdog_status = "paused_not_opted_in"
                _atomic_json(state_path, state)
                _append_log(
                    event_log,
                    "goal status is paused and --resume-paused is disabled; "
                    "watchdog exiting",
                )
                return 0
            if status == "blocked":
                state.watchdog_status = "waiting_on_blocked"
                _atomic_json(state_path, state)
                time.sleep(args.poll_interval)
                continue
            if status not in (None, "active", "paused"):
                state.watchdog_status = f"goal_{status}"
                _atomic_json(state_path, state)
                _append_log(event_log, f"goal status is {status}; watchdog exiting")
                return 0
            if idle < args.idle_timeout:
                if args.once:
                    print(
                        f"{status or 'unknown'}; last session activity {idle:.1f}s ago "
                        f"(< {args.idle_timeout:.1f}s)"
                    )
                    return 0
                time.sleep(args.poll_interval)
                continue

            # Recheck after a short guard interval to avoid racing a live response.
            guard = min(5.0, args.poll_interval)
            time.sleep(guard)
            guarded_logs = (
                [fixed_session_log]
                if fixed_session_log
                else _find_session_logs(codex_home, args.session_id)
            )
            guarded_log, guarded_modified = _latest_session_activity(guarded_logs)
            guarded_status = _goal_status(codex_home, args.session_id)
            if guarded_status != status:
                # In particular, an active-to-blocked transition must return to
                # the status gate instead of racing into a resume invocation.
                session_logs = guarded_logs
                continue
            if guarded_modified > modified or (
                guarded_log != session_log and guarded_modified >= modified
            ):
                session_logs = guarded_logs
                continue
            if args.dry_run:
                state.watchdog_status = "dry_run_stale"
                _atomic_json(state_path, state)
                _append_log(event_log, f"dry-run stale detection at {idle:.1f}s")
                print(
                    f"stale {status or 'unknown'} session detected after {idle:.1f}s"
                )
                return 2

            if attempts >= args.max_resume_attempts:
                state.watchdog_status = "retry_exhausted"
                _atomic_json(state_path, state)
                _append_log(
                    event_log,
                    "resume circuit breaker opened after "
                    f"{attempts} attempts without rollout progress",
                )
                print(
                    "resume circuit breaker opened without observed rollout progress",
                    file=sys.stderr,
                )
                return 75

            attempts += 1
            state.resume_attempts = attempts
            state.watchdog_status = "resuming"
            _atomic_json(state_path, state)
            _append_log(event_log, f"starting resume attempt {attempts}")
            code, output = _resume_once(
                args.codex,
                args.session_id,
                workspace,
                args.prompt,
                resume_output,
                args.child_idle_timeout,
            )
            last_exit_code = code
            state.last_exit_code = code
            state.resume_attempts = attempts
            state.watchdog_status = "resume_returned"
            _atomic_json(state_path, state)
            if code == 0:
                _append_log(
                    event_log,
                    "resume command returned successfully; awaiting rollout progress",
                )
                if args.once:
                    return 0
                time.sleep(args.poll_interval)
                continue

            delay = _retry_delay(
                output, attempts, args.retry_backoff, args.max_backoff
            )
            category = "rate-limited" if "429" in output else "failed"
            _append_log(
                event_log,
                f"resume attempt {category} with code {code}; retry in {delay:.1f}s",
            )
            if args.once:
                return code
            if attempts >= args.max_resume_attempts:
                continue
            time.sleep(delay)
    except KeyboardInterrupt:
        _append_log(event_log, "watchdog interrupted")
        return 130
    finally:
        _release_lock(lock_fd, lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
