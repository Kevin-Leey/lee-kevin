# Codex and API Watchdogs

These wrappers handle two separate failure modes. They do not contain API keys
and inherit the caller's environment.

## Codex goal monitor

`tools/codex_goal_watchdog.py` watches the modification time of every local
rollout associated with one root Codex session, including subagents. After 180
seconds without local rollout activity, it runs the supported non-interactive
command `codex exec resume <session-id> <prompt>`.

Start it as a hidden Windows process:

```powershell
powershell -ExecutionPolicy Bypass -File tools\start_codex_goal_watchdog.ps1 `
  -SessionId 019f7b04-d2f3-7371-922b-67305a972639
```

The launcher opts into resuming a stale `paused` goal because this project uses
that status for connection interruptions. Use `-DoNotResumePaused` when a
manual pause must remain paused. The local status does not identify why a goal
was paused.

Run one non-mutating check before enabling the daemon:

```powershell
powershell -ExecutionPolicy Bypass -File tools\start_codex_goal_watchdog.ps1 `
  -SessionId 019f7b04-d2f3-7371-922b-67305a972639 -DryRun
```

Dry-run exit code `0` means activity is newer than the timeout. Exit code `2`
means the goal is stale and would be resumed. Status, event, resume-output, and
PID files are written under `.agents/watchdog/`.

The resume circuit breaker defaults to eight consecutive attempts. Only new
rollout activity clears that counter; a CLI exit code of zero without new
activity does not. This prevents a permanently unavailable provider or an
invalid session from creating an infinite restart loop.

To stop a running daemon, read its PID file and terminate that exact process:

```powershell
$watchdogPid = Get-Content `
  .agents\watchdog\019f7b04-d2f3-7371-922b-67305a972639.daemon.pid
Stop-Process -Id $watchdogPid
```

## Network-bound child commands

`tools/run_with_watchdog.py` restarts a specified command after 180 seconds
without stdout or stderr, or after a nonzero exit. Retries are bounded by
default:

```powershell
$env:REMOTE_API_KEY = '<set outside source control>'
python tools\run_with_watchdog.py --idle-timeout 180 --attempts 8 `
  --event-log .agents\watchdog\remote-probe.log -- `
  python tools\remote_probe.py
```

Do not put a key in the watchdog command line. The child must emit progress or
heartbeat output; increase `--idle-timeout` for legitimate silent operations.
The retry count must be positive; unbounded retry mode is intentionally not
available.

## Recovery boundary

This monitor cannot restart the VS Code extension, repair a provider outage,
or directly force a cloud goal from `paused` to `active`. It can only record
local inactivity and ask an authenticated Codex CLI to create a continuation.
If authentication, the provider, or the cloud goal service remains unavailable,
the bounded retries stop and the state records `retry_exhausted`.
