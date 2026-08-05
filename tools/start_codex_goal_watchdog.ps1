[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9A-Fa-f-]{36}$')]
    [string]$SessionId,

    [string]$Workspace = (Get-Location).Path,

    [ValidateRange(1, 86400)]
    [int]$IdleSeconds = 180,

    [ValidateRange(1, 100)]
    [int]$MaxResumeAttempts = 8,

    [switch]$DoNotResumePaused,
    [switch]$DryRun,
    [switch]$Foreground
)

$ErrorActionPreference = 'Stop'

function ConvertTo-QuotedProcessArgument {
    param([Parameter(Mandatory = $true)][string]$Value)
    return '"' + $Value.Replace('"', '\"') + '"'
}

$workspacePath = (Resolve-Path -LiteralPath $Workspace).Path
$watchdogScript = (Resolve-Path -LiteralPath (
    Join-Path $PSScriptRoot 'codex_goal_watchdog.py'
)).Path
$pythonCommand = Get-Command python -CommandType Application -ErrorAction Stop |
    Select-Object -First 1

$stateDirectory = Join-Path $workspacePath '.agents\watchdog'
New-Item -ItemType Directory -Force -Path $stateDirectory | Out-Null
$pidPath = Join-Path $stateDirectory ($SessionId + '.daemon.pid')
$stdoutPath = Join-Path $stateDirectory ($SessionId + '.daemon.stdout.log')
$stderrPath = Join-Path $stateDirectory ($SessionId + '.daemon.stderr.log')

if (Test-Path -LiteralPath $pidPath) {
    $existingProcessId = 0
    [void][int]::TryParse(
        (Get-Content -Raw -LiteralPath $pidPath).Trim(),
        [ref]$existingProcessId
    )
    $existingProcess = $null
    if ($existingProcessId -gt 0) {
        $existingProcess = Get-CimInstance Win32_Process -Filter (
            'ProcessId = ' + $existingProcessId
        ) -ErrorAction SilentlyContinue
    }
    if (
        $null -ne $existingProcess -and
        $existingProcess.CommandLine -like '*codex_goal_watchdog.py*' -and
        $existingProcess.CommandLine -like ('*' + $SessionId + '*')
    ) {
        Write-Output (
            'Codex goal watchdog is already running (PID {0}).' -f
            $existingProcessId
        )
        return
    }
    Remove-Item -LiteralPath $pidPath -Force
}

$watchdogArgs = @(
    $watchdogScript,
    '--session-id', $SessionId,
    '--workspace', $workspacePath,
    '--idle-timeout', [string]$IdleSeconds,
    '--child-idle-timeout', [string]$IdleSeconds,
    '--poll-interval', '10',
    '--max-resume-attempts', [string]$MaxResumeAttempts,
    '--state-dir', '.agents/watchdog'
)
if (-not $DoNotResumePaused) {
    $watchdogArgs += '--resume-paused'
}
if ($DryRun) {
    $watchdogArgs += @('--dry-run', '--once')
}

if ($Foreground -or $DryRun) {
    & $pythonCommand.Source @watchdogArgs
    exit $LASTEXITCODE
}

$processArgs = foreach ($argument in $watchdogArgs) {
    ConvertTo-QuotedProcessArgument -Value $argument
}
$watchdogProcess = Start-Process `
    -FilePath $pythonCommand.Source `
    -ArgumentList $processArgs `
    -WorkingDirectory $workspacePath `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

Set-Content -LiteralPath $pidPath -Value $watchdogProcess.Id -Encoding ASCII
Start-Sleep -Milliseconds 300
$watchdogProcess.Refresh()
if ($watchdogProcess.HasExited) {
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
    throw (
        'Codex goal watchdog exited during startup with code {0}. See {1}' -f
        $watchdogProcess.ExitCode, $stderrPath
    )
}

Write-Output (
    'Started Codex goal watchdog (PID {0}); state: {1}' -f
    $watchdogProcess.Id, $stateDirectory
)
