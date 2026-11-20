param(
    [switch]$CheckOnly,
    [switch]$Resume,
    [string]$Output = 'artifacts/student-study/int8-extension'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repo

if (-not $CheckOnly) {
    $battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
    if (-not $battery -or -not @($battery | Where-Object { $_.BatteryStatus -eq 2 }).Count) {
        throw 'Connect AC power before the INT8 run. The earlier CPU benchmark used AC.'
    }
}

$arguments = @('run', 'python', '-u', 'scripts/run_int8_extension.py', '--output', $Output)
if ($CheckOnly) { $arguments += '--check-only' }
if ($Resume) { $arguments += '--resume' }

$log = Join-Path $repo 'artifacts/student-study/int8-extension-console.log'
Start-Transcript -Path $log -Append | Out-Null
try {
    & uv @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "INT8 extension failed with exit code $LASTEXITCODE. See $log"
    }
    if (-not $CheckOnly) {
        Write-Host "Completed. Read $Output/summary.json"
    }
} finally {
    Stop-Transcript | Out-Null
}
