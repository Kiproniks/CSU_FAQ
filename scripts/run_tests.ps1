param(
    [string]$PythonExe = "",
    [string[]]$ExtraPytestArgs = @()
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $root "test_runs"
$logsDir = Join-Path $runRoot "logs"
$resultsDir = Join-Path $runRoot "results"
$tmpDir = Join-Path $runRoot "tmp"

New-Item -ItemType Directory -Path $logsDir, $resultsDir, $tmpDir -Force | Out-Null

function Test-Python([string]$Candidate) {
    if ([string]::IsNullOrWhiteSpace($Candidate)) { return $false }
    try {
        & $Candidate -c "print('ok')" *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

$candidates = @()
if ($PythonExe) { $candidates += $PythonExe }
$candidates += @(
    (Join-Path $root "venv\Scripts\python.exe"),
    "python",
    "C:\Program Files\PostgreSQL\18\pgAdmin 4\python\python.exe"
)

$selected = $null
foreach ($candidate in $candidates) {
    if (Test-Python $candidate) {
        $selected = $candidate
        break
    }
}

if (-not $selected) {
    Write-Error "Не найден рабочий Python. Укажи путь: .\scripts\run_tests.ps1 -PythonExe '<path>'"
    exit 1
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logsDir ("pytest_{0}.log" -f $timestamp)
$junitPath = Join-Path $resultsDir ("pytest_{0}.xml" -f $timestamp)
$tmpPath = Join-Path $tmpDir "pytest"

$pytestArgs = @(
    "-q",
    "--import-mode=importlib",
    "--basetemp", $tmpPath,
    "--junitxml", $junitPath
) + $ExtraPytestArgs

$rootPy = $root -replace "\\", "\\\\"
$argsJson = ($pytestArgs | ConvertTo-Json -Compress)
$argsJsonPy = $argsJson -replace "\\", "\\\\"
$wrapperPath = Join-Path $tmpDir "run_pytest_wrapper.py"
$wrapperCode = @"
import json
import site
import sys
from pathlib import Path

root = Path(r"$rootPy")
site.addsitedir(str(root / "venv" / "Lib" / "site-packages"))
sys.path.insert(0, str(root))

import pytest

args = json.loads(r'''$argsJsonPy''')
raise SystemExit(pytest.main(args))
"@
Set-Content -Path $wrapperPath -Value $wrapperCode -Encoding UTF8

Write-Host "Python: $selected"
Write-Host "Log: $logPath"
Write-Host "JUnit: $junitPath"

& $selected $wrapperPath 2>&1 | Tee-Object -FilePath $logPath
$code = $LASTEXITCODE
Write-Host "Exit code: $code"
exit $code
