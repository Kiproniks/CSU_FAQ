param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$OriginUrl = "http://127.0.0.1:8000",
    [int]$WaitSeconds = 35
)

$ErrorActionPreference = "Stop"

function Find-CloudflaredPath {
    $candidate = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
    if (Test-Path $candidate) {
        return $candidate
    }

    $whereResult = & where.exe cloudflared 2>$null
    if ($LASTEXITCODE -eq 0 -and $whereResult) {
        return ($whereResult | Select-Object -First 1).Trim()
    }

    throw "cloudflared.exe not found. Install Cloudflare Tunnel first."
}

function Find-PythonPath {
    $preferred = "C:\Program Files\PostgreSQL\18\pgAdmin 4\python\python.exe"
    if (Test-Path $preferred) {
        return $preferred
    }

    $venvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    return "python"
}

$runtimeDir = Join-Path $ProjectRoot "test_runs\runtime"
New-Item -Path $runtimeDir -ItemType Directory -Force | Out-Null

$cfOutLog = Join-Path $runtimeDir "cloudflared.log"
$cfErrLog = Join-Path $runtimeDir "cloudflared.err.log"

$existingCloudflared = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "cloudflared.exe" }
foreach ($proc in $existingCloudflared) {
    try {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
    } catch {}
}

foreach ($logPath in @($cfOutLog, $cfErrLog)) {
    if (Test-Path $logPath) {
        Remove-Item $logPath -Force -ErrorAction SilentlyContinue
    }
}

$cloudflaredExe = Find-CloudflaredPath
Start-Process `
    -FilePath $cloudflaredExe `
    -ArgumentList @("tunnel", "--url", $OriginUrl, "--protocol", "http2", "--no-autoupdate") `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $cfOutLog `
    -RedirectStandardError $cfErrLog

$publicUrl = $null
$urlPattern = [regex]"https://[a-z0-9\-]+\.trycloudflare\.com"
$deadline = (Get-Date).AddSeconds($WaitSeconds)
while ((Get-Date) -lt $deadline) {
    if (Test-Path $cfErrLog) {
        $raw = Get-Content $cfErrLog -Raw -ErrorAction SilentlyContinue
        if ($raw) {
            $match = $urlPattern.Match($raw)
            if ($match.Success) {
                $publicUrl = $match.Value.ToLowerInvariant()
                break
            }
        }
    }
    Start-Sleep -Milliseconds 500
}

if (-not $publicUrl) {
    throw "Could not parse public URL from cloudflared log in ${WaitSeconds}s."
}

$envPath = Join-Path $ProjectRoot ".env"
$envRaw = if (Test-Path $envPath) { Get-Content $envPath -Raw } else { "" }
if ($envRaw -match "(?m)^PUBLIC_BASE_URL=.*$") {
    $envRaw = [regex]::Replace($envRaw, "(?m)^PUBLIC_BASE_URL=.*$", "PUBLIC_BASE_URL=$publicUrl")
} else {
    if ($envRaw.Length -gt 0 -and -not $envRaw.EndsWith("`n")) {
        $envRaw += "`r`n"
    }
    $envRaw += "PUBLIC_BASE_URL=$publicUrl`r`n"
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($envPath, $envRaw, $utf8NoBom)

$currentUrlPath = Join-Path $runtimeDir "current_public_url.txt"
[System.IO.File]::WriteAllText($currentUrlPath, $publicUrl + "`r`n", $utf8NoBom)

$runningBots = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like "python*.exe" -and $_.CommandLine -match "run_bot.py"
}
foreach ($proc in $runningBots) {
    try {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
    } catch {}
}

$pythonExe = Find-PythonPath
$botOutLog = Join-Path $ProjectRoot "test_runs\runtime_bot.log"
$botErrLog = Join-Path $ProjectRoot "test_runs\runtime_bot.err.log"
Start-Process `
    -FilePath $pythonExe `
    -ArgumentList @("-u", "scripts/bootstrap_runtime.py", "run_bot.py", "--project-root", $ProjectRoot) `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $botOutLog `
    -RedirectStandardError $botErrLog

Write-Output "PUBLIC_BASE_URL=$publicUrl"
Write-Output "cloudflared restarted and bot restarted."
