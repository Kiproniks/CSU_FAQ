param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$OriginUrl = "http://127.0.0.1:8000",
    [int]$BootAttempts = 8,
    [int]$WaitForUrlSeconds = 40,
    [int]$HealthIntervalSec = 20,
    [int]$FailureThreshold = 3,
    [switch]$EnableRemoteProbe,
    [switch]$RestartBotOnUrlChange
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

    throw "cloudflared.exe not found."
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

function Stop-Cloudflared {
    $existing = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "cloudflared.exe" }
    foreach ($proc in $existing) {
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        } catch {}
    }
}

function Test-CloudflaredAlive {
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "cloudflared.exe" -and $_.CommandLine -match [regex]::Escape($OriginUrl)
    }
    return @($procs).Count -gt 0
}

function Write-PublicBaseUrl([string]$PublicUrl) {
    $envPath = Join-Path $ProjectRoot ".env"
    $envRaw = if (Test-Path $envPath) { Get-Content $envPath -Raw } else { "" }

    if ($envRaw -match "(?m)^PUBLIC_BASE_URL=.*$") {
        $envRaw = [regex]::Replace($envRaw, "(?m)^PUBLIC_BASE_URL=.*$", "PUBLIC_BASE_URL=$PublicUrl")
    } else {
        if ($envRaw.Length -gt 0 -and -not $envRaw.EndsWith("`n")) {
            $envRaw += "`r`n"
        }
        $envRaw += "PUBLIC_BASE_URL=$PublicUrl`r`n"
    }

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($envPath, $envRaw, $utf8NoBom)

    $runtimeDir = Join-Path $ProjectRoot "test_runs\runtime"
    New-Item -Path $runtimeDir -ItemType Directory -Force | Out-Null
    $currentUrlPath = Join-Path $runtimeDir "current_public_url.txt"
    [System.IO.File]::WriteAllText($currentUrlPath, $PublicUrl + "`r`n", $utf8NoBom)
}

function Restart-Bot {
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
}

function Start-CloudflaredProcess([string]$CloudflaredExe, [string]$OutLog, [string]$ErrLog) {
    foreach ($logPath in @($OutLog, $ErrLog)) {
        if (Test-Path $logPath) {
            Remove-Item $logPath -Force -ErrorAction SilentlyContinue
        }
    }

    Start-Process `
        -FilePath $CloudflaredExe `
        -ArgumentList @("tunnel", "--url", $OriginUrl, "--protocol", "http2", "--ha-connections", "1", "--no-autoupdate") `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog
}

function Read-TunnelUrl([string]$ErrLog, [int]$WaitSeconds) {
    $urlPattern = [regex]"https://[a-z0-9\-]+\.trycloudflare\.com"
    $deadline = (Get-Date).AddSeconds($WaitSeconds)

    while ((Get-Date) -lt $deadline) {
        if (Test-Path $ErrLog) {
            $raw = Get-Content $ErrLog -Raw -ErrorAction SilentlyContinue
            if ($raw) {
                $all = $urlPattern.Matches($raw)
                if ($all.Count -gt 0) {
                    return $all[$all.Count - 1].Value.ToLowerInvariant()
                }
            }
        }
        Start-Sleep -Milliseconds 500
    }

    return ""
}

function Test-TunnelUrl([string]$PublicUrl) {
    if (-not $PublicUrl) {
        return $false
    }

    try {
        $resp = Invoke-WebRequest -UseBasicParsing -TimeoutSec 15 -Uri $PublicUrl
        $code = [int]$resp.StatusCode
        if ($code -ge 200 -and $code -lt 500 -and $code -ne 530) {
            return $true
        }
        return $false
    } catch {
        $statusCode = $null
        if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
            $statusCode = [int]$_.Exception.Response.StatusCode
        }

        if ($statusCode -and $statusCode -ge 200 -and $statusCode -lt 500 -and $statusCode -ne 530) {
            return $true
        }
        return $false
    }
}

function Ensure-HealthyTunnel([string]$CloudflaredExe, [string]$OutLog, [string]$ErrLog, [int]$Attempts, [int]$WaitSeconds) {
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        Stop-Cloudflared
        Start-CloudflaredProcess -CloudflaredExe $CloudflaredExe -OutLog $OutLog -ErrLog $ErrLog

        $publicUrl = Read-TunnelUrl -ErrLog $ErrLog -WaitSeconds $WaitSeconds
        if (-not $publicUrl) {
            Write-Host "Attempt ${attempt}/${Attempts}: cannot parse quick tunnel URL."
            continue
        }

        # Публикуем URL сразу, чтобы /admin всегда выдавал актуальную ссылку.
        Write-PublicBaseUrl $publicUrl
        Write-Host "Attempt ${attempt}/${Attempts}: parsed URL $publicUrl"

        if (-not $EnableRemoteProbe) {
            return $publicUrl
        }

        $probeDeadline = (Get-Date).AddSeconds(35)
        while ((Get-Date) -lt $probeDeadline) {
            if (Test-TunnelUrl -PublicUrl $publicUrl) {
                return $publicUrl
            }
            Start-Sleep -Seconds 2
        }

        Write-Host "Attempt ${attempt}/${Attempts}: tunnel URL is unhealthy ($publicUrl)."
    }

    return ""
}

$runtimeDir = Join-Path $ProjectRoot "test_runs\runtime"
New-Item -Path $runtimeDir -ItemType Directory -Force | Out-Null
$cfOutLog = Join-Path $runtimeDir "cloudflared.log"
$cfErrLog = Join-Path $runtimeDir "cloudflared.err.log"

$cloudflaredExe = Find-CloudflaredPath

$activeUrl = Ensure-HealthyTunnel `
    -CloudflaredExe $cloudflaredExe `
    -OutLog $cfOutLog `
    -ErrLog $cfErrLog `
    -Attempts $BootAttempts `
    -WaitSeconds $WaitForUrlSeconds

if (-not $activeUrl) {
    Write-PublicBaseUrl ""
    Write-Output "Failed to establish healthy tunnel."
    exit 1
}

Write-PublicBaseUrl $activeUrl
Write-Output "Tunnel ready: $activeUrl"

if ($RestartBotOnUrlChange) {
    Restart-Bot
}

$failureCount = 0
while ($true) {
    Start-Sleep -Seconds $HealthIntervalSec

    if (-not (Test-CloudflaredAlive)) {
        Write-Output "cloudflared process is down, recreating..."
        $newUrl = Ensure-HealthyTunnel `
            -CloudflaredExe $cloudflaredExe `
            -OutLog $cfOutLog `
            -ErrLog $cfErrLog `
            -Attempts $BootAttempts `
            -WaitSeconds $WaitForUrlSeconds

        if (-not $newUrl) {
            Write-PublicBaseUrl ""
            Write-Output "Tunnel recovery failed. Next retry in $HealthIntervalSec sec."
            continue
        }

        if ($newUrl -ne $activeUrl) {
            $activeUrl = $newUrl
            Write-PublicBaseUrl $activeUrl
            Write-Output "Tunnel rotated: $activeUrl"
            if ($RestartBotOnUrlChange) {
                Restart-Bot
            }
        } else {
            Write-Output "Tunnel recovered on same URL."
        }
        continue
    }

    if (-not $EnableRemoteProbe) {
        continue
    }

    if (Test-TunnelUrl -PublicUrl $activeUrl) {
        $failureCount = 0
        continue
    }

    $failureCount += 1
    Write-Output "Tunnel probe failed ($failureCount/$FailureThreshold) for $activeUrl"
    if ($failureCount -lt $FailureThreshold) {
        continue
    }
    $failureCount = 0
    Write-Output "Tunnel unhealthy, recreating..."
    $newUrl = Ensure-HealthyTunnel `
        -CloudflaredExe $cloudflaredExe `
        -OutLog $cfOutLog `
        -ErrLog $cfErrLog `
        -Attempts $BootAttempts `
        -WaitSeconds $WaitForUrlSeconds

    if (-not $newUrl) {
        Write-PublicBaseUrl ""
        Write-Output "Tunnel recovery failed. Next retry in $HealthIntervalSec sec."
        continue
    }

    if ($newUrl -ne $activeUrl) {
        $activeUrl = $newUrl
        Write-PublicBaseUrl $activeUrl
        Write-Output "Tunnel rotated: $activeUrl"
        if ($RestartBotOnUrlChange) {
            Restart-Bot
        }
    } else {
        Write-Output "Tunnel recovered on same URL."
    }
}
