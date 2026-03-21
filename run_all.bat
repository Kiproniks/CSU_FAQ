@echo off
setlocal
cd /d "%~dp0"
set "ROOT=%cd%"

echo [run_all.bat] Cleaning old web/bot processes from this project...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$root = [Regex]::Escape($env:ROOT); Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match $root -and ($_.CommandLine -like '*run_web.py*' -or $_.CommandLine -like '*run_bot.py*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

if not exist "venv\Scripts\python.exe" (
  echo [run_all.bat] venv not found. Create it first:
  echo   python -m venv venv
  echo   venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)

echo [run_all.bat] Starting stable supervisor...
"venv\Scripts\python.exe" run_all.py

endlocal
