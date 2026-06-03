@echo off
cd /d "%~dp0\..\.."
set PYTHONDONTWRITEBYTECODE=1
set PYTHONUNBUFFERED=1
set TK_SILENCE_DEPRECATION=1

if not exist runs\logs mkdir runs\logs
set LOG=runs\logs\equilibrium_mhd_desktop_launcher.log
echo [launcher] %DATE% %TIME% > "%LOG%"
echo [launcher] cwd=%CD% >> "%LOG%"

py -3 -B modules\equilibrium_mhd\equilibrium_mhd_desktop.py >> "%LOG%" 2>&1
if %ERRORLEVEL% EQU 0 exit /b 0

python -B modules\equilibrium_mhd\equilibrium_mhd_desktop.py >> "%LOG%" 2>&1
if %ERRORLEVEL% EQU 0 exit /b 0

echo [launcher] no usable Python/Tk runtime found >> "%LOG%"
notepad "%LOG%"
exit /b 1
