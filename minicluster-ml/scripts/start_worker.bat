@echo off
REM Start a MiniCluster ML worker on a lab PC (Windows).
REM Usage:  scripts\start_worker.bat 192.168.1.10 PC-01
cd /d "%~dp0\.."
if "%~1"=="" (echo Usage: start_worker.bat COORDINATOR_IP [NAME] & exit /b 1)
set NAME=%~2
if "%NAME%"=="" set NAME=%COMPUTERNAME%
if exist .venv\Scripts\python.exe (set PY=.venv\Scripts\python.exe) else (set PY=python)
%PY% -m minicluster.worker --coordinator %1 --port 9000 --name %NAME%
pause
