@echo off
REM Start the MiniCluster ML coordinator on the teacher / server PC (Windows).
REM Dashboard: http://<this-pc-ip>:8000
cd /d "%~dp0\.."
if exist .venv\Scripts\python.exe (set PY=.venv\Scripts\python.exe) else (set PY=python)
echo Allow ports 8000 and 9000 in Windows Firewall if workers cannot connect.
%PY% -m minicluster.coordinator --http-port 8000 --worker-port 9000
pause
