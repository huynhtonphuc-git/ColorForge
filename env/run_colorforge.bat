@echo off
REM ColorForge app window. Close this console to stop the app.
set "PY=%~dp0cf\python.exe"
if not exist "%PY%" set "PY=%~dp0cf\Scripts\python.exe"
"%PY%" -W ignore "%~dp0..\src\colorforge_app\server.py" %*
