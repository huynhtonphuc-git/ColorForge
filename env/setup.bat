@echo off
rem One-time setup for ColorForge: creates env\cf with Python 3.12 and the packages in requirements.txt.
setlocal
set "HERE=%~dp0"
if exist "%HERE%cf\python.exe" goto install
py -3.12 -m venv "%HERE%cf" || (echo Python 3.12 is required: https://www.python.org/downloads/ & exit /b 1)
:install
set "PY=%HERE%cf\Scripts\python.exe"
if exist "%HERE%cf\python.exe" set "PY=%HERE%cf\python.exe"
"%PY%" -m pip install --upgrade pip || exit /b 1
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || exit /b 1
"%PY%" -m pip install -r "%HERE%..\requirements.txt" || exit /b 1
echo Done. Start the app with env\run_colorforge.bat
