@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat first to install the project packages.
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 -m streamlit run app.py %*
