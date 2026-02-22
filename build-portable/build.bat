@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
echo [Build] Building ASR Portable...
python build-portable\build_portable.py
pause
