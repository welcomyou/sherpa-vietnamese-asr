@echo off
chcp 65001 >nul
setlocal

set "BASE_DIR=%~dp0"
set "PYTHON_EXE=%BASE_DIR%python\python.exe"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Khong tim thay Python embedded
    pause
    exit /b 1
)

set "PYTHONHOME=%BASE_DIR%python"
set "PYTHONDONTWRITEBYTECODE=1"
set "QT6_BIN=%BASE_DIR%python\Lib\site-packages\PyQt6\Qt6\bin"
set "PATH=%QT6_BIN%;%BASE_DIR%python;%BASE_DIR%python\Lib\site-packages;%BASE_DIR%;%PATH%"

echo ===================================
echo  Sherpa Vietnamese ASR - Admin GUI
echo ===================================
echo.
echo Yeu cau: Windows 10 1809+ / Server 2019+
echo Neu loi, hay dung start-server.bat thay the.
echo.

"%PYTHON_EXE%" "%BASE_DIR%server_gui.py" %*

if %errorlevel% neq 0 (
    echo.
    echo [Loi] Khong the khoi dong GUI (ma loi %errorlevel%)
    echo Hay dung start-server.bat de chay headless.
    pause
)

exit /b %errorlevel%
