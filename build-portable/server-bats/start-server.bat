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

rem Doc host va port tu config.ini
set "HOST=0.0.0.0"
set "PORT=8443"
set "HTTP_MODE=0"
for /f "tokens=2 delims== " %%a in ('findstr /i "^host " "%BASE_DIR%config.ini" 2^>nul') do set "HOST=%%a"
for /f "tokens=2 delims== " %%a in ('findstr /i "^port " "%BASE_DIR%config.ini" 2^>nul') do set "PORT=%%a"
for /f "tokens=2 delims== " %%a in ('findstr /i "^http_mode " "%BASE_DIR%config.ini" 2^>nul') do set "HTTP_MODE=%%a"

if "%HTTP_MODE%"=="1" (set "PROTO=http") else (set "PROTO=https")

echo ===================================
echo  Sherpa Vietnamese ASR - Server
echo ===================================
echo.
echo  Bind:  %HOST%:%PORT%
echo.
echo  Truy cap:
if "%HOST%"=="0.0.0.0" (
    echo    %PROTO%://localhost:%PORT%
    echo    %PROTO%://[IP-may-nay]:%PORT%
) else (
    echo    %PROTO%://%HOST%:%PORT%
)
echo.
echo  Dang nhap admin de quan tri qua web.
echo  Nhan Ctrl+C de dung server.
echo ===================================
echo.

"%PYTHON_EXE%" "%BASE_DIR%server_launcher.py" --no-gui

if %errorlevel% neq 0 (
    echo.
    echo [Loi] Server dung voi ma loi %errorlevel%
    pause
)

exit /b %errorlevel%
