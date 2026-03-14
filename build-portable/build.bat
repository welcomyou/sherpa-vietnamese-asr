@echo off
chcp 65001 >nul
cd /d "%~dp0\.."

echo ===================================
echo  Build Sherpa Vietnamese ASR
echo ===================================
echo.
echo [1] Desktop App (sherpa-vietnamese-asr)
echo [2] Web Service (sherpa-vietnamese-asr-service)
echo [3] Build cả 2
echo.
set /p choice="Chọn (1/2/3): "

if "%choice%"=="1" goto desktop
if "%choice%"=="2" goto service
if "%choice%"=="3" goto both
echo Lựa chọn không hợp lệ.
pause
exit /b 1

:desktop
echo.
echo [Build] Desktop App...
python build-portable\build_portable.py
goto done

:service
echo.
echo [Build] Web Service...
python build-portable\build_portable_online.py
goto done

:both
echo.
echo [Build] Desktop App...
python build-portable\build_portable.py
if errorlevel 1 (
    echo [ERROR] Desktop build failed!
    pause
    exit /b 1
)
echo.
echo [Build] Web Service...
python build-portable\build_portable_online.py
goto done

:done
echo.
echo ===================================
echo  Build hoàn tất!
echo ===================================
pause
