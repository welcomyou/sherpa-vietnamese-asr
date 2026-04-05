@echo off
chcp 65001 >nul
setlocal

set "BASE_DIR=%~dp0"
set "PYTHON_EXE=%BASE_DIR%python\python.exe"
set "NSSM_EXE=%BASE_DIR%nssm.exe"
set "SVC_NAME=SherpaASR"

echo ===================================
echo  Cai dat Windows Service
echo ===================================
echo.

rem Kiem tra quyen Admin
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [LOI] Can chay voi quyen Administrator!
    echo Click phai file nay ^> Run as Administrator
    pause
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo [LOI] Khong tim thay Python embedded
    pause
    exit /b 1
)

echo Chon thao tac:
echo   1. Cai dat service
echo   2. Go bo service
echo   3. Khoi dong service
echo   4. Dung service
echo   5. Xem trang thai
echo.
set /p CHOICE="Nhap so (1-5): "

if "%CHOICE%"=="1" goto :install
if "%CHOICE%"=="2" goto :remove
if "%CHOICE%"=="3" goto :svc_start
if "%CHOICE%"=="4" goto :svc_stop
if "%CHOICE%"=="5" goto :status
echo Lua chon khong hop le.
pause
exit /b 1

:install
echo.
echo Dang cai dat service "%SVC_NAME%"...

rem Uu tien nssm (don gian, khong can pywin32)
if exist "%NSSM_EXE%" (
    "%NSSM_EXE%" install %SVC_NAME% "%PYTHON_EXE%" "%BASE_DIR%server_launcher.py" --no-gui
    "%NSSM_EXE%" set %SVC_NAME% DisplayName "Sherpa Vietnamese ASR Service"
    "%NSSM_EXE%" set %SVC_NAME% Description "Web service ASR tieng Viet"
    "%NSSM_EXE%" set %SVC_NAME% AppDirectory "%BASE_DIR%"
    "%NSSM_EXE%" set %SVC_NAME% Start SERVICE_AUTO_START
    echo.
    echo [OK] Da cai dat service. Khoi dong:
    echo   net start %SVC_NAME%
    echo   hoac: nssm start %SVC_NAME%
) else (
    rem Fallback: dung sc.exe tao service goi bat file
    echo Dang tao service bang sc.exe...
    sc create %SVC_NAME% binPath= "\"%PYTHON_EXE%\" \"%BASE_DIR%server_launcher.py\" --no-gui" start= auto DisplayName= "Sherpa Vietnamese ASR Service"
    if %errorlevel% equ 0 (
        echo [OK] Da cai dat. Khoi dong: net start %SVC_NAME%
    ) else (
        echo [LOI] Khong the cai dat service.
        echo Goi y: tai nssm.exe tu https://nssm.cc/download va dat vao thu muc nay.
    )
)
pause
exit /b 0

:remove
echo.
echo Dang dung va go bo service "%SVC_NAME%"...
net stop %SVC_NAME% >nul 2>&1
if exist "%NSSM_EXE%" (
    "%NSSM_EXE%" remove %SVC_NAME% confirm
) else (
    sc delete %SVC_NAME%
)
echo [OK] Da go bo service.
pause
exit /b 0

:svc_start
echo.
net start %SVC_NAME%
pause
exit /b 0

:svc_stop
echo.
net stop %SVC_NAME%
pause
exit /b 0

:status
echo.
sc query %SVC_NAME%
pause
exit /b 0
