@echo off
REM ==========================================
REM Setup Trading AI Auto-Start
REM This script copies the autostart file to Windows Startup folder
REM ==========================================

cd /d "%~dp0"

echo ==========================================
echo   Trading AI Auto-Start Setup
echo ==========================================
echo.

REM Define paths
set STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set SOURCE_FILE=Trading_AI_Autostart.bat
set DEST_FILE=%STARTUP_FOLDER%\Trading_AI_Autostart.bat

echo Startup folder: %STARTUP_FOLDER%
echo.

REM Check if source exists, create if not
if not exist "%SOURCE_FILE%" (
    echo Creating %SOURCE_FILE%...
    (
        echo @echo off
        echo REM Auto-generated startup script
        echo timeout /t 30 /nobreak ^>nul
        echo.
        echo REM Launch MT5
        echo set MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
        echo if exist "%%MT5_PATH%%" (
        echo     echo Starting MT5 Terminal...
        echo     start "" "%%MT5_PATH%%"
        echo     echo Waiting for MT5 to initialize and login...
        echo     timeout /t 60 /nobreak ^>nul
        echo ^)
        echo.
        echo REM Launch Trading AI
        echo echo Starting Trading AI Bot...
        echo start "" /MIN "C:\Users\mehed\Herd\trading_ai\run.bat"
        echo echo Trading AI startup complete!
        echo exit
    ) > "%SOURCE_FILE%"
    echo Source file created.
    echo.
)

REM Check if destination is same as source (prevent self-copy)
if /i "%DEST_FILE%"=="%~dp0%SOURCE_FILE%" (
    echo ERROR: Destination is same as source!
    echo Please run this from the trading_ai folder.
    pause
    exit /b 1
)

REM Copy to startup folder
echo Copying %SOURCE_FILE% to Startup folder...
copy /Y "%SOURCE_FILE%" "%DEST_FILE%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==========================================
    echo   SUCCESS! Auto-start is now enabled.
    echo ==========================================
    echo.
    echo The trading bot will now start automatically
    echo when you turn on your computer.
    echo.
    echo MT5 path: C:\Program Files\MetaTrader 5\terminal64.exe
    echo.
    echo To DISABLE auto-start, delete this file:
    echo %DEST_FILE%
    echo.
) else (
    echo.
    echo ERROR: Failed to copy to Startup folder.
    echo Please run as Administrator.
)

pause
