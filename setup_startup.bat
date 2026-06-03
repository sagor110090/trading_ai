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

REM Check if source exists
if not exist "%SOURCE_FILE%" (
    echo ERROR: %SOURCE_FILE% not found!
    echo Creating it now...

    (
        echo @echo off
        echo REM Auto-generated startup script
        echo timeout /t 30 /nobreak ^>nul
        echo.
        echo REM Launch MT5 - UPDATE THIS PATH
        echo set MT5_PATH=C:\Program Files\FusionMarkets MetaTrader 5\terminal.exe
        echo if exist "%%MT5_PATH%%" start "" "%%MT5_PATH%%"
        echo timeout /t 15 /nobreak ^>nul
        echo.
        echo REM Launch Trading AI
        echo start "" /MIN "C:\Users\mehed\Herd\trading_ai\run.bat"
        echo exit
    ) > "%SOURCE_FILE%"
)

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
    echo To DISABLE auto-start, delete this file:
    echo %DEST_FILE%
    echo.
) else (
    echo.
    echo ERROR: Failed to copy to Startup folder.
    echo Please run as Administrator.
)

pause
