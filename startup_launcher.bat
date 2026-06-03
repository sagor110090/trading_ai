@echo off
REM ==========================================
REM Trading AI Auto-Start Launcher
REM Place this file (or a shortcut to it) in:
REM C:\Users\mehed\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup
REM ==========================================

REM Wait for Windows to fully load (prevents issues at startup)
timeout /t 30 /nobreak >nul

REM Change to trading_ai directory
cd /d "%~dp0"

REM ==========================================
REM STEP 1: Launch MT5 Terminal
REM ==========================================
REM Update this path to your MT5 terminal.exe location
REM Common locations:
REM   C:\Program Files\MetaTrader 5\terminal.exe
REM   C:\Program Files (x86)\MetaTrader 5\terminal.exe
REM   C:\Users\<username>\AppData\Local\MetaTrader 5\terminal.exe

set MT5_PATH=C:\Program Files\Fusion Markets MetaTrader 5\terminal64.exe

if exist "%MT5_PATH%" (
    echo Starting MT5 Terminal...
    start "" "%MT5_PATH%"

    REM Wait for MT5 to fully load (adjust if needed)
    echo Waiting for MT5 to initialize...
    timeout /t 15 /nobreak >nul
) else (
    echo MT5 not found at: %MT5_PATH%
    echo Please update MT5_PATH in this file.
    timeout /t 5 /nobreak >nul
)

REM ==========================================
REM STEP 2: Launch Trading AI Bot
REM ==========================================
echo Starting Trading AI Bot...
start "" /MIN "%~dp0run.bat"

echo Trading AI startup complete!
exit
