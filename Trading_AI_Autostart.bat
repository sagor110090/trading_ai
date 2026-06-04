@echo off
REM ==========================================
REM Trading AI Auto-Start
REM This file runs automatically when Windows starts
REM ==========================================

REM Wait for Windows to fully load
timeout /t 30 /nobreak >nul

REM Launch MT5
set MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
if exist "%MT5_PATH%" (
    echo Starting MT5 Terminal...
    start "" "%MT5_PATH%"
    REM Wait for MT5 to fully load AND login to trading account
    echo Waiting for MT5 to initialize and login...
    timeout /t 60 /nobreak >nul
)

REM Launch Trading AI Bot
echo Starting Trading AI Bot...
start "" /MIN "C:\Users\mehed\Herd\trading_ai\run.bat"

echo Trading AI autostart complete!
exit
