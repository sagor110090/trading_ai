@echo off
REM ==========================================
REM Trading AI Auto-Start
REM This file runs automatically when Windows starts
REM ==========================================

REM Wait for Windows to fully load
timeout /t 30 /nobreak >nul

REM Launch MT5 (update path if needed)
set MT5_PATH=C:\Program Files\Fusion Markets MetaTrader 5\terminal64.exe
if exist "%MT5_PATH%" (
    start "" "%MT5_PATH%"
    timeout /t 15 /nobreak >nul
)

REM Launch Trading AI Bot
start "" /MIN "C:\Users\mehed\Herd\trading_ai\run.bat"

exit
