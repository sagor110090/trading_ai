@echo off
cd /d "%~dp0"

REM Fix Unicode encoding for Windows terminal
set PYTHONIOENCODING=utf-8

REM Get Python path
set PYTHON_PATH=C:\Users\mehed\AppData\Local\Programs\Python\Python313\python.exe

REM Set PYTHONPATH to parent directory (so trading_ai package works)
set PYTHONPATH=C:\Users\mehed\Herd

REM Read interval from .env (default 300 = 5 min)
for /f "tokens=2 delims==" %%A in ('findstr /R "^RUN_INTERVAL=" .env 2^>nul') do set INTERVAL=%%A
if "%INTERVAL%"=="" set INTERVAL=300

REM Remove quotes and spaces
set INTERVAL=%INTERVAL:"=%
set INTERVAL=%INTERVAL: =%

REM Read trade mode
for /f "tokens=2 delims==" %%A in ('findstr /R "^TRADE_MODE=" .env 2^>nul') do set TRADE_MODE=%%A
if "%TRADE_MODE%"=="" set TRADE_MODE=day_trading

set TRADE_MODE=%TRADE_MODE:"=%
set TRADE_MODE=%TRADE_MODE: =%

echo ==========================================
echo   Trading AI  ^|  Mode: %TRADE_MODE%
echo   Database: SQLite (trading_ai.db)
echo   Main cycle:    every %INTERVAL% seconds
echo   Position monitor: every 60 seconds (background)
echo   Press Ctrl+C to stop
echo ==========================================

REM Use --loop so a single Python process stays alive.
REM This lets the background position-monitor thread tick every 60s
REM while the main signal cycle runs every %INTERVAL% seconds.
"%PYTHON_PATH%" -m trading_ai.main --loop --interval %INTERVAL%
