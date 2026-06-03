@echo off
REM ============================================================
REM Backtest BTC & XAU Pairs - 8 Weeks Each, Day Trading Mode
REM Tests: BTCUSD, XAUUSD
REM ============================================================

REM Change to script directory (project root)
cd /d "%~dp0"

REM Set PYTHONPATH so trading_ai package is importable
set PYTHONPATH=C:\Users\mehed\Herd

REM Fix Unicode encoding for Windows terminal
set PYTHONIOENCODING=utf-8

setlocal enabledelayedexpansion

set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "CYAN=[96m"
set "RESET=[0m"

echo.
echo %CYAN%============================================================%RESET%
echo %CYAN%  Backtesting BTC ^& XAU Pairs (8 weeks each)%RESET%
echo %CYAN%  Mode: day_trading ^| Reset: Yes%RESET%
echo %CYAN%============================================================%RESET%
echo.
echo This will backtest the following pairs sequentially:
echo   1. BTCUSD  (Bitcoin/US Dollar)
echo   2. XAUUSD  (Gold/US Dollar)
echo.
echo Weeks per pair: 8
echo Trade Mode: day_trading
echo Reset: Yes (clears old signals and ML models)
echo.
echo Starting in 3 seconds...
timeout /t 3 /nobreak >nul

REM ============================================================
REM Pair 1: BTCUSD
REM ============================================================
echo.
echo %CYAN%================================================================%RESET%
echo %CYAN%  Pair 1/2: BTCUSD (Bitcoin/US Dollar)%RESET%
echo %CYAN%================================================================%RESET%
echo.
python -m trading_ai.backtest_historical --weeks 8 --step 1 --mode day_trading --symbol BTCUSD --reset

if !errorlevel! neq 0 (
    echo.
    echo %RED%X BTCUSD backtest failed!%RESET%
    echo.
    pause
    goto :next_pair
)

echo.
echo %GREEN%V BTCUSD backtest complete!%RESET%
echo.
timeout /t 3 /nobreak >nul

:next_pair
REM ============================================================
REM Pair 2: XAUUSD
REM ============================================================
echo.
echo %CYAN%================================================================%RESET%
echo %CYAN%  Pair 2/2: XAUUSD (Gold/US Dollar)%RESET%
echo %CYAN%================================================================%RESET%
echo.
python -m trading_ai.backtest_historical --weeks 8 --step 1 --mode day_trading --symbol XAUUSD --reset

if !errorlevel! neq 0 (
    echo.
    echo %RED%X XAUUSD backtest failed!%RESET%
    echo.
    pause
    goto :finished
)

echo.
echo %GREEN%V XAUUSD backtest complete!%RESET%
echo.

:finished
REM ============================================================
REM Summary
REM ============================================================
echo.
echo %CYAN%============================================================%RESET%
echo %CYAN%  All Backtests Complete!%RESET%
echo %CYAN%============================================================%RESET%
echo.
echo Backtested pairs:
echo   * BTCUSD  (8 weeks, day_trading)
echo   * XAUUSD  (8 weeks, day_trading)
echo.
echo Check the results above for win rates.
echo Update your .env SYMBOLS to keep only profitable pairs.
echo.
echo %YELLOW%Next steps:%RESET%
echo   1. Review win rate for each pair
echo   2. Keep pairs with ^>40%% win rate
echo   3. Remove pairs with ^<30%% win rate
echo   4. Run: run.bat to start live trading
echo.
pause