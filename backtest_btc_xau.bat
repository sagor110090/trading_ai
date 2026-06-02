@echo off
REM ============================================================
REM Backtest BTC & XAU Pairs - 2 Weeks Each
REM Tests: BTCUSD, XAUUSD
REM ============================================================

REM Change to parent directory so Python imports work
cd /d "%~dp0.."

setlocal enabledelayedexpansion

REM Colors (if supported)
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "CYAN=[96m"
set "RESET=[0m"

echo.
echo %CYAN%============================================================%RESET%
echo %CYAN%  Backtesting BTC & XAU Pairs (2 weeks each)%RESET%
echo %CYAN%============================================================%RESET%
echo.
echo This will backtest the following pairs sequentially:
echo   1. BTCUSD  (Bitcoin/US Dollar)
echo   2. XAUUSD  (Gold/US Dollar)
echo.
echo Weeks per pair: 2
echo Trade Mode: day_trading (from .env)
echo.
echo Starting in 3 seconds...
timeout /t 3 /nobreak >nul

REM ============================================================
REM Pair 1: BTCUSD
REM ============================================================
echo.
echo %CYAN%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%RESET%
echo %CYAN%  Pair 1/2: BTCUSD (Bitcoin/US Dollar)%RESET%
echo %CYAN%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%RESET%
echo.
python -m trading_ai.backtest_historical --weeks 2 --symbol BTCUSD

if !errorlevel! neq 0 (
    echo.
    echo %RED%✗ BTCUSD backtest failed!%RESET%
    echo.
    pause
    goto :next_pair
)

echo.
echo %GREEN%✓ BTCUSD backtest complete!%RESET%
echo.
timeout /t 3 /nobreak >nul

:next_pair
REM ============================================================
REM Pair 2: XAUUSD
REM ============================================================
echo.
echo %CYAN%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%RESET%
echo %CYAN%  Pair 2/2: XAUUSD (Gold/US Dollar)%RESET%
echo %CYAN%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%RESET%
echo.
python -m trading_ai.backtest_historical --weeks 2 --symbol XAUUSD

if !errorlevel! neq 0 (
    echo.
    echo %RED%✗ XAUUSD backtest failed!%RESET%
    echo.
    pause
    goto :finished
)

echo.
echo %GREEN%✓ XAUUSD backtest complete!%RESET%
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
echo   • BTCUSD
echo   • XAUUSD
echo.
echo Check the results above for win rates.
echo Update your .env SYMBOLS to keep only profitable pairs.
echo.
echo %YELLOW%Next steps:%RESET%
echo   1. Review win rate for each pair
echo   2. Keep pairs with ^>40% win rate
echo   3. Remove pairs with ^<30% win rate
echo   4. Re-run AI training with updated data
echo.
pause
