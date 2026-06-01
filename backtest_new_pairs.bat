@echo off
REM ============================================================
REM Backtest New Pairs - 2 Weeks Each
REM Tests: ETHUSD, USDCAD, AUDUSD
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
echo %CYAN%  Backtesting New Trading Pairs (2 weeks each)%RESET%
echo %CYAN%============================================================%RESET%
echo.
echo This will backtest the following pairs sequentially:
echo   1. ETHUSD  (Crypto)
echo   2. USDCAD  (Forex - Canadian Dollar)
echo   3. AUDUSD  (Forex - Australian Dollar)
echo.
echo Weeks per pair: 2
echo Trade Mode: day_trading (from .env)
echo.
echo Starting in 3 seconds...
timeout /t 3 /nobreak >nul

REM ============================================================
REM Pair 1: ETHUSD
REM ============================================================
echo.
echo %CYAN%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%RESET%
echo %CYAN%  Pair 1/3: ETHUSD (Ethereum/US Dollar)%RESET%
echo %CYAN%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%RESET%
echo.
python -m trading_ai.backtest_historical --weeks 2 --symbol ETHUSD

if !errorlevel! neq 0 (
    echo.
    echo %RED%✗ ETHUSD backtest failed!%RESET%
    echo.
    pause
    goto :next_pair
)

echo.
echo %GREEN%✓ ETHUSD backtest complete!%RESET%
echo.
timeout /t 3 /nobreak >nul

:next_pair
REM ============================================================
REM Pair 2: USDCAD
REM ============================================================
echo.
echo %CYAN%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%RESET%
echo %CYAN%  Pair 2/3: USDCAD (US Dollar/Canadian Dollar)%RESET%
echo %CYAN%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%RESET%
echo.
python -m trading_ai.backtest_historical --weeks 2 --symbol USDCAD

if !errorlevel! neq 0 (
    echo.
    echo %RED%✗ USDCAD backtest failed!%RESET%
    echo.
    pause
    goto :next_pair2
)

echo.
echo %GREEN%✓ USDCAD backtest complete!%RESET%
echo.
timeout /t 3 /nobreak >nul

:next_pair2
REM ============================================================
REM Pair 3: AUDUSD
REM ============================================================
echo.
echo %CYAN%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%RESET%
echo %CYAN%  Pair 3/3: AUDUSD (Australian Dollar/US Dollar)%RESET%
echo %CYAN%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%RESET%
echo.
python -m trading_ai.backtest_historical --weeks 2 --symbol AUDUSD

if !errorlevel! neq 0 (
    echo.
    echo %RED%✗ AUDUSD backtest failed!%RESET%
    echo.
    pause
    goto :finished
)

echo.
echo %GREEN%✓ AUDUSD backtest complete!%RESET%
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
echo   • ETHUSD
echo   • USDCAD
echo   • AUDUSD
echo.
echo Check the results above for win rates.
echo Update your .env SYMBOLS to keep only profitable pairs.
echo.
echo %YELLOW%Next steps:%RESET%
echo   1. Review win rate for each pair
echo   2. Keep pairs with ^>40% win rate
echo   3. Remove pairs with ^<30% win rate
echo.
pause
