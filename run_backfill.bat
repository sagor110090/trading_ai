@echo off
REM Backfill runner for Windows
REM Usage: run_backfill.bat [symbol] [weeks]

set SYMBOL=%1
set WEEKS=%2

if "%SYMBOL%"=="" set SYMBOL=GBPUSD
if "%WEEKS%"=="" set WEEKS=4

echo Running backfill for %SYMBOL% (%WEEKS% weeks)...
echo.

REM Run the wrapper which sets up the path correctly
C:\Users\mehed\AppData\Local\Programs\Python\Python313\python.exe run_backfill_wrapper.py --symbol %SYMBOL% --weeks %WEEKS% --step 1

pause
