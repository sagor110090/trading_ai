@echo off
cd /d "C:\Users\mehed\Herd\trading_ai"

set PYTHONIOENCODING=utf-8
set PYTHON_PATH=C:\Users\mehed\AppData\Local\Programs\Python\Python313\python.exe
set PYTHONPATH=C:\Users\mehed\Herd

REM Wait 30s for MT5 terminal to fully load first
timeout /t 30 /nobreak >nul

:loop
"%PYTHON_PATH%" -m trading_ai.main
echo.
echo Restarting in 60 seconds...
timeout /t 60 /nobreak >nul
goto loop