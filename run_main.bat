@echo off
REM Main trading loop runner for Windows

echo Starting Trading AI...
echo.

REM Run the wrapper which sets up the path correctly
C:\Users\mehed\AppData\Local\Programs\Python\Python313\python.exe run_main_wrapper.py --loop --interval 300

pause
