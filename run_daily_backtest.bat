@echo off
REM Daily backtest and ML retraining script
REM Runs at 3 AM when market closes

echo ========================================
echo Starting Daily Backtest and ML Retrain
echo Time: %date% %time%
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist "venv\bin\activate" (
    echo Activating virtual environment...
    call venv\bin\activate
)

echo.
echo [1/2] Running auto_backtest.py...
python auto_backtest.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: auto_backtest.py failed with error code %ERRORLEVEL%
    goto :error
)

echo.
echo [2/2] Running retrain_ml.py...
python retrain_ml.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: retrain_ml.py failed with error code %ERRORLEVEL%
    goto :error
)

echo.
echo ========================================
echo Daily backest and ML retrain completed!
echo Time: %date% %time%
echo ========================================
goto :end

:error
echo.
echo ========================================
echo Daily task FAILED! Check logs above.
echo Time: %date% %time%
echo ========================================
exit /b 1

:end
exit /b 0
