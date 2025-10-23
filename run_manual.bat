@echo off
REM Manual Run Script for Spotify Data Ingestion
REM Run this to test the pipeline before scheduling

echo ============================================================
echo SPOTIFY DATA INGESTION - Manual Run
echo ============================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run the ingestion script
echo Starting data ingestion...
echo.
python run_ingestion.py

REM Check exit code
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS! Data ingestion completed successfully.
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo ERROR! Data ingestion failed with code %ERRORLEVEL%
    echo ============================================================
)

echo.
pause
