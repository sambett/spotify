@echo off
REM Spotify Analytics Pipeline - Windows Commands
REM Quick Docker operations for Windows users

if "%1"=="" goto :help
if "%1"=="help" goto :help
if "%1"=="build" goto :build
if "%1"=="up" goto :up
if "%1"=="down" goto :down
if "%1"=="logs" goto :logs
if "%1"=="shell" goto :shell
if "%1"=="test" goto :test
if "%1"=="preflight" goto :test
if "%1"=="run" goto :run
if "%1"=="clean" goto :clean
if "%1"=="rebuild" goto :rebuild
if "%1"=="quickstart" goto :quickstart
goto :help

:help
echo.
echo ========================================
echo Spotify Analytics Pipeline - Docker Commands
echo ========================================
echo.
echo Available commands:
echo   docker-run.bat build      - Build Docker image
echo   docker-run.bat up         - Start services
echo   docker-run.bat down       - Stop services
echo   docker-run.bat logs       - View logs
echo   docker-run.bat shell      - Open shell in container
echo   docker-run.bat test       - Run preflight tests
echo   docker-run.bat run        - Run ingestion pipeline
echo   docker-run.bat clean      - Clean up containers
echo   docker-run.bat rebuild    - Rebuild from scratch
echo   docker-run.bat quickstart - Build and run
echo.
goto :end

:build
echo.
echo Building Docker image...
docker-compose build
goto :end

:rebuild
echo.
echo Rebuilding Docker image (no cache)...
docker-compose build --no-cache
goto :end

:up
echo.
echo Starting services...
docker-compose up
goto :end

:down
echo.
echo Stopping services...
docker-compose down
goto :end

:logs
echo.
echo Viewing logs (Ctrl+C to exit)...
docker-compose logs -f
goto :end

:shell
echo.
echo Opening shell in container...
docker-compose run --rm spotify-pipeline bash
goto :end

:test
echo.
echo Running preflight checks...
docker-compose run --rm spotify-pipeline python test_setup.py
goto :end

:run
echo.
echo Running ingestion pipeline...
docker-compose run --rm spotify-pipeline python run_ingestion.py
goto :end

:clean
echo.
echo Cleaning up containers...
docker-compose down -v
goto :end

:quickstart
echo.
echo Quick Start: Building and running...
docker-compose build
docker-compose run --rm spotify-pipeline python run_ingestion.py
goto :end

:end
