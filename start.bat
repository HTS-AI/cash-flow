@echo off
REM Start both backend and frontend servers in parallel
REM Usage: start.bat

echo ======================================================================
echo CASH FLOW PREDICTION SYSTEM - DEVELOPMENT SERVER
echo ======================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.7+ and add it to your PATH
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo Please install Node.js and add it to your PATH
    pause
    exit /b 1
)

REM Check if root node_modules exists
if not exist "node_modules" (
    echo [INFO] Installing root dependencies...
    call npm install
    if errorlevel 1 (
        echo [ERROR] Failed to install root dependencies
        pause
        exit /b 1
    )
)

REM Check if frontend node_modules exists
if not exist "frontend\node_modules" (
    echo [INFO] Installing frontend dependencies...
    cd frontend
    call npm install
    if errorlevel 1 (
        echo [ERROR] Failed to install frontend dependencies
        cd ..
        pause
        exit /b 1
    )
    cd ..
)

echo.
echo ======================================================================
echo Starting both servers...
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:5000/docs
echo ======================================================================
echo.
echo Press Ctrl+C to stop both servers
echo.

REM Start both servers using concurrently
call npm run dev
