@echo off
REM Start both backend and frontend servers in parallel (Windows)
REM Usage: start_dev.bat

echo ======================================================================
echo CASH FLOW PREDICTION SYSTEM - DEVELOPMENT SERVER
echo ======================================================================
echo Starting both backend and frontend servers...
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:5000/docs
echo ======================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    pause
    exit /b 1
)

REM Start backend server in a new window
echo [INFO] Starting backend server...
start "Backend Server (Port 5000)" cmd /k "cd backend && python start_server.py"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend server in a new window
echo [INFO] Starting frontend server...
start "Frontend Server (Port 3000)" cmd /k "cd frontend && npm start"

echo.
echo [SUCCESS] Both servers are starting in separate windows
echo.
echo To stop the servers, close the respective command windows
echo or press Ctrl+C in each window
echo.
pause
