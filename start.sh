#!/bin/bash
# Start both backend and frontend servers in parallel (Linux/Mac)
# Usage: ./start.sh

echo "======================================================================"
echo "CASH FLOW PREDICTION SYSTEM - DEVELOPMENT SERVER"
echo "======================================================================"
echo "Starting both backend and frontend servers..."
echo "Backend:  http://localhost:5000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:5000/docs"
echo "======================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "[ERROR] Python is not installed or not in PATH"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js is not installed or not in PATH"
    exit 1
fi

# Check if root node_modules exists
if [ ! -d "node_modules" ]; then
    echo "[INFO] Installing root dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install root dependencies"
        exit 1
    fi
fi

# Check if frontend node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "[INFO] Installing frontend dependencies..."
    cd frontend
    npm install
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install frontend dependencies"
        cd ..
        exit 1
    fi
    cd ..
fi

# Start backend server in background
echo "[INFO] Starting backend server on port 5000..."
cd backend
$PYTHON_CMD start_server.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "[ERROR] Backend server failed to start"
    exit 1
fi

# Start frontend server in background
echo "[INFO] Starting frontend server on port 3000..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

# Wait a moment for frontend to start
sleep 2

echo ""
echo "[SUCCESS] Both servers are starting"
echo "Backend PID: $BACKEND_PID (Port 5000)"
echo "Frontend PID: $FRONTEND_PID (Port 3000)"
echo ""
echo "Backend:  http://localhost:5000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:5000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "[INFO] Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID 2>/dev/null
    wait $FRONTEND_PID 2>/dev/null
    echo "[INFO] Servers stopped. Goodbye!"
    exit 0
}

# Trap Ctrl+C and cleanup
trap cleanup INT TERM

# Wait for processes
wait
