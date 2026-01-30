#!/bin/bash
# Start both backend and frontend servers in parallel (Linux/Mac)
# Usage: ./start_dev.sh

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
    echo "[ERROR] Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js is not installed or not in PATH"
    exit 1
fi

# Start backend server in background
echo "[INFO] Starting backend server..."
cd backend
python3 start_server.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend server in background
echo "[INFO] Starting frontend server..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "[SUCCESS] Both servers are starting"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for user interrupt
trap "echo ''; echo 'Shutting down servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT

# Wait for processes
wait
