#!/bin/sh
set -e

echo "======================================================================"
echo "CASH FLOW PREDICTION SYSTEM"
echo "Runtime: Cloud Run / Docker"
echo "======================================================================"

# Cloud Run provides PORT environment variable (defaults to 8080)
PORT="${PORT:-8080}"

echo "[INFO] Starting FastAPI application"
echo "[INFO] Listening on port: ${PORT}"
echo "[INFO] Environment: ${ENV:-production}"
echo "======================================================================"

# Validate that frontend build exists (non-fatal, but helpful)
if [ ! -d "frontend_dist" ]; then
    echo "[WARN] frontend_dist not found. UI will not be served."
else
    echo "[INFO] frontend_dist found. Serving static frontend."
fi

# Start FastAPI using uvicorn (single process, Cloud Run compliant)
exec uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port "${PORT}"
