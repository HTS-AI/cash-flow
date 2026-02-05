#!/bin/sh
set -e

echo "======================================================"
echo "Starting Cash Flow Prediction API (Cloud Run)"
echo "Port: ${PORT:-8080}"
echo "======================================================"

exec uvicorn backend.main:app \
  --host 0.0.0.0 \
  --port ${PORT:-8080}
