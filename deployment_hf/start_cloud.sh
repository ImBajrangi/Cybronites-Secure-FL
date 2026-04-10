#!/bin/bash
set -e

# AI Guardian Cloud Orchestrator
echo "============================================================"
echo "  CYBRONITES | CLOUD INSTANCE STARTING"
echo "============================================================"

# Defaults (can be overridden by HuggingFace Secrets)
export FLOWER_PORT=${FLOWER_PORT:-8080}
export PORT=${PORT:-7860}
export ROUNDS=${ROUNDS:-5}
export PYTHONPATH=/app
export STP_API_PORT=${STP_API_PORT:-8100}

# Verify static assets exist
if [ -d "/app/dist" ]; then
    echo "  [INFO] Static dashboard assets (dist): $(ls /app/dist | wc -l) files"
elif [ -d "/app/static" ]; then
    echo "  [INFO] Static dashboard assets (static): $(ls /app/static | wc -l) files"
else
    echo "  [WARN] No static dashboard assets found!"
fi

# Verify DB can initialize
echo "  [INFO] Verifying database initialization..."
python -c "from Cybronites.server.db import init_db; init_db(); print('  [DB] guardian.db ready.')"

# Start Secure Training Platform in background (if available)
if [ -f "/app/secure_training_platform/main.py" ]; then
    echo "  [INFO] Starting Secure Training Platform on port $STP_API_PORT..."
    python -m secure_training_platform.main &
    STP_PID=$!
    echo "  [STP] PID: $STP_PID"
fi

# Start Unified Server (Flower gRPC + FastAPI Bridge)
echo "  [INFO] Launching Unified Server (Bridge on $PORT, Flower on $FLOWER_PORT)..."
exec python -m Cybronites.server.server --flower_port $FLOWER_PORT --rounds $ROUNDS
