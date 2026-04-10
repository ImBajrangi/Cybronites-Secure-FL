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

# Verify static assets exist
echo "  [INFO] Static dashboard assets: $(ls /app/static 2>/dev/null | wc -l) files"

# Verify DB can initialize (bridge startup calls init_db on import)
echo "  [INFO] Verifying database initialization..."
python -c "from Cybronites.server.db import init_db; init_db(); print('  [DB] guardian.db ready.')"

# Start Unified Server (Flower gRPC + FastAPI Bridge)
# They must run in the exact same process to share the in-memory WebSocket Bridge
echo "  [INFO] Launching Unified Server (Bridge on $PORT, Flower on $FLOWER_PORT)..."
exec python -m Cybronites.server.server --flower_port $FLOWER_PORT --rounds $ROUNDS
