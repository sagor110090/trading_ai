#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
source venv/bin/activate
export PYTHONPATH=/Users/sagor/Code

echo "=========================================="
echo "  Trading AI  |  Running every 5 minutes"
echo "  Press Ctrl+C to stop"
echo "=========================================="

cleanup() {
    echo ""
    echo "Stopping Trading AI..."
    # Kill any background python processes we spawned
    jobs -p | xargs kill 2>/dev/null
    echo "Stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

while true; do
    echo ""
    echo "--- Cycle at $(date '+%Y-%m-%d %H:%M:%S') ---"
    python3 -m trading_ai.main
    echo ""
    echo "Next cycle at $(date -v+5M '+%Y-%m-%d %H:%M:%S')"
    echo "Waiting 5 minutes... (Ctrl+C to stop)"
    sleep 300
done