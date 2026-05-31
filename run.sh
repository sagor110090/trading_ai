#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
source venv/bin/activate
export PYTHONPATH=/Users/sagor/Code

# Read interval from .env (default 300 = 5 min)
INTERVAL=$(grep -E '^RUN_INTERVAL=' .env 2>/dev/null | cut -d= -f2 | tr -d ' ')
INTERVAL=${INTERVAL:-300}

# Convert to human-readable
if [ "$INTERVAL" -le 60 ]; then
    INTERVAL_TEXT="${INTERVAL} seconds"
elif [ "$INTERVAL" -le 3600 ]; then
    MINUTES=$((INTERVAL / 60))
    INTERVAL_TEXT="${MINUTES} minutes"
else
    HOURS=$((INTERVAL / 3600))
    INTERVAL_TEXT="${HOURS} hours"
fi

# Read trade mode for display
TRADE_MODE=$(grep -E '^TRADE_MODE=' .env 2>/dev/null | cut -d= -f2 | tr -d ' ')
TRADE_MODE=${TRADE_MODE:-day_trading}

echo "=========================================="
echo "  Trading AI  |  Mode: $TRADE_MODE"
echo "  Running every $INTERVAL_TEXT"
echo "  Press Ctrl+C to stop"
echo "=========================================="

cleanup() {
    echo ""
    echo "Stopping Trading AI..."
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
    echo "Next cycle in $INTERVAL_TEXT at $(date -v+${INTERVAL}S '+%Y-%m-%d %H:%M:%S')"
    echo "Waiting... (Ctrl+C to stop)"
    sleep "$INTERVAL"
done