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

# Use --loop so PositionManager (cooldowns, position tracking) persists
# between cycles. Without --loop, cooldowns are lost each cycle.
python3 -m trading_ai.main --loop --interval "$INTERVAL"