#!/bin/bash
# Run the historical backfill to generate signals from old data and train ML.
#
# Usage:
#   ./run_backfill.sh                  # 4 weeks, default mode
#   ./run_backfill.sh --weeks 8        # 8 weeks of data
#   ./run_backfill.sh --step 12        # Check every hour instead of every 5 min
#   ./run_backfill.sh --mode scalping  # Override trade mode

set -e
cd "$(dirname "$0")"

source venv/bin/activate

echo "━━━ Historical Backfill ━━━"
echo "  Fetching old candle data, generating signals, and training ML"
echo ""

python3 -m trading_ai.backtest_historical "$@"