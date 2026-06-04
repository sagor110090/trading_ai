"""Quick command to auto-update data, backtest, and train ML models.

Auto-detects how many days of new data exist and processes all of them.

Usage:
    python auto_update.py              # Auto-update all stale symbols
    python auto_update.py --all        # Force update all symbols
    python auto_update.py XAGUSD       # Update specific symbol
"""

import sys
import argparse
from config.settings import Symbol

# Import the main function
from auto_backtest import run_auto_backtest, SETTINGS, TradeMode

def main():
    parser = argparse.ArgumentParser(
        description="Auto-update: fetch → backtest → train (auto-detects new data)"
    )
    parser.add_argument("symbols", nargs="*", default=None, help="Symbols to update")
    parser.add_argument("--all", action="store_true", help="Force update all symbols")
    parser.add_argument("--force", action="store_true", help="Force update (skip staleness check)")

    args = parser.parse_args()

    # Convert strings to Symbol enum
    symbols = None
    if args.symbols:
        try:
            symbols = [Symbol(s) for s in args.symbols]
        except ValueError as e:
            print(f"Invalid symbol: {e}")
            print(f"Available: {', '.join([s.value for s in SETTINGS.symbols])}")
            sys.exit(1)

    # If --all, use all configured symbols
    if args.all:
        symbols = list(SETTINGS.symbols)
        args.force = True

    # Run pipeline (auto-detects how many days to fetch)
    results = run_auto_backtest(
        symbols=symbols,
        force=args.force,
        mode=SETTINGS.trade_mode,
    )

    # Exit with appropriate code
    if "error" in results:
        sys.exit(1)

if __name__ == "__main__":
    main()
