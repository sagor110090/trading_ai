"""Quick CLI to view P&L report."""

import sys
import argparse
from datetime import datetime, timedelta, timezone

from trade.pnl_analysis import print_summary_report

def main():
    parser = argparse.ArgumentParser(description="Trading AI P&L Analysis")
    parser.add_argument("--days", type=int, default=30, help="Days to look back (default: 30)")
    parser.add_argument("--all", action="store_true", help="Show all-time statistics")

    args = parser.parse_args()

    if args.all:
        start = None
        end = None
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)

    print_summary_report(start, end)

if __name__ == "__main__":
    main()
