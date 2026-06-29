"""Simple script to query the database for XAUUSD trades."""
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent))

from data.database import get_session, create_tables
from data.models import TradeLog, Signal

create_tables()

print("=" * 80)
print("📊 DATABASE TRADE LOG - XAUUSD (last 30 days)")
print("=" * 80)

with get_session() as session:
    from_date = datetime.now(timezone.utc) - timedelta(days=30)

    # Query trade_log
    logs = session.query(TradeLog).filter(
        TradeLog.symbol == "XAUUSD",
        TradeLog.opened_at >= from_date
    ).order_by(TradeLog.opened_at.desc()).all()

    if not logs:
        print("\n❌ NO TRADE LOG RECORDS FOUND FOR XAUUSD")
    else:
        print(f"\n✅ Found {len(logs)} trade_log records:\n")

        for log in logs:
            print(f"--- Log ID: {log.id} ---")
            print(f"  Deal ID:       {log.deal_id}")
            print(f"  Symbol:        {log.symbol}")
            print(f"  Direction:     {log.direction}")
            print(f"  Size:          {log.size}")
            print(f"  Entry Price:   {log.entry_price}")
            print(f"  Stop Loss:     {log.stop_loss}")
            print(f"  Take Profit:   {log.take_profit}")
            print(f"  Opened At:     {log.opened_at}")
            print(f"  Closed At:     {log.closed_at}")
            print(f"  Final PnL:     {log.final_pnl}")
            print(f"  Close Reason:  {log.close_reason}")
            print(f"  Signal ID:     {log.signal_id}")
            print()

    # Query signals
    print("\n" + "=" * 80)
    print("📊 DATABASE SIGNALS - XAUUSD (last 30 days)")
    print("=" * 80)

    signals = session.query(Signal).filter(
        Signal.symbol == "XAUUSD",
        Signal.timestamp >= from_date
    ).order_by(Signal.timestamp.desc()).limit(20).all()

    if not signals:
        print("\n❌ NO SIGNAL RECORDS FOUND FOR XAUUSD")
    else:
        print(f"\n✅ Found {len(signals)} signals:\n")

        for sig in signals:
            print(f"--- Signal ID: {sig.signal_id} ---")
            print(f"  Symbol:        {sig.symbol}")
            print(f"  Direction:     {sig.signal}")
            print(f"  Confidence:    {sig.confidence}")
            print(f"  Entry:         {sig.entry}")
            print(f"  SL:            {sig.stop_loss}")
            print(f"  TP:            {sig.take_profit}")
            print(f"  Timestamp:     {sig.timestamp}")
            print(f"  Executed:      {sig.executed}")
            print(f"  ML Prob:       {sig.ml_probability}")
            print(f"  Blocked:       {sig.blocked_reason}")
            print()

print("=" * 80)
