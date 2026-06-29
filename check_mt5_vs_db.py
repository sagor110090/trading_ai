#!/usr/bin/env python3
"""Diagnostic script to check MT5 trades vs database records.

This script:
1. Queries MT5 directly for the last XAUUSD trade/order
2. Queries the trade_log table for XAUUSD records
3. Compares and identifies discrepancies
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.database import get_session, get_engine, create_tables
from data.models import TradeLog, Signal
from data.mt5_client import MT5Client
from config.settings import SETTINGS

# Ensure tables exist
create_tables()


def get_mt5_history(symbol="XAUUSD", days=7):
    """Get recent trade history from MT5 directly."""
    client = None
    try:
        client = MT5Client()
        if not client.connect():
            print("❌ Failed to connect to MT5")
            return None

        from_date = datetime.now(timezone.utc) - timedelta(days=days)
        to_date = datetime.now(timezone.utc)

        print(f"\n📡 Fetching MT5 history for {symbol} from {from_date} to {to_date}")

        import MetaTrader5 as mt5
        # Get history for the symbol
        deals = mt5.history_deals_get(from_date, to_date, group=f"*{symbol}*")

        if deals is None or len(deals) == 0:
            # Try without symbol filter
            deals = mt5.history_deals_get(from_date, to_date)
            if deals:
                # Filter manually
                deals = [d for d in deals if symbol.upper() in d.symbol.upper()]

        if not deals:
            print(f"❌ No deals found for {symbol} in MT5 history")
            return []

        print(f"✅ Found {len(deals)} deals in MT5 history")
        return deals

    except Exception as e:
        print(f"❌ Error fetching MT5 history: {e}")
        return None
    finally:
        if client:
            client.disconnect()


def get_db_trade_logs(symbol="XAUUSD", days=7):
    """Get trade logs from database."""
    try:
        from_date = datetime.now(timezone.utc) - timedelta(days=days)

        print(f"\n💾 Fetching trade_log for {symbol} since {from_date}")

        with get_session() as session:
            logs = session.query(TradeLog).filter(
                TradeLog.symbol == symbol,
                TradeLog.opened_at >= from_date
            ).order_by(TradeLog.opened_at.desc()).all()

        print(f"✅ Found {len(logs)} records in trade_log table")
        return logs

    except Exception as e:
        print(f"❌ Error fetching trade_log: {e}")
        return []


def get_db_signals(symbol="XAUUSD", days=7):
    """Get signals from database."""
    try:
        from_date = datetime.now(timezone.utc) - timedelta(days=days)

        print(f"\n💾 Fetching signals for {symbol} since {from_date}")

        with get_session() as session:
            signals = session.query(Signal).filter(
                Signal.symbol == symbol,
                Signal.timestamp >= from_date
            ).order_by(Signal.timestamp.desc()).all()

        print(f"✅ Found {len(signals)} signals in database")
        return signals

    except Exception as e:
        print(f"❌ Error fetching signals: {e}")
        return []


def format_mt5_deal(deal):
    """Format MT5 deal for display."""
    import MetaTrader5 as mt5

    direction = "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL" if deal.type == mt5.DEAL_TYPE_SELL else "BALANCE"
    return {
        "ticket": deal.ticket,
        "symbol": deal.symbol,
        "time": datetime.fromtimestamp(deal.time, tz=timezone.utc).isoformat(),
        "type": direction,
        "volume": deal.volume,
        "price": deal.price,
        "commission": deal.commission,
        "swap": deal.swap,
        "profit": deal.profit,
        "entry": "IN" if deal.entry == mt5.DEAL_ENTRY_IN else "OUT" if deal.entry == mt5.DEAL_ENTRY_OUT else "INOUT",
        "position_id": deal.position_id,
    }


def format_trade_log(log):
    """Format trade log for display."""
    return {
        "id": log.id,
        "deal_id": log.deal_id,
        "symbol": log.symbol,
        "direction": log.direction,
        "size": log.size,
        "entry_price": log.entry_price,
        "exit_price": log.exit_price,
        "stop_loss": log.stop_loss,
        "take_profit": log.take_profit,
        "opened_at": log.opened_at.isoformat() if log.opened_at else None,
        "closed_at": log.closed_at.isoformat() if log.closed_at else None,
        "final_pnl": log.final_pnl,
        "close_reason": log.close_reason,
        "signal_id": log.signal_id,
    }


def main():
    print("=" * 80)
    print("🔍 MT5 vs Database Trade Diagnostic")
    print("=" * 80)

    symbol = "XAUUSD"
    days = 30  # Check last 30 days

    # Get MT5 history
    mt5_deals = get_mt5_history(symbol, days)

    if mt5_deals:
        print("\n" + "=" * 80)
        print(f"📊 MT5 DEALS FOR {symbol}")
        print("=" * 80)

        # Convert to list of dicts for easier handling
        mt5_deal_list = [format_mt5_deal(d) for d in mt5_deals]

        for i, deal in enumerate(mt5_deal_list):
            print(f"\n--- MT5 Deal #{i+1} ---")
            print(f"  Ticket:        {deal['ticket']}")
            print(f"  Symbol:        {deal['symbol']}")
            print(f"  Time:          {deal['time']}")
            print(f"  Type:          {deal['type']} ({deal['entry']})")
            print(f"  Volume:        {deal['volume']}")
            print(f"  Price:         {deal['price']}")
            print(f"  Position ID:   {deal['position_id']}")
            print(f"  Commission:    {deal['commission']}")
            print(f"  Swap:          {deal['swap']}")
            print(f"  Profit:        {deal['profit']}")

        # Get last open deal (entry)
        entry_deals = [d for d in mt5_deal_list if d['entry'] == 'IN']
        if entry_deals:
            last_entry = max(entry_deals, key=lambda x: x['time'])
            print("\n" + "=" * 80)
            print("🎯 LAST ENTRY DEAL FROM MT5")
            print("=" * 80)
            print(f"  Ticket:        {last_entry['ticket']}")
            print(f"  Time:          {last_entry['time']}")
            print(f"  Direction:     {last_entry['type']}")
            print(f"  Volume:        {last_entry['volume']}")
            print(f"  Price:         {last_entry['price']}")
            print(f"  Position ID:   {last_entry['position_id']}")

    # Get database records
    trade_logs = get_db_trade_logs(symbol, days)

    if trade_logs:
        print("\n" + "=" * 80)
        print(f"📊 DATABASE TRADE_LOG FOR {symbol}")
        print("=" * 80)

        for i, log in enumerate(trade_logs):
            log_dict = format_trade_log(log)
            print(f"\n--- Trade Log #{i+1} (ID: {log_dict['id']}) ---")
            print(f"  Deal ID:       {log_dict['deal_id']}")
            print(f"  Symbol:        {log_dict['symbol']}")
            print(f"  Direction:     {log_dict['direction']}")
            print(f"  Size:          {log_dict['size']}")
            print(f"  Entry Price:   {log_dict['entry_price']}")
            print(f"  SL:            {log_dict['stop_loss']}")
            print(f"  TP:            {log_dict['take_profit']}")
            print(f"  Opened At:     {log_dict['opened_at']}")
            print(f"  Closed At:     {log_dict['closed_at']}")
            print(f"  Final PnL:     {log_dict['final_pnl']}")
            print(f"  Close Reason:  {log_dict['close_reason']}")
            print(f"  Signal ID:     {log_dict['signal_id']}")

    # Get signals
    signals = get_db_signals(symbol, days)

    # Compare and identify discrepancies
    print("\n" + "=" * 80)
    print("🔍 DISCREPANCY ANALYSIS")
    print("=" * 80)

    if mt5_deals:
        # Get unique position IDs from MT5
        mt5_position_ids = set(d['position_id'] for d in [format_mt5_deal(d) for d in mt5_deals] if d['entry'] == 'IN')

        # Get deal IDs from database
        db_deal_ids = set(str(log.deal_id) for log in trade_logs if log.deal_id)

        print(f"\n📊 MT5 Entry Deals (unique positions): {len(mt5_position_ids)}")
        print(f"📊 Database Trade Log records: {len(trade_logs)}")
        print(f"📊 Database signals (executed): {len([s for s in signals if s.executed])}")

        # Find MT5 deals not in database
        mt5_only = mt5_position_ids - db_deal_ids
        if mt5_only:
            print(f"\n⚠️  MT5 DEALS NOT IN DATABASE ({len(mt5_only)}):")
            for pos_id in mt5_only:
                deal = next((d for d in [format_mt5_deal(d) for d in mt5_deals] if d['position_id'] == pos_id), None)
                if deal:
                    print(f"  - Position ID: {deal['position_id']}")
                    print(f"    Ticket: {deal['ticket']}")
                    print(f"    Time: {deal['time']}")
                    print(f"    Type: {deal['type']} {deal['symbol']}")
                    print(f"    Volume: {deal['volume']} @ {deal['price']}")

        # Find database records not in MT5
        db_only = db_deal_ids - mt5_position_ids
        if db_only:
            print(f"\n⚠️  DATABASE RECORDS NOT IN MT5 ({len(db_only)}):")
            for deal_id in db_only:
                log = next((l for l in trade_logs if str(l.deal_id) == deal_id), None)
                if log:
                    print(f"  - Deal ID: {log.deal_id}")
                    print(f"    Log ID: {log.id}")
                    print(f"    Opened: {log.opened_at}")
                    print(f"    {log.direction} {log.symbol} size={log.size}")

        if not mt5_only and not db_only:
            print("\n✅ All MT5 deals match database records!")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
