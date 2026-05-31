"""Main orchestrator: wires all system components for the prediction loop.

Key behaviors:
- Signals are ALWAYS generated (for ML training), but trade execution is controlled
- If a new signal opposes an open position, the old position is CLOSED and reversed
- If a new signal agrees with an open position, the trade is skipped (already in it)
- After a loss, a cooldown prevents immediate re-entry
- Duplicate signals (same symbol+direction within a window) are not stored
"""

import logging
import sys
from typing import Optional

import pandas as pd
from rich.logging import RichHandler

from trading_ai.config.settings import SETTINGS, Symbol, Timeframe, TradingPlatform
from trading_ai.data.collector import CapitalComClient
from trading_ai.data.database import create_tables, get_session
from trading_ai.data.models import Candle, Signal, SignalVerification
from trading_ai.display import (
    console,
    display_auto_train_result,
    display_auto_train_skipped,
    display_auto_train_starting,
    display_auto_train_waiting,
    display_candle_fetch,
    display_cycle_header,
    display_cycle_summary,
    display_duplicate_signal,
    display_ml_prediction,
    display_no_ml_model,
    display_no_signal,
    display_position_reversed,
    display_signal,
    display_signal_skipped_cooldown,
    display_timeout_closed,
    display_trade_blocked,
    display_trade_failed,
    display_trade_placed,
    display_trade_skipped_cooldown,
    display_trade_skipped_existing,
    display_verification_summary,
)
from trading_ai.indicators.features import compute_all_features
from trading_ai.ml.predictor import Prediction, predict_signal
from trading_ai.signals.engine import TradeSignal, generate_signal
from trading_ai.trade.executor import (
    close_position_unified,
    get_trailing_stop_candidate,
    place_trade_unified,
    update_position_sl_tp_unified,
)
from trading_ai.trade.position_manager import PositionManager
from trading_ai.verification.verifier import verify_pending_signals

logger = logging.getLogger(__name__)


def _create_client():
    """Create the appropriate trading client based on platform selection."""
    if SETTINGS.trading_platform == TradingPlatform.MT5:
        from trading_ai.data.mt5_client import MT5Client
        return MT5Client()
    else:
        return CapitalComClient()


def _authenticate_client(client) -> bool:
    """Authenticate the trading client."""
    if SETTINGS.trading_platform == TradingPlatform.MT5:
        success = client.connect()
        if success:
            console.print("[green]✓ Connected to MetaTrader 5[/]")
            # Display account info
            account = client.get_account_info()
            if account:
                console.print(f"[dim]  Balance: ${account['balance']:.2f} | Equity: ${account['equity']:.2f}[/]")
        else:
            console.print("[bold red]✗ Cannot connect to MetaTrader 5. Make sure MT5 terminal is running.[/]")
        return success
    else:
        success = client.authenticate()
        if success:
            console.print("[green]✓ Authenticated with Capital.com[/]")
        else:
            console.print("[bold red]✗ Cannot authenticate with Capital.com. Check credentials.[/]")
        return success


def _setup_logging() -> None:
    """Configure logging with Rich handler for beautiful output."""
    logging.basicConfig(
        level="WARNING",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger(__name__).setLevel(logging.DEBUG)


def _store_candles(df: pd.DataFrame, symbol: str, timeframe: str) -> int:
    """Store fetched candles into the database. Returns count of new candles inserted."""
    session_gen = get_session()
    session = next(session_gen)
    count = 0

    for _, row in df.iterrows():
        ts = row["timestamp"]
        if isinstance(ts, str):
            ts = pd.Timestamp(ts, tz="UTC")

        exists = (
            session.query(Candle)
            .filter(
                Candle.symbol == symbol,
                Candle.timeframe == timeframe,
                Candle.timestamp == ts,
            )
            .first()
        )
        if exists:
            continue

        candle = Candle(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=ts,
            open_price=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0) or 0),
            spread=float(row.get("spread", 0) or 0),
        )
        session.add(candle)
        count += 1

    try:
        session.commit()
    except Exception:
        session.rollback()
        raise

    return count


def _store_signal(trade_signal: TradeSignal) -> Signal:
    """Persist a trade signal to the database."""
    session_gen = get_session()
    session = next(session_gen)

    features = trade_signal.features.to_dict()

    signal = Signal(
        signal_id=trade_signal.signal_id,
        symbol=trade_signal.symbol,
        timeframe=trade_signal.timeframe,
        signal=trade_signal.signal,
        confidence=trade_signal.confidence,
        entry=trade_signal.entry,
        stop_loss=trade_signal.stop_loss,
        take_profit=trade_signal.take_profit,
        timestamp=trade_signal.timestamp,
        ema20=features.get("ema20"),
        ema50=features.get("ema50"),
        ema200=features.get("ema200"),
        rsi14=features.get("rsi14"),
        macd_line=features.get("macd_line"),
        macd_signal=features.get("macd_signal"),
        macd_histogram=features.get("macd_histogram"),
        atr14=features.get("atr14"),
        candle_body_pct=features.get("candle_body_pct"),
        upper_wick_pct=features.get("upper_wick_pct"),
        lower_wick_pct=features.get("lower_wick_pct"),
        dist_ema20=features.get("dist_ema20"),
        dist_ema50=features.get("dist_ema50"),
        dist_ema200=features.get("dist_ema200"),
        trend_strength=features.get("trend_strength"),
        momentum_score=features.get("momentum_score"),
        session=features.get("session"),
        spread=features.get("spread"),
    )

    session.add(signal)

    try:
        session.commit()
    except Exception:
        session.rollback()
        raise

    return signal


def _load_candles_from_db(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Load recent candles from the database into a DataFrame."""
    session_gen = get_session()
    session = next(session_gen)

    candles = (
        session.query(Candle)
        .filter(Candle.symbol == symbol, Candle.timeframe == timeframe)
        .order_by(Candle.timestamp.desc())
        .limit(limit)
        .all()
    )

    if not candles:
        return pd.DataFrame()

    rows = [{
        "timestamp": c.timestamp,
        "open": c.open_price,
        "high": c.high,
        "low": c.low,
        "close": c.close,
        "volume": c.volume,
        "spread": c.spread,
    } for c in reversed(candles)]

    return pd.DataFrame(rows)


def run_signal_cycle(client: CapitalComClient, position_manager: PositionManager) -> None:
    """One complete cycle: fetch data for all timeframes, run confluence, verify, predict.

    The position_manager controls trade execution:
    - Same direction as open position → skip (already in this trade)
    - Opposite direction → close old position and take new one (reversal)
    - No position + no cooldown → place trade
    - Cooldown active → skip trade
    """
    # Clean up expired cooldowns
    position_manager.cleanup_expired_cooldowns()

    for symbol in SETTINGS.symbols:
        display_cycle_header(symbol.value)

        stats = {
            "signal_generated": False,
            "ml_prediction": None,
            "ml_blocked": False,
            "ml_approved": False,
            "no_ml": False,
            "trade_placed": False,
            "trade_blocked": False,
            "trade_skipped": False,
            "trade_cooldown": False,
            "trade_reversed": False,
            "trade_failed": False,
            "sl_updated": False,
            "verified": 0,
            "auto_train_started": False,
            "timeout_closed": False,
        }

        fetched: dict[str, pd.DataFrame] = {}

        # ── 0. Check position timeouts ──────────────────────────────────
        if SETTINGS.auto_trade:
            _check_timeouts(client, stats, position_manager, symbol.value)

        # ── 1. Fetch and store candles ──────────────────────────────────
        candle_results: dict[str, int] = {}
        for timeframe in SETTINGS.timeframes:
            try:
                df = client.fetch_candles(symbol, timeframe, limit=500)
            except Exception as e:
                logger.error("Failed to fetch %s %s: %s", symbol.value, timeframe.value, e)
                continue

            if df.empty:
                continue

            new_count = _store_candles(df, symbol.value, timeframe.value)
            candle_results[timeframe.value] = new_count
            fetched[timeframe.value] = _load_candles_from_db(symbol.value, timeframe.value, limit=500)

        display_candle_fetch(candle_results)

        # ── 2. Verify pending signals ────────────────────────────────────
        verified = verify_pending_signals(symbol.value)
        pending_count = (
            get_session()
            .__next__()
            .query(Signal)
            .outerjoin(Signal.verification)
            .filter(Signal.symbol == symbol.value, Signal.verification == None)
            .count()
        )
        display_verification_summary(verified, pending_count)
        stats["verified"] = verified

        # Register verified losses as cooldowns
        if verified > 0:
            _register_verified_results(symbol.value, position_manager)

        # ── 3. Check if we should skip signal generation ─────────────────
        should_generate, skip_reason = position_manager.should_generate_signal(symbol.value)
        if not should_generate:
            display_signal_skipped_cooldown(symbol.value, 0)  # reason already logged
            # Still verify and auto-train, but don't generate new signals
            _try_auto_train(symbol.value, stats)
            display_cycle_summary(stats)
            continue

        # ── 4. Multi-timeframe confluence signal ────────────────────────
        df_h4 = fetched.get(Timeframe.H4.value, pd.DataFrame())
        df_h1 = fetched.get(Timeframe.H1.value, pd.DataFrame())
        df_m5 = fetched.get(Timeframe.M5.value, pd.DataFrame())

        # Fetch live price so the signal entry reflects current market
        live_price = None
        price_data = None
        try:
            price_data = client.get_latest_price(symbol)
            if price_data:
                if SETTINGS.trading_platform == TradingPlatform.MT5:
                    live_price = price_data["ask"] if price_data.get("ask") else price_data.get("bid")
                else:
                    live_price = price_data.get("bid") or price_data.get("ask")
                logger.info("Live %s price: %.2f", symbol.value, live_price)
        except Exception as e:
            logger.warning("Could not fetch live price for %s: %s", symbol.value, e)

        if SETTINGS.auto_trade and SETTINGS.trailing_stop_enabled:
            _apply_profit_trailing(client, symbol, price_data, position_manager, stats)

        if not df_h4.empty and not df_h1.empty:
            trade_signal = generate_signal(
                df_h4, df_h1, df_m5, symbol, SETTINGS.trade_mode, live_price=live_price,
            )

            if trade_signal:
                # ── Signal dedup: skip storing if duplicate ──────────
                if position_manager.is_duplicate_signal(
                    trade_signal.symbol, trade_signal.signal.value, SETTINGS.trade_mode.value
                ):
                    display_duplicate_signal(trade_signal.symbol, trade_signal.signal.value)
                    # Don't store, don't trade, but still show the signal was detected
                    stats["signal_generated"] = True
                    stats["signal"] = trade_signal
                    display_signal(trade_signal)

                    # Even for duplicates, check if we should reverse
                    if SETTINGS.auto_trade:
                        _handle_trade_decision(client, trade_signal, position_manager, stats)

                    _try_auto_train(symbol.value, stats)
                    display_cycle_summary(stats)
                    continue

                _store_signal(trade_signal)
                display_signal(trade_signal)
                stats["signal_generated"] = True
                stats["signal"] = trade_signal

                # ── ML prediction ──────────────────────────────────────
                should_trade = True
                prediction = predict_signal(trade_signal)
                if prediction:
                    stats["ml_prediction"] = prediction
                    if prediction.probability < SETTINGS.ml.prediction_threshold:
                        should_trade = False
                        stats["ml_blocked"] = True
                        display_ml_prediction(prediction, SETTINGS.ml.prediction_threshold, blocked=True)
                    else:
                        stats["ml_approved"] = True
                        display_ml_prediction(prediction, SETTINGS.ml.prediction_threshold, blocked=False)
                else:
                    stats["no_ml"] = True
                    display_no_ml_model()

                # ── Auto-trade ──────────────────────────────────────────
                if SETTINGS.auto_trade and should_trade:
                    _handle_trade_decision(client, trade_signal, position_manager, stats)
                elif SETTINGS.auto_trade and not should_trade:
                    display_trade_blocked("ML filter")
                    stats["trade_blocked"] = True

            else:
                display_no_signal(symbol.value)
        else:
            logger.warning("Missing H4 or H1 data for confluence check on %s", symbol.value)

        # ── 5. Auto-train ───────────────────────────────────────────────
        _try_auto_train(symbol.value, stats)

        # ── Cycle summary ───────────────────────────────────────────────
        display_cycle_summary(stats)


def _handle_trade_decision(
    client: CapitalComClient,
    trade_signal: TradeSignal,
    position_manager: PositionManager,
    stats: dict,
) -> None:
    """Evaluate and execute a trade based on position state.

    Handles four scenarios:
    1. No position, no cooldown → place new trade
    2. Same direction position → UPDATE SL/TP (trail the stop)
    3. Opposite direction position → close old, place new (reversal)
    4. Cooldown active → skip
    """
    from trading_ai.trade.executor import OrderResult

    decision = position_manager.evaluate_trade(
        trade_signal.symbol,
        trade_signal.signal.value,
        trade_signal.stop_loss,
        trade_signal.take_profit,
    )

    if decision.action == "update_sl_tp":
        # Same direction — update SL/TP on existing position
        result = position_manager.execute_trade_decision(trade_signal, decision)

        if result.get("sl_updated"):
            console.print(
                f"  [cyan]📐 SL/TP Updated:[/] "
                f"SL→{trade_signal.stop_loss:,.2f}  TP→{trade_signal.take_profit:,.2f}  "
                f"[dim]({decision.reason})[/]"
            )
            stats["sl_updated"] = True
        else:
            # Update was rejected (unsafe) or failed — position still open, just not updated
            console.print(
                f"  [yellow]⏭  SL update skipped:[/] {result.get('reason', 'unknown')}"
            )
            stats["trade_skipped"] = True
        return

    if decision.action == "skip_cooldown":
        # Cooldown active after a loss/timeout
        display_trade_skipped_cooldown(trade_signal.symbol, decision.cooldown_remaining_minutes)
        stats["trade_cooldown"] = True
        return

    if decision.action == "reverse_and_place":
        # Close existing position, then place new one
        pos = decision.existing_position
        console.print(
            f"  [yellow]🔄 Reversing:[/] closing {pos.direction} {pos.epic} "
            f"to take {trade_signal.signal.value}"
        )
        close_result = close_position_unified(client, pos)

        if not close_result.success:
            console.print(
                f"  [red]✗ Failed to close position:[/] {close_result.error}"
            )
            stats["trade_failed"] = True
            return

        # Register the closed position
        position_manager.register_closed(trade_signal.symbol, "reversal")
        display_position_reversed(
            f"{pos.direction} {pos.epic} (deal {pos.deal_id})",
            trade_signal.signal.value,
            trade_signal.symbol,
        )
        stats["trade_reversed"] = True

    if decision.action in ("place", "reverse_and_place"):
        # Place the trade
        size = SETTINGS.get_trade_size(trade_signal.symbol)
        result = place_trade_unified(client, trade_signal, size=size)

        if result.success:
            display_trade_placed(result)
            stats["trade_placed"] = True
        else:
            if "already exists" in (result.error or "").lower():
                display_trade_skipped_existing(trade_signal.symbol)
                stats["trade_skipped"] = True
            else:
                display_trade_failed(result)
                stats["trade_failed"] = True


def _apply_profit_trailing(
    client,
    symbol: Symbol,
    price_data: Optional[dict],
    position_manager: PositionManager,
    stats: dict,
) -> None:
    """Move SL forward as profit milestones are reached."""
    if not price_data:
        return

    position_manager.sync_positions()
    status = position_manager.get_status(symbol.value)

    for pos in status["open_positions"]:
        new_sl, milestone = get_trailing_stop_candidate(pos, price_data)
        if new_sl is None or milestone is None:
            continue

        old_sl = pos.stop_level
        current_price = (
            price_data.get("bid") or price_data.get("ask")
            if pos.direction.upper() == "BUY"
            else price_data.get("ask") or price_data.get("bid")
        )
        old_sl_text = f"{old_sl:,.2f}" if old_sl is not None else "none"
        current_price_text = f"{current_price:,.2f}" if current_price is not None else "n/a"
        update_result = update_position_sl_tp_unified(client, pos, new_sl=new_sl)
        if not update_result.success:
            logger.warning(
                "Profit trail update failed for %s %s: %s",
                pos.direction,
                pos.epic,
                update_result.error,
            )
            continue

        console.print(
            f"  [cyan]📐 Profit trail:[/] "
            f"{pos.direction} {symbol.value} hit {milestone} TP "
            f"| [dim]price[/] {current_price_text} "
            f"| [dim]SL[/] {old_sl_text} → {new_sl:,.2f}"
        )
        stats["sl_updated"] = True
        position_manager.sync_positions()


def _register_verified_results(symbol: str, position_manager: PositionManager) -> None:
    """Check verified results from this cycle and register cooldowns for losses."""
    session = next(get_session())

    # Find recently verified losses for this symbol
    from datetime import datetime, timedelta, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)

    recent_losses = (
        session.query(SignalVerification)
        .join(Signal, SignalVerification.signal_id == Signal.id)
        .filter(
            Signal.symbol == symbol,
            SignalVerification.result == "LOSS",
            SignalVerification.verified_at >= cutoff,
        )
        .all()
    )

    for loss in recent_losses:
        position_manager.register_verified_result(symbol, "LOSS")


def _check_timeouts(
    client,
    stats: dict,
    position_manager: PositionManager,
    symbol: str,
) -> None:
    """Check open positions and close any that exceeded their max hold time."""
    from trading_ai.trade.executor import check_timeouts_unified

    max_minutes = SETTINGS.get_max_hold_minutes()
    results = check_timeouts_unified(client)

    for result in results:
        if result.success:
            stats["timeout_closed"] = True
            position_manager.register_closed(symbol, "timeout")
            console.print(
                f"  [yellow]⏰ Closed[/] {result.direction} {result.epic} "
                f"[dim](held over {max_minutes} min)[/]"
            )
        else:
            console.print(
                f"  [red]✗ Failed to close[/] {result.epic}: {result.error}"
            )


def _try_auto_train(symbol: str, stats: dict) -> None:
    """Auto-train ML models if enough verified signals exist."""
    from trading_ai.backtesting.walk_forward import run_walk_forward
    from trading_ai.data.models import MLModelMeta, SignalVerification

    session_gen = get_session()
    session = next(session_gen)

    verified_count = (
        session.query(SignalVerification)
        .join(Signal, SignalVerification.signal_id == Signal.id)
        .filter(Signal.symbol == symbol)
        .count()
    )

    min_signals = SETTINGS.walk_forward.min_signals

    if verified_count < min_signals:
        display_auto_train_waiting(verified_count, min_signals)
        return

    last_train_count = None
    try:
        latest_model = (
            session.query(MLModelMeta)
            .filter(MLModelMeta.symbol == symbol, MLModelMeta.is_active == True)
            .order_by(MLModelMeta.version.desc())
            .first()
        )
        if latest_model:
            last_train_count = latest_model.train_signal_count
    except Exception:
        pass

    if last_train_count is not None and verified_count <= last_train_count:
        display_auto_train_skipped(verified_count, last_train_count)
        return

    display_auto_train_starting(verified_count)
    stats["auto_train_started"] = True

    try:
        reports = run_walk_forward(symbol, timeframe=None)
        for report in reports:
            display_auto_train_result(
                report.model_type, report.symbol,
                report.mean_accuracy, report.mean_roc_auc,
            )
    except Exception as e:
        logger.error("Auto-train failed for %s: %s", symbol, e)


def run_loop(client: CapitalComClient, position_manager: PositionManager, interval: int = 300) -> None:
    """Run the signal cycle in a loop with the given interval in seconds.

    This keeps the PositionManager alive between cycles so cooldowns
    and position state persist correctly.
    """
    import time
    from rich.text import Text

    while True:
        try:
            run_signal_cycle(client, position_manager)
        except Exception as e:
            logger.error("Cycle error: %s", e)
            console.print(f"[red]✗ Cycle error: {e}[/]")

        console.print()
        console.print(f"[dim]Next cycle in {interval // 60} min {interval % 60}s...[/]")
        console.print(f"[dim]Press Ctrl+C to stop[/]")
        console.print()
        time.sleep(interval)


def main() -> None:
    """Main entry point. Supports two modes:
    - Single cycle (default): python -m trading_ai.main
    - Continuous loop: python -m trading_ai.main --loop [--interval SECONDS]
    """
    import argparse
    import time

    _setup_logging()

    parser = argparse.ArgumentParser(description="Trading AI")
    parser.add_argument("--loop", action="store_true", help="Run continuously in a loop")
    parser.add_argument("--interval", type=int, default=300, help="Loop interval in seconds (default: 300)")
    args = parser.parse_args()

    platform_name = "MetaTrader 5" if SETTINGS.trading_platform == TradingPlatform.MT5 else "Capital.com"
    console.print(f"[bold cyan]Trading AI[/] starting...")
    console.print(f"[dim]Platform: {platform_name}[/]")
    create_tables()

    client = _create_client()
    if not _authenticate_client(client):
        sys.exit(1)

    # Create position manager — persists across cycles
    position_manager = PositionManager(client)
    position_manager.sync_positions()

    if args.loop:
        console.print(f"[cyan]Running in loop mode (every {args.interval}s)[/]")
        run_loop(client, position_manager, interval=args.interval)
    else:
        # Single cycle mode (for external loops like run_all.sh)
        run_signal_cycle(client, position_manager)


if __name__ == "__main__":
    main()
