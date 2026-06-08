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
import threading
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from rich.logging import RichHandler

from config.settings import SETTINGS, Symbol, Timeframe, TradingPlatform
from config.utils import normalize_symbol
from data.collector import CapitalComClient
from data.database import create_tables, get_session
from data.models import Candle, Signal, SignalVerification
from display import (
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
    display_missing_confluence_data,
    display_no_ml_model,
    display_no_signal,
    display_position_external_close,
    display_position_reversed,
    display_signal,
    display_signal_skipped_cooldown,
    display_trade_blocked,
    display_trade_failed,
    display_trade_placed,
    display_trade_skipped_cooldown,
    display_trade_skipped_existing,
    display_verification_summary,
    display_weekly_close_positions_closed,
    display_weekly_close_status,
)
from indicators.features import compute_all_features
from ml.drift_detection import check_model_drift, sync_verified_signals_for_drift, track_ml_prediction, update_prediction_outcome
from ml.predictor import Prediction, predict_signal
from signals.engine import TradeSignal, generate_signal
from trade.executor import (
    close_position_unified,
    place_trade_unified,
)
from trade.position_manager import PositionManager
from trade.position_monitor import PositionMonitor
from trade.weekly_close import check_weekly_close_status
from rich.panel import Panel
from verification.verifier import verify_pending_signals

logger = logging.getLogger(__name__)


def _create_client():
    """Create the appropriate trading client based on platform selection."""
    if SETTINGS.trading_platform == TradingPlatform.MT5:
        from data.mt5_client import MT5Client
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
    count = 0
    normalized_symbol = normalize_symbol(symbol)  # Normalize for DB consistency

    with get_session() as session:
        for _, row in df.iterrows():
            ts = row["timestamp"]
            if isinstance(ts, str):
                ts = pd.Timestamp(ts, tz="UTC")

            exists = (
                session.query(Candle)
                .filter(
                    Candle.symbol == normalized_symbol,
                    Candle.timeframe == timeframe,
                    Candle.timestamp == ts,
                )
                .first()
            )
            if exists:
                continue

            candle = Candle(
                symbol=normalized_symbol,
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

    return count


def _store_signal(trade_signal: TradeSignal) -> Signal:
    """Persist a trade signal to the database.

    If a signal with the same signal_id already exists (same candle
    reprocessed in a later cycle), the existing row is returned silently.
    """
    from sqlalchemy.exc import IntegrityError

    with get_session() as session:
        # Check if already stored (same signal_id from a previous cycle)
        existing = (
            session.query(Signal)
            .filter(Signal.signal_id == trade_signal.signal_id)
            .first()
        )
        if existing:
            logger.debug("Signal %s already in DB, skipping insert", trade_signal.signal_id[:8])
            return existing

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
            session.flush()
        except IntegrityError:
            session.rollback()
            logger.debug("Signal %s duplicate on commit, skipping", trade_signal.signal_id[:8])
            existing = (
                session.query(Signal)
                .filter(Signal.signal_id == trade_signal.signal_id)
                .first()
            )
            return existing

        return signal


def _load_candles_from_db(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Load recent candles from the database into a DataFrame."""
    normalized_symbol = normalize_symbol(symbol)  # Normalize for DB consistency
    with get_session() as session:
        candles = (
            session.query(Candle)
            .filter(Candle.symbol == normalized_symbol, Candle.timeframe == timeframe)
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


def _check_position_strength(
    client: CapitalComClient,
    position_monitor: PositionMonitor,
    position_manager: PositionManager,
) -> None:
    """Check all open positions for TP progress and strength.

    Evaluates:
    - TP progress (if >= 60%, evaluate strength)
    - Trade strength score
    - Close position if score < 50

    Also logs decisions to the trade log database.
    """
    from indicators.features import compute_all_features
    from display import display_position_strength_exit
    from rich.table import Table

    def df_provider(symbol: str, timeframe: str) -> pd.DataFrame:
        """Provider function to get DataFrame for strength scoring.

        `symbol` and `timeframe` are strings coming from the position monitor,
        but `client.fetch_candles` expects Symbol/Timeframe enum values.
        We resolve them here.
        """
        try:
            try:
                sym_enum = Symbol(symbol) if symbol not in [s.value for s in Symbol] else Symbol(symbol)
            except ValueError:
                sym_enum = next((s for s in Symbol if s.value == symbol), None)
            if sym_enum is None:
                logger.error("Unknown symbol for strength check: %s", symbol)
                return pd.DataFrame()

            tf_enum = None
            for tf in Timeframe:
                if tf.value == timeframe:
                    tf_enum = tf
                    break
            if tf_enum is None:
                logger.error("Unknown timeframe for strength check: %s", timeframe)
                return pd.DataFrame()

            df = client.fetch_candles(sym_enum, tf_enum, limit=100)
            if df.empty:
                return pd.DataFrame()
            df = compute_all_features(df)
            return df
        except Exception as e:
            logger.error("Failed to fetch data for strength check: %s", e)
            return pd.DataFrame()

    # Check all positions
    decisions = position_monitor.check_all_positions(df_provider=df_provider)

    # Display results in a table
    if decisions:
        table = Table(title="[bold cyan]Position Monitor[/]", show_header=True, header_style="bold magenta")
        table.add_column("Symbol", style="cyan", width=10)
        table.add_column("Dir", style="cyan", width=4)
        table.add_column("TP Progress", justify="right", width=12)
        table.add_column("P&L", justify="right", width=10)
        table.add_column("Strength", justify="right", width=8)
        table.add_column("Status", style="bold", width=10)

        has_close = False

        for decision in decisions:
            if decision.action == "close":
                has_close = True
                table.add_row(
                    decision.symbol[:10],
                    decision.direction[:4],
                    f"[red]{decision.tp_progress:.1%}[/]",
                    f"[red]{decision.pnl or 0:+.2f}[/]",
                    f"[red]{decision.strength_score or 0:.1f}[/]",
                    "[red]CLOSED[/]",
                )
            elif decision.action == "external_close":
                has_close = True
                status_label = "EXTERNAL"
                if decision.close_reason == "tp_close":
                    status_label = "TP HIT"
                elif decision.close_reason == "sl_close":
                    status_label = "SL HIT"
                table.add_row(
                    decision.symbol[:10],
                    decision.direction[:4],
                    f"[yellow]{status_label}[/]",
                    f"[yellow]{decision.pnl or 0:+.2f}[/]",
                    "N/A",
                    "[yellow]EXTERNAL[/]",
                )
                active_cooldown = position_manager.get_active_cooldown(decision.symbol)
                actual_cooldown = 0
                if active_cooldown is not None:
                    actual_cooldown = round(
                        max((active_cooldown.until - datetime.now(timezone.utc)).total_seconds() / 60, 0.0)
                    )
                logger.info(
                    "External close detected: %s %s - cooldown already active for %dmin",
                    decision.symbol, decision.direction, actual_cooldown,
                )
                display_position_external_close(
                    decision.symbol,
                    decision.direction,
                    cooldown_minutes=actual_cooldown,
                    close_reason=decision.close_reason or "external_close",
                )
            elif decision.action == "trail_stop":
                table.add_row(
                    decision.symbol[:10],
                    decision.direction[:4],
                    f"[green]{decision.tp_progress:.1%}[/]",
                    f"[green]{decision.pnl or 0:+.2f}[/]",
                    f"{decision.strength_score:.1f}" if decision.strength_score is not None else "N/A",
                    f"[cyan]SL->{decision.new_sl:.2f}[/]",
                )
            elif decision.action == "hold":
                if decision.strength_score is not None:
                    # Strength was evaluated
                    if decision.strength_score >= 70:
                        status = "HOLD"
                        style = "[green]"
                    elif decision.strength_score >= 50:
                        status = "MONITOR"
                        style = "[yellow]"
                    else:
                        status = "WEAK"
                        style = "[red]"
                else:
                    # Just tracking TP progress
                    status = "TRACK"
                    style = "[dim]"

                progress_color = "[green]" if decision.tp_progress >= 0.6 else "[yellow]" if decision.tp_progress >= 0.3 else "[dim]"
                pnl_color = "[green]" if (decision.pnl or 0) > 0 else "[red]" if (decision.pnl or 0) < 0 else "[dim]"

                table.add_row(
                    decision.symbol[:10],
                    decision.direction[:4],
                    f"{progress_color}{decision.tp_progress:.1%}[/]",
                    f"{pnl_color}{decision.pnl or 0:+.2f}[/]",
                    f"{decision.strength_score:.1f}" if decision.strength_score is not None else "N/A",
                    f"{style}{status}[/]",
                )
            elif decision.action == "error":
                table.add_row(
                    "ERROR",
                    "",
                    "N/A",
                    "N/A",
                    "N/A",
                    f"[red]{decision.reason[:15]}...[/]",
                )

        console.print(table)

        # Also show individual close panels for positions that were closed
        if has_close:
            for decision in decisions:
                if decision.action == "close":
                    display_position_strength_exit(
                        symbol=decision.symbol,
                        direction=decision.direction,
                        tp_progress=decision.tp_progress,
                        strength_score=decision.strength_score or 0,
                        pnl=decision.pnl or 0,
                        reason=decision.reason,
                    )
                # external_close already displayed above with correct cooldown (30 min)


def run_signal_cycle(
    client: CapitalComClient,
    position_manager: PositionManager,
    position_monitor: Optional[PositionMonitor] = None,
) -> None:
    """One complete cycle: fetch data for all timeframes, run confluence, verify, predict.

    The position_manager controls trade execution:
    - Same direction as open position → skip (already in this trade)
    - Opposite direction → close old position and take new one (reversal)
    - No position + no cooldown → place trade
    - Cooldown active → skip trade
    """
    # Clean up expired cooldowns
    position_manager.cleanup_expired_cooldowns()
    timeout_closed_symbols: set[str] = set()

    # Sync broker state before evaluating symbols so external/manual closes
    # activate cooldowns in the same cycle they are discovered.
    if position_monitor is not None:
        position_monitor.sync_positions()

    # Timeout checks must run once per cycle, not once per symbol, otherwise
    # a single broker close can be processed multiple times under the wrong key.
    if SETTINGS.auto_trade:
        timeout_closed_symbols = _check_timeouts(client, position_manager)

    # NOTE: Position-monitor display is handled by the background
    # _monitor_loop (runs every 60s). It is intentionally NOT called
    # here so the monitor output isn't duplicated every main cycle.

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
            "trade_correlation_blocked": False,
            "ml_drift_detected": False,
            "verified": 0,
            "auto_train_started": False,
            "timeout_closed": symbol.value in timeout_closed_symbols,
            "strength_exit": False,
            "weekly_close_closed": 0,
        }

        fetched: dict[str, pd.DataFrame] = {}

        # Catch broker/manual closes as late as possible before analysis.
        if position_monitor is not None:
            position_monitor.sync_positions()

        # ── 0.5. Check weekly close ────────────────────────────────────────
        skip_symbol = False
        if SETTINGS.weekly_close.enabled:
            close_status = check_weekly_close_status(
                symbol=symbol.value,
                stop_new_trades_hours=SETTINGS.weekly_close.stop_new_trades_hours,
                close_positions_hours=SETTINGS.weekly_close.close_positions_hours,
            )
            # Display weekly close status (for all symbols with weekly close)
            if close_status.hours_until_close is not None or close_status.is_weekend:
                display_weekly_close_status(
                    symbol=symbol.value,
                    hours_until=close_status.hours_until_close,
                    should_stop=close_status.should_stop_new_trades,
                    should_close=close_status.should_close_positions,
                    is_weekend=close_status.is_weekend,
                    message=close_status.message,
                    show_always=True,
                )
            # Close positions if we're in the weekly close window
            if close_status.should_close_positions and SETTINGS.auto_trade:
                closed = position_manager.close_all_for_weekly_close(symbol.value)
                if closed:
                    display_weekly_close_positions_closed(len(closed), symbol.value)
                    stats["weekly_close_closed"] = len(closed)
            # Skip analysis during weekly close window (no new data)
            if close_status.should_stop_new_trades:
                logger.info("Skipping %s analysis during weekly close: %s", symbol.value, close_status.message)
                skip_symbol = True

        if skip_symbol:
            continue

        # ── 1. Fetch and store candles ──────────────────────────────────
        candle_results: dict[str, int] = {}
        loaded_counts: dict[str, int] = {}
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
            loaded_counts[timeframe.value] = len(fetched[timeframe.value])

        display_candle_fetch(candle_results, loaded_counts)

        # ── 2. Verify pending signals ────────────────────────────────────
        verified = verify_pending_signals(symbol.value)
        with get_session() as session:
            pending_count = (
                session.query(Signal)
                .outerjoin(Signal.verification)
                .filter(Signal.symbol == symbol.value, Signal.verification == None)
                .count()
            )
        display_verification_summary(verified, pending_count)
        stats["verified"] = verified

        # Register verified losses as cooldowns
        if verified > 0:
            _register_verified_results(symbol.value, position_manager)

            # Sync verified signals with drift detection
            if SETTINGS.ml.drift_detection_enabled:
                try:
                    synced = sync_verified_signals_for_drift(symbol.value)
                    if synced > 0:
                        logger.debug("Synced %d predictions with outcomes for drift detection", synced)

                    # Check for drift periodically (every 10 verifications)
                    if verified >= 10 or synced >= 10:
                        drift_report = check_model_drift(symbol.value)
                        if drift_report and drift_report.drift_detected:
                            console.print(Panel(
                                drift_report.message,
                                title="[bold red]⚠ ML Model Drift Alert[/]",
                                border_style="red",
                            ))
                            stats["ml_drift_detected"] = True
                except Exception as e:
                    logger.warning("Drift detection error: %s", e)

        # ── 3. Hard cooldown gate before signal generation ───────────────
        if position_manager.is_active(symbol.value):
            remaining_min = position_manager.get_cooldown_remaining_minutes(symbol.value)
            logger.info("%s: cooldown active, skipping signal generation", symbol.value)
            display_signal_skipped_cooldown(symbol.value, remaining_min)
            if SETTINGS.ml.auto_train:
                _try_auto_train(symbol.value, stats)
            display_cycle_summary(stats)
            continue

        # ── 4. Check if we should skip signal generation ─────────────────
        should_generate, skip_reason = position_manager.should_generate_signal(symbol.value)
        if not should_generate:
            # Extract remaining minutes from skip_reason (format: "Loss cooldown: X min remaining")
            remaining_min = 0
            if "min remaining" in skip_reason:
                try:
                    remaining_min = float(skip_reason.split(":")[1].split("min")[0].strip())
                except (IndexError, ValueError):
                    pass
            display_signal_skipped_cooldown(symbol.value, remaining_min)
            # Still verify and auto-train (if enabled), but don't generate new signals
            if SETTINGS.ml.auto_train:
                _try_auto_train(symbol.value, stats)
            display_cycle_summary(stats)
            continue

        # ── 5. Multi-timeframe confluence signal ────────────────────────
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

        if not df_h4.empty and not df_h1.empty:
            trade_signal = generate_signal(
                df_h4, df_h1, df_m5, symbol, SETTINGS.trade_mode, live_price=live_price,
                prefix="live_",
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

                    # ── ML prediction (even for duplicates) ─────────────
                    should_trade = True
                    prediction = predict_signal(trade_signal)
                    if prediction:
                        stats["ml_prediction"] = prediction
                        threshold = SETTINGS.get_prediction_threshold(trade_signal.symbol)
                        if prediction.probability < threshold:
                            should_trade = False
                            stats["ml_blocked"] = True
                            display_ml_prediction(prediction, threshold, blocked=True)
                        else:
                            stats["ml_approved"] = True
                            display_ml_prediction(prediction, threshold, blocked=False)

                        # Track prediction for drift detection
                        try:
                            track_ml_prediction(
                                signal_id=trade_signal.signal_id,
                                symbol=trade_signal.symbol,
                                model_type=prediction.model_type,
                                model_version=prediction.model_version,
                                ml_probability=prediction.probability,
                            )
                        except Exception as e:
                            logger.warning("Failed to track ML prediction: %s", e)
                    else:
                        stats["no_ml"] = True
                        display_no_ml_model()

                    # Even for duplicates, check if we should reverse
                    if SETTINGS.auto_trade and should_trade:
                        trade_placed = _handle_trade_decision(
                            client,
                            trade_signal,
                            position_manager,
                            stats,
                            position_monitor=position_monitor,
                        )
                        # Only save the signal if a new trade was actually placed
                        if trade_placed:
                            _store_signal(trade_signal)
                    elif SETTINGS.auto_trade and not should_trade:
                        display_trade_blocked("ML filter")
                        stats["trade_blocked"] = True

                    if SETTINGS.ml.auto_train:
                        _try_auto_train(symbol.value, stats)
                    display_cycle_summary(stats)
                    continue

                # Only save signal to database if NOT in auto-trade mode
                # In auto-trade mode, signals are only saved AFTER a trade is placed
                # This prevents false "wins" in verification for signals that never executed
                if not SETTINGS.auto_trade:
                    _store_signal(trade_signal)
                display_signal(trade_signal)
                stats["signal_generated"] = True
                stats["signal"] = trade_signal

                # ── ML prediction ──────────────────────────────────────
                should_trade = True
                prediction = predict_signal(trade_signal)
                if prediction:
                    stats["ml_prediction"] = prediction
                    threshold = SETTINGS.get_prediction_threshold(trade_signal.symbol)
                    if prediction.probability < threshold:
                        should_trade = False
                        stats["ml_blocked"] = True
                        display_ml_prediction(prediction, threshold, blocked=True)
                    else:
                        stats["ml_approved"] = True
                        display_ml_prediction(prediction, threshold, blocked=False)

                    # Track prediction for drift detection
                    try:
                        track_ml_prediction(
                            signal_id=trade_signal.signal_id,
                            symbol=trade_signal.symbol,
                            model_type=prediction.model_type,
                            model_version=prediction.model_version,
                            ml_probability=prediction.probability,
                        )
                    except Exception as e:
                        logger.warning("Failed to track ML prediction: %s", e)
                else:
                    stats["no_ml"] = True
                    display_no_ml_model()

                # ── Auto-trade ──────────────────────────────────────────
                if SETTINGS.auto_trade and should_trade:
                    _handle_trade_decision(
                        client,
                        trade_signal,
                        position_manager,
                        stats,
                        position_monitor=position_monitor,
                    )
                elif SETTINGS.auto_trade and not should_trade:
                    display_trade_blocked("ML filter")
                    stats["trade_blocked"] = True

            else:
                display_no_signal(symbol.value)
        else:
            logger.warning("Missing H4 or H1 data for confluence check on %s", symbol.value)
            display_missing_confluence_data(symbol.value, len(df_h4), len(df_h1), len(df_m5))

        # ── 6. Auto-train ───────────────────────────────────────────────
        if SETTINGS.ml.auto_train:
            _try_auto_train(symbol.value, stats)

        # ── Cycle summary ───────────────────────────────────────────────
        display_cycle_summary(stats)


def _handle_trade_decision(
    client: CapitalComClient,
    trade_signal: TradeSignal,
    position_manager: PositionManager,
    stats: dict,
    position_monitor: Optional[PositionMonitor] = None,
) -> bool:
    """Evaluate and execute a trade based on position state.

    Returns True if a NEW trade was placed, False otherwise.

    Handles three scenarios:
    1. No position, no cooldown → place new trade (returns True)
    2. Same direction position → SKIP (already in this trade) (returns False)
    3. Opposite direction position → close old, place new (reversal) (returns True)
    4. Cooldown active → skip (returns False)
    """
    from trade.executor import OrderResult

    if position_monitor is not None:
        position_monitor.sync_positions()

    if position_manager.is_active(trade_signal.symbol):
        display_trade_skipped_cooldown(
            trade_signal.symbol,
            position_manager.get_cooldown_remaining_minutes(trade_signal.symbol),
        )
        stats["trade_cooldown"] = True
        return False

    decision = position_manager.evaluate_trade(
        trade_signal.symbol,
        trade_signal.signal.value,
        trade_signal.stop_loss,
        trade_signal.take_profit,
    )

    if decision.action == "skip_cooldown":
        # Cooldown active - skip
        display_trade_skipped_cooldown(trade_signal.symbol, decision.cooldown_remaining_minutes)
        stats["trade_cooldown"] = True
        return False  # No new trade placed

    if decision.action == "skip_same_direction":
        # Same direction position already open - skip
        display_trade_skipped_existing(trade_signal.symbol)
        stats["trade_skipped"] = True
        return False  # No new trade placed

    if decision.action == "skip_correlation":
        # Correlation limit reached - skip
        console.print(
            f"  [yellow]⚠ Correlation limit:[/] {decision.reason}"
        )
        stats["trade_skipped"] = True
        stats["trade_correlation_blocked"] = True
        return False  # No new trade placed

    if decision.action == "reverse_and_place":
        # Close existing position, then place new one
        pos = decision.existing_position
        if pos is None:
            logger.error("Reverse requested without an existing position for %s", trade_signal.symbol)
            stats["trade_failed"] = True
            return False
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
            return False  # No new trade placed

        position_manager.register_position_close(
            symbol=trade_signal.symbol,
            reason="reversal",
            deal_id=pos.deal_id,
            source="main_reversal",
        )
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
            # Save signal to database ONLY when a new trade is actually placed
            # This prevents verification errors for signals that never executed
            _store_signal(trade_signal)
            display_trade_placed(result)
            stats["trade_placed"] = True
            return True  # New trade placed
        else:
            if "already exists" in (result.error or "").lower():
                display_trade_skipped_existing(trade_signal.symbol)
                stats["trade_skipped"] = True
            else:
                display_trade_failed(result)
                stats["trade_failed"] = True
            return False  # No new trade placed

    return False  # Default: no new trade placed


def _register_verified_results(symbol: str, position_manager: PositionManager) -> None:
    """Check verified results from this cycle and register cooldowns for losses."""
    with get_session() as session:
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
    position_manager: PositionManager,
 ) -> set[str]:
    """Check open positions and close any that exceeded their max hold time."""
    from trade.executor import check_timeouts_unified

    max_minutes = SETTINGS.get_max_hold_minutes()
    results = check_timeouts_unified(client)
    closed_symbols: set[str] = set()

    for result in results:
        if result.success:
            closed_symbol = position_manager.canonicalize_symbol(result.epic)
            position_manager.register_position_close(
                closed_symbol,
                "timeout",
                deal_id=result.deal_id,
                source="timeout_check",
            )
            closed_symbols.add(closed_symbol)
            console.print(
                f"  [yellow]⏰ Closed[/] {result.direction} {result.epic} "
                f"[dim](held over {max_minutes} min)[/]"
            )
        else:
            console.print(
                f"  [red]✗ Failed to close[/] {result.epic}: {result.error}"
            )

    return closed_symbols


def _try_auto_train(symbol: str, stats: dict) -> None:
    """Auto-train ML models if enough verified signals exist."""
    from backtesting.walk_forward import run_walk_forward
    from data.models import MLModelMeta, SignalVerification

    with get_session() as session:
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


_monitor_lock = threading.Lock()


def _monitor_loop(
    client: CapitalComClient,
    position_manager: PositionManager,
    position_monitor: PositionMonitor,
    interval: int = 60,
) -> None:
    """Background loop that runs the position monitor every `interval` seconds.

    This runs independently of the main signal cycle so open positions are
    tracked more frequently (default 30s) for better TP / strength decisions.
    A lock prevents overlap with the main cycle.
    """
    import time
    from rich.text import Text

    while True:
        try:
            from datetime import datetime
            from rich.panel import Panel
            now_str = datetime.now().strftime("%H:%M:%S")
            with _monitor_lock:
                banner = (
                    "[bold]Position Monitor[/]\n"
                    f"[dim]Tick: every {interval}s (background)[/]\n"
                    f"[dim]Time: {now_str}[/]"
                )
                console.print(
                    Panel(banner, border_style="cyan", expand=False, width=44)
                )
                _check_position_strength(client, position_monitor, position_manager)
        except Exception as e:
            logger.error("Monitor loop error: %s", e)
            console.print(f"[red]✗ Monitor loop error: {e}[/]")
        console.print()
        console.print(f"[dim]Next position-monitor check in {interval}s...[/]")
        time.sleep(interval)


def run_loop(client: CapitalComClient, position_manager: PositionManager, position_monitor: PositionMonitor, interval: int = 300) -> None:
    """Run the signal cycle in a loop with the given interval in seconds.

    This keeps the PositionManager alive between cycles so cooldowns
    and position state persist correctly.

    A background thread also runs the position monitor every 30s so open
    positions are tracked more frequently than the main cycle.
    """
    import time
    from rich.text import Text

    monitor_thread = threading.Thread(
        target=_monitor_loop,
        args=(client, position_manager, position_monitor, SETTINGS.position_monitor_interval),
        daemon=True,
        name="position-monitor",
    )
    monitor_thread.start()
    console.print(f"[cyan]Position monitor running every {SETTINGS.position_monitor_interval}s (background thread)[/]")

    while True:
        try:
            with _monitor_lock:
                run_signal_cycle(client, position_manager, position_monitor)
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

    # Create position monitor — tracks TP progress and manages exits
    position_monitor = PositionMonitor(
        client,
        on_position_closed=position_manager.register_position_close,
    )
    position_monitor.sync_positions()

    if args.loop:
        console.print(f"[cyan]Running in loop mode (every {args.interval}s)[/]")
        run_loop(client, position_manager, position_monitor, interval=args.interval)
    else:
        # Single cycle mode (for external loops like run_all.sh)
        run_signal_cycle(client, position_manager, position_monitor)


if __name__ == "__main__":
    main()
