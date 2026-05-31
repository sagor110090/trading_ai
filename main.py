"""Main orchestrator: wires all system components for the prediction loop."""

import logging
import sys
from typing import Optional

import pandas as pd
from rich.logging import RichHandler

from trading_ai.config.settings import SETTINGS, Symbol, Timeframe
from trading_ai.data.collector import CapitalComClient
from trading_ai.data.database import create_tables, get_session
from trading_ai.data.models import Candle, Signal
from trading_ai.display import (
    console,
    display_auto_train_result,
    display_auto_train_skipped,
    display_auto_train_starting,
    display_auto_train_waiting,
    display_candle_fetch,
    display_cycle_header,
    display_cycle_summary,
    display_ml_prediction,
    display_no_ml_model,
    display_no_signal,
    display_signal,
    display_timeout_closed,
    display_trade_blocked,
    display_trade_failed,
    display_trade_placed,
    display_trade_skipped_existing,
    display_verification_summary,
)
from trading_ai.indicators.features import compute_all_features
from trading_ai.ml.predictor import Prediction, predict_signal
from trading_ai.signals.engine import TradeSignal, generate_signal
from trading_ai.verification.verifier import verify_pending_signals

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging with Rich handler for beautiful output.

    Submodule logs at INFO level are suppressed — only WARNING and above
    appear in the console. The display module handles all INFO-level output.
    """
    # Rich handler for the root logger (catches WARNING+ from submodules)
    logging.basicConfig(
        level="WARNING",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    # Set __main__ logger to DEBUG so display-level info still works if needed
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


def run_signal_cycle(client: CapitalComClient) -> None:
    """One complete cycle: fetch data for all timeframes, run confluence, verify, predict."""
    for symbol in SETTINGS.symbols:
        display_cycle_header(symbol.value)

        # Track cycle results for summary
        stats = {
            "signal_generated": False,
            "ml_prediction": None,
            "ml_blocked": False,
            "ml_approved": False,
            "no_ml": False,
            "trade_placed": False,
            "trade_blocked": False,
            "trade_skipped": False,
            "trade_failed": False,
            "verified": 0,
            "auto_train_started": False,
            "timeout_closed": False,
        }

        fetched: dict[str, pd.DataFrame] = {}

        # ── 0. Check position timeouts ──────────────────────────────────
        if SETTINGS.auto_trade:
            _check_timeouts(client, stats)

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

        # ── 3. Multi-timeframe confluence signal ────────────────────────
        df_h4 = fetched.get(Timeframe.H4.value, pd.DataFrame())
        df_h1 = fetched.get(Timeframe.H1.value, pd.DataFrame())
        df_m5 = fetched.get(Timeframe.M5.value, pd.DataFrame())

        if not df_h4.empty and not df_h1.empty:
            trade_signal = generate_signal(df_h4, df_h1, df_m5, symbol, SETTINGS.trade_mode)

            if trade_signal:
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
                    _execute_trade(client, trade_signal, stats)
                elif SETTINGS.auto_trade and not should_trade:
                    display_trade_blocked("ML filter")
                    stats["trade_blocked"] = True

            else:
                display_no_signal(symbol.value)
        else:
            logger.warning("Missing H4 or H1 data for confluence check on %s", symbol.value)

        # ── 4. Auto-train ───────────────────────────────────────────────
        _try_auto_train(symbol.value, stats)

        # ── Cycle summary ───────────────────────────────────────────────
        display_cycle_summary(stats)


def _check_timeouts(client: CapitalComClient, stats: dict) -> None:
    """Check open positions and close any that exceeded their max hold time."""
    from trading_ai.trade.executor import check_timeouts, get_open_positions

    max_minutes = SETTINGS.get_max_hold_minutes()
    results = check_timeouts(client)

    for result in results:
        if result.success:
            stats["timeout_closed"] = True
            console.print(
                f"  [yellow]⏰ Closed[/] {result.direction} {result.epic} "
                f"[dim](held over {max_minutes} min)[/]"
            )
        else:
            console.print(
                f"  [red]✗ Failed to close[/] {result.epic}: {result.error}"
            )


def _execute_trade(client: CapitalComClient, trade_signal: TradeSignal, stats: dict) -> None:
    """Execute a trade on Capital.com."""
    from trading_ai.trade.executor import place_trade, has_open_position
    from trading_ai.data.collector import _EPIC_MAP

    symbol_enum = Symbol(trade_signal.symbol)
    epic = _EPIC_MAP.get(symbol_enum)

    # Check for existing position first
    if epic:
        existing = has_open_position(client, epic)
        if existing:
            display_trade_skipped_existing(trade_signal.symbol)
            stats["trade_skipped"] = True
            return

    size = SETTINGS.get_trade_size(trade_signal.symbol)
    result = place_trade(client, trade_signal, size=size)

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


def _try_auto_train(symbol: str, stats: dict) -> None:
    """Auto-train ML models if enough verified signals exist.

    Only trains when new verified signals have been added since the last training.
    """
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

    # Check if we already trained on this many verified signals
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


def main() -> None:
    """Main entry point."""
    _setup_logging()

    console.print("[bold cyan]Trading AI[/] starting...")
    create_tables()

    client = CapitalComClient()
    if not client.authenticate():
        console.print("[bold red]✗ Cannot authenticate with Capital.com. Check credentials.[/]")
        sys.exit(1)

    console.print("[green]✓ Authenticated with Capital.com[/]")
    run_signal_cycle(client)


if __name__ == "__main__":
    main()