"""Main orchestrator: wires all system components for the prediction loop."""

import logging
import sys
from typing import Optional

import pandas as pd

from trading_ai.config.settings import SETTINGS, Symbol, Timeframe
from trading_ai.data.collector import CapitalComClient
from trading_ai.data.database import create_tables, get_session
from trading_ai.data.models import Candle, Signal
from trading_ai.indicators.features import compute_all_features
from trading_ai.signals.engine import TradeSignal, generate_signal
from trading_ai.ml.predictor import Prediction, predict_signal
from trading_ai.verification.verifier import verify_pending_signals

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=SETTINGS.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


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

    logger.info("Stored signal %s to database", trade_signal.signal_id)
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


def _get_recent_candles_for_verification(symbol: str, timeframe: str, limit: int = 100) -> list:
    """Load recent candles as ORM objects for verification."""
    session_gen = get_session()
    session = next(session_gen)

    candles = (
        session.query(Candle)
        .filter(Candle.symbol == symbol, Candle.timeframe == timeframe)
        .order_by(Candle.timestamp.desc())
        .limit(limit)
        .all()
    )

    return list(reversed(candles))


def run_signal_cycle(client: CapitalComClient) -> None:
    """One complete cycle: fetch data for all timeframes, run confluence, verify, predict."""
    for symbol in SETTINGS.symbols:
        logger.info("=== Processing %s ===", symbol.value)

        fetched: dict[str, pd.DataFrame] = {}

        # 1. Fetch and store candles for all timeframes
        for timeframe in SETTINGS.timeframes:
            try:
                df = client.fetch_candles(symbol, timeframe, limit=500)
            except Exception as e:
                logger.error("Failed to fetch %s %s: %s", symbol.value, timeframe.value, e)
                continue

            if df.empty:
                logger.warning("No candle data for %s %s", symbol.value, timeframe.value)
                continue

            new_count = _store_candles(df, symbol.value, timeframe.value)
            logger.info("Stored %d new candles for %s %s", new_count, symbol.value, timeframe.value)

            fetched[timeframe.value] = _load_candles_from_db(symbol.value, timeframe.value, limit=500)

        # 2. Verify pending signals
        for timeframe in SETTINGS.timeframes:
            recent = _get_recent_candles_for_verification(symbol.value, timeframe.value)
            if recent:
                verified = verify_pending_signals(symbol.value, timeframe.value, recent)
                if verified:
                    logger.info("Verified %d signals for %s %s", verified, symbol.value, timeframe.value)

        # 3. Multi-timeframe confluence signal
        df_h4 = fetched.get(Timeframe.H4.value, pd.DataFrame())
        df_h1 = fetched.get(Timeframe.H1.value, pd.DataFrame())
        df_m5 = fetched.get(Timeframe.M5.value, pd.DataFrame())

        if not df_h4.empty and not df_h1.empty:
            mode = SETTINGS.trade_mode
            logger.info("Running %s signal for %s (H4=%d H1=%d M5=%d)",
                        mode.value, symbol.value, len(df_h4), len(df_h1), len(df_m5))

            trade_signal = generate_signal(df_h4, df_h1, df_m5, symbol, mode)

            if trade_signal:
                _store_signal(trade_signal)

                prediction = predict_signal(trade_signal)
                if prediction:
                    logger.info(
                        "ML: %s prob=%.4f ml_conf=%.2f model=%s_v%d [%s]",
                        prediction.signal, prediction.probability, prediction.confidence,
                        prediction.model_type, prediction.model_version,
                        trade_signal.confluence,
                    )
                else:
                    logger.info(
                        "Signal: %s %s confidence=%.2f entry=%.5f SL=%.5f TP=%.5f [%s]",
                        trade_signal.trade_mode, trade_signal.signal.value, trade_signal.confidence,
                        trade_signal.entry, trade_signal.stop_loss, trade_signal.take_profit,
                        trade_signal.confluence,
                    )
            else:
                logger.info("No confluence signal for %s this cycle", symbol.value)
        else:
            logger.warning("Missing H4 or H1 data for confluence check on %s", symbol.value)

        # 4. Auto-train if enough verified signals
        _try_auto_train(symbol.value)


def _try_auto_train(symbol: str) -> None:
    """Auto-train ML models if enough verified signals exist.

    Trains once per symbol when the minimum threshold is reached.
    Checks every cycle but only trains when new signals are available.
    """
    from trading_ai.backtesting.walk_forward import run_walk_forward
    from trading_ai.data.models import SignalVerification

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
        logger.info(
            "Auto-train: %d/%d verified signals for %s — need %d more",
            verified_count, min_signals, symbol, min_signals - verified_count,
        )
        return

    logger.info(
        "Auto-train: %d verified signals for %s — starting training!",
        verified_count, symbol,
    )

    try:
        reports = run_walk_forward(symbol, "confluence")
        for report in reports:
            logger.info(
                "Auto-train result: %s %s acc=%.4f auc=%.4f",
                report.model_type, report.symbol,
                report.mean_accuracy, report.mean_roc_auc,
            )
    except Exception as e:
        logger.error("Auto-train failed for %s: %s", symbol, e)
    """Run walk-forward training for all symbol/timeframe combinations."""
    from trading_ai.backtesting.walk_forward import run_walk_forward

    for symbol in SETTINGS.symbols:
        for timeframe in SETTINGS.timeframes:
            logger.info("Training models for %s %s", symbol.value, timeframe.value)
            reports = run_walk_forward(symbol.value, timeframe.value)

            for report in reports:
                logger.info(
                    "Training result: %s %s %s acc=%.4f auc=%.4f",
                    report.model_type, report.symbol, report.timeframe,
                    report.mean_accuracy, report.mean_roc_auc,
                )


def main() -> None:
    """Main entry point."""
    _setup_logging()
    logger.info("Trading AI system starting")

    create_tables()

    client = CapitalComClient()
    if not client.authenticate():
        logger.error("Cannot authenticate with Capital.com. Check credentials.")
        sys.exit(1)

    logger.info("System ready. Starting signal cycle.")
    run_signal_cycle(client)
    logger.info("Signal cycle complete.")


if __name__ == "__main__":
    main()