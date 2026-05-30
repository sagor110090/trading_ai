"""Signal verification: monitors future candles to determine signal outcome."""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from trading_ai.config.settings import SETTINGS, SignalResult, SignalType
from trading_ai.data.database import get_session
from trading_ai.data.models import Candle, Signal, SignalVerification

logger = logging.getLogger(__name__)

VERIFICATION_CONFIG = SETTINGS.verification


@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying a signal against subsequent price action."""
    signal_id: str
    result: SignalResult
    profit_points: Optional[float]
    risk_reward: Optional[float]
    holding_candles: Optional[int]
    max_favorable_excursion: Optional[float]
    max_adverse_excursion: Optional[float]


def verify_signal(
    signal: Signal,
    candles_after: list[Candle],
) -> Optional[VerificationResult]:
    """Verify a signal against subsequent candle data.

    Checks if price hit take_profit (WIN), stop_loss (LOSS), or expired (BREAKEVEN).

    Args:
        signal: The Signal record from the database.
        candles_after: Candles chronologically after the signal, sorted ascending.

    Returns:
        VerificationResult or None if not yet resolvable.
    """
    if not candles_after:
        return None

    entry = signal.entry
    sl = signal.stop_loss
    tp = signal.take_profit
    direction = SignalType(signal.signal)
    breakeven_thresh = VERIFICATION_CONFIG.breakeven_threshold_pips

    max_favorable = 0.0
    max_adverse = 0.0
    holding_candles = 0

    for i, candle in enumerate(candles_after):
        holding_candles = i + 1

        if direction == SignalType.BUY:
            # Check stop loss hit (low touches or goes below SL)
            if candle.low <= sl:
                mfe = max_favorable
                mae = entry - candle.low  # adverse in price units
                profit = sl - entry  # negative
                rr = abs(profit) / abs(tp - entry) if abs(tp - entry) > 0 else 0.0
                return VerificationResult(
                    signal_id=signal.signal_id,
                    result=SignalResult.LOSS,
                    profit_points=round(profit, 5),
                    risk_reward=round(-rr, 4),
                    holding_candles=holding_candles,
                    max_favorable_excursion=round(mfe, 5),
                    max_adverse_excursion=round(mae, 5),
                )

            # Check take profit hit (high touches or goes above TP)
            if candle.high >= tp:
                mfe = candle.high - entry
                mae = max_adverse
                profit = tp - entry
                rr = abs(profit) / abs(sl - entry) if abs(sl - entry) > 0 else 0.0
                return VerificationResult(
                    signal_id=signal.signal_id,
                    result=SignalResult.WIN,
                    profit_points=round(profit, 5),
                    risk_reward=round(rr, 4),
                    holding_candles=holding_candles,
                    max_favorable_excursion=round(mfe, 5),
                    max_adverse_excursion=round(mae, 5),
                )

            # Track MFE/MAE
            favorable = candle.high - entry
            adverse = entry - candle.low
            max_favorable = max(max_favorable, favorable)
            max_adverse = max(max_adverse, adverse)

        elif direction == SignalType.SELL:
            # Check stop loss hit (high touches or goes above SL)
            if candle.high >= sl:
                mfe = max_favorable
                mae = candle.high - entry
                profit = entry - sl  # negative
                rr = abs(profit) / abs(entry - tp) if abs(entry - tp) > 0 else 0.0
                return VerificationResult(
                    signal_id=signal.signal_id,
                    result=SignalResult.LOSS,
                    profit_points=round(profit, 5),
                    risk_reward=round(-rr, 4),
                    holding_candles=holding_candles,
                    max_favorable_excursion=round(mfe, 5),
                    max_adverse_excursion=round(mae, 5),
                )

            # Check take profit hit (low touches or goes below TP)
            if candle.low <= tp:
                mfe = entry - candle.low
                mae = max_adverse
                profit = entry - tp
                rr = abs(profit) / abs(entry - sl) if abs(entry - sl) > 0 else 0.0
                return VerificationResult(
                    signal_id=signal.signal_id,
                    result=SignalResult.WIN,
                    profit_points=round(profit, 5),
                    risk_reward=round(rr, 4),
                    holding_candles=holding_candles,
                    max_favorable_excursion=round(mfe, 5),
                    max_adverse_excursion=round(mae, 5),
                )

            # Track MFE/MAE
            favorable = entry - candle.low
            adverse = candle.high - entry
            max_favorable = max(max_favorable, favorable)
            max_adverse = max(max_adverse, adverse)

    # Exhausted lookahead candles without SL/TP hit
    if holding_candles >= VERIFICATION_CONFIG.lookahead_candles:
        # Determine breakeven vs loss by final position
        last_close = candles_after[-1].close
        if direction == SignalType.BUY:
            unrealized = last_close - entry
        else:
            unrealized = entry - last_close

        if abs(unrealized) <= breakeven_thresh:
            result = SignalResult.BREAKEVEN
        elif unrealized > 0:
            result = SignalResult.WIN
        else:
            result = SignalResult.LOSS

        return VerificationResult(
            signal_id=signal.signal_id,
            result=result,
            profit_points=round(unrealized, 5),
            risk_reward=None,
            holding_candles=holding_candles,
            max_favorable_excursion=round(max_favorable, 5),
            max_adverse_excursion=round(max_adverse, 5),
        )

    # Not yet resolved
    return None


def store_verification(result: VerificationResult, signal_db_id: int) -> None:
    """Persist a verification result to the database."""
    session_gen = get_session()
    session = next(session_gen)
    try:
        verification = SignalVerification(
            signal_id=signal_db_id,
            result=result.result,
            profit_points=result.profit_points,
            risk_reward=result.risk_reward,
            holding_candles=result.holding_candles,
            max_favorable_excursion=result.max_favorable_excursion,
            max_adverse_excursion=result.max_adverse_excursion,
        )
        session.add(verification)
        logger.info("Stored verification for signal %s: %s", result.signal_id, result.result.value)
    except Exception:
        session.rollback()
        raise


def verify_pending_signals(symbol: str, timeframe: str, recent_candles: list[Candle]) -> int:
    """Verify all pending signals for a symbol/timeframe that have enough subsequent data.

    Returns the number of signals newly verified.
    """
    session_gen = get_session()
    session = next(session_gen)
    verified_count = 0

    try:
        pending_signals = (
            session.query(Signal)
            .outerjoin(SignalVerification, Signal.id == SignalVerification.signal_id)
            .filter(
                Signal.symbol == symbol,
                Signal.timeframe == timeframe,
                SignalVerification.id.is_(None),
            )
            .order_by(Signal.timestamp)
            .all()
        )

        for sig in pending_signals:
            # Get candles after signal timestamp
            sig_candles = [
                c for c in recent_candles
                if c.timestamp > sig.timestamp
            ]

            if len(sig_candles) < VERIFICATION_CONFIG.lookahead_candles:
                continue

            result = verify_signal(sig, sig_candles[:VERIFICATION_CONFIG.lookahead_candles])
            if result:
                store_verification(result, sig.id)
                verified_count += 1
                logger.info(
                    "Verified signal %s: %s profit=%.5f",
                    result.signal_id, result.result.value, result.profit_points or 0,
                )

        session.commit()
    except Exception:
        session.rollback()
        raise

    logger.info("Verified %d pending signals for %s %s", verified_count, symbol, timeframe)
    return verified_count