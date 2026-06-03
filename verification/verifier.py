"""Signal verification: monitors future candles to determine signal outcome."""

import logging
from dataclasses import dataclass
from typing import Optional

from trading_ai.config.settings import SETTINGS, SignalResult, SignalType, Timeframe
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
            # Stop loss hit
            if candle.low <= sl:
                mfe = max_favorable
                mae = entry - candle.low
                profit = sl - entry
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

            # Take profit hit
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

            favorable = candle.high - entry
            adverse = entry - candle.low
            max_favorable = max(max_favorable, favorable)
            max_adverse = max(max_adverse, adverse)

        elif direction == SignalType.SELL:
            # Stop loss hit
            if candle.high >= sl:
                mfe = max_favorable
                mae = candle.high - entry
                profit = entry - sl
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

            # Take profit hit
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

            favorable = entry - candle.low
            adverse = candle.high - entry
            max_favorable = max(max_favorable, favorable)
            max_adverse = max(max_adverse, adverse)

    # Exhausted lookahead candles without SL/TP hit
    if holding_candles >= VERIFICATION_CONFIG.lookahead_candles:
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

    return None


def verify_pending_signals(symbol: str) -> int:
    """Verify all pending signals for a symbol using M5 candles.

    Uses M5 (5-minute) candles for all signals regardless of mode, because
    M5 gives the finest granularity to check SL/TP hits accurately.

    Uses a single database session for all operations to ensure atomicity.

    Returns the number of signals newly verified.
    """
    with get_session() as session:
        verified_count = 0

        # Get all unverified signals for this symbol
        pending_signals = (
            session.query(Signal)
            .outerjoin(SignalVerification, Signal.id == SignalVerification.signal_id)
            .filter(
                Signal.symbol == symbol,
                SignalVerification.id.is_(None),
            )
            .order_by(Signal.timestamp)
            .all()
        )

        if not pending_signals:
            return 0

        # Get M5 candles for verification — finest granularity
        m5_candles = (
            session.query(Candle)
            .filter(
                Candle.symbol == symbol,
                Candle.timeframe == Timeframe.M5.value,
            )
            .order_by(Candle.timestamp)
            .all()
        )

        logger.info(
            "Verifying %d pending signals for %s using %d M5 candles",
            len(pending_signals), symbol, len(m5_candles),
        )

        for sig in pending_signals:
            # Get M5 candles after signal timestamp
            sig_candles = [
                c for c in m5_candles
                if c.timestamp > sig.timestamp
            ]

            # Need at least min_lookahead_candles to attempt verification.
            # If SL/TP is hit within fewer candles, we verify immediately.
            # Only need full lookahead_candles for the BREAKEVEN/expired case.
            if len(sig_candles) < VERIFICATION_CONFIG.min_lookahead_candles:
                logger.debug(
                    "Signal %s: only %d candles after signal (need min %d)",
                    sig.signal_id, len(sig_candles), VERIFICATION_CONFIG.min_lookahead_candles,
                )
                continue

            # Check up to min(available candles, lookahead_candles)
            candles_to_check = sig_candles[:VERIFICATION_CONFIG.lookahead_candles]
            result = verify_signal(sig, candles_to_check)
            if result:
                verification = SignalVerification(
                    signal_id=sig.id,
                    result=result.result,
                    profit_points=result.profit_points,
                    risk_reward=result.risk_reward,
                    holding_candles=result.holding_candles,
                    max_favorable_excursion=result.max_favorable_excursion,
                    max_adverse_excursion=result.max_adverse_excursion,
                )
                session.add(verification)
                verified_count += 1
                logger.info(
                    "Verified signal %s: %s profit=%.5f",
                    result.signal_id, result.result.value, result.profit_points or 0,
                )

    logger.info("Verified %d/%d pending signals for %s", verified_count, len(pending_signals), symbol)
    return verified_count