"""Signal scoring: weighted multi-factor confidence calculation.

Direction is determined by the EMA trend (the primary directional indicator).
If EMAs show a clear trend, that's the direction — even if RSI or MACD are neutral.
RSI and MACD contribute to confidence (strength) but shouldn't veto direction.

Improvements over previous version:
- Removed "weak alignment" tier (close > ema20 alone) that was too permissive
- Added flat/choppy market detection: NO_TRADE when EMAs are nearly flat
- RSI scoring is now directional: confirms/contradicts instead of always positive
- Trend strength gate: penalizes signals in weak/flat markets
- Partial alignment scores lower (15% vs 65%) to reduce low-quality signals
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import SETTINGS, SignalType, SignalWeights

logger = logging.getLogger(__name__)

WEIGHTS = SETTINGS.signal_weights
THRESHOLDS = SETTINGS.signal_thresholds

# Minimum EMA separation to consider a trend valid.
# If |ema20 - ema50| / close < this, the market is flat/choppy → NO_TRADE.
FLAT_MARKET_THRESHOLD = 0.001  # 0.1% price separation = flat market
VERY_FLAT_THRESHOLD = 0.0003   # 0.03% = extremely flat, hard NO_TRADE


@dataclass(frozen=True)
class FactorScores:
    """Individual factor contribution scores for a signal."""
    ema_trend: float
    rsi: float
    macd: float
    atr: float
    structure: float
    momentum: float
    total: float


def _score_ema_trend(ema20: float, ema50: float, ema200: float, close: float,
                     max_score: float | None = None) -> float:
    """Score EMA trend alignment. Returns 0 to weights.ema_trend.

    Two-tier scoring (weak alignment removed — too many false signals):
    - Strong: all 3 EMAs aligned + price above/below → full score
    - Partial: EMA20 > EMA50 + price confirms → 50% score
    - Flat market: EMAs too close together → 0 (NO_TRADE handled elsewhere)
    """
    if pd.isna(ema20) or pd.isna(ema50) or close == 0:
        return 0.0

    if max_score is None:
        max_score = WEIGHTS.ema_trend
    ema_spread = abs(ema20 - ema50) / close

    # Check for flat/choppy market: EMAs too close together
    if ema_spread < FLAT_MARKET_THRESHOLD:
        return 0.0

    # Strong alignment: all 3 EMAs agree + price confirms
    if not pd.isna(ema200):
        if ema20 > ema50 > ema200 and close > ema20:
            return max_score
        if ema20 < ema50 < ema200 and close < ema20:
            return max_score

    # Partial alignment: EMA20 and EMA50 agree + price confirms
    if ema20 > ema50 and close > ema20:
        return max_score * 0.5
    if ema20 < ema50 and close < ema20:
        return max_score * 0.5

    # No clear alignment → score 0 (direction will be NO_TRADE or MACD fallback)
    return 0.0


def _score_rsi(rsi: float, direction: SignalType, max_score: float | None = None) -> float:
    """Score RSI factor — DIRECTIONAL. Returns -weights.rsi to +weights.rsi.

    RSI now contributes positively when confirming direction and
    negatively when contradicting. Neutral zone (40-60) scores 0.
    This prevents RSI from inflating confidence in directionless markets.
    """
    if pd.isna(rsi):
        return 0.0

    if max_score is None:
        max_score = WEIGHTS.rsi

    if direction == SignalType.BUY:
        # BUY signals: low RSI is bullish (oversold = bounce potential)
        if rsi <= 25:
            return max_score  # Strongly oversold — great buy signal
        elif rsi <= 35:
            return max_score * 0.7  # Moderately oversold
        elif rsi <= 45:
            return max_score * 0.3  # Slightly favorable
        elif rsi <= 55:
            return 0.0  # Neutral — RSI adds nothing
        elif rsi <= 65:
            return -max_score * 0.3  # Slightly overbought — caution
        elif rsi <= 75:
            return -max_score * 0.5  # Overbought — risky buy
        else:
            return -max_score  # Extremely overbought — strong warning

    elif direction == SignalType.SELL:
        # SELL signals: high RSI is bearish (overbought = drop potential)
        if rsi >= 75:
            return max_score  # Strongly overbought — great sell signal
        elif rsi >= 65:
            return max_score * 0.7  # Moderately overbought
        elif rsi >= 55:
            return max_score * 0.3  # Slightly favorable
        elif rsi >= 45:
            return 0.0  # Neutral — RSI adds nothing
        elif rsi >= 35:
            return -max_score * 0.3  # Slightly oversold — caution
        elif rsi >= 25:
            return -max_score * 0.5  # Oversold — risky sell
        else:
            return -max_score  # Extremely oversold — strong warning

    # NO_TRADE direction — RSI doesn't contribute
    return 0.0


def _score_macd(macd_hist: float, macd_line: float, macd_signal: float,
                max_score: float | None = None) -> float:
    """Score MACD factor. Returns 0 to weights.macd."""
    if pd.isna(macd_hist):
        return 0.0

    if max_score is None:
        max_score = WEIGHTS.macd
    score = 0.0

    # Histogram magnitude (tanh squash preserves magnitude)
    hist_abs = abs(macd_hist)
    if hist_abs > 0:
        magnitude = np.tanh(hist_abs / 2.0)
        score = max_score * 0.6 * magnitude

    # Crossover bonus
    if not pd.isna(macd_line) and not pd.isna(macd_signal):
        if abs(macd_line - macd_signal) < hist_abs * 0.3:
            score += max_score * 0.4

    return min(score, max_score)


def _score_atr(atr: float, close: float, max_score: float | None = None) -> float:
    """Score ATR factor — prefers moderate volatility. Returns 0 to weights.atr."""
    if pd.isna(atr) or close == 0:
        return 0.0

    if max_score is None:
        max_score = WEIGHTS.atr
    atr_pct = (atr / close) * 100.0

    # Sweet spot: 0.05-0.5% ATR relative to price
    if 0.05 <= atr_pct <= 0.5:
        return max_score
    elif atr_pct < 0.05:
        return max_score * atr_pct / 0.05
    elif atr_pct <= 1.0:
        return max_score * (1.0 - (atr_pct - 0.5) / 0.5) * 0.7
    else:
        return max_score * 0.2


def _score_structure(candle_body_pct: float, upper_wick_pct: float, lower_wick_pct: float,
                     max_score: float | None = None) -> float:
    """Score candle structure factor. Returns 0 to weights.structure."""
    if max_score is None:
        max_score = WEIGHTS.structure

    body_abs = abs(candle_body_pct) if not pd.isna(candle_body_pct) else 0.0
    upper = upper_wick_pct if not pd.isna(upper_wick_pct) else 0.0
    lower = lower_wick_pct if not pd.isna(lower_wick_pct) else 0.0

    total = body_abs + upper + lower
    if total == 0:
        return 0.0

    body_dominance = body_abs / total
    return max_score * body_dominance


def _score_momentum(momentum_score: float, trend_strength: float,
                    max_score: float | None = None) -> float:
    """Score momentum factor. Returns 0 to weights.momentum."""
    if pd.isna(momentum_score) or pd.isna(trend_strength):
        return 0.0

    if max_score is None:
        max_score = WEIGHTS.momentum

    alignment = abs(momentum_score + trend_strength) / 200.0
    magnitude = (abs(momentum_score) + abs(trend_strength)) / 200.0

    return max_score * alignment * magnitude


def _is_flat_market(ema20: float, ema50: float, close: float) -> bool:
    """Check if EMAs are too flat to generate a reliable signal.

    Returns True if |ema20 - ema50| / close < FLAT_MARKET_THRESHOLD,
    meaning the market is in a tight range and trend direction is unreliable.
    """
    if pd.isna(ema20) or pd.isna(ema50) or close == 0:
        return False
    ema_spread = abs(ema20 - ema50) / close
    return ema_spread < FLAT_MARKET_THRESHOLD


def _determine_ema_direction(ema20: float, ema50: float, ema200: float, close: float) -> SignalType:
    """Determine direction from EMA alignment.

    Two-tier system (no weak alignment):
    - Strong: all 3 EMAs agree + price confirms → BUY/SELL
    - Partial: EMA20 > EMA50 + price confirms → BUY/SELL
    - Flat market: EMAs too close → NO_TRADE
    - Mixed/unclear: MACD fallback
    """
    if pd.isna(ema20) or pd.isna(ema50):
        return SignalType.NO_TRADE

    # Flat market check: EMAs too close together = no clear trend
    if close > 0 and abs(ema20 - ema50) / close < FLAT_MARKET_THRESHOLD:
        return SignalType.NO_TRADE

    # Strong alignment: all EMAs agree
    if not pd.isna(ema200):
        if ema20 > ema50 > ema200 and close > ema20:
            return SignalType.BUY
        if ema20 < ema50 < ema200 and close < ema20:
            return SignalType.SELL

    # Partial alignment: EMA20 and EMA50 agree
    if ema20 > ema50 and close > ema20:
        return SignalType.BUY
    if ema20 < ema50 and close < ema20:
        return SignalType.SELL

    return SignalType.NO_TRADE


def compute_signal_scores(
    ema20: float,
    ema50: float,
    ema200: float,
    close: float,
    rsi: float,
    macd_line: float,
    macd_signal: float,
    macd_histogram: float,
    atr: float,
    candle_body_pct: float,
    upper_wick_pct: float,
    lower_wick_pct: float,
    momentum_score: float,
    trend_strength: float,
    weights: SignalWeights | None = None,
) -> tuple[FactorScores, SignalType]:
    """Compute weighted factor scores and determine signal direction.

    Direction is primarily determined by EMA trend alignment.
    Confidence (total score) measures HOW STRONG the signal is.
    RSI is now directional — confirming RSI adds, contradicting RSI subtracts.
    Flat markets (EMA20 ≈ EMA50) produce NO_TRADE.

    Args:
        weights: Per-symbol factor weights. If None, falls back to the global
            ``SETTINGS.signal_weights`` profile. This lets each instrument use a
            different strategy (e.g. trend-heavy for crypto, structure-heavy for
            gold) while sharing the same scoring engine.
    """
    w = weights if weights is not None else WEIGHTS

    # Direction from EMAs (primary)
    direction = _determine_ema_direction(ema20, ema50, ema200, close)

    if direction == SignalType.NO_TRADE:
        # EMAs are flat/unclear — try MACD as fallback
        if not pd.isna(macd_histogram):
            if macd_histogram > 0 and not pd.isna(macd_line) and macd_line > macd_signal:
                direction = SignalType.BUY
            elif macd_histogram < 0 and not pd.isna(macd_line) and macd_line < macd_signal:
                direction = SignalType.SELL
            else:
                return FactorScores(0, 0, 0, 0, 0, 0, 0), SignalType.NO_TRADE
        else:
            return FactorScores(0, 0, 0, 0, 0, 0, 0), SignalType.NO_TRADE

    # Compute all factor scores (threading per-symbol weights through)
    ema_score = _score_ema_trend(ema20, ema50, ema200, close, max_score=w.ema_trend)
    rsi_score = _score_rsi(rsi, direction, max_score=w.rsi)  # Now directional
    macd_score = _score_macd(macd_histogram, macd_line, macd_signal, max_score=w.macd)
    atr_score = _score_atr(atr, close, max_score=w.atr)
    struct_score = _score_structure(candle_body_pct, upper_wick_pct, lower_wick_pct,
                                    max_score=w.structure)
    mom_score = _score_momentum(momentum_score, trend_strength, max_score=w.momentum)

    total = ema_score + rsi_score + macd_score + atr_score + struct_score + mom_score

    # Direction confirmation bonus from RSI and MACD
    # These add/subtract from confidence, they don't veto direction
    confirmation_bonus = 0.0

    if direction == SignalType.BUY:
        # RSI confirms buy (oversold → bouncing)
        if not pd.isna(rsi) and rsi < 40:
            confirmation_bonus += 5.0
        # MACD confirms buy
        if not pd.isna(macd_histogram) and macd_histogram > 0:
            confirmation_bonus += 5.0
        # RSI contradicts (overbought → risky buy)
        if not pd.isna(rsi) and rsi > 70:
            confirmation_bonus -= 8.0
        # MACD contradicts
        if not pd.isna(macd_histogram) and macd_histogram < 0:
            confirmation_bonus -= 5.0

    elif direction == SignalType.SELL:
        # RSI confirms sell (overbought → dropping)
        if not pd.isna(rsi) and rsi > 60:
            confirmation_bonus += 5.0
        # MACD confirms sell
        if not pd.isna(macd_histogram) and macd_histogram < 0:
            confirmation_bonus += 5.0
        # RSI contradicts (oversold → risky sell)
        if not pd.isna(rsi) and rsi < 30:
            confirmation_bonus -= 8.0
        # MACD contradicts
        if not pd.isna(macd_histogram) and macd_histogram > 0:
            confirmation_bonus -= 5.0

    total = max(0.0, total + confirmation_bonus)

    # Flat market penalty: if EMAs are close but not flat enough for NO_TRADE,
    # reduce confidence to reflect weak trend
    if close > 0 and not pd.isna(ema20) and not pd.isna(ema50):
        ema_spread = abs(ema20 - ema50) / close
        if FLAT_MARKET_THRESHOLD <= ema_spread < FLAT_MARKET_THRESHOLD * 2:
            # Weak trend: reduce confidence by 30%
            total *= 0.7
            logger.debug(
                "Weak trend penalty: ema_spread=%.4f (< %.4f), confidence × 0.7",
                ema_spread, FLAT_MARKET_THRESHOLD * 2,
            )

    scores = FactorScores(
        ema_trend=ema_score,
        rsi=rsi_score,
        macd=macd_score,
        atr=atr_score,
        structure=struct_score,
        momentum=mom_score,
        total=total,
    )

    return scores, direction