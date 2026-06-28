"""Signal scoring: weighted multi-factor confidence calculation.

Direction is determined by the market-bias/regime pipeline in
:mod:`signals.bias` (EMA alignment → allowed direction, with EMA200 + distance
filters). MACD **never** sets direction — it only contributes to confidence and
serves as an entry-timing confirmation.

This module computes the factor scores that become the confidence score, given
a direction. The direction itself is decided upstream (``bias.decide_direction``)
so that no momentum indicator can override the higher-timeframe trend.

Improvements over previous version:
- Direction comes from ``signals.bias`` (EMA alignment + EMA200/distance filters)
  instead of an in-module MACD fallback that could BUY a downtrend.
- ``compute_factor_scores`` factors the scoring out of the direction decision,
  so the same scoring math is reused whether direction came from bias or was
  passed in directly.
- RSI scoring is directional: confirms/contradicts instead of always positive.
- Trend strength gate: penalizes signals in weak/flat markets.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import SETTINGS, SignalType, SignalWeights

logger = logging.getLogger(__name__)

WEIGHTS = SETTINGS.signal_weights
THRESHOLDS = SETTINGS.signal_thresholds

# Minimum EMA separation to consider a trend valid for the *weak-trend penalty*
# on confidence. Direction/no-trade is now decided in ``signals.bias``; this
# constant only affects the confidence scaling of borderline trends.
FLAT_MARKET_THRESHOLD = 0.001  # 0.1% price separation = weak trend penalty band


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


def _features_namespace(
    ema20, ema50, ema200, close, rsi, macd_line, macd_signal, macd_histogram,
    atr, momentum_score, trend_strength, atr_ratio,
):
    """Build a lightweight attribute-bag for ``bias.decide_direction``.

    ``decide_direction`` reads attributes off the passed object, so we wrap the
    raw indicator values in a ``SimpleNamespace`` rather than constructing a full
    ``FeatureSet``. ``close`` is attached here so the EMA200/distance filters work
    (``FeatureSet`` itself has no ``close`` field).
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        ema20=ema20, ema50=ema50, ema200=ema200, close=close,
        rsi14=rsi, macd_line=macd_line, macd_signal=macd_signal,
        macd_histogram=macd_histogram, atr14=atr, atr_ratio=atr_ratio,
        momentum_score=momentum_score, trend_strength=trend_strength,
    )


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

    Direction is decided by the market-bias/regime pipeline in
    :mod:`signals.bias` (EMA alignment + EMA200/distance filters). MACD never
    sets direction here — it only feeds confidence. Confidence (total score)
    measures HOW STRONG the signal is; RSI is directional (confirms/contradicts).

    Args:
        weights: Per-symbol factor weights. If None, falls back to the global
            ``SETTINGS.signal_weights`` profile. This lets each instrument use a
            different strategy (e.g. trend-heavy for crypto, structure-heavy for
            gold) while sharing the same scoring engine.
    """
    w = weights if weights is not None else WEIGHTS

    # Direction comes from the bias/regime pipeline — never from MACD. The
    # MACD-fallback that used to BUY downtrends is gone.
    from signals.bias import decide_direction

    feats = _features_namespace(
        ema20, ema50, ema200, close, rsi, macd_line, macd_signal, macd_histogram,
        atr, momentum_score, trend_strength, getattr(weights, "atr_ratio", None),
    )
    direction, _log = decide_direction(feats, strict=True, close=close)

    if direction == SignalType.NO_TRADE:
        return FactorScores(0, 0, 0, 0, 0, 0, 0), SignalType.NO_TRADE

    return compute_factor_scores(
        direction, ema20, ema50, ema200, close, rsi, macd_line, macd_signal,
        macd_histogram, atr, candle_body_pct, upper_wick_pct, lower_wick_pct,
        momentum_score, trend_strength, weights=w,
    ), direction


def compute_factor_scores(
    direction: SignalType,
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
) -> FactorScores:
    """Compute the six factor scores for a *given* direction.

    This is the pure scoring half of the old ``compute_signal_scores`` — it does
    NOT decide direction (that is the caller's job, usually
    ``bias.decide_direction``). Keeping direction separate is what guarantees no
    momentum indicator can override the trend.
    """
    w = weights if weights is not None else WEIGHTS

    # Compute all factor scores (threading per-symbol weights through)
    ema_score = _score_ema_trend(ema20, ema50, ema200, close, max_score=w.ema_trend)
    rsi_score = _score_rsi(rsi, direction, max_score=w.rsi)  # directional
    macd_score = _score_macd(macd_histogram, macd_line, macd_signal, max_score=w.macd)
    atr_score = _score_atr(atr, close, max_score=w.atr)
    struct_score = _score_structure(candle_body_pct, upper_wick_pct, lower_wick_pct,
                                    max_score=w.structure)
    mom_score = _score_momentum(momentum_score, trend_strength, max_score=w.momentum)

    total = ema_score + rsi_score + macd_score + atr_score + struct_score + mom_score

    # Direction confirmation bonus from RSI and MACD.
    # These add/subtract from confidence, they don't veto direction.
    confirmation_bonus = 0.0

    if direction == SignalType.BUY:
        if not pd.isna(rsi) and rsi < 40:
            confirmation_bonus += 5.0
        if not pd.isna(macd_histogram) and macd_histogram > 0:
            confirmation_bonus += 5.0
        if not pd.isna(rsi) and rsi > 70:
            confirmation_bonus -= 8.0
        if not pd.isna(macd_histogram) and macd_histogram < 0:
            confirmation_bonus -= 5.0

    elif direction == SignalType.SELL:
        if not pd.isna(rsi) and rsi > 60:
            confirmation_bonus += 5.0
        if not pd.isna(macd_histogram) and macd_histogram < 0:
            confirmation_bonus += 5.0
        if not pd.isna(rsi) and rsi < 30:
            confirmation_bonus -= 8.0
        if not pd.isna(macd_histogram) and macd_histogram > 0:
            confirmation_bonus -= 5.0

    total = max(0.0, total + confirmation_bonus)

    # Weak-trend penalty: borderline EMA separation reduces confidence.
    if close > 0 and not pd.isna(ema20) and not pd.isna(ema50):
        ema_spread = abs(ema20 - ema50) / close
        if FLAT_MARKET_THRESHOLD <= ema_spread < FLAT_MARKET_THRESHOLD * 2:
            total *= 0.7
            logger.debug(
                "Weak trend penalty: ema_spread=%.4f (< %.4f), confidence × 0.7",
                ema_spread, FLAT_MARKET_THRESHOLD * 2,
            )

    return FactorScores(
        ema_trend=ema_score,
        rsi=rsi_score,
        macd=macd_score,
        atr=atr_score,
        structure=struct_score,
        momentum=mom_score,
        total=total,
    )