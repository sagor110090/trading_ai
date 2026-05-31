"""Signal scoring: weighted multi-factor confidence calculation.

Direction is determined by the EMA trend (the primary directional indicator).
If EMAs show a clear trend, that's the direction — even if RSI or MACD are neutral.
RSI and MACD contribute to confidence (strength) but shouldn't veto direction.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_ai.config.settings import SETTINGS, SignalType

logger = logging.getLogger(__name__)

WEIGHTS = SETTINGS.signal_weights
THRESHOLDS = SETTINGS.signal_thresholds


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


def _score_ema_trend(ema20: float, ema50: float, ema200: float, close: float) -> float:
    """Score EMA trend alignment. Returns 0 to WEIGHTS.ema_trend."""
    if pd.isna(ema20) or pd.isna(ema50) or pd.isna(ema200):
        return 0.0

    score = 0.0
    max_score = WEIGHTS.ema_trend

    # Full alignment
    if ema20 > ema50 > ema200 and close > ema20:
        score = max_score
    elif ema20 < ema50 < ema200 and close < ema20:
        score = max_score
    # Partial alignment
    elif ema20 > ema50 and close > ema20:
        score = max_score * 0.65
    elif ema20 < ema50 and close < ema20:
        score = max_score * 0.65
    # Weak alignment
    elif close > ema20:
        score = max_score * 0.3
    elif close < ema20:
        score = max_score * 0.3

    return score


def _score_rsi(rsi: float) -> float:
    """Score RSI factor. Returns 0 to WEIGHTS.rsi.

    Scores signal STRENGTH, not direction.
    Overbought/oversold = strong signal, neutral = weak signal.
    """
    if pd.isna(rsi):
        return 0.0

    max_score = WEIGHTS.rsi

    # Strong zones
    if rsi <= 20:
        return max_score
    elif rsi <= 30:
        return max_score * (30 - rsi) / 10.0
    elif rsi >= 80:
        return max_score
    elif rsi >= 70:
        return max_score * (rsi - 70) / 10.0
    # Neutral zone — still scores a little (signal exists, just not strong)
    return max_score * 0.1


def _score_macd(macd_hist: float, macd_line: float, macd_signal: float) -> float:
    """Score MACD factor. Returns 0 to WEIGHTS.macd."""
    if pd.isna(macd_hist):
        return 0.0

    max_score = WEIGHTS.macd
    score = 0.0

    # Histogram direction and magnitude
    hist_abs = abs(macd_hist)
    if hist_abs > 0:
        magnitude = min(hist_abs / (hist_abs + 1e-9), 1.0)
        score = max_score * 0.6 * magnitude

    # Crossover bonus
    if not pd.isna(macd_line) and not pd.isna(macd_signal):
        if abs(macd_line - macd_signal) < hist_abs * 0.3:
            score += max_score * 0.4

    return min(score, max_score)


def _score_atr(atr: float, close: float) -> float:
    """Score ATR factor — prefers moderate volatility. Returns 0 to WEIGHTS.atr."""
    if pd.isna(atr) or close == 0:
        return 0.0

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


def _score_structure(candle_body_pct: float, upper_wick_pct: float, lower_wick_pct: float) -> float:
    """Score candle structure factor. Returns 0 to WEIGHTS.structure."""
    max_score = WEIGHTS.structure

    body_abs = abs(candle_body_pct) if not pd.isna(candle_body_pct) else 0.0
    upper = upper_wick_pct if not pd.isna(upper_wick_pct) else 0.0
    lower = lower_wick_pct if not pd.isna(lower_wick_pct) else 0.0

    total = body_abs + upper + lower
    if total == 0:
        return 0.0

    body_dominance = body_abs / total
    return max_score * body_dominance


def _score_momentum(momentum_score: float, trend_strength: float) -> float:
    """Score momentum factor. Returns 0 to WEIGHTS.momentum."""
    if pd.isna(momentum_score) or pd.isna(trend_strength):
        return 0.0

    max_score = WEIGHTS.momentum

    alignment = abs(momentum_score + trend_strength) / 200.0
    magnitude = (abs(momentum_score) + abs(trend_strength)) / 200.0

    return max_score * alignment * magnitude


def _determine_ema_direction(ema20: float, ema50: float, ema200: float, close: float) -> SignalType:
    """Determine direction from EMA alignment.

    EMAs are the primary direction indicator. If price is above key EMAs → BUY.
    If price is below key EMAs → SELL. If EMAs are mixed → NO_TRADE.
    """
    if pd.isna(ema20) or pd.isna(ema50):
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

    # Weak: price vs EMA20 only
    if close > ema20:
        return SignalType.BUY
    if close < ema20:
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
) -> tuple[FactorScores, SignalType]:
    """Compute weighted factor scores and determine signal direction.

    Direction is primarily determined by EMA trend alignment.
    Confidence (total score) measures HOW STRONG the signal is.
    RSI and MACD affect confidence but don't override EMA direction.

    This means: if EMAs say BUY but RSI is neutral (40-60), we still get BUY
    with moderate confidence. The old logic would give NO_TRADE because
    neutral RSI voted for neither side.
    """
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

    # Compute all factor scores
    ema_score = _score_ema_trend(ema20, ema50, ema200, close)
    rsi_score = _score_rsi(rsi)
    macd_score = _score_macd(macd_histogram, macd_line, macd_signal)
    atr_score = _score_atr(atr, close)
    struct_score = _score_structure(candle_body_pct, upper_wick_pct, lower_wick_pct)
    mom_score = _score_momentum(momentum_score, trend_strength)

    total = ema_score + rsi_score + macd_score + atr_score + struct_score + mom_score

    # Adjust total based on direction confirmation from RSI and MACD
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