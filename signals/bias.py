"""Market trend / regime / direction-decision core.

This module enforces the decision hierarchy that prevents counter-trend trades:

    Market Bias (EMA alignment)  →  Allowed Direction
    Market Regime                →  loosens/tightens the filter
    EMA200 + distance filters    →  hard trend-only when far from EMA200
    Momentum (MACD)               →  confirms only, NEVER sets direction
                                      (except picking within a RANGE)

No lower layer may override a higher layer. In particular, MACD momentum can
never turn a bearish EMA stack into a BUY — that was the original bug
(``scoring.py`` MACD fallback). Direction comes from EMA alignment; pullbacks
above/below EMA20 are setup quality, not a trend flip.

All functions accept either individual indicator values (for unit tests) or a
``FeatureSet`` (via ``decide_direction``). Nothing here reads or mutates the
database, ML models, or broker — it is a pure decision function.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from config.settings import SETTINGS, SignalType

logger = logging.getLogger(__name__)

# ── Tunable constants (env-overridable to match the repo's settings pattern) ──
# EMA20/EMA50 separation (as a fraction of close) below which the market is
# treated as flat/ranging.
FLAT_SPREAD = float(os.getenv("BIAS_FLAT_SPREAD", "0.0015"))
# ATR ratio (current ATR / 14-bar avg ATR) below which volatility is compressed
# → market is ranging regardless of EMA alignment.
ATR_COMPRESSED = float(os.getenv("BIAS_ATR_COMPRESSED", "0.8"))
# When price is this many ATRs away from EMA200, force trend-only direction
# (no counter-trend trades at all). Step 6 of the refactor spec.
DISTANCE_ATR_TREND_ONLY = float(os.getenv("BIAS_DISTANCE_ATR_TREND_ONLY", "2.0"))


class MarketBias(Enum):
    """Allowed trading direction implied by the EMA trend stack."""
    BUY_BIAS = "BUY_BIAS"
    SELL_BIAS = "SELL_BIAS"
    RANGE = "RANGE"


class MarketRegime(Enum):
    """Coarse market classification used to relax the trend filter."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"


@dataclass
class DecisionLog:
    """Structured record of a direction decision, for the Step-12 debug log."""
    bias: MarketBias = MarketBias.RANGE
    regime: MarketRegime = MarketRegime.RANGING
    allowed: tuple[SignalType, ...] = ()
    close: Optional[float] = None
    ema200: Optional[float] = None
    distance_atr: Optional[float] = None
    macd: Optional[str] = None
    rsi: Optional[float] = None
    momentum: Optional[str] = None
    setup: Optional[str] = None
    reason: str = ""
    decision: SignalType = SignalType.NO_TRADE

    def format_accepted(self, confidence: Optional[float] = None) -> str:
        conf_line = f"\n  Confidence: {confidence:.1f}" if confidence is not None else ""
        return (
            f"\n  Market Bias: {self.bias.value}"
            f"\n  Regime: {self.regime.value}"
            f"\n  Allowed: {', '.join(d.value for d in self.allowed) or 'NONE'}"
            f"\n  Close: {self.close}  EMA200: {self.ema200}  Dist ATR: {self.distance_atr}"
            f"\n  MACD: {self.macd}  RSI: {self.rsi}  Momentum: {self.momentum}"
            f"\n  Setup: {self.setup}{conf_line}"
            f"\n  Decision: {self.decision.value}"
        )

    def format_rejected(self, attempted: SignalType) -> str:
        return (
            f"\n  Market Bias: {self.bias.value}"
            f"\n  Attempted Direction: {attempted.value}"
            f"\n  Reason Rejected: {self.reason}"
            f"\n  Close: {self.close}  EMA200: {self.ema200}  Dist ATR: {self.distance_atr}"
            f"\n  Decision: NO TRADE"
        )


# ── Helpers ──────────────────────────────────────────────────────────────────
def _valid(x) -> bool:
    return x is not None and not pd.isna(x)


def _bias_from_emas(ema20: float, ema50: float, ema200: Optional[float]) -> MarketBias:
    """Pure EMA-alignment bias. No price condition (Step 1)."""
    if not _valid(ema20) or not _valid(ema50):
        return MarketBias.RANGE
    if _valid(ema200):
        if ema20 > ema50 > ema200:
            return MarketBias.BUY_BIAS
        if ema20 < ema50 < ema200:
            return MarketBias.SELL_BIAS
    # 2-EMA fallback when EMA200 is missing or EMAs are not fully stacked
    if ema20 > ema50:
        return MarketBias.BUY_BIAS
    if ema20 < ema50:
        return MarketBias.SELL_BIAS
    return MarketBias.RANGE


def get_market_bias(ema20: float, ema50: float, ema200: Optional[float]) -> MarketBias:
    """Market bias from EMA alignment only. Price location does NOT change bias."""
    return _bias_from_emas(ema20, ema50, ema200)


def detect_regime(
    ema20: float,
    ema50: float,
    ema200: Optional[float],
    close: float,
    atr: Optional[float],
    atr_ratio: Optional[float],
) -> MarketRegime:
    """Classify the market as trending up/down or ranging (Step 10).

    Uses EMA alignment + EMA spacing + ATR compression (no ADX — schema frozen).
    RANGING when EMAs are nearly flat OR volatility is compressed.
    """
    if not _valid(close) or close == 0 or not _valid(ema20) or not _valid(ema50):
        return MarketRegime.RANGING

    ema_spread = abs(ema20 - ema50) / close
    compressed = _valid(atr_ratio) and atr_ratio < ATR_COMPRESSED
    if ema_spread < FLAT_SPREAD or compressed:
        return MarketRegime.RANGING

    bias = _bias_from_emas(ema20, ema50, ema200)
    if bias == MarketBias.BUY_BIAS:
        return MarketRegime.TRENDING_UP
    if bias == MarketBias.SELL_BIAS:
        return MarketRegime.TRENDING_DOWN
    return MarketRegime.RANGING


def allowed_directions(
    bias: MarketBias,
    regime: MarketRegime,
    close: float,
    ema200: Optional[float],
    atr: Optional[float],
) -> tuple[set[SignalType], Optional[float]]:
    """Resolve the set of directions the bias permits (Steps 3, 5, 6).

    Returns (allowed_set, distance_atr). The EMA200 and distance filters are
    skipped when the regime is RANGING (counter-trend allowed in ranges).
    """
    if bias == MarketBias.BUY_BIAS:
        allowed: set[SignalType] = {SignalType.BUY}
    elif bias == MarketBias.SELL_BIAS:
        allowed = {SignalType.SELL}
    else:
        allowed = {SignalType.BUY, SignalType.SELL}

    distance_atr: Optional[float] = None

    # EMA200 trend filter (Step 5) — relaxed when RANGING (counter-trend allowed
    # in a range). A genuine uptrend does not forbid SELLs in a choppy market.
    if regime != MarketRegime.RANGING and _valid(close) and _valid(ema200):
        if close < ema200:
            allowed.discard(SignalType.BUY)
        elif close > ema200:
            allowed.discard(SignalType.SELL)

    # Distance filter (Step 6) — HARD trend-only gate. Applies even in RANGING:
    # price 2+ ATR from EMA200 is a real move, not noise, so counter-trend is
    # disallowed regardless of short-term EMA flatness.
    if _valid(close) and _valid(ema200) and _valid(atr) and atr > 0:
        distance_atr = abs(close - ema200) / atr
        if distance_atr >= DISTANCE_ATR_TREND_ONLY:
            if close < ema200:
                allowed = {SignalType.SELL}
            else:
                allowed = {SignalType.BUY}
    return allowed, distance_atr


def momentum_pick(
    macd_histogram: Optional[float],
    macd_line: Optional[float],
    macd_signal: Optional[float],
) -> SignalType:
    """Pick a direction from MACD momentum — used ONLY when RANGE leaves both.

    This never overrides a trend; it only breaks a tie in a ranging market.
    """
    if not _valid(macd_histogram):
        return SignalType.NO_TRADE
    if macd_histogram > 0 and _valid(macd_line) and _valid(macd_signal) and macd_line > macd_signal:
        return SignalType.BUY
    if macd_histogram < 0 and _valid(macd_line) and _valid(macd_signal) and macd_line < macd_signal:
        return SignalType.SELL
    return SignalType.NO_TRADE


def classify_setup(
    close: float,
    ema20: float,
    atr: Optional[float],
    direction: SignalType,
) -> str:
    """Setup quality from price location vs EMA20 (Step 2). Never changes direction.

    - ``pullback``: price has pulled back against the trend toward EMA20 (good entry)
    - ``extension``: price is extended far in the trend direction (late/chasing)
    - ``neutral``: anything else
    """
    if not _valid(close) or not _valid(ema20):
        return "neutral"
    if direction == SignalType.BUY:
        if close < ema20:
            return "pullback"
        if _valid(atr) and atr > 0 and (close - ema20) / atr >= 1.5:
            return "extension"
        return "neutral"
    if direction == SignalType.SELL:
        if close > ema20:
            return "pullback"
        if _valid(atr) and atr > 0 and (ema20 - close) / atr >= 1.5:
            return "extension"
        return "neutral"
    return "neutral"


def momentum_confirms(
    macd_histogram: Optional[float],
    macd_line: Optional[float],
    macd_signal: Optional[float],
    direction: SignalType,
) -> bool:
    """Entry-timing check: does MACD momentum agree with the trade direction?

    Confirmation only — never a veto of direction. Used as a confidence modifier
    and an M5 entry-timing gate in the engine.
    """
    if not _valid(macd_histogram) or not _valid(macd_line) or not _valid(macd_signal):
        return False
    if direction == SignalType.BUY:
        return macd_histogram > 0 and macd_line > macd_signal
    if direction == SignalType.SELL:
        return macd_histogram < 0 and macd_line < macd_signal
    return False


def _macd_label(macd_histogram: Optional[float]) -> Optional[str]:
    if not _valid(macd_histogram):
        return None
    if macd_histogram > 0:
        return "bullish"
    if macd_histogram < 0:
        return "bearish"
    return "flat"


# ── Orchestration ─────────────────────────────────────────────────────────────
def decide_direction(
    features,
    strict: bool = True,
    close: Optional[float] = None,
) -> tuple[SignalType, DecisionLog]:
    """Run the full Step 1-6 decision pipeline on a ``FeatureSet``.

    Args:
        features: A ``FeatureSet`` (indicators/features.py) with ema20/50/200,
            rsi14, macd_*, atr14, atr_ratio. Accepts any object with those
            attributes; tests may pass a SimpleNamespace.
        close: The latest candle close. ``FeatureSet`` has no ``close`` field,
            so the engine passes it explicitly. Falls back to
            ``ema200 + dist_ema200`` then ``ema20`` when not provided.
        strict: When True (H4 primary), apply the EMA200 + distance filters.
            When False (H1 confirmation / fallback), return the raw bias direction
            without the trend filters so a higher timeframe can do the filtering.

    Returns (direction, DecisionLog). direction is NO_TRADE when nothing is
    allowed or (in RANGE) MACD cannot break the tie.
    """
    ema20 = getattr(features, "ema20", None)
    ema50 = getattr(features, "ema50", None)
    ema200 = getattr(features, "ema200", None)
    rsi = getattr(features, "rsi14", None)
    macd_line = getattr(features, "macd_line", None)
    macd_signal = getattr(features, "macd_signal", None)
    macd_histogram = getattr(features, "macd_histogram", None)
    atr = getattr(features, "atr14", None)
    atr_ratio = getattr(features, "atr_ratio", None)
    dist_ema200 = getattr(features, "dist_ema200", None)
    # FeatureSet has no `close` field; reconstruct from ema200 + dist_ema200, then ema20.
    if not _valid(close) and _valid(ema200) and _valid(dist_ema200):
        close = ema200 + dist_ema200
    if not _valid(close):
        close = ema20

    bias = get_market_bias(ema20, ema50, ema200)
    regime = detect_regime(ema20, ema50, ema200, close, atr, atr_ratio)

    log = DecisionLog(
        bias=bias,
        regime=regime,
        close=close if _valid(close) else None,
        ema200=ema200 if _valid(ema200) else None,
        macd=_macd_label(macd_histogram),
        rsi=rsi if _valid(rsi) else None,
    )

    if not strict:
        # Raw bias direction for confirmation/fallback. RANGE → MACD pick.
        if bias == MarketBias.BUY_BIAS:
            log.decision = SignalType.BUY
        elif bias == MarketBias.SELL_BIAS:
            log.decision = SignalType.SELL
        else:
            log.decision = momentum_pick(macd_histogram, macd_line, macd_signal)
            log.reason = "RANGE: MACD momentum pick"
        return log.decision, log

    allowed, distance_atr = allowed_directions(bias, regime, close, ema200, atr)
    log.allowed = tuple(sorted(allowed, key=lambda d: d.value))
    log.distance_atr = round(distance_atr, 2) if distance_atr is not None else None

    if not allowed:
        log.reason = "No direction allowed by trend/regime filters"
        log.decision = SignalType.NO_TRADE
        return SignalType.NO_TRADE, log

    if len(allowed) == 1:
        decision = next(iter(allowed))
    else:
        # RANGE: MACD may choose either direction
        decision = momentum_pick(macd_histogram, macd_line, macd_signal)
        log.reason = "RANGE: MACD momentum pick"

    log.decision = decision
    log.momentum = "confirms" if momentum_confirms(
        macd_histogram, macd_line, macd_signal, decision) else "no-confirmation"
    return decision, log