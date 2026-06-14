"""Multi-timeframe confluence signal engine with 3 trade modes.

Scalping:  M5 entry + H1 trend only. Tight stops. Fast in/out.
Day trade: M5 entry + H1 trend + M15 confirmation (resampled from M5). Medium stops. Close same day.
Swing:     H4+H1 must agree + M5 trigger. Wide stops. Hold hours-days.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config.settings import (
    SETTINGS, SignalType, Symbol, Timeframe, TradeMode,
)
from indicators.features import FeatureSet, compute_all_features, extract_feature_set_with_lags
from signals.scoring import FactorScores, compute_signal_scores

logger = logging.getLogger(__name__)

WEIGHTS = SETTINGS.signal_weights
THRESHOLDS = SETTINGS.signal_thresholds

# Spread filter thresholds.
# If spread/close > HIGH_SPREAD_SKIP (0.5%), skip the signal entirely —
# entry cost is too high for a profitable trade.
# If spread/close > MODERATE_SPREAD_PENALTY (0.1%), reduce confidence by
# SPREAD_PENALTY_POINTS — spread is eating into profit but trade may still work.
HIGH_SPREAD_SKIP = 0.005     # 0.5% of price → skip signal
MODERATE_SPREAD_PENALTY = 0.001  # 0.1% of price → reduce confidence
SPREAD_PENALTY_POINTS = 15.0     # points to subtract when spread is moderate


@dataclass(frozen=True)
class TradeSignal:
    """Complete trade signal with entry, stop loss, take profit."""
    signal_id: str
    symbol: str
    timeframe: str
    signal: SignalType
    confidence: float
    entry: float
    stop_loss: float
    take_profit: float
    timestamp: pd.Timestamp
    features: FeatureSet
    factor_scores: FactorScores
    h4_direction: str
    h1_direction: str
    m5_direction: str
    confluence: str
    trade_mode: str


def _get_direction_from_df(df: pd.DataFrame) -> tuple[SignalType, FactorScores, FeatureSet, float]:
    """Compute direction, scores, and features from the latest candle."""
    df_features = compute_all_features(df)
    idx = len(df_features) - 1
    row = df_features.iloc[idx]
    features = extract_feature_set_with_lags(df_features, idx)

    if pd.isna(row.get("ema200")):
        return SignalType.NO_TRADE, FactorScores(0, 0, 0, 0, 0, 0, 0), features, row["close"]

    scores, direction = compute_signal_scores(
        ema20=features.ema20 or 0,
        ema50=features.ema50 or 0,
        ema200=features.ema200 or 0,
        close=row["close"],
        rsi=features.rsi14 or 50,
        macd_line=features.macd_line or 0,
        macd_signal=features.macd_signal or 0,
        macd_histogram=features.macd_histogram or 0,
        atr=features.atr14 or 0,
        candle_body_pct=features.candle_body_pct or 0,
        upper_wick_pct=features.upper_wick_pct or 0,
        lower_wick_pct=features.lower_wick_pct or 0,
        momentum_score=features.momentum_score or 0,
        trend_strength=features.trend_strength or 0,
    )

    return direction, scores, features, row["close"]


def _pips_to_price(pips: float, symbol: Symbol, entry: float) -> float:
    """Convert pips to price value based on the symbol.

    Different instruments have different pip values:
    - Forex (non-JPY): 1 pip = 0.0001
    - JPY pairs: 1 pip = 0.01
    - Gold (XAUUSD), Silver (XAGUSD): 1 pip = 0.01
    - Indices (US30, US500, NAS100): 1 pip = 0.1 or 1.0
    - Crypto (BTCUSD, ETHUSD): varies by price
    """
    # JPY pairs
    if symbol in [Symbol.USDJPY, Symbol.EURJPY, Symbol.GBPJPY]:
        return pips * 0.01

    # Metals (Gold, Silver)
    if symbol in [Symbol.XAUUSD, Symbol.XAGUSD]:
        return pips * 0.01

    # Indices
    if symbol in [Symbol.US30, Symbol.US500, Symbol.NAS100]:
        return pips * 0.1

    # Crypto - use percentage of current price
    if symbol in [Symbol.BTCUSD, Symbol.ETHUSD, Symbol.LTCUSD,
                  Symbol.BCHUSD, Symbol.XRPUSD, Symbol.BTCEUR, Symbol.ETHGBP]:
        return pips * entry * 0.0001

    # Default for forex majors (EURUSD, GBPUSD, etc.)
    return pips * 0.0001


def _compute_stop_take(
    signal: SignalType,
    entry: float,
    atr: Optional[float],
    symbol: Symbol,
    sl_mult: float,
    tp_mult: float,
    atr_ratio: Optional[float] = None,
    symbol_name: str = "",
) -> tuple[float, float]:
    """Calculate stop loss and take profit based on ATR multipliers.

    Applies two safety layers in fast markets:
    1. Fast-market reduction: when ATR ratio >= threshold, SL/TP multipliers
       are scaled down by SETTINGS.fast_market.sl_tp_factor (default 0.5).
    2. Max distance cap: SL/TP distances are clamped to per-symbol max values
       if configured (e.g., MAX_SL_DISTANCE_XAUUSD=30).

    Automatically adds broker spread buffer to SL to account for execution latency.
    """
    # Fast-market reduction: shrink SL/TP when ATR ratio exceeds threshold
    fast_applied = False
    if atr_ratio is not None and atr_ratio >= SETTINGS.fast_market.atr_threshold:
        original_sl, original_tp = sl_mult, tp_mult
        sl_mult *= SETTINGS.fast_market.sl_tp_factor
        tp_mult *= SETTINGS.fast_market.sl_tp_factor
        fast_applied = True
        logger.info(
            "Fast market SL/TP reduction for %s: ATR ratio %.2f >= %.2f, "
            "SL mult %.2f -> %.2f, TP mult %.2f -> %.2f",
            symbol_name or symbol.value, atr_ratio, SETTINGS.fast_market.atr_threshold,
            original_sl, sl_mult, original_tp, tp_mult,
        )

    if atr is None or atr <= 0 or pd.isna(atr):
        if symbol == Symbol.XAUUSD:
            sl_distance = entry * 0.0015
            tp_distance = entry * 0.0025
        else:
            sl_distance = entry * 0.02
            tp_distance = entry * 0.03
    else:
        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult

    # Add broker spread buffer to SL (makes SL wider to avoid premature stop-outs)
    broker_buffer = _pips_to_price(SETTINGS.broker_spread_pips, symbol, entry)
    sl_distance += broker_buffer

    # Cap SL/TP distances to per-symbol maximums (prevents runaway risk in volatile markets)
    max_sl, max_tp = SETTINGS.get_max_sl_tp(symbol_name or symbol.value)
    if max_sl is not None and sl_distance > max_sl:
        logger.info(
            "SL distance capped for %s: %.5f -> %.5f (max %.5f)",
            symbol_name or symbol.value, sl_distance, max_sl, max_sl,
        )
        sl_distance = max_sl
    if max_tp is not None and tp_distance > max_tp:
        logger.info(
            "TP distance capped for %s: %.5f -> %.5f (max %.5f)",
            symbol_name or symbol.value, tp_distance, max_tp, max_tp,
        )
        tp_distance = max_tp

    if signal == SignalType.BUY:
        stop_loss = entry - sl_distance
        take_profit = entry + tp_distance
    elif signal == SignalType.SELL:
        stop_loss = entry + sl_distance
        take_profit = entry - tp_distance
    else:
        stop_loss = entry
        take_profit = entry

    logger.debug(
        "SL/TP for %s: entry=%.5f SL=%.5f TP=%.5f (broker buffer=%.5f pips=%.2f%s)",
        symbol.value, entry, stop_loss, take_profit, broker_buffer, SETTINGS.broker_spread_pips,
        " [FAST-MKT]" if fast_applied else "",
    )

    return round(stop_loss, 5), round(take_profit, 5)


def _get_m5_direction(df_m5: pd.DataFrame) -> tuple[SignalType, float, float]:
    """Full scoring-based M5 direction check.

    Uses the same scoring engine as H1/H4 but with a minimum confidence gate.
    Returns (direction, confidence_pct, momentum_score).

    A minimum confidence of 15% is required to confirm M5 direction.
    This prevents choppy M5 candles from overriding H1/H4 trends.
    """
    if df_m5.empty or len(df_m5) < 50:
        return SignalType.NO_TRADE, 0.0, 0.0

    m5_dir, m5_scores, m5_feat, _ = _get_direction_from_df(df_m5)
    m5_momentum = float(m5_feat.momentum_score or 0)

    if m5_dir == SignalType.NO_TRADE:
        return SignalType.NO_TRADE, 0.0, m5_momentum

    # Minimum confidence gate: M5 must have meaningful direction strength
    m5_confidence = min(m5_scores.total / WEIGHTS.total * 100.0, 100.0)
    if m5_confidence < 15.0:
        logger.debug(
            "M5 direction %s too weak (%.1f%% < 15%%) — treating as NO_TRADE",
            m5_dir.value, m5_confidence,
        )
        return SignalType.NO_TRADE, m5_confidence, m5_momentum

    return m5_dir, m5_confidence, m5_momentum


def _apply_spread_filter(
    confidence: float,
    features: FeatureSet,
    symbol: Symbol,
) -> tuple[float, bool]:
    """Apply spread filter to reduce or skip signals with wide spreads.

    Returns (adjusted_confidence, should_skip).
    If should_skip is True, the signal should be discarded entirely.
    If should_skip is False, confidence may still be reduced for moderate spreads.
    """
    spread = features.spread
    # For the entry price, use a rough estimate from close price.
    # We need a price reference — use the feature set's EMA20 as proxy.
    close_proxy = features.ema20 or 0

    if not spread or spread <= 0 or close_proxy <= 0:
        # No spread data available — don't penalize
        return confidence, False

    spread_pct = spread / close_proxy

    if spread_pct > HIGH_SPREAD_SKIP:
        # Spread is too wide — skip this signal entirely
        logger.info(
            "Spread filter: %s spread too wide (%.4f > %.4f) — skipping signal",
            symbol.value, spread_pct, HIGH_SPREAD_SKIP,
        )
        return confidence, True

    if spread_pct > MODERATE_SPREAD_PENALTY:
        # Moderate spread — reduce confidence
        adjusted = confidence - SPREAD_PENALTY_POINTS
        logger.info(
            "Spread filter: %s spread moderate (%.4f > %.4f) — confidence %.1f -> %.1f",
            symbol.value, spread_pct, MODERATE_SPREAD_PENALTY, confidence, adjusted,
        )
        return max(0.0, adjusted), False

    return confidence, False


# ── SCALPING ──────────────────────────────────────────────────────────────────

def generate_signal_scalping(
    df_h1: pd.DataFrame,
    df_m5: pd.DataFrame,
    symbol: Symbol,
    live_price: Optional[float] = None,
    prefix: str = "",
) -> Optional[TradeSignal]:
    """Scalping signal: M5 entry in direction of H1 trend.

    Rules:
    - H1 provides trend direction (the only boss)
    - M5 momentum must agree for entry timing
    - Tight stops: 0.5x ATR SL, 0.8x ATR TP
    - Lower threshold (30%) — more signals, quick in/out

    Args:
        live_price: Current market price. If provided, used as entry instead
            of the last H1 candle close, so SL/TP reflect live conditions.
    """
    cfg = SETTINGS.scalping

    if df_h1.empty or len(df_h1) < 200:
        logger.info("Scalping: insufficient H1 data (%d)", len(df_h1))
        return None

    h1_dir, h1_scores, h1_feat, h1_close = _get_direction_from_df(df_h1)
    m5_dir, m5_conf, m5_momentum = _get_m5_direction(df_m5)

    logger.info("Scalping: H1=%s M5=%s (M5 conf=%.1f%% mom=%.1f)", h1_dir.value, m5_dir.value, m5_conf, m5_momentum)

    # H1 must have a direction
    if h1_dir == SignalType.NO_TRADE:
        logger.info("Scalping: H1 NO_TRADE — no trend")
        return None

    # M5 must agree (or at least not contradict)
    if m5_dir != SignalType.NO_TRADE and m5_dir != h1_dir:
        logger.info("Scalping: M5 opposes H1 — blocked")
        return None

    # M5 agreement gives bonus, disagreement gives penalty
    base_confidence = min(h1_scores.total / WEIGHTS.total * 100.0, 100.0)
    confluence_bonus = 0.0
    confluence_label = cfg.confluence_label

    if m5_dir == h1_dir:
        confluence_bonus += 10.0
        confluence_label = f"{cfg.confluence_label}+m5"
    elif m5_dir == SignalType.NO_TRADE:
        confluence_bonus += 3.0  # M5 neutral is OK for scalping

    confidence = max(0.0, min(100.0, base_confidence + confluence_bonus))

    # Spread filter: reduce confidence or skip signal for wide spreads
    confidence, skip = _apply_spread_filter(confidence, h1_feat, symbol)
    if skip:
        logger.info("Scalping: skipped — spread too wide")
        return None

    if confidence < cfg.confidence_threshold:
        logger.info("Scalping: confidence %.1f < threshold %.1f", confidence, cfg.confidence_threshold)
        return None

    entry = round(live_price if live_price is not None else h1_close, 5)
    # Use per-symbol SL/TP multipliers if set, otherwise use trade mode defaults
    sl_mult, tp_mult = SETTINGS.get_sl_tp_multipliers(symbol.value)
    stop_loss, take_profit = _compute_stop_take(
        h1_dir, entry, h1_feat.atr14, symbol, sl_mult, tp_mult,
        atr_ratio=h1_feat.atr_ratio, symbol_name=symbol.value,
    )

    return TradeSignal(
        signal_id=f"{prefix}{uuid.uuid4().hex[:16]}",
        symbol=symbol.value,
        timeframe=cfg.entry_timeframe,
        signal=h1_dir,
        confidence=round(confidence, 2),
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        timestamp=df_h1.iloc[-1]["timestamp"],
        features=h1_feat,
        factor_scores=h1_scores,
        h4_direction="n/a",
        h1_direction=h1_dir.value,
        m5_direction=m5_dir.value,
        confluence=confluence_label,
        trade_mode="scalping",
    )


# ── DAY TRADING ───────────────────────────────────────────────────────────────

def _resample_m5_to_m15(df_m5: pd.DataFrame) -> pd.DataFrame:
    """Resample M5 candle data to M15 for smoother confirmation.

    Groups every 3 consecutive M5 candles into one M15 candle.
    Requires at least 3 rows and a 'timestamp' column.
    Returns empty DataFrame if resampling fails.
    """
    if df_m5.empty or len(df_m5) < 3:
        return pd.DataFrame()

    try:
        df = df_m5.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        # Only include volume/spread if present
        if "volume" in df.columns:
            agg_dict["volume"] = "sum"
        if "spread" in df.columns:
            agg_dict["spread"] = "mean"

        m15 = df.resample("15min").agg(agg_dict)
        # Drop rows where OHLC is missing (incomplete 15-min windows)
        m15 = m15.dropna(subset=["open", "high", "low", "close"])

        m15 = m15.reset_index()
        return m15
    except Exception as e:
        logger.warning("Failed to resample M5 to M15: %s", e)
        return pd.DataFrame()


def generate_signal_day_trading(
    df_h4: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_m5: pd.DataFrame,
    symbol: Symbol,
    live_price: Optional[float] = None,
    prefix: str = "",
) -> Optional[TradeSignal]:
    """Day trading signal: M5 entry in H1 trend direction, M15 confirmation.

    Timeframes: H1 = trend direction, M15 = confirmation, M5 = entry timing.
    M15 is resampled from M5 data (no separate candle fetch needed).

    Rules:
    - H1 provides trend direction (the boss)
    - M15 opposition penalises confidence (smoother than M5, avoids noise)
    - M5 momentum must agree for entry timing
    - Medium stops: per-symbol SL/TP multipliers
    - Threshold: 30%
    """
    cfg = SETTINGS.day_trading

    if df_h1.empty or len(df_h1) < 200:
        logger.info("Day trade: insufficient H1 data (%d)", len(df_h1))
        return None

    h1_dir, h1_scores, h1_feat, h1_close = _get_direction_from_df(df_h1)
    m5_dir, m5_conf, m5_momentum = _get_m5_direction(df_m5)

    # Resample M5 → M15 for confirmation (smoother than raw M5)
    df_m15 = _resample_m5_to_m15(df_m5)
    m15_dir = SignalType.NO_TRADE
    m15_scores = None
    if not df_m15.empty and len(df_m15) >= 50:
        m15_dir, m15_scores, _, _ = _get_direction_from_df(df_m15)

    logger.info(
        "Day trade: M15=%s H1=%s M5=%s (M5 conf=%.1f%%)",
        m15_dir.value, h1_dir.value, m5_dir.value, m5_conf,
    )

    # H1 must have a direction
    if h1_dir == SignalType.NO_TRADE:
        logger.info("Day trade: H1 NO_TRADE")
        return None

    # M15 confirmation penalty/bonus (replaces H4 role)
    # M15 is smoother than M5, so it provides a more reliable mid-term direction
    m15_penalty = 0.0
    m15_opposes = False
    if m15_dir != SignalType.NO_TRADE and m15_dir != h1_dir:
        m15_penalty = -10.0
        m15_opposes = True
        logger.info("Day trade: M15=%s opposes H1=%s — penalty -10%%", m15_dir.value, h1_dir.value)

    # M5 must agree or be neutral
    if m5_dir != SignalType.NO_TRADE and m5_dir != h1_dir:
        m5_penalty = -8.0
    else:
        m5_penalty = 0.0

    base_confidence = min(h1_scores.total / WEIGHTS.total * 100.0, 100.0)
    confluence_bonus = 0.0
    confluence_label = cfg.confluence_label

    # M15 alignment bonus or opposition label
    if m15_dir == h1_dir and m15_dir != SignalType.NO_TRADE:
        confluence_bonus += 8.0
        confluence_label = f"{cfg.confluence_label}+m15"
    elif m15_opposes:
        confluence_label = f"{cfg.confluence_label}_vs_m15"

    # M5 alignment bonus
    if m5_dir == h1_dir:
        confluence_bonus += 6.0
        confluence_label = f"{confluence_label}+m5"

    confidence = max(0.0, min(100.0, base_confidence + confluence_bonus + m15_penalty + m5_penalty))

    # Spread filter: reduce confidence or skip signal for wide spreads
    confidence, skip = _apply_spread_filter(confidence, h1_feat, symbol)
    if skip:
        logger.info("Day trade: skipped — spread too wide")
        return None

    if confidence < cfg.confidence_threshold:
        logger.info("Day trade: confidence %.1f < threshold %.1f", confidence, cfg.confidence_threshold)
        return None

    entry = round(live_price if live_price is not None else h1_close, 5)
    # Use per-symbol SL/TP multipliers if set, otherwise use trade mode defaults
    sl_mult, tp_mult = SETTINGS.get_sl_tp_multipliers(symbol.value)
    stop_loss, take_profit = _compute_stop_take(
        h1_dir, entry, h1_feat.atr14, symbol, sl_mult, tp_mult,
        atr_ratio=h1_feat.atr_ratio, symbol_name=symbol.value,
    )

    return TradeSignal(
        signal_id=f"{prefix}{uuid.uuid4().hex[:16]}",
        symbol=symbol.value,
        timeframe=cfg.entry_timeframe,
        signal=h1_dir,
        confidence=round(confidence, 2),
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        timestamp=df_h1.iloc[-1]["timestamp"],
        features=h1_feat,
        factor_scores=h1_scores,
        h4_direction=m15_dir.value,  # M15 confirmation stored in h4_direction field
        h1_direction=h1_dir.value,
        m5_direction=m5_dir.value,
        confluence=confluence_label,
        trade_mode="day_trading",
    )


# ── SWING ────────────────────────────────────────────────────────────────────

def generate_signal_swing(
    df_h4: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_m5: pd.DataFrame,
    symbol: Symbol,
    live_price: Optional[float] = None,
    prefix: str = "",
) -> Optional[TradeSignal]:
    """Swing signal: H4+H1 alignment, M5 entry trigger.

    Rules:
    - H4 = trend direction (the boss)
    - H1 opposition penalises confidence (heavier penalty than day trade)
    - M5 supports entry timing
    - Wide stops: 2x ATR SL, 3x ATR TP
    - Threshold: 25%
    """
    cfg = SETTINGS.swing

    if df_h4.empty or len(df_h4) < 200 or df_h1.empty or len(df_h1) < 200:
        logger.info("Swing: insufficient data (H4=%d, H1=%d)", len(df_h4), len(df_h1))
        return None

    h4_dir, h4_scores, h4_feat, _ = _get_direction_from_df(df_h4)
    h1_dir, h1_scores, h1_feat, h1_close = _get_direction_from_df(df_h1)
    m5_dir, m5_conf, m5_momentum = _get_m5_direction(df_m5)

    logger.info("Swing: H4=%s H1=%s M5=%s (M5 conf=%.1f%%)", h4_dir.value, h1_dir.value, m5_dir.value, m5_conf)

    # H4 and H1 must both have a direction
    if h4_dir == SignalType.NO_TRADE or h1_dir == SignalType.NO_TRADE:
        logger.info("Swing: H4 or H1 is NO_TRADE")
        return None

    # H4 vs H1 conflict: heavy penalty instead of hard block.
    # Swing trades need more alignment, so the penalty is larger than day trade.
    # When they agree, H4 direction is used; when H1 opposes, H1 direction is used
    # (H1 is the entry timeframe — more relevant for timing).
    h4_h1_penalty = 0.0
    if h4_dir != h1_dir:
        h4_h1_penalty = -18.0
        logger.info("Swing: H4=%s vs H1=%s — penalty -18%%", h4_dir.value, h1_dir.value)

    signal_dir = h1_dir  # Use H1 direction (entry timing matters more)

    # Confidence calculation
    base_confidence = min(h1_scores.total / WEIGHTS.total * 100.0, 100.0)
    confluence_bonus = 0.0
    confluence_label = f"{cfg.confluence_label}"

    # H4 alignment label
    if h4_dir == signal_dir:
        confluence_label = f"{confluence_label}+h4"
    elif h4_dir != h1_dir:
        confluence_label = f"{confluence_label}_vs_h4"

    # H4 trend strength bonus (only when H4 agrees with signal direction)
    if h4_dir == signal_dir and h4_feat.trend_strength is not None and not pd.isna(h4_feat.trend_strength):
        h4_trend_abs = abs(h4_feat.trend_strength)
        if h4_trend_abs > 60:
            confluence_bonus += 10.0
        elif h4_trend_abs > 40:
            confluence_bonus += 5.0

    # M5 check
    if m5_dir == signal_dir:
        confluence_bonus += 8.0
        confluence_label = f"{confluence_label}+m5"
    elif m5_dir != SignalType.NO_TRADE and m5_dir != signal_dir:
        confluence_bonus -= 12.0
        confluence_label = f"{confluence_label}_vs_m5"

    confidence = max(0.0, min(100.0, base_confidence + confluence_bonus + h4_h1_penalty))

    # Spread filter: reduce confidence or skip signal for wide spreads
    confidence, skip = _apply_spread_filter(confidence, h1_feat, symbol)
    if skip:
        logger.info("Swing: skipped — spread too wide")
        return None

    if confidence < cfg.confidence_threshold:
        logger.info("Swing: confidence %.1f < threshold %.1f", confidence, cfg.confidence_threshold)
        return None

    entry = round(live_price if live_price is not None else h1_close, 5)
    # Use per-symbol SL/TP multipliers if set, otherwise use trade mode defaults
    sl_mult, tp_mult = SETTINGS.get_sl_tp_multipliers(symbol.value)
    stop_loss, take_profit = _compute_stop_take(
        signal_dir, entry, h1_feat.atr14, symbol, sl_mult, tp_mult,
        atr_ratio=h1_feat.atr_ratio, symbol_name=symbol.value,
    )

    return TradeSignal(
        signal_id=f"{prefix}{uuid.uuid4().hex[:16]}",
        symbol=symbol.value,
        timeframe="confluence",
        signal=signal_dir,
        confidence=round(confidence, 2),
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        timestamp=df_h1.iloc[-1]["timestamp"],
        features=h1_feat,
        factor_scores=h1_scores,
        h4_direction=h4_dir.value,
        h1_direction=h1_dir.value,
        m5_direction=m5_dir.value,
        confluence=confluence_label,
        trade_mode="swing",
    )


# ── DISPATCHER ────────────────────────────────────────────────────────────────

def generate_signal(
    df_h4: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_m5: pd.DataFrame,
    symbol: Symbol,
    mode: Optional[TradeMode] = None,
    live_price: Optional[float] = None,
    prefix: str = "",
) -> Optional[TradeSignal]:
    """Generate signal based on the active trade mode.

    Args:
        live_price: Current market price. If provided, used as entry instead
            of the last candle close, so SL/TP reflect live conditions.
        prefix: Prefix for signal_id (e.g., "live_", "backtest_") to distinguish sources.
    """
    if mode is None:
        mode = SETTINGS.trade_mode

    if mode == TradeMode.SCALPING:
        return generate_signal_scalping(df_h1, df_m5, symbol, live_price=live_price, prefix=prefix)
    elif mode == TradeMode.DAY_TRADING:
        return generate_signal_day_trading(df_h4, df_h1, df_m5, symbol, live_price=live_price, prefix=prefix)
    elif mode == TradeMode.SWING:
        return generate_signal_swing(df_h4, df_h1, df_m5, symbol, live_price=live_price, prefix=prefix)
    else:
        logger.error("Unknown trade mode: %s", mode)
        return None