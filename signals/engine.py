"""Multi-timeframe confluence signal engine with 3 trade modes.

Scalping:  M5 entry + H1 trend only. Tight stops. Fast in/out.
Day trade: M5 entry + H1 trend + H4 big picture. Medium stops. Close same day.
Swing:     H4+H1 must agree + M5 trigger. Wide stops. Hold hours-days.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    SETTINGS, SignalType, Symbol, Timeframe, TradeMode,
)
from indicators.features import FeatureSet, compute_all_features, extract_feature_set
from signals.scoring import FactorScores, compute_signal_scores
from trade.weekly_close import check_weekly_close_status

logger = logging.getLogger(__name__)

WEIGHTS = SETTINGS.signal_weights
THRESHOLDS = SETTINGS.signal_thresholds


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
    features = extract_feature_set(df_features, idx)

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
) -> tuple[float, float]:
    """Calculate stop loss and take profit based on ATR multipliers.

    Automatically adds broker spread buffer to SL to account for execution latency.
    """
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
        "SL/TP for %s: entry=%.5f SL=%.5f TP=%.5f (broker buffer=%.5f pips=%.2f)",
        symbol.value, entry, stop_loss, take_profit, broker_buffer, SETTINGS.broker_spread_pips,
    )

    return round(stop_loss, 5), round(take_profit, 5)


def _get_m5_direction(df_m5: pd.DataFrame) -> tuple[SignalType, float]:
    """Quick M5 direction check. Returns (direction, momentum_score)."""
    if df_m5.empty or len(df_m5) < 50:
        return SignalType.NO_TRADE, 0.0

    m5_features = compute_all_features(df_m5)
    m5_last = m5_features.iloc[-1]
    m5_momentum = float(m5_last.get("momentum_score", 0))

    if pd.isna(m5_last.get("ema20")):
        return SignalType.NO_TRADE, m5_momentum

    if m5_last["close"] > m5_last["ema20"] and m5_momentum > 0:
        return SignalType.BUY, m5_momentum
    elif m5_last["close"] < m5_last["ema20"] and m5_momentum < 0:
        return SignalType.SELL, m5_momentum

    return SignalType.NO_TRADE, m5_momentum


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

    # Check weekly close - block new trades if approaching weekly close
    if SETTINGS.weekly_close.enabled:
        close_status = check_weekly_close_status(
            symbol=symbol.value,
            close_day=SETTINGS.weekly_close.close_day,
            close_time_utc=SETTINGS.weekly_close.close_time_utc,
            stop_new_trades_hours=SETTINGS.weekly_close.stop_new_trades_hours,
        )
        if close_status.should_stop_new_trades:
            logger.info("Scalping: %s - %s", symbol.value, close_status.message)
            return None

    if df_h1.empty or len(df_h1) < 200:
        logger.info("Scalping: insufficient H1 data (%d)", len(df_h1))
        return None

    h1_dir, h1_scores, h1_feat, h1_close = _get_direction_from_df(df_h1)
    m5_dir, m5_momentum = _get_m5_direction(df_m5)

    logger.info("Scalping: H1=%s M5=%s (M5 mom=%.1f)", h1_dir.value, m5_dir.value, m5_momentum)

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

    if confidence < cfg.confidence_threshold:
        logger.info("Scalping: confidence %.1f < threshold %.1f", confidence, cfg.confidence_threshold)
        return None

    entry = round(live_price if live_price is not None else h1_close, 5)
    stop_loss, take_profit = _compute_stop_take(
        h1_dir, entry, h1_feat.atr14, symbol, cfg.sl_atr_mult, cfg.tp_atr_mult,
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

def generate_signal_day_trading(
    df_h4: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_m5: pd.DataFrame,
    symbol: Symbol,
    live_price: Optional[float] = None,
    prefix: str = "",
) -> Optional[TradeSignal]:
    """Day trading signal: M5 entry in H1 trend direction, H4 must not contradict.

    Rules:
    - H1 provides trend direction
    - H4 must NOT oppose (H4 opposite = blocked)
    - M5 momentum must agree for entry timing
    - Medium stops: 1x ATR SL, 1.5x ATR TP
    - Threshold: 35%
    """
    cfg = SETTINGS.day_trading

    # Check weekly close - block new trades if approaching weekly close
    if SETTINGS.weekly_close.enabled:
        close_status = check_weekly_close_status(
            symbol=symbol.value,
            close_day=SETTINGS.weekly_close.close_day,
            close_time_utc=SETTINGS.weekly_close.close_time_utc,
            stop_new_trades_hours=SETTINGS.weekly_close.stop_new_trades_hours,
        )
        if close_status.should_stop_new_trades:
            logger.info("Day trade: %s - %s", symbol.value, close_status.message)
            return None

    if df_h1.empty or len(df_h1) < 200:
        logger.info("Day trade: insufficient H1 data (%d)", len(df_h1))
        return None

    h1_dir, h1_scores, h1_feat, h1_close = _get_direction_from_df(df_h1)
    m5_dir, m5_momentum = _get_m5_direction(df_m5)

    # H4 check — only needed to avoid counter-trend
    h4_dir = SignalType.NO_TRADE
    if not df_h4.empty and len(df_h4) >= 200:
        h4_dir, _, _, _ = _get_direction_from_df(df_h4)

    logger.info(
        "Day trade: H4=%s H1=%s M5=%s",
        h4_dir.value, h1_dir.value, m5_dir.value,
    )

    # H1 must have a direction
    if h1_dir == SignalType.NO_TRADE:
        logger.info("Day trade: H1 NO_TRADE")
        return None

    # H4 must not oppose H1
    if h4_dir != SignalType.NO_TRADE and h4_dir != h1_dir:
        logger.info("Day trade: H4=%s opposes H1=%s — blocked", h4_dir.value, h1_dir.value)
        return None

    # M5 must agree or be neutral
    if m5_dir != SignalType.NO_TRADE and m5_dir != h1_dir:
        confluence_penalty = -8.0
    else:
        confluence_penalty = 0.0

    base_confidence = min(h1_scores.total / WEIGHTS.total * 100.0, 100.0)
    confluence_bonus = 0.0
    confluence_label = cfg.confluence_label

    # H4 alignment bonus
    if h4_dir == h1_dir and h4_dir != SignalType.NO_TRADE:
        confluence_bonus += 8.0
        confluence_label = f"{cfg.confluence_label}+h4"

    # M5 alignment bonus
    if m5_dir == h1_dir:
        confluence_bonus += 6.0
        confluence_label = f"{confluence_label}+m5"

    confidence = max(0.0, min(100.0, base_confidence + confluence_bonus + confluence_penalty))

    if confidence < cfg.confidence_threshold:
        logger.info("Day trade: confidence %.1f < threshold %.1f", confidence, cfg.confidence_threshold)
        return None

    entry = round(live_price if live_price is not None else h1_close, 5)
    stop_loss, take_profit = _compute_stop_take(
        h1_dir, entry, h1_feat.atr14, symbol, cfg.sl_atr_mult, cfg.tp_atr_mult,
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
        h4_direction=h4_dir.value,
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
    """Swing signal: H4+H1 must agree, M5 entry trigger.

    Rules:
    - H4 = trend direction (the boss)
    - H1 must confirm H4 direction
    - M5 supports entry timing
    - Wide stops: 2x ATR SL, 3x ATR TP
    - Threshold: 40%
    """
    cfg = SETTINGS.swing

    # Check weekly close - block new trades if approaching weekly close
    if SETTINGS.weekly_close.enabled:
        close_status = check_weekly_close_status(
            symbol=symbol.value,
            close_day=SETTINGS.weekly_close.close_day,
            close_time_utc=SETTINGS.weekly_close.close_time_utc,
            stop_new_trades_hours=SETTINGS.weekly_close.stop_new_trades_hours,
        )
        if close_status.should_stop_new_trades:
            logger.info("Swing: %s - %s", symbol.value, close_status.message)
            return None

    if df_h4.empty or len(df_h4) < 200 or df_h1.empty or len(df_h1) < 200:
        logger.info("Swing: insufficient data (H4=%d, H1=%d)", len(df_h4), len(df_h1))
        return None

    h4_dir, h4_scores, h4_feat, _ = _get_direction_from_df(df_h4)
    h1_dir, h1_scores, h1_feat, h1_close = _get_direction_from_df(df_h1)
    m5_dir, m5_momentum = _get_m5_direction(df_m5)

    logger.info("Swing: H4=%s H1=%s M5=%s", h4_dir.value, h1_dir.value, m5_dir.value)

    # H4 and H1 must both have a direction and agree
    if h4_dir == SignalType.NO_TRADE or h1_dir == SignalType.NO_TRADE:
        logger.info("Swing: H4 or H1 is NO_TRADE")
        return None

    if h4_dir != h1_dir:
        logger.info("Swing: H4=%s vs H1=%s — conflicting", h4_dir.value, h1_dir.value)
        return None

    signal_dir = h4_dir

    # Confidence calculation
    base_confidence = min(h1_scores.total / WEIGHTS.total * 100.0, 100.0)
    confluence_bonus = 0.0
    confluence_label = f"{cfg.confluence_label}"

    # H4 trend strength bonus
    if h4_feat.trend_strength is not None and not pd.isna(h4_feat.trend_strength):
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

    confidence = max(0.0, min(100.0, base_confidence + confluence_bonus))

    if confidence < cfg.confidence_threshold:
        logger.info("Swing: confidence %.1f < threshold %.1f", confidence, cfg.confidence_threshold)
        return None

    entry = round(live_price if live_price is not None else h1_close, 5)
    stop_loss, take_profit = _compute_stop_take(
        signal_dir, entry, h1_feat.atr14, symbol, cfg.sl_atr_mult, cfg.tp_atr_mult,
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