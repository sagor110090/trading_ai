"""Feature assembly: combines all indicators and derived features into a single DataFrame."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import Session
from trading_ai.indicators.ema import add_ema_columns
from trading_ai.indicators.rsi import add_rsi_column
from trading_ai.indicators.macd import add_macd_columns
from trading_ai.indicators.atr import add_atr_column

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSet:
    """Immutable snapshot of all features for a single candle."""
    ema20: Optional[float]
    ema50: Optional[float]
    ema200: Optional[float]
    rsi14: Optional[float]
    macd_line: Optional[float]
    macd_signal: Optional[float]
    macd_histogram: Optional[float]
    atr14: Optional[float]
    candle_body_pct: Optional[float]
    upper_wick_pct: Optional[float]
    lower_wick_pct: Optional[float]
    dist_ema20: Optional[float]
    dist_ema50: Optional[float]
    dist_ema200: Optional[float]
    trend_strength: Optional[float]
    momentum_score: Optional[float]
    session: Optional[str]
    spread: Optional[float]

    def to_dict(self) -> dict:
        return {
            "ema20": self.ema20,
            "ema50": self.ema50,
            "ema200": self.ema200,
            "rsi14": self.rsi14,
            "macd_line": self.macd_line,
            "macd_signal": self.macd_signal,
            "macd_histogram": self.macd_histogram,
            "atr14": self.atr14,
            "candle_body_pct": self.candle_body_pct,
            "upper_wick_pct": self.upper_wick_pct,
            "lower_wick_pct": self.lower_wick_pct,
            "dist_ema20": self.dist_ema20,
            "dist_ema50": self.dist_ema50,
            "dist_ema200": self.dist_ema200,
            "trend_strength": self.trend_strength,
            "momentum_score": self.momentum_score,
            "session": self.session,
            "spread": self.spread,
        }


def _determine_session(ts: pd.Timestamp) -> str:
    """Classify timestamp into trading session based on UTC hour.

    Asian: 00:00-08:00 UTC
    London: 08:00-16:00 UTC
    New York: 16:00-00:00 UTC
    """
    hour = ts.hour
    if 0 <= hour < 8:
        return Session.ASIAN.value
    elif 8 <= hour < 16:
        return Session.LONDON.value
    else:
        return Session.NEW_YORK.value


def _candle_body_pct(row: pd.Series) -> float:
    """Body as percentage of total candle range. Positive = bullish."""
    total_range = row["high"] - row["low"]
    if total_range == 0:
        return 0.0
    body = row["close"] - row["open"]
    return (body / total_range) * 100.0


def _upper_wick_pct(row: pd.Series) -> float:
    """Upper wick as percentage of total candle range."""
    total_range = row["high"] - row["low"]
    if total_range == 0:
        return 0.0
    upper_wick = row["high"] - max(row["open"], row["close"])
    return (upper_wick / total_range) * 100.0


def _lower_wick_pct(row: pd.Series) -> float:
    """Lower wick as percentage of total candle range."""
    total_range = row["high"] - row["low"]
    if total_range == 0:
        return 0.0
    lower_wick = min(row["open"], row["close"]) - row["low"]
    return (lower_wick / total_range) * 100.0


def _trend_strength(ema20: float, ema50: float, ema200: float, close: float) -> float:
    """Trend strength from -100 to +100 based on EMA alignment and price position."""
    if pd.isna(ema20) or pd.isna(ema50) or pd.isna(ema200):
        return 0.0

    score = 0.0

    # EMA stacking: bullish = ema20 > ema50 > ema200
    if ema20 > ema50 > ema200:
        score += 40.0
    elif ema20 < ema50 < ema200:
        score -= 40.0
    else:
        # Partial alignment
        if ema20 > ema50:
            score += 13.0
        else:
            score -= 13.0
        if ema50 > ema200:
            score += 7.0
        else:
            score -= 7.0

    # Price relative to EMAs
    if close > ema20:
        score += 20.0
    else:
        score -= 20.0
    if close > ema50:
        score += 13.0
    else:
        score -= 13.0
    if close > ema200:
        score += 7.0
    else:
        score -= 7.0

    # Width of EMA spread normalised
    avg_ema = (ema20 + ema50 + ema200) / 3.0
    if avg_ema > 0:
        spread_pct = (ema20 - ema200) / avg_ema * 100.0
        score += np.clip(spread_pct, -20.0, 20.0)

    return float(np.clip(score, -100.0, 100.0))


def _momentum_score(rsi: float, macd_hist: float, close: float, ema20: float) -> float:
    """Momentum score from -100 to +100 combining RSI, MACD, and price vs EMA20."""
    if pd.isna(rsi):
        return 0.0

    score = 0.0

    # RSI component: center around 50, scale to ±40
    rsi_component = (rsi - 50.0) * 0.8
    score += np.clip(rsi_component, -40.0, 40.0)

    # MACD histogram direction
    if not pd.isna(macd_hist):
        if macd_hist > 0:
            score += min(macd_hist / (abs(macd_hist) + 1e-9) * 30.0, 30.0)
        else:
            score += max(macd_hist / (abs(macd_hist) + 1e-9) * 30.0, -30.0)

    # Price momentum vs EMA20
    if not pd.isna(ema20) and ema20 > 0:
        dist_pct = (close - ema20) / ema20 * 100.0
        score += np.clip(dist_pct * 10.0, -30.0, 30.0)

    return float(np.clip(score, -100.0, 100.0))


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators and derived features for a candle DataFrame.

    Input DataFrame must have columns: timestamp, open, high, low, close, volume, spread.
    Returns enriched DataFrame with all feature columns.
    """
    if df.empty:
        return df

    df = df.copy()
    df = add_ema_columns(df)
    df = add_rsi_column(df)
    df = add_macd_columns(df)
    df = add_atr_column(df)

    # Candle structure features
    df["candle_body_pct"] = df.apply(_candle_body_pct, axis=1)
    df["upper_wick_pct"] = df.apply(_upper_wick_pct, axis=1)
    df["lower_wick_pct"] = df.apply(_lower_wick_pct, axis=1)

    # Distance from EMAs (percentage)
    df["dist_ema20"] = ((df["close"] - df["ema20"]) / df["ema20"]) * 100.0
    df["dist_ema50"] = ((df["close"] - df["ema50"]) / df["ema50"]) * 100.0
    df["dist_ema200"] = ((df["close"] - df["ema200"]) / df["ema200"]) * 100.0

    # Composite scores
    df["trend_strength"] = df.apply(
        lambda r: _trend_strength(r["ema20"], r["ema50"], r["ema200"], r["close"]),
        axis=1,
    )
    df["momentum_score"] = df.apply(
        lambda r: _momentum_score(r["rsi14"], r["macd_histogram"], r["close"], r["ema20"]),
        axis=1,
    )

    # Session classification
    df["session"] = df["timestamp"].apply(_determine_session)

    return df


def extract_feature_set(df: pd.DataFrame, idx: int) -> FeatureSet:
    """Extract a FeatureSet from a specific row index of the feature DataFrame."""
    row = df.iloc[idx]
    return FeatureSet(
        ema20=_safe_float(row, "ema20"),
        ema50=_safe_float(row, "ema50"),
        ema200=_safe_float(row, "ema200"),
        rsi14=_safe_float(row, "rsi14"),
        macd_line=_safe_float(row, "macd_line"),
        macd_signal=_safe_float(row, "macd_signal"),
        macd_histogram=_safe_float(row, "macd_histogram"),
        atr14=_safe_float(row, "atr14"),
        candle_body_pct=_safe_float(row, "candle_body_pct"),
        upper_wick_pct=_safe_float(row, "upper_wick_pct"),
        lower_wick_pct=_safe_float(row, "lower_wick_pct"),
        dist_ema20=_safe_float(row, "dist_ema20"),
        dist_ema50=_safe_float(row, "dist_ema50"),
        dist_ema200=_safe_float(row, "dist_ema200"),
        trend_strength=_safe_float(row, "trend_strength"),
        momentum_score=_safe_float(row, "momentum_score"),
        session=_safe_str(row, "session"),
        spread=_safe_float(row, "spread"),
    )


def _safe_float(row: pd.Series, col: str) -> Optional[float]:
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    return float(val)


def _safe_str(row: pd.Series, col: str) -> Optional[str]:
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    return str(val)