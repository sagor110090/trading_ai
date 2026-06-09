"""Feature assembly: combines all indicators and derived features into a single DataFrame."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import Session
from indicators.ema import add_ema_columns
from indicators.rsi import add_rsi_column
from indicators.macd import add_macd_columns
from indicators.atr import add_atr_column

logger = logging.getLogger(__name__)


# Columns for which lag features are computed (1, 2, 3 candles back)
LAG_COLUMNS = ["rsi14", "macd_histogram", "momentum_score", "trend_strength", "dist_ema200"]
LAG_PERIODS = [1, 2, 3]

# All lag feature column names (e.g., "rsi14_lag1", "rsi14_lag2", ...)
LAG_FEATURE_COLS = [f"{col}_lag{lag}" for col in LAG_COLUMNS for lag in LAG_PERIODS]


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
    # Lag features (previous candle values for temporal context)
    rsi14_lag1: Optional[float] = None
    rsi14_lag2: Optional[float] = None
    rsi14_lag3: Optional[float] = None
    macd_histogram_lag1: Optional[float] = None
    macd_histogram_lag2: Optional[float] = None
    macd_histogram_lag3: Optional[float] = None
    momentum_score_lag1: Optional[float] = None
    momentum_score_lag2: Optional[float] = None
    momentum_score_lag3: Optional[float] = None
    trend_strength_lag1: Optional[float] = None
    trend_strength_lag2: Optional[float] = None
    trend_strength_lag3: Optional[float] = None
    dist_ema200_lag1: Optional[float] = None
    dist_ema200_lag2: Optional[float] = None
    dist_ema200_lag3: Optional[float] = None
    # Volatility regime indicator: current ATR relative to its rolling mean
    atr_ratio: Optional[float] = None

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
            # Lag features
            "rsi14_lag1": self.rsi14_lag1,
            "rsi14_lag2": self.rsi14_lag2,
            "rsi14_lag3": self.rsi14_lag3,
            "macd_histogram_lag1": self.macd_histogram_lag1,
            "macd_histogram_lag2": self.macd_histogram_lag2,
            "macd_histogram_lag3": self.macd_histogram_lag3,
            "momentum_score_lag1": self.momentum_score_lag1,
            "momentum_score_lag2": self.momentum_score_lag2,
            "momentum_score_lag3": self.momentum_score_lag3,
            "trend_strength_lag1": self.trend_strength_lag1,
            "trend_strength_lag2": self.trend_strength_lag2,
            "trend_strength_lag3": self.trend_strength_lag3,
            "dist_ema200_lag1": self.dist_ema200_lag1,
            "dist_ema200_lag2": self.dist_ema200_lag2,
            "dist_ema200_lag3": self.dist_ema200_lag3,
            "atr_ratio": self.atr_ratio,
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

    # MACD histogram direction and magnitude (tanh squash preserves magnitude)
    if not pd.isna(macd_hist):
        # tanh maps real line to (-1, 1); dividing by 2.0 means a histogram of
        # ±2.0 reaches ~76% of the max component. This preserves magnitude
        # instead of the old bug where any non-zero value got ±30.
        macd_component = np.tanh(macd_hist / 2.0) * 30.0
        score += macd_component

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

    # ATR ratio: current ATR relative to its rolling mean (volatility regime)
    # Values >1.5 = fast market, <0.7 = quiet market, ~1.0 = normal
    atr_ma = df["atr14"].rolling(14, min_periods=5).mean()
    df["atr_ratio"] = df["atr14"] / atr_ma.replace(0, float("nan"))
    df["atr_ratio"] = df["atr_ratio"].fillna(1.0)  # Default: normal volatility

    # Session classification
    df["session"] = df["timestamp"].apply(_determine_session)

    return df


def extract_feature_set(df: pd.DataFrame, idx: int) -> FeatureSet:
    """Extract a FeatureSet from a specific row index of the feature DataFrame.

    Does NOT include lag features — use extract_feature_set_with_lags() for that.
    """
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
        atr_ratio=_safe_float(row, "atr_ratio"),
    )


def extract_feature_set_with_lags(df: pd.DataFrame, idx: int) -> FeatureSet:
    """Extract a FeatureSet with lag features from a feature DataFrame.

    Lag features capture the value of key indicators 1, 2, and 3 candles
    before the current one, giving the model temporal context.

    Args:
        df: Feature DataFrame (must have at least 4 rows before idx).
        idx: Row index to extract features for.

    Returns:
        FeatureSet with lag fields populated from previous candles.
    """
    feat = extract_feature_set(df, idx)

    # Extract lag features from previous candles
    lag_fields = {}
    for col in LAG_COLUMNS:
        for lag in LAG_PERIODS:
            key = f"{col}_lag{lag}"
            if idx - lag >= 0 and col in df.columns:
                val = df.iloc[idx - lag].get(col)
                lag_fields[key] = float(val) if pd.notna(val) else None
            else:
                lag_fields[key] = None

    return FeatureSet(**{**feat.to_dict(), **lag_fields})


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