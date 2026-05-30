"""ATR indicator calculation."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Average True Range using Wilder's smoothing."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def add_atr_column(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ATR14 column to a DataFrame with high, low, close columns."""
    df = df.copy()
    df["atr14"] = compute_atr(df["high"], df["low"], df["close"], period)
    return df