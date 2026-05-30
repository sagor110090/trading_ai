"""EMA indicator calculations."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average.

    Uses pandas' ewm which matches standard EMA definition.
    """
    return series.ewm(span=period, adjust=False).mean()


def add_ema_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA20, EMA50, EMA200 columns to a DataFrame with a 'close' column."""
    df = df.copy()
    df["ema20"] = compute_ema(df["close"], 20)
    df["ema50"] = compute_ema(df["close"], 50)
    df["ema200"] = compute_ema(df["close"], 200)
    return df