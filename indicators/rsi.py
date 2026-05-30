"""RSI indicator calculation."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.

    Uses Wilder's smoothing (exponential moving average of gains/losses).
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def add_rsi_column(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI14 column to a DataFrame with a 'close' column."""
    df = df.copy()
    df["rsi14"] = compute_rsi(df["close"], period)
    return df