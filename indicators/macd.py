"""MACD indicator calculation."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD line, signal line, and histogram.

    Returns:
        (macd_line, macd_signal, macd_histogram)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram


def add_macd_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add MACD, MACD_signal, MACD_histogram columns to a DataFrame."""
    df = df.copy()
    macd_line, macd_signal, macd_histogram = compute_macd(df["close"])
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_histogram"] = macd_histogram
    return df