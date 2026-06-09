"""Dataset builder: converts verified signals into ML-ready training data."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from config.settings import SignalResult
from data.database import get_engine
from data.models import Signal, SignalVerification

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "ema20",
    "ema50",
    "ema200",
    "rsi14",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "atr14",
    "momentum_score",
    "trend_strength",
    "dist_ema20",
    "dist_ema50",
    "dist_ema200",
    "candle_body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "spread",
]

SESSION_CATEGORIES = ["ASIAN", "LONDON", "NEW_YORK"]

SESSION_FEATURE_COLS = [f"session_{cat.lower()}" for cat in SESSION_CATEGORIES]
# Legacy column name for backward compatibility with old models
SESSION_ENC_COL = "session_enc"

TARGET_COLUMN = "win"

# Per-column NaN defaults: use neutral values instead of blanket 0.
# EMA and ATR columns are dropped entirely when NaN (early candles without
# enough history) rather than filled with a misleading 0.
NAN_DEFAULTS = {
    "rsi14": 50.0,           # Neutral RSI
    "macd_line": 0.0,        # Neutral MACD
    "macd_signal": 0.0,      # Neutral MACD
    "macd_histogram": 0.0,   # Neutral MACD
    "momentum_score": 0.0,    # Neutral momentum
    "trend_strength": 0.0,    # Neutral trend
    "dist_ema20": 0.0,       # At EMA = neutral
    "dist_ema50": 0.0,       # At EMA = neutral
    "dist_ema200": 0.0,      # At EMA = neutral
    "candle_body_pct": 0.0,  # Neutral candle
    "upper_wick_pct": 0.0,
    "lower_wick_pct": 0.0,
    "spread": 0.0,
    "session_asian": 0,
    "session_london": 0,
    "session_new_york": 0,
}

# Features where NaN rows should be dropped entirely (early data without enough
# history to compute these indicators — filling with 0 would create garbage
# values especially for dist_ema* columns where division by a zero EMA occurs).
DROP_IF_NAN = ["ema20", "ema50", "ema200", "atr14"]


def build_dataset(symbol: Optional[str] = None, timeframe: Optional[str] = None) -> pd.DataFrame:
    """Build an ML dataset from all verified signals.

    Each row = one signal with its feature snapshot and binary target (WIN=1, LOSS=0).
    BREAKEVEN signals are excluded (ambiguous label).

    Returns DataFrame with feature columns + target column + metadata.
    """
    engine = get_engine()

    query = """
        SELECT
            s.id,
            s.signal_id,
            s.symbol,
            s.timeframe,
            s.signal AS signal_type,
            s.confidence,
            s.timestamp,
            s.ema20, s.ema50, s.ema200,
            s.rsi14,
            s.macd_line, s.macd_signal, s.macd_histogram,
            s.atr14,
            s.trend_strength, s.momentum_score,
            s.dist_ema20, s.dist_ema50, s.dist_ema200,
            s.candle_body_pct, s.upper_wick_pct, s.lower_wick_pct,
            s.session, s.spread,
            v.result,
            v.profit_points,
            v.risk_reward,
            v.holding_candles,
            v.max_favorable_excursion,
            v.max_adverse_excursion
        FROM signals s
        JOIN signal_verifications v ON v.signal_id = s.id
        WHERE v.result IN ('WIN', 'LOSS')
    """

    conditions = []
    if symbol:
        conditions.append(f"s.symbol = '{symbol}'")
    if timeframe:
        conditions.append(f"s.timeframe = '{timeframe}'")

    if conditions:
        query += " AND " + " AND ".join(conditions)

    query += " ORDER BY s.timestamp"

    df = pd.read_sql(text(query), engine)

    if df.empty:
        logger.warning("No verified signals found for dataset")
        return df

    # Convert timestamp strings to datetime objects
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)

    # Encode target: WIN=1, LOSS=0
    df[TARGET_COLUMN] = (df["result"] == SignalResult.WIN.value).astype(int)

    # Encode session with one-hot encoding
    for cat in SESSION_CATEGORIES:
        df[f"session_{cat.lower()}"] = (df["session"] == cat).astype(int)

    # Drop rows where critical indicators are NaN (early candles without enough
    # history). These produce garbage values for dist_ema* when filled to 0.
    rows_before = len(df)
    for col in DROP_IF_NAN:
        if col in df.columns:
            df = df[df[col].notna()].copy()
    if len(df) < rows_before:
        logger.info(
            "Dropped %d rows with NaN in critical columns (%d → %d)",
            rows_before - len(df), rows_before, len(df),
        )

    # Replace remaining NaN features with neutral defaults
    feature_cols = FEATURE_COLUMNS + SESSION_FEATURE_COLS
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col in NAN_DEFAULTS:
                df[col] = df[col].fillna(NAN_DEFAULTS[col])
            else:
                # Columns not in NAN_DEFAULTS (e.g. EMA cols) should already
                # be clean after the DROP_IF_NaN pass, but fill any stragglers
                df[col] = df[col].fillna(0)

    logger.info(
        "Built dataset: %d rows, %d features, win rate=%.1f%%",
        len(df),
        len(feature_cols),
        df[TARGET_COLUMN].mean() * 100,
    )

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataset into feature matrix X and target vector y."""
    feature_cols = FEATURE_COLUMNS + SESSION_FEATURE_COLS
    available = [c for c in feature_cols if c in df.columns]

    X = df[available].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y
