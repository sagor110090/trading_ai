"""Dataset builder: converts verified signals into ML-ready training data."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from trading_ai.config.settings import SignalResult
from trading_ai.data.database import get_engine
from trading_ai.data.models import Signal, SignalVerification

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

SESSION_ENCODING = {"ASIAN": 0, "LONDON": 1, "NEW_YORK": 2}

TARGET_COLUMN = "win"


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
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Encode target: WIN=1, LOSS=0
    df[TARGET_COLUMN] = (df["result"] == SignalResult.WIN.value).astype(int)

    # Encode session
    df["session_enc"] = df["session"].map(SESSION_ENCODING).fillna(-1).astype(int)

    # Replace NaN features with 0 for ML
    feature_cols = FEATURE_COLUMNS + ["session_enc"]
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    logger.info(
        "Built dataset: %d rows, %d features, win rate=%.1f%%",
        len(df),
        len(feature_cols),
        df[TARGET_COLUMN].mean() * 100,
    )

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataset into feature matrix X and target vector y."""
    feature_cols = FEATURE_COLUMNS + ["session_enc"]
    available = [c for c in feature_cols if c in df.columns]

    X = df[available].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y