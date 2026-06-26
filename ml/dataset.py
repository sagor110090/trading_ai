"""Dataset builder: converts verified signals into ML-ready training data."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from config.settings import SignalResult
from data.database import get_engine
from data.models import Signal, SignalVerification
from indicators.features import LAG_COLUMNS, LAG_PERIODS, LAG_FEATURE_COLS

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
    "atr_ratio",
]

SESSION_CATEGORIES = ["ASIAN", "LONDON", "NEW_YORK"]

SESSION_FEATURE_COLS = [f"session_{cat.lower()}" for cat in SESSION_CATEGORIES]
# Legacy column name for backward compatibility with old models
SESSION_ENC_COL = "session_enc"

TRADE_MODES = ["scalping", "day_trading", "swing"]
MODE_FEATURE_COLS = [f"mode_{m}" for m in TRADE_MODES]

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
    "atr_ratio": 1.0,       # Normal volatility (neutral default)
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
            s.session, s.spread, s.atr_ratio, s.trade_mode,
            v.result,
            v.profit_points,
            v.risk_reward,
            v.holding_candles,
            v.max_favorable_excursion,
            v.max_adverse_excursion
        FROM signals s
        JOIN signal_verifications v ON v.signal_id = s.id
        WHERE v.result IN ('WIN', 'LOSS', 'BREAKEVEN')
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

    # Encode target: WIN=1, LOSS=0. BREAKEVEN rows are DROPPED — a breakeven
    # trade is a neutral outcome, not a losing pattern. Labeling it as LOSS
    # adds label noise that depresses probabilities for similar signals.
    target_map = {
        SignalResult.WIN.value: 1.0,
        SignalResult.LOSS.value: 0.0,
    }
    df[TARGET_COLUMN] = df["result"].map(target_map)
    be_count = int(df[TARGET_COLUMN].isna().sum())
    if be_count > 0:
        logger.info("Dropping %d BREAKEVEN/unmapped rows (not WIN or LOSS)", be_count)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    # Encode session with one-hot encoding
    for cat in SESSION_CATEGORIES:
        df[f"session_{cat.lower()}"] = (df["session"] == cat).astype(int)

    # Encode trade_mode with one-hot encoding
    if "trade_mode" in df.columns:
        for mode in TRADE_MODES:
            df[f"mode_{mode}"] = (df["trade_mode"] == mode).astype(int)
    else:
        # Fallback: no trade_mode column (old data)
        for mode in TRADE_MODES:
            df[f"mode_{mode}"] = 0

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

    # Add lag features (1, 2, 3 candles back) for key indicators.
    # These give the model temporal context — e.g., "RSI was 30 three candles
    # ago and is now 50" vs "RSI was 70 three candles ago and is now 50".
    df = _add_lag_features(df)

    # Compute ATR ratio (volatility regime indicator) from stored atr14 values.
    # Values >1.5 = fast market, <0.7 = quiet, ~1.0 = normal.
    if "atr14" in df.columns:
        atr_ma = df["atr14"].rolling(14, min_periods=5).mean()
        df["atr_ratio"] = df["atr14"] / atr_ma.replace(0, float("nan"))
        df["atr_ratio"] = df["atr_ratio"].fillna(1.0)

    # All feature columns including lag features
    all_feature_cols = FEATURE_COLUMNS + LAG_FEATURE_COLS + SESSION_FEATURE_COLS

    logger.info(
        "Built dataset: %d rows, %d features, win rate=%.1f%%",
        len(df),
        len(all_feature_cols),
        df[TARGET_COLUMN].mean() * 100,
    )

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataset into feature matrix X and target vector y."""
    feature_cols = FEATURE_COLUMNS + LAG_FEATURE_COLS + SESSION_FEATURE_COLS + MODE_FEATURE_COLS
    available = [c for c in feature_cols if c in df.columns]

    X = df[available].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features (1, 2, 3 candles back) for key indicators.

    Lags are computed within each (symbol, timeframe) group to prevent
    cross-contamination. When timeframe=None (the "all" model), signals from
    M5/H1/H4 interleave per symbol — grouping only by symbol would pull lags
    from a different timeframe, corrupting the temporal features.
    Rows where lag values are NaN (first few rows per group) are dropped.

    Must be called AFTER sorting by timestamp and AFTER filling NaN values.
    """
    if df.empty or "symbol" not in df.columns or "timeframe" not in df.columns:
        return df

    # Ensure sorted by timestamp within each (symbol, timeframe)
    df = df.sort_values(["symbol", "timeframe", "timestamp"]).reset_index(drop=True)

    # Compute lag features per (symbol, timeframe) group
    group_cols = ["symbol", "timeframe"]
    for (sym, tf), group_idx in df.groupby(group_cols).groups.items():
        for col in LAG_COLUMNS:
            if col not in df.columns:
                continue
            for lag in LAG_PERIODS:
                lag_col = f"{col}_lag{lag}"
                df.loc[group_idx, lag_col] = df.loc[group_idx, col].shift(lag)

    # Fill lag NaNs with 0 (first few rows per group where shift creates NaN).
    # For lag features, 0 means "no previous data" which is a reasonable neutral.
    for col_name in LAG_FEATURE_COLS:
        if col_name in df.columns:
            df[col_name] = df[col_name].fillna(0)

    return df
