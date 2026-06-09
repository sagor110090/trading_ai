"""Retrain ML models with updated configuration (class weights for imbalance)."""

import logging
import os
import sys
from pathlib import Path

# Setup project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

# Use SQLite
os.environ.setdefault('USE_SQLITE', 'true')

import numpy as np
import pandas as pd

# Direct imports from flat structure
from ml.dataset import build_dataset, split_features_target
from ml.trainer import train_models
from data.database import get_session
from data.models import Signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def retrain_models(symbols=None, model_types=None):
    """Retrain ML models with current configuration.

    Args:
        symbols: List of symbols to train (e.g., ["BTCUSD", "XAUUSD"]). None = all.
        model_types: List of model types (e.g., ["xgboost", "histgbrt", "catboost"]). None = all.
    """
    if model_types is None:
        model_types = ["xgboost", "histgbrt", "catboost"]

    if symbols is None:
        # Get unique symbols from database
        try:
            with get_session() as session:
                symbols = [s[0] for s in session.query(Signal.symbol).distinct().all()]
        except Exception as e:
            logger.error(f"Failed to get symbols from DB: {e}")
            symbols = ["BTCUSD", "XAUUSD"]

    logger.info(f"Retraining models for: {symbols}")
    logger.info(f"Model types: {model_types}")
    logger.info("=" * 60)

    all_results = {}

    for symbol in symbols:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {symbol}...")
        logger.info("=" * 60)

        # Build dataset for this symbol
        df = build_dataset(symbol=symbol, timeframe=None)

        if df.empty:
            logger.warning(f"No data for {symbol}, skipping")
            continue

        logger.info(f"Dataset: {len(df)} signals")
        win_rate = df["win"].mean() * 100
        wins = (df["win"] == 1.0).sum()
        losses = (df["win"] == 0.0).sum()
        breakevens = ((df["win"] > 0) & (df["win"] < 1)).sum()
        logger.info(f"Win rate: {win_rate:.1f}% ({wins} wins / {losses} losses / {breakevens} breakevens)")

        # Split into train/validation (70/30, time-ordered)
        df = df.sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(df) * 0.7)

        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()

        X_train, y_train = split_features_target(train_df)
        X_val, y_val = split_features_target(val_df)

        logger.info(f"Train: {len(X_train)} signals ({y_train.mean()*100:.1f}% win rate)")
        logger.info(f"Val: {len(X_val)} signals ({y_val.mean()*100:.1f}% win rate)")

        # Train models
        results = train_models(
            symbol=symbol,
            timeframe="all",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_types=model_types,
            train_signal_count=len(df),
        )

        all_results[symbol] = results

        # Print results
        logger.info(f"\n{'=' * 60}")
        logger.info(f"RESULTS for {symbol}:")
        logger.info("=" * 60)
        print(f"{'Model':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
        print("-" * 60)

        for result in results:
            m = result.metrics
            print(f"{result.model_type:<12} {m.accuracy:<10.4f} {m.precision:<10.4f} {m.recall:<10.4f} {m.f1:<10.4f} {m.roc_auc:<10.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info("\nAll models saved to: ml/models/")
    logger.info("Model metadata updated in database")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrain ML models with updated config")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to train (e.g., BTCUSD XAUUSD)")
    parser.add_argument("--models", nargs="+", default=None, help="Model types (e.g., xgboost histgbrt catboost)")

    args = parser.parse_args()

    retrain_models(symbols=args.symbols, model_types=args.models)
