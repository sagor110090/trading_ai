"""Walk-forward validation: strict time-series cross-validation with no data leakage."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from trading_ai.config.settings import SETTINGS
from trading_ai.ml.dataset import TARGET_COLUMN, build_dataset, split_features_target
from trading_ai.ml.trainer import (
    ModelMetrics,
    _compute_metrics,
    _ensure_model_dir,
    _store_model_meta,
    train_models,
)

logger = logging.getLogger(__name__)

WF_CONFIG = SETTINGS.walk_forward


@dataclass(frozen=True)
class FoldResult:
    """Metrics for a single walk-forward fold."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int
    metrics: ModelMetrics


@dataclass(frozen=True)
class WalkForwardReport:
    """Aggregated walk-forward validation report."""
    symbol: str
    timeframe: str
    model_type: str
    folds: list
    mean_accuracy: float
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_roc_auc: float
    win_rate: float


def _month_range(start: pd.Timestamp, months: int) -> pd.Timestamp:
    """Add months to a timestamp."""
    year = start.year + (start.month + months - 1) // 12
    month = (start.month + months - 1) % 12 + 1
    return start.replace(year=year, month=month)


def generate_walk_forward_folds(
    df: pd.DataFrame,
    train_months: int = WF_CONFIG.train_months,
    test_months: int = WF_CONFIG.test_months,
    min_signals: int = WF_CONFIG.min_signals,
) -> list[tuple[pd.DataFrame, pd.DataFrame, dict]]:
    """Generate walk-forward train/test splits.

    NO random splitting. NO data leakage.
    Each fold trains on expanding window, tests on the next period.

    Returns list of (train_df, test_df, fold_info).
    """
    if df.empty or "timestamp" not in df.columns:
        logger.warning("Dataset missing timestamp column or empty")
        return []

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Determine time range
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()

    # Align start to beginning of month
    current_test_start = _month_range(start_date, train_months)

    folds = []
    fold_idx = 0

    while current_test_start < end_date:
        current_test_end = _month_range(current_test_start, test_months)

        train_mask = df["timestamp"] < current_test_start
        test_mask = (df["timestamp"] >= current_test_start) & (df["timestamp"] < current_test_end)

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if len(train_df) < min_signals or len(test_df) < 5:
            logger.debug(
                "Fold %d skipped: train=%d test=%d (min=%d)",
                fold_idx, len(train_df), len(test_df), min_signals,
            )
            current_test_start = _month_range(current_test_start, test_months)
            fold_idx += 1
            continue

        fold_info = {
            "fold": fold_idx,
            "train_start": str(train_df["timestamp"].min()),
            "train_end": str(train_df["timestamp"].max()),
            "test_start": str(test_df["timestamp"].min()),
            "test_end": str(test_df["timestamp"].max()),
            "train_size": len(train_df),
            "test_size": len(test_df),
        }

        folds.append((train_df, test_df, fold_info))
        logger.info(
            "Fold %d: train %s → %s (%d), test %s → %s (%d)",
            fold_idx,
            fold_info["train_start"][:10], fold_info["train_end"][:10], len(train_df),
            fold_info["test_start"][:10], fold_info["test_end"][:10], len(test_df),
        )

        # Roll forward
        current_test_start = _month_range(current_test_start, test_months)
        fold_idx += 1

    logger.info("Generated %d walk-forward folds", len(folds))
    return folds


def run_walk_forward(
    symbol: str,
    timeframe: str,
    model_types: Optional[list[str]] = None,
) -> list[WalkForwardReport]:
    """Run walk-forward validation for all model types.

    Trains on each expanding window and evaluates on the out-of-sample test period.
    NO random splits. NO look-ahead. Strict temporal ordering.

    Returns a WalkForwardReport per model_type.
    """
    df = build_dataset(symbol=symbol, timeframe=timeframe)
    if df.empty:
        logger.warning("No data for walk-forward validation of %s %s", symbol, timeframe)
        return []

    folds = generate_walk_forward_folds(df)
    if not folds:
        logger.warning("No valid folds for walk-forward validation")
        return []

    if model_types is None:
        model_types = ["xgboost", "lightgbm", "catboost"]

    reports = []

    for model_type in model_types:
        fold_results = []

        for train_df, test_df, fold_info in folds:
            X_train, y_train = split_features_target(train_df)
            X_test, y_test = split_features_target(test_df)

            if X_train.empty or X_test.empty:
                continue

            # Train single model type on this fold
            results = train_models(
                symbol=symbol,
                timeframe=timeframe,
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                model_types=[model_type],
            )

            if not results:
                continue

            # The validation metrics from the trainer ARE the walk-forward metrics
            # since X_val/y_val = the out-of-sample test set
            metrics = results[0].metrics

            fold_result = FoldResult(
                fold=fold_info["fold"],
                train_start=fold_info["train_start"],
                train_end=fold_info["train_end"],
                test_start=fold_info["test_start"],
                test_end=fold_info["test_end"],
                train_size=fold_info["train_size"],
                test_size=fold_info["test_size"],
                metrics=metrics,
            )
            fold_results.append(fold_result)

        if not fold_results:
            continue

        # Aggregate across folds
        accs = [f.metrics.accuracy for f in fold_results]
        precs = [f.metrics.precision for f in fold_results]
        recs = [f.metrics.recall for f in fold_results]
        f1s = [f.metrics.f1 for f in fold_results]
        aucs = [f.metrics.roc_auc for f in fold_results]

        # Win rate = mean accuracy across folds (fraction of correct predictions)
        win_rate = np.mean(accs)

        report = WalkForwardReport(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            folds=[{
                "fold": f.fold,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "accuracy": f.metrics.accuracy,
                "precision": f.metrics.precision,
                "recall": f.metrics.recall,
                "f1": f.metrics.f1,
                "roc_auc": f.metrics.roc_auc,
            } for f in fold_results],
            mean_accuracy=round(float(np.mean(accs)), 4),
            mean_precision=round(float(np.mean(precs)), 4),
            mean_recall=round(float(np.mean(recs)), 4),
            mean_f1=round(float(np.mean(f1s)), 4),
            mean_roc_auc=round(float(np.mean(aucs)), 4),
            win_rate=round(float(win_rate), 4),
        )

        reports.append(report)

        logger.info(
            "Walk-forward %s %s %s: acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f over %d folds",
            model_type, symbol, timeframe,
            report.mean_accuracy, report.mean_precision,
            report.mean_recall, report.mean_f1,
            report.mean_roc_auc, len(fold_results),
        )

    return reports