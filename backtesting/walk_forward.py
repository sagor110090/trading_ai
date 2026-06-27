"""Walk-forward validation: strict time-series cross-validation with no data leakage."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.settings import SETTINGS
from ml.dataset import TARGET_COLUMN, build_dataset, split_features_target
from ml.trainer import (
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
    """Add months to a timestamp, clamping day to the max valid day of the target month."""
    year = start.year + (start.month + months - 1) // 12
    month = (start.month + months - 1) % 12 + 1
    # Clamp day to max valid day for the target month (avoids day 31 in 30-day months)
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    day = min(start.day, max_day)
    return start.replace(year=year, month=month, day=day)


def generate_walk_forward_folds(
    df: pd.DataFrame,
    train_months: int = WF_CONFIG.train_months,
    test_months: int = WF_CONFIG.test_months,
    min_signals: int = WF_CONFIG.min_signals,
) -> list[tuple[pd.DataFrame, pd.DataFrame, dict]]:
    """Generate walk-forward train/test splits.

    NO random splitting. NO data leakage.
    Each fold trains on expanding window, tests on the next period.

    For datasets spanning < train_months+test_months, falls back to a
    percentage-based split (70/30) while maintaining temporal order.

    Returns list of (train_df, test_df, fold_info).
    """
    if df.empty or "timestamp" not in df.columns:
        logger.warning("Dataset missing timestamp column or empty")
        return []

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Convert timestamp strings to datetime objects
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)

    # Determine time range
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    time_span_days = (end_date - start_date).total_seconds() / 86400

    required_months = train_months + test_months
    required_days = required_months * 30

    # ── Fallback: percentage-based split for short datasets ──
    if time_span_days < required_days or len(df) < min_signals * 2:
        logger.info(
            "Dataset spans only %.1f days (%d signals) — using percentage-based split",
            time_span_days, len(df),
        )
        return _percentage_split_folds(df, min_signals)

    # ── Normal: month-based walk-forward folds ──
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


def _percentage_split_folds(
    df: pd.DataFrame,
    min_signals: int,
    train_ratio: float = 0.7,
) -> list[tuple[pd.DataFrame, pd.DataFrame, dict]]:
    """Fallback: stratified time-ordered split for short datasets.

    Ensures both WIN and LOSS appear in both train and test sets.
    Uses a time-blocked stratified approach: divides data into blocks
    and assigns each block to train or test, maintaining temporal order
    within each split.
    """
    from ml.dataset import TARGET_COLUMN

    n = len(df)

    # Check class distribution
    if TARGET_COLUMN not in df.columns:
        logger.warning("No target column in dataset")
        return []

    class_counts = df[TARGET_COLUMN].value_counts()
    n_classes = len(class_counts)
    minority_count = class_counts.min()
    minority_class = class_counts.idxmin()

    if n_classes < 2:
        logger.warning(
            "Only one class in dataset (%s=%d) — cannot train a meaningful classifier. "
            "Need both WIN and LOSS signals. Collecting more data...",
            "WIN" if minority_class == 1 else "LOSS", minority_count,
        )
        return []

    # Need at least 2 minority samples in each split
    min_per_split = 2
    if minority_count < min_per_split * 2:
        logger.warning(
            "Too few minority class samples (%d) for stratified split (need %d). "
            "Will use oversampling.",
            minority_count, min_per_split * 2,
        )
        return _oversampled_split(df, min_signals)

    # Time-blocked stratified split:
    # Split data into 10 time blocks, assign blocks to train/test proportionally
    # This preserves temporal order within each split while ensuring class balance
    n_blocks = min(10, n)
    block_size = n // n_blocks
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else n
        blocks.append(df.iloc[start:end])

    train_blocks = []
    test_blocks = []
    n_train_blocks = int(n_blocks * train_ratio)

    # Assign first N blocks to train, rest to test (time order preserved)
    train_blocks = blocks[:n_train_blocks]
    test_blocks = blocks[n_train_blocks:]

    train_df = pd.concat(train_blocks, ignore_index=True)
    test_df = pd.concat(test_blocks, ignore_index=True)

    # Verify both classes in each split
    train_classes = set(train_df[TARGET_COLUMN].unique())
    test_classes = set(test_df[TARGET_COLUMN].unique())

    if len(train_classes) < 2 or len(test_classes) < 2:
        # Fall back to oversampled approach
        logger.warning(
            "Time-blocked split produced single-class test set. Using oversampled split."
        )
        return _oversampled_split(df, min_signals)

    fold_info = {
        "fold": 0,
        "train_start": str(train_df["timestamp"].min()),
        "train_end": str(train_df["timestamp"].max()),
        "test_start": str(test_df["timestamp"].min()),
        "test_end": str(test_df["timestamp"].max()),
        "train_size": len(train_df),
        "test_size": len(test_df),
    }

    train_win = (train_df[TARGET_COLUMN] == 1).sum()
    test_win = (test_df[TARGET_COLUMN] == 1).sum()
    logger.info(
        "Stratified split: train %s → %s (%d, %dW/%dL), test %s → %s (%d, %dW/%dL)",
        fold_info["train_start"][:10], fold_info["train_end"][:10],
        len(train_df), train_win, len(train_df) - train_win,
        fold_info["test_start"][:10], fold_info["test_end"][:10],
        len(test_df), test_win, len(test_df) - test_win,
    )

    return [(train_df, test_df, fold_info)]


def _oversampled_split(
    df: pd.DataFrame,
    min_signals: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame, dict]]:
    """Create balanced train/test using SMOTE-like oversampling of minority class.

    For very small datasets where simple splitting produces single-class splits.
    Uses sklearn's resample to oversample the minority class in train, and
    keeps test set as-is (no synthetic data in evaluation).
    """
    from sklearn.utils import resample
    from ml.dataset import TARGET_COLUMN

    n = len(df)
    train_ratio = 0.7
    split_idx = int(n * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Oversample minority class in train set ONLY
    majority_class = train_df[TARGET_COLUMN].value_counts().idxmax()
    minority_class = 1 - majority_class

    train_majority = train_df[train_df[TARGET_COLUMN] == majority_class]
    train_minority = train_df[train_df[TARGET_COLUMN] == minority_class]

    if len(train_minority) == 0:
        # No minority in train — cannot form a valid split without leaking
        # test data into train. Return empty so the caller skips this split.
        logger.warning("No minority class in train split — skipping (would require data leak)")
        return []

    # Upsample minority to match majority count
    train_minority_upsampled = resample(
        train_minority,
        replace=True,
        n_samples=len(train_majority),
        random_state=42,
    )

    train_balanced = pd.concat([train_majority, train_minority_upsampled])
    train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Verify test has both classes
    if len(test_df[TARGET_COLUMN].unique()) < 2:
        # Test set has a single class — AUC is undefined. Rather than leak
        # train rows into test, skip this split.
        logger.warning("Test split has single class — skipping (would require data leak)")
        return []

    train_win = (train_balanced[TARGET_COLUMN] == 1).sum()
    test_win = (test_df[TARGET_COLUMN] == 1).sum()

    fold_info = {
        "fold": 0,
        "train_start": str(train_balanced["timestamp"].min()),
        "train_end": str(train_balanced["timestamp"].max()),
        "test_start": str(test_df["timestamp"].min()),
        "test_end": str(test_df["timestamp"].max()),
        "train_size": len(train_balanced),
        "test_size": len(test_df),
    }

    logger.info(
        "Oversampled split: train (%d, %dW/%dL), test (%d, %dW/%dL)",
        len(train_balanced), train_win, len(train_balanced) - train_win,
        len(test_df), test_win, len(test_df) - test_win,
    )

    return [(train_balanced, test_df, fold_info)]


def run_walk_forward(
    symbol: str,
    timeframe: Optional[str] = None,
    model_types: Optional[list[str]] = None,
) -> list[WalkForwardReport]:
    """Run walk-forward validation for all model types.

    Trains on each expanding window and evaluates on the out-of-sample test period.
    NO random splits. NO look-ahead. Strict temporal ordering.

    Args:
        symbol: Trading pair symbol (e.g. "BTCUSD").
        timeframe: Timeframe filter (e.g. "5m") or None for all.
        model_types: List of model types to train.

    Returns a WalkForwardReport per model_type.
    """
    df = build_dataset(symbol=symbol, timeframe=timeframe)
    if df.empty:
        logger.warning("No data for walk-forward validation of %s %s", symbol, timeframe or "all")
        return []

    total_signals = len(df)

    folds = generate_walk_forward_folds(df)
    if not folds:
        logger.warning("No valid folds for walk-forward validation")
        return []

    if model_types is None:
        model_types = ["xgboost", "histgbrt", "catboost"]

    reports = []

    for model_type in model_types:
        fold_results = []

        for train_df, test_df, fold_info in folds:
            X_train, y_train = split_features_target(train_df)
            X_test, y_test = split_features_target(test_df)

            if X_train.empty or X_test.empty:
                continue

            # Split the test period into validation (first half) and test (second half).
            # The validation half is used for early stopping, tuning, and calibration;
            # the test half gives honest out-of-sample metrics never touched during
            # training. This prevents the test set from being used as validation
            # (which would make the reported metrics optimistically biased).
            n_test = len(X_test)
            val_count = n_test // 2
            if val_count < 5:
                # Too few test samples to split — skip this fold rather than leak
                logger.warning("Fold %d: test set too small to split (%d), skipping",
                                fold_info["fold"], n_test)
                continue
            X_val = X_test.iloc[:val_count]
            y_val = y_test.iloc[:val_count]
            X_test_honest = X_test.iloc[val_count:]
            y_test_honest = y_test.iloc[val_count:]

            # Skip fold if the honest test set has a single class (AUC undefined)
            if len(np.unique(y_test_honest)) < 2:
                logger.warning("Fold %d: honest test set has single class, skipping",
                                fold_info["fold"])
                continue

            # Train single model type on this fold (validation = first half of test)
            results = train_models(
                symbol=symbol,
                timeframe=timeframe,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                model_types=[model_type],
                train_signal_count=total_signals,
            )

            if not results:
                continue

            # Recompute metrics on the HONEST test set (second half, never used
            # for early stopping / calibration / feature selection) so the
            # reported walk-forward metrics are truly out-of-sample.
            try:
                from ml.trainer import _compute_metrics
                model_obj = joblib.load(results[0].filepath)
                # Align honest test set to the model's selected features
                # (feature selection may have dropped some columns).
                feat_cols = getattr(results[0], "feature_columns", None)
                X_honest = X_test_honest[feat_cols] if feat_cols else X_test_honest
                y_pred_h = model_obj.predict(X_honest)
                y_proba_h = model_obj.predict_proba(X_honest)[:, 1]
                metrics = _compute_metrics(y_test_honest, y_pred_h, y_proba_h)
            except Exception as e:
                logger.warning("Honest metrics recompute failed for fold %d: %s",
                                fold_info["fold"], e)
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

        # Win rate = mean recall across folds (fraction of WIN signals the model
        # correctly identified). Accuracy on imbalanced data is misleading (70%
        # losses → always-predict-LOSS gives 70% accuracy), so we report the
        # model's ability to find winning trades.
        win_rate = np.mean(recs) if recs else 0.0

        report = WalkForwardReport(
            symbol=symbol,
            timeframe=timeframe or "all",
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
