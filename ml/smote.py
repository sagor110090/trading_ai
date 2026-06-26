"""SMOTE oversampling for imbalanced classification datasets.

Applies Synthetic Minority Oversampling Technique only to training data
(never validation) to balance WIN/LOSS classes. Falls back to random
oversampling for very small minority classes (< 6 samples).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.utils import resample

logger = logging.getLogger(__name__)


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_smote: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE oversampling to balance minority class in training data.

    Args:
        X_train: Training features.
        y_train: Training labels (binary 0/1).
        use_smote: If False, return data unchanged.

    Returns:
        Resampled (X_train, y_train) with balanced classes.
    """
    if not use_smote:
        return X_train, y_train

    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())

    if pos_count == 0 or neg_count == 0:
        logger.warning("Cannot oversample: one class has 0 samples (neg=%d, pos=%d)", neg_count, pos_count)
        return X_train, y_train

    # If already balanced (within 10%), skip oversampling
    ratio = min(pos_count, neg_count) / max(pos_count, neg_count)
    if ratio >= 0.9:
        logger.info("Classes already balanced (%.1f%% ratio), skipping SMOTE", ratio * 100)
        return X_train, y_train

    minority_class = 1 if pos_count < neg_count else 0
    minority_count = min(pos_count, neg_count)

    # For very small minority (< 6 samples), use random oversampling instead
    # because SMOTE's k_neighbors requires at least k+1 minority samples
    if minority_count < 6:
        logger.info("Minority class too small for SMOTE (%d samples), using random oversampling", minority_count)
        return _random_oversample(X_train, y_train, minority_class)

    try:
        from imblearn.over_sampling import SMOTE

        k_neighbors = min(5, minority_count - 1)
        smote = SMOTE(sampling_strategy="auto", k_neighbors=k_neighbors, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # SMOTE generates fractional values for one-hot encoded columns
        # (e.g., session_asian=0.4), which breaks the binary semantics.
        # Round one-hot columns back to {0, 1} so synthetic samples are valid.
        onehot_cols = [c for c in X_resampled.columns if c.startswith("session_") or c.startswith("mode_")]
        for col in onehot_cols:
            X_resampled[col] = X_resampled[col].round().astype(int)

        new_pos = int((y_resampled == 1).sum())
        new_neg = int((y_resampled == 0).sum())
        logger.info(
            "SMOTE: %d/%d (neg/pos) → %d/%d (k_neighbors=%d)",
            neg_count, pos_count, new_neg, new_pos, k_neighbors,
        )

        return X_resampled, y_resampled

    except ImportError:
        logger.warning("imbalanced-learn not installed, falling back to random oversampling")
        return _random_oversample(X_train, y_train, minority_class)


def _random_oversample(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    minority_class: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Simple random oversampling: duplicate minority samples with replacement."""
    majority_class = 1 - minority_class

    df = X_train.copy()
    df["_target"] = y_train.values

    minority_df = df[df["_target"] == minority_class]
    majority_df = df[df["_target"] == majority_class]

    # Oversample minority to match majority count
    n_target = len(majority_df)
    minority_upsampled = resample(
        minority_df,
        replace=True,
        n_samples=n_target,
        random_state=42,
    )

    resampled = pd.concat([majority_df, minority_upsampled]).sample(frac=1, random_state=42)
    y_resampled = resampled["_target"]
    X_resampled = resampled.drop(columns=["_target"])

    new_pos = int((y_resampled == 1).sum())
    new_neg = int((y_resampled == 0).sum())
    logger.info("Random oversample: → %d/%d (neg/pos)", new_neg, new_pos)

    return X_resampled, y_resampled
