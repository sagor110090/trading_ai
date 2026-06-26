"""Probability calibration for ML models using Platt scaling.

Calibration maps raw predict_proba outputs to true probabilities.
Without calibration, a model outputting 0.15 might actually correspond
to a 30% win rate. After calibration, 0.15 means 15%.

Uses sklearn's CalibratedClassifierCV with cv=2 to fit a sigmoid
(Platt scaling) calibration. The base model is refit inside CV folds,
then calibrated on held-out data — no data leakage.

For sklearn >= 1.6, cv="prefit" was removed, so we use cv=2 which
refits the model internally. This is safe because calibration uses
the validation set only (separate from training).
"""

import json
import logging
import os

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from config.settings import SETTINGS

logger = logging.getLogger(__name__)

ML_CONFIG = SETTINGS.ml
MODEL_DIR = ML_CONFIG.model_dir


def calibrate_model(
    model,
    X_val: np.ndarray | list,
    y_val: np.ndarray | list,
    model_type: str,
    symbol: str,
    timeframe: str,
    version: int,
) -> object | None:
    """Fit Platt scaling calibration on validation data and save to disk.

    Uses CalibratedClassifierCV with cv=2, which refits the base model
    across 2 folds of the validation data and calibrates on held-out portions.
    This produces properly calibrated probability estimates.

    Args:
        model: Pre-fitted sklearn/xgboost/catboost model.
        X_val: Validation features.
        y_val: Validation labels (binary 0/1).
        model_type: e.g., "xgboost", "histgbrt", "catboost".
        symbol: e.g., "XAUUSD".
        timeframe: e.g., "all".
        version: Model version number.

    Returns:
        Calibrated model wrapper, or None if calibration fails.
    """
    if not ML_CONFIG.use_calibration:
        logger.info("Calibration disabled, skipping")
        return None

    try:
        # Check that we have both classes in validation data
        unique_classes = np.unique(y_val)
        if len(unique_classes) < 2:
            logger.warning(
                "Cannot calibrate %s: validation has only 1 class (%s), need 2",
                model_type, unique_classes,
            )
            return None

        # Need enough samples for cv=3 (at least 3 per class for 3-fold calibration)
        class_counts = np.bincount(y_val.astype(int))
        min_class_count = min(class_counts[class_counts > 0])
        if min_class_count < 3:
            logger.warning(
                "Cannot calibrate %s: minority class has only %d samples, need >= 3",
                model_type, min_class_count,
            )
            return None

        # Freeze the already-trained model so CalibratedClassifierCV keeps it
        # and ONLY fits the calibration layer (Platt sigmoid) on validation data.
        # The previous approach (cv=2 + model_class(**model_params)) discarded
        # the trained model and refit a fresh one on the small validation set,
        # producing a weaker calibrated model.
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(model),
            cv=3,
            method="sigmoid",
        )
        calibrated.fit(X_val, y_val)

        # Save calibrated model
        _ensure_model_dir()
        filename = f"{model_type}_{symbol}_{timeframe or 'all'}_v{version}_calibrated.joblib"
        filepath = os.path.join(MODEL_DIR, filename)
        joblib.dump(calibrated, filepath)
        logger.info("Saved calibration for %s to %s", model_type, filepath)

        return calibrated

    except Exception as e:
        logger.warning("Calibration failed for %s: %s", model_type, e)
        return None


def load_calibration(
    model_type: str,
    symbol: str,
    timeframe: str,
    version: int,
) -> object | None:
    """Load a saved calibration model from disk.

    Returns:
        CalibratedClassifierCV model, or None if not found.
    """
    filename = f"{model_type}_{symbol}_{timeframe or 'all'}_v{version}_calibrated.joblib"
    filepath = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(filepath):
        return None

    try:
        return joblib.load(filepath)
    except Exception as e:
        logger.warning("Failed to load calibration from %s: %s", filepath, e)
        return None


def _ensure_model_dir() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
