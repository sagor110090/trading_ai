"""ML trainer: trains XGBoost, LightGBM, and CatBoost models on verified signal data."""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.settings import SETTINGS
from data.database import get_session
from data.models import MLModelMeta
from ml.dataset import FEATURE_COLUMNS, build_dataset, split_features_target

logger = logging.getLogger(__name__)

ML_CONFIG = SETTINGS.ml
MODEL_DIR = ML_CONFIG.model_dir


@dataclass
class ModelMetrics:
    """Performance metrics for a trained model."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


@dataclass
class TrainingResult:
    """Result of training a single model type."""
    model_type: str
    metrics: ModelMetrics
    filepath: str
    version: int


def _ensure_model_dir() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)


def _train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Train XGBoost classifier and return metrics."""
    from xgboost import XGBClassifier

    model = XGBClassifier(**ML_CONFIG.xgboost)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    return {
        "model": model,
        "metrics": _compute_metrics(y_val, y_pred, y_proba),
    }


def _train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Train LightGBM classifier and return metrics."""
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(**ML_CONFIG.lightgbm)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    return {
        "model": model,
        "metrics": _compute_metrics(y_val, y_pred, y_proba),
    }


def _train_catboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Train CatBoost classifier and return metrics."""
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(**ML_CONFIG.catboost)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False,
    )
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    # CatBoost predict returns float arrays
    y_pred = y_pred.astype(int)

    return {
        "model": model,
        "metrics": _compute_metrics(y_val, y_pred, y_proba),
    }


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> ModelMetrics:
    """Compute classification metrics."""
    return ModelMetrics(
        accuracy=round(accuracy_score(y_true, y_pred), 4),
        precision=round(precision_score(y_true, y_pred, zero_division=0), 4),
        recall=round(recall_score(y_true, y_pred, zero_division=0), 4),
        f1=round(f1_score(y_true, y_pred, zero_division=0), 4),
        roc_auc=round(roc_auc_score(y_true, y_proba), 4),
    )


_TRAINERS = {
    "xgboost": _train_xgboost,
    "lightgbm": _train_lightgbm,
    "catboost": _train_catboost,
}


def train_models(
    symbol: str,
    timeframe: Optional[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_types: Optional[list[str]] = None,
    train_signal_count: Optional[int] = None,
) -> list[TrainingResult]:
    """Train all specified model types and save to disk.

    Args:
        symbol: Trading symbol (e.g., "XAUUSD").
        timeframe: Candle timeframe (e.g., "H1") or None for all-timeframe model.
        X_train, y_train: Training features and labels.
        X_val, y_val: Validation features and labels.
        model_types: List of model types to train. Defaults to all.

    Returns:
        List of TrainingResult for each trained model.
    """
    _ensure_model_dir()

    if model_types is None:
        model_types = list(_TRAINERS.keys())

    # Use "all" as label when no specific timeframe
    tf_label = timeframe or "all"

    results = []

    for model_type in model_types:
        trainer = _TRAINERS.get(model_type)
        if not trainer:
            logger.warning("Unknown model type: %s", model_type)
            continue

        logger.info("Training %s for %s %s...", model_type, symbol, tf_label)

        try:
            result = trainer(X_train, y_train, X_val, y_val)
        except Exception as e:
            logger.error("Failed to train %s: %s", model_type, e)
            continue

        model = result["model"]
        metrics = result["metrics"]

        # Determine version
        version = _get_next_version(symbol, tf_label, model_type)

        # Save model to disk
        filename = f"{model_type}_{symbol}_{tf_label}_v{version}.joblib"
        filepath = os.path.join(MODEL_DIR, filename)
        joblib.dump(model, filepath)
        logger.info("Saved %s model to %s", model_type, filepath)

        # Save feature columns for prediction alignment
        feat_path = filepath.replace(".joblib", "_features.json")
        with open(feat_path, "w") as f:
            json.dump(list(X_train.columns), f)

        # Store metadata in database (non-fatal if DB unavailable)
        try:
            # Replace NaN metrics with 0 — MySQL can't store NaN
            safe_metrics = ModelMetrics(
                accuracy=metrics.accuracy if not np.isnan(metrics.accuracy) else 0.0,
                precision=metrics.precision if not np.isnan(metrics.precision) else 0.0,
                recall=metrics.recall if not np.isnan(metrics.recall) else 0.0,
                f1=metrics.f1 if not np.isnan(metrics.f1) else 0.0,
                roc_auc=metrics.roc_auc if not np.isnan(metrics.roc_auc) else 0.0,
            )
            _store_model_meta(
                model_type=model_type,
                symbol=symbol,
                timeframe=tf_label,
                version=version,
                filepath=filepath,
                metrics=safe_metrics,
                train_signal_count=train_signal_count,
            )
        except Exception as e:
            logger.warning("Failed to store model metadata in DB: %s", e)

        results.append(TrainingResult(
            model_type=model_type,
            metrics=metrics,
            filepath=filepath,
            version=version,
        ))

        logger.info(
            "%s metrics: acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f",
            model_type, metrics.accuracy, metrics.precision,
            metrics.recall, metrics.f1, metrics.roc_auc,
        )

    return results


def _get_next_version(symbol: str, timeframe: Optional[str], model_type: str) -> int:
    """Get the next version number for a model type."""
    try:
        with get_session() as session:
            latest = (
                session.query(MLModelMeta)
                .filter(
                    MLModelMeta.model_type == model_type,
                    MLModelMeta.symbol == symbol,
                    MLModelMeta.timeframe == timeframe,
                )
                .order_by(MLModelMeta.version.desc())
                .first()
            )
            version = (latest.version + 1) if latest else 1

            # Check if this version already exists (race condition or import)
            existing = session.query(MLModelMeta).filter(
                MLModelMeta.model_type == model_type,
                MLModelMeta.symbol == symbol,
                MLModelMeta.timeframe == timeframe,
                MLModelMeta.version == version,
            ).first()
            if existing:
                # Find the next available version
                row = session.query(MLModelMeta.version).filter(
                    MLModelMeta.model_type == model_type,
                    MLModelMeta.symbol == symbol,
                    MLModelMeta.timeframe == timeframe,
                ).order_by(MLModelMeta.version.desc()).first()
                version = (row[0] + 1) if row else 1

            return version
    except Exception:
        return 1


def _store_model_meta(
    model_type: str,
    symbol: str,
    timeframe: Optional[str],
    version: int,
    filepath: str,
    metrics: ModelMetrics,
    train_signal_count: Optional[int] = None,
) -> None:
    """Store model metadata in the database."""
    with get_session() as session:
        # Deactivate previous active models for same type/symbol/timeframe
        session.query(MLModelMeta).filter(
            MLModelMeta.model_type == model_type,
            MLModelMeta.symbol == symbol,
            MLModelMeta.timeframe == timeframe,
            MLModelMeta.is_active == True,
        ).update({"is_active": False})

        # Check if this version already exists (from import)
        existing = session.query(MLModelMeta).filter(
            MLModelMeta.model_type == model_type,
            MLModelMeta.symbol == symbol,
            MLModelMeta.timeframe == timeframe,
            MLModelMeta.version == version,
        ).first()

        if existing:
            # Update existing record
            from datetime import datetime
            existing.filepath = filepath
            existing.accuracy = metrics.accuracy
            existing.precision = metrics.precision
            existing.recall = metrics.recall
            existing.f1 = metrics.f1
            existing.roc_auc = metrics.roc_auc
            existing.train_signal_count = train_signal_count
            existing.trained_at = datetime.utcnow()
            existing.is_active = True
        else:
            # Insert new record
            meta = MLModelMeta(
                model_type=model_type,
                symbol=symbol,
                timeframe=timeframe,
                version=version,
                filepath=filepath,
                accuracy=metrics.accuracy,
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
                roc_auc=metrics.roc_auc,
                train_signal_count=train_signal_count,
                is_active=True,
            )
            session.add(meta)