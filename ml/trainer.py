"""ML trainer: trains XGBoost, HistGradientBoosting, and CatBoost models on verified signal data.

Pipeline enhancements:
- SMOTE oversampling for class imbalance (70%+ LOSS)
- Optuna hyperparameter tuning (Bayesian search)
- Feature selection (drop <1% importance features and retrain)
- Probability calibration (Platt scaling)
"""

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
from ml.smote import apply_smote

logger = logging.getLogger(__name__)

# Surface training-progress INFO records (per-model "Training…", metrics,
# feature selection) even when the root logger is at WARNING — e.g. when run
# via backtest_historical.py. Otherwise a long Optuna tuning or model fit step
# produces no output and looks like a freeze.
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)

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
    feature_columns: list = None


def _ensure_model_dir() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)


def _train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, params: Optional[dict] = None) -> dict:
    """Train XGBoost classifier and return metrics.

    Targets are binary (0=LOSS, 1=WIN). Class imbalance is handled by SMOTE
    upstream (apply_smote), so we do NOT apply scale_pos_weight here —
    double-balancing (SMOTE + class weights) over-corrects toward WIN and
    inflates probabilities.
    """
    from xgboost import XGBClassifier

    final_params = {**ML_CONFIG.xgboost, **(params or {})}
    model = XGBClassifier(**final_params)
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


def _train_histgbrt(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, params: Optional[dict] = None) -> dict:
    """Train sklearn HistGradientBoosting classifier and return metrics.

    Class imbalance is handled by SMOTE upstream, so class_weight is NOT set
    here (double-balancing would over-correct). Built-in early stopping
    guards against overfitting on the validation set.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    final_params = {**ML_CONFIG.histgbrt, **(params or {})}
    model = HistGradientBoostingClassifier(**final_params)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    return {
        "model": model,
        "metrics": _compute_metrics(y_val, y_pred, y_proba),
    }


def _train_catboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, params: Optional[dict] = None) -> dict:
    """Train CatBoost classifier and return metrics.

    Targets are binary (0=LOSS, 1=WIN). Class imbalance is handled by SMOTE
    upstream, so auto_class_weights is NOT set here (double-balancing would
    over-correct toward WIN and inflate probabilities).
    """
    from catboost import CatBoostClassifier

    final_params = {**ML_CONFIG.catboost, **(params or {})}
    model = CatBoostClassifier(**final_params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False,
    )
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    return {
        "model": model,
        "metrics": _compute_metrics(y_val, y_pred.flatten(), y_proba),
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
    "histgbrt": _train_histgbrt,
    "catboost": _train_catboost,
}


def _log_feature_importance(model, model_type: str, feature_names: list[str]) -> None:
    """Extract and log feature importance from a trained model.

    Warns about features with near-zero importance (< 1% of total).
    """
    try:
        if model_type == "xgboost":
            importances = model.feature_importances_
        elif model_type == "catboost":
            importances = model.feature_importances_
        elif model_type == "histgbrt":
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                logger.info("HistGradientBoosting: feature_importances_ not available")
                return
        else:
            return

        total = sum(importances)
        if total == 0:
            return

        importance_dict = {
            name: float(imp / total)
            for name, imp in zip(feature_names, importances)
        }
        sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        # Log top 10 features
        top_10 = ", ".join(f"{k}:{v:.3f}" for k, v in sorted_imp[:10])
        logger.info("Top features for %s: %s", model_type, top_10)

        # Warn about near-zero features
        low_importance = [k for k, v in importance_dict.items() if v < 0.01]
        if low_importance:
            logger.warning(
                "Low importance features in %s (<1%%): %s",
                model_type, low_importance,
            )
    except Exception as e:
        logger.warning("Failed to log feature importance for %s: %s", model_type, e)


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
    """Train all specified model types with SMOTE, tuning, feature selection, and calibration.

    Pipeline:
    1. SMOTE oversampling on training data (if enabled)
    2. Optuna hyperparameter tuning (if enabled)
    3. Train model with best or default params
    4. Feature selection: drop <1% importance features and retrain
    5. Probability calibration (Platt scaling on validation data)

    Args:
        symbol: Trading symbol (e.g., "XAUUSD").
        timeframe: Candle timeframe (e.g., "H1") or None for all-timeframe model.
        X_train, y_train: Training features and labels.
        X_val, y_val: Validation features and labels.
        model_types: List of model types to train. Defaults to all.
        train_signal_count: Number of signals used for training (metadata).

    Returns:
        List of TrainingResult for each trained model.
    """
    _ensure_model_dir()

    if model_types is None:
        model_types = list(_TRAINERS.keys())

    # Use "all" as label when no specific timeframe
    tf_label = timeframe or "all"

    # Step 0: SMOTE oversampling on training data
    X_train_smoted, y_train_smoted = apply_smote(
        X_train, y_train, use_smote=ML_CONFIG.use_smote,
    )

    results = []

    for model_type in model_types:
        trainer = _TRAINERS.get(model_type)
        if not trainer:
            logger.warning("Unknown model type: %s", model_type)
            continue

        logger.info("Training %s for %s %s...", model_type, symbol, tf_label)

        # Use SMOTE-resampled data for training, original data for validation
        X_tr, y_tr = X_train_smoted, y_train_smoted
        X_vl, y_vl = X_val, y_val

        # Step 1: Optuna hyperparameter tuning
        tuned_params = None
        if ML_CONFIG.use_tuning:
            try:
                from ml.tuner import load_tuned_params, save_tuned_params, tune_hyperparams

                tuned_params = load_tuned_params(model_type, symbol, tf_label)
                if tuned_params is None:
                    tuned_params = tune_hyperparams(
                        model_type, X_tr, y_tr, X_vl, y_vl, n_trials=50,
                    )
                    if tuned_params:
                        save_tuned_params(tuned_params, model_type, symbol, tf_label)
                else:
                    logger.info("Using saved tuned params for %s", model_type)
            except Exception as e:
                logger.warning("Hyperparameter tuning failed for %s: %s", model_type, e)

        try:
            result = trainer(X_tr, y_tr, X_vl, y_vl, params=tuned_params)
        except Exception as e:
            logger.error("Failed to train %s: %s", model_type, e)
            continue

        model = result["model"]
        metrics = result["metrics"]

        # Step 2: Feature selection — drop <1% importance and retrain
        selected_cols = list(X_tr.columns)
        if ML_CONFIG.use_feature_selection:
            sel_model, sel_metrics, sel_cols = _select_features(
                model, model_type, X_tr, y_tr, X_vl, y_vl, trainer, params=tuned_params,
            )
            # Only use selected-feature results if they succeeded
            if sel_cols is not None:
                selected_cols = sel_cols
                model = sel_model
                if sel_metrics is not None:
                    metrics = sel_metrics
            if len(selected_cols) < len(X_tr.columns):
                logger.info(
                    "Feature selection for %s: %d → %d features",
                    model_type, len(X_tr.columns), len(selected_cols),
                )

        # Determine version (needed for calibration filename)
        version = _get_next_version(symbol, tf_label, model_type)

        # Step 3: Probability calibration (Platt scaling)
        # IMPORTANT: align validation set to the model's selected features.
        # After feature selection the model may expect fewer columns than X_vl,
        # so predict_proba would raise a feature_names mismatch.
        X_vl_aligned = X_vl[selected_cols] if selected_cols else X_vl
        calibrated_model = None
        if ML_CONFIG.use_calibration:
            try:
                from ml.calibration import calibrate_model

                calibrated_model = calibrate_model(
                    model, X_vl_aligned, y_vl, model_type, symbol, tf_label, version,
                )
            except Exception as e:
                logger.warning("Calibration failed for %s: %s", model_type, e)

        # Use calibrated model for final metrics if available
        if calibrated_model is not None:
            try:
                y_pred = calibrated_model.predict(X_vl_aligned)
                y_proba = calibrated_model.predict_proba(X_vl_aligned)[:, 1]
                metrics = _compute_metrics(y_vl, y_pred, y_proba)
                logger.info(
                    "%s calibrated metrics: acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f",
                    model_type, metrics.accuracy, metrics.precision,
                    metrics.recall, metrics.f1, metrics.roc_auc,
                )
            except Exception as e:
                logger.warning("Calibrated metrics computation failed for %s: %s", model_type, e)

        # Save model to disk (original model, not calibrated wrapper)
        filename = f"{model_type}_{symbol}_{tf_label}_v{version}.joblib"
        filepath = os.path.join(MODEL_DIR, filename)
        joblib.dump(model, filepath)
        logger.info("Saved %s model to %s", model_type, filepath)

        # Save feature columns for prediction alignment
        feat_path = filepath.replace(".joblib", "_features.json")
        with open(feat_path, "w") as f:
            json.dump(selected_cols, f)

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
            # Store the bare filename (not the absolute path) so the DB row is
            # portable across machines/OSes — the predictor resolves it against
            # the per-machine MODEL_DIR at load time.
            _store_model_meta(
                model_type=model_type,
                symbol=symbol,
                timeframe=tf_label,
                version=version,
                filepath=filename,
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
            feature_columns=selected_cols,
        ))

        logger.info(
            "%s metrics: acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f",
            model_type, metrics.accuracy, metrics.precision,
            metrics.recall, metrics.f1, metrics.roc_auc,
        )

        # Log feature importance
        _log_feature_importance(model, model_type, selected_cols)

    return results


def _select_features(
    model,
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    trainer_fn,
    params: Optional[dict] = None,
) -> tuple:
    """Drop features with <1% importance and retrain if beneficial.

    Returns:
        (model, metrics, selected_columns) — retrained model with fewer
        features if AUC improved, or original model if not.
    """
    try:
        if model_type == "xgboost":
            importances = model.feature_importances_
        elif model_type == "catboost":
            importances = model.feature_importances_
        elif model_type == "histgbrt":
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                return model, None, None
        else:
            return model, None, None

        total = sum(importances)
        if total == 0:
            return model, None, None

        # Keep features with >= 1% importance
        importance_dict = {
            name: float(imp / total)
            for name, imp in zip(X_train.columns, importances)
        }
        selected = [col for col, imp in importance_dict.items() if imp >= 0.01]

        # Don't reduce to fewer than 5 features
        if len(selected) < 5:
            logger.info(
                "Feature selection for %s: only %d features >= 1%%, keeping all",
                model_type, len(selected),
            )
            return model, None, list(X_train.columns)

        # If all features pass, no need to retrain
        if len(selected) == len(X_train.columns):
            return model, None, list(X_train.columns)

        # Retrain with selected features only
        X_train_sel = X_train[selected]
        X_val_sel = X_val[selected]
        result = trainer_fn(X_train_sel, y_train, X_val_sel, y_val, params=params)

        # Only use selected-feature model if AUC didn't drop significantly
        new_auc = result["metrics"].roc_auc
        y_proba_old = model.predict_proba(X_val)[:, 1]
        try:
            old_auc = round(roc_auc_score(y_val, y_proba_old), 4)
        except Exception:
            # If old AUC can't be computed (e.g. single class in y_val),
            # don't accept the new model blindly — keep the original.
            logger.warning("Could not compute old AUC for %s, keeping all features", model_type)
            return model, None, list(X_train.columns)

        if old_auc > 0 and new_auc < old_auc - 0.01:
            logger.info(
                "Feature selection for %s: AUC dropped (%.4f → %.4f), keeping all features",
                model_type, old_auc, new_auc,
            )
            return model, None, list(X_train.columns)

        logger.info(
            "Feature selection for %s: %d → %d features (AUC %.4f → %.4f)",
            model_type, len(X_train.columns), len(selected),
            old_auc, new_auc,
        )
        return result["model"], result["metrics"], selected

    except Exception as e:
        logger.warning("Feature selection failed for %s: %s", model_type, e)
        return model, None, None


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
