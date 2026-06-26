"""Live prediction: evaluates the probability that a rule-based signal will succeed.

Uses soft-voting ensemble: loads ALL active models for the symbol/timeframe
and averages their predicted probabilities, weighted by AUC. Falls back to
single-model prediction if only one model is available.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from config.settings import SETTINGS
from data.models import MLModelMeta
from indicators.features import FeatureSet
from ml.dataset import SESSION_CATEGORIES, SESSION_ENC_COL, TRADE_MODES
from signals.engine import TradeSignal

logger = logging.getLogger(__name__)

ML_CONFIG = SETTINGS.ml
MODEL_DIR = ML_CONFIG.model_dir

# Minimum AUC for a model to be included in the ensemble.
# Models below this threshold are skipped — they add noise, not signal.
MIN_ENSEMBLE_AUC = 0.62

# Model types that require an external library.  Checked once at startup
# so that models whose library is missing are never even attempted.
_MODEL_LIBRARY_MAP = {
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "histgbrt": "sklearn.ensemble",  # HistGradientBoosting is part of sklearn
}

_UNAVAILABLE_MODEL_TYPES: set[str] = set()


def _check_model_libraries() -> None:
    """Populate _UNAVAILABLE_MODEL_TYPES with model types whose library is not installed."""
    for model_type, module_name in _MODEL_LIBRARY_MAP.items():
        try:
            __import__(module_name)
        except (ImportError, OSError):
            # ImportError: package not installed
            # OSError: package installed but native DLL missing (e.g. lightgbm)
            _UNAVAILABLE_MODEL_TYPES.add(model_type)
            logger.warning(
                "ML library '%s' not importable — %s models will be skipped",
                module_name, model_type,
            )


_check_model_libraries()


@dataclass(frozen=True)
class Prediction:
    """ML-enhanced prediction for a trade signal."""
    signal: str
    confidence: float
    probability: float
    model_type: str
    model_version: int
    feature_count: int


def _extract_feature_names(model) -> Optional[list[str]]:
    """Extract feature names from a model in priority order.

    Checks multiple sources since different model types store features
    differently. CalibratedClassifierCV wraps the inner model, so we
    need to check the inner estimators.

    Priority:
    1. model.feature_names_in_ (sklearn standard)
    2. model.calibrated_classifiers_[0].estimator.feature_names_in_ (CalibratedClassifierCV inner)
    3. model.estimator.feature_names_in_ (CalibratedClassifierCV unfitted template — less reliable)
    4. model.feature_names_ (CatBoost native)
    5. model.calibrated_classifiers_[0].estimator.feature_names_ (CalibratedCV + CatBoost)
    6. model.estimator.feature_names_ (CalibratedCV template + CatBoost)
    """
    # 1. sklearn standard — set after fit() on XGBClassifier, HistGradientBoosting,
    #    and CalibratedClassifierCV (which sets it from the fit data)
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)

    # 2. CalibratedClassifierCV: check fitted inner estimators
    #    calibrated_classifiers_ is a list of _CalibratedClassifier instances,
    #    each with an .estimator attribute that is the fitted base model.
    if hasattr(model, 'calibrated_classifiers_'):
        try:
            inner = model.calibrated_classifiers_[0].estimator
            if hasattr(inner, 'feature_names_in_'):
                return list(inner.feature_names_in_)
            # CatBoost inner estimator
            if hasattr(inner, 'feature_names_'):
                cb_features = inner.feature_names_
                if cb_features is not None and len(cb_features) > 0:
                    return list(cb_features)
        except (IndexError, AttributeError):
            pass

    # 3. CalibratedClassifierCV.estimator (unfitted template)
    #    Less reliable since it's the constructor arg, not a fitted model,
    #    but may have feature_names_in_ if it was pre-fitted.
    if hasattr(model, 'estimator'):
        inner = model.estimator
        if hasattr(inner, 'feature_names_in_'):
            return list(inner.feature_names_in_)
        # CatBoost template
        if hasattr(inner, 'feature_names_'):
            cb_features = inner.feature_names_
            if cb_features is not None and len(cb_features) > 0:
                return list(cb_features)

    # 4. CatBoost native feature_names_ (direct model, no wrapper)
    if hasattr(model, 'feature_names_'):
        cb_features = model.feature_names_
        if cb_features is not None and len(cb_features) > 0:
            return list(cb_features)

    return None


def _load_active_models(symbol: str, timeframe: str) -> list[tuple]:
    """Load all active models for a symbol/timeframe, ordered by AUC.

    Returns list of (model, feature_columns, model_info) tuples.
    Models with AUC below MIN_ENSEMBLE_AUC or with missing libraries are skipped.
    If loading fails, the model is deactivated and the next one is tried.

    When calibration is enabled and a calibration file exists, the raw model
    is replaced with its calibrated wrapper (CalibratedClassifierCV) which
    produces more accurate probability estimates.
    """
    from data.database import get_session
    models = []

    try:
        with get_session() as session:
            # Get ALL active models for this symbol/timeframe, ordered by AUC.
            query = session.query(MLModelMeta).filter(
                MLModelMeta.symbol == symbol,
                MLModelMeta.timeframe == timeframe,
                MLModelMeta.is_active == True,
                MLModelMeta.roc_auc >= MIN_ENSEMBLE_AUC,
            )
            if _UNAVAILABLE_MODEL_TYPES:
                query = query.filter(
                    MLModelMeta.model_type.notin_(_UNAVAILABLE_MODEL_TYPES)
                )
            candidates = query.order_by(MLModelMeta.roc_auc.desc()).all()

            if not candidates:
                return []

            for meta in candidates:
                model_info = {
                    "model_type": meta.model_type,
                    "version": meta.version,
                    "filepath": meta.filepath,
                    "accuracy": meta.accuracy,
                    "precision": meta.precision,
                    "recall": meta.recall,
                    "f1": meta.f1,
                    "roc_auc": meta.roc_auc,
                    "train_signal_count": meta.train_signal_count,
                    "trained_at": meta.trained_at,
                }

                filepath = model_info["filepath"]
                if not os.path.isabs(filepath):
                    filepath = os.path.join(MODEL_DIR, os.path.basename(filepath))
                if not os.path.exists(filepath):
                    logger.warning("Model file not found: %s — skipping", filepath)
                    continue

                try:
                    model = joblib.load(filepath)
                except Exception as load_err:
                    logger.warning(
                        "Failed to load %s model (AUC=%.4f): %s — deactivating",
                        meta.model_type, meta.roc_auc, load_err,
                    )
                    try:
                        meta.is_active = False
                        session.commit()
                        logger.info("Deactivated broken %s model for %s", meta.model_type, symbol)
                    except Exception:
                        session.rollback()
                    continue

                # Try loading calibration if enabled
                if ML_CONFIG.use_calibration:
                    try:
                        from ml.calibration import load_calibration
                        calibrated = load_calibration(
                            meta.model_type, symbol, timeframe, meta.version,
                        )
                        if calibrated is not None:
                            model = calibrated
                            logger.debug("Using calibrated model for %s v%d", meta.model_type, meta.version)
                    except Exception as e:
                        logger.debug("No calibration for %s v%d: %s", meta.model_type, meta.version, e)

                # Determine feature columns from the most reliable source.
                # The model itself knows what features it was trained on —
                # this prevents count and ordering mismatches with _features.json.
                feature_cols = _extract_feature_names(model)

                # Fall back to _features.json sidecar file
                if feature_cols is None:
                    feat_path = filepath.replace(".joblib", "_features.json")
                    if os.path.exists(feat_path):
                        with open(feat_path) as f:
                            feature_cols = json.load(f)

                if feature_cols:
                    logger.debug(
                        "Model %s v%d: %d features from %s",
                        meta.model_type, meta.version, len(feature_cols),
                        "model" if _extract_feature_names(model) else "json",
                    )

                models.append((model, feature_cols, model_info))

    except Exception as e:
        logger.error("Failed to load models: %s", e)

    return models


def _build_feature_vector(signal: TradeSignal, feature_cols: Optional[list]) -> pd.DataFrame:
    """Build a feature DataFrame from a signal's feature snapshot.

    Handles session encoding, trade mode encoding, and NaN filling
    with per-column neutral defaults. Backward-compatible with old
    models that use ordinal session encoding.
    """
    features = signal.features.to_dict()

    # Encode session — backward compatible with old models
    session = features.pop("session", None) or ""
    if feature_cols and SESSION_ENC_COL in feature_cols:
        # Legacy model: ordinal encoding
        session_encoding = {"ASIAN": 0, "LONDON": 1, "NEW_YORK": 2}
        features[SESSION_ENC_COL] = session_encoding.get(session, -1)
    else:
        # New model: one-hot encoding
        for cat in SESSION_CATEGORIES:
            features[f"session_{cat.lower()}"] = 1 if session == cat else 0

    # Encode trade_mode with one-hot encoding
    trade_mode = getattr(signal, "trade_mode", "") or ""
    for mode in TRADE_MODES:
        features[f"mode_{mode}"] = 1 if trade_mode == mode else 0

    # Build DataFrame with explicit column ordering and guaranteed presence.
    # Use 0.0 for any feature missing from the signal's feature dict,
    # and replace None values with their neutral defaults.
    PREDICT_NAN_DEFAULTS = {
        "rsi14": 50.0, "macd_line": 0.0, "macd_signal": 0.0,
        "macd_histogram": 0.0, "momentum_score": 0.0, "trend_strength": 0.0,
        "dist_ema20": 0.0, "dist_ema50": 0.0, "dist_ema200": 0.0,
        "candle_body_pct": 0.0, "upper_wick_pct": 0.0, "lower_wick_pct": 0.0,
        "spread": 0.0, "atr_ratio": 1.0,
        "session_asian": 0, "session_london": 0, "session_new_york": 0,
        "session_enc": -1,
        "mode_scalping": 0, "mode_day_trading": 0, "mode_swing": 0,
    }

    if feature_cols:
        # Build row dict: get value from features, replace None with default,
        # fall back to default for missing keys. This guarantees every column
        # the model expects is present, even if the signal doesn't have it.
        row = {}
        for col in feature_cols:
            val = features.get(col)
            if val is None:
                val = PREDICT_NAN_DEFAULTS.get(col, 0.0)
            row[col] = val
        X = pd.DataFrame([row], columns=feature_cols)
    else:
        X = pd.DataFrame([features])

    # Fill any remaining NaN/None values
    for col, default in PREDICT_NAN_DEFAULTS.items():
        if col in X.columns:
            X[col] = X[col].fillna(default)
    X = X.fillna(0.0)
    X = X.replace([np.inf, -np.inf], 0.0)

    # Final safety: ensure column order exactly matches what the model expects.
    # This is critical for CatBoost which checks feature position, not just name.
    if feature_cols:
        X = X[feature_cols]

    return X


def predict_signal(signal: TradeSignal) -> Optional[Prediction]:
    """Evaluate a trade signal using soft-voting ensemble of ML models.

    Loads all active models for the symbol/timeframe, predicts with each,
    and averages their probabilities weighted by AUC. Falls back to a
    single model if only one is available.

    ML does NOT generate signals. It only evaluates the probability
    that the rule-based signal will succeed.

    Args:
        signal: The rule-based TradeSignal to evaluate.

    Returns:
        Prediction with ML-adjusted confidence, or None if no model available.
    """
    # Try "all" timeframe models first, then signal's own timeframe
    models = _load_active_models(signal.symbol, "all")
    if not models:
        models = _load_active_models(signal.symbol, signal.timeframe)
    if not models:
        logger.info("No active ML model for %s %s", signal.symbol, signal.timeframe)
        return None

    # Try each model, skipping those with incompatible feature sets.
    # Old models may have different feature columns than the current pipeline.
    probabilities = []
    auc_weights = []
    model_types = []
    best_auc = 0
    best_version = 0
    feature_count = 0

    for model, feature_cols, model_info in models:
        try:
            X = _build_feature_vector(signal, feature_cols)
            proba = model.predict_proba(X)[0, 1]
            probabilities.append(proba)
            auc_weights.append(model_info["roc_auc"])
            model_types.append(model_info["model_type"])
            if model_info["roc_auc"] > best_auc:
                best_auc = model_info["roc_auc"]
                best_version = model_info["version"]
            feature_count = X.shape[1]
        except Exception as e:
            logger.warning(
                "Prediction failed for %s v%d: %s — skipping",
                model_info["model_type"], model_info["version"], e,
            )
            continue

    if not probabilities:
        logger.error("All models failed for %s %s — using rule-based confidence only", signal.symbol, signal.timeframe)
        return None

    # Ensemble or single model result
    model_version = best_version
    if len(probabilities) == 1:
        proba = probabilities[0]
        ml_confidence = signal.confidence * proba
        model_type_str = model_types[0]
    else:
        # AUC-weighted average probability
        total_weight = sum(auc_weights)
        if total_weight > 0:
            proba = sum(p * w for p, w in zip(probabilities, auc_weights)) / total_weight
        else:
            proba = sum(probabilities) / len(probabilities)

        ml_confidence = signal.confidence * proba
        model_type_str = "+".join(model_types)

    logger.info(
        "Prediction: %d model(s) succeeded for %s",
        len(probabilities), signal.symbol,
    )
    if len(probabilities) > 1:
        logger.info(
            "Ensemble (%d models): probs=%s, weighted=%.4f",
            len(probabilities),
            [f"{p:.4f}" for p in probabilities],
            proba,
        )

    prediction = Prediction(
        signal=signal.signal.value,
        confidence=round(ml_confidence, 2),
        probability=round(float(proba), 4),
        model_type=model_type_str,
        model_version=model_version,
        feature_count=feature_count,
    )

    logger.info(
        "ML prediction: %s prob=%.4f ml_conf=%.2f (rule_conf=%.2f) model=%s_v%d",
        signal.signal.value, proba, ml_confidence, signal.confidence,
        model_type_str, model_version,
    )

    return prediction
