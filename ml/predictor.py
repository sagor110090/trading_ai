"""Live prediction: evaluates the probability that a rule-based signal will succeed."""

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


def _load_active_model(symbol: str, timeframe: str) -> Optional[tuple]:
    """Load the best active model for a symbol/timeframe.

    Tries models ordered by best AUC first. If loading a model fails
    (e.g. missing library like lightgbm), it automatically falls back
    to the next best model instead of giving up entirely.

    Returns (model, feature_columns, model_info) or None,
    where model_info is a dict with the model metadata.
    """
    from data.database import get_session
    try:
        with get_session() as session:
            # Get ALL active models for this symbol/timeframe, ordered by AUC.
            # Exclude model types whose library is not installed (e.g. lightgbm).
            query = session.query(MLModelMeta).filter(
                MLModelMeta.symbol == symbol,
                MLModelMeta.timeframe == timeframe,
                MLModelMeta.is_active == True,
            )
            if _UNAVAILABLE_MODEL_TYPES:
                query = query.filter(
                    MLModelMeta.model_type.notin_(_UNAVAILABLE_MODEL_TYPES)
                )
            candidates = query.order_by(MLModelMeta.roc_auc.desc()).all()

            if not candidates:
                return None

            for meta in candidates:
                # Extract all needed data while session is open
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
                # Resolve relative paths against MODEL_DIR
                if not os.path.isabs(filepath):
                    filepath = os.path.join(MODEL_DIR, os.path.basename(filepath))
                if not os.path.exists(filepath):
                    logger.warning("Model file not found: %s — skipping", filepath)
                    continue

                try:
                    model = joblib.load(filepath)
                except Exception as load_err:
                    logger.warning(
                        "Failed to load %s model (AUC=%.4f): %s — deactivating and trying next",
                        meta.model_type, meta.roc_auc, load_err,
                    )
                    # Deactivate the broken model so we don't retry it every cycle
                    try:
                        meta.is_active = False
                        session.commit()
                        logger.info("Deactivated broken %s model for %s", meta.model_type, symbol)
                    except Exception:
                        session.rollback()
                    continue

                feat_path = filepath.replace(".joblib", "_features.json")
                if os.path.exists(feat_path):
                    with open(feat_path) as f:
                        feature_cols = json.load(f)
                else:
                    feature_cols = None

                return model, feature_cols, model_info

            # All candidates failed
            logger.warning("No loadable model found for %s %s", symbol, timeframe)
            return None

    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return None


def predict_signal(signal: TradeSignal) -> Optional[Prediction]:
    """Evaluate a trade signal using the best available ML model.

    ML does NOT generate signals. It only evaluates the probability
    that the rule-based signal will succeed.

    Looks for models trained with timeframe="all" first (confluence/multi-TF),
    then falls back to models matching the signal's own timeframe.

    Args:
        signal: The rule-based TradeSignal to evaluate.

    Returns:
        Prediction with ML-adjusted confidence, or None if no model available.
    """
    # Prefer the "all" timeframe model (trained on all signals for the symbol)
    result = _load_active_model(signal.symbol, "all")
    if result is None:
        # Fallback: try signal's own timeframe
        result = _load_active_model(signal.symbol, signal.timeframe)
    if result is None:
        logger.info("No active ML model for %s %s", signal.symbol, signal.timeframe)
        return None

    model, feature_cols, model_info = result

    # Build feature vector from signal's feature snapshot
    features = signal.features.to_dict()

    # Encode session — backward compatible with old models
    # Old models use ordinal "session_enc"; new models use one-hot "session_*"
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

    # Build DataFrame with correct feature order
    if feature_cols:
        row = {}
        for col in feature_cols:
            row[col] = features.get(col, 0.0)
        X = pd.DataFrame([row])
    else:
        X = pd.DataFrame([features])

    # Replace NaN/None with per-column neutral defaults (for live prediction
    # we can't drop rows, so we use sensible fill values).
    PREDICT_NAN_DEFAULTS = {
        "rsi14": 50.0, "macd_line": 0.0, "macd_signal": 0.0,
        "macd_histogram": 0.0, "momentum_score": 0.0, "trend_strength": 0.0,
        "dist_ema20": 0.0, "dist_ema50": 0.0, "dist_ema200": 0.0,
        "candle_body_pct": 0.0, "upper_wick_pct": 0.0, "lower_wick_pct": 0.0,
        "spread": 0.0, "atr_ratio": 1.0,
        "session_asian": 0, "session_london": 0, "session_new_york": 0,
        "mode_scalping": 0, "mode_day_trading": 0, "mode_swing": 0,
    }
    for col, default in PREDICT_NAN_DEFAULTS.items():
        if col in X.columns:
            X[col] = X[col].fillna(default)
    # EMA/ATR columns: fill remaining NaN with 0 (signal data should not be
    # NaN at prediction time, but guard against it)
    X = X.fillna(0.0)
    X = X.replace([np.inf, -np.inf], 0.0)

    # Predict probability
    try:
        proba = model.predict_proba(X)[0, 1]
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        return None

    # ML confidence = rule confidence * ML probability
    # This penalises signals the ML model disagrees with
    ml_confidence = signal.confidence * proba

    prediction = Prediction(
        signal=signal.signal.value,
        confidence=round(ml_confidence, 2),
        probability=round(float(proba), 4),
        model_type=model_info["model_type"],
        model_version=model_info["version"],
        feature_count=X.shape[1],
    )

    logger.info(
        "ML prediction: %s prob=%.4f ml_conf=%.2f (rule_conf=%.2f) model=%s_v%d",
        signal.signal.value, proba, ml_confidence, signal.confidence,
        model_info["model_type"], model_info["version"],
    )

    return prediction