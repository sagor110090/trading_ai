"""Live prediction: evaluates the probability that a rule-based signal will succeed."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from trading_ai.config.settings import SETTINGS
from trading_ai.data.database import get_engine
from trading_ai.data.models import MLModelMeta
from trading_ai.indicators.features import FeatureSet
from trading_ai.signals.engine import TradeSignal

logger = logging.getLogger(__name__)

ML_CONFIG = SETTINGS.ml
MODEL_DIR = ML_CONFIG.model_dir


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

    Returns (model, feature_columns, model_meta) or None.
    """
    from trading_ai.data.database import get_session
    try:
        with get_session() as session:
            # Prefer best AUC among active models for this symbol/timeframe
            meta = (
                session.query(MLModelMeta)
                .filter(
                    MLModelMeta.symbol == symbol,
                    MLModelMeta.timeframe == timeframe,
                    MLModelMeta.is_active == True,
                )
                .order_by(MLModelMeta.roc_auc.desc())
                .first()
            )

            if not meta:
                return None

            filepath = meta.filepath
            # Resolve relative paths against MODEL_DIR
            if not os.path.isabs(filepath):
                filepath = os.path.join(MODEL_DIR, os.path.basename(filepath))
            if not os.path.exists(filepath):
                logger.warning("Model file not found: %s", filepath)
                return None

            model = joblib.load(filepath)

            feat_path = filepath.replace(".joblib", "_features.json")
            if os.path.exists(feat_path):
                with open(feat_path) as f:
                    feature_cols = json.load(f)
            else:
                feature_cols = None

            return model, feature_cols, meta

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

    model, feature_cols, meta = result

    # Build feature vector from signal's feature snapshot
    features = signal.features.to_dict()

    # Encode session
    session_encoding = {"ASIAN": 0, "LONDON": 1, "NEW_YORK": 2}
    features["session_enc"] = session_encoding.get(features.get("session", ""), -1)
    # Remove string session
    features.pop("session", None)

    # Build DataFrame with correct feature order
    if feature_cols:
        row = {}
        for col in feature_cols:
            row[col] = features.get(col, 0.0)
        X = pd.DataFrame([row])
    else:
        X = pd.DataFrame([features])

    # Replace any remaining NaN/None
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
        model_type=meta.model_type,
        model_version=meta.version,
        feature_count=X.shape[1],
    )

    logger.info(
        "ML prediction: %s prob=%.4f ml_conf=%.2f (rule_conf=%.2f) model=%s_v%d",
        signal.signal.value, proba, ml_confidence, signal.confidence,
        meta.model_type, meta.version,
    )

    return prediction