"""Hyperparameter optimization using Optuna (Bayesian search).

Finds better hyperparameters than hardcoded defaults by searching
a reasonable range around the current values. Runs N trials per
model type, optimizing for ROC AUC on the validation set.

Tuned parameters are saved to ml/models/{model_type}_{symbol}_params.json
and reused on subsequent training runs.
"""

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import SETTINGS

logger = logging.getLogger(__name__)

# Ensure this logger's INFO records reach the handlers even when the root
# logger is configured at WARNING (e.g. by backtest_historical.py). Without
# this, Optuna tuning and the per-trial heartbeat are completely silent and
# a long run looks indistinguishable from a freeze.
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)

ML_CONFIG = SETTINGS.ml
MODEL_DIR = ML_CONFIG.model_dir

# Default number of Optuna trials per model type
DEFAULT_N_TRIALS = 50


def _optuna_timeout() -> Optional[float]:
    """Optional wall-clock cap for an Optuna study, from ML_OPTUNA_TIMEOUT (sec).

    0 / unset / invalid → no cap (relies on n_trials). A positive value lets
    tuning terminate even if individual trials are unexpectedly slow, so the
    pipeline never hangs indefinitely.
    """
    raw = os.getenv("ML_OPTUNA_TIMEOUT", "").strip()
    if not raw:
        return None
    try:
        val = float(raw)
        return val if val > 0 else None
    except ValueError:
        logger.warning("Invalid ML_OPTUNA_TIMEOUT=%r, ignoring", raw)
        return None


def _xgboost_search_space(trial) -> dict:
    """Optuna search space for XGBoost."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "eval_metric": "logloss",
    }


def _histgbrt_search_space(trial) -> dict:
    """Optuna search space for HistGradientBoosting."""
    return {
        "max_iter": trial.suggest_int("max_iter", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.01, 10.0, log=True),
        "max_bins": trial.suggest_int("max_bins", 128, 255),
        "early_stopping": True,
        "n_iter_no_change": 50,
        "class_weight": "balanced",
        "scoring": "roc_auc",
        "verbose": 0,
    }


def _catboost_search_space(trial) -> dict:
    """Optuna search space for CatBoost."""
    return {
        "iterations": trial.suggest_int("iterations", 200, 800),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "bootstrap_type": "Bernoulli",
        "eval_metric": "Logloss",
        "auto_class_weights": "Balanced",
        "verbose": 0,
    }


_SEARCH_SPACES = {
    "xgboost": _xgboost_search_space,
    "histgbrt": _histgbrt_search_space,
    "catboost": _catboost_search_space,
}


def _create_model(model_type: str, params: dict):
    """Instantiate a model with the given parameters."""
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        # scale_pos_weight is set dynamically, not from params
        return XGBClassifier(**params)
    elif model_type == "histgbrt":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(**params)
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def tune_hyperparams(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = DEFAULT_N_TRIALS,
) -> Optional[dict]:
    """Use Optuna to find optimal hyperparameters for a model type.

    Args:
        model_type: One of "xgboost", "histgbrt", "catboost".
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        n_trials: Number of Optuna trials.

    Returns:
        Best parameters dict, or None if tuning is disabled or fails.
    """
    if not ML_CONFIG.use_tuning:
        logger.info("Hyperparameter tuning disabled, using defaults")
        return None

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed, skipping hyperparameter tuning")
        return None

    search_space_fn = _SEARCH_SPACES.get(model_type)
    if search_space_fn is None:
        logger.warning("No search space defined for %s, skipping tuning", model_type)
        return None

    from sklearn.metrics import roc_auc_score

    def objective(trial):
        params = search_space_fn(trial)

        # XGBoost needs scale_pos_weight computed from training data
        if model_type == "xgboost":
            neg_count = int((y_train == 0).sum())
            pos_count = int((y_train == 1).sum())
            params["scale_pos_weight"] = neg_count / pos_count if pos_count > 0 else 1.0

        try:
            model = _create_model(model_type, params)

            if model_type == "xgboost":
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            elif model_type == "histgbrt":
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            elif model_type == "catboost":
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

            y_proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_proba)

        except Exception:
            return 0.5  # Worst possible AUC (random classifier)

    study = optuna.create_study(direction="maximize")

    # Per-trial heartbeat so tuning is never silent. Without this a long
    # CatBoost / HistGBRT search produces no output and looks frozen.
    def _heartbeat(study, trial):
        try:
            completed = [t for t in study.trials if t.state.name == "COMPLETE"]
            best_val = None
            best_num = -1
            try:
                best_val = study.best_value
                best_num = study.best_trial.number
            except ValueError:
                pass
            logger.info(
                "Optuna %s trial %d/%d: value=%s best=%s (trial %d)",
                model_type,
                trial.number + 1, n_trials,
                f"{trial.value:.4f}" if trial.value is not None else "n/a",
                f"{best_val:.4f}" if best_val is not None else "n/a",
                best_num,
            )
        except Exception:
            pass

    timeout = _optuna_timeout()
    logger.info(
        "Optuna tuning %s: up to %d trials%s",
        model_type, n_trials,
        f" (timeout={timeout:.0f}s)" if timeout else "",
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
        callbacks=[_heartbeat],
    )

    best_params = study.best_params
    best_auc = study.best_value
    logger.info(
        "Optuna tuning for %s: best AUC=%.4f with params=%s",
        model_type, best_auc, best_params,
    )

    return best_params


def save_tuned_params(
    params: dict,
    model_type: str,
    symbol: str,
    timeframe: str,
) -> str:
    """Save tuned parameters to disk for reuse on subsequent training runs.

    Returns:
        Path to the saved JSON file.
    """
    _ensure_model_dir()
    filename = f"{model_type}_{symbol}_{timeframe or 'all'}_tuned_params.json"
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(params, f, indent=2)
    logger.info("Saved tuned params for %s to %s", model_type, filepath)
    return filepath


def load_tuned_params(
    model_type: str,
    symbol: str,
    timeframe: str,
) -> Optional[dict]:
    """Load previously saved tuned parameters from disk.

    Returns:
        Parameters dict, or None if not found.
    """
    filename = f"{model_type}_{symbol}_{timeframe or 'all'}_tuned_params.json"
    filepath = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load tuned params from %s: %s", filepath, e)
        return None


def _ensure_model_dir() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)