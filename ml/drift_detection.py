"""ML drift detection: monitors model performance degradation over time.

Tracks prediction vs actual outcomes and alerts when win rate drops
significantly below training performance (ROC AUC).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from config.settings import SETTINGS
from data.database import get_session
from data.models import MLModelMeta, Signal, SignalVerification

logger = logging.getLogger(__name__)


@dataclass
class PredictionTracking:
    """Tracks a single prediction and its outcome."""
    signal_id: str
    symbol: str
    model_type: str
    model_version: int
    ml_probability: float
    predicted_outcome: str  # "WIN" or "LOSS"
    actual_outcome: Optional[str] = None  # "WIN", "LOSS", "BREAKEVEN" or None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "ml_probability": self.ml_probability,
            "predicted_outcome": self.predicted_outcome,
            "actual_outcome": self.actual_outcome,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DriftReport:
    """Report on model drift for a symbol."""
    symbol: str
    model_type: str
    model_version: int
    training_auc: float
    recent_win_rate: float
    recent_count: int
    drift_detected: bool
    drift_percentage: float
    message: str


class DriftDetector:
    """Detects ML model performance drift over time."""

    def __init__(self, min_predictions: int = 20, drift_threshold: float = 0.15):
        """Initialize drift detector.

        Args:
            min_predictions: Minimum predictions before checking for drift
            drift_threshold: Alert if win rate drops below (training_auc - this)
        """
        self.min_predictions = min_predictions
        self.drift_threshold = drift_threshold
        self._tracked_predictions: list[PredictionTracking] = []

    def track_prediction(
        self,
        signal_id: str,
        symbol: str,
        model_type: str,
        model_version: int,
        ml_probability: float,
    ) -> None:
        """Track a prediction for later drift detection.

        Args:
            signal_id: Signal identifier
            symbol: Trading symbol
            model_type: Model type (xgboost, histgbrt, catboost)
            model_version: Model version number
            ml_probability: ML prediction probability (0-1)
        """
        # Predict WIN if probability >= 0.5, LOSS otherwise
        predicted_outcome = "WIN" if ml_probability >= 0.5 else "LOSS"

        tracking = PredictionTracking(
            signal_id=signal_id,
            symbol=symbol,
            model_type=model_type,
            model_version=model_version,
            ml_probability=ml_probability,
            predicted_outcome=predicted_outcome,
        )
        self._tracked_predictions.append(tracking)

        # Cleanup old predictions (> 30 days)
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        self._tracked_predictions = [
            t for t in self._tracked_predictions if t.timestamp > cutoff
        ]

    def update_outcome(self, signal_id: str, actual_outcome: str) -> None:
        """Update prediction with actual outcome.

        Args:
            signal_id: Signal identifier
            actual_outcome: Actual result ("WIN", "LOSS", "BREAKEVEN")
        """
        for tracking in self._tracked_predictions:
            if tracking.signal_id == signal_id and tracking.actual_outcome is None:
                tracking.actual_outcome = actual_outcome
                break

    def check_drift(self, symbol: str) -> Optional[DriftReport]:
        """Check if model has drifted for a symbol.

        Args:
            symbol: Trading symbol to check

        Returns:
            DriftReport if drift detected or enough data, None otherwise
        """
        # Get model metadata for this symbol
        with get_session() as session:
            model = (
                session.query(MLModelMeta)
                .filter(
                    MLModelMeta.symbol == symbol,
                    MLModelMeta.is_active == True,
                )
                .order_by(MLModelMeta.roc_auc.desc())
                .first()
            )

            if not model:
                return None

            training_auc = model.roc_auc
            model_type = model.model_type
            model_version = model.version

        # Get recent predictions with outcomes for this symbol
        recent_predictions = [
            t for t in self._tracked_predictions
            if t.symbol == symbol
            and t.actual_outcome is not None
            and t.model_type == model_type
            and t.model_version == model_version
        ]

        if len(recent_predictions) < self.min_predictions:
            return None

        # Calculate recent win rate
        wins = sum(1 for t in recent_predictions if t.actual_outcome == "WIN")
        recent_win_rate = wins / len(recent_predictions)

        # Expected win rate based on training AUC (simplified)
        # AUC 0.7 ~ 70% expected win rate in similar conditions
        expected_win_rate = training_auc

        # Check for drift
        drift_detected = recent_win_rate < (expected_win_rate - self.drift_threshold)
        drift_percentage = (expected_win_rate - recent_win_rate) * 100

        if drift_detected:
            message = (
                f"⚠️ DRIFT DETECTED: {symbol} {model_type}\n"
                f"   Training AUC: {training_auc:.2%}\n"
                f"   Recent win rate: {recent_win_rate:.2%} ({len(recent_predictions)} trades)\n"
                f"   Drift: {drift_percentage:.1f}% below expected"
            )
            logger.warning(message)
        else:
            message = (
                f"✓ {symbol} {model_type}: No drift\n"
                f"   Training AUC: {training_auc:.2%}\n"
                f"   Recent win rate: {recent_win_rate:.2%} ({len(recent_predictions)} trades)"
            )

        return DriftReport(
            symbol=symbol,
            model_type=model_type,
            model_version=model_version,
            training_auc=training_auc,
            recent_win_rate=recent_win_rate,
            recent_count=len(recent_predictions),
            drift_detected=drift_detected,
            drift_percentage=drift_percentage,
            message=message,
        )

    def get_pending_predictions(self, symbol: str) -> list[PredictionTracking]:
        """Get predictions waiting for outcome."""
        return [
            t for t in self._tracked_predictions
            if t.symbol == symbol and t.actual_outcome is None
        ]


# Global drift detector instance
_drift_detector: Optional[DriftDetector] = None


def get_drift_detector() -> DriftDetector:
    """Get or create global drift detector instance."""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector(
            min_predictions=SETTINGS.ml.min_signals_for_drift,
            drift_threshold=SETTINGS.ml.drift_threshold,
        )
    return _drift_detector


def track_ml_prediction(
    signal_id: str,
    symbol: str,
    model_type: str,
    model_version: int,
    ml_probability: float,
) -> None:
    """Track an ML prediction for drift detection.

    Call this after making an ML prediction.
    """
    detector = get_drift_detector()
    detector.track_prediction(
        signal_id=signal_id,
        symbol=symbol,
        model_type=model_type,
        model_version=model_version,
        ml_probability=ml_probability,
    )


def update_prediction_outcome(signal_id: str, actual_outcome: str) -> None:
    """Update a prediction's actual outcome after verification.

    Call this when a signal is verified.

    Args:
        signal_id: Signal identifier (without prefix)
        actual_outcome: "WIN", "LOSS", or "BREAKEVEN"
    """
    detector = get_drift_detector()
    # Try with signal_id as-is and with common prefixes
    for prefix in ["", "live_", "backtest_"]:
        detector.update_outcome(f"{prefix}{signal_id}", actual_outcome)


def check_model_drift(symbol: str) -> Optional[DriftReport]:
    """Check for ML model drift on a symbol.

    Returns:
        DriftReport if drift detected or enough data exists, None otherwise
    """
    detector = get_drift_detector()
    return detector.check_drift(symbol)


def sync_verified_signals_for_drift(symbol: str) -> int:
    """Sync recently verified signals with drift tracking.

    This ensures predictions are matched with their outcomes.

    Args:
        symbol: Trading symbol

    Returns:
        Number of predictions updated
    """
    detector = get_drift_detector()

    with get_session() as session:
        # Get recently verified signals (last 7 days)
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)

        verified = (
            session.query(Signal, SignalVerification)
            .join(SignalVerification, Signal.id == SignalVerification.signal_id)
            .filter(
                Signal.symbol == symbol,
                SignalVerification.verified_at >= cutoff,
            )
            .all()
        )

        updated = 0
        for signal, verification in verified:
            # Remove common prefixes from signal_id
            signal_id_base = signal.signal_id
            for prefix in ["live_", "backtest_"]:
                if signal_id_base.startswith(prefix):
                    signal_id_base = signal_id_base[len(prefix):]
                    break

            # Update if this prediction is being tracked
            for tracking in detector._tracked_predictions:
                if tracking.signal_id == signal_id_base and tracking.actual_outcome is None:
                    tracking.actual_outcome = verification.result
                    updated += 1
                    break

        return updated
