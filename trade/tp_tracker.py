"""TP Progress Tracker: monitors trade progress towards Take Profit.

Tracks:
- Current profit distance vs total TP distance
- Maximum Favorable Excursion (MFE): best price reached
- Maximum Adverse Excursion (MAE): worst price reached
- Highest TP progress reached

When TP progress reaches 60%, triggers strength evaluation.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class TPProgress:
    """Tracks a single trade's progress towards TP."""
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    opened_at: datetime

    # Tracking fields
    highest_tp_progress: float = 0.0  # 0.0 to 1.0+
    mfe_price: float = 0.0  # Maximum favorable price seen
    mae_price: float = 0.0  # Maximum adverse price seen
    mfe_pnl: float = 0.0  # Maximum favorable PnL
    mae_pnl: float = 0.0  # Maximum adverse PnL
    last_checked: Optional[datetime] = None
    strength_scores: list[dict] = field(default_factory=list)
    close_reason: Optional[str] = None
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    final_pnl: Optional[float] = None

    @property
    def tp_distance(self) -> float:
        """Total distance from entry to TP."""
        if self.direction.upper() == "BUY":
            return self.take_profit - self.entry_price
        else:
            return self.entry_price - self.take_profit

    @property
    def sl_distance(self) -> float:
        """Total distance from entry to SL."""
        if self.direction.upper() == "BUY":
            return self.entry_price - self.stop_loss
        else:
            return self.stop_loss - self.entry_price

    @property
    def risk_reward(self) -> float:
        """R:R ratio."""
        if self.sl_distance == 0:
            return 0.0
        return self.tp_distance / self.sl_distance

    def calculate_progress(self, current_price: float) -> float:
        """Calculate current TP progress (0.0 to 1.0+).

        Returns:
            TP progress ratio. Can exceed 1.0 if price moved past TP.
        """
        if self.direction.upper() == "BUY":
            profit_distance = current_price - self.entry_price
        else:
            profit_distance = self.entry_price - current_price

        progress = profit_distance / self.tp_distance if self.tp_distance > 0 else 0.0
        return max(0.0, progress)

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL at current price."""
        if self.direction.upper() == "BUY":
            pnl_per_unit = current_price - self.entry_price
        else:
            pnl_per_unit = self.entry_price - current_price

        return pnl_per_unit * self.size

    def update(self, current_price: float, high: float = None, low: float = None) -> dict:
        """Update tracking with current price data.

        Args:
            current_price: Current market price (close)
            high: High price of current candle (for MFE)
            low: Low price of current candle (for MAE)

        Returns:
            Dict with current state: progress, pnl, mfe_updated, mae_updated
        """
        result = {
            "progress": 0.0,
            "pnl": 0.0,
            "mfe_updated": False,
            "mae_updated": False,
            "highest_progress": self.highest_tp_progress,
        }

        # Calculate current progress
        progress = self.calculate_progress(current_price)
        result["progress"] = progress

        # Update highest progress
        if progress > self.highest_tp_progress:
            self.highest_tp_progress = progress
            result["highest_progress"] = progress

        # Calculate current PnL
        pnl = self.calculate_pnl(current_price)
        result["pnl"] = pnl

        # Update MFE/MAE using high/low if available
        if high is not None:
            mfe_pnl = self.calculate_pnl(high)
            if mfe_pnl > self.mfe_pnl:
                self.mfe_pnl = mfe_pnl
                self.mfe_price = high
                result["mfe_updated"] = True

        if low is not None:
            mae_pnl = self.calculate_pnl(low)
            if mae_pnl < self.mae_mae or self.mae_pnl == 0:
                self.mae_pnl = mae_pnl
                self.mae_price = low
                result["mae_updated"] = True

        # Also check current price for MFE/MAE
        if pnl > self.mfe_pnl:
            self.mfe_pnl = pnl
            self.mfe_price = current_price
            result["mfe_updated"] = True

        if pnl < self.mae_pnl or self.mae_pnl == 0:
            self.mae_pnl = pnl
            self.mae_price = current_price
            result["mae_updated"] = True

        self.last_checked = datetime.now(timezone.utc)
        return result

    def should_evaluate_strength(self, current_price: float) -> tuple[bool, str]:
        """Check if we should evaluate trade strength.

        Returns (should_evaluate, reason).

        Triggers when:
        - TP progress reaches 60% (first time)
        - TP progress drops 10% from highest (trend reversing)
        - Position is already above 60% when first tracked (first check)
        """
        progress = self.calculate_progress(current_price)

        # First time reaching 60% (or already above 60% on first check)
        if progress >= 0.60:
            # If we haven't evaluated yet and progress is >= 60%
            if self.highest_tp_progress < 0.60:
                logger.info(
                    "Strength eval triggered: TP >= 60%% (progress: %.1f%%, highest: %.1f%%)",
                    progress * 100, self.highest_tp_progress * 100,
                )
                return True, f"TP progress reached 60% (current: {progress:.1%})"
            # If progress is >= 60% but no strength scores recorded yet (first check for existing position)
            elif not self.strength_scores and self.last_checked is None:
                logger.info(
                    "Strength eval triggered: Already at %.1f%% with no prior evaluation",
                    progress * 100,
                )
                return True, f"Position at {progress:.1%} TP - initial evaluation"
            else:
                logger.debug(
                    "Strength eval NOT triggered: progress=%.1f%%, highest=%.1f%%, scores_count=%d, last_checked=%s",
                    progress * 100, self.highest_tp_progress * 100,
                    len(self.strength_scores), self.last_checked,
                )

        # Progress dropped significantly from peak
        if self.highest_tp_progress >= 0.50:
            drop_from_peak = self.highest_tp_progress - progress
            if drop_from_peak >= 0.10:
                logger.info(
                    "Strength eval triggered: Dropped %.1f%% from peak (peak: %.1f%%, current: %.1f%%)",
                    drop_from_peak * 100, self.highest_tp_progress * 100, progress * 100,
                )
                return True, (
                    f"TP progress dropped {drop_from_peak:.1%} from peak "
                    f"(peak: {self.highest_tp_progress:.1%}, current: {progress:.1%})"
                )

        return False, ""

    def close(self, exit_price: float, reason: str) -> dict:
        """Mark the position as closed.

        Returns:
            Dict with final trade statistics.
        """
        self.close_reason = reason
        self.closed_at = datetime.now(timezone.utc)
        self.exit_price = exit_price
        self.final_pnl = self.calculate_pnl(exit_price)

        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry": self.entry_price,
            "exit": exit_price,
            "sl": self.stop_loss,
            "tp": self.take_profit,
            "size": self.size,
            "pnl": self.final_pnl,
            "mfe_pnl": self.mfe_pnl,
            "mae_pnl": self.mae_pnl,
            "highest_tp_progress": self.highest_tp_progress,
            "reason": reason,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "strength_scores": self.strength_scores,
        }


class TPProgressTracker:
    """Manages TP progress tracking for all open positions."""

    def __init__(self):
        self.active_trades: dict[str, TPProgress] = {}

    def add_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        size: float,
        opened_at: datetime = None,
    ) -> TPProgress:
        """Add a new position to track."""
        if opened_at is None:
            opened_at = datetime.now(timezone.utc)

        tracker = TPProgress(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=size,
            opened_at=opened_at,
        )

        # Use symbol-direction as key (only one position per symbol-direction pair)
        key = f"{symbol}_{direction}"
        self.active_trades[key] = tracker

        logger.info(
            "Tracking TP progress: %s %s entry=%.2f SL=%.2f TP=%.2f",
            direction, symbol, entry_price, stop_loss, take_profit,
        )

        return tracker

    def update_position(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        high: float = None,
        low: float = None,
    ) -> Optional[dict]:
        """Update a tracked position with current price data.

        Returns:
            Update result dict, or None if position not found.
        """
        key = f"{symbol}_{direction}"
        tracker = self.active_trades.get(key)

        if tracker is None:
            return None

        return tracker.update(current_price, high, low)

    def should_evaluate_strength(
        self,
        symbol: str,
        direction: str,
        current_price: float,
    ) -> tuple[bool, str]:
        """Check if we should evaluate trade strength for this position."""
        key = f"{symbol}_{direction}"
        tracker = self.active_trades.get(key)

        if tracker is None:
            return False, "Position not tracked"

        return tracker.should_evaluate_strength(current_price)

    def record_strength_score(
        self,
        symbol: str,
        direction: str,
        score: float,
        decision: str,
        reason: str,
        components: dict = None,
    ) -> None:
        """Record a strength score evaluation for a position."""
        key = f"{symbol}_{direction}"
        tracker = self.active_trades.get(key)

        if tracker is None:
            return

        tracker.strength_scores.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "score": score,
            "decision": decision,
            "reason": reason,
            "components": components or {},
            "tp_progress": tracker.highest_tp_progress,
        })

    def close_position(
        self,
        symbol: str,
        direction: str,
        exit_price: float,
        reason: str,
    ) -> Optional[dict]:
        """Close a tracked position and return final stats."""
        key = f"{symbol}_{direction}"
        tracker = self.active_trades.get(key)

        if tracker is None:
            return None

        result = tracker.close(exit_price, reason)
        del self.active_trades[key]

        logger.info(
            "Closed tracked position: %s %s PnL=%.2f reason=%s",
            direction, symbol, result["pnl"], reason,
        )

        return result

    def get_position(self, symbol: str, direction: str) -> Optional[TPProgress]:
        """Get a tracked position."""
        key = f"{symbol}_{direction}"
        return self.active_trades.get(key)

    def get_all_positions(self) -> list[TPProgress]:
        """Get all tracked positions."""
        return list(self.active_trades.values())

    def remove_position(self, symbol: str, direction: str) -> bool:
        """Remove a position from tracking without closing it."""
        key = f"{symbol}_{direction}"
        if key in self.active_trades:
            del self.active_trades[key]
            return True
        return False


def calculate_tp_progress(
    entry: float,
    tp: float,
    current: float,
    direction: str,
) -> float:
    """Calculate TP progress ratio (0.0 to 1.0+)."""
    if direction.upper() == "BUY":
        profit_distance = current - entry
        tp_distance = tp - entry
    else:
        profit_distance = entry - current
        tp_distance = entry - tp

    if tp_distance == 0:
        return 0.0

    return max(0.0, profit_distance / tp_distance)
