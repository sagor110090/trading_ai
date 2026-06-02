"""Position Monitor: continuously checks open positions and manages exits.

Monitors every cycle:
1. Fetches current price for open positions
2. Calculates TP progress
3. At 60% TP progress, evaluates trade strength
4. Closes position if strength score < 50

Integrates with:
- TPProgressTracker: tracks MFE/MAE/TP progress
- TradeStrengthScore: evaluates trend strength
- Database: logs trade outcomes
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from trading_ai.config.settings import SETTINGS
from trading_ai.data.database import get_session
from trading_ai.data.models import TradeLog
from trading_ai.trade.executor import (
    OpenPosition,
    close_position_unified,
    get_open_positions_unified,
)
from trading_ai.trade.strength_score import TradeStrengthScore, StrengthScoreComponents, StrengthScoreResult
from trading_ai.trade.tp_tracker import TPProgressTracker

logger = logging.getLogger(__name__)


@dataclass
class MonitorDecision:
    """Decision made by position monitor."""
    action: str  # "hold", "close", "error"
    reason: str
    symbol: str = ""  # Symbol name
    direction: str = ""  # "BUY" or "SELL"
    tp_progress: float = 0.0
    strength_score: Optional[float] = None
    pnl: Optional[float] = None


class PositionMonitor:
    """Monitors open positions and manages exits based on TP progress and strength.

    Usage:
        monitor = PositionMonitor(client)
        decisions = monitor.check_all_positions()
        for decision in decisions:
            if decision.action == "close":
                # Position was closed
    """

    def __init__(self, client):
        self.client = client
        self.tp_tracker = TPProgressTracker()
        self.strength_scorer = TradeStrengthScore()
        self._synced_with_broker = False

    def sync_positions(self) -> None:
        """Sync tracked positions with broker's open positions.

        Adds new positions to tracker and removes closed ones.
        Uses broker deal_id/ticket as the unique key so that a new trade
        opened after a TP hit gets a fresh tracker (instead of reusing the
        old position's MFE/MAE/strength history).
        """
        broker_positions = get_open_positions_unified(self.client)
        tracked_keys = set(self.tp_tracker.get_tracked_keys())
        broker_keys = set()

        for pos in broker_positions:
            key = str(pos.deal_id)
            broker_keys.add(key)

            # Skip if already tracking
            if key in tracked_keys:
                continue

            # Add new position to tracker
            self.tp_tracker.add_position(
                symbol=pos.epic,
                direction=pos.direction,
                entry_price=pos.level,
                stop_loss=pos.stop_level if pos.stop_level else pos.level,
                take_profit=pos.profit_level if pos.profit_level else pos.level * 1.01,
                size=pos.size,
                opened_at=datetime.now(timezone.utc),
                deal_id=key,
            )
            logger.info(
                "Started tracking: %s %s entry=%.2f SL=%.2f TP=%.2f (deal_id=%s)",
                pos.direction, pos.epic, pos.level, pos.stop_level, pos.profit_level, key,
            )

        # Remove positions that are no longer open (TP hit, SL hit, closed externally)
        for key in tracked_keys - broker_keys:
            tracker = self.tp_tracker.active_trades.get(key)
            if tracker is None:
                continue
            final_stats = self.tp_tracker.close_position(
                tracker.symbol, tracker.direction,
                exit_price=0,  # Unknown - position closed externally
                reason="external_close",
                deal_id=key,
            )
            if final_stats:
                self._log_trade(final_stats)
            logger.info(
                "Stopped tracking (external close): key=%s %s %s",
                key, tracker.direction, tracker.symbol,
            )

        self._synced_with_broker = True

    def check_all_positions(
        self,
        df_provider: callable = None,
    ) -> list[MonitorDecision]:
        """Check all open positions and make decisions.

        Args:
            df_provider: Function that takes (symbol, timeframe) and returns
                         DataFrame with OHLCV and indicators for strength scoring.
                         If None, strength evaluation is skipped.

        Returns:
            List of MonitorDecision for each position checked.
        """
        # Re-sync every call so newly opened / closed positions are reflected.
        # sync_positions() is cheap (one broker query + dict diff) and runs
        # every 60s from the background monitor thread.
        self.sync_positions()

        decisions = []
        broker_positions = get_open_positions_unified(self.client)

        for pos in broker_positions:
            try:
                decision = self._check_position(pos, df_provider)
                decisions.append(decision)

                # Execute close if needed
                if decision.action == "close":
                    self._execute_close(pos, decision)

            except Exception as e:
                import traceback
                logger.error("Error checking position %s: %s\n%s", pos.epic, e, traceback.format_exc())
                decisions.append(MonitorDecision(
                    action="error",
                    reason=f"Error: {e}",
                    symbol=pos.epic,
                    direction=pos.direction,
                ))

        return decisions

    def _check_position(
        self,
        pos: OpenPosition,
        df_provider: callable = None,
    ) -> MonitorDecision:
        """Check a single position and make decision.

        Args:
            pos: The open position to check
            df_provider: Optional function to get DataFrame for strength scoring

        Returns:
            MonitorDecision with action and reasoning
        """
        # Get current price
        price_data = self.client.get_latest_price(pos.epic)
        if not price_data:
            return MonitorDecision(
                action="hold",
                reason="Could not fetch price",
                symbol=pos.epic,
                direction=pos.direction,
            )

        # Get current price based on platform
        # Handle both enum and string values for trading_platform
        is_mt5 = (
            SETTINGS.trading_platform == "MT5" or
            (hasattr(SETTINGS.trading_platform, "value") and SETTINGS.trading_platform.value == "MT5")
        )
        if is_mt5:
            current_price = price_data.get("ask") or price_data.get("bid")
        else:
            current_price = price_data.get("bid") or price_data.get("ask")

        if not current_price:
            return MonitorDecision(
                action="hold",
                reason="No valid price available",
                symbol=pos.epic,
                direction=pos.direction,
            )

        # Use broker deal_id/ticket as the unique tracker key
        deal_id = str(pos.deal_id)

        # Update TP tracker
        update_result = self.tp_tracker.update_position(
            pos.epic, pos.direction, current_price,
            deal_id=deal_id,
        )

        tp_progress = update_result.get("progress", 0.0) if update_result else 0.0
        # Use actual broker profit instead of calculating
        pnl = pos.upl

        # Check if we should evaluate strength
        should_eval, eval_reason = self.tp_tracker.should_evaluate_strength(
            pos.epic, pos.direction, current_price,
            deal_id=deal_id,
        )

        # Also force evaluation if TP is above 60% but no strength score recorded
        tracked = self.tp_tracker.get_position(pos.epic, pos.direction, deal_id=deal_id)
        force_eval = (
            tp_progress >= 0.60 and tracked is not None
            and len(tracked.strength_scores) == 0
        )

        if should_eval or force_eval:
            logger.info(
                "Evaluating strength: %s %s - TP: %.1f%% (should_eval=%s, force_eval=%s, df_provider=%s, deal_id=%s)",
                pos.direction, pos.epic, tp_progress * 100, should_eval, force_eval, df_provider is not None, deal_id,
            )

        if (should_eval or force_eval) and df_provider:
            try:
                strength_result = self._evaluate_strength(pos, df_provider)

                # Record the score (guard against missing components)
                components_dict = (
                    strength_result.components.to_dict()
                    if strength_result.components is not None
                    else {}
                )
                self.tp_tracker.record_strength_score(
                    pos.epic, pos.direction,
                    score=strength_result.score,
                    decision=strength_result.decision,
                    reason=strength_result.reason,
                    components=components_dict,
                    deal_id=deal_id,
                )

                # Make decision based on strength
                if strength_result.decision == "exit":
                    return MonitorDecision(
                        action="close",
                        reason=f"{eval_reason if should_eval else 'Forced'}; {strength_result.reason}",
                        symbol=pos.epic,
                        direction=pos.direction,
                        tp_progress=tp_progress,
                        strength_score=strength_result.score,
                        pnl=pnl,
                    )
                elif strength_result.decision == "monitor":
                    return MonitorDecision(
                        action="hold",
                        reason=f"{eval_reason if should_eval else 'Forced'}; {strength_result.reason}",
                        symbol=pos.epic,
                        direction=pos.direction,
                        tp_progress=tp_progress,
                        strength_score=strength_result.score,
                        pnl=pnl,
                    )
                else:  # "hold" decision from scorer
                    return MonitorDecision(
                        action="hold",
                        reason=strength_result.reason,
                        symbol=pos.epic,
                        direction=pos.direction,
                        tp_progress=tp_progress,
                        strength_score=strength_result.score,
                        pnl=pnl,
                    )
            except Exception as e:
                import traceback
                logger.error("Strength evaluation failed for %s: %s\n%s", pos.epic, e, traceback.format_exc())
                # Continue with hold but no strength score
                return MonitorDecision(
                    action="hold",
                    reason=f"Strength eval failed: {str(e)[:50]}",
                    symbol=pos.epic,
                    direction=pos.direction,
                    tp_progress=tp_progress,
                    pnl=pnl,
                )
        elif should_eval and not df_provider:
            logger.warning("Strength evaluation skipped: no df_provider for %s", pos.epic)

        # Default: hold
        return MonitorDecision(
            action="hold",
            reason=f"TP progress: {tp_progress:.1%}",
            symbol=pos.epic,
            direction=pos.direction,
            tp_progress=tp_progress,
            pnl=pnl,
        )

    def _evaluate_strength(
        self,
        pos: OpenPosition,
        df_provider: callable,
    ) -> StrengthScoreResult:
        """Evaluate trade strength for a position.

        Args:
            pos: The open position
            df_provider: Function to get DataFrame with indicators

        Returns:
            StrengthScoreResult
        """
        # Get M5 dataframe for strength calculation
        df = df_provider(pos.epic, "5m")

        if df is None or df.empty:
            logger.warning("No data available for strength evaluation of %s", pos.epic)
            return StrengthScoreResult(
                score=50.0,
                components=StrengthScoreComponents(),
                decision="monitor",
                reason="No data for evaluation",
            )

        # Compute features if not already present
        if "ema20" not in df.columns:
            from trading_ai.indicators.features import compute_all_features
            df = compute_all_features(df)

        # Evaluate strength
        result = self.strength_scorer.evaluate(df, pos.direction)

        logger.info(
            "Strength eval: %s %s score=%.1f decision=%s",
            pos.direction, pos.epic, result.score, result.decision,
        )

        return result

    def _execute_close(self, pos: OpenPosition, decision: MonitorDecision) -> None:
        """Execute a close decision.

        Closes the position and logs to database.
        """
        logger.info(
            "Closing %s %s: %s (TP progress: %.1%%, Strength: %.1f)",
            pos.direction, pos.epic, decision.reason,
            decision.tp_progress * 100,
            decision.strength_score or 0,
        )

        # Close the position
        close_result = close_position_unified(self.client, pos)

        if close_result.success:
            # Get final stats from tracker
            final_stats = self.tp_tracker.close_position(
                pos.epic, pos.direction,
                exit_price=close_result.close_level or pos.level,
                reason=f"strength_exit:{decision.reason[:50]}",
                deal_id=str(pos.deal_id),
            )

            if final_stats:
                self._log_trade(final_stats)
        else:
            logger.error("Failed to close position: %s", close_result.error)

    def _log_trade(self, stats: dict) -> None:
        """Log trade statistics to database."""
        session = next(get_session())

        trade_log = TradeLog(
            deal_id=str(stats.get("deal_id", "") or ""),
            symbol=stats["symbol"],
            direction=stats["direction"],
            size=stats["size"],
            entry_price=stats["entry"],
            exit_price=stats.get("exit"),
            stop_loss=stats["sl"],
            take_profit=stats["tp"],
            opened_at=datetime.fromisoformat(stats["opened_at"]),
            closed_at=datetime.fromisoformat(stats["closed_at"]) if stats.get("closed_at") else None,
            mfe_price=stats.get("mfe_price"),
            mae_price=stats.get("mae_price"),
            mfe_pnl=stats.get("mfe_pnl"),
            mae_pnl=stats.get("mae_pnl"),
            highest_tp_progress=stats.get("highest_tp_progress"),
            strength_scores=str(stats.get("strength_scores", [])),
            final_pnl=stats.get("pnl"),
            close_reason=stats.get("reason", "unknown"),
        )

        session.add(trade_log)
        try:
            session.commit()
            logger.debug("Logged trade to database: %s", stats["symbol"])
        except Exception as e:
            session.rollback()
            logger.error("Failed to log trade: %s", e)

    def add_manual_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        size: float,
        signal_id: str = None,
    ) -> None:
        """Add a position to tracking (e.g., right after placing a trade)."""
        tracker = self.tp_tracker.add_position(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=size,
        )

        logger.info(
            "Manually added to tracking: %s %s (signal: %s)",
            direction, symbol, signal_id or "N/A",
        )

    def get_position_stats(self, symbol: str, direction: str) -> Optional[dict]:
        """Get current tracking stats for a position."""
        tracker = self.tp_tracker.get_position(symbol, direction)
        if not tracker:
            return None

        return {
            "symbol": tracker.symbol,
            "direction": tracker.direction,
            "entry": tracker.entry_price,
            "sl": tracker.stop_loss,
            "tp": tracker.take_profit,
            "size": tracker.size,
            "highest_tp_progress": tracker.highest_tp_progress,
            "mfe_pnl": tracker.mfe_pnl,
            "mae_pnl": tracker.mae_pnl,
            "strength_scores": tracker.strength_scores,
            "opened_at": tracker.opened_at.isoformat(),
        }
