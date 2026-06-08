"""Position Monitor: continuously checks open positions and manages exits.

Monitors every cycle:
1. Fetches current price for open positions
2. Calculates TP progress
3. At 40% TP progress, evaluates trade strength
4. Closes position if strength score < 50

Integrates with:
- TPProgressTracker: tracks MFE/MAE/TP progress
- TradeStrengthScore: evaluates trend strength
- Database: logs trade outcomes
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd

from config.settings import SETTINGS
from config.utils import normalize_symbol
from data.database import get_session
from data.models import TradeLog
from trade.executor import (
    OpenPosition,
    close_position_unified,
    get_open_positions_unified,
    update_position_sl_tp_unified,
)
from trade.strength_score import TradeStrengthScore, StrengthScoreComponents, StrengthScoreResult
from trade.tp_tracker import TPProgress, TPProgressTracker

logger = logging.getLogger(__name__)


@dataclass
class MonitorDecision:
    """Decision made by position monitor."""
    action: str  # "hold", "close", "trail_stop", "error", "external_close"
    reason: str
    symbol: str = ""  # Symbol name
    direction: str = ""  # "BUY" or "SELL"
    tp_progress: float = 0.0
    strength_score: Optional[float] = None
    pnl: Optional[float] = None
    new_sl: Optional[float] = None
    deal_id: Optional[str] = None  # Deal ID for externally closed positions


class PositionMonitor:
    """Monitors open positions and manages exits based on TP progress and strength.

    Usage:
        monitor = PositionMonitor(client)
        decisions = monitor.check_all_positions()
        for decision in decisions:
            if decision.action == "close":
                # Position was closed
            elif decision.action == "external_close":
                # Position closed externally (SL/TP hit at broker)
    """

    def __init__(self, client, on_position_closed: Optional[Callable[..., object]] = None):
        self.client = client
        self.tp_tracker = TPProgressTracker()
        self.strength_scorer = TradeStrengthScore()
        self._synced_with_broker = False
        self._externally_closed: list[dict] = []  # Track positions closed by broker (SL/TP)
        self._on_position_closed = on_position_closed

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

            # Add new position to tracker (normalize symbol for DB consistency)
            normalized_symbol = normalize_symbol(pos.epic)
            self.tp_tracker.add_position(
                symbol=normalized_symbol,
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

            # Store externally closed position info for cooldown registration
            normalized_symbol = normalize_symbol(tracker.symbol)
            if self._on_position_closed:
                self._on_position_closed(
                    symbol=normalized_symbol,
                    reason="external_close",
                    deal_id=key,
                    source="position_monitor_external",
                )
            self._externally_closed.append({
                "deal_id": key,
                "symbol": normalized_symbol,
                "direction": tracker.direction,
                "entry": tracker.entry_price,
                "sl": tracker.stop_loss,
                "tp": tracker.take_profit,
            })

            final_stats = self.tp_tracker.close_position(
                tracker.symbol, tracker.direction,
                exit_price=0,  # Unknown - position closed externally
                reason="external_close",
                deal_id=key,
            )
            if final_stats:
                self._log_trade(final_stats, normalize_symbol(tracker.symbol))
            logger.info(
                "Stopped tracking (external close): key=%s %s %s - will register cooldown",
                key, tracker.direction, tracker.symbol,
            )

        self._synced_with_broker = True

    def check_all_positions(
        self,
        df_provider: Optional[Callable] = None,
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

        # First, process any externally closed positions (SL/TP hit at broker)
        # These need to be handled first so cooldowns are registered
        for closed in self._externally_closed:
            # Determine if it was a loss or win based on final PnL
            # Since we don't have final PnL for external closes, we assume SL hit = loss
            decisions.append(MonitorDecision(
                action="external_close",
                reason="Position closed by broker (SL/TP hit)",
                symbol=closed["symbol"],
                direction=closed["direction"],
                tp_progress=0.0,
                strength_score=None,
                pnl=None,
                deal_id=closed["deal_id"],
            ))

        # Clear the externally closed list after processing
        self._externally_closed.clear()

        for pos in broker_positions:
            try:
                decision = self._check_position(pos, df_provider)
                decisions.append(decision)

                # Execute close if needed
                if decision.action == "close":
                    self._execute_close(pos, decision)
                # Execute trailing stop if needed
                elif decision.action == "trail_stop":
                    self._execute_trail_stop(pos, decision)

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
        df_provider: Optional[Callable] = None,
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

        # Normalize symbol for tracker (tracker uses normalized symbols)
        normalized_symbol = normalize_symbol(pos.epic)

        # Update TP tracker
        update_result = self.tp_tracker.update_position(
            normalized_symbol, pos.direction, current_price,
            deal_id=deal_id,
        )

        tp_progress = update_result.get("progress", 0.0) if update_result else 0.0
        # Use actual broker profit instead of calculating
        pnl = pos.upl

        # Check if we should evaluate strength
        should_eval, eval_reason = self.tp_tracker.should_evaluate_strength(
            normalized_symbol, pos.direction, current_price,
            deal_id=deal_id,
        )

        tracked = self.tp_tracker.get_position(normalized_symbol, pos.direction, deal_id=deal_id)
        force_eval = (
            tp_progress >= 0.40 and tracked is not None
            and len(tracked.strength_scores) == 0
        )
        if force_eval and not eval_reason:
            eval_reason = f"TP progress at {tp_progress:.1%} - initial evaluation"

        if should_eval or force_eval:
            logger.info(
                "Evaluating strength: %s %s - TP: %.1f%% (should_eval=%s, force_eval=%s, df_provider=%s, deal_id=%s)",
                pos.direction, pos.epic, tp_progress * 100, should_eval, force_eval, df_provider is not None, deal_id,
            )

        # Track strength score for the final decision
        tracked_strength_score = None

        if (should_eval or force_eval) and df_provider:
            try:
                strength_result = self._evaluate_strength(pos, df_provider)
                tracked_strength_score = strength_result.score

                if strength_result.score == 50.0 and strength_result.reason == "No data for evaluation":
                    logger.warning("Strength evaluation returned no-data for %s %s - using score=50", pos.direction, pos.epic)
                    # tracked_strength_score is already 50.0, continue to record it

                components_dict = (
                    strength_result.components.to_dict()
                    if strength_result.components is not None
                    else {}
                )
                self.tp_tracker.record_strength_score(
                    normalized_symbol, pos.direction,
                    score=strength_result.score,
                    decision=strength_result.decision,
                    reason=strength_result.reason,
                    components=components_dict,
                    deal_id=deal_id,
                )

                if strength_result.decision == "exit":
                    close_decision = MonitorDecision(
                        action="close",
                        reason=f"{eval_reason}; {strength_result.reason}",
                        symbol=pos.epic,
                        direction=pos.direction,
                        tp_progress=tp_progress,
                        strength_score=strength_result.score,
                        pnl=pnl,
                    )
                    # Close takes priority - no need to check trailing stop
                    return close_decision
                # hold/monitor: fall through to trailing stop check
            except Exception as e:
                import traceback
                logger.error("Strength evaluation failed for %s: %s\n%s", pos.epic, e, traceback.format_exc())
                # Set score to None on error to indicate evaluation failed
                tracked_strength_score = None
        elif (should_eval or force_eval) and not df_provider:
            logger.warning("Strength evaluation skipped: no df_provider for %s", pos.epic)

        # Trailing stop: move SL to protect profits as TP progress increases
        trail_decision = self._check_trailing_stop(pos, tracked, current_price, tp_progress)
        if trail_decision is not None:
            return trail_decision

        # Default: hold
        return MonitorDecision(
            action="hold",
            reason=f"TP progress: {tp_progress:.1%}",
            symbol=pos.epic,
            direction=pos.direction,
            tp_progress=tp_progress,
            pnl=pnl,
            strength_score=tracked_strength_score,
        )

    def _check_trailing_stop(
        self,
        pos: OpenPosition,
        tracked: Optional[TPProgress],
        current_price: float,
        tp_progress: float,
    ) -> Optional[MonitorDecision]:
        """Check if stop loss should be moved to protect profits.

        Trailing stop levels based on TP progress:
        - >= 40%: move SL to breakeven (entry price)
        - >= 60%: move SL to 40% of TP distance
        - >= 80%: move SL to 60% of TP distance
        - >= 95%: move SL to 80% of TP distance (very close to TP)

        Only moves SL in the protective direction (tighter).
        """
        if tracked is None or tp_progress < 0.40:
            return None

        entry = tracked.entry_price
        sl = tracked.stop_loss
        tp = tracked.take_profit
        direction = pos.direction.upper()

        if direction == "BUY":
            if tp <= entry:
                return None
            tp_distance = tp - entry
            if tp_progress >= 0.95:
                target_sl = entry + tp_distance * 0.80
            elif tp_progress >= 0.80:
                target_sl = entry + tp_distance * 0.60
            elif tp_progress >= 0.60:
                target_sl = entry + tp_distance * 0.40
            else:
                target_sl = entry  # breakeven

            # Only move SL up (tighter for BUY)
            # If no SL set, use entry as baseline so breakeven triggers
            current_sl = sl if sl and sl > 0 else entry
            if target_sl <= current_sl + 0.00001:  # small buffer for float comparison
                return None

            logger.info(
                "Trailing stop: %s %s SL %.2f -> %.2f (TP progress: %.1f%%)",
                direction, pos.epic, current_sl, target_sl, tp_progress * 100,
            )
            return MonitorDecision(
                action="trail_stop",
                reason=f"Trailing stop: SL {current_sl:.2f} -> {target_sl:.2f} (TP {tp_progress:.1%})",
                symbol=pos.epic,
                direction=pos.direction,
                tp_progress=tp_progress,
                pnl=pos.upl,
                new_sl=target_sl,
            )
        else:  # SELL
            if tp >= entry:
                return None
            tp_distance = entry - tp
            if tp_progress >= 0.95:
                target_sl = entry - tp_distance * 0.80
            elif tp_progress >= 0.80:
                target_sl = entry - tp_distance * 0.60
            elif tp_progress >= 0.60:
                target_sl = entry - tp_distance * 0.40
            else:
                target_sl = entry  # breakeven

            # Only move SL down (tighter for SELL)
            # If no SL set, use entry as baseline so breakeven triggers
            current_sl = sl if sl and sl > 0 else entry
            if target_sl >= current_sl - 0.00001:  # small buffer for float comparison
                return None

            logger.info(
                "Trailing stop: %s %s SL %.2f -> %.2f (TP progress: %.1f%%)",
                direction, pos.epic, current_sl, target_sl, tp_progress * 100,
            )
            return MonitorDecision(
                action="trail_stop",
                reason=f"Trailing stop: SL {current_sl:.2f} -> {target_sl:.2f} (TP {tp_progress:.1%})",
                symbol=pos.epic,
                direction=pos.direction,
                tp_progress=tp_progress,
                pnl=pos.upl,
                new_sl=target_sl,
            )

    def _execute_trail_stop(self, pos: OpenPosition, decision: MonitorDecision) -> None:
        """Move stop loss to protect profits."""
        result = update_position_sl_tp_unified(self.client, pos, new_sl=decision.new_sl)

        if result.success:
            # Update the tracker's stop_loss so next cycle compares against the new SL
            deal_id = str(pos.deal_id)
            tracked = self.tp_tracker.get_position(pos.epic, pos.direction, deal_id=deal_id)
            if tracked is not None and decision.new_sl is not None:
                tracked.stop_loss = decision.new_sl

            logger.info(
                "Trailing stop applied: %s %s SL %.2f -> %.2f",
                pos.direction, pos.epic, pos.stop_level or 0, decision.new_sl,
            )
        else:
            logger.error(
                "Trailing stop failed: %s %s - %s",
                pos.direction, pos.epic, result.error,
            )

    def _evaluate_strength(
        self,
        pos: OpenPosition,
        df_provider: Callable,
    ) -> StrengthScoreResult:
        """Evaluate trade strength for a position.

        Args:
            pos: The open position
            df_provider: Function to get DataFrame with indicators

        Returns:
            StrengthScoreResult
        """
        # Normalize symbol for database queries (df_provider queries DB)
        normalized_symbol = normalize_symbol(pos.epic)
        df = df_provider(normalized_symbol, "5m")

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
            from indicators.features import compute_all_features
            df = compute_all_features(df)

        # Evaluate strength
        result = self.strength_scorer.evaluate(df, pos.direction)

        logger.info(
            "Strength eval: %s %s score=%.1f decision=%s",
            pos.direction, pos.epic, result.score, result.decision,
        )

        return result

    def _execute_close(self, pos: OpenPosition, decision: MonitorDecision) -> None:
        """Execute a close decision with retry and verification."""
        logger.info(
            "Closing %s %s: %s (TP progress: %.1f%%, Strength: %.1f)",
            pos.direction, pos.epic, decision.reason,
            decision.tp_progress * 100,
            decision.strength_score or 0,
        )

        max_retries = 3
        close_result = None

        for attempt in range(1, max_retries + 1):
            close_result = close_position_unified(self.client, pos)

            if close_result.success:
                break

            logger.warning(
                "Close attempt %d/%d failed for %s %s: %s",
                attempt, max_retries, pos.direction, pos.epic, close_result.error,
            )
            if attempt < max_retries:
                import time
                time.sleep(2 * attempt)

        if close_result is None or not close_result.success:
            logger.critical(
                "FAILED to close position after %d attempts: %s %s (deal_id=%s) - MANUAL INTERVENTION REQUIRED",
                max_retries, pos.direction, pos.epic, pos.deal_id,
            )
            return

        # Verify position is actually closed
        import time
        time.sleep(2)
        broker_positions = get_open_positions_unified(self.client)
        still_open = any(str(p.deal_id) == str(pos.deal_id) for p in broker_positions)

        if still_open:
            logger.critical(
                "Position %s %s (deal_id=%s) still open after close confirmation - MANUAL INTERVENTION REQUIRED",
                pos.direction, pos.epic, pos.deal_id,
            )
            return

        # Get final stats from tracker (use normalized symbol)
        normalized_symbol = normalize_symbol(pos.epic)
        if self._on_position_closed:
            self._on_position_closed(
                symbol=normalized_symbol,
                reason="strength_exit",
                deal_id=str(pos.deal_id),
                source="position_monitor_strength",
            )
        final_stats = self.tp_tracker.close_position(
            normalized_symbol, pos.direction,
            exit_price=close_result.close_level or pos.level,
            reason=f"strength_exit:{decision.reason[:50]}",
            deal_id=str(pos.deal_id),
        )

        if final_stats:
            self._log_trade(final_stats, normalized_symbol)

        logger.info(
            "Position closed and verified: %s %s (deal_id=%s)",
            pos.direction, pos.epic, pos.deal_id,
        )

    def _log_trade(self, stats: dict, normalized_symbol: Optional[str] = None) -> None:
        """Log trade statistics to database.

        Args:
            stats: Trade statistics dictionary
            normalized_symbol: Pre-normalized symbol (optional, for efficiency)
        """
        # Use provided normalized symbol or normalize from stats
        symbol_to_store = normalized_symbol or normalize_symbol(stats["symbol"])

        with get_session() as session:
            trade_log = TradeLog(
                deal_id=str(stats.get("deal_id", "") or ""),
                symbol=symbol_to_store,
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
                session.flush()
                logger.debug("Logged trade to database: %s", symbol_to_store)
            except Exception as e:
                logger.error("Failed to log trade: %s", e)

    def add_manual_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        size: float,
        signal_id: Optional[str] = None,
    ) -> None:
        """Add a position to tracking (e.g., right after placing a trade)."""
        # Normalize symbol for database consistency
        normalized_symbol = normalize_symbol(symbol)
        tracker = self.tp_tracker.add_position(
            symbol=normalized_symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=size,
        )

        logger.info(
            "Manually added to tracking: %s %s (signal: %s)",
            direction, normalized_symbol, signal_id or "N/A",
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
