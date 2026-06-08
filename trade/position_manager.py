"""Position manager: tracks open positions, enforces cooldowns, prevents duplicate signals.

The core problem this solves:
- A losing trade stays open and blocks new winning signals
- Every 5-min cycle creates a redundant signal in the database
- After a loss, the system immediately re-enters without cooldown
- Opposite-direction signals are blocked instead of closing the old position
- Same-direction signals are skipped instead of updating SL/TP

Solution:
- Generate signals ALWAYS (for ML training), but control TRADE EXECUTION
- If new signal is OPPOSITE direction → close old position, take new (reversal)
- If new signal is SAME direction → UPDATE SL/TP (trail the stop)
- No position + no cooldown → place new trade
- After a loss, enforce a cooldown before re-entering
- Deduplicate signals: don't store a new one if a recent one matches
- SL updates are safety-checked: can only tighten, never loosen, never too aggressive
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from config.settings import SETTINGS, Symbol, TradeMode, TradingPlatform
from config.utils import broker_symbol, normalize_symbol
from data.collector import CapitalComClient
from data.database import get_session
from data.models import Signal, SignalVerification
from trade.executor import (
    OpenPosition,
    close_position,
    get_open_positions,
    get_open_positions_unified,
    has_open_position_unified,
    place_trade_unified,
    close_position_unified,
)
from trade.weekly_close import check_weekly_close_status

logger = logging.getLogger(__name__)


# ── Cooldown durations (minutes) after a position closes ──────────────────

COOLDOWN_AFTER_LOSS: dict[str, int] = {
    "scalping": 15,
    "day_trading": 30,
    "swing": 60,
}

COOLDOWN_AFTER_BREAKEVEN: dict[str, int] = {
    "scalping": 8,
    "day_trading": 15,
    "swing": 30,
}

COOLDOWN_AFTER_WIN: dict[str, int] = {
    "scalping": 5,
    "day_trading": 10,
    "swing": 15,
}

# Cooldown after strength exit (weak trend, position closed early)
# Longer than normal win because the trade was struggling
COOLDOWN_AFTER_STRENGTH_EXIT: dict[str, int] = {
    "scalping": 15,   # Same as breakeven
    "day_trading": 30,  # Same as loss cooldown
    "swing": 45,   # Between win (15) and loss (60)
}

# ── Signal dedup window (minutes) ────────────────────────────────────────

SIGNAL_DEDUP_WINDOWS: dict[str, int] = {
    "scalping": 15,
    "day_trading": 60,
    "swing": 240,
}


@dataclass
class CooldownEntry:
    """Tracks cooldown after a position closes."""
    symbol: str
    reason: str  # "win", "loss", "timeout", "external_close", "sl_close", "tp_close", "breakeven", "reversal", "strength_exit", "weekly_close", "manual_close"
    until: datetime


@dataclass
class TradeDecision:
    """Result of checking whether a trade should be placed."""
    action: str  # "place", "skip_cooldown", "reverse_and_place", "skip_correlation"
    reason: str
    existing_position: Optional[OpenPosition] = None
    cooldown_remaining_minutes: float = 0.0


class PositionManager:
    """Manages position state, cooldowns, and trade decisions.

    Key rules:
    - ALWAYS generate signals (for ML training data)
    - But control whether a TRADE is placed based on position state:
      * Same direction as open position → SKIP (already in position)
      * Opposite direction → close old position, place new trade (reversal)
      * No position + no cooldown → place trade
      * Cooldown active → skip trade
    - Deduplicate signal storage (don't store redundant signals)
    """

    def __init__(self, client: CapitalComClient):
        self.client = client
        self.cooldowns: list[CooldownEntry] = []
        self._broker_positions: list[OpenPosition] = []
        self._closed_position_keys: dict[str, datetime] = {}
        self._lock = threading.RLock()

    # ── Sync with broker ──────────────────────────────────────────────────

    def sync_positions(self) -> list:
        """Fetch open positions from the trading platform."""
        self._broker_positions = get_open_positions_unified(self.client)
        return self._broker_positions

    # ── Signal dedup ──────────────────────────────────────────────────────

    def is_duplicate_signal(
        self, symbol: str, direction: str, trade_mode: str
    ) -> bool:
        """Check if a recent signal already exists for this symbol+direction.

        Prevents storing redundant signals that would pollute ML training.
        """
        dedup_minutes = SIGNAL_DEDUP_WINDOWS.get(trade_mode, 60)
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=dedup_minutes)

        with get_session() as session:
            existing = (
                session.query(Signal)
                .filter(
                    Signal.symbol == symbol,
                    Signal.signal == direction,
                    Signal.created_at >= cutoff,
                )
                .first()
            )

            if existing:
                logger.info(
                    "Duplicate signal: %s %s already exists (id=%s, created %s)",
                    direction, symbol, existing.signal_id[:8], existing.created_at,
                )
                return True
            return False

    # ── Should we generate a signal at all? ────────────────────────────────

    def should_generate_signal(self, symbol: str) -> tuple[bool, str]:
        """Check if signal GENERATION should be skipped.

        We ALWAYS generate signals (for ML training) unless a cooldown is active
        for this symbol.

        Returns (should_generate, reason).
        """
        active_entry = self.get_active_cooldown(symbol)
        if active_entry is not None:
            remaining = self.get_cooldown_remaining_minutes(symbol)
            return False, f"Cooldown: {remaining:.0f} min remaining after {active_entry.reason}"

        return True, ""

    # ── Trade decision: should we place a trade? ──────────────────────────

    def evaluate_trade(self, symbol: str, direction: str, new_sl: float, new_tp: float) -> TradeDecision:
        """Decide what to do with a new signal.

        Returns a TradeDecision with action:
        - "place": no position, no cooldown → go ahead
        - "skip_cooldown": cooldown active → wait
        - "reverse_and_place": opposite direction → close old, place new
        - Same direction position → SKIP (already in this trade)
        """
        canonical_symbol = self.canonicalize_symbol(symbol)

        # 1. Check cooldowns
        active_cooldown = self.get_active_cooldown(canonical_symbol)
        if active_cooldown is not None:
            remaining_minutes = self.get_cooldown_remaining_minutes(canonical_symbol)
            logger.info(
                "%s: cooldown check before trade evaluation -> ACTIVE (%s, %.1f min remaining)",
                canonical_symbol,
                active_cooldown.reason,
                remaining_minutes,
            )
            return TradeDecision(
                action="skip_cooldown",
                reason=f"Cooldown: {remaining_minutes:.0f} min left after {active_cooldown.reason}",
                cooldown_remaining_minutes=remaining_minutes,
            )

        logger.info("%s: cooldown check before trade evaluation -> CLEAR", canonical_symbol)

        # 2. Check open positions
        self.sync_positions()

        for pos in self._broker_positions:
            if self._position_matches_symbol(pos.epic, canonical_symbol):
                pos_dir = pos.direction.upper()
                new_dir = direction.upper()

                if pos_dir == new_dir:
                    # Same direction — skip, already in this trade
                    return TradeDecision(
                        action="skip_same_direction",
                        reason=f"Open {pos_dir} {canonical_symbol} — same direction, skipping",
                        cooldown_remaining_minutes=0.0,
                    )

                # Opposite direction — close old, take new
                return TradeDecision(
                    action="reverse_and_place",
                    reason=(
                        f"Open {pos_dir} {canonical_symbol} opposes new {new_dir} — "
                        f"close position {pos.deal_id} and reverse"
                    ),
                    existing_position=pos,
                )

        # 3. Check correlation limits (prevent overexposure in correlated groups)
        from trade.correlation import check_correlation_limit

        correlation_check = check_correlation_limit(symbol, self.client)
        if not correlation_check.allowed:
            return TradeDecision(
                action="skip_correlation",
                reason=correlation_check.reason,
                cooldown_remaining_minutes=0.0,
            )

        # 4. No position, no cooldown, no correlation issue → place trade
        return TradeDecision(
            action="place",
            reason="No open position or cooldown",
        )

    # ── Execute a trade decision ──────────────────────────────────────────

    def execute_trade_decision(
        self, trade_signal, decision: TradeDecision
    ) -> dict:
        """Execute the trade based on the decision.

        Returns a dict with:
            success: bool
            action: str (what was done)
            reason: str
            deal_reference: Optional[str]
            closed_position: Optional[str]  # if we reversed
        """
        from trade.executor import place_trade_unified

        result = {
            "success": False,
            "action": decision.action,
            "reason": decision.reason,
            "deal_reference": None,
            "closed_position": None,
        }

        if decision.action == "skip_cooldown":
            logger.info("Trade skipped: %s", decision.reason)
            return result

        if decision.action == "reverse_and_place":
            # Close the existing position first
            pos = decision.existing_position
            if pos is None:
                result["reason"] = "Missing existing position for reversal"
                logger.error(result["reason"])
                return result
            logger.info(
                "Reversing position: closing %s %s (deal %s) to take opposite trade",
                pos.direction, pos.epic, pos.deal_id,
            )
            close_result = close_position_unified(self.client, pos)
            if not close_result.success:
                logger.error("Failed to close position %s: %s", pos.deal_id, close_result.error)
                result["reason"] = f"Failed to close existing position: {close_result.error}"
                return result

            self.register_position_close(
                symbol=trade_signal.symbol,
                reason="reversal",
                deal_id=pos.deal_id,
                source="reverse_and_place",
            )

            result["closed_position"] = f"{pos.direction} {pos.epic} (deal {pos.deal_id})"
            logger.info("Position closed successfully: %s", close_result.deal_id)

        if decision.action in ("place", "reverse_and_place"):
            # Place the new trade
            size = SETTINGS.get_trade_size(trade_signal.symbol)
            order_result = place_trade_unified(self.client, trade_signal, size=size)

            if order_result.success:
                result["success"] = True
                result["deal_reference"] = order_result.deal_reference
                logger.info(
                    "Trade placed: %s %s size=%s deal=%s",
                    trade_signal.signal.value, trade_signal.symbol, size,
                    order_result.deal_reference,
                )
            else:
                result["reason"] = order_result.error or "Order failed"
                logger.error("Trade failed: %s", order_result.error)

        return result

    # ── Register events ────────────────────────────────────────────────────

    def register_position_close(
        self,
        symbol: str,
        reason: str,
        deal_id: Optional[str] = None,
        source: str = "unknown",
    ) -> CooldownEntry:
        """Register a confirmed position close and start the symbol cooldown."""
        canonical_symbol = self.canonicalize_symbol(symbol)
        trade_mode = SETTINGS.trade_mode.value
        now = datetime.now(timezone.utc)
        close_key = str(deal_id) if deal_id else None

        with self._lock:
            self._cleanup_expired_close_keys(now)

            if close_key and close_key in self._closed_position_keys:
                existing_entry = self.get_active_cooldown(canonical_symbol)
                logger.info(
                    "Position close already registered: symbol=%s deal_id=%s reason=%s source=%s",
                    canonical_symbol,
                    close_key,
                    reason,
                    source,
                )
                return existing_entry or CooldownEntry(
                    symbol=canonical_symbol,
                    reason=reason,
                    until=now,
                )

            cooldown_minutes = self._resolve_cooldown_minutes(trade_mode, reason)
            until = now + timedelta(minutes=cooldown_minutes)

            # Keep the longest active cooldown for the symbol to avoid stacking
            # duplicate entries from competing close paths.
            active_for_symbol = [
                cd for cd in self.cooldowns
                if cd.symbol == canonical_symbol and cd.until > now
            ]
            if active_for_symbol:
                longest_until = max(cd.until for cd in active_for_symbol)
                if longest_until > until:
                    until = longest_until

            self.cooldowns = [
                cd for cd in self.cooldowns
                if not (cd.symbol == canonical_symbol and cd.until > now)
            ]
            entry = CooldownEntry(
                symbol=canonical_symbol,
                reason=reason,
                until=until,
            )
            self.cooldowns.append(entry)
            if close_key:
                self._closed_position_keys[close_key] = until

        logger.info(
            "Position closed: symbol=%s reason=%s deal_id=%s source=%s",
            canonical_symbol,
            reason,
            close_key or "n/a",
            source,
        )
        logger.info("Close reason: %s %s", canonical_symbol, reason)
        logger.info(
            "Cooldown started: %s — %d min after %s (expires %s UTC)",
            canonical_symbol,
            cooldown_minutes,
            reason,
            until.strftime("%H:%M:%S"),
        )
        logger.info(
            "Cooldown remaining: %s %.1f min",
            canonical_symbol,
            max((until - now).total_seconds() / 60, 0.0),
        )
        return entry

    def register_closed(
        self,
        symbol: str,
        reason: str,
        deal_id: Optional[str] = None,
        source: str = "legacy",
    ) -> CooldownEntry:
        """Backward-compatible alias for close registration."""
        return self.register_position_close(
            symbol=symbol,
            reason=reason,
            deal_id=deal_id,
            source=source,
        )

    def register_verified_result(self, symbol: str, result: str, signal_id: Optional[str] = None) -> None:
        """Register a verified signal result. Starts cooldown based on outcome.

        Uses signal_id (when available) as the deal_id for dedup so the same
        verified result cannot register a duplicate cooldown across cycles.
        """
        # Build a dedup key from signal_id, or fall back to a stable key
        # based on symbol+result so that re-processing the same outcome
        # within the query window doesn't stack cooldowns.
        dedup_key = signal_id if signal_id else None

        if result == "LOSS":
            self.register_position_close(symbol, "loss", deal_id=dedup_key, source="verified_result")
        elif result == "BREAKEVEN":
            self.register_position_close(symbol, "breakeven", deal_id=dedup_key, source="verified_result")
        elif result == "WIN":
            self.register_position_close(symbol, "win", deal_id=dedup_key, source="verified_result")

    # ── Cleanup ────────────────────────────────────────────────────────────

    def cleanup_expired_cooldowns(self) -> None:
        """Remove expired cooldowns."""
        now = datetime.now(timezone.utc)
        with self._lock:
            expired = [cd for cd in self.cooldowns if cd.until <= now]
            for cd in expired:
                logger.debug("Cooldown expired: %s after %s", cd.symbol, cd.reason)
            self.cooldowns = [cd for cd in self.cooldowns if cd.until > now]
            self._cleanup_expired_close_keys(now)

    # ── Weekly Close Handling ───────────────────────────────────────────────

    def should_close_for_weekly_close(self, symbol: str) -> bool:
        """Check if positions should be closed due to weekly close.

        Args:
            symbol: The symbol to check

        Returns:
            True if positions for this symbol should be closed
        """
        if not SETTINGS.weekly_close.enabled:
            return False

        close_status = check_weekly_close_status(
            symbol=symbol,
            stop_new_trades_hours=SETTINGS.weekly_close.stop_new_trades_hours,
            close_positions_hours=SETTINGS.weekly_close.close_positions_hours,
        )
        return close_status.should_close_positions

    def close_all_for_weekly_close(self, symbol: str) -> list:
        """Close all positions for a symbol due to weekly close.

        Args:
            symbol: The symbol to close positions for

        Returns:
            List of close results for positions that were closed
        """
        if not SETTINGS.weekly_close.enabled:
            logger.debug("Weekly close handling disabled, skipping position close")
            return []

        # Check if we should close
        if not self.should_close_for_weekly_close(symbol):
            logger.debug("Not in weekly close window for %s", symbol)
            return []

        logger.info("Weekly close: closing all positions for %s", symbol)

        # Get all open positions
        self.sync_positions()
        epic = self._get_epic(symbol)

        if not epic:
            logger.warning("No epic found for symbol %s", symbol)
            return []

        closed_positions = []
        for pos in self._broker_positions:
            # Normalize both for comparison (handle 'm' suffix)
            normalized_epic = normalize_symbol(pos.epic)
            normalized_symbol = normalize_symbol(symbol)

            if normalized_epic == normalized_symbol:
                logger.info(
                    "Closing position for weekly close: %s %s (deal %s)",
                    pos.direction, pos.epic, pos.deal_id
                )
                close_result = close_position_unified(self.client, pos)
                if close_result.success:
                    closed_positions.append(close_result)
                    self.register_position_close(
                        symbol=symbol,
                        reason="weekly_close",
                        deal_id=pos.deal_id,
                        source="weekly_close",
                    )
                else:
                    logger.error(
                        "Failed to close position %s: %s",
                        pos.deal_id, close_result.error
                    )

        logger.info("Weekly close: closed %d positions for %s", len(closed_positions), symbol)
        return closed_positions

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _get_epic(symbol: str) -> Optional[str]:
        """Get the broker symbol identifier for the active platform.

        For MT5, adds 'm' suffix for broker communication.
        For Capital.com, maps to the epic identifier.
        """
        if SETTINGS.trading_platform == TradingPlatform.MT5:
            # MT5 broker uses 'm' suffix (e.g., BTCUSDm)
            return broker_symbol(symbol)
        from data.collector import _EPIC_MAP
        try:
            symbol_enum = Symbol(symbol)
            return _EPIC_MAP.get(symbol_enum)
        except (ValueError, KeyError):
            return None

    def canonicalize_symbol(self, symbol: str) -> str:
        """Resolve broker/strategy symbol variants to the configured symbol key."""
        normalized = normalize_symbol(symbol)
        if not normalized:
            return normalized

        active_candidates = [s.value if hasattr(s, "value") else str(s) for s in SETTINGS.symbols]
        all_candidates = active_candidates + [
            s.value for s in Symbol
            if s.value not in active_candidates
        ]

        matches: list[str] = []
        for candidate in all_candidates:
            candidate_key = normalize_symbol(candidate)
            if candidate_key == normalized:
                return candidate

            candidate_epic = self._get_epic(candidate)
            if candidate_epic and normalize_symbol(candidate_epic) == normalized:
                matches.append(candidate)

        if len(matches) > 1:
            logger.warning(
                "Ambiguous symbol normalization for %s -> %s; using %s",
                symbol,
                matches,
                matches[0],
            )
        if matches:
            return matches[0]

        return normalized

    def get_active_cooldown(self, symbol: str) -> Optional[CooldownEntry]:
        """Return the active cooldown for a symbol, if any."""
        canonical_symbol = self.canonicalize_symbol(symbol)
        now = datetime.now(timezone.utc)
        with self._lock:
            active_entries = [
                cd for cd in self.cooldowns
                if cd.symbol == canonical_symbol and cd.until > now
            ]

        if not active_entries:
            return None

        return max(active_entries, key=lambda entry: entry.until)

    def get_cooldown_remaining_minutes(self, symbol: str) -> float:
        """Return active cooldown minutes remaining for the symbol."""
        active_entry = self.get_active_cooldown(symbol)
        if active_entry is None:
            return 0.0

        remaining_seconds = (active_entry.until - datetime.now(timezone.utc)).total_seconds()
        return max(remaining_seconds / 60, 0.0)

    def is_active(self, symbol: str) -> bool:
        """Check whether a symbol is currently in cooldown."""
        canonical_symbol = self.canonicalize_symbol(symbol)
        active_entry = self.get_active_cooldown(canonical_symbol)
        if active_entry is None:
            logger.info("%s: cooldown check before signal generation -> CLEAR", canonical_symbol)
            return False

        remaining = self.get_cooldown_remaining_minutes(canonical_symbol)
        logger.info(
            "%s: cooldown check before signal generation -> ACTIVE (%s, %.1f min remaining)",
            canonical_symbol,
            active_entry.reason,
            remaining,
        )
        return True

    def _position_matches_symbol(self, position_symbol: str, requested_symbol: str) -> bool:
        """Compare broker symbols and strategy symbols using the canonical key."""
        return self.canonicalize_symbol(position_symbol) == self.canonicalize_symbol(requested_symbol)

    @staticmethod
    def _resolve_cooldown_minutes(trade_mode: str, reason: str) -> int:
        """Resolve cooldown duration for a given close reason."""
        if reason in ("loss", "timeout", "external_close", "sl_close", "weekly_close"):
            return COOLDOWN_AFTER_LOSS.get(trade_mode, 30)
        if reason == "breakeven":
            return COOLDOWN_AFTER_BREAKEVEN.get(trade_mode, 15)
        if reason in ("win", "tp_close"):
            return COOLDOWN_AFTER_WIN.get(trade_mode, 10)
        if reason == "reversal":
            return 2
        if reason == "strength_exit":
            return COOLDOWN_AFTER_STRENGTH_EXIT.get(trade_mode, 30)
        if reason == "manual_close":
            return 5
        return 5

    def _cleanup_expired_close_keys(self, now: datetime) -> None:
        """Drop dedupe keys after their associated cooldown has expired."""
        self._closed_position_keys = {
            key: until for key, until in self._closed_position_keys.items()
            if until > now
        }

    def get_status(self, symbol: str) -> dict:
        """Get current position manager status for a symbol."""
        now = datetime.now(timezone.utc)
        canonical_symbol = self.canonicalize_symbol(symbol)

        with self._lock:
            active_cooldowns = [
                cd for cd in self.cooldowns
                if cd.symbol == canonical_symbol and cd.until > now
            ]

        open_for_symbol = []
        for pos in self._broker_positions:
            if self._position_matches_symbol(pos.epic, canonical_symbol):
                open_for_symbol.append(pos)

        return {
            "has_open_position": len(open_for_symbol) > 0,
            "open_positions": open_for_symbol,
            "cooldowns": active_cooldowns,
            "can_trade": not active_cooldowns and not open_for_symbol,
        }
