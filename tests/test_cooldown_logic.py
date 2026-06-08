"""Tests for cooldown logic in position_manager.

Covers:
- Cooldown registration and expiry
- Breakeven vs loss cooldown tiers
- Verified result cooldowns (LOSS, BREAKEVEN, WIN)
- Deal-ID dedup preventing duplicate cooldowns
- _resolve_cooldown_minutes for all reason types
"""
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

from config.settings import SETTINGS, Symbol, TradeMode
from trade.executor import OpenPosition
from trade.position_manager import (
    COOLDOWN_AFTER_BREAKEVEN,
    COOLDOWN_AFTER_LOSS,
    COOLDOWN_AFTER_WIN,
    PositionManager,
)
from trade.position_monitor import PositionMonitor
from trade.tp_tracker import TPProgress

# Current trade mode from config (frozen dataclass — read only)
TRADE_MODE = SETTINGS.trade_mode.value  # e.g. "day_trading"


class CooldownLogicTests(unittest.TestCase):
    """Original cooldown tests + new bug-fix tests."""

    def test_timeout_close_starts_cooldown(self) -> None:
        manager = PositionManager(Mock())

        entry = manager.register_position_close(
            symbol="BTCUSD",
            reason="timeout",
            deal_id="timeout-1",
            source="unit_test",
        )

        self.assertEqual(entry.symbol, "BTCUSD")
        self.assertEqual(entry.reason, "timeout")
        self.assertTrue(manager.is_active("BTCUSD"))
        self.assertGreaterEqual(manager.get_cooldown_remaining_minutes("BTCUSD"), 29.0)

    def test_external_close_starts_cooldown(self) -> None:
        client = Mock()
        manager = PositionManager(client)
        monitor = PositionMonitor(client, on_position_closed=manager.register_position_close)
        position = OpenPosition(
            deal_id="external-1",
            epic="BTCUSDm",
            direction="BUY",
            size=0.1,
            level=100000.0,
            stop_level=99000.0,
            profit_level=101000.0,
            upl=0.0,
            created_date=datetime.now(timezone.utc).isoformat(),
        )

        with patch("trade.position_monitor.get_open_positions_unified", side_effect=[[position], []]):
            monitor.sync_positions()
            self.assertFalse(manager.is_active("BTCUSD"))

            monitor.sync_positions()

        active_entry = manager.get_active_cooldown("BTCUSD")
        if active_entry is None:
            self.fail("Expected an active cooldown after external close detection")
        self.assertEqual(active_entry.reason, "external_close")
        self.assertTrue(manager.is_active("BTCUSD"))

    def test_no_trade_can_be_opened_during_cooldown(self) -> None:
        manager = PositionManager(Mock())
        manager.register_position_close(
            symbol="BTCUSD",
            reason="timeout",
            deal_id="timeout-2",
            source="unit_test",
        )

        with patch.object(manager, "sync_positions"), patch(
            "trade.correlation.check_correlation_limit",
            return_value=SimpleNamespace(allowed=True, reason=""),
        ):
            decision = manager.evaluate_trade("BTCUSD", "BUY", 0.0, 0.0)

        self.assertEqual(decision.action, "skip_cooldown")
        self.assertGreater(decision.cooldown_remaining_minutes, 0.0)

    def test_trade_can_open_after_cooldown_expires(self) -> None:
        manager = PositionManager(Mock())
        manager.register_position_close(
            symbol="BTCUSD",
            reason="timeout",
            deal_id="timeout-3",
            source="unit_test",
        )
        manager.cooldowns[0].until = datetime.now(timezone.utc) - timedelta(minutes=1)

        with patch.object(manager, "sync_positions"), patch(
            "trade.correlation.check_correlation_limit",
            return_value=SimpleNamespace(allowed=True, reason=""),
        ):
            decision = manager.evaluate_trade("BTCUSD", "BUY", 0.0, 0.0)

        self.assertEqual(decision.action, "place")

    def test_trade_decision_sync_blocks_trade_after_external_close(self) -> None:
        from main import _handle_trade_decision

        manager = PositionManager(Mock())
        monitor = Mock()
        monitor.sync_positions.side_effect = lambda: manager.register_position_close(
            symbol="BTCUSD",
            reason="external_close",
            deal_id="external-2",
            source="unit_test",
        )
        trade_signal = cast(Any, SimpleNamespace(
            symbol="BTCUSD",
            signal=SimpleNamespace(value="BUY"),
            stop_loss=99000.0,
            take_profit=101000.0,
        ))
        stats = {"trade_cooldown": False}

        with patch("main.display_trade_skipped_cooldown"), patch("main.display_trade_skipped_existing"), patch(
            "main.display_trade_failed"
        ), patch("main.display_trade_placed"):
            placed = _handle_trade_decision(
                client=Mock(),
                trade_signal=trade_signal,
                position_manager=manager,
                stats=stats,
                position_monitor=monitor,
            )

        self.assertFalse(placed)
        self.assertTrue(stats["trade_cooldown"])
        self.assertTrue(manager.is_active("BTCUSD"))

    def test_external_close_classifies_tp_when_tp_was_observed(self) -> None:
        monitor = PositionMonitor(Mock())
        tracker = TPProgress(
            symbol="BTCUSD",
            direction="BUY",
            entry_price=100000.0,
            stop_loss=99000.0,
            take_profit=101000.0,
            size=0.1,
            opened_at=datetime.now(timezone.utc),
        )
        tracker.highest_tp_progress = 1.05

        close_reason, reason_text = monitor._classify_external_close(tracker)

        self.assertEqual(close_reason, "tp_close")
        self.assertIn("Take profit", reason_text)

    # ── Bug-fix tests ─────────────────────────────────────────────────────

    def test_breakeven_uses_breakeven_cooldown_not_loss(self) -> None:
        """BREAKEVEN should use shorter breakeven cooldown, not loss cooldown."""
        manager = PositionManager(Mock())

        entry = manager.register_position_close(
            symbol="BTCUSD",
            reason="breakeven",
            deal_id="be-1",
            source="unit_test",
        )

        # The breakeven cooldown should be shorter than the loss cooldown
        expected_be = COOLDOWN_AFTER_BREAKEVEN.get(TRADE_MODE, 15)
        expected_loss = COOLDOWN_AFTER_LOSS.get(TRADE_MODE, 30)
        remaining = manager.get_cooldown_remaining_minutes("BTCUSD")

        self.assertGreater(remaining, 0.0)
        self.assertLessEqual(remaining, expected_be + 0.5,
                             f"Breakeven cooldown ({remaining:.1f} min) should be ≤ {expected_be} min, "
                             f"not the loss cooldown of {expected_loss} min")

    def test_verified_result_breakeven_uses_breakeven_tier(self) -> None:
        """register_verified_result with BREAKEVEN should use breakeven tier, not loss."""
        manager = PositionManager(Mock())

        manager.register_verified_result("BTCUSD", "BREAKEVEN", signal_id="sig-be-1")

        remaining = manager.get_cooldown_remaining_minutes("BTCUSD")
        expected_be = COOLDOWN_AFTER_BREAKEVEN.get(TRADE_MODE, 15)
        expected_loss = COOLDOWN_AFTER_LOSS.get(TRADE_MODE, 30)

        self.assertGreater(remaining, 0.0)
        self.assertLessEqual(remaining, expected_be + 0.5,
                             f"BREAKEVEN verified result cooldown ({remaining:.1f} min) "
                             f"should use breakeven tier ({expected_be} min), not loss tier ({expected_loss} min)")

    def test_verified_result_win_registers_cooldown(self) -> None:
        """WIN verified results should register a win cooldown."""
        manager = PositionManager(Mock())

        manager.register_verified_result("BTCUSD", "WIN", signal_id="sig-win-1")

        remaining = manager.get_cooldown_remaining_minutes("BTCUSD")
        expected = COOLDOWN_AFTER_WIN.get(TRADE_MODE, 10)

        self.assertGreater(remaining, 0.0)
        self.assertLessEqual(remaining, expected + 0.5,
                             f"WIN verified result cooldown ({remaining:.1f} min) "
                             f"should be ≤ {expected} min")

    def test_verified_result_dedup_prevents_duplicate_cooldown(self) -> None:
        """Passing the same signal_id twice should not create duplicate cooldowns."""
        manager = PositionManager(Mock())

        # First registration
        manager.register_verified_result("BTCUSD", "LOSS", signal_id="sig-dedup-1")
        remaining1 = manager.get_cooldown_remaining_minutes("BTCUSD")
        self.assertGreater(remaining1, 0.0)

        # Second registration with same signal_id — should be deduped
        manager.register_verified_result("BTCUSD", "LOSS", signal_id="sig-dedup-1")

        # Count cooldowns for BTCUSD — should be exactly 1
        canonical = manager.canonicalize_symbol("BTCUSD")
        btc_cooldowns = [cd for cd in manager.cooldowns if cd.symbol == canonical]
        self.assertEqual(len(btc_cooldowns), 1,
                         f"Expected 1 cooldown after dedup, got {len(btc_cooldowns)}")

    def test_verified_result_without_signal_id_still_works(self) -> None:
        """register_verified_result should still work when signal_id is None."""
        manager = PositionManager(Mock())

        manager.register_verified_result("BTCUSD", "LOSS")
        self.assertTrue(manager.is_active("BTCUSD"))
        remaining = manager.get_cooldown_remaining_minutes("BTCUSD")
        self.assertGreater(remaining, 0.0)

    def test_resolve_cooldown_minutes_all_reasons(self) -> None:
        """Verify _resolve_cooldown_minutes returns correct durations for all reasons."""
        manager = PositionManager(Mock())

        # Loss-tier reasons
        for reason in ("loss", "timeout", "external_close", "sl_close", "weekly_close"):
            self.assertEqual(
                manager._resolve_cooldown_minutes(TRADE_MODE, reason, "BTCUSD"),
                COOLDOWN_AFTER_LOSS.get(TRADE_MODE, 30),
                f"Reason '{reason}' should use LOSS cooldown tier",
            )

        # Breakeven
        self.assertEqual(
            manager._resolve_cooldown_minutes(TRADE_MODE, "breakeven", "BTCUSD"),
            COOLDOWN_AFTER_BREAKEVEN.get(TRADE_MODE, 15),
        )

        # Win-tier reasons
        for reason in ("win", "tp_close"):
            self.assertEqual(
                manager._resolve_cooldown_minutes(TRADE_MODE, reason, "BTCUSD"),
                COOLDOWN_AFTER_WIN.get(TRADE_MODE, 10),
                f"Reason '{reason}' should use WIN cooldown tier",
            )

        # Reversal — fixed 2 min
        self.assertEqual(manager._resolve_cooldown_minutes(TRADE_MODE, "reversal", "BTCUSD"), 2)

        # Strength exit
        self.assertEqual(
            manager._resolve_cooldown_minutes(TRADE_MODE, "strength_exit", "BTCUSD"),
            30,  # COOLDOWN_AFTER_STRENGTH_EXIT["day_trading"]
        )

        # Manual close — fixed 5 min
        self.assertEqual(manager._resolve_cooldown_minutes(TRADE_MODE, "manual_close", "BTCUSD"), 5)

        # Unknown — default 5 min
        self.assertEqual(manager._resolve_cooldown_minutes(TRADE_MODE, "unknown_reason", "BTCUSD"), 5)

    def test_breakeven_cooldown_shorter_than_loss(self) -> None:
        """Breakeven cooldown must always be shorter than loss cooldown across all modes."""
        for mode in ("scalping", "day_trading", "swing"):
            be = COOLDOWN_AFTER_BREAKEVEN.get(mode, 15)
            loss = COOLDOWN_AFTER_LOSS.get(mode, 30)
            self.assertLessEqual(be, loss,
                                 f"Breakeven ({be} min) should be ≤ loss ({loss} min) for {mode}")

    def test_win_cooldown_shorter_than_loss(self) -> None:
        """Win cooldown must always be shorter than loss cooldown across all modes."""
        for mode in ("scalping", "day_trading", "swing"):
            win = COOLDOWN_AFTER_WIN.get(mode, 10)
            loss = COOLDOWN_AFTER_LOSS.get(mode, 30)
            self.assertLessEqual(win, loss,
                                 f"Win ({win} min) should be ≤ loss ({loss} min) for {mode}")

    # ── Fast market cooldown tests ────────────────────────────────────────

    def test_fast_market_doubles_cooldown(self) -> None:
        """When ATR ratio >= threshold, cooldowns should be multiplied by the configured multiplier."""
        manager = PositionManager(Mock())

        # Set ATR ratio above threshold (default 1.5)
        manager.update_atr_ratio("BTCUSD", 1.8)

        # Register a loss cooldown — should be extended
        entry = manager.register_position_close(
            symbol="BTCUSD",
            reason="loss",
            deal_id="fast-market-1",
            source="unit_test",
        )

        # Default trade mode is swing, so loss cooldown is 60 min
        # With 2x multiplier, it should be 120 min
        expected_base = COOLDOWN_AFTER_LOSS.get(TRADE_MODE, 30)
        expected_extended = int(expected_base * SETTINGS.fast_market.cooldown_multiplier)
        remaining = manager.get_cooldown_remaining_minutes("BTCUSD")

        self.assertGreaterEqual(remaining, expected_extended - 1.0,
                                 f"Fast market cooldown ({remaining:.1f} min) should be ~{expected_extended} min "
                                 f"(base {expected_base} x {SETTINGS.fast_market.cooldown_multiplier})")
        self.assertEqual(entry.fast_market_multiplier, SETTINGS.fast_market.cooldown_multiplier)

    def test_normal_market_no_extension(self) -> None:
        """When ATR ratio is below threshold, cooldowns should NOT be extended."""
        manager = PositionManager(Mock())

        # Set ATR ratio below threshold
        manager.update_atr_ratio("BTCUSD", 1.1)

        entry = manager.register_position_close(
            symbol="BTCUSD",
            reason="loss",
            deal_id="normal-market-1",
            source="unit_test",
        )

        expected_base = COOLDOWN_AFTER_LOSS.get(TRADE_MODE, 30)
        remaining = manager.get_cooldown_remaining_minutes("BTCUSD")

        # Should be the base cooldown, not extended
        self.assertLessEqual(remaining, expected_base + 1.0,
                             f"Normal market cooldown ({remaining:.1f} min) should be {expected_base} min")
        self.assertEqual(entry.fast_market_multiplier, 1.0)

    def test_no_atr_ratio_defaults_to_normal(self) -> None:
        """When no ATR ratio is cached for a symbol, cooldowns should use base duration."""
        manager = PositionManager(Mock())

        # Don't call update_atr_ratio at all — should default to 1.0 multiplier
        entry = manager.register_position_close(
            symbol="ETHUSD",
            reason="loss",
            deal_id="no-atr-1",
            source="unit_test",
        )

        expected_base = COOLDOWN_AFTER_LOSS.get(TRADE_MODE, 30)
        remaining = manager.get_cooldown_remaining_minutes("ETHUSD")

        self.assertLessEqual(remaining, expected_base + 1.0,
                             f"No-ATR cooldown ({remaining:.1f} min) should be base {expected_base} min")
        self.assertEqual(entry.fast_market_multiplier, 1.0)

    def test_fast_market_multiplier_varies_by_reason(self) -> None:
        """Fast market multiplier should apply to all cooldown reasons."""
        manager = PositionManager(Mock())
        manager.update_atr_ratio("BTCUSD", 2.0)  # Well above threshold

        multiplier = SETTINGS.fast_market.cooldown_multiplier

        # Test a few different reasons
        for reason in ("loss", "breakeven", "win"):
            # Clean up cooldowns between iterations
            manager.cooldowns.clear()
            manager._closed_position_keys.clear()

            entry = manager.register_position_close(
                symbol="BTCUSD",
                reason=reason,
                deal_id=f"fast-{reason}-1",
                source="unit_test",
            )
            self.assertEqual(entry.fast_market_multiplier, multiplier,
                             f"Reason '{reason}' should have fast market multiplier {multiplier}")

    def test_atr_threshold_boundary(self) -> None:
        """ATR ratio exactly at threshold should trigger fast market multiplier."""
        manager = PositionManager(Mock())

        # Exactly at threshold (default 1.5)
        manager.update_atr_ratio("BTCUSD", SETTINGS.fast_market.atr_threshold)

        entry = manager.register_position_close(
            symbol="BTCUSD",
            reason="loss",
            deal_id="boundary-1",
            source="unit_test",
        )
        self.assertEqual(entry.fast_market_multiplier, SETTINGS.fast_market.cooldown_multiplier)

        # Just below threshold
        manager.update_atr_ratio("BTCUSD", SETTINGS.fast_market.atr_threshold - 0.01)
        manager.cooldowns.clear()
        manager._closed_position_keys.clear()

        entry = manager.register_position_close(
            symbol="BTCUSD",
            reason="loss",
            deal_id="boundary-2",
            source="unit_test",
        )
        self.assertEqual(entry.fast_market_multiplier, 1.0)

    # ── Late verified result cooldown fix ────────────────────────────────────

    def test_verified_result_with_closed_at_skips_expired_cooldown(self) -> None:
        """A verified result with closed_at in the past should NOT register a
        new cooldown if the cooldown would have already expired by now.

        This is the core fix for the bug where verified results arriving late
        (after the original cooldown expired) would re-register a fresh cooldown
        from `now`, extending the cooldown far beyond its intended expiry.
        """
        manager = PositionManager(Mock())

        # Simulate a trade that closed 60 minutes ago
        closed_at = datetime.now(timezone.utc) - timedelta(minutes=60)

        # LOSS cooldown for day_trading is 30 min — would have expired 30 min ago
        manager.register_verified_result("XAUUSD", "LOSS", signal_id="sig-late-1", closed_at=closed_at)

        # No cooldown should be active — it would have already expired
        self.assertFalse(manager.is_active("XAUUSD"),
                         "Late verified result should NOT register a cooldown that already expired")
        self.assertEqual(manager.get_cooldown_remaining_minutes("XAUUSD"), 0.0)

    def test_verified_result_with_recent_closed_at_registers_cooldown(self) -> None:
        """A verified result with a recent closed_at (cooldown not yet expired)
        should register a cooldown anchored to the actual close time.
        """
        manager = PositionManager(Mock())

        # Simulate a trade that closed 5 minutes ago
        closed_at = datetime.now(timezone.utc) - timedelta(minutes=5)

        # LOSS cooldown for day_trading is 30 min — should expire in 25 min
        manager.register_verified_result("XAUUSD", "LOSS", signal_id="sig-recent-1", closed_at=closed_at)

        self.assertTrue(manager.is_active("XAUUSD"))
        remaining = manager.get_cooldown_remaining_minutes("XAUUSD")
        expected = COOLDOWN_AFTER_LOSS.get(TRADE_MODE, 30) - 5  # 30 min - 5 min already passed
        # Allow 1 min tolerance for test execution time
        self.assertAlmostEqual(remaining, expected, delta=1.0,
                               msg=f"Expected ~{expected:.0f} min remaining, got {remaining:.1f}")

    def test_verified_result_without_closed_at_uses_now(self) -> None:
        """Backward compatibility: verified results without closed_at use `now`
        as the cooldown start (original behavior).
        """
        manager = PositionManager(Mock())

        manager.register_verified_result("XAUUSD", "LOSS", signal_id="sig-no-closed-at")

        self.assertTrue(manager.is_active("XAUUSD"))
        remaining = manager.get_cooldown_remaining_minutes("XAUUSD")
        expected = COOLDOWN_AFTER_LOSS.get(TRADE_MODE, 30)
        self.assertAlmostEqual(remaining, expected, delta=1.0)

    def test_no_double_cooldown_from_verified_after_original_expires(self) -> None:
        """End-to-end reproduction of the original bug:

        1. Position close registers a cooldown (e.g., 10 min for WIN)
        2. Original cooldown expires
        3. Verified result arrives and should NOT register a new cooldown
           because closed_at proves the trade closed long ago.
        """
        manager = PositionManager(Mock())

        # Step 1: Original close registers a cooldown
        closed_at = datetime.now(timezone.utc) - timedelta(minutes=15)
        manager.register_position_close(
            "XAUUSD", "win", deal_id="deal-1", source="position_monitor",
            closed_at=closed_at,
        )
        # WIN cooldown for day_trading is 10 min — already expired (closed 15 min ago)
        self.assertFalse(manager.is_active("XAUUSD"),
                         "WIN cooldown from 15 min ago should already be expired")

        # Step 2: Verified result arrives — should NOT re-register
        manager.register_verified_result("XAUUSD", "WIN", signal_id="sig-2", closed_at=closed_at)

        self.assertFalse(manager.is_active("XAUUSD"),
                         "Late verified result should not re-register an expired cooldown")


if __name__ == "__main__":
    unittest.main()