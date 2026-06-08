import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

from config.settings import SETTINGS, Symbol, TradeMode
from trade.executor import OpenPosition
from trade.position_manager import PositionManager
from trade.position_monitor import PositionMonitor
from trade.tp_tracker import TPProgress


class CooldownLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trade_mode_patch = patch.object(SETTINGS, "trade_mode", TradeMode.DAY_TRADING)
        self.symbols_patch = patch.object(SETTINGS, "symbols", (Symbol.BTCUSD,))
        self.trade_mode_patch.start()
        self.symbols_patch.start()

    def tearDown(self) -> None:
        self.symbols_patch.stop()
        self.trade_mode_patch.stop()

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


if __name__ == "__main__":
    unittest.main()
