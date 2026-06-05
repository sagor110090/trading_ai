"""Quick test script for all recent changes."""

import sys

def test_imports():
    """Test all new modules can be imported."""
    print("Testing imports...")
    try:
        from trade.weekly_close import check_weekly_close_status, get_asset_class
        print("  ✓ weekly_close")
    except Exception as e:
        print(f"  ✗ weekly_close: {e}")
        return False

    try:
        from config.settings import SETTINGS, WeeklyCloseConfig
        print("  ✓ settings (WeeklyCloseConfig)")
    except Exception as e:
        print(f"  ✗ settings: {e}")
        return False

    try:
        from trade.position_manager import (
            PositionManager, COOLDOWN_AFTER_STRENGTH_EXIT
        )
        print("  ✓ position_manager (COOLDOWN_AFTER_STRENGTH_EXIT)")
    except Exception as e:
        print(f"  ✗ position_manager: {e}")
        return False

    return True


def test_weekly_close():
    """Test weekly close logic."""
    print("\nTesting weekly close logic...")
    from trade.weekly_close import check_weekly_close_status, get_asset_class

    # Test asset classes
    assert get_asset_class("BTCUSD") == "crypto", "BTCUSD should be crypto"
    assert get_asset_class("XAUUSD") == "metals", "XAUUSD should be metals"
    assert get_asset_class("EURUSD") == "forex", "EURUSD should be forex"
    print("  ✓ Asset class detection")

    # Test weekly close status
    symbols = ["BTCUSD", "XAUUSD", "EURUSD"]
    for sym in symbols:
        status = check_weekly_close_status(sym)
        print(f"  {sym:10} | {status.asset_class:10} | {status.message}")

    print("  ✓ Weekly close status check")


def test_cooldowns():
    """Test cooldown values."""
    print("\nTesting cooldowns...")
    from trade.position_manager import (
        COOLDOWN_AFTER_WIN,
        COOLDOWN_AFTER_STRENGTH_EXIT,
        COOLDOWN_AFTER_BREAKEVEN,
        COOLDOWN_AFTER_LOSS,
    )

    print("  Cooldowns (day_trading):")
    print(f"    WIN:           {COOLDOWN_AFTER_WIN.get('day_trading')} min")
    print(f"    STRENGTH_EXIT: {COOLDOWN_AFTER_STRENGTH_EXIT.get('day_trading')} min")
    print(f"    BREAKEVEN:      {COOLDOWN_AFTER_BREAKEVEN.get('day_trading')} min")
    print(f"    LOSS:          {COOLDOWN_AFTER_LOSS.get('day_trading')} min")

    # Verify strength_exit is longer than win
    assert COOLDOWN_AFTER_STRENGTH_EXIT['day_trading'] > COOLDOWN_AFTER_WIN['day_trading']
    print("  ✓ Strength exit cooldown > win cooldown")


def test_settings():
    """Test settings loaded correctly."""
    print("\nTesting settings...")
    from config.settings import SETTINGS

    assert hasattr(SETTINGS, 'weekly_close'), "Missing weekly_close in settings"
    assert SETTINGS.weekly_close.enabled == True, "Weekly close should be enabled"
    assert SETTINGS.weekly_close.stop_new_trades_hours == 2.0
    assert SETTINGS.weekly_close.close_positions_hours == 0.5

    print(f"  ✓ WEEKLY_CLOSE_ENABLED={SETTINGS.weekly_close.enabled}")
    print(f"  ✓ STOP_NEW_TRADES_HOURS={SETTINGS.weekly_close.stop_new_trades_hours}")
    print(f"  ✓ CLOSE_POSITIONS_HOURS={SETTINGS.weekly_close.close_positions_hours}")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Recent Changes")
    print("=" * 50)

    # Test imports
    if not test_imports():
        print("\n✗ Import tests failed")
        sys.exit(1)

    # Test functionality
    try:
        test_weekly_close()
        test_cooldowns()
        test_settings()
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
