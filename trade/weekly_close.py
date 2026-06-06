"""Weekly close management for forex and related markets.

Forex markets close at Friday 22:00 UTC (5 PM EST). This module handles:
- Detecting when we approach weekly close
- Determining if new trades should be blocked
- Determining if existing positions should be closed
- Handling different asset classes (forex vs crypto which is 24/7)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, time

from config.utils import normalize_symbol

logger = logging.getLogger(__name__)


# Asset class definitions
ASSET_CLASS_CRYPTO = "crypto"
ASSET_CLASS_FOREX = "forex"
ASSET_CLASS_METALS = "metals"
ASSET_CLASS_INDICES = "indices"
ASSET_CLASS_UNKNOWN = "unknown"


# Symbol lists for each asset class (uppercase)
CRYPTO_SYMBOLS = {
    "BTCUSD", "ETHUSD", "LTCUSD", "BCHUSD", "XRPUSD", "SOLUSD",
    "BTCEUR", "ETHGBP"
}

FOREX_SYMBOLS = {
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "EURGBP", "EURJPY", "GBPJPY",
    "AUDUSD", "NZDUSD", "USDCAD", "EURAUD", "GBPCHF"
}

METALS_SYMBOLS = {"XAUUSD", "XAGUSD"}

INDICES_SYMBOLS = {"US30", "US500", "NAS100"}


@dataclass
class WeeklyCloseStatus:
    """Status of weekly close for a symbol."""
    is_active: bool  # True if weekly close handling is active for this symbol
    hours_until_close: float | None  # Hours until close (None if not applicable)
    should_stop_new_trades: bool  # True if new trades should be blocked
    should_close_positions: bool  # True if existing positions should be closed
    message: str  # Human-readable status message
    asset_class: str  # Asset class of the symbol


def get_asset_class(symbol: str) -> str:
    """Categorize symbol into asset class.

    Args:
        symbol: The symbol name (will be normalized)

    Returns:
        Asset class: crypto, forex, metals, indices, or unknown
    """
    # Normalize symbol to handle MT5 'm' suffix
    normalized = normalize_symbol(symbol).upper()

    if normalized in CRYPTO_SYMBOLS:
        return ASSET_CLASS_CRYPTO
    elif normalized in FOREX_SYMBOLS:
        return ASSET_CLASS_FOREX
    elif normalized in METALS_SYMBOLS:
        return ASSET_CLASS_METALS
    elif normalized in INDICES_SYMBOLS:
        return ASSET_CLASS_INDICES
    else:
        return ASSET_CLASS_UNKNOWN


def has_weekly_close(asset_class: str) -> bool:
    """Check if an asset class has a weekly close.

    Crypto markets are 24/7 and have no weekly close.
    Forex, metals, and indices follow the forex weekly close schedule.

    Args:
        asset_class: The asset class to check

    Returns:
        True if the asset class has a weekly close
    """
    return asset_class in (ASSET_CLASS_FOREX, ASSET_CLASS_METALS, ASSET_CLASS_INDICES)


def check_weekly_close_status(
    symbol: str,
    close_day: int = 4,  # Friday (0=Monday, 6=Sunday)
    close_time_utc: str = "22:00",  # 5 PM EST
    stop_new_trades_hours: float = 2.0,
    close_positions_hours: float = 0.5,
) -> WeeklyCloseStatus:
    """Check if we're approaching weekly close for a symbol.

    Args:
        symbol: The symbol to check
        close_day: Day of week for close (5 = Friday)
        close_time_utc: Close time in UTC (HH:MM format)
        stop_new_trades_hours: Hours before close to stop new trades
        close_positions_hours: Hours before close to close positions

    Returns:
        WeeklyCloseStatus with current state
    """
    asset_class = get_asset_class(symbol)

    # Crypto has no weekly close - always allow trading
    if not has_weekly_close(asset_class):
        return WeeklyCloseStatus(
            is_active=False,
            hours_until_close=None,
            should_stop_new_trades=False,
            should_close_positions=False,
            message=f"{asset_class} - 24/7 market, no weekly close",
            asset_class=asset_class,
        )

    now = datetime.now(timezone.utc)
    current_day = now.weekday()  # 0=Monday, 6=Sunday
    current_time = now.time()

    # Parse close time
    try:
        close_hour, close_minute = map(int, close_time_utc.split(":"))
        close_time = time(close_hour, close_minute)
    except (ValueError, AttributeError):
        logger.error("Invalid close_time_utc format: %s", close_time_utc)
        close_hour, close_minute = 22, 0
        close_time = time(close_hour, close_minute)

    # Calculate seconds until close
    # Convert times to seconds since midnight for comparison
    current_seconds = current_time.hour * 3600 + current_time.minute * 60 + current_time.second
    close_seconds = close_time.hour * 3600 + close_time.minute * 60

    # Calculate days until close (handling week wrap)
    if current_day < close_day:
        days_until = close_day - current_day
    elif current_day > close_day:
        days_until = 7 - current_day + close_day
    else:  # Same day
        if current_seconds < close_seconds:
            days_until = 0
        else:
            days_until = 7  # Next week

    # Calculate total seconds and hours until close
    if days_until == 0:
        seconds_until = close_seconds - current_seconds
    else:
        seconds_until = days_until * 86400 + (close_seconds - current_seconds)

    hours_until = seconds_until / 3600

    # Determine if we should stop new trades or close positions
    should_stop = hours_until <= stop_new_trades_hours
    should_close = hours_until <= close_positions_hours

    # Build message
    if should_close:
        message = f"Weekly close in {hours_until:.1f}h - positions closing"
    elif should_stop:
        message = f"Weekly close in {hours_until:.1f}h - no new trades"
    elif hours_until < 24:
        message = f"Weekly close in {hours_until:.1f}h"
    else:
        message = f"Weekly close in {hours_until / 24:.1f} days"

    return WeeklyCloseStatus(
        is_active=should_stop or should_close,
        hours_until_close=hours_until,
        should_stop_new_trades=should_stop,
        should_close_positions=should_close,
        message=message,
        asset_class=asset_class,
    )


def get_symbols_with_weekly_close(symbols: list[str]) -> dict[str, str]:
    """Get mapping of symbols to their asset classes for those with weekly close.

    Args:
        symbols: List of symbols to check

    Returns:
        Dict mapping symbol to asset class (only for symbols with weekly close)
    """
    result = {}
    for symbol in symbols:
        asset_class = get_asset_class(symbol)
        if has_weekly_close(asset_class):
            result[symbol] = asset_class
    return result
