"""Weekly close management for forex and related markets.

Uses Bangladesh Time (Asia/Dhaka UTC+6) for Exness market hours:
- Forex: Close Saturday 02:59 AM BST, Open Monday 03:05 AM BST
- Gold (XAUUSD): Close Saturday 02:58 AM BST, Open Monday 04:05 AM BST
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta

from config.utils import normalize_symbol

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Bangladesh Time (UTC+6)
TZ_DHAKA = ZoneInfo("Asia/Dhaka")


# Asset class definitions
ASSET_CLASS_CRYPTO = "crypto"
ASSET_CLASS_FOREX = "forex"
ASSET_CLASS_METALS = "metals"
ASSET_CLASS_INDICES = "indices"
ASSET_CLASS_UNKNOWN = "unknown"


# Symbol lists for each asset class (uppercase)
CRYPTO_SYMBOLS = set()

FOREX_SYMBOLS = set()

METALS_SYMBOLS = {"XAUUSD"}

INDICES_SYMBOLS = set()


@dataclass
class WeeklyCloseStatus:
    """Status of weekly close for a symbol."""
    is_active: bool  # True if weekly close handling is active for this symbol
    hours_until_close: float | None  # Hours until close (None if not applicable)
    should_stop_new_trades: bool  # True if new trades should be blocked
    should_close_positions: bool  # True if existing positions should be closed
    message: str  # Human-readable status message
    asset_class: str  # Asset class of the symbol
    is_weekend: bool = False  # True if market is closed for weekend


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


# Market hours in Bangladesh Time (Exness)
# Close: Saturday morning, Open: Monday morning
FOREX_CLOSE_DAY = 5  # Saturday (0=Monday, 6=Sunday)
FOREX_CLOSE_TIME = "02:59"
FOREX_OPEN_DAY = 0   # Monday
FOREX_OPEN_TIME = "03:05"

GOLD_CLOSE_DAY = 5   # Saturday
GOLD_CLOSE_TIME = "02:58"
GOLD_OPEN_DAY = 0    # Monday
GOLD_OPEN_TIME = "04:05"


def _get_market_hours(asset_class: str) -> tuple:
    """Get market close/open times for asset class.

    Returns:
        (close_day, close_time, open_day, open_time) in BST
    """
    if asset_class == ASSET_CLASS_METALS:
        return (GOLD_CLOSE_DAY, GOLD_CLOSE_TIME, GOLD_OPEN_DAY, GOLD_OPEN_TIME)
    else:
        return (FOREX_CLOSE_DAY, FOREX_CLOSE_TIME, FOREX_OPEN_DAY, FOREX_OPEN_TIME)


def _parse_time_str(time_str: str) -> time:
    """Parse HH:MM string to time object."""
    hour, minute = map(int, time_str.split(":"))
    return time(hour=hour, minute=minute)


def _get_next_weekday(base_date: datetime, target_day: int) -> datetime:
    """Get next occurrence of target weekday.

    Returns today if base_date is already on target_day, otherwise returns the
    next occurrence (adding 7 days only if we're past the target day).
    """
    current_day = base_date.weekday()
    days_ahead = target_day - current_day
    if days_ahead < 0:  # Only add 7 days if we're PAST the target day, not ON it
        days_ahead += 7
    return base_date + timedelta(days=days_ahead)


def check_weekly_close_status(
    symbol: str,
    stop_new_trades_hours: float = 2.0,
    close_positions_hours: float = 0.5,
) -> WeeklyCloseStatus:
    """Check if we're approaching weekly close for a symbol.

    Args:
        symbol: The symbol to check
        stop_new_trades_hours: Hours before close to stop new trades
        close_positions_hours: Hours before close to close positions

    Returns:
        WeeklyCloseStatus with current state (uses Bangladesh Time)
    """
    asset_class = get_asset_class(symbol)

    # Crypto has no weekly close - always allow trading
    if not has_weekly_close(asset_class):
        return WeeklyCloseStatus(
            is_active=False,
            hours_until_close=None,
            should_stop_new_trades=False,
            should_close_positions=False,
            message=f"{asset_class} - 24/7 market",
            asset_class=asset_class,
            is_weekend=False,
        )

    now = datetime.now(TZ_DHAKA)
    close_day, close_time_str, open_day, open_time_str = _get_market_hours(asset_class)
    close_time = _parse_time_str(close_time_str)
    open_time = _parse_time_str(open_time_str)

    # Determine the relevant Saturday close (the one before this Monday)
    # and the coming Saturday close for calculating hours_until_close
    this_saturday_close = _get_next_weekday(now - timedelta(days=7), close_day)
    this_saturday_close = this_saturday_close.replace(
        hour=close_time.hour,
        minute=close_time.minute,
        second=0,
        microsecond=0,
    )

    next_close = _get_next_weekday(now, close_day)
    next_close = next_close.replace(
        hour=close_time.hour,
        minute=close_time.minute,
        second=0,
        microsecond=0,
    )

    # Get this week's Monday open for weekend check
    this_monday_open = _get_next_weekday(this_saturday_close, open_day)
    this_monday_open = this_monday_open.replace(
        hour=open_time.hour,
        minute=open_time.minute,
        second=0,
        microsecond=0,
    )

    # Get next Monday open for message display
    next_monday_open = _get_next_weekday(now, open_day)
    next_monday_open = next_monday_open.replace(
        hour=open_time.hour,
        minute=open_time.minute,
        second=0,
        microsecond=0,
    )
    if next_monday_open < now:
        next_monday_open = next_monday_open + timedelta(days=7)

    # Check if we're in weekend (after this Saturday close, before this Monday open)
    is_weekend = now > this_saturday_close and now < this_monday_open
    next_open = next_monday_open  # For message display

    # Use this_saturday_close as last_close for the weekend check
    last_close = this_saturday_close

    if is_weekend:
        # Market is closed for weekend
        hours_until_open = (next_open - now).total_seconds() / 3600

        if hours_until_open < 24:
            message = f"Weekend - Opens in {hours_until_open:.1f}h"
        else:
            message = f"Weekend - Opens in {hours_until_open / 24:.1f} days"

        return WeeklyCloseStatus(
            is_active=True,
            hours_until_close=None,
            should_stop_new_trades=True,
            should_close_positions=True,
            message=message,
            asset_class=asset_class,
            is_weekend=True,
        )

    # Market is open, calculate hours until close
    hours_until_close = (next_close - now).total_seconds() / 3600

    # Determine if we should stop new trades or close positions
    should_stop = hours_until_close <= stop_new_trades_hours
    should_close = hours_until_close <= close_positions_hours

    # Build message
    if should_close:
        message = f"Weekly close in {hours_until_close:.1f}h - positions closing"
    elif should_stop:
        message = f"Weekly close in {hours_until_close:.1f}h - no new trades"
    elif hours_until_close < 24:
        message = f"Weekly close in {hours_until_close:.1f}h"
    else:
        message = f"Weekly close in {hours_until_close / 24:.1f} days"

    return WeeklyCloseStatus(
        is_active=should_stop or should_close,
        hours_until_close=hours_until_close,
        should_stop_new_trades=should_stop,
        should_close_positions=should_close,
        message=message,
        asset_class=asset_class,
        is_weekend=False,
    )
