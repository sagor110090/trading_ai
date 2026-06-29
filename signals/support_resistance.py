"""Swing high/low detection for true support and resistance levels.

This module identifies actual price levels where the market has previously
reversed (swing highs = resistance, swing lows = support). These are more
reliable than EMA-based levels for determining entry direction.

A swing high is a bar with higher high than N bars before and after.
A swing low is a bar with lower low than N bars before and after.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Default swing lookback: how many bars on each side to confirm a swing point
DEFAULT_SWING_PERIOD = 5


@dataclass
class SwingPoint:
    """A confirmed swing high or low point."""
    index: int
    price: float
    is_high: bool  # True = swing high (resistance), False = swing low (support)
    timestamp: Optional[pd.Timestamp] = None


def detect_swing_points(
    df: pd.DataFrame,
    period: int = DEFAULT_SWING_PERIOD,
) -> List[SwingPoint]:
    """Detect swing highs and lows in candle data.

    A swing high: high is highest among period bars on each side.
    A swing low: low is lowest among period bars on each side.

    Args:
        df: DataFrame with 'high', 'low', 'timestamp' columns
        period: Number of bars on each side required to confirm a swing

    Returns:
        List of SwingPoint objects sorted by index (most recent last)
    """
    if df is None or df.empty or len(df) < period * 2 + 1:
        logger.debug("Not enough data for swing detection (need %d bars)", period * 2 + 1)
        return []

    swings: List[SwingPoint] = []
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df.get("timestamp", pd.Series([None] * len(df))).values

    for i in range(period, len(df) - period):
        # Check for swing high
        is_swing_high = True
        for j in range(i - period, i + period + 1):
            if j != i and highs[i] <= highs[j]:
                is_swing_high = False
                break

        # Check for swing low
        is_swing_low = True
        for j in range(i - period, i + period + 1):
            if j != i and lows[i] >= lows[j]:
                is_swing_low = False
                break

        if is_swing_high:
            swings.append(SwingPoint(
                index=i,
                price=highs[i],
                is_high=True,
                timestamp=timestamps[i] if timestamps[i] else None
            ))

        if is_swing_low:
            swings.append(SwingPoint(
                index=i,
                price=lows[i],
                is_high=False,
                timestamp=timestamps[i] if timestamps[i] else None
            ))

    logger.debug("Detected %d swing points (%d highs, %d lows)",
                 len(swings),
                 sum(1 for s in swings if s.is_high),
                 sum(1 for s in swings if not s.is_high))
    return swings


def find_nearest_support(
    swings: List[SwingPoint],
    current_price: float,
    max_distance_pip: float = 500.0,
    pip_value: float = 0.01,
) -> Optional[SwingPoint]:
    """Find the nearest swing low (support) below current price.

    Args:
        swings: List of detected swing points
        current_price: Current market price
        max_distance_pip: Maximum distance in pips to consider as "near"
        pip_value: Value of 1 pip for the symbol (0.01 for BTC/XAU)

    Returns:
        Nearest swing low below price, or None if not found
    """
    swing_lows = [s for s in swings if not s.is_high and s.price < current_price]

    if not swing_lows:
        return None

    # Find the closest swing low by PRICE (not by index/time)
    nearest = min(swing_lows, key=lambda s: abs(current_price - s.price))

    # Check if it's within max distance
    distance_pip = abs(current_price - nearest.price) / pip_value
    if distance_pip > max_distance_pip:
        return None

    return nearest


def find_nearest_resistance(
    swings: List[SwingPoint],
    current_price: float,
    max_distance_pip: float = 500.0,
    pip_value: float = 0.01,
) -> Optional[SwingPoint]:
    """Find the nearest swing high (resistance) above current price.

    Args:
        swings: List of detected swing points
        current_price: Current market price
        max_distance_pip: Maximum distance in pips to consider as "near"
        pip_value: Value of 1 pip for the symbol (0.01 for BTC/XAU)

    Returns:
        Nearest swing high above price, or None if not found
    """
    swing_highs = [s for s in swings if s.is_high and s.price > current_price]

    if not swing_highs:
        return None

    # Find the closest swing high by PRICE (not by index/time)
    nearest = min(swing_highs, key=lambda s: abs(s.price - current_price))

    # Check if it's within max distance
    distance_pip = abs(current_price - nearest.price) / pip_value
    if distance_pip > max_distance_pip:
        return None

    return nearest


def is_at_support(
    swings: List[SwingPoint],
    current_price: float,
    atr: Optional[float] = None,
    tolerance_pip: float = 100.0,
    pip_value: float = 0.01,
) -> bool:
    """Check if current price is at or near a support level (swing low).

    Args:
        swings: List of detected swing points
        current_price: Current market price
        atr: ATR value for dynamic tolerance (uses tolerance_pip if not provided)
        tolerance_pip: Static tolerance in pips if ATR not provided
        pip_value: Value of 1 pip for the symbol

    Returns:
        True if price is within tolerance of a swing low
    """
    support = find_nearest_support(swings, current_price, max_distance_pip=tolerance_pip, pip_value=pip_value)

    if support is None:
        return False

    # Use ATR-based tolerance if available, otherwise use static
    if atr is not None and atr > 0:
        tolerance = atr * 0.5  # Half ATR tolerance
    else:
        tolerance = tolerance_pip * pip_value

    distance = abs(current_price - support.price)
    return distance <= tolerance


def is_at_resistance(
    swings: List[SwingPoint],
    current_price: float,
    atr: Optional[float] = None,
    tolerance_pip: float = 100.0,
    pip_value: float = 0.01,
) -> bool:
    """Check if current price is at or near a resistance level (swing high).

    Args:
        swings: List of detected swing points
        current_price: Current market price
        atr: ATR value for dynamic tolerance (uses tolerance_pip if not provided)
        tolerance_pip: Static tolerance in pips if ATR not provided
        pip_value: Value of 1 pip for the symbol

    Returns:
        True if price is within tolerance of a swing high
    """
    resistance = find_nearest_resistance(swings, current_price, max_distance_pip=tolerance_pip, pip_value=pip_value)

    if resistance is None:
        return False

    # Use ATR-based tolerance if available, otherwise use static
    if atr is not None and atr > 0:
        tolerance = atr * 0.5  # Half ATR tolerance
    else:
        tolerance = tolerance_pip * pip_value

    distance = abs(current_price - resistance.price)
    return distance <= tolerance


def should_buy_at_support(
    swings: List[SwingPoint],
    current_price: float,
    ema_direction: str,  # "BUY" or "SELL" from EMA analysis
    atr: Optional[float] = None,
    tolerance_pip: float = 100.0,
    pip_value: float = 0.01,
) -> bool:
    """Determine if we should BUY based on support proximity.

    Even if EMA says downtrend, if price is at support and pulling back,
    we should BUY the support bounce.

    Args:
        swings: List of detected swing points
        current_price: Current market price
        ema_direction: Direction from EMA analysis ("BUY" or "SELL")
        atr: ATR value for dynamic tolerance
        tolerance_pip: Static tolerance in pips
        pip_value: Value of 1 pip

    Returns:
        True if we should BUY at support (overrides EMA SELL)
    """
    # If EMA already says BUY, no override needed
    if ema_direction == "BUY":
        return False

    # Check if we're at a support level
    if not is_at_support(swings, current_price, atr, tolerance_pip, pip_value):
        return False

    # Price is at support in a supposed "downtrend"
    # This is likely a pullback to support - BUY the bounce
    logger.info(
        "Support override: price %.5f at support level in EMA downtrend -> BUY signal",
        current_price
    )
    return True


def should_sell_at_resistance(
    swings: List[SwingPoint],
    current_price: float,
    ema_direction: str,
    atr: Optional[float] = None,
    tolerance_pip: float = 100.0,
    pip_value: float = 0.01,
) -> bool:
    """Determine if we should SELL based on resistance proximity.

    Even if EMA says uptrend, if price is at resistance and pulling back,
    we should SELL the resistance rejection.

    Args:
        swings: List of detected swing points
        current_price: Current market price
        ema_direction: Direction from EMA analysis ("BUY" or "SELL")
        atr: ATR value for dynamic tolerance
        tolerance_pip: Static tolerance in pips
        pip_value: Value of 1 pip

    Returns:
        True if we should SELL at resistance (overrides EMA BUY)
    """
    # If EMA already says SELL, no override needed
    if ema_direction == "SELL":
        return False

    # Check if we're at a resistance level
    if not is_at_resistance(swings, current_price, atr, tolerance_pip, pip_value):
        return False

    # Price is at resistance in a supposed "uptrend"
    # This is likely a pullback to resistance - SELL the rejection
    logger.info(
        "Resistance override: price %.5f at resistance level in EMA uptrend -> SELL signal",
        current_price
    )
    return True


def get_support_resistance_info(
    swings: List[SwingPoint],
    current_price: float,
    atr: Optional[float] = None,
    pip_value: float = 0.01,
) -> dict:
    """Get comprehensive support/resistance information for logging/analysis.

    Returns dict with nearest support/resistance and whether price is at each.
    """
    # Calculate tolerance based on ATR - used for both finding and checking
    tolerance_pip = 100.0
    if atr is not None and atr > 0:
        tolerance_pip = (atr * 0.5) / pip_value

    # Use the SAME tolerance for finding and checking
    support = find_nearest_support(swings, current_price, max_distance_pip=tolerance_pip, pip_value=pip_value)
    resistance = find_nearest_resistance(swings, current_price, max_distance_pip=tolerance_pip, pip_value=pip_value)

    # Build result dict with proper price values
    result = {
        "nearest_support_price": None,
        "nearest_support_distance_pip": None,
        "at_support": False,
        "nearest_resistance_price": None,
        "nearest_resistance_distance_pip": None,
        "at_resistance": False,
        "total_swings": len(swings),
        "swing_highs": sum(1 for s in swings if s.is_high),
        "swing_lows": sum(1 for s in swings if not s.is_high),
    }

    if support is not None:
        result["nearest_support_price"] = support.price
        result["nearest_support_distance_pip"] = (current_price - support.price) / pip_value

    if resistance is not None:
        result["nearest_resistance_price"] = resistance.price
        result["nearest_resistance_distance_pip"] = (resistance.price - current_price) / pip_value

    # Mutual exclusivity: a price cannot be "at" both support and resistance
    # at the same time. When the ATR-based tolerance is wide enough that both
    # zones overlap, only flag the NEAREST level. This prevents the engine from
    # logging contradictory "At Support: True  At Resistance: True" state and
    # from flipping direction based on stale/overlapping zones.
    sup_close = support is not None
    res_close = resistance is not None
    if sup_close and res_close:
        # Both within tolerance — pick the closer one only
        sup_dist = abs(current_price - support.price)
        res_dist = abs(resistance.price - current_price)
        if sup_dist <= res_dist:
            result["at_support"] = True
        else:
            result["at_resistance"] = True
    elif sup_close:
        result["at_support"] = True
    elif res_close:
        result["at_resistance"] = True

    return result
