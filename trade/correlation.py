"""Correlation limiter: prevents multiple positions on correlated symbols.

Correlation groups:
- CRYPTO: BTCUSD, ETHUSD, SOLUSD, LTCUSD, BCHUSD, XRPUSD
- METALS: XAUUSD, XAGUSD
- FOREX_MAJOR: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, NZDUSD, USDCAD
- FOREX_MINOR: EURGBP, EURJPY, GBPJPY, EURAUD, GBPCHF
- INDICES: US30, US500, NAS100
"""

import logging
from dataclasses import dataclass
from typing import Optional

from config.settings import SETTINGS
from trade.executor import get_open_positions_unified

logger = logging.getLogger(__name__)


# Correlation groups - symbols that tend to move together
CORRELATION_GROUPS = {
    "METALS": [
        "XAUUSD",  # Gold
    ],
}


@dataclass
class CorrelationCheck:
    """Result of correlation check."""
    allowed: bool
    group: str
    current_count: int
    max_allowed: int
    reason: str


def get_symbol_group(symbol: str) -> Optional[str]:
    """Get the correlation group for a symbol.

    Returns:
        Group name or None if symbol not in any group.
    """
    for group, symbols in CORRELATION_GROUPS.items():
        if symbol in symbols:
            return group
    return None


def count_positions_in_group(
    open_positions: list,
    group: str,
) -> int:
    """Count how many open positions exist in a correlation group.

    Args:
        open_positions: List of OpenPosition objects
        group: Correlation group name

    Returns:
        Number of positions in this group
    """
    count = 0
    group_symbols = set(CORRELATION_GROUPS.get(group, []))

    for pos in open_positions:
        # Normalize symbol name (handle 'm' suffix for MT5)
        symbol = pos.epic.rstrip("m")
        if symbol in group_symbols:
            count += 1

    return count


def check_correlation_limit(
    symbol: str,
    client,
) -> CorrelationCheck:
    """Check if a new trade would exceed correlation limits.

    Args:
        symbol: Symbol to check
        client: Trading client (MetaTrader 5)

    Returns:
        CorrelationCheck with decision
    """
    group = get_symbol_group(symbol)

    # No group defined = no restriction
    if not group:
        return CorrelationCheck(
            allowed=True,
            group="NONE",
            current_count=0,
            max_allowed=999,
            reason=f"Symbol {symbol} not in a correlation group",
        )

    # Get max allowed for this group from settings or use default
    max_allowed = getattr(SETTINGS, "max_positions_per_group", {}).get(group, 1)

    # Get current open positions
    open_positions = get_open_positions_unified(client)

    # Count positions in this group
    current_count = count_positions_in_group(open_positions, group)

    # Check if limit would be exceeded
    if current_count >= max_allowed:
        return CorrelationCheck(
            allowed=False,
            group=group,
            current_count=current_count,
            max_allowed=max_allowed,
            reason=(
                f"Already have {current_count}/{max_allowed} positions in {group} group. "
                f"Would exceed limit with {symbol}."
            ),
        )

    return CorrelationCheck(
        allowed=True,
        group=group,
        current_count=current_count,
        max_allowed=max_allowed,
        reason=f"Have {current_count}/{max_allowed} positions in {group} group - OK",
    )
