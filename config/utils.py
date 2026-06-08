"""Common utility functions for the trading system."""


def normalize_symbol(symbol: str) -> str:
    """Normalize broker symbol variants into a stable uppercase key.

    Handles common broker suffixes so cooldowns, DB rows, and trade decisions
    all use the same symbol key.

    Examples:
        BTCUSDm -> BTCUSD
        XAUUSDm -> XAUUSD
        US30.cash -> US30
        btcusd -> BTCUSD
    """
    if not symbol:
        return symbol

    normalized = str(symbol).strip().upper()

    # Strip common CFD suffixes before the MT5 broker suffix.
    for suffix in (".CASH",):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]

    if normalized.endswith("M") and len(normalized) > 1:
        normalized = normalized[:-1]

    return normalized


def broker_symbol(symbol: str) -> str:
    """Add 'm' suffix for MT5 broker communication.

    This is the inverse of normalize_symbol() - use when you need to
    communicate with the MT5 broker which expects the 'm' suffix.

    Args:
        symbol: The normalized symbol name (without 'm')

    Returns:
        Symbol name with 'm' suffix for broker communication
    """
    return f"{symbol}m" if not symbol.endswith("m") else symbol
