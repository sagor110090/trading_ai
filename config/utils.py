"""Common utility functions for the trading system."""

def normalize_symbol(symbol: str) -> str:
    """Strip 'm' suffix from broker symbols for database/ML consistency.

    Many MT5 brokers add 'm' suffix to currency pairs (e.g., BTCUSDm, AUDUSDm).
    This function normalizes them to the base symbol name for consistent
    database storage and ML model training.

    Examples:
        BTCUSDm -> BTCUSD
        XAUUSDm -> XAUUSD
        AUDUSDm -> AUDUSD
        XAGUSDm -> XAGUSD
        US30m -> US30
        BTCUSD -> BTCUSD (no change)

    Args:
        symbol: The symbol name (with or without 'm' suffix)

    Returns:
        Normalized symbol name without 'm' suffix
    """
    return symbol[:-1] if symbol.endswith("m") and len(symbol) > 1 else symbol


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
