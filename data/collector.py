"""Market data collector module.

Capital.com integration has been removed. Market data and trade execution
 are handled exclusively via the MetaTrader 5 client (see data.mt5_client).
"""
from config.settings import SETTINGS, Symbol, Timeframe  # noqa: F401
