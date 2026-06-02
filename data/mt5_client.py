"""MetaTrader 5 (MT5) client for candle data and trading."""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from trading_ai.config.settings import SETTINGS, Symbol, Timeframe

logger = logging.getLogger(__name__)

# MT5 timeframe constants must come from the MetaTrader5 module itself.
# Minute frames happen to match their minute count, but hourly/daily frames do not.
_TIMEFRAME_MAP: dict[Timeframe, str] = {
    Timeframe.M1: "TIMEFRAME_M1",
    Timeframe.M5: "TIMEFRAME_M5",
    Timeframe.M15: "TIMEFRAME_M15",
    Timeframe.H1: "TIMEFRAME_H1",
    Timeframe.H4: "TIMEFRAME_H4",
    Timeframe.D1: "TIMEFRAME_D1",
}

# MT5 symbol names (may differ from Capital.com)
_SYMBOL_MAP: dict[Symbol, str] = {
    Symbol.XAUUSD: "XAUUSD",
    Symbol.BTCUSD: "BTCUSD",
    Symbol.ETHUSD: "ETHUSD",
    Symbol.EURUSD: "EURUSD",
    Symbol.GBPUSD: "GBPUSD",
    # Additional forex pairs
    Symbol.USDJPY: "USDJPY",
    Symbol.USDCHF: "USDCHF",
    Symbol.EURGBP: "EURGBP",
    Symbol.EURJPY: "EURJPY",
    Symbol.GBPJPY: "GBPJPY",
    Symbol.AUDUSD: "AUDUSD",
    Symbol.NZDUSD: "NZDUSD",
    Symbol.USDCAD: "USDCAD",
    # Metals
    Symbol.XAGUSD: "XAGUSD",
}


class MT5Client:
    """MetaTrader 5 client for market data and trading."""

    def __init__(self) -> None:
        self._connected = False
        self._mt5 = None

    def connect(self) -> bool:
        """Connect to MT5 terminal using credentials from SETTINGS.

        Note: MT5 terminal must be installed on your computer.
        """
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5

            # Get credentials from settings
            mt5_cfg = SETTINGS.mt5

            logger.info("Attempting MT5 connection...")
            logger.info("  Login: %s", mt5_cfg.login if mt5_cfg.login else "N/A")
            logger.info("  Server: %s", mt5_cfg.server if mt5_cfg.server else "N/A")
            logger.info("  Timeout: %d ms", mt5_cfg.timeout)

            # Try to initialize with login if credentials provided
            if mt5_cfg.login and mt5_cfg.server:
                password = mt5_cfg.password if mt5_cfg.password else ""

                logger.info("Initializing MT5 with credentials...")
                if not mt5.initialize(
                    login=mt5_cfg.login,
                    password=password,
                    server=mt5_cfg.server,
                    timeout=mt5_cfg.timeout // 1000,  # Convert ms to seconds
                ):
                    error_code = mt5.last_error()
                    logger.error("MT5 initialize() failed, error code = %s", error_code)
                    logger.error("Possible issues:")
                    logger.error("  1. MT5 terminal not installed")
                    logger.error("  2. Incorrect login, password, or server name")
                    logger.error("  3. Server name format (check with your broker)")
                    logger.error("  4. Account is disabled or blocked")
                    return False

                # Verify login was successful
                account_info = mt5.account_info()
                if account_info is None:
                    logger.error("MT5 initialized but no account info - login may have failed")
                    mt5.shutdown()
                    return False

                logger.info("MT5 initialized successfully: %s@%s", mt5_cfg.login, mt5_cfg.server)
                logger.info("  Account: %s", account_info.login)
                logger.info("  Server: %s", account_info.server)
                logger.info("  Balance: %s", account_info.balance)
            else:
                # Try without login (terminal must be running and logged in)
                logger.info("No MT5 credentials in .env, trying terminal connection...")
                if not mt5.initialize():
                    logger.error("MT5 initialize() failed, error code = %s", mt5.last_error())
                    logger.error("MT5 terminal must be running and logged in")
                    return False
                logger.info("MT5 initialized via terminal")

            self._connected = True
            logger.info("Connected to MetaTrader 5")
            return True
        except ImportError:
            logger.error("MetaTrader5 library not installed. Install with: pip install MetaTrader5")
            return False
        except Exception as e:
            logger.error("Failed to connect to MT5: %s", e)
            return False

    def _ensure_connected(self) -> None:
        if not self._connected:
            if not self.connect():
                raise ConnectionError("Cannot connect to MetaTrader 5")

    def shutdown(self) -> None:
        """Shutdown MT5 connection."""
        if self._connected and self._mt5:
            self._mt5.shutdown()
            self._connected = False
            logger.info("MT5 connection closed")

    def fetch_candles(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        limit: int = 500,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch candle data from MT5.

        Args:
            symbol: Trading symbol.
            timeframe: Candle timeframe.
            limit: Max candles to fetch (MT5 max is ~100000 depending on timeframe).
            from_date: Optional start datetime.
            to_date: Optional end datetime.

        Returns DataFrame with columns: timestamp, open, high, low, close, volume, spread.
        """
        self._ensure_connected()

        mt5_symbol = _SYMBOL_MAP.get(symbol, symbol.value)
        mt5_tf_name = _TIMEFRAME_MAP.get(timeframe)
        if not mt5_tf_name:
            raise ValueError(f"No timeframe mapping for {timeframe}")
        mt5_tf = getattr(self._mt5, mt5_tf_name, None)
        if mt5_tf is None:
            raise ValueError(f"MT5 module missing timeframe constant {mt5_tf_name}")

        # Check if symbol exists in MT5
        symbol_info = self._mt5.symbol_info(mt5_symbol)
        if not symbol_info:
            logger.error("Symbol '%s' not found in MT5. Available symbols: %s",
                        mt5_symbol, self._get_available_symbols())
            return pd.DataFrame()
        if not symbol_info.visible and not self._mt5.symbol_select(mt5_symbol, True):
            logger.error("Failed to select MT5 symbol '%s' in Market Watch", mt5_symbol)
            return pd.DataFrame()

        # Fetch rates
        if from_date and to_date:
            # Fetch range
            rates = self._mt5.copy_rates_range(mt5_symbol, mt5_tf, from_date, to_date)
        elif from_date:
            # Fetch from date to now
            rates = self._mt5.copy_rates_from(mt5_symbol, mt5_tf, from_date, limit)
        else:
            # Fetch last N candles
            rates = self._mt5.copy_rates_from_pos(mt5_symbol, mt5_tf, 0, limit)

        if rates is None or len(rates) == 0:
            logger.warning(
                "No candle data returned for %s %s (mt5 timeframe=%s, last_error=%s)",
                symbol.value,
                timeframe.value,
                mt5_tf,
                self._mt5.last_error(),
            )
            return pd.DataFrame()

        df = pd.DataFrame(rates)

        # MT5 returns time as Unix timestamp in seconds
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)

        # Rename MT5 columns to match our schema
        df = df.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tick_volume": "volume",
        })

        # MT5 doesn't provide spread in historical data, calculate from bid/ask if needed
        df["spread"] = 0.0

        # Select and order columns
        df = df[["timestamp", "open", "high", "low", "close", "volume", "spread"]]

        # Drop rows with zero prices
        df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]

        logger.info("Fetched %d candles for %s %s", len(df), symbol.value, timeframe.value)
        return df

    def get_latest_price(self, symbol) -> Optional[dict]:
        """Get the latest bid/ask for a symbol.

        Args:
            symbol: Symbol enum or string (e.g., "BTCUSD")

        Returns dict with bid, ask, spread, timestamp.
        """
        self._ensure_connected()

        # Handle both Symbol enum and string
        if hasattr(symbol, "value"):
            symbol_str = symbol.value
        else:
            symbol_str = str(symbol)

        mt5_symbol = _SYMBOL_MAP.get(Symbol(symbol_str) if symbol_str in [s.value for s in Symbol] else symbol_str, symbol_str)
        tick = self._mt5.symbol_info_tick(mt5_symbol)

        if tick is None:
            logger.error("Failed to get tick for %s", mt5_symbol)
            return None

        return {
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "spread": float(tick.ask - tick.bid),
            "timestamp": datetime.now(tz=timezone.utc),
        }

    def get_account_info(self) -> Optional[dict]:
        """Get account information."""
        self._ensure_connected()
        info = self._mt5.account_info()
        if info is None:
            return None
        return {
            "balance": float(info.balance),
            "equity": float(info.equity),
            "margin": float(info.margin),
            "free_margin": float(info.margin_free),
            "profit": float(info.profit),
        }

    def get_positions(self) -> list:
        """Get all open positions."""
        self._ensure_connected()
        positions = self._mt5.positions_get()
        if positions is None or len(positions) == 0:
            return []
        return [{
            "ticket": pos.ticket,
            "symbol": pos.symbol,
            "type": "BUY" if pos.type == 0 else "SELL",
            "volume": float(pos.volume),
            "price_open": float(pos.price_open),
            "sl": float(pos.sl),
            "tp": float(pos.tp),
            "profit": float(pos.profit),
            "comment": pos.comment,
        } for pos in positions]

    def _get_available_symbols(self) -> list[str]:
        """Get list of available symbols in MT5."""
        if not self._connected:
            return []
        symbols = self._mt5.symbols_get()
        return [s.name for s in symbols] if symbols else []


def place_order(
    symbol: str,
    order_type: str,  # "BUY" or "SELL"
    volume: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    comment: str = "",
) -> Optional[int]:
    """Place an order on MT5.

    Returns order ticket if successful, None otherwise.
    """
    import MetaTrader5 as mt5

    # Get credentials from settings
    mt5_cfg = SETTINGS.mt5

    # Initialize with credentials if available
    if mt5_cfg.login and mt5_cfg.password and mt5_cfg.server:
        if not mt5.initialize(
            login=mt5_cfg.login,
            password=mt5_cfg.password,
            server=mt5_cfg.server,
        ):
            logger.error("MT5 initialize failed")
            return None
    elif not mt5.initialize():
        logger.error("MT5 initialize failed - terminal must be running")
        return None

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error("Symbol %s not found", symbol)
        mt5.shutdown()
        return None

    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error("Failed to get tick for %s", symbol)
        mt5.shutdown()
        return None

    # Determine order type and price
    if order_type.upper() == "BUY":
        order_type_mt5 = mt5.ORDER_TYPE_BUY
        price = tick.ask
    elif order_type.upper() == "SELL":
        order_type_mt5 = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        logger.error("Invalid order type: %s", order_type)
        mt5.shutdown()
        return None

    # Normalize volume to symbol's lot step
    lot_step = symbol_info.volume_step
    volume = round(volume / lot_step) * lot_step

    # Validate volume
    if volume < symbol_info.volume_min:
        volume = symbol_info.volume_min
    if volume > symbol_info.volume_max:
        volume = symbol_info.volume_max

    # Respect the broker's minimum stop distance for market orders.
    point = float(symbol_info.point or 0.0)
    stops_level = max(int(getattr(symbol_info, "trade_stops_level", 0) or 0), 0)
    min_stop_distance = stops_level * point

    if order_type.upper() == "BUY":
        if stop_loss is not None and stop_loss >= price:
            stop_loss = price - max(min_stop_distance, point)
        if take_profit is not None and take_profit <= price:
            take_profit = price + max(min_stop_distance, point)
        if stop_loss is not None and min_stop_distance > 0 and (price - stop_loss) < min_stop_distance:
            stop_loss = price - min_stop_distance
        if take_profit is not None and min_stop_distance > 0 and (take_profit - price) < min_stop_distance:
            take_profit = price + min_stop_distance
    else:
        if stop_loss is not None and stop_loss <= price:
            stop_loss = price + max(min_stop_distance, point)
        if take_profit is not None and take_profit >= price:
            take_profit = price - max(min_stop_distance, point)
        if stop_loss is not None and min_stop_distance > 0 and (stop_loss - price) < min_stop_distance:
            stop_loss = price + min_stop_distance
        if take_profit is not None and min_stop_distance > 0 and (price - take_profit) < min_stop_distance:
            take_profit = price - min_stop_distance

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type_mt5,
        "price": price,
        "deviation": 20,  # Max slippage points
        "magic": 234000,  # Magic number for identification
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,  # Good Till Cancelled
        "type_filling": mt5.ORDER_FILLING_FOK,  # Fill or Kill
    }

    if stop_loss:
        request["sl"] = round(stop_loss, 5)
    if take_profit:
        request["tp"] = round(take_profit, 5)

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(
            "Order failed: %s (price=%.5f sl=%s tp=%s stops_level=%s point=%s retcode=%s)",
            result.comment,
            price,
            request.get("sl"),
            request.get("tp"),
            stops_level,
            point,
            result.retcode,
        )
        return None

    logger.info("Order placed: %s %s @ %s, ticket=%d",
                order_type, symbol, price, result.order)
    return result.order


def close_position(ticket: int) -> bool:
    """Close a position by ticket number."""
    import MetaTrader5 as mt5

    # Get credentials from settings
    mt5_cfg = SETTINGS.mt5

    # Initialize with credentials if available
    if mt5_cfg.login and mt5_cfg.password and mt5_cfg.server:
        if not mt5.initialize(
            login=mt5_cfg.login,
            password=mt5_cfg.password,
            server=mt5_cfg.server,
        ):
            logger.error("MT5 initialize failed")
            return False
    elif not mt5.initialize():
        logger.error("MT5 initialize failed - terminal must be running")
        return False

    position = mt5.positions_get(ticket=ticket)
    if position is None or len(position) == 0:
        logger.error("Position %d not found", ticket)
        mt5.shutdown()
        return False

    pos = position[0]

    # Determine opposite order type
    if pos.type == mt5.POSITION_TYPE_BUY:
        close_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(pos.symbol).bid
    else:
        close_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(pos.symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": close_type,
        "position": ticket,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": "Close by trading AI",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error("Close position failed: %s", result.comment)
        mt5.shutdown()
        return False

    logger.info("Position %d closed", ticket)
    return True


def update_position_levels(
    ticket: int,
    new_sl: Optional[float] = None,
    new_tp: Optional[float] = None,
) -> bool:
    """Update an open MT5 position's stop loss and/or take profit."""
    import MetaTrader5 as mt5

    if new_sl is None and new_tp is None:
        logger.error("No MT5 SL/TP update requested for ticket %d", ticket)
        return False

    # Get credentials from settings
    mt5_cfg = SETTINGS.mt5

    # Initialize with credentials if available
    if mt5_cfg.login and mt5_cfg.password and mt5_cfg.server:
        if not mt5.initialize(
            login=mt5_cfg.login,
            password=mt5_cfg.password,
            server=mt5_cfg.server,
        ):
            logger.error("MT5 initialize failed")
            return False
    elif not mt5.initialize():
        logger.error("MT5 initialize failed - terminal must be running")
        return False

    position = mt5.positions_get(ticket=ticket)
    if position is None or len(position) == 0:
        logger.error("Position %d not found", ticket)
        mt5.shutdown()
        return False

    pos = position[0]
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": pos.symbol,
        "position": ticket,
        "sl": round(new_sl if new_sl is not None else float(pos.sl), 5),
        "tp": round(new_tp if new_tp is not None else float(pos.tp), 5),
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(
            "Update position failed: %s (ticket=%d sl=%s tp=%s retcode=%s)",
            result.comment,
            ticket,
            request["sl"],
            request["tp"],
            result.retcode,
        )
        mt5.shutdown()
        return False

    logger.info(
        "Position %d levels updated: SL=%s TP=%s",
        ticket,
        request["sl"],
        request["tp"],
    )
    return True
