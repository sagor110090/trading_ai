"""Trade executor: places orders on MetaTrader 5 when auto-trade is enabled."""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union

from config.settings import SETTINGS, SignalType, Symbol, TradeMode
from config.utils import normalize_symbol
from signals.engine import TradeSignal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenPosition:
    """An existing open position on MetaTrader 5."""
    deal_id: str
    epic: str
    direction: str
    size: float
    level: float
    stop_level: Optional[float]
    profit_level: Optional[float]
    upl: float
    created_date: str


@dataclass(frozen=True)
class CloseResult:
    """Result of closing a position."""
    success: bool
    deal_id: str
    epic: str
    direction: str
    size: float
    close_level: Optional[float] = None
    profit: Optional[float] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class OrderResult:
    """Result of placing an order on MetaTrader 5."""
    success: bool
    deal_reference: Optional[str]
    epic: str
    direction: str
    size: float
    stop_level: Optional[float]
    profit_level: Optional[float]
    error: Optional[str] = None


def _normalize_level(level: Optional[Union[int, float]]) -> Optional[float]:
    """Convert broker level values to floats and treat zero/None as missing."""
    if level is None:
        return None

    try:
        parsed = float(level)
    except (TypeError, ValueError):
        return None

    return parsed if parsed > 0 else None


@dataclass(frozen=True)
class UpdateResult:
    """Result of updating a position's SL/TP."""
    success: bool
    deal_id: str
    epic: str
    direction: str
    old_sl: Optional[float] = None
    new_sl: Optional[float] = None
    old_tp: Optional[float] = None
    new_tp: Optional[float] = None
    error: Optional[str] = None


# ==================== MT5 FUNCTIONS ====================

from data.mt5_client import place_order as mt5_place_order, close_position as mt5_close_position
from data.mt5_client import update_position_levels as mt5_update_position_levels


def mt5_get_open_positions(client) -> list[OpenPosition]:
    """Fetch all currently open positions from MT5 in the unified position shape."""
    positions = []
    for pos in client.get_positions():
        positions.append(
            OpenPosition(
                deal_id=str(pos["ticket"]),
                epic=pos["symbol"],
                direction=pos["type"],
                size=float(pos["volume"]),
                level=float(pos["price_open"]),
                stop_level=_normalize_level(pos.get("sl")),
                profit_level=_normalize_level(pos.get("tp")),
                upl=float(pos["profit"]),
                created_date="",
            )
        )
    return positions


def mt5_has_open_position(client, symbol: str) -> Optional[OpenPosition]:
    """Check if there's an open position for a symbol in MT5.

    Handles broker symbol suffixes (e.g., AUDUSDm) by comparing both
    the base symbol and the 'm' suffix variant.
    """
    positions = mt5_get_open_positions(client)
    normalized_symbol = normalize_symbol(symbol)
    for pos in positions:
        # Direct match
        if pos.epic == symbol:
            return pos
        # Handle "m" suffix: AUDUSD matches AUDUSDm and vice versa
        normalized_epic = normalize_symbol(pos.epic)
        if normalized_epic == normalized_symbol:
            return pos
    return None


def _adjust_mt5_market_levels(client, signal: TradeSignal) -> tuple[Optional[float], Optional[float]]:
    """Shift signal-based SL/TP distances onto the live MT5 execution price."""
    try:
        symbol = Symbol(signal.symbol)
    except ValueError:
        logger.warning("Unknown MT5 symbol for level adjustment: %s", signal.symbol)
        return signal.stop_loss, signal.take_profit

    latest_price = client.get_latest_price(symbol)
    if not latest_price:
        return signal.stop_loss, signal.take_profit

    if signal.signal.value == "BUY":
        execution_price = latest_price["ask"]
        risk_distance = max(signal.entry - signal.stop_loss, 0.0)
        reward_distance = max(signal.take_profit - signal.entry, 0.0)
        stop_loss = execution_price - risk_distance if risk_distance > 0 else signal.stop_loss
        take_profit = execution_price + reward_distance if reward_distance > 0 else signal.take_profit
    else:
        execution_price = latest_price["bid"]
        risk_distance = max(signal.stop_loss - signal.entry, 0.0)
        reward_distance = max(signal.entry - signal.take_profit, 0.0)
        stop_loss = execution_price + risk_distance if risk_distance > 0 else signal.stop_loss
        take_profit = execution_price - reward_distance if reward_distance > 0 else signal.take_profit

    logger.info(
        "MT5 market levels adjusted for %s %s: signal entry=%.2f live=%.2f SL %.2f->%.2f TP %.2f->%.2f",
        signal.signal.value,
        signal.symbol,
        signal.entry,
        execution_price,
        signal.stop_loss,
        stop_loss,
        signal.take_profit,
        take_profit,
    )
    return round(stop_loss, 5), round(take_profit, 5)


def mt5_place_trade(
    client,
    signal: TradeSignal,
    size: float = 0.01,
) -> OrderResult:
    """Place a market order on MT5 based on a trade signal."""
    symbol = signal.symbol

    # Check for existing open position
    existing = mt5_has_open_position(client, symbol)
    if existing:
        logger.warning(
            "SKIP: already have open %s %s size=%.2f — no duplicate order",
            existing.direction, existing.epic, existing.size,
        )
        return OrderResult(
            success=False, deal_reference=None, epic=symbol,
            direction=signal.signal.value, size=size,
            stop_level=signal.stop_loss, profit_level=signal.take_profit,
            error=f"Open position already exists for {symbol} (ticket={existing.deal_id})",
        )

    stop_level, profit_level = _adjust_mt5_market_levels(client, signal)

    # Place the order
    ticket = mt5_place_order(
        symbol=symbol,
        order_type=signal.signal.value,
        volume=size,
        stop_loss=stop_level,
        take_profit=profit_level,
        comment=f"AI {signal.signal.value} {symbol}",
    )

    if ticket:
        logger.info("Order placed on MT5: ticket=%d", ticket)
        return OrderResult(
            success=True,
            deal_reference=str(ticket),
            epic=symbol,
            direction=signal.signal.value,
            size=size,
            stop_level=stop_level,
            profit_level=profit_level,
        )
    else:
        return OrderResult(
            success=False,
            deal_reference=None,
            epic=symbol,
            direction=signal.signal.value,
            size=size,
            stop_level=stop_level,
            profit_level=profit_level,
            error="Failed to place order on MT5",
        )


def mt5_close_position_by_symbol(client, position: OpenPosition) -> CloseResult:
    """Close all open MT5 positions for the same symbol as the given position."""
    positions = mt5_get_open_positions(client)
    closed = []

    for pos in positions:
        if pos.epic == position.epic:
            success = mt5_close_position(int(pos.deal_id))
            if success:
                closed.append(pos)

    if closed:
        return CloseResult(
            success=True,
            deal_id=",".join(p.deal_id for p in closed),
            epic=position.epic,
            direction=closed[0].direction,
            size=sum(p.size for p in closed),
        )
    else:
        return CloseResult(
            success=False,
            deal_id="",
            epic=position.epic,
            direction="",
            size=0,
            error=f"No positions found for {position.epic}",
        )


def mt5_update_position_sl_tp(
    client,
    position: OpenPosition,
    new_sl: Optional[float] = None,
    new_tp: Optional[float] = None,
) -> UpdateResult:
    """Update the stop loss and/or take profit of an MT5 position."""
    if new_sl is None and new_tp is None:
        return UpdateResult(
            success=False,
            deal_id=position.deal_id,
            epic=position.epic,
            direction=position.direction,
            error="No new SL or TP provided",
        )

    try:
        ticket = int(position.deal_id)
    except ValueError:
        return UpdateResult(
            success=False,
            deal_id=position.deal_id,
            epic=position.epic,
            direction=position.direction,
            error=f"Invalid MT5 ticket: {position.deal_id}",
        )

    success = mt5_update_position_levels(ticket, new_sl=new_sl, new_tp=new_tp)
    if not success:
        return UpdateResult(
            success=False,
            deal_id=position.deal_id,
            epic=position.epic,
            direction=position.direction,
            error="Failed to update position on MT5",
        )

    return UpdateResult(
        success=True,
        deal_id=position.deal_id,
        epic=position.epic,
        direction=position.direction,
        old_sl=position.stop_level,
        new_sl=new_sl,
        old_tp=position.profit_level,
        new_tp=new_tp,
    )


def mt5_check_timeouts(client) -> list[CloseResult]:
    """Check all open MT5 positions and close any that have exceeded max hold time."""
    import MetaTrader5 as mt5

    max_hold_minutes = SETTINGS.get_max_hold_minutes()
    max_hold_seconds = max_hold_minutes * 60

    # Get positions directly from MT5 to access open time
    # client.get_positions() doesn't include time field
    try:
        mt5_positions = mt5.positions_get() if client._mt5 else []
        if mt5_positions is None:
            mt5_positions = []
    except Exception as e:
        logger.error("Failed to get MT5 positions for timeout check: %s", e)
        return []

    if not mt5_positions:
        return []

    results = []
    now = datetime.now(timezone.utc)

    for pos in mt5_positions:
        try:
            # MT5 position.time is Unix timestamp in seconds
            opened_at = datetime.fromtimestamp(pos.time, tz=timezone.utc)
            hold_seconds = (now - opened_at).total_seconds()

            if hold_seconds > max_hold_seconds:
                logger.info(
                    "MT5 TIMEOUT: %s %s (ticket %d) held %.0f min (max %d min) — closing",
                    "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                    pos.symbol,
                    pos.ticket,
                    hold_seconds / 60,
                    max_hold_minutes,
                )

                # Create OpenPosition for the close function
                open_pos = OpenPosition(
                    deal_id=str(pos.ticket),
                    epic=pos.symbol,
                    direction="BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                    size=float(pos.volume),
                    level=float(pos.price_open),
                    stop_level=_normalize_level(pos.sl),
                    profit_level=_normalize_level(pos.tp),
                    upl=float(pos.profit),
                    created_date=opened_at.isoformat(),
                )

                close_result = mt5_close_position_by_symbol(client, open_pos)
                results.append(close_result)

            else:
                logger.debug(
                    "MT5 position %s %s (ticket %d) held %.0f/%d min — OK",
                    pos.symbol,
                    "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                    pos.ticket,
                    hold_seconds / 60,
                    max_hold_minutes,
                )

        except Exception as e:
            logger.warning("Failed to check timeout for MT5 position %s: %s", getattr(pos, 'ticket', 'unknown'), e)
            continue

    return results


# ==================== UNIFIED INTERFACE ====================

def get_open_positions_unified(client) -> list:
    """Get open positions from MetaTrader 5."""
    return mt5_get_open_positions(client)


def has_open_position_unified(client, epic_or_symbol: str):
    """Check if there's an open position for a symbol on MetaTrader 5."""
    return mt5_has_open_position(client, epic_or_symbol)


def place_trade_unified(
    client,
    signal: TradeSignal,
    size: float = 0.01,
) -> OrderResult:
    """Place a trade on MetaTrader 5."""
    return mt5_place_trade(client, signal, size)


def close_position_unified(client, position) -> CloseResult:
    """Close a position on MetaTrader 5."""
    return mt5_close_position_by_symbol(client, position)


def update_position_sl_tp_unified(
    client,
    position,
    new_sl: Optional[float] = None,
    new_tp: Optional[float] = None,
) -> UpdateResult:
    """Update a position's SL/TP on MetaTrader 5."""
    return mt5_update_position_sl_tp(client, position, new_sl=new_sl, new_tp=new_tp)


def check_timeouts_unified(client) -> list[CloseResult]:
    """Check position timeouts on MetaTrader 5.

    Returns empty list if position timeout is disabled via .env.
    """
    if not SETTINGS.position_timeout_enabled:
        logger.debug("Position timeout disabled — skipping timeout check")
        return []
    return mt5_check_timeouts(client)
