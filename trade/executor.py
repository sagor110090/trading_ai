"""Trade executor: places orders on Capital.com or MT5 when auto-trade is enabled."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union

import requests

from config.settings import SETTINGS, SignalType, Symbol, TradeMode, TradingPlatform
from data.collector import CapitalComClient, _EPIC_MAP
from signals.engine import TradeSignal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenPosition:
    """An existing open position on Capital.com."""
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
    """Result of placing an order on Capital.com."""
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


def get_open_positions(client: CapitalComClient) -> list[OpenPosition]:
    """Fetch all currently open positions from Capital.com."""
    url = f"{client._base_url}/api/v1/positions"

    try:
        resp = client._session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error("Failed to fetch open positions: %s", e)
        return []

    positions = []
    for item in data.get("positions", []):
        pos = item.get("position", {})
        market = item.get("market", {})

        positions.append(OpenPosition(
            deal_id=pos.get("dealId", ""),
            epic=market.get("epic", ""),
            direction=pos.get("direction", ""),
            size=float(pos.get("size", 0)),
            level=float(pos.get("level", 0)),
            stop_level=_normalize_level(pos.get("stopLevel")),
            profit_level=_normalize_level(pos.get("profitLevel", pos.get("limitLevel"))),
            upl=float(pos.get("upl", 0)),
            created_date=pos.get("createdDateUTC", pos.get("createdDate", "")),
        ))

    return positions


def has_open_position(client: CapitalComClient, epic: str) -> Optional[OpenPosition]:
    """Check if there's already an open position for a given epic.

    Returns the position if found, None if no position exists.
    """
    positions = get_open_positions(client)
    for pos in positions:
        if pos.epic == epic:
            return pos
    return None


def place_trade(
    client: CapitalComClient,
    signal: TradeSignal,
    size: float = 0.01,
    guaranteed_stop: bool = False,
) -> OrderResult:
    """Place a market order on Capital.com based on a trade signal.

    Checks for existing open positions first — skips if one already exists
    for the same epic (prevents duplicate orders).

    Args:
        client: Authenticated Capital.com client.
        signal: The trade signal to execute.
        size: Position size (contracts/lots).
        guaranteed_stop: Use guaranteed stop loss.

    Returns:
        OrderResult with deal reference or error.
    """
    symbol = Symbol(signal.symbol)
    epic = _EPIC_MAP.get(symbol)
    if not epic:
        return OrderResult(
            success=False, deal_reference=None, epic="",
            direction=signal.signal.value, size=size,
            stop_level=signal.stop_loss, profit_level=signal.take_profit,
            error=f"No epic mapping for {signal.symbol}",
        )

    # ── Check for existing open position ──
    existing = has_open_position(client, epic)
    if existing:
        logger.warning(
            "SKIP: already have open %s %s size=%.2f entry=%.2f upl=%.2f — no duplicate order",
            existing.direction, existing.epic, existing.size,
            existing.level, existing.upl,
        )
        return OrderResult(
            success=False, deal_reference=None, epic=epic,
            direction=signal.signal.value, size=size,
            stop_level=signal.stop_loss, profit_level=signal.take_profit,
            error=f"Open position already exists for {epic} (dealId={existing.deal_id})",
        )

    direction = signal.signal.value

    payload = {
        "epic": epic,
        "direction": direction,
        "size": size,
        "guaranteedStop": guaranteed_stop,
    }

    if signal.stop_loss and signal.signal != SignalType.NO_TRADE:
        payload["stopLevel"] = round(signal.stop_loss, 2)

    if signal.take_profit and signal.signal != SignalType.NO_TRADE:
        payload["profitLevel"] = round(signal.take_profit, 2)

    url = f"{client._base_url}/api/v1/positions"

    logger.info(
        "Placing %s %s size=%.4f SL=%.2f TP=%.2f",
        direction, epic, size, signal.stop_loss, signal.take_profit,
    )

    try:
        resp = client._session.post(url, json=payload, timeout=30)
        data = resp.json()

        if resp.status_code == 200 and "dealReference" in data:
            deal_ref = data["dealReference"]
            logger.info("Order accepted: dealReference=%s", deal_ref)

            confirmed = _confirm_deal(client, deal_ref)

            return OrderResult(
                success=confirmed,
                deal_reference=deal_ref,
                epic=epic,
                direction=direction,
                size=size,
                stop_level=signal.stop_loss,
                profit_level=signal.take_profit,
                error=None if confirmed else "Deal not confirmed",
            )
        else:
            error_msg = data.get("errorCode", data.get("errorMessage", str(data)))
            logger.error("Order rejected: %s", error_msg)
            return OrderResult(
                success=False, deal_reference=None, epic=epic,
                direction=direction, size=size,
                stop_level=signal.stop_loss, profit_level=signal.take_profit,
                error=error_msg,
            )

    except requests.RequestException as e:
        logger.error("Order request failed: %s", e)
        return OrderResult(
            success=False, deal_reference=None, epic=epic,
            direction=direction, size=size,
            stop_level=signal.stop_loss, profit_level=signal.take_profit,
            error=str(e),
        )


def _confirm_deal(client: CapitalComClient, deal_reference: str, retries: int = 3) -> bool:
    """Confirm a deal was executed by checking its status."""
    url = f"{client._base_url}/api/v1/confirms/{deal_reference}"

    for attempt in range(retries):
        try:
            time.sleep(1)
            resp = client._session.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("dealStatus", "")
                reason = data.get("reason", "")
                if status == "ACCEPTED":
                    logger.info("Deal confirmed: %s", deal_reference)
                    return True
                else:
                    logger.warning("Deal status: %s reason: %s", status, reason)
                    return False
            else:
                logger.warning("Confirm request failed: %d", resp.status_code)
        except requests.RequestException as e:
            logger.warning("Confirm attempt %d failed: %s", attempt + 1, e)

    return False


def close_position(client: CapitalComClient, position: OpenPosition) -> CloseResult:
    """Close an open position on Capital.com.

    Args:
        client: Authenticated Capital.com client.
        position: The OpenPosition to close.

    Returns:
        CloseResult with success status and details.
    """
    url = f"{client._base_url}/api/v1/positions/{position.deal_id}"

    payload = {"size": position.size}

    try:
        resp = client._session.delete(url, json=payload, timeout=30)
        data = resp.json()

        if resp.status_code == 200 and "dealReference" in data:
            deal_ref = data["dealReference"]
            logger.info("Position closed: %s %s dealRef=%s", position.direction, position.epic, deal_ref)

            confirmed = _confirm_deal(client, deal_ref)

            return CloseResult(
                success=confirmed,
                deal_id=position.deal_id,
                epic=position.epic,
                direction=position.direction,
                size=position.size,
                close_level=position.level,
                profit=position.upl,
                error=None if confirmed else "Deal not confirmed",
            )
        else:
            error_msg = data.get("errorCode", data.get("errorMessage", str(data)))
            logger.error("Close position rejected: %s", error_msg)
            return CloseResult(
                success=False,
                deal_id=position.deal_id,
                epic=position.epic,
                direction=position.direction,
                size=position.size,
                error=error_msg,
            )

    except requests.RequestException as e:
        logger.error("Close position request failed: %s", e)
        return CloseResult(
            success=False,
            deal_id=position.deal_id,
            epic=position.epic,
            direction=position.direction,
            size=position.size,
            error=str(e),
        )


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


def update_position_sl_tp(
    client: CapitalComClient,
    position: OpenPosition,
    new_sl: Optional[float] = None,
    new_tp: Optional[float] = None,
) -> UpdateResult:
    """Update the stop loss and/or take profit of an open position.

    IMPORTANT: SL can only be moved in the protective direction:
    - For BUY positions: new SL must be LOWER than entry (tightening = moving up, loosening = moving down)
    - For SELL positions: new SL must be HIGHER than entry

    For BUY: we only move SL UP (tighter), never DOWN (looser).
    For SELL: we only move SL DOWN (tighter), never UP (looser).

    TP can be updated freely.

    Args:
        client: Authenticated Capital.com client.
        position: The open position to update.
        new_sl: New stop loss level, or None to keep current.
        new_tp: New take profit level, or None to keep current.

    Returns:
        UpdateResult with success status and details.
    """
    if not new_sl and not new_tp:
        return UpdateResult(
            success=False, deal_id=position.deal_id,
            epic=position.epic, direction=position.direction,
            error="No new SL or TP provided",
        )

    url = f"{client._base_url}/api/v1/positions/{position.deal_id}"

    payload = {}
    if new_sl is not None:
        payload["stopLevel"] = round(new_sl, 2)
    if new_tp is not None:
        payload["profitLevel"] = round(new_tp, 2)

    try:
        resp = client._session.put(url, json=payload, timeout=15)
        data = resp.json()

        if resp.status_code == 200:
            logger.info(
                "Updated position %s: SL %s → %s, TP %s → %s",
                position.deal_id,
                f"{position.level - 0:.2f}" if not new_sl else f"{new_sl:.2f}",
                f"{new_sl:.2f}" if new_sl else "unchanged",
                "unchanged" if not new_tp else f"{new_tp:.2f}",
                f"{new_tp:.2f}" if new_tp else "unchanged",
            )
            return UpdateResult(
                success=True,
                deal_id=position.deal_id,
                epic=position.epic,
                direction=position.direction,
                new_sl=new_sl,
                new_tp=new_tp,
            )
        else:
            error_msg = data.get("errorCode", data.get("errorMessage", str(data)))
            logger.error("Update position %s failed: %s", position.deal_id, error_msg)
            return UpdateResult(
                success=False, deal_id=position.deal_id,
                epic=position.epic, direction=position.direction,
                error=error_msg,
            )

    except requests.RequestException as e:
        logger.error("Update position request failed: %s", e)
        return UpdateResult(
            success=False, deal_id=position.deal_id,
            epic=position.epic, direction=position.direction,
            error=str(e),
        )


def check_timeouts(client: CapitalComClient) -> list[CloseResult]:
    """Check all open positions and close any that have exceeded their max hold time.

    The max hold time depends on the current trade mode:
    - Scalping: 30 minutes
    - Day trading: 2 hours
    - Swing: 8 hours

    Returns a list of CloseResult for any positions that were closed.
    """
    max_hold_minutes = SETTINGS.get_max_hold_minutes()
    positions = get_open_positions(client)

    if not positions:
        return []

    results = []
    now = datetime.now(timezone.utc)

    for pos in positions:
        # Parse the position's created date
        try:
            # Capital.com returns ISO format like "2024-01-15T10:30:00.000"
            created_str = pos.created_date
            if not created_str:
                logger.warning("Position %s has no created date, skipping timeout check", pos.deal_id)
                continue

            # Parse the timestamp
            created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)

            hold_minutes = (now - created).total_seconds() / 60

            if hold_minutes > max_hold_minutes:
                logger.info(
                    "TIMEOUT: %s %s held for %.0f min (max %d min) — closing",
                    pos.direction, pos.epic, hold_minutes, max_hold_minutes,
                )
                result = close_position(client, pos)
                results.append(result)
            else:
                logger.debug(
                    "Position %s %s held %.0f/%d min — OK",
                    pos.direction, pos.epic, hold_minutes, max_hold_minutes,
                )

        except Exception as e:
            logger.warning("Failed to check timeout for position %s: %s", pos.deal_id, e)
            continue

    return results


# ==================== MT5 FUNCTIONS ====================

from trading_ai.data.mt5_client import place_order as mt5_place_order, close_position as mt5_close_position
from trading_ai.data.mt5_client import update_position_levels as mt5_update_position_levels


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
    """Check if there's an open position for a symbol in MT5."""
    positions = mt5_get_open_positions(client)
    for pos in positions:
        if pos.epic == symbol:
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
    max_hold_minutes = SETTINGS.get_max_hold_minutes()
    positions = client.get_positions()

    if not positions:
        return []

    results = []
    now = datetime.now(timezone.utc)

    for pos in positions:
        # MT5 doesn't provide open time in positions_get(), need to get order info
        # For now, skip timeout check for MT5 or use a different approach
        logger.debug("MT5 timeout check not yet implemented for ticket %s", pos["ticket"])

    return results


# ==================== UNIFIED INTERFACE ====================

def get_open_positions_unified(client) -> list:
    """Get open positions for the current platform."""
    if SETTINGS.trading_platform == TradingPlatform.MT5:
        return mt5_get_open_positions(client)
    else:
        return get_open_positions(client)


def has_open_position_unified(client, epic_or_symbol: str):
    """Check if there's an open position for a symbol."""
    if SETTINGS.trading_platform == TradingPlatform.MT5:
        return mt5_has_open_position(client, epic_or_symbol)
    else:
        return has_open_position(client, epic_or_symbol)


def place_trade_unified(
    client,
    signal: TradeSignal,
    size: float = 0.01,
) -> OrderResult:
    """Place a trade for the current platform."""
    if SETTINGS.trading_platform == TradingPlatform.MT5:
        return mt5_place_trade(client, signal, size)
    else:
        return place_trade(client, signal, size)


def close_position_unified(client, position) -> CloseResult:
    """Close a position for the current platform."""
    if SETTINGS.trading_platform == TradingPlatform.MT5:
        return mt5_close_position_by_symbol(client, position)
    else:
        return close_position(client, position)


def update_position_sl_tp_unified(
    client,
    position,
    new_sl: Optional[float] = None,
    new_tp: Optional[float] = None,
) -> UpdateResult:
    """Update a position's SL/TP for the current platform."""
    if SETTINGS.trading_platform == TradingPlatform.MT5:
        return mt5_update_position_sl_tp(client, position, new_sl=new_sl, new_tp=new_tp)
    return update_position_sl_tp(client, position, new_sl=new_sl, new_tp=new_tp)


def check_timeouts_unified(client) -> list[CloseResult]:
    """Check position timeouts for the current platform."""
    if SETTINGS.trading_platform == TradingPlatform.MT5:
        return mt5_check_timeouts(client)
    else:
        return check_timeouts(client)
