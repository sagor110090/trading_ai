"""Trade executor: places orders on Capital.com when auto-trade is enabled."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

from trading_ai.config.settings import SETTINGS, SignalType, Symbol, TradeMode
from trading_ai.data.collector import CapitalComClient, _EPIC_MAP
from trading_ai.signals.engine import TradeSignal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenPosition:
    """An existing open position on Capital.com."""
    deal_id: str
    epic: str
    direction: str
    size: float
    level: float
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


def should_update_sl(
    direction: str,
    current_sl: float,
    new_sl: float,
    entry_price: float,
) -> tuple[bool, str]:
    """Check if a new SL is safe to apply.

    Rules:
    - For BUY: SL can only move UP (tighter). Moving SL DOWN (looser) is dangerous.
    - For SELL: SL can only move DOWN (tighter). Moving SL UP (looser) is dangerous.
    - New SL must not be too close to entry (min buffer).
    - New SL must not be too aggressive (moved way too close to current price).

    Returns (is_safe, reason).
    """
    min_sl_buffer_pct = 0.002  # SL must be at least 0.2% away from entry
    max_sl_tighten_pct = 0.5   # SL can tighten at most 50% of original distance

    if direction.upper() == "BUY":
        # BUY: SL is below entry
        original_distance = entry_price - current_sl
        new_distance = entry_price - new_sl

        if new_distance <= 0:
            return False, f"New SL {new_sl:.2f} is above entry {entry_price:.2f} for BUY"

        # Don't loosen SL (move it further down) — too risky
        if new_sl < current_sl:
            return False, (
                f"New SL {new_sl:.2f} is BELOW current SL {current_sl:.2f} — "
                f"loosening stop is dangerous"
            )

        # Don't tighten SL too aggressively (too close to entry)
        min_buffer = entry_price * min_sl_buffer_pct
        if new_distance < min_buffer:
            return False, (
                f"New SL too aggressive: only {new_distance:.2f} from entry "
                f"(min {min_buffer:.2f}) — risk of noise stop-out"
            )

        # Don't tighten more than 50% of original distance
        max_tighten = original_distance * (1 - max_sl_tighten_pct)
        if new_distance < original_distance * max_sl_tighten_pct:
            return False, (
                f"New SL too tight: tightened {((1 - new_distance/original_distance)*100):.0f}% "
                f"from original distance (max 50%)"
            )

        return True, "OK"

    elif direction.upper() == "SELL":
        # SELL: SL is above entry
        original_distance = current_sl - entry_price
        new_distance = new_sl - entry_price

        if new_distance <= 0:
            return False, f"New SL {new_sl:.2f} is below entry {entry_price:.2f} for SELL"

        # Don't loosen SL (move it further up)
        if new_sl > current_sl:
            return False, (
                f"New SL {new_sl:.2f} is ABOVE current SL {current_sl:.2f} — "
                f"loosening stop is dangerous"
            )

        # Don't tighten SL too aggressively
        min_buffer = entry_price * min_sl_buffer_pct
        if new_distance < min_buffer:
            return False, (
                f"New SL too aggressive: only {new_distance:.2f} from entry "
                f"(min {min_buffer:.2f}) — risk of noise stop-out"
            )

        # Don't tighten more than 50% of original distance
        if new_distance < original_distance * max_sl_tighten_pct:
            return False, (
                f"New SL too tight: tightened {((1 - new_distance/original_distance)*100):.0f}% "
                f"from original distance (max 50%)"
            )

        return True, "OK"

    return False, f"Unknown direction: {direction}"


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