"""Trade executor: places orders on Capital.com when auto-trade is enabled."""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

from trading_ai.config.settings import SETTINGS, SignalType, Symbol
from trading_ai.data.collector import CapitalComClient, _EPIC_MAP
from trading_ai.signals.engine import TradeSignal

logger = logging.getLogger(__name__)


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


def place_trade(
    client: CapitalComClient,
    signal: TradeSignal,
    size: float = 0.01,
    guaranteed_stop: bool = False,
) -> OrderResult:
    """Place a market order on Capital.com based on a trade signal.

    Args:
        client: Authenticated Capital.com client.
        signal: The trade signal to execute.
        size: Position size (contracts/lots).
        guaranteed_stop: Use guaranteed stop loss (wider but guaranteed fill).

    Returns:
        OrderResult with deal reference or error.
    """
    # Map signal symbol to Capital.com epic
    symbol = Symbol(signal.symbol)
    epic = _EPIC_MAP.get(symbol)
    if not epic:
        return OrderResult(
            success=False, deal_reference=None, epic="",
            direction=signal.signal.value, size=size,
            stop_level=signal.stop_loss, profit_level=signal.take_profit,
            error=f"No epic mapping for {signal.symbol}",
        )

    # Map signal direction to Capital.com direction
    direction = signal.signal.value  # BUY or SELL — matches Capital.com format

    # Build order payload
    payload = {
        "epic": epic,
        "direction": direction,
        "size": size,
        "guaranteedStop": guaranteed_stop,
    }

    # Stop loss
    if signal.stop_loss and signal.signal != SignalType.NO_TRADE:
        payload["stopLevel"] = round(signal.stop_loss, 2)

    # Take profit
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

            # Confirm the deal
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
            time.sleep(1)  # Give server time to process
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