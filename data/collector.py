"""Capital.com API data collector for candle data."""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from config.settings import SETTINGS, Symbol, Timeframe

logger = logging.getLogger(__name__)

# Capital.com API resolution codes
_TIMEFRAME_MAP: dict[Timeframe, str] = {
    Timeframe.M1: "MINUTE",
    Timeframe.M5: "MINUTE_5",
    Timeframe.M15: "MINUTE_15",
    Timeframe.H1: "HOUR",
    Timeframe.H4: "HOUR_4",
    Timeframe.D1: "DAY",
}

# Capital.com epic identifiers for instruments
_EPIC_MAP: dict[Symbol, str] = {
    Symbol.XAUUSD: "GOLD",
    Symbol.BTCUSD: "BTCUSD",
}


class CapitalComClient:
    """REST client for Capital.com market data."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        self._base_url = SETTINGS.capital_com.active_url
        self._api_key = SETTINGS.capital_com.api_key
        self._identifier = SETTINGS.capital_com.identifier
        self._password = SETTINGS.capital_com.password
        self._cst_token: Optional[str] = None
        self._x_security_token: Optional[str] = None
        self._authenticated = False

    def authenticate(self) -> bool:
        """Authenticate with Capital.com API.

        POST /api/v1/session with X-CAP-API-KEY header.
        Response headers contain CST and X-SECURITY-TOKEN for subsequent requests.
        """
        url = f"{self._base_url}/api/v1/session"
        payload = {
            "identifier": self._identifier,
            "password": self._password,
            "encryptedPassword": False,
        }
        headers = {
            "X-CAP-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }
        try:
            resp = self._session.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()

            self._cst_token = resp.headers.get("CST")
            self._x_security_token = resp.headers.get("X-SECURITY-TOKEN")

            if self._cst_token and self._x_security_token:
                self._session.headers.update({
                    "CST": self._cst_token,
                    "X-SECURITY-TOKEN": self._x_security_token,
                })
                self._authenticated = True
                logger.info("Capital.com authentication successful")
                return True

            # Log response body for debugging
            body = resp.json() if resp.text else ""
            logger.error(
                "Auth response missing tokens. CST=%s X-SECURITY-TOKEN=%s body=%s",
                self._cst_token, self._x_security_token, body,
            )
            return False
        except requests.RequestException as e:
            logger.error("Capital.com authentication failed: %s", e)
            return False

    def _ensure_auth(self) -> None:
        if not self._authenticated:
            if not self.authenticate():
                raise ConnectionError("Cannot authenticate with Capital.com API")

    def search_market(self, search_term: str) -> list[dict]:
        """Search for market instruments. Useful for finding correct epic codes."""
        self._ensure_auth()
        url = f"{self._base_url}/api/v1/markets"
        params = {"searchTerm": search_term}

        try:
            resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            markets = data.get("markets", [])
            return [{
                "epic": m.get("epic", ""),
                "name": m.get("instrumentName", ""),
                "symbol": m.get("symbol", ""),
            } for m in markets]
        except requests.RequestException as e:
            logger.error("Market search failed for '%s': %s", search_term, e)
            return []

    def fetch_candles(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        limit: int = 500,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch candle data from Capital.com.

        GET /api/v1/prices/{epic}?resolution=MINUTE_5&max=500

        Args:
            symbol: Trading symbol.
            timeframe: Candle timeframe.
            limit: Max candles per request (capped at 1000 by API).
            from_date: Optional start date (ISO format: "2026-05-24T00:00:00").
            to_date: Optional end date (same format as from_date).

        Returns DataFrame with columns: timestamp, open, high, low, close, volume, spread.
        """
        self._ensure_auth()
        epic = _EPIC_MAP.get(symbol)
        if not epic:
            raise ValueError(f"No epic mapping for symbol {symbol}")

        resolution = _TIMEFRAME_MAP.get(timeframe)
        if not resolution:
            raise ValueError(f"No resolution mapping for timeframe {timeframe}")

        # Build URL manually when date params are present — requests.get()
        # URL-encodes params which breaks the ISO colons in the date format.
        url = f"{self._base_url}/api/v1/prices/{epic}"
        query = f"resolution={resolution}&max={min(limit, 1000)}"
        if from_date:
            query += f"&from={from_date}"
        if to_date:
            query += f"&to={to_date}"
        full_url = f"{url}?{query}"

        try:
            resp = self._session.get(full_url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error("Failed to fetch candles for %s %s: %s", symbol, timeframe, e)
            return pd.DataFrame()

        prices = data.get("prices", [])
        if not prices:
            logger.warning("No candle data returned for %s %s", symbol, timeframe)
            return pd.DataFrame()

        rows = []
        for p in prices:
            try:
                # Capital.com returns snapshotTime as ISO 8601 string or epoch ms
                raw_ts = p.get("snapshotTime") or p.get("snapshotTimeUTC", "")
                if isinstance(raw_ts, (int, float)):
                    ts = pd.Timestamp(raw_ts, unit="ms", tz="UTC")
                else:
                    ts = pd.Timestamp(raw_ts, tz="UTC")

                # Price fields: openPrice/highPrice/lowPrice/closePrice each have bid/ask
                open_price = _extract_price(p, "openPrice") or _extract_price(p, "open")
                high_price = _extract_price(p, "highPrice") or _extract_price(p, "high")
                low_price = _extract_price(p, "lowPrice") or _extract_price(p, "low")
                close_price = _extract_price(p, "closePrice") or _extract_price(p, "close")

                # Spread from close bid/ask
                close_ask = _extract_price(p, "closePrice", ask=True) or _extract_price(p, "close", ask=True)
                spread = close_ask - close_price if close_ask and close_price else 0.0

                rows.append({
                    "timestamp": ts,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": float(p.get("lastTradedVolume", 0) or 0),
                    "spread": round(spread, 5),
                })
            except (KeyError, ValueError, TypeError) as e:
                logger.warning("Skipping malformed candle: %s", e)
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Drop rows with zero prices
        df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            "Fetched %d candles for %s %s", len(df), symbol.value, timeframe.value
        )
        return df

    def get_latest_price(self, symbol: Symbol) -> Optional[dict]:
        """Get the latest bid/ask for a symbol.

        GET /api/v1/markets/{epic}
        """
        self._ensure_auth()
        epic = _EPIC_MAP.get(symbol)
        url = f"{self._base_url}/api/v1/markets/{epic}"

        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            snapshot = data.get("snapshot", {})
            return {
                "bid": float(snapshot.get("bid", 0)),
                "ask": float(snapshot.get("ask", 0)),
                "spread": float(snapshot.get("spread", 0)),
                "timestamp": datetime.now(tz=timezone.utc),
            }
        except requests.RequestException as e:
            logger.error("Failed to fetch latest price for %s: %s", symbol, e)
            return None


def _extract_price(
    candle: dict,
    field: str,
    ask: bool = False,
) -> float:
    """Extract bid (default) or ask price from a candle data dict.

    Capital.com price fields are nested: { "openPrice": { "bid": 3340, "ask": 3341 } }
    """
    price_obj = candle.get(field, {})
    if isinstance(price_obj, dict):
        key = "ask" if ask else "bid"
        val = price_obj.get(key, 0)
    elif isinstance(price_obj, (int, float)):
        return float(price_obj)
    else:
        val = 0

    if val is None or val == 0:
        return 0.0
    return float(val)