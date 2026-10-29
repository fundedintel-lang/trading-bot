"""TwelveData REST adapter — secondary/tertiary redundancy feed."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from apex_smc_bot.data.candle_manager import Candle, DataFeed, TIMEFRAME_SECONDS, utcnow
from apex_smc_bot.data.feed_base import FeedError, FeedUnavailable, with_retries

logger = logging.getLogger("apex.feed.twelve")

BASE_URL = "https://api.twelvedata.com"
INTERVALS = {"M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1day"}


def to_twelve_symbol(instrument: str) -> str:
    """``XAU_USD`` → ``XAU/USD`` (TwelveData forex convention)."""

    return instrument.replace("_", "/")


class TwelveDataFeed(DataFeed):
    """Candle + quote adapter for TwelveData."""

    name = "twelve"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
    ) -> None:
        if not api_key:
            raise FeedUnavailable("TwelveData API key missing: set TWELVE_DATA_API_KEY")
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            )
        return self._session

    async def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        session = await self._get_session()
        params = {**params, "apikey": self._api_key}

        async def request() -> dict[str, Any]:
            async with session.get(f"{BASE_URL}{path}", params=params) as response:
                text = await response.text()
                if response.status == 429 or response.status >= 500:
                    raise FeedError(f"TwelveData HTTP {response.status}")
                if response.status >= 400:
                    raise FeedUnavailable(f"TwelveData rejected request ({response.status}): {text[:200]}")
                try:
                    payload = await response.json(content_type=None)
                except Exception as exc:  # noqa: BLE001
                    raise FeedError(f"Invalid JSON from TwelveData: {exc}") from exc
            if isinstance(payload, dict) and payload.get("status") == "error":
                message = str(payload.get("message", "unknown error"))
                if "limit" in message.lower() or "credits" in message.lower():
                    raise FeedError(f"TwelveData quota: {message}")
                raise FeedUnavailable(f"TwelveData error: {message}")
            return payload

        return await with_retries(
            request,
            attempts=self.max_retries,
            backoff_seconds=self.backoff_seconds,
            label="TwelveData request",
        )

    async def fetch_candles(self, instrument: str, timeframe: str, count: int) -> list[Candle]:
        if timeframe not in INTERVALS:
            raise FeedError(f"Unsupported timeframe {timeframe!r} for TwelveData")
        payload = await self._get(
            "/time_series",
            {
                "symbol": to_twelve_symbol(instrument),
                "interval": INTERVALS[timeframe],
                "outputsize": str(max(1, min(int(count), 5000))),
                "timezone": "UTC",
                "order": "ASC",
            },
        )
        values = payload.get("values") or []
        candles: list[Candle] = []
        for raw in values:
            try:
                timestamp = datetime.strptime(raw["datetime"], "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
                candles.append(
                    Candle(
                        timestamp=timestamp,
                        open=float(raw["open"]),
                        high=float(raw["high"]),
                        low=float(raw["low"]),
                        close=float(raw["close"]),
                        volume=float(raw.get("volume", 0) or 0),
                    )
                )
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping malformed TwelveData candle: %s", exc)
                continue
        if not candles:
            raise FeedError("TwelveData returned no candles")
        candles.sort(key=lambda c: c.timestamp)
        cutoff = utcnow() - timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
        return [bar for bar in candles if bar.timestamp <= cutoff]

    async def fetch_price(self, instrument: str) -> tuple[float, float]:
        payload = await self._get("/price", {"symbol": to_twelve_symbol(instrument)})
        try:
            price = float(payload["price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FeedError(f"Malformed TwelveData quote: {payload}") from exc
        # TwelveData quotes are mid prices; assume a typical 2 pip gold spread (not an executable quote).
        half = 0.10
        return round(price - half, 2), round(price + half, 2)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
