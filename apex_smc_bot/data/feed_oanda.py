"""OANDA v20 REST market data adapter (primary feed).

Uses the public v20 REST API over aiohttp; no SDK required.  Streaming is
deliberately not used: the strategy only acts on **closed candles**, so polling
REST at candle close is simpler, cheaper and avoids partial-candle races.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import aiohttp

from apex_smc_bot.config import Secrets
from apex_smc_bot.data.candle_manager import Candle, DataFeed
from apex_smc_bot.data.feed_base import FeedError, FeedUnavailable, with_retries

logger = logging.getLogger("apex.feed.oanda")

PRACTICE_URL = "https://api-fxpractice.oanda.com"
LIVE_URL = "https://api-fxtrade.oanda.com"

GRANULARITY = {"M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30", "H1": "H1", "H4": "H4", "D1": "D"}


def parse_oanda_time(raw: str) -> datetime:
    """Parse OANDA's RFC3339 timestamps (9-digit fractional seconds included)."""

    cleaned = raw.strip().replace("Z", "+00:00")
    if "." in cleaned:
        head, tail = cleaned.split(".", 1)
        fraction, _, offset = tail.partition("+")
        fraction = (fraction + "000000")[:6]
        cleaned = f"{head}.{fraction}+{offset}" if offset else f"{head}.{fraction}+00:00"
    try:
        return datetime.fromisoformat(cleaned).astimezone(timezone.utc)
    except ValueError as exc:  # pragma: no cover - defensive
        raise FeedError(f"Unparseable OANDA timestamp {raw!r}") from exc


class OandaFeed(DataFeed):
    """Market data + account snapshots from OANDA v20."""

    name = "oanda"

    def __init__(
        self,
        account_id: str,
        access_token: str,
        environment: str = "practice",
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
    ) -> None:
        if not account_id or not access_token:
            raise FeedUnavailable(
                "OANDA credentials missing: set OANDA_ACCOUNT_ID and OANDA_ACCESS_TOKEN"
            )
        if environment not in {"practice", "live"}:
            raise ValueError("environment must be 'practice' or 'live'")
        self.account_id = account_id
        self._token = access_token
        self.environment = environment
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._session: aiohttp.ClientSession | None = None

    @classmethod
    def from_secrets(cls, secrets: Secrets, environment: str = "practice") -> "OandaFeed":
        return cls(secrets.oanda_account_id, secrets.oanda_access_token, environment)

    @property
    def base_url(self) -> str:
        return LIVE_URL if self.environment == "live" else PRACTICE_URL

    @property
    def configured(self) -> bool:
        return bool(self.account_id and self._token)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Accept-Datetime-Format": "RFC3339",
                    "Content-Type": "application/json",
                },
            )
        return self._session

    async def _get(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        session = await self._get_session()

        async def request() -> dict[str, Any]:
            try:
                async with session.get(f"{self.base_url}{path}", params=params) as response:
                    text = await response.text()
                    if response.status == 429 or response.status >= 500:
                        raise FeedError(f"OANDA HTTP {response.status}: {text[:200]}")
                    if response.status >= 400:
                        raise FeedUnavailable(
                            f"OANDA rejected request ({response.status}): {text[:300]}"
                        )
                    try:
                        return await response.json(content_type=None)
                    except Exception as exc:  # noqa: BLE001
                        raise FeedError(f"Invalid JSON from OANDA: {exc}") from exc
            except asyncio.TimeoutError as exc:
                raise FeedError("OANDA request timed out") from exc
            except aiohttp.ClientError as exc:
                raise FeedError(f"OANDA connection error: {exc}") from exc

        return await with_retries(
            request,
            attempts=self.max_retries,
            backoff_seconds=self.backoff_seconds,
            label=f"OANDA GET {path}",
        )

    # ------------------------------------------------------------------ candles
    async def fetch_candles(self, instrument: str, timeframe: str, count: int) -> list[Candle]:
        if timeframe not in GRANULARITY:
            raise FeedError(f"Unsupported timeframe {timeframe!r} for OANDA")
        count = max(1, min(int(count), 5000))
        payload = await self._get(
            f"/v3/instruments/{instrument}/candles",
            {"granularity": GRANULARITY[timeframe], "count": str(count), "price": "M"},
        )
        raw_candles = payload.get("candles") or []
        candles: list[Candle] = []
        for raw in raw_candles:
            if not raw.get("complete", True):
                continue  # ignore the forming candle — no look-ahead
            mid = raw.get("mid") or {}
            try:
                candle = Candle(
                    timestamp=parse_oanda_time(raw["time"]),
                    open=float(mid["o"]),
                    high=float(mid["h"]),
                    low=float(mid["l"]),
                    close=float(mid["c"]),
                    volume=float(raw.get("volume", 0) or 0),
                    spread=0.0,
                )
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping malformed OANDA candle: %s", exc)
                continue
            candles.append(candle)
        if not candles:
            raise FeedError("OANDA returned no completed candles")
        candles.sort(key=lambda c: c.timestamp)
        return candles

    async def fetch_price(self, instrument: str) -> tuple[float, float]:
        payload = await self._get(
            f"/v3/accounts/{self.account_id}/pricing",
            {"instruments": instrument},
        )
        prices = payload.get("prices") or []
        if not prices:
            raise FeedError("OANDA returned no pricing data")
        price = prices[0]
        bid = float(price["bids"][0]["price"])
        ask = float(price["asks"][0]["price"])
        if ask < bid:
            raise FeedError(f"OANDA returned inverted quote bid={bid} ask={ask}")
        return bid, ask

    async def fetch_spread(self, instrument: str) -> float:
        bid, ask = await self.fetch_price(instrument)
        return round(ask - bid, 5)

    # ------------------------------------------------------------------ account
    async def account_summary(self) -> dict[str, float]:
        """Return balance/NAV/equity snapshot for risk sizing."""

        payload = await self._get(f"/v3/accounts/{self.account_id}/summary")
        account = payload.get("account") or {}
        try:
            return {
                "balance": float(account.get("balance", 0.0)),
                "nav": float(account.get("NAV", account.get("balance", 0.0))),
                "equity": float(account.get("NAV", account.get("balance", 0.0))),
                "margin_used": float(account.get("marginUsed", 0.0)),
                "margin_available": float(account.get("marginAvailable", 0.0)),
                "open_trade_count": float(account.get("openTradeCount", 0)),
                "unrealized_pnl": float(account.get("unrealizedPL", 0.0)),
            }
        except (TypeError, ValueError) as exc:
            raise FeedError(f"Malformed OANDA account summary: {exc}") from exc

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def __repr__(self) -> str:  # never leak the token
        suffix = self.account_id[-4:] if len(self.account_id) > 4 else "****"
        return f"OandaFeed(account=***{suffix}, env={self.environment})"
