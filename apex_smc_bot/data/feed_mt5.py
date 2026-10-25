"""MetaTrader 5 adapter (Windows-only broker feed / execution fallback).

The ``MetaTrader5`` package is imported lazily so the rest of the engine runs on
Linux/macOS without it.  All MT5 calls are synchronous and therefore executed in
a worker thread to keep the asyncio event loop responsive.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from apex_smc_bot.data.candle_manager import Candle, DataFeed
from apex_smc_bot.data.feed_base import FeedError, FeedUnavailable

logger = logging.getLogger("apex.feed.mt5")

_TIMEFRAMES = {"M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4"}


class MT5Feed(DataFeed):
    """Thin MT5 wrapper; requires the Windows terminal to be running."""

    name = "mt5"

    def __init__(
        self,
        login: str = "",
        password: str = "",
        server: str = "",
        symbol: str = "XAUUSD",
        path: str | None = None,
    ) -> None:
        self.login = login
        self.password = password
        self.server = server
        self.symbol = symbol
        self.path = path
        self._mt5: Any | None = None
        self._initialised = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ lifecycle
    def _import(self) -> Any:
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception as exc:  # noqa: BLE001 - import error varies by platform
            raise FeedUnavailable(
                "MetaTrader5 package is unavailable (Windows only). "
                "Install with `pip install MetaTrader5` on a Windows host."
            ) from exc
        return mt5

    def _ensure_initialised(self) -> Any:
        mt5 = self._mt5 or self._import()
        if not self._initialised:
            kwargs: dict[str, Any] = {}
            if self.login:
                kwargs["login"] = int(self.login)
            if self.password:
                kwargs["password"] = self.password
            if self.server:
                kwargs["server"] = self.server
            if self.path:
                kwargs["path"] = self.path
            if not mt5.initialize(**kwargs):
                code, message = mt5.last_error()
                raise FeedUnavailable(f"MT5 initialize failed ({code}): {message}")
            self._initialised = True
        self._mt5 = mt5
        return mt5

    # ------------------------------------------------------------------ data
    def _fetch_sync(self, timeframe: str, count: int) -> list[Candle]:
        mt5 = self._ensure_initialised()
        tf_name = _TIMEFRAMES.get(timeframe)
        if tf_name is None:
            raise FeedError(f"Unsupported timeframe {timeframe!r} for MT5")
        rates = mt5.copy_rates_from_pos(self.symbol, getattr(mt5, tf_name), 0, int(count))
        if rates is None:
            code, message = mt5.last_error()
            raise FeedError(f"MT5 copy_rates failed ({code}): {message}")
        candles: list[Candle] = []
        for rate in rates:
            candles.append(
                Candle(
                    timestamp=datetime.fromtimestamp(int(rate["time"]), tz=timezone.utc),
                    open=float(rate["open"]),
                    high=float(rate["high"]),
                    low=float(rate["low"]),
                    close=float(rate["close"]),
                    volume=float(rate["tick_volume"]),
                    spread=float(rate["spread"]) * 0.01,  # MT5 reports spread in points
                )
            )
        # The last rate may still be forming; drop it to avoid look-ahead.
        return candles[:-1] if candles else candles

    async def fetch_candles(self, instrument: str, timeframe: str, count: int) -> list[Candle]:
        async with self._lock:
            return await asyncio.to_thread(self._fetch_sync, timeframe, count)

    def _tick_sync(self) -> tuple[float, float]:
        mt5 = self._ensure_initialised()
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            code, message = mt5.last_error()
            raise FeedError(f"MT5 tick unavailable ({code}): {message}")
        return float(tick.bid), float(tick.ask)

    async def fetch_price(self, instrument: str) -> tuple[float, float]:
        async with self._lock:
            return await asyncio.to_thread(self._tick_sync)

    async def close(self) -> None:
        if self._mt5 is not None and self._initialised:
            await asyncio.to_thread(self._mt5.shutdown)
        self._initialised = False
        self._mt5 = None
