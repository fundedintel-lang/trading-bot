"""Deterministic replay feed used by the backtester and tests.

``ReplayFeed`` exposes an already-generated multi-timeframe dataset as a
:class:`~apex_smc_bot.data.candle_manager.DataFeed`, serving only the bars up to
an explicit cursor.  That makes the live engine loop replayable bar by bar with
no look-ahead bias.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, Sequence

from apex_smc_bot.data.candle_manager import Candle, DataFeed, TIMEFRAME_SECONDS, to_utc


class ReplayFeed(DataFeed):
    """Cursor-driven feed over an in-memory dataset."""

    name = "replay"

    def __init__(
        self,
        data: Mapping[str, Sequence[Candle]],
        spread_price: float = 0.20,
        timeframes: Iterable[str] = ("M5", "M15", "H1", "H4"),
    ) -> None:
        if "M5" not in data or not data["M5"]:
            raise ValueError("ReplayFeed requires a non-empty M5 series")
        self._data: dict[str, list[Candle]] = {
            timeframe: sorted(data[timeframe], key=lambda candle: candle.timestamp)
            for timeframe in timeframes
            if timeframe in data
        }
        self.spread_price = max(0.0, float(spread_price))
        self._cursor = 0
        self._timestamps = {tf: [bar.timestamp for bar in bars] for tf, bars in self._data.items()}
        for tf, timestamps in self._timestamps.items():
            if len(set(timestamps)) != len(timestamps):
                raise ValueError(f"Duplicate {tf} candle timestamps")

    # ------------------------------------------------------------------ cursor
    @property
    def bars(self) -> list[Candle]:
        return self._data["M5"]

    @property
    def cursor(self) -> int:
        return self._cursor

    @cursor.setter
    def cursor(self, index: int) -> None:
        self._cursor = max(0, min(int(index), len(self.bars) - 1))

    @property
    def now(self) -> datetime:
        return self.bars[self._cursor].timestamp

    def __len__(self) -> int:
        return len(self.bars)

    def seek(self, moment: datetime) -> int:
        """Move the cursor to the last M5 bar at or before ``moment``."""

        target = to_utc(moment)
        for index in range(len(self.bars) - 1, -1, -1):
            if self.bars[index].timestamp <= target:
                self._cursor = index
                return index
        self._cursor = 0
        return 0

    # ------------------------------------------------------------------ feed API
    async def fetch_candles(self, instrument: str, timeframe: str, count: int) -> list[Candle]:
        bars = self._data.get(timeframe)
        if not bars:
            return []
        # Candle timestamps are OPEN times. At the close of this M5 bar,
        # a precomputed H4 candle may still contain 3h55m of future prices.
        as_of = self.now + timedelta(seconds=TIMEFRAME_SECONDS["M5"])
        latest_open = as_of - timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
        stop = bisect_right(self._timestamps[timeframe], latest_open)
        start = max(0, stop - count) if count > 0 else 0
        return bars[start:stop]

    async def fetch_price(self, instrument: str) -> tuple[float, float]:
        close = self.bars[self._cursor].close
        half = self.spread_price / 2.0
        return round(close - half, 3), round(close + half, 3)

    async def close(self) -> None:
        return None


def fixed_end(year: int = 2026, month: int = 10, day: int = 7) -> datetime:
    """A stable Wednesday midnight UTC — keeps research runs reproducible."""

    return datetime(year, month, day, tzinfo=timezone.utc)
