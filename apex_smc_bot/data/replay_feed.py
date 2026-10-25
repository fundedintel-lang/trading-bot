"""Deterministic replay feed used by the backtester and tests.

``ReplayFeed`` exposes an already-generated multi-timeframe dataset as a
:class:`~apex_smc_bot.data.candle_manager.DataFeed`, serving only the bars up to
an explicit cursor.  That makes the live engine loop replayable bar by bar with
no look-ahead bias.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

from apex_smc_bot.data.candle_manager import Candle, DataFeed, floor_to_timeframe, to_utc


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
            timeframe: list(data[timeframe])
            for timeframe in timeframes
            if timeframe in data
        }
        self.spread_price = max(0.0, float(spread_price))
        self._cursor = 0
        self._by_timeframe: dict[str, dict[datetime, int]] = {
            timeframe: {candle.timestamp: index for index, candle in enumerate(bars)}
            for timeframe, bars in self._data.items()
        }

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
        cutoff = self.now
        index = self._by_timeframe[timeframe]
        position = index.get(floor_to_timeframe(cutoff, timeframe))
        if position is None:
            # Binary-search-free fallback: last bar at or before the cutoff.
            position = -1
            for offset in range(len(bars) - 1, -1, -1):
                if bars[offset].timestamp <= cutoff:
                    position = offset
                    break
            if position < 0:
                return []
        start = max(0, position + 1 - max(1, int(count)))
        return bars[start : position + 1]

    async def fetch_price(self, instrument: str) -> tuple[float, float]:
        close = self.bars[self._cursor].close
        half = self.spread_price / 2.0
        return round(close - half, 3), round(close + half, 3)

    async def close(self) -> None:
        return None


def fixed_end(year: int = 2026, month: int = 10, day: int = 7) -> datetime:
    """A stable Wednesday midnight UTC — keeps research runs reproducible."""

    return datetime(year, month, day, tzinfo=timezone.utc)
