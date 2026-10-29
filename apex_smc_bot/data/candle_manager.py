"""Candle domain model, resampling utilities and the in-memory candle manager.

The rest of the engine consumes plain ``Sequence[Candle]`` objects.  Keeping the
data layer free of pandas makes the analysis deterministic, fast and trivially
unit-testable.  pandas is only used for optional reporting / dashboards.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, runtime_checkable

TIMEFRAME_SECONDS: dict[str, int] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}

#: Timeframes required by the strategy, fastest first.
REQUIRED_TIMEFRAMES: tuple[str, ...] = ("M5", "M15", "H1", "H4")

#: Instrument convention used across the engine.  Gold quotes with 2 decimals;
#: one "pip" is 0.10 USD (so a 2.70 USD stop is a 27 pip stop).
PIP_SIZE_XAUUSD: float = 0.10
CONTRACT_SIZE_XAUUSD: float = 100.0  # ounces per 1.00 lot
PIP_VALUE_PER_LOT_XAUUSD: float = PIP_SIZE_XAUUSD * CONTRACT_SIZE_XAUUSD  # $10 / pip / lot


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""

    return datetime.now(timezone.utc)


def to_utc(value: datetime) -> datetime:
    """Normalise a datetime to timezone-aware UTC.

    Naive datetimes are interpreted as UTC (never local time) so that session
    logic is deterministic regardless of host timezone.
    """

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def floor_to_timeframe(value: datetime, timeframe: str) -> datetime:
    """Floor a timestamp to the opening boundary of its timeframe bucket."""

    seconds = TIMEFRAME_SECONDS[timeframe]
    utc_value = to_utc(value)
    epoch = int(utc_value.timestamp())
    return datetime.fromtimestamp((epoch // seconds) * seconds, tz=timezone.utc)


@dataclass(frozen=True, slots=True)
class Candle:
    """A single OHLCV bar.

    Attributes:
        timestamp: Bar open time (UTC).
        open/high/low/close: Prices in quote currency (USD for XAU/USD).
        volume: Tick volume from the broker (proxy for real volume on OTC gold).
        spread: Quoted spread in price units at bar close (0.0 if unknown).
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    spread: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", to_utc(self.timestamp))
        for name in ("open", "high", "low", "close", "volume", "spread"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Candle.{name} must be a finite number, got {value!r}")
        if self.high < self.low:
            raise ValueError(
                f"Invalid candle {self.timestamp.isoformat()}: high {self.high} < low {self.low}"
            )
        if self.high + 1e-9 < max(self.open, self.close):
            raise ValueError(
                f"Invalid candle {self.timestamp.isoformat()}: high {self.high} below open/close"
            )
        if self.low - 1e-9 > min(self.open, self.close):
            raise ValueError(
                f"Invalid candle {self.timestamp.isoformat()}: low {self.low} above open/close"
            )
        if self.volume < 0:
            raise ValueError("Candle.volume must be >= 0")
        if self.spread < 0:
            raise ValueError("Candle.spread must be >= 0")

    @property
    def body(self) -> float:
        """Absolute body size."""

        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        """High-low range."""

        return self.high - self.low

    @property
    def upper_wick(self) -> float:
        """Size of the upper wick."""

        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        """Size of the lower wick."""

        return min(self.open, self.close) - self.low

    @property
    def body_ratio(self) -> float:
        """Body size as a fraction of total range (0.0 when range is zero)."""

        return self.body / self.range if self.range > 0 else 0.0

    @property
    def direction(self) -> str:
        """``BULLISH``, ``BEARISH`` or ``DOJI``."""

        if self.close > self.open:
            return "BULLISH"
        if self.close < self.open:
            return "BEARISH"
        return "DOJI"

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "spread": self.spread,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candle":
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return cls(
            timestamp=timestamp,
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data.get("volume", 0.0) or 0.0),
            spread=float(data.get("spread", 0.0) or 0.0),
        )


@dataclass(slots=True)
class CandleSeries:
    """An immutable-by-convention, chronologically sorted list of candles."""

    candles: list[Candle] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.candles = sorted(self.candles, key=lambda c: c.timestamp)
        seen: set[datetime] = set()
        for candle in self.candles:
            if candle.timestamp in seen:
                raise ValueError(f"Duplicate candle timestamp {candle.timestamp.isoformat()}")
            seen.add(candle.timestamp)

    def __len__(self) -> int:
        return len(self.candles)

    def __iter__(self) -> Iterator[Candle]:
        return iter(self.candles)

    def __getitem__(self, index: int | slice) -> Candle | list[Candle]:
        return self.candles[index]

    def __bool__(self) -> bool:
        return bool(self.candles)

    @property
    def timestamps(self) -> list[datetime]:
        return [c.timestamp for c in self.candles]

    @property
    def opens(self) -> list[float]:
        return [c.open for c in self.candles]

    @property
    def highs(self) -> list[float]:
        return [c.high for c in self.candles]

    @property
    def lows(self) -> list[float]:
        return [c.low for c in self.candles]

    @property
    def closes(self) -> list[float]:
        return [c.close for c in self.candles]

    @property
    def volumes(self) -> list[float]:
        return [c.volume for c in self.candles]

    @property
    def first(self) -> Candle | None:
        return self.candles[0] if self.candles else None

    @property
    def last(self) -> Candle | None:
        return self.candles[-1] if self.candles else None

    def tail(self, count: int) -> list[Candle]:
        """Return the most recent ``count`` candles (order preserved)."""

        if count <= 0:
            return []
        return self.candles[-count:]

    def closed_before(self, timestamp: datetime) -> list[Candle]:
        """Candles that closed strictly before ``timestamp`` (no look-ahead)."""

        return [c for c in self.candles if c.timestamp < timestamp]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.candles]

    @classmethod
    def from_iterable(cls, candles: Iterable[Candle]) -> "CandleSeries":
        return cls(candles=list(candles))


def true_range(previous_close: float, candle: Candle) -> float:
    """Wilder true range for one candle."""

    return max(candle.high - candle.low, abs(candle.high - previous_close), abs(candle.low - previous_close))


def atr(candles: Sequence[Candle], period: int = 14) -> float:
    """Average True Range (Wilder smoothing) over the final candles.

    Returns 0.0 when there is insufficient data.
    """

    if period <= 0:
        raise ValueError("ATR period must be positive")
    if len(candles) < period + 1:
        return 0.0
    window = candles[-(period + 1):]
    ranges = [true_range(window[i - 1].close, window[i]) for i in range(1, len(window))]
    value = sum(ranges) / len(ranges)
    return round(value, 6)


def average_volume(candles: Sequence[Candle], period: int = 20) -> float:
    """Mean tick volume over the trailing ``period`` candles (excluding the last)."""

    if period <= 0:
        raise ValueError("volume period must be positive")
    if len(candles) < 2:
        return 0.0
    window = candles[-(period + 1):-1]
    if not window:
        return 0.0
    return sum(c.volume for c in window) / len(window)


def resample(candles: Sequence[Candle], timeframe: str) -> list[Candle]:
    """Aggregate candles into a higher timeframe using bucket-open timestamps.

    Candles that share a bucket are merged: open of the first, high/low extremes,
    close of the last, summed volume and the last observed spread.
    """

    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe {timeframe!r}")
    if not candles:
        return []
    buckets: dict[datetime, list[Candle]] = {}
    for candle in candles:
        key = floor_to_timeframe(candle.timestamp, timeframe)
        buckets.setdefault(key, []).append(candle)
    out: list[Candle] = []
    for key in sorted(buckets):
        group = buckets[key]
        out.append(
            Candle(
                timestamp=key,
                open=group[0].open,
                high=max(c.high for c in group),
                low=min(c.low for c in group),
                close=group[-1].close,
                volume=sum(c.volume for c in group),
                spread=group[-1].spread,
            )
        )
    return out


@runtime_checkable
class DataFeed(Protocol):
    """Minimal contract every market-data adapter must satisfy."""

    name: str

    async def fetch_candles(self, instrument: str, timeframe: str, count: int) -> list[Candle]:
        """Return up to ``count`` of the most recent completed candles."""

    async def fetch_price(self, instrument: str) -> tuple[float, float]:
        """Return ``(bid, ask)`` for the instrument."""

    async def close(self) -> None:
        """Release network resources."""


class CandleManager:
    """In-memory + database backed OHLCV cache shared by the whole engine.

    Responsibilities:
      * keep all four strategy timeframes sorted and de-duplicated,
      * bootstrap history from the configured feed,
      * persist candles (optional) for restart resilience and research,
      * expose the latest quote/spread to execution and risk layers.
    """

    def __init__(
        self,
        instrument: str = "XAU_USD",
        feed: DataFeed | None = None,
        database: Any | None = None,
        max_candles_per_timeframe: int = 3000,
    ) -> None:
        self.instrument = instrument
        self.feed = feed
        self.database = database
        self.max_candles_per_timeframe = max_candles_per_timeframe
        self._candles: dict[str, list[Candle]] = {tf: [] for tf in TIMEFRAME_SECONDS}
        self._lock = asyncio.Lock()
        self._last_quote: tuple[float, float] = (0.0, 0.0)
        self.quote_received_at: datetime | None = None

    # ------------------------------------------------------------------ ingest
    def ingest(self, timeframe: str, candles: Iterable[Candle]) -> int:
        """Insert/merge candles for a timeframe, returning how many were new."""

        if timeframe not in TIMEFRAME_SECONDS:
            raise ValueError(f"Unsupported timeframe {timeframe!r}")
        bucket = self._candles.setdefault(timeframe, [])
        known = {c.timestamp: i for i, c in enumerate(bucket)}
        added = 0
        for candle in candles:
            existing = known.get(candle.timestamp)
            if existing is None:
                bucket.append(candle)
                known[candle.timestamp] = len(bucket) - 1
                added += 1
            else:
                bucket[existing] = candle
        bucket.sort(key=lambda c: c.timestamp)
        if len(bucket) > self.max_candles_per_timeframe:
            self._candles[timeframe] = bucket[-self.max_candles_per_timeframe:]
        return added

    def ingest_many(self, data: dict[str, Iterable[Candle]]) -> dict[str, int]:
        """Ingest multiple timeframes at once."""

        return {tf: self.ingest(tf, candles) for tf, candles in data.items()}

    def get_candles(
        self, timeframe: str, count: int | None = None, *, as_of: datetime | None = None
    ) -> list[Candle]:
        """Return cached candles for ``timeframe`` (most recent ``count``)."""

        if timeframe not in self._candles:
            raise ValueError(f"Unsupported timeframe {timeframe!r}")
        candles = self._candles[timeframe]
        if as_of is not None:
            cutoff = to_utc(as_of) - timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
            candles = [bar for bar in candles if bar.timestamp <= cutoff]
        if count is None or count <= 0:
            return list(candles)
        return list(candles[-count:])

    def get_series(self, timeframe: str, count: int | None = None) -> CandleSeries:
        return CandleSeries(self.get_candles(timeframe, count))

    def count(self, timeframe: str) -> int:
        return len(self._candles.get(timeframe, []))

    def latest_candle(self, timeframe: str) -> Candle | None:
        candles = self._candles.get(timeframe) or []
        return candles[-1] if candles else None

    @property
    def latest_price(self) -> float:
        """Latest mid price (from quote when available, else last M5 close)."""

        bid, ask = self._last_quote
        if bid > 0 and ask > 0:
            return round((bid + ask) / 2.0, 2)
        for timeframe in ("M5", "M15", "H1", "H4"):
            candle = self.latest_candle(timeframe)
            if candle is not None:
                return candle.close
        return 0.0

    @property
    def latest_spread(self) -> float:
        """Latest spread in **price units** (USD for gold)."""

        bid, ask = self._last_quote
        if bid > 0 and ask > 0:
            return round(ask - bid, 4)
        candle = self.latest_candle("M5")
        return round(candle.spread, 4) if candle is not None else 0.0

    def set_quote(self, bid: float, ask: float, now: datetime | None = None) -> None:
        """Cache the latest quote (used by ``latest_price``/spread)."""

        if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask <= 0 or ask < bid:
            raise ValueError(f"Invalid quote bid={bid} ask={ask}")
        self._last_quote = (float(bid), float(ask))
        self.quote_received_at = to_utc(now or utcnow())

    # ------------------------------------------------------------------ network
    async def refresh(self, timeframe: str, count: int = 500) -> int:
        """Fetch and ingest fresh candles for one timeframe."""

        if self.feed is None:
            return 0
        candles = await self.feed.fetch_candles(self.instrument, timeframe, count)
        known = {bar.timestamp: bar for bar in self._candles[timeframe]}
        changed = [bar for bar in candles if known.get(bar.timestamp) != bar]
        added = self.ingest(timeframe, candles)
        if self.database is not None and changed:
            try:
                self.database.upsert_candles(self.instrument, timeframe, changed)
            except Exception:  # pragma: no cover - persistence must never break trading
                pass
        return added

    async def refresh_all(self, counts: dict[str, int] | None = None) -> dict[str, int]:
        """Refresh all timeframes concurrently."""

        counts = counts or {tf: 500 for tf in REQUIRED_TIMEFRAMES}
        async with self._lock:
            results = await asyncio.gather(
                *(self.refresh(tf, counts.get(tf, 500)) for tf in counts),
                return_exceptions=True,
            )
        summary: dict[str, int] = {}
        for timeframe, result in zip(counts, results):
            summary[timeframe] = result if isinstance(result, int) else 0
        return summary

    async def bootstrap(self, minimum: int = 500) -> dict[str, int]:
        """Load historical candles from DB first, then top-up from the feed."""

        loaded: dict[str, int] = {}
        for timeframe in REQUIRED_TIMEFRAMES:
            if self.database is not None:
                try:
                    cached = self.database.get_candles(self.instrument, timeframe, limit=minimum)
                    if cached:
                        self.ingest(timeframe, cached)
                        loaded[timeframe] = len(cached)
                except Exception:  # pragma: no cover
                    pass
        if self.feed is not None:
            for timeframe in REQUIRED_TIMEFRAMES:
                if self.count(timeframe) < minimum:
                    try:
                        await self.refresh(timeframe, count=minimum)
                    except Exception:
                        continue
        return loaded

    async def refresh_quote(self, now: datetime | None = None) -> tuple[float, float]:
        """Refresh and cache the live bid/ask quote."""

        if self.feed is None:
            return self._last_quote
        bid, ask = await self.feed.fetch_price(self.instrument)
        self.set_quote(bid, ask, now)
        return bid, ask

    async def close(self) -> None:
        if self.feed is not None:
            await self.feed.close()

    # ------------------------------------------------------------------ helpers
    def snapshot(self) -> dict[str, int]:
        """Counts per timeframe (for health endpoints / dashboard)."""

        return {tf: len(self._candles.get(tf, [])) for tf in TIMEFRAME_SECONDS}

    def seed(self, data: dict[str, Iterable[Candle]]) -> None:
        """Synchronously load pre-fetched history (used by backtests/tests)."""

        for timeframe, candles in data.items():
            self.ingest(timeframe, candles)
