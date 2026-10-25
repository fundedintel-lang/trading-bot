"""Deterministic synthetic XAU/USD market data.

Synthetic data exists for three reasons:

* run the engine, dashboard and backtester end-to-end without broker credentials,
* provide reproducible fixtures for the test suite,
* let operators validate plumbing before subscribing to a paid feed.

The generator produces regime-switching trends, session-dependent volatility and
spreads, displacement impulses, liquidity sweeps (long wicks) and realistic
weekend closures.  It is **not** a market simulator for profitability claims.
"""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

from apex_smc_bot.data.candle_manager import (
    Candle,
    DataFeed,
    TIMEFRAME_SECONDS,
    floor_to_timeframe,
    resample,
    to_utc,
    utcnow,
)

# Session volatility / spread profile per UTC hour, mirroring the kill-zone map.
_SESSION_VOL = {
    0: 0.55, 1: 0.55, 2: 0.60, 3: 0.65, 4: 0.70, 5: 0.70, 6: 0.75,
    7: 1.25, 8: 1.35, 9: 1.30, 10: 1.15, 11: 1.05,
    12: 1.30, 13: 1.45, 14: 1.40, 15: 1.10, 16: 1.00,
    17: 0.90, 18: 0.85, 19: 0.80, 20: 0.65,
    21: 0.85, 22: 0.60, 23: 0.55,
}
_SESSION_SPREAD = {
    0: 0.30, 1: 0.30, 2: 0.30, 3: 0.32, 4: 0.34, 5: 0.34, 6: 0.32,
    7: 0.20, 8: 0.18, 9: 0.19, 10: 0.20, 11: 0.24,
    12: 0.22, 13: 0.18, 14: 0.19, 15: 0.22, 16: 0.26,
    17: 0.26, 18: 0.28, 19: 0.30, 20: 0.34,
    21: 0.75, 22: 0.55, 23: 0.36,
}


def _is_market_open(ts: datetime) -> bool:
    """Gold CFD session: Sun 22:00 UTC → Fri 21:00 UTC (approx.)."""

    weekday = ts.weekday()  # Monday=0 ... Sunday=6
    if weekday == 5:  # Saturday
        return False
    if weekday == 6:  # Sunday
        return ts.hour >= 22
    if weekday == 4:  # Friday
        return ts.hour < 21
    return True


def _bar_sigma(timeframe: str, reference_sigma: float = 0.95) -> float:
    """Scale base M5 noise to the requested timeframe (square-root of time)."""

    seconds = TIMEFRAME_SECONDS.get(timeframe)
    if seconds is None:
        raise ValueError(f"Unsupported timeframe {timeframe!r}")
    return reference_sigma * math.sqrt(seconds / 300.0)


def generate_candles(
    start: datetime,
    days: int,
    timeframe: str = "M5",
    seed: int = 42,
    start_price: float = 2350.0,
    volatility: float = 1.8,
) -> list[Candle]:
    """Generate deterministic OHLCV candles ending at ``start + days``.

    Args:
        start: First bar timestamp (UTC; naive values treated as UTC).
        days: Number of calendar days to simulate.
        timeframe: Output timeframe (market sessions are always modelled on M5).
        seed: RNG seed — identical inputs return identical candles.
        start_price: Initial mid price.
        volatility: Multiplier on the base bar sigma (1.0 = realistic gold).
    """

    if days <= 0:
        raise ValueError("days must be positive")
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe {timeframe!r}")
    if start_price <= 0:
        raise ValueError("start_price must be positive")
    if volatility <= 0:
        raise ValueError("volatility must be positive")

    start = floor_to_timeframe(to_utc(start), "M5")
    end = start + timedelta(days=days)
    rng = random.Random(seed)

    price = float(start_price)
    anchor = float(start_price)
    sigma_base = _bar_sigma("M5", 0.55) * volatility

    direction = 1
    regime_left = rng.randint(150, 400)
    regime_strength = rng.uniform(0.4, 1.0)
    candles: list[Candle] = []

    ts = start
    while ts < end:
        if not _is_market_open(ts):
            ts += timedelta(minutes=5)
            continue

        hour = ts.hour
        session_vol = _SESSION_VOL.get(hour, 0.8)
        sigma = sigma_base * session_vol

        if regime_left <= 0:
            direction = rng.choice((-1, 1, 1, -1, 0))
            regime_left = rng.randint(120, 480)
            regime_strength = rng.uniform(0.3, 1.0)
        regime_left -= 1

        anchor += rng.gauss(0.0, 0.35)  # slowly drifting fair value
        pull = (anchor - price) * 0.0015

        drift = direction * regime_strength * sigma * 0.10
        noise = rng.gauss(0.0, sigma * 0.55)
        impulse = 0.0
        impulse_probability = 0.012 if direction != 0 else 0.004
        if rng.random() < impulse_probability:
            impulse_dir = direction if direction != 0 else rng.choice((-1, 1))
            impulse = impulse_dir * sigma * rng.uniform(2.8, 5.2)

        open_price = price
        close_price = max(50.0, open_price + drift + noise + impulse + pull)

        body = abs(close_price - open_price)
        upper_wick = abs(rng.gauss(0.0, sigma * 0.35))
        lower_wick = abs(rng.gauss(0.0, sigma * 0.35))

        # Liquidity raids: occasionally print a long wick beyond the body.
        if rng.random() < 0.045:
            if rng.random() < 0.5:
                upper_wick += sigma * rng.uniform(1.2, 3.2)
            else:
                lower_wick += sigma * rng.uniform(1.2, 3.2)

        high = max(open_price, close_price) + upper_wick
        low = min(open_price, close_price) - lower_wick
        if high - low < 0.03:  # never emit a degenerate zero-range bar
            high, low = max(high, close_price + 0.02), min(low, close_price - 0.02)

        typical_range = max(0.05, high - low)
        session_volume = 90.0 * (session_vol ** 1.4)
        volume = session_volume * (0.75 + typical_range / max(sigma, 1e-6) * 0.35)
        volume *= rng.uniform(0.75, 1.35)
        if rng.random() < impulse_probability:
            volume *= 1.8
        spread = max(0.05, _SESSION_SPREAD.get(hour, 0.3) + rng.gauss(0.0, 0.03))

        candles.append(
            Candle(
                timestamp=ts,
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close_price, 2),
                volume=round(volume, 2),
                spread=round(spread, 2),
            )
        )
        price = close_price
        ts += timedelta(minutes=5)

    if timeframe == "M5":
        return candles
    return resample(candles, timeframe)


def generate_market_data(
    days: int = 60,
    seed: int = 42,
    end: datetime | None = None,
    start_price: float = 2350.0,
    volatility: float = 1.8,
) -> dict[str, list[Candle]]:
    """Generate a consistent multi-timeframe dataset (M5 → M15/H1/H4).

    Higher timeframes are resampled from the M5 series so all four are aligned.
    """

    end = to_utc(end or utcnow())
    start = floor_to_timeframe(end - timedelta(days=days), "M5")
    m5 = generate_candles(start, days, "M5", seed, start_price, volatility)
    return {
        "M5": m5,
        "M15": resample(m5, "M15"),
        "H1": resample(m5, "H1"),
        "H4": resample(m5, "H4"),
    }


def load_candles_from_csv(path: str | Path, timeframe: str = "M5") -> list[Candle]:
    """Load OHLCV candles from a CSV with header row.

    Accepted columns: timestamp/date/time, open, high, low, close, volume,
    spread. Timestamps may be ISO-8601 or UNIX seconds/milliseconds.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV not found: {file_path}")
    candles: list[Candle] = []
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        required = {"open", "high", "low", "close"}
        if not required.issubset(fields):
            raise ValueError(f"CSV missing required columns: {sorted(required - set(fields))}")
        time_key = next(
            (fields[key] for key in ("timestamp", "date", "time", "datetime") if key in fields),
            None,
        )
        if time_key is None:
            raise ValueError("CSV missing a timestamp/date column")
        for line_number, row in enumerate(reader, start=2):
            try:
                raw_time = row[time_key].strip()
                if raw_time.replace(".", "", 1).isdigit():
                    value = float(raw_time)
                    if value > 1e11:  # milliseconds
                        value /= 1000.0
                    timestamp = datetime.fromtimestamp(value, tz=timezone.utc)
                else:
                    timestamp = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                candles.append(
                    Candle(
                        timestamp=to_utc(timestamp),
                        open=float(row[fields["open"]]),
                        high=float(row[fields["high"]]),
                        low=float(row[fields["low"]]),
                        close=float(row[fields["close"]]),
                        volume=float(row[fields["volume"]]) if "volume" in fields else 0.0,
                        spread=float(row[fields["spread"]]) if "spread" in fields else 0.0,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid CSV row {line_number}: {exc}") from exc
    if not candles:
        raise ValueError("CSV contains no candles")
    candles.sort(key=lambda c: c.timestamp)
    if timeframe != "M5":
        candles = resample(candles, timeframe)
    return candles


class SyntheticFeed(DataFeed):
    """Offline feed backed by :func:`generate_candles`.

    The generator is a strict forward simulation from one fixed ``start``, so a
    longer run always reproduces the same prefix.  That lets this feed behave
    like a *live* market: bars keep appearing as the wall clock advances, while
    everything already delivered stays byte-identical.

    Only closed bars are ever exposed (``timestamp <= last closed M5 bar``), so
    a caller can never see a candle that has not finished — the same no
    look-ahead guarantee the replay feed gives a backtest.
    """

    name = "synthetic"

    def __init__(
        self,
        days: int = 30,
        seed: int = 7,
        start_price: float = 2350.0,
        volatility: float = 1.8,
        now: datetime | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.days = max(5, days)
        self.seed = seed
        self.start_price = start_price
        self.volatility = volatility
        self._clock = clock or utcnow
        anchor = to_utc(now or self._clock())
        self._start = floor_to_timeframe(anchor - timedelta(days=self.days), "M5")
        self._m5: list[Candle] = []
        self._generated_days = 0

    # ------------------------------------------------------------------ internals
    def _ensure_horizon(self, now: datetime) -> None:
        """(Re)generate the series whenever the clock needs bars we do not have.

        The horizon carries one day of slack so regeneration happens roughly
        once a day instead of on every poll.
        """

        last_bar = floor_to_timeframe(now, "M5")
        span = last_bar + timedelta(days=1) - self._start
        needed = max(1, math.ceil(span.total_seconds() / 86400.0))
        if self._m5 and needed <= self._generated_days:
            return
        self._generated_days = needed
        self._m5 = generate_candles(
            self._start, needed, "M5", self.seed, self.start_price, self.volatility
        )

    def _closed(self) -> list[Candle]:
        """Closed M5 bars only, newest last."""

        now = to_utc(self._clock())
        self._ensure_horizon(now)
        cutoff = floor_to_timeframe(now, "M5")
        closed = [bar for bar in self._m5 if bar.timestamp < cutoff]
        return closed or [self._m5[0]]

    # ------------------------------------------------------------------ DataFeed
    async def fetch_candles(self, instrument: str, timeframe: str, count: int) -> list[Candle]:
        history = self._closed()
        candles = history if timeframe == "M5" else resample(history, timeframe)
        return candles[-count:] if count > 0 else list(candles)

    async def fetch_price(self, instrument: str) -> tuple[float, float]:
        last = self._closed()[-1]
        half = max(0.05, last.spread) / 2.0
        return round(last.close - half, 2), round(last.close + half, 2)

    async def close(self) -> None:
        return None
