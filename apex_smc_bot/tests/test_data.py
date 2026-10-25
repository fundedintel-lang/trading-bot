"""Data layer: synthetic feed liveness, no look-ahead, candle store and DB."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from apex_smc_bot.data.candle_manager import CandleManager, TIMEFRAME_SECONDS, resample
from apex_smc_bot.data.database import Database
from apex_smc_bot.data.synthetic import SyntheticFeed, generate_candles, generate_market_data


class Clock:
    """Mutable stand-in for ``utcnow`` so tests can drive the wall clock."""

    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment


def _snapshot(bars):
    return {
        bar.timestamp: (bar.open, bar.high, bar.low, bar.close, bar.volume, bar.spread)
        for bar in bars
    }


@pytest.mark.asyncio
async def test_synthetic_feed_never_exposes_an_unclosed_bar():
    clock = Clock(datetime(2026, 10, 26, 12, 7, tzinfo=timezone.utc))
    feed = SyntheticFeed(days=10, seed=7, clock=clock)
    bars = await feed.fetch_candles("XAU_USD", "M5", 0)
    assert bars, "feed returned no candles"
    # 12:07 → the 12:05 bar is still forming, so the newest closed bar is 12:00.
    assert bars[-1].timestamp == datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
    assert all(bar.timestamp <= bars[-1].timestamp for bar in bars)


@pytest.mark.asyncio
async def test_synthetic_feed_grows_without_rewriting_history():
    """The feed must behave like a live market: new bars appear, old ones never change."""

    clock = Clock(datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc))
    feed = SyntheticFeed(days=10, seed=7, clock=clock)

    first = _snapshot(await feed.fetch_candles("XAU_USD", "M5", 0))
    clock.moment += timedelta(minutes=45)
    second = _snapshot(await feed.fetch_candles("XAU_USD", "M5", 0))

    assert len(second) == len(first) + 9
    shared = set(first) & set(second)
    assert shared == set(first)
    assert all(first[ts] == second[ts] for ts in shared)


@pytest.mark.asyncio
async def test_synthetic_feed_regeneration_keeps_the_prefix_identical():
    """Crossing a day boundary regenerates the series — history must survive it."""

    clock = Clock(datetime(2026, 10, 26, 23, 0, tzinfo=timezone.utc))
    feed = SyntheticFeed(days=10, seed=7, clock=clock)
    before = _snapshot(await feed.fetch_candles("XAU_USD", "M5", 0))

    clock.moment += timedelta(days=1, minutes=30)
    after = _snapshot(await feed.fetch_candles("XAU_USD", "M5", 0))

    assert len(after) > len(before)
    assert all(before[ts] == after[ts] for ts in before)


@pytest.mark.asyncio
async def test_synthetic_feed_pauses_over_the_weekend():
    friday_late = datetime(2026, 10, 30, 20, 30, tzinfo=timezone.utc)  # Friday
    clock = Clock(friday_late)
    feed = SyntheticFeed(days=10, seed=7, clock=clock)
    bars = await feed.fetch_candles("XAU_USD", "M5", 5)
    last_friday = bars[-1].timestamp
    assert last_friday <= friday_late

    clock.moment = datetime(2026, 10, 31, 15, 0, tzinfo=timezone.utc)  # Saturday
    still_closed = await feed.fetch_candles("XAU_USD", "M5", 5)
    # Friday's session closed at 21:00 UTC — nothing may be emitted after that.
    assert still_closed[-1].timestamp.weekday() == 4
    assert still_closed[-1].timestamp.hour < 21
    assert still_closed[-1].timestamp > last_friday

    # Gold reopens Sunday 22:00 UTC — bars resume from there.
    clock.moment = datetime(2026, 11, 2, 0, 0, tzinfo=timezone.utc)
    reopened = await feed.fetch_candles("XAU_USD", "M5", 5)
    assert reopened[-1].timestamp > last_friday
    assert reopened[-1].timestamp.weekday() == 6 or reopened[-1].timestamp.weekday() == 0


@pytest.mark.asyncio
async def test_quote_is_two_sided_with_a_realistic_spread():
    clock = Clock(datetime(2026, 10, 26, 13, 0, tzinfo=timezone.utc))
    feed = SyntheticFeed(days=5, seed=7, clock=clock)
    bid, ask = await feed.fetch_price("XAU_USD")
    assert 0 < bid < ask
    assert (ask - bid) / 0.10 <= 5.0  # pips, inside the configurable spread guard


def test_generator_is_deterministic_and_prefix_stable():
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    short = generate_candles(start, 5, "M5", 7)
    long = generate_candles(start, 10, "M5", 7)
    assert _snapshot(short) == _snapshot(long[: len(short)])
    assert _snapshot(short) == _snapshot(generate_candles(start, 5, "M5", 7))
    assert _snapshot(short) != _snapshot(generate_candles(start, 5, "M5", 8))


def test_resample_aggregates_ohlc_correctly():
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    m5 = generate_candles(start, 1, "M5", 3, start_price=2000.0)
    m15 = resample(m5, "M15")
    assert len(m15) == pytest.approx(len(m5) / 3, abs=1)
    first_three = m5[:3]
    assert m15[0].open == first_three[0].open
    assert m15[0].close == first_three[-1].close
    assert m15[0].high == pytest.approx(max(bar.high for bar in first_three))
    assert m15[0].low == pytest.approx(min(bar.low for bar in first_three))
    for timeframe in ("M15", "H1", "H4"):
        assert TIMEFRAME_SECONDS[timeframe] % 300 == 0


@pytest.mark.asyncio
async def test_candle_manager_ingests_without_duplicating():
    data = generate_market_data(days=3, seed=5)
    feed = SyntheticFeed(days=3, seed=5)
    manager = CandleManager(feed=feed, database=None)
    manager.ingest("M15", data["M15"])
    count = manager.count("M15")
    manager.ingest("M15", data["M15"])  # same bars again
    assert manager.count("M15") == count
    assert manager.get_candles("M15")[-1].close == data["M15"][-1].close


def test_database_round_trips_candles_and_signals(tmp_path):
    database = Database(tmp_path / "roundtrip.sqlite3")
    candles = generate_market_data(days=2, seed=9)["M15"]
    database.upsert_candles("XAU_USD", "M15", candles)
    loaded = database.get_candles("XAU_USD", "M15", limit=500)
    assert len(loaded) == len(candles)
    assert loaded[0].timestamp == candles[0].timestamp
    assert loaded[-1].close == pytest.approx(candles[-1].close, abs=1e-6)

    database.record_signal(
        signal_id="S-1",
        timestamp=candles[-1].timestamp,
        instrument="XAU_USD",
        direction="BEARISH",
        payload={"signal_id": "S-1"},
        confluence_score=8.0,
        status="GENERATED",
    )
    rows = database.get_signals(limit=5)
    assert rows and rows[0]["signal_id"] == "S-1"
    database.close()
