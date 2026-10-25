"""Shared pytest fixtures.

All market data is deterministic: ``fixed_end`` pins the synthetic generator so
tests never depend on the wall clock.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest

from apex_smc_bot.config import Settings
from apex_smc_bot.data.candle_manager import Candle, resample, to_utc, utcnow
from apex_smc_bot.data.database import Database
from apex_smc_bot.data.replay_feed import fixed_end
from apex_smc_bot.data.synthetic import generate_candles, generate_market_data


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def market_data() -> dict[str, list[Candle]]:
    return generate_market_data(days=12, seed=11, end=fixed_end())


@pytest.fixture(scope="session")
def m5(market_data: dict[str, list[Candle]]) -> list[Candle]:
    return market_data["M5"]


@pytest.fixture(scope="session")
def m15(market_data: dict[str, list[Candle]]) -> list[Candle]:
    return market_data["M15"]


@pytest.fixture(scope="session")
def h1(market_data: dict[str, list[Candle]]) -> list[Candle]:
    return market_data["H1"]


@pytest.fixture(scope="session")
def h4(market_data: dict[str, list[Candle]]) -> list[Candle]:
    return market_data["H4"]


@pytest.fixture()
def session_candles(m15: list[Candle]) -> list[Candle]:
    """A 400-bar slice — the window the engine analyses in production."""

    return m15[-400:]


@pytest.fixture()
def database(tmp_path: Path) -> Iterator[Database]:
    db = Database(tmp_path / "test_state.sqlite3")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def trade_time() -> datetime:
    """A Wednesday inside the NY kill zone (2026-10-07 13:30 UTC)."""

    return datetime(2026, 10, 7, 13, 30, tzinfo=timezone.utc)


@pytest.fixture()
def make_candle() -> Any:
    """Factory for hand-built candles (used for surgical unit tests)."""

    def _make(
        open_: float,
        high: float,
        low: float,
        close: float,
        timestamp: datetime | None = None,
        volume: float = 1000.0,
        spread: float = 0.20,
    ) -> Candle:
        return Candle(
            timestamp=to_utc(timestamp or utcnow()),
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            spread=spread,
        )

    return _make


@pytest.fixture()
def trend_candles() -> list[Candle]:
    """A clean impulsive downtrend with one pullback (for structure tests)."""

    start = datetime(2026, 10, 5, 8, 0, tzinfo=timezone.utc)
    candles: list[Candle] = []
    price = 2400.0
    for index in range(60):
        drift = -1.0 if index % 12 not in (0, 1) else 0.6
        open_ = price
        close = price + drift
        high = max(open_, close) + 0.35
        low = min(open_, close) - 0.35
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=15 * index),
                open=round(open_, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                volume=1000 + index,
                spread=0.2,
            )
        )
        price = close
    return candles
