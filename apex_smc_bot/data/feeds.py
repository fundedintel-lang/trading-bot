"""Feed factory + failover router.

Configured feeds are tried in order (primary → secondary → tertiary).  A failing
provider never crashes the engine: the router logs the failure and moves on.  If
every provider fails the error is surfaced as :class:`FeedError` to the caller.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from apex_smc_bot.config import Settings
from apex_smc_bot.data.candle_manager import Candle, DataFeed
from apex_smc_bot.data.feed_base import FeedError, FeedUnavailable
from apex_smc_bot.data.feed_mt5 import MT5Feed
from apex_smc_bot.data.feed_oanda import OandaFeed
from apex_smc_bot.data.feed_twelve import TwelveDataFeed
from apex_smc_bot.data.synthetic import SyntheticFeed

logger = logging.getLogger("apex.feeds")

SUPPORTED_FEEDS = ("oanda", "synthetic", "twelve", "mt5")


class FallbackFeed(DataFeed):
    """Tries providers in priority order and remembers the healthy one."""

    name = "fallback"

    def __init__(self, feeds: Sequence[DataFeed]) -> None:
        if not feeds:
            raise ValueError("FallbackFeed requires at least one feed")
        self.feeds = list(feeds)
        self.active_index = 0

    @property
    def active(self) -> DataFeed:
        return self.feeds[self.active_index]

    async def _try_all(self, method: str, *args, **kwargs):
        errors: list[str] = []
        order = list(range(len(self.feeds)))
        # Prefer the provider that worked last time.
        if self.active_index in order:
            order.remove(self.active_index)
            order.insert(0, self.active_index)
        for index in order:
            feed = self.feeds[index]
            try:
                result = await getattr(feed, method)(*args, **kwargs)
                if index != self.active_index:
                    logger.warning("Feed failover: switching active feed to %s", feed.name)
                    self.active_index = index
                return result
            except (FeedError, FeedUnavailable, OSError, ValueError, KeyError) as exc:
                errors.append(f"{feed.name}: {exc}")
                logger.warning("Feed %s failed for %s: %s", feed.name, method, exc)
        raise FeedError(f"All feeds failed for {method}: {' | '.join(errors)}")

    async def fetch_candles(self, instrument: str, timeframe: str, count: int) -> list[Candle]:
        return await self._try_all("fetch_candles", instrument, timeframe, count)

    async def fetch_price(self, instrument: str) -> tuple[float, float]:
        return await self._try_all("fetch_price", instrument)

    async def close(self) -> None:
        for feed in self.feeds:
            try:
                await feed.close()
            except Exception:  # noqa: BLE001
                pass


def _build_single(name: str, settings: Settings) -> DataFeed | None:
    name = (name or "").strip().lower()
    if name in {"", "none"}:
        return None
    if name == "synthetic":
        return SyntheticFeed(
            days=max(90, settings.feed.historical_bars // 12 + 30),
            seed=settings.feed.synthetic_seed,
            volatility=1.0,
        )
    if name == "oanda":
        if not settings.secrets.has_oanda:
            logger.warning("Feed 'oanda' selected but credentials are missing — skipping")
            return None
        try:
            return OandaFeed(
                account_id=settings.secrets.oanda_account_id,
                access_token=settings.secrets.oanda_access_token,
                environment=settings.execution.oanda_environment,
                timeout_seconds=settings.feed.request_timeout_seconds,
                max_retries=settings.feed.max_retries,
            )
        except FeedUnavailable as exc:
            logger.warning("OANDA feed unavailable: %s", exc)
            return None
    if name == "twelve":
        if not settings.secrets.twelve_data_api_key:
            logger.warning("Feed 'twelve' selected but TWELVE_DATA_API_KEY is missing — skipping")
            return None
        try:
            return TwelveDataFeed(
                settings.secrets.twelve_data_api_key,
                timeout_seconds=settings.feed.request_timeout_seconds,
                max_retries=settings.feed.max_retries,
            )
        except FeedUnavailable as exc:
            logger.warning("TwelveData feed unavailable: %s", exc)
            return None
    if name == "mt5":
        try:
            return MT5Feed(
                login=settings.secrets.mt5_login,
                password=settings.secrets.mt5_password,
                server=settings.secrets.mt5_server,
                symbol=settings.bot.instrument.replace("_", ""),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MT5 feed unavailable: %s", exc)
            return None
    raise FeedError(f"Unknown feed {name!r}; supported: {SUPPORTED_FEEDS}")


def build_feed(settings: Settings, override: str | None = None) -> DataFeed | None:
    """Construct the configured feed chain.

    Args:
        settings: Loaded settings.
        override: Force one provider (used by ``main.py --feed synthetic``).
    """

    if override:
        names = [override]
    else:
        names = [settings.feed.primary, settings.feed.secondary, settings.feed.tertiary]
    feeds: list[DataFeed] = []
    seen: set[str] = set()
    for name in names:
        key = (name or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        feed = _build_single(key, settings)
        if feed is not None:
            feeds.append(feed)
    if not feeds:
        logger.warning("No live data feed configured — using synthetic feed")
        feeds.append(
            SyntheticFeed(
                days=max(90, settings.feed.historical_bars // 12 + 30),
                seed=settings.feed.synthetic_seed,
            )
        )
    if len(feeds) == 1:
        return feeds[0]
    return FallbackFeed(feeds)
