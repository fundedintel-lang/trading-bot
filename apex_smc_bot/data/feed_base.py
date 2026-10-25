"""Shared feed errors and retry helpers."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from apex_smc_bot.data.candle_manager import DataFeed

logger = logging.getLogger("apex.feed")
T = TypeVar("T")


class FeedError(RuntimeError):
    """Raised when a market-data provider cannot satisfy a request."""


class FeedUnavailable(FeedError):
    """Raised when a provider is not configured or its SDK is missing."""


async def with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    backoff_seconds: float = 1.5,
    label: str = "feed request",
    retry_on: tuple[type[BaseException], ...] = (FeedError, asyncio.TimeoutError, OSError),
) -> T:
    """Execute an async operation with exponential backoff.

    The final failure is re-raised as :class:`FeedError` so callers have a single
    predictable exception type.
    """

    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await operation()
        except retry_on as exc:  # noqa: PERF203 - retries are intentional
            last_error = exc
            if attempt == attempts - 1:
                break
            delay = backoff_seconds * (2 ** attempt) + random.uniform(0.0, 0.15)
            logger.warning(
                "%s failed (attempt %s/%s): %s — retrying in %.2fs",
                label,
                attempt + 1,
                attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    raise FeedError(f"{label} failed after {attempts} attempts: {last_error}") from last_error


def describe_feed(feed: DataFeed | None) -> dict[str, Any]:
    """Small serialisable description used in health logs."""

    if feed is None:
        return {"name": "none", "configured": False}
    return {"name": getattr(feed, "name", type(feed).__name__), "configured": True}
