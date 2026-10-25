"""Economic-calendar news filter (ForexFactory weekly JSON).

Rules: no new entries within ±30 minutes of a high-impact USD/XAU event
(FOMC, NFP, CPI, PPI, JOLTS, ISM, Fed speeches, GDP, PCE…).  Existing positions
are not force-closed by this module; the orchestrator may decide to de-risk.

If the calendar cannot be fetched the filter degrades gracefully: it uses the
last cached events; with no cache at all it allows trading only when
``news.fail_closed`` is false (default), and always logs the degraded state.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

import aiohttp

from apex_smc_bot.config import NewsConfig
from apex_smc_bot.data.candle_manager import to_utc, utcnow

logger = logging.getLogger("apex.risk.news")

HIGH = "High"


@dataclass(frozen=True)
class NewsEvent:
    title: str
    currency: str
    impact: str
    timestamp: datetime
    forecast: str = ""
    previous: str = ""

    @property
    def is_high_impact(self) -> bool:
        return self.impact.lower() == "high"

    @property
    def is_usd(self) -> bool:
        return self.currency.upper() in {"USD", "XAU", "XAUUSD", "GOLD"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "currency": self.currency,
            "impact": self.impact,
            "date": self.timestamp.isoformat(),
            "forecast": self.forecast,
            "previous": self.previous,
        }


@dataclass
class NewsCheck:
    safe: bool
    events: list[NewsEvent] = field(default_factory=list)
    next_event: NewsEvent | None = None
    source: str = "live"
    error: str = ""
    calendar_age_minutes: float | None = None

    @property
    def blocked_titles(self) -> list[str]:
        return [f"{event.title} ({event.currency})" for event in self.events]

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe": self.safe,
            "blocking_events": [event.to_dict() for event in self.events],
            "next_event": self.next_event.to_dict() if self.next_event else None,
            "source": self.source,
            "error": self.error,
            "calendar_age_minutes": round(self.calendar_age_minutes, 1)
            if self.calendar_age_minutes is not None
            else None,
        }


def parse_calendar(payload: Any) -> list[NewsEvent]:
    """Parse the ForexFactory JSON array into UTC-aware events."""

    if not isinstance(payload, list):
        raise ValueError("Calendar payload must be a JSON array")
    events: list[NewsEvent] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_date = item.get("date") or item.get("datetime") or item.get("timestamp")
        title = str(item.get("title") or "").strip()
        if not raw_date or not title:
            continue
        try:
            timestamp = to_utc(datetime.fromisoformat(str(raw_date).replace("Z", "+00:00")))
        except ValueError:
            continue
        events.append(
            NewsEvent(
                title=title,
                currency=str(item.get("currency") or item.get("country") or "").strip().upper(),
                impact=str(item.get("impact") or "Low").strip(),
                timestamp=timestamp,
                forecast=str(item.get("forecast") or ""),
                previous=str(item.get("previous") or ""),
            )
        )
    events.sort(key=lambda event: event.timestamp)
    return events


class NewsFilter:
    """Fetches and caches the weekly calendar; evaluates event proximity."""

    def __init__(
        self,
        config: NewsConfig | None = None,
        database: Any | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.config = config or NewsConfig()
        self.database = database
        self.timeout_seconds = timeout_seconds
        self._events: list[NewsEvent] = []
        self._last_fetch: datetime | None = None
        self._last_attempt: datetime | None = None
        self._error: str = ""
        self._lock = asyncio.Lock()
        self._load_persisted()

    # ------------------------------------------------------------------ cache
    def _load_persisted(self) -> None:
        if self.database is None:
            return
        try:
            rows = self.database.get_runtime_state("news_events_cache") or []
            events = []
            for row in rows:
                try:
                    events.append(
                        NewsEvent(
                            title=row["title"],
                            currency=row.get("currency", ""),
                            impact=row.get("impact", ""),
                            timestamp=datetime.fromisoformat(row["date"]),
                            forecast=row.get("forecast", ""),
                            previous=row.get("previous", ""),
                        )
                    )
                except Exception:  # noqa: BLE001
                    continue
            if events:
                self._events = sorted(events, key=lambda event: event.timestamp)
                self._last_fetch = utcnow()
                self._last_attempt = self._last_fetch
        except Exception:  # noqa: BLE001
            pass

    def _persist(self) -> None:
        if self.database is None:
            return
        try:
            self.database.set_runtime_state(
                "news_events_cache",
                [
                    {
                        "title": event.title,
                        "currency": event.currency,
                        "impact": event.impact,
                        "date": event.timestamp.isoformat(),
                        "forecast": event.forecast,
                        "previous": event.previous,
                    }
                    for event in self._events
                ],
            )
        except Exception:  # noqa: BLE001
            pass

    def _attempt_due(self, now: datetime) -> bool:
        """True when a fetch should be attempted (success *or* failure aware).

        A failed fetch must not be retried on every 5-minute cycle: with a dead
        calendar endpoint that would add seconds of latency to each decision and
        could make the engine miss a bar close.
        """

        if self._last_fetch is not None and self._last_attempt is None:
            self._last_attempt = self._last_fetch
        anchor = self._last_attempt or self._last_fetch
        if anchor is None:
            return True
        age = (now - to_utc(anchor)).total_seconds() / 60.0
        if age >= self.config.cache_minutes:
            return True
        return not self._events and age >= 1.0

    def cache_age_minutes(self, now: datetime | None = None) -> float | None:
        if self._last_fetch is None:
            return None
        return (to_utc(now or utcnow()) - self._last_fetch).total_seconds() / 60.0

    # ------------------------------------------------------------------ fetch
    async def refresh(self, now: datetime | None = None, force: bool = False) -> list[NewsEvent]:
        """Fetch the calendar if the cache is older than ``cache_minutes``."""

        now = to_utc(now or utcnow())
        if not force and not self._attempt_due(now):
            return self._events
        async with self._lock:
            if not force and not self._attempt_due(now):
                return self._events
            self._last_attempt = now
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
                headers = {"User-Agent": "APEX-SMC/1.0 (+https://example.invalid)"}
                async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                    async with session.get(self.config.calendar_url) as response:
                        if response.status != 200:
                            raise ValueError(f"HTTP {response.status}")
                        payload = await response.json(content_type=None)
                events = parse_calendar(payload)
                self._events = events
                self._last_fetch = now
                self._error = ""
                self._persist()
                logger.info("Economic calendar refreshed: %s events", len(events))
            except Exception as exc:  # noqa: BLE001 - network failures must not crash
                self._error = str(exc)
                logger.warning("Economic calendar refresh failed: %s (using cache)", exc)
            return self._events

    def set_events(self, events: Sequence[NewsEvent | dict[str, Any]], fetched_at: datetime | None = None) -> None:
        """Inject events (tests, offline ops, manual calendar)."""

        parsed: list[NewsEvent] = []
        for event in events:
            if isinstance(event, NewsEvent):
                parsed.append(event)
            else:
                parsed.append(parse_calendar([event])[0])
        self._events = sorted(parsed, key=lambda event: event.timestamp)
        self._last_fetch = to_utc(fetched_at or utcnow())
        self._error = ""
        self._persist()

    # ------------------------------------------------------------------ logic
    def _is_relevant(self, event: NewsEvent) -> bool:
        if not event.is_usd or not event.is_high_impact:
            return False
        if any(keyword.lower() in event.title.lower() for keyword in self.config.high_impact_keywords):
            return True
        # All high-impact USD events matter on gold, even if unlisted.
        return True

    def blocking_events(
        self,
        now: datetime | None = None,
        before_minutes: int | None = None,
        after_minutes: int | None = None,
    ) -> list[NewsEvent]:
        now = to_utc(now or utcnow())
        before = before_minutes if before_minutes is not None else self.config.block_minutes_before
        after = after_minutes if after_minutes is not None else self.config.block_minutes_after
        window_start = now - timedelta(minutes=after)
        window_end = now + timedelta(minutes=before)
        return [
            event
            for event in self._events
            if window_start <= event.timestamp <= window_end and self._is_relevant(event)
        ]

    def check(self, now: datetime | None = None) -> NewsCheck:
        """Return whether new entries are allowed right now."""

        now = to_utc(now or utcnow())
        blocking = self.blocking_events(now)
        upcoming = [
            event
            for event in self._events
            if event.timestamp >= now - timedelta(minutes=self.config.block_minutes_after)
            and self._is_relevant(event)
        ]
        next_event = min(upcoming, key=lambda event: event.timestamp) if upcoming else None
        age = self.cache_age_minutes(now)
        stale = age is None or age > self.config.cache_minutes * 1.5
        error = self._error
        if stale and not self._events:
            # No calendar at all: fail-open unless explicitly configured closed.
            safe = not self.config.fail_closed
            source = "unavailable"
            if not error:
                error = "calendar not available"
        else:
            safe = not blocking
            source = "cache" if stale else "live"
        return NewsCheck(
            safe=safe,
            events=blocking,
            next_event=next_event,
            source=source,
            error=error,
            calendar_age_minutes=age,
        )

    def is_safe_to_trade(self, now: datetime | None = None) -> bool:
        return self.check(now).safe

    def upcoming(self, now: datetime | None = None, hours: float = 12.0) -> list[NewsEvent]:
        now = to_utc(now or utcnow())
        horizon = now + timedelta(hours=hours)
        return [event for event in self._events if now <= event.timestamp <= horizon and self._is_relevant(event)]

    def snapshot(self, now: datetime | None = None) -> dict[str, Any]:
        check = self.check(now)
        return {
            "events_loaded": len(self._events),
            "error": self._error,
            "age_minutes": self.cache_age_minutes(now),
            "check": check.to_dict(),
            "upcoming": [event.to_dict() for event in self.upcoming(now, hours=24)[:10]],
        }


def hardcoded_high_impact_titles() -> Iterable[str]:
    """Titles recognised as high impact when the feed omits the impact flag."""

    return (
        "FOMC",
        "Non-Farm Employment Change",
        "CPI",
        "PPI",
        "JOLTS",
        "ISM Manufacturing PMI",
        "ISM Services PMI",
        "Federal Funds Rate",
        "RBA Rate Decision",
    )
