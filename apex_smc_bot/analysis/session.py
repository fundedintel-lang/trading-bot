"""ICT kill-zone session engine (all times UTC, DST-proof by construction).

The engine decides *when* a setup is allowed to execute.  Prices are only one
half of an ICT trade; time is the other.

Session map (UTC):
    00:00–02:00  ASIA_OPEN        monitor only
    02:00–07:00  ASIA_SESSION     avoid (chop / fake moves)
    07:00–10:00  LONDON_KILL_ZONE prime (Judas window 07:00–07:30)
    10:00–12:00  LONDON_SESSION   continuation, medium priority
    12:00–12:30  NY_JUDAS_WINDOW  wait for the fake move to complete
    12:30–13:00  NY_KILL_ZONE     prime, lunch caution flag
    13:00–15:00  NY_KILL_ZONE     prime
    15:00–17:00  NY_AFTERNOON     caution, low priority
    17:00–20:00  NY_CLOSE         reversal scalps, medium priority
    20:00–24:00  DEAD_ZONE        no institutional activity
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Sequence

from apex_smc_bot.analysis.structure import BEARISH, BULLISH
from apex_smc_bot.config import SessionConfig
from apex_smc_bot.data.candle_manager import Candle, to_utc

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"
NONE = "NONE"

LONDON_KILL_ZONE = "LONDON_KILL_ZONE"
LONDON_SESSION = "LONDON_SESSION"
NY_JUDAS_WINDOW = "NY_JUDAS_WINDOW"
NY_KILL_ZONE = "NY_KILL_ZONE"
NY_AFTERNOON = "NY_AFTERNOON"
NY_CLOSE = "NY_CLOSE"
ASIA_OPEN = "ASIA_OPEN"
ASIA_SESSION = "ASIA_SESSION"
DEAD_ZONE = "DEAD_ZONE"
WEEKEND = "WEEKEND"


@dataclass(frozen=True, slots=True)
class Session:
    """A tradeable (or forbidden) time window."""

    name: str
    start: time
    end: time
    tradeable: bool
    priority: str
    is_kill_zone: bool = False
    watch_judas: bool = False
    judas_window_end: time | None = None
    caution: bool = False
    note: str = ""

    def contains(self, moment: datetime) -> bool:
        current = to_utc(moment).time()
        return self.start <= current < self.end

    @property
    def minutes(self) -> int:
        start_minutes = self.start.hour * 60 + self.start.minute
        end_minutes = self.end.hour * 60 + self.end.minute
        return max(0, end_minutes - start_minutes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start.strftime("%H:%M"),
            "end": self.end.strftime("%H:%M"),
            "tradeable": self.tradeable,
            "priority": self.priority,
            "is_kill_zone": self.is_kill_zone,
            "caution": self.caution,
            "watch_judas": self.watch_judas,
            "note": self.note,
        }


def _at(moment: datetime, hour: int, minute: int = 0) -> time:
    return time(hour=hour, minute=minute)


class SessionEngine:
    """Maps a UTC timestamp to a :class:`Session` and tracks kill-zone timing."""

    def __init__(self, config: SessionConfig | None = None) -> None:
        self.config = config or SessionConfig()

    # ------------------------------------------------------------------ sessions
    def get_current_session(self, moment: datetime | None = None) -> Session:
        now = to_utc(moment or datetime.now(timezone.utc))
        weekday = now.weekday()
        current_time = now.time()

        # Weekend closure (gold CFD: Fri 21:00 → Sun 22:00 UTC).
        if weekday == 5 or (weekday == 6 and current_time < time(22, 0)) or (
            weekday == 4 and current_time >= time(21, 0)
        ):
            return Session(
                WEEKEND,
                time(0, 0),
                time(23, 59),
                tradeable=False,
                priority=NONE,
                is_kill_zone=False,
                note="Market closed — no execution",
            )

        cfg = self.config
        london_start, london_end = cfg.london_kill_zone_start, cfg.london_kill_zone_end
        if london_start <= current_time < london_end:
            judas_end = _add_minutes(london_start, cfg.judas_window_minutes)
            return Session(
                LONDON_KILL_ZONE,
                london_start,
                london_end,
                tradeable=True,
                priority=HIGH,
                is_kill_zone=True,
                watch_judas=True,
                judas_window_end=judas_end,
                note="London kill zone — Judas swing window 07:00-07:30 UTC",
            )
        if london_end <= current_time < cfg.ny_kill_zone_start:
            return Session(
                LONDON_SESSION,
                london_end,
                cfg.ny_kill_zone_start,
                tradeable=True,
                priority=MEDIUM,
                caution=True,
                note="London continuation — requires strong confluence",
            )
        ny_start = cfg.ny_kill_zone_start
        ny_judas_end = _add_minutes(ny_start, cfg.judas_window_minutes)
        if ny_start <= current_time < ny_judas_end:
            return Session(
                NY_JUDAS_WINDOW,
                ny_start,
                ny_judas_end,
                tradeable=False,
                priority=NONE,
                is_kill_zone=True,
                watch_judas=True,
                judas_window_end=ny_judas_end,
                note="NY open Judas window — wait for the fake move to resolve",
            )
        if ny_judas_end <= current_time < cfg.ny_kill_zone_end:
            lunch_caution = current_time < time(13, 0)
            return Session(
                NY_KILL_ZONE,
                ny_judas_end,
                cfg.ny_kill_zone_end,
                tradeable=True,
                priority=HIGH,
                is_kill_zone=True,
                caution=lunch_caution,
                note="NY kill zone — post-Judas, highest gold volatility"
                if not lunch_caution
                else "NY kill zone — lunch caution until 13:00 UTC",
            )
        if cfg.ny_kill_zone_end <= current_time < cfg.ny_close_start:
            return Session(
                NY_AFTERNOON,
                cfg.ny_kill_zone_end,
                cfg.ny_close_start,
                tradeable=True,
                priority=LOW,
                caution=True,
                note="NY afternoon — late moves, reduced quality",
            )
        if cfg.ny_close_start <= current_time < cfg.ny_close_end:
            return Session(
                NY_CLOSE,
                cfg.ny_close_start,
                cfg.ny_close_end,
                tradeable=True,
                priority=MEDIUM,
                note="NY close — reversal / end-of-day positioning scalps",
            )
        if time(0, 0) <= current_time < time(2, 0):
            return Session(
                ASIA_OPEN,
                time(0, 0),
                time(2, 0),
                tradeable=False,
                priority=NONE,
                note="Asia open — monitoring only, low liquidity",
            )
        if time(2, 0) <= current_time < london_start:
            return Session(
                ASIA_SESSION,
                time(2, 0),
                london_start,
                tradeable=False,
                priority=NONE,
                note="Asia session — choppy, engineered fake moves",
            )
        return Session(
            DEAD_ZONE,
            cfg.ny_close_end,
            time(23, 59, 59),
            tradeable=False,
            priority=NONE,
            note="Dead zone — no institutional activity",
        )

    def is_tradeable(self, moment: datetime | None = None) -> bool:
        """True inside an approved tradeable window (subject to priority rules)."""

        session = self.get_current_session(moment)
        if session.tradeable:
            return True
        return self.config.allow_outside_kill_zone and session.name not in {WEEKEND, DEAD_ZONE}

    def session_priority(self, moment: datetime | None = None) -> str:
        return self.get_current_session(moment).priority

    def time_until_session_end(self, moment: datetime | None = None) -> timedelta:
        now = to_utc(moment or datetime.now(timezone.utc))
        session = self.get_current_session(now)
        end = datetime.combine(now.date(), session.end, tzinfo=timezone.utc)
        if end <= now:
            end += timedelta(days=1)
        return end - now

    def time_until_next_kill_zone(self, moment: datetime | None = None) -> timedelta:
        now = to_utc(moment or datetime.now(timezone.utc))
        for offset in range(0, 8 * 24 * 60, 5):  # 5-minute resolution over a week
            probe = now + timedelta(minutes=offset)
            session = self.get_current_session(probe)
            if session.is_kill_zone and session.tradeable:
                return probe - now
        return timedelta(days=7)

    def judas_session_start(self, moment: datetime | None = None) -> datetime | None:
        """Return today's opening time for the Judas window of the active session."""

        now = to_utc(moment or datetime.now(timezone.utc))
        session = self.get_current_session(now)
        if not session.is_kill_zone:
            return None
        if session.name == LONDON_KILL_ZONE:
            return datetime.combine(now.date(), self.config.london_kill_zone_start, tzinfo=timezone.utc)
        if session.name in {NY_JUDAS_WINDOW, NY_KILL_ZONE}:
            return datetime.combine(now.date(), self.config.ny_kill_zone_start, tzinfo=timezone.utc)
        return None

    def describe(self, moment: datetime | None = None) -> dict[str, Any]:
        now = to_utc(moment or datetime.now(timezone.utc))
        session = self.get_current_session(now)
        return {
            "name": session.name,
            "time_utc": now.strftime("%H:%M"),
            "tradeable": session.tradeable,
            "priority": session.priority,
            "caution": session.caution,
            "is_kill_zone": session.is_kill_zone,
            "note": session.note,
            "time_until_end_minutes": round(self.time_until_session_end(now).total_seconds() / 60),
        }


def _add_minutes(moment: time, minutes: int) -> time:
    anchor = datetime.combine(datetime(2000, 1, 1).date(), moment)
    shifted = anchor + timedelta(minutes=minutes)
    return shifted.time()


# --------------------------------------------------------------------------- Judas
@dataclass
class JudasSignal:
    """A detected opening fake-out for the session."""

    direction: str  # expected reversal direction (BULLISH / BEARISH)
    judas_level: float
    session: str
    confidence: str  # HIGH | MEDIUM | LOW
    resolved: bool = False
    sweep_index: int | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "judas_level": round(self.judas_level, 5),
            "session": self.session,
            "confidence": self.confidence,
            "resolved": self.resolved,
            "detail": self.detail,
        }


def _aggregate_first_window(
    candles: Sequence[Candle], session_start: datetime, window_minutes: int
) -> Candle | None:
    window_end = session_start + timedelta(minutes=window_minutes)
    window = [c for c in candles if session_start <= c.timestamp < window_end]
    if not window:
        return None
    return Candle(
        timestamp=window[0].timestamp,
        open=window[0].open,
        high=max(c.high for c in window),
        low=min(c.low for c in window),
        close=window[-1].close,
        volume=sum(c.volume for c in window),
        spread=window[-1].spread,
    )


def detect_judas_swing(
    candles: Sequence[Candle],
    session: Session,
    session_start: datetime,
    window_minutes: int = 30,
) -> JudasSignal | None:
    """Detect the opening Judas swing and whether it has been resolved.

    The first ``window_minutes`` of a kill zone often move against the true
    direction to raid liquidity.  A bearish first leg implies a bullish reversal
    (and vice versa).  The signal is *resolved* once price sweeps the Judas
    extreme and closes back inside the opening range.
    """

    if not session.is_kill_zone:
        return None
    session_start = to_utc(session_start)
    first = _aggregate_first_window(candles, session_start, window_minutes)
    if first is None or first.range <= 0:
        return None
    if first.close < first.open:
        direction = BULLISH
        level = first.low
        confidence = "HIGH" if first.body_ratio > 0.5 else "MEDIUM"
        detail = "Bearish opening drive — expect bullish reversal after SSL raid"
    elif first.close > first.open:
        direction = BEARISH
        level = first.high
        confidence = "HIGH" if first.body_ratio > 0.5 else "MEDIUM"
        detail = "Bullish opening drive — expect bearish reversal after BSL raid"
    else:
        return None

    window_end = session_start + timedelta(minutes=window_minutes)
    resolved = False
    sweep_index: int | None = None
    for index, candle in enumerate(candles):
        if candle.timestamp < window_end:
            continue
        if direction == BULLISH:
            if candle.low < level and candle.close > level:
                resolved = True
                sweep_index = index
                break
        else:
            if candle.high > level and candle.close < level:
                resolved = True
                sweep_index = index
                break

    return JudasSignal(
        direction=direction,
        judas_level=level,
        session=session.name,
        confidence=confidence,
        resolved=resolved,
        sweep_index=sweep_index,
        detail=detail,
    )


def judas_swing_detector(
    candles_30min: Sequence[Candle], session: Session, pip_size: float = 0.10
) -> JudasSignal | None:
    """Spec-compatible wrapper operating on pre-aggregated session candles."""

    if not session.watch_judas or not candles_30min:
        return None
    first = candles_30min[0]
    start = first.timestamp.replace(
        minute=(first.timestamp.minute // 30) * 30, second=0, microsecond=0
    )
    return detect_judas_swing(candles_30min, session, start, window_minutes=30)


def get_current_session(moment: datetime | None = None) -> Session:
    """Module-level convenience used by scripts and tests."""

    return SessionEngine().get_current_session(moment)
