"""Real-time spread guard with spike detection and a 15-minute pause.

XAU/USD spreads widen around rollover, news and liquidity gaps.  A sudden
doubling/tripling is treated as an institutional event: cancel pending entries
and pause for the configured window.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from apex_smc_bot.config import RiskConfig
from apex_smc_bot.data.candle_manager import to_utc, utcnow

logger = logging.getLogger("apex.risk.spread")


@dataclass
class SpreadState:
    current_price: float
    current_pips: float
    average_pips: float
    max_pips: float
    spike: bool
    allowed: bool
    paused_until: datetime | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_price": round(self.current_price, 4),
            "current_pips": round(self.current_pips, 2),
            "average_pips": round(self.average_pips, 2),
            "max_pips": round(self.max_pips, 2),
            "spike": self.spike,
            "allowed": self.allowed,
            "paused_until": self.paused_until.isoformat() if self.paused_until else None,
            "reason": self.reason,
        }


class SpreadMonitor:
    """Tracks a rolling spread baseline and rejects entries above the limit."""

    def __init__(
        self,
        config: RiskConfig | None = None,
        window: int = 30,
        spike_multiplier: float = 2.0,
        pause_minutes: int = 15,
    ) -> None:
        self.config = config or RiskConfig()
        self.window = max(5, window)
        self.spike_multiplier = spike_multiplier
        self.pause_minutes = pause_minutes
        self._history: deque[float] = deque(maxlen=self.window)
        self._paused_until: datetime | None = None
        self._last: SpreadState | None = None

    def update(self, spread_price: float, now: datetime | None = None) -> SpreadState:
        """Record a quote spread (price units) and evaluate the guard."""

        now = to_utc(now or utcnow())
        if spread_price is None or spread_price < 0:
            raise ValueError("spread must be non-negative")
        pip = self.config.pip_size
        pips = spread_price / pip
        self._history.append(pips)
        average = sum(self._history) / len(self._history)
        spike = (
            len(self._history) >= 5
            and average > 0
            and pips >= max(self.spike_multiplier * average, self.config.max_spread_pips)
        )
        if spike:
            candidate = now + timedelta(minutes=self.pause_minutes)
            if self._paused_until is None or candidate > self._paused_until:
                self._paused_until = candidate
                logger.warning(
                    "SPREAD SPIKE %.2f pips vs avg %.2f — pausing entries until %s",
                    pips,
                    average,
                    candidate.isoformat(),
                )
        paused = self._paused_until is not None and now < self._paused_until
        allowed = (pips <= self.config.max_spread_pips) and not paused
        if pips <= 0:
            allowed = False
        reason = "OK"
        if paused:
            reason = f"SPREAD_SPIKE_PAUSE until {self._paused_until.isoformat()}"
        elif pips > self.config.max_spread_pips:
            reason = f"SPREAD_TOO_HIGH ({pips:.2f} > {self.config.max_spread_pips:.2f} pips)"
        state = SpreadState(
            current_price=spread_price,
            current_pips=round(pips, 3),
            average_pips=round(average, 3),
            max_pips=self.config.max_spread_pips,
            spike=spike,
            allowed=allowed,
            paused_until=self._paused_until,
            reason=reason,
        )
        self._last = state
        return state

    def is_tradeable(self, now: datetime | None = None) -> bool:
        now = to_utc(now or utcnow())
        if self._paused_until is not None and now < self._paused_until:
            return False
        return self._last.allowed if self._last is not None else True

    @property
    def last(self) -> SpreadState | None:
        return self._last

    @property
    def normal_pips(self) -> float:
        if not self._history:
            return 0.0
        return round(sum(self._history) / len(self._history), 3)

    def clear_pause(self) -> None:
        self._paused_until = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "last": self._last.to_dict() if self._last else None,
            "normal_pips": self.normal_pips,
            "samples": len(self._history),
            "paused_until": self._paused_until.isoformat() if self._paused_until else None,
        }
