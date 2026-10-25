"""Daily drawdown tracking, loss cooldowns and the 5% circuit breaker.

State is persisted through the database runtime state so a restart cannot reset
a circuit breaker or erase a loss streak.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

from apex_smc_bot.config import RiskConfig
from apex_smc_bot.data.candle_manager import to_utc, utcnow

logger = logging.getLogger("apex.risk.drawdown")

STATE_KEY = "drawdown_state"


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return to_utc(value)
    try:
        return to_utc(datetime.fromisoformat(str(value)))
    except ValueError:
        return None


@dataclass
class DrawdownState:
    day: str = ""
    daily_start_equity: float = 0.0
    current_equity: float = 0.0
    peak_equity: float = 0.0
    daily_pnl: float = 0.0
    daily_drawdown_percent: float = 0.0
    halted_until: datetime | None = None
    halt_reason: str = ""
    consecutive_losses: int = 0
    cooldown_until: datetime | None = None
    last_loss_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("halted_until", "cooldown_until", "last_loss_at", "updated_at"):
            value = data.get(key)
            data[key] = value.isoformat() if isinstance(value, datetime) else value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DrawdownState":
        data = data or {}
        return cls(
            day=str(data.get("day", "")),
            daily_start_equity=float(data.get("daily_start_equity", 0.0) or 0.0),
            current_equity=float(data.get("current_equity", 0.0) or 0.0),
            peak_equity=float(data.get("peak_equity", 0.0) or 0.0),
            daily_pnl=float(data.get("daily_pnl", 0.0) or 0.0),
            daily_drawdown_percent=float(data.get("daily_drawdown_percent", 0.0) or 0.0),
            halted_until=_parse_dt(data.get("halted_until")),
            halt_reason=str(data.get("halt_reason", "")),
            consecutive_losses=int(data.get("consecutive_losses", 0) or 0),
            cooldown_until=_parse_dt(data.get("cooldown_until")),
            last_loss_at=_parse_dt(data.get("last_loss_at")),
            updated_at=_parse_dt(data.get("updated_at")),
        )


@dataclass
class DrawdownDecision:
    allow_trading: bool
    risk_multiplier: float
    reason: str
    daily_drawdown_percent: float
    daily_pnl_percent: float
    halted_until: datetime | None = None
    cooldown_until: datetime | None = None
    breaker_triggered: bool = False
    reduced_risk: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_trading": self.allow_trading,
            "risk_multiplier": self.risk_multiplier,
            "reason": self.reason,
            "daily_drawdown_percent": round(self.daily_drawdown_percent, 2),
            "daily_pnl_percent": round(self.daily_pnl_percent, 2),
            "halted_until": self.halted_until.isoformat() if self.halted_until else None,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "breaker_triggered": self.breaker_triggered,
            "reduced_risk": self.reduced_risk,
        }


class DrawdownMonitor:
    """Tracks equity, enforces the daily loss limit and manages cool-downs."""

    def __init__(
        self,
        config: RiskConfig | None = None,
        database: Any | None = None,
        state: DrawdownState | None = None,
    ) -> None:
        self.config = config or RiskConfig()
        self.database = database
        self.state = state or self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> DrawdownState:
        if self.database is None:
            return DrawdownState()
        try:
            raw = self.database.get_runtime_state(STATE_KEY)
            return DrawdownState.from_dict(raw)
        except Exception:  # noqa: BLE001
            return DrawdownState()

    def save(self) -> None:
        if self.database is None:
            return
        try:
            self.state.updated_at = utcnow()
            self.database.set_runtime_state(STATE_KEY, self.state.to_dict())
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ equity
    def roll_day(self, now: datetime, equity: float) -> None:
        """Anchor the daily baseline at the first observation of a UTC day."""

        day = to_utc(now).date().isoformat()
        if self.state.day != day:
            self.state.day = day
            self.state.daily_start_equity = float(equity)
            self.state.current_equity = float(equity)
            self.state.daily_pnl = 0.0
            self.state.daily_drawdown_percent = 0.0
            self.save()

    def update_equity(self, equity: float, now: datetime | None = None) -> DrawdownDecision:
        """Update equity, potentially trip the 24H circuit breaker."""

        now = to_utc(now or utcnow())
        if equity <= 0:
            raise ValueError("equity must be positive")
        self.roll_day(now, equity)
        if self.state.daily_start_equity <= 0:
            self.state.daily_start_equity = float(equity)
        self.state.current_equity = float(equity)
        self.state.peak_equity = max(self.state.peak_equity, float(equity))
        self.state.daily_pnl = round(equity - self.state.daily_start_equity, 2)
        self.state.daily_drawdown_percent = round(
            max(0.0, (self.state.daily_start_equity - equity) / self.state.daily_start_equity * 100.0),
            4,
        )

        breaker = False
        if self.state.daily_drawdown_percent >= self.config.max_daily_loss_percent:
            halt_until = now + timedelta(hours=self.config.daily_halt_hours)
            if self.state.halted_until is None or self.state.halted_until < halt_until:
                self.state.halted_until = halt_until
                self.state.halt_reason = (
                    f"Daily drawdown {self.state.daily_drawdown_percent:.2f}% "
                    f">= {self.config.max_daily_loss_percent:.2f}%"
                )
                breaker = True
                logger.error(
                    "CIRCUIT BREAKER: %s — trading halted until %s",
                    self.state.halt_reason,
                    halt_until.isoformat(),
                )
        self.save()
        decision = self.decision(now)
        decision.breaker_triggered = breaker
        return decision

    # ------------------------------------------------------------------ decisions
    def decision(self, now: datetime | None = None) -> DrawdownDecision:
        now = to_utc(now or utcnow())
        if (
            self.state.halted_until is not None
            and now < self.state.halted_until
        ):
            return DrawdownDecision(
                allow_trading=False,
                risk_multiplier=0.0,
                reason=f"CIRCUIT_BREAKER active until {self.state.halted_until.isoformat()}",
                daily_drawdown_percent=self.state.daily_drawdown_percent,
                daily_pnl_percent=self.daily_pnl_percent(),
                halted_until=self.state.halted_until,
                cooldown_until=self.state.cooldown_until,
            )
        if self.state.cooldown_until is not None and now < self.state.cooldown_until:
            return DrawdownDecision(
                allow_trading=False,
                risk_multiplier=0.0,
                reason=(
                    f"LOSS_COOLDOWN active until {self.state.cooldown_until.isoformat()} "
                    f"({self.state.consecutive_losses} consecutive loss(es))"
                ),
                daily_drawdown_percent=self.state.daily_drawdown_percent,
                daily_pnl_percent=self.daily_pnl_percent(),
                cooldown_until=self.state.cooldown_until,
            )
        if self.state.daily_drawdown_percent >= self.config.reduce_risk_threshold_percent:
            multiplier = round(
                self.config.reduced_risk_percent / self.config.max_risk_per_trade_percent, 3
            )
            return DrawdownDecision(
                allow_trading=True,
                risk_multiplier=multiplier,
                reason=(
                    f"Drawdown {self.state.daily_drawdown_percent:.2f}% — risk reduced to "
                    f"{self.config.reduced_risk_percent}% per trade (half size)"
                ),
                daily_drawdown_percent=self.state.daily_drawdown_percent,
                daily_pnl_percent=self.daily_pnl_percent(),
                reduced_risk=True,
            )
        return DrawdownDecision(
            allow_trading=True,
            risk_multiplier=1.0,
            reason="OK",
            daily_drawdown_percent=self.state.daily_drawdown_percent,
            daily_pnl_percent=self.daily_pnl_percent(),
        )

    def daily_pnl_percent(self) -> float:
        if self.state.daily_start_equity <= 0:
            return 0.0
        return round(self.state.daily_pnl / self.state.daily_start_equity * 100.0, 4)

    # ------------------------------------------------------------------ trades
    def record_trade_result(self, pnl: float, now: datetime | None = None) -> None:
        """Update the loss streak and arm the mandatory cool-down."""

        now = to_utc(now or utcnow())
        if pnl < 0:
            self.state.consecutive_losses += 1
            self.state.last_loss_at = now
            hours = (
                self.config.cooldown_after_two_losses_hours
                if self.state.consecutive_losses >= self.config.max_consecutive_losses
                else self.config.cooldown_after_loss_hours
            )
            self.state.cooldown_until = now + timedelta(hours=hours)
            logger.warning(
                "Loss recorded (%s consecutive). Trading paused until %s",
                self.state.consecutive_losses,
                self.state.cooldown_until.isoformat(),
            )
        else:
            self.state.consecutive_losses = 0
            self.state.cooldown_until = None
        self.save()

    def reset_cooldown(self) -> None:
        self.state.cooldown_until = None
        self.state.consecutive_losses = 0
        self.save()

    def reset_halt(self) -> None:
        self.state.halted_until = None
        self.state.halt_reason = ""
        self.save()

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "decision": self.decision().to_dict(),
            "config": {
                "max_daily_loss_percent": self.config.max_daily_loss_percent,
                "reduce_risk_threshold_percent": self.config.reduce_risk_threshold_percent,
                "max_consecutive_losses": self.config.max_consecutive_losses,
            },
        }
