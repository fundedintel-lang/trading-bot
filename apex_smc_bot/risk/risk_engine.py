"""Portfolio-level risk gate.

Order of operations in the live engine:

    risk_engine.evaluate(...)  → allowed? effective risk %?
    signal_generator.generate_full_signal(..., risk_percent=effective)
    signal_validator.validate(signal, ...)
    execution.order_manager.submit(signal)

The risk engine is the only component allowed to reduce or veto risk.  It cannot
raise risk above the configured maximum, ever.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from apex_smc_bot.analysis.order_blocks import OrderBlock
from apex_smc_bot.analysis.session import Session
from apex_smc_bot.config import Settings
from apex_smc_bot.data.candle_manager import Candle, to_utc, utcnow
from apex_smc_bot.risk.caution_zones import CautionResult, evaluate_caution_zones
from apex_smc_bot.risk.drawdown_monitor import DrawdownDecision, DrawdownMonitor
from apex_smc_bot.risk.news_filter import NewsCheck
from apex_smc_bot.risk.spread_monitor import SpreadMonitor, SpreadState
from apex_smc_bot.signals.models import PendingSetup, Signal


@dataclass
class RiskDecision:
    allow: bool
    risk_percent: float
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    drawdown: dict[str, Any] = field(default_factory=dict)
    spread: dict[str, Any] = field(default_factory=dict)
    caution: dict[str, Any] = field(default_factory=dict)
    news: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow": self.allow,
            "risk_percent": round(self.risk_percent, 3),
            "checks": self.checks,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "drawdown": self.drawdown,
            "spread": self.spread,
            "caution": self.caution,
            "news": self.news,
        }

    @property
    def blocked_by(self) -> list[str]:
        return [name for name, passed in self.checks.items() if not passed]

    def summary(self) -> str:
        if self.allow:
            return f"RISK OK — {self.risk_percent:.2f}% per trade"
        return "RISK BLOCK — " + ", ".join(self.blocked_by or self.reasons)


class RiskEngine:
    """Owns drawdown, spread and caution checks; exposes a single decision."""

    def __init__(self, settings: Settings, database: Any | None = None) -> None:
        self.settings = settings
        self.risk = settings.risk
        self.strategy = settings.strategy
        self.drawdown = DrawdownMonitor(settings.risk, database)
        self.spread = SpreadMonitor(
            settings.risk,
            spike_multiplier=settings.risk.spread_spike_multiplier,
            pause_minutes=settings.risk.spread_pause_minutes,
        )
        self.database = database

    # ------------------------------------------------------------------ updates
    def update_equity(self, equity: float, now: datetime | None = None) -> DrawdownDecision:
        return self.drawdown.update_equity(equity, now)

    def update_spread(self, spread_price: float, now: datetime | None = None) -> SpreadState:
        return self.spread.update(spread_price, now)

    def note_trade_closed(self, pnl: float, now: datetime | None = None) -> None:
        self.drawdown.record_trade_result(pnl, now)

    def check_drawdown_circuit(self, equity: float, now: datetime | None = None) -> DrawdownDecision:
        return self.drawdown.update_equity(equity, now)

    # ------------------------------------------------------------------ evaluation
    def evaluate(
        self,
        now: datetime | None = None,
        equity: float | None = None,
        open_trade_count: int = 0,
        session: Session | None = None,
        spread_price: float = 0.0,
        news_check: NewsCheck | None = None,
        pending: PendingSetup | None = None,
        candles_15m: Sequence[Candle] | None = None,
        order_blocks: Sequence[OrderBlock] | None = None,
        market_price: float | None = None,
        extra_risk_multiplier: float = 1.0,
    ) -> RiskDecision:
        """Aggregate every portfolio and market-condition gate."""

        now = to_utc(now or utcnow())
        equity = float(equity if equity is not None else self.risk.starting_equity)
        checks: dict[str, bool] = {}
        reasons: list[str] = []
        warnings: list[str] = []

        checks["equity_sufficient"] = math.isfinite(equity) and equity > 0 and equity >= self.risk.min_equity
        if not checks["equity_sufficient"]:
            reasons.append(f"Equity ${equity:.2f} below minimum ${self.risk.min_equity:.2f}")

        drawdown_decision = self.drawdown.update_equity(equity, now)
        checks["drawdown_ok"] = drawdown_decision.allow_trading
        if not drawdown_decision.allow_trading:
            reasons.append(drawdown_decision.reason)
        elif drawdown_decision.reduced_risk:
            warnings.append(drawdown_decision.reason)

        spread_state = self.spread.update(spread_price, now)
        checks["spread_ok"] = spread_state.allowed
        if not spread_state.allowed:
            reasons.append(spread_state.reason)

        if news_check is None:
            checks["news_safe"] = True
            warnings.append("News calendar not supplied — proximity check skipped")
            news_payload: dict[str, Any] = {"safe": True, "source": "not_checked"}
        else:
            checks["news_safe"] = news_check.safe
            news_payload = news_check.to_dict()
            if not news_check.safe:
                reasons.append("NEWS_FILTER: " + ", ".join(news_check.blocked_titles))
            elif news_check.source == "unavailable":
                warnings.append(f"News calendar unavailable: {news_check.error}")

        if session is None:
            checks["session_ok"] = True
            warnings.append("Session not supplied — kill-zone check skipped")
        else:
            checks["session_ok"] = session.tradeable
            if not session.tradeable:
                reasons.append(f"Session {session.name} is not tradeable")

        checks["max_open_trades"] = open_trade_count < self.risk.max_open_trades
        if not checks["max_open_trades"]:
            reasons.append(
                f"{open_trade_count} open trades — maximum is {self.risk.max_open_trades}"
            )

        caution = CautionResult()
        if pending is not None and session is not None and candles_15m is not None:
            price = market_price if market_price is not None else pending.entry_price
            caution = evaluate_caution_zones(
                pending,
                session,
                price,
                candles_15m,
                spread_state=spread_state,
                drawdown=drawdown_decision,
                order_blocks=order_blocks,
                strategy=self.strategy,
                risk=self.risk,
                structure=pending.structure,
                now=now,
            )
            checks["caution_zones"] = not caution.abort
            if caution.abort:
                reasons.extend(caution.flags or ["CAUTION_ZONE_ABORT"])
        else:
            checks["caution_zones"] = True

        risk_percent = (
            self.risk.max_risk_per_trade_percent
            * drawdown_decision.risk_multiplier
            * (caution.risk_multiplier if not caution.abort else 0.0)
            * max(0.0, extra_risk_multiplier)
        )
        risk_percent = min(risk_percent, self.risk.max_risk_per_trade_percent)

        allow = all(checks.values())
        if not allow:
            risk_percent = 0.0

        return RiskDecision(
            allow=allow,
            risk_percent=round(risk_percent, 3),
            checks=checks,
            reasons=reasons,
            warnings=warnings,
            drawdown=drawdown_decision.to_dict(),
            spread=spread_state.to_dict(),
            caution=caution.to_dict(),
            news=news_payload,
        )

    def approve_signal(
        self,
        signal: Signal,
        now: datetime | None = None,
        equity: float | None = None,
        open_trade_count: int = 0,
        risk_percent: float | None = None,
    ) -> RiskDecision:
        """Final portfolio check of an already-sized signal."""

        now = to_utc(now or utcnow())
        equity_value = float(equity if equity is not None else signal.position_sizing.account_equity)
        drawdown_decision = self.drawdown.update_equity(equity_value, now)
        checks = {
            "drawdown_ok": drawdown_decision.allow_trading,
            "max_open_trades": open_trade_count < self.risk.max_open_trades,
            "signal_risk_within_limit": (
                signal.position_sizing.risk_percent <= self.risk.max_risk_per_trade_percent + 1e-9
            ),
            "signal_confluence": signal.confluence_score >= self.strategy.min_confluence_score,
        }
        reasons: list[str] = []
        warnings: list[str] = []
        if not checks["drawdown_ok"]:
            reasons.append(drawdown_decision.reason)
        if not checks["max_open_trades"]:
            reasons.append(
                f"{open_trade_count} open trades — maximum is {self.risk.max_open_trades}"
            )
        if not checks["signal_risk_within_limit"]:
            reasons.append("Signal risk exceeds maximum per-trade risk")
        if not checks["signal_confluence"]:
            reasons.append("Signal below minimum confluence")
        if drawdown_decision.reduced_risk:
            warnings.append(drawdown_decision.reason)
        allowed_risk = self.risk.max_risk_per_trade_percent * drawdown_decision.risk_multiplier
        if risk_percent is not None:
            allowed_risk = min(allowed_risk, risk_percent)
        if signal.position_sizing.risk_percent > allowed_risk + 1e-9:
            checks["signal_risk_matches_gate"] = False
            reasons.append(
                f"Signal sizes {signal.position_sizing.risk_percent:.2f}% but the risk gate allowed "
                f"{allowed_risk:.2f}%"
            )
        allow = all(checks.values())
        return RiskDecision(
            allow=allow,
            risk_percent=signal.position_sizing.risk_percent if allow else 0.0,
            checks=checks,
            reasons=reasons,
            warnings=warnings,
            drawdown=drawdown_decision.to_dict(),
            spread=self.spread.last.to_dict() if self.spread.last else {},
            caution={},
            news={},
        )

    # ------------------------------------------------------------------ status
    def status(self, equity: float | None = None, now: datetime | None = None) -> dict[str, Any]:
        now = to_utc(now or utcnow())
        if equity is not None:
            self.update_equity(equity, now)
        return {
            "drawdown": self.drawdown.snapshot(),
            "spread": self.spread.snapshot(),
            "limits": {
                "max_risk_per_trade_percent": self.risk.max_risk_per_trade_percent,
                "max_daily_loss_percent": self.risk.max_daily_loss_percent,
                "max_open_trades": self.risk.max_open_trades,
                "max_spread_pips": self.risk.max_spread_pips,
                "max_sl_pips": self.risk.max_sl_pips,
            },
        }
