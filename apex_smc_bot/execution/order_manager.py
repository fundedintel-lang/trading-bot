"""Order lifecycle orchestration: safety checks → approval → broker → audit.

Semi-auto flow
--------------
``submit()`` stores the signal in an in-memory approval queue and returns
``PENDING_APPROVAL``.  A human approves via the Telegram command listener or
``main.py approve <signal_id>``; ``approve()`` then routes the order.

Pending approvals intentionally do not survive a restart: replaying a stale gold
entry after downtime is more dangerous than losing a signal.  They expire with
the signal ``valid_until`` timestamp.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable

from apex_smc_bot.config import Settings
from apex_smc_bot.data.candle_manager import to_utc, utcnow
from apex_smc_bot.execution.broker import (
    AccountSnapshot,
    Broker,
    BrokerError,
    BrokerOutcomeUnknown,
    OrderRequest,
    OrderResult,
    Position,
)
from apex_smc_bot.signals.models import Signal
from apex_smc_bot.signals.signal_validator import SignalValidator
from apex_smc_bot.risk.risk_engine import RiskEngine

logger = logging.getLogger("apex.execution.orders")


@dataclass
class PendingApproval:
    signal: Signal
    created_at: datetime
    expires_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal.to_dict(),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }


@dataclass
class OrderManager:
    """Routes validated signals to the broker with hard safety guards."""

    settings: Settings
    broker: Broker
    database: Any | None = None
    pending_approvals: dict[str, PendingApproval] = field(default_factory=dict)
    executed_signals: set[str] = field(default_factory=set)
    risk_engine: RiskEngine | None = None
    quote_provider: Callable[[], Awaitable[tuple[float, float]]] | None = None
    execution_guard: Callable[[Signal, float, float, datetime], Awaitable[str]] | None = None
    review_guard: Callable[[Signal, float, float, datetime], Awaitable[str]] | None = None
    final_guard: Callable[[Signal, datetime], str] | None = None
    clock: Callable[[], datetime] | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.risk_engine = self.risk_engine or RiskEngine(self.settings, self.database)
        if self.database is not None:
            self.executed_signals.update(self.database.get_runtime_state("routed_signals", []) or [])

    def _consume(self, signal_id: str) -> None:
        self.executed_signals.add(signal_id)
        if self.database is not None:
            self.database.set_runtime_state("routed_signals", sorted(self.executed_signals)[-2000:])

    @staticmethod
    def _rejected(signal: Signal, message: str) -> OrderResult:
        return OrderResult(False, "REJECTED", message=message, signal_id=signal.signal_id)


    # ------------------------------------------------------------------ helpers
    @property
    def is_live_money(self) -> bool:
        return self.broker.name == "oanda" and self.settings.execution.oanda_environment == "live"

    def _auto_execute(self) -> bool:
        if self.settings.bot.execution_mode == "FULL_AUTO":
            return True
        if self.broker.name == "paper" and self.settings.execution.paper_auto_approve:
            return True
        return False

    def _build_request(self, signal: Signal, now: datetime | None = None) -> OrderRequest:
        expires = to_utc(now or signal.timestamp) + timedelta(minutes=self.settings.execution.pending_order_ttl_minutes)
        if signal.expires_at is not None:
            expires = min(expires, to_utc(signal.expires_at))
        return OrderRequest(
            signal_id=signal.signal_id,
            instrument=signal.instrument,
            direction=signal.direction,
            units=signal.position_sizing.units,
            entry_price=signal.entry.price,
            stop_loss=signal.stop_loss.price,
            take_profit=(signal.take_profits.tp3 or signal.take_profits.tp2).price,
            order_type=signal.entry.type.upper(),
            expires_at=expires,
            metadata={
                "tp1": signal.take_profits.tp1.price,
                "tp2": signal.take_profits.tp2.price,
                "tp3": signal.take_profits.tp3.price if signal.take_profits.tp3 else None,
                "lots": signal.position_sizing.lot_size,
                "confluence": signal.confluence_score,
                "confidence": signal.confidence_percent,
                "risk_amount": signal.risk_amount,
                "max_slippage_price": self.settings.risk.max_slippage_pips * self.settings.risk.pip_size,
            },
        )

    def _validate_execution_price(self, signal: Signal, market_price: float) -> str | None:
        if not math.isfinite(market_price) or market_price <= 0:
            return "INVALID_MARKET_PRICE"
        pip = self.settings.risk.pip_size
        if signal.is_long and market_price <= signal.stop_loss.price:
            return "PRICE_BEYOND_STOP"
        if not signal.is_long and market_price >= signal.stop_loss.price:
            return "PRICE_BEYOND_STOP"
        if signal.entry.type.upper() == "MARKET":
            slippage_pips = abs(market_price - signal.entry.price) / pip
            if slippage_pips > self.settings.risk.max_slippage_pips:
                return (
                    f"SLIPPAGE_GUARD: {slippage_pips:.1f} pips > "
                    f"{self.settings.risk.max_slippage_pips:.1f}"
                )
        return None

    async def _safety_check(
        self, signal: Signal, market_price: float, spread_price: float, now: datetime,
    ) -> str:
        """Re-run at the point of routing, including after a delayed approval."""

        if not math.isfinite(spread_price) or not 0 < spread_price <= self.settings.risk.max_spread_pips * self.settings.risk.pip_size:
            return "SPREAD_GUARD"
        if self.is_live_money and not (self.settings.execution.allow_live_trading and self.settings.live_trading_confirmed):
            return "LIVE_TRADING_INTERLOCK"
        account = await self.broker.get_account()  # failure must not use cached equity
        if self.broker.name == "paper":
            account.equity = self.broker.equity(market_price, spread_price=spread_price)
        if not math.isfinite(account.equity) or account.equity < self.settings.risk.min_equity or account.equity <= 0:
            return "ACCOUNT_EQUITY_UNAVAILABLE"
        if account.currency != "USD":
            return "ACCOUNT_CURRENCY_UNSUPPORTED: USD sizing requires a USD account"
        count = await self.get_open_trade_count()
        validation = SignalValidator(self.settings).validate(
            signal, market_price=market_price, now=now, open_trade_count=count, account_equity=account.equity
        )
        if not validation.valid:
            return validation.summary()
        assert self.risk_engine is not None
        spread = self.risk_engine.update_spread(spread_price, now)
        if not spread.allowed:
            return spread.reason
        risk = self.risk_engine.approve_signal(signal, now, account.equity, count)
        if not risk.allow:
            return risk.summary()
        # Reserve for an adverse market fill and exit slippage. Never route a
        # larger monetary exposure just because risk_percent metadata is small.
        size = signal.position_sizing
        if signal.entry.type == "MARKET":
            adverse = spread_price / 2 + self.settings.risk.max_slippage_pips * self.settings.risk.pip_size
            fill = market_price + adverse if signal.is_long else market_price - adverse
            amount = size.units * (abs(fill - signal.stop_loss.price) +
                                   self.settings.risk.max_slippage_pips * self.settings.risk.pip_size)
            if amount > account.equity * size.risk_percent / 100 + 1e-6:
                return "FILL_RISK_EXCEEDS_BUDGET"
        if self.execution_guard is not None:
            return await self.execution_guard(signal, market_price, spread_price, now)
        return ""

    async def _execute(
        self, signal: Signal, market_price: float, spread_price: float, now: datetime,
    ) -> OrderResult:
        if signal.signal_id in self.executed_signals:
            return self._rejected(signal, "signal already executed or awaiting reconciliation")
        try:
            if self.quote_provider is not None:
                market_price, spread_price = await self.quote_provider()
            now = to_utc(self.clock() if self.clock is not None else now)
            if self.review_guard is not None:
                problem = await self.review_guard(signal, market_price, spread_price, now)
                if problem:
                    self._mark_signal(signal.signal_id, "REJECTED")
                    return self._rejected(signal, problem)
                # AI/network latency is not permission to route stale prices.
                if self.quote_provider is not None:
                    market_price, spread_price = await self.quote_provider()
                now = to_utc(self.clock() if self.clock is not None else now)
            problem = self._validate_execution_price(signal, market_price)
            problem = problem or await self._safety_check(signal, market_price, spread_price, now)
            if problem:
                self._mark_signal(signal.signal_id, "REJECTED")
                return self._rejected(signal, problem)
            now = to_utc(self.clock() if self.clock is not None else now)
            if signal.expires_at is not None and to_utc(signal.expires_at) <= now:
                return self._rejected(signal, "SIGNAL_EXPIRED_DURING_PREFLIGHT")
            if self.final_guard is not None:
                problem = self.final_guard(signal, now)
                if problem:
                    return self._rejected(signal, problem)
            request = self._build_request(signal, now)
            result = await self.broker.submit_order(request, market_price, spread_price)
        except BrokerOutcomeUnknown:
            self._consume(signal.signal_id)  # a timeout does NOT mean no fill
            self._mark_signal(signal.signal_id, "UNKNOWN")
            return OrderResult(False, "UNKNOWN", signal_id=signal.signal_id,
                               message="Broker outcome unknown; reconcile broker before resuming")
        except Exception:  # fail closed; no provider bodies or credentials in the error
            self._mark_signal(signal.signal_id, "REJECTED")
            logger.warning("Execution/preflight unavailable for %s", signal.signal_id)
            return self._rejected(signal, "EXECUTION_PREFLIGHT_OR_BROKER_UNAVAILABLE")
        if result.status == "UNKNOWN":
            self._consume(signal.signal_id)
            self._mark_signal(signal.signal_id, "UNKNOWN")
            return result
        if result.success:
            self._consume(signal.signal_id)
            self._mark_signal(signal.signal_id, "EXECUTED" if result.status == "FILLED" else "PENDING")
            logger.info("ORDER %s %s %s units (%s)", result.status, signal.side,
                        signal.position_sizing.units, signal.signal_id)
        else:
            self._mark_signal(signal.signal_id, "REJECTED")
        return result

    def _mark_signal(self, signal_id: str, status: str) -> None:
        if self.database is not None:
            try:
                self.database.update_signal_status(signal_id, status)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ public
    async def submit(
        self, signal: Signal, market_price: float, spread_price: float = 0.0,
        now: datetime | None = None,
    ) -> OrderResult:
        """Serialize capacity reservation and broker writes (including approval)."""

        async with self._lock:
            now = to_utc(now or utcnow())
            self.expire_pending(now)
            if signal.signal_id in self.executed_signals or signal.signal_id in self.pending_approvals:
                return self._rejected(signal, "duplicate signal submission")
            if signal.expires_at is not None and to_utc(signal.expires_at) <= now:
                return self._rejected(signal, "signal expired before submission")
            if self._auto_execute():
                return await self._execute(signal, market_price, spread_price, now)
            try:
                problem = await self._safety_check(signal, market_price, spread_price, now)
            except Exception:
                problem = "EXECUTION_PREFLIGHT_UNAVAILABLE"
            if problem:
                return self._rejected(signal, problem)
            ttl = now + timedelta(minutes=self.settings.execution.pending_order_ttl_minutes)
            expires = min(ttl, to_utc(signal.expires_at)) if signal.expires_at else ttl
            self.pending_approvals[signal.signal_id] = PendingApproval(signal, now, expires)
            self._mark_signal(signal.signal_id, "PENDING_APPROVAL")
            return OrderResult(True, "PENDING_APPROVAL", signal_id=signal.signal_id, message="awaiting human approval")

    async def approve(
        self, signal_id: str, market_price: float, spread_price: float = 0.0,
        now: datetime | None = None,
    ) -> OrderResult:
        async with self._lock:
            now = to_utc(now or utcnow())
            self.expire_pending(now)
            approval = self.pending_approvals.pop(signal_id, None)
            if approval is None:
                return OrderResult(False, "REJECTED", message="no pending approval with that id", signal_id=signal_id)
            return await self._execute(approval.signal, market_price, spread_price, now)

    def reject(self, signal_id: str) -> bool:
        approval = self.pending_approvals.pop(signal_id, None)
        if approval is None:
            return False
        self._mark_signal(signal_id, "REJECTED")
        return True

    def expire_pending(self, now: datetime | None = None) -> list[str]:
        now = to_utc(now or utcnow())
        expired = [
            signal_id
            for signal_id, approval in self.pending_approvals.items()
            if to_utc(approval.expires_at) <= now
        ]
        for signal_id in expired:
            self.pending_approvals.pop(signal_id, None)
            self._mark_signal(signal_id, "EXPIRED")
        return expired

    def get_pending_approvals(self) -> list[dict[str, Any]]:
        return [approval.to_dict() for approval in self.pending_approvals.values()]

    # ------------------------------------------------------------------ broker proxy
    async def modify_stop(self, trade_id: str, stop_loss: float) -> OrderResult:
        return await self.broker.modify_stop(trade_id, stop_loss)

    async def partial_close(self, trade_id: str, units: float, price: float | None = None) -> OrderResult:
        return await self.broker.partial_close(trade_id, units, price)

    async def close_position(self, trade_id: str, price: float | None = None, reason: str = "") -> OrderResult:
        return await self.broker.close_position(trade_id, price, reason)

    async def close_all(self, reason: str = "CIRCUIT_BREAKER") -> list[OrderResult]:
        return await self.broker.close_all(reason=reason) if hasattr(self.broker, "close_all") else []

    async def get_positions(self) -> list[Position]:
        return await self.broker.get_positions()

    async def get_account(self) -> AccountSnapshot:
        return await self.broker.get_account()

    async def _working_entries(self) -> dict[str, Any]:
        getter = getattr(self.broker, "get_pending_orders", None)
        if not callable(getter):
            if self.broker.name != "paper":
                raise BrokerError("Cannot verify broker pending exposure")
            return {}
        pending = getter()
        if inspect.isawaitable(pending):
            pending = await pending
        if isinstance(pending, Mapping):
            return dict(pending)
        # OANDA also returns stop-loss/take-profit orders; those are protection,
        # NOT new-entry exposure, and must never be cancelled by this manager.
        return {str(order["id"]): order for order in pending or []
                if order.get("type") in {"MARKET", "LIMIT", "STOP", "MARKET_IF_TOUCHED"}}

    async def get_working_order_count(self) -> int:
        return len(await self._working_entries()) + len(self.pending_approvals)

    async def cancel_pending_entries(self) -> int:
        """Remove new-entry exposure before a halt; leave protective orders alone."""
        async with self._lock:
            cancelled = 0
            for signal_id in list(self.pending_approvals):
                self.reject(signal_id)
                cancelled += 1
            for order_id in await self._working_entries():
                result = await self.broker.cancel_order(order_id)
                if not result.success:
                    raise BrokerError("Failed to cancel resting entry")
                cancelled += 1
            return cancelled

    async def get_open_trade_count(self) -> int:
        """Total exposure: open positions, resting orders and pending approvals."""
        return len(await self.broker.get_positions()) + await self.get_working_order_count()

    async def close(self) -> None:
        await self.broker.close()
