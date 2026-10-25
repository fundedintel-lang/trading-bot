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

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from apex_smc_bot.config import Settings
from apex_smc_bot.data.candle_manager import to_utc, utcnow
from apex_smc_bot.execution.broker import (
    AccountSnapshot,
    Broker,
    BrokerError,
    OrderRequest,
    OrderResult,
    Position,
)
from apex_smc_bot.signals.models import Signal

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

    def _build_request(self, signal: Signal) -> OrderRequest:
        return OrderRequest(
            signal_id=signal.signal_id,
            instrument=signal.instrument,
            direction=signal.direction,
            units=signal.position_sizing.units,
            entry_price=signal.entry.price,
            stop_loss=signal.stop_loss.price,
            take_profit=signal.take_profits.tp2.price,
            order_type=signal.entry.type.upper(),
            expires_at=signal.expires_at,
            metadata={
                "tp1": signal.take_profits.tp1.price,
                "tp2": signal.take_profits.tp2.price,
                "tp3": signal.take_profits.tp3.price if signal.take_profits.tp3 else None,
                "lots": signal.position_sizing.lot_size,
                "confluence": signal.confluence_score,
                "confidence": signal.confidence_percent,
            },
        )

    def _validate_execution_price(self, signal: Signal, market_price: float) -> str | None:
        if market_price <= 0:
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

    async def _execute(self, signal: Signal, market_price: float, spread_price: float) -> OrderResult:
        problem = self._validate_execution_price(signal, market_price)
        if problem:
            self._mark_signal(signal.signal_id, "REJECTED")
            return OrderResult(
                success=False,
                status="REJECTED",
                message=problem,
                signal_id=signal.signal_id,
            )
        if signal.signal_id in self.executed_signals:
            return OrderResult(
                success=False,
                status="REJECTED",
                message="signal already executed",
                signal_id=signal.signal_id,
            )
        request = self._build_request(signal)
        try:
            result = await self.broker.submit_order(request, market_price, spread_price)
        except BrokerError as exc:
            logger.error("Broker rejected %s: %s", signal.signal_id, exc)
            self._mark_signal(signal.signal_id, "REJECTED")
            return OrderResult(
                success=False, status="REJECTED", message=str(exc), signal_id=signal.signal_id
            )
        if result.success:
            self.executed_signals.add(signal.signal_id)
            self._mark_signal(signal.signal_id, "EXECUTED" if result.status == "FILLED" else "PENDING")
            logger.info(
                "ORDER %s %s %s units @ %s (%s)",
                result.status,
                signal.side,
                signal.position_sizing.units,
                result.filled_price or signal.entry.price,
                signal.signal_id,
            )
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
        self,
        signal: Signal,
        market_price: float,
        spread_price: float = 0.0,
        now: datetime | None = None,
    ) -> OrderResult:
        """Execute in auto mode or queue for human approval in semi-auto."""

        now = to_utc(now or utcnow())
        if signal.expires_at is not None and to_utc(signal.expires_at) < now:
            return OrderResult(
                success=False,
                status="REJECTED",
                message="signal expired before submission",
                signal_id=signal.signal_id,
            )
        if signal.signal_id in self.executed_signals:
            return OrderResult(
                success=False,
                status="REJECTED",
                message="duplicate signal submission",
                signal_id=signal.signal_id,
            )
        self.expire_pending(now)
        if self._auto_execute():
            return await self._execute(signal, market_price, spread_price)

        self.pending_approvals[signal.signal_id] = PendingApproval(
            signal=signal,
            created_at=now,
            expires_at=to_utc(signal.expires_at) if signal.expires_at else now,
        )
        self._mark_signal(signal.signal_id, "PENDING_APPROVAL")
        logger.info("Signal %s awaiting human approval", signal.signal_id)
        return OrderResult(
            success=True,
            status="PENDING_APPROVAL",
            signal_id=signal.signal_id,
            message="awaiting human approval",
        )

    async def approve(
        self,
        signal_id: str,
        market_price: float,
        spread_price: float = 0.0,
        now: datetime | None = None,
    ) -> OrderResult:
        now = to_utc(now or utcnow())
        self.expire_pending(now)
        approval = self.pending_approvals.pop(signal_id, None)
        if approval is None:
            return OrderResult(
                success=False,
                status="REJECTED",
                message="no pending approval with that id",
                signal_id=signal_id,
            )
        if to_utc(approval.expires_at) < now:
            self._mark_signal(signal_id, "EXPIRED")
            return OrderResult(
                success=False,
                status="REJECTED",
                message="approval window expired",
                signal_id=signal_id,
            )
        return await self._execute(approval.signal, market_price, spread_price)

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
            if to_utc(approval.expires_at) < now
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

    async def get_working_order_count(self) -> int:
        """Orders that will become positions if price cooperates.

        Counting filled positions alone lets a burst of resting limits all fill
        at once and blow through ``max_open_trades``.  Exposure therefore means
        "positions + working orders + orders awaiting approval".
        """

        count = 0
        getter = getattr(self.broker, "get_pending_orders", None)
        if callable(getter):
            try:
                pending = getter()
                if isinstance(pending, Mapping):
                    count += len(pending)
                else:
                    count += len(list(pending or []))
            except Exception:  # noqa: BLE001
                logger.debug("Broker pending order lookup failed", exc_info=True)
        count += len(self.pending_approvals)
        return count

    async def get_open_trade_count(self) -> int:
        """Total exposure: open positions, resting orders and pending approvals."""
        return len(await self.broker.get_positions()) + await self.get_working_order_count()

    async def close(self) -> None:
        await self.broker.close()
