"""Broker protocol and execution value objects shared by paper/OANDA engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from apex_smc_bot.analysis.structure import BEARISH
from apex_smc_bot.data.candle_manager import to_utc


class BrokerError(RuntimeError):
    """Broker rejected or failed an operation."""


class BrokerNotConfigured(BrokerError):
    """Credentials/SDK missing — execution cannot proceed."""


@dataclass
class OrderRequest:
    """Broker-agnostic order instruction."""

    signal_id: str
    instrument: str
    direction: str  # BULLISH (buy) | BEARISH (sell)
    units: float
    entry_price: float
    stop_loss: float
    take_profit: float | None = None
    order_type: str = "LIMIT"  # LIMIT | MARKET
    expires_at: datetime | None = None
    client_tag: str = "APEX-SMC"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def signed_units(self) -> int:
        units = int(round(abs(self.units)))
        units = max(1, units)
        return units if self.direction != BEARISH else -units

    @property
    def side(self) -> str:
        return "BUY" if self.direction != BEARISH else "SELL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "instrument": self.instrument,
            "direction": self.direction,
            "units": self.units,
            "signed_units": self.signed_units,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "order_type": self.order_type,
            "expires_at": to_utc(self.expires_at).isoformat() if self.expires_at else None,
        }


@dataclass
class OrderResult:
    success: bool
    status: str  # FILLED | PENDING | REJECTED | PENDING_APPROVAL | CANCELLED
    order_id: str = ""
    trade_id: str = ""
    filled_price: float | None = None
    units: float = 0.0
    message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    signal_id: str = ""
    realized_pnl: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "order_id": self.order_id,
            "trade_id": self.trade_id,
            "filled_price": self.filled_price,
            "units": self.units,
            "message": self.message,
            "signal_id": self.signal_id,
            "realized_pnl": self.realized_pnl,
        }


@dataclass
class Position:
    """An open (or closed) broker position."""

    trade_id: str
    signal_id: str
    instrument: str
    direction: str
    units: float  # remaining units, always positive
    initial_units: float
    entry_price: float
    stop_loss: float
    take_profit: float | None
    opened_at: datetime
    status: str = "OPEN"
    current_price: float = 0.0
    realized_pnl: float = 0.0
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    closed_at: datetime | None = None
    exit_price: float | None = None

    @property
    def is_long(self) -> bool:
        return self.direction != BEARISH

    @property
    def is_open(self) -> bool:
        return self.status == "OPEN" and self.units > 0

    @property
    def price_precision(self) -> int:
        return 2

    def unrealized_pnl(self, price: float | None = None) -> float:
        if not self.is_open:
            return 0.0
        mark = price if price is not None else self.current_price
        if not mark:
            return 0.0
        delta = mark - self.entry_price if self.is_long else self.entry_price - mark
        return round(delta * self.units, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "signal_id": self.signal_id,
            "instrument": self.instrument,
            "direction": self.direction,
            "units": round(self.units, 2),
            "initial_units": round(self.initial_units, 2),
            "entry_price": round(self.entry_price, 5),
            "stop_loss": round(self.stop_loss, 5),
            "take_profit": round(self.take_profit, 5) if self.take_profit else None,
            "opened_at": to_utc(self.opened_at).isoformat(),
            "closed_at": to_utc(self.closed_at).isoformat() if self.closed_at else None,
            "exit_price": round(self.exit_price, 5) if self.exit_price else None,
            "status": self.status,
            "current_price": self.current_price,
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl(), 2),
            "tp1_hit": self.tp1_hit,
            "tp2_hit": self.tp2_hit,
            "tp3_hit": self.tp3_hit,
            "metadata": self.metadata,
            "close_reason": self.metadata.get("close_reason", ""),
        }


@dataclass
class AccountSnapshot:
    balance: float
    equity: float
    margin_available: float = 0.0
    open_trade_count: int = 0
    currency: str = "USD"

    def to_dict(self) -> dict[str, Any]:
        return {
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "margin_available": round(self.margin_available, 2),
            "open_trade_count": self.open_trade_count,
            "currency": self.currency,
        }


@runtime_checkable
class Broker(Protocol):
    """Minimal execution contract shared by paper and OANDA engines."""

    name: str

    async def submit_order(self, request: OrderRequest, market_price: float, spread_price: float = 0.0) -> OrderResult:
        ...

    async def modify_stop(self, trade_id: str, stop_loss: float) -> OrderResult:
        ...

    async def partial_close(self, trade_id: str, units: float, price: float | None = None) -> OrderResult:
        ...

    async def close_position(self, trade_id: str, price: float | None = None, reason: str = "") -> OrderResult:
        ...

    async def get_positions(self) -> list[Position]:
        ...

    async def get_account(self) -> AccountSnapshot:
        ...

    async def close(self) -> None:
        ...
