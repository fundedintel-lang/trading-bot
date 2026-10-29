"""Deterministic paper broker used for demo mode, semi-auto approval and tests.

Fills:
    MARKET  at mid ∓ half spread + adverse slippage
    LIMIT   only when touched; filled at the limit price (no positive slippage)

The broker tracks balance, equity, partial closes and realized PnL.  It does not
invent fills beyond the provided quote.  Stop-loss / take-profit management is
performed by :class:`~apex_smc_bot.execution.position_manager.PositionManager`
so live and paper behaviour follow the same state machine.
"""

from __future__ import annotations

import copy
import logging
import math
from datetime import datetime
from typing import Any

from apex_smc_bot.analysis.structure import BEARISH, BULLISH
from apex_smc_bot.data.candle_manager import Candle, to_utc, utcnow
from apex_smc_bot.execution.broker import (
    AccountSnapshot,
    BrokerError,
    OrderRequest,
    OrderResult,
    Position,
)

logger = logging.getLogger("apex.execution.paper")


class PaperBroker:
    """In-memory broker with realistic fills and account accounting."""

    name = "paper"

    def __init__(
        self,
        starting_equity: float = 10_000.0,
        database: Any | None = None,
        slippage_price: float = 0.0,
        contract_size: float = 100.0,
        commission_per_lot: float = 0.0,  # per side, charged pro rata
    ) -> None:
        if not all(math.isfinite(v) for v in (starting_equity, slippage_price, contract_size, commission_per_lot)):
            raise ValueError("Broker inputs must be finite")
        if contract_size <= 0 or commission_per_lot < 0 or slippage_price < 0:
            raise ValueError("Invalid contract size or trading costs")
        if starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        self.balance = float(starting_equity)
        self.starting_equity = float(starting_equity)
        self.database = database
        self.slippage_price = max(0.0, slippage_price)
        self.contract_size = contract_size
        self.commission_per_lot = commission_per_lot
        self._positions: dict[str, Position] = {}
        self._history: list[Position] = []
        self._pending: dict[str, OrderRequest] = {}
        self._last_mid: float = 0.0
        self._last_spread: float = 0.0
        self._trade_seq = 0
        self._order_seq = 0
        self._equity_seq = 0
        # Simulated clock (set from mark_price / fills).  Backtests drive the
        # broker with historical timestamps, so nothing here may call utcnow()
        # for trade lifecycle fields.
        self._clock: datetime = utcnow()

    # ------------------------------------------------------------------ helpers
    def _next_trade_id(self) -> str:
        self._trade_seq += 1
        return f"PAPER-{self._trade_seq:06d}"

    def _next_order_id(self) -> str:
        self._order_seq += 1
        return f"ORD-{self._order_seq:06d}"

    def _archive(self, position: Position) -> None:
        """Append an immutable snapshot of a closed position to the history.

        Positions are mutated in place while open, so the history must hold a
        copy — otherwise every historical record ends up showing the final
        units/P&L and the performance report double counts trades.
        """

        snapshot = copy.deepcopy(position)
        snapshot.metadata = dict(position.metadata)
        self._history = [record for record in self._history if record.trade_id != snapshot.trade_id]
        self._history.append(snapshot)

    def _close_out(self, position: Position, exit_price: float, reason: str) -> None:
        position.units = 0.0
        position.status = "CLOSED"
        position.exit_price = round(exit_price, 3)
        position.closed_at = self._clock
        position.metadata["close_reason"] = reason
        self._archive(position)

    def _exit_price(self, mid: float, is_long: bool, slippage: bool = True) -> float:
        cost = self._last_spread / 2.0 + (self.slippage_price if slippage else 0.0)
        return round(mid - cost if is_long else mid + cost, 3)

    def _commission(self, units: float) -> float:
        return round(units / self.contract_size * self.commission_per_lot, 2)

    def _entry_price(self, mid: float, spread: float, is_long: bool) -> float:
        half = spread / 2.0
        price = mid + half if is_long else mid - half
        price += self.slippage_price if is_long else -self.slippage_price
        return round(price, 3)

    # ------------------------------------------------------------------ orders
    async def submit_order(
        self, request: OrderRequest, market_price: float, spread_price: float = 0.0
    ) -> OrderResult:
        values = (request.units, request.entry_price, request.stop_loss, market_price, spread_price)
        if any(not math.isfinite(value) for value in values) or spread_price < 0:
            return OrderResult(False, "REJECTED", message="non-finite order or quote", signal_id=request.signal_id)
        if request.order_type.upper() not in {"LIMIT", "MARKET"}:
            return OrderResult(False, "REJECTED", message="unsupported order type", signal_id=request.signal_id)
        direction = str(request.direction).upper()
        if direction not in {"BULLISH", "BEARISH", "LONG", "SHORT", "BUY", "SELL"}:
            return OrderResult(False, "REJECTED", message="invalid direction", signal_id=request.signal_id)
        if any(order.signal_id == request.signal_id for order in self._pending.values()):
            return OrderResult(False, "REJECTED", message="signal already has a resting order", signal_id=request.signal_id)
        if request.units <= 0:
            return OrderResult(
                success=False,
                status="REJECTED",
                message="units must be positive",
                signal_id=request.signal_id,
            )
        if market_price <= 0:
            return OrderResult(
                success=False,
                status="REJECTED",
                message="invalid market price",
                signal_id=request.signal_id,
            )
        if any(p.signal_id == request.signal_id and p.is_open for p in self._positions.values()):
            return OrderResult(
                success=False,
                status="REJECTED",
                message="signal already has an open position",
                signal_id=request.signal_id,
            )
        self._last_mid = market_price
        self._last_spread = max(0.0, spread_price)
        is_long = direction in {"BULLISH", "LONG", "BUY"}
        if request.order_type.upper() == "MARKET":
            return self._fill(request, market_price, self._last_spread, is_long)

        # LIMIT: fill only if the market has reached the price.
        executable = market_price + self._last_spread / 2 if is_long else market_price - self._last_spread / 2
        reached = executable <= request.entry_price if is_long else executable >= request.entry_price
        if reached:
            return self._fill(request, request.entry_price, self._last_spread, is_long, limit=True)
        order_id = self._next_order_id()
        self._pending[order_id] = request
        logger.info(
            "Paper limit order %s pending: %s %s units @ %.2f (market %.2f)",
            order_id,
            request.side,
            request.units,
            request.entry_price,
            market_price,
        )
        return OrderResult(
            success=True,
            status="PENDING",
            order_id=order_id,
            message="limit order resting",
            signal_id=request.signal_id,
        )

    def _fill(
        self,
        request: OrderRequest,
        mid_price: float,
        spread: float,
        is_long: bool,
        limit: bool = False,
    ) -> OrderResult:
        if limit:
            fill_price = round(request.entry_price, 3)
        else:
            fill_price = self._entry_price(mid_price, spread, is_long)
        if not is_long and (fill_price >= request.stop_loss):
            return OrderResult(
                success=False,
                status="REJECTED",
                message="fill price would already be through the stop",
                signal_id=request.signal_id,
            )
        if is_long and (fill_price <= request.stop_loss):
            return OrderResult(
                success=False,
                status="REJECTED",
                message="fill price would already be through the stop",
                signal_id=request.signal_id,
            )
        trade_id = self._next_trade_id()
        # Normalise broker vocabulary: the position manager compares against
        # BULLISH/BEARISH, so a "LONG"/"BUY" request must not silently become a
        # short in the stop/target logic.
        direction = BEARISH if str(request.direction).upper() in {"BEARISH", "SHORT", "SELL"} else BULLISH
        position = Position(
            trade_id=trade_id,
            signal_id=request.signal_id,
            instrument=request.instrument,
            direction=direction,
            units=float(request.units),
            initial_units=float(request.units),
            entry_price=fill_price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            opened_at=self._clock,
            current_price=mid_price,
            metadata=dict(request.metadata),
        )
        fee = self._commission(request.units)
        self.balance = round(self.balance - fee, 2)
        position.realized_pnl = -fee
        position.metadata["commission_paid"] = fee
        position.metadata["initial_stop_loss"] = request.stop_loss
        self._positions[trade_id] = position
        self._persist_position(position)
        logger.info(
            "PAPER FILL %s %s %s units @ %.2f SL %.2f",
            trade_id,
            request.side,
            request.units,
            fill_price,
            request.stop_loss,
        )
        return OrderResult(
            success=True,
            status="FILLED",
            order_id=self._next_order_id(),
            trade_id=trade_id,
            filled_price=fill_price,
            units=float(request.units),
            message="paper fill",
            signal_id=request.signal_id,
        )

    async def cancel_order(self, order_id: str) -> OrderResult:
        request = self._pending.pop(order_id, None)
        if request is None:
            return OrderResult(success=False, status="REJECTED", message="unknown order", order_id=order_id)
        return OrderResult(success=True, status="CANCELLED", order_id=order_id, message="cancelled")

    # ------------------------------------------------------------------ market data
    async def mark_price(
        self, mid_price: float, spread_price: float = 0.0, now: datetime | None = None,
        *, bar: Candle | None = None,
    ) -> list[Position]:
        """Update marks, fill touched limit orders and return open positions."""

        if not math.isfinite(mid_price) or not math.isfinite(spread_price) or mid_price <= 0 or spread_price < 0:
            raise BrokerError("mark price must be positive")
        now = to_utc(now or utcnow())
        self._clock = now
        self._last_mid = mid_price
        self._last_spread = max(0.0, spread_price)

        for order_id, request in list(self._pending.items()):
            if request.expires_at is not None and to_utc(request.expires_at) <= now:
                self._pending.pop(order_id, None)
                continue
            is_long = str(request.direction).upper() not in {"BEARISH", "SHORT", "SELL"}
            low = bar.low if bar is not None else mid_price
            high = bar.high if bar is not None else mid_price
            reached = low + self._last_spread / 2 <= request.entry_price if is_long else high - self._last_spread / 2 >= request.entry_price
            if reached:
                self._pending.pop(order_id, None)
                self._fill(request, request.entry_price, self._last_spread, is_long, limit=True)

        for position in self._positions.values():
            if position.is_open:
                position.current_price = mid_price

        self._equity_seq += 1
        if self.database is not None and self._equity_seq % 10 == 0:
            self.database.record_equity(self.equity(mid_price), self.balance)
        return [p for p in self._positions.values() if p.is_open]

    def equity(self, price: float | None = None, *, spread_price: float | None = None) -> float:
        mark = price if price is not None else self._last_mid
        half = (self._last_spread if spread_price is None else spread_price) / 2
        unrealized = sum(p.unrealized_pnl(mark - half if p.is_long else mark + half)
                         for p in self._positions.values() if p.is_open)
        return round(self.balance + unrealized, 2)

    # ------------------------------------------------------------------ management
    async def modify_stop(self, trade_id: str, stop_loss: float) -> OrderResult:
        position = self._positions.get(trade_id)
        if position is None or not position.is_open:
            return OrderResult(success=False, status="REJECTED", message="position not found", trade_id=trade_id)
        executable = self._exit_price(self._last_mid, position.is_long, slippage=False)
        if not math.isfinite(stop_loss) or stop_loss <= 0 or (
            stop_loss >= executable if position.is_long else stop_loss <= executable
        ):
            return OrderResult(False, "REJECTED", trade_id=trade_id, message="invalid protective stop")
        position.stop_loss = round(float(stop_loss), 3)
        self._persist_position(position)
        return OrderResult(
            success=True,
            status="MODIFIED",
            trade_id=trade_id,
            message=f"stop moved to {position.stop_loss}",
        )

    async def partial_close(self, trade_id: str, units: float, price: float | None = None) -> OrderResult:
        position = self._positions.get(trade_id)
        if position is None or not position.is_open:
            return OrderResult(success=False, status="REJECTED", message="position not found", trade_id=trade_id)
        units = float(units)
        if units <= 0 or units > position.units + 1e-9:
            return OrderResult(
                success=False,
                status="REJECTED",
                message=f"invalid partial close units {units} of {position.units}",
                trade_id=trade_id,
            )
        exit_price = price if price is not None else self._exit_price(position.current_price, position.is_long)
        pnl = (exit_price - position.entry_price) * units if position.is_long else (
            position.entry_price - exit_price
        ) * units
        fee = self._commission(units)
        pnl = round(pnl - fee, 2)
        position.metadata["commission_paid"] = round(position.metadata.get("commission_paid", 0.0) + fee, 2)
        self.balance = round(self.balance + pnl, 2)
        position.realized_pnl = round(position.realized_pnl + pnl, 2)
        position.units = round(position.units - units, 4)
        if position.units <= 0.01:
            self._close_out(position, exit_price, "PARTIAL_FULL_FILL")
        self._persist_position(position)
        return OrderResult(
            success=True,
            status="PARTIAL_CLOSE",
            trade_id=trade_id,
            filled_price=exit_price,
            units=units,
            message=f"closed {units} units at {exit_price} for {pnl:+.2f}",
            realized_pnl=pnl,
        )

    async def close_position(
        self, trade_id: str, price: float | None = None, reason: str = ""
    ) -> OrderResult:
        position = self._positions.get(trade_id)
        if position is None:
            return OrderResult(success=False, status="REJECTED", message="position not found", trade_id=trade_id)
        if not position.is_open:
            return OrderResult(success=True, status="CLOSED", trade_id=trade_id, message="already closed")
        exit_price = price if price is not None else self._exit_price(position.current_price, position.is_long)
        pnl = (exit_price - position.entry_price) * position.units if position.is_long else (
            position.entry_price - exit_price
        ) * position.units
        fee = self._commission(position.units)
        pnl = round(pnl - fee, 2)
        position.metadata["commission_paid"] = round(position.metadata.get("commission_paid", 0.0) + fee, 2)
        self.balance = round(self.balance + pnl, 2)
        position.realized_pnl = round(position.realized_pnl + pnl, 2)
        self._close_out(position, exit_price, reason)
        self._persist_position(position)
        logger.info("PAPER CLOSE %s at %.2f (%+.2f) — %s", trade_id, exit_price, pnl, reason)
        return OrderResult(
            success=True,
            status="CLOSED",
            trade_id=trade_id,
            filled_price=exit_price,
            message=f"closed at {exit_price} for {pnl:+.2f} ({reason})",
            realized_pnl=pnl,
        )

    async def close_all(self, reason: str = "MANUAL") -> list[OrderResult]:
        return [await self.close_position(position.trade_id, reason=reason) for position in self.get_open_positions()]

    # ------------------------------------------------------------------ reads
    def get_open_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.is_open]

    async def get_positions(self) -> list[Position]:
        return self.get_open_positions()

    def get_position(self, trade_id: str) -> Position | None:
        return self._positions.get(trade_id)

    def get_history(self) -> list[Position]:
        """Closed trades, oldest first (immutable snapshots)."""

        return [copy.deepcopy(record) for record in self._history]

    def get_pending_orders(self) -> dict[str, OrderRequest]:
        return dict(self._pending)

    async def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            balance=self.balance,
            equity=self.equity(),
            margin_available=round(self.equity() * 0.9, 2),
            open_trade_count=len(self.get_open_positions()),
        )

    async def close(self) -> None:
        return None

    # ------------------------------------------------------------------ persistence
    def _persist_position(self, position: Position) -> None:
        if self.database is None:
            return
        self.database.record_trade(
            {
                "trade_id": position.trade_id,
                "signal_id": position.signal_id,
                "instrument": position.instrument,
                "direction": position.direction,
                "status": position.status,
                "entry_price": position.entry_price,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "lots": round(position.initial_units / self.contract_size, 2),
                "units": position.units,
                "opened_at": position.opened_at.isoformat(),
                "closed_at": to_utc(position.closed_at).isoformat() if position.closed_at else None,
                "realized_pnl": position.realized_pnl,
                "payload": position.to_dict(),
            }
        )
