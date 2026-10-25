"""Open-position state machine: SL checks, partial TPs, break-even and trails.

Works identically for the paper broker and OANDA:

    TP1 → close 50 %, stop to breakeven
    TP2 → close 25 % of the original size, stop to TP1
    TP3 → close the runner

Management progress is tracked per trade id so the state survives the fact that
OANDA returns fresh position snapshots on every poll.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from apex_smc_bot.analysis.structure import BEARISH, BULLISH
from apex_smc_bot.config import Settings
from apex_smc_bot.data.candle_manager import Candle, to_utc, utcnow
from apex_smc_bot.execution.broker import Broker, OrderResult, Position
from apex_smc_bot.execution.trail_stop import next_stop, partial_units

logger = logging.getLogger("apex.execution.positions")


@dataclass
class PositionEvent:
    kind: str
    trade_id: str
    signal_id: str
    price: float
    units: float = 0.0
    pnl: float = 0.0
    message: str = ""
    timestamp: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "trade_id": self.trade_id,
            "signal_id": self.signal_id,
            "price": round(self.price, 2),
            "units": round(self.units, 2),
            "pnl": round(self.pnl, 2),
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class _TradeState:
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    realized: float = 0.0


class PositionManager:
    """Monitors open trades and executes the mandated management sequence."""

    def __init__(
        self,
        settings: Settings,
        broker: Broker,
        database: Any | None = None,
        risk_engine: Any | None = None,
    ) -> None:
        self.settings = settings
        self.broker = broker
        self.database = database
        self.risk_engine = risk_engine
        self._known: dict[str, Position] = {}
        self._state: dict[str, _TradeState] = {}
        self._tp1_fraction = settings.strategy.tp1_close_fraction
        self._tp2_fraction = settings.strategy.tp2_close_fraction
        # Clock of the most recent evaluation.  Backtests and replays drive the
        # manager with a simulated clock, so every downstream timestamp (loss
        # cool-downs, drawdown windows) must use it instead of the wall clock.
        self._clock: datetime | None = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _tp_levels(position: Position) -> tuple[float | None, float | None, float | None]:
        metadata = position.metadata or {}
        tp1 = metadata.get("tp1")
        tp2 = metadata.get("tp2") or position.take_profit
        tp3 = metadata.get("tp3")
        return (
            float(tp1) if tp1 is not None else None,
            float(tp2) if tp2 is not None else None,
            float(tp3) if tp3 is not None else None,
        )

    def _state_for(self, position: Position) -> _TradeState:
        state = self._state.get(position.trade_id)
        if state is None:
            state = _TradeState(
                tp1_hit=position.tp1_hit,
                tp2_hit=position.tp2_hit,
                tp3_hit=position.tp3_hit,
            )
            self._state[position.trade_id] = state
        return state

    def _notify_closed(self, position: Position, pnl: float, reason: str) -> None:
        if self.risk_engine is not None:
            try:
                self.risk_engine.note_trade_closed(pnl, self._clock)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to notify risk engine of trade close")
        if self.database is not None:
            try:
                self.database.record_event(
                    "TRADE_CLOSED",
                    {
                        "trade_id": position.trade_id,
                        "signal_id": position.signal_id,
                        "pnl": round(pnl, 2),
                        "reason": reason,
                    },
                )
            except Exception:  # noqa: BLE001
                pass
        logger.info("TRADE CLOSED %s pnl=%+.2f (%s)", position.trade_id, pnl, reason)

    async def _safe_partial(
        self, position: Position, units: float, price: float
    ) -> tuple[bool, float]:
        before = position.realized_pnl
        try:
            result: OrderResult = await self.broker.partial_close(position.trade_id, units, price)
        except Exception as exc:  # noqa: BLE001
            logger.error("Partial close failed for %s: %s", position.trade_id, exc)
            return False, 0.0
        if not result.success:
            logger.warning("Partial close rejected for %s: %s", position.trade_id, result.message)
            return False, 0.0
        pnl = result.realized_pnl if result.realized_pnl is not None else round(
            position.realized_pnl - before, 2
        )
        if self.broker.name != "paper":
            position.units = max(0.0, round(position.units - units, 4))
        return True, round(pnl, 2)

    async def _safe_close(self, position: Position, price: float, reason: str) -> float:
        before = position.realized_pnl
        try:
            result = await self.broker.close_position(position.trade_id, price, reason)
        except Exception as exc:  # noqa: BLE001
            logger.error("Close failed for %s: %s", position.trade_id, exc)
            return 0.0
        if not result.success:
            logger.warning("Close rejected for %s: %s", position.trade_id, result.message)
            return 0.0
        pnl = result.realized_pnl if result.realized_pnl is not None else round(
            position.realized_pnl - before, 2
        )
        position.units = 0.0
        position.status = "CLOSED"
        return round(pnl, 2)

    async def _apply_stop(self, position: Position, stop: float, reason: str) -> bool:
        try:
            result = await self.broker.modify_stop(position.trade_id, stop)
        except Exception as exc:  # noqa: BLE001
            logger.error("Stop modification failed for %s: %s", position.trade_id, exc)
            return False
        if not result.success:
            logger.warning("Stop modification rejected for %s: %s", position.trade_id, result.message)
            return False
        position.stop_loss = round(stop, 2)
        logger.info("STOP %s → %.2f (%s)", position.trade_id, stop, reason)
        return True

    # ------------------------------------------------------------------ main loop
    async def on_price(
        self,
        market_price: float,
        spread_price: float = 0.0,
        candles_5m: Sequence[Candle] | None = None,
        now: datetime | None = None,
    ) -> list[PositionEvent]:
        """Re-evaluate every open position at the latest price."""

        now = to_utc(now or utcnow())
        self._clock = now
        if market_price <= 0:
            raise ValueError("market_price must be positive")
        if hasattr(self.broker, "mark_price"):
            try:
                await self.broker.mark_price(market_price, spread_price, now)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                logger.exception("Paper mark-to-market failed")

        positions = await self.broker.get_positions()
        events: list[PositionEvent] = []
        for position in positions:
            if not position.is_open:
                continue
            events.extend(await self._manage(position, market_price, candles_5m, now))

        events.extend(await self._reconcile(positions, now))
        self._known = {p.trade_id: copy.deepcopy(p) for p in positions}
        return events

    async def _manage(
        self,
        position: Position,
        market_price: float,
        candles_5m: Sequence[Candle] | None,
        now: datetime,
    ) -> list[PositionEvent]:
        events: list[PositionEvent] = []
        bullish = position.direction == BULLISH
        tp1, tp2, tp3 = self._tp_levels(position)
        state = self._state_for(position)

        # 1) Protective stop (paper broker: emulate the broker-side stop).
        hit_stop = market_price <= position.stop_loss if bullish else market_price >= position.stop_loss
        if hit_stop and self.broker.name == "paper":
            fill = min(position.stop_loss, market_price) if bullish else max(position.stop_loss, market_price)
            # A stop that has been trailed past the entry (or to breakeven) closes
            # the trade in profit — labelling that STOP_LOSS would make the trade
            # journal lie about what happened.  The reason is decided *before* the
            # close so the archived record carries the right label.
            remaining = (fill - position.entry_price) if bullish else (position.entry_price - fill)
            projected = round(remaining * position.units, 2)
            projected_total = round(state.realized + projected, 2)
            reason = (
                "STOP_LOSS"
                if projected_total < 0
                else ("BREAKEVEN_STOP" if abs(projected_total) < 0.01 else "TRAIL_STOP")
            )
            pnl = await self._safe_close(position, fill, reason)
            total = round(state.realized + pnl, 2)
            self._notify_closed(position, total, reason)
            self._state.pop(position.trade_id, None)
            events.append(
                PositionEvent(
                    "STOP_HIT",
                    position.trade_id,
                    position.signal_id,
                    fill,
                    position.initial_units,
                    pnl,
                    f"{reason.replace('_', ' ').title()} ({total:+.2f} total)",
                    now,
                    {"total_pnl": total},
                )
            )
            return events

        # 2) TP1 — close 50 %, move to breakeven.
        if tp1 is not None and not state.tp1_hit:
            reached = market_price >= tp1 if bullish else market_price <= tp1
            if reached:
                units = partial_units(position.initial_units, self._tp1_fraction, position.units)
                ok, pnl = await self._safe_partial(position, units, tp1)
                if ok:
                    state.tp1_hit = True
                    state.realized = round(state.realized + pnl, 2)
                    position.tp1_hit = True
                    position.realized_pnl = state.realized
                    if position.units <= 0.01:
                        position.units = 0.0
                        position.status = "CLOSED"
                        self._notify_closed(position, state.realized, "TP1_FULL_FILL")
                        self._state.pop(position.trade_id, None)
                    else:
                        trail = next_stop(
                            entry=position.entry_price,
                            current_stop=position.stop_loss,
                            direction=position.direction,
                            market_price=market_price,
                            tp1=tp1,
                            tp2=tp2 or position.take_profit or tp1,
                            tp1_hit=True,
                            tp2_hit=False,
                            pip_size=self.settings.risk.pip_size,
                            buffer_pips=self.settings.risk.trailing_breakeven_buffer_pips,
                        )
                        if trail is not None:
                            await self._apply_stop(position, trail.stop_loss, trail.reason)
                    events.append(
                        PositionEvent(
                            "TP1_HIT",
                            position.trade_id,
                            position.signal_id,
                            tp1,
                            units,
                            pnl,
                            "TP1 hit — 50% closed, stop to breakeven",
                            now,
                            {"stop_loss": position.stop_loss, "total_pnl": state.realized},
                        )
                    )

        # 3) TP2 — close 25 % of original, lock stop at TP1.
        if tp2 is not None and state.tp1_hit and not state.tp2_hit and position.is_open:
            reached = market_price >= tp2 if bullish else market_price <= tp2
            if reached:
                units = partial_units(position.initial_units, self._tp2_fraction, position.units)
                ok, pnl = await self._safe_partial(position, units, tp2)
                if ok:
                    state.tp2_hit = True
                    state.realized = round(state.realized + pnl, 2)
                    position.tp2_hit = True
                    position.realized_pnl = state.realized
                    if position.units <= 0.01:
                        position.units = 0.0
                        position.status = "CLOSED"
                        self._notify_closed(position, state.realized, "TP2_FULL_FILL")
                        self._state.pop(position.trade_id, None)
                    elif tp1 is not None:
                        improved = tp1 > position.stop_loss if bullish else tp1 < position.stop_loss
                        if improved:
                            await self._apply_stop(position, tp1, "TP2 hit — lock TP1")
                    events.append(
                        PositionEvent(
                            "TP2_HIT",
                            position.trade_id,
                            position.signal_id,
                            tp2,
                            units,
                            pnl,
                            "TP2 hit — 25% closed, stop locked at TP1",
                            now,
                            {"stop_loss": position.stop_loss, "total_pnl": state.realized},
                        )
                    )

        # 4) TP3 — close the runner.
        if tp3 is not None and state.tp2_hit and not state.tp3_hit and position.is_open:
            reached = market_price >= tp3 if bullish else market_price <= tp3
            if reached:
                units = position.units
                pnl = await self._safe_close(position, tp3, "TP3")
                state.tp3_hit = True
                state.realized = round(state.realized + pnl, 2)
                self._notify_closed(position, state.realized, "TP3")
                self._state.pop(position.trade_id, None)
                events.append(
                    PositionEvent(
                        "TP3_HIT",
                        position.trade_id,
                        position.signal_id,
                        tp3,
                        units,
                        pnl,
                        f"TP3 hit — runner closed ({state.realized:+.2f} total)",
                        now,
                        {"total_pnl": state.realized},
                    )
                )

        # 5) Trail the runner (structure/ATR) after TP2.
        if position.is_open and state.tp2_hit and candles_5m:
            trail = next_stop(
                entry=position.entry_price,
                current_stop=position.stop_loss,
                direction=position.direction,
                market_price=market_price,
                tp1=tp1 or position.take_profit or position.entry_price,
                tp2=tp2 or position.take_profit or position.entry_price,
                tp1_hit=True,
                tp2_hit=True,
                candles=candles_5m,
                pip_size=self.settings.risk.pip_size,
                buffer_pips=self.settings.risk.trailing_breakeven_buffer_pips,
            )
            if trail is not None:
                if await self._apply_stop(position, trail.stop_loss, trail.reason):
                    events.append(
                        PositionEvent(
                            "TRAIL_UPDATED",
                            position.trade_id,
                            position.signal_id,
                            market_price,
                            position.units,
                            0.0,
                            trail.reason,
                            now,
                            {"stop_loss": position.stop_loss, "stage": trail.stage},
                        )
                    )
        return events

    async def _reconcile(self, positions: Sequence[Position], now: datetime) -> list[PositionEvent]:
        """Detect positions closed broker-side (SL/TP) between polls."""

        current = {p.trade_id for p in positions}
        events: list[PositionEvent] = []
        for trade_id, known in list(self._known.items()):
            if trade_id in current or not known.is_open:
                continue
            state = self._state.pop(trade_id, None)
            pnl = state.realized if state else known.realized_pnl
            closed: dict[str, Any] | None = None
            getter = getattr(self.broker, "get_closed_trades", None)
            if callable(getter):
                try:
                    closed_trades = await getter(20)
                    closed = next((t for t in closed_trades if str(t.get("id")) == trade_id), None)
                except Exception:  # noqa: BLE001
                    closed = None
            if closed:
                try:
                    pnl = float(closed.get("realizedPL", pnl) or pnl)
                except (TypeError, ValueError):
                    pass
            events.append(
                PositionEvent(
                    "BROKER_CLOSED",
                    trade_id,
                    known.signal_id,
                    known.current_price or known.stop_loss,
                    known.units,
                    pnl,
                    "Position closed broker-side (SL/TP)",
                    now,
                    {"closed_trade": closed} if closed else {},
                )
            )
            self._notify_closed(known, pnl, "BROKER_CLOSED")
        return events

    # ------------------------------------------------------------------ commands
    async def force_close(
        self,
        trade_id: str,
        reason: str = "MANUAL",
        price: float | None = None,
        now: datetime | None = None,
    ) -> PositionEvent | None:
        now = to_utc(now or utcnow())
        positions = {p.trade_id: p for p in await self.broker.get_positions()}
        position = positions.get(trade_id) or self._known.get(trade_id)
        if position is None:
            return None
        units = position.units
        state = self._state.pop(trade_id, None) or _TradeState()
        pnl = await self._safe_close(position, price or position.current_price, reason)
        total = round(state.realized + pnl, 2)
        self._notify_closed(position, total, reason)
        return PositionEvent(
            "CLOSED",
            trade_id,
            position.signal_id,
            price or position.current_price,
            units,
            pnl,
            f"Closed ({reason}) — total {total:+.2f}",
            now,
            {"total_pnl": total},
        )

    async def force_close_all(
        self, reason: str = "CIRCUIT_BREAKER", price: float | None = None, now: datetime | None = None
    ) -> list[PositionEvent]:
        self._clock = to_utc(now or self._clock or utcnow())
        positions = await self.broker.get_positions()
        events = []
        for position in positions:
            event = await self.force_close(position.trade_id, reason, price, now)
            if event:
                events.append(event)
        return events

    async def open_positions(self) -> list[Position]:
        return await self.broker.get_positions()

    def managed_state(self) -> dict[str, Any]:
        return {
            trade_id: {
                "tp1_hit": state.tp1_hit,
                "tp2_hit": state.tp2_hit,
                "tp3_hit": state.tp3_hit,
                "realized": state.realized,
            }
            for trade_id, state in self._state.items()
        }
