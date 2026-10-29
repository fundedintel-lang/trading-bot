"""Execution layer: fills, partials, trailing stops, order routing and ledger."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from apex_smc_bot.config import Settings
from apex_smc_bot.execution.broker import OrderRequest
from apex_smc_bot.execution.order_manager import OrderManager
from apex_smc_bot.execution.paper_broker import PaperBroker
from apex_smc_bot.execution.position_manager import PositionManager
from apex_smc_bot.execution.trail_stop import next_stop
from apex_smc_bot.signals.models import BEARISH
from apex_smc_bot.tests.factories import make_signal

T0 = datetime(2026, 10, 7, 13, 0, tzinfo=timezone.utc)


def _request(signal_id: str = "S1", direction: str = BEARISH, units: float = 37.0, **kwargs) -> OrderRequest:
    defaults = dict(
        signal_id=signal_id,
        instrument="XAU_USD",
        direction=direction,
        units=units,
        entry_price=2350.0,
        stop_loss=2352.7,
        take_profit=2345.95,
        order_type="MARKET",
    )
    defaults.update(kwargs)
    return OrderRequest(**defaults)


# ------------------------------------------------------------------ paper broker
@pytest.mark.asyncio
async def test_market_fill_pays_the_spread_and_slippage():
    broker = PaperBroker(starting_equity=10_000, slippage_price=0.05, contract_size=100)
    await broker.mark_price(2350.0, 0.20, T0)
    result = await broker.submit_order(_request(direction=BEARISH), 2350.0, 0.20)
    assert result.status == "FILLED"
    position = broker.get_position(result.trade_id)
    # Short: mid - half spread - slippage.
    assert position.entry_price == pytest.approx(2349.85, abs=0.01)
    assert position.opened_at == T0


@pytest.mark.asyncio
async def test_limit_order_rests_until_touched():
    broker = PaperBroker(starting_equity=10_000)
    await broker.mark_price(2350.0, 0.20, T0)
    request = _request(order_type="LIMIT", entry_price=2355.0, stop_loss=2357.0, take_profit=2350.0)
    first = await broker.submit_order(request, 2350.0, 0.20)
    assert first.status == "PENDING"
    assert broker.get_pending_orders()
    # Price rises through the limit.
    await broker.mark_price(2355.5, 0.20, T0 + timedelta(minutes=5))
    assert broker.get_pending_orders() == {}
    assert len(broker.get_open_positions()) == 1


@pytest.mark.asyncio
async def test_limit_is_not_filled_on_the_wrong_side():
    broker = PaperBroker(starting_equity=10_000)
    await broker.mark_price(2340.0, 0.20, T0)
    request = _request(order_type="LIMIT", entry_price=2355.0, stop_loss=2357.0, take_profit=2350.0)
    result = await broker.submit_order(request, 2340.0, 0.20)
    assert result.status == "PENDING"
    assert not broker.get_open_positions()


@pytest.mark.asyncio
async def test_partial_close_realizes_half_the_position():
    broker = PaperBroker(starting_equity=10_000, slippage_price=0.0)
    await broker.mark_price(2350.0, 0.20, T0)
    fill = await broker.submit_order(_request(units=40), 2350.0, 0.20)
    await broker.mark_price(2348.0, 0.20, T0 + timedelta(minutes=15))
    partial = await broker.partial_close(fill.trade_id, 20.0, price=2348.0)
    assert partial.success
    entry = broker.get_position(fill.trade_id).entry_price
    assert broker.balance == pytest.approx(10_000 + (entry - 2348.0) * 20, abs=0.01)
    position = broker.get_position(fill.trade_id)
    assert position.units == pytest.approx(20.0)
    assert position.status == "OPEN"
    assert position.realized_pnl > 0


@pytest.mark.asyncio
async def test_full_close_archives_a_single_immutable_snapshot():
    broker = PaperBroker(starting_equity=10_000, slippage_price=0.0)
    await broker.mark_price(2350.0, 0.20, T0)
    fill = await broker.submit_order(_request(units=40), 2350.0, 0.20)
    await broker.mark_price(2346.0, 0.20, T0 + timedelta(minutes=30))
    await broker.partial_close(fill.trade_id, 20.0, price=2346.0)
    await broker.close_position(fill.trade_id, price=2346.0, reason="TP2")
    await broker.close_position(fill.trade_id, price=2346.0, reason="TP2")  # idempotent

    history = broker.get_history()
    assert len(history) == 1
    record = history[0]
    assert record.status == "CLOSED" and record.units == 0.0
    assert record.closed_at == T0 + timedelta(minutes=30)
    assert record.metadata["close_reason"] == "TP2"
    # History must be a snapshot: mutating the live position cannot rewrite it.
    broker.get_position(fill.trade_id).realized_pnl = 999.0
    assert broker.get_history()[0].realized_pnl != 999.0


@pytest.mark.asyncio
async def test_equity_tracks_balance_and_unrealized():
    broker = PaperBroker(starting_equity=10_000, slippage_price=0.0)
    await broker.mark_price(2350.0, 0.0, T0)
    await broker.submit_order(_request(units=100), 2350.0, 0.0)
    await broker.mark_price(2340.0, 0.0, T0 + timedelta(minutes=10))
    assert broker.equity(2340.0) == pytest.approx(11_000.0, abs=0.01)


@pytest.mark.asyncio
async def test_duplicate_signal_cannot_open_twice():
    broker = PaperBroker(starting_equity=10_000)
    await broker.mark_price(2350.0, 0.20, T0)
    await broker.submit_order(_request(), 2350.0, 0.20)
    again = await broker.submit_order(_request(), 2350.0, 0.20)
    assert not again.success
    assert "already" in again.message


@pytest.mark.asyncio
async def test_stop_already_through_the_fill_is_rejected():
    broker = PaperBroker(starting_equity=10_000)
    await broker.mark_price(2350.0, 0.20, T0)
    result = await broker.submit_order(
        _request(direction=BEARISH, entry_price=2350.0, stop_loss=2349.0), 2350.0, 0.20
    )
    assert not result.success


# ----------------------------------------------------------------- position mgmt
@pytest.mark.asyncio
async def test_position_manager_closes_on_stop_and_notifies_risk():
    settings = Settings()
    broker = PaperBroker(starting_equity=10_000, slippage_price=0.0)
    await broker.mark_price(2350.0, 0.20, T0)
    fill = await broker.submit_order(_request(units=100), 2350.0, 0.0)

    class RiskSpy:
        calls: list[tuple[float, datetime | None]] = []

        def note_trade_closed(self, pnl: float, now=None) -> None:
            RiskSpy.calls.append((pnl, now))

    manager = PositionManager(settings, broker, risk_engine=RiskSpy())
    events = await manager.on_price(2360.0, 0.20, [], T0 + timedelta(minutes=20))
    assert any(event.kind == "STOP_HIT" for event in events)
    assert broker.get_history()
    # The simulated clock — not the wall clock — must reach the risk engine.
    assert RiskSpy.calls and RiskSpy.calls[-1][1] == T0 + timedelta(minutes=20)
    assert fill.trade_id in broker.get_history()[0].trade_id


@pytest.mark.asyncio
async def test_tp1_partial_then_breakeven_stop():
    settings = Settings()
    broker = PaperBroker(starting_equity=10_000, slippage_price=0.0)
    await broker.mark_price(2350.0, 0.20, T0)
    signal = make_signal(direction=BEARISH, entry=2350.0, stop_loss=2353.0)
    request = _request(
        units=40,
        entry_price=signal.entry.price,
        stop_loss=signal.stop_loss.price,
        take_profit=signal.take_profits.tp2.price,
        metadata={
            "tp1": signal.take_profits.tp1.price,
            "tp2": signal.take_profits.tp2.price,
            "tp3": signal.take_profits.tp3.price if signal.take_profits.tp3 else None,
        },
    )
    fill = await broker.submit_order(request, 2350.0, 0.0)
    manager = PositionManager(settings, broker)
    events = await manager.on_price(signal.take_profits.tp1.price, 0.0, [], T0 + timedelta(minutes=20))
    assert any(event.kind == "TP1_HIT" for event in events)
    position = broker.get_position(fill.trade_id)
    assert position.units < position.initial_units
    assert position.stop_loss <= signal.entry.price  # moved to breakeven-or-better


# ------------------------------------------------------------------ order manager
@pytest.mark.asyncio
async def test_semi_auto_queues_then_approves():
    settings = Settings()
    settings = replace(
        settings,
        bot=replace(settings.bot, execution_mode="SEMI_AUTO"),
        execution=replace(settings.execution, paper_auto_approve=False),
    )
    broker = PaperBroker(starting_equity=10_000)
    await broker.mark_price(2350.0, 0.20, T0)
    orders = OrderManager(settings=settings, broker=broker)
    signal = make_signal(direction=BEARISH, entry=2350.0, stop_loss=2352.7, timestamp=T0)
    queued = await orders.submit(signal, 2350.0, 0.20, T0)
    assert queued.status == "PENDING_APPROVAL"
    assert orders.get_pending_approvals()
    approved = await orders.approve(signal.signal_id, 2350.0, 0.20, T0)
    assert approved.success
    # A sell limit requires the BID to touch, not just the midpoint.
    await broker.mark_price(2350.2, 0.20, T0 + timedelta(seconds=1))
    assert broker.get_open_positions()


@pytest.mark.asyncio
async def test_exposure_counts_resting_orders():
    settings = Settings()
    settings = replace(
        settings,
        bot=replace(settings.bot, execution_mode="FULL_AUTO"),
        execution=replace(settings.execution, paper_auto_approve=True),
    )
    broker = PaperBroker(starting_equity=10_000)
    await broker.mark_price(2340.0, 0.20, T0)
    orders = OrderManager(settings=settings, broker=broker)
    assert await orders.get_open_trade_count() == 0
    signal = make_signal(direction=BEARISH, entry=2355.0, stop_loss=2357.0, timestamp=T0)
    result = await orders.submit(signal, 2340.0, 0.20, T0)
    assert result.status in {"PENDING", "FILLED"}
    assert await orders.get_open_trade_count() >= 1


# -------------------------------------------------------------------- trailing
def test_trail_stop_only_moves_in_the_profit_direction():
    plan = next_stop(
        entry=2350.0,
        current_stop=2353.0,
        direction=BEARISH,
        market_price=2340.0,
        tp1=2345.5,
        tp2=2341.0,
        tp1_hit=True,
        tp2_hit=False,
    )
    assert plan is not None
    assert plan.stop_loss < 2353.0  # a short's stop only ratchets down
    assert plan.stage == "BREAKEVEN"


def test_trail_stop_returns_none_before_tp1():
    assert (
        next_stop(
            entry=2350.0,
            current_stop=2353.0,
            direction=BEARISH,
            market_price=2340.0,
            tp1=2345.5,
            tp2=2341.0,
            tp1_hit=False,
            tp2_hit=False,
        )
        is None
    )


@pytest.mark.asyncio
async def test_profitable_stop_is_labelled_a_trail_not_a_loss():
    """A stop trailed above the entry must not be journalled as STOP_LOSS."""

    settings = Settings()
    broker = PaperBroker(starting_equity=10_000, slippage_price=0.0)
    await broker.mark_price(2350.0, 0.20, T0)
    await broker.submit_order(_request(units=40), 2350.0, 0.0)
    await broker.mark_price(2344.0, 0.0, T0 + timedelta(minutes=25))
    await broker.modify_stop(broker.get_open_positions()[0].trade_id, 2345.0)
    manager = PositionManager(settings, broker)
    events = await manager.on_price(2346.0, 0.0, [], T0 + timedelta(minutes=30))
    assert events
    record = broker.get_history()[0]
    assert record.realized_pnl > 0
    assert record.metadata["close_reason"] in {"TRAIL_STOP", "BREAKEVEN_STOP"}
