"""Risk layer: position sizing, spread guard, drawdown breaker, news blackout."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from apex_smc_bot.config import RiskConfig, Settings
from apex_smc_bot.execution.lot_calculator import LotSizeError, calculate_position, calculate_position_size
from apex_smc_bot.risk.drawdown_monitor import DrawdownMonitor, DrawdownState
from apex_smc_bot.risk.news_filter import NewsFilter
from apex_smc_bot.risk.risk_engine import RiskEngine
from apex_smc_bot.risk.spread_monitor import SpreadMonitor
from apex_smc_bot.tests.factories import make_session


# --------------------------------------------------------------------- sizing
def test_position_size_matches_the_law():
    """$10,000 at 1% with a 27-pip stop (10 $/pip/lot) → 0.37 lots."""

    lots = calculate_position_size(10_000, 1.0, 27.0, pip_value_per_lot=10.0)
    assert lots == 0.37
    sizing = calculate_position(10_000, 1.0, 2350.0, 2347.3, RiskConfig())
    assert sizing.lot_size == pytest.approx(0.37, abs=0.01)
    assert sizing.risk_amount_usd == pytest.approx(100.0, abs=1.0)
    assert sizing.sl_pips == pytest.approx(27.0, abs=0.1)


def test_tiny_account_or_huge_stop_is_unsizeable():
    assert calculate_position_size(50.0, 1.0, 50.0) == 0.0
    with pytest.raises(LotSizeError):
        calculate_position_size(-1.0, 1.0, 20.0)
    with pytest.raises(LotSizeError):
        calculate_position_size(10_000, 0.0, 20.0)


def test_lot_floor_respects_the_risk_budget():
    """The floor-rounded lot must never risk more than the budget by >10%."""

    for equity in (500, 2_000, 10_000, 100_000):
        for sl_pips in (5, 15, 27, 50):
            lots = calculate_position_size(equity, 1.0, sl_pips)
            if lots == 0.0:
                continue
            risk = lots * sl_pips * 10.0
            assert risk <= equity * 0.01 * 1.10 + 1e-9


# --------------------------------------------------------------------- spread
def test_spread_guard_blocks_wide_and_spiking_spreads():
    monitor = SpreadMonitor(RiskConfig(), window=10, spike_multiplier=2.0, pause_minutes=15)
    now = _t0()
    for _ in range(6):
        state = monitor.update(0.20, now)
    assert state.allowed
    wide = monitor.update(RiskConfig().max_spread_pips * 10 + 1.0, now + timedelta(minutes=1))
    assert not wide.allowed
    assert wide.reason


def test_spread_monitor_rejects_negative_input():
    monitor = SpreadMonitor(RiskConfig())
    with pytest.raises(ValueError):
        monitor.update(-0.1)


# ------------------------------------------------------------------- drawdown
def test_daily_drawdown_breaker_halts_for_24h():
    monitor = DrawdownMonitor(RiskConfig(), database=None, state=DrawdownState())
    now = _t0()
    monitor.update_equity(10_000, now)
    decision = monitor.update_equity(9_400, now + timedelta(hours=1))  # -6%
    assert decision.breaker_triggered
    assert not decision.allow_trading
    blocked = monitor.decision(now + timedelta(hours=2))
    assert not blocked.allow_trading
    assert "CIRCUIT_BREAKER" in blocked.reason


def test_reduced_risk_band_before_the_breaker():
    monitor = DrawdownMonitor(RiskConfig(), database=None, state=DrawdownState())
    now = _t0()
    monitor.update_equity(10_000, now)
    decision = monitor.update_equity(9_700, now + timedelta(minutes=30))  # -3%
    assert decision.allow_trading
    assert decision.reduced_risk
    assert decision.risk_multiplier == pytest.approx(RiskConfig().reduced_risk_percent / RiskConfig().max_risk_per_trade_percent, abs=0.01)


def test_two_losses_arm_the_cooldown():
    monitor = DrawdownMonitor(RiskConfig(), database=None, state=DrawdownState())
    now = _t0()
    monitor.update_equity(10_000, now)
    monitor.record_trade_result(-100, now)
    monitor.record_trade_result(-100, now + timedelta(minutes=30))
    decision = monitor.decision(now + timedelta(minutes=45))
    assert not decision.allow_trading
    assert "LOSS_COOLDOWN" in decision.reason
    # A win clears the streak.
    monitor.record_trade_result(+100, now + timedelta(hours=5))
    assert monitor.decision(now + timedelta(hours=5)).allow_trading


def test_day_rollover_resets_the_daily_baseline():
    monitor = DrawdownMonitor(RiskConfig(), database=None, state=DrawdownState())
    day_one = _t0()
    monitor.update_equity(10_000, day_one)
    monitor.update_equity(9_800, day_one + timedelta(hours=2))
    assert monitor.state.daily_drawdown_percent == pytest.approx(2.0, abs=1e-6)
    monitor.update_equity(9_800, day_one + timedelta(days=1))
    assert monitor.state.daily_drawdown_percent == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------- news
def test_news_filter_fails_open_without_a_calendar():
    config = replace(Settings().news, enabled=True, fail_closed=False)
    filt = NewsFilter(config)
    check = filt.check(_t0())
    assert check.safe


def test_news_filter_blocks_inside_the_window():
    config = replace(Settings().news, enabled=True, fail_closed=False, block_minutes_before=30, block_minutes_after=30)
    filt = NewsFilter(config)
    now = _t0()
    filt.set_events(
        [
            {
                "title": "FOMC Statement",
                "timestamp": now + timedelta(minutes=10),
                "impact": "HIGH",
                "currency": "USD",
            }
        ],
        fetched_at=now,
    )
    check = filt.check(now)
    assert not check.safe
    assert check.events


# ---------------------------------------------------------------- risk engine
def test_risk_engine_blocks_a_wide_spread():
    settings = Settings()
    engine = RiskEngine(settings, database=None)
    decision = engine.evaluate(
        now=_t0(),
        equity=10_000,
        open_trade_count=0,
        session=make_session(),
        spread_price=settings.risk.max_spread_pips * settings.risk.pip_size * 5,
    )
    assert not decision.allow
    assert "SPREAD" in decision.summary().upper()


def test_risk_engine_blocks_when_trade_slots_are_full():
    settings = Settings()
    engine = RiskEngine(settings, database=None)
    decision = engine.evaluate(
        now=_t0(),
        equity=10_000,
        open_trade_count=settings.risk.max_open_trades,
        session=make_session(),
        spread_price=0.15,
    )
    assert not decision.allow
    assert "max_open_trades" in decision.checks or "TRADE" in decision.summary().upper()


def test_risk_engine_blocks_an_untradeable_session():
    settings = Settings()
    engine = RiskEngine(settings, database=None)
    decision = engine.evaluate(
        now=_t0(),
        equity=10_000,
        open_trade_count=0,
        session=make_session(name="DEAD_ZONE", tradeable=False, is_kill_zone=False),
        spread_price=0.15,
    )
    assert not decision.allow


def test_risk_engine_allows_a_clean_setup():
    settings = Settings()
    engine = RiskEngine(settings, database=None)
    decision = engine.evaluate(
        now=_t0(),
        equity=10_000,
        open_trade_count=0,
        session=make_session(),
        spread_price=0.15,
    )
    assert decision.allow
    assert 0 < decision.risk_percent <= settings.risk.max_risk_per_trade_percent


def test_risk_engine_refuses_equity_below_the_floor():
    settings = Settings()
    engine = RiskEngine(settings, database=None)
    decision = engine.evaluate(
        now=_t0(),
        equity=settings.risk.min_equity / 2,
        open_trade_count=0,
        session=make_session(),
        spread_price=0.15,
    )
    assert not decision.allow


def _t0():
    from datetime import datetime, timezone

    return datetime(2026, 10, 7, 13, 0, tzinfo=timezone.utc)
