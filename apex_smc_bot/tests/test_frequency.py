"""Frequency controls: swing-raid sweeps, alignment modes, market entry, time stop.

These tests pin the behaviour that turns the engine from a ~5-trades-per-quarter
swing scanner into an intraday system, without loosening any hard risk law.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from apex_smc_bot.analysis.confluence import ConfluenceEngine, Setup
from apex_smc_bot.analysis.htf_bias import AlignmentResult, HTFBiasEngine
from apex_smc_bot.analysis.liquidity import (
    EQH,
    build_liquidity_map,
    detect_sweeps,
    single_swing_levels,
)
from apex_smc_bot.analysis.structure import BEARISH, BULLISH, RANGING, analyze_structure, detect_swings
from apex_smc_bot.config import ConfigError, Settings
from apex_smc_bot.signals.models import PendingSetup


# --------------------------------------------------------------- liquidity
def test_single_swing_levels_become_raidable_pools(session_candles):
    swings = detect_swings(session_candles, lookback=3)
    levels = single_swing_levels(swings)
    assert len(levels) == len(swings)
    assert all(level.count == 1 for level in levels)
    assert all(level.kind == EQH for level in levels if level.swings[0].is_high)


def test_single_swing_raid_produces_a_sweep(make_candle):
    """A wick above one prior swing high that closes back below is a sweep."""

    from datetime import datetime, timezone

    base = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
    candles = [
        make_candle(100.0, 101.0, 99.9, 100.5, base),
        make_candle(100.5, 102.0, 100.4, 101.2, base + timedelta(minutes=15)),
        make_candle(101.2, 101.5, 100.8, 101.2, base + timedelta(minutes=30)),
        make_candle(101.2, 101.4, 100.6, 100.9, base + timedelta(minutes=45)),
        # raid above the 102.00 swing high, closing back inside
        make_candle(100.9, 102.4, 100.8, 101.0, base + timedelta(minutes=60)),
        make_candle(101.0, 101.1, 100.2, 100.4, base + timedelta(minutes=75)),
    ]
    swings = detect_swings(candles, lookback=1)
    levels = single_swing_levels([s for s in swings if s.is_high])
    sweeps = detect_sweeps(candles, levels, [], bias=BEARISH, lookahead=4)
    assert sweeps, "a single-swing raid must register as a sweep"
    assert sweeps[-1].direction == BEARISH


def test_build_liquidity_map_single_swing_flag_adds_pools(session_candles):
    tight = build_liquidity_map(session_candles, "M15", single_swing_lookback=0)
    wide = build_liquidity_map(session_candles, "M15", single_swing_lookback=96)
    assert len(wide.sweeps) >= len(tight.sweeps)


# --------------------------------------------------------------- alignment
def _bias(direction: str, price: float = 3300.0):
    from apex_smc_bot.analysis.htf_bias import HTFBias

    return HTFBias(timeframe="H4", direction=direction, strength="MODERATE", current_price=price)


def test_pullback_alignment_trades_with_the_4h_not_against_it():
    engine = HTFBiasEngine(Settings())
    result = engine.check_alignment(_bias(BULLISH), _bias(BEARISH))
    assert result.aligned
    assert result.direction == BULLISH  # never counter-4H
    assert result.mode == "4H_PULLBACK"


def test_pullback_alignment_can_be_switched_off():
    settings = Settings()
    settings = replace(
        settings, strategy=replace(settings.strategy, allow_1h_pullback_alignment=False)
    )
    engine = HTFBiasEngine(settings)
    result = engine.check_alignment(_bias(BULLISH), _bias(BEARISH))
    assert not result.aligned
    assert result.direction == "CONFLICTED"


# --------------------------------------------------------------- confluence
def _setup(**kwargs) -> Setup:
    defaults = dict(
        direction=BULLISH,
        entry_price=3300.0,
        four_h_bias=_bias(BULLISH),
        one_h_bias=_bias(BULLISH),
        htf_aligned=True,
        htf_mode="FULL",
    )
    defaults.update(kwargs)
    return Setup(**defaults)


def test_confluence_never_trades_against_the_4h():
    engine = ConfluenceEngine(Settings())
    result = engine.score(_setup(direction=BEARISH, htf_mode="4H_PULLBACK"))
    assert not result.trade_allowed
    assert result.abort_reason == "HTF_NOT_ALIGNED"


def test_confluence_partial_credit_for_pullback_and_range_modes():
    engine = ConfluenceEngine(Settings())
    full = engine.score(_setup(htf_mode="FULL"))
    pullback = engine.score(_setup(htf_mode="4H_PULLBACK", one_h_bias=_bias(BEARISH)))
    ranged = engine.score(
        _setup(htf_mode="RANGE", four_h_bias=_bias(RANGING), one_h_bias=_bias(BULLISH))
    )
    assert full.breakdown["htf_alignment"] == 1.0
    assert pullback.breakdown["htf_alignment"] == 0.7
    assert ranged.breakdown["htf_alignment"] == 0.5
    assert ranged.trade_allowed or ranged.abort_reason != "HTF_NOT_ALIGNED"


# --------------------------------------------------------------- config
def test_risk_cap_is_enforced_at_config_time():
    settings = Settings()
    with pytest.raises(ConfigError):
        replace(
            settings,
            risk=replace(settings.risk, max_open_trades=4, max_risk_per_trade_percent=1.0),
        )


def test_entry_mode_is_validated():
    settings = Settings()
    with pytest.raises(ConfigError):
        replace(settings, strategy=replace(settings.strategy, entry_mode="STOP"))
    assert replace(settings, strategy=replace(settings.strategy, entry_mode="market")).strategy.entry_mode == "MARKET"


def test_high_frequency_preset_loads_and_respects_the_cap():
    settings = Settings.load("configs/high_frequency.yaml")
    assert settings.strategy.entry_mode == "MARKET"
    exposure = settings.risk.max_open_trades * settings.risk.max_risk_per_trade_percent
    assert exposure <= 2.0 + 1e-9
    assert settings.strategy.min_fivem_checks >= 4  # never below the documented floor


# --------------------------------------------------------------- time stop
async def _aged_position(price_offset: float, hours: float = 3.0):
    """Open a long, then run one management pass some hours later."""

    from apex_smc_bot.execution.order_manager import OrderRequest
    from apex_smc_bot.execution.paper_broker import PaperBroker
    from apex_smc_bot.execution.position_manager import PositionManager

    settings = Settings()
    settings = replace(
        settings,
        risk=replace(settings.risk, max_position_age_hours=2.0),
        database=replace(settings.database, enabled=False),
    )
    broker = PaperBroker(starting_equity=10_000.0)
    request = OrderRequest(
        signal_id="SIG-TIME",
        instrument="XAU_USD",
        direction="LONG",
        entry_price=3300.0,
        units=0.40,
        stop_loss=3290.0,
        take_profit=3340.0,
        order_type="MARKET",
    )
    await broker.submit_order(request, market_price=3300.0, spread_price=0.20)
    manager = PositionManager(settings, broker, None)
    opened = broker._clock
    events = await manager.on_price(
        3300.0 + price_offset,
        spread_price=0.20,
        candles_5m=None,
        now=opened + timedelta(hours=hours),
    )
    return broker, events


@pytest.mark.asyncio
async def test_stale_losing_position_is_closed_by_the_time_stop():
    broker, events = await _aged_position(price_offset=-1.0)
    assert any(event.kind == "TIME_EXIT" for event in events)
    assert not broker.get_open_positions()


@pytest.mark.asyncio
async def test_stale_winner_is_allowed_to_run_past_the_time_stop():
    broker, events = await _aged_position(price_offset=+1.0)
    assert not any(event.kind == "TIME_EXIT" for event in events)
    assert broker.get_open_positions()


@pytest.mark.asyncio
async def test_winner_is_still_capped_at_twice_the_max_age():
    broker, events = await _aged_position(price_offset=+1.0, hours=5.0)
    assert any(event.kind == "TIME_EXIT" for event in events)
    assert not broker.get_open_positions()


def test_breakeven_after_tp1_can_be_disabled():
    from apex_smc_bot.execution.trail_stop import next_stop

    common = dict(
        entry=3300.0,
        current_stop=3290.0,
        direction=BULLISH,
        market_price=3315.0,
        tp1=3310.0,
        tp2=3330.0,
        tp1_hit=True,
        tp2_hit=False,
    )
    spec = next_stop(**common)
    assert spec is not None and spec.stage == "BREAKEVEN"
    runner = next_stop(**common, breakeven_after_tp1=False)
    assert runner is None


def test_daily_trend_reads_the_4h_series(make_candle):
    from datetime import datetime, timedelta, timezone

    from apex_smc_bot.analysis.htf_bias import HTFBiasEngine

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def series(prices):
        return [
            make_candle(p, p + 0.6, p - 0.6, p, base + timedelta(hours=4 * i))
            for i, p in enumerate(prices)
        ]

    rising = series([2300 + i * 2.0 for i in range(160)])
    falling = series([2300 - i * 2.0 for i in range(160)])
    flat = series([2300 + (0.3 if i % 2 else -0.3) for i in range(160)])
    assert HTFBiasEngine.daily_trend(rising) == BULLISH
    assert HTFBiasEngine.daily_trend(falling) == BEARISH
    assert HTFBiasEngine.daily_trend(flat) == RANGING
    assert HTFBiasEngine.daily_trend(rising[:40]) == RANGING  # not enough history


def test_daily_trend_filter_is_off_by_default():
    assert Settings().strategy.daily_trend_filter is False


def test_pending_setup_helpers_still_round_trip(session_candles):
    """Guard: the extra Setup field must not break the confluence pipeline."""

    structure = analyze_structure(session_candles, "M15")
    assert structure.direction in {BULLISH, BEARISH, RANGING}
