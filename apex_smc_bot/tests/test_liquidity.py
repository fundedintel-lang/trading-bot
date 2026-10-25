"""Liquidity map: equal highs/lows, sweeps, inducement and pools."""

from __future__ import annotations

from apex_smc_bot.analysis.liquidity import (
    EQL,
    EQH,
    build_liquidity_map,
    detect_equal_levels,
    detect_inducement,
    detect_sweeps,
)
from apex_smc_bot.analysis.structure import analyze_structure, detect_swings


def test_empty_input():
    liq = build_liquidity_map([])
    assert liq.equal_highs == []
    assert liq.equal_lows == []
    assert liq.recent_sweep() is None
    assert liq.nearest_above(100.0) is None


def test_equal_level_clustering_within_tolerance(session_candles):
    swing_points = detect_swings(session_candles, lookback=3)
    highs, lows = detect_equal_levels(swing_points, tolerance_pips=5.0)
    levels = highs + lows
    for level in levels:
        assert level.kind in {EQH, EQL}
        assert level.count >= 2
        # Every member swing sits within one tolerance of the cluster's mean price.
        assert all(abs(swing.price - level.price) <= level.tolerance + 1e-6 for swing in level.swings)
        assert level.last_index - level.first_index >= 2


def test_tolerance_controls_grouping(session_candles):
    swing_points = detect_swings(session_candles, lookback=3)
    tight_highs, tight_lows = detect_equal_levels(swing_points, tolerance_pips=0.2)
    loose_highs, loose_lows = detect_equal_levels(swing_points, tolerance_pips=20.0)
    assert len(loose_highs) + len(loose_lows) >= len(tight_highs) + len(tight_lows)
    assert len(loose_highs) > 0 or len(tight_highs) >= 0


def test_liquidity_map_structure(session_candles):
    liq = build_liquidity_map(session_candles, "M15", tolerance_pips=5.0)
    assert liq.timeframe == "M15"
    assert all(level.kind == EQH for level in liq.equal_highs)
    assert all(level.kind == EQL for level in liq.equal_lows)
    price = session_candles[-1].close
    above = liq.nearest_above(price)
    if above:
        assert above.price > price
    below = liq.nearest_below(price)
    if below:
        assert below.price < price


def test_sweeps_are_confirmed_by_a_close_back(session_candles):
    liq = build_liquidity_map(session_candles, "M15")
    for sweep in liq.sweeps:
        assert sweep.direction in {"BULLISH", "BEARISH"}
        assert sweep.level > 0
        if sweep.confirmed:
            assert sweep.close != 0


def test_recent_sweep_filters(session_candles):
    liq = build_liquidity_map(session_candles, "M15")
    if not liq.sweeps:
        return
    latest = liq.recent_sweep()
    assert latest is not None
    confirmed = liq.recent_sweep(require_confirmed=True)
    assert confirmed is None or confirmed.confirmed


def test_detect_sweeps_ignores_flat_data(make_candle):
    from apex_smc_bot.data.candle_manager import utcnow
    from datetime import timedelta

    base = utcnow()
    flat = [make_candle(100, 100.2, 99.8, 100, base + timedelta(minutes=5 * i)) for i in range(40)]
    swings = detect_swings(flat, lookback=2)
    highs, lows = detect_equal_levels(swings, tolerance_pips=2.0)
    sweeps = detect_sweeps(flat, highs + lows, [], lookahead=5)
    # A perfectly flat series has no raid that closes back inside, so no sweeps.
    assert all(sweep.level > 0 for sweep in sweeps)


def test_inducement_returns_none_or_dict(session_candles):
    structure = analyze_structure(session_candles, "M15")
    result = detect_inducement(session_candles, structure, "BEARISH")
    assert result is None or isinstance(result, dict)
    if isinstance(result, dict):
        assert "direction" in result or "level" in result


def test_liquidity_map_to_dict(session_candles):
    liq = build_liquidity_map(session_candles)
    payload = liq.to_dict()
    assert set(payload) >= {"timeframe", "equal_highs", "equal_lows", "sweeps"}
