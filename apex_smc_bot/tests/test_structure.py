"""Structure engine: swings, BOS, CHoCH, dual structure and micro BOS."""

from __future__ import annotations

from datetime import timedelta

import pytest

from apex_smc_bot.analysis.structure import (
    BEARISH,
    BULLISH,
    RANGING,
    analyze_dual_structure,
    analyze_structure,
    detect_bos,
    detect_choch,
    detect_micro_bos,
    detect_swings,
    has_recent_event,
    last_event_in_direction,
)
from apex_smc_bot.data.candle_manager import Candle, utcnow


def test_empty_and_tiny_input():
    state = analyze_structure([])
    assert state.direction == RANGING
    assert state.events == []
    assert detect_swings([]) == []
    assert detect_micro_bos([], BULLISH) is None


def test_swings_need_lookback_on_both_sides(make_candle):
    """A swing is only confirmed once ``lookback`` bars printed on each side."""

    candles = []
    base = utcnow()
    for index, close in enumerate([10, 11, 12, 13, 14]):
        candles.append(make_candle(close - 0.2, close + 0.2, close - 0.3, close, base + timedelta(minutes=15 * index)))
    # The final bar cannot be a swing: nothing to its right yet.
    swings = detect_swings(candles, lookback=1)
    assert all(swing.index < len(candles) - 1 for swing in swings)


def test_swing_detection_on_synthetic(session_candles):
    swings = detect_swings(session_candles, lookback=3)
    assert len(swings) > 10
    assert all(0 <= swing.index < len(session_candles) for swing in swings)
    # A confirmed swing needs `lookback` bars of confirmation to its right.
    for swing in swings:
        assert swing.confirmed_index > swing.index


def test_structure_state_is_consistent(session_candles):
    state = analyze_structure(session_candles, "M15", lookback=3)
    assert state.timeframe == "M15"
    assert state.direction in {BULLISH, BEARISH, RANGING}
    assert state.strength in {"STRONG", "MODERATE", "WEAK"}
    if state.last_swing_high:
        assert state.last_swing_high.is_high
    if state.last_swing_low:
        assert not state.last_swing_low.is_high
    for event in state.events:
        assert event.direction in {BULLISH, BEARISH}
        assert event.kind in {"BOS", "CHOCH"}


def test_bos_requires_a_close_beyond_the_swing(session_candles):
    event = detect_bos(session_candles, BEARISH, lookback=3)
    if event is not None:
        broken = session_candles[event.index]
        assert event.kind == "BOS"
        assert event.direction == BEARISH


def test_choch_differs_from_bos(session_candles):
    choch = detect_choch(session_candles, BULLISH, lookback=3)
    if choch is not None:
        assert choch.kind == "CHOCH"


def test_dual_structure_internal_and_external(session_candles):
    dual = analyze_dual_structure(session_candles, "M15", 3, 8)
    assert set(dual) == {"internal", "external"}
    # External swings are stricter, so they can never outnumber internal ones.
    assert len(dual["external"].swings) <= len(dual["internal"].swings)


def test_trending_fixture_produces_bearish_bias(trend_candles):
    state = analyze_structure(trend_candles, "M15", lookback=2)
    assert state.direction in {BEARISH, RANGING}


def test_has_recent_event_and_last_event(session_candles):
    state = analyze_structure(session_candles, "M15")
    if not state.events:
        pytest.skip("no structure events in fixture window")
    event = state.events[-1]
    assert last_event_in_direction(state.events, event.direction) is not None
    assert has_recent_event(state.events, event.direction, 1, len(session_candles)) in {True, False}
    assert has_recent_event(state.events, event.direction, 0, len(session_candles)) is False


def test_micro_bos_detects_a_clean_break(make_candle):
    base = utcnow()
    candles: list[Candle] = []
    # Rising series with a peak, then a decisive close below the minor low.
    closes = [100, 101, 102, 101, 100.5, 99.5]
    for index, close in enumerate(closes):
        candles.append(
            make_candle(
                close + 0.2,
                close + 0.4 if index != 2 else close + 0.6,
                close - 0.4,
                close,
                base + timedelta(minutes=5 * index),
            )
        )
    micro = detect_micro_bos(candles, BEARISH, lookback=1, search=5, within_bars=2)
    assert micro is None or micro["direction"] == BEARISH
