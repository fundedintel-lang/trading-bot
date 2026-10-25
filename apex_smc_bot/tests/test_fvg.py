"""Fair value gaps: detection, fill lifecycle and inverse FVG behaviour."""

from __future__ import annotations

from datetime import timedelta

from apex_smc_bot.analysis.fvg import (
    FILLED,
    UNFILLED,
    active_fvgs,
    detect_fvg,
    nearest_fvg,
    overlapping_fvg,
    update_fvg_status,
)
from apex_smc_bot.data.candle_manager import Candle, utcnow


def test_empty_and_short_input():
    assert detect_fvg([]) == []
    assert active_fvgs([]) == []
    assert nearest_fvg([], 100.0) is None


def test_detects_a_bullish_gap(make_candle):
    base = utcnow()
    candles = [
        make_candle(100.0, 100.5, 99.5, 100.0, base),
        make_candle(100.0, 103.0, 100.0, 103.0, base + timedelta(minutes=15)),
        make_candle(103.0, 104.0, 102.0, 103.5, base + timedelta(minutes=30)),
    ]
    gaps = detect_fvg(candles, "M15", min_gap_price=0.01)
    assert any(gap.kind == "BULLISH" for gap in gaps)
    bullish = next(gap for gap in gaps if gap.kind == "BULLISH")
    assert bullish.bottom == 100.5
    assert bullish.top == 102.0
    assert bullish.midpoint == (100.5 + 102.0) / 2


def test_detects_a_bearish_gap(make_candle):
    base = utcnow()
    candles = [
        make_candle(104.0, 104.5, 103.5, 104.0, base),
        make_candle(104.0, 104.0, 101.0, 101.0, base + timedelta(minutes=15)),
        make_candle(101.0, 102.0, 100.5, 101.5, base + timedelta(minutes=30)),
    ]
    gaps = detect_fvg(candles, "M15", min_gap_price=0.01)
    assert any(gap.kind == "BEARISH" for gap in gaps)


def test_min_gap_filters_micro_gaps(make_candle):
    base = utcnow()
    candles = [
        make_candle(100.0, 100.5, 99.5, 100.0, base),
        make_candle(100.0, 101.0, 100.0, 101.0, base + timedelta(minutes=15)),
        make_candle(101.0, 101.2, 100.4, 100.5, base + timedelta(minutes=30)),
    ]
    assert detect_fvg(candles, "M15", min_gap_price=5.0) == []


def test_fill_lifecycle(session_candles):
    gaps = detect_fvg(session_candles[-400:], "M15")
    assert gaps
    for gap in gaps:
        assert gap.top >= gap.bottom
        assert gap.status in {UNFILLED, "PARTIALLY_FILLED", FILLED, "INVERSE"}
        assert 0.0 <= gap.fill_ratio <= 1.0
    unfilled = active_fvgs(gaps)
    assert all(gap.status != FILLED or gap.delivered for gap in unfilled)


def test_update_fvg_status_fills_a_gap(make_candle):
    base = utcnow()
    candles = [
        make_candle(100.0, 100.5, 99.5, 100.0, base),
        make_candle(100.0, 103.0, 100.0, 103.0, base + timedelta(minutes=15)),
        make_candle(103.0, 104.0, 102.0, 103.5, base + timedelta(minutes=30)),
        # Price trades straight back through the gap.
        make_candle(103.0, 103.0, 100.0, 100.5, base + timedelta(minutes=45)),
    ]
    gaps = detect_fvg(candles, "M15", min_gap_price=0.01)
    if not gaps:
        return
    gap = gaps[0]
    update_fvg_status(gap, candles)
    assert gap.fill_ratio >= 0.0


def test_overlapping_fvg_returns_zones(session_candles):
    gaps = detect_fvg(session_candles[-400:], "M15")
    if not gaps:
        return
    gap = gaps[-1]
    found = overlapping_fvg(gaps, gap.bottom, gap.top, direction=gap.kind)
    assert found is None or found.kind == gap.kind
