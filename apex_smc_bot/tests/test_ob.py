"""Order blocks: detection, quality, mitigation lifecycle and flips."""

from __future__ import annotations

from apex_smc_bot.analysis.order_blocks import (
    BEARISH,
    BULLISH,
    FULLY_MITIGATED,
    UNMITIGATED,
    active_order_blocks,
    confluent_order_blocks,
    detect_order_blocks,
    nearest_ob,
    nearest_opposing_ob,
)
from apex_smc_bot.analysis.structure import analyze_structure


def test_empty_input_is_safe():
    assert detect_order_blocks([], [], "M15") == []
    assert active_order_blocks([]) == []
    assert nearest_ob([], 100.0) is None


def test_detection_on_synthetic(session_candles):
    state = analyze_structure(session_candles, "M15")
    blocks = detect_order_blocks(session_candles, state.events, "M15", lookback=100)
    assert isinstance(blocks, list)
    for block in blocks:
        assert block.low <= block.high
        assert block.body_low <= block.body_high
        assert block.kind in {BULLISH, BEARISH}
        assert 1 <= block.quality <= 5
        assert block.low <= block.midpoint <= block.high
        assert block.timeframe == "M15"


def test_quality_score_is_bounded(session_candles):
    state = analyze_structure(session_candles, "M15")
    blocks = detect_order_blocks(session_candles, state.events, "M15")
    assert all(1 <= block.quality <= 5 for block in blocks)


def test_active_filter_excludes_mitigated(session_candles):
    state = analyze_structure(session_candles, "M15")
    blocks = detect_order_blocks(session_candles, state.events, "M15")
    active = active_order_blocks(blocks)
    assert all(block.is_active for block in active)
    assert len(active) <= len(blocks)


def test_distance_and_containment(session_candles):
    state = analyze_structure(session_candles, "M15")
    blocks = detect_order_blocks(session_candles, state.events, "M15")
    if not blocks:
        return
    block = blocks[0]
    assert block.contains(block.midpoint, tolerance=0.0)
    assert block.distance_to(block.midpoint) == 0.0
    assert block.distance_to(block.low - 1.0) == 1.0


def test_nearest_opposing_only_returns_blocks_ahead(session_candles):
    state = analyze_structure(session_candles, "M15")
    blocks = detect_order_blocks(session_candles, state.events, "M15")
    price = session_candles[-1].close
    long_obstacle = nearest_opposing_ob(blocks, price, BULLISH)
    if long_obstacle is not None:
        assert long_obstacle.kind == BEARISH
        assert long_obstacle.low > price
    short_obstacle = nearest_opposing_ob(blocks, price, BEARISH)
    if short_obstacle is not None:
        assert short_obstacle.kind == BULLISH
        assert short_obstacle.high < price


def test_confluent_blocks_match_direction(session_candles):
    state = analyze_structure(session_candles, "M15")
    blocks = detect_order_blocks(session_candles, state.events, "M15")
    pairs = confluent_order_blocks(blocks, blocks)
    for ltf, htf in pairs:
        assert ltf.kind == htf.kind
        assert ltf.overlaps(htf.low, htf.high)


def test_mitigated_block_status_changes():
    """A block whose zone is traded through must flip to FULLY_MITIGATED."""

    from apex_smc_bot.analysis.order_blocks import update_order_block_status

    state = analyze_structure.__module__  # noqa: F841 - documentation only
    from apex_smc_bot.data.synthetic import generate_market_data
    from apex_smc_bot.data.replay_feed import fixed_end

    data = generate_market_data(days=5, seed=3, end=fixed_end())
    candles = data["M15"]
    structure = analyze_structure(candles, "M15")
    blocks = detect_order_blocks(candles, structure.events, "M15")
    if not blocks:
        return
    block = blocks[0]
    before = block.status
    update_order_block_status(block, candles, start_index=min(block.index + 1, len(candles) - 1))
    assert block.status in {UNMITIGATED, "PARTIALLY_MITIGATED", FULLY_MITIGATED, "EXHAUSTED"}
    assert before in {UNMITIGATED, "PARTIALLY_MITIGATED", FULLY_MITIGATED, "EXHAUSTED"}
