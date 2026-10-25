"""Order Blocks (OB) with quality scoring, mitigation lifecycle and flip zones.

An order block is the last opposing candle before a displacement/BOS.  We keep:

* the **full wick zone** (conservative entry) and the **body zone** (aggressive),
* a 1-5 quality score (freshness, displacement, FVG confluence, HTF alignment,
  proximity to an HTF level),
* a live mitigation state: UNMITIGATED → PARTIALLY_MITIGATED → EXHAUSTED /
  FULLY_MITIGATED,
* an inverted **breaker / flip** zone once the OB is closed through.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

from apex_smc_bot.analysis.fvg import FVG, active_fvgs
from apex_smc_bot.analysis.structure import StructureEvent
from apex_smc_bot.data.candle_manager import Candle, atr as compute_atr, to_utc

BULLISH = "BULLISH"
BEARISH = "BEARISH"

UNMITIGATED = "UNMITIGATED"
PARTIALLY_MITIGATED = "PARTIALLY_MITIGATED"
FULLY_MITIGATED = "FULLY_MITIGATED"
EXHAUSTED = "EXHAUSTED"


@dataclass
class OrderBlock:
    """A single order block zone."""

    kind: str
    high: float
    low: float
    body_high: float
    body_low: float
    index: int
    timestamp: datetime
    timeframe: str
    bos_index: int
    bos_timestamp: datetime
    status: str = UNMITIGATED
    quality: int = 3
    test_count: int = 0
    is_flip: bool = False
    is_breaker: bool = False
    origin: str = "BOS"
    mitigated_at: datetime | None = None
    notes: list[str] | None = None

    def __post_init__(self) -> None:
        if self.high < self.low:
            self.high, self.low = self.low, self.high
        if self.body_high < self.body_low:
            self.body_high, self.body_low = self.body_low, self.body_high
        if self.kind not in {BULLISH, BEARISH}:
            raise ValueError(f"Invalid order block kind {self.kind!r}")
        self.quality = max(1, min(5, int(self.quality)))

    # ------------------------------------------------------------------ geometry
    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def height(self) -> float:
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        return self.kind == BULLISH

    @property
    def is_active(self) -> bool:
        return self.status in {UNMITIGATED, PARTIALLY_MITIGATED} and self.test_count < 2

    @property
    def is_valid(self) -> bool:
        return self.is_active

    def contains(self, price: float, tolerance: float = 0.0) -> bool:
        return (self.low - tolerance) <= price <= (self.high + tolerance)

    def overlaps(self, low: float, high: float) -> bool:
        return not (high < self.low or low > self.high)

    def distance_to(self, price: float) -> float:
        """Signed distance from price to the zone (0 when inside)."""

        if price < self.low:
            return self.low - price
        if price > self.high:
            return price - self.high
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "high": round(self.high, 5),
            "low": round(self.low, 5),
            "body_high": round(self.body_high, 5),
            "body_low": round(self.body_low, 5),
            "midpoint": round(self.midpoint, 5),
            "index": self.index,
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe,
            "bos_index": self.bos_index,
            "status": self.status,
            "quality": self.quality,
            "test_count": self.test_count,
            "is_flip": self.is_flip,
            "is_breaker": self.is_breaker,
            "origin": self.origin,
            "mitigated_at": self.mitigated_at.isoformat() if self.mitigated_at else None,
        }


# --------------------------------------------------------------------------- quality
def rate_ob_quality(
    ob: OrderBlock,
    bos_candle: Candle,
    candles: Sequence[Candle],
    atr_value: float | None = None,
    fvgs: Sequence[FVG] | None = None,
    htf_bias: str | None = None,
    htf_levels: Sequence[float] | None = None,
    pip_size: float = 0.10,
) -> int:
    """Score an OB from 1 (weak) to 5 (institutional quality)."""

    score = 1

    reference_time = candles[-1].timestamp if candles else ob.bos_timestamp
    age = to_utc(reference_time) - to_utc(ob.timestamp)
    if age <= timedelta(hours=48):
        score += 1

    atr_value = atr_value if atr_value and atr_value > 0 else compute_atr(list(candles), 14)
    displacement = abs(bos_candle.close - (ob.high if ob.is_bullish else ob.low))
    if atr_value > 0 and displacement >= 0.8 * atr_value:
        score += 1
    elif atr_value > 0 and bos_candle.body_ratio > 0.65 and bos_candle.body >= atr_value * 0.7:
        score += 1

    if fvgs:
        for fvg in active_fvgs(fvgs, direction=ob.kind, include_inverse=False):
            if fvg.overlaps(ob.low, ob.high) or (
                ob.is_bullish and fvg.bottom >= ob.low
            ) or (not ob.is_bullish and fvg.top <= ob.high):
                score += 1
                break

    if htf_bias == ob.kind:
        score += 1

    if htf_levels:
        tolerance = 15 * pip_size
        if any(abs(level - ob.midpoint) <= tolerance or ob.low - tolerance <= level <= ob.high + tolerance for level in htf_levels):
            score += 1

    return max(1, min(5, score))


# --------------------------------------------------------------------------- lifecycle
def update_order_block_status(ob: OrderBlock, candles: Sequence[Candle], start_index: int | None = None) -> OrderBlock:
    """Replay candles after the BOS to update mitigation/test state."""

    start = ob.bos_index + 1 if start_index is None else start_index
    if start >= len(candles):
        return ob
    inside = False
    for j in range(start, len(candles)):
        candle = candles[j]
        if ob.is_bullish:
            if candle.close < ob.low:
                ob.status = FULLY_MITIGATED
                ob.mitigated_at = candle.timestamp
                break
            if candle.low <= ob.high:
                if not inside:
                    ob.test_count += 1
                    inside = True
                ob.status = PARTIALLY_MITIGATED if ob.test_count == 1 else EXHAUSTED
                ob.mitigated_at = ob.mitigated_at or candle.timestamp
            else:
                inside = False
        else:
            if candle.close > ob.high:
                ob.status = FULLY_MITIGATED
                ob.mitigated_at = candle.timestamp
                break
            if candle.high >= ob.low:
                if not inside:
                    ob.test_count += 1
                    inside = True
                ob.status = PARTIALLY_MITIGATED if ob.test_count == 1 else EXHAUSTED
                ob.mitigated_at = ob.mitigated_at or candle.timestamp
            else:
                inside = False
    return ob


def make_inverted_block(
    ob: OrderBlock, candles: Sequence[Candle], reason: str = "OB closed through"
) -> OrderBlock | None:
    """Create the flip/breaker zone that forms when an OB is violated."""

    if ob.status != FULLY_MITIGATED:
        return None
    break_index = ob.bos_index
    for j in range(ob.bos_index + 1, len(candles)):
        candle = candles[j]
        if (ob.is_bullish and candle.close < ob.low) or (not ob.is_bullish and candle.close > ob.high):
            break_index = j
            break
    if break_index >= len(candles):
        return None
    break_candle = candles[break_index]
    flipped = OrderBlock(
        kind=BEARISH if ob.is_bullish else BULLISH,
        high=ob.high,
        low=ob.low,
        body_high=ob.body_high,
        body_low=ob.body_low,
        index=ob.index,
        timestamp=ob.timestamp,
        timeframe=ob.timeframe,
        bos_index=break_index,
        bos_timestamp=break_candle.timestamp,
        status=UNMITIGATED,
        quality=max(2, ob.quality - 1),
        is_flip=True,
        is_breaker=True,
        origin="BREAKER",
        notes=[reason],
    )
    update_order_block_status(flipped, candles, start_index=break_index + 1)
    return flipped


# --------------------------------------------------------------------------- detection
def _find_origin_candle(
    candles: Sequence[Candle], event: StructureEvent, lookback: int
) -> tuple[int, Candle] | None:
    """Last opposing candle before the event (the institutional origin)."""

    start = max(0, event.index - lookback)
    wanted_bearish = event.direction == BULLISH
    for j in range(event.index - 1, start - 1, -1):
        candle = candles[j]
        if wanted_bearish and candle.close < candle.open:
            return j, candle
        if not wanted_bearish and candle.close > candle.open:
            return j, candle
    return None


def detect_order_blocks(
    candles: Sequence[Candle],
    events: Sequence[StructureEvent],
    timeframe: str = "M15",
    lookback: int = 30,
    fvgs: Sequence[FVG] | None = None,
    htf_bias: str | None = None,
    htf_levels: Sequence[float] | None = None,
    pip_size: float = 0.10,
    include_flips: bool = True,
) -> list[OrderBlock]:
    """Detect OBs from BOS/CHoCH events and track their lifecycle.

    Returns zones sorted by recency (newest first), including inverted breaker
    zones when ``include_flips`` is True.
    """

    if not candles or not events:
        return []
    atr_value = compute_atr(list(candles), 14)
    blocks: list[OrderBlock] = []
    seen: set[tuple[int, str]] = set()

    for event in events:
        if event.index <= 0 or event.index >= len(candles):
            continue
        origin_match = _find_origin_candle(candles, event, lookback)
        if origin_match is None:
            continue
        origin_index, origin = origin_match
        if (origin_index, event.direction) in seen:
            continue
        seen.add((origin_index, event.direction))
        if event.direction == BULLISH:
            body_high, body_low = origin.open, origin.close
        else:
            body_high, body_low = origin.close, origin.open
        ob = OrderBlock(
            kind=event.direction,
            high=origin.high,
            low=origin.low,
            body_high=body_high,
            body_low=body_low,
            index=origin_index,
            timestamp=origin.timestamp,
            timeframe=timeframe,
            bos_index=event.index,
            bos_timestamp=event.timestamp,
        )
        ob.quality = rate_ob_quality(
            ob,
            candles[event.index],
            candles,
            atr_value=atr_value,
            fvgs=fvgs,
            htf_bias=htf_bias,
            htf_levels=htf_levels,
            pip_size=pip_size,
        )
        update_order_block_status(ob, candles)
        blocks.append(ob)

    if include_flips:
        for block in list(blocks):
            if block.status == FULLY_MITIGATED:
                flipped = make_inverted_block(block, candles)
                if flipped is not None:
                    blocks.append(flipped)

    # Newest first; ties broken by quality.
    blocks.sort(key=lambda b: (b.bos_index, b.quality), reverse=True)
    return blocks


# --------------------------------------------------------------------------- queries
def active_order_blocks(
    blocks: Sequence[OrderBlock],
    direction: str | None = None,
    timeframe: str | None = None,
    max_age_hours: float | None = None,
    now: datetime | None = None,
) -> list[OrderBlock]:
    """Valid (unmitigated / first-test) blocks, newest first."""

    result = []
    for block in blocks:
        if not block.is_active:
            continue
        if direction and block.kind != direction:
            continue
        if timeframe and block.timeframe != timeframe:
            continue
        if max_age_hours is not None:
            reference = to_utc(now) if now else block.bos_timestamp
            if to_utc(block.timestamp) < reference - timedelta(hours=max_age_hours):
                continue
        result.append(block)
    return sorted(result, key=lambda b: (b.quality, b.bos_index), reverse=True)


def order_block_at_price(
    blocks: Sequence[OrderBlock],
    price: float,
    tolerance: float = 0.0,
    direction: str | None = None,
) -> OrderBlock | None:
    """Best active block containing (or within tolerance of) ``price``."""

    candidates = [
        b for b in active_order_blocks(blocks, direction=direction) if b.contains(price, tolerance)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda b: (b.quality, b.bos_index))


def nearest_opposing_ob(
    blocks: Sequence[OrderBlock],
    price: float,
    trade_direction: str,
    max_distance: float | None = None,
) -> OrderBlock | None:
    """Nearest active opposing OB in front of the trade (risk check)."""

    opposing = BEARISH if trade_direction == BULLISH else BULLISH
    candidates = [
        b for b in active_order_blocks(blocks, direction=opposing)
        if (b.low > price if trade_direction == BULLISH else b.high < price)
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda b: abs(b.distance_to(price)))
    if max_distance is not None and best.distance_to(price) > max_distance:
        return None
    return best


def nearest_ob(
    blocks: Sequence[OrderBlock],
    price: float,
    direction: str | None = None,
    above: bool | None = None,
    max_distance: float | None = None,
) -> OrderBlock | None:
    """Closest active OB to ``price`` with optional side filter."""

    candidates = active_order_blocks(blocks, direction=direction)
    if above is True:
        candidates = [b for b in candidates if b.midpoint > price]
    elif above is False:
        candidates = [b for b in candidates if b.midpoint < price]
    if not candidates:
        return None
    best = min(candidates, key=lambda b: b.distance_to(price))
    if max_distance is not None and best.distance_to(price) > max_distance:
        return None
    return best


def confluent_order_blocks(
    ltf_blocks: Sequence[OrderBlock], htf_blocks: Sequence[OrderBlock], tolerance: float = 0.0
) -> list[tuple[OrderBlock, OrderBlock]]:
    """Pairs of overlapping LTF/HTF blocks (highest-probability zones)."""

    pairs: list[tuple[OrderBlock, OrderBlock]] = []
    for ltf in active_order_blocks(ltf_blocks):
        for htf in active_order_blocks(htf_blocks):
            if ltf.kind != htf.kind:
                continue
            if ltf.overlaps(htf.low - tolerance, htf.high + tolerance):
                pairs.append((ltf, htf))
    return pairs


def ob_summary(blocks: Sequence[OrderBlock], limit: int = 5) -> list[dict[str, Any]]:
    """Compact representation for logs and dashboards."""

    return [b.to_dict() for b in active_order_blocks(blocks)[:limit]]
