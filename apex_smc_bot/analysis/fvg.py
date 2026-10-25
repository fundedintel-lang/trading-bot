"""Fair Value Gaps (imbalances) with lifecycle tracking.

A bullish FVG is the void between ``candle[i-1].high`` and ``candle[i+1].low``
when ``candle[i+1].low > candle[i-1].high`` (and the mirror for bearish gaps).
The middle candle is the displacement candle institutions left behind.

Lifecycle:
    UNFILLED → PARTIALLY_FILLED → FILLED
    FILLED + strong rejection → INVERSE (IFVG, acts as support/resistance)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Sequence

from apex_smc_bot.data.candle_manager import Candle, to_utc

BULLISH = "BULLISH"
BEARISH = "BEARISH"

UNFILLED = "UNFILLED"
PARTIALLY_FILLED = "PARTIALLY_FILLED"
FILLED = "FILLED"
INVERSE = "INVERSE"


@dataclass
class FVG:
    """A single fair value gap."""

    kind: str
    top: float
    bottom: float
    index: int
    timestamp: datetime
    timeframe: str = "M15"
    status: str = UNFILLED
    fill_ratio: float = 0.0
    tested: int = 0
    formed_on_bos: bool = False
    delivered: bool = False
    quality: int = 3

    def __post_init__(self) -> None:
        if self.top < self.bottom:
            self.top, self.bottom = self.bottom, self.top
        if self.kind not in {BULLISH, BEARISH}:
            raise ValueError(f"Invalid FVG kind {self.kind!r}")

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def is_bullish(self) -> bool:
        return self.kind == BULLISH

    @property
    def is_active(self) -> bool:
        return self.status in {UNFILLED, PARTIALLY_FILLED}

    def contains(self, price: float, tolerance: float = 0.0) -> bool:
        return (self.bottom - tolerance) <= price <= (self.top + tolerance)

    def overlaps(self, low: float, high: float) -> bool:
        return not (high < self.bottom or low > self.top)

    def distance_to(self, price: float) -> float:
        if price < self.bottom:
            return self.bottom - price
        if price > self.top:
            return price - self.top
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "top": round(self.top, 5),
            "bottom": round(self.bottom, 5),
            "midpoint": round(self.midpoint, 5),
            "size": round(self.size, 5),
            "index": self.index,
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe,
            "status": self.status,
            "fill_ratio": round(self.fill_ratio, 3),
            "tested": self.tested,
            "quality": self.quality,
            "inverse": self.status == INVERSE,
        }


def _score_gap(candles: Sequence[Candle], index: int, kind: str) -> int:
    """Quality 1-5 based on displacement strength and freshness."""

    if index <= 0 or index >= len(candles) - 1:
        return 1
    middle = candles[index]
    score = 1
    if middle.body_ratio > 0.6:
        score += 1
    if middle.range > 0:
        recent_ranges = [c.range for c in candles[max(0, index - 10):index]]
        if recent_ranges and middle.range > 1.3 * (sum(recent_ranges) / len(recent_ranges)):
            score += 1
    # Gap created by a close beyond the prior candle's extreme is cleaner.
    prior = candles[index - 1]
    if (kind == BULLISH and middle.close > prior.high) or (kind == BEARISH and middle.close < prior.low):
        score += 1
    if middle.direction == kind:
        score += 1
    return max(1, min(5, score))


def detect_fvg(
    candles: Sequence[Candle],
    timeframe: str = "M15",
    min_gap_price: float = 0.02,
    include_filled: bool = True,
) -> list[FVG]:
    """Detect 3-candle imbalances and replay their fill lifecycle.

    Args:
        candles: Chronologically ordered candles.
        timeframe: Timeframe label stored with each gap.
        min_gap_price: Ignore micro-gaps smaller than this (price units).
        include_filled: Keep filled gaps in the output (needed for IFVG logic).
    """

    if len(candles) < 3:
        return []
    fvgs: list[FVG] = []
    for i in range(1, len(candles) - 1):
        before, middle, after = candles[i - 1], candles[i], candles[i + 1]
        if after.low - before.high > max(min_gap_price, 0.0):
            fvgs.append(
                FVG(
                    kind=BULLISH,
                    top=after.low,
                    bottom=before.high,
                    index=i,
                    timestamp=middle.timestamp,
                    timeframe=timeframe,
                    quality=_score_gap(candles, i, BULLISH),
                )
            )
        elif before.low - after.high > max(min_gap_price, 0.0):
            fvgs.append(
                FVG(
                    kind=BEARISH,
                    top=before.low,
                    bottom=after.high,
                    index=i,
                    timestamp=middle.timestamp,
                    timeframe=timeframe,
                    quality=_score_gap(candles, i, BEARISH),
                )
            )

    for fvg in fvgs:
        update_fvg_status(fvg, candles)
    return [fvg for fvg in fvgs if include_filled or fvg.is_active]


def update_fvg_status(fvg: FVG, candles: Sequence[Candle]) -> FVG:
    """Replay subsequent price action to update fill state / IFVG conversion."""

    start = fvg.index + 2  # gap is only complete at candle i+1
    if start >= len(candles):
        return fvg
    extreme = fvg.top if fvg.is_bullish else fvg.bottom
    filled_at: int | None = None
    rejection_seen = False

    for j in range(start, len(candles)):
        candle = candles[j]
        if fvg.is_bullish:
            if candle.low < fvg.top:
                fvg.tested += 1
            penetration = max(0.0, fvg.top - candle.low)
            fvg.fill_ratio = max(fvg.fill_ratio, min(1.0, penetration / fvg.size)) if fvg.size > 0 else 1.0
            if candle.close <= fvg.bottom:
                fvg.status = FILLED
                fvg.fill_ratio = 1.0
                filled_at = j
                break
            if candle.low <= fvg.bottom:
                fvg.status = FILLED
                fvg.fill_ratio = 1.0
                filled_at = j
                break
            if candle.low <= fvg.top:
                fvg.status = PARTIALLY_FILLED
        else:
            if candle.high > fvg.bottom:
                fvg.tested += 1
            penetration = max(0.0, candle.high - fvg.bottom)
            fvg.fill_ratio = max(fvg.fill_ratio, min(1.0, penetration / fvg.size)) if fvg.size > 0 else 1.0
            if candle.close >= fvg.top:
                fvg.status = FILLED
                fvg.fill_ratio = 1.0
                filled_at = j
                break
            if candle.high >= fvg.top:
                fvg.status = FILLED
                fvg.fill_ratio = 1.0
                filled_at = j
                break
            if candle.high >= fvg.bottom:
                fvg.status = PARTIALLY_FILLED

    # Inverse FVG: after a fill, a strong rejection candle reclaims the gap.
    if filled_at is not None and filled_at + 3 < len(candles):
        for j in range(filled_at, min(filled_at + 4, len(candles))):
            candle = candles[j]
            if fvg.is_bullish and candle.close > fvg.top and candle.body_ratio > 0.55:
                fvg.status = INVERSE
                rejection_seen = True
                break
            if not fvg.is_bullish and candle.close < fvg.bottom and candle.body_ratio > 0.55:
                fvg.status = INVERSE
                rejection_seen = True
                break

    # A gap that price has run away from without rebalancing is "delivered".
    if fvg.is_active and len(candles) - 1 > fvg.index + 10:
        last = candles[-1]
        if fvg.is_bullish and last.low > fvg.top:
            fvg.delivered = True
        if not fvg.is_bullish and last.high < fvg.bottom:
            fvg.delivered = True

    if rejection_seen:
        fvg.status = INVERSE
    return fvg


def active_fvgs(fvgs: Sequence[FVG], direction: str | None = None, include_inverse: bool = True) -> list[FVG]:
    """Filter to tradeable gaps, newest first."""

    allowed = {UNFILLED, PARTIALLY_FILLED}
    if include_inverse:
        allowed.add(INVERSE)
    result = [f for f in fvgs if f.status in allowed and (direction is None or f.kind == direction)]
    return sorted(result, key=lambda f: f.index, reverse=True)


def overlapping_fvg(fvgs: Sequence[FVG], low: float, high: float, direction: str | None = None) -> FVG | None:
    """Return the highest-quality active gap overlapping a price zone."""

    candidates = [f for f in active_fvgs(fvgs, direction) if f.overlaps(low, high)]
    if not candidates:
        return None
    return max(candidates, key=lambda f: (f.quality, f.index))


def nearest_fvg(
    fvgs: Sequence[FVG], price: float, direction: str | None = None, above: bool | None = None, max_distance: float | None = None
) -> FVG | None:
    """Find the closest active gap to ``price`` (optionally above/below only)."""

    candidates = active_fvgs(fvgs, direction)
    if above is True:
        candidates = [f for f in candidates if f.midpoint > price]
    elif above is False:
        candidates = [f for f in candidates if f.midpoint < price]
    if not candidates:
        return None
    best = min(candidates, key=lambda f: f.distance_to(price))
    if max_distance is not None and best.distance_to(price) > max_distance:
        return None
    return best


def is_fresh(fvg: FVG, now: datetime, max_age_hours: float = 24.0) -> bool:
    """True when the gap formed within ``max_age_hours`` of ``now``."""

    return to_utc(now) - to_utc(fvg.timestamp) <= timedelta(hours=max_age_hours)
