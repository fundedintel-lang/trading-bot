"""Volume profile, delta proxy, climax and divergence analysis.

XAU/USD is OTC, so true tape volume is unavailable from retail brokers.  We use
broker tick volume as a proxy (highly correlated with institutional activity on
gold futures arbitrage) and a body-based delta estimate when tick-side data is
absent.  Every function states which proxy it uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from apex_smc_bot.data.candle_manager import Candle, average_volume

BULLISH = "BULLISH"
BEARISH = "BEARISH"


@dataclass
class VolumeProfile:
    """Volume-at-price distribution with HVN/LVN and value area."""

    hvns: list[float] = field(default_factory=list)
    lvns: list[float] = field(default_factory=list)
    poc: float = 0.0  # point of control (highest volume price)
    value_area_high: float = 0.0
    value_area_low: float = 0.0
    volume_by_price: dict[float, float] = field(default_factory=dict)
    bin_size: float = 0.0

    @property
    def total_volume(self) -> float:
        return sum(self.volume_by_price.values())

    def nearest_hvn(self, price: float) -> float | None:
        return min(self.hvns, key=lambda level: abs(level - price)) if self.hvns else None

    def nearest_lvn(self, price: float) -> float | None:
        return min(self.lvns, key=lambda level: abs(level - price)) if self.lvns else None

    def is_hvn(self, price: float, tolerance: float = 0.5) -> bool:
        return any(abs(price - level) <= tolerance for level in self.hvns)

    def in_value_area(self, price: float) -> bool:
        return self.value_area_low <= price <= self.value_area_high

    def to_dict(self) -> dict[str, Any]:
        return {
            "poc": round(self.poc, 2),
            "value_area_high": round(self.value_area_high, 2),
            "value_area_low": round(self.value_area_low, 2),
            "hvns": [round(v, 2) for v in self.hvns[:10]],
            "lvns": [round(v, 2) for v in self.lvns[:10]],
        }


@dataclass(frozen=True, slots=True)
class VolumeSignal:
    """A climax/divergence/absorption observation."""

    kind: str
    direction: str
    strength: float
    detail: str
    index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "direction": self.direction,
            "strength": round(self.strength, 3),
            "detail": self.detail,
        }


def candle_delta(candle: Candle) -> float:
    """Body-weighted delta proxy (positive = net buying pressure)."""

    if candle.range <= 0:
        return 0.0
    body_ratio = (candle.close - candle.open) / candle.range
    return candle.volume * body_ratio


def calculate_delta(candle: Candle, tick_data: Sequence[Any] | None = None) -> float:
    """Volume delta for one candle.

    If ``tick_data`` is provided (objects with ``volume`` and ``side``), the true
    delta is computed; otherwise a body-weighted proxy is returned.
    """

    if tick_data:
        buy = sum(float(t.volume) for t in tick_data if str(getattr(t, "side", "")).upper() == "BUY")
        sell = sum(float(t.volume) for t in tick_data if str(getattr(t, "side", "")).upper() == "SELL")
        return buy - sell
    return candle_delta(candle)


def delta_series(candles: Sequence[Candle]) -> list[float]:
    return [candle_delta(c) for c in candles]


def cumulative_delta(candles: Sequence[Candle]) -> float:
    return sum(candle_delta(c) for c in candles)


def build_volume_profile(candles: Sequence[Candle], bins: int = 100) -> VolumeProfile:
    """Build a volume-at-price histogram, HVN/LVN lists and the 70% value area."""

    if not candles or bins < 1:
        return VolumeProfile()
    price_min = min(c.low for c in candles)
    price_max = max(c.high for c in candles)
    if price_max <= price_min:
        return VolumeProfile(
            hvns=[price_max],
            lvns=[],
            poc=price_max,
            value_area_high=price_max,
            value_area_low=price_min,
            volume_by_price={price_max: sum(c.volume for c in candles)},
            bin_size=0.0,
        )
    bin_size = (price_max - price_min) / bins
    volume_by_price: dict[float, float] = {}
    for candle in candles:
        candle_range = candle.high - candle.low
        if candle_range <= 0:
            key = round((candle.close - price_min) / bin_size) * bin_size + price_min
            volume_by_price[key] = volume_by_price.get(key, 0.0) + candle.volume
            continue
        covered = max(1, int(round(candle_range / bin_size)) + 1)
        volume_per_bin = candle.volume / covered
        steps = max(1, int(candle_range / bin_size))
        for step in range(steps + 1):
            price = candle.low + step * (candle_range / steps) if steps else candle.close
            index = int(round((price - price_min) / bin_size))
            key = round((index * bin_size) + price_min, 5)
            volume_by_price[key] = volume_by_price.get(key, 0.0) + volume_per_bin

    ordered = sorted(volume_by_price.items(), key=lambda kv: kv[1], reverse=True)
    hvn_count = max(1, len(ordered) // 10)
    hvns = [price for price, _ in ordered[:hvn_count]]
    lvns = [price for price, _ in ordered[-hvn_count:]]
    poc = ordered[0][0]

    # 70% value area around the POC.
    total = sum(volume_by_price.values())
    target = total * 0.70
    index_by_price = {price: i for i, (price, _) in enumerate(ordered)}
    poc_rank = index_by_price[poc]
    accumulated = ordered[poc_rank][1]
    low_rank = high_rank = poc_rank
    while accumulated < target and (low_rank > 0 or high_rank < len(ordered) - 1):
        next_low = ordered[low_rank - 1][1] if low_rank > 0 else -1.0
        next_high = ordered[high_rank + 1][1] if high_rank < len(ordered) - 1 else -1.0
        if next_high >= next_low:
            high_rank += 1
            accumulated += ordered[high_rank][1]
        else:
            low_rank -= 1
            accumulated += ordered[low_rank][1]
    values = [price for price, _ in ordered[low_rank:high_rank + 1]]
    return VolumeProfile(
        hvns=sorted(hvns),
        lvns=sorted(lvns),
        poc=round(poc, 2),
        value_area_high=round(max(values + [poc]), 2),
        value_area_low=round(min(values + [poc]), 2),
        volume_by_price=volume_by_price,
        bin_size=round(bin_size, 5),
    )


def detect_volume_climax(
    candles: Sequence[Candle], period: int = 20, multiplier: float = 2.0
) -> VolumeSignal | None:
    """Flag a 2x-average volume spike on the latest candle and its implication."""

    if len(candles) < period + 1:
        return None
    latest = candles[-1]
    baseline = average_volume(candles, period)
    if baseline <= 0 or latest.volume < multiplier * baseline:
        return None
    direction = BEARISH if latest.close > latest.open else BULLISH
    kind = "CLIMAX_TOP" if direction == BEARISH else "CLIMAX_BOTTOM"
    detail = (
        f"Volume {latest.volume:.0f} vs {multiplier}x avg {baseline:.0f} — "
        f"{'buyers absorbed at the high' if direction == BEARISH else 'sellers absorbed at the low'}"
    )
    return VolumeSignal(
        kind=kind, direction=direction, strength=latest.volume / baseline, detail=detail, index=len(candles) - 1
    )


def detect_delta_divergence(candles: Sequence[Candle], lookback: int = 12) -> VolumeSignal | None:
    """Price/delta divergence over the trailing window.

    Bullish: price lower low while cumulative delta rises (accumulation).
    Bearish: price higher high while cumulative delta falls (distribution).
    """

    if len(candles) < lookback + 2:
        return None
    recent = list(candles[-lookback:])
    prior = list(candles[-2 * lookback:-lookback])
    if not prior:
        return None
    recent_high, recent_low = max(c.high for c in recent), min(c.low for c in recent)
    prior_high, prior_low = max(c.high for c in prior), min(c.low for c in prior)
    recent_delta = cumulative_delta(recent)
    prior_delta = cumulative_delta(prior)

    if recent_high > prior_high and recent_delta < prior_delta:
        return VolumeSignal(
            kind="BEARISH_DELTA_DIVERGENCE",
            direction=BEARISH,
            strength=abs(prior_delta - recent_delta),
            detail="Higher high on falling delta — distribution",
        )
    if recent_low < prior_low and recent_delta > prior_delta:
        return VolumeSignal(
            kind="BULLISH_DELTA_DIVERGENCE",
            direction=BULLISH,
            strength=abs(recent_delta - prior_delta),
            detail="Lower low on rising delta — accumulation",
        )
    return None


def volume_confirmation(
    candles: Sequence[Candle], direction: str, period: int = 20, threshold: float = 1.2
) -> tuple[bool, str]:
    """Confirm entry volume: expansion OR a delta flip in the trade direction."""

    if len(candles) < 2:
        return False, "insufficient volume history"
    latest = candles[-1]
    baseline = average_volume(candles, period)
    expanded = baseline > 0 and latest.volume >= baseline * threshold
    delta_now = candle_delta(latest)
    delta_prev = candle_delta(candles[-2])
    flipped = (direction == BULLISH and delta_now > 0 and delta_prev <= 0) or (
        direction == BEARISH and delta_now < 0 and delta_prev >= 0
    )
    if expanded and (direction == BULLISH and latest.close >= latest.open or direction == BEARISH and latest.close <= latest.open):
        return True, f"volume {latest.volume:.0f} >= {threshold}x avg {baseline:.0f}"
    if flipped:
        return True, f"delta flipped {direction.lower()} ({delta_prev:.1f} → {delta_now:.1f})"
    if expanded:
        return True, f"volume expansion {latest.volume:.0f} vs avg {baseline:.0f}"
    return False, f"volume {latest.volume:.0f} below {threshold}x avg {baseline:.0f}"


def absorption_at_level(candles: Sequence[Candle], direction: str) -> VolumeSignal | None:
    """Detect an absorption candle in the last three bars."""

    if len(candles) < 3:
        return None
    for index in range(len(candles) - 3, len(candles)):
        candle = candles[index]
        delta = candle_delta(candle)
        if direction == BULLISH and candle.close < candle.open and delta > 0:
            return VolumeSignal(
                kind="BULLISH_ABSORPTION",
                direction=BULLISH,
                strength=abs(delta),
                detail="Down candle with net buying — sellers absorbed",
                index=index,
            )
        if direction == BEARISH and candle.close > candle.open and delta < 0:
            return VolumeSignal(
                kind="BEARISH_ABSORPTION",
                direction=BEARISH,
                strength=abs(delta),
                detail="Up candle with net selling — buyers absorbed",
                index=index,
            )
    return None


def volume_summary(candles: Sequence[Candle]) -> dict[str, Any]:
    """Compact volume snapshot for the signal audit trail."""

    if not candles:
        return {}
    latest = candles[-1]
    baseline = average_volume(candles, 20)
    profile = build_volume_profile(candles[-200:]) if len(candles) >= 30 else None
    return {
        "latest_volume": round(latest.volume, 2),
        "average_volume_20": round(baseline, 2),
        "volume_ratio": round(latest.volume / baseline, 2) if baseline > 0 else None,
        "delta": round(candle_delta(latest), 2),
        "cumulative_delta": round(cumulative_delta(candles[-20:]), 2),
        "profile": profile.to_dict() if profile else None,
    }
