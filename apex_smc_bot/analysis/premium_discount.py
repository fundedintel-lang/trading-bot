"""Premium / discount pricing and the Optimal Trade Entry (OTE) zone.

Institutions buy cheap (discount, below equilibrium) and sell expensive
(premium, above equilibrium) inside the current dealing range.  The OTE window is
the 61.8%–79% retracement of the impulse leg where reloads are most likely.

Price mapping by bias:
    BULLISH leg: 0% = swing high, 100% = swing low (retracement measured down)
    BEARISH leg: 0% = swing low,  100% = swing high (retracement measured up)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from apex_smc_bot.analysis.structure import BEARISH, BULLISH, StructureState

PREMIUM = "PREMIUM"
DISCOUNT = "DISCOUNT"
EQUILIBRIUM = "EQUILIBRIUM"

DEFAULT_OTE_LOW = 0.618
DEFAULT_OTE_HIGH = 0.79
OTE_MID = 0.705


def _validate(swing_low: float, swing_high: float) -> None:
    if swing_high <= swing_low:
        raise ValueError(f"swing_high ({swing_high}) must be greater than swing_low ({swing_low})")


@dataclass(frozen=True, slots=True)
class DealingRange:
    """A structured impulse leg with fib levels and premium/discount zones."""

    low: float
    high: float
    bias: str
    equilibrium: float
    ote_low: float
    ote_high: float
    fibs: dict[float, float]

    @property
    def size(self) -> float:
        return self.high - self.low

    @property
    def premium_zone(self) -> tuple[float, float]:
        return (self.equilibrium, self.high)

    @property
    def discount_zone(self) -> tuple[float, float]:
        return (self.low, self.equilibrium)

    @property
    def ote_zone(self) -> tuple[float, float]:
        return (self.ote_low, self.ote_high)

    def zone_of(self, price: float) -> str:
        if price > self.equilibrium:
            return PREMIUM
        if price < self.equilibrium:
            return DISCOUNT
        return EQUILIBRIUM

    def in_discount(self, price: float) -> bool:
        return price <= self.equilibrium

    def in_premium(self, price: float) -> bool:
        return price >= self.equilibrium

    def in_ote(self, price: float) -> bool:
        return self.ote_low <= price <= self.ote_high

    def retracement(self, price: float) -> float:
        """Retracement fraction (0 = leg extreme, 1 = leg origin)."""

        if self.size <= 0:
            return 0.0
        if self.bias == BULLISH:
            return max(0.0, min(1.0, (self.high - price) / self.size))
        return max(0.0, min(1.0, (price - self.low) / self.size))

    def ote_quality(self, price: float) -> str:
        if self.in_ote(price):
            mid_distance = abs(price - self.fibs.get(OTE_MID, self.ote_low))
            return "SNIPER" if mid_distance <= self.size * 0.03 else "GOOD"
        if self.zone_of(price) == (DISCOUNT if self.bias == BULLISH else PREMIUM):
            return "DIRECTIONAL"
        return "OUTSIDE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bias": self.bias,
            "low": round(self.low, 5),
            "high": round(self.high, 5),
            "equilibrium": round(self.equilibrium, 5),
            "ote_low": round(self.ote_low, 5),
            "ote_high": round(self.ote_high, 5),
            "premium_zone": [round(self.premium_zone[0], 5), round(self.premium_zone[1], 5)],
            "discount_zone": [round(self.discount_zone[0], 5), round(self.discount_zone[1], 5)],
            "fibs": {str(k): round(v, 5) for k, v in self.fibs.items()},
        }


def calculate_dealing_range(
    swing_low: float,
    swing_high: float,
    bias: str,
    ote_low_fib: float = DEFAULT_OTE_LOW,
    ote_high_fib: float = DEFAULT_OTE_HIGH,
) -> DealingRange:
    """Build the dealing range + OTE for an impulse leg."""

    _validate(swing_low, swing_high)
    if bias not in {BULLISH, BEARISH}:
        raise ValueError(f"bias must be {BULLISH} or {BEARISH}, got {bias!r}")
    if not 0 < ote_low_fib < ote_high_fib < 1:
        raise ValueError("OTE fib bounds must satisfy 0 < low < high < 1")
    size = swing_high - swing_low
    if bias == BULLISH:
        fibs = {
            0.0: swing_high,
            0.5: swing_high - size * 0.5,
            ote_low_fib: swing_high - size * ote_low_fib,
            OTE_MID: swing_high - size * OTE_MID,
            ote_high_fib: swing_high - size * ote_high_fib,
            1.0: swing_low,
        }
        ote_high_price = swing_high - size * ote_low_fib
        ote_low_price = swing_high - size * ote_high_fib
    else:
        fibs = {
            0.0: swing_low,
            0.5: swing_low + size * 0.5,
            ote_low_fib: swing_low + size * ote_low_fib,
            OTE_MID: swing_low + size * OTE_MID,
            ote_high_fib: swing_low + size * ote_high_fib,
            1.0: swing_high,
        }
        ote_low_price = swing_low + size * ote_low_fib
        ote_high_price = swing_low + size * ote_high_fib
    equilibrium = swing_low + size * 0.5
    return DealingRange(
        low=swing_low,
        high=swing_high,
        bias=bias,
        equilibrium=equilibrium,
        ote_low=min(ote_low_price, ote_high_price),
        ote_high=max(ote_low_price, ote_high_price),
        fibs=fibs,
    )


def calculate_ote_zones(
    swing_low: float,
    swing_high: float,
    bias: str,
    ote_low_fib: float = DEFAULT_OTE_LOW,
    ote_high_fib: float = DEFAULT_OTE_HIGH,
) -> dict[str, Any]:
    """Spec-compatible dictionary form of the dealing range."""

    dr = calculate_dealing_range(swing_low, swing_high, bias, ote_low_fib, ote_high_fib)
    base = {
        "bias": bias,
        "equilibrium": dr.equilibrium,
        "ote_zone_high": dr.ote_high,
        "ote_zone_low": dr.ote_low,
        "premium_zone": dr.premium_zone,
        "discount_zone": dr.discount_zone,
    }
    if bias == BULLISH:
        base["ideal_buy_zone"] = dr.ote_zone
    else:
        base["ideal_sell_zone"] = dr.ote_zone
    return base


def dealing_range_from_structure(
    structure: StructureState,
    ote_low_fib: float = DEFAULT_OTE_LOW,
    ote_high_fib: float = DEFAULT_OTE_HIGH,
    *,
    bias: str | None = None,
) -> DealingRange | None:
    """Construct the dealing range from a structure's most recent swings."""

    pair = structure.dealing_range
    if pair is None:
        return None
    low, high = pair
    if high <= low:
        return None
    bias = bias if bias in {BULLISH, BEARISH} else None
    if bias is None:
        bias = structure.direction if structure.direction in {BULLISH, BEARISH} else None
    if bias is None:
        # Fall back to the leg direction: rising low→high is a bullish leg.
        if structure.last_swing_low is None or structure.last_swing_high is None:
            return None
        bias = BULLISH if structure.last_swing_low.index < structure.last_swing_high.index else BEARISH
    return calculate_dealing_range(low, high, bias, ote_low_fib, ote_high_fib)


def premium_discount_position(price: float, dealing_range: DealingRange) -> dict[str, Any]:
    """Describe where price sits inside the range (for signal analysis)."""

    retracement = dealing_range.retracement(price)
    return {
        "zone": dealing_range.zone_of(price),
        "retracement_percent": round(retracement * 100, 1),
        "in_ote": dealing_range.in_ote(price),
        "ote_quality": dealing_range.ote_quality(price),
        "equilibrium": round(dealing_range.equilibrium, 5),
    }


def fib_extension_targets(dealing_range: DealingRange, entry: float, direction: str) -> list[float]:
    """Projection targets beyond the current leg extreme (TP candidates)."""

    size = dealing_range.size
    if size <= 0:
        return []
    if direction == BULLISH:
        return [dealing_range.high + size * factor for factor in (0.272, 0.618, 1.0)]
    return [dealing_range.low - size * factor for factor in (0.272, 0.618, 1.0)]


def select_dealing_range(
    candles_lows: Sequence[float], candles_highs: Sequence[float], lookback: int = 60
) -> tuple[float, float] | None:
    """Simple significant swing range for dashboards / quick analysis."""

    if not candles_lows or not candles_highs:
        return None
    lows = list(candles_lows)[-lookback:]
    highs = list(candles_highs)[-lookback:]
    low, high = min(lows), max(highs)
    return (low, high) if high > low else None
