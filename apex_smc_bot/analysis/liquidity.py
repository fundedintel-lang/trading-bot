"""Liquidity engineering: equal highs/lows, raids (sweeps), inducement.

Liquidity pools form where retail stop losses cluster: above equal/swing highs
(BSL) and below equal/swing lows (SSL).  A valid sweep requires **three** things:

1. a wick that pierces the pool,
2. a close back on the original side (the raid is rejected), and
3. a BOS/CHoCH in the reversal direction shortly afterwards.

Without step 3 the move is *inducement* (a trap), not a sweep, and is never
tradeable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from apex_smc_bot.analysis.structure import (
    BEARISH,
    BULLISH,
    StructureEvent,
    StructureState,
    SwingPoint,
    analyze_dual_structure,
    detect_swings,
)
from apex_smc_bot.data.candle_manager import Candle

EQH = "EQH"
EQL = "EQL"
BSL_SWEEP = "BSL_SWEEP"
SSL_SWEEP = "SSL_SWEEP"


@dataclass(frozen=True, slots=True)
class EqualLevel:
    """A cluster of equal swing highs (EQH) or lows (EQL)."""

    kind: str
    price: float
    swings: tuple[SwingPoint, ...]
    tolerance: float = 0.5
    internal: bool = True

    @property
    def count(self) -> int:
        return len(self.swings)

    @property
    def strength(self) -> int:
        return max(2, self.count * 2)

    @property
    def last_index(self) -> int:
        return max(s.index for s in self.swings)

    @property
    def first_index(self) -> int:
        return min(s.index for s in self.swings)

    @property
    def is_buy_side(self) -> bool:
        return self.kind == EQH

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "price": round(self.price, 5),
            "count": self.count,
            "strength": self.strength,
            "tolerance": self.tolerance,
            "last_index": self.last_index,
            "internal": self.internal,
        }


@dataclass
class SweepEvent:
    """A confirmed or potential liquidity raid."""

    kind: str  # BSL_SWEEP | SSL_SWEEP
    level: float
    index: int
    timestamp: Any
    extreme: float
    close: float
    direction: str  # expected reversal direction
    confirmed_bos: StructureEvent | None = None
    quality: str = "LOW"  # HIGH | MEDIUM | LOW
    raid_count: int = 1
    internal: bool = True
    level_kind: str = EQH

    @property
    def confirmed(self) -> bool:
        return self.confirmed_bos is not None

    @property
    def is_buy_side(self) -> bool:
        return self.kind == BSL_SWEEP

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "level": round(self.level, 5),
            "index": self.index,
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else str(self.timestamp),
            "extreme": round(self.extreme, 5),
            "close": round(self.close, 5),
            "direction": self.direction,
            "confirmed": self.confirmed,
            "confirmed_bos": self.confirmed_bos.to_dict() if self.confirmed_bos else None,
            "quality": self.quality,
            "raid_count": self.raid_count,
            "internal": self.internal,
            "level_kind": self.level_kind,
        }


@dataclass
class LiquidityMap:
    """All liquidity information for one timeframe."""

    timeframe: str = "M15"
    internal_levels: list[EqualLevel] = field(default_factory=list)
    external_levels: list[EqualLevel] = field(default_factory=list)
    sweeps: list[SweepEvent] = field(default_factory=list)
    bias: str | None = None

    @property
    def all_levels(self) -> list[EqualLevel]:
        return self.internal_levels + self.external_levels

    @property
    def equal_highs(self) -> list[EqualLevel]:
        return [lvl for lvl in self.all_levels if lvl.kind == EQH]

    @property
    def equal_lows(self) -> list[EqualLevel]:
        return [lvl for lvl in self.all_levels if lvl.kind == EQL]

    def nearest_above(self, price: float) -> EqualLevel | None:
        """Closest unswept buy-side pool above price."""

        candidates = [lvl for lvl in self.equal_highs if lvl.price > price]
        return min(candidates, key=lambda lvl: lvl.price - price) if candidates else None

    def nearest_below(self, price: float) -> EqualLevel | None:
        """Closest unswept sell-side pool below price."""

        candidates = [lvl for lvl in self.equal_lows if lvl.price < price]
        return min(candidates, key=lambda lvl: price - lvl.price) if candidates else None

    def recent_sweep(
        self,
        direction: str | None = None,
        max_age_candles: int | None = None,
        total_candles: int | None = None,
        require_confirmed: bool = False,
    ) -> SweepEvent | None:
        """Most recent matching sweep (newest first)."""

        for sweep in sorted(self.sweeps, key=lambda s: s.index, reverse=True):
            if direction and sweep.direction != direction:
                continue
            if require_confirmed and not sweep.confirmed:
                continue
            if max_age_candles is not None and total_candles is not None:
                if total_candles - 1 - sweep.index > max_age_candles:
                    continue
            return sweep
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "equal_highs": [lvl.to_dict() for lvl in self.equal_highs[-5:]],
            "equal_lows": [lvl.to_dict() for lvl in self.equal_lows[-5:]],
            "sweeps": [s.to_dict() for s in sorted(self.sweeps, key=lambda s: s.index, reverse=True)[:5]],
        }


# --------------------------------------------------------------------------- equal levels
def _cluster(swings: Sequence[SwingPoint], tolerance: float, kind: str, internal: bool) -> list[EqualLevel]:
    if not swings:
        return []
    ordered = sorted(swings, key=lambda s: s.price)
    clusters: list[list[SwingPoint]] = []
    for swing in ordered:
        placed = False
        for cluster in clusters:
            mean_price = sum(s.price for s in cluster) / len(cluster)
            if abs(swing.price - mean_price) <= tolerance:
                cluster.append(swing)
                placed = True
                break
        if not placed:
            clusters.append([swing])
    levels: list[EqualLevel] = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        # Keep clusters that are genuinely separated in time (>= 2 candles).
        indices = sorted(s.index for s in cluster)
        if indices[-1] - indices[0] < 2:
            continue
        levels.append(
            EqualLevel(
                kind=kind,
                price=round(sum(s.price for s in cluster) / len(cluster), 5),
                swings=tuple(sorted(cluster, key=lambda s: s.index)),
                tolerance=tolerance,
                internal=internal,
            )
        )
    return sorted(levels, key=lambda lvl: lvl.price)


def detect_equal_levels(
    swings: Sequence[SwingPoint],
    tolerance_pips: float = 5.0,
    pip_size: float = 0.10,
    internal: bool = True,
) -> tuple[list[EqualLevel], list[EqualLevel]]:
    """Group swing points into EQH/EQL pools.

    Returns ``(equal_highs, equal_lows)``.
    """

    tolerance = max(1e-6, tolerance_pips * pip_size)
    highs = [s for s in swings if s.is_high]
    lows = [s for s in swings if not s.is_high]
    return (
        _cluster(highs, tolerance, EQH, internal),
        _cluster(lows, tolerance, EQL, internal),
    )


def detect_liquidity_levels(
    candles: Sequence[Candle], tolerance_pips: float = 5.0, lookback: int = 3, pip_size: float = 0.10
) -> tuple[list[EqualLevel], list[EqualLevel]]:
    """Convenience: detect swings then equal levels on a candle series."""

    swings = detect_swings(candles, lookback=lookback)
    return detect_equal_levels(swings, tolerance_pips, pip_size, internal=(lookback <= 3))


# --------------------------------------------------------------------------- sweeps
def _first_bos_after(
    events: Sequence[StructureEvent], index: int, direction: str, lookahead: int
) -> StructureEvent | None:
    for event in events:
        if event.index <= index:
            continue
        if event.index > index + lookahead:
            break
        if event.direction == direction and event.kind in {"BOS", "CHOCH"}:
            return event
    return None


def detect_sweeps(
    candles: Sequence[Candle],
    levels: Sequence[EqualLevel],
    events: Sequence[StructureEvent],
    bias: str | None = None,
    lookahead: int = 16,
) -> list[SweepEvent]:
    """Detect raids on equal highs/lows and whether a BOS confirmed them."""

    sweeps: list[SweepEvent] = []
    for level in levels:
        start = level.last_index + 1
        raids: list[int] = []
        for j in range(start, len(candles)):
            candle = candles[j]
            if level.kind == EQH:
                if candle.high > level.price and candle.close < level.price:
                    raids.append(j)
            else:
                if candle.low < level.price and candle.close > level.price:
                    raids.append(j)
        if not raids:
            continue
        # Evaluate each raid; prefer the latest raid that gained a confirming BOS.
        chosen: tuple[int, StructureEvent | None] | None = None
        for j in raids:
            direction = BEARISH if level.kind == EQH else BULLISH
            bos = _first_bos_after(events, j, direction, lookahead)
            if chosen is None or (bos is not None):
                chosen = (j, bos)
            if bos is not None:
                break
        if chosen is None:
            continue
        j, bos = chosen
        candle = candles[j]
        direction = BEARISH if level.kind == EQH else BULLISH
        if bos is not None and (bias is None or bias == direction):
            quality = "HIGH"
        elif bias == direction:
            quality = "MEDIUM"
        else:
            quality = "LOW"
        if len(raids) >= 2 and bos is not None:
            quality = "HIGH"
        sweeps.append(
            SweepEvent(
                kind=BSL_SWEEP if level.kind == EQH else SSL_SWEEP,
                level=level.price,
                index=j,
                timestamp=candle.timestamp,
                extreme=candle.high if level.kind == EQH else candle.low,
                close=candle.close,
                direction=direction,
                confirmed_bos=bos,
                quality=quality,
                raid_count=len(raids),
                internal=level.internal,
                level_kind=level.kind,
            )
        )
    return sorted(sweeps, key=lambda s: s.index)


def detect_sweep(
    candles: Sequence[Candle],
    liquidity_level: EqualLevel,
    bias: str,
    events: Sequence[StructureEvent] | None = None,
    lookahead: int = 16,
) -> SweepEvent | None:
    """Single-level sweep helper (spec-compatible signature)."""

    if events is None:
        state = analyze_dual_structure(candles, "M15", internal_lookback=3, external_lookback=8)
        events = state["internal"].events
    results = detect_sweeps(candles, [liquidity_level], events, bias=bias, lookahead=lookahead)
    return results[0] if results else None


# --------------------------------------------------------------------------- inducement
def detect_inducement(
    candles: Sequence[Candle],
    structure: StructureState,
    direction: str,
    lookback_candles: int = 12,
) -> dict[str, Any] | None:
    """Detect a resolved inducement swing in the trade direction.

    Inducement = a minor swing that lures retail traders before being raided and
    reversed.  We look for the most recent minor swing opposite to the trade
    direction that was pierced and reclaimed within ``lookback_candles``.
    """

    if len(candles) < 5:
        return None
    minor = detect_swings(candles, lookback=1)
    window_start = max(0, len(candles) - lookback_candles)
    if direction == BULLISH:
        candidates = [s for s in minor if not s.is_high and s.index >= window_start]
        if not candidates:
            return None
        level = candidates[-1]
        later = candles[level.index + 1:]
        for candle in later:
            if candle.low < level.price and candle.close > level.price:
                return {
                    "kind": "INDUCEMENT_SWEEP",
                    "level": level.price,
                    "index": level.index,
                    "sweep_timestamp": candle.timestamp.isoformat(),
                    "resolved": True,
                    "direction": BULLISH,
                }
    else:
        candidates = [s for s in minor if s.is_high and s.index >= window_start]
        if not candidates:
            return None
        level = candidates[-1]
        later = candles[level.index + 1:]
        for candle in later:
            if candle.high > level.price and candle.close < level.price:
                return {
                    "kind": "INDUCEMENT_SWEEP",
                    "level": level.price,
                    "index": level.index,
                    "sweep_timestamp": candle.timestamp.isoformat(),
                    "resolved": True,
                    "direction": BEARISH,
                }
    return None


# --------------------------------------------------------------------------- map
def build_liquidity_map(
    candles: Sequence[Candle],
    timeframe: str = "M15",
    tolerance_pips: float = 5.0,
    internal_lookback: int = 3,
    external_lookback: int = 8,
    bias: str | None = None,
    lookahead: int = 16,
    pip_size: float = 0.10,
) -> LiquidityMap:
    """Build internal + external liquidity maps with sweep confirmation."""

    dual = analyze_dual_structure(candles, timeframe, internal_lookback, external_lookback)
    internal = dual["internal"]
    external = dual["external"]
    eqh_i, eql_i = detect_equal_levels(internal.swings, tolerance_pips, pip_size, internal=True)
    eqh_e, eql_e = detect_equal_levels(external.swings, tolerance_pips, pip_size, internal=False)
    sweeps = detect_sweeps(candles, eqh_i + eql_i, internal.events, bias=bias, lookahead=lookahead)
    return LiquidityMap(
        timeframe=timeframe,
        internal_levels=eqh_i + eql_i,
        external_levels=eqh_e + eql_e,
        sweeps=sweeps,
        bias=bias,
    )
