"""Market structure: swings, HH/HL/LH/LL labels, BOS and CHoCH.

All detection is strictly causal: a swing at index ``i`` only becomes *known* at
``i + lookback`` and a BOS can only be confirmed by a candle that closes after
that confirmation.  This makes the same functions safe to use in backtests and
live trading without look-ahead bias.

Definitions (per the APEX-SMC specification):
  * Swing high: high greater than ``N`` candles left **and** right (N=3 default).
  * Swing low:  low lower than ``N`` candles left **and** right.
  * BOS:        candle **close** beyond a confirmed swing (wicks do not count).
  * CHoCH:      first break against the prevailing structure direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from apex_smc_bot.data.candle_manager import Candle

SWING_HIGH = "SWING_HIGH"
SWING_LOW = "SWING_LOW"
BULLISH = "BULLISH"
BEARISH = "BEARISH"
RANGING = "RANGING"


@dataclass(frozen=True, slots=True)
class SwingPoint:
    """A confirmed fractal swing point."""

    index: int
    timestamp: datetime
    price: float
    kind: str  # SWING_HIGH | SWING_LOW
    confirmed_index: int
    label: str = ""  # HH | HL | LH | LL (assigned later)

    @property
    def is_high(self) -> bool:
        return self.kind == SWING_HIGH

    def __str__(self) -> str:
        return f"{self.kind}({self.label or '?'}) @ {self.price:.2f} idx={self.index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp.isoformat(),
            "price": round(self.price, 5),
            "kind": self.kind,
            "confirmed_index": self.confirmed_index,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class StructureEvent:
    """A confirmed BOS or CHoCH."""

    kind: str  # BOS | CHOCH
    direction: str  # BULLISH | BEARISH
    level: float
    index: int
    timestamp: datetime
    broken_swing_index: int
    lookback: int = 3

    @property
    def is_bos(self) -> bool:
        return self.kind == "BOS"

    @property
    def is_choch(self) -> bool:
        return self.kind == "CHOCH"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "direction": self.direction,
            "level": round(self.level, 5),
            "index": self.index,
            "timestamp": self.timestamp.isoformat(),
            "broken_swing_index": self.broken_swing_index,
        }


@dataclass
class StructureState:
    """Result of analysing one timeframe's structure."""

    timeframe: str
    direction: str = RANGING
    strength: str = "WEAK"
    swings: list[SwingPoint] = field(default_factory=list)
    events: list[StructureEvent] = field(default_factory=list)
    last_bos: StructureEvent | None = None
    last_choch: StructureEvent | None = None
    last_swing_high: SwingPoint | None = None
    last_swing_low: SwingPoint | None = None
    lookback: int = 3

    @property
    def is_bullish(self) -> bool:
        return self.direction == BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.direction == BEARISH

    @property
    def swing_highs(self) -> list[SwingPoint]:
        return [s for s in self.swings if s.is_high]

    @property
    def swing_lows(self) -> list[SwingPoint]:
        return [s for s in self.swings if not s.is_high]

    @property
    def dealing_range(self) -> tuple[float, float] | None:
        """Most recent swing low/high pair, ordered ``(low, high)``."""

        if self.last_swing_high is None or self.last_swing_low is None:
            return None
        low, high = self.last_swing_low.price, self.last_swing_high.price
        return (low, high) if low < high else (high, low)

    def recent_highs(self, count: int = 3) -> list[SwingPoint]:
        return self.swing_highs[-count:]

    def recent_lows(self, count: int = 3) -> list[SwingPoint]:
        return self.swing_lows[-count:]

    def labels(self) -> list[str]:
        return [s.label for s in self.swings if s.label]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "direction": self.direction,
            "strength": self.strength,
            "last_bos": self.last_bos.to_dict() if self.last_bos else None,
            "last_choch": self.last_choch.to_dict() if self.last_choch else None,
            "last_swing_high": self.last_swing_high.to_dict() if self.last_swing_high else None,
            "last_swing_low": self.last_swing_low.to_dict() if self.last_swing_low else None,
            "swings": [s.to_dict() for s in self.swings[-12:]],
            "events": [e.to_dict() for e in self.events[-8:]],
            "labels": self.labels()[-8:],
        }


# --------------------------------------------------------------------------- swings
def detect_swings(candles: Sequence[Candle], lookback: int = 3) -> list[SwingPoint]:
    """Detect confirmed fractal swing highs and lows.

    A swing is confirmed ``lookback`` candles after it prints, which is recorded
    in :attr:`SwingPoint.confirmed_index`.
    """

    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    n = len(candles)
    swings: list[SwingPoint] = []
    for i in range(lookback, n - lookback):
        candle = candles[i]
        left = candles[i - lookback:i]
        right = candles[i + 1:i + 1 + lookback]
        if candle.high > max(c.high for c in left) and candle.high > max(c.high for c in right):
            swings.append(
                SwingPoint(
                    index=i,
                    timestamp=candle.timestamp,
                    price=candle.high,
                    kind=SWING_HIGH,
                    confirmed_index=i + lookback,
                )
            )
        if candle.low < min(c.low for c in left) and candle.low < min(c.low for c in right):
            swings.append(
                SwingPoint(
                    index=i,
                    timestamp=candle.timestamp,
                    price=candle.low,
                    kind=SWING_LOW,
                    confirmed_index=i + lookback,
                )
            )
    swings.sort(key=lambda s: s.index)
    return label_swings(swings)


def label_swings(swings: Sequence[SwingPoint]) -> list[SwingPoint]:
    """Assign HH/HL/LH/LL labels by comparing consecutive swings of the same kind."""

    labelled: list[SwingPoint] = []
    last_high: float | None = None
    last_low: float | None = None
    for swing in swings:
        label = ""
        if swing.is_high:
            if last_high is not None:
                label = "HH" if swing.price > last_high else ("LH" if swing.price < last_high else "EQH")
            last_high = swing.price
        else:
            if last_low is not None:
                label = "HL" if swing.price > last_low else ("LL" if swing.price < last_low else "EQL")
            last_low = swing.price
        labelled.append(
            SwingPoint(
                index=swing.index,
                timestamp=swing.timestamp,
                price=swing.price,
                kind=swing.kind,
                confirmed_index=swing.confirmed_index,
                label=label,
            )
        )
    return labelled


# --------------------------------------------------------------------------- events
def detect_structure_events(
    candles: Sequence[Candle], swings: Sequence[SwingPoint]
) -> list[StructureEvent]:
    """Single causal pass detecting BOS/CHoCH events.

    A close beyond the most recent unbroken, already-confirmed swing triggers an
    event.  If the break agrees with the prevailing direction it is a BOS, if it
    opposes it the event is a CHoCH and the direction flips.
    """

    if not candles:
        return []
    highs = [s for s in swings if s.is_high]
    lows = [s for s in swings if not s.is_high]
    broken_highs: set[int] = set()
    broken_lows: set[int] = set()
    events: list[StructureEvent] = []
    trend = RANGING

    for i, candle in enumerate(candles):
        confirmed_highs = [s for s in highs if s.confirmed_index <= i and s.index not in broken_highs]
        confirmed_lows = [s for s in lows if s.confirmed_index <= i and s.index not in broken_lows]

        if confirmed_highs:
            candidate = confirmed_highs[-1]
            if candle.close > candidate.price:
                kind = "BOS" if trend == BULLISH else "CHOCH"
                events.append(
                    StructureEvent(
                        kind=kind,
                        direction=BULLISH,
                        level=candidate.price,
                        index=i,
                        timestamp=candle.timestamp,
                        broken_swing_index=candidate.index,
                    )
                )
                trend = BULLISH
                for swing in confirmed_highs:
                    broken_highs.add(swing.index)

        if confirmed_lows:
            candidate = confirmed_lows[-1]
            if candle.close < candidate.price:
                kind = "BOS" if trend == BEARISH else "CHOCH"
                events.append(
                    StructureEvent(
                        kind=kind,
                        direction=BEARISH,
                        level=candidate.price,
                        index=i,
                        timestamp=candle.timestamp,
                        broken_swing_index=candidate.index,
                    )
                )
                trend = BEARISH
                for swing in confirmed_lows:
                    broken_lows.add(swing.index)
    return events


def _determine_direction(swings: Sequence[SwingPoint], events: Sequence[StructureEvent]) -> tuple[str, str]:
    """Score recent labels + last event to derive direction and strength."""

    recent = [s for s in swings if s.label][-6:]
    score = 0
    for swing in recent:
        if swing.label == "HH":
            score += 1
        elif swing.label == "LH":
            score -= 1
        elif swing.label == "HL":
            score += 1
        elif swing.label == "LL":
            score -= 1

    if events:
        last = events[-1]
        score += 2 if last.direction == BULLISH else -2
        # A CHoCH immediately before the last event has lower weight.
        if len(events) >= 2 and events[-2].kind == "CHOCH" and events[-2].direction == last.direction:
            score += 1 if last.direction == BULLISH else -1

    if score >= 3:
        direction = BULLISH
    elif score <= -3:
        direction = BEARISH
    else:
        direction = RANGING

    recent_labels = [s.label for s in recent]
    aligned = (
        direction == BULLISH
        and "HH" in recent_labels
        and "HL" in recent_labels
        and not any(lbl in {"LL"} for lbl in recent_labels[-2:])
    ) or (
        direction == BEARISH
        and "LL" in recent_labels
        and "LH" in recent_labels
        and not any(lbl in {"HH"} for lbl in recent_labels[-2:])
    )
    strength = "STRONG" if aligned else ("MODERATE" if direction != RANGING else "WEAK")
    return direction, strength


def analyze_structure(
    candles: Sequence[Candle], timeframe: str = "M15", lookback: int = 3
) -> StructureState:
    """Full structure analysis for one timeframe."""

    if not candles:
        return StructureState(timeframe=timeframe, lookback=lookback)
    swings = detect_swings(candles, lookback=lookback)
    events = detect_structure_events(candles, swings)
    direction, strength = _determine_direction(swings, events)
    bos_events = [e for e in events if e.kind == "BOS"]
    choch_events = [e for e in events if e.kind == "CHOCH"]
    highs = [s for s in swings if s.is_high]
    lows = [s for s in swings if not s.is_high]
    return StructureState(
        timeframe=timeframe,
        direction=direction,
        strength=strength,
        swings=swings,
        events=events,
        last_bos=bos_events[-1] if bos_events else None,
        last_choch=choch_events[-1] if choch_events else None,
        last_swing_high=highs[-1] if highs else None,
        last_swing_low=lows[-1] if lows else None,
        lookback=lookback,
    )


def analyze_dual_structure(
    candles: Sequence[Candle],
    timeframe: str,
    internal_lookback: int = 3,
    external_lookback: int = 8,
) -> dict[str, StructureState]:
    """Return internal (LTF precision) and external (HTF major) structure."""

    return {
        "internal": analyze_structure(candles, timeframe, internal_lookback),
        "external": analyze_structure(candles, timeframe, external_lookback),
    }


def last_event_in_direction(
    events: Sequence[StructureEvent], direction: str, kind: str | None = None
) -> StructureEvent | None:
    """Return the most recent event matching ``direction`` (and optional ``kind``)."""

    for event in reversed(events):
        if event.direction != direction:
            continue
        if kind is not None and event.kind != kind:
            continue
        return event
    return None


def has_recent_event(
    events: Sequence[StructureEvent],
    direction: str,
    within_candles: int,
    total_candles: int,
    kind: str | None = None,
) -> bool:
    """True when a matching event occurred within the last ``within_candles`` bars."""

    cutoff = total_candles - within_candles
    for event in reversed(events):
        if event.index < cutoff:
            break
        if event.direction == direction and (kind is None or event.kind == kind):
            return True
    return False


def detect_micro_bos(
    candles: Sequence[Candle],
    direction: str,
    lookback: int = 1,
    search: int = 8,
    within_bars: int = 1,
) -> dict[str, Any] | None:
    """Detect a 5M micro break of the most recent minor swing (entry trigger).

    Args:
        candles: 5M series, oldest → newest.
        direction: Trade direction the break must support.
        lookback: Fractal strength for the minor swings.
        search: How many bars back a swing may sit and still count.
        within_bars: How many of the most recent bars may contain the break.
            ``1`` requires the break on the current bar; ``3`` allows the
            structure shift to have printed on the zone-touch bar.

    Returns:
        A dict with the broken level/index, or ``None``.
    """

    if len(candles) < lookback * 2 + 2:
        return None
    swings = detect_swings(candles, lookback=lookback)
    within = max(1, min(int(within_bars), len(candles) - 1))
    last_index = len(candles) - 1
    if direction == BULLISH:
        candidates = [s for s in swings if s.is_high and last_index - search - 1 <= s.index < last_index]
        if not candidates:
            return None
        level = candidates[-1].price
        # A close beyond the level, with the previous close still inside.
        broke = [
            offset
            for offset in range(within)
            if candles[last_index - offset].close > level
            and candles[last_index - offset - 1].close <= level
        ]
        if broke:
            return {
                "level": level,
                "index": candidates[-1].index,
                "direction": "BULLISH",
                "confirmed": True,
                "bars_ago": broke[0],
            }
    else:
        candidates = [s for s in swings if not s.is_high and last_index - search - 1 <= s.index < last_index]
        if not candidates:
            return None
        level = candidates[-1].price
        broke = [
            offset
            for offset in range(within)
            if candles[last_index - offset].close < level
            and candles[last_index - offset - 1].close >= level
        ]
        if broke:
            return {
                "level": level,
                "index": candidates[-1].index,
                "direction": "BEARISH",
                "confirmed": True,
                "bars_ago": broke[0],
            }
    return None


def detect_bos(candles: Sequence[Candle], direction: str, lookback: int = 3) -> StructureEvent | None:
    """Convenience wrapper: most recent BOS in ``direction``."""

    state = analyze_structure(candles, lookback=lookback)
    return last_event_in_direction(state.events, direction, kind="BOS")


def detect_choch(candles: Sequence[Candle], direction: str, lookback: int = 3) -> StructureEvent | None:
    """Convenience wrapper: most recent CHoCH in ``direction``."""

    state = analyze_structure(candles, lookback=lookback)
    return last_event_in_direction(state.events, direction, kind="CHOCH")
