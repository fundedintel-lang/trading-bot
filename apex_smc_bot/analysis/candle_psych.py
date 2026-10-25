"""Candle anatomy / psychology scoring used by the 5M trigger layer.

The engine never enters on a *zone touch* alone: the candle that reacts inside
the zone must show intent (engulfing, pin, absorption, momentum).  Every pattern
is scored so the confluence engine can consume a single ``signal_strength``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from apex_smc_bot.analysis.structure import BEARISH, BULLISH
from apex_smc_bot.data.candle_manager import Candle

# Pattern names
BULLISH_ENGULFING = "BULLISH_ENGULFING"
BEARISH_ENGULFING = "BEARISH_ENGULFING"
HAMMER = "HAMMER"
SHOOTING_STAR = "SHOOTING_STAR"
DOJI = "DOJI"
INSIDE_BAR = "INSIDE_BAR"
OUTSIDE_BAR = "OUTSIDE_BAR"
BULLISH_ABSORPTION = "BULLISH_ABSORPTION"
BEARISH_ABSORPTION = "BEARISH_ABSORPTION"
EXHAUSTION_WICK = "EXHAUSTION_WICK"
STRONG_BULLISH_MOMENTUM = "STRONG_BULLISH_MOMENTUM"
STRONG_BEARISH_MOMENTUM = "STRONG_BEARISH_MOMENTUM"
NEUTRAL = "NEUTRAL"


@dataclass(frozen=True, slots=True)
class CandleAnalysis:
    """Result of scoring one candle."""

    candle: Candle
    type: str
    direction: str
    body_ratio: float
    upper_wick: float
    lower_wick: float
    range: float
    signal_strength: int  # -3..+3 ; positive supports longs
    volume_delta: float | None = None
    structure_pattern: str = NEUTRAL
    notes: tuple[str, ...] = ()

    @property
    def is_bullish(self) -> bool:
        return self.direction == BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.direction == BEARISH

    @property
    def is_rejection(self) -> bool:
        """True when the pattern is a valid rejection for its own direction."""

        return self.signal_strength >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "direction": self.direction,
            "body_ratio": round(self.body_ratio, 3),
            "upper_wick": round(self.upper_wick, 5),
            "lower_wick": round(self.lower_wick, 5),
            "range": round(self.range, 5),
            "signal_strength": self.signal_strength,
            "volume_delta": round(self.volume_delta, 3) if self.volume_delta is not None else None,
            "timestamp": self.candle.timestamp.isoformat(),
            "close": self.candle.close,
            "notes": list(self.notes),
        }


def _engulfs(current: Candle, previous: Candle, direction: str) -> bool:
    if direction == BULLISH:
        return (
            current.close > current.open
            and previous.close < previous.open
            and current.close >= previous.open
            and current.open <= previous.close
        )
    return (
        current.close < current.open
        and previous.close > previous.open
        and current.close <= previous.open
        and current.open >= previous.close
    )


def analyze_candle(
    candle: Candle,
    previous: Candle | None = None,
    volume_delta: float | None = None,
    average_volume: float | None = None,
    prior_move: float | None = None,
) -> CandleAnalysis:
    """Score a candle's anatomy and (optional) volume behaviour.

    Args:
        candle: Candle to analyse.
        previous: Preceding candle, required for engulfing/inside/outside patterns.
        volume_delta: Estimated buy-sell volume delta for this candle.
        average_volume: Trailing average volume (for exhaustion confirmation).
        prior_move: Signed size of the move into this candle (for exhaustion).
    """

    body = candle.body
    upper_wick = candle.upper_wick
    lower_wick = candle.lower_wick
    total_range = candle.range
    body_ratio = body / total_range if total_range > 0 else 0.0
    direction = candle.direction
    notes: list[str] = []
    pattern = NEUTRAL
    strength = 0

    # 1) Indecision first — a doji is never a trigger.
    if body_ratio < 0.1:
        pattern = DOJI
        notes.append("Doji — indecision, wait for the next candle")
        analysis_type = DOJI
    # 2) Engulfing (needs a previous candle)
    elif previous is not None and _engulfs(candle, previous, BULLISH):
        pattern = BULLISH_ENGULFING
        strength = 3 if body_ratio > 0.7 else 2
        analysis_type = BULLISH_ENGULFING
        notes.append(f"Bullish engulfing (body ratio {body_ratio:.2f})")
    elif previous is not None and _engulfs(candle, previous, BEARISH):
        pattern = BEARISH_ENGULFING
        strength = 3 if body_ratio > 0.7 else 2
        analysis_type = BEARISH_ENGULFING
        notes.append(f"Bearish engulfing (body ratio {body_ratio:.2f})")
    # 3) Pin bars
    elif lower_wick > 2 * body and upper_wick < 0.5 * body and body_ratio < 0.3:
        pattern = HAMMER
        strength = 2
        analysis_type = HAMMER
        notes.append("Hammer / bullish pin — sellers rejected at the low")
    elif upper_wick > 2 * body and lower_wick < 0.5 * body and body_ratio < 0.3:
        pattern = SHOOTING_STAR
        strength = 2
        analysis_type = SHOOTING_STAR
        notes.append("Shooting star / bearish pin — buyers rejected at the high")
    # 4) Momentum candles
    elif body_ratio > 0.7:
        pattern = STRONG_BULLISH_MOMENTUM if direction == BULLISH else STRONG_BEARISH_MOMENTUM
        strength = 3 if direction == BULLISH else -3
        analysis_type = pattern
        notes.append(f"Strong {direction.lower()} momentum (body ratio {body_ratio:.2f})")
    # 5) Compression / expansion
    elif previous is not None and candle.high <= previous.high and candle.low >= previous.low:
        pattern = INSIDE_BAR
        analysis_type = INSIDE_BAR
        notes.append("Inside bar — compression before expansion")
    elif previous is not None and candle.high > previous.high and candle.low < previous.low:
        pattern = OUTSIDE_BAR
        analysis_type = OUTSIDE_BAR
        notes.append("Outside bar — volatile rejection / trap zone")
    else:
        analysis_type = NEUTRAL

    # 6) Absorption: body direction disagrees with the underlying delta.
    if volume_delta is not None:
        if candle.close > candle.open and volume_delta < 0:
            pattern = BEARISH_ABSORPTION
            strength = -2
            analysis_type = BEARISH_ABSORPTION
            notes.append("Bearish absorption — green candle on net selling")
        elif candle.close < candle.open and volume_delta > 0:
            pattern = BULLISH_ABSORPTION
            strength = 2
            analysis_type = BULLISH_ABSORPTION
            notes.append("Bullish absorption — red candle on net buying")

    # 7) Exhaustion wick with volume confirmation.
    if total_range > 0 and average_volume is not None and candle.volume > 1.5 * average_volume:
        wick_ratio = max(upper_wick, lower_wick) / total_range
        if wick_ratio > 0.6 and prior_move is not None and abs(prior_move) > total_range * 1.5:
            if upper_wick > lower_wick:
                pattern = EXHAUSTION_WICK
                strength = -1
                notes.append("Exhaustion wick at the high with volume spike")
            else:
                pattern = EXHAUSTION_WICK
                strength = 1
                notes.append("Exhaustion wick at the low with volume spike")
            analysis_type = EXHAUSTION_WICK

    return CandleAnalysis(
        candle=candle,
        type=analysis_type,
        direction=direction,
        body_ratio=body_ratio,
        upper_wick=upper_wick,
        lower_wick=lower_wick,
        range=total_range,
        signal_strength=max(-3, min(3, strength)),
        volume_delta=volume_delta,
        structure_pattern=pattern,
        notes=tuple(notes),
    )


def is_rejection_for(analysis: CandleAnalysis, direction: str, min_strength: int = 2) -> bool:
    """Whether the candle is a valid rejection in the trade direction."""

    if direction == BULLISH:
        return analysis.signal_strength >= min_strength and analysis.direction in {BULLISH, "DOJI"}
    return analysis.signal_strength <= -min_strength and analysis.direction in {BEARISH, "DOJI"}


def validate_wick_ratio(candle: Candle, direction: str, max_ratio: float = 0.30) -> bool:
    """Reject entries taken after price has already run far through the zone.

    For longs the **lower** wick beyond the zone must stay below 30% of range;
    for shorts the upper wick must.
    """

    if candle.range <= 0:
        return False
    if direction == BULLISH:
        return candle.lower_wick / candle.range < max_ratio
    return candle.upper_wick / candle.range < max_ratio


def analyse_sequence(candles: Sequence[Candle], lookback: int = 3) -> list[CandleAnalysis]:
    """Analyse the final ``lookback`` candles (useful for dashboards/tests)."""

    if not candles:
        return []
    window = candles[-max(1, lookback):]
    output: list[CandleAnalysis] = []
    for index, candle in enumerate(window):
        global_index = len(candles) - len(window) + index
        previous = candles[global_index - 1] if global_index > 0 else None
        output.append(analyze_candle(candle, previous=previous))
    return output
