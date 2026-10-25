"""Stop-management rules.

Immutable sequence (LAW 11/12):
    TP1 hit  → close 50%, stop to breakeven (+ buffer)
    TP2 hit  → close 25% more, stop to TP1
    runner   → ATR / structure trail toward TP3

Stops only ever move in the trade's favour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from apex_smc_bot.analysis.structure import BEARISH, BULLISH, detect_swings
from apex_smc_bot.data.candle_manager import Candle, atr as compute_atr


@dataclass(frozen=True)
class TrailPlan:
    stop_loss: float
    stage: str
    reason: str
    triggered: bool = True

    def to_dict(self) -> dict[str, str | float | bool]:
        return {
            "stop_loss": round(self.stop_loss, 2),
            "stage": self.stage,
            "reason": self.reason,
            "triggered": self.triggered,
        }


def breakeven_stop(
    entry: float, direction: str, pip_size: float = 0.10, buffer_pips: float = 1.0
) -> float:
    """Entry ± a small buffer so commissions/spread do not stop the runner out."""

    buffer = buffer_pips * pip_size
    stop = entry + buffer if direction == BULLISH else entry - buffer
    return round(stop, 2)


def protective_stop(
    entry: float,
    tp1: float,
    tp2: float,
    direction: str,
    stage: str,
    pip_size: float = 0.10,
    buffer_pips: float = 1.0,
) -> float:
    """Stop to use after a given TP stage."""

    buffer = buffer_pips * pip_size
    if stage.upper() in {"TP1", "BREAKEVEN", "TP1_HIT"}:
        return breakeven_stop(entry, direction, pip_size, buffer_pips)
    if stage.upper() in {"TP2", "TP2_HIT", "TP1_LOCK"}:
        return round(tp1 + buffer if direction == BULLISH else tp1 - buffer, 2)
    return round(entry if stage == "ENTRY" else tp2, 2)


def atr_trail_stop(
    candles: Sequence[Candle],
    direction: str,
    multiplier: float = 2.0,
    period: int = 14,
    min_distance: float = 0.0,
) -> float | None:
    """Volatility trail from the latest close."""

    if len(candles) < period + 1:
        return None
    value = compute_atr(list(candles), period)
    if value <= 0:
        return None
    close = candles[-1].close
    distance = max(value * multiplier, min_distance)
    stop = close - distance if direction == BULLISH else close + distance
    return round(stop, 2)


def structure_trail_stop(
    candles: Sequence[Candle],
    direction: str,
    lookback: int = 2,
    buffer: float = 0.20,
) -> float | None:
    """Trail behind the most recent confirmed 5M swing."""

    if len(candles) < 10:
        return None
    swings = detect_swings(list(candles), lookback=lookback)
    if direction == BULLISH:
        lows = [s for s in swings if not s.is_high]
        return round(lows[-1].price - buffer, 2) if lows else None
    highs = [s for s in swings if s.is_high]
    return round(highs[-1].price + buffer, 2) if highs else None


def next_stop(
    entry: float,
    current_stop: float,
    direction: str,
    market_price: float,
    tp1: float,
    tp2: float,
    tp1_hit: bool,
    tp2_hit: bool,
    candles: Sequence[Candle] | None = None,
    pip_size: float = 0.10,
    buffer_pips: float = 1.0,
    atr_multiplier: float = 2.0,
    breakeven_after_tp1: bool = True,
) -> TrailPlan | None:
    """Compute the next improved stop, or ``None`` when it must not move.

    Args:
        breakeven_after_tp1: When False the stop is left untouched after TP1 and
            only protected once TP2 fills.  The runner keeps its original risk,
            which is what makes the 3R/5R targets reachable.

    The returned stop is guaranteed to be on the protective side of the current
    market price and strictly better than ``current_stop``.
    """

    if not tp1_hit:
        return None
    bullish = direction == BULLISH
    candidates: list[tuple[float, str, str]] = []

    if not tp2_hit:
        if breakeven_after_tp1:
            candidates.append(
                (
                    breakeven_stop(entry, direction, pip_size, buffer_pips),
                    "BREAKEVEN",
                    "TP1 hit — stop to breakeven",
                )
            )
    else:
        candidates.append(
            (
                protective_stop(entry, tp1, tp2, direction, "TP2", pip_size, buffer_pips),
                "TP1_LOCK",
                "TP2 hit — stop locked at TP1",
            )
        )
        if candles:
            atr_candidate = atr_trail_stop(list(candles), direction, atr_multiplier)
            if atr_candidate is not None:
                candidates.append((atr_candidate, "ATR_TRAIL", "ATR volatility trail"))
            structure_candidate = structure_trail_stop(list(candles), direction)
            if structure_candidate is not None:
                candidates.append(
                    (structure_candidate, "STRUCTURE_TRAIL", "Trail behind 5M structure")
                )

        # Direction-safety: the new stop must remain on the protective side.
        safe = [(stop, stage, reason) for stop, stage, reason in candidates if stop < market_price] if bullish else [
            (stop, stage, reason) for stop, stage, reason in candidates if stop > market_price
        ]
        candidates = safe

    best: TrailPlan | None = None
    for stop, stage, reason in candidates:
        improved = stop > current_stop + 0.02 if bullish else stop < current_stop - 0.02
        if not improved:
            continue
        if bullish and stop >= market_price:
            continue
        if not bullish and stop <= market_price:
            continue
        plan = TrailPlan(stop_loss=round(stop, 2), stage=stage, reason=reason)
        if best is None:
            best = plan
        elif bullish and plan.stop_loss > best.stop_loss:
            best = plan
        elif not bullish and plan.stop_loss < best.stop_loss:
            best = plan
    return best


def partial_units(initial_units: float, fraction: float, remaining_units: float | None = None) -> float:
    """Units to close for a partial take-profit, bounded by what remains open."""

    if initial_units <= 0:
        return 0.0
    units = initial_units * max(0.0, min(1.0, fraction))
    if remaining_units is not None:
        units = min(units, remaining_units)
    return round(units, 2)
