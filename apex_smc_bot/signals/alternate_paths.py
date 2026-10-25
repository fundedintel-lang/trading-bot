"""Plan A / Plan B scenario generation for every signal.

Plan A describes the path that confirms the setup and how it is managed.
Plan B describes exactly what invalidates it, where the next valid zone is and
how to flip if structure changes.  A signal without both plans is rejected.
"""

from __future__ import annotations

from typing import Any, Sequence

from apex_smc_bot.analysis.fvg import active_fvgs
from apex_smc_bot.analysis.order_blocks import OrderBlock, active_order_blocks
from apex_smc_bot.analysis.structure import BEARISH, BULLISH
from apex_smc_bot.signals.models import PendingSetup, PlanA, PlanB


def find_next_zone(
    blocks: Sequence[OrderBlock], direction: str, beyond: float, fvgs: Sequence[Any] | None = None
) -> tuple[float | None, str]:
    """Closest valid same-direction zone beyond the invalidation price."""

    candidates: list[tuple[float, str]] = []
    for ob in active_order_blocks(blocks, direction=direction):
        if direction == BULLISH and ob.midpoint < beyond:
            candidates.append((ob.midpoint, f"{ob.timeframe} {ob.kind} OB {ob.low:.2f}-{ob.high:.2f}"))
        elif direction == BEARISH and ob.midpoint > beyond:
            candidates.append((ob.midpoint, f"{ob.timeframe} {ob.kind} OB {ob.low:.2f}-{ob.high:.2f}"))
    if fvgs:
        for fvg in active_fvgs(fvgs, direction=direction):
            if direction == BULLISH and fvg.midpoint < beyond:
                candidates.append((fvg.midpoint, f"{fvg.timeframe} {fvg.kind} FVG {fvg.bottom:.2f}-{fvg.top:.2f}"))
            elif direction == BEARISH and fvg.midpoint > beyond:
                candidates.append((fvg.midpoint, f"{fvg.timeframe} {fvg.kind} FVG {fvg.bottom:.2f}-{fvg.top:.2f}"))
    if not candidates:
        return None, "No fresh zone below/above invalidations — stand down and re-map after the next BOS"
    price, description = min(candidates, key=lambda item: abs(item[0] - beyond))
    return price, description


def generate_alternate_paths(
    pending: PendingSetup,
    entry: float,
    stop_loss: float,
    tp1: float,
    tp2: float,
    tp3: float,
    order_blocks: Sequence[OrderBlock] | None = None,
    fvgs: Sequence[Any] | None = None,
) -> tuple[PlanA, PlanB]:
    """Build the two mandated scenarios for a trade plan."""

    direction = pending.direction
    bullish = direction == BULLISH
    opposite_word = "BEARISH" if bullish else "BULLISH"

    plan_a = PlanA(
        scenario=f"{direction} CONTINUATION",
        required=[
            f"Price holds above {stop_loss:.2f}" if bullish else f"Price holds below {stop_loss:.2f}",
            f"15M structure keeps making {'HL' if bullish else 'LH'} — no opposing CHoCH",
            f"TP1 at {tp1:.2f} prints before invalidation",
            f"5M closes stay on the {'buy' if bullish else 'sell'} side of the entry zone",
        ],
        next_milestone=tp1,
        trail_sl_after_tp1=entry,
        management=(
            "Close 50% at TP1, move SL to breakeven, trail the remainder toward TP2; "
            "close 25% at TP2 and trail the runner toward TP3 while HTF bias holds"
        ),
        tp1=tp1,
        tp2=tp2,
    )

    next_zone: float | None = None
    description = ""
    if order_blocks:
        next_zone, description = find_next_zone(order_blocks, direction, stop_loss, fvgs)
    else:
        description = "Re-map zones after the next BOS"

    htf_ob = None
    if pending.alignment is not None and getattr(pending.alignment, "one_h_bias", None):
        for ob in active_order_blocks(pending.alignment.one_h_bias.order_blocks, direction=direction):
            if (bullish and ob.midpoint < stop_loss) or (not bullish and ob.midpoint > stop_loss):
                htf_ob = ob
                break

    plan_b = PlanB(
        scenario=f"{('BULLISH' if bullish else 'BEARISH')} OB FAILURE — {opposite_word} CONTINUATION",
        invalidation=stop_loss,
        what_happens_if_sl_hit=(
            "Zone invalidated. Do not re-enter in the same direction until a fresh "
            "sweep + BOS prints at a higher-timeframe zone."
        ),
        next_zone=next_zone,
        next_zone_description=description,
        wait_period="Wait for 2 x 15M closes and re-run the full pipeline before any re-entry",
        possible_flip=(
            f"If 15M prints a {opposite_word} BOS/CHoCH, stand down on {direction.lower()} setups "
            + (
                f"and monitor 1H demand at {htf_ob.low:.2f}-{htf_ob.high:.2f} for continuation"
                if htf_ob and bullish
                else f"and monitor 1H supply at {htf_ob.low:.2f}-{htf_ob.high:.2f} for a reversal"
                if htf_ob
                else "and wait for 1H alignment before attempting a counter-trend entry"
            )
        ),
    )
    return plan_a, plan_b


def default_do_not_enter_if(
    pending: PendingSetup, spread_limit_pips: float = 3.5, pip_size: float = 0.10
) -> list[str]:
    """Standard invalidation checklist embedded in every signal."""

    side = "below" if pending.direction == BULLISH else "above"
    return [
        f"Price closes {side} the invalidation level {pending.stop_loss:.2f}",
        "No 5M micro-BOS within 3 candles of the zone touch",
        "Rejection candle fails (doji or continuation candle inside the zone)",
        "Volume shows continuation against the trade (absorption fails)",
        f"Spread exceeds {spread_limit_pips:.0f} pips ({spread_limit_pips * pip_size:.2f} USD)",
        "High-impact news within 30 minutes",
        "Confluence score drops below 7",
        "HTF bias flips or 1H closes through the opposing structure",
        "Price runs 15+ pips away from the entry zone before triggering (no chasing)",
    ]
