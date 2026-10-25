"""Caution-zone detectors from Section 17 of the specification.

These checks are deliberately *hard*: chopboxes, spread spikes, revenge-trade
windows, dead sessions and missed entries abort the setup.  An opposing order
block only halves the risk (unless it is an HTF block, which the confluence
engine vetoes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from apex_smc_bot.analysis.order_blocks import OrderBlock, nearest_opposing_ob
from apex_smc_bot.analysis.session import Session
from apex_smc_bot.analysis.structure import StructureState
from apex_smc_bot.config import RiskConfig, StrategyConfig
from apex_smc_bot.data.candle_manager import Candle
from apex_smc_bot.risk.drawdown_monitor import DrawdownDecision
from apex_smc_bot.risk.spread_monitor import SpreadState
from apex_smc_bot.signals.models import PendingSetup


@dataclass
class CautionResult:
    flags: list[str] = field(default_factory=list)
    abort: bool = False
    risk_multiplier: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flags": self.flags,
            "abort": self.abort,
            "risk_multiplier": self.risk_multiplier,
            "details": self.details,
            "notes": self.notes,
        }


def detect_chopbox(
    candles: Sequence[Candle],
    lookback: int = 10,
    max_range_pips: float = 20.0,
    pip_size: float = 0.10,
    breakout_index: int | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Tight range without a fresh BOS = chopbox (skip entries)."""

    if len(candles) < lookback:
        return False, {"reason": "insufficient candles"}
    window = candles[-lookback:]
    high = max(c.high for c in window)
    low = min(c.low for c in window)
    range_pips = (high - low) / pip_size if pip_size > 0 else 0.0
    total = len(candles) - 1
    broke_out = breakout_index is not None and total - breakout_index <= 2
    chopbox = range_pips < max_range_pips and not broke_out
    return chopbox, {
        "range_pips": round(range_pips, 1),
        "high": round(high, 2),
        "low": round(low, 2),
        "lookback": lookback,
        "recent_bos": broke_out,
    }


def evaluate_caution_zones(
    pending: PendingSetup,
    session: Session,
    price: float,
    candles_15m: Sequence[Candle],
    spread_state: SpreadState | None = None,
    drawdown: DrawdownDecision | None = None,
    order_blocks: Sequence[OrderBlock] | None = None,
    strategy: StrategyConfig | None = None,
    risk: RiskConfig | None = None,
    structure: StructureState | None = None,
    now: datetime | None = None,
) -> CautionResult:
    """Run all Section-17 detectors for a pending setup."""

    strategy = strategy or StrategyConfig()
    risk = risk or RiskConfig()
    pip = risk.pip_size
    result = CautionResult()

    # 1) Chopbox
    bos_index = getattr(pending.bos_event, "index", None)
    chopbox, chop_detail = detect_chopbox(
        candles_15m,
        strategy.chopbox_lookback,
        strategy.chopbox_max_range_pips,
        pip,
        breakout_index=bos_index,
    )
    result.details["chopbox"] = chop_detail
    if chopbox:
        result.flags.append("CHOPBOX_DETECTED")
        result.abort = True
        result.notes.append("Tight 15M range without a BOS — engineered retail trap")

    # 2) Spread spike / pause
    if spread_state is not None:
        result.details["spread"] = spread_state.to_dict()
        if spread_state.spike:
            result.flags.append("SPREAD_SPIKE")
            result.abort = True
        if not spread_state.allowed:
            result.flags.append("SPREAD_NOT_TRADEABLE")
            result.abort = True

    # 3) Revenge-trade / loss cooldown
    if drawdown is not None:
        result.details["drawdown"] = drawdown.to_dict()
        if not drawdown.allow_trading:
            if "COOLDOWN" in drawdown.reason.upper():
                result.flags.append("REVENGE_TRADE_COOLDOWN")
            if "CIRCUIT_BREAKER" in drawdown.reason.upper():
                result.flags.append("CIRCUIT_BREAKER")
            result.abort = True

    # 4) Dead session
    if not session.tradeable:
        result.flags.append("DEAD_SESSION")
        result.abort = True
        result.notes.append(f"{session.name} is not an approved execution window")

    # 5) Opposing OB nearby (halve risk; HTF blocks veto in confluence)
    combined = list(order_blocks or [])
    if pending.alignment is not None:
        for bias in (getattr(pending.alignment, "one_h", None), getattr(pending.alignment, "four_h", None)):
            if bias is not None:
                combined.extend(getattr(bias, "order_blocks", []) or [])
    opposing = nearest_opposing_ob(combined, pending.entry_price, pending.direction)
    if opposing is not None:
        distance_pips = opposing.distance_to(pending.entry_price) / pip
        result.details["opposing_ob"] = {
            "timeframe": opposing.timeframe,
            "kind": opposing.kind,
            "zone": [round(opposing.low, 2), round(opposing.high, 2)],
            "distance_pips": round(distance_pips, 1),
        }
        if distance_pips < strategy.opposing_ob_warning_pips:
            result.flags.append("OPPOSING_OB_NEARBY")
            result.risk_multiplier = min(result.risk_multiplier, 0.5)
            result.notes.append(
                f"Active {opposing.timeframe} {opposing.kind} OB {distance_pips:.1f} pips away — half size"
            )

    # 6) Spring / fake compression: no sweep before the OB touch
    if pending.sweep is None:
        result.flags.append("NO_SWEEP_BEFORE_ZONE")
        result.abort = True
        result.notes.append("No liquidity raid preceded the zone — spring trap risk")

    # 7) Missed entry / chase guard
    distance_pips = abs(price - pending.entry_price) / pip
    result.details["distance_to_entry_pips"] = round(distance_pips, 1)
    if distance_pips > strategy.max_entry_distance_pips:
        result.flags.append("MISSED_ENTRY")
        result.abort = True
        result.notes.append("Price is too far from the entry zone — do not chase")

    # 8) Structure conflict at execution time
    if (
        strategy.require_15m_structure
        and structure is not None
        and structure.direction not in {pending.direction, "RANGING"}
    ):
        result.flags.append("15M_STRUCTURE_FLIPPED")
        result.abort = True
        result.notes.append("15M structure flipped against the setup while pending")

    return result
