"""Smart-money simulation: would a $500M desk take this trade?

This is a *sanity check*, not a crystal ball.  It asks whether the setup sits
where institutions actually transact: OTE/discount, after a confirmed raid, from
a high-quality OB, with absorption and (optionally) a volume node nearby.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from apex_smc_bot.analysis.premium_discount import DealingRange
from apex_smc_bot.analysis.structure import BEARISH, BULLISH
from apex_smc_bot.signals.models import PendingSetup


@dataclass
class SmartMoneyAssessment:
    score: int
    would_institutions_trade: bool
    phase: str  # ACCUMULATION | DISTRIBUTION
    assessment: list[str]
    hundred_million_risk: bool
    verdict: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "would_institutions_trade": self.would_institutions_trade,
            "phase": self.phase,
            "assessment": self.assessment,
            "100m_risk_assessment": "YES" if self.hundred_million_risk else "NO",
            "verdict": self.verdict,
        }


def simulate_smart_money(
    pending: PendingSetup,
    confluence_score: float,
    dealing_range: DealingRange | None = None,
    hvn_levels: Sequence[float] | None = None,
    absorption: bool = False,
    volume_confirmed: bool = False,
) -> SmartMoneyAssessment:
    """Score institutional intent from 0 to 8 points."""

    score = 0
    notes: list[str] = []
    direction = pending.direction
    entry = pending.entry_price
    bullish = direction == BULLISH

    # 1) Is the entry in the institutional OTE window?
    if dealing_range is not None and dealing_range.in_ote(entry):
        score += 2
        notes.append("✅ Entry inside the 61.8–79% OTE window")
    elif dealing_range is not None and (
        (bullish and dealing_range.in_discount(entry)) or (not bullish and dealing_range.in_premium(entry))
    ):
        score += 1
        notes.append(
            "✅ Entry on the correct side of equilibrium "
            f"({'discount' if bullish else 'premium'})"
        )

    # 2) Was liquidity taken before the entry?
    sweep = pending.sweep
    if sweep is not None and getattr(sweep, "confirmed", False):
        score += 2
        notes.append(
            f"✅ {sweep.kind} at {sweep.level:.2f} confirmed by BOS — stop hunt completed"
        )

    # 3) Order-block quality
    ob = pending.ob
    if ob is not None and getattr(ob, "quality", 0) >= 4:
        score += 1
        notes.append(f"✅ High-quality {ob.kind} OB ({ob.quality}/5) at the origin")
    elif ob is not None:
        notes.append(f"• OB quality {getattr(ob, 'quality', 0)}/5 — acceptable, not pristine")

    # 4) Absorption / displacement evidence
    if absorption:
        score += 2
        notes.append("✅ Absorption candle at the zone — retail flow is being filled")
    elif volume_confirmed:
        score += 1
        notes.append("✅ Volume expansion supports the reaction")

    # 5) Nearby high-volume node
    if hvn_levels:
        nearest = min(hvn_levels, key=lambda level: abs(level - entry))
        if abs(nearest - entry) <= 1.0:  # 10 pips
            score += 1
            notes.append(f"✅ HVN at {nearest:.2f} — institutional positioning level")

    would_trade = score >= 6
    hundred_million = would_trade and confluence_score >= 8
    phase = "ACCUMULATION" if bullish else "DISTRIBUTION"
    if would_trade:
        verdict = (
            f"{phase} — institutional flow supports a {direction.lower()} position "
            "with the stop beyond the raid"
        )
    else:
        verdict = "Setup is tradable only with reduced size — institutional confluence is thin"
    return SmartMoneyAssessment(
        score=score,
        would_institutions_trade=would_trade,
        phase=phase,
        assessment=notes,
        hundred_million_risk=hundred_million,
        verdict=verdict,
    )
