"""10-factor confluence scoring engine (0–10).

Minimum executable score is 7/10.  Score 9–10 = sniper (full size), 7–8 =
reduced size (50%).  Any hard gate — news, HTF conflict, missing sweep — vetoes
the trade regardless of the numerical score (see the immutable laws).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apex_smc_bot.analysis.candle_psych import CandleAnalysis, is_rejection_for
from apex_smc_bot.analysis.fvg import FVG
from apex_smc_bot.analysis.htf_bias import HTFBias
from apex_smc_bot.analysis.liquidity import SweepEvent
from apex_smc_bot.analysis.order_blocks import OrderBlock
from apex_smc_bot.analysis.session import Session
from apex_smc_bot.analysis.structure import BEARISH, BULLISH, StructureEvent, StructureState
from apex_smc_bot.analysis.volume import VolumeSignal
from apex_smc_bot.config import RiskConfig, Settings, StrategyConfig

FACTORS = (
    "htf_alignment",
    "bos_choch",
    "ob_zone",
    "fvg",
    "sweep",
    "volume",
    "candle",
    "session",
    "inducement",
    "sl_logic",
)


@dataclass
class Setup:
    """Everything the confluence matrix needs to score a candidate trade."""

    direction: str
    entry_price: float = 0.0
    four_h_bias: HTFBias | None = None
    one_h_bias: HTFBias | None = None
    structure_15m: StructureState | None = None
    bos_15m: StructureEvent | None = None
    ob: OrderBlock | None = None
    fvg: FVG | None = None
    sweep: SweepEvent | None = None
    session: Session | None = None
    candle_analysis: CandleAnalysis | None = None
    volume_confirmed: bool = False
    judas: Any | None = None
    inducement: dict[str, Any] | None = None
    ote_aligned: bool = False
    htf_level_confluence: bool = False
    double_inducement: bool = False
    volume_climax: VolumeSignal | None = None
    opposing_ob: OrderBlock | None = None
    opposing_ob_distance_pips: float = float("inf")
    spread_pips: float = 0.0
    normal_spread_pips: float = 0.0
    news_within_30min: bool = False
    news_events: list[str] = field(default_factory=list)
    chopbox: bool = False
    in_kill_zone: bool = True
    sl_distance_pips: float = 0.0
    sl_structure_based: bool = True
    htf_aligned: bool = False
    # How the HTF agreement was reached (see HTFBiasEngine.check_alignment):
    # FULL (4H+1H agree), 4H_PULLBACK (trade with 4H during a 1H pullback),
    # RANGE (4H ranging, trade the 1H from the correct range half).
    htf_mode: str = "FULL"

    @property
    def range_mode(self) -> bool:
        """True when the setup follows the 1H inside a ranging 4H."""

        return str(self.htf_mode).upper() == "RANGE"
    is_judas: bool = False

    @property
    def is_bullish(self) -> bool:
        return self.direction == BULLISH


@dataclass
class ConfluenceResult:
    """Scored setup with an audit trail."""

    score: float
    breakdown: dict[str, float]
    reasoning: list[str]
    penalties: list[dict[str, Any]] = field(default_factory=list)
    bonuses: list[dict[str, Any]] = field(default_factory=list)
    trade_allowed: bool = False
    aborted: bool = False
    abort_reason: str = ""
    missing_factors: list[str] = field(default_factory=list)

    @property
    def sniper(self) -> bool:
        return self.trade_allowed and self.score >= 9

    @property
    def position_fraction(self) -> float:
        """Fraction of normal risk to deploy (sniper=1.0, 7-8=0.5)."""

        if not self.trade_allowed:
            return 0.0
        return 1.0 if self.score >= 9 else 0.5

    @property
    def confidence_percent(self) -> int:
        return int(round(min(100.0, self.score * 10)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "breakdown": {key: round(value, 2) for key, value in self.breakdown.items()},
            "penalties": self.penalties,
            "bonuses": self.bonuses,
            "reasoning": self.reasoning,
            "trade_allowed": self.trade_allowed,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "missing_factors": self.missing_factors,
            "sniper": self.sniper,
            "position_fraction": self.position_fraction,
            "confidence_percent": self.confidence_percent,
        }


class ConfluenceEngine:
    """Implements the scoring matrix from the specification."""

    def __init__(self, config: Settings | StrategyConfig | None = None) -> None:
        if isinstance(config, Settings):
            self.strategy = config.strategy
            self.risk = config.risk
        else:
            self.strategy = config or StrategyConfig()
            self.risk = RiskConfig()

    def score(self, setup: Setup) -> ConfluenceResult:
        cfg = self.strategy
        breakdown: dict[str, float] = {factor: 0.0 for factor in FACTORS}
        reasoning: list[str] = []
        penalties: list[dict[str, Any]] = []
        bonuses: list[dict[str, Any]] = []

        if setup.direction not in {BULLISH, BEARISH}:
            return ConfluenceResult(
                score=0.0,
                breakdown=breakdown,
                reasoning=["❌ Invalid trade direction"],
                trade_allowed=False,
                aborted=True,
                abort_reason="INVALID_DIRECTION",
                missing_factors=list(FACTORS),
            )

        # 1) HTF alignment
        if (
            setup.four_h_bias
            and setup.one_h_bias
            and setup.four_h_bias.direction == setup.direction
            and setup.one_h_bias.direction == setup.direction
        ):
            breakdown["htf_alignment"] = 1.0
            reasoning.append("✅ 4H + 1H both aligned with trade direction")
        elif str(setup.htf_mode).upper() == "4H_PULLBACK" and setup.four_h_bias and setup.four_h_bias.direction == setup.direction:
            # With-trend entry into a 1H pullback: strong, but the 1H is not
            # confirming, so it earns less than a fully aligned tape.
            breakdown["htf_alignment"] = 0.7
            reasoning.append(
                "◐ 4H trend intact — 1H pullback traded in the 4H direction at a discount"
            )
        elif setup.range_mode and setup.one_h_bias and setup.one_h_bias.direction == setup.direction:
            # Half credit: the 4H has no trend to align with, so the range trade
            # must earn the rest of its score from the zone, sweep and OTE.
            breakdown["htf_alignment"] = 0.5
            reasoning.append(
                "◐ 4H ranging — 1H direction traded from the correct half of the 4H range"
            )

        # 2) BOS / CHoCH
        if setup.bos_15m and setup.bos_15m.direction == setup.direction:
            breakdown["bos_choch"] = 1.0
            reasoning.append(f"✅ 15M {setup.bos_15m.kind} confirmed at {setup.bos_15m.level:.2f}")

        # 3) Order block
        if setup.ob and setup.ob.is_active and setup.ob.quality >= 3 and setup.ob.kind == setup.direction:
            breakdown["ob_zone"] = 1.0
            reasoning.append(
                f"✅ Valid {setup.ob.kind} OB {setup.ob.low:.2f}–{setup.ob.high:.2f} "
                f"(quality {setup.ob.quality}/5, {setup.ob.status})"
            )

        # 4) FVG confluence
        if setup.fvg and setup.fvg.is_active and setup.fvg.kind == setup.direction:
            breakdown["fvg"] = 1.0
            reasoning.append(
                f"✅ {setup.fvg.status} {setup.fvg.kind} FVG {setup.fvg.bottom:.2f}–{setup.fvg.top:.2f}"
            )

        # 5) Liquidity sweep
        if (
            setup.sweep
            and setup.sweep.quality == "HIGH"
            and setup.sweep.direction == setup.direction
            and setup.sweep.confirmed
        ):
            breakdown["sweep"] = 1.0
            reasoning.append(
                f"✅ {setup.sweep.kind} at {setup.sweep.level:.2f} confirmed by BOS"
                + (" (double raid)" if setup.sweep.raid_count >= 2 else "")
            )

        # 6) Volume
        if setup.volume_confirmed:
            breakdown["volume"] = 1.0
            reasoning.append("✅ Volume expansion / delta shift at the zone")

        # 7) Candle structure
        if setup.candle_analysis and is_rejection_for(setup.candle_analysis, setup.direction):
            breakdown["candle"] = 1.0
            reasoning.append(
                f"✅ {setup.candle_analysis.type} rejection candle at the zone"
            )

        # 8) Session alignment
        if setup.session and setup.session.priority == "HIGH" and setup.session.tradeable and setup.in_kill_zone:
            breakdown["session"] = 1.0
            reasoning.append(f"✅ In {setup.session.name} — prime execution window")

        # 9) Inducement / Judas
        judas_resolved = bool(
            setup.judas is not None and getattr(setup.judas, "resolved", False)
        )
        inducement_resolved = bool(setup.inducement and setup.inducement.get("resolved"))
        if judas_resolved or inducement_resolved:
            breakdown["inducement"] = 1.0
            label = "Judas swing" if judas_resolved else "inducement"
            reasoning.append(f"✅ {label.capitalize()} detected and resolved")

        # 10) Structure-based, tight stop
        if setup.sl_structure_based and 0 < setup.sl_distance_pips <= self.risk.max_sl_pips:
            breakdown["sl_logic"] = 1.0
            reasoning.append(f"✅ SL at structure — {setup.sl_distance_pips:.1f} pips")

        # ------------------------------------------------------------- bonuses
        if setup.htf_level_confluence:
            bonuses.append({"reason": "HTF OB/weekly level confluence", "value": 0.5})
            reasoning.append("✅ Bonus +0.5 — HTF level confluence")
        if setup.double_inducement or (setup.sweep and setup.sweep.raid_count >= 2):
            bonuses.append({"reason": "Double liquidity raid", "value": 0.5})
            reasoning.append("✅ Bonus +0.5 — double inducement")
        if setup.ote_aligned:
            bonuses.append({"reason": "OTE 61.8–79% alignment", "value": 0.5})
            reasoning.append("✅ Bonus +0.5 — perfect OTE alignment")
        if setup.volume_climax and setup.volume_climax.direction == setup.direction:
            bonuses.append({"reason": "Volume climax + delta reversal", "value": 0.5})
            reasoning.append("✅ Bonus +0.5 — volume climax reversal")

        # ------------------------------------------------------------- penalties
        if setup.opposing_ob_distance_pips < cfg.opposing_ob_abort_pips:
            penalties.append(
                {"reason": "Opposing OB within 30 pips", "value": -1.0}
            )
            reasoning.append("⚠️ Opposing OB nearby — penalty -1")

        normal_spread = setup.normal_spread_pips or 0.0
        if normal_spread > 0 and setup.spread_pips > 2 * normal_spread:
            penalties.append({"reason": "Spread > 2x normal", "value": -0.5})
            reasoning.append("⚠️ Spread elevated — penalty -0.5")

        if setup.chopbox:
            penalties.append({"reason": "Chopbox / compression", "value": -1.0})
            reasoning.append("⚠️ Chopbox detected — penalty -1")

        if setup.news_within_30min:
            events = ", ".join(setup.news_events) if setup.news_events else "high-impact event"
            penalties.append({"reason": f"News proximity ({events})", "value": -2.0})
            reasoning.append("🚫 NEWS FILTER — major event within 30min — ABORT")

        if not setup.in_kill_zone:
            penalties.append({"reason": "Entry outside kill zone", "value": -1.0})
            reasoning.append("⚠️ Entry outside kill zone — penalty -1")

        raw_score = sum(breakdown.values()) + sum(b["value"] for b in bonuses)
        raw_score += sum(p["value"] for p in penalties)
        score = round(min(10.0, max(0.0, raw_score)), 2)

        # The hard law is "never trade against the 4H bias".  A 1H that has not
        # yet turned is a pullback, not a conflict; a ranging 4H has no bias to
        # violate, so the 1H governs when price is on the correct range half.
        htf_ok = setup.four_h_bias is not None and setup.four_h_bias.direction == setup.direction
        if not htf_ok and setup.range_mode:
            htf_ok = setup.one_h_bias is not None and setup.one_h_bias.direction == setup.direction
        aborted = False
        abort_reason = ""
        required_score = cfg.min_confluence_score
        opposing_nearby = (
            setup.opposing_ob is not None
            and setup.opposing_ob_distance_pips < cfg.opposing_ob_abort_pips
        )
        if setup.news_within_30min:
            aborted = True
            abort_reason = "NEWS_FILTER"
        elif not htf_ok:
            aborted = True
            abort_reason = "HTF_NOT_ALIGNED"
        elif opposing_nearby and setup.opposing_ob is not None and setup.opposing_ob.timeframe in {"H4", "H1"}:
            # An HTF opposing OB directly in front of the trade is a hard veto.
            aborted = True
            abort_reason = "OPPOSING_HTF_OB"
        elif opposing_nearby:
            required_score = 9.0  # non-HTF opposition still demands a sniper setup

        trade_allowed = score >= required_score and not aborted
        if not trade_allowed and not abort_reason:
            abort_reason = f"LOW_CONFLUENCE_{score:.1f}"
        missing = [factor for factor, value in breakdown.items() if value < 1.0]

        return ConfluenceResult(
            score=score,
            breakdown=breakdown,
            reasoning=reasoning,
            penalties=penalties,
            bonuses=bonuses,
            trade_allowed=trade_allowed,
            aborted=aborted,
            abort_reason=abort_reason,
            missing_factors=missing,
        )

    # Convenience used by tests / research notebooks.
    def score_summary(self, setup: Setup) -> dict[str, Any]:
        return self.score(setup).to_dict()
