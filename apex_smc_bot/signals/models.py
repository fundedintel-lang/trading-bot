"""Typed trade-plan models serialised into the APEX-SMC signal JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from apex_smc_bot.analysis.structure import BEARISH, BULLISH
from apex_smc_bot.data.candle_manager import to_utc

LONG = "LONG"
SHORT = "SHORT"


def side_for(direction: str) -> str:
    return LONG if direction == BULLISH else SHORT


@dataclass
class EntryPlan:
    price: float
    zone_low: float
    zone_high: float
    zone: str
    type: str = "LIMIT"
    valid_until: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "price": round(self.price, 2),
            "zone": self.zone,
            "zone_low": round(self.zone_low, 2),
            "zone_high": round(self.zone_high, 2),
            "type": self.type,
            "valid_until": to_utc(self.valid_until).isoformat().replace("+00:00", "Z")
            if self.valid_until
            else None,
        }


@dataclass
class StopLossPlan:
    price: float
    distance_pips: float
    basis: str
    structure_basis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "price": round(self.price, 2),
            "distance_pips": round(self.distance_pips, 1),
            "basis": self.basis,
            "structure_basis": self.structure_basis,
        }


@dataclass
class TakeProfit:
    price: float
    distance_pips: float
    basis: str
    size_to_close: str
    rr: float = 0.0
    optional: bool = False
    condition: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "price": round(self.price, 2),
            "distance_pips": round(self.distance_pips, 1),
            "basis": self.basis,
            "size_to_close": self.size_to_close,
            "rr": f"1:{self.rr:.2f}",
        }
        if self.optional:
            payload["condition"] = self.condition or "Optional — only if bias holds"
        return payload


@dataclass
class TakeProfitPlan:
    tp1: TakeProfit
    tp2: TakeProfit
    tp3: TakeProfit | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"tp1": self.tp1.to_dict(), "tp2": self.tp2.to_dict()}
        if self.tp3:
            data["tp3"] = self.tp3.to_dict()
        return data

    def all_levels(self) -> list[float]:
        levels = [self.tp1.price, self.tp2.price]
        if self.tp3:
            levels.append(self.tp3.price)
        return levels


@dataclass
class RiskReward:
    rr_to_tp1: float
    rr_to_tp2: float
    rr_to_tp3: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "rr_to_tp1": f"1:{self.rr_to_tp1:.2f}",
            "rr_to_tp2": f"1:{self.rr_to_tp2:.2f}",
        }
        if self.rr_to_tp3 is not None:
            data["rr_to_tp3"] = f"1:{self.rr_to_tp3:.2f}"
        return data


@dataclass
class PositionSizing:
    account_equity: float
    risk_percent: float
    risk_amount_usd: float
    sl_pips: float
    pip_value_per_lot: float
    lot_size: float
    units: float
    contract_size: float = 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_equity": round(self.account_equity, 2),
            "risk_percent": round(self.risk_percent, 2),
            "risk_amount_usd": round(self.risk_amount_usd, 2),
            "sl_pips": round(self.sl_pips, 1),
            "pip_value_per_lot": self.pip_value_per_lot,
            "lot_size": round(self.lot_size, 2),
            "units": round(self.units, 2),
            "notional_oz": round(self.lot_size * self.contract_size, 2),
        }


@dataclass
class PendingSetup:
    """A 15M armed setup waiting for the 5M trigger."""

    direction: str
    entry_price: float
    zone_low: float
    zone_high: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    created_at: datetime
    expires_at: datetime
    signal_bias: str
    session_name: str
    timeframe: str = "M15"
    zone_basis: str = ""
    structure_basis: str = ""
    sweep: Any | None = None
    ob: Any | None = None
    fvg: Any | None = None
    structure: Any | None = None
    bos_event: Any | None = None
    dealing_range: Any | None = None
    alignment: Any | None = None
    judas: Any | None = None
    inducement: dict[str, Any] | None = None
    confluent_zones: list[Any] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    tp1_basis: str = ""
    tp2_basis: str = ""
    tp3_basis: str = ""

    @property
    def is_long(self) -> bool:
        return self.direction == BULLISH

    @property
    def side(self) -> str:
        return side_for(self.direction)

    @property
    def fingerprint(self) -> str:
        """Stable identity for de-duplicating re-armed copies of the same setup.

        Deliberately built from *price geography* rather than detector indices:
        order blocks, sweeps and FVGs are re-computed on every cycle, so their
        ``index`` values drift while the trade is still the same trade.  Keying
        on the zone plus entry/stop levels means one setup can only ever be
        armed once, no matter how many cycles rediscover it.
        """

        return (
            f"{self.direction}:{round(self.zone_low, 1)}:{round(self.zone_high, 1)}:"
            f"{round(self.entry_price, 1)}:{round(self.stop_loss, 1)}"
        )

    def is_expired(self, now: datetime) -> bool:
        return to_utc(now) > to_utc(self.expires_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "side": self.side,
            "entry_price": round(self.entry_price, 2),
            "entry_zone": [round(self.zone_low, 2), round(self.zone_high, 2)],
            "zone_basis": self.zone_basis,
            "stop_loss": round(self.stop_loss, 2),
            "tp1": round(self.tp1, 2),
            "tp2": round(self.tp2, 2),
            "tp3": round(self.tp3, 2),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "session": self.session_name,
            "sweep": self.sweep.to_dict() if hasattr(self.sweep, "to_dict") else None,
            "ob": self.ob.to_dict() if hasattr(self.ob, "to_dict") else None,
            "fvg": self.fvg.to_dict() if hasattr(self.fvg, "to_dict") else None,
            "tp_basis": [self.tp1_basis, self.tp2_basis, self.tp3_basis],
            "notes": self.notes,
        }


@dataclass
class PlanA:
    scenario: str
    required: list[str]
    next_milestone: float
    trail_sl_after_tp1: float
    management: str
    tp1: float = 0.0
    tp2: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "required": self.required,
            "next_milestone": round(self.next_milestone, 2),
            "trail_sl_after_tp1": round(self.trail_sl_after_tp1, 2),
            "management": self.management,
        }


@dataclass
class PlanB:
    scenario: str
    invalidation: float
    what_happens_if_sl_hit: str
    next_zone: float | None
    next_zone_description: str
    wait_period: str
    possible_flip: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "invalidation": round(self.invalidation, 2),
            "what_happens_if_sl_hit": self.what_happens_if_sl_hit,
            "next_zone": round(self.next_zone, 2) if self.next_zone is not None else None,
            "next_zone_description": self.next_zone_description,
            "wait_period": self.wait_period,
            "possible_flip": self.possible_flip,
        }


@dataclass
class Signal:
    """A fully-formed, executable trade plan."""

    signal_id: str
    timestamp: datetime
    instrument: str
    direction: str
    entry: EntryPlan
    stop_loss: StopLossPlan
    take_profits: TakeProfitPlan
    risk_reward: RiskReward
    position_sizing: PositionSizing
    confluence_score: float
    confidence_percent: int
    signal_type: str
    session: dict[str, Any]
    analysis: dict[str, Any]
    plan_a: PlanA
    plan_b: PlanB
    do_not_enter_if: list[str]
    smart_money_simulation: dict[str, Any]
    confluence: dict[str, Any] = field(default_factory=dict)
    status: str = "GENERATED"
    expires_at: datetime | None = None

    @property
    def side(self) -> str:
        return side_for(self.direction)

    @property
    def risk_amount(self) -> float:
        return self.position_sizing.risk_amount_usd

    @property
    def is_long(self) -> bool:
        return self.direction == BULLISH

    def opposite(self) -> str:
        return BEARISH if self.is_long else BULLISH

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "timestamp": to_utc(self.timestamp).isoformat().replace("+00:00", "Z"),
            "instrument": "XAU/USD" if self.instrument.upper().startswith("XAU") else self.instrument,
            "instrument_code": self.instrument,
            "direction": "BUY" if self.is_long else "SELL",
            "bias": self.direction,
            "status": self.status,
            "entry": self.entry.to_dict(),
            "stop_loss": self.stop_loss.to_dict(),
            "take_profits": self.take_profits.to_dict(),
            "risk_reward": self.risk_reward.to_dict(),
            "position_sizing": self.position_sizing.to_dict(),
            "confluence_score": round(self.confluence_score, 1),
            "confidence_percent": self.confidence_percent,
            "signal_type": self.signal_type,
            "session": self.session,
            "analysis": self.analysis,
            "plan_a": self.plan_a.to_dict(),
            "plan_b": self.plan_b.to_dict(),
            "do_not_enter_if": self.do_not_enter_if,
            "smart_money_simulation": self.smart_money_simulation,
            "confluence": self.confluence,
            "expires_at": to_utc(self.expires_at).isoformat().replace("+00:00", "Z")
            if self.expires_at
            else None,
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str, ensure_ascii=False)


@dataclass
class TriggerResult:
    """Outcome of the 7-check 5M confirmation layer."""

    execute: bool
    checks_passed: int
    confidence: float
    entry_price: float
    reason: str = ""
    missing_checks: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    candle_analysis: Any | None = None
    volume_confirmed: bool = False
    micro_bos: dict[str, Any] | None = None
    opposing_ob: Any | None = None
    spread_pips: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "execute": self.execute,
            "checks_passed": self.checks_passed,
            "checks_total": len(self.checks) or 7,
            "confidence": round(self.confidence, 1),
            "entry_price": round(self.entry_price, 2),
            "reason": self.reason,
            "missing_checks": self.missing_checks,
            "checks": self.checks,
            "volume_confirmed": self.volume_confirmed,
            "spread_pips": round(self.spread_pips, 2),
            "candle": self.candle_analysis.to_dict() if self.candle_analysis else None,
            "micro_bos": self.micro_bos,
        }
