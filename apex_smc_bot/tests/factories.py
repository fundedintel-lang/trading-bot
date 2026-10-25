"""Test factories that build fully-formed domain objects without the pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from apex_smc_bot.analysis.candle_psych import CandleAnalysis
from apex_smc_bot.analysis.order_blocks import OrderBlock
from apex_smc_bot.analysis.session import Session
from apex_smc_bot.signals.models import (
    BEARISH,
    BULLISH,
    EntryPlan,
    PlanA,
    PlanB,
    PositionSizing,
    RiskReward,
    Signal,
    StopLossPlan,
    TakeProfit,
    TakeProfitPlan,
)


def make_session(
    name: str = "NY_KILL_ZONE",
    tradeable: bool = True,
    is_kill_zone: bool = True,
    priority: str = "HIGH",
) -> Session:
    from datetime import time

    return Session(
        name=name,
        start=time(12, 0),
        end=time(15, 0),
        tradeable=tradeable,
        priority=priority,
        is_kill_zone=is_kill_zone,
    )


def make_ob(
    kind: str = BEARISH,
    low: float = 2400.0,
    high: float = 2403.0,
    quality: int = 4,
    index: int = 100,
    status: str = "UNMITIGATED",
    timeframe: str = "M15",
) -> OrderBlock:
    return OrderBlock(
        kind=kind,
        high=high,
        low=low,
        body_high=high - 0.2,
        body_low=low + 0.2,
        index=index,
        timestamp=datetime(2026, 10, 7, 9, 0, tzinfo=timezone.utc),
        timeframe=timeframe,
        bos_index=index + 2,
        bos_timestamp=datetime(2026, 10, 7, 9, 30, tzinfo=timezone.utc),
        status=status,
        quality=quality,
    )


def make_candle_analysis(direction: str = BEARISH, strength: int = -3) -> CandleAnalysis:
    return CandleAnalysis(
        type="BEARISH_REJECTION" if direction == BEARISH else "BULLISH_REJECTION",
        signal_strength=strength,
        direction=direction,
        is_rejection=True,
        body_ratio=0.45,
        upper_wick_ratio=0.4,
        lower_wick_ratio=0.15,
    )


def make_signal(
    direction: str = BEARISH,
    entry: float = 2400.0,
    stop_loss: float = 2403.0,
    tp1: float | None = None,
    tp2: float | None = None,
    tp3: float | None = None,
    lots: float = 0.33,
    equity: float = 10_000.0,
    risk_percent: float = 1.0,
    confluence: float = 8.5,
    timestamp: datetime | None = None,
) -> Signal:
    """A consistent, valid signal (short by default)."""

    bullish = direction == BULLISH
    sl_distance = abs(entry - stop_loss)
    sl_pips = round(sl_distance / 0.10, 1)
    risk_amount = equity * risk_percent / 100.0
    if tp1 is None:
        tp1 = entry + (1.5 * sl_distance if bullish else -1.5 * sl_distance)
    if tp2 is None:
        tp2 = entry + (3.0 * sl_distance if bullish else -3.0 * sl_distance)
    if tp3 is None:
        tp3 = entry + (5.0 * sl_distance if bullish else -5.0 * sl_distance)
    moment = timestamp or datetime(2026, 10, 7, 13, 30, tzinfo=timezone.utc)

    def tp(price: float, fraction: str, basis: str) -> TakeProfit:
        distance = abs(price - entry) / 0.10
        rr = distance / sl_pips if sl_pips else 0.0
        return TakeProfit(price=price, distance_pips=round(distance, 1), basis=basis, size_to_close=fraction, rr=round(rr, 2))

    return Signal(
        signal_id="APEX-TEST-XAUUSD-SELL-001",
        timestamp=moment,
        instrument="XAU_USD",
        direction=direction,
        entry=EntryPlan(
            price=entry,
            zone_low=min(entry, stop_loss),
            zone_high=max(entry, stop_loss) - 0.5,
            zone="TEST ZONE",
            type="LIMIT",
            valid_until=moment + timedelta(minutes=120),
        ),
        stop_loss=StopLossPlan(
            price=stop_loss,
            distance_pips=sl_pips,
            basis="OB + buffer",
            structure_basis="Above OB",
        ),
        take_profits=TakeProfitPlan(
            tp1=tp(tp1, "50%", "1.5R"),
            tp2=tp(tp2, "25%", "3R"),
            tp3=tp(tp3, "runner", "5R"),
        ),
        risk_reward=RiskReward(
            rr_to_tp1=round(abs(tp1 - entry) / sl_distance, 2),
            rr_to_tp2=round(abs(tp2 - entry) / sl_distance, 2),
            rr_to_tp3=round(abs(tp3 - entry) / sl_distance, 2),
        ),
        position_sizing=PositionSizing(
            account_equity=equity,
            risk_percent=risk_percent,
            risk_amount_usd=round(risk_amount, 2),
            sl_pips=sl_pips,
            pip_value_per_lot=10.0,
            lot_size=lots,
            units=lots * 100.0,
        ),
        confluence_score=confluence,
        confidence_percent=85,
        signal_type="SNIPER_ENTRY",
        session={"name": "NY_KILL_ZONE", "time_utc": moment.strftime("%H:%M")},
        analysis={"four_h_bias": direction, "one_h_bias": direction, "summary": "test"},
        plan_a=PlanA(
            scenario="Continuation",
            required=["5M close beyond the zone"],
            next_milestone=entry,
            trail_sl_after_tp1=entry,
            management="Manage per plan",
            tp1=tp1,
            tp2=tp2,
        ),
        plan_b=PlanB(
            scenario="Invalidation",
            invalidation=stop_loss,
            what_happens_if_sl_hit="Re-assess",
            next_zone=None,
            next_zone_description="No zone",
            wait_period="4 hours",
            possible_flip="Flip if a sweep prints",
        ),
        do_not_enter_if=["Spread > 3.5 pips"],
        smart_money_simulation={"narrative": "test"},
        expires_at=moment + timedelta(minutes=120),
    )
