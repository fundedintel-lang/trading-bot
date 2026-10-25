"""Signal generation: 15M setup arming → 7-check 5M trigger → full trade plan.

Pipeline (strict, top-down):

1. HTF gate — 4H and 1H must agree (``AlignmentResult.aligned``).
2. Session gate — only approved kill zones / sessions.
3. 15M structure + liquidity gate — a confirmed sweep against the pools and a
   BOS/CHoCH in the trade direction must precede the zone.
4. Zone selection — best unmitigated 15M OB (preferably stacked with an FVG and
   an HTF confluent zone) on the correct side of equilibrium.
5. 15M arming — entry / structural stop / liquidity targets become a
   :class:`PendingSetup` with a hard expiry.
6. 5M confirmation — all seven checks; at least six must pass.
7. Full signal — confluence scoring, dynamic sizing, Plan A/B, SM simulation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Sequence

from apex_smc_bot.analysis.candle_psych import (
    CandleAnalysis,
    analyze_candle,
    is_rejection_for,
    validate_wick_ratio,
)
from apex_smc_bot.analysis.confluence import ConfluenceEngine, ConfluenceResult, Setup
from apex_smc_bot.analysis.fvg import FVG, active_fvgs, overlapping_fvg
from apex_smc_bot.analysis.htf_bias import AlignmentResult, HTFBias
from apex_smc_bot.analysis.liquidity import LiquidityMap, SweepEvent, detect_inducement
from apex_smc_bot.analysis.order_blocks import (
    OrderBlock,
    active_order_blocks,
    nearest_opposing_ob,
)
from apex_smc_bot.analysis.premium_discount import (
    DealingRange,
    dealing_range_from_structure,
    premium_discount_position,
)
from apex_smc_bot.analysis.session import Session
from apex_smc_bot.analysis.structure import (
    BEARISH,
    BULLISH,
    RANGING,
    StructureState,
    detect_micro_bos,
    has_recent_event,
)
from apex_smc_bot.analysis.volume import (
    candle_delta,
    volume_confirmation,
)
from apex_smc_bot.config import Settings
from apex_smc_bot.data.candle_manager import Candle, atr, average_volume, to_utc, utcnow
from apex_smc_bot.execution.lot_calculator import calculate_position, sl_distance_pips
from apex_smc_bot.signals.alternate_paths import (
    default_do_not_enter_if,
    generate_alternate_paths,
)
from apex_smc_bot.signals.models import (
    EntryPlan,
    PendingSetup,
    RiskReward,
    Signal,
    StopLossPlan,
    TakeProfit,
    TakeProfitPlan,
    TriggerResult,
    side_for,
)
from apex_smc_bot.signals.smart_money import simulate_smart_money

logger = logging.getLogger("apex.signals")

IN_ZONE_TOLERANCE_PIPS = 3.0


class SignalGenerationError(RuntimeError):
    """Raised when a validated setup cannot be converted into a trade plan."""


def _recent_sweep(
    sweeps: LiquidityMap | Sequence[SweepEvent] | None,
    direction: str,
    total_candles: int,
    max_age: int,
    require_confirmed: bool = True,
) -> SweepEvent | None:
    if sweeps is None:
        return None
    if isinstance(sweeps, LiquidityMap):
        return sweeps.recent_sweep(
            direction=direction,
            max_age_candles=max_age,
            total_candles=total_candles,
            require_confirmed=require_confirmed,
        )
    for sweep in sorted(sweeps, key=lambda s: s.index, reverse=True):
        if sweep.direction != direction:
            continue
        if require_confirmed and not sweep.confirmed:
            continue
        if total_candles - 1 - sweep.index > max_age:
            continue
        return sweep
    return None


class SignalGenerator:
    """Stateless-ish orchestrator for the signal pipeline (sequence counter aside)."""

    def __init__(
        self,
        settings: Settings,
        confluence_engine: ConfluenceEngine | None = None,
        database: Any | None = None,
    ) -> None:
        self.settings = settings
        self.strategy = settings.strategy
        self.risk = settings.risk
        self.confluence = confluence_engine or ConfluenceEngine(settings)
        self.database = database
        self._sequence: dict[str, int] = {}
        self.last_rejection: str | None = None

    # ------------------------------------------------------------------ ids
    def next_signal_id(self, now: datetime, direction: str, instrument: str | None = None) -> str:
        date_key = to_utc(now).strftime("%Y%m%d")
        instrument = (instrument or self.settings.bot.instrument).replace("_", "")
        side = "LONG" if direction == BULLISH else "SHORT"
        key = f"{date_key}-{side}"
        if key not in self._sequence:
            existing = 0
            if self.database is not None:
                try:
                    existing = len(
                        [
                            s
                            for s in self.database.get_signals(limit=500)
                            if str(s.get("signal_id", "")).startswith(f"APEX-{date_key}-")
                            and f"-{side}-" in str(s.get("signal_id", ""))
                        ]
                    )
                except Exception:  # noqa: BLE001
                    existing = 0
            self._sequence[key] = existing
        self._sequence[key] += 1
        return f"APEX-{date_key}-{instrument}-{side}-{self._sequence[key]:03d}"

    # ------------------------------------------------------------------ 15M
    def check_15m_setup(
        self,
        structure: StructureState,
        obs: Sequence[OrderBlock],
        fvgs: Sequence[FVG],
        sweeps: LiquidityMap | Sequence[SweepEvent] | None,
        htf_bias: AlignmentResult,
        session: Session,
        candles: Sequence[Candle] | None = None,
        dealing_range: DealingRange | None = None,
        now: datetime | None = None,
        judas: Any | None = None,
        current_price: float | None = None,
    ) -> PendingSetup | None:
        """Arm a 15M setup if every gate passes; otherwise return ``None``."""

        self.last_rejection = None
        now = to_utc(now or utcnow())
        if candles is None or len(candles) < 30:
            self.last_rejection = "INSUFFICIENT_15M_DATA"
            return None
        if not htf_bias.aligned or htf_bias.direction not in {BULLISH, BEARISH}:
            self.last_rejection = "HTF_NOT_ALIGNED"
            return None
        if not session.tradeable:
            self.last_rejection = f"SESSION_NOT_TRADEABLE:{session.name}"
            return None
        direction = htf_bias.direction
        price = float(current_price if current_price is not None else candles[-1].close)
        pip = self.risk.pip_size

        # 15M structure must agree, be a tradeable range, or have produced a
        # fresh event in-direction.  A *ranging* 15M is the absence of structure
        # rather than a conflict: the sweep, the zone and the HTF bias are what
        # provide the edge, and most reversal entries print out of consolidation.
        structure_blocks = self.strategy.require_15m_structure and (
            structure.direction not in {direction, RANGING}
            or (structure.direction == RANGING and not self.strategy.allow_15m_ranging)
        )
        if structure_blocks and not has_recent_event(
            structure.events,
            direction,
            self.strategy.structure_event_window_bars,
            len(candles),
        ):
            self.last_rejection = "15M_STRUCTURE_CONFLICT"
            return None

        # Liquidity must already have been raided (LAW 2).
        sweep = _recent_sweep(
            sweeps,
            direction,
            total_candles=len(candles),
            max_age=max(36, self.strategy.inducement_lookback_candles + 24),
            require_confirmed=self.strategy.require_sweep_confirmation,
        )
        quality_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        min_rank = quality_rank.get(self.strategy.min_sweep_quality.upper(), 2)
        if sweep is None or quality_rank.get(sweep.quality, 0) < min_rank:
            self.last_rejection = (
                "NO_CONFIRMED_SWEEP" if sweep is None else f"SWEEP_QUALITY_{sweep.quality}"
            )
            return None

        # Best demand/supply zone on the correct side of price.
        selected = self._select_zone(obs, fvgs, direction, price, htf_bias.confluent_zones, pip)
        if selected is None:
            self.last_rejection = "NO_VALID_ZONE"
            return None
        zone_low, zone_high, basis, ob, fvg = selected
        entry = self._entry_price(ob, fvg, zone_low, zone_high, direction, price)
        if entry is None:
            self.last_rejection = "ZONE_INVALIDATED_BY_PRICE"
            return None

        arm_radius = (
            self.strategy.arm_entry_distance_pips
            if self.strategy.arm_entry_distance_pips > 0
            else self.strategy.max_entry_distance_pips
        )
        if abs(price - entry) > arm_radius * pip:
            self.last_rejection = "MISSED_ENTRY"
            return None

        if dealing_range is None:
            dealing_range = dealing_range_from_structure(
                structure, self.strategy.ote_low_fib, self.strategy.ote_high_fib
            )
        if dealing_range is not None and self.strategy.require_premium_discount:
            if direction == BULLISH and entry > dealing_range.equilibrium:
                self.last_rejection = "BUY_IN_PREMIUM"
                return None
            if direction == BEARISH and entry < dealing_range.equilibrium:
                self.last_rejection = "SELL_IN_DISCOUNT"
                return None

        atr_15m = atr(list(candles), 14) if len(candles) >= 15 else 0.0
        stop_loss = self._structural_stop(
            direction,
            entry,
            ob,
            fvg,
            sweep,
            pip,
            atr_value=atr_15m,
            zone_low=zone_low,
            zone_high=zone_high,
        )
        if stop_loss is None:
            self.last_rejection = "STOP_OUT_OF_BOUNDS"
            return None

        risk_distance = abs(entry - stop_loss)
        tp1, tp2, tp3, tp1_basis, tp2_basis, tp3_basis = self._project_targets(
            direction, entry, risk_distance, structure, htf_bias, sweeps, pip
        )

        inducement = detect_inducement(
            candles, structure, direction, self.strategy.inducement_lookback_candles
        )
        bos_event = next(
            (e for e in reversed(structure.events) if e.direction == direction and e.index >= len(candles) - 24),
            None,
        ) or next(
            (e for e in reversed(structure.events) if e.direction == direction),
            None,
        )
        notes = [
            f"Entry zone: {basis}",
            f"Confirmed {sweep.kind} at {sweep.level:.2f} "
            f"({sweep.raid_count} raid(s), {sweep.quality})",
            f"15M structure: {structure.direction} ({structure.strength})",
        ]
        if ob is not None:
            notes.append(f"OB quality {ob.quality}/5, status {ob.status}")
        if fvg is not None:
            notes.append(f"FVG {fvg.status} {fvg.bottom:.2f}-{fvg.top:.2f}")

        expires_at = now + timedelta(minutes=self.strategy.max_signal_age_minutes)
        return PendingSetup(
            direction=direction,
            entry_price=round(entry, 2),
            zone_low=round(zone_low, 2),
            zone_high=round(zone_high, 2),
            stop_loss=round(stop_loss, 2),
            tp1=round(tp1, 2),
            tp2=round(tp2, 2),
            tp3=round(tp3, 2),
            created_at=now,
            expires_at=expires_at,
            signal_bias=direction,
            session_name=session.name,
            zone_basis=basis,
            structure_basis=(
                f"Below {sweep.kind} extreme {sweep.extreme:.2f}"
                if direction == BULLISH
                else f"Above {sweep.kind} extreme {sweep.extreme:.2f}"
            ),
            sweep=sweep,
            ob=ob,
            fvg=fvg,
            structure=structure,
            bos_event=bos_event,
            dealing_range=dealing_range,
            alignment=htf_bias,
            judas=judas,
            inducement=inducement,
            confluent_zones=list(htf_bias.confluent_zones),
            notes=notes,
            tp1_basis=tp1_basis,
            tp2_basis=tp2_basis,
            tp3_basis=tp3_basis,
        )

    # ------------------------------------------------------------------ helpers
    def _select_zone(
        self,
        obs: Sequence[OrderBlock],
        fvgs: Sequence[FVG],
        direction: str,
        price: float,
        confluent_zones: Sequence[Any],
        pip: float,
    ) -> tuple[float, float, str, OrderBlock | None, FVG | None] | None:
        bullish = direction == BULLISH
        best: tuple[float, OrderBlock] | None = None
        min_width = self.strategy.min_zone_width_pips * pip
        max_width = self.strategy.max_zone_width_pips * pip
        for ob in active_order_blocks(obs, direction=direction):
            if ob.quality < self.strategy.min_zone_ob_quality:
                continue
            if not min_width <= ob.height <= max_width:
                # Micro blocks are noise; 50-pip "zones" are structure, not entries.
                continue
            if bullish and ob.low > price:
                continue  # demand zone above price is not a discount entry
            if not bullish and ob.high < price:
                continue
            confluent = any(
                getattr(zone, "direction", None) == direction
                and zone.overlaps(ob.low, ob.high)
                for zone in confluent_zones
            )
            fvg_match = overlapping_fvg(fvgs, ob.low, ob.high, direction)
            distance = ob.distance_to(price)
            # Proximity dominates: a zone price can actually reach in a few
            # hours beats a "perfect" zone 60 pips away that never fills.
            # Quality and HTF confluence break ties.
            distance_pips = distance / pip if pip > 0 else 0.0
            score = ob.quality * 3.0 + (3.0 if confluent else 0.0)
            score += 1.5 if fvg_match else 0.0
            score -= distance_pips * 0.25
            if distance_pips > self.strategy.max_entry_distance_pips * 0.5:
                # Beyond half the arming radius the fill probability collapses.
                score -= 3.0
            best = (score, ob) if best is None or score > best[0] else best
        if best is not None:
            ob = best[1]
            fvg = overlapping_fvg(fvgs, ob.low, ob.high, direction)
            label = f"15M {ob.kind} OB ({ob.low:.2f} - {ob.high:.2f})"
            if fvg:
                label += f" + FVG ({fvg.bottom:.2f} - {fvg.top:.2f})"
            return ob.low, ob.high, label, ob, fvg

        # FVG-only fallback (no valid OB remains).
        for fvg in sorted(active_fvgs(fvgs, direction), key=lambda f: f.index, reverse=True):
            if not min_width <= fvg.size <= max_width:
                continue
            if bullish and fvg.bottom > price:
                continue
            if not bullish and fvg.top < price:
                continue
            return (
                fvg.bottom,
                fvg.top,
                f"15M {fvg.kind} FVG ({fvg.bottom:.2f} - {fvg.top:.2f})",
                None,
                fvg,
            )
        return None

    def _entry_price(
        self,
        ob: OrderBlock | None,
        fvg: FVG | None,
        zone_low: float,
        zone_high: float,
        direction: str,
        price: float,
    ) -> float | None:
        if direction == BULLISH:
            if price < zone_low:
                return None
            proximal = zone_high
            if ob is not None:
                proximal = min(zone_high, max(ob.body_high, zone_low))
            elif fvg is not None:
                proximal = fvg.midpoint
            entry = min(proximal, price)
            return entry if zone_low <= entry <= zone_high else None
        if price > zone_high:
            return None
        proximal = zone_low
        if ob is not None:
            proximal = max(zone_low, min(ob.body_low, zone_high))
        elif fvg is not None:
            proximal = fvg.midpoint
        entry = max(proximal, price)
        return entry if zone_low <= entry <= zone_high else None

    def _structural_stop(
        self,
        direction: str,
        entry: float,
        ob: OrderBlock | None,
        fvg: FVG | None,
        sweep: SweepEvent | None,
        pip: float,
        atr_value: float = 0.0,
        zone_low: float | None = None,
        zone_high: float | None = None,
    ) -> float | None:
        """Stop outside the zone, but never inside one ATR of noise.

        LAW: the stop always sits beyond the far edge of the entry zone (never
        inside the OB/FVG) and outside the sweep extreme.
        """

        buffer = self.strategy.sl_buffer_pips * pip
        min_distance = max(
            self.risk.min_sl_pips * pip,
            self.strategy.min_sl_atr_fraction * max(0.0, atr_value),
        )
        max_distance = self.risk.max_sl_pips * pip
        candidates: list[float] = []
        if zone_low is not None and zone_high is not None:
            if direction == BULLISH:
                candidates.append(zone_low - buffer)
            else:
                candidates.append(zone_high + buffer)
        if direction == BULLISH:
            if ob is not None:
                candidates.append(ob.low - buffer)
            if fvg is not None:
                candidates.append(fvg.bottom - buffer)
            if sweep is not None:
                candidates.append(sweep.extreme - buffer)
            candidates = [c for c in candidates if c < entry]
            candidates.sort()
        else:
            if ob is not None:
                candidates.append(ob.high + buffer)
            if fvg is not None:
                candidates.append(fvg.top + buffer)
            if sweep is not None:
                candidates.append(sweep.extreme + buffer)
            candidates = [c for c in candidates if c > entry]
            candidates.sort(reverse=True)
        for candidate in candidates:
            distance = abs(entry - candidate)
            if min_distance <= distance <= max_distance:
                return round(candidate, 2)
        # No structural level far enough away: fall back to a volatility stop.
        if candidates and max_distance >= min_distance:
            sign = 1 if direction == BULLISH else -1
            fallback = entry - sign * min(min_distance, max_distance)
            if all(abs(entry - candidate) > max_distance for candidate in candidates):
                return None
            return round(fallback, 2)
        return None

    def _collect_targets(
        self,
        direction: str,
        entry: float,
        structure: StructureState | None,
        alignment: AlignmentResult,
        sweeps: LiquidityMap | Sequence[SweepEvent] | None,
    ) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
        internal: list[tuple[float, str]] = []
        external: list[tuple[float, str]] = []
        bullish = direction == BULLISH

        def acceptable(level: float) -> bool:
            return level > entry + 0.01 if bullish else level < entry - 0.01

        if structure is not None:
            for swing in structure.swings:
                if bullish and swing.is_high and acceptable(swing.price):
                    internal.append((swing.price, f"15M swing high ({swing.label or 'SH'})"))
                if not bullish and not swing.is_high and acceptable(swing.price):
                    internal.append((swing.price, f"15M swing low ({swing.label or 'SL'})"))
        if isinstance(sweeps, LiquidityMap):
            for level in sweeps.all_levels:
                if bullish and level.kind == "EQH" and acceptable(level.price):
                    internal.append((level.price, f"15M EQH pool ({level.count} touches)"))
                if not bullish and level.kind == "EQL" and acceptable(level.price):
                    internal.append((level.price, f"15M EQL pool ({level.count} touches)"))

        for bias in (alignment.one_h, getattr(alignment, "four_h", None)):
            if bias is None:
                continue
            label = bias.timeframe
            for ob in active_order_blocks(bias.order_blocks):
                target = ob.low if bullish else ob.high
                if acceptable(target):
                    external.append((target, f"{label} {ob.kind} OB ({ob.low:.2f}-{ob.high:.2f})"))
            for fvg in active_fvgs(bias.fvgs):
                target = fvg.bottom if bullish else fvg.top
                if acceptable(target):
                    external.append((target, f"{label} {fvg.kind} FVG ({fvg.bottom:.2f}-{fvg.top:.2f})"))
            if bias.liquidity:
                for level in bias.liquidity.equal_highs + bias.liquidity.equal_lows:
                    if level.kind == "EQH" and bullish and acceptable(level.price):
                        external.append((level.price, f"{label} EQH liquidity"))
                    if level.kind == "EQL" and not bullish and acceptable(level.price):
                        external.append((level.price, f"{label} EQL liquidity"))

        internal.sort(key=lambda item: item[0], reverse=not bullish)
        external.sort(key=lambda item: item[0], reverse=not bullish)
        return internal, external

    def _project_targets(
        self,
        direction: str,
        entry: float,
        risk_distance: float,
        structure: StructureState | None,
        alignment: AlignmentResult,
        sweeps: LiquidityMap | Sequence[SweepEvent] | None,
        pip: float,
    ) -> tuple[float, float, float, str, str, str]:
        bullish = direction == BULLISH
        sign = 1 if bullish else -1
        internal, external = self._collect_targets(direction, entry, structure, alignment, sweeps)
        min1 = entry + sign * risk_distance * self.strategy.min_rr_tp1
        min2 = entry + sign * risk_distance * self.strategy.min_rr_tp2
        min3 = entry + sign * risk_distance * self.strategy.target_rr_tp3

        def meets(level: float, threshold: float) -> bool:
            return level >= threshold if bullish else level <= threshold

        def far_enough(candidate: float, reference: float) -> bool:
            return candidate >= reference + pip * 2 if bullish else candidate <= reference - pip * 2

        tp1, basis1 = entry + sign * risk_distance * self.strategy.min_rr_tp1, (
            f"{self.strategy.min_rr_tp1:.1f}R projection"
        )
        for level, label in internal:
            if meets(level, min1):
                tp1, basis1 = level, label
                break

        tp2, basis2 = entry + sign * risk_distance * self.strategy.min_rr_tp2, (
            f"{self.strategy.min_rr_tp2:.1f}R projection"
        )
        for level, label in external + internal:
            if meets(level, min2) and far_enough(level, tp1):
                tp2, basis2 = level, label
                break
        if not far_enough(tp2, tp1):
            tp2 = tp1 + sign * max(risk_distance * 0.75, pip * 10)
            basis2 = "Extended projection (no clean HTF target)"

        tp3, basis3 = entry + sign * risk_distance * self.strategy.target_rr_tp3, (
            f"{self.strategy.target_rr_tp3:.1f}R projection"
        )
        for level, label in external:
            if meets(level, min3) and far_enough(level, tp2):
                tp3, basis3 = level, label
                break
        if not far_enough(tp3, tp2):
            tp3 = tp2 + sign * max(risk_distance * 0.75, pip * 10)
            basis3 = "Extended projection — trail only"

        return (
            round(float(tp1), 2),
            round(float(tp2), 2),
            round(float(tp3), 2),
            basis1,
            basis2,
            basis3,
        )

    # ------------------------------------------------------------------ 5M
    def five_minute_confirmation(
        self,
        pending: PendingSetup,
        candles_5m: Sequence[Candle],
        spread_price: float,
        order_blocks: Sequence[OrderBlock] | None = None,
        htf_order_blocks: Sequence[OrderBlock] | None = None,
        now: datetime | None = None,
    ) -> TriggerResult:
        """Run the seven 5M checks (ALL are evaluated; ≥6 required)."""

        now = to_utc(now or utcnow())
        pip = self.risk.pip_size
        spread_pips = max(0.0, float(spread_price)) / pip if spread_price else (
            candles_5m[-1].spread / pip if candles_5m and candles_5m[-1].spread else 0.0
        )
        checks: dict[str, bool] = {}
        if candles_5m is None or len(candles_5m) < 5:
            return TriggerResult(
                execute=False,
                checks_passed=0,
                confidence=0.0,
                entry_price=pending.entry_price,
                reason="INSUFFICIENT_5M_DATA",
                spread_pips=spread_pips,
            )
        candle = candles_5m[-1]
        previous = candles_5m[-2]
        direction = pending.direction
        bullish = direction == BULLISH
        tolerance = IN_ZONE_TOLERANCE_PIPS * pip

        if pending.is_expired(now):
            return TriggerResult(
                execute=False,
                checks_passed=0,
                confidence=0.0,
                entry_price=candle.close,
                reason="PENDING_EXPIRED",
                spread_pips=spread_pips,
            )

        # Hard invalidation before scoring.
        if (bullish and candle.close < pending.stop_loss) or (not bullish and candle.close > pending.stop_loss):
            return TriggerResult(
                execute=False,
                checks_passed=0,
                confidence=0.0,
                entry_price=candle.close,
                reason="SETUP_INVALIDATED",
                spread_pips=spread_pips,
            )

        # 1) Price has reached the entry zone.
        #
        # The plan is a LIMIT order resting at the zone, so the objective test is
        # "did price trade into the zone?" — a wick counts, because that is what
        # fills a limit order.  Requiring the confirmation bar's *close* to sit
        # inside the zone would reject the ideal fast-rejection entry (gold wicks
        # in and closes 10+ pips away inside a single 5M bar).
        #
        # Two guards keep that honest:
        #   * ``zone_touch_bars`` bounds how stale the tap may be (15 min by
        #     default), so a tap from two hours ago cannot confirm a setup now;
        #   * price may not currently sit more than ``max_entry_distance_pips``
        #     from the entry, otherwise the move has already left and the signal
        #     would be a chase of a zone that is no longer reachable.
        lookback = max(1, self.strategy.zone_touch_bars)
        recent = list(candles_5m[-lookback:])
        touched = any(
            (bar.low <= pending.zone_high + tolerance and bar.high >= pending.zone_low - tolerance)
            for bar in recent
        )
        distance_pips = abs(candle.close - pending.entry_price) / pip
        reachable = distance_pips <= self.strategy.max_entry_distance_pips
        in_zone = bool(touched and reachable)
        checks["1_price_in_zone"] = in_zone

        # 2) Rejection candle with body/delta confirmation.
        #
        # The rejection is the *reaction* to the zone, so it may print on any of
        # the bars in the touch window — not only on the confirmation candle.
        # The latest bar is still analysed for reporting and for check 5.
        avg_volume = average_volume(candles_5m, 20)
        analyses: list[CandleAnalysis] = []
        for offset in range(lookback):
            index = len(candles_5m) - 1 - offset
            if index < 1:
                break
            bar = candles_5m[index]
            prior = candles_5m[index - 4].close if index >= 4 else None
            analyses.append(
                analyze_candle(
                    bar,
                    previous=candles_5m[index - 1],
                    volume_delta=candle_delta(bar),
                    average_volume=avg_volume,
                    prior_move=bar.close - prior if prior is not None else None,
                )
            )
        analysis = analyses[0] if analyses else analyze_candle(candle, previous=previous)
        rejection_hits = [a for a in analyses if is_rejection_for(a, direction)]
        checks["2_rejection_candle"] = bool(rejection_hits)

        # 3) Micro BOS through the prior minor swing.
        micro = detect_micro_bos(
            list(candles_5m), direction, lookback=1, search=8, within_bars=lookback
        )
        checks["3_micro_bos"] = micro is not None

        # 4) Volume expansion / delta flip.
        volume_ok, volume_reason = volume_confirmation(list(candles_5m), direction)
        checks["4_volume_confirmation"] = volume_ok

        # 5) Wick ratio guard.
        checks["5_wick_ratio"] = validate_wick_ratio(candle, direction)

        # 6) No dangerous opposing OB in front of the trade.
        #
        # Only quality >= ``min_obstacle_quality`` blocks count: on gold every
        # 5M bar spawns marginal blocks, so an unfiltered test never passes.
        # Distance is measured from the *entry* rather than the live price so the
        # check is stable while the position is being armed.
        combined_obs = [
            block
            for block in list(order_blocks or []) + list(htf_order_blocks or [])
            if getattr(block, "is_active", False)
            and getattr(block, "quality", 0) >= self.strategy.min_obstacle_quality
        ]
        opposing = nearest_opposing_ob(combined_obs, pending.entry_price, direction)
        opposing_distance_pips = (
            opposing.distance_to(pending.entry_price) / pip
            if opposing is not None and pip > 0
            else float("inf")
        )
        fatal_opposing = (
            opposing is not None
            and opposing.timeframe in {"H4", "H1"}
            and opposing_distance_pips <= self.strategy.opposing_ob_abort_pips
        )
        checks["6_no_opposing_ob"] = not (
            opposing is not None
            and opposing_distance_pips <= self.strategy.opposing_ob_warning_pips
        )

        # 7) Spread guard.
        checks["7_spread_ok"] = 0 < spread_pips <= self.risk.max_spread_pips

        passed = sum(1 for value in checks.values() if value)
        confidence = passed / len(checks) * 100.0
        missing = [name for name, value in checks.items() if not value]
        # Price at the zone and a sane spread are non-negotiable: the ">= 6/7"
        # rule is a tolerance for the softer confirmations, never a licence to
        # enter a market with price 100 pips away from the plan.
        mandatory_ok = checks["1_price_in_zone"] and checks["7_spread_ok"]
        execute = passed >= self.strategy.min_fivem_checks and mandatory_ok and not fatal_opposing
        reason = ""
        if fatal_opposing:
            reason = (
                f"OPPOSING_HTF_OB — {opposing.timeframe} {opposing.kind} at "
                f"{opposing.low:.2f}-{opposing.high:.2f} ({opposing_distance_pips:.1f} pips away)"
            )
        elif not checks["1_price_in_zone"]:
            reason = "NO_ENTRY_PROXIMITY — price is not at the entry zone"
        elif not execute:
            reason = f"Only {passed}/7 checks passed — insufficient confirmation"
        elif passed < len(checks):
            reason = f"{passed}/7 checks passed — executing with {', '.join(missing)}"
        else:
            reason = "All 7 checks passed"

        return TriggerResult(
            execute=execute,
            checks_passed=passed,
            confidence=confidence,
            entry_price=candle.close,
            reason=reason,
            missing_checks=missing,
            checks=checks,
            candle_analysis=analysis,
            volume_confirmed=volume_ok,
            micro_bos=micro,
            opposing_ob=opposing,
            spread_pips=round(spread_pips, 2),
        )

    # ------------------------------------------------------------------ setup
    def build_setup(
        self,
        pending: PendingSetup,
        trigger: TriggerResult,
        alignment: AlignmentResult,
        spread_pips: float,
        normal_spread_pips: float = 0.0,
        news_within_30min: bool = False,
        news_events: Sequence[str] | None = None,
        chopbox: bool = False,
        session: Session | None = None,
        order_blocks: Sequence[OrderBlock] | None = None,
        htf_order_blocks: Sequence[OrderBlock] | None = None,
    ) -> Setup:
        """Assemble the confluence inputs from the armed setup and 5M trigger."""

        direction = pending.direction
        dealing_range = pending.dealing_range
        ote_aligned = bool(dealing_range and dealing_range.in_ote(pending.entry_price))
        confluent = any(
            getattr(zone, "direction", None) == direction
            and zone.low <= pending.entry_price <= zone.high
            for zone in pending.confluent_zones
        )
        combined_obs = list(order_blocks or []) + list(htf_order_blocks or [])
        opposing = nearest_opposing_ob(combined_obs, pending.entry_price, direction)
        opposing_pips = (
            opposing.distance_to(pending.entry_price) / self.risk.pip_size
            if opposing is not None
            else float("inf")
        )
        in_kill_zone = bool(
            session.is_kill_zone if session is not None else pending.session_name.endswith("KILL_ZONE")
        )
        return Setup(
            direction=direction,
            entry_price=pending.entry_price,
            four_h_bias=alignment.four_h,
            one_h_bias=alignment.one_h,
            structure_15m=pending.structure,
            bos_15m=pending.bos_event,
            ob=pending.ob,
            fvg=pending.fvg,
            sweep=pending.sweep,
            session=session,
            candle_analysis=trigger.candle_analysis,
            volume_confirmed=trigger.volume_confirmed,
            judas=pending.judas,
            inducement=pending.inducement,
            ote_aligned=ote_aligned,
            htf_level_confluence=confluent,
            double_inducement=bool(pending.sweep and pending.sweep.raid_count >= 2),
            opposing_ob=opposing,
            opposing_ob_distance_pips=opposing_pips,
            spread_pips=spread_pips,
            normal_spread_pips=normal_spread_pips,
            news_within_30min=news_within_30min,
            news_events=list(news_events or []),
            chopbox=chopbox,
            in_kill_zone=in_kill_zone,
            sl_distance_pips=sl_distance_pips(
                pending.entry_price, pending.stop_loss, self.risk.pip_size
            ),
            sl_structure_based=True,
            htf_aligned=alignment.aligned,
            htf_mode=getattr(alignment, "mode", "FULL"),
        )

    # ------------------------------------------------------------------ signal
    def generate_full_signal(
        self,
        pending: PendingSetup,
        trigger: TriggerResult,
        alignment: AlignmentResult,
        confluence: ConfluenceResult | None = None,
        setup: Setup | None = None,
        account_equity: float | None = None,
        risk_percent: float | None = None,
        order_blocks: Sequence[OrderBlock] | None = None,
        fvgs: Sequence[FVG] | None = None,
        spread_pips: float = 0.0,
        normal_spread_pips: float = 0.0,
        now: datetime | None = None,
        market_price: float | None = None,
    ) -> Signal | None:
        """Convert an approved trigger into a complete :class:`Signal`.

        Args:
            pending: The armed setup.
            trigger: The confirmed 5M trigger.
            alignment: Current HTF alignment.
            confluence: Pre-computed confluence result (optional).
            setup: Pre-computed confluence input (optional).
            account_equity: Live equity used for sizing.
            risk_percent: Base risk percent before the confluence size factor.
            order_blocks: 15M blocks for the alternate-path plans.
            fvgs: 15M FVGs for the alternate-path plans.
            spread_pips: Current spread.
            normal_spread_pips: Baseline spread.
            now: Clock.
            market_price: Live price.  Required for ``strategy.entry_mode=MARKET``
                so the fill price is the price the order is actually sent at.
        """

        self.last_rejection = None
        now = to_utc(now or utcnow())
        if pending.is_expired(now):
            self.last_rejection = "PENDING_EXPIRED"
            return None
        if not trigger.execute:
            self.last_rejection = trigger.reason or "TRIGGER_NOT_CONFIRMED"
            return None
        if setup is None:
            setup = self.build_setup(
                pending, trigger, alignment, spread_pips, normal_spread_pips
            )
        if confluence is None:
            confluence = self.confluence.score(setup)
        if not confluence.trade_allowed:
            self.last_rejection = confluence.abort_reason or f"CONFLUENCE_{confluence.score}"
            return None

        # Entry price: either the resting limit at the zone edge, or the live
        # price when the operator runs the MARKET profile (fills the trigger
        # immediately instead of waiting for a pullback that often never comes).
        entry_price = float(pending.entry_price)
        entry_type = "LIMIT"
        if str(self.strategy.entry_mode).upper() == "MARKET":
            if market_price is None or market_price <= 0:
                self.last_rejection = "MARKET_ENTRY_WITHOUT_PRICE"
                return None
            entry_price = round(float(market_price), 2)
            entry_type = "MARKET"
            if pending.direction == BULLISH and pending.stop_loss >= entry_price:
                self.last_rejection = "STOP_ON_WRONG_SIDE_OF_MARKET"
                return None
            if pending.direction == BEARISH and pending.stop_loss <= entry_price:
                self.last_rejection = "STOP_ON_WRONG_SIDE_OF_MARKET"
                return None

        equity = float(account_equity if account_equity is not None else self.risk.starting_equity)
        base_risk = float(
            risk_percent if risk_percent is not None else self.risk.max_risk_per_trade_percent
        )
        effective_risk = max(0.01, base_risk * confluence.position_fraction)
        try:
            sizing = calculate_position(
                equity, effective_risk, entry_price, pending.stop_loss, self.risk, pending.direction
            )
        except Exception as exc:  # noqa: BLE001 - sizing failures are business rejections
            self.last_rejection = f"SIZING_FAILED: {exc}"
            return None

        risk_distance = abs(entry_price - pending.stop_loss)
        tp1, tp2, tp3 = pending.tp1, pending.tp2, pending.tp3
        if entry_type == "MARKET" and risk_distance > 0:
            # A market fill can sit above (long) or below (short) the zone entry,
            # which shrinks the distance to the zone-based TP1.  Floor TP1 at the
            # configured R multiple so the trade never risks 1R to make 0.6R.
            signed = 1.0 if pending.direction == BULLISH else -1.0
            floor_tp1 = entry_price + signed * risk_distance * self.strategy.min_rr_tp1
            if (signed > 0 and tp1 < floor_tp1) or (signed < 0 and tp1 > floor_tp1):
                tp1 = round(floor_tp1, 2)
            # Keep the ladder ordered after the floor moved TP1.
            tp2 = tp1 + signed * max(abs(tp2 - tp1), 0.1)
            tp3 = tp2 + signed * max(abs(tp3 - tp2), 0.1)
        rr1 = abs(tp1 - entry_price) / risk_distance if risk_distance else 0.0
        rr2 = abs(tp2 - entry_price) / risk_distance if risk_distance else 0.0
        rr3 = abs(tp3 - entry_price) / risk_distance if risk_distance else 0.0
        if rr1 < 1.0:
            self.last_rejection = "TP1_BELOW_1R"
            return None

        direction = pending.direction
        bullish = direction == BULLISH
        pip = self.risk.pip_size
        risk_rr = RiskReward(rr_to_tp1=round(rr1, 2), rr_to_tp2=round(rr2, 2), rr_to_tp3=round(rr3, 2))
        tps = TakeProfitPlan(
            tp1=TakeProfit(
                price=tp1,
                distance_pips=round(abs(tp1 - entry_price) / pip, 1),
                basis=pending.tp1_basis or self._basis_for_target(pending, 1),
                size_to_close=f"{self.strategy.tp1_close_fraction:.0%}",
                rr=round(rr1, 2),
            ),
            tp2=TakeProfit(
                price=tp2,
                distance_pips=round(abs(tp2 - entry_price) / pip, 1),
                basis=pending.tp2_basis or self._basis_for_target(pending, 2),
                size_to_close=f"{self.strategy.tp2_close_fraction:.0%}",
                rr=round(rr2, 2),
            ),
            tp3=TakeProfit(
                price=tp3,
                distance_pips=round(abs(tp3 - entry_price) / pip, 1),
                basis=pending.tp3_basis or self._basis_for_target(pending, 3),
                size_to_close="remaining (trail after TP2)",
                rr=round(rr3, 2),
                optional=True,
                condition="Only active while HTF bias holds; otherwise close at TP2",
            ),
        )

        session_info = {
            "name": pending.session_name,
            "time_utc": now.strftime("%H:%M"),
            "judas_detected": bool(pending.judas is not None),
            "judas_resolved": bool(getattr(pending.judas, "resolved", False)),
            "judas_detail": getattr(pending.judas, "detail", ""),
        }

        smart_money = simulate_smart_money(
            pending,
            confluence.score,
            dealing_range=pending.dealing_range,
            hvn_levels=None,
            absorption=bool(
                trigger.candle_analysis
                and "ABSORPTION" in trigger.candle_analysis.type
            ),
            volume_confirmed=trigger.volume_confirmed,
        )

        candle = trigger.candle_analysis
        structure = pending.structure
        analysis = {
            "four_h_bias": alignment.four_h.rationale() if alignment.four_h else "n/a",
            "one_h_bias": alignment.one_h.rationale() if alignment.one_h else "n/a",
            "fifteen_m_setup": (
                f"15M {structure.direction if structure else 'n/a'} structure; "
                f"{pending.sweep.kind if pending.sweep else 'no'} sweep; "
                f"{pending.zone_basis}"
            ),
            "five_m_trigger": (
                f"{candle.type if candle else 'n/a'} at zone; "
                f"micro-BOS {trigger.micro_bos.get('level') if trigger.micro_bos else 'n/a'}; "
                f"{trigger.checks_passed}/7 checks; spread {trigger.spread_pips:.2f} pips"
            ),
            "sweep_detail": self._sweep_detail(pending.sweep),
            "ob_detail": self._ob_detail(pending.ob),
            "fvg_detail": self._fvg_detail(pending.fvg),
            "premium_discount": self._premium_discount_detail(pending),
            "ote": self._ote_detail(pending),
            "volume": f"delta {candle_delta(trigger.candle_analysis.candle):.1f}" if candle else "n/a",
            "session_detail": session_info,
            "confluence_breakdown": confluence.to_dict(),
            "news_filter": (
                "BLOCKED — high-impact event within 30 minutes"
                if setup.news_within_30min
                else "clear"
            ),
            "smart_money_assessment": smart_money.verdict,
            "entry_checks": trigger.checks,
            "missing_checks": trigger.missing_checks,
        }

        plan_a, plan_b = generate_alternate_paths(
            pending,
            entry=entry_price,
            stop_loss=pending.stop_loss,
            tp1=pending.tp1,
            tp2=pending.tp2,
            tp3=pending.tp3,
            order_blocks=list(order_blocks or []) + list(htf_order_blocks_from(alignment)),
            fvgs=list(fvgs or []),
        )

        confidence = int(
            round(
                min(
                    99.0,
                    confluence.confidence_percent * 0.6 + trigger.confidence * 0.4,
                )
            )
        )
        signal_type = "SNIPER_ENTRY" if confluence.score >= 9 else "REDUCED_SIZE_ENTRY"

        return Signal(
            signal_id=self.next_signal_id(now, direction),
            timestamp=now,
            instrument=self.settings.bot.instrument,
            direction=direction,
            entry=EntryPlan(
                price=entry_price,
                zone_low=pending.zone_low,
                zone_high=pending.zone_high,
                zone=pending.zone_basis,
                type=entry_type,
                valid_until=pending.expires_at,
            ),
            stop_loss=StopLossPlan(
                price=pending.stop_loss,
                distance_pips=sl_distance_pips(entry_price, pending.stop_loss, pip),
                basis=f"Outside zone + {self.strategy.sl_buffer_pips:.0f} pip buffer",
                structure_basis=pending.structure_basis,
            ),
            take_profits=tps,
            risk_reward=risk_rr,
            position_sizing=sizing,
            confluence_score=confluence.score,
            confidence_percent=confidence,
            signal_type=signal_type,
            session=session_info,
            analysis=analysis,
            plan_a=plan_a,
            plan_b=plan_b,
            do_not_enter_if=default_do_not_enter_if(
                pending, self.risk.max_spread_pips, self.risk.pip_size
            ),
            smart_money_simulation=smart_money.to_dict(),
            confluence=confluence.to_dict(),
            status="GENERATED",
            expires_at=pending.expires_at,
        )

    # ------------------------------------------------------------------ text details
    @staticmethod
    def _basis_for_target(pending: PendingSetup, index: int) -> str:
        labels = {
            1: "Internal 15M liquidity — nearest opposing swing/pool",
            2: "External 1H/4H liquidity — major pool / HTF zone",
            3: "HTF extension target — trail-only runner",
        }
        return labels.get(index, "liquidity target")

    @staticmethod
    def _sweep_detail(sweep: SweepEvent | None) -> str:
        if sweep is None:
            return "n/a"
        return (
            f"{sweep.kind} of {sweep.level:.2f} — wick to {sweep.extreme:.2f}, "
            f"closed at {sweep.close:.2f}; {sweep.raid_count} raid(s), quality {sweep.quality}"
        )

    @staticmethod
    def _ob_detail(ob: OrderBlock | None) -> str:
        if ob is None:
            return "No order block — FVG-only entry"
        return (
            f"{ob.timeframe} {ob.kind} OB: {ob.low:.2f}-{ob.high:.2f}, "
            f"body {ob.body_low:.2f}-{ob.body_high:.2f}, quality {ob.quality}/5, status {ob.status}"
        )

    @staticmethod
    def _fvg_detail(fvg: FVG | None) -> str:
        if fvg is None:
            return "No FVG stacked with the zone"
        return f"{fvg.timeframe} {fvg.kind} FVG: {fvg.bottom:.2f}-{fvg.top:.2f} ({fvg.status})"

    @staticmethod
    def _premium_discount_detail(pending: PendingSetup) -> str:
        if pending.dealing_range is None:
            return "Dealing range unavailable"
        position = premium_discount_position(pending.entry_price, pending.dealing_range)
        return (
            f"{position['zone']} — retracement {position['retracement_percent']}% of range "
            f"{pending.dealing_range.low:.2f}-{pending.dealing_range.high:.2f}"
        )

    @staticmethod
    def _ote_detail(pending: PendingSetup) -> str:
        if pending.dealing_range is None:
            return "n/a"
        return (
            f"OTE {pending.dealing_range.ote_low:.2f}-{pending.dealing_range.ote_high:.2f}; "
            f"entry quality {pending.dealing_range.ote_quality(pending.entry_price)}"
        )


def htf_order_blocks_from(alignment: AlignmentResult) -> list[OrderBlock]:
    """Flatten HTF order blocks for targeting / Plan B."""

    blocks: list[OrderBlock] = []
    for bias in (alignment.one_h, alignment.four_h):
        if isinstance(bias, HTFBias):
            blocks.extend(bias.order_blocks)
    return blocks
