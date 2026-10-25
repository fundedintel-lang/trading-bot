"""Top-down HTF bias engine (4H master, 1H confirmation).

The 4H defines the institutional direction; the 1H must agree before any 15M
setup may be armed.  Conflicting timeframes mean **no trade**, never a "feeling".
The engine also produces a ranked key-level map handed to the 15M/5M layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from apex_smc_bot.analysis.fvg import FVG, active_fvgs, detect_fvg
from apex_smc_bot.analysis.liquidity import LiquidityMap, build_liquidity_map
from apex_smc_bot.analysis.order_blocks import (
    OrderBlock,
    active_order_blocks,
    detect_order_blocks,
)
from apex_smc_bot.analysis.premium_discount import (
    DealingRange,
    dealing_range_from_structure,
    premium_discount_position,
)
from apex_smc_bot.analysis.structure import (
    BEARISH,
    BULLISH,
    RANGING,
    StructureEvent,
    StructureState,
    analyze_dual_structure,
)
from apex_smc_bot.config import Settings, StrategyConfig
from apex_smc_bot.data.candle_manager import Candle

CONFLUENT = "CONFLUENT"


@dataclass
class HTFBias:
    """Complete directional state for one higher timeframe."""

    timeframe: str
    direction: str = RANGING
    strength: str = "WEAK"
    structure: StructureState | None = None
    order_blocks: list[OrderBlock] = field(default_factory=list)
    fvgs: list[FVG] = field(default_factory=list)
    liquidity: LiquidityMap | None = None
    dealing_range: DealingRange | None = None
    premium_discount: str = "UNKNOWN"
    current_price: float = 0.0
    last_bos: StructureEvent | None = None
    last_choch: StructureEvent | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def aligned_direction(self) -> str | None:
        return self.direction if self.direction in {BULLISH, BEARISH} else None

    @property
    def key_obs(self) -> list[OrderBlock]:
        return active_order_blocks(self.order_blocks)[:5]

    @property
    def key_fvgs(self) -> list[FVG]:
        return active_fvgs(self.fvgs)[:10]

    def rationale(self) -> str:
        parts: list[str] = []
        if self.structure:
            parts.append(
                f"{self.timeframe} {self.direction} ({self.strength}) — "
                f"labels {', '.join(self.structure.labels()[-4:]) or 'n/a'}"
            )
        if self.last_bos:
            parts.append(f"last BOS {self.last_bos.direction} at {self.last_bos.level:.2f}")
        if self.last_choch:
            parts.append(f"CHoCH {self.last_choch.direction} at {self.last_choch.level:.2f}")
        if self.dealing_range:
            parts.append(f"{self.premium_discount} of range {self.dealing_range.low:.2f}-{self.dealing_range.high:.2f}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "direction": self.direction,
            "strength": self.strength,
            "last_bos": self.last_bos.to_dict() if self.last_bos else None,
            "last_choch": self.last_choch.to_dict() if self.last_choch else None,
            "premium_discount": self.premium_discount,
            "dealing_range": self.dealing_range.to_dict() if self.dealing_range else None,
            "key_obs": [ob.to_dict() for ob in self.key_obs],
            "key_fvgs": [fvg.to_dict() for fvg in self.key_fvgs],
            "liquidity": self.liquidity.to_dict() if self.liquidity else None,
            "rationale": self.rationale(),
            "notes": self.notes,
        }


@dataclass
class ConfluentZone:
    """Overlap between HTF zones (or HTF zone and near price)."""

    kind: str  # OB_OB | OB_FVG | FVG_FVG
    direction: str
    low: float
    high: float
    timeframes: tuple[str, str]
    score: float
    components: tuple[Any, Any]

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0

    def overlaps(self, low: float, high: float) -> bool:
        return not (high < self.low or low > self.high)

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "direction": self.direction,
            "low": round(self.low, 5),
            "high": round(self.high, 5),
            "timeframes": list(self.timeframes),
            "score": round(self.score, 2),
        }


@dataclass
class AlignmentResult:
    """Outcome of the 4H/1H agreement check."""

    aligned: bool
    direction: str
    strength: str
    action: str
    # FULL       — 4H and 1H point the same way
    # 4H_PULLBACK— 4H trend intact, 1H is pulling back against it (entry, not reversal)
    # RANGE      — 4H ranging, 1H direction traded from the correct range half
    mode: str = "FULL"
    monitor: str = ""
    confluent_zones: list[ConfluentZone] = field(default_factory=list)
    four_h: HTFBias | None = None
    one_h: HTFBias | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "aligned": self.aligned,
            "direction": self.direction,
            "mode": self.mode,
            "strength": self.strength,
            "action": self.action,
            "monitor": self.monitor,
            "confluent_zones": [zone.to_dict() for zone in self.confluent_zones],
        }

    def describe(self) -> str:
        if self.aligned:
            return f"HTF aligned {self.direction} ({self.strength}, {self.mode}) — {self.action}"
        return f"HTF conflict ({self.direction}) — {self.action}"


@dataclass
class KeyLevel:
    """A ranked level in the key-level map."""

    kind: str  # OB | FVG | EQH | EQL | OTE
    timeframe: str
    low: float
    high: float
    direction: str
    role: str  # SUPPORT | RESISTANCE | TARGET
    quality: float
    distance: float
    score: float
    reference: Any = None

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "timeframe": self.timeframe,
            "low": round(self.low, 5),
            "high": round(self.high, 5),
            "direction": self.direction,
            "role": self.role,
            "quality": round(self.quality, 2),
            "distance": round(self.distance, 2),
            "score": round(self.score, 2),
        }


class HTFBiasEngine:
    """Builds :class:`HTFBias` objects and the 4H/1H alignment verdict."""

    def __init__(self, config: Settings | StrategyConfig | None = None) -> None:
        if isinstance(config, Settings):
            self.strategy = config.strategy
        else:
            self.strategy = config or StrategyConfig()

    # ------------------------------------------------------------------ daily trend
    @staticmethod
    def daily_trend(candles_4h: Sequence[Candle], period: int = 20) -> str:
        """Coarse D1 trend from the 4H series (six 4H bars per trading day).

        Returns ``BULLISH``, ``BEARISH`` or ``RANGING``.  A trend only counts
        when price sits on one side of a rising/falling EMA *and* the EMA has
        actually moved over the last three days; otherwise the market is
        ranging and the filter stays out of the way.
        """

        if len(candles_4h) < period * 6 + 24:
            return RANGING
        daily = [c.close for c in candles_4h[5::6]]
        if len(daily) < period + 3:
            return RANGING
        k = 2.0 / (period + 1.0)
        ema = sum(daily[:period]) / period
        emas: list[float] = []
        for price in daily[period:]:
            ema = price * k + ema * (1.0 - k)
            emas.append(ema)
        if len(emas) < 4:
            return RANGING
        price = daily[-1]
        slope = emas[-1] - emas[-4]
        tolerance = max(abs(emas[-1]) * 0.0005, 0.05)
        if price > emas[-1] and slope > tolerance:
            return BULLISH
        if price < emas[-1] and slope < -tolerance:
            return BEARISH
        return RANGING

    # ------------------------------------------------------------------ analysis
    def analyze(
        self,
        candles: Sequence[Candle],
        timeframe: str,
        current_price: float | None = None,
        htf_levels: Sequence[float] | None = None,
    ) -> HTFBias:
        """Analyse a single HTF candle series."""

        if len(candles) < 30:
            return HTFBias(
                timeframe=timeframe,
                notes=[f"insufficient candles ({len(candles)}) for HTF analysis"],
                current_price=current_price or (candles[-1].close if candles else 0.0),
            )

        strategy = self.strategy
        dual = analyze_dual_structure(
            candles,
            timeframe,
            internal_lookback=strategy.swing_lookback,
            external_lookback=strategy.external_swing_lookback,
        )
        external, internal = dual["external"], dual["internal"]
        direction = external.direction
        strength = external.strength
        notes: list[str] = []
        if direction == RANGING:
            direction = internal.direction
            strength = "MODERATE" if direction != RANGING else "WEAK"
            notes.append("External structure ranging — using internal direction")

        price = current_price or candles[-1].close
        levels = list(htf_levels or [])
        if external.last_swing_high:
            levels.append(external.last_swing_high.price)
        if external.last_swing_low:
            levels.append(external.last_swing_low.price)

        fvgs = detect_fvg(candles, timeframe)
        obs = detect_order_blocks(
            candles,
            external.events,
            timeframe=timeframe,
            lookback=strategy.ob_lookback_candles,
            fvgs=fvgs,
            htf_bias=direction if direction in {BULLISH, BEARISH} else None,
            htf_levels=levels,
            pip_size=0.10,
        )
        liquidity = build_liquidity_map(
            candles,
            timeframe,
            tolerance_pips=strategy.equal_level_tolerance_pips,
            internal_lookback=strategy.swing_lookback,
            external_lookback=strategy.external_swing_lookback,
            bias=direction if direction in {BULLISH, BEARISH} else None,
            lookahead=strategy.sweep_lookahead_candles,
            single_swing_lookback=strategy.single_swing_lookback_bars,
        )
        dealing_range = dealing_range_from_structure(
            external, strategy.ote_low_fib, strategy.ote_high_fib
        )
        pd_state = "UNKNOWN"
        if dealing_range:
            pd_state = premium_discount_position(price, dealing_range)["zone"]

        return HTFBias(
            timeframe=timeframe,
            direction=direction,
            strength=strength,
            structure=external,
            order_blocks=obs,
            fvgs=fvgs,
            liquidity=liquidity,
            dealing_range=dealing_range,
            premium_discount=pd_state,
            current_price=price,
            last_bos=external.last_bos,
            last_choch=external.last_choch,
            notes=notes,
        )

    def analyze_4h(self, candles: Sequence[Candle], current_price: float | None = None) -> HTFBias:
        return self.analyze(candles, "H4", current_price)

    def analyze_1h(
        self,
        candles: Sequence[Candle],
        current_price: float | None = None,
        four_h: HTFBias | None = None,
    ) -> HTFBias:
        levels: list[float] = []
        if four_h:
            levels.extend(ob.midpoint for ob in four_h.key_obs)
            levels.extend(fvg.midpoint for fvg in four_h.key_fvgs)
        return self.analyze(candles, "H1", current_price, htf_levels=levels)

    # ------------------------------------------------------------------ alignment
    def check_alignment(self, four_h: HTFBias, one_h: HTFBias) -> AlignmentResult:
        """Hard gate: both HTFs must agree, otherwise no execution.

        One documented exception: when the 4H is **ranging** there is no HTF
        bias to violate, so the 1H direction may be traded — but only from the
        correct half of the 4H dealing range (discount for longs, premium for
        shorts).  That is range trading at a range edge, not counter-trend
        trading; a 4H that leans the opposite way always blocks.
        """

        zones = self.find_confluent_zones(four_h, one_h)
        if four_h.direction == one_h.direction and four_h.direction in {BULLISH, BEARISH}:
            strength = "STRONG" if four_h.strength == "STRONG" and one_h.strength != "WEAK" else "MODERATE"
            return AlignmentResult(
                aligned=True,
                direction=four_h.direction,
                mode="FULL",
                strength=strength,
                action=f"Trade only {four_h.direction.lower()} setups with 15M confirmation",
                confluent_zones=zones,
                four_h=four_h,
                one_h=one_h,
            )
        if four_h.direction == RANGING and one_h.direction in {BULLISH, BEARISH}:
            range_trade = self._range_alignment(four_h, one_h, zones)
            if range_trade is not None:
                return range_trade
        if (
            self.strategy.allow_1h_pullback_alignment
            and four_h.direction in {BULLISH, BEARISH}
            and one_h.direction in {BULLISH, BEARISH}
            and one_h.direction != four_h.direction
        ):
            # The 4H trend is the law; the 1H is pulling back against it.  That
            # pullback is where the discount/premium entry lives, so the trade
            # still goes *with* the 4H — never against it.
            return AlignmentResult(
                aligned=True,
                direction=four_h.direction,
                mode="4H_PULLBACK",
                strength="MODERATE",
                action=(
                    f"4H {four_h.direction} intact — 1H pullback is the entry; "
                    f"trade {four_h.direction.lower()} at the 4H discount/premium only"
                ),
                monitor="Abort if the 4H prints a CHoCH against the position",
                confluent_zones=zones,
                four_h=four_h,
                one_h=one_h,
            )
        if four_h.direction == RANGING or one_h.direction == RANGING:
            return AlignmentResult(
                aligned=False,
                direction=RANGING,
                strength="WEAK",
                action="NO_TRADE — HTF structure ranging, wait for a clean BOS",
                monitor="Monitor for 4H BOS + 1H CHoCH alignment",
                confluent_zones=zones,
                four_h=four_h,
                one_h=one_h,
            )
        monitor = "NO_TRADE — wait for 1H to align with 4H"
        if one_h.last_choch and one_h.last_choch.direction != four_h.direction:
            monitor = (
                f"1H CHoCH against 4H {four_h.direction} — potential HTF reversal forming; "
                "monitor 4H CHoCH for Plan B"
            )
        return AlignmentResult(
            aligned=False,
            direction="CONFLICTED",
            strength="WEAK",
            action=monitor,
            monitor=monitor,
            confluent_zones=zones,
            four_h=four_h,
            one_h=one_h,
        )

    def _range_alignment(
        self, four_h: HTFBias, one_h: HTFBias, zones: list[ConfluentZone]
    ) -> AlignmentResult | None:
        """Trade the 1H inside a ranging 4H, only from the correct range half.

        Returns ``None`` when the setup is not a valid range trade (price on the
        wrong side of equilibrium, no usable 4H dealing range, or range too
        small to offer an edge).
        """

        if not self.strategy.allow_range_alignment:
            return None
        dealing_range = four_h.dealing_range
        if dealing_range is None or dealing_range.size <= 0:
            return None
        price = float(four_h.current_price or one_h.current_price or 0.0)
        if price <= 0:
            return None
        equity = dealing_range.equilibrium
        direction = one_h.direction
        if direction == BULLISH and price >= equity:
            return None  # buying the premium of a range is chasing
        if direction == BEARISH and price <= equity:
            return None  # selling the discount of a range is chasing
        # A range narrower than one 15M ATR offers no room to pay the spread.
        if dealing_range.size < self.strategy.min_zone_width_pips * 4.0 * 0.10:
            return None
        return AlignmentResult(
            aligned=True,
            direction=direction,
            mode="RANGE",
            strength="MODERATE",
            action=(
                f"RANGE MODE — 4H ranging, trade {direction.lower()} from the "
                f"{'discount' if direction == BULLISH else 'premium'} half of the 4H range"
            ),
            monitor="Range edge trade: target the opposite side of the 4H range; abort on a 4H BOS",
            confluent_zones=zones,
            four_h=four_h,
            one_h=one_h,
        )

    # ------------------------------------------------------------------ maps
    def find_confluent_zones(self, four_h: HTFBias, one_h: HTFBias) -> list[ConfluentZone]:
        """Overlapping 4H/1H OBs, OB↔FVG and FVG↔FVG zones."""

        zones: list[ConfluentZone] = []
        obs_4h = active_order_blocks(four_h.order_blocks)
        obs_1h = active_order_blocks(one_h.order_blocks)
        fvg_4h = active_fvgs(four_h.fvgs)
        fvg_1h = active_fvgs(one_h.fvgs)

        for a in obs_4h:
            for b in obs_1h:
                if a.kind == b.kind and a.overlaps(b.low, b.high):
                    low, high = max(a.low, b.low), min(a.high, b.high)
                    zones.append(
                        ConfluentZone(
                            kind="OB_OB",
                            direction=a.kind,
                            low=low,
                            high=high,
                            timeframes=("H4", "H1"),
                            score=min(5.0, (a.quality + b.quality) / 2 + 1.0),
                            components=(a, b),
                        )
                    )
            for fvg in fvg_1h:
                if a.kind == fvg.kind and a.overlaps(fvg.bottom, fvg.top):
                    zones.append(
                        ConfluentZone(
                            kind="OB_FVG",
                            direction=a.kind,
                            low=max(a.low, fvg.bottom),
                            high=min(a.high, fvg.top),
                            timeframes=("H4", "H1"),
                            score=min(5.0, a.quality + 1.0),
                            components=(a, fvg),
                        )
                    )
        for a in fvg_4h:
            for b in fvg_1h:
                if a.kind == b.kind and a.overlaps(b.bottom, b.top):
                    zones.append(
                        ConfluentZone(
                            kind="FVG_FVG",
                            direction=a.kind,
                            low=max(a.bottom, b.bottom),
                            high=min(a.top, b.top),
                            timeframes=("H4", "H1"),
                            score=min(5.0, (a.quality + b.quality) / 2),
                            components=(a, b),
                        )
                    )
                for ob in obs_1h:
                    if ob.kind == a.kind and a.overlaps(ob.low, ob.high):
                        zones.append(
                            ConfluentZone(
                                kind="OB_FVG",
                                direction=a.kind,
                                low=max(a.bottom, ob.low),
                                high=min(a.top, ob.high),
                                timeframes=("H4", "H1"),
                                score=min(5.0, ob.quality + 1.0),
                                components=(ob, a),
                            )
                        )
        # Deduplicate near-identical overlap boxes.
        deduped: list[ConfluentZone] = []
        for zone in sorted(zones, key=lambda z: z.score, reverse=True):
            if any(
                abs(zone.low - other.low) < 0.25
                and abs(zone.high - other.high) < 0.25
                and zone.direction == other.direction
                for other in deduped
            ):
                continue
            deduped.append(zone)
        return deduped[:10]

    def build_key_level_map(
        self,
        alignment: AlignmentResult,
        price: float,
        pip_size: float = 0.10,
        max_levels: int = 12,
    ) -> list[KeyLevel]:
        """Rank zones by proximity, quality and timeframe weight."""

        if alignment.four_h is None or alignment.one_h is None:
            return []
        weights = {"H4": 1.5, "H1": 1.2}
        levels: list[KeyLevel] = []

        for bias in (alignment.four_h, alignment.one_h):
            for ob in active_order_blocks(bias.order_blocks):
                direction = ob.kind
                role = "SUPPORT" if direction == BULLISH else "RESISTANCE"
                if (direction == BULLISH and ob.high < price) or (direction == BEARISH and ob.low > price):
                    role = "TARGET" if direction != alignment.direction else role
                distance = ob.distance_to(price)
                proximity = 1.0 / (1.0 + distance / (10 * pip_size))
                score = weights.get(bias.timeframe, 1.0) * (ob.quality / 5.0) * 0.6 + proximity * 0.4
                levels.append(
                    KeyLevel(
                        kind="OB",
                        timeframe=bias.timeframe,
                        low=ob.low,
                        high=ob.high,
                        direction=direction,
                        role=role,
                        quality=ob.quality,
                        distance=distance,
                        score=score,
                        reference=ob,
                    )
                )
            for fvg in active_fvgs(bias.fvgs):
                direction = fvg.kind
                role = "SUPPORT" if direction == BULLISH else "RESISTANCE"
                distance = fvg.distance_to(price)
                proximity = 1.0 / (1.0 + distance / (10 * pip_size))
                score = weights.get(bias.timeframe, 1.0) * (fvg.quality / 5.0) * 0.55 + proximity * 0.45
                levels.append(
                    KeyLevel(
                        kind="FVG",
                        timeframe=bias.timeframe,
                        low=fvg.bottom,
                        high=fvg.top,
                        direction=direction,
                        role=role,
                        quality=fvg.quality,
                        distance=distance,
                        score=score,
                        reference=fvg,
                    )
                )
            if bias.liquidity:
                for level in bias.liquidity.equal_highs:
                    if level.price > price:
                        distance = level.price - price
                        levels.append(
                            KeyLevel(
                                kind="EQH",
                                timeframe=bias.timeframe,
                                low=level.price,
                                high=level.price,
                                direction=BEARISH,
                                role="TARGET",
                                quality=float(level.strength),
                                distance=distance,
                                score=weights.get(bias.timeframe, 1.0)
                                * (0.4 + 1.0 / (1.0 + distance / (10 * pip_size))),
                            )
                        )
                for level in bias.liquidity.equal_lows:
                    if level.price < price:
                        distance = price - level.price
                        levels.append(
                            KeyLevel(
                                kind="EQL",
                                timeframe=bias.timeframe,
                                low=level.price,
                                high=level.price,
                                direction=BULLISH,
                                role="TARGET",
                                quality=float(level.strength),
                                distance=distance,
                                score=weights.get(bias.timeframe, 1.0)
                                * (0.4 + 1.0 / (1.0 + distance / (10 * pip_size))),
                            )
                        )
        levels.sort(key=lambda level: level.score, reverse=True)
        return levels[:max_levels]

    def key_levels_for(
        self, bias: HTFBias, price: float, pip_size: float = 0.10
    ) -> list[dict[str, Any]]:
        """Lighter level map for a single timeframe (used in signal JSON)."""

        out = []
        for ob in bias.key_obs:
            out.append(
                {
                    "kind": f"OB_{ob.kind}",
                    "timeframe": bias.timeframe,
                    "zone": [round(ob.low, 2), round(ob.high, 2)],
                    "distance": round(ob.distance_to(price), 2),
                    "quality": ob.quality,
                }
            )
        for fvg in bias.key_fvgs:
            out.append(
                {
                    "kind": f"FVG_{fvg.kind}",
                    "timeframe": bias.timeframe,
                    "zone": [round(fvg.bottom, 2), round(fvg.top, 2)],
                    "distance": round(fvg.distance_to(price), 2),
                    "quality": fvg.quality,
                }
            )
        return out
