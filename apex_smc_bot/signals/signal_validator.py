"""Pre-execution validation of a trade plan.

This validator is intentionally independent of the confluence engine: it checks
that the *plan itself* is internally consistent, that risk limits are respected
and that the signal has not expired or been overtaken by price.  Account-level
and portfolio checks live in :mod:`apex_smc_bot.risk.risk_engine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from apex_smc_bot.config import Settings
from apex_smc_bot.data.candle_manager import to_utc, utcnow
from apex_smc_bot.execution.lot_calculator import sl_distance_pips
from apex_smc_bot.signals.models import Signal


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = "ERROR"  # ERROR | WARNING

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "severity": self.severity}


@dataclass
class ValidationResult:
    valid: bool
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "checks": self.checks,
        }

    def summary(self) -> str:
        if self.valid:
            return "valid" + (f" ({len(self.warnings)} warning(s))" if self.warnings else "")
        return "invalid: " + "; ".join(issue.code for issue in self.errors)


class SignalValidator:
    """Runs every structural check required before an order can be placed."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.risk = settings.risk
        self.strategy = settings.strategy

    def validate(
        self,
        signal: Signal,
        market_price: float | None = None,
        now: datetime | None = None,
        open_trade_count: int = 0,
        account_equity: float | None = None,
    ) -> ValidationResult:
        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []
        checks: dict[str, bool] = {}
        now = to_utc(now or utcnow())
        pip = self.risk.pip_size
        entry = signal.entry.price
        sl = signal.stop_loss.price
        bullish = signal.is_long

        def fail(code: str, message: str) -> None:
            errors.append(ValidationIssue(code, message))

        def warn(code: str, message: str) -> None:
            warnings.append(ValidationIssue(code, message, "WARNING"))

        # 1) Direction & stop placement ---------------------------------------
        checks["stop_on_correct_side"] = sl < entry if bullish else sl > entry
        if not checks["stop_on_correct_side"]:
            fail("STOP_WRONG_SIDE", f"Stop {sl} is on the wrong side of entry {entry}")

        sl_pips = sl_distance_pips(entry, sl, pip)
        checks["sl_within_bounds"] = self.risk.min_sl_pips <= sl_pips <= self.risk.max_sl_pips
        if not checks["sl_within_bounds"]:
            fail(
                "SL_DISTANCE_INVALID",
                f"Stop distance {sl_pips:.1f} pips outside [{self.risk.min_sl_pips}, {self.risk.max_sl_pips}]",
            )

        # 2) Targets -----------------------------------------------------------
        levels = signal.take_profits.all_levels()
        ordered = all(
            levels[i + 1] > levels[i] + pip for i in range(len(levels) - 1)
        ) if bullish else all(levels[i + 1] < levels[i] - pip for i in range(len(levels) - 1))
        checks["targets_ordered"] = ordered
        if not ordered:
            fail("TPS_OUT_OF_ORDER", f"Take profits are not in trade direction: {levels}")

        # Ordering alone is not enough: a short with TP1 above entry can still be
        # "ordered" (2404 > 2391 > 2385).  Every target must be on the winning
        # side of the entry, and the quoted R:R must match the actual levels.
        tp1 = signal.take_profits.tp1.price
        checks["tp1_beyond_entry"] = tp1 > entry + pip if bullish else tp1 < entry - pip
        if not checks["tp1_beyond_entry"]:
            fail("TP1_WRONG_SIDE", f"TP1 {tp1:.2f} is not beyond entry {entry:.2f}")

        risk_distance = abs(entry - sl)
        computed_rr = abs(tp1 - entry) / risk_distance if risk_distance > 0 else 0.0
        quoted_rr = signal.risk_reward.rr_to_tp1
        checks["rr_matches_prices"] = abs(computed_rr - quoted_rr) <= max(0.15, quoted_rr * 0.1)
        if not checks["rr_matches_prices"]:
            fail(
                "RR_MISMATCH",
                f"Quoted TP1 R:R {quoted_rr:.2f} does not match the levels ({computed_rr:.2f})",
            )

        checks["tp1_min_rr"] = signal.risk_reward.rr_to_tp1 >= min(1.0, self.strategy.min_rr_tp1)
        if not checks["tp1_min_rr"]:
            fail("TP1_RR_TOO_LOW", f"TP1 R:R {signal.risk_reward.rr_to_tp1:.2f} < 1.0")
        elif signal.risk_reward.rr_to_tp1 < self.strategy.min_rr_tp1:
            warn(
                "TP1_RR_BELOW_TARGET",
                f"TP1 R:R {signal.risk_reward.rr_to_tp1:.2f} below configured minimum "
                f"{self.strategy.min_rr_tp1}",
            )
        checks["tp2_above_tp1_rr"] = signal.risk_reward.rr_to_tp2 > signal.risk_reward.rr_to_tp1
        if not checks["tp2_above_tp1_rr"]:
            fail("TP2_RR_NOT_ABOVE_TP1", "TP2 risk/reward must exceed TP1")

        # 3) Sizing ------------------------------------------------------------
        sizing = signal.position_sizing
        checks["lot_within_bounds"] = self.risk.min_lot <= sizing.lot_size <= self.risk.max_lot
        if not checks["lot_within_bounds"]:
            fail("LOT_OUT_OF_BOUNDS", f"Lot {sizing.lot_size} outside broker limits")
        expected_risk = sizing.lot_size * sl_pips * self.risk.pip_value_per_lot
        checks["risk_matches_sizing"] = abs(expected_risk - sizing.risk_amount_usd) <= max(
            2.0, sizing.risk_amount_usd * 0.05
        )
        if not checks["risk_matches_sizing"]:
            fail(
                "RISK_MISMATCH",
                f"Sized risk ${expected_risk:.2f} does not match stated ${sizing.risk_amount_usd:.2f}",
            )
        checks["risk_within_limit"] = sizing.risk_percent <= self.risk.max_risk_per_trade_percent + 1e-9
        if not checks["risk_within_limit"]:
            fail(
                "RISK_PERCENT_EXCEEDED",
                f"Risk {sizing.risk_percent}% exceeds max {self.risk.max_risk_per_trade_percent}%",
            )
        if account_equity is not None and account_equity > 0:
            checks["risk_within_equity"] = (
                sizing.risk_amount_usd <= account_equity * self.risk.max_risk_per_trade_percent / 100
            )
            if not checks["risk_within_equity"]:
                fail("RISK_EXCEEDS_EQUITY_LIMIT", "Risk amount exceeds configured % of live equity")

        # 4) Confluence & confidence ------------------------------------------
        checks["confluence_threshold"] = signal.confluence_score >= self.strategy.min_confluence_score
        if not checks["confluence_threshold"]:
            fail(
                "CONFLUENCE_TOO_LOW",
                f"Score {signal.confluence_score} < {self.strategy.min_confluence_score}",
            )
        checks["confidence_sane"] = 0 <= signal.confidence_percent <= 100
        if not checks["confidence_sane"]:
            fail("CONFIDENCE_INVALID", "Confidence must be 0–100")

        # 5) Timing / expiry ---------------------------------------------------
        expires = to_utc(signal.expires_at) if signal.expires_at else None
        checks["not_expired"] = expires is None or expires >= now
        if not checks["not_expired"]:
            fail("SIGNAL_EXPIRED", f"Signal expired at {expires.isoformat() if expires else 'n/a'}")
        checks["timestamp_not_future"] = to_utc(signal.timestamp) <= now
        if not checks["timestamp_not_future"]:
            fail("SIGNAL_FROM_FUTURE", "Signal timestamp is in the future")

        # 6) Market sanity -----------------------------------------------------
        if market_price is not None and market_price > 0:
            if (bullish and market_price <= sl) or (not bullish and market_price >= sl):
                fail("PRICE_BEYOND_STOP", f"Market {market_price:.2f} already beyond the stop {sl:.2f}")
            distance_pips = abs(market_price - entry) / pip
            if distance_pips > self.strategy.max_entry_distance_pips:
                warn(
                    "ENTRY_FAR_FROM_MARKET",
                    f"Entry is {distance_pips:.0f} pips from market — verify the limit is still wanted",
                )
            moved_beyond_entry = (
                (bullish and market_price < entry - pip) or (not bullish and market_price > entry + pip)
            )
            if moved_beyond_entry and distance_pips > 15:
                warn(
                    "ENTRY_OVERRUN",
                    "Price has already moved past the entry zone — do not chase; wait for a retrace",
                )

        # 7) Completeness ------------------------------------------------------
        checks["has_plan_a"] = bool(signal.plan_a and signal.plan_a.required)
        checks["has_plan_b"] = bool(signal.plan_b and signal.plan_b.invalidation > 0)
        checks["has_invalidation_list"] = bool(signal.do_not_enter_if)
        if not checks["has_plan_a"]:
            fail("PLAN_A_MISSING", "Plan A is required")
        if not checks["has_plan_b"]:
            fail("PLAN_B_MISSING", "Plan B is required")
        if not checks["has_invalidation_list"]:
            fail("INVALIDATION_LIST_MISSING", "do_not_enter_if must not be empty")

        # 8) Portfolio capacity -----------------------------------------------
        checks["open_trade_capacity"] = open_trade_count < self.risk.max_open_trades
        if not checks["open_trade_capacity"]:
            fail(
                "MAX_OPEN_TRADES",
                f"{open_trade_count} open trades — limit is {self.risk.max_open_trades}",
            )

        valid = not errors
        return ValidationResult(valid=valid, errors=errors, warnings=warnings, checks=checks)
