"""Dynamic position sizing for XAU/USD.

    Lots = Risk_amount / (SL_pips × pip_value_per_lot)

Gold convention used everywhere in this repo:
    pip size          = 0.10 USD
    1.00 lot          = 100 oz
    pip value / lot   = 0.10 × 100 = 10 USD

Sizing always **floors** to the broker lot step so the realised risk cannot
exceed the configured percentage.  If even the minimum lot would breach the risk
budget the trade is rejected (lots = 0). Costs may be reserved in the stop distance.
"""

from __future__ import annotations

import math
from typing import Any

from apex_smc_bot.analysis.structure import BEARISH, BULLISH
from apex_smc_bot.config import RiskConfig
from apex_smc_bot.signals.models import PositionSizing


class LotSizeError(ValueError):
    """Raised when position sizing cannot satisfy the risk constraints."""


def sl_distance_pips(entry: float, stop_loss: float, pip_size: float = 0.10) -> float:
    """Absolute stop distance expressed in pips."""

    if pip_size <= 0:
        raise LotSizeError("pip_size must be positive")
    distance = abs(float(entry) - float(stop_loss)) / pip_size
    if not math.isfinite(distance):
        raise LotSizeError("Non-finite stop distance")
    return round(distance, 2)


def calculate_position_size(
    account_equity: float,
    risk_percent: float,
    sl_pips: float,
    pip_value_per_lot: float = 10.0,
    min_lot: float = 0.01,
    max_lot: float = 10.0,
    lot_step: float = 0.01,
    risk_tolerance: float = 1.0,
) -> float:
    """Return lot size (floor-rounded) or ``0.0`` when the trade is unsizeable."""

    values = (account_equity, risk_percent, sl_pips, pip_value_per_lot, min_lot, max_lot, lot_step, risk_tolerance)
    if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
        raise LotSizeError("Sizing inputs must be finite numbers")
    if not 0 < risk_tolerance <= 1:
        raise LotSizeError("risk_tolerance cannot permit exceeding the risk budget")
    if account_equity <= 0:
        raise LotSizeError("account_equity must be positive")
    if not 0 < risk_percent <= 100:
        raise LotSizeError("risk_percent must be within (0, 100]")
    if sl_pips <= 0:
        raise LotSizeError("sl_pips must be positive")
    if pip_value_per_lot <= 0:
        raise LotSizeError("pip_value_per_lot must be positive")
    if min_lot <= 0 or max_lot < min_lot or lot_step <= 0:
        raise LotSizeError("invalid lot bounds")

    risk_amount = account_equity * risk_percent / 100.0
    raw_lots = risk_amount / (sl_pips * pip_value_per_lot)
    if raw_lots <= 0 or not math.isfinite(raw_lots):
        return 0.0

    steps = math.floor(min(raw_lots, max_lot) / lot_step + 1e-9)
    lot_size = round(steps * lot_step, 8)
    if lot_size < min_lot:
        # Never round UP into risk the account cannot afford.
        min_risk = min_lot * sl_pips * pip_value_per_lot
        if min_risk > risk_amount * risk_tolerance:
            return 0.0
        lot_size = min_lot
    return lot_size


def calculate_position(
    account_equity: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float,
    config: RiskConfig,
    direction: str = BULLISH,
    cost_reserve_pips: float = 0.0,
) -> PositionSizing:
    """Build a :class:`PositionSizing` with validation of stop direction."""

    if direction not in {BULLISH, BEARISH}:
        raise LotSizeError("Unknown trade direction")
    if not math.isfinite(cost_reserve_pips) or cost_reserve_pips < 0:
        raise LotSizeError("Cost reserve must be finite and non-negative")
    if direction == BULLISH and stop_loss >= entry_price:
        raise LotSizeError(f"Long stop {stop_loss} must be below entry {entry_price}")
    if direction == BEARISH and stop_loss <= entry_price:
        raise LotSizeError(f"Short stop {stop_loss} must be above entry {entry_price}")
    pips = sl_distance_pips(entry_price, stop_loss, config.pip_size)
    lot_size = calculate_position_size(
        account_equity,
        risk_percent,
        pips + cost_reserve_pips,
        config.pip_value_per_lot,
        config.min_lot,
        config.max_lot,
    )
    risk_amount = account_equity * risk_percent / 100.0
    if lot_size <= 0:
        raise LotSizeError(
            f"Cannot size {direction} trade: {pips} pip stop at {risk_percent}% risk "
            f"would breach the risk budget with the minimum lot"
        )
    return PositionSizing(
        account_equity=account_equity,
        risk_percent=risk_percent,
        risk_amount_usd=round(lot_size * (pips + cost_reserve_pips) * config.pip_value_per_lot, 4),
        sl_pips=pips,
        pip_value_per_lot=config.pip_value_per_lot,
        lot_size=lot_size,
        units=round(lot_size * config.contract_size, 2),
        contract_size=config.contract_size,
        cost_reserve_pips=cost_reserve_pips,
    )


def risk_of_lot(lot_size: float, sl_pips: float, pip_value_per_lot: float = 10.0) -> float:
    """USD risk for a given lot size and stop distance."""

    return round(lot_size * sl_pips * pip_value_per_lot, 2)


def pips_to_price(pips: float, pip_size: float = 0.10) -> float:
    return round(pips * pip_size, 5)


def price_to_pips(price_distance: float, pip_size: float = 0.10) -> float:
    return round(price_distance / pip_size, 5)


def trade_summary(sizing: PositionSizing) -> dict[str, Any]:
    return sizing.to_dict()
