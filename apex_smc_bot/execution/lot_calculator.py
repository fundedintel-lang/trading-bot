"""Dynamic position sizing for XAU/USD.

    Lots = Risk_amount / (SL_pips × pip_value_per_lot)

Gold convention used everywhere in this repo:
    pip size          = 0.10 USD
    1.00 lot          = 100 oz
    pip value / lot   = 0.10 × 100 = 10 USD

Sizing always **floors** to the broker lot step so the realised risk cannot
exceed the configured percentage.  If even the minimum lot would breach the risk
budget by more than ``risk_tolerance`` the trade is rejected (lots = 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    risk_tolerance: float = 1.10,
) -> float:
    """Return lot size (floor-rounded) or ``0.0`` when the trade is unsizeable."""

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

    steps = math.floor(raw_lots / lot_step + 1e-9)
    lot_size = round(steps * lot_step, 2)
    lot_size = min(lot_size, max_lot)
    if lot_size < min_lot:
        # Minimum lot would exceed the budget by too much -> reject.
        min_risk = min_lot * sl_pips * pip_value_per_lot
        if min_risk > risk_amount * risk_tolerance:
            return 0.0
        lot_size = min_lot
    return round(lot_size, 2)


def calculate_position(
    account_equity: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float,
    config: RiskConfig,
    direction: str = BULLISH,
) -> PositionSizing:
    """Build a :class:`PositionSizing` with validation of stop direction."""

    if direction == BULLISH and stop_loss >= entry_price:
        raise LotSizeError(f"Long stop {stop_loss} must be below entry {entry_price}")
    if direction == BEARISH and stop_loss <= entry_price:
        raise LotSizeError(f"Short stop {stop_loss} must be above entry {entry_price}")
    pips = sl_distance_pips(entry_price, stop_loss, config.pip_size)
    lot_size = calculate_position_size(
        account_equity,
        risk_percent,
        pips,
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
        risk_amount_usd=risk_amount,
        sl_pips=pips,
        pip_value_per_lot=config.pip_value_per_lot,
        lot_size=lot_size,
        units=round(lot_size * config.contract_size, 2),
        contract_size=config.contract_size,
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
