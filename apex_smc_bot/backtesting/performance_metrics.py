"""Performance analytics for backtests and live results.

All functions are pure: pass a list of closed trades (dicts or dataclasses with
``pnl``/``realized_pnl``) and/or an equity curve.  Nothing here touches the DB or
the network, so the metrics are trivially testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence

TRADING_DAYS_PER_YEAR = 252


def _pnl(trade: Any) -> float:
    if isinstance(trade, dict):
        for key in ("pnl", "realized_pnl", "profit", "net_pnl"):
            if key in trade and trade[key] is not None:
                return float(trade[key])
        return 0.0
    for attribute in ("realized_pnl", "pnl", "profit"):
        if hasattr(trade, attribute):
            return float(getattr(trade, attribute) or 0.0)
    return 0.0


def _closed_at(trade: Any) -> datetime | None:
    if isinstance(trade, dict):
        value = trade.get("closed_at") or trade.get("timestamp")
    else:
        value = getattr(trade, "closed_at", None) or getattr(trade, "opened_at", None)
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


# --------------------------------------------------------------------------- core
def win_rate(pnls: Sequence[float]) -> float:
    """Percentage of trades that closed positive."""

    if not pnls:
        return 0.0
    return round(sum(1 for pnl in pnls if pnl > 0) / len(pnls) * 100.0, 2)


def profit_factor(pnls: Sequence[float]) -> float:
    """Gross profit / gross loss (``inf`` when there are no losses)."""

    gross_profit = sum(pnl for pnl in pnls if pnl > 0)
    gross_loss = -sum(pnl for pnl in pnls if pnl < 0)
    if gross_loss <= 1e-9:
        return float("inf") if gross_profit > 0 else 0.0
    return round(gross_profit / gross_loss, 3)


def expectancy(pnls: Sequence[float]) -> float:
    """Average P/L per trade."""

    return round(sum(pnls) / len(pnls), 2) if pnls else 0.0


def average_win_loss(pnls: Sequence[float]) -> tuple[float, float]:
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    return (
        round(sum(wins) / len(wins), 2) if wins else 0.0,
        round(sum(losses) / len(losses), 2) if losses else 0.0,
    )


def max_drawdown(equity_curve: Sequence[float]) -> tuple[float, float]:
    """Return ``(max_drawdown_abs, max_drawdown_percent)``."""

    if not equity_curve:
        return 0.0, 0.0
    peak = equity_curve[0]
    worst_abs = 0.0
    worst_pct = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        drop = peak - value
        if drop > worst_abs:
            worst_abs = drop
            worst_pct = (drop / peak * 100.0) if peak > 0 else 0.0
    return round(worst_abs, 2), round(worst_pct, 2)


def sharpe_ratio(
    returns: Sequence[float], risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Annualised Sharpe ratio of periodic returns (0 when undefined)."""

    if len(returns) < 2:
        return 0.0
    excess = [value - risk_free_rate / periods_per_year for value in returns]
    mean = sum(excess) / len(excess)
    variance = sum((value - mean) ** 2 for value in excess) / (len(excess) - 1)
    if variance <= 0:
        return 0.0
    return round(mean / math.sqrt(variance) * math.sqrt(periods_per_year), 3)


def sortino_ratio(returns: Sequence[float], periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sortino ratio (downside deviation only)."""

    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    downside = [value for value in returns if value < 0]
    if not downside:
        return 0.0
    downside_dev = math.sqrt(sum(value**2 for value in downside) / len(returns))
    if downside_dev <= 0:
        return 0.0
    return round(mean / downside_dev * math.sqrt(periods_per_year), 3)


def streak_stats(pnls: Sequence[float]) -> dict[str, int]:
    """Longest win/loss streaks."""

    best_win = best_loss = current_win = current_loss = 0
    for pnl in pnls:
        if pnl > 0:
            current_win += 1
            current_loss = 0
        elif pnl < 0:
            current_loss += 1
            current_win = 0
        best_win = max(best_win, current_win)
        best_loss = max(best_loss, current_loss)
    return {"max_win_streak": best_win, "max_loss_streak": best_loss}


def equity_to_returns(curve: Sequence[float]) -> list[float]:
    """Period-over-period simple returns."""

    returns: list[float] = []
    for previous, current in zip(curve, curve[1:]):
        if previous > 0:
            returns.append((current - previous) / previous)
    return returns


def daily_returns(trades: Iterable[Any], starting_equity: float) -> list[float]:
    """Aggregate trade P/L by calendar day and return daily returns."""

    buckets: dict[str, float] = {}
    for trade in trades:
        moment = _closed_at(trade)
        key = moment.date().isoformat() if moment else "unknown"
        buckets[key] = buckets.get(key, 0.0) + _pnl(trade)
    if not buckets:
        return []
    equity = starting_equity
    returns = []
    for key in sorted(buckets):
        pnl = buckets[key]
        new_equity = equity + pnl
        returns.append(pnl / equity if equity > 0 else 0.0)
        equity = new_equity
    return returns


# --------------------------------------------------------------------------- report
@dataclass
class PerformanceReport:
    """Full metrics bundle for a run."""

    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    return_percent: float = 0.0
    final_equity: float = 0.0
    initial_equity: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    best_day: float = 0.0
    worst_day: float = 0.0
    days_traded: int = 0
    equity_curve: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = {
            key: value
            for key, value in self.__dict__.items()
            if key not in {"equity_curve"}
        }
        data["profit_factor"] = (
            "inf" if self.profit_factor == float("inf") else self.profit_factor
        )
        return data

    def summary(self) -> str:
        pf = "inf" if self.profit_factor == float("inf") else f"{self.profit_factor:.2f}"
        lines = [
            "=" * 62,
            "APEX-SMC BACKTEST PERFORMANCE",
            "=" * 62,
            f"Trades            : {self.trades} ({self.wins}W / {self.losses}L)",
            f"Win rate          : {self.win_rate:.1f}%",
            f"Profit factor     : {pf}",
            f"Net P/L           : ${self.net_profit:+,.2f} ({self.return_percent:+.2f}%)",
            f"Equity            : ${self.initial_equity:,.2f} → ${self.final_equity:,.2f}",
            f"Expectancy        : ${self.expectancy:+,.2f}/trade",
            f"Average win/loss  : ${self.average_win:+,.2f} / ${self.average_loss:+,.2f}",
            f"Largest win/loss  : ${self.largest_win:+,.2f} / ${self.largest_loss:+,.2f}",
            f"Max drawdown      : ${self.max_drawdown:,.2f} ({self.max_drawdown_percent:.2f}%)",
            f"Sharpe / Sortino  : {self.sharpe:.2f} / {self.sortino:.2f}",
            f"Streaks (W/L)     : {self.max_win_streak} / {self.max_loss_streak}",
            f"Best / worst day  : ${self.best_day:+,.2f} / ${self.worst_day:+,.2f}",
            "=" * 62,
        ]
        lines.extend(self.notes)
        return "\n".join(lines)


def build_report(
    trades: Sequence[Any],
    equity_curve: Sequence[float] | None = None,
    initial_equity: float = 10_000.0,
    min_trades_for_report: int = 20,
) -> PerformanceReport:
    """Compute the complete metric set from closed trades."""

    pnls = [_pnl(trade) for trade in trades]
    curve = list(equity_curve) if equity_curve else [initial_equity]
    if not equity_curve and pnls:
        equity = initial_equity
        curve = [initial_equity]
        for pnl in pnls:
            equity += pnl
            curve.append(round(equity, 2))
    dd_abs, dd_pct = max_drawdown(curve)
    daily = daily_returns(trades, initial_equity)
    streaks = streak_stats(pnls)
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    avg_win, avg_loss = average_win_loss(pnls)
    final_equity = round(curve[-1], 2) if curve else initial_equity
    notes: list[str] = []
    if len(pnls) < min_trades_for_report:
        notes.append(
            f"NOTE: only {len(pnls)} trades — below the {min_trades_for_report}-trade "
            "statistical minimum. Treat these figures as indicative only."
        )
    if not pnls:
        notes.append("NOTE: no completed trades in this window.")

    return PerformanceReport(
        trades=len(pnls),
        wins=len(wins),
        losses=len(losses),
        win_rate=win_rate(pnls),
        gross_profit=round(sum(wins), 2),
        gross_loss=round(sum(losses), 2),
        net_profit=round(sum(pnls), 2),
        profit_factor=profit_factor(pnls),
        expectancy=expectancy(pnls),
        average_win=avg_win,
        average_loss=avg_loss,
        largest_win=round(max(wins), 2) if wins else 0.0,
        largest_loss=round(min(losses), 2) if losses else 0.0,
        max_drawdown=dd_abs,
        max_drawdown_percent=dd_pct,
        sharpe=sharpe_ratio(daily),
        sortino=sortino_ratio(daily),
        return_percent=round((final_equity - initial_equity) / initial_equity * 100.0, 2)
        if initial_equity
        else 0.0,
        final_equity=final_equity,
        initial_equity=round(initial_equity, 2),
        max_win_streak=streaks["max_win_streak"],
        max_loss_streak=streaks["max_loss_streak"],
        best_day=round(max([0.0] + [value for value in _daily_pnl(trades)]), 2),
        worst_day=round(min([0.0] + [value for value in _daily_pnl(trades)]), 2),
        days_traded=len(_daily_pnl(trades)),
        equity_curve=[round(value, 2) for value in curve],
        notes=notes,
    )


def _daily_pnl(trades: Sequence[Any]) -> list[float]:
    buckets: dict[str, float] = {}
    for trade in trades:
        moment = _closed_at(trade)
        key = moment.date().isoformat() if moment else "unknown"
        buckets[key] = buckets.get(key, 0.0) + _pnl(trade)
    return list(buckets.values())
