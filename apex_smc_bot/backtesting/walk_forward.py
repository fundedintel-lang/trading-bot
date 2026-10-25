"""Walk-forward analysis.

A single in-sample backtest tells you almost nothing about robustness.  This
module splits history into consecutive (in-sample → out-of-sample) folds, runs
the *same* production engine on each window, and compares the results.  The
parameter set is fixed — walk-forward here measures stability, it does not
optimise (optimising SMC thresholds on synthetic data would be curve fitting).

Usage::

    python -m apex_smc_bot.backtesting.walk_forward --days 120 --folds 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from apex_smc_bot.backtesting.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    load_dataset,
)
from apex_smc_bot.backtesting.performance_metrics import PerformanceReport, build_report
from apex_smc_bot.config import Settings
from apex_smc_bot.data.candle_manager import to_utc, utcnow


@dataclass
class WalkForwardFold:
    index: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    is_report: PerformanceReport | None = None
    oos_report: PerformanceReport | None = None
    is_signals: int = 0
    oos_signals: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def degradation(self) -> float:
        """OOS vs IS net profit (negative = worse out-of-sample)."""

        if not self.is_report or not self.oos_report:
            return 0.0
        return round(self.oos_report.net_profit - self.is_report.net_profit, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "in_sample": {
                "start": self.is_start.isoformat(),
                "end": self.is_end.isoformat(),
                "signals": self.is_signals,
                "report": self.is_report.to_dict() if self.is_report else None,
            },
            "out_of_sample": {
                "start": self.oos_start.isoformat(),
                "end": self.oos_end.isoformat(),
                "signals": self.oos_signals,
                "report": self.oos_report.to_dict() if self.oos_report else None,
            },
            "degradation": self.degradation,
            "notes": self.notes,
        }


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold]
    combined_oos: PerformanceReport
    settings_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "folds": [fold.to_dict() for fold in self.folds],
            "combined_out_of_sample": self.combined_oos.to_dict(),
            "settings": self.settings_summary,
        }

    def summary(self) -> str:
        lines = ["=" * 74, "APEX-SMC WALK-FORWARD ANALYSIS", "=" * 74]
        header = f"{'Fold':<5}{'IS trades':>10}{'IS P/L':>12}{'OOS trades':>12}{'OOS P/L':>12}{'OOS PF':>9}"
        lines.append(header)
        lines.append("-" * 74)
        for fold in self.folds:
            is_trades = fold.is_report.trades if fold.is_report else 0
            is_pnl = fold.is_report.net_profit if fold.is_report else 0.0
            oos_trades = fold.oos_report.trades if fold.oos_report else 0
            oos_pnl = fold.oos_report.net_profit if fold.oos_report else 0.0
            pf = fold.oos_report.profit_factor if fold.oos_report else 0.0
            pf_text = "inf" if pf == float("inf") else f"{pf:.2f}"
            lines.append(
                f"{fold.index:<5}{is_trades:>10}{is_pnl:>+12.2f}{oos_trades:>12}{oos_pnl:>+12.2f}{pf_text:>9}"
            )
        lines.append("-" * 74)
        lines.append(self.combined_oos.summary())
        return "\n".join(lines)


class WalkForwardAnalyzer:
    """Runs the engine over rolling in-sample / out-of-sample windows."""

    def __init__(
        self,
        settings: Settings,
        days: int = 120,
        folds: int = 4,
        is_days: int | None = None,
        oos_days: int | None = None,
        seed: int | None = None,
        data_csv: str | None = None,
        research_mode: bool = False,
        end: datetime | None = None,
    ) -> None:
        if folds < 2:
            raise ValueError("walk-forward needs at least 2 folds")
        self.settings = settings
        self.days = max(30, days)
        self.folds = max(2, folds)
        self.seed = seed if seed is not None else settings.backtest.seed
        self.data_csv = data_csv or settings.backtest.data_csv
        self.research_mode = research_mode
        self.end = to_utc(end or utcnow())
        window = self.days / self.folds
        self.is_days = max(5, int(is_days or window * 0.6))
        self.oos_days = max(3, int(oos_days or window * 0.4))

    def _base_config(self) -> BacktestConfig:
        return BacktestConfig(
            days=self.days,
            initial_equity=self.settings.backtest.initial_equity,
            spread_pips=self.settings.backtest.spread_pips,
            slippage_pips=self.settings.backtest.slippage_pips,
            risk_percent=self.settings.backtest.risk_percent,
            seed=self.seed,
            data_csv=self.data_csv,
            min_trades_for_report=self.settings.backtest.min_trades_for_report,
            end=self.end,
            research_mode=self.research_mode,
        )

    async def run(self) -> WalkForwardResult:
        dataset = load_dataset(self._base_config())
        m5 = dataset["M5"]
        if not m5:
            raise ValueError("No data for walk-forward analysis")
        first, last = m5[0].timestamp, m5[-1].timestamp
        span = (last - first) / self.folds
        folds: list[WalkForwardFold] = []
        oos_trades: list[dict[str, Any]] = []

        for index in range(self.folds):
            fold_start = first + span * index
            is_start = fold_start
            is_end = min(is_start + timedelta(days=self.is_days), last)
            oos_start = is_end
            oos_end = min(oos_start + timedelta(days=self.oos_days), last)
            if oos_end <= oos_start:
                continue

            is_result = await self._run_window(dataset, is_start, is_end)
            oos_result = await self._run_window(dataset, oos_start, oos_end)
            folds.append(
                WalkForwardFold(
                    index=index + 1,
                    is_start=is_start,
                    is_end=is_end,
                    oos_start=oos_start,
                    oos_end=oos_end,
                    is_report=is_result.report if is_result else None,
                    oos_report=oos_result.report if oos_result else None,
                    is_signals=len(is_result.signals) if is_result else 0,
                    oos_signals=len(oos_result.signals) if oos_result else 0,
                )
            )
            if oos_result:
                oos_trades.extend(oos_result.trades)

        combined = build_report(
            oos_trades,
            initial_equity=self.settings.backtest.initial_equity,
            min_trades_for_report=self.settings.backtest.min_trades_for_report,
        )
        return WalkForwardResult(
            folds=folds,
            combined_oos=combined,
            settings_summary={
                "instrument": self.settings.bot.instrument,
                "confluence_min": self.settings.strategy.min_confluence_score,
                "five_m_checks": self.settings.strategy.min_fivem_checks,
                "risk_percent": self.settings.risk.max_risk_per_trade_percent,
                "research_mode": self.research_mode,
                "seed": self.seed,
                "folds": self.folds,
                "is_days": self.is_days,
                "oos_days": self.oos_days,
            },
        )

    async def _run_window(
        self, dataset: dict[str, Any], start: datetime, end: datetime
    ) -> BacktestResult | None:
        config = self._base_config()
        config.start = start
        config.end = end
        config.days = max(1, (end - start).days)
        engine = BacktestEngine(self.settings, config)
        # Feed the pre-loaded dataset through a temporary cache so each fold does
        # not regenerate (or reload CSV) data.
        original_loader = engine.run
        try:
            return await self._run_with_dataset(engine, dataset)
        except Exception as exc:  # noqa: BLE001
            print(f"  fold window {start:%Y-%m-%d} → {end:%Y-%m-%d} failed: {exc}", flush=True)
            return None

    async def _run_with_dataset(self, engine: BacktestEngine, dataset: dict[str, Any]) -> BacktestResult:
        import apex_smc_bot.backtesting.backtest_engine as module

        original = module.load_dataset
        module.load_dataset = lambda config: dataset  # type: ignore[assignment]
        try:
            return await engine.run()
        finally:
            module.load_dataset = original  # type: ignore[assignment]

    def run_sync(self) -> WalkForwardResult:
        return asyncio.run(self.run())


# --------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="APEX-SMC walk-forward analysis")
    parser.add_argument("--config", default=None)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--research", action="store_true", help="relax the two quality gates")
    parser.add_argument("--json", default=None, help="write the full result to this path")
    args = parser.parse_args(argv)

    settings = Settings.load(args.config)
    analyzer = WalkForwardAnalyzer(
        settings,
        days=args.days,
        folds=args.folds,
        seed=args.seed,
        data_csv=args.csv,
        research_mode=args.research,
    )
    result = analyzer.run_sync()
    print(result.summary(), flush=True)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
        print(f"\nWritten to {path}", flush=True)
    return 0 if result.combined_oos.trades > 0 else 0  # zero trades is not an error


if __name__ == "__main__":
    raise SystemExit(main())
