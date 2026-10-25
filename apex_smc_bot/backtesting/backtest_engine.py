"""Bar-by-bar backtester that replays the **production engine**.

Design choice: rather than re-implementing the strategy inside a simulator (the
classic source of "it worked in backtest" bugs), every 5M bar is fed through
:class:`~apex_smc_bot.core.engine.TradingEngine.cycle` with a
:class:`~apex_smc_bot.data.replay_feed.ReplayFeed` and a
:class:`~apex_smc_bot.execution.paper_broker.PaperBroker`.

That guarantees the exact same confluence, risk gate, sizing, validation and
position-management code that trades live.

No look-ahead: the feed serves only bars up to the cursor, and each cycle is
stamped with the *close* of the bar being evaluated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

from apex_smc_bot.backtesting.performance_metrics import PerformanceReport, build_report
from apex_smc_bot.config import Settings
from apex_smc_bot.core.engine import TradingEngine
from apex_smc_bot.data.candle_manager import Candle, to_utc, utcnow
from apex_smc_bot.data.database import Database
from apex_smc_bot.data.replay_feed import ReplayFeed
from apex_smc_bot.data.synthetic import generate_market_data, load_candles_from_csv
from apex_smc_bot.execution.paper_broker import PaperBroker
from apex_smc_bot.logging_setup import setup_logging
from apex_smc_bot.notifications.notifier import Notifier

logger = logging.getLogger("apex.backtest")


@dataclass
class BacktestConfig:
    """Runtime knobs (defaults come from ``settings.backtest``)."""

    days: int = 60
    initial_equity: float = 10_000.0
    spread_pips: float = 2.0
    slippage_pips: float = 0.5
    risk_percent: float = 1.0
    seed: int = 42
    data_csv: str | None = None
    min_trades_for_report: int = 20
    start: datetime | None = None
    end: datetime | None = None
    research_mode: bool = False

    @classmethod
    def from_settings(cls, settings: Settings) -> "BacktestConfig":
        cfg = settings.backtest
        return cls(
            days=cfg.days,
            initial_equity=cfg.initial_equity,
            spread_pips=cfg.spread_pips,
            slippage_pips=cfg.slippage_pips,
            risk_percent=cfg.risk_percent,
            seed=cfg.seed,
            data_csv=cfg.data_csv,
            min_trades_for_report=cfg.min_trades_for_report,
        )


@dataclass
class BacktestResult:
    config: BacktestConfig
    report: PerformanceReport
    trades: list[dict[str, Any]] = field(default_factory=list)
    signals: list[dict[str, Any]] = field(default_factory=list)
    cycles: int = 0
    duration_seconds: float = 0.0
    counters: dict[str, Any] = field(default_factory=dict)
    start: datetime | None = None
    end: datetime | None = None

    @property
    def ok(self) -> bool:
        return self.report.trades > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "report": self.report.to_dict(),
            "trades": self.trades,
            "signals": self.signals,
            "cycles": self.cycles,
            "duration_seconds": round(self.duration_seconds, 2),
            "counters": self.counters,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
        }

    def summary(self) -> str:
        header = (
            f"APEX-SMC backtest · {self.config.days}d · seed {self.config.seed} · "
            f"{self.cycles} cycles · {self.duration_seconds:.1f}s"
        )
        return "\n".join([header, self.report.summary()])


def parse_end(value: str | None) -> datetime | None:
    """Parse ``YYYY-MM-DD`` or a full ISO timestamp into an aware UTC datetime."""

    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"--end must be ISO-8601 (e.g. 2026-10-07 or 2026-10-07T14:00), got {value!r}") from exc
    return to_utc(moment)


def load_dataset(config: BacktestConfig) -> dict[str, list[Candle]]:
    """CSV if provided, else deterministic synthetic data."""

    if config.data_csv:
        m5 = load_candles_from_csv(config.data_csv, "M5")
        if not m5:
            raise ValueError(f"No candles loaded from {config.data_csv}")
        from apex_smc_bot.data.candle_manager import resample

        return {"M5": m5, "M15": resample(m5, "M15"), "H1": resample(m5, "H1"), "H4": resample(m5, "H4")}
    end = config.end or utcnow()
    return generate_market_data(days=config.days, seed=config.seed, end=end)


class BacktestEngine:
    """Replays historical data through the live trading engine."""

    def __init__(
        self,
        settings: Settings,
        config: BacktestConfig | None = None,
        database: Database | None = None,
    ) -> None:
        self.settings = settings
        self.config = config or BacktestConfig.from_settings(settings)
        self.database = database

    def _settings_for_run(self) -> Settings:
        """Apply backtest overrides to a copy of the settings."""

        notifications = replace(
            self.settings.notifications,
            telegram_enabled=False,
            discord_enabled=False,
            email_enabled=False,
            console_enabled=False,
        )
        risk = replace(
            self.settings.risk,
            max_risk_per_trade_percent=min(
                self.settings.risk.max_risk_per_trade_percent, self.config.risk_percent
            ),
            starting_equity=self.config.initial_equity,
            min_equity=0.0,
        )
        strategy = self.settings.strategy
        if self.config.research_mode:
            # Explicit opt-in: relaxes the two quality gates so a synthetic data
            # run can produce a statistically useful sample.  Never default.
            strategy = replace(
                strategy,
                min_confluence_score=max(6.0, strategy.min_confluence_score - 1.0),
                min_fivem_checks=max(5, strategy.min_fivem_checks - 1),
            )
        return replace(
            self.settings,
            notifications=notifications,
            risk=risk,
            strategy=strategy,
            news=replace(self.settings.news, enabled=False),
            database=replace(self.settings.database, enabled=False),
        )

    async def run(self, progress: Callable[[int, int], None] | None = None) -> BacktestResult:
        settings = self._settings_for_run()
        data = load_dataset(self.config)
        m5 = data["M5"]
        if not m5:
            raise ValueError("Backtest dataset is empty")

        feed = ReplayFeed(data, spread_price=self.config.spread_pips * settings.risk.pip_size)
        broker = PaperBroker(
            starting_equity=self.config.initial_equity,
            database=None,
            slippage_price=self.config.slippage_pips * settings.risk.pip_size,
            contract_size=settings.risk.contract_size,
        )
        database = self.database
        if database is None:
            database = Database(":memory:")
        # A replay emits one event per cycle; keep the audit log readable.
        bundle = setup_logging(settings, console=False, level="WARNING")
        engine = TradingEngine(
            settings=settings,
            broker=broker,
            feed=feed,
            database=database,
            logging_bundle=bundle,
            notifier=Notifier(settings, database=database, logging_bundle=bundle, banner=False),
        )

        await engine.start()
        start_index = 0
        if self.config.start is not None:
            start_index = feed.seek(self.config.start)
        elif len(m5) > 60 * 24 * self.config.days:
            start_index = len(m5) - 60 * 24 * self.config.days
        end_index = len(m5) - 1
        if self.config.end is not None:
            end_index = feed.seek(self.config.end)

        feed.cursor = start_index
        started = time.perf_counter()
        cycles = 0
        signals: list[dict[str, Any]] = []
        equity_curve: list[float] = [self.config.initial_equity]
        try:
            for index in range(start_index, end_index + 1):
                feed.cursor = index
                now = m5[index].timestamp + timedelta(minutes=5)
                report = await engine.cycle(now=now, force=True)
                cycles += 1
                if report.errors:
                    logger.warning("Cycle %d errors: %s", cycles, report.errors)
                equity_curve.append(broker.equity(m5[index].close))
                if progress is not None and cycles % 500 == 0:
                    progress(index - start_index, end_index - start_index)
            # Close any remaining position at the last bar so metrics are complete.
            if broker.get_open_positions():
                await engine.positions.force_close_all(
                    reason="BACKTEST_END",
                    price=m5[end_index].close,
                    now=m5[end_index].timestamp + timedelta(minutes=5),
                )
                equity_curve[-1] = broker.equity()
            signals = engine.latest_signals(limit=1000)
        finally:
            await engine.shutdown("backtest")

        duration = time.perf_counter() - started
        trades = [position.to_dict() for position in broker.get_history()]
        report_metrics = build_report(
            broker.get_history(),
            equity_curve=equity_curve,
            initial_equity=self.config.initial_equity,
            min_trades_for_report=self.config.min_trades_for_report,
        )
        return BacktestResult(
            config=self.config,
            report=report_metrics,
            trades=trades,
            signals=signals,
            cycles=cycles,
            duration_seconds=duration,
            counters=engine.state.counters.to_dict(),
            start=m5[start_index].timestamp,
            end=m5[end_index].timestamp,
        )

    def run_sync(self, progress: Callable[[int, int], None] | None = None) -> BacktestResult:
        return asyncio.run(self.run(progress=progress))


# --------------------------------------------------------------------------- CLI
def run_backtest_cli(
    settings: Settings,
    days: int | None = None,
    csv: str | None = None,
    seed: int | None = None,
    research_mode: bool | None = None,
    output: str | None = None,
    end: str | None = None,
) -> int:
    """``python main.py backtest`` implementation."""

    config = BacktestConfig.from_settings(settings)
    if days:
        config.days = days
    if csv:
        config.data_csv = csv
    if seed is not None:
        config.seed = seed
    if research_mode is not None:
        config.research_mode = research_mode
    if end:
        config.end = parse_end(end)

    engine = BacktestEngine(settings, config)
    print(
        f"Running backtest: {config.days} days · seed {config.seed} · "
        f"${config.initial_equity:,.0f} · spread {config.spread_pips} pips"
        + (" · RESEARCH MODE" if config.research_mode else ""),
        flush=True,
    )

    def progress(done: int, total: int) -> None:
        pct = (done / total * 100.0) if total else 100.0
        print(f"  ... {done}/{total} bars ({pct:5.1f}%)", flush=True)

    try:
        result = engine.run_sync(progress=progress)
    except KeyboardInterrupt:
        print("Backtest interrupted.", flush=True)
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("Backtest failed")
        print(f"Backtest failed: {exc}", flush=True)
        return 1

    print(result.summary(), flush=True)
    if result.trades:
        print("\nTrades:", flush=True)
        for trade in result.trades[-15:]:
            print(
                f"  {trade.get('trade_id')} {trade.get('direction'):7s} "
                f"{trade.get('initial_units', 0):8.2f}u @ {trade.get('entry_price'):.2f} → "
                f"P/L {trade.get('realized_pnl', 0):+.2f} ({trade.get('close_reason', '')})",
                flush=True,
            )

    out_path = Path(output) if output else Path("reports") / f"backtest_{utcnow():%Y%m%d_%H%M%S}.json"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
        print(f"\nReport written to {out_path}", flush=True)
    except OSError as exc:
        print(f"Could not write report: {exc}", flush=True)
    return 0
