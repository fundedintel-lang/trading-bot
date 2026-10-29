"""APScheduler orchestration.

Jobs
----
``trading_cycle``  every 5 minutes, offset 5 s past the close (cron ``*/5``)
``manage_only``    every 30 s outside kill zones — position management still runs
``news_refresh``   every 30 min
``state_flush``    every minute — ledger/counter persistence
``daily_reset``    00:01 UTC — roll the drawdown day

The engine loop is serialised by an internal ``asyncio.Lock`` and guarded by a
watchdog so an overrun cycle cannot pile up.  ``misfire_grace_time`` prevents a
late wake-up (laptop sleep, CPU stall) from triggering a burst of cycles.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from apex_smc_bot.config import Settings
from apex_smc_bot.core.engine import CycleReport, TradingEngine
from apex_smc_bot.data.candle_manager import utcnow

logger = logging.getLogger("apex.core.scheduler")


class EngineScheduler:
    """Owns the APScheduler instance and the trading loop callbacks."""

    def __init__(self, engine: TradingEngine, settings: Settings) -> None:
        self.engine = engine
        self.settings = settings
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self._cycle_busy = asyncio.Lock()
        self._last_reports: list[dict[str, Any]] = []
        self._cycle_hook: Callable[[CycleReport], Any] | None = None
        self._job_errors: dict[str, int] = {}

    # ------------------------------------------------------------------ hooks
    def on_cycle(self, hook: Callable[[CycleReport], Any]) -> None:
        """Register a callback invoked after every trading cycle."""

        self._cycle_hook = hook

    async def _record(self, report: CycleReport) -> None:
        self._last_reports.append(report.to_dict())
        self._last_reports = self._last_reports[-50:]
        if self._cycle_hook is not None:
            try:
                result = self._cycle_hook(report)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001
                logger.exception("Cycle hook failed")

    # ------------------------------------------------------------------ jobs
    async def trading_cycle(self, force: bool = False) -> CycleReport | None:
        """One trading cycle; skipped if the previous one is still running."""

        if self._cycle_busy.locked():
            logger.warning("Skipping cycle — previous cycle still running")
            return None
        async with self._cycle_busy:
            try:
                report = await self.engine.cycle(force=force)
            except Exception:  # noqa: BLE001
                logger.exception("Trading cycle crashed")  # engine.cycle already guards
                self._job_errors["trading_cycle"] = self._job_errors.get("trading_cycle", 0) + 1
                return None
            await self._record(report)
            return report

    async def management_cycle(self) -> None:
        """Manage open positions between 5M closes (paper stops, trails)."""

        if self._cycle_busy.locked():
            return
        async with self._cycle_busy:
            report = await self.engine.management_cycle()
            if report.errors:
                self._job_errors["management_cycle"] = self._job_errors.get("management_cycle", 0) + 1

    async def refresh_news(self) -> None:
        if self.engine.news is None:
            return
        try:
            await self.engine.news.refresh(utcnow())
        except Exception:  # noqa: BLE001
            logger.exception("News refresh failed")
            self._job_errors["news_refresh"] = self._job_errors.get("news_refresh", 0) + 1

    async def flush_state(self) -> None:
        try:
            self.engine.state.flush()
        except Exception:  # noqa: BLE001
            logger.exception("State flush failed")

    async def daily_reset(self) -> None:
        now = utcnow()
        try:
            equity = self.engine.state.last_equity(self.settings.risk.starting_equity)
            self.engine.risk.drawdown.roll_day(now, equity)
            self.engine.state.flush()
            logger.info("Daily drawdown baseline rolled at %s", now.isoformat())
        except Exception:  # noqa: BLE001
            logger.exception("Daily reset failed")

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        self.scheduler.add_job(
            self.trading_cycle,
            CronTrigger(minute="*/5", second=5),
            id="trading_cycle",
            name="5M trading cycle",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=120,
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.management_cycle,
            IntervalTrigger(seconds=30),
            id="management_cycle",
            name="Position management",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=20,
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.refresh_news,
            IntervalTrigger(minutes=30),
            id="news_refresh",
            name="News calendar refresh",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=120,
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.flush_state,
            IntervalTrigger(minutes=1),
            id="state_flush",
            name="State flush",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.daily_reset,
            CronTrigger(hour=0, minute=1),
            id="daily_reset",
            name="Daily drawdown reset",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=600,
            replace_existing=True,
        )
        self.scheduler.start()
        logger.info(
            "Scheduler started with jobs: %s",
            ", ".join(job.id for job in self.scheduler.get_jobs()),
        )

    def shutdown(self, wait: bool = False) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("Scheduler stopped")

    @property
    def running(self) -> bool:
        return self.scheduler.running

    def jobs(self) -> list[dict[str, Any]]:
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "errors": self._job_errors.get(job.id, 0),
            }
            for job in self.scheduler.get_jobs()
        ]

    def recent_reports(self, limit: int = 10) -> list[dict[str, Any]]:
        return self._last_reports[-limit:]

    async def run_forever(self) -> None:
        """Start the scheduler and block until the engine is shut down."""

        self.start()
        try:
            await self.engine.wait()
        finally:
            self.shutdown(wait=False)
