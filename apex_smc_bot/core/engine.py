"""APEX-SMC trading engine.

One :meth:`TradingEngine.cycle` call is one complete decision loop and is safe to
run repeatedly (the scheduler fires it on every 5M close):

    1. refresh candles + quote          (feed, with retries)
    2. reconcile account/equity         (drawdown monitor)
    3. manage open positions            (TP1/TP2/TP3, break-even, trail, SL)
    4. hard gates                       (pause, weekend, dead session, news)
    5. arm 15M setups                   (deduped by fingerprint ledger)
    6. evaluate 5M triggers             (7 checks, ≥6 to execute)
    7. confluence score                 (≥7/10)
    8. build signal → validate → risk   (portfolio gate)
    9. submit order → notify            (semi-auto approval aware)

Everything is wrapped so that a single failing stage logs and continues: the
engine must never die silently or stop trading because one provider hiccupped.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from apex_smc_bot.analysis.confluence import ConfluenceEngine, ConfluenceResult
from apex_smc_bot.analysis.fvg import detect_fvg
from apex_smc_bot.analysis.htf_bias import AlignmentResult, HTFBiasEngine
from apex_smc_bot.analysis.liquidity import build_liquidity_map, detect_inducement
from apex_smc_bot.analysis.order_blocks import detect_order_blocks
from apex_smc_bot.analysis.premium_discount import dealing_range_from_structure
from apex_smc_bot.analysis.session import Session, SessionEngine, detect_judas_swing
from apex_smc_bot.analysis.structure import BEARISH, BULLISH, analyze_structure
from apex_smc_bot.config import Settings
from apex_smc_bot.core.state_manager import StateManager
from apex_smc_bot.data.candle_manager import (
    REQUIRED_TIMEFRAMES,
    Candle,
    CandleManager,
    to_utc,
    utcnow,
)
from apex_smc_bot.data.database import Database
from apex_smc_bot.execution.broker import Broker, BrokerError, OrderResult
from apex_smc_bot.execution.order_manager import OrderManager
from apex_smc_bot.execution.position_manager import PositionEvent, PositionManager
from apex_smc_bot.logging_setup import log_analysis, log_engine, log_exception, setup_logging
from apex_smc_bot.notifications.notifier import Notifier
from apex_smc_bot.risk.news_filter import NewsFilter
from apex_smc_bot.risk.risk_engine import RiskDecision, RiskEngine
from apex_smc_bot.signals.models import PendingSetup, Signal, TriggerResult
from apex_smc_bot.signals.signal_formatter import format_rejection
from apex_smc_bot.signals.signal_generator import SignalGenerator
from apex_smc_bot.signals.signal_validator import SignalValidator

logger = logging.getLogger("apex.core.engine")

TRIGGER_WINDOW_5M = 40
# Bounded analysis windows keep each cycle O(window) instead of O(history).
M15_WINDOW = 400
H1_WINDOW = 300
H4_WINDOW = 200


@dataclass
class CycleReport:
    """What happened in one cycle (returned for tests, dashboard and logs)."""

    cycle_id: int
    started_at: datetime
    duration_ms: float = 0.0
    price: float = 0.0
    spread_pips: float = 0.0
    session: str = ""
    tradeable_session: bool = False
    armed: int = 0
    triggers: int = 0
    signals: int = 0
    orders: int = 0
    rejections: list[str] = field(default_factory=list)
    position_events: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped: str = ""
    signal_ids: list[str] = field(default_factory=list)
    risk: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at.isoformat(),
            "duration_ms": round(self.duration_ms, 1),
            "price": round(self.price, 2),
            "spread_pips": round(self.spread_pips, 2),
            "session": self.session,
            "tradeable_session": self.tradeable_session,
            "armed": self.armed,
            "triggers": self.triggers,
            "signals": self.signals,
            "orders": self.orders,
            "rejections": self.rejections[-20:],
            "position_events": self.position_events[-20:],
            "errors": self.errors,
            "skipped": self.skipped,
            "signal_ids": self.signal_ids,
            "risk": self.risk,
        }

    @property
    def ok(self) -> bool:
        return not self.errors


class TradingEngine:
    """Wires every layer together and owns the trading loop."""

    def __init__(
        self,
        settings: Settings,
        broker: Broker,
        feed: Any | None = None,
        database: Database | None = None,
        notifier: Notifier | None = None,
        logging_bundle: Any | None = None,
    ) -> None:
        self.settings = settings
        self.broker = broker
        self.database = database
        self.logging_bundle = logging_bundle or setup_logging(settings)
        self.notifier = notifier or Notifier.from_settings(
            settings, database=database, logging_bundle=self.logging_bundle
        )
        self.candles = CandleManager(
            instrument=settings.bot.instrument, feed=feed, database=database
        )
        self.state = StateManager(database)
        self.risk = RiskEngine(settings, database)
        self.news = NewsFilter(settings.news, database) if settings.news.enabled else None
        self.generator = SignalGenerator(settings, database=database)
        self.confluence = ConfluenceEngine(settings)
        self.validator = SignalValidator(settings)
        self.htf = HTFBiasEngine(settings)
        self.sessions = SessionEngine(settings.sessions)
        self.orders = OrderManager(settings, broker, database)
        self.positions = PositionManager(settings, broker, database, risk_engine=self.risk)

        self._cycle_id = 0
        self._last_bar_ts: datetime | None = None
        self._last_arm_ts: datetime | None = None
        self._cache_alignment: AlignmentResult | None = None
        self._cache_structure: Any = None
        self._cache_blocks: list[Any] = []
        self._running = False
        self._shutdown_done = False
        self._stopping = asyncio.Event()
        self._last_report: CycleReport | None = None
        self._halted_notified = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Bootstrap history and announce readiness. Never raises."""

        log_engine(self.logging_bundle, "ENGINE_STARTING", mode=self.settings.bot.execution_mode)
        try:
            if self.candles.feed is not None:
                counts = await self.candles.bootstrap(minimum=self.settings.feed.historical_bars)
                log_engine(self.logging_bundle, "HISTORY_LOADED", counts=counts)
            else:
                logger.warning("No data feed configured — engine will use cached candles only")
        except Exception as exc:  # noqa: BLE001
            self.state.note_error(str(exc))
            log_exception(logger, "bootstrap_failed", exc)
        self._running = True
        await self.notifier.notify_startup(
            {
                "mode": self.settings.bot.execution_mode,
                "broker": getattr(self.broker, "name", "unknown"),
                "instrument": self.settings.bot.instrument,
                "candles": self.candles.snapshot(),
            }
        )

    async def shutdown(self, reason: str = "shutdown") -> None:
        """Flush state, close clients. Positions are left untouched.

        Idempotent: a signal handler and the command wrapper may both call it.
        """

        if self._shutdown_done:
            logger.debug("shutdown() already completed — ignoring %r", reason)
            return
        self._shutdown_done = True
        self._running = False
        self._stopping.set()
        self.state.flush()
        if self.database is not None:
            try:
                self.database.close()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to close database")
        await self.notifier.close()
        try:
            await self.broker.close()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to close broker")
        if self.candles.feed is not None:
            try:
                await self.candles.feed.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        log_engine(self.logging_bundle, "ENGINE_STOPPED", reason=reason)
        print(f"APEX-SMC engine stopped ({reason}).", flush=True)

    # ------------------------------------------------------------------ data
    async def _refresh_data(self) -> None:
        """Pull fresh candles and quote; tolerate individual timeframe failures."""

        if self.candles.feed is not None:
            try:
                await self.candles.refresh_all()
            except Exception as exc:  # noqa: BLE001
                self.state.note_error(f"candle refresh: {exc}")
                log_exception(logger, "candle_refresh_failed", exc)
        try:
            bid, ask = await self.candles.refresh_quote()
            if bid > 0 and ask > 0:
                self.candles.set_quote(bid, ask)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Quote refresh failed: %s", exc)

        if self.database is not None:
            for timeframe in REQUIRED_TIMEFRAMES:
                bars = self.candles.get_candles(timeframe)
                if bars:
                    try:
                        self.database.upsert_candles(
                            self.settings.bot.instrument, timeframe, bars[-300:]
                        )
                    except Exception:  # noqa: BLE001
                        logger.debug("Candle persistence failed for %s", timeframe, exc_info=True)

    def _market_price(self, timeframe: str = "M5") -> float:
        price = self.candles.latest_price
        if price > 0:
            return price
        bars = self.candles.get_candles(timeframe)
        return bars[-1].close if bars else 0.0

    def _spread_price(self, candle: Candle | None = None) -> float:
        spread = self.candles.latest_spread
        if spread > 0:
            return spread
        if candle is not None and candle.spread > 0:
            return candle.spread
        return self.settings.risk.max_spread_pips * self.settings.risk.pip_size

    # ------------------------------------------------------------------ cycle
    async def cycle(self, now: datetime | None = None, force: bool = False) -> CycleReport:
        """Run one full decision loop. Safe to call concurrently (serialised)."""

        async with self._lock:
            return await self._cycle(now=now, force=force)

    async def _cycle(self, now: datetime | None = None, force: bool = False) -> CycleReport:
        started = time.perf_counter()
        now = to_utc(now or utcnow())
        self._cycle_id += 1
        report = CycleReport(cycle_id=self._cycle_id, started_at=now)
        self.state.tick_cycle(now)

        try:
            await self._refresh_data()
            bars_5m = self.candles.get_candles("M5")
            bars_15m = self.candles.get_candles("M15")
            bars_1h = self.candles.get_candles("H1")
            bars_4h = self.candles.get_candles("H4")

            if len(bars_5m) < 10 or len(bars_15m) < 30:
                report.skipped = "INSUFFICIENT_DATA"
                return report

            last_bar = bars_5m[-1].timestamp
            report.price = self._market_price()
            report.spread_pips = self._spread_price(bars_5m[-1]) / self.settings.risk.pip_size
            session = self.sessions.get_current_session(now)
            report.session = session.name
            report.tradeable_session = session.tradeable

            # -------- account + drawdown -------------------------------------
            equity = await self._account_equity(report)
            self.state.set_last_equity(equity)
            drawdown = self.risk.update_equity(equity, now)
            self.risk.update_spread(self._spread_price(bars_5m[-1]), now)
            report.risk = {"equity": round(equity, 2), "drawdown": drawdown.to_dict()}
            await self._handle_circuit_breaker(drawdown, report)

            # -------- position management ------------------------------------
            await self._manage_positions(report, bars_5m, now)

            # -------- gating --------------------------------------------------
            skip_reason = await self._hard_gates(now, session, report)
            if skip_reason:
                report.skipped = skip_reason
                self.state.prune(now)
                return report

            if not force and self._last_bar_ts is not None and last_bar <= self._last_bar_ts:
                report.skipped = "NO_NEW_BAR"
                return report
            self._last_bar_ts = last_bar

            # -------- 15M arming (only when a new 15M bar closed) ------------
            new_15m = self._last_arm_ts is None or bars_15m[-1].timestamp > self._last_arm_ts
            if new_15m or force or self._cache_alignment is None:
                alignment, structure, order_blocks = await self._arm_setups(
                    bars_15m, bars_1h, bars_4h, session, now, report
                )
                if alignment is not None:
                    self._cache_alignment = alignment
                    self._cache_structure = structure
                    self._cache_blocks = list(order_blocks)
                    self._last_arm_ts = bars_15m[-1].timestamp
            else:
                alignment = self._cache_alignment
                structure = self._cache_structure
                order_blocks = self._cache_blocks

            # -------- 5M triggers --------------------------------------------
            if alignment is not None and self.state.get_pending(now=now):
                await self._evaluate_triggers(
                    alignment, structure, order_blocks, bars_5m, session, now, report
                )

            self.state.prune(now)
        except Exception as exc:  # noqa: BLE001
            self.state.note_error(str(exc))
            report.errors.append(str(exc))
            log_exception(logger, "cycle_failed", exc, cycle=self._cycle_id)
        finally:
            self._finish(report, started)
        return report

    def _finish(self, report: CycleReport, started: float) -> None:
        report.duration_ms = (time.perf_counter() - started) * 1000.0
        self._last_report = report
        if self.logging_bundle is not None:
            log_engine(
                self.logging_bundle,
                "CYCLE_COMPLETE",
                cycle=report.cycle_id,
                duration_ms=round(report.duration_ms, 1),
                armed=report.armed,
                triggers=report.triggers,
                signals=report.signals,
                orders=report.orders,
                skipped=report.skipped,
                errors=report.errors,
            )
        if self.database is not None:
            try:
                self.database.record_event("CYCLE", report.to_dict())
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ stages
    async def _account_equity(self, report: CycleReport) -> float:
        try:
            account = await self.broker.get_account()
            return float(account.equity or account.balance)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"account: {exc}")
            logger.warning("Account snapshot failed (%s) — using cached equity", exc)
            return self.state.last_equity(self.settings.risk.starting_equity)

    async def _handle_circuit_breaker(self, drawdown: Any, report: CycleReport) -> None:
        """Flatten everything and alert when the daily breaker trips."""

        if drawdown.allow_trading:
            self._halted_notified = False
            return
        reason = str(getattr(drawdown, "reason", "drawdown halt"))
        if "COOLDOWN" in reason.upper() and not getattr(drawdown, "breaker_triggered", False):
            report.skipped = reason
            return
        if self._halted_notified and not getattr(drawdown, "breaker_triggered", False):
            return
        logger.error("CIRCUIT BREAKER: %s", reason)
        events = await self.positions.force_close_all(reason="CIRCUIT_BREAKER")
        self._halted_notified = True
        report.position_events.extend(event.kind for event in events)
        await self.notifier.notify_trade_event(
            "CIRCUIT_BREAKER",
            {"reason": reason, "closed": len(events)},
            text=f"🚨 CIRCUIT BREAKER — {reason}. Closed {len(events)} position(s).",
        )

    async def _manage_positions(
        self, report: CycleReport, bars_5m: Sequence[Candle], now: datetime
    ) -> None:
        price = self._market_price()
        if price <= 0:
            return
        try:
            events: list[PositionEvent] = await self.positions.on_price(
                price, self._spread_price(bars_5m[-1] if bars_5m else None), bars_5m, now
            )
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"positions: {exc}")
            log_exception(logger, "position_management_failed", exc)
            return
        for event in events:
            report.position_events.append(f"{event.kind}:{event.trade_id}")
            await self.notifier.notify_trade_event(event.kind, event.to_dict(), text=self._format_event(event))
            if event.kind in {"STOP_HIT", "TP3_HIT", "CLOSED"} and self.database is not None:
                try:
                    self.database.update_trade(
                        event.trade_id,
                        status="CLOSED",
                        realized_pnl=event.pnl,
                        closed_at=now.isoformat(),
                    )
                except Exception:  # noqa: BLE001
                    pass

    @staticmethod
    def _format_event(event: PositionEvent) -> str:
        emoji = {
            "TP1_HIT": "✅",
            "TP2_HIT": "✅",
            "TP3_HIT": "🏁",
            "STOP_HIT": "🛑",
            "TRAIL_UPDATED": "🔒",
            "CLOSED": "⚪",
            "BROKER_CLOSED": "⚪",
        }.get(event.kind, "ℹ️")
        pnl = f" P/L {event.pnl:+.2f}" if event.pnl else ""
        return f"{emoji} {event.kind} {event.trade_id} @ {event.price:.2f}{pnl} — {event.message}"

    async def _hard_gates(self, now: datetime, session: Session, report: CycleReport) -> str:
        """Return a skip reason, or an empty string when trading may proceed."""

        if self.state.paused:
            return "PAUSED"
        if now.weekday() >= 5:
            return "WEEKEND"
        if not session.tradeable:
            return f"SESSION_NOT_TRADEABLE:{session.name}"
        drawdown = self.risk.drawdown.decision(now)
        report.risk["drawdown"] = drawdown.to_dict()
        if not drawdown.allow_trading:
            return f"RISK_HALT:{drawdown.reason}"
        if self.news is not None:
            try:
                await self.news.refresh(now)
                check = self.news.check(now)
                report.risk["news"] = check.to_dict()
                if not check.safe:
                    return f"NEWS_BLACKOUT:{','.join(event.title for event in check.events)[:80]}"
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"news: {exc}")
                log_exception(logger, "news_check_failed", exc)
        if not self._within_session_window(now, session):
            return f"OUTSIDE_KILL_ZONE:{session.name}"
        return ""

    def _within_session_window(self, now: datetime, session: Session) -> bool:
        """Is this a window the strategy may arm in?

        Kill zones (London 07-10, NY 12-15 UTC) are always allowed.  With
        ``sessions.trade_secondary_sessions`` the London afternoon and the
        London close / NY afternoon are allowed too — they are liquid sessions,
        unlike the dead zones which the session table already marks
        non-tradeable.
        """

        if session.is_kill_zone:
            return self._time_in_window(now, session)
        if not self.settings.sessions.trade_secondary_sessions:
            return False
        return self._time_in_window(now, session)

    @staticmethod
    def _time_in_window(now: datetime, session: Session) -> bool:
        minutes = now.hour * 60 + now.minute
        start = session.start.hour * 60 + session.start.minute
        end = session.end.hour * 60 + session.end.minute
        if end <= start:  # window wraps midnight
            return minutes >= start or minutes < end
        return start <= minutes < end

    async def _arm_setups(
        self,
        bars_15m: Sequence[Candle],
        bars_1h: Sequence[Candle],
        bars_4h: Sequence[Candle],
        session: Session,
        now: datetime,
        report: CycleReport,
    ) -> tuple[AlignmentResult | None, Any, list[Any]]:
        try:
            m15 = list(bars_15m[-M15_WINDOW:])
            h1 = list(bars_1h[-H1_WINDOW:])
            h4 = list(bars_4h[-H4_WINDOW:])
            structure = analyze_structure(
                m15, "M15", lookback=self.settings.strategy.swing_lookback
            )
            bars_15m = m15
            order_blocks = detect_order_blocks(
                list(bars_15m),
                structure.events,
                "M15",
                lookback=self.settings.strategy.ob_lookback_candles,
            )
            fvgs = detect_fvg(list(bars_15m), "M15")
            price = self._market_price()
            four_h = self.htf.analyze(h4, "H4", current_price=price)
            one_h = self.htf.analyze_1h(h1, current_price=price, four_h=four_h)
            alignment = self.htf.check_alignment(four_h, one_h)
            if self.settings.strategy.daily_trend_filter and alignment.aligned:
                daily = self.htf.daily_trend(h4)
                if daily in {BULLISH, BEARISH} and daily != alignment.direction:
                    report.rejections.append("D1_TREND_CONFLICT")
                    self.state.note_rejection()
                    return alignment, structure, order_blocks
            # Built *after* the bias so sweeps are graded against the direction
            # actually being traded (bias=None can only ever yield HIGH/LOW).
            liquidity = build_liquidity_map(
                list(bars_15m),
                "M15",
                tolerance_pips=self.settings.strategy.equal_level_tolerance_pips,
                internal_lookback=self.settings.strategy.swing_lookback,
                external_lookback=self.settings.strategy.external_swing_lookback,
                bias=alignment.direction if alignment.aligned else None,
                lookahead=self.settings.strategy.sweep_lookahead_candles,
                pip_size=self.settings.risk.pip_size,
                single_swing_lookback=self.settings.strategy.single_swing_lookback_bars,
            )
            dealing_range = dealing_range_from_structure(
                structure,
                self.settings.strategy.ote_low_fib,
                self.settings.strategy.ote_high_fib,
                bias=alignment.direction,
            )
            judas = None
            try:
                start_at = self.sessions.judas_session_start(now)
                if start_at is not None:
                    judas = detect_judas_swing(
                        list(bars_15m),
                        session,
                        start_at,
                        self.settings.sessions.judas_window_minutes,
                    )
            except Exception:  # noqa: BLE001
                judas = None

            pending = self.generator.check_15m_setup(
                structure,
                order_blocks,
                fvgs,
                liquidity,
                alignment,
                session,
                candles=bars_15m,
                dealing_range=dealing_range,
                now=now,
                judas=judas,
                current_price=price,
            )
            if pending is None:
                rejection = self.generator.last_rejection or "NO_SETUP"
                report.rejections.append(rejection)
                self.state.note_rejection()
                if self.logging_bundle is not None:
                    log_analysis(
                        self.logging_bundle,
                        "setup_rejected",
                        reason=rejection,
                        session=session.name,
                    )
                return alignment, structure, order_blocks

            if self.state.is_spent(pending):
                report.rejections.append("DUPLICATE_FINGERPRINT")
                return alignment, structure, order_blocks

            added = self.state.add_pending(pending, now)
            if added:
                report.armed += 1
                logger.info(
                    "ARMED %s %s zone %.2f-%.2f (%s)",
                    pending.side,
                    pending.fingerprint,
                    pending.zone_low,
                    pending.zone_high,
                    pending.session_name,
                )
                if self.logging_bundle is not None:
                    log_analysis(self.logging_bundle, "setup_armed", **pending.to_dict())
            return alignment, structure, order_blocks
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"arm: {exc}")
            log_exception(logger, "arming_failed", exc)
            return None, None, []

    async def _evaluate_triggers(
        self,
        alignment: AlignmentResult,
        structure: Any,
        order_blocks: Sequence[Any],
        bars_5m: Sequence[Candle],
        session: Session,
        now: datetime,
        report: CycleReport,
    ) -> None:
        window = list(bars_5m[-TRIGGER_WINDOW_5M:])
        spread_price = self._spread_price(bars_5m[-1])
        htf_blocks = self._htf_order_blocks(alignment)
        for pending in self.state.get_pending(now=now):
            if pending.direction != alignment.direction and alignment.aligned:
                continue
            try:
                trigger = self.generator.five_minute_confirmation(
                    pending,
                    window,
                    spread_price=spread_price,
                    order_blocks=order_blocks,
                    htf_order_blocks=htf_blocks,
                    now=now,
                )
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"trigger: {exc}")
                log_exception(logger, "trigger_failed", exc, fingerprint=pending.fingerprint)
                continue

            if trigger.execute:
                report.triggers += 1
                self.state.note_trigger(True)
                await self._process_signal(
                    pending,
                    trigger,
                    alignment,
                    order_blocks,
                    htf_blocks,
                    session,
                    spread_price,
                    now,
                    report,
                )
            elif trigger.reason.startswith(("SETUP_INVALIDATED", "PENDING_EXPIRED")):
                self.state.drop_pending(pending.fingerprint)
                if self.logging_bundle is not None:
                    log_analysis(
                        self.logging_bundle, "setup_dropped", fingerprint=pending.fingerprint, reason=trigger.reason
                    )

    @staticmethod
    def _htf_order_blocks(alignment: AlignmentResult | None) -> list[Any]:
        if alignment is None:
            return []
        blocks: list[Any] = []
        for bias in (getattr(alignment, "four_h", None), getattr(alignment, "one_h", None)):
            if bias is None:
                continue
            blocks.extend(list(getattr(bias, "order_blocks", []) or []))
        return blocks

    async def _process_signal(
        self,
        pending: PendingSetup,
        trigger: TriggerResult,
        alignment: AlignmentResult,
        order_blocks: Sequence[Any],
        htf_blocks: Sequence[Any],
        session: Session,
        spread_price: float,
        now: datetime,
        report: CycleReport,
    ) -> None:
        price = self._market_price()
        spread_pips = spread_price / self.settings.risk.pip_size

        # Portfolio gate BEFORE sizing.
        risk_decision: RiskDecision = self.risk.evaluate(
            now=now,
            equity=self.state.last_equity(self.settings.risk.starting_equity),
            open_trade_count=await self.orders.get_open_trade_count(),
            session=session,
            spread_price=spread_price,
            pending=pending,
            candles_15m=self.candles.get_candles("M15"),
            order_blocks=order_blocks,
            market_price=price,
        )
        report.risk["gate"] = risk_decision.to_dict()
        if not risk_decision.allow:
            report.rejections.append(f"RISK:{risk_decision.summary()}")
            self.state.note_rejection()
            await self.notifier.notify_rejection("RISK_GATE", risk_decision.to_dict())
            return

        try:
            setup = self.generator.build_setup(
                pending,
                trigger,
                alignment,
                spread_pips,
                normal_spread_pips=self.risk.spread.normal_pips / self.settings.risk.pip_size,
                session=session,
                order_blocks=order_blocks,
                htf_order_blocks=htf_blocks,
            )
            confluence: ConfluenceResult = self.confluence.score(setup)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"confluence: {exc}")
            log_exception(logger, "confluence_failed", exc)
            return

        if not confluence.trade_allowed:
            report.rejections.append(
                f"CONFLUENCE:{confluence.abort_reason or f'{confluence.score:.1f}/10'}"
            )
            self.state.note_rejection()
            if self.logging_bundle is not None:
                log_analysis(
                    self.logging_bundle,
                    "confluence_rejected",
                    score=confluence.score,
                    reason=confluence.abort_reason,
                    breakdown=confluence.breakdown,
                )
            return

        signal = self.generator.generate_full_signal(
            pending,
            trigger,
            alignment,
            confluence=confluence,
            setup=setup,
            account_equity=self.state.last_equity(self.settings.risk.starting_equity),
            risk_percent=risk_decision.risk_percent,
            order_blocks=order_blocks,
            spread_pips=spread_pips,
            now=now,
            market_price=price,
        )
        if signal is None:
            report.rejections.append(f"SIGNAL_BUILD:{self.generator.last_rejection or 'unknown'}")
            self.state.note_rejection()
            return

        validation = self.validator.validate(
            signal,
            market_price=price,
            now=now,
            open_trade_count=await self.orders.get_open_trade_count(),
            account_equity=self.state.last_equity(self.settings.risk.starting_equity),
        )
        if not validation.valid:
            report.rejections.append(f"VALIDATION:{validation.summary()}")
            self.state.note_rejection()
            await self.notifier.notify_rejection("VALIDATION_FAILED", validation.to_dict())
            return

        # Persist + broadcast the signal before routing the order.
        self.state.mark_signalled(pending, signal)
        report.signals += 1
        report.signal_ids.append(signal.signal_id)
        await self.notifier.notify_signal(signal)

        result: OrderResult = await self.orders.submit(signal, price, spread_price, now)
        if result.status == "PENDING_APPROVAL":
            await self.notifier.notify_trade_event(
                "AWAITING_APPROVAL",
                result.to_dict(),
                text=(
                    f"⏳ {signal.signal_id} waiting for approval\n"
                    f"/approve {signal.signal_id}  or  /reject {signal.signal_id}"
                ),
            )
        elif result.success:
            report.orders += 1
            self.state.note_order()
            await self.notifier.notify_trade_event(
                "ORDER_EXECUTED", result.to_dict(), text=f"🟢 {result.status} {signal.signal_id} @ {result.filled_price}"
            )
        else:
            report.rejections.append(f"ORDER:{result.message}")
            self.state.note_rejection()
            await self.notifier.notify_trade_event(
                "ORDER_REJECTED", result.to_dict(), text=f"❌ Order rejected: {result.message}"
            )

    # ------------------------------------------------------------------ commands
    async def handle_command(self, command: str, args: list[str]) -> str:
        """Telegram/CLI command dispatch. Returns the reply text."""

        command = command.lower().lstrip("/")
        price = self._market_price()
        spread = self._spread_price()

        if command in {"help", "start"}:
            return (
                "APEX-SMC commands:\n"
                "/status — engine, account and risk\n"
                "/positions — open trades\n"
                "/approve <signal_id> — execute a pending signal\n"
                "/reject <signal_id> — discard it\n"
                "/close <trade_id> — close one trade\n"
                "/closeall — flatten everything\n"
                "/pause — stop opening new trades\n"
                "/resume — resume trading\n"
                "/signals — last 5 signals"
            )
        if command == "status":
            return self.status_text()
        if command == "positions":
            positions = await self.orders.get_positions()
            if not positions:
                return "No open positions."
            return "\n".join(
                f"{p.trade_id} {p.direction} {p.units:.2f}u @ {p.entry_price:.2f} "
                f"SL {p.stop_loss:.2f} P/L {p.unrealized_pnl():+.2f}"
                for p in positions
            )
        if command == "approve":
            if not args:
                return "Usage: /approve <signal_id>"
            result = await self.orders.approve(args[0], price, spread)
            if result.success:
                return f"✅ Approved {args[0]} — {result.status} @ {result.filled_price}"
            return f"❌ Could not approve {args[0]}: {result.message}"
        if command == "reject":
            if not args:
                return "Usage: /reject <signal_id>"
            ok = self.orders.reject(args[0])
            return f"🗑 Rejected {args[0]}" if ok else f"Unknown signal {args[0]}"
        if command == "close":
            if not args:
                return "Usage: /close <trade_id>"
            event = await self.positions.force_close(args[0], "MANUAL", price)
            return f"Closed {args[0]} ({event.pnl:+.2f})" if event else f"Unknown trade {args[0]}"
        if command == "closeall":
            events = await self.positions.force_close_all("MANUAL", price)
            return f"Closed {len(events)} position(s)."
        if command == "pause":
            self.state.pause("telegram")
            return "⏸ Trading paused — positions still managed."
        if command == "resume":
            self.state.resume()
            return "▶️ Trading resumed."
        if command == "signals":
            if self.database is None:
                return "No database configured."
            rows = self.database.get_signals(limit=5)
            if not rows:
                return "No signals recorded yet."
            return "\n".join(
                f"{row['created_at'][:16]} {row['direction']} {row['signal_id']} "
                f"score {row.get('confluence_score')} [{row['status']}]"
                for row in rows
            )
        return f"Unknown command /{command}. Try /help."

    def status_text(self) -> str:
        report = self._last_report
        lines = [
            "APEX-SMC status",
            f"mode: {self.settings.bot.execution_mode} · broker: {getattr(self.broker, 'name', '?')}",
            f"paused: {self.state.paused}",
            f"equity: ${self.state.last_equity(self.settings.risk.starting_equity):,.2f}",
            f"price: {self._market_price():.2f} · spread {self._spread_price():.2f}",
            f"session: {self.sessions.get_current_session().name}",
        ]
        if report is not None:
            lines.append(
                f"last cycle: {report.started_at:%H:%M:%S} · armed {report.armed} · "
                f"signals {report.signals} · orders {report.orders} · {report.duration_ms:.0f} ms"
            )
            if report.errors:
                lines.append(f"errors: {report.errors[-1]}")
        counters = self.state.counters
        lines.append(
            f"counters: cycles {counters.cycles} · setups {counters.setups_armed} · "
            f"signals {counters.signals} · orders {counters.orders}"
        )
        return "\n".join(lines)

    def status(self) -> dict[str, Any]:
        """Structured status used by the dashboard and ``main.py status``."""

        positions = []
        try:
            positions = [p.to_dict() for p in getattr(self.broker, "get_open_positions", lambda: [])()]
        except Exception:  # noqa: BLE001
            positions = []
        return {
            "engine": {
                "mode": self.settings.bot.execution_mode,
                "broker": getattr(self.broker, "name", "unknown"),
                "instrument": self.settings.bot.instrument,
                "running": self._running,
                "candles": self.candles.snapshot(),
                "last_report": self._last_report.to_dict() if self._last_report else None,
            },
            "state": self.state.status(),
            "risk": self.risk.status(self.state.last_equity(self.settings.risk.starting_equity)),
            "positions": positions,
            "pending_approvals": self.orders.get_pending_approvals(),
            "notifications": self.notifier.active_channels,
            "price": self._market_price(),
            "spread": self._spread_price(),
        }

    # ------------------------------------------------------------------ helpers
    def latest_signals(self, limit: int = 20) -> list[dict[str, Any]]:
        if self.database is None:
            return []
        try:
            return self.database.get_signals(limit=limit)
        except Exception:  # noqa: BLE001
            return []

    def remaining_pending(self) -> list[PendingSetup]:
        return self.state.get_pending()

    async def wait(self) -> None:
        """Block until shutdown is requested (used by the scheduler runner)."""

        await self._stopping.wait()
