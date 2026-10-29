#!/usr/bin/env python3
"""APEX-SMC Institutional Algo Trading Engine — command line entrypoint.

Usage
-----
    python main.py run                    # live/paper engine on the scheduler
    python main.py once                   # one decision cycle (cron-friendly)
    python main.py demo                   # one cycle on synthetic data (no keys)
    python main.py status                 # engine/account/risk snapshot
    python main.py backtest               # run the backtester + report
    python main.py approve <signal_id>    # approve a queued semi-auto signal
    python main.py reject <signal_id>
    python main.py dashboard              # launch the Streamlit dashboard
    python main.py validate-config        # config sanity check
    python main.py selftest               # offline component smoke test

Exit codes: 0 ok, 1 runtime/config error, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any

from apex_smc_bot.config import ConfigError, Settings
from apex_smc_bot.core.engine import TradingEngine
from apex_smc_bot.core.scheduler import EngineScheduler
from apex_smc_bot.data.database import Database
from apex_smc_bot.data.feeds import SUPPORTED_FEEDS, build_feed
from apex_smc_bot.data.synthetic import SyntheticFeed
from apex_smc_bot.execution.broker import Broker
from apex_smc_bot.execution.oanda_executor import OANDAExecutionEngine
from apex_smc_bot.execution.paper_broker import PaperBroker
from apex_smc_bot.logging_setup import setup_logging

logger = logging.getLogger("apex.main")


# ---------------------------------------------------------------------------- setup
def build_database(settings: Settings) -> Database:
    """Open (or no-op) the SQLite state database."""

    if not settings.database.enabled:
        return Database(":memory:", enabled=False)
    settings.ensure_runtime_dirs()
    return Database(settings.database_path)


def build_broker(settings: Settings, database: Database | None = None) -> Broker:
    """Use the selected broker; explicitly selected OANDA fails closed."""

    broker_name = settings.execution.broker
    if broker_name == "oanda":
        # An explicitly selected broker must not silently become a paper account.
        return OANDAExecutionEngine.from_settings(settings, database=database)
    return PaperBroker(
        starting_equity=settings.execution.paper_starting_equity,
        database=None,  # engine records trades; avoids duplicate writes
        slippage_price=settings.risk.max_slippage_pips * settings.risk.pip_size / 2,
        contract_size=settings.risk.contract_size,
    )


def build_engine(
    settings: Settings,
    feed_override: str | None = None,
    database: Database | None = None,
    broker: Broker | None = None,
    synthetic: bool = False,
) -> TradingEngine:
    """Assemble the engine graph."""

    settings.ensure_runtime_dirs()
    database = database or build_database(settings)
    broker = broker or build_broker(settings, database=database)
    if synthetic:
        feed: Any | None = SyntheticFeed(
            days=max(30, settings.backtest.days), seed=settings.feed.synthetic_seed
        )
    else:
        feed = build_feed(settings, override=feed_override)
    bundle = setup_logging(settings)
    engine = TradingEngine(
        settings=settings,
        broker=broker,
        feed=feed,
        database=database,
        logging_bundle=bundle,
    )
    return engine


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop, engine: TradingEngine, scheduler: EngineScheduler | None
) -> None:
    """SIGINT/SIGTERM → graceful shutdown (finish cycle, flush state)."""

    stop = asyncio.Event()

    def _handler(signame: str) -> None:
        if stop.is_set():
            return
        logger.warning("Received %s — shutting down", signame)
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handler, sig.name)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - Windows
            pass

    async def _waiter() -> None:
        await stop.wait()
        if scheduler is not None:
            scheduler.shutdown(wait=False)
        await engine.shutdown("signal")

    return loop.create_task(_waiter())  # type: ignore[return-value]


# ---------------------------------------------------------------------------- commands
async def cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    engine = build_engine(settings, feed_override=args.feed, synthetic=args.synthetic)
    scheduler = EngineScheduler(engine, settings)
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop, engine, scheduler)

    await engine.start()
    print(
        f"APEX-SMC running · mode {settings.bot.execution_mode} · broker {getattr(engine.broker, 'name')} "
        f"· instrument {settings.bot.instrument}\nPress Ctrl+C to stop.",
        flush=True,
    )
    await scheduler.run_forever()
    await engine.shutdown("stopped")
    return 0


async def cmd_once(settings: Settings, args: argparse.Namespace) -> int:
    engine = build_engine(settings, feed_override=args.feed, synthetic=args.synthetic)
    await engine.start()
    report = await engine.cycle(force=True)
    print(json.dumps(report.to_dict(), indent=2), flush=True)
    await engine.shutdown("once")
    return 0 if report.ok else 1


async def cmd_status(settings: Settings, args: argparse.Namespace) -> int:
    engine = build_engine(settings, feed_override=args.feed, synthetic=args.synthetic)
    await engine.start()
    snapshot = engine.status()
    print(engine.status_text(), flush=True)
    print(json.dumps(snapshot, indent=2, default=str), flush=True)
    await engine.shutdown("status")
    return 0


async def cmd_approve(settings: Settings, args: argparse.Namespace, approve: bool = True) -> int:
    """Semi-auto approval from the CLI (same code path as Telegram)."""

    engine = build_engine(settings, feed_override=args.feed, synthetic=args.synthetic)
    await engine.start()
    try:
        if approve:
            result = await engine.orders.approve(args.signal_id, engine._market_price(), engine._spread_price())
        else:
            ok = engine.orders.reject(args.signal_id)
            print("rejected" if ok else "unknown signal id", flush=True)
            await engine.shutdown("approve")
            return 0 if ok else 1
        print(json.dumps(result.to_dict(), indent=2), flush=True)
        await engine.shutdown("approve")
        return 0 if result.success else 1
    finally:
        if engine._running:
            await engine.shutdown("approve")


def cmd_backtest(settings: Settings, args: argparse.Namespace) -> int:
    from apex_smc_bot.backtesting.backtest_engine import run_backtest_cli

    return run_backtest_cli(
        settings,
        days=args.days,
        csv=args.csv,
        seed=args.seed,
        research_mode=getattr(args, "research", False),
        end=getattr(args, "end", None),
    )


def cmd_walkforward(settings: Settings, args: argparse.Namespace) -> int:
    from apex_smc_bot.backtesting.walk_forward import WalkForwardAnalyzer

    from apex_smc_bot.backtesting.backtest_engine import parse_end

    analyzer = WalkForwardAnalyzer(
        settings,
        days=args.days,
        folds=args.folds,
        seed=args.seed,
        data_csv=args.csv,
        research_mode=getattr(args, "research", False),
        end=parse_end(getattr(args, "end", None)),
    )
    result = analyzer.run_sync()
    print(result.summary(), flush=True)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
        print(f"\nWritten to {out}", flush=True)
    return 0


def cmd_dashboard(settings: Settings, args: argparse.Namespace) -> int:
    """Launch Streamlit in-process (blocking)."""

    import subprocess

    app = Path(__file__).resolve().parent / "apex_smc_bot" / "dashboard" / "streamlit_app.py"
    if not app.exists():
        print(f"Dashboard not found at {app}", file=sys.stderr)
        return 1
    print(f"Launching dashboard from {app} ...", flush=True)
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port),
         "--server.address", "0.0.0.0"]
    )


def cmd_validate_config(settings: Settings, args: argparse.Namespace) -> int:
    try:
        settings.validate()
    except ConfigError as exc:
        print(f"INVALID CONFIG: {exc}", file=sys.stderr)
        return 1
    print("Config OK\n" + json.dumps(settings.public_dict(), indent=2, default=str), flush=True)
    return 0


def cmd_selftest(settings: Settings, args: argparse.Namespace) -> int:
    """Offline smoke test: data → analysis → signal → paper execution."""

    from apex_smc_bot.core.selftest import run_selftest

    return run_selftest(settings)


# ---------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="APEX-SMC institutional XAU/USD trading engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--env-file", default=".env", help="path to .env file")
    parser.add_argument("--log-level", default=None, help="override logging level")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--feed", choices=SUPPORTED_FEEDS + ("fallback",), default=None,
                        help="force a specific data feed")
        sp.add_argument("--synthetic", action="store_true", help="use synthetic data (no network)")

    p_run = sub.add_parser("run", help="run the live/paper engine")
    add_common(p_run)

    p_once = sub.add_parser("once", help="run a single decision cycle")
    add_common(p_once)

    p_demo = sub.add_parser("demo", help="one cycle on synthetic data")
    add_common(p_demo)

    p_status = sub.add_parser("status", help="print engine status")
    add_common(p_status)

    p_bt = sub.add_parser("backtest", help="run a backtest")
    p_bt.add_argument("--days", type=int, default=None)
    p_bt.add_argument("--csv", default=None, help="CSV of M5 candles (optional)")
    p_bt.add_argument("--seed", type=int, default=None)
    p_bt.add_argument(
        "--end",
        default=None,
        help="UTC end boundary (last included candle must CLOSE by it), ISO-8601",
    )
    p_bt.add_argument(
        "--research",
        action="store_true",
        help="relax the confluence / 5M gates (synthetic data studies only)",
    )

    p_wf = sub.add_parser("walkforward", help="walk-forward stability analysis")
    p_wf.add_argument("--days", type=int, default=120)
    p_wf.add_argument("--folds", type=int, default=4)
    p_wf.add_argument("--csv", default=None, help="CSV of M5 candles (optional)")
    p_wf.add_argument("--seed", type=int, default=None)
    p_wf.add_argument("--json", default=None, help="write the full result to this path")
    p_wf.add_argument("--end", default=None, help="UTC close-time end boundary, ISO-8601 (default: now)")
    p_wf.add_argument("--research", action="store_true")

    p_approve = sub.add_parser("approve", help="approve a queued signal")
    p_approve.add_argument("signal_id")
    add_common(p_approve)

    p_reject = sub.add_parser("reject", help="reject a queued signal")
    p_reject.add_argument("signal_id")
    add_common(p_reject)

    p_dash = sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    p_dash.add_argument("--port", type=int, default=8501)

    sub.add_parser("validate-config", help="validate configuration")
    sub.add_parser("selftest", help="offline smoke test")
    return parser


def load_env_file(path: str) -> None:
    """Minimal .env loader (no third-party dependency)."""

    import os

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file(args.env_file)
    try:
        settings = Settings.load(args.config)
        if args.log_level:
            from dataclasses import replace

            settings = replace(
                settings,
                logging=replace(settings.logging, level=args.log_level.upper()),
            )
        settings.validate()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load settings: {exc}", file=sys.stderr)
        return 1

    settings.ensure_runtime_dirs()
    setup_logging(settings)

    try:
        if args.command == "run":
            return asyncio.run(cmd_run(settings, args))
        if args.command in {"once", "demo"}:
            if args.command == "demo":
                from dataclasses import replace
                args.synthetic = True
                settings = replace(settings, execution=replace(settings.execution, broker="paper"),
                                   ai=replace(settings.ai, enabled=False), news=replace(settings.news, enabled=False),
                                   notifications=replace(settings.notifications, telegram_enabled=False,
                                                         discord_enabled=False, email_enabled=False))
            return asyncio.run(cmd_once(settings, args))
        if args.command == "status":
            return asyncio.run(cmd_status(settings, args))
        if args.command == "approve":
            return asyncio.run(cmd_approve(settings, args, approve=True))
        if args.command == "reject":
            return asyncio.run(cmd_approve(settings, args, approve=False))
        if args.command == "backtest":
            return cmd_backtest(settings, args)
        if args.command == "walkforward":
            return cmd_walkforward(settings, args)
        if args.command == "dashboard":
            return cmd_dashboard(settings, args)
        if args.command == "validate-config":
            return cmd_validate_config(settings, args)
        if args.command == "selftest":
            return cmd_selftest(settings, args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error")
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
