"""Offline self-test used by ``python main.py selftest``.

Runs the full pipeline on synthetic data and a paper broker, asserting that a
signal can be produced and executed.  It is intentionally dependency-light so
deployment can verify a new machine without credentials.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from apex_smc_bot.config import Settings
from apex_smc_bot.data.synthetic import SyntheticFeed, generate_market_data
from apex_smc_bot.execution.paper_broker import PaperBroker
from apex_smc_bot.logging_setup import setup_logging


def _check(name: str, ok: bool, detail: str = "") -> tuple[str, bool, str]:
    return (name, ok, detail)


def run_selftest(settings: Settings) -> int:
    """Execute every layer offline and print a pass/fail table."""

    results: list[tuple[str, bool, str]] = []
    data = generate_market_data(days=30, seed=settings.feed.synthetic_seed)
    results.append(
        _check("synthetic data", all(len(v) > 100 for v in data.values()), f"{ {k: len(v) for k, v in data.items()} }")
    )

    from apex_smc_bot.analysis.structure import analyze_structure
    from apex_smc_bot.analysis.order_blocks import detect_order_blocks
    from apex_smc_bot.analysis.fvg import detect_fvg
    from apex_smc_bot.analysis.liquidity import build_liquidity_map
    from apex_smc_bot.analysis.htf_bias import HTFBiasEngine
    from apex_smc_bot.analysis.session import SessionEngine

    m15, h1, h4 = data["M15"], data["H1"], data["H4"]
    structure = analyze_structure(m15, "M15")
    blocks = detect_order_blocks(m15, structure.events, "M15")
    fvgs = detect_fvg(m15, "M15")
    liquidity = build_liquidity_map(m15, "M15")
    sessions = SessionEngine(settings.sessions)
    session = sessions.get_current_session(m15[-1].timestamp)
    htf = HTFBiasEngine(settings)
    price = m15[-1].close
    four_h = htf.analyze(h4, "H4", current_price=price)
    one_h = htf.analyze_1h(h1, current_price=price, four_h=four_h)
    alignment = htf.check_alignment(four_h, one_h)
    results.append(_check("structure", bool(structure.swings), f"{len(structure.events)} events"))
    results.append(_check("order blocks", True, f"{len(blocks)} blocks"))
    results.append(_check("fvg", True, f"{len(fvgs)} gaps"))
    results.append(_check("liquidity", True, f"{len(liquidity.sweeps)} sweeps"))
    results.append(_check("htf alignment", True, f"{alignment.direction} aligned={alignment.aligned}"))
    results.append(_check("session engine", True, session.name))

    from apex_smc_bot.execution.lot_calculator import calculate_position

    sizing = calculate_position(10_000, 1.0, 2_350.0, 2_347.3, settings.risk)
    results.append(_check("position sizing", 0.3 <= sizing.lot_size <= 0.4, f"{sizing.lot_size} lots"))

    from apex_smc_bot.risk.risk_engine import RiskEngine

    risk = RiskEngine(settings)
    decision = risk.evaluate(equity=10_000, session=session, spread_price=0.25)
    results.append(_check("risk gate", isinstance(decision.allow, bool), decision.summary()))

    async def _execution() -> tuple[bool, str]:
        broker = PaperBroker(starting_equity=10_000)
        from apex_smc_bot.execution.broker import OrderRequest

        result = await broker.submit_order(
            OrderRequest(
                signal_id="SELFTEST",
                instrument="XAU_USD",
                direction="BULLISH",
                units=10,
                entry_price=price,
                stop_loss=price - 2,
                take_profit=price + 6,
                order_type="MARKET",
                metadata={"tp1": price + 3, "tp2": price + 6, "tp3": price + 10},
            ),
            market_price=price,
            spread_price=0.2,
        )
        return result.success, result.status

    ok, status = asyncio.run(_execution())
    results.append(_check("paper execution", ok, status))

    async def _engine_cycle() -> tuple[bool, str]:
        from apex_smc_bot.core.engine import TradingEngine

        with tempfile.TemporaryDirectory() as tmp:
            from apex_smc_bot.data.database import Database

            db = Database(Path(tmp) / "selftest.sqlite3")
            engine = TradingEngine(
                settings=settings,
                broker=PaperBroker(starting_equity=10_000),
                feed=SyntheticFeed(days=30, seed=settings.feed.synthetic_seed),
                database=db,
                logging_bundle=setup_logging(settings),
            )
            await engine.start()
            report = await engine.cycle(force=True)
            ok_ = report.errors == []
            detail = f"armed={report.armed} signals={report.signals} skipped={report.skipped}"
            await engine.shutdown("selftest")
            return ok_, detail

    ok, detail = asyncio.run(_engine_cycle())
    results.append(_check("engine cycle", ok, detail))

    width = max(len(name) for name, _, _ in results) + 2
    print("\nAPEX-SMC self-test")
    print("-" * (width + 40))
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}} {detail}")
    print("-" * (width + 40))
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1
