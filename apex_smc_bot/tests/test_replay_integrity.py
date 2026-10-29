"""No future candles, honest windows/costs, and conservative intrabar execution."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from apex_smc_bot.backtesting.backtest_engine import BacktestConfig, BacktestEngine
from apex_smc_bot.backtesting.performance_metrics import build_report, expectancy_interval, max_drawdown, win_rate_interval
from apex_smc_bot.config import Secrets, Settings
from apex_smc_bot.core.engine import CycleReport, TradingEngine
from apex_smc_bot.data.candle_manager import Candle, CandleManager, TIMEFRAME_SECONDS, resample
from apex_smc_bot.data.replay_feed import ReplayFeed
from apex_smc_bot.data.synthetic import SyntheticFeed, load_candles_from_csv
from apex_smc_bot.execution.broker import BrokerOutcomeUnknown, OrderRequest, OrderResult
from apex_smc_bot.execution.paper_broker import PaperBroker
from apex_smc_bot.execution.position_manager import PositionManager

T0 = datetime(2026, 10, 7, 8, tzinfo=timezone.utc)


def dataset(count=100):
    bars = [Candle(T0 + timedelta(minutes=5 * i), 100, 101, 99, 100, 10) for i in range(count)]
    return {"M5": bars, **{tf: resample(bars, tf) for tf in ("M15", "H1", "H4")}}


def long_request(**overrides):
    fields = dict(signal_id="long", instrument="XAU_USD", direction="BULLISH", units=10,
                  entry_price=100, stop_loss=99, take_profit=105, order_type="MARKET",
                  metadata={"tp1": 101.5, "tp2": 103, "tp3": 105})
    fields.update(overrides)
    return OrderRequest(**fields)


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", [0, 1, 2, 11, 47, 48, 80])
async def test_replay_higher_timeframes_are_visible_only_after_close(cursor):
    feed = ReplayFeed(dataset())
    feed.cursor = cursor
    as_of = feed.now + timedelta(minutes=5)
    for timeframe, seconds in TIMEFRAME_SECONDS.items():
        if timeframe not in ("M5", "M15", "H1", "H4"):
            continue
        bars = await feed.fetch_candles("XAU_USD", timeframe, 500)
        assert all(bar.timestamp + timedelta(seconds=seconds) <= as_of for bar in bars)
        if cursor == 0 and timeframe != "M5":
            assert not bars
    if cursor == 47:
        assert (await feed.fetch_candles("XAU_USD", "H4", 500))[-1].timestamp == T0


@pytest.mark.asyncio
async def test_changing_future_price_cannot_change_the_current_htf_analysis_input():
    original = dataset()
    altered = dataset()
    # At 08:05 the 08:00-12:00 H4 bar has NOT closed. Its eventual high
    # must not influence an 08:05 decision, regardless of how extreme it is.
    altered["M5"][20] = replace(altered["M5"][20], high=10000, close=9999)
    for tf in ("M15", "H1", "H4"):
        altered[tf] = resample(altered["M5"], tf)
    a, b = ReplayFeed(original), ReplayFeed(altered)
    for tf in ("M15", "H1", "H4"):
        assert await a.fetch_candles("XAU_USD", tf, 500) == await b.fetch_candles("XAU_USD", tf, 500)


@pytest.mark.asyncio
async def test_synthetic_htf_bars_never_change_after_publication():
    clock = [datetime(2026, 10, 7, 13, 7, tzinfo=timezone.utc)]
    feed = SyntheticFeed(days=5, clock=lambda: clock[0])
    before = await feed.fetch_candles("XAU_USD", "H4", 500)
    assert before[-1].timestamp + timedelta(hours=4) <= clock[0]
    clock[0] += timedelta(hours=4)
    after = {bar.timestamp: bar for bar in await feed.fetch_candles("XAU_USD", "H4", 500)}
    assert all(after[bar.timestamp] == bar for bar in before)


def test_completed_candle_cache_filters_forming_and_future_bars():
    manager = CandleManager()
    manager.seed(dataset())
    assert not manager.get_candles("H4", as_of=T0 + timedelta(hours=1))
    assert len(manager.get_candles("M15", as_of=T0 + timedelta(minutes=15))) == 1


def test_duplicate_csv_timestamps_are_rejected(tmp_path):
    path = tmp_path / "duplicate.csv"
    row = "2026-10-07T08:00Z,100,101,99,100\n"
    path.write_text("timestamp,open,high,low,close\n" + row * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_candles_from_csv(path)


@pytest.mark.asyncio
async def test_both_stop_and_targets_inside_a_bar_is_a_stop_not_a_win():
    broker = PaperBroker()
    await broker.mark_price(100, 0, T0)
    await broker.submit_order(long_request(), 100, 0)
    manager = PositionManager(Settings(), broker)
    bar = Candle(T0, 100, 106, 98, 104)
    events = await manager.on_price(bar.close, 0, [], T0 + timedelta(minutes=5), bar=bar)
    assert [event.kind for event in events] == ["STOP_HIT"]
    assert broker.get_history()[0].realized_pnl == -10


@pytest.mark.asyncio
async def test_limit_touch_uses_ask_and_cannot_claim_an_earlier_intrabar_target():
    broker = PaperBroker()
    await broker.mark_price(101, .2, T0)
    order = long_request(order_type="LIMIT")
    assert (await broker.submit_order(order, 101, .2)).status == "PENDING"
    # Mid touches 100 but ask=100.1; a buy limit at 100 is NOT filled.
    await broker.mark_price(100, .2, T0 + timedelta(minutes=5))
    assert not broker.get_open_positions()
    # A subsequent candle reaches the ask limit; its high might precede the fill.
    bar = Candle(T0 + timedelta(minutes=5), 101, 106, 99.5, 100.5)
    manager = PositionManager(Settings(), broker)
    events = await manager.on_price(100.5, .2, [], T0 + timedelta(minutes=10), bar=bar)
    assert broker.get_open_positions() and not events
    assert not broker.get_open_positions()[0].tp1_hit


@pytest.mark.asyncio
async def test_round_trip_and_partials_charge_entry_and_exit_commissions():
    broker = PaperBroker(slippage_price=.05, commission_per_lot=3.5)
    await broker.mark_price(100, .2, T0)
    fill = await broker.submit_order(long_request(units=100), 100, .2)
    await broker.partial_close(fill.trade_id, 50)
    await broker.close_position(fill.trade_id)
    history = broker.get_history()[0]
    # Entry 100.15, exit 99.85, 100 units, $7 round-trip commission.
    assert history.realized_pnl == pytest.approx(-37)
    assert history.metadata["commission_paid"] == 7
    assert broker.balance == pytest.approx(10000 + history.realized_pnl)


@pytest.mark.asyncio
async def test_failed_close_does_not_emit_a_fictitious_closed_event_or_erase_state():
    broker = PaperBroker()
    await broker.mark_price(100, 0, T0)
    fill = await broker.submit_order(long_request(), 100, 0)
    broker.close_position = AsyncMock(return_value=OrderResult(False, "REJECTED", message="offline"))
    risk = type("Risk", (), {"note_trade_closed": lambda *args: pytest.fail("not closed")})()
    manager = PositionManager(Settings(), broker, risk_engine=risk)
    assert not await manager.on_price(98, 0, [], T0 + timedelta(minutes=5))
    assert broker.get_open_positions()
    assert fill.trade_id in manager.managed_state()
    assert await manager.force_close(fill.trade_id, "MANUAL", 98, T0 + timedelta(minutes=10)) is None
    assert fill.trade_id in manager.managed_state()


@pytest.mark.asyncio
async def test_management_state_survives_restart_and_includes_entry_commission(database):
    broker = PaperBroker(commission_per_lot=10)
    await broker.mark_price(100, 0, T0)
    fill = await broker.submit_order(long_request(units=40), 100, 0)
    settings = Settings()
    manager = PositionManager(settings, broker, database)
    await manager.on_price(101.5, 0, [], T0 + timedelta(minutes=5))
    remaining = broker.get_position(fill.trade_id).units
    broker.get_position(fill.trade_id).tp1_hit = False  # emulate a fresh remote snapshot
    restarted = PositionManager(settings, broker, database)
    events = await restarted.on_price(101.6, 0, [], T0 + timedelta(minutes=6))
    assert not any(event.kind == "TP1_HIT" for event in events)
    assert broker.get_position(fill.trade_id).units == remaining
    assert restarted.managed_state()[fill.trade_id]["realized"] == broker.get_position(fill.trade_id).realized_pnl


@pytest.mark.asyncio
async def test_unknown_partial_close_is_never_automatically_retried(database):
    class Remote(PaperBroker):
        name = "oanda"

    broker = Remote()
    await broker.mark_price(100, 0, T0)
    await broker.submit_order(long_request(), 100, 0)
    broker.partial_close = AsyncMock(side_effect=BrokerOutcomeUnknown("timeout after write"))
    manager = PositionManager(Settings(), broker, database)
    await manager.on_price(102, 0, [], T0 + timedelta(minutes=5))
    assert manager.has_uncertain_operations
    restarted = PositionManager(Settings(), broker, database)
    await restarted.on_price(102, 0, [], T0 + timedelta(minutes=6))
    assert broker.partial_close.await_count == 1 and restarted.has_uncertain_operations


def test_backtesting_forces_paper_auto_execution_and_never_uses_ai_keys():
    settings = Settings()
    settings = replace(settings, ai=replace(settings.ai, enabled=True, mode="veto"),
                       secrets=Secrets(groq_api_key="gsk-fake-never-send"),
                       execution=replace(settings.execution, broker="oanda", oanda_environment="live",
                                         allow_live_trading=True), live_trading_confirmed=True)
    run = BacktestEngine(settings)._settings_for_run()
    run.validate()
    assert run.execution.broker == "paper" and run.bot.execution_mode == "FULL_AUTO"
    assert run.execution.oanda_environment == "practice" and not run.execution.allow_live_trading
    assert not run.ai.enabled and not run.secrets.groq_api_key and not run.news.enabled


@pytest.mark.asyncio
async def test_backtest_windows_use_actual_m5_duration_and_do_not_overlap(tmp_path):
    settings = Settings()
    settings = replace(settings, logging=replace(settings.logging, directory=str(tmp_path / "logs")))
    data = dataset(12)
    split = T0 + timedelta(minutes=30)
    a = BacktestConfig(days=1, start=T0, end=split)
    b = BacktestConfig(days=1, start=split, end=T0 + timedelta(hours=1))
    first = await BacktestEngine(settings, a, dataset=data).run()
    second = await BacktestEngine(settings, b, dataset=data).run()
    assert first.cycles == second.cycles == 6
    assert first.end == second.start == split
    assert first.ok and second.ok
    assert first.assumptions["ai_enabled"] is False
    assert any("SYNTHETIC" in note for note in first.report.notes)


@pytest.mark.asyncio
async def test_runtime_errors_cannot_be_reported_as_a_successful_backtest(monkeypatch, tmp_path):
    async def broken_cycle(self, now=None, force=False):
        return CycleReport(1, now, errors=["test-error"])

    monkeypatch.setattr(TradingEngine, "cycle", broken_cycle)
    settings = Settings()
    settings = replace(settings, logging=replace(settings.logging, directory=str(tmp_path / "logs")))
    result = await BacktestEngine(settings, BacktestConfig(days=1), dataset=dataset(3)).run()
    assert not result.ok and result.cycle_errors == 3
    assert "INVALID RUN" in result.report.summary()


def test_research_mode_cannot_relax_real_data_evaluation():
    with pytest.raises(ValueError, match="synthetic"):
        BacktestConfig(data_csv="real.csv", research_mode=True).validate()
    with pytest.raises(ValueError):
        BacktestConfig(days=0).validate()


def test_percent_drawdown_is_not_assumed_to_peak_with_dollar_drawdown():
    # $80 at 100 is an 80% DD, whereas the later $300 loss is only 30%.
    assert max_drawdown([100, 20, 1000, 700]) == (300, 80)


def test_uncertainty_metrics_are_reproducible_and_do_not_claim_live_profitability():
    pnls = [100, -100] * 20
    assert expectancy_interval(pnls) == expectancy_interval(pnls)
    low, high = expectancy_interval(pnls)
    assert low < 0 < high
    low, high = win_rate_interval(20, 40)
    assert low < 50 < high
    report = build_report([{"pnl": value} for value in pnls])
    assert report.evidence == "inconclusive"
    assert build_report([{"pnl": 10}]).evidence == "insufficient_sample"


@pytest.mark.asyncio
async def test_unchanged_history_is_not_written_twice_to_sqlite():
    db = type("DB", (), {"upsert_candles": lambda self, instrument, tf, bars: self.batches.append(list(bars)),
                          "batches": []})()
    feed = ReplayFeed(dataset())
    manager = CandleManager(feed=feed, database=db)
    await manager.refresh("M5")
    await manager.refresh("M5")
    assert len(db.batches) == 1 and len(db.batches[0]) == 1
    feed.cursor = 1
    await manager.refresh("M5")
    assert len(db.batches) == 2 and len(db.batches[1]) == 1


def test_csv_rejects_misaligned_m5_times_that_could_leak_aggregate_future_prices(tmp_path):
    path = tmp_path / "misaligned.csv"
    path.write_text("timestamp,open,high,low,close\n2026-10-07T08:03Z,100,101,99,100\n")
    with pytest.raises(ValueError, match="M5 bar-open"):
        load_candles_from_csv(path)


@pytest.mark.asyncio
async def test_trade_audit_updates_remaining_units_after_a_partial(database):
    broker = PaperBroker()
    await broker.mark_price(100, 0, T0)
    fill = await broker.submit_order(long_request(units=40), 100, 0)
    manager = PositionManager(Settings(), broker, database)
    await manager.on_price(100, 0, [], T0 + timedelta(minutes=1))
    await manager.on_price(101.5, 0, [], T0 + timedelta(minutes=5))
    rows = database.get_trades(status="OPEN")
    assert len(rows) == 1 and rows[0]["trade_id"] == fill.trade_id
    assert rows[0]["units"] == 20
