"""Mocked OANDA writes/reads and engine fail-closed integration checks."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from apex_smc_bot.config import Settings
from apex_smc_bot.core.engine import CycleReport, TradingEngine
from apex_smc_bot.data.candle_manager import Candle, floor_to_timeframe
from apex_smc_bot.data.feed_base import FeedUnavailable
from apex_smc_bot.data.feeds import build_feed
from apex_smc_bot.data.synthetic import SyntheticFeed
from apex_smc_bot.execution.broker import BrokerError, BrokerOutcomeUnknown, OrderRequest
from apex_smc_bot.execution.oanda_executor import OANDAExecutionEngine
from apex_smc_bot.execution.paper_broker import PaperBroker
from apex_smc_bot.tests.factories import make_session

T0 = datetime(2026, 10, 7, 13, tzinfo=timezone.utc)


class Response:
    def __init__(self, status, data=None):
        self.status, self.data = status, data or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def json(self, **kwargs):
        return self.data


class Session:
    closed = False

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def executor(session):
    return OANDAExecutionEngine("fake-account", "fake-not-a-secret", session=session,
                                max_retries=3, backoff_seconds=0)


@pytest.mark.asyncio
async def test_transient_reads_retry_but_permanent_rejections_do_not():
    session = Session([Response(500), Response(429), Response(200, {"account": {}})])
    assert await executor(session)._request("GET", "/read") == {"account": {}}
    assert len(session.calls) == 3
    session = Session([Response(401)])
    with pytest.raises(BrokerError):
        await executor(session)._request("GET", "/read")
    assert len(session.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["POST", "PUT"])
@pytest.mark.parametrize("failure", [Response(500), TimeoutError()])
async def test_ambiguous_writes_never_replay(method, failure):
    session = Session([failure])
    with pytest.raises(BrokerOutcomeUnknown):
        await executor(session)._request(method, "/orders", {"units": "10"})
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_order_payload_has_native_stop_runner_target_fill_bound_and_trade_identity(database):
    session = Session([Response(200, {"orderCreateTransaction": {"id": "order-1"}})])
    broker = OANDAExecutionEngine("fake-account", "fake-token", session=session, database=database)
    request = OrderRequest("signal-1", "XAU_USD", "BULLISH", 10, 2400, 2397, 2415, "MARKET",
                           metadata={"tp1": 2404.5, "tp2": 2409, "tp3": 2415, "max_slippage_price": .2})
    result = await broker.submit_order(request, 2400, .2)
    assert result.success
    order = session.calls[0][2]["json"]["order"]
    assert order["stopLossOnFill"]["price"] == "2397.00"
    assert order["takeProfitOnFill"]["price"] == "2415.00"
    assert order["priceBound"] == "2400.30"
    assert order["tradeClientExtensions"]["id"] == "signal-1"
    assert database.get_runtime_state("oanda_trade_plans")["signal-1"]["tp1"] == 2404.5


@pytest.mark.asyncio
async def test_cancelled_fok_is_not_misreported_as_a_resting_order():
    broker = executor(Session([]))
    request = OrderRequest("signal-1", "XAU_USD", "BULLISH", 10, 2400, 2397)
    result = broker._parse_order_response({"orderCreateTransaction": {"id": "1"},
                                           "orderCancelTransaction": {"id": "2", "reason": "FOK"}}, request)
    assert not result.success and result.status == "CANCELLED"


def test_synthetic_fallback_is_for_paper_only():
    settings = Settings()
    settings = replace(settings, execution=replace(settings.execution, broker="oanda"),
                       feed=replace(settings.feed, primary="oanda", secondary="", tertiary=""))
    with pytest.raises(FeedUnavailable):
        build_feed(settings)
    with pytest.raises(FeedUnavailable):
        build_feed(settings, override="synthetic")


def test_engine_refuses_to_drive_a_real_broker_with_simulated_prices():
    broker = type("Remote", (), {"name": "oanda"})()
    with pytest.raises(ValueError, match="Simulated"):
        TradingEngine(Settings(), broker, SyntheticFeed())


def seed_history(engine, now=T0):
    for tf, minutes in (("M5", 5), ("M15", 15), ("H1", 60), ("H4", 240)):
        last_open = floor_to_timeframe(now, tf) - timedelta(minutes=minutes)
        engine.candles.ingest(tf, [Candle(last_open - timedelta(minutes=minutes * i), 100, 101, 99, 100)
                                   for i in range(50)])
    engine.candles.set_quote(99.9, 100.1, now)


def offline_settings():
    settings = Settings()
    return replace(settings, news=replace(settings.news, enabled=False),
                   notifications=replace(settings.notifications, console_enabled=False))


@pytest.mark.asyncio
async def test_management_cycle_fetches_new_quote_instead_of_reusing_the_old_mark():
    feed = type("Feed", (), {"name": "test", "fetch_price": AsyncMock(return_value=(96.9, 97.1)),
                             "close": AsyncMock()})()
    broker = PaperBroker()
    await broker.mark_price(100, .2, T0)
    request = OrderRequest("test", "XAU_USD", "BULLISH", 10, 100, 99, order_type="MARKET")
    await broker.submit_order(request, 100, .2)
    engine = TradingEngine(offline_settings(), broker, feed)
    seed_history(engine)
    report = await engine.management_cycle(T0 + timedelta(seconds=30))
    assert not report.errors
    assert feed.fetch_price.await_count == 1
    assert not broker.get_open_positions()
    assert broker.get_history()[0].exit_price < 99
    await engine.shutdown("test")


@pytest.mark.asyncio
async def test_freshness_gates_block_stale_quote_and_candles_but_not_a_valid_snapshot():
    engine = TradingEngine(offline_settings(), PaperBroker())
    seed_history(engine)
    assert engine._data_gate(T0) == ""
    assert "QUOTE" in engine._data_gate(T0 + timedelta(seconds=31))
    assert engine._data_gate(T0 + timedelta(hours=1)) == "STALE_CANDLES"
    await engine.shutdown("test")


@pytest.mark.asyncio
async def test_news_exceptions_fail_closed_when_requested():
    settings = offline_settings()
    settings = replace(settings, news=replace(settings.news, enabled=True, fail_closed=True))
    engine = TradingEngine(settings, PaperBroker())
    seed_history(engine)
    engine.news.refresh = AsyncMock(side_effect=TimeoutError("offline"))
    report = CycleReport(1, T0)
    assert await engine._hard_gates(T0, make_session(), report) == "NEWS_UNAVAILABLE"
    await engine.shutdown("test")


@pytest.mark.asyncio
async def test_breaker_cancels_resting_entries_before_they_can_be_filled():
    broker = PaperBroker()
    await broker.mark_price(100, .2, T0)
    request = OrderRequest("pending", "XAU_USD", "BULLISH", 10, 95, 93, order_type="LIMIT")
    assert (await broker.submit_order(request, 100, .2)).status == "PENDING"
    engine = TradingEngine(offline_settings(), broker)
    seed_history(engine)
    engine.risk.update_equity(10000, T0)
    decision = engine.risk.update_equity(9400, T0 + timedelta(seconds=1))
    await engine._handle_circuit_breaker(decision, CycleReport(1, T0))
    assert not broker.get_pending_orders()
    await broker.mark_price(94, .2, T0 + timedelta(minutes=5))
    assert not broker.get_open_positions()
    await engine.shutdown("test")


@pytest.mark.asyncio
async def test_native_broker_quote_takes_precedence_over_an_assumed_feed_spread():
    class Remote(PaperBroker):
        name = "oanda"
        get_quote = AsyncMock(return_value=(100.0, 100.3))

    feed = type("Feed", (), {"name": "auxiliary", "fetch_price": AsyncMock(side_effect=AssertionError()),
                             "close": AsyncMock()})()
    broker = Remote()
    engine = TradingEngine(offline_settings(), broker, feed)
    await engine._refresh_quote(T0)
    assert engine.candles.latest_spread == .3
    assert engine.candles.quote_received_at == T0
    feed.fetch_price.assert_not_called()
    await engine.shutdown("test")


@pytest.mark.asyncio
async def test_broker_market_closed_quote_is_not_executable():
    session = Session([Response(200, {"prices": [{"tradeable": False, "bids": [], "asks": []}]})])
    with pytest.raises(BrokerError, match="tradeable"):
        await executor(session).get_quote("XAU_USD")


def test_stale_calendar_cannot_be_rejuvenated_by_restart(database):
    from apex_smc_bot.risk.news_filter import NewsFilter
    settings = Settings()
    filt = NewsFilter(settings.news, database)
    filt.set_events([{"title": "CPI", "date": T0.isoformat(), "impact": "High", "currency": "USD"}], fetched_at=T0)
    restarted = NewsFilter(settings.news, database)
    assert restarted.cache_age_minutes(T0 + timedelta(hours=2)) == 120
    check = restarted.check(T0 + timedelta(hours=2))
    assert not check.safe and check.source == "unavailable"


def test_fail_open_news_still_respects_a_known_blackout():
    from apex_smc_bot.risk.news_filter import NewsFilter
    config = replace(Settings().news, fail_closed=False)
    filt = NewsFilter(config)
    filt.set_events([{"title": "CPI", "date": T0.isoformat(), "impact": "High", "currency": "USD"}],
                    fetched_at=T0 - timedelta(hours=2))
    assert not filt.check(T0).safe
