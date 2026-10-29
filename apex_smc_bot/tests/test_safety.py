"""Regression coverage for risk/approval bugs not covered by the original suite."""

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from apex_smc_bot.config import ConfigError, Settings
from apex_smc_bot.data.candle_manager import CandleManager
from apex_smc_bot.execution.broker import BrokerError, BrokerOutcomeUnknown, OrderResult
from apex_smc_bot.execution.lot_calculator import LotSizeError, calculate_position, calculate_position_size
from apex_smc_bot.execution.order_manager import OrderManager
from apex_smc_bot.execution.paper_broker import PaperBroker
from apex_smc_bot.signals.signal_validator import SignalValidator
from apex_smc_bot.tests.factories import make_signal

T0 = datetime(2026, 10, 7, 13, tzinfo=timezone.utc)


def semi_auto():
    settings = Settings()
    return replace(settings, execution=replace(settings.execution, paper_auto_approve=False))


def test_env_mapping_is_respected_and_secrets_never_appear_in_public_output(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-ambient-should-not-be-used")
    settings = Settings.load(tmp_path / "absent.yaml", env={"GROQ_API_KEY": "gsk-explicit-test", "APEX_AI_ENABLED": "true"})
    assert settings.secrets.groq_api_key == "gsk-explicit-test" and settings.ai.enabled
    assert "gsk-" not in repr(settings) + repr(settings.secrets) + json.dumps(settings.public_dict(), default=str)
    assert settings.public_dict()["secrets_configured"]["groq_configured"]
    assert not Settings.load(tmp_path / "absent.yaml", env={}).secrets.groq_api_key


@pytest.mark.parametrize("yaml", ["secrets:\n  groq_api_key: not-allowed", "ai:\n  enabled: 'false'",
                                 "risk:\n  max_spread_pips: .nan", "ai:\n  max_tokens: 32.5",
                                 "risk:\n  max_cost_to_stop_ratio: 2", "feed:\n  max_quote_age_seconds: 0"])
def test_unsafe_or_mistyped_configuration_is_rejected(tmp_path, yaml):
    path = tmp_path / "config.yaml"
    path.write_text(yaml)
    with pytest.raises(ConfigError):
        Settings.load(path, env={})


@pytest.mark.parametrize("config_switch,env_switch,allowed", [(True, "true", True), (True, "false", False),
                                                           (False, "true", False), (True, None, False)])
def test_live_environment_requires_both_independent_interlocks(tmp_path, config_switch, env_switch, allowed):
    path = tmp_path / "config.yaml"
    path.write_text(f"execution:\n  broker: oanda\n  oanda_environment: live\n  allow_live_trading: {str(config_switch).lower()}\n")
    env = {} if env_switch is None else {"APEX_ALLOW_LIVE": env_switch}
    if allowed:
        assert Settings.load(path, env=env).live_trading_confirmed
    else:
        with pytest.raises(ConfigError):
            Settings.load(path, env=env)


def test_minimum_lot_must_not_round_up_even_one_percent_over_budget():
    assert calculate_position_size(495, 1.0, 50) == 0
    assert calculate_position_size(500, 1.0, 50) == 0.01
    with pytest.raises(LotSizeError):
        calculate_position_size(495, 1.0, 50, risk_tolerance=1.1)
    for value in (float("nan"), float("inf")):
        with pytest.raises(LotSizeError):
            calculate_position_size(value, 1, 20)


def test_cost_reserve_lowers_size_without_exceeding_the_budget():
    ordinary = calculate_position(10_000, 1, 2400, 2397, Settings().risk)
    reserved = calculate_position(10_000, 1, 2400, 2397, Settings().risk, cost_reserve_pips=6)
    assert reserved.lot_size < ordinary.lot_size
    assert reserved.risk_amount_usd <= 100
    assert reserved.risk_amount_usd == pytest.approx(reserved.units * 3.6)


@pytest.mark.parametrize("field,value", [("units", 10000), ("units", float("nan")), ("risk_percent", -1),
                                        ("cost_reserve_pips", -1)])
def test_sizing_metadata_cannot_hide_real_exposure(field, value):
    signal = make_signal(timestamp=T0)
    setattr(signal.position_sizing, field, value)
    assert not SignalValidator(Settings()).validate(signal, now=T0, account_equity=10_000).valid


def test_all_configured_rr_minima_are_hard_gates():
    signal = make_signal(timestamp=T0, tp1=2396.4, tp2=2393.0)
    result = SignalValidator(Settings()).validate(signal, now=T0)
    assert not result.valid
    assert {"TP1_RR_TOO_LOW", "TP2_RR_INVALID"} <= {error.code for error in result.errors}


@pytest.mark.parametrize("bid,ask", [(float("nan"), 100), (100, float("inf")), (101, 100)])
def test_invalid_quotes_cannot_poison_the_cache(bid, ask):
    with pytest.raises(ValueError):
        CandleManager().set_quote(bid, ask)


@pytest.mark.asyncio
async def test_delayed_approval_rechecks_spread_and_current_equity():
    broker = PaperBroker()
    manager = OrderManager(semi_auto(), broker)
    signal = make_signal(timestamp=T0)
    assert (await manager.submit(signal, 2400, .2, T0)).status == "PENDING_APPROVAL"
    rejected = await manager.approve(signal.signal_id, 2400, 1.0, T0 + timedelta(minutes=10))
    assert not rejected.success and "SPREAD" in rejected.message
    assert not broker.get_open_positions()

    signal.signal_id += "-equity"
    assert (await manager.submit(signal, 2400, .2, T0)).success
    broker.balance = 500
    rejected = await manager.approve(signal.signal_id, 2400, .2, T0 + timedelta(minutes=10))
    assert not rejected.success


@pytest.mark.asyncio
async def test_delayed_approval_rechecks_reduced_risk_and_loss_cooldown():
    broker = PaperBroker()
    manager = OrderManager(semi_auto(), broker)
    signal = make_signal(timestamp=T0, lots=.25)  # valid below-budget lot
    assert (await manager.submit(signal, 2400, .2, T0)).success
    broker.balance = 9700
    rejected = await manager.approve(signal.signal_id, 2400, .2, T0 + timedelta(minutes=10))
    assert not rejected.success and "signal_risk_matches_gate" in rejected.message

    manager = OrderManager(semi_auto(), PaperBroker())
    signal.signal_id += "-loss"
    assert (await manager.submit(signal, 2400, .2, T0)).success
    manager.risk_engine.note_trade_closed(-50, T0)
    rejected = await manager.approve(signal.signal_id, 2400, .2, T0 + timedelta(minutes=10))
    assert not rejected.success and "drawdown_ok" in rejected.message


@pytest.mark.asyncio
async def test_simultaneous_orders_and_duplicate_approvals_cannot_exceed_capacity():
    settings = Settings()
    settings = replace(settings, risk=replace(settings.risk, max_open_trades=1))
    broker = PaperBroker()
    manager = OrderManager(settings, broker)
    a = make_signal(timestamp=T0)
    b = make_signal(timestamp=T0)
    b.signal_id += "-second"
    results = await asyncio.gather(manager.submit(a, 2390, .2, T0), manager.submit(b, 2390, .2, T0))
    assert sum(result.success for result in results) == 1
    assert await manager.get_open_trade_count() == 1

    broker = PaperBroker()
    manager = OrderManager(semi_auto(), broker)
    assert (await manager.submit(a, 2390, .2, T0)).success
    results = await asyncio.gather(manager.approve(a.signal_id, 2390, .2, T0), manager.approve(a.signal_id, 2390, .2, T0))
    assert sum(result.success for result in results) == 1


@pytest.mark.asyncio
async def test_async_pending_lookup_counts_entries_not_protective_orders():
    class Remote(PaperBroker):
        name = "oanda"

        async def get_pending_orders(self):
            return [{"id": "entry", "type": "LIMIT"}, {"id": "sl", "type": "STOP_LOSS"},
                    {"id": "tp", "type": "TAKE_PROFIT"}]

        async def cancel_order(self, order_id):
            assert order_id == "entry"
            return OrderResult(True, "CANCELLED", order_id=order_id)

    manager = OrderManager(Settings(), Remote())
    assert await manager.get_working_order_count() == 1
    assert await manager.cancel_pending_entries() == 1


@pytest.mark.asyncio
async def test_pending_lookup_failure_is_not_zero_exposure():
    broker = PaperBroker()
    broker.get_pending_orders = AsyncMock(side_effect=BrokerError("offline"))
    manager = OrderManager(Settings(), broker)
    result = await manager.submit(make_signal(timestamp=T0), 2400, .2, T0)
    assert not result.success and not broker.get_open_positions()


@pytest.mark.asyncio
async def test_quote_is_refreshed_again_after_ai_and_slippage_is_rechecked():
    broker = PaperBroker()
    quote = AsyncMock(side_effect=[(2400, .2), (2402, .2)])
    review = AsyncMock(return_value="")  # PASS does not bypass the deterministic checks
    signal = make_signal(timestamp=T0)
    signal.entry.type = "MARKET"
    manager = OrderManager(Settings(), broker, quote_provider=quote, review_guard=review)
    result = await manager.submit(signal, 2400, .2, T0)
    assert not result.success and "SLIPPAGE" in result.message
    assert quote.await_count == 2 and review.await_count == 1


@pytest.mark.asyncio
async def test_quote_failure_or_approval_callback_failure_never_uses_cached_price():
    broker = PaperBroker()
    manager = OrderManager(Settings(), broker, quote_provider=AsyncMock(side_effect=TimeoutError()))
    result = await manager.submit(make_signal(timestamp=T0), 2400, .2, T0)
    assert not result.success and not broker.get_open_positions()


@pytest.mark.asyncio
async def test_unknown_write_is_consumed_and_cannot_be_replayed_after_restart(database):
    broker = PaperBroker()
    broker.submit_order = AsyncMock(side_effect=BrokerOutcomeUnknown("may already be filled"))
    manager = OrderManager(Settings(), broker, database)
    signal = make_signal(timestamp=T0)
    result = await manager.submit(signal, 2400, .2, T0)
    assert result.status == "UNKNOWN"
    restarted = OrderManager(Settings(), broker, database)
    duplicate = await restarted.submit(signal, 2400, .2, T0)
    assert not duplicate.success and broker.submit_order.await_count == 1


def test_broker_protection_is_at_the_final_runner_not_tp2():
    signal = make_signal(timestamp=T0)
    request = OrderManager(Settings(), PaperBroker())._build_request(signal)
    assert request.take_profit == signal.take_profits.tp3.price


def test_equity_tracking_does_not_require_a_database():
    from apex_smc_bot.core.state_manager import StateManager
    state = StateManager()
    state.set_last_equity(9000)
    assert state.last_equity(10000) == 9000
    state.set_last_equity(0)
    assert state.last_equity(10000) == 0


@pytest.mark.asyncio
async def test_slow_preflight_cannot_route_an_expired_signal():
    clock = [T0]
    signal = make_signal(timestamp=T0)
    signal.expires_at = T0 + timedelta(seconds=10)

    async def slow_guard(*args):
        clock[0] += timedelta(seconds=20)
        return ""

    broker = PaperBroker()
    manager = OrderManager(Settings(), broker, execution_guard=slow_guard, clock=lambda: clock[0])
    result = await manager.submit(signal, 2400, .2, T0)
    assert not result.success and result.message == "SIGNAL_EXPIRED_DURING_PREFLIGHT"
    assert not broker.get_open_positions() and not broker.get_pending_orders()


def test_programmatic_string_booleans_cannot_arm_live_interlocks():
    settings = Settings()
    invalid = replace(settings, execution=replace(settings.execution, allow_live_trading="false"))
    with pytest.raises(ConfigError):
        invalid.validate()


def test_resting_orders_obey_the_execution_ttl_as_well_as_signal_expiry():
    settings = Settings()
    settings = replace(settings, execution=replace(settings.execution, pending_order_ttl_minutes=10))
    signal = make_signal(timestamp=T0)
    request = OrderManager(settings, PaperBroker())._build_request(signal, T0)
    assert request.expires_at == T0 + timedelta(minutes=10)
