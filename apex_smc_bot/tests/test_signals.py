"""Signals: validation rules, formatting channels and setup de-duplication."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta

import pytest

from apex_smc_bot.config import Settings
from apex_smc_bot.signals import signal_formatter as fmt
from apex_smc_bot.signals.models import BEARISH, BULLISH, PendingSetup
from apex_smc_bot.signals.signal_validator import SignalValidator
from apex_smc_bot.tests.factories import make_signal


@pytest.fixture(scope="module")
def validator() -> SignalValidator:
    return SignalValidator(Settings())


def test_reference_signal_is_valid(validator, trade_time):
    signal = make_signal(timestamp=trade_time)
    result = validator.validate(signal, market_price=signal.entry.price, now=trade_time)
    assert result.valid, [error.message for error in result.errors]
    assert result.checks


def test_stop_on_wrong_side_is_rejected(validator, trade_time):
    signal = make_signal(direction=BEARISH, entry=2400.0, stop_loss=2397.0, timestamp=trade_time)
    result = validator.validate(signal, market_price=2400.0, now=trade_time)
    assert not result.valid
    assert any("STOP" in error.code for error in result.errors)


def test_take_profit_on_wrong_side_is_rejected(validator, trade_time):
    signal = make_signal(direction=BEARISH, entry=2400.0, stop_loss=2403.0, timestamp=trade_time)
    # TP1 above the entry of a short: still "ordered" (2404 > 2391 > 2385) but
    # on the losing side, so it must be rejected.
    signal.take_profits.tp1.price = 2404.0
    result = validator.validate(signal, market_price=2400.0, now=trade_time)
    assert not result.valid
    assert any(error.code == "TP1_WRONG_SIDE" for error in result.errors)


def test_stale_rr_metadata_is_rejected(validator, trade_time):
    signal = make_signal(timestamp=trade_time)
    signal.risk_reward.rr_to_tp1 = 9.0  # metadata no longer matches the levels
    result = validator.validate(signal, market_price=2400.0, now=trade_time)
    assert not result.valid
    assert any(error.code == "RR_MISMATCH" for error in result.errors)


def test_risk_over_two_percent_is_rejected(validator, trade_time):
    signal = make_signal(risk_percent=2.5, timestamp=trade_time, equity=10_000.0)
    signal.position_sizing.risk_percent = 2.5
    signal.position_sizing.risk_amount_usd = 250.0
    result = validator.validate(signal, market_price=2400.0, now=trade_time, account_equity=10_000.0)
    assert not result.valid
    assert any("RISK" in error.code for error in result.errors)


def test_expired_signal_is_rejected(validator, trade_time):
    signal = make_signal(timestamp=trade_time)
    later = trade_time + timedelta(hours=6)
    result = validator.validate(signal, market_price=2400.0, now=later)
    assert not result.valid


def test_open_trade_limit_blocks(validator, trade_time):
    signal = make_signal(timestamp=trade_time)
    result = validator.validate(
        signal, market_price=2400.0, now=trade_time, open_trade_count=5
    )
    assert not result.valid


def test_signal_json_round_trips(validator, trade_time):
    signal = make_signal(timestamp=trade_time)
    payload = json.loads(fmt.format_signal_json(signal))
    assert payload["instrument"] == "XAU/USD"
    assert payload["direction"] == "SELL"
    assert payload["entry"]["price"] == pytest.approx(signal.entry.price, abs=0.01)
    assert "stop_loss" in payload and "take_profits" in payload
    assert payload["plan_a"] and payload["plan_b"]
    assert payload["position_sizing"]["lot_size"] > 0
    # The JSON must survive a second serialisation (no datetimes leak through).
    json.dumps(payload)


def test_text_and_console_channels_cover_the_plan(trade_time):
    signal = make_signal(timestamp=trade_time)
    text = fmt.format_signal_text(signal)
    console = fmt.format_console_signal(signal)
    telegram = fmt.format_telegram_signal(signal)
    discord = fmt.format_discord_signal(signal)
    for rendered in (text, console, telegram, discord):
        assert signal.signal_id in rendered
        assert "2397" in rendered or "2403" in rendered  # stop/TP level present
    assert len(text) > 100


def test_rejection_formatting_is_two_args():
    rendered = fmt.format_rejection("HTF_NOT_ALIGNED", {"direction": "BEARISH", "score": 3})
    assert "HTF_NOT_ALIGNED" in rendered


def test_fingerprint_is_geography_based():
    base = make_signal()
    pending_a = PendingSetup(
        direction=BEARISH,
        entry_price=base.entry.price,
        zone_low=base.entry.zone_low,
        zone_high=base.entry.zone_high,
        stop_loss=base.stop_loss.price,
        tp1=base.take_profits.tp1.price,
        tp2=base.take_profits.tp2.price,
        tp3=base.take_profits.tp3.price if base.take_profits.tp3 else None,
        created_at=base.timestamp,
        expires_at=base.timestamp + timedelta(hours=4),
        session_name="NY_KILL_ZONE",
        zone_basis="test",
        signal_bias=BEARISH,
    )
    pending_b = replace(pending_a, created_at=pending_a.created_at + timedelta(minutes=15))
    assert pending_a.fingerprint == pending_b.fingerprint
    moved = replace(pending_a, zone_low=pending_a.zone_low - 2.0)
    assert moved.fingerprint != pending_a.fingerprint


def test_bullish_signal_geometry():
    signal = make_signal(direction=BULLISH, entry=2400.0, stop_loss=2397.0)
    assert signal.is_long and signal.side == "LONG"
    assert signal.take_profits.tp1.price > signal.entry.price
    assert signal.stop_loss.price < signal.entry.price


@pytest.mark.asyncio
async def test_signal_reaches_console_log_and_database(tmp_path, trade_time):
    """The dispatch contract: one signal → console, log file and DB in one call."""

    from apex_smc_bot.config import Settings
    from apex_smc_bot.data.database import Database
    from apex_smc_bot.logging_setup import setup_logging
    from apex_smc_bot.notifications.notifier import Notifier

    settings = Settings()
    database = Database(tmp_path / "dispatch.sqlite3")
    bundle = setup_logging(settings, console=False)
    notifier = Notifier(settings, database=database, logging_bundle=bundle, banner=False)
    signal = make_signal(timestamp=trade_time)

    report = await notifier.notify_signal(signal)
    await notifier.close()

    assert report.channels["console"] is True
    assert report.channels["log"] is True
    assert report.channels["database"] is True

    rows = database.get_signals(limit=5)
    assert rows and rows[0]["signal_id"] == signal.signal_id
    payload = rows[0]["payload"]
    assert payload["direction"] == "SELL"
    assert payload["confluence_score"] == pytest.approx(signal.confluence_score, abs=0.1)
    database.close()
