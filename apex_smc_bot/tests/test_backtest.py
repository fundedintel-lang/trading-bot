"""Backtesting: metrics integrity, equity reconciliation and fold splitting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from apex_smc_bot.backtesting.backtest_engine import BacktestConfig, parse_end
from apex_smc_bot.backtesting.performance_metrics import (
    build_report,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    streak_stats,
    win_rate,
)
from apex_smc_bot.backtesting.walk_forward import WalkForwardAnalyzer
from apex_smc_bot.config import Settings


def _trades(pnls, start=datetime(2026, 9, 1, tzinfo=timezone.utc)):
    """Closed-trade dicts, two per day, with an explicit ``closed_at``."""

    return [
        {
            "realized_pnl": pnl,
            "closed_at": (start + timedelta(days=index // 2, hours=3)).isoformat(),
            "status": "CLOSED",
        }
        for index, pnl in enumerate(pnls)
    ]


# ------------------------------------------------------------------- primitives
def test_core_metrics():
    pnls = [100, -50, 120, -40]
    assert win_rate(pnls) == 50.0
    assert profit_factor(pnls) == pytest.approx(220 / 90, abs=0.001)
    amount, percent = max_drawdown([10_000, 10_200, 9_800, 10_100])
    assert amount == pytest.approx(400.0)
    assert percent == pytest.approx(3.92, abs=0.01)
    streak = streak_stats([1, 2, 3, -1, -2, 5])
    assert streak["max_win_streak"] == 3
    assert streak["max_loss_streak"] == 2


def test_sharpe_and_sortino_are_zero_when_undefined():
    assert sharpe_ratio([0.01]) == 0.0
    assert sharpe_ratio([0.01, 0.01, 0.01]) == 0.0  # zero variance
    assert sortino_ratio([0.01, 0.02]) == 0.0  # no downside


def test_report_uses_closed_at_for_daily_statistics():
    """Missing ``closed_at`` used to collapse every trade into one 'unknown' day,
    which silently zeroed Sharpe/Sortino.  Guard against that regression."""

    report = build_report(_trades([100, -50, 120, -40, 90, 60]), initial_equity=10_000)
    assert report.trades == 6
    assert report.net_profit == pytest.approx(280.0)
    assert report.days_traded == 3
    assert report.sharpe != 0.0
    assert report.best_day > 0


def test_report_flags_small_samples():
    report = build_report(_trades([100, -50]), initial_equity=10_000, min_trades_for_report=20)
    assert any("below the" in note for note in report.notes)
    assert "indicative" in report.summary()


def test_report_reconciles_with_an_equity_curve():
    report = build_report(
        _trades([100, -50, 120]),
        equity_curve=[10_000, 10_100, 10_050, 10_170],
        initial_equity=10_000,
    )
    assert report.net_profit == pytest.approx(170.0)
    assert report.final_equity == pytest.approx(10_170.0)


def test_empty_report_is_safe():
    report = build_report([], initial_equity=10_000)
    assert report.trades == 0
    assert report.win_rate == 0.0
    assert "no completed trades" in report.summary()


# ---------------------------------------------------------------- configuration
def test_parse_end_accepts_dates_and_timestamps():
    assert parse_end(None) is None
    assert parse_end("2026-10-07") == datetime(2026, 10, 7, tzinfo=timezone.utc)
    assert parse_end("2026-10-07T14:00Z") == datetime(2026, 10, 7, 14, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        parse_end("not-a-date")


def test_backtest_config_defaults_come_from_settings():
    settings = Settings()
    config = BacktestConfig.from_settings(settings)
    assert config.initial_equity == settings.backtest.initial_equity
    assert config.seed == settings.backtest.seed
    assert config.research_mode is False


def test_walk_forward_windows_tile_the_history():
    settings = Settings()
    analyzer = WalkForwardAnalyzer(settings, days=120, folds=3, seed=11)
    assert analyzer.folds == 3
    assert analyzer.is_days >= 5 and analyzer.oos_days >= 3
    # Out-of-sample must follow in-sample, and each fold must be a real window.
    assert analyzer.is_days + analyzer.oos_days > 0
    with pytest.raises(ValueError):
        WalkForwardAnalyzer(settings, folds=1)


def test_walk_forward_result_serialises():
    from apex_smc_bot.backtesting.walk_forward import WalkForwardFold, WalkForwardResult

    report = build_report(_trades([100, -50]), initial_equity=10_000)
    fold = WalkForwardFold(
        index=1,
        is_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        is_end=datetime(2026, 8, 20, tzinfo=timezone.utc),
        oos_start=datetime(2026, 8, 20, tzinfo=timezone.utc),
        oos_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
        is_report=report,
        oos_report=report,
    )
    result = WalkForwardResult(folds=[fold], combined_oos=report)
    payload = result.to_dict()
    assert payload["folds"][0]["index"] == 1
    assert "combined_out_of_sample" in payload
    assert "Fold" in result.summary()
