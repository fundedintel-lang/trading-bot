"""APEX-SMC Streamlit dashboard.

Run with::

    python main.py dashboard
    # or
    streamlit run apex_smc_bot/dashboard/streamlit_app.py --server.port 8501

The dashboard is read-only: it renders signals, trades, equity, open positions and
risk state from SQLite.  It never places orders — that stays in the engine/Telegram
approval flow.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from apex_smc_bot.backtesting.performance_metrics import build_report
from apex_smc_bot.config import Settings
from apex_smc_bot.dashboard.chart_renderer import (
    candlestick_chart,
    drawdown_chart,
    equity_curve_chart,
    score_histogram,
)
from apex_smc_bot.data.database import Database

st.set_page_config(page_title="APEX-SMC · XAU/USD", page_icon="🏆", layout="wide")

PIP = 0.10


# --------------------------------------------------------------------------- data
@st.cache_resource
def get_settings() -> Settings:
    return Settings.load()


@st.cache_resource
def get_database() -> Database:
    settings = get_settings()
    path = settings.database_path
    if not path.exists():
        return Database(path, enabled=False)
    return Database(path)


@st.cache_data(ttl=20)
def load_signals(limit: int = 200, status: str | None = None) -> pd.DataFrame:
    rows = get_database().get_signals(limit=limit, status=status)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    return frame.sort_values("created_at", ascending=False)


@st.cache_data(ttl=20)
def load_trades(limit: int = 500) -> pd.DataFrame:
    rows = get_database().get_trades(limit=limit)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    for column in ("opened_at", "closed_at"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame.sort_values("opened_at", ascending=False)


@st.cache_data(ttl=20)
def load_equity(limit: int = 2000) -> pd.DataFrame:
    rows = get_database().get_equity_curve(limit=limit)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame


@st.cache_data(ttl=20)
def load_candles(timeframe: str = "M15", limit: int = 300) -> pd.DataFrame:
    rows = get_database().get_candles(get_settings().bot.instrument, timeframe, limit=limit)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.sort_values("timestamp")


def load_runtime_state() -> dict[str, Any]:
    state = {}
    for key in ("drawdown_state", "trading_paused", "last_equity", "engine_counters"):
        try:
            state[key] = get_database().get_runtime_state(key)
        except Exception:  # noqa: BLE001
            state[key] = None
    return state


# --------------------------------------------------------------------------- ui
def render_header() -> None:
    settings = get_settings()
    state = load_runtime_state()
    equity = state.get("last_equity") or settings.risk.starting_equity
    drawdown = state.get("drawdown_state") or {}
    paused = bool(state.get("trading_paused"))

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:16px">
          <h1 style="margin:0">🏆 APEX-SMC</h1>
          <span style="padding:4px 10px;border-radius:12px;background:#1e293b;font-size:13px">
            {settings.bot.instrument} · {settings.bot.execution_mode} · {settings.execution.broker}
          </span>
          <span style="padding:4px 10px;border-radius:12px;background:{'#7f1d1d' if paused else '#14532d'};font-size:13px">
            {'PAUSED' if paused else 'ACTIVE'}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    cols[0].metric("Equity", f"${equity:,.2f}")
    daily_dd = float(drawdown.get("daily_drawdown_percent") or 0.0)
    cols[1].metric("Daily DD", f"{daily_dd:.2f}%", delta=f"-{daily_dd:.2f}%" if daily_dd else None, delta_color="inverse")
    limit = float(settings.risk.max_daily_loss_percent)
    cols[2].metric("DD limit", f"{limit:.1f}%")
    trades = load_trades()
    cols[3].metric("Open trades", int((trades["status"] == "OPEN").sum()) if not trades.empty else 0)
    cols[4].metric("Risk / trade", f"{settings.risk.max_risk_per_trade_percent:.2f}%")
    if daily_dd >= limit * 0.6:
        st.warning(f"Daily drawdown at {daily_dd:.2f}% — circuit breaker trips at {limit:.1f}%.")


def render_chart() -> None:
    st.subheader("Market structure")
    controls = st.columns([2, 1, 1])
    timeframe = controls[0].selectbox("Timeframe", ["M15", "H1", "M5", "H4"], index=0)
    candles = load_candles(timeframe)
    if candles.empty:
        st.info("No candles stored yet — run the engine to populate the database.")
        return

    from apex_smc_bot.data.candle_manager import Candle

    has_volume = "volume" in candles.columns
    has_spread = "spread" in candles.columns
    bars = [
        Candle(
            timestamp=pd.Timestamp(row.timestamp).to_pydatetime(),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(getattr(row, "volume", 0.0) or 0.0) if has_volume else 0.0,
            spread=float(getattr(row, "spread", 0.0) or 0.0) if has_spread else 0.0,
        )
        for row in candles.itertuples()
    ]
    levels: dict[str, float] = {}
    open_trades = load_trades()
    if not open_trades.empty:
        active = open_trades[open_trades["status"] == "OPEN"]
        if not active.empty:
            row = active.iloc[0]
            levels = {
                "entry": float(row["entry_price"]),
                "stop_loss": float(row["stop_loss"]),
            }
            take_profit = row["take_profit"] if "take_profit" in active.columns else None
            if take_profit is not None and pd.notna(take_profit):
                levels["tp3"] = float(take_profit)
    figure = candlestick_chart(bars, title=f"XAU/USD · {timeframe}", levels=levels)
    st.plotly_chart(figure, use_container_width=True)


def render_signals() -> None:
    st.subheader("Signals")
    signals = load_signals()
    if signals.empty:
        st.info("No signals recorded yet.")
        return
    if "confluence_score" in signals:
        st.plotly_chart(
            score_histogram(signals["confluence_score"].dropna().tolist()),
            use_container_width=True,
        )
    display = signals.copy()
    if "payload" in display:
        display = display.drop(columns=["payload"])
    st.dataframe(display, use_container_width=True, height=320)

    st.markdown("#### Signal detail")
    options = signals.get("signal_id", pd.Series(dtype=str)).tolist()
    if options:
        chosen = st.selectbox("Signal", options)
        raw = signals.loc[signals["signal_id"] == chosen, "payload"].iloc[0]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {"raw": raw}
        st.json(raw)


def render_performance() -> None:
    st.subheader("Performance")
    trades = load_trades()
    equity = load_equity()

    closed = trades[trades["status"] == "CLOSED"] if not trades.empty and "status" in trades else pd.DataFrame()
    initial = get_settings().risk.starting_equity
    report = build_report(
        closed.to_dict("records") if not closed.empty else [],
        equity_curve=equity["equity"].tolist() if not equity.empty else None,
        initial_equity=initial,
    )
    cols = st.columns(6)
    cols[0].metric("Trades", report.trades)
    cols[1].metric("Win rate", f"{report.win_rate:.1f}%")
    pf = report.profit_factor
    cols[2].metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}")
    cols[3].metric("Net P/L", f"${report.net_profit:+,.2f}")
    cols[4].metric("Max DD", f"{report.max_drawdown_percent:.2f}%")
    cols[5].metric("Sharpe", f"{report.sharpe:.2f}")

    if not equity.empty:
        st.plotly_chart(
            equity_curve_chart(equity[["timestamp", "equity"]].to_dict("records"), initial_equity=initial),
            use_container_width=True,
        )
        st.plotly_chart(drawdown_chart(equity["equity"].tolist()), use_container_width=True)
    if not closed.empty:
        st.dataframe(closed, use_container_width=True, height=260)
    for note in report.notes:
        st.caption(note)


def render_positions() -> None:
    st.subheader("Open positions")
    trades = load_trades()
    if trades.empty:
        st.info("No trades recorded.")
        return
    active = trades[trades["status"] == "OPEN"] if "status" in trades else pd.DataFrame()
    if active.empty:
        st.success("Flat.")
    else:
        st.dataframe(active, use_container_width=True, height=220)
    st.markdown("#### Recent closed")
    closed = trades[trades["status"] == "CLOSED"] if "status" in trades else pd.DataFrame()
    st.dataframe(closed.head(30), use_container_width=True, height=260)


def render_risk() -> None:
    st.subheader("Risk state")
    settings = get_settings()
    state = load_runtime_state()
    drawdown = state.get("drawdown_state") or {}
    counters = state.get("engine_counters") or {}

    left, right = st.columns(2)
    with left:
        st.markdown("**Drawdown monitor**")
        st.json(drawdown or {"status": "no data"})
        st.markdown("**Limits**")
        st.json(
            {
                "max_risk_per_trade_percent": settings.risk.max_risk_per_trade_percent,
                "max_daily_loss_percent": settings.risk.max_daily_loss_percent,
                "max_open_trades": settings.risk.max_open_trades,
                "max_spread_pips": settings.risk.max_spread_pips,
                "max_sl_pips": settings.risk.max_sl_pips,
                "min_confluence_score": settings.strategy.min_confluence_score,
                "min_five_m_checks": settings.strategy.min_fivem_checks,
            }
        )
    with right:
        st.markdown("**Engine counters**")
        st.json(counters or {"status": "no data"})
        st.markdown("**News blackout**")
        st.caption(
            f"±{settings.news.block_minutes_before} min around high-impact USD events "
            f"({', '.join(settings.news.high_impact_keywords[:6])}…)"
        )
        news = get_database().get_events(event_type="NEWS_BLOCK", limit=5)
        st.json(news or [])


def render_ai() -> None:
    settings = get_settings()
    st.subheader("AI second opinion")
    st.caption("The strategy and risk limits remain authoritative. Model confidence is NOT a win probability.")
    st.json({"enabled": settings.ai.enabled, "provider": settings.ai.provider,
             "model": settings.ai.model_name, "mode": settings.ai.mode,
             "configured": bool(settings.secrets.groq_api_key if settings.ai.provider == "groq" else settings.secrets.xai_api_key),
             "daily_request_budget": settings.ai.max_calls_per_day,
             "usage": get_database().get_runtime_state("ai_daily_budget", {})})
    reviews = get_database().get_events("AI_REVIEW", limit=30)
    if not reviews:
        st.info("No AI reviews yet. Requests happen only after a deterministic signal passes its gates; replay never calls AI.")
    else:
        st.dataframe([{"timestamp": row["timestamp"], **row["payload"]} for row in reviews],
                     use_container_width=True)


def main() -> None:
    settings = get_settings()
    render_header()
    tabs = st.tabs(["Chart", "Signals", "Performance", "Positions", "Risk", "AI Review", "Config"])
    with tabs[0]:
        render_chart()
    with tabs[1]:
        render_signals()
    with tabs[2]:
        render_performance()
    with tabs[3]:
        render_positions()
    with tabs[4]:
        render_risk()
    with tabs[5]:
        render_ai()
    with tabs[6]:
        public = settings.public_dict()
        st.json(public)
        st.caption(f"Database: {settings.database_path} · logs: {settings.logs_dir}")
    st.caption(
        f"Read-only dashboard · refreshed {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC"
        f" · data TTL 20s"
    )


if __name__ == "__main__":
    main()
