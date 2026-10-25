"""Plotly chart builders for the Streamlit dashboard and HTML reports.

Everything is pure: pass candles/zones in, get a ``plotly.graph_objects.Figure``
back.  Plotly is imported lazily so the trading engine never depends on it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from apex_smc_bot.data.candle_manager import Candle

BUY_COLOR = "#26a69a"
SELL_COLOR = "#ef5350"
OB_BULL_COLOR = "rgba(38, 166, 154, 0.22)"
OB_BEAR_COLOR = "rgba(239, 83, 80, 0.22)"
FVG_COLOR = "rgba(255, 193, 7, 0.18)"


def _go():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("plotly is required for charts (pip install plotly)") from exc
    return go


def _add_zones(fig: Any, blocks: Sequence[Any] | None, fvgs: Sequence[Any] | None) -> None:
    """Draw order blocks and fair value gaps as shaded rectangles."""

    if blocks:
        for block in blocks[:20]:
            bullish = getattr(block, "kind", "BULLISH") == "BULLISH"
            fig.add_hrect(
                y0=getattr(block, "low", 0.0),
                y1=getattr(block, "high", 0.0),
                fillcolor=OB_BULL_COLOR if bullish else OB_BEAR_COLOR,
                line_width=0,
                layer="below",
                annotation_text=f"OB {block.quality}/5" if hasattr(block, "quality") else "OB",
                annotation_position="top left",
                annotation_font_size=9,
            )
    if fvgs:
        for gap in fvgs[:20]:
            fig.add_hrect(
                y0=getattr(gap, "bottom", 0.0),
                y1=getattr(gap, "top", 0.0),
                fillcolor=FVG_COLOR,
                line_width=0,
                layer="below",
            )


def _add_levels(fig: Any, levels: dict[str, float] | None) -> None:
    if not levels:
        return
    styles = {
        "entry": ("#2196f3", "dot", "Entry"),
        "stop_loss": ("#e53935", "dash", "SL"),
        "tp1": ("#43a047", "dash", "TP1"),
        "tp2": ("#2e7d32", "dash", "TP2"),
        "tp3": ("#1b5e20", "dash", "TP3"),
    }
    for key, (color, dash, label) in styles.items():
        price = levels.get(key)
        if price:
            fig.add_hline(
                y=price,
                line_dash=dash,
                line_color=color,
                annotation_text=f"{label} {price:.2f}",
                annotation_position="right",
                annotation_font_size=10,
            )


def candlestick_chart(
    candles: Sequence[Candle],
    title: str = "XAU/USD",
    order_blocks: Sequence[Any] | None = None,
    fvgs: Sequence[Any] | None = None,
    levels: dict[str, float] | None = None,
    markers: Sequence[dict[str, Any]] | None = None,
    show_volume: bool = True,
    height: int = 620,
    max_candles: int = 220,
) -> Any:
    """OHLC candlestick chart with OB/FVG overlays and trade levels."""

    go = _go()
    from plotly.subplots import make_subplots

    data = list(candles)[-max_candles:]
    if not data:
        figure = go.Figure()
        figure.update_layout(title=f"{title} — no data", height=height)
        return figure

    rows = 2 if show_volume else 1
    figure = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22] if show_volume else [1.0],
    )
    figure.add_trace(
        go.Candlestick(
            x=[c.timestamp for c in data],
            open=[c.open for c in data],
            high=[c.high for c in data],
            low=[c.low for c in data],
            close=[c.close for c in data],
            name="Price",
            increasing_line_color=BUY_COLOR,
            decreasing_line_color=SELL_COLOR,
        ),
        row=1,
        col=1,
    )
    if show_volume:
        colors = [BUY_COLOR if c.close >= c.open else SELL_COLOR for c in data]
        figure.add_trace(
            go.Bar(
                x=[c.timestamp for c in data],
                y=[c.volume for c in data],
                marker_color=colors,
                name="Volume",
                opacity=0.55,
            ),
            row=2,
            col=1,
        )
    _add_zones(figure, order_blocks, fvgs)
    _add_levels(figure, levels)

    if markers:
        for marker in markers:
            figure.add_trace(
                go.Scatter(
                    x=[marker["timestamp"]],
                    y=[marker["price"]],
                    mode="markers+text",
                    marker={
                        "symbol": marker.get("symbol", "circle"),
                        "size": marker.get("size", 11),
                        "color": marker.get("color", "#ff9800"),
                    },
                    text=[marker.get("text", "")],
                    textposition="top center",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    figure.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend={"orientation": "h", "y": 1.02},
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    return figure


def equity_curve_chart(
    curve: Sequence[dict[str, Any]] | Sequence[float],
    title: str = "Equity curve",
    initial_equity: float | None = None,
    height: int = 360,
) -> Any:
    """Equity over time (accepts DB rows ``{timestamp, equity}`` or raw floats)."""

    go = _go()
    figure = go.Figure()
    if not curve:
        figure.update_layout(title=f"{title} — no data", height=height, template="plotly_dark")
        return figure

    if isinstance(curve[0], dict):
        x = [row.get("timestamp") for row in curve]
        y = [float(row.get("equity", 0.0)) for row in curve]
    else:
        x = list(range(len(curve)))
        y = [float(value) for value in curve]

    baseline = initial_equity if initial_equity is not None else (y[0] if y else 0.0)
    line_color = BUY_COLOR if (y[-1] >= baseline) else SELL_COLOR
    figure.add_trace(
        go.Scatter(x=x, y=y, mode="lines", line={"color": line_color, "width": 2}, name="Equity")
    )
    figure.add_hline(y=baseline, line_dash="dot", line_color="rgba(255,255,255,0.4)")
    figure.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        yaxis_title="Equity (USD)",
    )
    return figure


def drawdown_chart(curve: Sequence[float], height: int = 260, title: str = "Drawdown (%)") -> Any:
    """Underwater (drawdown) chart from an equity series."""

    go = _go()
    figure = go.Figure()
    peak = 0.0
    drawdowns = []
    for value in curve:
        peak = max(peak, value)
        drawdowns.append(((value - peak) / peak * 100.0) if peak else 0.0)
    figure.add_trace(
        go.Scatter(
            x=list(range(len(drawdowns))),
            y=drawdowns,
            fill="tozeroy",
            line={"color": SELL_COLOR, "width": 1},
            name="Drawdown",
        )
    )
    figure.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        yaxis_title="%",
    )
    return figure


def score_histogram(scores: Sequence[float], height: int = 280, title: str = "Confluence scores") -> Any:
    """Distribution of confluence scores (0-10)."""

    go = _go()
    figure = go.Figure(
        go.Histogram(x=list(scores), nbinsx=10, marker_color="#42a5f5", name="Score")
    )
    figure.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        xaxis_title="Score /10",
        yaxis_title="Signals",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return figure


def render_html_report(
    candles: Sequence[Candle],
    output_path: str,
    title: str = "APEX-SMC report",
    **kwargs: Any,
) -> str:
    """Write a standalone HTML chart (used by ``main.py`` and CI artifacts)."""

    figure = candlestick_chart(candles, title=title, **kwargs)
    html = figure.to_html(include_plotlyjs="cdn", full_html=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return output_path
