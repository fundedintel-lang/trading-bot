"""Human-readable + JSON signal rendering for console, Telegram and Discord."""

from __future__ import annotations

import json
from typing import Any

from apex_smc_bot.signals.models import Signal

MAX_TELEGRAM_CHARS = 3900


def _money(value: float) -> str:
    return f"{value:,.2f}"


def _rr(value: float) -> str:
    return f"1:{value:.2f}"


def format_telegram_signal(signal: Signal) -> str:
    """Plain-text (no parse_mode) signal card safe for Telegram/Discord."""

    data = signal.to_dict()
    entry = data["entry"]
    sl = data["stop_loss"]
    tps = data["take_profits"]
    rr = data["risk_reward"]
    sizing = data["position_sizing"]
    analysis = data.get("analysis") or {}
    plan_b = data.get("plan_b") or {}
    arrow = "🟢 BUY" if signal.is_long else "🔴 SELL"

    tp3_line = ""
    if "tp3" in tps:
        tp3_line = f"✅ TP3: {tps['tp3']['price']:.2f} ({tps['tp3']['distance_pips']:.0f} pips, trail)\n"

    reasons = [
        analysis.get("four_h_bias", ""),
        analysis.get("one_h_bias", ""),
        analysis.get("fifteen_m_setup", ""),
        analysis.get("five_m_trigger", ""),
    ]
    reason_block = "\n".join(f"• {line}" for line in reasons if line)

    message = f"""🏆 APEX-SMC INSTITUTIONAL SIGNAL 🏆
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 Instrument: {data['instrument']}
📌 Direction: {arrow}
📌 Signal Type: {data['signal_type']}
📌 Confluence: {data['confluence_score']}/10 ({data['confidence_percent']}% confidence)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 ENTRY: {entry['price']:.2f} ({entry['type']})
🛑 STOP LOSS: {sl['price']:.2f} ({sl['distance_pips']:.0f} pips)
✅ TP1: {tps['tp1']['price']:.2f} ({tps['tp1']['distance_pips']:.0f} pips) — close {tps['tp1']['size_to_close']}
✅ TP2: {tps['tp2']['price']:.2f} ({tps['tp2']['distance_pips']:.0f} pips) — close {tps['tp2']['size_to_close']}
{tp3_line}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 R:R → TP1: {rr['rr_to_tp1']}
📊 R:R → TP2: {rr['rr_to_tp2']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ SESSION: {data['session']['name']} ({data['session']['time_utc']} UTC)
💼 SIZE: {sizing['lot_size']} lots ({sizing['units']:.0f} oz) risking ${_money(sizing['risk_amount_usd'])} ({sizing['risk_percent']}%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📖 REASON:
{reason_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚫 INVALIDATION: {plan_b['invalidation']:.2f}
⚠️ PLAN B: {plan_b['what_happens_if_sl_hit']}
🧭 NEXT ZONE: {plan_b['next_zone_description']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signal ID: {data['signal_id']}
Expires: {data['expires_at']}
"""
    if len(message) > MAX_TELEGRAM_CHARS:
        message = message[: MAX_TELEGRAM_CHARS - 20] + "\n… (truncated)"
    return message.strip()


def format_discord_signal(signal: Signal) -> str:
    """Compact Discord card (same content, slightly tighter)."""

    data = signal.to_dict()
    tps = data["take_profits"]
    analysis = data.get("analysis") or {}
    plan_b = data.get("plan_b") or {}
    arrow = "🟢 BUY" if signal.is_long else "🔴 SELL"
    return (
        f"**🏆 APEX-SMC SIGNAL — {arrow} {data['instrument']}**\n"
        f"Score **{data['confluence_score']}/10** · {data['confidence_percent']}% confidence · "
        f"{data['session']['name']}\n"
        f"Entry **{data['entry']['price']:.2f}** · SL **{data['stop_loss']['price']:.2f}** "
        f"({data['stop_loss']['distance_pips']:.0f} pips)\n"
        f"TP1 **{tps['tp1']['price']:.2f}** ({tps['tp1']['distance_pips']:.0f} pips) · "
        f"TP2 **{tps['tp2']['price']:.2f}** ({tps['tp2']['distance_pips']:.0f} pips)\n"
        f"R:R {data['risk_reward']['rr_to_tp1']} / {data['risk_reward']['rr_to_tp2']} · "
        f"{data['position_sizing']['lot_size']} lots · risk ${_money(data['position_sizing']['risk_amount_usd'])}\n"
        f"_{analysis.get('fifteen_m_setup', 'setup')}_\n"
        f"Invalidation **{float(plan_b.get('invalidation', 0.0)):.2f}** · ID `{data['signal_id']}`"
    )


def format_console_signal(signal: Signal) -> str:
    """Single-screen summary for operators."""

    data = signal.to_dict()
    side = "BUY " if signal.is_long else "SELL"
    return (
        f"{data['signal_id']} | {side} | entry {data['entry']['price']:.2f} | "
        f"SL {data['stop_loss']['price']:.2f} ({data['stop_loss']['distance_pips']:.0f}p) | "
        f"TP1 {data['take_profits']['tp1']['price']:.2f} | TP2 {data['take_profits']['tp2']['price']:.2f} | "
        f"score {data['confluence_score']}/10 | {data['position_sizing']['lot_size']} lots | "
        f"{data['session']['name']}"
    )


def format_signal_json(signal: Signal, indent: int | None = 2) -> str:
    return json.dumps(signal.to_dict(), indent=indent, default=str, ensure_ascii=False)


def format_signal_text(signal: Signal) -> str:
    """Full multi-line report used for email and audit logs."""

    data = signal.to_dict()
    # Every field below is optional in practice (signals can be reconstructed
    # from the database after an upgrade), so rendering must never KeyError.
    analysis = data.get("analysis") or {}
    lines = [
        "=" * 64,
        f"APEX-SMC SIGNAL {data['signal_id']}",
        "=" * 64,
        f"Time          : {data['timestamp']}",
        f"Instrument    : {data['instrument']}",
        f"Direction     : {data['direction']} ({data['signal_type']})",
        f"Entry         : {data['entry']['price']:.2f} [{data['entry']['zone']}]",
        f"Stop loss     : {data['stop_loss']['price']:.2f} ({data['stop_loss']['distance_pips']:.1f} pips)",
        f"TP1           : {data['take_profits']['tp1']['price']:.2f} "
        f"({data['take_profits']['tp1']['distance_pips']:.1f} pips)",
        f"TP2           : {data['take_profits']['tp2']['price']:.2f} "
        f"({data['take_profits']['tp2']['distance_pips']:.1f} pips)",
        f"Risk          : ${data['position_sizing']['risk_amount_usd']:.2f} "
        f"({data['position_sizing']['risk_percent']}% of {data['position_sizing']['account_equity']:.2f})",
        f"Size          : {data['position_sizing']['lot_size']} lots",
        f"Confluence    : {data['confluence_score']}/10 ({data['confidence_percent']}%)",
        f"Session       : {data['session']['name']} {data['session']['time_utc']} UTC",
        "",
        "Analysis",
        "-" * 64,
        f"4H bias       : {analysis.get('four_h_bias', 'n/a')}",
        f"1H bias       : {analysis.get('one_h_bias', 'n/a')}",
        f"15M setup     : {analysis.get('fifteen_m_setup', 'n/a')}",
        f"5M trigger    : {analysis.get('five_m_trigger', 'n/a')}",
        f"Sweep         : {analysis.get('sweep_detail', 'n/a')}",
        f"Order block   : {analysis.get('ob_detail', 'n/a')}",
        f"FVG           : {analysis.get('fvg_detail', 'n/a')}",
        f"Premium/disc. : {analysis.get('premium_discount', 'n/a')}",
        f"OTE           : {analysis.get('ote', 'n/a')}",
        f"Smart money   : {analysis.get('smart_money_assessment', 'n/a')}",
        "",
        "Plan A",
        "-" * 64,
    ]
    lines.extend(f"  • {item}" for item in data["plan_a"]["required"])
    lines.append(f"  Management: {data['plan_a']['management']}")
    lines.extend(
        [
            "",
            "Plan B",
            "-" * 64,
            f"  {data['plan_b']['scenario']}",
            f"  Invalidation: {data['plan_b']['invalidation']:.2f}",
            f"  {data['plan_b']['next_zone_description']}",
            f"  {data['plan_b']['possible_flip']}",
            "",
            "Do NOT enter if:",
            "-" * 64,
        ]
    )
    lines.extend(f"  ✗ {item}" for item in data["do_not_enter_if"])
    lines.append("=" * 64)
    return "\n".join(lines)


def format_rejection(reason: str, context: dict[str, Any] | None = None) -> str:
    """Compact one-line rejection log for the engine loop."""

    if context:
        details = " ".join(f"{key}={value}" for key, value in context.items())
        return f"REJECTED: {reason} | {details}"
    return f"REJECTED: {reason}"
