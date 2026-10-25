"""Fan-out delivery of signals, trade events and alerts.

Every channel is best-effort: one failing channel must never stop the others or
crash the engine.  Results are returned so the engine can log delivery status.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from apex_smc_bot.config import Settings
from apex_smc_bot.data.candle_manager import to_utc, utcnow
from apex_smc_bot.notifications.discord_webhook import DiscordWebhook
from apex_smc_bot.notifications.email_alert import EmailAlerter
from apex_smc_bot.notifications.telegram_bot import TelegramBot
from apex_smc_bot.signals.models import Signal
from apex_smc_bot.signals.signal_formatter import (
    format_console_signal,
    format_discord_signal,
    format_signal_json,
    format_signal_text,
    format_telegram_signal,
)

logger = logging.getLogger("apex.notifications")


@dataclass
class DeliveryReport:
    channels: dict[str, bool] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utcnow)

    @property
    def ok(self) -> bool:
        return any(self.channels.values())

    @property
    def failed(self) -> list[str]:
        return [name for name, delivered in self.channels.items() if not delivered]

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat(),
        }


class Notifier:
    """Routes notifications to console, log, database, Telegram, Discord, email."""

    def __init__(
        self,
        settings: Settings,
        telegram: TelegramBot | None = None,
        discord: DiscordWebhook | None = None,
        email: EmailAlerter | None = None,
        database: Any | None = None,
        logging_bundle: Any | None = None,
        banner: bool = True,
    ) -> None:
        self.settings = settings
        self.telegram = telegram
        self.discord = discord
        self.email = email
        self.database = database
        self.logging_bundle = logging_bundle
        self.banner = banner
        self._startup_banner_shown = False
        self._last_console: str = ""

    # ------------------------------------------------------------------ startup
    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        database: Any | None = None,
        logging_bundle: Any | None = None,
    ) -> "Notifier":
        return cls(
            settings=settings,
            telegram=TelegramBot.from_settings(settings),
            discord=DiscordWebhook.from_settings(settings),
            email=EmailAlerter.from_settings(settings),
            database=database,
            logging_bundle=logging_bundle,
        )

    @property
    def active_channels(self) -> list[str]:
        channels = []
        if self.settings.notifications.console_enabled:
            channels.append("console")
        if self.telegram is not None and self.telegram.enabled:
            channels.append("telegram")
        if self.discord is not None and self.discord.enabled:
            channels.append("discord")
        if self.email is not None and self.email.enabled:
            channels.append("email")
        if self.database is not None:
            channels.append("database")
        return channels

    # ------------------------------------------------------------------ signals
    async def notify_signal(self, signal: Signal, status: str = "GENERATED") -> DeliveryReport:
        """Deliver a signal to every configured channel plus console/log/DB."""

        report = DeliveryReport()
        payload = signal.to_dict()
        text = format_signal_text(signal)

        if self.settings.notifications.console_enabled:
            try:
                print(format_console_signal(signal), flush=True)
                if self.banner:
                    print(text, flush=True)
                report.channels["console"] = True
            except Exception as exc:  # noqa: BLE001
                report.errors["console"] = str(exc)
                logger.exception("Console delivery failed")

        if self.logging_bundle is not None:
            try:
                from apex_smc_bot.logging_setup import log_signal

                log_signal(self.logging_bundle, payload)
                report.channels["log"] = True
            except Exception as exc:  # noqa: BLE001
                report.errors["log"] = str(exc)
                logger.exception("Signal log delivery failed")

        if self.database is not None:
            try:
                self.database.record_signal(
                    signal_id=signal.signal_id,
                    timestamp=signal.timestamp,
                    instrument=signal.instrument,
                    direction=signal.direction,
                    payload=payload,
                    confluence_score=signal.confluence_score,
                    status=status,
                )
                report.channels["database"] = True
            except Exception as exc:  # noqa: BLE001
                report.errors["database"] = str(exc)
                logger.exception("Signal DB write failed")

        if self.telegram is not None and self.telegram.enabled:
            try:
                report.channels["telegram"] = await self.telegram.send_message(format_telegram_signal(signal))
                if not report.channels["telegram"]:
                    report.errors["telegram"] = self.telegram.last_error or "send failed"
            except Exception as exc:  # noqa: BLE001
                report.channels["telegram"] = False
                report.errors["telegram"] = str(exc)
                logger.exception("Telegram delivery failed")

        if self.discord is not None and self.discord.enabled:
            try:
                report.channels["discord"] = await self.discord.send_message(
                    format_discord_signal(signal), signal.direction
                )
                if not report.channels["discord"]:
                    report.errors["discord"] = self.discord.last_error or "send failed"
            except Exception as exc:  # noqa: BLE001
                report.channels["discord"] = False
                report.errors["discord"] = str(exc)
                logger.exception("Discord delivery failed")

        if self.email is not None and self.email.enabled:
            try:
                report.channels["email"] = await self.email.send_message(
                    text, subject=f"APEX-SMC {signal.side} {signal.instrument} — {signal.confluence_score}/10"
                )
                if not report.channels["email"]:
                    report.errors["email"] = self.email.last_error or "send failed"
            except Exception as exc:  # noqa: BLE001
                report.channels["email"] = False
                report.errors["email"] = str(exc)
                logger.exception("Email delivery failed")

        logger.info(
            "Signal %s delivered to %s",
            signal.signal_id,
            ", ".join(name for name, ok in report.channels.items() if ok) or "nothing",
        )
        return report

    # ------------------------------------------------------------------ events
    async def notify_trade_event(self, event: str, payload: dict[str, Any] | None = None, text: str | None = None) -> DeliveryReport:
        """Broadcast a trade/engine event (fills, TP hits, halts)."""

        report = DeliveryReport()
        payload = payload or {}
        body = text or f"{event}: " + ", ".join(f"{k}={v}" for k, v in payload.items() if v is not None)

        try:
            if self.settings.notifications.console_enabled:
                print(f"[{to_utc(utcnow()).strftime('%H:%M:%S')}] {body}", flush=True)
                report.channels["console"] = True
        except Exception:  # noqa: BLE001
            report.channels["console"] = False

        if self.logging_bundle is not None:
            try:
                from apex_smc_bot.logging_setup import log_trade_event

                log_trade_event(self.logging_bundle, event, **payload)
                report.channels["log"] = True
            except Exception:  # noqa: BLE001
                report.channels["log"] = False

        if self.database is not None:
            try:
                self.database.record_event(event, payload)
                report.channels["database"] = True
            except Exception:  # noqa: BLE001
                report.channels["database"] = False

        for name, client in (("telegram", self.telegram), ("discord", self.discord)):
            if client is None or not getattr(client, "enabled", False):
                continue
            try:
                if name == "telegram":
                    delivered = await client.send_message(body)  # type: ignore[union-attr]
                else:
                    delivered = await client.send_message(body)  # type: ignore[union-attr]
                report.channels[name] = bool(delivered)
            except Exception as exc:  # noqa: BLE001
                report.channels[name] = False
                report.errors[name] = str(exc)
        return report

    async def alert(self, message: str, level: str = "INFO") -> DeliveryReport:
        """Operational alert (startup, shutdown, circuit breaker, errors)."""

        return await self.notify_trade_event(f"{level}_ALERT", {"message": message}, text=f"[{level}] {message}")

    async def notify_rejection(self, reason: str, context: dict[str, Any] | None = None) -> None:
        """Rejections are logged locally only — Telegram would be too noisy."""

        logger.info("REJECTED %s %s", reason, context or "")
        if self.database is not None:
            try:
                self.database.record_event("SIGNAL_REJECTED", {"reason": reason, **(context or {})})
            except Exception:  # noqa: BLE001
                pass

    async def notify_startup(self, summary: dict[str, Any]) -> DeliveryReport:
        if self._startup_banner_shown:
            return DeliveryReport()
        self._startup_banner_shown = True
        lines = [
            "APEX-SMC engine online",
            f"mode: {summary.get('mode')}",
            f"broker: {summary.get('broker')}",
            f"instrument: {summary.get('instrument')}",
            f"channels: {', '.join(self.active_channels) or 'console only'}",
        ]
        return await self.notify_trade_event("ENGINE_STARTED", summary, text="\n".join(lines))

    async def close(self) -> None:
        for client in (self.telegram, self.discord, self.email):
            closer = getattr(client, "close", None)
            if closer is None:
                continue
            try:
                await closer()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to close notification client")


def summarise_signals(signals: Sequence[Signal]) -> str:
    """Human-readable digest used by the dashboard/CLI."""

    if not signals:
        return "No signals."
    return "\n".join(
        f"{signal.timestamp:%Y-%m-%d %H:%M} {signal.signal_id} {signal.side} "
        f"score {signal.confluence_score:.1f} entry {signal.entry.price:.2f}"
        for signal in signals
    )
