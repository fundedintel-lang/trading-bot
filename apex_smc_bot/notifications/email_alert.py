"""SMTP email alerts (optional, disabled by default)."""

from __future__ import annotations

import asyncio
import logging
import smtplib
from email.message import EmailMessage
from typing import Any

from apex_smc_bot.config import NotificationConfig

logger = logging.getLogger("apex.notifications.email")


class EmailAlerter:
    """Sends plain-text alerts through an SMTP server with STARTTLS."""

    name = "email"

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: list[str],
        config: NotificationConfig | None = None,
        use_tls: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.from_addr = from_addr or username
        self.to_addrs = [addr for addr in to_addrs if addr]
        self.config = config or NotificationConfig()
        self.max_retries = max(1, self.config.max_retries)
        self.use_tls = use_tls
        self._last_error: str | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> "EmailAlerter | None":
        if not settings.notifications.email_enabled:
            return None
        secrets = settings.secrets
        missing = [
            name
            for name in ("smtp_host", "smtp_user", "smtp_password", "email_to")
            if not getattr(secrets, name, None)
        ]
        if missing:
            logger.warning("Email enabled but %s missing", ", ".join(missing))
            return None
        recipients = [addr.strip() for addr in str(secrets.email_to).split(",") if addr.strip()]
        return cls(
            host=secrets.smtp_host,
            port=int(secrets.smtp_port or 587),
            username=secrets.smtp_user,
            password=secrets.smtp_password,
            from_addr=secrets.smtp_user,
            to_addrs=recipients,
            config=settings.notifications,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.host and self.to_addrs)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _send_sync(self, subject: str, body: str) -> None:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.from_addr
        message["To"] = ", ".join(self.to_addrs)
        message.set_content(body)
        with smtplib.SMTP(self.host, self.port, timeout=20) as smtp:
            if self.use_tls:
                smtp.starttls()
            if self.username:
                smtp.login(self.username, self.password)
            smtp.send_message(message)

    async def send_message(self, text: str, subject: str = "APEX-SMC alert") -> bool:
        """Non-blocking wrapper around the blocking SMTP client."""

        if not self.enabled or not text:
            return False
        for attempt in range(self.max_retries):
            try:
                await asyncio.to_thread(self._send_sync, subject, text)
                self._last_error = None
                return True
            except Exception as exc:  # noqa: BLE001
                self._last_error = str(exc)
                logger.warning("Email attempt %d failed: %s", attempt + 1, exc)
                await asyncio.sleep(1.5 * (2 ** attempt))
        return False

    async def close(self) -> None:
        return None
