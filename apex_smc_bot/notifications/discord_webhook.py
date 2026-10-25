"""Discord webhook delivery for signal broadcasts."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from apex_smc_bot.config import NotificationConfig

logger = logging.getLogger("apex.notifications.discord")

EMBED_COLORS = {"BUY": 0x2ECC71, "BULLISH": 0x2ECC71, "SELL": 0xE74C3C, "BEARISH": 0xE74C3C}


class DiscordWebhook:
    """Posts embed messages to a Discord webhook URL."""

    name = "discord"

    def __init__(
        self,
        webhook_url: str,
        config: NotificationConfig | None = None,
        username: str = "APEX-SMC",
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.webhook_url = webhook_url
        self.config = config or NotificationConfig()
        self.username = username
        self.timeout = self.config.timeout_seconds
        self.max_retries = max(1, self.config.max_retries)
        self._session = session
        self._owns_session = session is None
        self._last_error: str | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> "DiscordWebhook | None":
        if not settings.notifications.discord_enabled:
            return None
        url = settings.secrets.discord_webhook_url
        if not url:
            logger.warning("Discord enabled but DISCORD_WEBHOOK_URL is missing")
            return None
        return cls(url, settings.notifications)

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self._session

    async def send_message(self, text: str, direction: str = "") -> bool:
        """Send plain text, split to Discord's 2000-character limit."""

        if not self.enabled or not text:
            return False
        session = await self._get_session()
        chunks = [text[i : i + 1900] for i in range(0, len(text), 1900)]
        ok = True
        for chunk in chunks:
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        self.webhook_url,
                        json={"content": chunk, "username": self.username},
                    ) as response:
                        if response.status in {200, 204}:
                            self._last_error = None
                            break
                        body = await response.text()
                        self._last_error = f"HTTP {response.status}: {body[:200]}"
                        if response.status == 429 or response.status >= 500:
                            retry_after = float(response.headers.get("Retry-After", 2))
                            await asyncio.sleep(min(30.0, retry_after))
                            continue
                        logger.warning("Discord rejected message: %s", self._last_error)
                        return False
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    self._last_error = str(exc)
                    logger.warning("Discord attempt %d failed: %s", attempt + 1, exc)
                    await asyncio.sleep(1.5 * (2 ** attempt))
            else:
                ok = False
        return ok

    async def send_embed(
        self,
        title: str,
        description: str,
        direction: str = "",
        fields: list[dict[str, Any]] | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        payload = {
            "username": self.username,
            "embeds": [
                {
                    "title": title[:256],
                    "description": description[:4000],
                    "color": EMBED_COLORS.get(direction.upper(), 0x3498DB),
                    "fields": (fields or [])[:25],
                    "footer": {"text": "APEX-SMC v1.0 · XAU/USD"},
                }
            ],
        }
        session = await self._get_session()
        try:
            async with session.post(self.webhook_url, json=payload) as response:
                if response.status in {200, 204}:
                    return True
                self._last_error = f"HTTP {response.status}"
                return False
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self._last_error = str(exc)
            return False

    async def close(self) -> None:
        if self._session is not None and not self._session.closed and self._owns_session:
            await self._session.close()
        self._session = None
