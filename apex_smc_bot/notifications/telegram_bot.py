"""Telegram notifications and the semi-auto approval command loop.

Commands (only accepted from the configured ``TELEGRAM_CHAT_ID``):
    /status              engine + account + risk snapshot
    /approve <signal_id> queue the pending signal
    /reject <signal_id>  discard it
    /positions           open trades
    /close <trade_id>    close one trade
    /closeall            flatten everything
    /pause, /resume      trading switch
    /help
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

import aiohttp

from apex_smc_bot.config import NotificationConfig

logger = logging.getLogger("apex.notifications.telegram")

API_ROOT = "https://api.telegram.org"

CommandHandler = Callable[[str, list[str]], Awaitable[str]]


class RateLimiter:
    """Simple token bucket so a burst cannot get the bot rate-limited."""

    def __init__(self, max_tokens: int = 20, refill_seconds: float = 60.0) -> None:
        self.max_tokens = max_tokens
        self.refill_seconds = refill_seconds
        self._tokens = float(max_tokens)
        self._last = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0.0

    async def acquire(self) -> bool:
        now = asyncio.get_running_loop().time()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self.max_tokens, self._tokens + elapsed / self.refill_seconds * self.max_tokens)
        if self._tokens < 1:
            return False
        self._tokens -= 1
        return True


class TelegramBot:
    """Text delivery plus optional long-poll command listener."""

    name = "telegram"

    def __init__(
        self,
        token: str,
        chat_id: str,
        config: NotificationConfig | None = None,
        session: aiohttp.ClientSession | None = None,
        command_handler: CommandHandler | None = None,
    ) -> None:
        self.token = token
        self.chat_id = str(chat_id)
        self.config = config or NotificationConfig()
        self.max_retries = max(1, self.config.max_retries)
        self.timeout = self.config.timeout_seconds
        self._session = session
        self._owns_session = session is None
        self.command_handler = command_handler
        self._limiter = RateLimiter(max_tokens=18, refill_seconds=60.0)
        self._offset = 0
        self._last_error: str | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> "TelegramBot | None":
        if not settings.notifications.telegram_enabled:
            return None
        secrets = settings.secrets
        if not secrets.telegram_bot_token or not secrets.telegram_chat_id:
            logger.warning("Telegram enabled but TELEGRAM_BOT_TOKEN/CHAT_ID are missing")
            return None
        return cls(secrets.telegram_bot_token, secrets.telegram_chat_id, settings.notifications)

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def _call(self, method: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        session = await self._get_session()
        url = f"{API_ROOT}/bot{self.token}/{method}"
        for attempt in range(self.max_retries):
            try:
                async with session.post(url, json=payload) as response:
                    data = await response.json(content_type=None)
                    if response.status == 429:
                        retry_after = float(data.get("parameters", {}).get("retry_after", 2))
                        await asyncio.sleep(min(30.0, retry_after))
                        continue
                    if response.status >= 400 or not data.get("ok", False):
                        self._last_error = str(data.get("description", response.status))
                        logger.warning("Telegram %s failed: %s", method, self._last_error)
                        if response.status >= 500:
                            raise aiohttp.ClientError(self._last_error)
                        return None
                    self._last_error = None
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                self._last_error = str(exc)
                logger.warning("Telegram %s attempt %d failed: %s", method, attempt + 1, exc)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1.5 * (2 ** attempt))
        return None

    # ------------------------------------------------------------------ sending
    def _split(self, text: str, limit: int = 4000) -> list[str]:
        if len(text) <= limit:
            return [text]
        chunks: list[str] = []
        current = ""
        for block in text.split("\n"):
            candidate = f"{current}\n{block}" if current else block
            if len(candidate) > limit:
                if current:
                    chunks.append(current)
                current = block[:limit]
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    async def send_message(
        self, text: str, chat_id: str | None = None, parse_mode: str | None = None
    ) -> bool:
        """Send text; HTML/Markdown is never used because signal text is plain."""

        if not self.enabled or not text:
            return False
        if not await self._limiter.acquire():
            logger.warning("Telegram rate limit guard — message dropped")
            return False
        ok = True
        for chunk in self._split(text):
            payload: dict[str, Any] = {
                "chat_id": chat_id or self.chat_id,
                "text": chunk,
                "disable_web_page_preview": True,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            result = await self._call("sendMessage", payload)
            ok = ok and result is not None
        return ok

    async def send_signal(self, text: str) -> bool:
        return await self.send_message(text)

    # ------------------------------------------------------------------ commands
    async def _handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message") or update.get("edited_message") or {}
        chat = message.get("chat") or {}
        text = (message.get("text") or "").strip()
        if not text.startswith("/"):
            return
        if str(chat.get("id")) != self.chat_id:
            logger.warning("Ignoring command from unauthorised chat %s", chat.get("id"))
            await self.send_message("Unauthorised chat.", chat_id=str(chat.get("id")))
            return
        parts = text.split()
        command = parts[0].split("@")[0].lower()
        args = parts[1:]
        if self.command_handler is None:
            reply = "Command handler not attached."
        else:
            try:
                reply = await self.command_handler(command, args)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Command %s failed", command)
                reply = f"Command failed: {exc}"
        if reply:
            await self.send_message(reply)

    async def poll_once(self, timeout: int = 20) -> int:
        """Long-poll for updates; returns how many were handled."""

        data = await self._call(
            "getUpdates",
            {"offset": self._offset, "timeout": timeout, "allowed_updates": ["message"]},
        )
        if not data:
            return 0
        updates = data.get("result") or []
        for update in updates:
            self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)
            await self._handle_update(update)
        return len(updates)

    async def listen(self, poll_timeout: int = 20, stop_event: asyncio.Event | None = None) -> None:
        """Background command loop (only when a handler is attached)."""

        if self.command_handler is None:
            return
        logger.info("Telegram command listener started")
        while stop_event is None or not stop_event.is_set():
            try:
                await self.poll_once(poll_timeout)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("Telegram polling error")
                await asyncio.sleep(5)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed and self._owns_session:
            await self._session.close()
        self._session = None
