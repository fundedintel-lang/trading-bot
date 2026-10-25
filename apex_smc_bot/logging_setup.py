"""Structured JSON logging with dedicated signal / trade / error / analysis sinks.

Every decision is written as one JSON object per line (``*.log``), including an
ISO-8601 UTC timestamp, the event name and arbitrary structured fields.  Console
output is human readable for operators.

structlog is used for JSON rendering when installed; a stdlib fallback keeps the
application fully functional without it.
"""

from __future__ import annotations

import functools
import json
import logging
import logging.handlers
import sys
import traceback
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from enum import Enum
from pathlib import Path
from typing import Any

try:  # optional
    import structlog

    _JSON_RENDERER = structlog.processors.JSONRenderer()
except Exception:  # pragma: no cover - fallback
    structlog = None  # type: ignore[assignment]
    _JSON_RENDERER = None

from apex_smc_bot.config import LoggingConfig, Settings


class _JsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:  # noqa: D401
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (set, tuple)):
            return list(obj)
        if hasattr(obj, "to_dict"):
            try:
                return obj.to_dict()
            except Exception:  # pragma: no cover
                pass
        return str(obj)


def _dumps(payload: dict[str, Any]) -> str:
    if _JSON_RENDERER is not None:
        try:
            return _JSON_RENDERER(None, "info", payload)  # type: ignore[arg-type,return-value]
        except Exception:  # pragma: no cover
            pass
    return json.dumps(payload, cls=_JsonEncoder, ensure_ascii=False, sort_keys=False)


class JsonFormatter(logging.Formatter):
    """Format stdlib log records as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in getattr(record, "context", {}).items():
            payload.setdefault(key, value)
        if record.exc_info:
            payload["exception"] = "".join(traceback.format_exception(*record.exc_info)).strip()
        return _dumps(payload)


class ConsoleFormatter(logging.Formatter):
    """Compact, readable console format."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S")
        base = f"{ts} | {record.levelname:<7} | {record.name:<28} | {record.getMessage()}"
        context = getattr(record, "context", None)
        if context:
            rendered = " ".join(f"{k}={v}" for k, v in context.items() if k not in {"payload"})
            if rendered:
                base = f"{base} | {rendered}"
        if record.exc_info:
            base = f"{base}\n{''.join(traceback.format_exception(*record.exc_info))}"
        return base


class _ContextAdapter(logging.LoggerAdapter):
    """Adds a persistent context dict to every record."""

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        context = dict(self.extra or {})
        extra = dict(kwargs.get("extra") or {})
        merged = {**context, **(extra.get("context") or {})}
        if merged:
            kwargs["extra"] = {"context": merged}
        return msg, kwargs

    def bind(self, **fields: Any) -> "_ContextAdapter":
        return _ContextAdapter(self.logger, {**(self.extra or {}), **fields})


@dataclass
class LoggingBundle:
    """Handles for each log sink plus the configured loggers."""

    engine: _ContextAdapter
    signals: _ContextAdapter
    trades: _ContextAdapter
    analysis: _ContextAdapter
    errors: _ContextAdapter
    directory: Path
    handlers: list[logging.Handler] = field(default_factory=list)

    def bind(self, **fields: Any) -> "LoggingBundle":
        return LoggingBundle(
            engine=self.engine.bind(**fields),
            signals=self.signals.bind(**fields),
            trades=self.trades.bind(**fields),
            analysis=self.analysis.bind(**fields),
            errors=self.errors.bind(**fields),
            directory=self.directory,
            handlers=self.handlers,
        )

    def close(self) -> None:
        for handler in self.handlers:
            try:
                handler.flush()
                handler.close()
            except Exception:  # pragma: no cover
                pass


def _rotating_handler(path: Path, cfg: LoggingConfig, formatter: logging.Formatter) -> logging.Handler:
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=cfg.max_bytes,
        backupCount=cfg.backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    return handler


_LOGGER_NAMES = {
    "engine": "apex.engine",
    "signals": "apex.signals",
    "trades": "apex.trades",
    "analysis": "apex.analysis",
    "errors": "apex.errors",
}


def setup_logging(
    settings: Settings, console: bool = True, level: str | int | None = None
) -> LoggingBundle:
    """Configure all sinks and return bound loggers.

    Args:
        settings: Loaded settings (uses ``settings.logging``).
        console: Emit human-readable records to stdout as well.
        level: Override the configured level (name or number).  Backtests pass
            ``WARNING`` so an 18,000-cycle replay does not flood the audit log.
    """

    cfg = settings.logging
    directory = settings.logs_dir
    directory.mkdir(parents=True, exist_ok=True)
    json_formatter = JsonFormatter()
    console_formatter = ConsoleFormatter()
    if isinstance(level, int):
        level_value = level
    elif isinstance(level, str):
        level_value = getattr(logging, level.upper(), logging.INFO)
    else:
        level_value = getattr(logging, cfg.level.upper(), logging.INFO)
    level = level_value

    root = logging.getLogger("apex")
    root.setLevel(level)
    root.propagate = False
    for existing in list(root.handlers):
        root.removeHandler(existing)

    handlers: list[logging.Handler] = []
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(level)
        root.addHandler(console_handler)
        handlers.append(console_handler)

    file_map = {
        # Engine/orchestrator decisions are part of the audit trail.
        "apex.engine": cfg.analysis_log,
        "apex.signals": cfg.signal_log,
        "apex.trades": cfg.trade_log,
        "apex.analysis": cfg.analysis_log,
        "apex.errors": cfg.error_log,
    }
    for logger_name, filename in file_map.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = True
        # Idempotent: drop handlers from a previous setup_logging() call so a
        # re-configured process never writes each record twice.
        for stale in list(logger.handlers):
            logger.removeHandler(stale)
            with suppress(Exception):
                stale.close()
        if not filename:
            continue
        path = directory / filename
        handler = _rotating_handler(path, cfg, json_formatter)
        handler.setLevel(level)
        logger.addHandler(handler)
        handlers.append(handler)
        # The dedicated logger hands off to "apex" for console output; avoid
        # duplicate file writes by keeping its own handler only.

    # Engine events are persisted to analysis.log as an audit trail.
    engine = _ContextAdapter(logging.getLogger("apex.engine"), {})
    signals = _ContextAdapter(logging.getLogger("apex.signals"), {})
    trades = _ContextAdapter(logging.getLogger("apex.trades"), {})
    analysis = _ContextAdapter(logging.getLogger("apex.analysis"), {})
    errors = _ContextAdapter(logging.getLogger("apex.errors"), {})
    return LoggingBundle(
        engine=engine,
        signals=signals,
        trades=trades,
        analysis=analysis,
        errors=errors,
        directory=directory,
        handlers=handlers,
    )


# --------------------------------------------------------------------------- helpers
def log_exception(logger: _ContextAdapter, event: str, exc: BaseException, **fields: Any) -> None:
    """Log an exception with traceback at error level (never raises)."""

    try:
        logger.error(
            event,
            extra={"context": {**fields, "error": str(exc)}},
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    except Exception:  # pragma: no cover
        pass


def log_signal(bundle: LoggingBundle, signal_payload: dict[str, Any]) -> None:
    """Persist a complete signal (JSON line) to ``signals.log``."""

    bundle.signals.info(
        "SIGNAL_GENERATED",
        extra={"context": {"payload": signal_payload, "signal_id": signal_payload.get("signal_id")}},
    )


def log_trade_event(bundle: LoggingBundle, event: str, **fields: Any) -> None:
    bundle.trades.info(event, extra={"context": fields})


def log_analysis(bundle: LoggingBundle, event: str, **fields: Any) -> None:
    bundle.analysis.info(event, extra={"context": fields})


def log_engine(bundle: LoggingBundle, event: str, **fields: Any) -> None:
    bundle.engine.info(event, extra={"context": fields})


def get_logger(name: str = "apex.engine") -> _ContextAdapter:
    """Return a context-capable logger (used by standalone modules/tests)."""

    return _ContextAdapter(logging.getLogger(name), {})


def configure_fallback_logging(level: str = "INFO") -> None:
    """Minimal console logging for scripts/tests that do not load Settings."""

    root = logging.getLogger("apex")
    if root.handlers:
        return
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ConsoleFormatter())
    root.addHandler(handler)
    root.propagate = False


def silence_noisy_libraries() -> None:
    """Reduce third-party log noise (aiohttp/urllib3 access logs)."""

    for name in ("asyncio", "urllib3", "aiohttp.access", "apscheduler.executors.default"):
        logging.getLogger(name).setLevel(logging.WARNING)


# Convenience decorator used by background loops.
def log_calls(logger: _ContextAdapter, event: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            logger.debug(event)
            return func(*args, **kwargs)

        return wrapper

    return decorator
