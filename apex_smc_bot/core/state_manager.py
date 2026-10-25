"""Runtime state that must survive a restart *and* in-process de-duplication.

Two different jobs:

1. **Live memory** — the armed :class:`PendingSetup` objects between the 15M scan
   and the 5M trigger.  They are rebuilt from fresh candles after a restart, so
   only their identity is persisted.
2. **Persistence** — a fingerprint ledger (``SETUP_LEDGER`` runtime key), the
   trading pause switch, engine counters and last-known account equity.

The fingerprint ledger is what prevents the "same zone re-armed on every bar"
bug from producing duplicate signals: a fingerprint may generate at most one
signal per TTL window, and a fingerprint that produced a signal is retired for
the rest of its life.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable

from apex_smc_bot.data.candle_manager import to_utc, utcnow
from apex_smc_bot.signals.models import PendingSetup, Signal

logger = logging.getLogger("apex.core.state")

LEDGER_KEY = "setup_ledger"
PAUSED_KEY = "trading_paused"
EQUITY_KEY = "last_equity"
COUNTERS_KEY = "engine_counters"


@dataclass
class LedgerEntry:
    fingerprint: str
    first_seen: datetime
    expires_at: datetime
    signal_id: str | None = None
    attempts: int = 0

    @property
    def signalled(self) -> bool:
        return self.signal_id is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "first_seen": self.first_seen.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "signal_id": self.signal_id,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LedgerEntry":
        return cls(
            fingerprint=str(payload["fingerprint"]),
            first_seen=to_utc(datetime.fromisoformat(str(payload["first_seen"]))),
            expires_at=to_utc(datetime.fromisoformat(str(payload["expires_at"]))),
            signal_id=payload.get("signal_id"),
            attempts=int(payload.get("attempts", 0)),
        )


@dataclass
class EngineCounters:
    cycles: int = 0
    setups_armed: int = 0
    triggers: int = 0
    signals: int = 0
    rejections: int = 0
    orders: int = 0
    errors: int = 0
    started_at: datetime = field(default_factory=utcnow)
    last_cycle_at: datetime | None = None
    last_signal_at: datetime | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycles": self.cycles,
            "setups_armed": self.setups_armed,
            "triggers": self.triggers,
            "signals": self.signals,
            "rejections": self.rejections,
            "orders": self.orders,
            "errors": self.errors,
            "started_at": self.started_at.isoformat(),
            "last_cycle_at": self.last_cycle_at.isoformat() if self.last_cycle_at else None,
            "last_signal_at": self.last_signal_at.isoformat() if self.last_signal_at else None,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EngineCounters":
        counters = cls()
        for name in ("cycles", "setups_armed", "triggers", "signals", "rejections", "orders", "errors"):
            setattr(counters, name, int(payload.get(name, 0) or 0))
        for name in ("started_at", "last_cycle_at", "last_signal_at"):
            raw = payload.get(name)
            setattr(counters, name, to_utc(datetime.fromisoformat(str(raw))) if raw else None)
        counters.last_error = payload.get("last_error")
        return counters


class StateManager:
    """Armed setups, dedupe ledger, pause switch and counters."""

    def __init__(self, database: Any | None = None, max_pending_age_minutes: int = 120) -> None:
        self.database = database
        self.max_pending_age_minutes = max_pending_age_minutes
        self._pending: dict[str, PendingSetup] = {}
        self._ledger: dict[str, LedgerEntry] = {}
        self.counters = EngineCounters()
        self._paused = False
        self._loaded = False
        self.load()

    # ------------------------------------------------------------------ persistence
    def load(self) -> None:
        if self._loaded or self.database is None:
            self._loaded = True
            return
        try:
            raw_ledger = self.database.get_runtime_state(LEDGER_KEY, {}) or {}
            for fingerprint, payload in raw_ledger.items():
                try:
                    self._ledger[fingerprint] = LedgerEntry.from_dict(payload)
                except Exception:  # noqa: BLE001
                    logger.warning("Dropping unreadable ledger entry %s", fingerprint)
            self._paused = bool(self.database.get_runtime_state(PAUSED_KEY, False))
            counters = self.database.get_runtime_state(COUNTERS_KEY)
            if counters:
                self.counters = EngineCounters.from_dict(counters)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load engine state — starting fresh")
        self._loaded = True

    def flush(self) -> None:
        """Persist the ledger, pause switch and counters (safe to call often)."""

        if self.database is None:
            return
        try:
            self.database.set_runtime_state(
                LEDGER_KEY, {key: entry.to_dict() for key, entry in self._ledger.items()}
            )
            self.database.set_runtime_state(PAUSED_KEY, self._paused)
            self.database.set_runtime_state(COUNTERS_KEY, self.counters.to_dict())
        except Exception:  # noqa: BLE001
            logger.exception("Failed to persist engine state")

    # ------------------------------------------------------------------ pause
    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self, reason: str = "manual") -> None:
        if not self._paused:
            logger.warning("Trading PAUSED (%s)", reason)
        self._paused = True
        self.flush()

    def resume(self) -> None:
        if self._paused:
            logger.warning("Trading RESUMED")
        self._paused = False
        self.flush()

    # ------------------------------------------------------------------ pending setups
    def add_pending(self, pending: PendingSetup, now: datetime | None = None) -> bool:
        """Register an armed setup; returns False if this fingerprint is spent."""

        now = to_utc(now or utcnow())
        entry = self._ledger.get(pending.fingerprint)
        if entry is not None:
            if entry.signalled:
                return False
            if to_utc(entry.expires_at) > now:
                entry.attempts += 1
                self._pending[pending.fingerprint] = pending
                return False
        self._ledger[pending.fingerprint] = LedgerEntry(
            fingerprint=pending.fingerprint,
            first_seen=to_utc(pending.created_at),
            expires_at=to_utc(pending.expires_at),
        )
        self._pending[pending.fingerprint] = pending
        self.counters.setups_armed += 1
        return True

    def get_pending(
        self, direction: str | None = None, now: datetime | None = None
    ) -> list[PendingSetup]:
        """Live (unexpired) setups; ``now`` keeps backtests deterministic."""

        now = to_utc(now or utcnow())
        items = [pending for pending in self._pending.values() if not pending.is_expired(now)]
        if direction is not None:
            items = [pending for pending in items if pending.direction == direction]
        return sorted(items, key=lambda item: item.created_at, reverse=True)

    def drop_pending(self, fingerprint: str) -> None:
        self._pending.pop(fingerprint, None)

    def mark_signalled(
        self, pending: PendingSetup, signal: Signal, now: datetime | None = None
    ) -> None:
        """Retire a fingerprint once it produced a signal."""

        entry = self._ledger.get(pending.fingerprint)
        if entry is None:
            entry = LedgerEntry(
                fingerprint=pending.fingerprint,
                first_seen=to_utc(pending.created_at),
                expires_at=to_utc(pending.expires_at),
            )
            self._ledger[pending.fingerprint] = entry
        entry.signal_id = signal.signal_id
        self._pending.pop(pending.fingerprint, None)
        self.counters.signals += 1
        self.counters.last_signal_at = to_utc(now or utcnow())
        self.flush()

    def is_spent(self, pending: PendingSetup) -> bool:
        entry = self._ledger.get(pending.fingerprint)
        return bool(entry and entry.signalled)

    def prune(self, now: datetime | None = None) -> int:
        """Remove expired ledger entries and pending setups."""

        now = to_utc(now or utcnow())
        stale = [key for key, entry in self._ledger.items() if to_utc(entry.expires_at) < now]
        for key in stale:
            self._ledger.pop(key, None)
            self._pending.pop(key, None)
        expired_pending = [
            key for key, pending in self._pending.items() if pending.is_expired(now)
        ]
        for key in expired_pending:
            self._pending.pop(key, None)
        if stale or expired_pending:
            logger.debug(
                "Pruned %d ledger entries and %d pending setups", len(stale), len(expired_pending)
            )
        return len(stale) + len(expired_pending)

    def ledger_size(self) -> int:
        return len(self._ledger)

    # ------------------------------------------------------------------ equity
    def set_last_equity(self, equity: float) -> None:
        if self.database is not None:
            try:
                self.database.set_runtime_state(EQUITY_KEY, float(equity))
            except Exception:  # noqa: BLE001
                pass

    def last_equity(self, default: float = 0.0) -> float:
        if self.database is None:
            return default
        try:
            return float(self.database.get_runtime_state(EQUITY_KEY, default) or default)
        except Exception:  # noqa: BLE001
            return default

    # ------------------------------------------------------------------ counters
    def tick_cycle(self, now: datetime | None = None) -> None:
        self.counters.cycles += 1
        self.counters.last_cycle_at = to_utc(now or utcnow())

    def note_trigger(self, executed: bool) -> None:
        if executed:
            self.counters.triggers += 1

    def note_rejection(self) -> None:
        self.counters.rejections += 1

    def note_order(self) -> None:
        self.counters.orders += 1

    def note_error(self, message: str) -> None:
        self.counters.errors += 1
        self.counters.last_error = message[:500]

    def status(self) -> dict[str, Any]:
        return {
            "paused": self._paused,
            "pending_setups": len(self._pending),
            "ledger_entries": len(self._ledger),
            "counters": self.counters.to_dict(),
            "pending": [pending.to_dict() for pending in self.get_pending()],
        }

    def reset(self) -> None:
        self._pending.clear()
        self._ledger.clear()
        if self.database is not None:
            try:
                self.database.set_runtime_state(LEDGER_KEY, {})
            except Exception:  # noqa: BLE001
                pass

    def pending_fingerprints(self) -> Iterable[str]:
        return list(self._pending.keys())
