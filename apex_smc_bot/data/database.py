"""SQLite persistence: candles, signals, trades, equity curve, events, runtime state.

SQLite is intentionally used instead of TimescaleDB for a single-instrument bot:
zero-ops, WAL concurrency and trivially portable.  The schema is compatible with
a later PostgreSQL migration (JSON payload columns keep the audit trail intact).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apex_smc_bot.data.candle_manager import Candle, to_utc

logger = logging.getLogger("apex.database")

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    instrument TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    timestamp  TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL DEFAULT 0,
    spread REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (instrument, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_candles_lookup ON candles (instrument, timeframe, timestamp DESC);

CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    instrument TEXT NOT NULL,
    direction TEXT NOT NULL,
    status TEXT NOT NULL,
    confluence_score REAL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals (created_at DESC);

CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    signal_id TEXT,
    instrument TEXT NOT NULL,
    direction TEXT NOT NULL,
    status TEXT NOT NULL,
    entry_price REAL,
    stop_loss REAL,
    take_profit REAL,
    lots REAL,
    units REAL,
    opened_at TEXT,
    closed_at TEXT,
    realized_pnl REAL NOT NULL DEFAULT 0,
    payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_opened ON trades (opened_at DESC);

CREATE TABLE IF NOT EXISTS equity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    equity REAL NOT NULL,
    balance REAL NOT NULL,
    daily_drawdown_percent REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity (timestamp DESC);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_type ON events (event_type, timestamp DESC);

CREATE TABLE IF NOT EXISTS news_events (
    event_key TEXT PRIMARY KEY,
    event_time TEXT NOT NULL,
    title TEXT NOT NULL,
    impact TEXT,
    currency TEXT,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runtime_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class Database:
    """Thread-safe SQLite wrapper used by the engine, backtester and dashboard.

    The database never raises into the trading loop for non-critical writes: errors
    are logged and returned as ``False``/empty values so a disk hiccup cannot take
    the engine down.  Critical read paths still raise on programming errors.
    """

    def __init__(self, path: str | Path, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        if enabled:
            self._connect()

    # ------------------------------------------------------------------ setup
    def _connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database is disabled or not connected")
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                with suppress(Exception):
                    self._conn.commit()
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ candles
    def upsert_candles(self, instrument: str, timeframe: str, candles: Iterable[Candle]) -> int:
        """Insert or replace candles; returns number of rows written."""

        if not self.enabled:
            return 0
        rows = [
            (
                instrument,
                timeframe,
                c.timestamp.isoformat(),
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
                c.spread,
            )
            for c in candles
        ]
        if not rows:
            return 0
        try:
            with self._lock, self.connection as conn:
                conn.executemany(
                    """INSERT INTO candles
                       (instrument, timeframe, timestamp, open, high, low, close, volume, spread)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(instrument, timeframe, timestamp) DO UPDATE SET
                         open=excluded.open, high=excluded.high, low=excluded.low,
                         close=excluded.close, volume=excluded.volume, spread=excluded.spread""",
                    rows,
                )
            return len(rows)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist candles: %s", exc)
            return 0

    def get_candles(
        self,
        instrument: str,
        timeframe: str,
        limit: int = 500,
        since: datetime | None = None,
    ) -> list[Candle]:
        """Return the most recent ``limit`` candles in ascending time order."""

        if not self.enabled:
            return []
        sql = (
            "SELECT timestamp, open, high, low, close, volume, spread FROM candles "
            "WHERE instrument=? AND timeframe=?"
        )
        params: list[Any] = [instrument, timeframe]
        if since is not None:
            sql += " AND timestamp>=?"
            params.append(to_utc(since).isoformat())
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(int(limit))
        try:
            with self._lock:
                rows = self.connection.execute(sql, params).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load candles: %s", exc)
            return []
        candles: list[Candle] = []
        for row in reversed(rows):
            try:
                candles.append(
                    Candle(
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"],
                        spread=row["spread"],
                    )
                )
            except ValueError:
                continue
        return candles

    def candle_count(self, instrument: str, timeframe: str) -> int:
        if not self.enabled:
            return 0
        try:
            with self._lock:
                row = self.connection.execute(
                    "SELECT COUNT(*) AS n FROM candles WHERE instrument=? AND timeframe=?",
                    (instrument, timeframe),
                ).fetchone()
            return int(row["n"]) if row else 0
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------------ signals
    def record_signal(
        self,
        signal_id: str,
        timestamp: datetime,
        instrument: str,
        direction: str,
        payload: Mapping[str, Any],
        confluence_score: float | None = None,
        status: str = "GENERATED",
    ) -> bool:
        if not self.enabled:
            return False
        try:
            with self._lock, self.connection as conn:
                conn.execute(
                    """INSERT INTO signals
                       (signal_id, created_at, instrument, direction, status, confluence_score, payload)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(signal_id) DO UPDATE SET status=excluded.status, payload=excluded.payload""",
                    (
                        signal_id,
                        to_utc(timestamp).isoformat(),
                        instrument,
                        direction,
                        status,
                        confluence_score,
                        json.dumps(payload, default=str),
                    ),
                )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to record signal %s: %s", signal_id, exc)
            return False

    def update_signal_status(self, signal_id: str, status: str) -> bool:
        if not self.enabled:
            return False
        try:
            with self._lock, self.connection as conn:
                conn.execute("UPDATE signals SET status=? WHERE signal_id=?", (status, signal_id))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to update signal %s: %s", signal_id, exc)
            return False

    def get_signals(self, limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        sql = "SELECT * FROM signals"
        params: list[Any] = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        try:
            with self._lock:
                rows = self.connection.execute(sql, params).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read signals: %s", exc)
            return []
        out = []
        for row in rows:
            record = dict(row)
            with suppress(Exception):
                record["payload"] = json.loads(record["payload"])
            out.append(record)
        return out

    # ------------------------------------------------------------------ trades
    def record_trade(self, trade: Mapping[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            with self._lock, self.connection as conn:
                conn.execute(
                    """INSERT INTO trades
                       (trade_id, signal_id, instrument, direction, status, entry_price, stop_loss,
                        take_profit, lots, units, opened_at, closed_at, realized_pnl, payload)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(trade_id) DO UPDATE SET
                         status=excluded.status, stop_loss=excluded.stop_loss,
                         take_profit=excluded.take_profit, closed_at=excluded.closed_at,
                         realized_pnl=excluded.realized_pnl, payload=excluded.payload""",
                    (
                        trade["trade_id"],
                        trade.get("signal_id"),
                        trade.get("instrument", "XAU_USD"),
                        trade.get("direction", ""),
                        trade.get("status", "OPEN"),
                        trade.get("entry_price"),
                        trade.get("stop_loss"),
                        trade.get("take_profit"),
                        trade.get("lots"),
                        trade.get("units"),
                        str(trade.get("opened_at") or "") or None,
                        str(trade.get("closed_at") or "") or None,
                        float(trade.get("realized_pnl", 0.0) or 0.0),
                        json.dumps(trade.get("payload", {}), default=str),
                    ),
                )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to record trade: %s", exc)
            return False

    def update_trade(self, trade_id: str, **fields: Any) -> bool:
        if not self.enabled or not fields:
            return False
        allowed = {"status", "stop_loss", "take_profit", "closed_at", "realized_pnl", "payload"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            raise ValueError(f"No updatable trade fields in {sorted(fields)}")
        if "payload" in updates and not isinstance(updates["payload"], str):
            updates["payload"] = json.dumps(updates["payload"], default=str)
        assignments = ", ".join(f"{key}=?" for key in updates)
        try:
            with self._lock, self.connection as conn:
                conn.execute(
                    f"UPDATE trades SET {assignments} WHERE trade_id=?",
                    [*updates.values(), trade_id],
                )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to update trade %s: %s", trade_id, exc)
            return False

    def get_trades(self, limit: int = 200, status: str | None = None) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        sql = "SELECT * FROM trades"
        params: list[Any] = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY opened_at DESC LIMIT ?"
        params.append(limit)
        try:
            with self._lock:
                rows = self.connection.execute(sql, params).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read trades: %s", exc)
            return []
        out = []
        for row in rows:
            record = dict(row)
            with suppress(Exception):
                record["payload"] = json.loads(record["payload"]) if record["payload"] else {}
            out.append(record)
        return out

    # ------------------------------------------------------------------ equity
    def record_equity(
        self,
        equity: float,
        balance: float | None = None,
        daily_drawdown_percent: float = 0.0,
        timestamp: datetime | None = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            with self._lock, self.connection as conn:
                conn.execute(
                    "INSERT INTO equity (timestamp, equity, balance, daily_drawdown_percent) VALUES (?, ?, ?, ?)",
                    (
                        to_utc(timestamp or datetime.now(timezone.utc)).isoformat(),
                        float(equity),
                        float(balance if balance is not None else equity),
                        float(daily_drawdown_percent),
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to record equity: %s", exc)

    def get_equity_curve(self, limit: int = 500) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            with self._lock:
                rows = self.connection.execute(
                    "SELECT timestamp, equity, balance, daily_drawdown_percent FROM equity "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        except Exception:  # noqa: BLE001
            return []
        return [dict(row) for row in reversed(rows)]

    def latest_equity(self) -> float | None:
        curve = self.get_equity_curve(limit=1)
        return float(curve[0]["equity"]) if curve else None

    def daily_start_equity(self, day: datetime | None = None) -> float | None:
        """First recorded equity for a UTC calendar day (drawdown anchoring)."""

        target = (day or datetime.now(timezone.utc)).date().isoformat()
        if not self.enabled:
            return None
        try:
            with self._lock:
                row = self.connection.execute(
                    "SELECT equity FROM equity WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT 1",
                    (f"{target}T00:00:00",),
                ).fetchone()
            return float(row["equity"]) if row else None
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ events
    def record_event(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            with self._lock, self.connection as conn:
                conn.execute(
                    "INSERT INTO events (timestamp, event_type, payload) VALUES (?, ?, ?)",
                    (
                        to_utc(timestamp or datetime.now(timezone.utc)).isoformat(),
                        event_type,
                        json.dumps(payload or {}, default=str),
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to record event %s: %s", event_type, exc)

    def get_events(self, event_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        sql = "SELECT timestamp, event_type, payload FROM events"
        params: list[Any] = []
        if event_type:
            sql += " WHERE event_type=?"
            params.append(event_type)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        try:
            with self._lock:
                rows = self.connection.execute(sql, params).fetchall()
        except Exception:  # noqa: BLE001
            return []
        out = []
        for row in reversed(rows):
            record = dict(row)
            with suppress(Exception):
                record["payload"] = json.loads(record["payload"])
            out.append(record)
        return out

    # ------------------------------------------------------------------ news cache
    def store_news_events(self, events: Iterable[Mapping[str, Any]]) -> int:
        if not self.enabled:
            return 0
        rows = []
        now = datetime.now(timezone.utc).isoformat()
        for event in events:
            key = event.get("event_key") or f"{event.get('date')}|{event.get('title')}"
            rows.append(
                (
                    key,
                    str(event.get("date", "")),
                    str(event.get("title", "")),
                    str(event.get("impact", "")),
                    str(event.get("currency", "")),
                    now,
                )
            )
        if not rows:
            return 0
        try:
            with self._lock, self.connection as conn:
                conn.executemany(
                    """INSERT INTO news_events (event_key, event_time, title, impact, currency, fetched_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(event_key) DO UPDATE SET
                         event_time=excluded.event_time, title=excluded.title,
                         impact=excluded.impact, fetched_at=excluded.fetched_at""",
                    rows,
                )
            return len(rows)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to cache news events: %s", exc)
            return 0

    # ------------------------------------------------------------------ state
    def set_runtime_state(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        try:
            with self._lock, self.connection as conn:
                conn.execute(
                    """INSERT INTO runtime_state (key, value, updated_at) VALUES (?, ?, ?)
                       ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
                    (key, json.dumps(value, default=str), datetime.now(timezone.utc).isoformat()),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist state %s: %s", key, exc)

    def get_runtime_state(self, key: str, default: Any = None) -> Any:
        if not self.enabled:
            return default
        try:
            with self._lock:
                row = self.connection.execute(
                    "SELECT value FROM runtime_state WHERE key=?", (key,)
                ).fetchone()
            if row is None:
                return default
            return json.loads(row["value"])
        except Exception:  # noqa: BLE001
            return default

    # ------------------------------------------------------------------ reporting
    def stats(self) -> dict[str, Any]:
        """Row counts and file size (dashboard/health endpoint)."""

        if not self.enabled:
            return {"enabled": False}
        out: dict[str, Any] = {"enabled": True, "path": str(self.path)}
        for table in ("candles", "signals", "trades", "equity", "events"):
            try:
                with self._lock:
                    row = self.connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
                out[table] = int(row["n"]) if row else 0
            except Exception:  # noqa: BLE001
                out[table] = 0
        with suppress(Exception):
            out["size_mb"] = round(self.path.stat().st_size / (1024 * 1024), 3)
        return out
