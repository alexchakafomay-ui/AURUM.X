from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from typing import Any, Optional

from utils import to_db_ts, utcnow

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    pair TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence INTEGER NOT NULL,
    entry REAL,
    sl REAL,
    tp1 REAL,
    tp2 REAL,
    rr_tp1 REAL,
    rr_tp2 REAL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    result_pips REAL DEFAULT 0,
    result_r REAL DEFAULT 0,
    news_bias TEXT,
    reasoning TEXT,
    valid_until TEXT,
    raw_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts_utc);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
CREATE INDEX IF NOT EXISTS idx_signals_pair ON signals(pair);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    meta TEXT
);

CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.init_db()

    def init_db(self) -> None:
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # ---------------------------------------------------------
    # SIGNALS
    # ---------------------------------------------------------
    def log_signal(self, signal: dict) -> int:
        with self._lock:
            cur = self.conn.execute(
                """
                INSERT INTO signals (
                    ts_utc, pair, direction, confidence,
                    entry, sl, tp1, tp2, rr_tp1, rr_tp2,
                    status, news_bias, reasoning, valid_until, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?)
                """,
                (
                    to_db_ts(signal.get("timestamp_utc")),
                    signal["pair"],
                    signal["direction"],
                    int(signal["confidence"]),
                    signal.get("entry"),
                    signal.get("sl"),
                    signal.get("tp1"),
                    signal.get("tp2"),
                    signal.get("risk_reward_tp1"),
                    signal.get("risk_reward_tp2"),
                    signal.get("news_bias", "Neutral"),
                    signal.get("reasoning", ""),
                    to_db_ts(signal.get("valid_until")),
                    json.dumps(signal, ensure_ascii=False, default=str),
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def update_status(
        self,
        signal_id: int,
        status: str,
        result_pips: Optional[float] = None,
        result_r: Optional[float] = None,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE signals
                SET status = ?, result_pips = COALESCE(?, result_pips),
                    result_r = COALESCE(?, result_r)
                WHERE id = ?
                """,
                (status, result_pips, result_r, signal_id),
            )
            self.conn.commit()

    def get_open_signals(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT * FROM signals
                WHERE status IN ('PENDING', 'ACTIVE')
                  AND julianday(valid_until) > julianday('now', '-2 hours')
                ORDER BY id ASC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def count_signals_today(self) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM signals WHERE date(ts_utc) = date('now')"
            ).fetchone()
            return int(row["n"])

    def open_trades_count(self) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM signals WHERE status IN ('PENDING', 'ACTIVE')"
            ).fetchone()
            return int(row["n"])

    def daily_realized_r(self) -> float:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT COALESCE(SUM(result_r), 0) AS r FROM signals
                WHERE date(ts_utc) = date('now')
                  AND status IN ('TP1', 'TP2', 'SL')
                """
            ).fetchone()
            return float(row["r"])

    def recent_pair_signal(self, pair: str, hours: int = 2) -> Optional[dict]:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT * FROM signals
                WHERE pair = ?
                  AND julianday(ts_utc) >= julianday('now', ?)
                ORDER BY id DESC LIMIT 1
                """,
                (pair, f"-{hours} hours"),
            ).fetchone()
            return dict(row) if row else None

    # ---------------------------------------------------------
    # STATS
    # ---------------------------------------------------------
    def get_stats(self, days: Optional[int] = None) -> dict:
        with self._lock:
            if days:
                rows = self.conn.execute(
                    "SELECT * FROM signals WHERE julianday(ts_utc) >= julianday('now', ?)",
                    (f"-{days} days",),
                ).fetchall()
            else:
                rows = self.conn.execute("SELECT * FROM signals").fetchall()

        records = [dict(r) for r in rows]
        total = len(records)
        counts = {k: 0 for k in ("TP1", "TP2", "SL", "EXPIRED", "CANCELLED", "PENDING", "ACTIVE")}
        for rec in records:
            status = rec.get("status", "PENDING")
            counts[status] = counts.get(status, 0) + 1

        wins = counts["TP1"] + counts["TP2"]
        losses = counts["SL"]
        closed = wins + losses
        winrate = (wins / closed * 100) if closed else 0.0

        pips = [float(r.get("result_pips") or 0) for r in records if r.get("result_pips")]
        r_values = [float(r.get("result_r") or 0) for r in records if r.get("result_r")]
        gross_win = sum(x for x in pips if x > 0)
        gross_loss = abs(sum(x for x in pips if x < 0))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
        avg_r = sum(r_values) / len(r_values) if r_values else 0.0

        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "closed": closed,
            "tp1": counts["TP1"],
            "tp2": counts["TP2"],
            "sl": counts["SL"],
            "expired": counts["EXPIRED"],
            "cancelled": counts["CANCELLED"],
            "pending": counts["PENDING"] + counts["ACTIVE"],
            "winrate": winrate,
            "total_pips": sum(pips),
            "profit_factor": profit_factor,
            "avg_r": avg_r,
        }

    # ---------------------------------------------------------
    # RISK
    # ---------------------------------------------------------
    def can_trade(self, settings) -> tuple[bool, str]:
        if self.is_paused():
            return False, "Bot sedang dijeda."
        if self.count_signals_today() >= settings.max_signals_per_day:
            return False, "Batas sinyal harian tercapai."
        if self.open_trades_count() >= settings.max_open_trades:
            return False, "Batas open trade tercapai."
        max_loss_r = settings.max_daily_loss_pct / max(settings.risk_percent, 0.1)
        if self.daily_realized_r() <= -max_loss_r:
            return False, "Max daily loss tercapai. Stop sampai besok."
        return True, "OK"

    # ---------------------------------------------------------
    # STATE
    # ---------------------------------------------------------
    def set_state(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO state(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self.conn.commit()

    def get_state(self, key: str, default: str = "") -> str:
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM state WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default

    def set_paused(self, paused: bool) -> None:
        self.set_state("paused", "1" if paused else "0")

    def is_paused(self) -> bool:
        return self.get_state("paused", "0") == "1"

    # ---------------------------------------------------------
    # EVENTS
    # ---------------------------------------------------------
    def log_event(self, level: str, message: str, meta: Optional[dict] = None) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO events(ts_utc, level, message, meta) VALUES(?, ?, ?, ?)",
                (
                    to_db_ts(utcnow()),
                    level,
                    message,
                    json.dumps(meta or {}, ensure_ascii=False, default=str),
                ),
            )
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()
