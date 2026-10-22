from __future__ import annotations

import html
import json
import math
import re
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd

UTC = timezone.utc


def utcnow() -> datetime:
    return datetime.now(UTC)


def market_is_open(dt: Optional[datetime] = None) -> bool:
    """
    Pasar forex dianggap buka Senin 00:00 UTC sampai Jumat 21:00 UTC.
    Weekend: Jumat >= 21:00, Sabtu, dan Minggu.
    """
    dt = dt or utcnow()
    wd = dt.weekday()  # Mon=0 ... Sun=6
    if wd == 4 and dt.hour >= 21:
        return False
    if wd in (5, 6):
        return False
    return True


def next_market_open(dt: Optional[datetime] = None) -> datetime:
    """Mengembalikan Senin 00:00 UTC berikutnya."""
    dt = dt or utcnow()
    days_ahead = (0 - dt.weekday()) % 7
    candidate = (dt + timedelta(days=days_ahead)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    if candidate <= dt:
        candidate += timedelta(days=7)
    return candidate


def weekend_remaining(dt: Optional[datetime] = None) -> tuple[int, int, int]:
    dt = dt or utcnow()
    delta = next_market_open(dt) - dt
    total_seconds = max(0, int(delta.total_seconds()))
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    return days, hours, minutes


def pip_size(pair: str) -> float:
    pair = pair.upper()
    if pair.endswith("JPY"):
        return 0.01
    if pair == "XAUUSD":
        return 0.1
    return 0.0001


def price_decimals(pair: str) -> int:
    pair = pair.upper()
    if pair.endswith("JPY"):
        return 3
    if pair == "XAUUSD":
        return 2
    return 5


def fmt_price(pair: str, value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{price_decimals(pair)}f}"


def pips_between(pair: str, a: float, b: float) -> float:
    return abs(a - b) / pip_size(pair)


def rr(entry: float, sl: float, tp: float) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    return abs(tp - entry) / risk


def escape_html(text: Any) -> str:
    return html.escape(str(text), quote=False)


def extract_json(text: str) -> Optional[dict]:
    """Mengambil objek JSON pertama dari output LLM."""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def to_db_ts(value: Any) -> str:
    """Normalisasi timestamp ke format SQLite UTC: YYYY-MM-DD HH:MM:SS."""
    if isinstance(value, str):
        ts = pd.Timestamp(value)
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    else:
        ts = pd.Timestamp.now(tz="UTC")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def parse_db_ts(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


class RateLimiter:
    """Token bucket sederhana untuk API dengan limit per menit."""

    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period = period_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()

                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return

                sleep_for = self.period - (now - self._calls[0]) + 0.05

            time.sleep(max(0.05, sleep_for))
