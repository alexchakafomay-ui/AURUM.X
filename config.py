from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_list(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = os.getenv(name)
    if not raw:
        return default
    return tuple(x.strip().upper() for x in raw.split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    # Telegram
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: int = field(default_factory=lambda: _env_int("TELEGRAM_CHAT_ID", 0))

    # Data
    twelve_data_api_key: str = field(default_factory=lambda: os.getenv("TWELVE_DATA_API_KEY", ""))
    fxnewsbias_api_key: str = field(default_factory=lambda: os.getenv("FXNEWSBIAS_API_KEY", ""))

    # AI
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
    )
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    ai_timeout: int = field(default_factory=lambda: _env_int("AI_TIMEOUT", 45))

    # Risk
    paper_trading: bool = field(default_factory=lambda: _env_bool("PAPER_TRADING", True))
    risk_percent: float = field(default_factory=lambda: _env_float("RISK_PERCENT", 1.5))
    max_signals_per_day: int = field(default_factory=lambda: _env_int("MAX_SIGNALS_PER_DAY", 6))
    max_open_trades: int = field(default_factory=lambda: _env_int("MAX_OPEN_TRADES", 2))
    max_daily_loss_pct: float = field(default_factory=lambda: _env_float("MAX_DAILY_LOSS_PCT", 3.0))
    min_confidence: int = field(default_factory=lambda: _env_int("MIN_CONFIDENCE", 70))
    signal_valid_hours: int = field(default_factory=lambda: _env_int("SIGNAL_VALID_HOURS", 2))

    # Runtime
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "aurum_x.db"))
    send_no_signal: bool = field(default_factory=lambda: _env_bool("SEND_NO_SIGNAL", True))
    include_m1: bool = field(default_factory=lambda: _env_bool("INCLUDE_M1", False))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Market
    watchlist: Tuple[str, ...] = field(
        default_factory=lambda: _env_list(
            "WATCHLIST",
            ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF", "XAUUSD"),
        )
    )
    session_hours_utc: Tuple[int, ...] = (7, 11, 15, 19)
    prealert_minutes: int = 2
    min_confluence: int = 4

    @property
    def pair_map(self) -> dict[str, str]:
        """EURUSD -> EUR/USD, XAUUSD -> XAU/USD."""
        result = {}
        for pair in self.watchlist:
            if pair == "XAUUSD":
                result[pair] = "XAU/USD"
            else:
                result[pair] = f"{pair[:3]}/{pair[3:]}"
        return result

    def validate(self) -> None:
        missing = []
        if not self.telegram_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not self.telegram_chat_id:
            missing.append("TELEGRAM_CHAT_ID")
        if not self.twelve_data_api_key and not self.openrouter_api_key:
            # yfinance fallback masih bisa jalan, tapi tanpa AI key bot akan NO_SIGNAL.
            pass
        if missing:
            raise RuntimeError(f"Environment variable wajib belum diisi: {', '.join(missing)}")
