from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import requests

from config import Settings
from utils import RateLimiter

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Twelve Data sebagai sumber utama.
    yfinance sebagai fallback.

    Optimasi kuota:
      - H1 5000 bar -> resample D1 & H4.
      - M5 5000 bar -> resample M15 & M30.
      - Hanya 2 request per pair per siklus.
    """

    TD_URL = "https://api.twelvedata.com/time_series"
    TD_INTERVALS = {
        "D1": "1day",
        "H4": "4h",
        "H1": "1h",
        "M30": "30min",
        "M15": "15min",
        "M5": "5min",
        "M1": "1min",
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        # Free Twelve Data: 8 credits/menit. Sisakan margin.
        self.rate_limiter = RateLimiter(max_calls=8, period_seconds=60)
        self._bundle_cache: dict[str, tuple[float, dict]] = {}
        self._cache_ttl = 300  # 5 menit

    # ---------------------------------------------------------
    # SYMBOL
    # ---------------------------------------------------------
    def td_symbol(self, pair: str) -> str:
        if pair == "XAUUSD":
            return "XAU/USD"
        return f"{pair[:3]}/{pair[3:]}"

    def yf_symbol(self, pair: str) -> str:
        if pair == "XAUUSD":
            return "GC=F"
        return f"{pair}=X"

    # ---------------------------------------------------------
    # TWELVE DATA
    # ---------------------------------------------------------
    def fetch_twelve(
        self, pair: str, interval: str, outputsize: int = 5000
    ) -> Optional[pd.DataFrame]:
        if not self.settings.twelve_data_api_key:
            return None

        params = {
            "symbol": self.td_symbol(pair),
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.settings.twelve_data_api_key,
            "format": "JSON",
            "timezone": "UTC",
        }

        last_error = None
        for attempt in range(3):
            try:
                self.rate_limiter.acquire()
                response = requests.get(self.TD_URL, params=params, timeout=30)
                data = response.json()

                if data.get("status") == "error" or "values" not in data:
                    raise RuntimeError(str(data)[:300])

                df = pd.DataFrame(data["values"])
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                df = df.set_index("datetime").sort_index()

                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    elif col == "volume":
                        df["volume"] = 0.0

                df = df.dropna(subset=["open", "high", "low", "close"])
                return df

            except Exception as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning(
                    "Twelve Data gagal %s %s (attempt %s): %s",
                    pair,
                    interval,
                    attempt + 1,
                    exc,
                )
                time.sleep(wait)

        logger.error("Twelve Data menyerah %s %s: %s", pair, interval, last_error)
        return None

    # ---------------------------------------------------------
    # YFINANCE FALLBACK
    # ---------------------------------------------------------
    def fetch_yfinance(
        self, pair: str, interval: str = "1h", period: str = "60d"
    ) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf

            ticker = self.yf_symbol(pair)
            df = yf.download(
                ticker,
                interval=interval,
                period=period,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if df is None or df.empty:
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.rename(columns=str.lower)
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()

            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    df[col] = 0.0
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["open", "high", "low", "close"])
            return df if not df.empty else None

        except Exception as exc:
            logger.warning("yfinance gagal %s %s %s: %s", pair, interval, period, exc)
            return None

    # ---------------------------------------------------------
    # RESAMPLE
    # ---------------------------------------------------------
    @staticmethod
    def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        out = (
            df.resample(rule, label="right", closed="right")
            .agg(agg)
            .dropna(subset=["open", "high", "low", "close"])
        )
        return out

    # ---------------------------------------------------------
    # BUNDLE
    # ---------------------------------------------------------
    def fetch_pair_bundle(self, pair: str) -> Optional[dict[str, pd.DataFrame]]:
        now = time.time()
        cached = self._bundle_cache.get(pair)
        if cached and now - cached[0] < self._cache_ttl:
            return cached[1]

        h1 = self.fetch_twelve(pair, self.TD_INTERVALS["H1"], 5000)
        m5 = self.fetch_twelve(pair, self.TD_INTERVALS["M5"], 5000)

        # Fallback HTF
        if h1 is None or len(h1) < 250:
            d1 = self.fetch_yfinance(pair, "1d", "2y")
            h1 = self.fetch_yfinance(pair, "1h", "60d")
            if d1 is None or h1 is None or len(h1) < 250:
                logger.error("Data HTF tidak lengkap untuk %s", pair)
                return None
            h4 = self.resample(h1, "4h")
        else:
            d1 = self.resample(h1, "1D")
            h4 = self.resample(h1, "4h")

        # Fallback LTF
        if m5 is None or len(m5) < 100:
            m5 = self.fetch_yfinance(pair, "5m", "5d")
            if m5 is None or len(m5) < 100:
                logger.error("Data LTF tidak lengkap untuk %s", pair)
                return None

        m15 = self.resample(m5, "15min")
        m30 = self.resample(m5, "30min")

        bundle = {
            "D1": d1,
            "H4": h4,
            "H1": h1,
            "M30": m30,
            "M15": m15,
            "M5": m5,
        }

        if len(d1) < 200 or len(h4) < 100 or len(m30) < 60:
            logger.warning("Bar kurang untuk %s: D1=%s H4=%s M30=%s", pair, len(d1), len(h4), len(m30))
            return None

        self._bundle_cache[pair] = (now, bundle)
        return bundle

    def fetch_watchlist(self) -> dict[str, Optional[dict[str, pd.DataFrame]]]:
        result = {}
        for pair in self.settings.watchlist:
            logger.info("Fetching %s...", pair)
            result[pair] = self.fetch_pair_bundle(pair)
        return result
