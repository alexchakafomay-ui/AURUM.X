from __future__ import annotations

import asyncio
import logging
from typing import Optional

import pandas as pd

from data_fetcher import DataFetcher
from logger import Database
from telegram_sender import TelegramSender
from utils import parse_db_ts, pip_size, utcnow

logger = logging.getLogger(__name__)


class SignalTracker:
    """
    Paper-trading tracker.
    Tidak mengirim order ke broker. Hanya mengecek harga pasar
    dan memperbarui status sinyal di SQLite.
    """

    def __init__(self, db: Database, fetcher: DataFetcher, sender: TelegramSender):
        self.db = db
        self.fetcher = fetcher
        self.sender = sender

    async def run(self) -> None:
        signals = self.db.get_open_signals()
        for signal in signals:
            try:
                df = await asyncio.to_thread(
                    self.fetcher.fetch_yfinance, signal["pair"], "5m", "5d"
                )
                result = self.evaluate(signal, df)
                if result is None:
                    continue

                status, pips, result_r = result
                if status == signal["status"]:
                    continue

                self.db.update_status(signal["id"], status, pips, result_r)

                if status in {"TP1", "TP2", "SL", "EXPIRED", "CANCELLED"}:
                    await self.sender.send_result(signal, status, pips, result_r)

            except Exception:
                logger.exception("Tracker error untuk signal %s", signal.get("id"))

    def evaluate(self, signal: dict, df: Optional[pd.DataFrame]) -> Optional[tuple[str, float, float]]:
        if df is None or df.empty:
            return None

        try:
            ts = parse_db_ts(signal["ts_utc"])
            valid_until = parse_db_ts(signal["valid_until"])
        except Exception:
            return None

        df = df[df.index >= ts - pd.Timedelta(minutes=5)]
        if df.empty:
            return None

        entry = float(signal["entry"])
        sl = float(signal["sl"])
        tp1 = float(signal["tp1"])
        tp2 = float(signal["tp2"])
        direction = signal["direction"]
        pair = signal["pair"]
        now = pd.Timestamp(utcnow())
        pip = pip_size(pair)

        if direction == "BUY":
            touched_mask = df["low"] <= entry
        else:
            touched_mask = df["high"] >= entry

        if not touched_mask.any():
            if now > valid_until:
                return ("EXPIRED", 0.0, 0.0)

            last_close = float(df["close"].iloc[-1])
            if direction == "BUY" and last_close >= entry + 0.5 * (tp1 - entry):
                return ("CANCELLED", 0.0, 0.0)
            if direction == "SELL" and last_close <= entry - 0.5 * (entry - tp1):
                return ("CANCELLED", 0.0, 0.0)

            return None

        first_idx = touched_mask[touched_mask].index[0]
        after = df.loc[first_idx:]

        for _, bar in after.iterrows():
            if direction == "BUY":
                # Konservatif: SL dicek lebih dulu jika satu bar menyentuh keduanya.
                if bar["low"] <= sl:
                    return ("SL", -abs(entry - sl) / pip, -1.0)
                if bar["high"] >= tp1:
                    return ("TP1", abs(tp1 - entry) / pip, float(signal["rr_tp1"]))
            else:
                if bar["high"] >= sl:
                    return ("SL", -abs(sl - entry) / pip, -1.0)
                if bar["low"] <= tp1:
                    return ("TP1", abs(entry - tp1) / pip, float(signal["rr_tp1"]))

        # Entry tersentuh tetapi belum TP/SL.
        return ("ACTIVE", 0.0, 0.0)
