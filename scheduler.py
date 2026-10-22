from __future__ import annotations

import asyncio
import logging
from datetime import time, timezone

from telegram.ext import Application, ContextTypes

from ai_analyzer import AIAnalyzer
from config import Settings
from data_fetcher import DataFetcher
from indicators import build_candidate, build_snapshot
from logger import Database
from news_filter import NewsFilter
from telegram_sender import TelegramSender
from tracker import SignalTracker
from utils import market_is_open, utcnow

logger = logging.getLogger(__name__)


class AurumScheduler:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        fetcher: DataFetcher,
        news: NewsFilter,
        analyzer: AIAnalyzer,
        sender: TelegramSender,
        tracker: SignalTracker,
    ):
        self.settings = settings
        self.db = db
        self.fetcher = fetcher
        self.news = news
        self.analyzer = analyzer
        self.sender = sender
        self.tracker = tracker
        self._analysis_lock = asyncio.Lock()

    # ---------------------------------------------------------
    # SCHEDULE
    # ---------------------------------------------------------
    def schedule(self, app: Application) -> None:
        jq = app.job_queue
        tz = timezone.utc
        days = tuple(range(5))  # Mon-Fri

        for hour in self.settings.session_hours_utc:
            # Greeting hanya di sesi pertama (07:00).
            if hour == self.settings.session_hours_utc[0]:
                jq.run_daily(
                    self.greeting_job,
                    time=time(hour=hour - 1, minute=55, tzinfo=tz),
                    days=days,
                    name=f"greeting_{hour}",
                )

            pre_minute = 60 - self.settings.prealert_minutes
            pre_hour = hour - 1
            if pre_minute == 60:
                pre_minute = 0
                pre_hour = hour

            jq.run_daily(
                self.pre_alert_job,
                time=time(hour=pre_hour, minute=pre_minute, tzinfo=tz),
                days=days,
                name=f"prealert_{hour}",
            )
            jq.run_daily(
                self.analysis_job,
                time=time(hour=hour, minute=0, tzinfo=tz),
                days=days,
                name=f"analysis_{hour}",
            )

        jq.run_repeating(
            self.weekend_job,
            interval=6 * 3600,
            first=60,
            name="weekend",
        )
        jq.run_repeating(
            self.tracker_job,
            interval=300,
            first=30,
            name="tracker",
        )

    # ---------------------------------------------------------
    # JOBS
    # ---------------------------------------------------------
    async def greeting_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not market_is_open():
            return
        today = utcnow().strftime("%Y-%m-%d")
        if self.db.get_state("greeted_date") == today:
            return

        await asyncio.to_thread(self.news.refresh)
        sentiment = self.news.sentiment()
        events = [
            e
            for e in self.news.upcoming(24)
            if e["impact"] == "High"
            and e["datetime_utc"].date() == utcnow().date()
        ]
        await self.sender.send_greeting(sentiment, events)
        self.db.set_state("greeted_date", today)

    async def pre_alert_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not market_is_open():
            return
        if self.db.is_paused():
            return
        bias = self.db.get_state("last_bias", "Mixed")
        await self.sender.send_prealert(bias)

    async def analysis_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not market_is_open():
            return

        async with self._analysis_lock:
            try:
                await self.run_analysis()
            except Exception:
                logger.exception("Analysis job error")
                await self.sender.send_no_signal("Terjadi error internal saat analisis.")

    async def weekend_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        if market_is_open():
            return
        await self.sender.send_weekend()

    async def tracker_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not market_is_open():
            return
        await self.tracker.run()

    # ---------------------------------------------------------
    # ANALYSIS
    # ---------------------------------------------------------
    async def run_analysis(self) -> None:
        ok, reason = self.db.can_trade(self.settings)
        if not ok:
            if self.settings.send_no_signal:
                await self.sender.send_no_signal(reason)
            return

        # Pre-fetch news cache (sync requests di thread).
        await asyncio.to_thread(self.news.refresh)

        logger.info("Fetching watchlist...")
        bundles = await asyncio.to_thread(self.fetcher.fetch_watchlist)

        candidates = []
        for pair, bundle in bundles.items():
            if bundle is None:
                continue
            try:
                candidate = build_candidate(pair, bundle, self.news)
                if candidate:
                    snapshot = build_snapshot(bundle)
                    candidates.append((candidate, snapshot, bundle))
            except Exception:
                logger.exception("Candidate error untuk %s", pair)

        if not candidates:
            if self.settings.send_no_signal:
                await self.sender.send_no_signal("Tidak ada konfluensi yang memenuhi standar.")
            return

        candidates.sort(key=lambda x: x[0]["confidence"], reverse=True)

        slots = min(
            self.settings.max_open_trades - self.db.open_trades_count(),
            self.settings.max_signals_per_day - self.db.count_signals_today(),
        )
        if slots <= 0:
            if self.settings.send_no_signal:
                await self.sender.send_no_signal("Slot trade penuh.")
            return

        sent = 0
        for candidate, snapshot, bundle in candidates[:slots]:
            # Hindari sinyal duplikat pair dalam 2 jam.
            if self.db.recent_pair_signal(candidate["pair"], hours=2):
                continue

            final = await asyncio.to_thread(
                self.analyzer.analyze,
                candidate,
                snapshot,
                candidate.get("news_meta", {}),
            )

            if final.get("direction") == "NO_SIGNAL":
                continue

            ok, reason = self.db.can_trade(self.settings)
            if not ok:
                break

            self.db.log_signal(final)
            await self.sender.send_signal(final)
            sent += 1

            # Update bias terakhir untuk pre-alert berikutnya.
            self.db.set_state("last_bias", candidate["direction"])

        if sent == 0 and self.settings.send_no_signal:
            await self.sender.send_no_signal("Semua kandidat ditolak filter AI/risiko.")
