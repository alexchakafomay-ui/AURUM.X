from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from ai_analyzer import AIAnalyzer
from config import Settings
from data_fetcher import DataFetcher
from logger import Database
from news_filter import NewsFilter
from scheduler import AurumScheduler
from stats import format_stats, format_weekly
from telegram_sender import TelegramSender
from tracker import SignalTracker
from utils import market_is_open

logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)


def authorized(update: Update, settings: Settings) -> bool:
    return bool(
        update.effective_chat
        and update.effective_chat.id == settings.telegram_chat_id
    )


# ============================================================
# COMMAND HANDLERS
# ============================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not authorized(update, settings):
        return

    market = "BUKA" if market_is_open() else "TUTUP (weekend)"
    text = (
        "🚀 <b>AURUM-X — ONLINE</b>\n\n"
        "<i>Presisi institusional, di saku kamu.</i>\n\n"
        f"Status pasar: <b>{market}</b>\n"
        f"Watchlist: {', '.join(settings.watchlist)}\n"
        f"Mode: <b>{'PAPER TRADING' if settings.paper_trading else 'SIGNAL ONLY'}</b>\n\n"
        "Perintah:\n"
        "/stats — statistik keseluruhan\n"
        "/weekly — performa 7 hari\n"
        "/calendar — berita 48 jam ke depan\n"
        "/pause — jeda sinyal\n"
        "/resume — aktifkan lagi\n\n"
        "⚠️ Bot ini tidak mengeksekusi order ke broker. "
        "Risiko 1-2% per trade. Bukan jaminan profit."
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not authorized(update, settings):
        return
    db: Database = context.bot_data["db"]
    stats = db.get_stats()
    await update.message.reply_text(format_stats(stats), parse_mode="HTML")


async def cmd_weekly(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not authorized(update, settings):
        return
    db: Database = context.bot_data["db"]
    stats = db.get_stats(days=7)
    await update.message.reply_text(format_weekly(stats), parse_mode="HTML")


async def cmd_calendar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not authorized(update, settings):
        return
    news: NewsFilter = context.bot_data["news"]
    sender: TelegramSender = context.bot_data["sender"]

    events = await asyncio.to_thread(news.upcoming, 48)
    await sender.send_calendar(events)


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not authorized(update, settings):
        return
    db: Database = context.bot_data["db"]
    db.set_paused(True)
    await update.message.reply_text(
        "⏸ <b>AURUM-X dijeda.</b>\nSinyal baru tidak akan dikirim sampai /resume.",
        parse_mode="HTML",
    )


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not authorized(update, settings):
        return
    db: Database = context.bot_data["db"]
    db.set_paused(False)
    await update.message.reply_text(
        "▶️ <b>AURUM-X aktif kembali.</b>\nSinyal akan dikirim pada siklus berikutnya.",
        parse_mode="HTML",
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not authorized(update, settings):
        return
    db: Database = context.bot_data["db"]
    status = "PAUSED" if db.is_paused() else "ACTIVE"
    market = "BUKA" if market_is_open() else "TUTUP"
    text = (
        f"⚙️ <b>AURUM-X STATUS</b>\n\n"
        f"Bot: <b>{status}</b>\n"
        f"Market: <b>{market}</b>\n"
        f"Sinyal hari ini: <b>{db.count_signals_today()}</b> / {settings.max_signals_per_day}\n"
        f"Open trade: <b>{db.open_trades_count()}</b> / {settings.max_open_trades}\n"
        f"Daily realized R: <b>{db.daily_realized_r():+.2f}R</b>"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Telegram error: %s", context.error, exc_info=context.error)


# ============================================================
# MAIN
# ============================================================

async def post_init(application: Application) -> None:
    scheduler: AurumScheduler = application.bot_data["scheduler"]
    scheduler.schedule(application)
    logger.info("AURUM-X scheduler aktif.")


def main() -> None:
    settings = Settings()
    setup_logging(settings.log_level)
    settings.validate()

    db = Database(settings.db_path)
    fetcher = DataFetcher(settings)
    news = NewsFilter(settings)
    analyzer = AIAnalyzer(settings)

    application = (
        Application.builder()
        .token(settings.telegram_token)
        .post_init(post_init)
        .build()
    )

    sender = TelegramSender(application.bot, settings.telegram_chat_id)
    tracker = SignalTracker(db, fetcher, sender)
    scheduler = AurumScheduler(
        settings, db, fetcher, news, analyzer, sender, tracker
    )

    application.bot_data.update(
        {
            "settings": settings,
            "db": db,
            "fetcher": fetcher,
            "news": news,
            "analyzer": analyzer,
            "sender": sender,
            "tracker": tracker,
            "scheduler": scheduler,
        }
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("stats", cmd_stats))
    application.add_handler(CommandHandler("weekly", cmd_weekly))
    application.add_handler(CommandHandler("calendar", cmd_calendar))
    application.add_handler(CommandHandler("pause", cmd_pause))
    application.add_handler(CommandHandler("resume", cmd_resume))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_error_handler(error_handler)

    logger.info("AURUM-X starting polling...")
    application.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
    )


if __name__ == "__main__":
    main()
