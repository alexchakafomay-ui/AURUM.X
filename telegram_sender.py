from __future__ import annotations

import asyncio
import html
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from telegram import Bot

from utils import (
    escape_html,
    fmt_price,
    next_market_open,
    pips_between,
    utcnow,
    weekend_remaining,
)

logger = logging.getLogger(__name__)


class TelegramSender:
    def __init__(self, bot: Bot, chat_id: int):
        self.bot = bot
        self.chat_id = chat_id

    # ---------------------------------------------------------
    # CORE
    # ---------------------------------------------------------
    async def send_message(self, text: str, chat_id: Optional[int] = None) -> None:
        target = chat_id or self.chat_id
        for attempt in range(3):
            try:
                await self.bot.send_message(
                    chat_id=target,
                    text=text,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
                return
            except Exception as exc:
                logger.warning("Telegram send gagal (attempt %s): %s", attempt + 1, exc)
                await asyncio.sleep(2 ** attempt)
        logger.error("Telegram send menyerah.")

    # ---------------------------------------------------------
    # GREETING
    # ---------------------------------------------------------
    async def send_greeting(self, sentiment: dict, events: list[dict]) -> None:
        lines = []
        for currency in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"):
            item = sentiment.get(currency, {"score": 50, "bias": "Neutral"})
            lines.append(f"{currency} {item['score']} {item['bias']}")

        news_lines = []
        for event in events[:5]:
            dt = pd.Timestamp(event["datetime_utc"]).strftime("%H:%M UTC")
            news_lines.append(f"• {dt} {event['currency']} — {event['title']}")
        news_text = "\n".join(news_lines) if news_lines else "• Tidak ada high-impact hari ini."

        text = (
            "☀️ <b>AURUM-X — SELAMAT PAGI, BOS!</b>\n\n"
            "Pagi ini saya sudah bangun duluan buat analisa pasar buat Bos. 😎\n"
            "📊 <b>Watchlist:</b> EURUSD · GBPUSD · USDJPY · XAUUSD · dll\n"
            f"🌍 <b>Sentimen global:</b> {' · '.join(lines)}\n"
            f"📰 <b>Berita penting hari ini:</b>\n{escape_html(news_text)}\n\n"
            "Saya sedang menyiapkan setup terbaik. Kasih saya 2 menit ya Bos, "
            "siapkan platform trading-nya. Sinyal menyusul. 🎯"
        )
        await self.send_message(text)

    # ---------------------------------------------------------
    # PRE-ALERT
    # ---------------------------------------------------------
    async def send_prealert(self, bias: str = "Mixed") -> None:
        text = (
            "🟡 <b>AURUM-X — PERSIAPAN</b>\n\n"
            "Oke Bos, saya sudah selesai analisa multi-timeframe\n"
            "(D1 → H4 → H1 → M30 → M15 → M5).\n"
            f"Arah bias: <b>{escape_html(bias)}</b>\n"
            "Sinyal akan dikirim dalam 2 menit.\n\n"
            "Siapkan posisi Bos. 🎯"
        )
        await self.send_message(text)

    # ---------------------------------------------------------
    # SIGNAL
    # ---------------------------------------------------------
    async def send_signal(self, signal: dict) -> None:
        pair = signal["pair"]
        direction = signal["direction"]
        entry = signal["entry"]
        sl = signal["sl"]
        tp1 = signal["tp1"]
        tp2 = signal["tp2"]

        sl_pips = pips_between(pair, entry, sl)
        tp1_pips = pips_between(pair, entry, tp1)
        tp2_pips = pips_between(pair, entry, tp2)

        alignment = signal.get("tf_alignment", {})
        marks = []
        for tf in ("D1", "H4", "H1", "M30", "M15", "M5"):
            bias = alignment.get(tf, "Neutral")
            if direction == "BUY":
                mark = "✅" if bias == "Bullish" else ("⚠️" if bias == "Neutral" else "❌")
            else:
                mark = "✅" if bias == "Bearish" else ("⚠️" if bias == "Neutral" else "❌")
            marks.append(f"{tf} {mark}")

        emoji = "🟢" if direction == "BUY" else "🔴"
        label = "BUY" if direction == "BUY" else "SELL"

        text = (
            f"{emoji} <b>AURUM-X — SIGNAL TERVERIFIKASI</b>\n\n"
            f"📈 <b>Pair</b>       : {pair}\n"
            f"🎯 <b>Arah</b>       : {label}\n"
            f"💰 <b>Entry</b>      : {fmt_price(pair, entry)}\n"
            f"🛑 <b>SL</b>         : {fmt_price(pair, sl)} ({sl_pips:.0f} pips)\n"
            f"✅ <b>TP1</b>        : {fmt_price(pair, tp1)} ({tp1_pips:.0f} pips | RR 1:{signal['risk_reward_tp1']:.2f})\n"
            f"✅ <b>TP2</b>        : {fmt_price(pair, tp2)} ({tp2_pips:.0f} pips | RR 1:{signal['risk_reward_tp2']:.2f})\n\n"
            f"📊 <b>Confidence</b> : {signal['confidence']}%\n"
            f"📰 <b>News Bias</b>  : {signal.get('news_bias', 'Neutral')} "
            f"({signal.get('news_score', {}).get('base', 50)} vs {signal.get('news_score', {}).get('quote', 50)})\n"
            f"🔍 <b>TF Alignment</b>: {' · '.join(marks)}\n\n"
            f"📝 <b>Alasan Entry:</b>\n{escape_html(signal.get('reasoning', '-'))}\n\n"
            f"⏳ <b>Valid hingga:</b> {signal.get('valid_until', '-')}\n"
            f"⚠️ {escape_html(signal.get('risk_note', 'Risiko 1-2% per trade. Bukan jaminan profit.'))}"
        )
        await self.send_message(text)

    # ---------------------------------------------------------
    # NO SIGNAL
    # ---------------------------------------------------------
    async def send_no_signal(self, reason: str = "") -> None:
        text = (
            "⚪ <b>AURUM-X — TIDAK ADA SETUP</b>\n\n"
            "Bos, saya sudah scan semua pair. Hari ini belum ada konfluensi "
            "yang memenuhi standar. Sabar ya Bos — lebih baik tidak entry "
            "daripada entry asal-asalan. Saya pantau terus. 👀"
        )
        if reason:
            text += f"\n\n<i>{escape_html(reason[:500])}</i>"
        await self.send_message(text)

    # ---------------------------------------------------------
    # WEEKEND
    # ---------------------------------------------------------
    async def send_weekend(self) -> None:
        days, hours, minutes = weekend_remaining()
        open_dt = next_market_open().strftime("%A, %d %B %Y")

        text = (
            "🌙 <b>AURUM-X — PASAR TUTUP</b>\n\n"
            "Halo Bos! 👋\n"
            "Pasar forex sedang tutup untuk akhir pekan.\n"
            f"🕐 Buka kembali: <b>{open_dt} pukul 00:00 UTC</b>\n"
            f"⏳ Sisa waktu: <b>{days} hari {hours} jam {minutes} menit</b> lagi\n\n"
            "💡 Selama menunggu, ini yang bisa Bos lakukan:\n"
            "   • Review performa minggu ini dengan /stats\n"
            "   • Baca rangkuman mingguan dengan /weekly\n"
            "   • Cek berita ekonomi minggu depan dengan /calendar\n\n"
            "Santai dulu Bos, recharge energi. AURUM-X standby 24/7. 🚀"
        )
        await self.send_message(text)

    # ---------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------
    async def send_result(
        self,
        signal: dict,
        status: str,
        pips: float,
        result_r: float,
    ) -> None:
        pair = signal["pair"]
        direction = signal["direction"]

        if status == "TP1":
            text = (
                "✅ <b>AURUM-X — TP1 HIT! 🎉</b>\n"
                f"Pair {pair} | {direction} | {pips:+.1f} pips | RR 1:{result_r:.2f}\n"
                "Mantap Bos! Statistik: /stats"
            )
        elif status == "TP2":
            text = (
                "🏆 <b>AURUM-X — TP2 HIT! 🔥</b>\n"
                f"Pair {pair} | {direction} | {pips:+.1f} pips | RR 1:{result_r:.2f}\n"
                "Full target Bos! Statistik: /stats"
            )
        elif status == "SL":
            text = (
                "🔴 <b>AURUM-X — SL HIT</b>\n"
                f"Pair {pair} | {direction} | {pips:+.1f} pips\n"
                "Tenang Bos, ini bagian dari proses. Selanjutnya lebih baik. 💪"
            )
        elif status == "EXPIRED":
            text = (
                "⚪ <b>AURUM-X — SINYAL EXPIRED</b>\n"
                f"Pair {pair} | Entry tidak tersentuh dalam 2 jam.\n"
                "Sinyal dibatalkan otomatis. Tidak ada eksekusi."
            )
        elif status == "CANCELLED":
            text = (
                "⚪ <b>AURUM-X — SINYAL DIBATALKAN</b>\n"
                f"Pair {pair} | Harga sudah lewat 50% jarak ke TP1 sebelum entry.\n"
                "Menghindari entry telat."
            )
        else:
            return

        await self.send_message(text)

    # ---------------------------------------------------------
    # CALENDAR
    # ---------------------------------------------------------
    async def send_calendar(self, events: list[dict]) -> None:
        if not events:
            await self.send_message(
                "📅 <b>AURUM-X — KALENDER</b>\n\nTidak ada berita high/medium impact dalam 48 jam ke depan."
            )
            return

        lines = ["📅 <b>AURUM-X — KALENDER 48 JAM</b>\n"]
        for event in events:
            dt = pd.Timestamp(event["datetime_utc"]).strftime("%d/%m %H:%M UTC")
            icon = "🔴" if event["impact"] == "High" else "🟡"
            lines.append(
                f"{icon} {dt} | <b>{event['currency']}</b> | {escape_html(event['title'])}"
            )
        await self.send_message("\n".join(lines))
