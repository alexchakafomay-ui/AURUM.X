from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Any, Optional

import pandas as pd
import requests

from config import Settings
from utils import utcnow

logger = logging.getLogger(__name__)

CURRENCIES = ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")

COUNTRY_TO_CURRENCY = {
    "US": "USD", "EU": "EUR", "GB": "GBP", "JP": "JPY", "AU": "AUD",
    "CA": "CAD", "CH": "CHF", "NZ": "NZD", "DE": "EUR", "FR": "EUR",
    "IT": "EUR", "ES": "EUR",
}

class NewsFilter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._sentiment: dict[str, dict] = {}
        self._sentiment_ts = 0.0
        self._sentiment_ttl = 3 * 3600
        self._calendar: list[dict] = []
        self._calendar_ts = 0.0
        self._calendar_ttl = 3600

    def _fetch_sentiment(self) -> dict[str, dict]:
        key = self.settings.fxnewsbias_api_key
        if not key:
            return {c: {"score": 50, "bias": "Neutral"} for c in CURRENCIES}
        try:
            response = requests.get(
                "https://fxnewsbias.com/api/v1/sentiment",
                headers={"Authorization": f"Bearer {key}"},
                timeout=15,
            )
            data = response.json()
            result = {}
            for item in data.get("data", []):
                currency = str(item.get("currency", "")).upper()
                if currency:
                    result[currency] = {"score": int(item.get("score", 50)), "bias": item.get("bias", "Neutral")}
            for currency in CURRENCIES:
                result.setdefault(currency, {"score": 50, "bias": "Neutral"})
            return result
        except Exception as exc:
            logger.warning("FXNewsBias gagal: %s", exc)
            return {}

    def sentiment(self) -> dict[str, dict]:
        now = time.time()
        if self._sentiment and now - self._sentiment_ts < self._sentiment_ttl:
            return self._sentiment
        fetched = self._fetch_sentiment()
        if fetched:
            self._sentiment = fetched
            self._sentiment_ts = now
        elif not self._sentiment:
            self._sentiment = {c: {"score": 50, "bias": "Neutral"} for c in CURRENCIES}
        return self._sentiment

    def _fetch_calendar(self) -> list[dict]:
        try:
            response = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=20)
            raw = response.json()
            events = []
            for item in raw:
                impact = str(item.get("impact", "")).strip().title()
                if impact not in {"High", "Medium"}:
                    continue
                dt = pd.Timestamp(item.get("date"))
                if dt.tzinfo is None:
                    dt = dt.tz_localize("UTC")
                else:
                    dt = dt.tz_convert("UTC")
                country = str(item.get("country", "")).upper()
                currency = COUNTRY_TO_CURRENCY.get(country, country)
                events.append({
                    "title": item.get("title", ""),
                    "currency": currency,
                    "datetime_utc": dt.to_pydatetime(),
                    "impact": impact,
                    "forecast": item.get("forecast", ""),
                    "previous": item.get("previous", ""),
                })
            events.sort(key=lambda x: x["datetime_utc"])
            return events
        except Exception as exc:
            logger.warning("Kalender ForexFactory gagal: %s", exc)
            return []

    def calendar(self) -> list[dict]:
        now = time.time()
        if self._calendar and now - self._calendar_ts < self._calendar_ttl:
            return self._calendar
        fetched = self._fetch_calendar()
        if fetched:
            self._calendar = fetched
            self._calendar_ts = now
        return self._calendar

    def refresh(self) -> None:
        self.sentiment()
        self.calendar()

    def high_impact_within(self, currency: str, minutes: int = 30) -> Optional[dict]:
        now = utcnow()
        limit = now + timedelta(minutes=minutes)
        for event in self.calendar():
            if event["currency"] != currency:
                continue
            if event["impact"] != "High":
                continue
            event_dt = pd.Timestamp(event["datetime_utc"])
            if event_dt.tzinfo is None:
                event_dt = event_dt.tz_localize("UTC")
            if now <= event_dt <= limit:
                return event
        return None

    def gate(self, pair: str, direction: str) -> tuple[bool, str, dict]:
        base = pair[:3].upper()
        quote = pair[3:].upper()
        sentiment = self.sentiment()
        base_score = sentiment.get(base, {}).get("score", 50)
        quote_score = sentiment.get(quote, {}).get("score", 50)
        news_bias = "Neutral"
        if base_score >= 60 and quote_score <= 45:
            news_bias = "Bullish"
        elif base_score <= 40 and quote_score >= 60:
            news_bias = "Bearish"

        meta = {"base_score": base_score, "quote_score": quote_score, "news_bias": news_bias}

        for currency in (base, quote):
            event = self.high_impact_within(currency, 30)
            if event:
                return False, f"High-impact news {currency}: {event['title']}", meta

        if direction == "BUY":
            if base_score < 40 or quote_score > 65:
                return False, f"Sentimen berlawanan: {base} {base_score} / {quote} {quote_score}", meta
        elif direction == "SELL":
            if base_score > 60 or quote_score < 35:
                return False, f"Sentimen berlawanan: {base} {base_score} / {quote} {quote_score}", meta

        now = utcnow()
        for currency in (base, quote):
            if self.high_impact_within(currency, 15):
                return False, "Spread berpotensi melebar menjelang news.", meta
        if now.weekday() == 4 and now.hour >= 19:
            return False, "Spread Jumat sore berpotensi melebar.", meta

        return True, "OK", meta

    def get_major_news(self, days_ahead: int = 7) -> list[dict]:
        keywords = ["CPI", "PPI", "FOMC", "INTEREST RATE", "NFP", "NON-FARM", "PMI", "GDP", "UNEMPLOYMENT"]
        now = utcnow()
        limit = now + timedelta(days=days_ahead)
        result = []
        
        for event in self.calendar():
            if event["impact"] != "High":
                continue
            title_upper = event["title"].upper()
            if not any(kw in title_upper for kw in keywords):
                continue
                
            event_dt = pd.Timestamp(event["datetime_utc"])
            if event_dt.tzinfo is None:
                event_dt = event_dt.tz_localize("UTC")
                
            if now <= event_dt <= limit:
                result.append(event)
        return result

    def upcoming(self, hours: int = 48) -> list[dict]:
        now = utcnow()
        limit = now + timedelta(hours=hours)
        result = []
        for event in self.calendar():
            event_dt = pd.Timestamp(event["datetime_utc"])
            if event_dt.tzinfo is None:
                event_dt = event_dt.tz_localize("UTC")
            if now <= event_dt <= limit:
                result.append(event)
        return result[:30]
