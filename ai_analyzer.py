from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any, Optional

from config import Settings
from utils import extract_json, utcnow, to_db_ts

logger = logging.getLogger(__name__)

# ============================================================
# SYSTEM PROMPT: MULTI-AGENT DEBATE (TRADINGAGENTS LITE)
# ============================================================
SYSTEM_PROMPT = """Kamu adalah AURUM-X, sistem AI trading institusional yang terdiri dari 4 agen spesialis di dalam satu otak.

TUGASMU:
1. Baca snapshot data pasar dan kandidat sinyal dari rule-engine.
2. Lakukan "debat internal" secara singkat:
   - AGEN BULL: Cari alasan terkuat mengapa harga bisa naik (bantah jika lemah).
   - AGEN BEAR: Cari alasan terkuat mengapa harga bisa turun (bantah jika lemah).
   - RISK MANAGER: Periksa apakah SL/TP logis, RR memadai, dan berita aman. Tolak jika berisiko tinggi.
3. KEPUTUSAN AKHIR: Tentukan arah (BUY/SELL/NO_SIGNAL) berdasarkan konsensus debat tadi.

ATURAN KETAT:
- Arah final harus sama dengan kandidat, atau pilih NO_SIGNAL.
- SL harus logis: BUY sl < entry < tp1 < tp2; SELL tp2 < tp1 < entry < sl.
- RR TP1 minimal 1.45, RR TP2 minimal 2.9.
- Confidence minimal 70. Jika ragu, NO_SIGNAL.
- Jangan ubah pair.
- Jika news bias berlawanan arah, NO_SIGNAL.

Kamu HARUS mengembalikan JSON murni tanpa markdown, tanpa penjelasan di luar JSON. Bahasa reasoning: Indonesia, 4-6 kalimat (rangkuman hasil debat agen).
"""

USER_PROMPT_TEMPLATE = """Validasi kandidat sinyal berikut dengan simulasi debat multi-agen.

FORMAT JSON WAJIB (Hanya keluarkan ini, tidak ada teks lain):
{{
  "bot": "AURUM-X",
  "timestamp_utc": "ISO8601",
  "pair": "EURUSD",
  "direction": "BUY" | "SELL" | "NO_SIGNAL",
  "confidence": 0-100,
  "entry": number,
  "sl": number,
  "tp1": number,
  "tp2": number,
  "risk_reward_tp1": number,
  "risk_reward_tp2": number,
  "news_bias": "Bullish" | "Bearish" | "Neutral",
  "news_score": {{"base": number, "quote": number}},
  "tf_alignment": {{"D1": "...", "H4": "...", "H1": "...", "M30": "...", "M15": "...", "M5": "..."}},
  "key_levels": {{"support": number, "resistance": number}},
  "reasoning": "Hasil debat: Agen Bull menilai... Agen Bear menilai... Risk Manager memutuskan...",
  "valid_until": "ISO8601",
  "risk_note": "Risiko 1-2% per trade. Bukan jaminan profit."
}}

SNAPSHOT MULTI-TIMEFRAME:
{snapshot}

KANDIDAT RULE ENGINE:
{candidate}

NEWS META:
{news_meta}
"""


class AIAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _call_openai_compatible(
        self,
        base_url: str,
        api_key: str,
        model: str,
        messages: list[dict],
        extra_headers: Optional[dict] = None,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self.settings.ai_timeout,
        )

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1500,
        }

        try:
            response = client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
                extra_headers=extra_headers or {},
            )
        except Exception:
            response = client.chat.completions.create(
                **kwargs,
                extra_headers=extra_headers or {},
            )

        return response.choices[0].message.content or ""

    def analyze(
        self,
        candidate: dict,
        snapshot: dict,
        news_meta: dict,
    ) -> dict:
        if candidate is None:
            return self.no_signal("-", "Tidak ada kandidat dari rule engine.")

        prompt = USER_PROMPT_TEMPLATE.format(
            snapshot=json.dumps(snapshot, ensure_ascii=False, default=str)[:18000],
            candidate=json.dumps(candidate, ensure_ascii=False, default=str)[:4000],
            news_meta=json.dumps(news_meta, ensure_ascii=False, default=str)[:2000],
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        raw = ""
        try:
            if self.settings.openrouter_api_key:
                raw = self._call_openai_compatible(
                    "https://openrouter.ai/api/v1",
                    self.settings.openrouter_api_key,
                    self.settings.openrouter_model,
                    messages,
                    {
                        "HTTP-Referer": "https://github.com/aurum-x",
                        "X-Title": "AURUM-X TradingAgents Lite",
                    },
                )
            elif self.settings.groq_api_key:
                raw = self._call_openai_compatible(
                    "https://api.groq.com/openai/v1",
                    self.settings.groq_api_key,
                    self.settings.groq_model,
                    messages,
                )
            else:
                logger.warning("Tidak ada API key AI, memakai rule-engine fallback.")
                return self.fallback(candidate, news_meta)
        except Exception as exc:
            logger.warning("OpenRouter gagal: %s", exc)
            if self.settings.groq_api_key:
                try:
                    raw = self._call_openai_compatible(
                        "https://api.groq.com/openai/v1",
                        self.settings.groq_api_key,
                        self.settings.groq_model,
                        messages,
                    )
                except Exception as exc2:
                    logger.error("Groq juga gagal: %s", exc2)
                    return self.fallback(candidate, news_meta)
            else:
                return self.fallback(candidate, news_meta)

        parsed = extract_json(raw)
        if not parsed:
            logger.warning("AI output bukan JSON valid: %s", raw[:300])
            return self.fallback(candidate, news_meta)

        final = self.validate(parsed, candidate, news_meta)
        if final is None:
            logger.warning("AI output tidak lolos validasi, fallback rule engine.")
            return self.fallback(candidate, news_meta)

        return final

    def validate(
        self,
        data: dict,
        candidate: dict,
        news_meta: dict,
    ) -> Optional[dict]:
        direction = str(data.get("direction", "")).upper()
        if direction == "NO_SIGNAL":
            return self.no_signal(
                candidate["pair"],
                str(data.get("reasoning", "AI memilih NO_SIGNAL.")),
            )
        if direction not in {"BUY", "SELL"}:
            return None
        if direction != candidate["direction"]:
            return None
        if str(data.get("pair", "")).upper() != candidate["pair"]:
            return None

        try:
            entry = float(data["entry"])
            sl = float(data["sl"])
            tp1 = float(data["tp1"])
            tp2 = float(data["tp2"])
            confidence = int(data.get("confidence", 0))
        except (KeyError, TypeError, ValueError):
            return None

        if direction == "BUY" and not (sl < entry < tp1 < tp2):
            return None
        if direction == "SELL" and not (tp2 < tp1 < entry < sl):
            return None

        risk = abs(entry - sl)
        if risk <= 0:
            return None
        rr1 = abs(tp1 - entry) / risk
        rr2 = abs(tp2 - entry) / risk
        if rr1 < 1.45 or rr2 < 2.9:
            return None
        if confidence < self.settings.min_confidence:
            return None

        now = utcnow()
        return {
            "bot": "AURUM-X",
            "timestamp_utc": now.isoformat(),
            "pair": candidate["pair"],
            "direction": direction,
            "confidence": min(95, confidence),
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "risk_reward_tp1": round(rr1, 2),
            "risk_reward_tp2": round(rr2, 2),
            "news_bias": data.get("news_bias", news_meta.get("news_bias", "Neutral")),
            "news_score": data.get(
                "news_score",
                {
                    "base": news_meta.get("base_score", 50),
                    "quote": news_meta.get("quote_score", 50),
                },
            ),
            "tf_alignment": data.get("tf_alignment", candidate["tf_alignment"]),
            "key_levels": data.get("key_levels", candidate["key_levels"]),
            "reasoning": str(data.get("reasoning", candidate["reasoning"]))[:1200],
            "valid_until": (now + timedelta(hours=self.settings.signal_valid_hours)).isoformat(),
            "risk_note": "Risiko 1-2% per trade. Bukan jaminan profit.",
        }

    def fallback(self, candidate: dict, news_meta: dict) -> dict:
        now = utcnow()
        return {
            "bot": "AURUM-X",
            "timestamp_utc": now.isoformat(),
            "pair": candidate["pair"],
            "direction": candidate["direction"],
            "confidence": min(75, int(candidate["confidence"])),
            "entry": float(candidate["entry"]),
            "sl": float(candidate["sl"]),
            "tp1": float(candidate["tp1"]),
            "tp2": float(candidate["tp2"]),
            "risk_reward_tp1": float(candidate["risk_reward_tp1"]),
            "risk_reward_tp2": float(candidate["risk_reward_tp2"]),
            "news_bias": news_meta.get("news_bias", "Neutral"),
            "news_score": {
                "base": news_meta.get("base_score", 50),
                "quote": news_meta.get("quote_score", 50),
            },
            "tf_alignment": candidate["tf_alignment"],
            "key_levels": candidate["key_levels"],
            "reasoning": (
                "Rule-engine fallback: AI tidak tersedia atau output tidak valid. "
                + candidate["reasoning"]
            )[:1200],
            "valid_until": (now + timedelta(hours=self.settings.signal_valid_hours)).isoformat(),
            "risk_note": "Risiko 1-2% per trade. Bukan jaminan profit.",
        }

    @staticmethod
    def no_signal(pair: str, reasoning: str) -> dict:
        now = utcnow()
        return {
            "bot": "AURUM-X",
            "timestamp_utc": now.isoformat(),
            "pair": pair,
            "direction": "NO_SIGNAL",
            "confidence": 0,
            "entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "risk_reward_tp1": None,
            "risk_reward_tp2": None,
            "news_bias": "Neutral",
            "news_score": {"base": 50, "quote": 50},
            "tf_alignment": {},
            "key_levels": {},
            "reasoning": reasoning[:1200],
            "valid_until": now.isoformat(),
            "risk_note": "Risiko 1-2% per trade. Bukan jaminan profit.",
        }
