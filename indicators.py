from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


# ============================================================
# INDIKATOR DASAR
# ============================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(series, fast) - ema(series, slow)
    signal_line = ema(line, signal)
    hist = line - signal_line
    return line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def _last(series: pd.Series, default: float = float("nan")) -> float:
    try:
        value = series.iloc[-1]
        return float(value) if pd.notna(value) else default
    except Exception:
        return default


# ============================================================
# STRUKTUR & LEVEL
# ============================================================

def swing_points(df: pd.DataFrame, left: int = 3, right: int = 3):
    highs = df["high"].values
    lows = df["low"].values
    swing_high = np.zeros(len(df), dtype=bool)
    swing_low = np.zeros(len(df), dtype=bool)

    for i in range(left, len(df) - right):
        window_high = highs[i - left : i + right + 1]
        window_low = lows[i - left : i + right + 1]
        if highs[i] == window_high.max():
            swing_high[i] = True
        if lows[i] == window_low.min():
            swing_low[i] = True

    return swing_high, swing_low


def last_swings(df: pd.DataFrame, n: int = 4):
    sh, sl = swing_points(df)
    swing_highs = df["high"][sh].tail(n).tolist()
    swing_lows = df["low"][sl].tail(n).tolist()
    return swing_highs, swing_lows


def sr_levels(df: pd.DataFrame, lookback: int = 200, tolerance_factor: float = 0.5):
    """
    Support/resistance sederhana dengan clustering dan jumlah touch.
    Return: (support_dict, resistance_dict)
    """
    data = df.tail(lookback).copy()
    if len(data) < 30:
        price = float(data["close"].iloc[-1])
        return (
            {"level": price * 0.995, "touches": 0},
            {"level": price * 1.005, "touches": 0},
        )

    price = float(data["close"].iloc[-1])
    atr_value = _last(atr(data), price * 0.001)
    tolerance = max(atr_value * tolerance_factor, price * 0.0002)

    highs, lows = last_swings(data, 12)
    candidates = [(float(x), "resistance") for x in highs] + [
        (float(x), "support") for x in lows
    ]

    levels: list[dict[str, Any]] = []
    for level, kind in candidates:
        merged = False
        for item in levels:
            if abs(item["level"] - level) <= tolerance:
                total = item["touches"] + 1
                item["level"] = (item["level"] * item["touches"] + level) / total
                item["touches"] = total
                merged = True
                break
        if not merged:
            levels.append({"level": level, "touches": 1, "kind": kind})

    supports = [x for x in levels if x["level"] < price]
    resistances = [x for x in levels if x["level"] > price]

    support = max(supports, key=lambda x: x["level"]) if supports else {
        "level": price - 2 * atr_value,
        "touches": 0,
    }
    resistance = min(resistances, key=lambda x: x["level"]) if resistances else {
        "level": price + 2 * atr_value,
        "touches": 0,
    }
    return support, resistance


def detect_fvg(df: pd.DataFrame, lookback: int = 60) -> Optional[dict]:
    """Fair Value Gap terbaru yang belum terisi."""
    data = df.tail(lookback)
    for i in range(len(data) - 1, 1, -1):
        if data["low"].iloc[i] > data["high"].iloc[i - 2]:
            return {
                "type": "bullish",
                "bottom": float(data["high"].iloc[i - 2]),
                "top": float(data["low"].iloc[i]),
            }
        if data["high"].iloc[i] < data["low"].iloc[i - 2]:
            return {
                "type": "bearish",
                "bottom": float(data["high"].iloc[i]),
                "top": float(data["low"].iloc[i - 2]),
            }
    return None


def detect_order_block(df: pd.DataFrame, lookback: int = 60) -> Optional[dict]:
    """Order block sederhana: candle terakhir sebelum impulse."""
    data = df.tail(lookback)
    for i in range(len(data) - 2, 0, -1):
        current = data.iloc[i]
        prev = data.iloc[i - 1]

        if current["close"] > prev["high"] and prev["close"] < prev["open"]:
            return {
                "type": "bullish",
                "low": float(prev["low"]),
                "high": float(prev["high"]),
            }
        if current["close"] < prev["low"] and prev["close"] > prev["open"]:
            return {
                "type": "bearish",
                "low": float(prev["low"]),
                "high": float(prev["high"]),
            }
    return None


def candle_patterns(df: pd.DataFrame) -> list[str]:
    if len(df) < 3:
        return []

    last = df.iloc[-1]
    prev = df.iloc[-2]
    patterns = []

    body = abs(last["close"] - last["open"])
    upper_wick = last["high"] - max(last["close"], last["open"])
    lower_wick = min(last["close"], last["open"]) - last["low"]

    if last["close"] > last["open"] and prev["close"] < prev["open"]:
        if last["close"] >= prev["open"] and last["open"] <= prev["close"]:
            patterns.append("bullish_engulfing")
    if last["close"] < last["open"] and prev["close"] > prev["open"]:
        if last["close"] <= prev["open"] and last["open"] >= prev["close"]:
            patterns.append("bearish_engulfing")

    if body > 0:
        if lower_wick > 2 * body and upper_wick < body:
            patterns.append("bullish_pin")
        if upper_wick > 2 * body and lower_wick < body:
            patterns.append("bearish_pin")

    return patterns


# ============================================================
# ANALISIS PER TIMEFRAME
# ============================================================

def analyze_timeframe(df: pd.DataFrame, tf: str) -> Optional[dict]:
    if df is None or len(df) < 60:
        return None

    close = df["close"]
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    e200 = ema(close, 200)
    rsi_series = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    atr_series = atr(df, 14)
    bb_upper, bb_mid, bb_lower = bollinger(close, 20, 2.0)

    last_close = _last(close)
    last_e20 = _last(e20)
    last_e50 = _last(e50)
    last_e200 = _last(e200)

    bias = "Neutral"
    if pd.notna(last_e20) and pd.notna(last_e50) and pd.notna(last_e200):
        if last_close > last_e50 and last_e20 > last_e50:
            bias = "Bullish"
        elif last_close < last_e50 and last_e20 < last_e50:
            bias = "Bearish"

    swing_highs, swing_lows = last_swings(df, 4)
    structure = "Ranging"
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        if swing_highs[-1] > swing_highs[-2] and swing_lows[-1] > swing_lows[-2]:
            structure = "HH/HL"
        elif swing_highs[-1] < swing_highs[-2] and swing_lows[-1] < swing_lows[-2]:
            structure = "LH/LL"

    support, resistance = sr_levels(df)
    fvg = detect_fvg(df)
    ob = detect_order_block(df)

    candles = []
    for idx, row in df.tail(3).iterrows():
        candles.append(
            {
                "time": idx.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        )

    return {
        "tf": tf,
        "close": last_close,
        "open": float(df["open"].iloc[-1]),
        "high": float(df["high"].iloc[-1]),
        "low": float(df["low"].iloc[-1]),
        "ema20": last_e20,
        "ema50": last_e50,
        "ema200": last_e200,
        "rsi": _last(rsi_series),
        "macd": _last(macd_line),
        "macd_signal": _last(signal_line),
        "macd_hist": _last(hist),
        "atr": _last(atr_series),
        "bb_upper": _last(bb_upper),
        "bb_mid": _last(bb_mid),
        "bb_lower": _last(bb_lower),
        "bias": bias,
        "structure": structure,
        "swing_highs": [float(x) for x in swing_highs],
        "swing_lows": [float(x) for x in swing_lows],
        "support": float(support["level"]),
        "support_touches": int(support["touches"]),
        "resistance": float(resistance["level"]),
        "resistance_touches": int(resistance["touches"]),
        "fvg": fvg,
        "order_block": ob,
        "patterns": candle_patterns(df),
        "last_candles": candles,
    }


# ============================================================
# TRIGGER & CANDIDATE
# ============================================================

def trigger_m5(df: pd.DataFrame, direction: str) -> bool:
    if df is None or len(df) < 30:
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if direction == "BUY":
        breakout = last["close"] > df["high"].iloc[-21:-1].max()
        engulf = (
            last["close"] > last["open"]
            and prev["close"] < prev["open"]
            and last["close"] >= prev["open"]
        )
        return bool(breakout or engulf)

    breakout = last["close"] < df["low"].iloc[-21:-1].min()
    engulf = (
        last["close"] < last["open"]
        and prev["close"] > prev["open"]
        and last["close"] <= prev["open"]
    )
    return bool(breakout or engulf)


def build_snapshot(bundle: dict[str, pd.DataFrame]) -> dict[str, Any]:
    snapshot = {}
    keep = (
        "close",
        "ema20",
        "ema50",
        "ema200",
        "rsi",
        "macd_hist",
        "atr",
        "bb_upper",
        "bb_lower",
        "bias",
        "structure",
        "support",
        "support_touches",
        "resistance",
        "resistance_touches",
        "fvg",
        "order_block",
        "patterns",
        "last_candles",
    )
    for tf, df in bundle.items():
        analysis = analyze_timeframe(df, tf)
        if analysis:
            snapshot[tf] = {k: analysis[k] for k in keep if k in analysis}
    return snapshot


def build_candidate(
    pair: str,
    bundle: dict[str, pd.DataFrame],
    news_filter=None,
) -> Optional[dict]:
    """
    Rule engine deterministik. AI hanya memvalidasi kandidat yang lolos.
    Return None jika tidak ada kandidat.
    """
    required = ["D1", "H4", "H1", "M30", "M15", "M5"]
    analyses = {}
    for tf in required:
        df = bundle.get(tf)
        analysis = analyze_timeframe(df, tf) if df is not None else None
        if analysis is None:
            return None
        analyses[tf] = analysis

    d1 = analyses["D1"]
    h4 = analyses["H4"]
    h1 = analyses["H1"]
    m30 = analyses["M30"]
    m15 = analyses["M15"]
    m5 = analyses["M5"]

    # Aturan emas: D1 & H4 wajib searah.
    if d1["bias"] == "Neutral" or h4["bias"] == "Neutral":
        return None
    if d1["bias"] != h4["bias"]:
        return None

    direction = d1["bias"]
    confluence = 0
    reasons = []

    # H1 momentum
    if h1["bias"] == direction:
        confluence += 1
        reasons.append(f"H1 {direction.lower()} searah")
    elif h1["bias"] != "Neutral":
        return None

    # M30 konfirmasi
    if direction == "BUY":
        if m30["rsi"] is not None and 38 <= m30["rsi"] <= 68:
            confluence += 1
            reasons.append("RSI M30 sehat")
        if m30["macd_hist"] > 0:
            confluence += 1
            reasons.append("MACD M30 positif")
    else:
        if m30["rsi"] is not None and 32 <= m30["rsi"] <= 62:
            confluence += 1
            reasons.append("RSI M30 sehat")
        if m30["macd_hist"] < 0:
            confluence += 1
            reasons.append("MACD M30 negatif")

    # M15 trigger
    if direction == "BUY":
        if m15["close"] > m15["ema20"] and m15["close"] > m15["open"]:
            confluence += 1
            reasons.append("M15 bullish trigger")
    else:
        if m15["close"] < m15["ema20"] and m15["close"] < m15["open"]:
            confluence += 1
            reasons.append("M15 bearish trigger")

    # M5 trigger
    if trigger_m5(bundle["M5"], direction):
        confluence += 1
        reasons.append("M5 breakout struktur")

    # Level touch
    if direction == "BUY" and m15["support_touches"] >= 2:
        confluence += 1
        reasons.append("Support M15 teruji")
    if direction == "SELL" and m15["resistance_touches"] >= 2:
        confluence += 1
        reasons.append("Resistance M15 teruji")

    if confluence < 4:
        return None

    entry = float(m5["close"])
    atr5 = float(m5["atr"]) if m5["atr"] and m5["atr"] > 0 else entry * 0.0005
    m5_df = bundle["M5"]

    if direction == "BUY":
        swing_low = float(m5_df["low"].tail(10).min())
        sl = min(swing_low, entry - atr5 * 1.5) - atr5 * 0.2
        risk = entry - sl
        if risk <= 0:
            return None

        tp1 = float(m15["resistance"])
        if tp1 <= entry or rr(entry, sl, tp1) < 1.5:
            tp1 = entry + risk * 1.5

        tp2 = float(h1["resistance"])
        if tp2 <= tp1 or rr(entry, sl, tp2) < 3.0:
            tp2 = entry + risk * 3.0

    else:
        swing_high = float(m5_df["high"].tail(10).max())
        sl = max(swing_high, entry + atr5 * 1.5) + atr5 * 0.2
        risk = sl - entry
        if risk <= 0:
            return None

        tp1 = float(m15["support"])
        if tp1 >= entry or rr(entry, sl, tp1) < 1.5:
            tp1 = entry - risk * 1.5

        tp2 = float(h1["support"])
        if tp2 >= tp1 or rr(entry, sl, tp2) < 3.0:
            tp2 = entry - risk * 3.0

    rr1 = rr(entry, sl, tp1)
    rr2 = rr(entry, sl, tp2)

    if rr1 < 1.45 or rr2 < 2.9:
        return None

    all_aligned = all(
        analyses[tf]["bias"] == direction for tf in ["D1", "H4", "H1"]
    )
    confidence = min(92, 68 + confluence * 4 + (6 if all_aligned else 0))
    if confidence < 70:
        return None

    # News gate
    news_meta = {}
    if news_filter is not None:
        allowed, reason, meta = news_filter.gate(pair, direction)
        news_meta = meta
        if not allowed:
            return None

    return {
        "pair": pair,
        "direction": direction,
        "confidence": confidence,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk_reward_tp1": rr1,
        "risk_reward_tp2": rr2,
        "tf_alignment": {tf: analyses[tf]["bias"] for tf in required},
        "key_levels": {
            "support": float(h1["support"]),
            "resistance": float(h1["resistance"]),
        },
        "reasoning": (
            f"Bias {direction} dari D1 dan H4. "
            + "; ".join(reasons)
            + f". Entry {entry:.5f}, SL {sl:.5f}, TP1 {tp1:.5f}, TP2 {tp2:.5f}."
        ),
        "news_meta": news_meta,
    }
