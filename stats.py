from __future__ import annotations

import math


def _fmt_pf(value: float) -> str:
    if math.isinf(value):
        return "∞"
    return f"{value:.2f}"


def format_stats(stats: dict, title: str = "STATISTIK AURUM-X") -> str:
    return (
        f"📊 <b>{title}</b>\n\n"
        f"Total sinyal : <b>{stats['total']}</b>\n"
        f"Closed       : <b>{stats['closed']}</b>\n"
        f"TP1 / TP2    : <b>{stats['tp1']} / {stats['tp2']}</b>\n"
        f"SL           : <b>{stats['sl']}</b>\n"
        f"Expired      : <b>{stats['expired']}</b>\n"
        f"Cancelled    : <b>{stats['cancelled']}</b>\n"
        f"Pending      : <b>{stats['pending']}</b>\n\n"
        f"Winrate      : <b>{stats['winrate']:.1f}%</b>\n"
        f"Total pips   : <b>{stats['total_pips']:+.1f}</b>\n"
        f"Profit factor: <b>{_fmt_pf(stats['profit_factor'])}</b>\n"
        f"Avg R        : <b>{stats['avg_r']:+.2f}R</b>\n\n"
        f"⚠️ Winrate nyata hanya valid setelah minimal 100 trade. "
        f"Bukan jaminan profit."
    )


def format_weekly(stats: dict) -> str:
    return format_stats(stats, "STATISTIK 7 HARI TERAKHIR")
