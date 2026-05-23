"""FX / currency adapter — Bank-of-China historical mid-prices via sina."""

from __future__ import annotations

import akshare as ak
import pandas as pd

from data_api import cache
from data_api.markets._common import (
    DAILY_COLS,
    retry,
    to_date_compact,
    to_date_iso,
)

name = "fx"

# Sina expects Chinese currency names. Provide friendly aliases.
_ALIASES = {
    "USD": "美元",
    "USDCNY": "美元",
    "EUR": "欧元",
    "EURCNY": "欧元",
    "JPY": "日元",
    "JPYCNY": "日元",
    "GBP": "英镑",
    "GBPCNY": "英镑",
    "HKD": "港币",
    "HKDCNY": "港币",
    "AUD": "澳元",
    "CAD": "加拿大元",
    "CHF": "瑞士法郎",
    "SGD": "新加坡元",
    "KRW": "韩元",
}


def _native(symbol: str) -> str:
    s = symbol.strip().upper()
    if s in _ALIASES:
        return _ALIASES[s]
    # Already in Chinese
    return symbol.strip()


def to_canonical(symbol: str) -> str:
    return f"FX.{symbol.strip().upper()}"


def to_native(symbol: str) -> str:
    return _native(symbol)


def fetch_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    native = _native(symbol)
    freq = "daily_fx"
    ts_col = "date"

    start_iso, end_iso = to_date_iso(start), to_date_iso(end)
    start_c, end_c = to_date_compact(start), to_date_compact(end)

    cached = cache.load(symbol.upper(), freq, ts_col)
    if cached is not None:
        sliced = cache.slice_range(cached, ts_col, start_iso, end_iso)
        if not sliced.empty and cached[ts_col].min() <= start_iso and cached[ts_col].max() >= end_iso:
            return sliced

    raw = retry(
        lambda: ak.currency_boc_sina(symbol=native, start_date=start_c, end_date=end_c)
    )
    # Bank-of-China table — use 中行折算价 as close, 央行中间价 if available.
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw["日期"]).dt.date.astype(str)
    mid = raw.get("央行中间价") if "央行中间价" in raw.columns else raw.get("中行折算价")
    out["close"] = pd.to_numeric(mid, errors="coerce")
    out["open"] = out["close"]
    out["high"] = out["close"]
    out["low"] = out["close"]
    out["volume"] = 0.0
    out["amount"] = 0.0
    out["pct_chg"] = (out["close"].pct_change() * 100).fillna(0.0)
    out["turnover_rate"] = 0.0
    out = out[DAILY_COLS].dropna(subset=["close"]).reset_index(drop=True)
    cache.save(symbol.upper(), freq, out, ts_col)
    return cache.slice_range(out, ts_col, start_iso, end_iso)


def fetch_minute(*args, **kwargs):
    raise NotImplementedError("FX minute bars not implemented")


def universe(name: str) -> list[str]:
    return sorted({k for k in _ALIASES if k.endswith("CNY") or len(k) == 3})
