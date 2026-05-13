"""HK-stock market adapter (akshare sina)."""

from __future__ import annotations

import re
from functools import lru_cache

import akshare as ak
import pandas as pd

from data_api import cache
from data_api.markets._common import finalize_daily, retry, to_date_iso

name = "hk_stock"

_HK_CODE = re.compile(r"^\d{4,5}$")
_HK_PREFIXED = re.compile(r"^(HK|HKEX)[.:]?(\d{4,5})$", re.IGNORECASE)


def _bare(symbol: str) -> str:
    s = symbol.strip()
    m = _HK_PREFIXED.match(s)
    if m:
        s = m.group(2)
    if _HK_CODE.match(s):
        return s.zfill(5)
    raise ValueError(f"unrecognised HK symbol: {symbol!r}")


def to_canonical(symbol: str) -> str:
    return f"HK.{_bare(symbol)}"


def to_native(symbol: str) -> str:
    # akshare HK daily expects the raw 5-digit form, leading zeros included
    return _bare(symbol)


def fetch_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    sym = _bare(symbol)
    freq = f"daily_hk_{adjust or 'raw'}"
    ts_col = "date"

    start_iso, end_iso = to_date_iso(start), to_date_iso(end)

    cached = cache.load(sym, freq, ts_col)
    if cached is not None:
        sliced = cache.slice_range(cached, ts_col, start_iso, end_iso)
        if not sliced.empty and cached[ts_col].min() <= start_iso and cached[ts_col].max() >= end_iso:
            return sliced

    raw = retry(lambda: ak.stock_hk_daily(symbol=sym, adjust=adjust or "qfq"))
    df = finalize_daily(raw)
    cache.save(sym, freq, df, ts_col)
    return cache.slice_range(df, ts_col, start_iso, end_iso)


def fetch_minute(*args, **kwargs):
    raise NotImplementedError("HK-stock minute bars not supported (akshare EM blocked)")


@lru_cache(maxsize=1)
def _hk_codes() -> pd.DataFrame:
    return ak.stock_hk_spot()


def universe(name: str) -> list[str]:
    df = _hk_codes()
    if "symbol" in df.columns:
        return df["symbol"].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()
