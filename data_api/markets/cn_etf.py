"""China-ETF market adapter (akshare sina)."""

from __future__ import annotations

import re
from functools import lru_cache

import akshare as ak
import pandas as pd

from data_api import cache
from data_api.markets._common import finalize_daily, retry, to_date_iso

name = "cn_etf"

_BARE = re.compile(r"^\d{6}$")
_LOWER = re.compile(r"^(sh|sz)(\d{6})$")
_PREFIXED = re.compile(r"^(SHSE|SZSE)\.(\d{6})$", re.IGNORECASE)


def _bare(symbol: str) -> str:
    s = symbol.strip()
    if _BARE.match(s):
        return s
    m = _PREFIXED.match(s) or _LOWER.match(s.lower())
    if m:
        return m.group(2)
    raise ValueError(f"unrecognised ETF symbol: {symbol!r}")


def _exch(code: str) -> str:
    return "sh" if code[0] == "5" else "sz"


def to_canonical(symbol: str) -> str:
    code = _bare(symbol)
    return f"{'SHSE' if _exch(code) == 'sh' else 'SZSE'}.{code}"


def to_native(symbol: str) -> str:
    code = _bare(symbol)
    return f"{_exch(code)}{code}"


def fetch_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    code = _bare(symbol)
    native = to_native(symbol)
    freq = "daily_etf"
    ts_col = "date"

    start_iso, end_iso = to_date_iso(start), to_date_iso(end)

    cached = cache.load(code, freq, ts_col)
    if cached is not None:
        sliced = cache.slice_range(cached, ts_col, start_iso, end_iso)
        if not sliced.empty and cached[ts_col].min() <= start_iso and cached[ts_col].max() >= end_iso:
            return sliced

    raw = retry(lambda: ak.fund_etf_hist_sina(symbol=native))
    df = finalize_daily(raw)
    cache.save(code, freq, df, ts_col)
    return cache.slice_range(df, ts_col, start_iso, end_iso)


def fetch_minute(*args, **kwargs):
    raise NotImplementedError("ETF minute bars not supported")


@lru_cache(maxsize=1)
def _etf_list() -> pd.DataFrame:
    return ak.fund_etf_category_sina(symbol="ETF基金")


def universe(name: str) -> list[str]:
    df = _etf_list()
    # symbol column comes as 'sh510300' style; strip exchange prefix
    col = "代码" if "代码" in df.columns else df.columns[0]
    raw = df[col].astype(str)
    return [s[2:] if s[:2].lower() in {"sh", "sz"} else s for s in raw]
