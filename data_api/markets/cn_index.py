"""China-Index market adapter (akshare sina)."""

from __future__ import annotations

import re

import akshare as ak
import pandas as pd

from data_api import cache
from data_api.markets._common import finalize_daily, retry, to_date_iso

name = "cn_index"

_BARE = re.compile(r"^\d{6}$")
_LOWER = re.compile(r"^(sh|sz)(\d{6})$")
_DOT = re.compile(r"^(SHSE|SZSE)\.(\d{6})$", re.IGNORECASE)

_ALIASES = {
    "HS300": "sh000300",
    "CSI300": "sh000300",
    "SSE50": "sh000016",
    "ZZ500": "sh000905",
    "CSI500": "sh000905",
    "ZZ1000": "sh000852",
    "CSI1000": "sh000852",
    "SZ50": "sz399330",
    "CYB": "sz399006",  # 创业板指
    "ZXB": "sz399005",  # 中小板指
    "SHCOMP": "sh000001",
    "SZCOMP": "sz399001",
}


def _native(symbol: str) -> str:
    s = symbol.strip()
    if s.upper() in _ALIASES:
        return _ALIASES[s.upper()]
    if _LOWER.match(s.lower()):
        return s.lower()
    m = _DOT.match(s)
    if m:
        prefix = "sh" if m.group(1).upper() == "SHSE" else "sz"
        return f"{prefix}{m.group(2)}"
    if _BARE.match(s):
        # default heuristic: 000xxx → sh, 399xxx → sz
        return ("sh" if s.startswith("000") else "sz") + s
    raise ValueError(f"unrecognised index symbol: {symbol!r}")


def to_canonical(symbol: str) -> str:
    n = _native(symbol)
    code = n[2:]
    return f"{'SHSE' if n.startswith('sh') else 'SZSE'}.{code}"


def to_native(symbol: str) -> str:
    return _native(symbol)


def fetch_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    native = _native(symbol)
    code = native[2:]
    freq = "daily_index"
    ts_col = "date"

    start_iso, end_iso = to_date_iso(start), to_date_iso(end)

    cached = cache.load(code, freq, ts_col)
    if cached is not None:
        sliced = cache.slice_range(cached, ts_col, start_iso, end_iso)
        if not sliced.empty and cached[ts_col].min() <= start_iso and cached[ts_col].max() >= end_iso:
            return sliced

    raw = retry(lambda: ak.stock_zh_index_daily(symbol=native))
    df = finalize_daily(raw)
    cache.save(code, freq, df, ts_col)
    return cache.slice_range(df, ts_col, start_iso, end_iso)


def fetch_minute(*args, **kwargs):
    raise NotImplementedError("Index minute bars not implemented")


def universe(name: str) -> list[str]:
    """Return the alias keys we know about plus a couple of raw codes."""
    return sorted(_ALIASES.keys())
