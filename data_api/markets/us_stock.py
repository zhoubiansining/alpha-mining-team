"""US-stock market adapter (akshare sina)."""

from __future__ import annotations

from functools import lru_cache

import akshare as ak
import pandas as pd

from data_api import cache
from data_api.markets._common import finalize_daily, retry, to_date_iso

name = "us_stock"


def _norm(symbol: str) -> str:
    """Accept 'AAPL', 'us.AAPL', 'NASDAQ:AAPL' — return upper-case ticker."""
    s = symbol.strip().upper()
    for prefix in ("US.", "US:", "NASDAQ:", "NYSE:", "AMEX:"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    return s


def to_canonical(symbol: str) -> str:
    return f"US.{_norm(symbol)}"


def to_native(symbol: str) -> str:
    return _norm(symbol)


def fetch_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    sym = _norm(symbol)
    freq = f"daily_us_{adjust or 'raw'}"
    ts_col = "date"

    start_iso, end_iso = to_date_iso(start), to_date_iso(end)

    cached = cache.load(sym, freq, ts_col)
    if cached is not None:
        sliced = cache.slice_range(cached, ts_col, start_iso, end_iso)
        if not sliced.empty and cached[ts_col].min() <= start_iso and cached[ts_col].max() >= end_iso:
            return sliced

    raw = retry(lambda: ak.stock_us_daily(symbol=sym, adjust=adjust or "qfq"))
    # sina returns date as Timestamp; volume only (no amount)
    df = finalize_daily(raw)
    cache.save(sym, freq, df, ts_col)
    return cache.slice_range(df, ts_col, start_iso, end_iso)


def fetch_minute(*args, **kwargs):
    raise NotImplementedError("US-stock minute bars not supported (akshare EM blocked)")


@lru_cache(maxsize=1)
def _us_codes() -> pd.DataFrame:
    return ak.get_us_stock_name()


def universe(name: str) -> list[str]:
    """``name`` is currently a no-op — returns the full list of US tickers."""
    df = _us_codes()
    if "symbol" in df.columns:
        return df["symbol"].astype(str).str.upper().tolist()
    for col in df.columns:
        if df[col].dtype == object:
            return df[col].astype(str).str.upper().tolist()
    return []
