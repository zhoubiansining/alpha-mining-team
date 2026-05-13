"""China futures market adapter (akshare sina main contract)."""

from __future__ import annotations

import akshare as ak
import pandas as pd

from data_api import cache
from data_api.markets._common import finalize_daily, retry, to_date_compact, to_date_iso

name = "cn_future"

_CN_COLS = {
    "日期": "date",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量": "volume",
    "持仓量": "open_interest",
    "动态结算价": "settle",
}


def to_canonical(symbol: str) -> str:
    s = symbol.strip().upper()
    if not s.endswith("0"):
        s += "0"
    return f"CN.FUT.{s}"


def to_native(symbol: str) -> str:
    s = symbol.strip().upper()
    if not s.endswith("0"):
        s += "0"
    return s


def fetch_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    sym = to_native(symbol)
    freq = "daily_future"
    ts_col = "date"

    start_iso, end_iso = to_date_iso(start), to_date_iso(end)
    start_c, end_c = to_date_compact(start), to_date_compact(end)

    cached = cache.load(sym, freq, ts_col)
    if cached is not None:
        sliced = cache.slice_range(cached, ts_col, start_iso, end_iso)
        if not sliced.empty and cached[ts_col].min() <= start_iso and cached[ts_col].max() >= end_iso:
            return sliced

    raw = retry(
        lambda: ak.futures_main_sina(symbol=sym, start_date=start_c, end_date=end_c)
    )
    raw = raw.rename(columns=_CN_COLS)
    # No 'amount' column; use volume*settle as a coarse stand-in if you need it
    df = finalize_daily(raw)
    cache.save(sym, freq, df, ts_col)
    return cache.slice_range(df, ts_col, start_iso, end_iso)


def fetch_minute(*args, **kwargs):
    raise NotImplementedError("future minute bars not implemented")


def universe(name: str) -> list[str]:
    """Common main-contract tickers — full list lives on sina but is huge."""
    return [
        "IF0",  # CSI300 future
        "IC0",  # CSI500 future
        "IH0",  # SSE50 future
        "IM0",  # CSI1000 future
        "T0",   # 10y T-bond
        "TF0",  # 5y T-bond
        "RB0",  # rebar
        "HC0",  # hot-rolled coil
        "I0",   # iron ore
        "J0",   # coke
        "CU0",  # copper
        "AL0",  # aluminum
        "ZN0",  # zinc
        "AU0",  # gold
        "AG0",  # silver
        "SC0",  # crude oil
        "M0",   # soybean meal
        "Y0",   # soybean oil
        "P0",   # palm oil
        "C0",   # corn
    ]
