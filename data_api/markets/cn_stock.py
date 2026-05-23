"""A-share stock market adapter (akshare)."""

from __future__ import annotations

import re
from datetime import date
from functools import lru_cache

import akshare as ak  # noqa: E402
import pandas as pd

from data_api import cache
from data_api.markets._common import (
    finalize_daily,
    finalize_minute,
    retry,
    to_date_compact,
    to_date_iso,
)

name = "cn_stock"

_BARE_CODE = re.compile(r"^\d{6}$")
_PREFIXED = re.compile(r"^(SHSE|SZSE|SH|SZ)[.:]?(\d{6})$", re.IGNORECASE)
_LOWER = re.compile(r"^(sh|sz)(\d{6})$")


def _bare(symbol: str) -> str:
    s = symbol.strip()
    if _BARE_CODE.match(s):
        return s
    m = _PREFIXED.match(s)
    if m:
        return m.group(2)
    m = _LOWER.match(s.lower())
    if m:
        return m.group(2)
    raise ValueError(f"unrecognised A-share symbol: {symbol!r}")


def _exchange(code: str) -> str:
    head = code[0]
    if head in {"6", "9"} or code.startswith("688"):
        return "SH"
    return "SZ"


def to_canonical(symbol: str) -> str:
    code = _bare(symbol)
    ex = _exchange(code)
    return f"{'SHSE' if ex == 'SH' else 'SZSE'}.{code}"


def to_native(symbol: str) -> str:
    code = _bare(symbol)
    return f"{_exchange(code).lower()}{code}"


_INDEX_NAME_TO_CODE = {
    "HS300": "000300",
    "ZZ500": "000905",
    "ZZ1000": "000852",
    "SSE50": "000016",
    "CSI300": "000300",
    "CSI500": "000905",
    "CSI1000": "000852",
}


@lru_cache(maxsize=1)
def _calendar() -> list[str]:
    df = ak.tool_trade_date_hist_sina()
    days = pd.to_datetime(df["trade_date"]).dt.date
    return [d.isoformat() for d in sorted(days)]


def trade_days_in_range(start: str, end: str) -> list[str]:
    return [d for d in _calendar() if start <= d <= end]


def minute_window_for_recent_days(n_days: int = 5) -> tuple[str, str]:
    today = date.today().isoformat()
    past = [d for d in _calendar() if d <= today][-n_days:]
    if not past:
        raise RuntimeError("calendar empty")
    return f"{past[0]} 09:00:00", f"{past[-1]} 16:00:00"


def universe(name: str) -> list[str]:
    key = name.upper()
    if key in {"A-SHARE", "ASHARE", "ALL"}:
        spot = ak.stock_zh_a_spot_em()
        return sorted(spot["代码"].astype(str).tolist())
    code = _INDEX_NAME_TO_CODE.get(key)
    if code is None:
        raise ValueError(f"unknown CN universe {name!r}")
    df = ak.index_stock_cons_csindex(symbol=code)
    return df["成分券代码"].astype(str).tolist()


def fetch_minute(
    symbol: str, start: str, end: str, period: str = "1", adjust: str = "qfq"
) -> pd.DataFrame:
    code = _bare(symbol)
    sina_sym = to_native(symbol)
    freq = f"min{period}_{adjust or 'raw'}"
    ts_col = "datetime"

    cached = cache.load(code, freq, ts_col)
    if cached is not None:
        sliced = cache.slice_range(cached, ts_col, start, end)
        if not sliced.empty:
            if pd.to_datetime(cached[ts_col]).min() <= pd.to_datetime(start) and pd.to_datetime(
                cached[ts_col]
            ).max() >= pd.to_datetime(end):
                return sliced

    try:
        raw = retry(
            lambda: ak.stock_zh_a_minute(
                symbol=sina_sym, period=str(period), adjust=adjust or ""
            )
        )
        raw = raw.rename(columns={"day": "datetime"})
        df = finalize_minute(raw)
    except Exception:
        raw = retry(
            lambda: ak.stock_zh_a_hist_min_em(
                symbol=code,
                period=str(period),
                start_date=start,
                end_date=end,
                adjust=adjust or "",
            )
        )
        raw = raw.rename(
            columns={
                "时间": "datetime",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "均价": "vwap",
            }
        )
        df = finalize_minute(raw)

    cache.save(code, freq, df, ts_col)
    return cache.slice_range(df, ts_col, start, end)


def fetch_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    code = _bare(symbol)
    sina_sym = to_native(symbol)
    freq = f"daily_{adjust or 'raw'}"
    ts_col = "date"

    start_iso, end_iso = to_date_iso(start), to_date_iso(end)
    start_c, end_c = to_date_compact(start), to_date_compact(end)

    cached = cache.load(code, freq, ts_col)
    if cached is not None:
        sliced = cache.slice_range(cached, ts_col, start_iso, end_iso)
        if not sliced.empty and cached[ts_col].min() <= start_iso and cached[ts_col].max() >= end_iso:
            return sliced

    try:
        raw = retry(
            lambda: ak.stock_zh_a_daily(
                symbol=sina_sym, start_date=start_c, end_date=end_c, adjust=adjust or "qfq"
            )
        )
        raw = raw.rename(columns={"turnover": "turnover_rate"})
        df = finalize_daily(raw)
    except Exception:
        raw = retry(
            lambda: ak.stock_zh_a_hist(
                symbol=code, period="daily", start_date=start_c, end_date=end_c, adjust=adjust
            )
        )
        raw = raw.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "涨跌幅": "pct_chg",
                "换手率": "turnover_rate",
            }
        )
        df = finalize_daily(raw)

    cache.save(code, freq, df, ts_col)
    return cache.slice_range(df, ts_col, start_iso, end_iso)
