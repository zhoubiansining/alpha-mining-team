"""Thin façade over the per-market adapter registry.

Kept so older callers (and tests) can do `from data_api import source`
without knowing the market dispatch internals.
"""

from __future__ import annotations

from data_api import markets
from data_api.markets.cn_stock import (
    minute_window_for_recent_days,
    trade_days_in_range,
    universe as _cn_universe,
)

__all__ = [
    "minute_window_for_recent_days",
    "trade_days_in_range",
    "trade_calendar",
    "index_constituents",
    "universe_codes",
    "fetch_minute",
    "fetch_daily",
]


def trade_calendar() -> list[str]:
    from data_api.markets.cn_stock import _calendar

    return _calendar()


def universe_codes(name: str) -> list[str]:
    """A-share universe lookup (kept for backwards compat with v1 tests)."""
    return _cn_universe(name)


def index_constituents(name: str):  # pragma: no cover - retained for callers
    import akshare as ak

    code_map = {"HS300": "000300", "ZZ500": "000905", "ZZ1000": "000852", "SSE50": "000016"}
    code = code_map.get(name.upper())
    if code is None:
        raise ValueError(name)
    df = ak.index_stock_cons_csindex(symbol=code)
    return df.rename(
        columns={
            "成分券代码": "code",
            "成分券名称": "name",
            "交易所": "exchange",
            "日期": "as_of",
        }
    )[["code", "name", "exchange", "as_of"]]


def fetch_minute(symbol: str, start: str, end: str, period: str = "1", adjust: str = "qfq", market: str = "cn_stock"):
    return markets.get(market).fetch_minute(symbol, start, end, period=period, adjust=adjust)


def fetch_daily(symbol: str, start: str, end: str, adjust: str = "qfq", market: str = "cn_stock"):
    return markets.get(market).fetch_daily(symbol, start, end, adjust=adjust)
