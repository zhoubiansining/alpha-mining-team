"""Per-market data adapters.

Each adapter exposes the same minimal contract::

    name: str
    fetch_daily(symbol, start, end, adjust) -> DataFrame[date,open,high,low,close,volume,amount,pct_chg,turnover_rate]
    fetch_minute(symbol, start, end, period, adjust) -> DataFrame[datetime,open,...,vwap]   # may raise NotImplementedError
    universe(name) -> list[str]                                                              # may raise NotImplementedError
    to_canonical(symbol) -> str
    to_native(symbol) -> str                                                                 # for caching key

The router in ``data_api.main`` dispatches the ``market`` query param onto
this registry. Unknown markets → HTTP 400.
"""

from __future__ import annotations

from importlib import import_module
from typing import Protocol

import pandas as pd


class MarketAdapter(Protocol):
    name: str

    def fetch_daily(
        self, symbol: str, start: str, end: str, adjust: str = "qfq"
    ) -> pd.DataFrame: ...

    def fetch_minute(
        self, symbol: str, start: str, end: str, period: str = "1", adjust: str = "qfq"
    ) -> pd.DataFrame: ...

    def universe(self, name: str) -> list[str]: ...

    def to_canonical(self, symbol: str) -> str: ...

    def to_native(self, symbol: str) -> str: ...


_REGISTRY: dict[str, MarketAdapter] = {}


def _register(name: str, module_path: str) -> None:
    mod = import_module(module_path)
    _REGISTRY[name] = mod  # type: ignore[assignment]


# Lazy-register the built-in markets. Import errors here would mask test
# failures, so we let any import problem propagate.
_register("cn_stock", "data_api.markets.cn_stock")
_register("us_stock", "data_api.markets.us_stock")
_register("hk_stock", "data_api.markets.hk_stock")
_register("cn_etf", "data_api.markets.cn_etf")
_register("cn_index", "data_api.markets.cn_index")
_register("cn_future", "data_api.markets.cn_future")
_register("fx", "data_api.markets.fx")


def get(market: str) -> MarketAdapter:
    key = market.lower().replace("-", "_")
    aliases = {
        "cn": "cn_stock",
        "a": "cn_stock",
        "ashare": "cn_stock",
        "a_share": "cn_stock",
        "us": "us_stock",
        "hk": "hk_stock",
        "etf": "cn_etf",
        "index": "cn_index",
        "future": "cn_future",
        "futures": "cn_future",
        "forex": "fx",
    }
    key = aliases.get(key, key)
    if key not in _REGISTRY:
        raise ValueError(
            f"unknown market {market!r}. known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def list_markets() -> list[str]:
    return sorted(_REGISTRY)
