import asyncio
import logging
import os
from typing import Dict

import httpx
import numpy as np
import pandas as pd

DATA_API_BASE = os.getenv("DATA_API_BASE", "http://localhost:8001")
_MARKET_DATA_CACHE = {}
logger = logging.getLogger(__name__)


async def fetch_stock_data(client: httpx.AsyncClient, symbol: str, start: str, end: str):
    """拉取单只股票的日线数据"""
    url = f"{DATA_API_BASE}/bars/daily"
    params = {"symbol": symbol, "start": start, "end": end, "market": "cn_stock", "adjust": "qfq"}
    try:
        response = await client.get(url, params=params, timeout=15.0)
        if response.status_code == 200:
            bars = response.json().get("bars", [])
            return symbol, bars
        logger.warning("Daily bars fetch failed | symbol=%s | status=%s | body=%s", symbol, response.status_code, response.text[:200])
    except Exception as e:
        logger.warning("Daily bars fetch exception | symbol=%s | error=%s", symbol, e)
    return symbol, []


def _build_shadow_bars(symbols: list[str], start_date: str, end_date: str) -> list[pd.DataFrame]:
    """Build deterministic non-constant market data for offline contract tests."""
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    raw_data: list[pd.DataFrame] = []

    for symbol_index, symbol in enumerate(symbols):
        t = np.arange(len(dates), dtype=float)
        base_price = 8.0 + symbol_index * 3.0
        trend = (0.0008 + symbol_index * 0.00025) * t
        seasonal = 0.035 * np.sin(t / 6.0 + symbol_index * 0.7)
        short_cycle = 0.012 * np.cos(t / (2.5 + symbol_index * 0.2))
        close = base_price * (1.0 + trend + seasonal + short_cycle)
        close = np.maximum(close, 1.0)

        open_price = close * (1.0 + 0.004 * np.cos(t / 3.0 + symbol_index))
        high = np.maximum(open_price, close) * (1.0 + 0.008 + 0.002 * (symbol_index % 3))
        low = np.minimum(open_price, close) * (1.0 - 0.008 - 0.001 * (symbol_index % 2))
        volume = 50000.0 * (1.0 + 0.15 * np.sin(t / 4.0 + symbol_index)) * (1.0 + symbol_index * 0.08)
        volume = np.maximum(volume, 1000.0)
        amount = close * volume

        raw_data.append(pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
            "symbol": symbol,
        }))

    return raw_data


async def load_market_data(eval_config: dict) -> Dict[str, pd.DataFrame]:
    """对接 Data API 获取股票池并构建因子计算所需的面板数据"""
    universe = eval_config.get("universe", "HS300")
    start_date = eval_config.get("start_date", "2023-01-01")
    end_date = eval_config.get("end_date", "2023-06-30")

    cache_key = f"{DATA_API_BASE}_{universe}_{start_date}_{end_date}"
    if cache_key in _MARKET_DATA_CACHE:
        return _MARKET_DATA_CACHE[cache_key]

    async with httpx.AsyncClient() as client:
        try:
            univ_resp = await client.get(f"{DATA_API_BASE}/universe?name={universe}&market=cn_stock", timeout=15.0)
            symbols = univ_resp.json().get("symbols", []) if univ_resp.status_code == 200 else []
        except Exception as exc:
            logger.warning("Universe fetch failed | universe=%s | error=%s", universe, exc)
            symbols = []

        if not symbols:
            logger.warning("Universe empty; using standard debug symbol set")
            symbols = ["000001", "600000", "000002", "600016", "600030"]
        else:
            symbols = symbols[:10]

        tasks = [fetch_stock_data(client, symbol, start_date, end_date) for symbol in symbols]
        results = await asyncio.gather(*tasks)

    raw_data = []
    for symbol, bars in results:
        if not bars:
            continue
        df = pd.DataFrame(bars)
        if df.empty:
            continue
        df["symbol"] = symbol
        raw_data.append(df)

    data_source = "real"
    if not raw_data:
        logger.warning("All daily bars empty; using deterministic shadow market data")
        raw_data = _build_shadow_bars(symbols, start_date, end_date)
        data_source = "shadow"

    master_df = pd.concat(raw_data, ignore_index=True)
    master_df["date"] = pd.to_datetime(master_df["date"])

    market_data = {}
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        pivot_df = master_df.pivot(index="date", columns="symbol", values=col)
        market_data[col] = pivot_df.sort_index().ffill().bfill()

    close = market_data["close"]
    logger.info(
        "Market data loaded | universe=%s | source=%s | rows=%d | symbols=%d | close_std=%.6f",
        universe,
        data_source,
        close.shape[0],
        close.shape[1],
        float(close.stack().std()) if not close.empty else 0.0,
    )

    _MARKET_DATA_CACHE[cache_key] = market_data
    return market_data
