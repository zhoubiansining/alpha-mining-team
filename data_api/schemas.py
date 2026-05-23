"""Pydantic request / response models for the data-API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str = "data-api"
    source: str = "akshare"


class UniverseResponse(BaseModel):
    name: str
    count: int
    symbols: list[str]


class TradeCalendarResponse(BaseModel):
    start: str
    end: str
    trade_days: list[str]


class Bar(BaseModel):
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    vwap: Optional[float] = None


class MinuteBarsResponse(BaseModel):
    symbol: str
    period: str
    start: str
    end: str
    count: int
    bars: list[Bar]


class DailyBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    pct_chg: Optional[float] = None
    turnover_rate: Optional[float] = None


class DailyBarsResponse(BaseModel):
    symbol: str
    start: str
    end: str
    count: int
    bars: list[DailyBar]


class PanelRequest(BaseModel):
    symbols: list[str] = Field(min_length=1)
    start: str
    end: str
    period: str = "1"
    adjust: str = ""
    market: str = "cn_stock"


class PanelEntry(BaseModel):
    symbol: str
    count: int
    bars: list[Bar]
    error: Optional[str] = None


class PanelResponse(BaseModel):
    period: str
    start: str
    end: str
    entries: list[PanelEntry]
