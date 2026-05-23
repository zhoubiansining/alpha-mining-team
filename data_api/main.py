"""FastAPI service exposing real A-share market data via akshare."""

from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import ORJSONResponse  # type: ignore[import-not-found]

# ORJSON is optional; fall back to FastAPI default if unavailable.
try:  # pragma: no cover - import guard only
    import orjson  # noqa: F401
    _default_response_class = ORJSONResponse
except Exception:  # pragma: no cover
    from fastapi.responses import JSONResponse as _default_response_class  # type: ignore[assignment]

from data_api import markets, source
from data_api.config import settings
from data_api.schemas import (
    Bar,
    DailyBar,
    DailyBarsResponse,
    HealthResponse,
    MinuteBarsResponse,
    PanelEntry,
    PanelRequest,
    PanelResponse,
    TradeCalendarResponse,
    UniverseResponse,
)

logger = logging.getLogger("data_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(
    title="Alpha-Mining Data API",
    version="0.1.0",
    description="Real A-share market data (bars / universe / calendar) for the alpha mining pipeline.",
    default_response_class=_default_response_class,
)


def _clean_float(v) -> float:
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


def _bars_from_minute_df(df: pd.DataFrame) -> list[Bar]:
    if df is None or df.empty:
        return []
    out: list[Bar] = []
    for row in df.itertuples(index=False):
        out.append(
            Bar(
                datetime=pd.Timestamp(row.datetime).strftime("%Y-%m-%d %H:%M:%S"),
                open=_clean_float(row.open),
                high=_clean_float(row.high),
                low=_clean_float(row.low),
                close=_clean_float(row.close),
                volume=_clean_float(row.volume),
                amount=_clean_float(row.amount),
                vwap=_clean_float(row.vwap),
            )
        )
    return out


def _bars_from_daily_df(df: pd.DataFrame) -> list[DailyBar]:
    if df is None or df.empty:
        return []
    out: list[DailyBar] = []
    for row in df.itertuples(index=False):
        out.append(
            DailyBar(
                date=str(row.date),
                open=_clean_float(row.open),
                high=_clean_float(row.high),
                low=_clean_float(row.low),
                close=_clean_float(row.close),
                volume=_clean_float(row.volume),
                amount=_clean_float(row.amount),
                pct_chg=_clean_float(getattr(row, "pct_chg", 0.0)),
                turnover_rate=_clean_float(getattr(row, "turnover_rate", 0.0)),
            )
        )
    return out


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/markets")
def list_markets_endpoint() -> dict:
    return {"markets": markets.list_markets()}


@app.get("/universe", response_model=UniverseResponse)
def universe(
    name: str = Query("HS300", description="A-share | HS300 | ZZ500 | ZZ1000 | SSE50 | (market-specific)"),
    market: str = Query("cn_stock", description="cn_stock | us_stock | hk_stock | cn_etf | cn_index | cn_future | fx"),
    canonical: bool = Query(False, description="If true, emit canonical symbol form"),
) -> UniverseResponse:
    try:
        adapter = markets.get(market)
        codes = adapter.universe(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:  # network / upstream error
        logger.exception("universe fetch failed")
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")
    symbols = [adapter.to_canonical(c) for c in codes] if canonical else codes
    return UniverseResponse(name=name.upper(), count=len(symbols), symbols=symbols)


@app.get("/trade_calendar", response_model=TradeCalendarResponse)
def trade_calendar(
    start: str = Query(..., description="YYYY-MM-DD inclusive"),
    end: str = Query(..., description="YYYY-MM-DD inclusive"),
) -> TradeCalendarResponse:
    try:
        days = source.trade_days_in_range(start, end)
    except Exception as e:
        logger.exception("calendar fetch failed")
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")
    return TradeCalendarResponse(start=start, end=end, trade_days=days)


@app.get("/bars/minute", response_model=MinuteBarsResponse)
def minute_bars(
    symbol: str = Query(..., description="e.g. 600000, SHSE.600000"),
    start: Optional[str] = Query(None, description="YYYY-MM-DD HH:MM:SS — defaults to last N trade days"),
    end: Optional[str] = Query(None, description="YYYY-MM-DD HH:MM:SS"),
    period: str = Query("1", description="1 | 5 | 15 | 30 | 60"),
    adjust: str = Query("qfq", description="qfq | hfq | ''"),
    market: str = Query("cn_stock", description="only cn_stock supports minute today"),
) -> MinuteBarsResponse:
    if start is None or end is None:
        s, e = source.minute_window_for_recent_days(settings.minute_history_days)
        start = start or s
        end = end or e
    try:
        adapter = markets.get(market)
        df = adapter.fetch_minute(symbol, start, end, period=period, adjust=adjust)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.exception("minute fetch failed for %s", symbol)
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")
    bars = _bars_from_minute_df(df)
    return MinuteBarsResponse(
        symbol=adapter.to_canonical(symbol),
        period=period,
        start=start,
        end=end,
        count=len(bars),
        bars=bars,
    )


@app.get("/bars/daily", response_model=DailyBarsResponse)
def daily_bars(
    symbol: str = Query(...),
    start: str = Query(..., description="YYYY-MM-DD or YYYYMMDD"),
    end: str = Query(..., description="YYYY-MM-DD or YYYYMMDD"),
    adjust: str = Query("qfq", description="qfq | hfq | ''"),
    market: str = Query("cn_stock", description="cn_stock | us_stock | hk_stock | cn_etf | cn_index | cn_future | fx"),
) -> DailyBarsResponse:
    try:
        adapter = markets.get(market)
        df = adapter.fetch_daily(symbol, start, end, adjust=adjust)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.exception("daily fetch failed for %s", symbol)
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")
    bars = _bars_from_daily_df(df)
    return DailyBarsResponse(
        symbol=adapter.to_canonical(symbol),
        start=start,
        end=end,
        count=len(bars),
        bars=bars,
    )


@app.post("/bars/minute/panel", response_model=PanelResponse)
def minute_panel(req: PanelRequest) -> PanelResponse:
    if len(req.symbols) > settings.max_panel_symbols:
        raise HTTPException(
            status_code=400,
            detail=f"too many symbols: {len(req.symbols)} > {settings.max_panel_symbols}",
        )
    market = getattr(req, "market", None) or "cn_stock"
    try:
        adapter = markets.get(market)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    entries: list[PanelEntry] = []
    for sym in req.symbols:
        try:
            df = adapter.fetch_minute(sym, req.start, req.end, period=req.period, adjust=req.adjust)
            bars = _bars_from_minute_df(df)
            entries.append(
                PanelEntry(symbol=adapter.to_canonical(sym), count=len(bars), bars=bars)
            )
        except Exception as e:
            logger.warning("panel: %s failed: %s", sym, e)
            entries.append(
                PanelEntry(symbol=sym, count=0, bars=[], error=str(e))
            )
    return PanelResponse(period=req.period, start=req.start, end=req.end, entries=entries)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "data_api.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
