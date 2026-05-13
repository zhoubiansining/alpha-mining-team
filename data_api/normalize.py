"""Symbol + column normalisation helpers.

backtest.md uses the canonical form ``SHSE.600000`` / ``SZSE.000001`` while
akshare expects the bare 6-digit code. We accept both at the API surface and
keep the SHSE/SZSE form on the wire.
"""

from __future__ import annotations

import re

import pandas as pd

_BARE_CODE_RE = re.compile(r"^\d{6}$")
_PREFIXED_RE = re.compile(r"^(SHSE|SZSE|SH|SZ)[.:]?(\d{6})$", re.IGNORECASE)


def to_bare_code(symbol: str) -> str:
    """Return the 6-digit code that akshare consumes."""
    s = symbol.strip()
    if _BARE_CODE_RE.match(s):
        return s
    m = _PREFIXED_RE.match(s)
    if m:
        return m.group(2)
    raise ValueError(f"Unrecognised A-share symbol: {symbol!r}")


def _exchange_of(code: str) -> str:
    """Return SH or SZ for the 6-digit code."""
    head = code[0]
    if head in {"6", "9"} or code.startswith("688"):
        return "SH"
    return "SZ"


def to_canonical(symbol: str) -> str:
    """Return the SHSE.XXXXXX / SZSE.XXXXXX canonical form used in responses."""
    code = to_bare_code(symbol)
    ex = _exchange_of(code)
    return f"{'SHSE' if ex == 'SH' else 'SZSE'}.{code}"


def to_sina_symbol(symbol: str) -> str:
    """Sina uses lower-case 'sh600000' / 'sz000001'."""
    code = to_bare_code(symbol)
    return f"{_exchange_of(code).lower()}{code}"


_MIN_COL_MAP_EM = {
    "时间": "datetime",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "均价": "vwap",
}

_MIN_COL_MAP_SINA = {
    "day": "datetime",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
}

_DAILY_COL_MAP_EM = {
    "日期": "date",
    "股票代码": "code",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_chg",
    "涨跌额": "change",
    "换手率": "turnover_rate",
}

_DAILY_COL_MAP_SINA = {
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
    "outstanding_share": "outstanding_share",
    "turnover": "turnover_rate",
}


_MIN_COLS = ["datetime", "open", "high", "low", "close", "volume", "amount", "vwap"]


def normalize_minute_frame(df: pd.DataFrame, source: str = "em") -> pd.DataFrame:
    """Map akshare minute frame to canonical English schema.

    ``source`` selects the column mapping: ``em`` (东方财富) or ``sina`` (新浪).
    Sina has no vwap column, so we derive it as amount/volume.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=_MIN_COLS)
    if source == "sina":
        out = df.rename(columns=_MIN_COL_MAP_SINA).copy()
    else:
        out = df.rename(columns=_MIN_COL_MAP_EM).copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    for col in ("open", "high", "low", "close", "volume", "amount"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "vwap" not in out.columns:
        out["vwap"] = (out["amount"] / out["volume"]).where(out["volume"] > 0, out["close"])
    # First bar after opening call auction may report open=0 on EM — patch it.
    if len(out) and out["open"].iloc[0] == 0:
        out.loc[out.index[0], "open"] = out["close"].iloc[0]
    return out[_MIN_COLS].dropna(subset=["close"]).reset_index(drop=True)


_DAILY_COLS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pct_chg",
    "turnover_rate",
]


def normalize_daily_frame(df: pd.DataFrame, source: str = "em") -> pd.DataFrame:
    """Map akshare daily frame to canonical English schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=_DAILY_COLS)
    if source == "sina":
        out = df.rename(columns=_DAILY_COL_MAP_SINA).copy()
    else:
        out = df.rename(columns=_DAILY_COL_MAP_EM).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "pct_chg" not in out.columns:
        out["pct_chg"] = (out["close"].pct_change() * 100).fillna(0.0)
    if "turnover_rate" not in out.columns:
        out["turnover_rate"] = 0.0
    return out[_DAILY_COLS].dropna(subset=["close"]).reset_index(drop=True)
