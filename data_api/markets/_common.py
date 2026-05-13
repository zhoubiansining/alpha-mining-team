"""Shared helpers for market adapters."""

from __future__ import annotations

import time
import warnings
from typing import Callable, Optional, TypeVar

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="py_mini_racer")

T = TypeVar("T")


def retry(fn: Callable[[], T], attempts: int = 3, base_delay: float = 0.7) -> T:
    """Run ``fn`` with exponential backoff."""
    last: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(base_delay * (2 ** i))
    assert last is not None
    raise last


def to_date_compact(s: str) -> str:
    return s.replace("-", "")


def to_date_iso(s: str) -> str:
    s = s.replace("-", "")
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def derive_pct_chg(df: pd.DataFrame) -> pd.DataFrame:
    if "pct_chg" not in df.columns:
        df = df.copy()
        df["pct_chg"] = (df["close"].pct_change() * 100).fillna(0.0)
    return df


DAILY_COLS = [
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


def finalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all canonical daily columns exist and types are sane."""
    if df is None or df.empty:
        return pd.DataFrame(columns=DAILY_COLS)
    df = df.copy()
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0
    df = derive_pct_chg(df)
    if "turnover_rate" not in df.columns:
        df["turnover_rate"] = 0.0
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df[DAILY_COLS].dropna(subset=["close"]).reset_index(drop=True)


MINUTE_COLS = ["datetime", "open", "high", "low", "close", "volume", "amount", "vwap"]


def finalize_minute(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=MINUTE_COLS)
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0
    if "vwap" not in df.columns:
        df["vwap"] = (df["amount"] / df["volume"]).where(df["volume"] > 0, df["close"])
    if len(df) and df["open"].iloc[0] == 0:
        df.loc[df.index[0], "open"] = df["close"].iloc[0]
    return df[MINUTE_COLS].dropna(subset=["close"]).reset_index(drop=True)
