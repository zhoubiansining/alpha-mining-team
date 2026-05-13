"""Parquet-based disk cache.

Each (symbol, freq) is stored as a single parquet file. On read we slice by
the requested date range; on write we union with any existing rows so an
incremental fetch only pulls fresh days from upstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from data_api.config import settings


def _path(symbol: str, freq: str) -> Path:
    return settings.cache_dir / freq / f"{symbol}.parquet"


def load(symbol: str, freq: str, ts_col: str) -> Optional[pd.DataFrame]:
    p = _path(symbol, freq)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    if ts_col not in df.columns or df.empty:
        return None
    return df


def save(symbol: str, freq: str, df: pd.DataFrame, ts_col: str) -> None:
    if df is None or df.empty:
        return
    p = _path(symbol, freq)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = load(symbol, freq, ts_col)
    if existing is not None:
        merged = (
            pd.concat([existing, df], ignore_index=True)
            .drop_duplicates(subset=[ts_col], keep="last")
            .sort_values(ts_col)
            .reset_index(drop=True)
        )
    else:
        merged = df.sort_values(ts_col).reset_index(drop=True)
    merged.to_parquet(p, index=False)


def slice_range(
    df: pd.DataFrame,
    ts_col: str,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    s = df.copy()
    ts = pd.to_datetime(s[ts_col])
    if start is not None:
        s = s[ts >= pd.to_datetime(start)]
        ts = pd.to_datetime(s[ts_col])
    if end is not None:
        s = s[ts <= pd.to_datetime(end)]
    return s.reset_index(drop=True)
