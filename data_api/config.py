"""Data-API settings."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    host: str = Field(default_factory=lambda: os.getenv("DATA_API_HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("DATA_API_PORT", "8001")))

    cache_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv(
                "DATA_API_CACHE_DIR",
                str(Path(__file__).resolve().parent / "cache_store"),
            )
        )
    )

    # akshare minute API limit: only the most recent ~5 trading days are
    # available via stock_zh_a_hist_min_em. Surface the limit so callers
    # can pre-validate ranges instead of getting empty frames.
    minute_history_days: int = Field(default=5)

    # Cap how many symbols a single panel request may include — fetching is
    # serial and per-symbol latency dominates.
    max_panel_symbols: int = Field(default=50)

    request_timeout_seconds: float = Field(default=30.0)


settings = Settings()
settings.cache_dir.mkdir(parents=True, exist_ok=True)
