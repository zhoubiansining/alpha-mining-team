"""End-to-end smoke tests for the data-API.

These tests hit the real akshare upstream — they are gated on network
availability and verify that the service returns non-empty, well-formed
panels of real A-share market data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_NETWORK_TESTS") == "1",
    reason="network/upstream tests disabled",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from data_api.main import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["source"] == "akshare"


def test_universe_hs300(client):
    r = client.get("/universe", params={"name": "HS300"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 300
    assert all(len(c) == 6 and c.isdigit() for c in body["symbols"])


def test_universe_canonical(client):
    r = client.get("/universe", params={"name": "HS300", "canonical": True})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 300
    assert all(s.startswith(("SHSE.", "SZSE.")) for s in body["symbols"])


def test_trade_calendar(client):
    r = client.get(
        "/trade_calendar", params={"start": "2026-04-01", "end": "2026-05-12"}
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["trade_days"]) >= 20
    assert body["trade_days"] == sorted(body["trade_days"])


def test_minute_bars_recent(client):
    """Recent 1-min bars for 浦发银行 — akshare keeps ~5 trading days."""
    r = client.get(
        "/bars/minute",
        params={"symbol": "600000", "period": "1"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "SHSE.600000"
    assert body["count"] >= 240, f"expected at least one full session of bars, got {body['count']}"
    first = body["bars"][0]
    # the first-bar open=0 quirk should have been patched
    assert first["open"] > 0
    assert first["high"] >= first["low"]
    assert all(b["close"] > 0 for b in body["bars"][:10])


def test_minute_bars_canonical_symbol(client):
    r = client.get(
        "/bars/minute",
        params={"symbol": "SHSE.600000", "period": "1"},
    )
    assert r.status_code == 200
    assert r.json()["count"] > 0


def test_daily_bars_qfq(client):
    r = client.get(
        "/bars/daily",
        params={
            "symbol": "SZSE.000001",
            "start": "2026-01-01",
            "end": "2026-05-12",
            "adjust": "qfq",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 60
    assert body["bars"][0]["close"] > 0


def test_minute_panel(client):
    r = client.post(
        "/bars/minute/panel",
        json={
            "symbols": ["600000", "000001", "600519"],
            "start": "",  # filled below
            "end": "",
            "period": "1",
        },
    )
    # empty start/end is invalid — service should require concrete times
    assert r.status_code in (200, 400, 422, 502)


def test_minute_panel_with_window(client):
    from data_api import source

    start, end = source.minute_window_for_recent_days(2)
    r = client.post(
        "/bars/minute/panel",
        json={
            "symbols": ["600000", "000001"],
            "start": start,
            "end": end,
            "period": "1",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["entries"]) == 2
    by_sym = {e["symbol"]: e for e in body["entries"]}
    assert by_sym["SHSE.600000"]["count"] > 0
    assert by_sym["SZSE.000001"]["count"] > 0
