"""Smoke tests for the multi-market data adapters.

Each test hits the real upstream and verifies non-empty, well-formed bars.
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


def test_list_markets(client):
    r = client.get("/markets")
    assert r.status_code == 200
    body = r.json()
    expected = {"cn_stock", "us_stock", "hk_stock", "cn_etf", "cn_index", "cn_future", "fx"}
    assert expected.issubset(set(body["markets"]))


def test_us_daily_aapl(client):
    r = client.get(
        "/bars/daily",
        params={
            "market": "us_stock",
            "symbol": "AAPL",
            "start": "2026-01-01",
            "end": "2026-05-12",
            "adjust": "qfq",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["symbol"] == "US.AAPL"
    assert body["count"] >= 60, f"too few US bars: {body['count']}"
    bar = body["bars"][-1]
    assert bar["close"] > 0
    assert bar["high"] >= bar["low"]


def test_hk_daily_tencent(client):
    r = client.get(
        "/bars/daily",
        params={
            "market": "hk_stock",
            "symbol": "00700",
            "start": "2026-01-01",
            "end": "2026-05-12",
            "adjust": "qfq",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["symbol"] == "HK.00700"
    assert body["count"] >= 60
    assert body["bars"][-1]["close"] > 0


def test_cn_etf_daily_hs300(client):
    r = client.get(
        "/bars/daily",
        params={
            "market": "cn_etf",
            "symbol": "510300",
            "start": "2026-01-01",
            "end": "2026-05-12",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["symbol"] == "SHSE.510300"
    assert body["count"] >= 60
    assert body["bars"][-1]["close"] > 0


def test_cn_index_daily_hs300(client):
    r = client.get(
        "/bars/daily",
        params={
            "market": "cn_index",
            "symbol": "HS300",
            "start": "2026-01-01",
            "end": "2026-05-12",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["symbol"] == "SHSE.000300"
    assert body["count"] >= 60
    assert body["bars"][-1]["close"] > 0


def test_cn_future_daily_if(client):
    r = client.get(
        "/bars/daily",
        params={
            "market": "cn_future",
            "symbol": "IF0",
            "start": "2026-01-01",
            "end": "2026-05-12",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["symbol"] == "CN.FUT.IF0"
    assert body["count"] >= 60
    assert body["bars"][-1]["close"] > 0


def test_fx_usd_cny(client):
    r = client.get(
        "/bars/daily",
        params={
            "market": "fx",
            "symbol": "USD",
            "start": "2026-01-01",
            "end": "2026-05-13",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] >= 30
    assert body["bars"][-1]["close"] > 0


def test_us_minute_not_implemented(client):
    r = client.get(
        "/bars/minute",
        params={"market": "us_stock", "symbol": "AAPL"},
    )
    assert r.status_code == 501


def test_unknown_market_400(client):
    r = client.get(
        "/bars/daily",
        params={"market": "bogus", "symbol": "X", "start": "2026-01-01", "end": "2026-05-12"},
    )
    assert r.status_code == 400


def test_universe_us(client):
    r = client.get("/universe", params={"market": "us_stock", "name": "all"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] > 1000  # full US list is large


def test_universe_cn_index_aliases(client):
    r = client.get("/universe", params={"market": "cn_index", "name": "all"})
    assert r.status_code == 200
    body = r.json()
    assert "HS300" in body["symbols"]
    assert "CYB" in body["symbols"]


def test_universe_cn_future(client):
    r = client.get("/universe", params={"market": "cn_future", "name": "main"})
    assert r.status_code == 200
    body = r.json()
    assert "IF0" in body["symbols"]
    assert "CU0" in body["symbols"]
