"""Tests for the backtesting engine execution contract."""

import pandas as pd
import pytest

from back_test.engine import execute_and_evaluate
from back_test.schemas import EvaluateRequest


@pytest.mark.asyncio
async def test_engine_accepts_alpha_factor_template_inheritance(monkeypatch):
    """Factor code may inherit AlphaFactorTemplate without importing it."""
    dates = pd.date_range("2023-01-01", periods=30, freq="B")
    symbols = ["000001", "600000", "000002"]
    close = pd.DataFrame(
        [[10 + day + stock for stock in range(len(symbols))] for day in range(len(dates))],
        index=dates,
        columns=symbols,
        dtype=float,
    )
    market_data = {
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": close * 1000,
        "amount": close * close * 1000,
    }

    async def fake_load_market_data(eval_config):
        return market_data

    monkeypatch.setattr("back_test.engine.load_market_data", fake_load_market_data)

    request = EvaluateRequest(
        alpha_id="alpha-template-test",
        alpha_description="template inheritance smoke test",
        alpha_code='''class TemplateMomentum(AlphaFactorTemplate):
    def __init__(self, window: int = 5, **kwargs):
        self.window = window

    def compute(self, data: dict):
        close = data["close"]
        return close / close.shift(self.window) - 1.0

    def get_name(self):
        return "TemplateMomentum"
''',
        parameters={"window": 5},
        eval_config={},
    )

    status, metrics, error_msg = await execute_and_evaluate(request)

    assert status == "success"
    assert metrics is not None
    assert error_msg is None


@pytest.mark.asyncio
async def test_engine_produces_distinct_metrics_for_distinct_factors(monkeypatch):
    """Different factor values should not collapse to identical fallback metrics."""
    dates = pd.date_range("2023-01-01", periods=80, freq="B")
    symbols = ["000001", "600000", "000002", "600030", "600016"]
    close = pd.DataFrame(index=dates, columns=symbols, dtype=float)
    for idx, symbol in enumerate(symbols):
        t = pd.Series(range(len(dates)), index=dates, dtype=float)
        close[symbol] = 10 + idx * 2 + t * (0.02 + idx * 0.003) + (idx + 1) * 0.2 * ((t % 7) - 3) / 7
    volume = close * (10000 + pd.Series(range(len(dates)), index=dates).values.reshape(-1, 1) * 20)
    market_data = {
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": volume,
        "amount": close * volume,
    }

    async def fake_load_market_data(eval_config):
        return market_data

    monkeypatch.setattr("back_test.engine.load_market_data", fake_load_market_data)

    momentum_code = '''class MomentumAlpha(AlphaFactorTemplate):
    def __init__(self, window: int = 5, **kwargs):
        self.window = window
    def compute(self, data: dict):
        close = data["close"]
        return close / close.shift(self.window) - 1.0
    def get_name(self):
        return "MomentumAlpha"
'''
    reversal_code = '''class ReversalAlpha(AlphaFactorTemplate):
    def __init__(self, window: int = 5, **kwargs):
        self.window = window
    def compute(self, data: dict):
        close = data["close"]
        return -(close / close.shift(self.window) - 1.0)
    def get_name(self):
        return "ReversalAlpha"
'''

    status_a, metrics_a, _ = await execute_and_evaluate(EvaluateRequest(alpha_code=momentum_code, parameters={"window": 5}))
    status_b, metrics_b, _ = await execute_and_evaluate(EvaluateRequest(alpha_code=reversal_code, parameters={"window": 5}))

    assert status_a == "success"
    assert status_b == "success"
    assert metrics_a is not None
    assert metrics_b is not None
    assert metrics_a.ic_mean != metrics_b.ic_mean or metrics_a.sharpe != metrics_b.sharpe
