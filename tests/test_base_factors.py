"""Tests for reusable baseline factor libraries."""

import numpy as np
import pandas as pd

from tests.base_factors import (
    get_all_base_factor_libraries,
    get_base_factor_library,
    get_combined_base_factor_library,
    list_base_factor_names,
)


def _make_market_data():
    dates = pd.date_range("2023-01-01", periods=40, freq="B")
    symbols = ["000001", "600000", "000002"]
    base = pd.DataFrame(
        np.arange(len(dates) * len(symbols), dtype=float).reshape(len(dates), len(symbols)) + 10.0,
        index=dates,
        columns=symbols,
    )
    return {
        "open": base,
        "high": base + 1.0,
        "low": base - 1.0,
        "close": base + 0.5,
        "volume": base * 1000.0 + 10000.0,
        "amount": (base + 0.5) * (base * 1000.0 + 10000.0),
    }


def _instantiate_factor(factor: dict):
    local_env = {}
    exec(factor["code"], {}, local_env)
    classes = [obj for obj in local_env.values() if isinstance(obj, type)]
    assert len(classes) == 1
    return classes[0](**factor.get("parameters", {}))


def test_base_factor_names_available():
    names = list_base_factor_names()
    assert "momentum_20d" in names
    assert "mean_reversion_20d" in names
    assert "amplitude_20d" in names
    assert "volume_ratio_20d" in names


def test_one_factor_library_contract():
    library = get_base_factor_library("momentum_20d")
    assert len(library) == 1
    factor = library[0]
    assert {"id", "name", "code", "description", "parameters", "evaluation"}.issubset(factor)


def test_all_base_factors_are_executable():
    market_data = _make_market_data()
    for library in get_all_base_factor_libraries().values():
        factor = library[0]
        instance = _instantiate_factor(factor)
        values = instance.compute(market_data)
        assert isinstance(values, pd.DataFrame)
        assert values.shape == market_data["close"].shape
        assert hasattr(instance, "get_name")


def test_combined_base_factor_library():
    combined = get_combined_base_factor_library()
    assert len(combined) == len(list_base_factor_names())
    assert len({factor["id"] for factor in combined}) == len(combined)
