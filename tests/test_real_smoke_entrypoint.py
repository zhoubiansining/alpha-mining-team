"""Entrypoint tests for real smoke test assets."""

from tests.base_factors import get_base_factor_library


def test_smoke_baseline_available():
    baseline = get_base_factor_library("momentum_20d")
    assert len(baseline) == 1
    assert baseline[0]["name"] == "Momentum 20D"
