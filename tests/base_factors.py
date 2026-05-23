"""Reusable daily baseline factor libraries for integration debugging.

Each helper returns a one-factor library by default. The returned dictionaries
match the baseline_factor_library contract accepted by alpha_mining.workflow.run_mining.
"""

from __future__ import annotations

from copy import deepcopy


MOMENTUM_20D_CODE = '''class Momentum20D:
    def __init__(self, window: int = 20, **kwargs):
        self.window = window

    def compute(self, data: dict):
        close = data["close"]
        return close / close.shift(self.window) - 1.0

    def get_name(self) -> str:
        return f"Momentum_{self.window}D"
'''

MEAN_REVERSION_20D_CODE = '''class MeanReversion20D:
    def __init__(self, window: int = 20, **kwargs):
        self.window = window

    def compute(self, data: dict):
        close = data["close"]
        mean = close.rolling(self.window).mean()
        std = close.rolling(self.window).std().replace(0, 1e-12)
        return -((close - mean) / std)

    def get_name(self) -> str:
        return f"MeanReversion_{self.window}D"
'''

AMPLITUDE_20D_CODE = '''class Amplitude20D:
    def __init__(self, window: int = 20, **kwargs):
        self.window = window

    def compute(self, data: dict):
        high = data["high"]
        low = data["low"]
        close = data["close"]
        amplitude = (high.rolling(self.window).max() - low.rolling(self.window).min()) / close
        return -amplitude

    def get_name(self) -> str:
        return f"Amplitude_{self.window}D"
'''

VOLUME_RATIO_20D_CODE = '''class VolumeRatio20D:
    def __init__(self, window: int = 20, **kwargs):
        self.window = window

    def compute(self, data: dict):
        volume = data["volume"]
        avg_volume = volume.rolling(self.window).mean().replace(0, 1e-12)
        return volume / avg_volume - 1.0

    def get_name(self) -> str:
        return f"VolumeRatio_{self.window}D"
'''

VWAP_DEVIATION_10D_CODE = '''class VwapDeviation10D:
    def __init__(self, window: int = 10, **kwargs):
        self.window = window

    def compute(self, data: dict):
        close = data["close"]
        amount = data["amount"]
        volume = data["volume"].replace(0, 1e-12)
        vwap = amount / volume
        deviation = close / vwap.replace(0, 1e-12) - 1.0
        return -deviation.rolling(self.window).mean()

    def get_name(self) -> str:
        return f"VwapDeviation_{self.window}D"
'''

_BASE_FACTORS = {
    "momentum_20d": {
        "id": "baseline-momentum-20d",
        "name": "Momentum 20D",
        "code": MOMENTUM_20D_CODE,
        "description": "20-day price momentum factor based on close / delayed close - 1.",
        "parameters": {"window": 20},
        "intuition": "Stocks with stronger recent price trends may continue to outperform over short horizons.",
        "category": "momentum",
        "evaluation": {},
    },
    "mean_reversion_20d": {
        "id": "baseline-mean-reversion-20d",
        "name": "Mean Reversion 20D",
        "code": MEAN_REVERSION_20D_CODE,
        "description": "Negative 20-day close-price z-score factor.",
        "parameters": {"window": 20},
        "intuition": "Extreme short-term deviations from the rolling mean may revert.",
        "category": "mean_reversion",
        "evaluation": {},
    },
    "amplitude_20d": {
        "id": "baseline-amplitude-20d",
        "name": "Amplitude 20D",
        "code": AMPLITUDE_20D_CODE,
        "description": "Negative 20-day high-low range scaled by close.",
        "parameters": {"window": 20},
        "intuition": "Lower recent trading range can proxy for stability and lower idiosyncratic noise.",
        "category": "amplitude",
        "evaluation": {},
    },
    "volume_ratio_20d": {
        "id": "baseline-volume-ratio-20d",
        "name": "Volume Ratio 20D",
        "code": VOLUME_RATIO_20D_CODE,
        "description": "Current volume relative to 20-day average volume.",
        "parameters": {"window": 20},
        "intuition": "Abnormal volume can capture liquidity shocks and attention effects.",
        "category": "liquidity",
        "evaluation": {},
    },
    "vwap_deviation_10d": {
        "id": "baseline-vwap-deviation-10d",
        "name": "VWAP Deviation 10D",
        "code": VWAP_DEVIATION_10D_CODE,
        "description": "Negative rolling mean of close-to-VWAP deviation.",
        "parameters": {"window": 10},
        "intuition": "Price deviations from volume-weighted trading levels can contain short-term reversal signals.",
        "category": "price_volume",
        "evaluation": {},
    },
}


def get_base_factor(name: str) -> dict:
    """Return one baseline factor by key."""
    return deepcopy(_BASE_FACTORS[name])


def get_base_factor_library(name: str = "momentum_20d") -> list[dict]:
    """Return a one-factor baseline library for debugging one optimization path."""
    return [get_base_factor(name)]


def get_all_base_factor_libraries() -> dict[str, list[dict]]:
    """Return one baseline library per built-in baseline factor."""
    return {name: [deepcopy(factor)] for name, factor in _BASE_FACTORS.items()}


def get_combined_base_factor_library() -> list[dict]:
    """Return all built-in baseline factors as one library."""
    return [deepcopy(factor) for factor in _BASE_FACTORS.values()]


def list_base_factor_names() -> list[str]:
    """List available built-in baseline factor keys."""
    return list(_BASE_FACTORS.keys())
